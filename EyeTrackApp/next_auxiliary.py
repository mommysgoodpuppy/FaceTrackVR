"""Auxiliary feature extraction for the NEXT eye tracker.

NEXT remains authoritative for gaze and the expressions it was trained to
predict.  This module supplies independent channels that are absent from the
NEXT model, beginning with pupil size.  ``AuxiliaryEyeFeatures.expressions`` is
the extension point for future models that predict additional Unified
Expressions shapes.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from ellseg_pupil import EllSegMeasurement, EllSegPupilDetector

logger = logging.getLogger(__name__)

# Fixed EllSeg/BSB model-space baseline. Real anatomical pupil/iris ratios can
# reach higher values, but the oblique BSB images and EllSeg's reconstructed
# masks consistently report about 0.52-0.55 for the supplied dilated samples.
# Keeping this fixed prevents prolonged dilation from being relearned as
# neutral. The user-facing output range remains an independent final remap.
ELLSEG_RATIO_CONSTRICTED = 0.25
ELLSEG_RATIO_DILATED = 0.55


@dataclass(frozen=True)
class AuxiliaryEyeFeatures:
    pupil_dilation: float | None = None
    pupil_dilation_raw: float | None = None
    pupil_locked: bool = False
    pupil_center: tuple[float, float] | None = None
    pupil_axes: tuple[float, float] | None = None
    pupil_angle_degrees: float = 0.0
    pupil_radius_px: float | None = None
    pupil_to_iris_ratio: float | None = None
    iris_center: tuple[float, float] | None = None
    iris_axes: tuple[float, float] | None = None
    iris_angle_degrees: float = 0.0
    frame_size: tuple[int, int] | None = None
    radius_range: tuple[float, float] | None = None
    ratio_range: tuple[float, float] | None = None
    sample_count: int = 0
    detector_name: str = "EllSeg"
    model_rate_hz: float = 0.0
    debug_rate: bool = False
    dilation_output_range: tuple[float, float] = (0.0, 1.0)
    preprocess_gamma: float = 0.8
    calibration_mode: str = ""
    expressions: dict[str, float] = field(default_factory=dict)

    def diagnostics(self) -> dict:
        return {
            "pupil_locked": self.pupil_locked,
            "pupil_dilation_raw": self.pupil_dilation_raw,
            "pupil_dilation_mapped": self.pupil_dilation,
            "pupil_center": self.pupil_center,
            "pupil_axes": self.pupil_axes,
            "pupil_angle_degrees": self.pupil_angle_degrees,
            "pupil_radius_px": self.pupil_radius_px,
            "pupil_to_iris_ratio": self.pupil_to_iris_ratio,
            "iris_center": self.iris_center,
            "iris_axes": self.iris_axes,
            "iris_angle_degrees": self.iris_angle_degrees,
            "frame_size": self.frame_size,
            "radius_range": self.radius_range,
            "ratio_range": self.ratio_range,
            "sample_count": self.sample_count,
            "detector_name": self.detector_name,
            "model_rate_hz": self.model_rate_hz,
            "debug_rate": self.debug_rate,
            "dilation_output_range": self.dilation_output_range,
            "preprocess_gamma": self.preprocess_gamma,
            "calibration_mode": self.calibration_mode,
        }


@dataclass(frozen=True)
class PupilGeometry:
    center: tuple[float, float]
    axes: tuple[float, float]
    angle_degrees: float
    equivalent_radius: float


class NextAuxiliaryTracker:
    """Low-rate pupil geometry sidecar for NEXT.

    EllSeg runs asynchronously at low rate and predicts complete pupil and iris
    ellipses even when an eyelid hides part of their edge. The pupil-to-iris
    size ratio is normalized against a fixed model-space baseline, cancelling
    most camera foreshortening without learning prolonged dilation as neutral.
    NEXT gaze is untouched.
    """

    def __init__(
        self,
        detector=None,
        rate_hz: float = 1.0,
        history_size: int = 300,
        minimum_samples: int = 4,
        use_gpu: bool = False,
        debug_rate: bool = False,
        dilation_min: float = 0.0,
        dilation_max: float = 1.0,
        preprocess_gamma: float = 0.8,
    ):
        self._detector = (
            detector if detector is not None else EllSegPupilDetector(use_gpu=use_gpu, debug_rate=debug_rate)
        )
        self._normal_rate_hz = max(0.1, float(rate_hz))
        self._debug_rate = bool(debug_rate)
        self._dilation_min = 0.0
        self._dilation_max = 1.0
        self._preprocess_gamma = 0.8
        self.set_dilation_output_range(dilation_min, dilation_max)
        self.set_preprocess_gamma(preprocess_gamma)
        # Debug mode submits on every tracking tick when the worker is free.
        # With a 60 Hz camera this targets 60 Hz without building a stale-frame
        # queue when the selected ONNX provider cannot keep up.
        self._interval = 0.0 if self._debug_rate else 1.0 / self._normal_rate_hz
        self._last_run = float("-inf")
        self._size_signals = deque(maxlen=max(minimum_samples, history_size))
        self._minimum_samples = minimum_samples
        self._smoothed = 0.5
        self._target = 0.5
        self._last_update: float | None = None
        self._radius_range: tuple[float, float] | None = None
        self._ratio_range: tuple[float, float] | None = None
        self._future = None
        self._asynchronous_detector = isinstance(self._detector, EllSegPupilDetector)
        self._geometry: PupilGeometry | None = None
        self._iris_center: tuple[float, float] | None = None
        self._iris_axes: tuple[float, float] | None = None
        self._iris_angle = 0.0
        self._pupil_to_iris_ratio: float | None = None
        self._frame_size: tuple[int, int] | None = None
        self._last_measurement = float("-inf")
        self._measurement_times = deque(maxlen=30)
        self._detector_name = "EllSeg" if self._asynchronous_detector else "HSF"
        self._features = AuxiliaryEyeFeatures(pupil_dilation=0.5)
        self._warning_logged = False

    @property
    def features(self) -> AuxiliaryEyeFeatures:
        return self._features

    def set_debug_rate(self, enabled: bool):
        """Switch between normal 1 Hz sampling and the 60 Hz debug target."""
        self._debug_rate = bool(enabled)
        self._interval = 0.0 if self._debug_rate else 1.0 / self._normal_rate_hz
        if self._asynchronous_detector:
            self._detector.set_debug_rate(self._debug_rate)

    def set_dilation_output_range(self, minimum: float, maximum: float):
        """Map the selected measured-dilation range onto the 0..1 output."""
        minimum = float(np.clip(minimum, 0.0, 0.99))
        maximum = float(np.clip(maximum, 0.01, 1.0))
        if maximum <= minimum:
            maximum = min(1.0, minimum + 0.01)
            minimum = min(minimum, maximum - 0.01)
        self._dilation_min = minimum
        self._dilation_max = maximum

    def set_preprocess_gamma(self, gamma: float):
        """Set nonlinear input brightening (values below 1 brighten shadows)."""
        self._preprocess_gamma = float(np.clip(gamma, 0.5, 1.5))

    def update(self, bgr_frame: np.ndarray, now: float | None = None) -> AuxiliaryEyeFeatures:
        now = time.monotonic() if now is None else float(now)
        try:
            if bgr_frame.ndim == 2:
                gray = bgr_frame
            else:
                gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            self._frame_size = (gray.shape[1], gray.shape[0])

            if self._future is not None and self._future.done():
                future, self._future = self._future, None
                measurement = future.result()
                if measurement is not None:
                    self._accept_ellseg(measurement, gray.shape, now)

            if now - self._last_run >= self._interval:
                if self._asynchronous_detector:
                    if self._future is None:
                        submitted = self._detector.submit(
                            bgr_frame, gamma=self._preprocess_gamma
                        )
                        if submitted is not None:
                            self._future = submitted
                            self._last_run = now
                else:
                    # Synchronous injection is retained for tests and as a
                    # compatibility path for the former HSF detector.
                    self._last_run = now
                    center_x, center_y, _preview, _search_radius = self._detector.run(gray)
                    geometry = self._measure_pupil_geometry(gray, center_x, center_y)
                    if geometry is not None:
                        self._accept_geometry(
                            geometry,
                            geometry.equivalent_radius,
                            gray.shape,
                            now,
                            ratio_based=False,
                        )

            self._advance_interpolation(now)
            locked = self._geometry is not None and (now - self._last_measurement <= max(3.0, self._interval * 3.0))
            mapped_dilation = self._map_dilation_output(self._smoothed)
            self._features = AuxiliaryEyeFeatures(
                pupil_dilation=mapped_dilation,
                pupil_dilation_raw=self._smoothed,
                pupil_locked=locked,
                pupil_center=self._geometry.center if locked else None,
                pupil_axes=self._geometry.axes if locked else None,
                pupil_angle_degrees=(self._geometry.angle_degrees if locked else 0.0),
                pupil_radius_px=(self._geometry.equivalent_radius if locked else None),
                pupil_to_iris_ratio=(self._pupil_to_iris_ratio if locked else None),
                iris_center=(self._iris_center if locked else None),
                iris_axes=(self._iris_axes if locked else None),
                iris_angle_degrees=(self._iris_angle if locked else 0.0),
                frame_size=self._frame_size,
                radius_range=self._radius_range,
                ratio_range=self._ratio_range,
                sample_count=len(self._size_signals),
                detector_name=self._detector_name,
                model_rate_hz=self._measured_rate_hz(),
                debug_rate=self._debug_rate,
                dilation_output_range=(self._dilation_min, self._dilation_max),
                preprocess_gamma=self._preprocess_gamma,
                calibration_mode=("fixed" if self._ratio_range else "rolling"),
                expressions=self._features.expressions,
            )
        except (
            cv2.error,
            TypeError,
            ValueError,
            IndexError,
            AttributeError,
            RuntimeError,
        ) as exc:
            if not self._warning_logged:
                logger.warning("NEXT auxiliary pupil detection failed: %s", exc)
                self._warning_logged = True
        return self._features

    def _map_dilation_output(self, value: float) -> float:
        span = self._dilation_max - self._dilation_min
        return float(np.clip((float(value) - self._dilation_min) / span, 0.0, 1.0))

    def _accept_ellseg(
        self,
        measurement: EllSegMeasurement,
        frame_shape,
        now: float,
    ):
        geometry = PupilGeometry(
            center=measurement.center,
            axes=measurement.axes,
            angle_degrees=measurement.angle_degrees,
            equivalent_radius=measurement.equivalent_radius,
        )
        self._iris_center = measurement.iris_center
        self._iris_axes = measurement.iris_axes
        self._iris_angle = measurement.iris_angle_degrees
        self._pupil_to_iris_ratio = measurement.pupil_to_iris_ratio
        self._accept_geometry(
            geometry,
            measurement.pupil_to_iris_ratio,
            frame_shape,
            now,
            ratio_based=True,
        )

    def _accept_geometry(
        self,
        geometry: PupilGeometry,
        size_signal: float,
        frame_shape,
        now: float,
        *,
        ratio_based: bool,
    ):
        if self._set_normalization_target(
            size_signal,
            geometry.center[0],
            geometry.center[1],
            frame_shape,
            ratio_based=ratio_based,
        ):
            self._geometry = geometry
            self._last_measurement = now
            self._measurement_times.append(now)
            self._warning_logged = False

    def _measured_rate_hz(self) -> float:
        if len(self._measurement_times) < 2:
            return 0.0
        elapsed = self._measurement_times[-1] - self._measurement_times[0]
        if elapsed <= 0.0:
            return 0.0
        return float((len(self._measurement_times) - 1) / elapsed)

    @staticmethod
    def _measure_pupil_geometry(gray: np.ndarray, center_x, center_y) -> PupilGeometry | None:
        """Measure the dark pupil contour near HSF's located center.

        HSF's returned radius controls its search kernel and is intentionally
        adaptive; it is not a stable pupil-size measurement. The equivalent
        radius of the local dark contour is the actual auxiliary signal.
        """
        height, width = gray.shape[:2]
        center_x = int(round(float(center_x)))
        center_y = int(round(float(center_y)))
        half = max(24, int(round(min(height, width) * 0.15)))
        x0, x1 = max(0, center_x - half), min(width, center_x + half + 1)
        y0, y1 = max(0, center_y - half), min(height, center_y + half + 1)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0 or min(roi.shape[:2]) < 12:
            return None

        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        _threshold, dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        contours, _hierarchy = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        local_x, local_y = center_x - x0, center_y - y0
        max_area = math.pi * (half * 0.8) ** 2
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not 20.0 <= area <= max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if max(w, h) == 0 or min(w, h) / max(w, h) < 0.35:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            contour_x = moments["m10"] / moments["m00"]
            contour_y = moments["m01"] / moments["m00"]
            distance = math.hypot(contour_x - local_x, contour_y - local_y)
            if distance > half * 0.55:
                continue
            candidates.append((distance, -area, area, contour))

        if not candidates:
            return None
        _distance, _negative_area, area, contour = min(candidates, key=lambda item: item[:2])
        moments = cv2.moments(contour)
        pupil_x = x0 + moments["m10"] / moments["m00"]
        pupil_y = y0 + moments["m01"] / moments["m00"]
        equivalent_radius = math.sqrt(area / math.pi)
        if len(contour) >= 5:
            (_ellipse_x, _ellipse_y), axes, angle = cv2.fitEllipse(contour)
            axes = (float(axes[0]), float(axes[1]))
            angle = float(angle)
        else:
            diameter = equivalent_radius * 2.0
            axes = (diameter, diameter)
            angle = 0.0
        return PupilGeometry(
            center=(float(pupil_x), float(pupil_y)),
            axes=axes,
            angle_degrees=angle,
            equivalent_radius=equivalent_radius,
        )

    def _set_normalization_target(
        self,
        size_signal,
        center_x,
        center_y,
        frame_shape,
        *,
        ratio_based: bool,
    ) -> bool:
        size_signal = float(size_signal)
        center_x = float(center_x)
        center_y = float(center_y)
        height, width = frame_shape[:2]
        if (
            not all(math.isfinite(v) for v in (size_signal, center_x, center_y))
            or size_signal <= 0.0
            or not 0 <= center_x < width
            or not 0 <= center_y < height
        ):
            return False
        if ratio_based:
            if not 0.08 <= size_signal <= 0.9:
                return False
        elif size_signal < 1.0 or size_signal > min(height, width) / 3.0:
            return False

        self._size_signals.append(size_signal)
        if ratio_based:
            # Do not infer min/max from recent history: a pupil that remains
            # dilated for minutes must stay dilated rather than slowly becoming
            # the new midpoint. Iris-relative scale makes one fixed baseline
            # substantially more portable than fixed pixel diameters.
            low = ELLSEG_RATIO_CONSTRICTED
            high = ELLSEG_RATIO_DILATED
            self._ratio_range = (low, high)
            target = float(np.clip((size_signal - low) / (high - low), 0.0, 1.0))
        elif len(self._size_signals) < self._minimum_samples:
            target = 0.5
        else:
            low, high = np.percentile(np.asarray(self._size_signals), [5.0, 95.0])
            self._radius_range = (float(low), float(high))
            span = float(high - low)
            # Until the observed pupil actually changes, neutral is more
            # honest than amplifying detector noise into full-range dilation.
            noise_floor = max(0.25, float(low) * 0.03)
            if span < noise_floor:
                target = 0.5
            else:
                target = float(np.clip((size_signal - low) / span, 0.0, 1.0))

        self._target = target
        return True

    def _advance_interpolation(self, now: float):
        if self._last_update is None:
            self._last_update = now
            return
        elapsed = max(0.0, min(0.25, now - self._last_update))
        self._last_update = now
        # Roughly 90% of a step is reached in one second, while per-frame
        # output remains smooth at normal camera rates.
        alpha = 1.0 - math.exp(-elapsed / 0.43)
        self._smoothed += alpha * (self._target - self._smoothed)
        self._smoothed = float(np.clip(self._smoothed, 0.0, 1.0))
