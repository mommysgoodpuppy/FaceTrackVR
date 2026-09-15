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

from haar_surround_feature import External_Run_HSF

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuxiliaryEyeFeatures:
    pupil_dilation: float | None = None
    pupil_locked: bool = False
    pupil_center: tuple[float, float] | None = None
    pupil_axes: tuple[float, float] | None = None
    pupil_angle_degrees: float = 0.0
    pupil_radius_px: float | None = None
    frame_size: tuple[int, int] | None = None
    radius_range: tuple[float, float] | None = None
    sample_count: int = 0
    expressions: dict[str, float] = field(default_factory=dict)

    def diagnostics(self) -> dict:
        return {
            "pupil_locked": self.pupil_locked,
            "pupil_center": self.pupil_center,
            "pupil_axes": self.pupil_axes,
            "pupil_angle_degrees": self.pupil_angle_degrees,
            "pupil_radius_px": self.pupil_radius_px,
            "frame_size": self.frame_size,
            "radius_range": self.radius_range,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class PupilGeometry:
    center: tuple[float, float]
    axes: tuple[float, float]
    angle_degrees: float
    equivalent_radius: float


class NextAuxiliaryTracker:
    """Low-rate pupil geometry sidecar for NEXT.

    HSF is used only to measure the pupil radius; its gaze result does not
    replace NEXT gaze. Radius is normalized over a rolling window and gently
    smoothed because physiological pupil changes are much slower than gaze.
    """

    def __init__(
        self,
        detector=None,
        rate_hz: float = 15.0,
        history_size: int = 900,
        minimum_samples: int = 8,
    ):
        self._detector = detector
        self._interval = 1.0 / max(1.0, float(rate_hz))
        self._last_run = float("-inf")
        self._radii = deque(maxlen=max(minimum_samples, history_size))
        self._minimum_samples = minimum_samples
        self._smoothed = 0.5
        self._radius_range: tuple[float, float] | None = None
        self._features = AuxiliaryEyeFeatures(pupil_dilation=0.5)
        self._warning_logged = False

    @property
    def features(self) -> AuxiliaryEyeFeatures:
        return self._features

    def update(self, bgr_frame: np.ndarray, now: float | None = None) -> AuxiliaryEyeFeatures:
        now = time.monotonic() if now is None else float(now)
        if now - self._last_run < self._interval:
            return self._features
        self._last_run = now

        try:
            if bgr_frame.ndim == 2:
                gray = bgr_frame
            else:
                gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            if self._detector is None:
                # Autoradius must remain enabled: the changing radius is the
                # signal this sidecar exists to measure.
                self._detector = External_Run_HSF(False)
            center_x, center_y, _preview, _search_radius = self._detector.run(gray)
            geometry = self._measure_pupil_geometry(gray, center_x, center_y)
            if geometry is None:
                self._features = AuxiliaryEyeFeatures(
                    pupil_dilation=self._features.pupil_dilation,
                    frame_size=(gray.shape[1], gray.shape[0]),
                    radius_range=self._radius_range,
                    sample_count=len(self._radii),
                    expressions=self._features.expressions,
                )
                return self._features
            dilation = self._normalize_radius(
                geometry.equivalent_radius,
                geometry.center[0],
                geometry.center[1],
                gray.shape,
            )
            if dilation is not None:
                self._features = AuxiliaryEyeFeatures(
                    pupil_dilation=dilation,
                    pupil_locked=True,
                    pupil_center=geometry.center,
                    pupil_axes=geometry.axes,
                    pupil_angle_degrees=geometry.angle_degrees,
                    pupil_radius_px=geometry.equivalent_radius,
                    frame_size=(gray.shape[1], gray.shape[0]),
                    radius_range=self._radius_range,
                    sample_count=len(self._radii),
                    expressions=self._features.expressions,
                )
            self._warning_logged = False
        except (cv2.error, TypeError, ValueError, IndexError, AttributeError) as exc:
            if not self._warning_logged:
                logger.warning("NEXT auxiliary pupil detection failed: %s", exc)
                self._warning_logged = True
        return self._features

    @staticmethod
    def _measure_pupil_geometry(
        gray: np.ndarray, center_x, center_y
    ) -> PupilGeometry | None:
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
        _threshold, dark = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        contours, _hierarchy = cv2.findContours(
            dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

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

    def _normalize_radius(self, radius, center_x, center_y, frame_shape) -> float | None:
        radius = float(radius)
        center_x = float(center_x)
        center_y = float(center_y)
        height, width = frame_shape[:2]
        if (
            not all(math.isfinite(v) for v in (radius, center_x, center_y))
            or radius < 1.0
            or radius > min(height, width) / 3.0
            or not 0 <= center_x < width
            or not 0 <= center_y < height
        ):
            return None

        self._radii.append(radius)
        if len(self._radii) < self._minimum_samples:
            target = 0.5
        else:
            low, high = np.percentile(np.asarray(self._radii), [5.0, 95.0])
            self._radius_range = (float(low), float(high))
            span = float(high - low)
            # Until the observed pupil actually changes, neutral is more
            # honest than amplifying detector noise into full-range dilation.
            if span < max(0.25, float(low) * 0.03):
                target = 0.5
            else:
                target = float(np.clip((radius - low) / span, 0.0, 1.0))

        self._smoothed += 0.2 * (target - self._smoothed)
        return float(np.clip(self._smoothed, 0.0, 1.0))
