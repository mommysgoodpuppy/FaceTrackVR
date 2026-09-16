"""Low-rate EllSeg pupil segmentation for NEXT's pupil-dilation sidecar.

The published EllSeg model predicts complete elliptical eye structures rather
than only their visible pixels.  That makes it a good fit for oblique headset
cameras where an eyelid hides part of the pupil.  Inference is serialized on a
single daemon worker shared by both eyes so it never stalls NEXT's tracking
threads.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
import onnxruntime

from utils.misc_utils import resource_path
from utils.onnx_runtime import DML_INFERENCE_LOCK, create_inference_session

logger = logging.getLogger(__name__)

_MODEL_PATH = "Models/EllSeg_all.onnx"
_INPUT_WIDTH = 320
_INPUT_HEIGHT = 240


@dataclass(frozen=True)
class EllSegMeasurement:
    center: tuple[float, float]
    axes: tuple[float, float]
    angle_degrees: float
    equivalent_radius: float
    iris_center: tuple[float, float]
    iris_axes: tuple[float, float]
    iris_angle_degrees: float
    pupil_to_iris_ratio: float


class _EllSegRuntime:
    """One lazy ONNX session and worker shared by every eye tracker."""

    def __init__(self, use_gpu: bool, high_rate: bool):
        self._use_gpu = bool(use_gpu)
        self._high_rate = bool(high_rate)
        self._uses_directml = False
        self._jobs: queue.Queue = queue.Queue(maxsize=2)
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="EllSegPupil",
        )
        self._thread.start()

    def submit(self, frame: np.ndarray, gamma: float = 0.8) -> Future | None:
        future = Future()
        try:
            self._jobs.put_nowait((frame.copy(), future, float(gamma)))
        except queue.Full:
            return None
        return future

    def _create_session(self):
        model_path = resource_path(_MODEL_PATH)
        logger.info(
            "EllSeg pupil: loading %s (GPU preferred: %s, high rate: %s)",
            model_path,
            self._use_gpu,
            self._high_rate,
        )
        onnxruntime.disable_telemetry_events()
        # EllSeg is unusually expensive on CPU even at the normal one-update-
        # per-eye cadence (~460 ms of CPU per pass on a 5800X3D).  WebGPU cuts
        # that to a small asynchronous Vulkan dispatch and reproduces the same
        # fitted geometry on BSB input.  Use it whenever GPU inference is
        # requested; high_rate controls cadence/threading, not provider choice.
        if self._use_gpu:
            webgpu_session = _create_webgpu_session(model_path)
            if webgpu_session is not None:
                return webgpu_session

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 0 if self._high_rate else 1
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_mem_pattern = False
        session, self._uses_directml = create_inference_session(
            model_path,
            options,
            use_gpu=self._use_gpu,
            component="EllSeg pupil",
            logger=logger,
        )
        return session

    def _worker(self):
        session = None
        input_name = None
        while True:
            frame, future, gamma = self._jobs.get()
            if future.cancelled():
                continue
            try:
                if session is None:
                    session = self._create_session()
                    input_name = session.get_inputs()[0].name
                tensor, transform = _preprocess(frame, gamma=gamma)
                if self._uses_directml:
                    with DML_INFERENCE_LOCK:
                        logits = session.run(None, {input_name: tensor})[0]
                else:
                    logits = session.run(None, {input_name: tensor})[0]
                future.set_result(_measurement_from_logits(logits, transform))
            except Exception as exc:
                future.set_exception(exc)


_RUNTIMES: dict[tuple[bool, bool], _EllSegRuntime] = {}
_RUNTIME_LOCK = threading.Lock()


def _get_runtime(use_gpu: bool, high_rate: bool) -> _EllSegRuntime:
    key = (bool(use_gpu), bool(high_rate))
    if key not in _RUNTIMES:
        with _RUNTIME_LOCK:
            if key not in _RUNTIMES:
                _RUNTIMES[key] = _EllSegRuntime(
                    use_gpu=key[0],
                    high_rate=key[1],
                )
    return _RUNTIMES[key]


class EllSegPupilDetector:
    def __init__(self, use_gpu: bool = False, debug_rate: bool = False):
        self._use_gpu = bool(use_gpu)
        self._debug_rate = bool(debug_rate)

    def set_debug_rate(self, enabled: bool):
        self._debug_rate = bool(enabled)

    def submit(self, frame: np.ndarray, gamma: float = 0.8) -> Future | None:
        return _get_runtime(self._use_gpu, self._debug_rate).submit(frame, gamma)


def _create_webgpu_session(model_path):
    """Create the optional Linux WebGPU/Vulkan session, or return None."""
    try:
        import onnxruntime_ep_webgpu as webgpu_ep

        if "WebGpuExecutionProvider" not in onnxruntime.get_available_providers():
            onnxruntime.register_execution_provider_library(
                "webgpu_ep_registration",
                webgpu_ep.get_library_path(),
            )
        devices = [device for device in onnxruntime.get_ep_devices() if device.ep_name == webgpu_ep.get_ep_name()]
        if not devices:
            raise RuntimeError("no WebGPU/Vulkan device was discovered")

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_provider_for_devices(
            devices,
            {
                # NHWC is faster in plugin 0.1.0 but changes EllSeg's output
                # geometry. Preserve the exported model's NCHW numerics.
                "preferredLayout": "NCHW",
                # Plugin 0.1.0 currently returns an empty output list after
                # the first captured run for this model. Regular dispatch is
                # still fast enough for the debug-rate target.
                "enableGraphCapture": "0",
                "powerPreference": "high-performance",
            },
        )
        session = onnxruntime.InferenceSession(model_path, sess_options=options)
        logger.info("EllSeg pupil WebGPU providers: %s", session.get_providers())
        return session
    except Exception as exc:
        logger.warning(
            "EllSeg WebGPU unavailable (%s); using ONNX Runtime CPU fallback.",
            exc,
        )
        return None


@dataclass(frozen=True)
class _FrameTransform:
    scale: float
    vertical_shift: float
    source_width: int
    source_height: int


@lru_cache(maxsize=32)
def _gamma_lut(gamma: float) -> np.ndarray:
    gamma = round(float(np.clip(gamma, 0.5, 1.5)), 3)
    values = np.arange(256, dtype=np.float32) / 255.0
    return np.uint8(np.clip(np.power(values, gamma) * 255.0, 0.0, 255.0))


def _preprocess(
    frame: np.ndarray, gamma: float = 0.8
) -> tuple[np.ndarray, _FrameTransform]:
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    source_height, source_width = gray.shape[:2]
    if source_height < 2 or source_width < 2:
        raise ValueError("eye frame is empty")

    # EllSeg standardizes each frame, so a linear brightness multiplier would
    # be cancelled by the later z-score. Gamma is intentionally nonlinear:
    # values below 1 lift the dark BSB eye-camera shadows while retaining the
    # pupil/iris separation. It cannot reconstruct eyelashes' occluded pixels;
    # EllSeg's ellipse prediction handles that missing boundary instead.
    gamma = float(np.clip(gamma, 0.5, 1.5))
    if abs(gamma - 1.0) > 1e-3:
        gray = cv2.LUT(gray, _gamma_lut(gamma))

    # Match EllSeg's published inference preprocessing: width-align to 320,
    # center-pad/crop to 240 high, then normalize each image by its own stats.
    scale = _INPUT_WIDTH / float(source_width)
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        gray,
        (_INPUT_WIDTH, resized_height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    vertical_shift = (_INPUT_HEIGHT - resized_height) / 2.0
    if resized_height < _INPUT_HEIGHT:
        padding = _INPUT_HEIGHT - resized_height
        top = padding // 2
        prepared = np.pad(resized, ((top, padding - top), (0, 0)))
        vertical_shift = float(top)
    elif resized_height > _INPUT_HEIGHT:
        top = (resized_height - _INPUT_HEIGHT) // 2
        prepared = resized[top : top + _INPUT_HEIGHT]
        vertical_shift = float(-top)
    else:
        prepared = resized
        vertical_shift = 0.0

    prepared = prepared.astype(np.float32)
    standard_deviation = float(prepared.std())
    if standard_deviation < 1e-6:
        raise ValueError("eye frame has no usable contrast")
    prepared = (prepared - float(prepared.mean())) / standard_deviation
    tensor = np.ascontiguousarray(prepared[None, None], dtype=np.float32)
    return tensor, _FrameTransform(
        scale=scale,
        vertical_shift=vertical_shift,
        source_width=source_width,
        source_height=source_height,
    )


def _measurement_from_logits(
    logits: np.ndarray,
    transform: _FrameTransform,
) -> EllSegMeasurement | None:
    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError(f"unexpected EllSeg output shape: {logits.shape}")
    labels = np.argmax(logits[0], axis=0)
    pupil_ellipse = _largest_class_ellipse(labels, 2, minimum_area=50.0)
    iris_ellipse = _largest_class_ellipse(labels, 1, minimum_area=100.0)
    if pupil_ellipse is None or iris_ellipse is None:
        return None

    (center_x, center_y), (axis_a, axis_b), angle = pupil_ellipse
    (iris_x, iris_y), (iris_axis_a, iris_axis_b), iris_angle = iris_ellipse
    scale = transform.scale
    center_x /= scale
    center_y = (center_y - transform.vertical_shift) / scale
    axis_a /= scale
    axis_b /= scale
    iris_x /= scale
    iris_y = (iris_y - transform.vertical_shift) / scale
    iris_axis_a /= scale
    iris_axis_b /= scale

    values = (
        center_x,
        center_y,
        axis_a,
        axis_b,
        angle,
        iris_x,
        iris_y,
        iris_axis_a,
        iris_axis_b,
        iris_angle,
    )
    if not all(math.isfinite(value) for value in values):
        return None
    if not (
        0.0 <= center_x < transform.source_width
        and 0.0 <= center_y < transform.source_height
        and 0.0 <= iris_x < transform.source_width
        and 0.0 <= iris_y < transform.source_height
        and 4.0 <= min(axis_a, axis_b)
        and max(axis_a, axis_b) <= min(transform.source_width, transform.source_height)
        and min(axis_a, axis_b) / max(axis_a, axis_b) >= 0.12
        and 8.0 <= min(iris_axis_a, iris_axis_b)
        and max(iris_axis_a, iris_axis_b) <= min(transform.source_width, transform.source_height) * 1.5
    ):
        return None

    # Geometric mean of the semi-axes: equivalent radius of the full ellipse,
    # independent of its camera-angle-induced aspect ratio.
    equivalent_radius = math.sqrt((axis_a * 0.5) * (axis_b * 0.5))
    iris_equivalent_radius = math.sqrt((iris_axis_a * 0.5) * (iris_axis_b * 0.5))
    pupil_to_iris_ratio = equivalent_radius / iris_equivalent_radius
    if not 0.08 <= pupil_to_iris_ratio <= 0.9:
        return None
    return EllSegMeasurement(
        center=(float(center_x), float(center_y)),
        axes=(float(axis_a), float(axis_b)),
        angle_degrees=float(angle),
        equivalent_radius=float(equivalent_radius),
        iris_center=(float(iris_x), float(iris_y)),
        iris_axes=(float(iris_axis_a), float(iris_axis_b)),
        iris_angle_degrees=float(iris_angle),
        pupil_to_iris_ratio=float(pupil_to_iris_ratio),
    )


def _largest_class_ellipse(
    labels: np.ndarray,
    class_index: int,
    *,
    minimum_area: float,
):
    mask = np.uint8(labels == class_index) * 255
    contours, _hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    candidates = [contour for contour in contours if len(contour) >= 5 and cv2.contourArea(contour) >= minimum_area]
    if not candidates:
        return None
    return cv2.fitEllipse(max(candidates, key=cv2.contourArea))
