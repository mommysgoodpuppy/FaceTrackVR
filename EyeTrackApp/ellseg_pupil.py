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

import cv2
import numpy as np
import onnxruntime

from utils.misc_utils import resource_path

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

    def __init__(self):
        self._jobs: queue.Queue = queue.Queue(maxsize=2)
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="EllSegPupil",
        )
        self._thread.start()

    def submit(self, frame: np.ndarray) -> Future | None:
        future = Future()
        try:
            self._jobs.put_nowait((frame.copy(), future))
        except queue.Full:
            return None
        return future

    @staticmethod
    def _create_session():
        model_path = resource_path(_MODEL_PATH)
        logger.info("EllSeg pupil: loading %s (CPU, low-rate worker)", model_path)
        onnxruntime.disable_telemetry_events()
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_mem_pattern = False
        return onnxruntime.InferenceSession(
            model_path,
            options,
            providers=["CPUExecutionProvider"],
        )

    def _worker(self):
        session = None
        input_name = None
        while True:
            frame, future = self._jobs.get()
            if future.cancelled():
                continue
            try:
                if session is None:
                    session = self._create_session()
                    input_name = session.get_inputs()[0].name
                tensor, transform = _preprocess(frame)
                logits = session.run(None, {input_name: tensor})[0]
                future.set_result(_measurement_from_logits(logits, transform))
            except Exception as exc:
                future.set_exception(exc)


_RUNTIME: _EllSegRuntime | None = None
_RUNTIME_LOCK = threading.Lock()


def _get_runtime() -> _EllSegRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        with _RUNTIME_LOCK:
            if _RUNTIME is None:
                _RUNTIME = _EllSegRuntime()
    return _RUNTIME


class EllSegPupilDetector:
    def submit(self, frame: np.ndarray) -> Future | None:
        return _get_runtime().submit(frame)


@dataclass(frozen=True)
class _FrameTransform:
    scale: float
    vertical_shift: float
    source_width: int
    source_height: int


def _preprocess(frame: np.ndarray) -> tuple[np.ndarray, _FrameTransform]:
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    source_height, source_width = gray.shape[:2]
    if source_height < 2 or source_width < 2:
        raise ValueError("eye frame is empty")

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
