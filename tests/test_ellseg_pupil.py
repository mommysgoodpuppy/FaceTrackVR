import cv2
import numpy as np
import onnxruntime
from unittest import mock

from ellseg_pupil import (
    _EllSegRuntime,
    _FrameTransform,
    _measurement_from_logits,
    _preprocess,
)


def test_preprocess_width_aligns_and_center_crops_square_bsb_frame():
    frame = np.tile(np.arange(400, dtype=np.uint8), (400, 1))

    tensor, transform = _preprocess(frame)

    assert tensor.shape == (1, 1, 240, 320)
    assert transform.scale == 0.8
    assert transform.vertical_shift == -40.0
    assert abs(float(tensor.mean())) < 1e-5
    assert abs(float(tensor.std()) - 1.0) < 1e-5


def test_fast_preprocess_uses_reduced_spatial_resolution():
    frame = np.tile(np.arange(400, dtype=np.uint8), (400, 1))

    tensor, transform = _preprocess(frame, input_size=(288, 224))

    assert tensor.shape == (1, 1, 224, 288)
    assert transform.scale == 0.72
    assert transform.vertical_shift == -32.0


def test_bsb_gamma_brightening_is_nonlinear_and_remains_standardized():
    ramp = np.tile(np.arange(256, dtype=np.uint8), (192, 1))

    brightened, _transform = _preprocess(ramp, gamma=0.8)
    neutral, _transform = _preprocess(ramp, gamma=1.0)

    assert brightened.shape == neutral.shape == (1, 1, 240, 320)
    assert abs(float(brightened.mean())) < 1e-5
    assert abs(float(brightened.std()) - 1.0) < 1e-5
    assert not np.allclose(brightened, neutral)


def test_measurement_uses_pupil_to_iris_ratio_to_cancel_ellipse_angle():
    labels = np.zeros((240, 320), dtype=np.uint8)
    cv2.ellipse(labels, (160, 120), (80, 40), 28, 0, 360, 1, -1)
    cv2.ellipse(labels, (160, 120), (40, 20), 28, 0, 360, 2, -1)
    logits = np.full((1, 3, 240, 320), -8.0, dtype=np.float32)
    for class_index in range(3):
        logits[0, class_index][labels == class_index] = 8.0

    measurement = _measurement_from_logits(
        logits,
        _FrameTransform(1.0, 0.0, 320, 240),
    )

    assert measurement is not None
    assert 0.48 < measurement.pupil_to_iris_ratio < 0.52
    assert 27.5 < measurement.equivalent_radius < 29.0


def test_ellseg_onnx_artifact_loads_and_has_expected_contract():
    session = onnxruntime.InferenceSession(
        "EyeTrackApp/Models/EllSeg_all.onnx",
        providers=["CPUExecutionProvider"],
    )

    output = session.run(
        None,
        {session.get_inputs()[0].name: np.zeros((1, 1, 240, 320), np.float32)},
    )[0]

    assert output.shape == (1, 3, 240, 320)

    fast_output = session.run(
        None,
        {session.get_inputs()[0].name: np.zeros((1, 1, 224, 288), np.float32)},
    )[0]
    assert fast_output.shape == (1, 3, 224, 288)


def _unstarted_runtime(*, use_gpu, high_rate=False):
    runtime = object.__new__(_EllSegRuntime)
    runtime._use_gpu = use_gpu
    runtime._high_rate = high_rate
    runtime._uses_directml = False
    return runtime


def test_normal_rate_uses_cpu_when_gpu_is_requested(monkeypatch):
    expected = mock.Mock()
    expected.get_providers.return_value = ["CPUExecutionProvider"]
    monkeypatch.setattr(
        "ellseg_pupil._create_webgpu_session",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("normal-rate tracking should not use WebGPU")
        ),
    )
    monkeypatch.setattr(
        "ellseg_pupil.create_inference_session",
        lambda *_args, **_kwargs: (expected, False),
    )

    runtime = _unstarted_runtime(use_gpu=True, high_rate=False)

    assert runtime._create_session() is expected
    assert runtime._uses_cpu is True


def test_high_rate_uses_webgpu_when_gpu_is_requested(monkeypatch):
    expected = object()
    monkeypatch.setattr("ellseg_pupil._create_webgpu_session", lambda _path: expected)
    runtime = _unstarted_runtime(use_gpu=True, high_rate=True)

    assert runtime._create_session() is expected


def test_force_webgpu_allows_normal_rate_profiling(monkeypatch):
    expected = object()
    monkeypatch.setenv("ELLSEG_FORCE_WEBGPU", "1")
    monkeypatch.setattr("ellseg_pupil._create_webgpu_session", lambda _path: expected)
    runtime = _unstarted_runtime(use_gpu=True, high_rate=False)

    assert runtime._create_session() is expected


def test_force_cpu_skips_webgpu(monkeypatch):
    expected = mock.Mock()
    expected.get_providers.return_value = ["CPUExecutionProvider"]
    monkeypatch.setenv("ELLSEG_FORCE_CPU", "1")
    monkeypatch.setattr(
        "ellseg_pupil._create_webgpu_session",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("WebGPU should be skipped when CPU is forced")
        ),
    )
    monkeypatch.setattr(
        "ellseg_pupil.create_inference_session",
        lambda *_args, **_kwargs: (expected, False),
    )

    runtime = _unstarted_runtime(use_gpu=True, high_rate=False)

    assert runtime._create_session() is expected
    assert runtime._uses_cpu is True


def test_normal_rate_falls_back_when_webgpu_is_unavailable(monkeypatch):
    expected = object()
    monkeypatch.setattr("ellseg_pupil._create_webgpu_session", lambda _path: None)
    monkeypatch.setattr(
        "ellseg_pupil.create_inference_session",
        lambda *_args, **_kwargs: (expected, False),
    )

    runtime = _unstarted_runtime(use_gpu=True, high_rate=False)

    assert runtime._create_session() is expected
