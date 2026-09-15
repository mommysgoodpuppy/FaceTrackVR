import cv2
import numpy as np
import onnxruntime

from ellseg_pupil import (
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
