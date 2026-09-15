from collections import deque

import numpy as np

from camera_widget import CameraWidget
from eye import EyeId, EyeInfo, EyeInfoOrigin


def test_tracking_preview_annotates_pupil_and_diagnostics():
    widget = CameraWidget.__new__(CameraWidget)
    widget.eye_id = EyeId.LEFT
    widget._dilation_history = deque(maxlen=120)
    widget._radius_history = deque(maxlen=120)
    widget._last_aux_sample_count = -1
    image = np.full((300, 300), 90, dtype=np.uint8)
    eye_info = EyeInfo(
        EyeInfoOrigin.NEXT,
        x=0.25,
        y=-0.2,
        pupil_dilation=0.7,
        blink=0.8,
        avg_velocity=0.0,
        eyebrow=0.4,
        squeeze=0.2,
        auxiliary_diagnostics={
            "pupil_locked": True,
            "pupil_center": (200.0, 180.0),
            "pupil_axes": (42.0, 36.0),
            "pupil_angle_degrees": 12.0,
            "pupil_radius_px": 19.0,
            "pupil_to_iris_ratio": 0.48,
            "iris_center": (200.0, 180.0),
            "iris_axes": (100.0, 80.0),
            "iris_angle_degrees": 0.0,
            "frame_size": (400, 400),
            "ratio_range": (0.42, 0.56),
            "sample_count": 12,
            "detector_name": "EllSeg",
            "pupil_dilation_raw": 0.6,
            "pupil_dilation_mapped": 0.7,
            "comparison_pupil_locked": True,
            "comparison_pupil_center": (120.0, 120.0),
            "comparison_pupil_axes": (30.0, 22.0),
            "comparison_pupil_angle_degrees": 5.0,
            "comparison_frame_size": (300, 300),
            "comparison_pupil_dilation_mapped": 0.55,
        },
    )

    annotated = widget._annotate_tracking_preview(image, eye_info)

    assert annotated.shape == (300, 300, 3)
    assert np.any(annotated[:, :, 0] != annotated[:, :, 1])
    assert np.any(
        (annotated[:, :, 0] > 180)
        & (annotated[:, :, 1] < 160)
        & (annotated[:, :, 2] > 220)
    )
    # The segmentation visualization is outline-only; the iris interior stays
    # as the unobscured grayscale camera image.
    assert tuple(annotated[135, 175]) == (90, 90, 90)
    assert list(widget._dilation_history) == [0.7]
    assert list(widget._radius_history) == [19.0]
