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
            "frame_size": (400, 400),
            "radius_range": (15.0, 21.0),
            "sample_count": 12,
        },
    )

    annotated = widget._annotate_tracking_preview(image, eye_info)

    assert annotated.shape == (300, 300, 3)
    assert np.any(annotated[:, :, 0] != annotated[:, :, 1])
    assert list(widget._dilation_history) == [0.7]
    assert list(widget._radius_history) == [19.0]
