from eye import EyeInfoOrigin
from eye_processor import EyeProcessor


def _processor(origin, *, radius=12.0, width=30.0, height=20.0):
    processor = EyeProcessor.__new__(EyeProcessor)
    processor.current_algo = origin
    processor.radius = radius
    processor.pupil_width = width
    processor.pupil_height = height
    processor._pupil_geometry_source = None
    processor._pupil_geometry_label = ""
    return processor


def test_ransac_preserves_its_fitted_ellipse_for_legacy_dilation():
    processor = _processor(EyeInfoOrigin.RANSAC)
    processor._pupil_geometry_source = "ellipse"

    assert processor._prepare_pupil_axes_for_dilation() == "ellipse"
    assert (processor.pupil_width, processor.pupil_height) == (30.0, 20.0)


def test_hsf_rejects_search_radius_and_stale_ellipse_axes():
    processor = _processor(EyeInfoOrigin.HSF, radius=12.0)

    assert processor._prepare_pupil_axes_for_dilation() is None
    assert (processor.pupil_width, processor.pupil_height) == (0.0, 0.0)


def test_ahsf_preserves_its_detected_rectangle_as_a_labeled_proxy():
    processor = _processor(EyeInfoOrigin.HSF)
    processor._pupil_geometry_source = "rectangle"

    assert processor._prepare_pupil_axes_for_dilation() == "rectangle proxy"
    assert (processor.pupil_width, processor.pupil_height) == (30.0, 20.0)


def test_leap_rejects_stale_pupil_geometry():
    processor = _processor(EyeInfoOrigin.LEAP)

    assert processor._prepare_pupil_axes_for_dilation() is None
    assert (processor.pupil_width, processor.pupil_height) == (0.0, 0.0)
