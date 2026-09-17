from unittest import mock

import numpy as np

from next_auxiliary import (
    ELLSEG_RATIO_CONSTRICTED,
    ELLSEG_RATIO_DILATED,
    NextAuxiliaryTracker,
)


class RadiusDetector:
    def run(self, gray):
        return 50, 50, gray, 10


def pupil_frame(radius):
    frame = np.full((100, 100, 3), 220, dtype=np.uint8)
    yy, xx = np.ogrid[:100, :100]
    frame[(xx - 50) ** 2 + (yy - 50) ** 2 <= radius**2] = 10
    return frame


def test_next_auxiliary_normalizes_measured_pupil_radius():
    detector = RadiusDetector()
    tracker = NextAuxiliaryTracker(
        detector=detector,
        rate_hz=1000,
        minimum_samples=4,
    )
    radii = [8, 8.5, 9, 9.5, 10, 10.5, 11, 12, 8]
    values = [tracker.update(pupil_frame(radius), now=i).pupil_dilation for i, radius in enumerate(radii)]

    assert values[7] > 0.5
    assert values[8] < values[7]


def test_next_auxiliary_is_rate_limited_and_preserves_feature_schema():
    detector = mock.Mock()
    detector.run.return_value = (50, 50, np.zeros((100, 100)), 10)
    tracker = NextAuxiliaryTracker(detector=detector, rate_hz=10)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    first = tracker.update(frame, now=1.0)
    second = tracker.update(frame, now=1.01)

    assert first.pupil_dilation == second.pupil_dilation
    assert first.expressions == {}
    detector.run.assert_called_once()


def test_next_auxiliary_debug_mode_submits_on_every_tracking_tick():
    detector = RadiusDetector()
    tracker = NextAuxiliaryTracker(detector=detector, rate_hz=1.0)
    frame = pupil_frame(10)

    tracker.update(frame, now=1.0)
    normal = tracker.update(frame, now=1.02)
    tracker.set_debug_rate(True)
    debug = tracker.update(frame, now=1.04)

    assert normal.sample_count == 1
    assert debug.sample_count == 2
    assert debug.debug_rate is True
    assert 24.0 < debug.model_rate_hz < 26.0


def test_ellseg_quality_mode_controls_normal_sampling_cadence():
    tracker = NextAuxiliaryTracker(detector=RadiusDetector())

    assert tracker._normal_rate_hz == 0.5
    assert tracker._interval == 2.0

    tracker.set_high_quality(True)
    assert tracker._normal_rate_hz == 1.0
    assert tracker._interval == 1.0

    tracker.set_high_quality(False)
    assert tracker._normal_rate_hz == 0.5
    assert tracker._interval == 2.0


def test_manual_dilation_range_maps_eighty_percent_to_full_output():
    tracker = NextAuxiliaryTracker(detector=RadiusDetector())
    tracker.set_dilation_output_range(0.0, 0.8)

    assert tracker._map_dilation_output(0.0) == 0.0
    assert tracker._map_dilation_output(0.4) == 0.5
    assert tracker._map_dilation_output(0.8) == 1.0
    assert tracker._map_dilation_output(1.0) == 1.0


def test_ellseg_uses_fixed_ratio_baseline_instead_of_rolling_history():
    tracker = NextAuxiliaryTracker(detector=RadiusDetector(), minimum_samples=4)
    midpoint = (ELLSEG_RATIO_CONSTRICTED + ELLSEG_RATIO_DILATED) / 2.0

    for _ in range(100):
        assert tracker._set_normalization_target(
            ELLSEG_RATIO_DILATED,
            50,
            50,
            (100, 100),
            ratio_based=True,
        )
        assert tracker._target == 1.0

    tracker._set_normalization_target(
        midpoint,
        50,
        50,
        (100, 100),
        ratio_based=True,
    )
    assert abs(tracker._target - 0.5) < 1e-9
    assert tracker._ratio_range == (
        ELLSEG_RATIO_CONSTRICTED,
        ELLSEG_RATIO_DILATED,
    )
