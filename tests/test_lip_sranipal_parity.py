import numpy as np
import pytest
from types import SimpleNamespace

from lip import (
    LipTracker,
    MouthBounceFilter,
    NewSmoothFilter,
    apply_global_response,
    apply_vrcft_parameter_adjustment,
    map_to_improved_v2,
    map_to_unified,
)
from lip_post_v2 import LIP_SHAPE_V2, LipPostprocessV2
from pyvrcft.expressions import UnifiedTrackingData, compute_outputs
from pyvrcft.client import VRCFTClient
from config import EyeTrackSettingsConfig
from settings.modules.LipSettingsModule import (
    LIP_PROCESSING_FIELDS,
    lip_processing_defaults,
)


def _weights(**overrides):
    values = {name: 0.0 for name in LIP_SHAPE_V2}
    values.update(overrides)
    return values


def _raw(**channels):
    values = np.zeros(30, dtype=np.float32)
    values[0] = 1.0
    for index, value in channels.items():
        values[int(index)] = value
    return values


def test_postprocess_initial_state_matches_compatibility_behavior():
    post = LipPostprocessV2()
    assert post.norm == pytest.approx([0.5, 0.5, 0.7] + [0.5] * 29)
    assert post.th == pytest.approx([0.5] * 4)
    assert post.ths == pytest.approx([0.5] * 4)
    assert post.thacc == pytest.approx([0.5] * 4)


def test_tracking_loss_resets_normalizers_to_compatibility_defaults():
    post = LipPostprocessV2()
    post.norm = [2.0] * 32
    lost = np.zeros(30, dtype=np.float32)
    for _ in range(27):
        post(lost)
    assert post.norm == pytest.approx([0.5, 0.5, 0.7] + [0.5] * 29)


def test_first_tracked_output_uses_compatible_normalization_and_smoothing():
    post = LipPostprocessV2()
    raw = _raw(**{"1": 0.4, "2": 0.8, "3": 0.3, "5": 0.45, "6": 0.3})
    for _ in range(25):
        assert not np.any(post(raw))
    out = post(raw)
    # First active EMA rate is .25 / (.25 + .5) = 1/3.
    assert out[0] == pytest.approx(((0.5 - 0.4) / 0.5) / 0.5 / 3.0)
    assert out[2] == pytest.approx((0.8 - 0.5) / 0.7 / 3.0)
    assert out[3] == pytest.approx(0.3 / 0.5 / 3.0)
    assert out[5] == pytest.approx(((0.5 - 0.45) / 0.5) / 0.5 / 3.0)
    assert out[7] == pytest.approx(((0.5 - 0.3) / 0.5) / 0.5 / 3.0)


def test_threshold_learning_uses_current_frame_and_absolute_delta():
    post = LipPostprocessV2()
    steady = _raw(**{"1": 0.5, "2": 0.5, "5": 0.5, "6": 0.5})
    for _ in range(162):
        post(steady)
    changed = steady.copy()
    changed[1] = 0.6
    post(changed)
    assert post.th[0] > 0.5


def test_tongue_up_branch_reuses_point_two_gate():
    post = LipPostprocessV2()
    post.track_hold = 26
    raw = _raw(**{"3": 0.1, "24": 0.15, "27": 0.21})
    out = post(raw)
    # Step1 normalizes to .3. TongueUp remains .21 (the >=27 normalizer has a
    # .502 activation floor), so the branch input is .21 + .12 = .33.
    # First-frame output EMA is 1/3.
    assert out[32] == pytest.approx(0.11)


def test_unified_mapping_clears_zeroes_and_matches_compatibility_emulation():
    mapped = map_to_unified(
        _weights(MouthSmileLeft=0.4, MouthSmileRight=0.3, MouthSadRight=0.2)
    )
    assert mapped["JawOpen"] == 0.0  # zero must be emitted to clear stale data
    assert mapped["CheekSquintLeft"] == pytest.approx(0.4)
    assert mapped["CheekSquintRight"] == pytest.approx(0.3)
    assert mapped["MouthDimpleLeft"] == pytest.approx(0.4)
    assert mapped["MouthDimpleRight"] == pytest.approx(0.3)
    # Compatibility behavior intentionally preserves this asymmetry.
    assert mapped["MouthStretchLeft"] == pytest.approx(0.2)
    assert mapped["MouthStretchRight"] == pytest.approx(0.2)


def test_unified_mapping_preserves_unclamped_compatibility_arithmetic():
    mapped = map_to_unified(
        _weights(
            JawOpen=0.8,
            MouthApeShape=0.7,
            MouthUpperUpLeft=0.1,
            MouthUpperOverturn=0.4,
            MouthLowerOverlay=0.2,
            MouthUpperInside=0.6,
        )
    )
    assert mapped["JawOpen"] == pytest.approx(1.5)
    assert mapped["MouthUpperUpLeft"] == pytest.approx(-0.3)
    assert mapped["MouthRaiserUpper"] == pytest.approx(-0.4)


def test_optional_etvr_mouth_smoothing_is_bypassed_for_compatibility():
    tracker = LipTracker(SimpleNamespace(gui_lip_use_etvr_smoothing=False))
    tracker._lip_smoother = lambda values: values * 0.5
    values = np.arange(len(LIP_SHAPE_V2), dtype=np.float32)
    assert np.array_equal(tracker._apply_optional_smoothing(values), values)


def test_mouth_bounce_zero_mix_is_exact_source_output():
    bounce = MouthBounceFilter()
    bounce({"JawOpen": 0.0}, now=1.0, mix=0.0)
    for index in range(1, 12):
        target = 1.0 if index < 6 else 0.25
        result = bounce({"JawOpen": target}, now=1.0 + index / 60.0, mix=0.0)
        assert result["JawOpen"] == target


def test_mouth_bounce_underdamped_response_overshoots_and_settles():
    bounce = MouthBounceFilter()
    bounce({"JawOpen": 0.0}, now=1.0, mix=100.0)
    samples = []
    internal = []
    for index in range(1, 121):
        samples.append(bounce(
            {"JawOpen": 1.0}, now=1.0 + index / 120.0,
            response_hz=4.0, damping=0.35, mix=100.0,
        )["JawOpen"])
        internal.append(bounce.position["JawOpen"])
    assert max(internal) > 1.05  # the physical spring still overshoots
    assert all(0.0 <= value <= 1.0 for value in samples)
    # The whole outward lobe is represented as a smooth return. It neither
    # disappears into a clipped plateau nor acquires a corner at the cap.
    assert len(set(samples[10:15])) == 5
    assert min(samples[10:25]) < 0.95
    assert samples[-1] == pytest.approx(1.0, abs=0.001)


def test_mouth_bounce_fold_is_smooth_at_cap_and_exact_at_equilibrium():
    epsilon = 1e-5
    below = MouthBounceFilter._visible_position(
        "JawOpen", target=1.0, spring_position=1.0 - epsilon
    )
    cap = MouthBounceFilter._visible_position(
        "JawOpen", target=1.0, spring_position=1.0
    )
    above = MouthBounceFilter._visible_position(
        "JawOpen", target=1.0, spring_position=1.0 + epsilon
    )
    assert below == pytest.approx(above, abs=1e-12)
    assert cap == pytest.approx(1.0)
    # Both one-sided slopes tend to zero instead of changing direction at a
    # sharp triangle-wave cusp.
    assert (cap - below) / epsilon < 1e-4
    assert (above - cap) / epsilon > -1e-4
    assert MouthBounceFilter._visible_position(
        "JawOpen", target=0.63, spring_position=0.63
    ) == pytest.approx(0.63)


@pytest.mark.parametrize(
    ("name", "target"),
    (
        ("JawOpen", 1e-9),
        ("JawOpen", 1.0 - 1e-9),
        ("v2/JawX", -1.0 + 1e-9),
        ("v2/JawX", 1.0 - 1e-9),
    ),
)
def test_mouth_bounce_handles_targets_that_fold_to_float_endpoints(name, target):
    bounce = MouthBounceFilter()
    bounce({name: 0.5}, now=1.0, mix=10.0)

    result = bounce({name: target}, now=1.0 + 1.0 / 60.0, mix=10.0)

    low, high = bounce._bounds(name)
    assert np.isfinite(result[name])
    assert low <= result[name] <= high


def test_mouth_bounce_preserves_source_overflow():
    bounce = MouthBounceFilter()
    result = bounce({"MouthUpperUpLeft": -0.3}, now=1.0, mix=100.0)
    assert result["MouthUpperUpLeft"] == pytest.approx(-0.3)
    assert "MouthUpperUpLeft" not in bounce.position


def test_new_smooth_then_bounce_preserves_visible_spring_motion():
    smooth = NewSmoothFilter()
    bounce = MouthBounceFilter()
    settings = dict(deadband=0.006)
    initial = smooth({"v2/JawOpen": 0.0}, now=1.0, **settings)
    bounce(initial, now=1.0, response_hz=4.0, damping=0.35, mix=100.0)

    # New Smooth passes non-tongue expressions straight through, so the spring
    # sees the real low-latency step and its transient remains observable.
    first = smooth({"v2/JawOpen": 1.0}, now=1.01, **settings)
    animated = bounce(
        first, now=1.01, response_hz=4.0, damping=0.35, mix=100.0
    )
    assert 0.0 < animated["v2/JawOpen"] < 1.0

    samples = []
    internal = []
    for index in range(1, 121):
        accepted = smooth(
            {"v2/JawOpen": 1.0}, now=1.01 + index / 120.0, **settings
        )
        samples.append(bounce(
            accepted, now=1.01 + index / 120.0,
            response_hz=4.0, damping=0.35, mix=100.0,
        )["v2/JawOpen"])
        internal.append(bounce.position["v2/JawOpen"])
    assert max(internal) > 1.05
    assert all(0.0 <= value <= 1.0 for value in samples)
    assert min(samples[10:35]) < 0.95


def test_mouth_bounce_exclusion_keeps_new_smooth_tongue_untouched():
    bounce = MouthBounceFilter()
    excluded = frozenset(("v2/TongueX", "v2/TongueY", "v2/TongueOut"))
    bounce(
        {"v2/JawOpen": 0.0, "v2/TongueY": 0.0}, now=1.0,
        response_hz=4.0, damping=0.35, mix=100.0, exclude=excluded,
    )
    result = bounce(
        {"v2/JawOpen": 1.0, "v2/TongueY": 0.4}, now=1.01,
        response_hz=4.0, damping=0.35, mix=100.0, exclude=excluded,
    )
    assert 0.0 < result["v2/JawOpen"] < 1.0
    assert result["v2/TongueY"] == 0.4


def test_disabled_mouth_bounce_returns_original_mapping_object():
    tracker = LipTracker(SimpleNamespace(gui_lip_bounce_enable=False))
    values = {"JawOpen": 0.5}
    assert tracker._apply_optional_bounce(values) is values


def test_global_response_deadzone_saturation_and_signed_direction():
    result = apply_global_response(
        {"small": 0.1, "middle": 0.5, "full": 0.8, "signed": -0.5},
        minimum=0.2, maximum=0.8, curve=1.0,
    )
    assert result["small"] == 0.0
    assert result["middle"] == pytest.approx(0.5)
    assert result["full"] == 1.0
    assert result["signed"] == pytest.approx(-0.5)


def test_global_response_curve_controls_sensitivity():
    source = {"Smile": 0.25}
    exaggerated = apply_global_response(source, 0.0, 1.0, 0.5)
    dampened = apply_global_response(source, 0.0, 1.0, 2.0)
    assert exaggerated["Smile"] == pytest.approx(0.5)
    assert dampened["Smile"] == pytest.approx(0.0625)


def test_disabled_global_response_returns_original_mapping_object():
    tracker = LipTracker(SimpleNamespace(gui_lip_response_enable=False))
    values = {"JawOpen": 0.5}
    assert tracker._apply_optional_global_response(values) is values


def test_optional_etvr_mouth_smoothing_uses_shared_one_euro_stage():
    tracker = LipTracker(
        SimpleNamespace(
            gui_lip_use_etvr_smoothing=True,
            gui_min_cutoff="0.003162",
            gui_speed_coefficient="1.5250",
        )
    )

    class HalfFilter:
        min_cutoff = np.zeros(len(LIP_SHAPE_V2), dtype=np.float32)
        beta = np.zeros(len(LIP_SHAPE_V2), dtype=np.float32)

        def __call__(self, values):
            return values * 0.5

    tracker._lip_smoother = HalfFilter()
    values = np.ones(len(LIP_SHAPE_V2), dtype=np.float32)
    assert tracker._apply_optional_smoothing(values) == pytest.approx(0.5)


def test_vrcft_parameter_adjustment_matches_unclamped_formula():
    source = {
        "JawOpen": 0.5,
        "MouthCornerPullLeft": 0.1,
        "MouthCornerPullRight": 1.1,
        "TongueOut": 0.7,
    }
    adjusted = apply_vrcft_parameter_adjustment(
        source,
        {
            "jaw_open": (0.2, 0.8),
            "smile": (0.2, 0.6),
            "tongue_out": (0.0, 1.0),
        },
    )
    assert adjusted["JawOpen"] == pytest.approx(0.5)
    assert adjusted["MouthCornerPullLeft"] == pytest.approx(-0.25)
    assert adjusted["MouthCornerPullRight"] == pytest.approx(2.25)
    assert adjusted["TongueOut"] == pytest.approx(0.7)
    assert source["MouthCornerPullLeft"] == 0.1  # input is not mutated


def test_vrcft_adjustment_disabled_preserves_exact_mapping_object():
    tracker = LipTracker(SimpleNamespace(gui_lip_vrcft_adjust=False))
    mapped = {"JawOpen": 0.37}
    assert tracker._apply_optional_vrcft_adjustment(mapped) is mapped


def test_preview_buffers_are_copies_and_do_not_change_model_input():
    class Session:
        def __init__(self):
            self.x = None

        def run(self, _outputs, feeds):
            self.x = feeds["input"].copy()
            return [np.zeros((1, 30), dtype=np.float32)]

    tracker = LipTracker(SimpleNamespace(gui_lip_preview=True))
    tracker.session = Session()
    yuyv = np.arange(400 * 400 * 2, dtype=np.uint8).reshape(400, 400, 2)
    original = yuyv.copy()
    tracker._infer(yuyv)

    assert np.array_equal(yuyv, original)
    assert np.array_equal(tracker._raw_preview, original.reshape(400, 800))
    assert tracker._model_preview.shape == (400, 800)
    assert tracker.session.x.shape == (1, 2, 100, 100)
    tracker._raw_preview[:] = 0
    assert np.array_equal(yuyv, original)


def test_mouth_model_accepts_generic_bgr_camera_frames():
    class Session:
        def __init__(self):
            self.x = None

        def run(self, _outputs, feeds):
            self.x = feeds["input"]
            return [np.zeros((1, 30), dtype=np.float32)]

    tracker = LipTracker(SimpleNamespace(gui_lip_preview=False))
    tracker.session = Session()
    tracker._infer(np.zeros((400, 800, 3), dtype=np.uint8))

    assert tracker.session.x.shape == (1, 2, 100, 100)


def test_model_contract_uses_declared_tensor_names():
    class Tensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape
            self.type = "tensor(float)"

    class Session:
        def get_inputs(self):
            return [Tensor("camera_tensor", [1, 2, 100, 100])]

        def get_outputs(self):
            return [Tensor("face_signals", [1, 30])]

    assert LipTracker._validate_model_contract(Session()) == (
        "camera_tensor",
        "face_signals",
    )


@pytest.mark.parametrize(
    ("input_shape", "output_shape", "message"),
    [
        ([1, 1, 100, 100], [1, 30], "input shape"),
        ([1, 2, 100, 100], [1, 29], "output shape"),
    ],
)
def test_model_contract_rejects_incompatible_shapes(
    input_shape, output_shape, message
):
    class Tensor:
        type = "tensor(float)"

        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class Session:
        def get_inputs(self):
            return [Tensor("input", input_shape)]

        def get_outputs(self):
            return [Tensor("output", output_shape)]

    with pytest.raises(ValueError, match=message):
        LipTracker._validate_model_contract(Session())


def test_face_model_path_must_be_explicit(tmp_path):
    tracker = LipTracker(SimpleNamespace(gui_lip_onnx_path=""))
    with pytest.raises(FileNotFoundError, match="No mouth-tracking model"):
        tracker._model_path()

    model = tmp_path / "compatible.onnx"
    model.touch()
    tracker.config.gui_lip_onnx_path = str(model)
    assert tracker._model_path() == str(model.resolve())


def test_mouth_device_must_be_explicit():
    tracker = LipTracker(SimpleNamespace(gui_lip_device=""))
    assert tracker._device() is None

    tracker.config.gui_lip_device = "  /dev/video7  "
    assert tracker._device() == "/dev/video7"


def test_preview_capture_is_bit_exactly_observational():
    class Session:
        def __init__(self):
            self.x = None

        def run(self, _outputs, feeds):
            self.x = feeds["input"].copy()
            return [np.zeros((1, 30), dtype=np.float32)]

    yuyv = np.random.default_rng(42).integers(
        0, 256, size=(400, 400, 2), dtype=np.uint8
    )
    off = LipTracker(SimpleNamespace(gui_lip_preview=False))
    on = LipTracker(SimpleNamespace(gui_lip_preview=True))
    off.session, on.session = Session(), Session()
    off._infer(yuyv)
    on._infer(yuyv)
    assert np.array_equal(off.session.x, on.session.x)


def test_improved_v2_mapping_bypasses_noisy_compatibility_expansion():
    weights = _weights(
        JawOpen=0.4,
        MouthApeShape=0.3,
        MouthUpperOverturn=0.2,
        MouthLowerOverturn=0.6,
        MouthUpperUpRight=0.1,
        MouthUpperUpLeft=0.5,
        MouthLowerDownRight=0.4,
        MouthLowerDownLeft=0.8,
        MouthSmileRight=0.7,
        MouthSadRight=0.2,
    )
    compatibility = map_to_unified(weights)
    improved = map_to_improved_v2(weights)

    assert compatibility["JawOpen"] == pytest.approx(0.7)
    assert improved["v2/JawOpen"] == pytest.approx(0.4)
    assert improved["v2/MouthClosed"] == pytest.approx(0.3)
    assert improved["v2/LipFunnel"] == pytest.approx(0.4)
    assert improved["v2/MouthUpperUpRight"] == 0.0
    assert improved["v2/MouthUpperUpLeft"] == pytest.approx(0.3)
    assert improved["v2/MouthLowerDown"] == pytest.approx(0.1)
    assert improved["v2/SmileFrownRight"] == pytest.approx(0.5)
    assert improved["v2/MouthStretchRight"] == pytest.approx(0.2)


def test_improved_v2_optional_jaw_boost_matches_vrcft_formula():
    weights = _weights(JawOpen=0.45, MouthApeShape=0.3)
    normal = map_to_improved_v2(weights)
    boosted = map_to_improved_v2(weights, boost_jaw_open=True)
    assert normal["v2/JawOpen"] == pytest.approx(0.45)
    assert boosted["v2/JawOpen"] == pytest.approx(0.75)
    assert boosted["v2/MouthClosed"] == pytest.approx(0.3)


def test_improved_v2_mapping_preserves_independent_mouth_stretch_sides():
    improved = map_to_improved_v2(
        _weights(MouthSadRight=0.25, MouthSadLeft=0.75)
    )
    assert improved["v2/MouthStretchRight"] == pytest.approx(0.25)
    assert improved["v2/MouthStretchLeft"] == pytest.approx(0.75)


def test_improved_values_override_only_compact_v2_output_layer():
    data = UnifiedTrackingData()
    data.shapes["JawOpen"] = 0.8
    data._lip_v2_overrides = {
        "v2/JawOpen": 0.35,
        "v2/LipFunnel": 0.2,
    }
    outputs, _model = compute_outputs(data)
    assert outputs["v2/JawOpen"] == pytest.approx(0.35)
    assert outputs["v2/LipFunnel"] == pytest.approx(0.2)
    # Unrelated eye and expression outputs still use the regular pipeline.
    assert outputs["v2/EyeOpen"] == pytest.approx(1.0)
    assert outputs["v2/CheekPuffSuck"] == pytest.approx(0.0)


def test_improved_sender_transmits_immediately_and_deduplicates_next_frame():
    class Socket:
        def __init__(self):
            self.packets = []

        def sendto(self, packet, destination):
            self.packets.append((packet, destination))

    client = VRCFTClient(force_relevant=True)
    client._send_sock = Socket()
    assert client.send_immediate({"v2/JawOpen": 0.42}) == 1
    assert len(client._send_sock.packets) == 1
    # The OutputSlot is shared with the periodic loop, so an unchanged frame
    # cannot be sent again on either path.
    assert client.send_immediate({"v2/JawOpen": 0.42}) == 0
    assert len(client._send_sock.packets) == 1


def test_new_smooth_limits_impossible_tongue_velocity_without_waiting():
    smooth = NewSmoothFilter()
    assert smooth(
        {"v2/TongueX": 0.0, "v2/TongueY": 0.0, "v2/TongueOut": 0.0},
        now=1.0,
    )["v2/TongueX"] == 0.0
    moved = smooth(
        {"v2/TongueX": 1.0, "v2/TongueY": 0.0, "v2/TongueOut": 0.0},
        now=1.01, tongue_speed=4.0, deadband=0.0,
    )
    assert moved["v2/TongueX"] == pytest.approx(0.04)


@pytest.mark.parametrize(
    ("protrusion", "expected_y"),
    ((0.0, 0.35), (0.5, 0.675), (1.0, 1.0)),
)
def test_new_smooth_tongue_vertical_range_tracks_protrusion(
    protrusion, expected_y
):
    smooth = NewSmoothFilter()
    result = smooth(
        {
            "v2/TongueX": 0.0,
            "v2/TongueY": 1.0,
            "v2/TongueOut": protrusion,
        },
        now=1.0,
        tongue_retracted_range=0.35,
        deadband=0.0,
    )
    assert result["v2/TongueY"] == pytest.approx(expected_y)


def test_new_smooth_tongue_range_expands_continuously_with_protrusion():
    smooth = NewSmoothFilter()
    values = {
        "v2/TongueX": 0.0,
        "v2/TongueY": 1.0,
        "v2/TongueOut": 0.0,
    }
    first = smooth(values, now=1.0, tongue_retracted_range=0.35, deadband=0.0)
    values["v2/TongueOut"] = 1.0
    moved = smooth(
        values,
        now=1.01,
        tongue_speed=4.0,
        tongue_retracted_range=0.35,
        deadband=0.0,
    )
    assert first["v2/TongueY"] == pytest.approx(0.35)
    assert moved["v2/TongueOut"] == pytest.approx(0.04)
    assert moved["v2/TongueY"] == pytest.approx(0.376)


def test_new_smooth_low_passes_small_tongue_protrusion_chatter():
    smooth = NewSmoothFilter()
    tongue = {"v2/TongueOut": 0.5}
    assert smooth(tongue, now=1.0)["v2/TongueOut"] == pytest.approx(0.5)
    tongue["v2/TongueOut"] = 0.6
    moved = smooth(
        tongue,
        now=1.01,
        tongue_speed=100.0,
        tongue_protrusion_hz=12.0,
    )["v2/TongueOut"]
    assert 0.5 < moved < 0.6


def test_new_smooth_protrusion_filter_is_frame_rate_independent():
    slow = NewSmoothFilter()
    fast = NewSmoothFilter()
    slow({"v2/TongueOut": 0.0}, now=1.0)
    fast({"v2/TongueOut": 0.0}, now=1.0)
    slow_value = slow(
        {"v2/TongueOut": 1.0}, now=1.02,
        tongue_speed=100.0, tongue_protrusion_hz=12.0,
    )["v2/TongueOut"]
    fast({"v2/TongueOut": 1.0}, now=1.01,
         tongue_speed=100.0, tongue_protrusion_hz=12.0)
    fast_value = fast(
        {"v2/TongueOut": 1.0}, now=1.02,
        tongue_speed=100.0, tongue_protrusion_hz=12.0,
    )["v2/TongueOut"]
    assert fast_value == pytest.approx(slow_value)


def test_new_smooth_retracts_before_opposite_tongue_direction():
    smooth = NewSmoothFilter()
    up = {"v2/TongueX": 0.0, "v2/TongueY": 1.0}
    down = {"v2/TongueX": 0.0, "v2/TongueY": -1.0}
    assert smooth(up, now=1.0, deadband=0.0)["v2/TongueY"] == pytest.approx(1.0)
    first = smooth(
        down, now=1.01, tongue_speed=4.0,
        tongue_reversal_frames=1, deadband=0.0
    )
    assert first["v2/TongueY"] == pytest.approx(0.96)
    assert abs(first["v2/TongueX"]) < 1e-6

    result = first
    for index in range(1, 27):
        result = smooth(
            down, now=1.01 + index * 0.01, tongue_speed=4.0,
            tongue_reversal_frames=1, deadband=0.0
        )
    assert result["v2/TongueY"] < 0.0
    assert abs(result["v2/TongueX"]) < 1e-6


def test_new_smooth_rejects_one_frame_tongue_direction_flicker():
    smooth = NewSmoothFilter()
    up = {"v2/TongueX": 0.0, "v2/TongueY": 0.8}
    down = {"v2/TongueX": 0.0, "v2/TongueY": -0.8}
    smooth(up, now=1.0, deadband=0.0)

    flicker = smooth(
        down, now=1.01, tongue_speed=4.0, deadband=0.0
    )
    assert flicker["v2/TongueY"] == pytest.approx(0.8)

    resumed = smooth(
        up, now=1.02, tongue_speed=4.0, deadband=0.0
    )
    assert resumed["v2/TongueY"] == pytest.approx(0.8)

    # A persistent reversal is accepted on its second frame and begins a
    # continuous physical retraction rather than flipping direction.
    held = smooth(down, now=1.03, tongue_speed=4.0, deadband=0.0)
    retracting = smooth(down, now=1.04, tongue_speed=4.0, deadband=0.0)
    assert held["v2/TongueY"] == pytest.approx(0.8)
    assert retracting["v2/TongueY"] == pytest.approx(0.76)


def test_new_smooth_acquires_direction_directly_when_leaving_neutral():
    smooth = NewSmoothFilter()
    neutral = {"v2/TongueX": 0.0, "v2/TongueY": 0.0}
    up = {"v2/TongueX": 0.0, "v2/TongueY": 1.0}
    smooth(neutral, now=1.0, deadband=0.0)
    result = smooth(up, now=1.01, tongue_speed=4.0, deadband=0.0)
    assert result["v2/TongueY"] == pytest.approx(0.04)
    assert abs(result["v2/TongueX"]) < 1e-6


def test_new_smooth_non_tongue_values_are_exact_direct_passthrough():
    smooth = NewSmoothFilter()
    samples = (0.1, 0.9, 0.1, 0.62, 0.0)
    for index, target in enumerate(samples):
        result = smooth({"v2/JawOpen": target}, now=1.0 + index * 0.01)
        assert result["v2/JawOpen"] == target


def test_legacy_improved_toggle_migrates_to_direct_selector_mode():
    config = EyeTrackSettingsConfig.model_validate(
        {"gui_lip_improved_output": True}
    )
    assert config.gui_lip_output_mode == "direct"


def test_face_processing_defaults_capture_tested_new_smooth_setup():
    defaults = lip_processing_defaults()
    assert defaults == {
        field: getattr(EyeTrackSettingsConfig(), field)
        for field in LIP_PROCESSING_FIELDS
    }
    assert defaults["gui_lip_output_mode"] == "new_smooth"
    assert defaults["gui_lip_smooth_tongue_stability"] == "balanced"
    assert defaults["gui_lip_smooth_boost_jaw_open"] is True
    assert defaults["gui_lip_smooth_tongue_retracted_range"] == pytest.approx(0.35)
    assert defaults["gui_lip_use_etvr_smoothing"] is True
    assert defaults["gui_lip_response_enable"] is True
    assert defaults["gui_lip_response_min"] == pytest.approx(0.0)
    assert defaults["gui_lip_response_max"] == pytest.approx(1.0)
    assert defaults["gui_lip_response_curve"] == pytest.approx(1.0)
    assert defaults["gui_lip_bounce_enable"] is True
    assert defaults["gui_lip_bounce_response_hz"] == pytest.approx(5.0)
    assert defaults["gui_lip_bounce_damping"] == pytest.approx(0.6)
    assert defaults["gui_lip_bounce_mix"] == pytest.approx(10.0)


def test_face_processing_reset_does_not_own_hardware_configuration():
    assert "gui_lip_enable" not in LIP_PROCESSING_FIELDS
    assert "gui_lip_device" not in LIP_PROCESSING_FIELDS
    assert "gui_lip_onnx_path" not in LIP_PROCESSING_FIELDS
