import pytest

from lip import _with_legacy_lip, map_to_unified
from lip_post_v2 import LIP_SHAPE_V2
from pyvrcft.avatar import AvatarInfo, AvatarParameter
from pyvrcft.client import VRCFTClient
from pyvrcft.expressions import (
    SRANIPAL_MERGED_SHAPES,
    ShapeWeights,
    UnifiedTrackingData,
    compute_legacy_lip_outputs,
    compute_legacy_outputs,
    compute_sranipal_lip_shapes,
)


def _shapes(**values):
    w = ShapeWeights()
    for name, value in values.items():
        w[name] = value
    return w


def test_every_sranipal_shape_and_merged_shape_is_emitted():
    out = compute_legacy_lip_outputs(ShapeWeights())
    assert set(LIP_SHAPE_V2) <= set(out)
    assert set(SRANIPAL_MERGED_SHAPES) <= set(out)
    # Spot-check names from the VRCFT v4 lip parameter documentation.
    for name in ("JawX", "SmileSadRight", "TongueSteps", "PuffSuck",
                 "MouthUpperUpRightApe", "MouthLowerDownInside", "PuffOverturn"):
        assert name in out


def test_unified_sranipal_mapper_formulas():
    w = _shapes(JawOpen=0.9, MouthClosed=0.3, TongueOut=0.75,
                MouthCornerPullRight=0.5, MouthDimpleRight=0.2,
                MouthFrownRight=0.6, MouthFrownLeft=0.2)
    sr = compute_sranipal_lip_shapes(w)
    assert sr["JawOpen"] == pytest.approx(0.6)
    assert sr["MouthApeShape"] == pytest.approx(0.3)
    assert sr["TongueLongStep1"] == pytest.approx(1.0)
    assert sr["TongueLongStep2"] == pytest.approx(0.5)
    assert sr["MouthSmileRight"] == pytest.approx(0.4)
    # Sad uses the averaged frown, minus the transformed smile.
    assert sr["MouthSadRight"] == pytest.approx(0.0)
    assert sr["MouthSadLeft"] == pytest.approx(0.4)


def test_merged_shape_kinds():
    w = _shapes(TongueOut=0.75, JawRight=0.2, JawLeft=0.5,
                MouthUpperRight=0.8, MouthLowerRight=0.2, MouthUpperLeft=0.1)
    out = compute_legacy_lip_outputs(w)
    assert out["JawX"] == pytest.approx(-0.3)
    assert out["TongueSteps"] == pytest.approx(0.5)  # 1.0 + 0.5 - 1
    assert out["MouthX"] == pytest.approx(0.7)  # max(right) - max(left)


def test_direct_sranipal_shapes_round_trip_through_unified():
    w = {name: 0.0 for name in LIP_SHAPE_V2}
    w.update(JawOpen=0.4, MouthApeShape=0.2, MouthPout=0.3, CheekSuck=0.1,
             MouthUpperInside=0.25, MouthLowerOverlay=0.15, TongueUp=0.6,
             JawLeft=0.35, MouthLowerRight=0.45)
    sr = compute_sranipal_lip_shapes(_shapes(**map_to_unified(w)))
    for name in ("JawOpen", "MouthApeShape", "MouthPout", "CheekSuck",
                 "MouthUpperInside", "MouthLowerOverlay", "TongueUp",
                 "JawLeft", "MouthLowerRight"):
        assert sr[name] == pytest.approx(w[name]), name


def test_legacy_outputs_include_lip_and_eye():
    data = UnifiedTrackingData()
    data.shapes["JawOpen"] = 0.5
    out = compute_legacy_outputs(data)
    assert out["JawOpen"] == pytest.approx(0.5)
    assert "EyesX" in out


def _client_for(names):
    client = VRCFTClient(use_oscquery=False)
    client.avatar = AvatarInfo(id="avtr_test", parameters=[
        AvatarParameter(name.rsplit("/", 1)[-1], "/avatar/parameters/" + name, float)
        for name in names
    ])
    return client


def test_sranipal_avatar_resolves_bare_lip_names():
    client = _client_for(["JawOpen", "SmileSadLeft"])
    data = UnifiedTrackingData()
    data.shapes["JawOpen"] = 0.5
    client.update_tracking(data)
    assert [s.address for s in client._slots["JawOpen"]] == ["/avatar/parameters/JawOpen"]
    assert client._slots["SmileSadLeft"]


def test_unified_avatar_does_not_capture_legacy_names():
    client = _client_for(["FT/v2/JawOpen", "FT/v2/TongueX"])
    client.update_tracking(UnifiedTrackingData())
    assert client._slots["JawOpen"] == []
    assert client._slots["TongueX"] == []
    assert client._slots["v2/JawOpen"]


def test_direct_mode_packet_carries_legacy_lip_values():
    shapes = _shapes(JawOpen=0.7)
    packet = _with_legacy_lip({"v2/JawOpen": 0.65}, shapes)
    assert packet["v2/JawOpen"] == 0.65
    assert packet["JawOpen"] == pytest.approx(0.7)
    assert _with_legacy_lip({}, shapes) is None
