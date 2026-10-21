import socket
import time

import pytest

from pyvrcft import osc
from pyvrcft.avatar import AvatarInfo, AvatarParameter
from pyvrcft.client import VRCFTClient
from pyvrcft.expressions import UNIFIED_EXPRESSIONS, UnifiedTrackingData
from pyvrcft.native_face import (
    ADDRESS,
    TRACKING_KEYS,
    compute_native_face_keys,
    encode_native_face,
)


def test_key_order_matches_vrchat_indices():
    assert len(TRACKING_KEYS) == len(set(TRACKING_KEYS)) == 102
    assert TRACKING_KEYS.index("EyeLookOutRight") == 0
    assert TRACKING_KEYS.index("EyeClosedRight") == 8
    assert TRACKING_KEYS.index("BrowPinchRight") == 18
    assert TRACKING_KEYS.index("JawOpen") == 36
    assert TRACKING_KEYS.index("LipSuckUpperRight") == 44
    assert TRACKING_KEYS.index("MouthUpperUpRight") == 58
    assert TRACKING_KEYS.index("MouthCornerPullRight") == 70
    assert TRACKING_KEYS.index("TongueOut") == 86
    assert TRACKING_KEYS[-1] == "NeckFlexLeft"


def test_non_eye_keys_are_unified_expressions_shapes():
    assert set(TRACKING_KEYS[14:]) <= set(UNIFIED_EXPRESSIONS)


def test_payload_is_102_bytes_and_neutral_by_default():
    payload = encode_native_face(UnifiedTrackingData())
    assert len(payload) == 102
    assert set(payload) == {0}


def test_payload_values():
    data = UnifiedTrackingData()
    data.shapes["JawOpen"] = 1.0
    data.shapes["TongueOut"] = 0.5
    data.shapes["MouthClosed"] = 2.0  # clamped
    data.eye.left.openness = 0.25
    payload = encode_native_face(data)
    assert payload[36] == 255
    assert payload[TRACKING_KEYS.index("TongueOut")] == 128
    assert payload[TRACKING_KEYS.index("MouthClosed")] == 255
    assert payload[TRACKING_KEYS.index("EyeClosedLeft")] == 191
    assert payload[TRACKING_KEYS.index("EyeClosedRight")] == 0


def test_gaze_splits_into_look_keys():
    data = UnifiedTrackingData()
    data.eye.left.gaze = (0.5, -0.25)
    data.eye.right.gaze = (0.5, -0.25)
    keys = compute_native_face_keys(data)
    # Looking right: right eye out, left eye in; looking down.
    assert keys["EyeLookOutRight"] == pytest.approx(0.5)
    assert keys["EyeLookInLeft"] == pytest.approx(0.5)
    assert keys["EyeLookInRight"] == 0.0
    assert keys["EyeLookOutLeft"] == 0.0
    assert keys["EyeLookDownRight"] == pytest.approx(0.25)
    assert keys["EyeLookUpLeft"] == 0.0


def test_pupil_splits_around_neutral():
    data = UnifiedTrackingData()
    neutral = compute_native_face_keys(data, 0.5)
    assert neutral["EyeDilationLeft"] == neutral["EyeConstrictLeft"] == 0.0
    dilated = compute_native_face_keys(data, 0.75)
    assert dilated["EyeDilationRight"] == pytest.approx(0.5)
    assert dilated["EyeConstrictRight"] == 0.0
    constricted = compute_native_face_keys(data, 0.0)
    assert constricted["EyeConstrictLeft"] == pytest.approx(1.0)


def test_blob_round_trips_through_osc_codec():
    payload = bytes(range(102))
    message = osc.encode_message(ADDRESS, [payload])
    assert len(message) % 4 == 0
    assert osc.parse(message) == [(ADDRESS, [payload])]


def _client(names=()):
    client = VRCFTClient(use_oscquery=False)
    client.avatar = AvatarInfo(id="avtr_test", parameters=[
        AvatarParameter(name.rsplit("/", 1)[-1], "/avatar/parameters/" + name, float)
        for name in names
    ])
    return client


def test_native_face_sent_to_avatar_without_ft_parameters():
    client = _client(["SomeToggle"])
    client.update_tracking(UnifiedTrackingData())
    assert client.native_face_relevant()
    assert len(client._direct[ADDRESS].pending) == 102


@pytest.mark.parametrize("names", [["FT/v2/JawOpen"], ["JawOpen"], ["EyesX"]])
def test_ft_parameter_avatars_keep_ft_parameters(names):
    client = _client(names)
    client.update_tracking(UnifiedTrackingData())
    assert not client.native_face_relevant()
    assert ADDRESS not in client._direct


def test_native_face_can_be_disabled():
    client = _client()
    client.native_face = False
    client.update_tracking(UnifiedTrackingData())
    assert ADDRESS not in client._direct


def test_native_face_reaches_the_wire():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(0.5)
    client = VRCFTClient(send_port=rx.getsockname()[1], recv_port=0,
                         use_oscquery=False)
    client.start()
    try:
        data = UnifiedTrackingData()
        data.shapes["JawOpen"] = 1.0
        client.update_tracking(data)
        deadline = time.monotonic() + 1.0
        received = None
        while received is None and time.monotonic() < deadline:
            packet, _ = rx.recvfrom(65535)
            for address, args in osc.parse(packet):
                if address == ADDRESS:
                    received = args
    finally:
        client.stop()
        rx.close()
    assert received is not None and len(received) == 1
    assert len(received[0]) == 102 and received[0][36] == 255
