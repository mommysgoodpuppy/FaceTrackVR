"""Babble-OSC backend: mapping table + loopback wire proof.

Sends through a real BabbleSender into a UDP loopback socket and decodes
each datagram with the stdlib pyvrcft OSC codec. Asserts the stable
Baballonia parameter set (single message per datagram, float args) and
that the experimental eye expressions (squint/lower) stay off the wire.
"""

import socket

import pytest

from eye import EyeId, EyeInfo, EyeInfoOrigin
from osc.BabbleSender import (
    COLLAPSED_ADDRESSES,
    SHAPE_ADDRESSES,
    BabbleSender,
    frame_to_babble_params,
)
from osc.OSCMessage import OSCMessage, OSCMessageType
from pyvrcft import osc as osc_codec


class _Config:
    gui_babble_address = "127.0.0.1"
    gui_babble_port = 0  # replaced with the bound loopback port
    osc_invert_eye_close = False


class _MainConfig:
    eye_display_id = 2  # dual-eye path


def _eye_info(x=0.25, y=-0.5, blink=0.8, squeeze=0.3):
    return EyeInfo(
        EyeInfoOrigin.NEXT,
        x,
        y,
        0.5,
        blink,
        0.0,
        float("nan"),
        squeeze,
        {"JawOpen": 0.6, "EyeWideLeft": 0.7},
        {},
    )


def _receive_all(sock, count, timeout=2.0):
    sock.settimeout(timeout)
    out = []
    for _ in range(count):
        data, _ = sock.recvfrom(4096)
        parsed = osc_codec.parse(data)
        assert len(parsed) == 1, f"one OSC message per datagram, got {parsed}"
        address, args = parsed[0]
        assert len(args) == 1, f"one float per message, got {parsed}"
        out.append((address, args[0]))
    return dict(out)
def test_frame_mapping_matches_babble_receiver_set():
    sender = BabbleSender()
    sender.data.eye.left.gaze = (0.25, -0.5)
    sender.data.eye.left.openness = 0.8
    sender.data.eye.right.gaze = (-0.1, 0.2)
    sender.data.eye.right.openness = 0.6
    sender.data.shapes["EyeWideLeft"] = 0.7
    sender.data.shapes["JawOpen"] = 0.6
    sender.data.shapes["LipFunnelUpperLeft"] = 0.2
    sender.data.shapes["LipFunnelLowerRight"] = 0.9

    params = dict(frame_to_babble_params(sender.data))

    assert params["/LeftEyeX"] == 0.25
    assert params["/LeftEyeY"] == -0.5
    assert params["/LeftEyeLid"] == 0.8
    assert params["/LeftEyeWiden"] == 0.7
    assert params["/jawOpen"] == 0.6
    assert params["/mouthFunnel"] == 0.9  # max() collapse
    # Experimental expressions stay off the wire (Resonite-Issues #6821).
    assert not any(
        a in ("/LeftEyeBrow", "/RightEyeBrow", "/LeftEyeLower", "/RightEyeLower")
        for a in params
    )
    assert len(params) == 8 + len(SHAPE_ADDRESSES) + len(COLLAPSED_ADDRESSES)



def test_babble_sender_loopback_wire():
    recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv.bind(("127.0.0.1", 0))
    cfg = _Config()
    cfg.gui_babble_port = recv.getsockname()[1]

    sender = BabbleSender()
    sender.start(cfg)
    try:
        sender.output_osc_info(
            OSCMessage(OSCMessageType.EYE_INFO, (EyeId.RIGHT, _eye_info())),
            _MainConfig(),
            cfg,
        )
        n = len(frame_to_babble_params(sender.data))
        got = _receive_all(recv, n)
    finally:
        sender.stop()
        recv.close()

    assert got["/RightEyeX"] == pytest.approx(0.25)
    assert got["/RightEyeY"] == pytest.approx(-0.5)
    assert got["/RightEyeLid"] == pytest.approx(0.8)
    assert got["/RightEyeWiden"] == pytest.approx(0.0)
    assert got["/LeftEyeWiden"] == pytest.approx(0.7)  # auxiliary shape lands left
    assert got["/jawOpen"] == pytest.approx(0.6)
    assert all(isinstance(v, float) for v in got.values())


def test_mouth_only_mapping_omits_eye_addresses():
    sender = BabbleSender()
    sender.data.shapes["JawOpen"] = 0.4
    params = dict(frame_to_babble_params(sender.data, include_eyes=False))
    assert params["/jawOpen"] == 0.4
    assert not any("Eye" in address for address in params)
    assert len(params) == len(SHAPE_ADDRESSES) + len(COLLAPSED_ADDRESSES)


def test_lip_frame_flushes_mouth_only_sender():
    from lip import LipTracker

    recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv.bind(("127.0.0.1", 0))
    cfg = _Config()
    cfg.gui_babble_port = recv.getsockname()[1]

    sender = BabbleSender()
    sender.start(cfg, send_eyes=False)
    tracker = LipTracker.__new__(LipTracker)
    tracker.shared_data = sender.data
    tracker.shared_data_provider = None
    try:
        tracker._emit({"JawOpen": 0.75})
        got = _receive_all(
            recv, len(SHAPE_ADDRESSES) + len(COLLAPSED_ADDRESSES)
        )
    finally:
        sender.stop()
        recv.close()

    assert got["/jawOpen"] == pytest.approx(0.75)
    assert "/LeftEyeX" not in got
    assert sender.data._lip_frame_sender is None


@pytest.mark.parametrize(
    "flags, started, send_eyes",
    [
        ({"gui_babble": True}, True, True),
        ({"gui_use_module": True, "gui_osc_vrcft_v2": True}, True, False),
        ({}, False, None),  # pyVRCFT default
    ],
)
def test_osc_sender_babble_routing(flags, started, send_eyes):
    import queue
    import threading
    import time

    from config import EyeTrackConfig
    from osc.osc import OSCSender

    config = EyeTrackConfig()
    s = config.settings
    s.gui_pyvrcft = not flags
    s.gui_babble_port = 9  # discard; nothing listens
    for name, value in flags.items():
        setattr(s, name, value)

    cancel = threading.Event()
    sender = OSCSender(cancel, queue.Queue(), config)
    seen = {}
    real_stop = sender.babble_sender.stop

    def stop():
        seen["sock"] = sender.babble_sender.sock is not None
        seen["send_eyes"] = sender.babble_sender.send_eyes
        real_stop()

    sender.babble_sender.stop = stop
    sender.pyvrcft_sender.start = lambda config: None
    sender.pyvrcft_sender.stop = lambda: None
    thread = threading.Thread(target=sender.run)
    thread.start()
    time.sleep(0.2)
    cancel.set()
    thread.join(timeout=2)

    assert seen.get("sock", False) is started
    if started:
        assert seen["send_eyes"] is send_eyes
