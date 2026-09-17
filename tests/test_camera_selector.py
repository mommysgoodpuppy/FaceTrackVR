from types import SimpleNamespace
import ctypes

import numpy as np

from camera_enum import label_uvc_cameras
import camera_behaviors
from camera_behaviors import (
    HtcStreamController,
    WindowsHtcStreamController,
    _run_htc_handshake,
    stream_controller_for,
)
from settings.modules.LipSettingsModule import LipSettingsModule
from lip import MouthCamera


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def test_known_multi_interface_cameras_collapse_to_video_index_zero():
    cameras = [
        {
            "name": "Bigeye: Bigeye",
            "address": "/dev/v4l/by-id/usb-Bigscreen_Bigeye-video-index1",
        },
        {
            "name": "HTC Multimedia Camera: HTC Mult",
            "address": "/dev/v4l/by-id/usb-HTC_Multimedia_Camera-video-index1",
        },
        {
            "name": "Bigeye: Bigeye",
            "address": "/dev/v4l/by-id/usb-Bigscreen_Bigeye-video-index0",
        },
        {
            "name": "HTC Multimedia Camera: HTC Mult",
            "address": "/dev/v4l/by-id/usb-HTC_Multimedia_Camera-video-index0",
        },
    ]

    choices = label_uvc_cameras(cameras)

    assert [label for label, _camera in choices] == [
        "Bigeye: Bigeye",
        "HTC Multimedia Camera: HTC Mult",
    ]
    assert all(
        camera["address"].endswith("video-index0")
        for _label, camera in choices
    )


def test_ordinary_duplicate_camera_names_remain_separately_selectable():
    cameras = [
        {"name": "Webcam", "address": "/dev/video4"},
        {"name": "Webcam", "address": "/dev/video6"},
    ]

    choices = label_uvc_cameras(cameras)

    assert [label for label, _camera in choices] == ["Webcam (1)", "Webcam (2)"]


def test_camera_behavior_follows_device_instead_of_selector_slot():
    by_name = stream_controller_for(
        "HTC Multimedia Camera: HTC Mult", "/dev/video2"
    )
    by_address = stream_controller_for(
        "", "/dev/v4l/by-id/usb-HTC_Multimedia_Camera-video-index0"
    )
    ordinary = stream_controller_for("Bigeye: Bigeye", "/dev/video0")

    assert isinstance(by_name, HtcStreamController)
    assert isinstance(by_address, HtcStreamController)
    assert ordinary is None


def test_windows_camera_behavior_uses_resolved_directshow_index(monkeypatch):
    monkeypatch.setattr(camera_behaviors.sys, "platform", "win32")

    controller = stream_controller_for(
        "HTC Multimedia Camera", "USB\\VID_0BB4&PID_0581", device_index=3
    )

    assert isinstance(controller, WindowsHtcStreamController)
    assert controller.device_index == 3


def test_htc_handshake_builds_expected_enable_commands(monkeypatch):
    monkeypatch.setattr(camera_behaviors.time, "sleep", lambda _seconds: None)
    payloads = []

    _run_htc_handshake(64, True, lambda payload: payloads.append(bytes(payload)))

    assert payloads[0][:2] == bytes((0x51, 0x52))
    assert payloads[1][:4] == bytes((0x50, 0x14, 0x00, 0x00))
    assert payloads[-1][:4] == bytes((0x50, 0x14, 0x00, 0x01))
    led_writes = [
        payload for payload in payloads
        if payload[:5] == bytes((0x50, 0xAB, 0x60, 0x01, 0x01))
        and payload[8] in (0x02, 0x03, 0x04)
    ]
    assert len(led_writes) == 3
    assert all(payload[16] == 0xFF for payload in led_writes)


def test_htc_handshake_disable_turns_emitters_off_without_restarting_stream(
    monkeypatch,
):
    monkeypatch.setattr(camera_behaviors.time, "sleep", lambda _seconds: None)
    payloads = []

    _run_htc_handshake(384, False, lambda payload: payloads.append(bytes(payload)))

    assert payloads[0][254:256] == bytes((0x53, 0x54))
    assert payloads[1][:4] == bytes((0x50, 0x14, 0x00, 0x00))
    assert not any(
        payload[:4] == bytes((0x50, 0x14, 0x00, 0x01))
        for payload in payloads
    )
    led_writes = [
        payload for payload in payloads
        if payload[:5] == bytes((0x50, 0xAB, 0x60, 0x01, 0x01))
        and payload[8] in (0x02, 0x03, 0x04)
    ]
    assert len(led_writes) == 3
    assert all(payload[16] == 0x00 for payload in led_writes)


def test_mouth_camera_accepts_only_unconverted_vft_frames():
    assert MouthCamera._is_raw_vft_frame(
        np.zeros((400, 400, 2), dtype=np.uint8)
    )
    assert MouthCamera._is_raw_vft_frame(
        np.zeros((400, 800), dtype=np.uint8)
    )
    assert not MouthCamera._is_raw_vft_frame(
        np.zeros((400, 400, 3), dtype=np.uint8)
    )


def test_windows_ks_property_uses_selector_two_and_topology_flag(monkeypatch):
    class FakeGuid(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_uint8 * 16)]

        def __init__(self, _text=None):
            super().__init__()

    class FakeProperty(ctypes.Structure):
        _fields_ = [
            ("Set", FakeGuid),
            ("Id", ctypes.c_uint32),
            ("Flags", ctypes.c_uint32),
        ]

    class FakeNode(ctypes.Structure):
        _fields_ = [
            ("Property", FakeProperty),
            ("NodeId", ctypes.c_uint32),
            ("Reserved", ctypes.c_uint32),
        ]

    calls = []

    class FakeControl:
        def KsProperty(self, request_ptr, request_size, data_ptr, data_size):
            request = ctypes.cast(
                request_ptr, ctypes.POINTER(FakeNode)
            ).contents
            calls.append(
                (
                    request.Property.Id,
                    request.Property.Flags,
                    request.NodeId,
                    request_size,
                    data_size,
                )
            )
            ctypes.memset(data_ptr, 0x56, data_size)

    monkeypatch.setattr(
        camera_behaviors,
        "_windows_ks_types",
        lambda: (FakeGuid, FakeNode, object, object),
    )

    result = WindowsHtcStreamController._ks_property(
        FakeControl(), 9, 64, get=True
    )

    assert calls == [(2, 0x10000001, 9, 32, 64)]
    assert result == bytes((0x56,)) * 64


def test_mouth_camera_friendly_label_saves_stable_named_source():
    module = LipSettingsModule(SimpleNamespace(), "mouth")
    module._device_display_map = {
        "Vive Facial Tracker": (
            "uvc:Vive Facial Tracker@/dev/v4l/by-id/vive-video-index0"
        )
    }
    module.tk_vars = {
        module.gui_lip_device: _Value("Vive Facial Tracker"),
        module.gui_lip_onnx_path: _Value("/models/mouth.onnx"),
    }

    values = module.get_values_map()

    assert values[module.gui_lip_device] == (
        "uvc:Vive Facial Tracker@/dev/v4l/by-id/vive-video-index0"
    )
    assert values[module.gui_lip_onnx_path] == "/models/mouth.onnx"


def test_mouth_camera_disconnect_label_saves_empty_device():
    module = LipSettingsModule(SimpleNamespace(), "mouth")
    module._device_display_map = {"None (disconnect)": ""}
    module.tk_vars = {
        module.gui_lip_device: _Value("None (disconnect)"),
    }

    values = module.get_values_map()

    assert values[module.gui_lip_device] == ""
