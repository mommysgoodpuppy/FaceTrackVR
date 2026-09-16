from types import SimpleNamespace

from camera_enum import label_uvc_cameras
from camera_behaviors import HtcStreamController, stream_controller_for
from settings.modules.LipSettingsModule import LipSettingsModule


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


def test_mouth_camera_friendly_label_saves_stable_device_address():
    module = LipSettingsModule(SimpleNamespace(), "mouth")
    module._device_display_map = {
        "Vive Facial Tracker": "/dev/v4l/by-id/vive-video-index0"
    }
    module.tk_vars = {
        module.gui_lip_device: _Value("Vive Facial Tracker"),
        module.gui_lip_onnx_path: _Value("/models/mouth.onnx"),
    }

    values = module.get_values_map()

    assert values[module.gui_lip_device] == (
        "/dev/v4l/by-id/vive-video-index0"
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
