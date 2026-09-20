"""Tests for the BSB2E userspace (libuvc) capture backend.

The kernel V4L2 path for the Bigscreen Bigeye is broken, so the camera is
detected from sysfs and routed through libuvc. These tests pin the observable
behavior: detection, source routing, libuvc shutdown, and the frame-decode
path into the capture queue. The libuvc library itself is stubbed out.
"""

import ctypes
import logging
import queue
import sys
import threading

import cv2
import numpy as np
import pytest

import camera as camera_module
import libuvc_capture
from camera import Camera, CameraState

BIGEYE_BY_ID = (
    "uvc:Bigeye: Bigeye@/dev/v4l/by-id/usb-Bigscreen_Bigeye-video-index0"
)


def _write_usb_ids(directory, vendor, product):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "idVendor").write_text(f"{vendor:04x}\n")
    (directory / "idProduct").write_text(f"{product:04x}\n")


def _fake_sysfs(monkeypatch, tmp_path, mapping):
    """Point video-node sysfs lookups at tmp dirs. ``mapping`` maps a node
    name (e.g. "video0") to ``(vendor, product)`` or None for a missing node."""
    interface_dirs = {}
    for node, ids in mapping.items():
        if ids is None:
            continue
        # ids live on the USB device dir; the video node points at the
        # interface dir beneath it, so the lookup must walk up.
        usb_dir = tmp_path / f"{node}-usb"
        _write_usb_ids(usb_dir, *ids)
        interface = usb_dir / "iface"
        interface.mkdir()
        interface_dirs[node] = interface

    real_realpath = libuvc_capture.os.path.realpath

    def fake_realpath(path, *args, **kwargs):
        for node, interface in interface_dirs.items():
            if str(path) == f"/sys/class/video4linux/{node}/device":
                return str(interface)
        return real_realpath(path, *args, **kwargs)

    monkeypatch.setattr(libuvc_capture.os.path, "realpath", fake_realpath)


def test_usb_ids_are_read_by_walking_up_from_the_video_node(monkeypatch, tmp_path):
    _fake_sysfs(monkeypatch, tmp_path, {"video0": (0x35BD, 0x0202)})

    assert libuvc_capture._usb_ids_for_video_node("video0") == (0x35BD, 0x0202)
    assert libuvc_capture._usb_ids_for_video_node("video9") is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux-only")
def test_bigeye_is_recognized_and_other_cameras_are_not(monkeypatch, tmp_path):
    _fake_sysfs(
        monkeypatch,
        tmp_path,
        {"video0": (0x35BD, 0x0202), "video2": (0x0BB4, 0x0321)},
    )
    real_realpath = libuvc_capture.os.path.realpath
    monkeypatch.setattr(
        libuvc_capture.os.path,
        "realpath",
        lambda p, *a, **k: "/dev/video0"
        if str(p).endswith("usb-Bigscreen_Bigeye-video-index0")
        else real_realpath(p, *a, **k),
    )

    assert libuvc_capture.is_bsb2e_video_device("/dev/video0") is True
    assert libuvc_capture.is_bsb2e_video_device("/dev/video2") is False
    # The stored (by-id) address is returned verbatim; only the sysfs lookup
    # has to peel the symlink back to the /dev/videoN node.
    assert (
        libuvc_capture.bsb2e_video_path(BIGEYE_BY_ID)
        == "/dev/v4l/by-id/usb-Bigscreen_Bigeye-video-index0"
    )
    assert libuvc_capture.bsb2e_video_path("uvc:Bigeye: Bigeye@index:0") is None
    assert libuvc_capture.bsb2e_video_path("http://etvr.local/") is None
    assert libuvc_capture.bsb2e_video_path(0) is None


def test_non_linux_hosts_never_route_to_libuvc(monkeypatch):
    monkeypatch.setattr(libuvc_capture.sys, "platform", "win32")
    monkeypatch.setattr(libuvc_capture, "_bsb2e_usb_attached", lambda: True)

    assert libuvc_capture.is_bsb2e_video_device("/dev/video0") is False
    assert libuvc_capture.bsb2e_video_path(BIGEYE_BY_ID) is None


def test_detached_camera_routes_by_usb_identity(monkeypatch):
    """libuvc removes the /dev node while streaming, so a camera we already own
    must still be recognized from its USB identity."""
    monkeypatch.setattr(libuvc_capture, "is_bsb2e_video_device", lambda path: False)
    monkeypatch.setattr(libuvc_capture, "_bsb2e_usb_attached", lambda: True)

    assert libuvc_capture.bsb2e_video_path(BIGEYE_BY_ID) == (
        "/dev/v4l/by-id/usb-Bigscreen_Bigeye-video-index0"
    )
    # Without a Bigeye on the bus, the fallback must not invent one.
    monkeypatch.setattr(libuvc_capture, "_bsb2e_usb_attached", lambda: False)
    assert libuvc_capture.bsb2e_video_path(BIGEYE_BY_ID) is None


def test_usb_identity_scan_matches_vendor_and_product(tmp_path):
    bigeye = tmp_path / "3-2.1.2"
    bigeye.mkdir()
    (bigeye / "idVendor").write_text("35bd\n")
    (bigeye / "idProduct").write_text("0202\n")
    htc = tmp_path / "1-1"
    htc.mkdir()
    (htc / "idVendor").write_text("0bb4\n")
    (htc / "idProduct").write_text("0321\n")

    assert libuvc_capture._bsb2e_usb_attached(str(tmp_path)) is True

    # Same vendor, different product (the Beyond HMD): not the eye camera.
    (bigeye / "idProduct").write_text("0101\n")
    assert libuvc_capture._bsb2e_usb_attached(str(tmp_path)) is False


def _mjpeg_frame(payload: bytes, fmt: int = libuvc_capture.UVC_FRAME_FORMAT_MJPEG):
    buf = ctypes.create_string_buffer(payload)
    frame = libuvc_capture._Frame()
    frame.data = ctypes.cast(buf, ctypes.c_void_p)
    frame.data_bytes = len(payload)
    frame.frame_format = fmt
    return frame


def test_payload_extraction_copies_mjpeg_and_ignores_other_formats():
    stream = libuvc_capture.LibuvcCapture()
    assert stream._payload_from_frame(_mjpeg_frame(b"\xff\xd8first\xff\xd9")) == (
        b"\xff\xd8first\xff\xd9"
    )
    assert stream.received_frames == 1

    # 0 is UVC_FRAME_FORMAT_UNKNOWN and is never what the Bigeye streams.
    assert stream._payload_from_frame(_mjpeg_frame(b"\x00\x01", fmt=0)) is None
    assert stream.dropped_format_frames == 1

    # A zero-length payload must not produce an empty frame.
    assert stream._payload_from_frame(_mjpeg_frame(b"")) is None


class _FakeLib:
    """Stands in for the loaded libuvc CDLL inside LibuvcCapture.read()."""

    def __init__(self, frame, rc=0):
        self._frame = frame
        self._rc = rc
        self.calls = []

    def uvc_stream_get_frame(self, stream, frame_ref, timeout_us):
        self.calls.append(timeout_us)
        if self._rc == 0 and self._frame is not None:
            out = ctypes.cast(
                frame_ref, ctypes.POINTER(ctypes.POINTER(libuvc_capture._Frame))
            )
            out[0] = ctypes.pointer(self._frame)
        return self._rc


def _stream_with_fake_lib(monkeypatch, lib):
    stream = libuvc_capture.LibuvcCapture()
    stream._stream = ctypes.c_void_p(1)
    stream._streaming = True
    monkeypatch.setattr(libuvc_capture, "_load_lib", lambda: (lib, None))
    return stream


def test_read_returns_the_payload_and_passes_the_timeout(monkeypatch):
    lib = _FakeLib(_mjpeg_frame(b"\xff\xd8body\xff\xd9"))
    stream = _stream_with_fake_lib(monkeypatch, lib)

    assert stream.read(0.05) == b"\xff\xd8body\xff\xd9"
    assert lib.calls == [50_000]


def test_read_returns_none_on_timeout_and_without_a_stream(monkeypatch):
    stream = _stream_with_fake_lib(monkeypatch, _FakeLib(None, rc=-50))
    assert stream.read(0.01) is None

    idle = libuvc_capture.LibuvcCapture()
    assert idle.read(0.01) is None


def test_read_returns_none_once_streaming_stops(monkeypatch):
    stream = _stream_with_fake_lib(monkeypatch, _FakeLib(_mjpeg_frame(b"x")))
    stream._streaming = False

    assert stream.read(0.01) is None


# -- Camera integration ----------------------------------------------------


def _camera(camera_config, settings_config, source):
    config = camera_config.model_copy(update={"capture_source": source})
    return Camera(
        config=config,
        settings=settings_config,
        camera_index=0,
        cancellation_event=threading.Event(),
        capture_event=threading.Event(),
        camera_status_outgoing=queue.Queue(),
        camera_output_outgoing=queue.Queue(maxsize=20),
    )


def test_path_lookup_is_cached_per_source(monkeypatch, eyetrack_camera_config, eyetrack_settings_config):
    cam = _camera(eyetrack_camera_config, eyetrack_settings_config, BIGEYE_BY_ID)
    probed = []

    def fake_path(source):
        probed.append(source)
        return "/dev/video0"

    monkeypatch.setattr(camera_module.libuvc_capture, "bsb2e_video_path", fake_path)
    monkeypatch.setattr(camera_module.libuvc_capture, "libuvc_available", lambda: True)

    assert cam._libuvc_path_for(BIGEYE_BY_ID) == "/dev/video0"
    assert cam._libuvc_path_for(BIGEYE_BY_ID) == "/dev/video0"
    assert probed == [BIGEYE_BY_ID]

    assert cam._libuvc_path_for("uvc:Bigeye: Bigeye@/dev/video1") == "/dev/video0"
    assert len(probed) == 2


def test_missing_libuvc_falls_back_with_a_warning(
    monkeypatch, caplog, eyetrack_camera_config, eyetrack_settings_config
):
    cam = _camera(eyetrack_camera_config, eyetrack_settings_config, BIGEYE_BY_ID)
    monkeypatch.setattr(
        camera_module.libuvc_capture, "bsb2e_video_path", lambda source: "/dev/video0"
    )
    monkeypatch.setattr(camera_module.libuvc_capture, "libuvc_available", lambda: False)
    caplog.set_level(logging.WARNING)

    assert cam._libuvc_path_for(BIGEYE_BY_ID) is None
    assert "libuvc" in caplog.text


class _FakeStream:
    def __init__(self, frames=(), fail=None):
        self._frames = list(frames)
        self.fail = fail
        self.opened = False
        self.started = False
        self.closed = False

    def open(self):
        if self.fail:
            raise self.fail
        self.opened = True

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def read(self, timeout):
        if self.fail:
            raise self.fail
        return self._frames.pop(0) if self._frames else None


def test_open_failure_leaves_camera_disconnected(monkeypatch, eyetrack_camera_config, eyetrack_settings_config):
    cam = _camera(eyetrack_camera_config, eyetrack_settings_config, BIGEYE_BY_ID)
    monkeypatch.setattr(
        camera_module.libuvc_capture, "LibuvcCapture", lambda: _FakeStream(fail=RuntimeError("nope"))
    )

    cam._open_libuvc_stream(BIGEYE_BY_ID, "/dev/video0")

    assert cam._libuvc_stream is None
    assert cam.camera_status == CameraState.DISCONNECTED


def test_open_waits_for_a_frame_before_reporting_success(
    monkeypatch, eyetrack_camera_config, eyetrack_settings_config
):
    cam = _camera(eyetrack_camera_config, eyetrack_settings_config, BIGEYE_BY_ID)
    stream = _FakeStream()
    monkeypatch.setattr(camera_module.libuvc_capture, "LibuvcCapture", lambda: stream)
    monkeypatch.setattr(camera_module.libuvc_capture, "wait_for_first_frame", lambda s, t: False)

    cam._open_libuvc_stream(BIGEYE_BY_ID, "/dev/video0")

    assert cam._libuvc_stream is None
    assert stream.closed is True
    assert cam.camera_status == CameraState.DISCONNECTED


def test_release_closes_the_active_stream(monkeypatch, eyetrack_camera_config, eyetrack_settings_config):
    cam = _camera(eyetrack_camera_config, eyetrack_settings_config, BIGEYE_BY_ID)
    stream = _FakeStream(frames=[b"x"])
    cam._libuvc_stream = stream

    cam._release_libuvc_stream()

    assert stream.closed is True
    assert cam._libuvc_stream is None


def test_libuvc_frames_are_decoded_and_pushed_once_per_frame(
    eyetrack_camera_config, eyetrack_settings_config
):
    cam = _camera(eyetrack_camera_config, eyetrack_settings_config, BIGEYE_BY_ID)
    ok, encoded = cv2.imencode(".jpg", np.full((400, 800, 3), 128, np.uint8))
    assert ok
    payload = encoded.tobytes()
    cam._libuvc_stream = _FakeStream(frames=[payload, payload])

    cam.get_libuvc_camera_picture(True)

    image, frame_number, fps, _ts = cam.camera_output_outgoing.get_nowait()
    assert image.shape == (340, 680, 3)
    assert frame_number == 1

    # Each delivered frame is published; a read that yields nothing (timeout)
    # must not publish or re-publish anything.
    cam.get_libuvc_camera_picture(True)
    assert cam.camera_output_outgoing.get_nowait()[1] == 2
    cam.get_libuvc_camera_picture(True)
    assert cam.camera_output_outgoing.empty()


def test_downscale_caps_width_and_preserves_aspect():
    wide = np.zeros((400, 800, 3), np.uint8)
    assert Camera._downscale_wide_frame(wide).shape == (340, 680, 3)

    small = np.zeros((480, 640, 3), np.uint8)
    assert Camera._downscale_wide_frame(small) is small


def test_push_failure_tears_down_the_stream(eyetrack_camera_config, eyetrack_settings_config):
    cam = _camera(eyetrack_camera_config, eyetrack_settings_config, BIGEYE_BY_ID)

    class _Broken(_FakeStream):
        def read(self, timeout):
            raise RuntimeError("device gone")

    stream = _Broken()
    cam._libuvc_stream = stream

    cam.get_libuvc_camera_picture(True)

    assert cam.camera_status == CameraState.DISCONNECTED
    assert cam._libuvc_stream is None
    assert stream.closed is True
