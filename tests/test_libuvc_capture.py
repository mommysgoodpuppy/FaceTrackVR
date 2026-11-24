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


def test_find_usb_device_path_reads_busnum_and_devnum(tmp_path):
    bigeye = tmp_path / "3-2.1.2"
    bigeye.mkdir()
    (bigeye / "idVendor").write_text("35bd\n")
    (bigeye / "idProduct").write_text("0202\n")
    (bigeye / "busnum").write_text("3\n")
    (bigeye / "devnum").write_text("5\n")

    assert (
        libuvc_capture._find_usb_device_path(0x35BD, 0x0202, str(tmp_path))
        == "/dev/bus/usb/003/005"
    )
    assert libuvc_capture._find_usb_device_path(0x35BD, 0x0101, str(tmp_path)) is None


def test_frame_assembler_ignores_uvc_boundaries_inside_jpegs():
    asm = libuvc_capture._UvcFrameAssembler()
    # Captured failure: EOF/FID change in the JPEG, then EOI + SOI inside
    # a later packet. Neither UVC flag is a reliable image boundary.
    first = b"\xff\xd8first image\xff\xd9"
    second = b"\xff\xd8second image\xff\xd9"
    asm.feed_packet(bytes([12, 0x8c]) + bytes(10) + b"old tail" + first[:6])
    asm.feed_packet(bytes([12, 0x8e]) + bytes(10) + first[6:9])
    assert not asm.ready
    asm.feed_packet(bytes([12, 0x8d]) + bytes(10) + first[9:] + second[:8])
    assert asm.ready.popleft() == first
    asm.feed_packet(bytes([12, 0x8f]) + bytes(10) + second[8:])
    assert list(asm.ready) == [second]
    assert asm.received_frames == 2


def test_frame_assembler_handles_split_markers_and_multiple_images():
    asm = libuvc_capture._UvcFrameAssembler()
    for body in [b"junk\xff", b"\xd8one\xff", b"\xd9\xff\xd8two\xff\xd9\xff\xd8three\xff\xd9"]:
        asm.feed_packet(memoryview(bytes([2, 0x00]) + body))
    assert list(asm.ready) == [b"\xff\xd8one\xff\xd9", b"\xff\xd8two\xff\xd9", b"\xff\xd8three\xff\xd9"]


def test_frame_assembler_recovers_after_error_oversize_and_missing_eoi():
    asm = libuvc_capture._UvcFrameAssembler(max_frame_bytes=32)
    asm.feed_packet(bytes([2, 0x00]) + b"\xff\xd8partial")
    asm.feed_packet(bytes([2, 0x40]) + b"bad")
    asm.feed_packet(bytes([2, 0x40]) + b"still_bad")
    asm.feed_packet(bytes([2, 0x02]) + b"tail\xff\xd9")
    assert not asm.ready
    assert asm.dropped_error_frames == 1

    asm.feed_packet(bytes([2, 0x01]) + b"\xff\xd8" + b"x" * 20)
    asm.feed_packet(bytes([2, 0x03]) + b"y" * 20)
    assert not asm.ready
    assert len(asm._buf) <= 1
    assert asm.dropped_error_frames == 2

    asm.feed_packet(bytes([2, 0x00]) + b"\xff\xd8no end")
    good = b"\xff\xd8recovered\xff\xd9"
    asm.feed_packet(bytes([2, 0x00]) + good)
    assert list(asm.ready) == [good]
    assert asm.dropped_error_frames == 3

    asm.feed_packet(b"")
    asm.feed_packet(b"\x01")
    asm.feed_packet(bytes([99, 0x02]) + b"short")
    asm.feed_packet(bytes([2, 0x00]) + b"x" * 100)
    assert len(asm._buf) <= 1


def test_real_jpegs_decode_despite_uvc_boundary_drift():
    ok, encoded = cv2.imencode(".jpg", np.full((400, 800, 3), 128, np.uint8))
    assert ok
    jpeg = encoded.tobytes()
    payload = b"old tail" + jpeg * 3
    asm = libuvc_capture._UvcFrameAssembler()
    for i, offset in enumerate(range(0, len(payload), 1012)):
        # Deliberately unrelated FID/EOF flags, like the live camera.
        asm.feed_packet(bytes([12, 0x8c | (i % 4)]) + bytes(10) + payload[offset:offset + 1012])
    assert list(asm.ready) == [jpeg] * 3
    for frame in asm.ready:
        assert cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR).shape == (400, 800, 3)


class _FakePoller:
    def __init__(self, events):
        self._events = list(events)
        self.timeouts = []

    def poll(self, timeout_ms):
        self.timeouts.append(timeout_ms)
        return self._events.pop(0) if self._events else []


def test_read_returns_queued_frame_and_handles_timeout_and_disconnect(monkeypatch):
    stream = libuvc_capture.LibuvcCapture()
    assert stream.read(0.01) is None

    stream._fd = 99
    stream._streaming = True
    stream._poller = _FakePoller([])
    stream._assembler.ready.append(b"\xff\xd8body\xff\xd9")

    assert stream.read(0.05) == b"\xff\xd8body\xff\xd9"
    # Timeout when poller returns no events.
    assert stream.read(0.01) is None
    assert len(stream._poller.timeouts) == 1

    # Disconnect event raises LibuvcError so Camera tears down cleanly.
    stream._poller = _FakePoller([[(99, libuvc_capture.select.POLLHUP)]])
    with pytest.raises(libuvc_capture.LibuvcError):
        stream.read(0.05)


def test_read_returns_none_once_streaming_stops():
    stream = libuvc_capture.LibuvcCapture()
    stream._fd = 99
    stream._streaming = False
    stream._poller = _FakePoller([[(99, libuvc_capture.select.POLLOUT)]])
    stream._assembler.ready.append(b"x")

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


@pytest.mark.parametrize("payload", [None, b"not a jpeg"])
def test_libuvc_reconnects_after_sustained_invalid_frames(
    monkeypatch, eyetrack_camera_config, eyetrack_settings_config, payload
):
    cam = _camera(eyetrack_camera_config, eyetrack_settings_config, BIGEYE_BY_ID)
    stream = _FakeStream(frames=[payload] * 3)
    cam._libuvc_stream = stream
    clock = [10.0]
    monkeypatch.setattr(camera_module.time, "monotonic", lambda: clock[0])
    cam.get_libuvc_camera_picture(True)
    assert not stream.closed
    clock[0] = 11.9
    cam.get_libuvc_camera_picture(True)
    assert not stream.closed
    clock[0] = 12.1
    cam.get_libuvc_camera_picture(True)
    assert stream.closed
    assert cam._libuvc_stream is None
    assert cam.camera_status == CameraState.DISCONNECTED


def test_valid_libuvc_frame_resets_recovery_deadline(
    monkeypatch, eyetrack_camera_config, eyetrack_settings_config
):
    cam = _camera(eyetrack_camera_config, eyetrack_settings_config, BIGEYE_BY_ID)
    ok, encoded = cv2.imencode(".jpg", np.full((400, 800, 3), 128, np.uint8))
    assert ok
    stream = _FakeStream(frames=[None, encoded.tobytes(), None, None])
    cam._libuvc_stream = stream
    clock = [10.0]
    monkeypatch.setattr(camera_module.time, "monotonic", lambda: clock[0])
    for instant in [10.0, 11.0, 12.5, 14.0]:
        clock[0] = instant
        cam.get_libuvc_camera_picture(True)
    assert not stream.closed
    assert cam.frame_number == 1
