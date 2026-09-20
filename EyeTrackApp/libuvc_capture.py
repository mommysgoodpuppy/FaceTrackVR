"""Userspace UVC capture for the Bigscreen Bigeye (BSB2E) eye cameras.

The Beyond 2E eye-tracking camera (USB ``35bd:0202``) advertises an MJPEG
800x400@90 format whose ``dwMaxVideoFrameSize`` the Linux kernel ``uvcvideo``
driver reads as zero. ``VIDIOC_REQBUFS`` therefore fails with ``Invalid
argument`` (OpenCV logs "no support for memory mapping" / "Insufficient buffer
memory") and V4L2 capture never starts, even though the camera streams fine on
Windows and through userspace UVC stacks. We bypass V4L2 entirely and drive
libuvc directly, as the reference plugin at ``babl-bsb2e``
(``Baballonia.LibuvcCapture``) does.

Call sequence: ``uvc_init`` -> ``uvc_find_device`` -> ``uvc_open`` ->
``uvc_get_stream_ctrl_format_size`` -> ``uvc_stream_open_ctrl`` ->
``uvc_stream_start``, then ``uvc_stream_get_frame`` per frame.

The capture uses libuvc's *synchronous* API rather than its frame callback.
``uvc_stream_get_frame`` blocks inside libuvc until the next assembled frame
arrives, and ctypes releases the GIL for the duration of that call, so waiting
costs no Python CPU and cannot starve the other threads. A ctypes callback
would run on a libuvc-owned thread and need the GIL 90 times a second, which
measurably throttled the whole pipeline.

``uvc_stream_get_frame`` returns libuvc's own frame buffer, valid until the
next call; it must NOT be freed by the caller. We copy the payload out
immediately.

Only Linux needs this: Windows/macOS are served by cv2's own backends, so the
module is inert elsewhere and simply reports ``LIBUVC_AVAILABLE == False``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import sys
import threading

logger = logging.getLogger(__name__)

# Bigscreen Bigeye (Beyond 2E eye tracking camera).
BSB2E_VENDOR_ID = 0x35BD
BSB2E_PRODUCT_ID = 0x0202
# The single format the camera exposes (see `v4l2-ctl --list-formats-ext`).
BSB2E_WIDTH = 800
BSB2E_HEIGHT = 400
BSB2E_FPS = 90

# enum uvc_frame_format: UNKNOWN/ANY=0, UNCOMPRESSED, COMPRESSED, YUYV, UYVY,
# RGB, BGR, MJPEG.
UVC_FRAME_FORMAT_MJPEG = 7
_UVC_SUCCESS = 0
# How far to walk up the sysfs device chain looking for idVendor/idProduct.
# A video4linux node points at the USB *interface* directory, so the USB
# device directory (which carries the IDs) is one or two levels up.
_USB_ID_WALK_DEPTH = 6


class LibuvcError(RuntimeError):
    """Raised when libuvc is missing or a UVC request fails."""


class _StreamControl(ctypes.Structure):
    """``uvc_stream_ctrl_t`` from libuvc.h."""

    _fields_ = [
        ("bmHint", ctypes.c_uint16),
        ("bFormatIndex", ctypes.c_uint8),
        ("bFrameIndex", ctypes.c_uint8),
        ("dwFrameInterval", ctypes.c_uint32),
        ("wKeyFrameRate", ctypes.c_uint16),
        ("wPFrameRate", ctypes.c_uint16),
        ("wCompQuality", ctypes.c_uint16),
        ("wCompWindowSize", ctypes.c_uint16),
        ("wDelay", ctypes.c_uint16),
        ("dwMaxVideoFrameSize", ctypes.c_uint32),
        ("dwMaxPayloadTransferSize", ctypes.c_uint32),
        ("dwClockFrequency", ctypes.c_uint32),
        ("bmFramingInfo", ctypes.c_uint8),
        ("bPreferredVersion", ctypes.c_uint8),
        ("bMinVersion", ctypes.c_uint8),
        ("bMaxVersion", ctypes.c_uint8),
        ("bInterfaceNumber", ctypes.c_uint8),
    ]


class _Frame(ctypes.Structure):
    """``uvc_frame_t`` from libuvc.h (fields up to ``sequence`` are read)."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("data_bytes", ctypes.c_size_t),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("frame_format", ctypes.c_int),
        ("step", ctypes.c_size_t),
        ("sequence", ctypes.c_uint32),
        ("capture_time_sec", ctypes.c_long),
        ("capture_time_usec", ctypes.c_long),
        ("capture_time_finished_sec", ctypes.c_long),
        ("capture_time_finished_nsec", ctypes.c_long),
        ("source", ctypes.c_void_p),
        ("library_owns_data", ctypes.c_uint8),
        ("metadata", ctypes.c_void_p),
        ("metadata_bytes", ctypes.c_size_t),
    ]


_lib_lock = threading.Lock()
_lib = None
_lib_error: str | None = None
_lib_loaded = False


def _load_lib():
    """Load libuvc once and declare every prototype we use.

    Returns ``(libuvc, None)`` on success or ``(None, message)`` when libuvc is
    unavailable, so callers can log the reason instead of raising on import.
    """
    global _lib, _lib_error, _lib_loaded
    if _lib_loaded:
        return _lib, _lib_error
    with _lib_lock:
        if _lib_loaded:
            return _lib, _lib_error
        _lib_loaded = True
        if not sys.platform.startswith("linux"):
            _lib_error = "libuvc capture is Linux-only; cv2's own backend is used"
            return None, _lib_error
        name = ctypes.util.find_library("uvc") or "libuvc.so.0"
        try:
            lib = ctypes.CDLL(name)
        except OSError as exc:
            _lib_error = f"libuvc not found ({exc}); install libuvc to use BSB cameras"
            logger.warning("libuvc unavailable: %s", _lib_error)
            return None, _lib_error

        lib.uvc_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        lib.uvc_init.restype = ctypes.c_int
        lib.uvc_exit.argtypes = [ctypes.c_void_p]
        lib.uvc_find_device.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
        ]
        lib.uvc_find_device.restype = ctypes.c_int
        lib.uvc_unref_device.argtypes = [ctypes.c_void_p]
        lib.uvc_open.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.uvc_open.restype = ctypes.c_int
        lib.uvc_close.argtypes = [ctypes.c_void_p]
        lib.uvc_get_stream_ctrl_format_size.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_StreamControl),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.uvc_get_stream_ctrl_format_size.restype = ctypes.c_int
        lib.uvc_stream_open_ctrl.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_StreamControl),
        ]
        lib.uvc_stream_open_ctrl.restype = ctypes.c_int
        lib.uvc_stream_start.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint8,
        ]
        lib.uvc_stream_start.restype = ctypes.c_int
        lib.uvc_stream_get_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(_Frame)),
            ctypes.c_int32,
        ]
        lib.uvc_stream_get_frame.restype = ctypes.c_int
        lib.uvc_stream_stop.argtypes = [ctypes.c_void_p]
        lib.uvc_stream_close.argtypes = [ctypes.c_void_p]
        lib.uvc_strerror.argtypes = [ctypes.c_int]
        lib.uvc_strerror.restype = ctypes.c_char_p

        _lib = lib
        _lib_error = None
        logger.info("libuvc loaded from %s", name)
        return _lib, None


def libuvc_available() -> bool:
    """True when the libuvc userspace stack can be used on this machine."""
    lib, _ = _load_lib()
    return lib is not None


def _usb_ids_for_video_node(node: str) -> tuple[int, int] | None:
    """Read the USB vendor/product IDs backing a ``videoN`` sysfs node."""
    directory = os.path.realpath(f"/sys/class/video4linux/{node}/device")
    if not directory:
        return None
    for _ in range(_USB_ID_WALK_DEPTH):
        try:
            with open(os.path.join(directory, "idVendor")) as fh:
                vendor = int(fh.read().strip(), 16)
            with open(os.path.join(directory, "idProduct")) as fh:
                product = int(fh.read().strip(), 16)
            return vendor, product
        except (OSError, ValueError):
            parent = os.path.dirname(directory)
            if parent == directory:
                break
            directory = parent
    return None


# Substrings that identify the Bigscreen Bigeye in a stored capture source: the
# OS device name ("Bigeye: Bigeye") and the by-id symlink ("usb-Bigscreen_Bigeye").
_BSB2E_SOURCE_MARKERS = ("bigeye", "bigscreen")


def _bsb2e_usb_attached(base: str = "/sys/bus/usb/devices") -> bool:
    """True when a ``35bd:0202`` device is on the USB bus, node or not.

    libuvc detaches the ``uvcvideo`` driver while it streams, which removes the
    camera's ``/dev/videoN`` nodes (and their by-id symlinks) for as long as it
    holds the interface. A crash there leaves them detached until the next
    clean open or a replug, so detection must not depend on the node existing.
    """
    try:
        entries = os.listdir(base)
    except OSError:
        return False
    for entry in entries:
        directory = os.path.join(base, entry)
        try:
            with open(os.path.join(directory, "idVendor")) as fh:
                if int(fh.read().strip(), 16) != BSB2E_VENDOR_ID:
                    continue
            with open(os.path.join(directory, "idProduct")) as fh:
                if int(fh.read().strip(), 16) == BSB2E_PRODUCT_ID:
                    return True
        except (OSError, ValueError):
            continue
    return False


def is_bsb2e_video_device(path: str | None) -> bool:
    """True if ``path`` is a video4linux node backed by a Bigscreen Bigeye.

    Accepts ``/dev/videoN`` or the ``/dev/v4l/by-id/...`` symlink the camera
    enumerator stores, so callers can test a configured capture source without
    first resolving it to an index.
    """
    if not path or not sys.platform.startswith("linux"):
        return False
    try:
        real = os.path.realpath(str(path))
    except OSError:
        return False
    if not real.startswith("/dev/video"):
        return False
    node = os.path.basename(real)
    return _usb_ids_for_video_node(node) == (BSB2E_VENDOR_ID, BSB2E_PRODUCT_ID)


def bsb2e_video_path(capture_source) -> str | None:
    """Return the ``/dev`` node for a BSB2E capture source, else ``None``.

    ``capture_source`` is either the raw ``/dev/videoN`` / by-id path or the
    app's ``uvc:<name>@<address>`` form. Only the address portion is inspected;
    the device name is deliberately ignored so a renamed camera still routes
    correctly.
    """
    if capture_source is None or isinstance(capture_source, int):
        return None
    value = str(capture_source)
    address = value
    if value.startswith("uvc:"):
        # Camera names may contain '@' (DirectShow does this); the address is
        # everything after the last one, matching parse_uvc_named_source.
        address = value[value.rfind("@") + 1 :]
    if not address.startswith("/dev/"):
        return None
    if is_bsb2e_video_device(address):
        return address
    # No resolvable video node. It may simply not exist yet, because libuvc
    # detached the kernel driver (see _bsb2e_usb_attached): fall back to the USB
    # identity plus the source's own naming, so a camera we already own is not
    # mistaken for a V4L2 camera. A raw /dev/videoN path is too ambiguous to
    # fall back on; it keeps the sysfs answer.
    if (
        sys.platform.startswith("linux")
        and any(marker in value.casefold() for marker in _BSB2E_SOURCE_MARKERS)
        and _bsb2e_usb_attached()
    ):
        return address
    return None


class LibuvcCapture:
    """One synchronous libuvc stream, owning libuvc's whole lifetime.

    ``open()`` negotiates the format and opens the stream; ``start()`` begins
    streaming; ``read()`` blocks (without the GIL) until the next frame and
    returns its raw MJPEG payload. Not thread-safe: one consumer at a time,
    which is how the capture loop uses it.
    """

    def __init__(
        self,
        vendor_id: int = BSB2E_VENDOR_ID,
        product_id: int = BSB2E_PRODUCT_ID,
        width: int = BSB2E_WIDTH,
        height: int = BSB2E_HEIGHT,
        fps: int = BSB2E_FPS,
    ):
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.width = width
        self.height = height
        self.fps = fps
        self._ctx = None
        self._device = None
        self._handle = None
        self._stream = None
        self._controller = None
        self._streaming = False
        self.received_frames = 0
        self.dropped_format_frames = 0

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        """Init libuvc, find the camera and negotiate the stream format."""
        lib, error = _load_lib()
        if lib is None:
            raise LibuvcError(error or "libuvc unavailable")

        ctx = ctypes.c_void_p()
        self._check(lib, lib.uvc_init(ctypes.byref(ctx), None), "uvc_init")
        self._ctx = ctx

        device = ctypes.c_void_p()
        rc = lib.uvc_find_device(
            ctx, ctypes.byref(device), self.vendor_id, self.product_id, None
        )
        if rc != _UVC_SUCCESS or not device:
            lib.uvc_exit(ctx)
            self._ctx = None
            raise LibuvcError(
                f"camera {self.vendor_id:04x}:{self.product_id:04x} not found "
                f"({self._error_text(lib, rc)})"
            )
        self._device = device

        handle = ctypes.c_void_p()
        try:
            self._check(lib, lib.uvc_open(device, ctypes.byref(handle)), "uvc_open")
        except LibuvcError:
            self.close()
            raise
        self._handle = handle

        controller = _StreamControl()
        try:
            self._check(
                lib,
                lib.uvc_get_stream_ctrl_format_size(
                    handle,
                    ctypes.byref(controller),
                    UVC_FRAME_FORMAT_MJPEG,
                    self.width,
                    self.height,
                    self.fps,
                ),
                "uvc_get_stream_ctrl_format_size",
            )
            self._controller = controller
            stream = ctypes.c_void_p()
            self._check(
                lib,
                lib.uvc_stream_open_ctrl(handle, ctypes.byref(stream), ctypes.byref(controller)),
                "uvc_stream_open_ctrl",
            )
        except LibuvcError:
            self.close()
            raise
        self._stream = stream
        logger.info(
            "libuvc: negotiated %dx%d @ %d fps MJPEG (interface %d, "
            "max frame %d bytes, max payload %d bytes)",
            self.width,
            self.height,
            self.fps,
            controller.bInterfaceNumber,
            controller.dwMaxVideoFrameSize,
            controller.dwMaxPayloadTransferSize,
        )

    def start(self) -> None:
        """Begin streaming.

        Passing no callback selects libuvc's polling mode, which is what makes
        ``read()`` work (``uvc_stream_get_frame`` rejects streams that have a
        callback).
        """
        if self._stream is None:
            raise LibuvcError("start() called before open()")
        lib, _ = _load_lib()
        self._check(
            lib,
            lib.uvc_stream_start(self._stream, None, None, 0),
            "uvc_stream_start",
        )
        self._streaming = True

    def read(self, timeout: float) -> bytes | None:
        """Block up to ``timeout`` seconds for the next frame's MJPEG payload.

        Returns ``None`` on timeout or when the stream has no frame for us.
        The GIL is released for the whole wait, so this is cheap to call in a
        loop and never spins.
        """
        stream = self._stream
        if stream is None or not self._streaming:
            return None
        lib, _ = _load_lib()
        frame_ptr = ctypes.POINTER(_Frame)()
        rc = lib.uvc_stream_get_frame(
            stream, ctypes.byref(frame_ptr), int(timeout * 1_000_000)
        )
        if rc != _UVC_SUCCESS or not frame_ptr:
            # Timeout, or the stream stopped underneath us.
            return None
        return self._payload_from_frame(frame_ptr.contents)

    def _payload_from_frame(self, frame) -> bytes | None:
        """Copy a frame's MJPEG payload; ``None`` for anything unusable.

        The buffer belongs to libuvc and is only valid until the next
        ``uvc_stream_get_frame`` call, so it has to be copied here.
        """
        if frame.frame_format != UVC_FRAME_FORMAT_MJPEG:
            self.dropped_format_frames += 1
            return None
        if not frame.data or not frame.data_bytes:
            return None
        self.received_frames += 1
        return ctypes.string_at(frame.data, frame.data_bytes)

    def stop(self) -> None:
        """Stop streaming. Safe to call when not streaming; never raises."""
        if not self._streaming:
            return
        self._streaming = False
        lib, _ = _load_lib()
        if lib is None or self._stream is None:
            return
        try:
            lib.uvc_stream_stop(self._stream)
        except Exception as exc:  # noqa: BLE001 - teardown must not raise
            logger.warning("libuvc stream_stop failed: %s", exc)

    def close(self) -> None:
        """Release every libuvc resource. Idempotent and never raises."""
        self.stop()
        lib, _ = _load_lib()
        if lib is None:
            self._forget()
            return
        if self._stream is not None:
            try:
                lib.uvc_stream_close(self._stream)
            except Exception as exc:  # noqa: BLE001
                logger.warning("libuvc stream_close failed: %s", exc)
        if self._handle is not None:
            try:
                lib.uvc_close(self._handle)
            except Exception as exc:  # noqa: BLE001
                logger.warning("libuvc uvc_close failed: %s", exc)
        if self._device is not None:
            try:
                lib.uvc_unref_device(self._device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("libuvc uvc_unref_device failed: %s", exc)
        if self._ctx is not None:
            try:
                lib.uvc_exit(self._ctx)
            except Exception as exc:  # noqa: BLE001
                logger.warning("libuvc uvc_exit failed: %s", exc)
        self._forget()

    def _forget(self) -> None:
        self._stream = None
        self._handle = None
        self._device = None
        self._ctx = None
        self._controller = None
        self._streaming = False

    def _check(self, lib, rc: int, what: str) -> None:
        if rc != _UVC_SUCCESS:
            raise LibuvcError(f"{what} failed: {self._error_text(lib, rc)}")

    @staticmethod
    def _error_text(lib, rc: int) -> str:
        try:
            text = lib.uvc_strerror(rc)
            if text:
                return f"{text.decode('utf-8', 'replace')} (code {rc})"
        except Exception:  # noqa: BLE001 - diagnostics must not mask the error
            pass
        return f"code {rc}"


def wait_for_first_frame(stream: LibuvcCapture, timeout: float) -> bool:
    """Wait for the camera's first frame; that frame is consumed.

    Used right after ``start()`` so a device that opens but never streams is
    treated as a failed start instead of a silent black feed.
    """
    return stream.read(timeout) is not None
