"""Self-contained userspace UVC capture for the Bigscreen Bigeye (BSB2E) eye cameras.

The Beyond 2E eye-tracking camera (USB ``35bd:0202``) advertises an MJPEG
800x400@90 format whose ``dwMaxVideoFrameSize`` the Linux kernel ``uvcvideo``
driver reads as zero. ``VIDIOC_REQBUFS`` therefore fails with ``Invalid
argument`` (OpenCV logs "no support for memory mapping" / "Insufficient buffer
memory") and V4L2 capture never starts, even though the camera streams fine on
Windows and through userspace UVC stacks.

Instead of requiring an external ``libuvc`` shared library, this module drives
Linux's kernel ``usbdevfs`` interface (``/dev/bus/usb/BBB/DDD``) directly via
``ioctl`` and ``select.poll``:

1. Locate ``35bd:0202`` in ``/sys/bus/usb/devices`` and open its usbdevfs node.
2. Detach ``uvcvideo`` and claim VideoControl (0) and VideoStreaming (1).
3. Negotiate UVC 1.1 ``VS_PROBE_CONTROL`` / ``VS_COMMIT_CONTROL`` for MJPEG
   800x400@90 (ignoring the camera's zero ``dwMaxVideoFrameSize``).
4. Select VideoStreaming alternate setting 1 (isochronous EP ``0x82``, 1024 B)
   and keep a ring of asynchronous isochronous URBs in flight.
5. Reap completed URBs synchronously in ``read()`` using ``select.poll`` (which
   releases the GIL while waiting) and reassemble UVC payload headers into
   complete MJPEG frames.
6. On ``close()``, discard/reap in-flight URBs, switch back to altsetting 0,
   release both interfaces, and reattach ``uvcvideo`` so ``/dev/videoN``
   reappears cleanly.

Only Linux needs this: Windows/macOS are served by cv2's own backends, so the
module is inert elsewhere and simply reports ``libuvc_available() == False``.
"""

from __future__ import annotations

from collections import deque
import ctypes
import errno
import logging
import os
import select
import struct
import sys
import time

logger = logging.getLogger(__name__)

# Bigscreen Bigeye (Beyond 2E eye tracking camera).
BSB2E_VENDOR_ID = 0x35BD
BSB2E_PRODUCT_ID = 0x0202
# The single format the camera exposes (see `v4l2-ctl --list-formats-ext`).
BSB2E_WIDTH = 800
BSB2E_HEIGHT = 400
BSB2E_FPS = 90

# Kept for compatibility with tests/callers inspecting frame format constants.
UVC_FRAME_FORMAT_MJPEG = 7

# Bigeye USB interface / endpoint layout (from USB descriptors).
_VC_INTERFACE = 0
_VS_INTERFACE = 1
_VS_ALTSETTING_IDLE = 0
_VS_ALTSETTING_STREAM = 1
_VS_ENDPOINT_IN = 0x82
_VS_FORMAT_INDEX = 1
_VS_FRAME_INDEX = 1

# UVC 1.1 class-specific VideoStreaming requests.
_UVC_REQ_TYPE_OUT = 0x21  # Host-to-device | Class | Interface
_UVC_REQ_TYPE_IN = 0xA1  # Device-to-host | Class | Interface
_UVC_SET_CUR = 0x01
_UVC_GET_CUR = 0x81
_UVC_VS_PROBE_CONTROL = 0x0100
_UVC_VS_COMMIT_CONTROL = 0x0200
_UVC_PROBE_LEN = 34  # UVC 1.1 probe/commit control block length
_UVC_CTRL_TIMEOUT_MS = 1000

# UVC payload header bitfield masks (byte 1: bmHeaderInfo).
_UVC_STREAM_FID = 0x01
_UVC_STREAM_EOF = 0x02
_UVC_STREAM_ERR = 0x40

# Isochronous URB ring sizing: 32 packets * 125 us = 4 ms per URB,
# 8 URBs = 32 ms of kernel-side buffering.
_NUM_URBS = 8
_PACKETS_PER_URB = 32
_ISO_PACKET_SIZE = 1024
# Hard cap on a single assembled MJPEG frame (normal frames are ~18 KB;
# descriptor dwMaxVideoFrameBufferSize is 640 KB).
_MAX_FRAME_BYTES = 1024 * 1024

# How far to walk up the sysfs device chain looking for idVendor/idProduct.
# A video4linux node points at the USB *interface* directory, so the USB
# device directory (which carries the IDs) is one or two levels up.
_USB_ID_WALK_DEPTH = 6

# Linux usbdevfs constants (<linux/usbdevice_fs.h>).
_USBDEVFS_URB_TYPE_ISO = 0
_USBDEVFS_URB_ISO_ASAP = 0x02

_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2


def _ioc(direction: int, ioc_type: str, nr: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (ord(ioc_type) << 8) | nr


def _io(ioc_type: str, nr: int) -> int:
    return _ioc(_IOC_NONE, ioc_type, nr, 0)


def _ior(ioc_type: str, nr: int, size: int) -> int:
    return _ioc(_IOC_READ, ioc_type, nr, size)


def _iow(ioc_type: str, nr: int, size: int) -> int:
    return _ioc(_IOC_WRITE, ioc_type, nr, size)


def _iowr(ioc_type: str, nr: int, size: int) -> int:
    return _ioc(_IOC_READ | _IOC_WRITE, ioc_type, nr, size)


class LibuvcError(RuntimeError):
    """Raised when userspace UVC capture fails or the device is unavailable."""


class _CtrlTransfer(ctypes.Structure):
    """``struct usbdevfs_ctrltransfer``."""

    _fields_ = [
        ("bRequestType", ctypes.c_uint8),
        ("bRequest", ctypes.c_uint8),
        ("wValue", ctypes.c_uint16),
        ("wIndex", ctypes.c_uint16),
        ("wLength", ctypes.c_uint16),
        ("timeout", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
    ]


class _SetInterface(ctypes.Structure):
    """``struct usbdevfs_setinterface``."""

    _fields_ = [
        ("interface", ctypes.c_uint),
        ("altsetting", ctypes.c_uint),
    ]


class _DisconnectClaim(ctypes.Structure):
    """``struct usbdevfs_disconnect_claim``."""

    _fields_ = [
        ("interface", ctypes.c_uint),
        ("flags", ctypes.c_uint),
        ("driver", ctypes.c_char * 256),
    ]


class _UsbIoctl(ctypes.Structure):
    """``struct usbdevfs_ioctl``."""

    _fields_ = [
        ("ifno", ctypes.c_int),
        ("ioctl_code", ctypes.c_int),
        ("data", ctypes.c_void_p),
    ]


class _IsoPacketDesc(ctypes.Structure):
    """``struct usbdevfs_iso_packet_desc``."""

    _fields_ = [
        ("length", ctypes.c_uint),
        ("actual_length", ctypes.c_uint),
        ("status", ctypes.c_uint),
    ]


class _UrbHeader(ctypes.Structure):
    """Fixed header of ``struct usbdevfs_urb`` (used for ioctl size)."""

    _fields_ = [
        ("type", ctypes.c_ubyte),
        ("endpoint", ctypes.c_ubyte),
        ("status", ctypes.c_int),
        ("flags", ctypes.c_uint),
        ("buffer", ctypes.c_void_p),
        ("buffer_length", ctypes.c_int),
        ("actual_length", ctypes.c_int),
        ("start_frame", ctypes.c_int),
        ("number_of_packets", ctypes.c_int),
        ("error_count", ctypes.c_int),
        ("signr", ctypes.c_uint),
        ("usercontext", ctypes.c_void_p),
    ]


class _IsoUrb(ctypes.Structure):
    """``struct usbdevfs_urb`` with ``_PACKETS_PER_URB`` trailing iso descs."""

    _fields_ = _UrbHeader._fields_ + [
        ("iso_frame_desc", _IsoPacketDesc * _PACKETS_PER_URB)
    ]


_USBDEVFS_CONTROL = _iowr("U", 0, ctypes.sizeof(_CtrlTransfer))
_USBDEVFS_SETINTERFACE = _ior("U", 4, ctypes.sizeof(_SetInterface))
_USBDEVFS_SUBMITURB = _ior("U", 10, ctypes.sizeof(_UrbHeader))
_USBDEVFS_DISCARDURB = _io("U", 11)
_USBDEVFS_REAPURBNDELAY = _iow("U", 13, ctypes.sizeof(ctypes.c_void_p))
_USBDEVFS_RELEASEINTERFACE = _ior("U", 16, ctypes.sizeof(ctypes.c_uint))
_USBDEVFS_IOCTL = _iowr("U", 18, ctypes.sizeof(_UsbIoctl))
_USBDEVFS_CONNECT = _io("U", 23)
_USBDEVFS_DISCONNECT_CLAIM = _ior("U", 27, ctypes.sizeof(_DisconnectClaim))

_DISCONNECT_ERRNOS = frozenset(
    {
        errno.ENODEV,
        errno.ESHUTDOWN,
        errno.ENOENT,
        errno.EPROTO,
        errno.ENXIO,
    }
)

_libc = None


def _get_libc():
    """Return a cached libc handle with ``ioctl`` configured and ``use_errno=True``.

    We call ``libc.ioctl`` instead of ``fcntl.ioctl`` because Python's
    ``fcntl.ioctl`` copies buffers <= 1024 bytes onto a temporary C stack
    buffer. ``USBDEVFS_SUBMITURB`` stores the userspace pointer to the URB
    struct and writes packet completion lengths back to that address, so the
    kernel must receive the real stable address of our ``_IsoUrb`` instance.
    """
    global _libc
    if _libc is None:
        lib = ctypes.CDLL(None, use_errno=True)
        lib.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]
        lib.ioctl.restype = ctypes.c_int
        _libc = lib
    return _libc


def _ioctl(fd: int, request: int, arg) -> int:
    """Invoke ``ioctl(fd, request, arg)`` and raise ``OSError`` on failure."""
    rc = _get_libc().ioctl(fd, request, arg)
    if rc < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return rc


def libuvc_available() -> bool:
    """True when the built-in Linux usbdevfs UVC backend can be used."""
    return sys.platform.startswith("linux")


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


def _find_usb_device_path(
    vendor_id: int = BSB2E_VENDOR_ID,
    product_id: int = BSB2E_PRODUCT_ID,
    base: str = "/sys/bus/usb/devices",
) -> str | None:
    """Return ``/dev/bus/usb/BBB/DDD`` for the matching USB device, or ``None``."""
    try:
        entries = os.listdir(base)
    except OSError:
        return None
    for entry in entries:
        directory = os.path.join(base, entry)
        try:
            with open(os.path.join(directory, "idVendor")) as fh:
                if int(fh.read().strip(), 16) != vendor_id:
                    continue
            with open(os.path.join(directory, "idProduct")) as fh:
                if int(fh.read().strip(), 16) != product_id:
                    continue
            with open(os.path.join(directory, "busnum")) as fh:
                busnum = int(fh.read().strip())
            with open(os.path.join(directory, "devnum")) as fh:
                devnum = int(fh.read().strip())
            return f"/dev/bus/usb/{busnum:03d}/{devnum:03d}"
        except (OSError, ValueError):
            continue
    return None


def _bsb2e_usb_attached(base: str = "/sys/bus/usb/devices") -> bool:
    """True when a ``35bd:0202`` device is on the USB bus, node or not.

    Userspace UVC detaches the ``uvcvideo`` driver while it streams, which
    removes the camera's ``/dev/videoN`` nodes (and their by-id symlinks) for
    as long as it holds the interface. A crash there leaves them detached until
    the next clean open or a replug, so detection must not depend on the node
    existing.
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
    # No resolvable video node. It may simply not exist yet, because userspace
    # capture detached the kernel driver (see _bsb2e_usb_attached): fall back
    # to the USB identity plus the source's own naming, so a camera we already
    # own is not mistaken for a V4L2 camera. A raw /dev/videoN path is too
    # ambiguous to fall back on; it keeps the sysfs answer.
    if (
        sys.platform.startswith("linux")
        and any(marker in value.casefold() for marker in _BSB2E_SOURCE_MARKERS)
        and _bsb2e_usb_attached()
    ):
        return address
    return None


class _UvcFrameAssembler:
    """Reassemble MJPEG by JPEG markers; Bigeye UVC frame flags can drift."""

    def __init__(self, max_frame_bytes: int = _MAX_FRAME_BYTES) -> None:
        self._max_frame_bytes = max_frame_bytes
        self._buf = bytearray()
        self._cur_fid: int = -1
        self._frame_bad: bool = False
        self.ready: deque[bytes] = deque()
        self.received_frames = 0
        self.dropped_error_frames = 0

    def reset(self) -> None:
        self._buf.clear()
        self._cur_fid = -1
        self._frame_bad = False
        self.ready.clear()

    def feed_packet(self, packet: bytes | memoryview) -> None:
        """Process one UVC payload packet (header + optional MJPEG data)."""
        alen = len(packet)
        if alen < 2:
            return
        hlen = packet[0]
        bfh = packet[1]
        if hlen < 2 or hlen > alen:
            return

        # Bigeye's FID/EOF flags can drift into the middle of a JPEG while
        # the payload remains intact. Use JPEG markers across packet boundaries
        # rather than flushing on those unreliable UVC frame boundaries.
        if bfh & _UVC_STREAM_ERR:
            if not self._frame_bad:
                self.dropped_error_frames += 1
            self._frame_bad = True
            self._buf.clear()
            return

        self._buf.extend(packet[hlen:alen])
        while self._buf:
            start = self._buf.find(b"\xff\xd8")
            if start < 0:
                # Keep a possible first byte of a split SOI marker.
                self._buf[:] = b"\xff" if self._buf[-1] == 0xff else b""
                return
            if start:
                del self._buf[:start]
            self._frame_bad = False
            end = self._buf.find(b"\xff\xd9", 2)
            next_start = self._buf.find(b"\xff\xd8", 2)
            if next_start >= 0 and (end < 0 or next_start < end):
                # A new JPEG before EOI means a damaged/incomplete frame.
                self.dropped_error_frames += 1
                del self._buf[:next_start]
                continue
            if end >= 0:
                frame_size = end + 2
                if frame_size <= self._max_frame_bytes:
                    self.ready.append(bytes(self._buf[:frame_size]))
                    self.received_frames += 1
                else:
                    self.dropped_error_frames += 1
                del self._buf[:frame_size]
                continue
            if len(self._buf) > self._max_frame_bytes:
                self.dropped_error_frames += 1
                self._buf[:] = b"\xff" if self._buf[-1] == 0xff else b""
            return


class LibuvcCapture:
    """Self-contained synchronous Linux usbdevfs UVC stream for the Bigeye.

    ``open()`` claims the USB interfaces and negotiates the UVC 1.1 format;
    ``start()`` selects the isochronous alternate setting and submits the URB
    ring; ``read()`` blocks (without the GIL, via ``select.poll``) until the
    next MJPEG frame completes and returns its bytes. Not thread-safe: one
    consumer at a time, which is how the capture loop uses it.
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
        self._fd: int | None = None
        self._dev_path: str | None = None
        self._poller: select.poll | None = None
        self._claimed_interfaces: list[int] = []
        self._urbs: list[tuple[_IsoUrb, bytearray, ctypes.Array, memoryview]] = []
        self._urb_map: dict[int, tuple[_IsoUrb, bytearray, ctypes.Array, memoryview]] = {}
        self._submitted_urbs: set[int] = set()
        self._streaming = False
        self._assembler = _UvcFrameAssembler()
        self.dropped_format_frames = 0

    @property
    def received_frames(self) -> int:
        return self._assembler.received_frames

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        """Open the USB device, claim interfaces, and negotiate MJPEG 800x400@90."""
        if not libuvc_available():
            raise LibuvcError(
                "userspace BSB2E capture is Linux-only; cv2's own backend is used"
            )

        dev_path = _find_usb_device_path(self.vendor_id, self.product_id)
        if dev_path is None:
            raise LibuvcError(
                f"camera {self.vendor_id:04x}:{self.product_id:04x} not found on USB bus"
            )

        try:
            fd = os.open(dev_path, os.O_RDWR)
        except PermissionError as exc:
            raise LibuvcError(
                f"permission denied opening {dev_path} ({exc}); "
                f"check udev rules for {self.vendor_id:04x}:{self.product_id:04x}"
            ) from exc
        except OSError as exc:
            raise LibuvcError(f"failed to open {dev_path}: {exc}") from exc

        self._fd = fd
        self._dev_path = dev_path

        try:
            for ifno in (_VC_INTERFACE, _VS_INTERFACE):
                dc = _DisconnectClaim(interface=ifno, flags=0, driver=b"")
                _ioctl(fd, _USBDEVFS_DISCONNECT_CLAIM, ctypes.byref(dc))
                self._claimed_interfaces.append(ifno)

            max_frame, max_payload = self._negotiate_uvc_stream(fd)
        except OSError as exc:
            self.close()
            raise LibuvcError(
                f"USB setup failed on {dev_path}: {exc}"
            ) from exc
        except Exception:
            self.close()
            raise

        poller = select.poll()
        poller.register(fd, select.POLLOUT)
        self._poller = poller

        logger.info(
            "usbdevfs UVC: negotiated %dx%d @ %d fps MJPEG on %s "
            "(interface %d, max frame %d bytes, max payload %d bytes)",
            self.width,
            self.height,
            self.fps,
            dev_path,
            _VS_INTERFACE,
            max_frame,
            max_payload,
        )

    def _negotiate_uvc_stream(self, fd: int) -> tuple[int, int]:
        """Run UVC 1.1 VS_PROBE_CONTROL (SET/GET) and VS_COMMIT_CONTROL (SET)."""
        frame_interval = max(1, int(round(10_000_000 / self.fps)))
        probe = bytearray(_UVC_PROBE_LEN)
        # bmHint=1 (dwFrameInterval), bFormatIndex=1 (MJPEG), bFrameIndex=1 (800x400)
        struct.pack_into(
            "<HBBI",
            probe,
            0,
            1,
            _VS_FORMAT_INDEX,
            _VS_FRAME_INDEX,
            frame_interval,
        )
        cbuf = (ctypes.c_ubyte * _UVC_PROBE_LEN).from_buffer(probe)
        buf_addr = ctypes.addressof(cbuf)

        for req_type, req, wvalue in (
            (_UVC_REQ_TYPE_OUT, _UVC_SET_CUR, _UVC_VS_PROBE_CONTROL),
            (_UVC_REQ_TYPE_IN, _UVC_GET_CUR, _UVC_VS_PROBE_CONTROL),
            (_UVC_REQ_TYPE_OUT, _UVC_SET_CUR, _UVC_VS_COMMIT_CONTROL),
        ):
            ctrl = _CtrlTransfer(
                bRequestType=req_type,
                bRequest=req,
                wValue=wvalue,
                wIndex=_VS_INTERFACE,
                wLength=_UVC_PROBE_LEN,
                timeout=_UVC_CTRL_TIMEOUT_MS,
                data=buf_addr,
            )
            _ioctl(fd, _USBDEVFS_CONTROL, ctypes.byref(ctrl))

        max_frame, max_payload = struct.unpack_from("<II", probe, 18)
        return max_frame, max_payload

    def start(self) -> None:
        """Select the streaming alternate setting and submit the isochronous URB ring."""
        if self._fd is None:
            raise LibuvcError("start() called before open()")
        if self._streaming:
            return

        fd = self._fd
        self._assembler.reset()
        try:
            si = _SetInterface(
                interface=_VS_INTERFACE, altsetting=_VS_ALTSETTING_STREAM
            )
            _ioctl(fd, _USBDEVFS_SETINTERFACE, ctypes.byref(si))

            buf_size = _PACKETS_PER_URB * _ISO_PACKET_SIZE
            for idx in range(_NUM_URBS):
                data = bytearray(buf_size)
                cdata = (ctypes.c_ubyte * buf_size).from_buffer(data)
                mv = memoryview(data)
                urb = _IsoUrb()
                urb.type = _USBDEVFS_URB_TYPE_ISO
                urb.endpoint = _VS_ENDPOINT_IN
                urb.flags = _USBDEVFS_URB_ISO_ASAP
                urb.buffer = ctypes.addressof(cdata)
                urb.buffer_length = buf_size
                urb.number_of_packets = _PACKETS_PER_URB
                urb.usercontext = idx
                for p in range(_PACKETS_PER_URB):
                    urb.iso_frame_desc[p].length = _ISO_PACKET_SIZE
                addr = ctypes.addressof(urb)
                self._urbs.append((urb, data, cdata, mv))
                self._urb_map[addr] = (urb, data, cdata, mv)
                _ioctl(fd, _USBDEVFS_SUBMITURB, ctypes.byref(urb))
                self._submitted_urbs.add(addr)
        except OSError as exc:
            self._streaming = True  # let stop() unwind any partially submitted URBs
            self.stop()
            raise LibuvcError(f"failed to start USB isochronous stream: {exc}") from exc

        self._streaming = True

    def read(self, timeout: float) -> bytes | None:
        """Block up to ``timeout`` seconds for the next frame's MJPEG payload.

        Returns ``None`` on timeout or when not streaming. Raises
        ``LibuvcError`` if the USB device disconnects or errors out.
        """
        if self._fd is None or not self._streaming or self._poller is None:
            return None
        if self._assembler.ready:
            return self._assembler.ready.popleft()

        fd = self._fd
        poller = self._poller
        deadline = time.monotonic() + max(0.0, timeout)

        while True:
            remaining_ms = int(max(0.0, (deadline - time.monotonic()) * 1000.0))
            events = poller.poll(remaining_ms)
            if not events:
                return None

            _ev_fd, ev_mask = events[0]
            if ev_mask & (select.POLLERR | select.POLLHUP | select.POLLNVAL):
                raise LibuvcError("USB device disconnected")

            self._reap_available_urbs(fd)
            if self._assembler.ready:
                return self._assembler.ready.popleft()
            if time.monotonic() >= deadline:
                return None

    def _reap_available_urbs(self, fd: int) -> None:
        """Reap all completed URBs, feed packets into the assembler, and resubmit."""
        while True:
            ptr = ctypes.c_void_p()
            try:
                _ioctl(fd, _USBDEVFS_REAPURBNDELAY, ctypes.byref(ptr))
            except OSError as exc:
                if exc.errno == errno.EAGAIN:
                    break
                if exc.errno in _DISCONNECT_ERRNOS:
                    raise LibuvcError(f"USB stream lost ({exc})") from exc
                raise LibuvcError(f"USB reap failed ({exc})") from exc

            addr = ptr.value
            self._submitted_urbs.discard(addr)
            entry = self._urb_map.get(addr)
            if entry is None:
                continue
            urb, _data, _cdata, mv = entry

            if urb.status in (-errno.ENODEV, -errno.ESHUTDOWN):
                raise LibuvcError(f"USB device disconnected (urb status {urb.status})")

            for p in range(_PACKETS_PER_URB):
                desc = urb.iso_frame_desc[p]
                alen = desc.actual_length
                if desc.status != 0 or alen < 2:
                    continue
                off = p * _ISO_PACKET_SIZE
                self._assembler.feed_packet(mv[off : off + alen])

            # Reset packet descriptors before resubmitting.
            for p in range(_PACKETS_PER_URB):
                desc = urb.iso_frame_desc[p]
                desc.length = _ISO_PACKET_SIZE
                desc.actual_length = 0
                desc.status = 0

            try:
                _ioctl(fd, _USBDEVFS_SUBMITURB, ctypes.byref(urb))
                self._submitted_urbs.add(addr)
            except OSError as exc:
                raise LibuvcError(f"USB URB resubmit failed ({exc})") from exc

    def stop(self) -> None:
        """Stop streaming and reap all in-flight URBs. Safe to call anytime."""
        if not self._streaming and not self._submitted_urbs:
            return
        self._streaming = False
        fd = self._fd
        if fd is not None:
            for addr in list(self._submitted_urbs):
                entry = self._urb_map.get(addr)
                if entry is None:
                    continue
                try:
                    _ioctl(fd, _USBDEVFS_DISCARDURB, ctypes.byref(entry[0]))
                except OSError:
                    pass
            # Drain discarded/completed URBs so the kernel cannot write to our
            # buffers after we drop references to them.
            drain_deadline = time.monotonic() + 0.5
            while self._submitted_urbs and time.monotonic() < drain_deadline:
                ptr = ctypes.c_void_p()
                try:
                    _ioctl(fd, _USBDEVFS_REAPURBNDELAY, ctypes.byref(ptr))
                    self._submitted_urbs.discard(ptr.value)
                except OSError as exc:
                    if exc.errno == errno.EAGAIN:
                        if self._poller is not None:
                            self._poller.poll(10)
                        else:
                            time.sleep(0.005)
                    else:
                        break
            try:
                si = _SetInterface(
                    interface=_VS_INTERFACE, altsetting=_VS_ALTSETTING_IDLE
                )
                _ioctl(fd, _USBDEVFS_SETINTERFACE, ctypes.byref(si))
            except OSError as exc:
                if exc.errno not in _DISCONNECT_ERRNOS:
                    logger.warning("usbdevfs set altsetting 0 failed: %s", exc)

        for _urb, _data, _cdata, mv in self._urbs:
            try:
                mv.release()
            except Exception:
                pass
        self._urbs.clear()
        self._urb_map.clear()
        self._submitted_urbs.clear()
        self._assembler.reset()

    def close(self) -> None:
        """Release every USB resource and reattach uvcvideo. Idempotent."""
        self.stop()
        fd = self._fd
        self._fd = None
        self._poller = None
        self._dev_path = None
        claimed = list(self._claimed_interfaces)
        self._claimed_interfaces.clear()

        if fd is None:
            return

        for ifno in reversed(claimed):
            try:
                cif = ctypes.c_uint(ifno)
                _ioctl(fd, _USBDEVFS_RELEASEINTERFACE, ctypes.byref(cif))
            except OSError as exc:
                if exc.errno not in _DISCONNECT_ERRNOS:
                    logger.warning("usbdevfs release interface %d failed: %s", ifno, exc)

        # Reattach the kernel uvcvideo driver so /dev/videoN nodes return.
        for ifno in claimed:
            try:
                ui = _UsbIoctl(ifno=ifno, ioctl_code=_USBDEVFS_CONNECT, data=None)
                _ioctl(fd, _USBDEVFS_IOCTL, ctypes.byref(ui))
            except OSError:
                pass

        try:
            os.close(fd)
        except OSError as exc:
            logger.warning("usbdevfs close failed: %s", exc)


def wait_for_first_frame(stream: LibuvcCapture, timeout: float) -> bool:
    """Wait for the camera's first frame; that frame is consumed.

    Used right after ``start()`` so a device that opens but never streams is
    treated as a failed start instead of a silent black feed.
    """
    return stream.read(timeout) is not None
