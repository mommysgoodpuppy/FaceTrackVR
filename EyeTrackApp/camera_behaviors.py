from __future__ import annotations

import ctypes
import os
import re
import sys
import time
from collections.abc import Iterable


_VIDEO_INTERFACE_RE = re.compile(r"^(.*)-video-index(\d+)$")
_COLLAPSED_INTERFACE_CAMERAS = ("bigeye", "htc multimedia camera")
_HTC_STREAM_CAMERAS = ("htc multimedia camera",)

_UVCIOC_CTRL_QUERY = 0xC0107521
_XU_SET_CUR = 0x01
_XU_GET_CUR = 0x81
_XU_GET_LEN = 0x85
_XU_UNIT = 4
_XU_SELECTOR = 2

_HTC_REGISTER_WRITES = (
    (0x00, 0x40), (0x08, 0x01), (0x70, 0x00),
    (0x02, 0xFF), (0x03, 0xFF), (0x04, 0xFF),
    (0x0E, 0x00), (0x05, 0xB2), (0x06, 0xB2), (0x07, 0xB2), (0x0F, 0x03),
)


def collapse_camera_interfaces(cameras: Iterable[dict]) -> list[dict]:
    selected: list[dict] = []
    grouped: dict[tuple[str, str], tuple[int, int]] = {}
    for camera in cameras:
        name = str(camera.get("name", ""))
        address = str(camera.get("address", ""))
        match = _VIDEO_INTERFACE_RE.match(address)
        collapsible = any(
            marker in name.casefold()
            for marker in _COLLAPSED_INTERFACE_CAMERAS
        )
        if not collapsible or match is None:
            selected.append(camera)
            continue

        interface = int(match.group(2))
        key = (name.casefold(), match.group(1))
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = (len(selected), interface)
            selected.append(camera)
        elif interface < previous[1]:
            position, _old_interface = previous
            grouped[key] = (position, interface)
            selected[position] = camera
    return selected


def stream_controller_for(name: str, address: str):
    if not sys.platform.startswith("linux"):
        return None
    identity = re.sub(r"[^a-z0-9]+", " ", f"{name} {address}".casefold())
    if any(marker in identity for marker in _HTC_STREAM_CAMERAS):
        return HtcStreamController(address)
    return None


class _XuControlQuery(ctypes.Structure):
    _fields_ = [
        ("unit", ctypes.c_uint8),
        ("selector", ctypes.c_uint8),
        ("query", ctypes.c_uint8),
        ("size", ctypes.c_uint16),
        ("data", ctypes.c_void_p),
    ]


def _xu_query(fd, request, selector, payload):
    import fcntl

    if isinstance(payload, int):
        buf = ctypes.create_string_buffer(payload)
    else:
        buf = ctypes.create_string_buffer(bytes(payload), len(payload))
    query = _XuControlQuery(
        unit=_XU_UNIT,
        selector=selector,
        query=request,
        size=len(buf),
        data=ctypes.cast(buf, ctypes.c_void_p),
    )
    fcntl.ioctl(fd, _UVCIOC_CTRL_QUERY, query)
    return buf.raw


class HtcStreamController:
    def __init__(self, device: str):
        self.device = device
        self.enabled = False

    def set_enabled(self, enabled: bool) -> None:
        fd = os.open(self.device, os.O_RDWR)
        try:
            size_bytes = _xu_query(fd, _XU_GET_LEN, _XU_SELECTOR, 2)
            size = size_bytes[0] | (size_bytes[1] << 8)
            if size not in (384, 64):
                raise RuntimeError(f"unexpected XU buffer size {size}")

            def set_cur(payload):
                _xu_query(fd, _XU_SET_CUR, _XU_SELECTOR, payload)
                deadline = time.time() + 1.0
                while time.time() < deadline:
                    result = _xu_query(fd, _XU_GET_CUR, _XU_SELECTOR, size)
                    if result[0] == 0x55:
                        time.sleep(0.001)
                    elif result[0] == 0x56:
                        if result[1:17] != bytes(payload[:16]):
                            raise RuntimeError("XU response mismatch")
                        return
                    else:
                        raise RuntimeError(f"bad XU response {result[0]:#x}")
                raise TimeoutError("XU timeout")

            def magic():
                data = bytearray(size)
                data[0], data[1] = 0x51, 0x52
                if size >= 256:
                    data[254], data[255] = 0x53, 0x54
                set_cur(data)

            def enable_stream(on):
                data = bytearray(size)
                data[0:4] = bytes((0x50, 0x14, 0x00, int(on)))
                set_cur(data)

            def register(address, value):
                data = bytearray(size)
                data[0:5] = bytes((0x50, 0xAB, 0x60, 0x01, 0x01))
                data[8] = address
                data[9:13] = bytes((0x90, 0x01, 0x00, 0x01))
                data[16] = value
                set_cur(data)

            magic()
            enable_stream(False)
            time.sleep(0.1)
            magic()
            ir_led = 0xFF if enabled else 0x00
            for address, value in _HTC_REGISTER_WRITES:
                register(
                    address,
                    ir_led if address in (0x02, 0x03, 0x04) else value,
                )
            if enabled:
                time.sleep(0.1)
                magic()
                enable_stream(True)
            self.enabled = enabled
        finally:
            os.close(fd)

    def enable(self) -> None:
        self.set_enabled(True)

    def disable(self) -> None:
        if self.enabled:
            self.set_enabled(False)
