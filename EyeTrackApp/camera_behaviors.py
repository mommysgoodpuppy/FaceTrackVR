from __future__ import annotations

import ctypes
import os
import re
import sys
import time
from collections.abc import Iterable
from functools import lru_cache


_VIDEO_INTERFACE_RE = re.compile(r"^(.*)-video-index(\d+)$")
_COLLAPSED_INTERFACE_CAMERAS = ("bigeye", "htc multimedia camera")
_HTC_STREAM_CAMERAS = (
    "htc multimedia camera",
    "vid 0bb4 pid 0581",
)

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


def stream_controller_for(
    name: str, address: str, *, device_index: int | None = None
):
    identity = re.sub(r"[^a-z0-9]+", " ", f"{name} {address}".casefold())
    if any(marker in identity for marker in _HTC_STREAM_CAMERAS):
        if sys.platform.startswith("linux"):
            return HtcStreamController(address)
        if sys.platform == "win32" and device_index is not None:
            return WindowsHtcStreamController(device_index)
    return None


def _run_htc_handshake(size: int, enabled: bool, set_cur) -> None:
    """Run the command sequence shared by the Linux and Windows transports."""

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

            _run_htc_handshake(size, enabled, set_cur)
            self.enabled = enabled
        finally:
            os.close(fd)

    def enable(self) -> None:
        self.set_enabled(True)

    def disable(self) -> None:
        if self.enabled:
            self.set_enabled(False)


# The HTC extension-unit property-set GUID recovered from LCameraModule.dll.
# Selector 2 is the same control used by the Linux UVCIOC_CTRL_QUERY path.
_HTC_XU_PROPERTY_SET = "{2CCB0BDA-6331-4FDB-850E-79054DBD5671}"
_KSNODETYPE_DEV_SPECIFIC = "{941C7AC0-C559-11D0-8A2B-00A0C9255AC1}"
_KSPROPERTY_TYPE_GET = 0x00000001
_KSPROPERTY_TYPE_SET = 0x00000002
_KSPROPERTY_TYPE_TOPOLOGY = 0x10000000


@lru_cache(maxsize=1)
def _windows_ks_types():
    """Create the small part of the KS COM API that comtypes does not ship."""
    if sys.platform != "win32":
        raise RuntimeError("Windows KS controls are only available on Windows")

    from ctypes import HRESULT, POINTER, Structure, c_void_p
    from ctypes.wintypes import DWORD, ULONG
    from comtypes import COMMETHOD, GUID, IUnknown

    class KsProperty(Structure):
        _fields_ = [
            ("Set", GUID),
            ("Id", ULONG),
            ("Flags", ULONG),
        ]

    class KsNode(Structure):
        _fields_ = [
            ("Property", KsProperty),
            ("NodeId", ULONG),
            ("Reserved", ULONG),
        ]

    class IKsControl(IUnknown):
        _iid_ = GUID("{28F54685-06FD-11D2-B27A-00A0C9223196}")
        _idlflags_ = []

    IKsControl._methods_ = [
        COMMETHOD(
            [], HRESULT, "KsProperty",
            (["in"], c_void_p, "property"),
            (["in"], ULONG, "property_length"),
            (["in", "out"], c_void_p, "data"),
            (["in"], ULONG, "data_length"),
            (["out"], POINTER(ULONG), "bytes_returned"),
        ),
        COMMETHOD(
            [], HRESULT, "KsMethod",
            (["in"], c_void_p, "method"),
            (["in"], ULONG, "method_length"),
            (["in", "out"], c_void_p, "data"),
            (["in"], ULONG, "data_length"),
            (["out"], POINTER(ULONG), "bytes_returned"),
        ),
        COMMETHOD(
            [], HRESULT, "KsEvent",
            (["in"], c_void_p, "event"),
            (["in"], ULONG, "event_length"),
            (["in", "out"], c_void_p, "event_data"),
            (["in"], ULONG, "data_length"),
            (["out"], POINTER(ULONG), "bytes_returned"),
        ),
    ]

    class IKsTopologyInfo(IUnknown):
        _iid_ = GUID("{720D4AC0-7533-11D0-A5D6-28DB04C10000}")
        _idlflags_ = []

    IKsTopologyInfo._methods_ = [
        COMMETHOD([], HRESULT, "get_NumCategories", (["out"], POINTER(DWORD), "count")),
        COMMETHOD(
            [], HRESULT, "get_Category",
            (["in"], DWORD, "index"),
            (["out"], POINTER(GUID), "category"),
        ),
        COMMETHOD([], HRESULT, "get_NumConnections", (["out"], POINTER(DWORD), "count")),
        COMMETHOD(
            [], HRESULT, "get_ConnectionInfo",
            (["in"], DWORD, "index"),
            (["out"], c_void_p, "connection"),
        ),
        COMMETHOD(
            [], HRESULT, "get_NodeName",
            (["in"], DWORD, "node"),
            (["out"], c_void_p, "name"),
            (["in"], DWORD, "name_chars"),
            (["out"], POINTER(DWORD), "required_chars"),
        ),
        COMMETHOD([], HRESULT, "get_NumNodes", (["out"], POINTER(DWORD), "count")),
        COMMETHOD(
            [], HRESULT, "get_NodeType",
            (["in"], DWORD, "node"),
            (["out"], POINTER(GUID), "node_type"),
        ),
        COMMETHOD(
            [], HRESULT, "CreateNodeInstance",
            (["in"], DWORD, "node"),
            (["in"], POINTER(GUID), "iid"),
            (["out"], POINTER(POINTER(IKsControl)), "instance"),
        ),
    ]
    return GUID, KsNode, IKsControl, IKsTopologyInfo


class WindowsHtcStreamController:
    """HTC UVC extension-unit transport through DirectShow/IKsControl."""

    def __init__(self, device_index: int):
        self.device_index = int(device_index)
        self.enabled = False

    @staticmethod
    def _ks_property(control, node_id, payload, *, get):
        from ctypes import addressof, c_void_p, sizeof

        GUID, KsNode, _IKsControl, _IKsTopologyInfo = _windows_ks_types()
        request = KsNode()
        request.Property.Set = GUID(_HTC_XU_PROPERTY_SET)
        request.Property.Id = _XU_SELECTOR
        request.Property.Flags = (
            (_KSPROPERTY_TYPE_GET if get else _KSPROPERTY_TYPE_SET)
            | _KSPROPERTY_TYPE_TOPOLOGY
        )
        request.NodeId = node_id
        request.Reserved = 0
        if isinstance(payload, int):
            buffer = ctypes.create_string_buffer(payload)
        else:
            buffer = ctypes.create_string_buffer(bytes(payload), len(payload))
        control.KsProperty(
            c_void_p(addressof(request)),
            sizeof(request),
            c_void_p(addressof(buffer)),
            len(buffer),
        )
        return buffer.raw

    def _with_control(self, callback):
        import comtypes
        from comtypes import GUID
        from pygrabber.dshow_graph import FilterGraph

        _GUID, _KsNode, IKsControl, IKsTopologyInfo = _windows_ks_types()
        graph = None
        source = None
        topology = None
        control = None
        initialized = False
        try:
            comtypes.CoInitialize()
            initialized = True
            graph = FilterGraph()
            graph.add_video_input_device(self.device_index)
            source = graph.get_input_device().instance
            topology = source.QueryInterface(IKsTopologyInfo)
            expected_type = GUID(_KSNODETYPE_DEV_SPECIFIC)
            for node_id in range(int(topology.get_NumNodes())):
                if topology.get_NodeType(node_id) != expected_type:
                    continue
                control = topology.CreateNodeInstance(
                    node_id, ctypes.byref(IKsControl._iid_)
                )
                result = callback(control, node_id)
                return result
            raise RuntimeError("HTC extension-unit node was not found")
        finally:
            # Release node/filter COM references before CoUninitialize.
            control = None
            topology = None
            source = None
            if graph is not None:
                try:
                    graph.remove_filters()
                except Exception:
                    pass
            if initialized:
                comtypes.CoUninitialize()

    def set_enabled(self, enabled: bool) -> None:
        def run(control, node_id):
            last_error = None
            # SRanipal probes the original 384-byte protocol first, then the
            # 64-byte firmware variant. Running the harmless magic transaction
            # is also the most reliable way to select the correct payload size.
            for size in (384, 64):
                try:
                    def set_cur(payload):
                        self._ks_property(control, node_id, payload, get=False)
                        deadline = time.time() + 1.0
                        while time.time() < deadline:
                            result = self._ks_property(
                                control, node_id, size, get=True
                            )
                            if result[0] == 0x55:
                                time.sleep(0.001)
                            elif result[0] == 0x56:
                                if result[1:17] != bytes(payload[:16]):
                                    raise RuntimeError("XU response mismatch")
                                return
                            else:
                                raise RuntimeError(
                                    f"bad XU response {result[0]:#x}"
                                )
                        raise TimeoutError("XU timeout")

                    _run_htc_handshake(size, enabled, set_cur)
                    return
                except Exception as exc:
                    last_error = exc
            raise RuntimeError(
                f"HTC extension-unit handshake failed: {last_error}"
            ) from last_error

        self._with_control(run)
        self.enabled = enabled

    def enable(self) -> None:
        self.set_enabled(True)

    def disable(self) -> None:
        if self.enabled:
            self.set_enabled(False)
