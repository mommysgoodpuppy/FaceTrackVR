"""Raw YUY2 capture for Windows cameras through a DirectShow graph."""

from __future__ import annotations

from ctypes import string_at
import threading
from typing import Callable

import numpy as np


YUY2_SUBTYPE = "{32595559-0000-0010-8000-00AA00389B71}"


def _select_yuy2_format(
    formats: list[dict], width: int, height: int, fps: float
) -> dict:
    candidates = [
        fmt
        for fmt in formats
        if fmt["media_type_str"] == "YUY2"
        and int(fmt["width"]) == width
        and abs(int(fmt["height"])) == height
    ]
    if not candidates:
        offered = sorted(
            {
                (fmt["media_type_str"], fmt["width"], fmt["height"])
                for fmt in formats
            }
        )
        raise RuntimeError(
            f"camera does not offer {width}x{height} YUY2; formats={offered}"
        )

    def distance(fmt: dict) -> float:
        rates = (float(fmt["min_framerate"]), float(fmt["max_framerate"]))
        return min(abs(rate - fps) for rate in rates)

    return min(candidates, key=distance)


def capture_yuy2(
    device_index: int,
    stop_event: threading.Event,
    on_frame: Callable[[np.ndarray], None],
    *,
    width: int = 400,
    height: int = 400,
    fps: float = 60.0,
) -> None:
    """Capture packed YUY2 samples until ``stop_event`` is set.

    OpenCV's DirectShow adapter always allocates a three-channel destination
    for YUY2, even with CAP_PROP_CONVERT_RGB disabled.  A native Sample Grabber
    preserves the camera's media sample byte-for-byte.
    """
    import comtypes
    from comtypes import COMObject
    from pygrabber.dshow_core import qedit
    from pygrabber.dshow_graph import FilterGraph, FilterType
    from pygrabber.dshow_ids import MediaTypes

    expected_bytes = width * height * 2

    class RawSampleCallback(COMObject):
        _com_interfaces_ = [qedit.ISampleGrabberCB]

        def __init__(self):
            self.error: Exception | None = None
            super().__init__()

        def SampleCB(self, this, sample_time, sample):
            return 0

        def BufferCB(self, this, sample_time, buffer, length):
            try:
                if int(length) != expected_bytes:
                    raise RuntimeError(
                        f"DirectShow returned {length} bytes; "
                        f"expected {expected_bytes} bytes of YUY2"
                    )
                # DirectShow owns and reuses the sample after this callback.
                # string_at makes the required owned copy before returning.
                frame = np.frombuffer(
                    string_at(buffer, length), dtype=np.uint8
                ).reshape(height, width * 2)
                on_frame(frame)
            except Exception as exc:  # COM callbacks must not leak exceptions.
                self.error = exc
                stop_event.set()
            return 0

    graph = None
    callback = RawSampleCallback()
    comtypes.CoInitialize()
    try:
        graph = FilterGraph()
        graph.add_video_input_device(int(device_index))
        video_input = graph.get_input_device()
        selected = _select_yuy2_format(
            video_input.get_formats(), width, height, fps
        )
        video_input.set_format(selected["index"])

        # pygrabber's public helper initially requests RGB24. Replace that
        # preference before connecting the graph so no converter is inserted.
        graph.add_sample_grabber(lambda _frame: None)
        grabber = graph.filters[FilterType.sample_grabber]
        grabber.set_callback(callback, 1)
        grabber.set_media_type(MediaTypes.Video, YUY2_SUBTYPE)
        graph.add_null_render()
        graph.prepare_preview_graph()
        graph.run()

        while not stop_event.wait(0.01):
            if callback.error is not None:
                raise callback.error
        if callback.error is not None:
            raise callback.error
    finally:
        if graph is not None:
            try:
                graph.stop()
            finally:
                graph.remove_filters()
        comtypes.CoUninitialize()
