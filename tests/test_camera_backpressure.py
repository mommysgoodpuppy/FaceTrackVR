import logging
from queue import Queue

import numpy as np

import camera
from camera import Camera


def test_capture_backpressure_warning_is_aggregated(monkeypatch, caplog):
    capture = object.__new__(Camera)
    capture.serial_connection = None
    capture.camera_output_outgoing = Queue(maxsize=20)
    capture._extra_output_queues = []
    capture._capture_backpressure_events = 0
    capture._capture_backpressure_window_started = 0.0
    for item in (object(), object()):
        capture.camera_output_outgoing.put(item)

    times = iter((1.0, 2.0, 5.1))
    monkeypatch.setattr(camera.time, "monotonic", lambda: next(times))
    caplog.set_level(logging.WARNING)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    capture.push_image_to_queue(frame, 1, 90.0)
    capture.push_image_to_queue(frame, 2, 90.0)
    assert "Capture queue backpressure" not in caplog.text

    capture.push_image_to_queue(frame, 3, 90.0)
    assert "3 events in 5.1s" in caplog.text
    assert caplog.text.count("Capture queue backpressure") == 1
