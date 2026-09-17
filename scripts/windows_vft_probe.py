"""Probe direct Vive Facial Tracker capture on Windows.

Run from the repository root with the Windows project environment active:

    python scripts/windows_vft_probe.py

The probe performs the HTC extension-unit startup sequence, captures raw
400x400 YUY2 samples through a native DirectShow graph, and always attempts
to turn the emitters off before exiting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import threading
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "EyeTrackApp"
sys.path.insert(0, str(APP_DIR))

from camera_behaviors import stream_controller_for  # noqa: E402
from camera_enum import list_uvc_cameras  # noqa: E402
from windows_dshow_capture import capture_yuy2  # noqa: E402


def _is_vft(camera: dict) -> bool:
    identity = f"{camera.get('name', '')} {camera.get('address', '')}".casefold()
    return (
        "htc multimedia camera" in identity
        or ("vid_0bb4" in identity and "pid_0581" in identity)
    )


def _choose_camera(cameras: list[dict], requested_index: int | None) -> dict:
    if requested_index is not None:
        for camera in cameras:
            if int(camera["index"]) == requested_index:
                return camera
        raise RuntimeError(f"camera index {requested_index} was not found")

    matches = [camera for camera in cameras if _is_vft(camera)]
    if not matches:
        raise RuntimeError("HTC Multimedia Camera was not found")
    if len(matches) > 1:
        indices = ", ".join(str(camera["index"]) for camera in matches)
        raise RuntimeError(
            f"multiple HTC camera entries were found ({indices}); pass --index"
        )
    return matches[0]


def _frame_is_raw_yuy2(frame) -> bool:
    return (
        frame.dtype.name == "uint8"
        and frame.nbytes == 400 * 400 * 2
        and frame.shape in ((400, 400, 2), (400, 800))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, help="DirectShow camera index")
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    if sys.platform != "win32":
        print("This hardware probe must be run from native Windows.", file=sys.stderr)
        return 2

    cameras = list_uvc_cameras()
    print("Detected cameras:")
    for camera in cameras:
        print(
            f"  [{camera['index']}] {camera['name']} :: {camera['address']}"
        )

    camera = _choose_camera(cameras, args.index)
    index = int(camera["index"])
    controller = stream_controller_for(
        str(camera["name"]), str(camera["address"]), device_index=index
    )
    if controller is None:
        raise RuntimeError("selected device did not activate the HTC controller")

    enabled = False
    frames = 0
    raw_frames = 0
    shapes: set[tuple[int, ...]] = set()
    stop_event = threading.Event()
    timer = None
    capture_elapsed = 0.0
    try:
        print(f"Enabling HTC stream controls on DirectShow index {index}...")
        controller.enable()
        enabled = True

        def receive(frame):
            nonlocal frames, raw_frames
            frames += 1
            shapes.add(tuple(frame.shape))
            raw_frames += int(_frame_is_raw_yuy2(frame))

        timer = threading.Timer(max(args.seconds, 0.25), stop_event.set)
        timer.start()
        capture_started = time.monotonic()
        capture_yuy2(index, stop_event, receive)
        capture_elapsed = time.monotonic() - capture_started
    finally:
        stop_event.set()
        if timer is not None:
            timer.cancel()
        if enabled:
            print("Disabling HTC stream controls...")
            controller.disable()

    elapsed = max(capture_elapsed, 1e-6)
    print(
        f"Captured {frames} frames in {elapsed:.2f}s "
        f"({frames / elapsed:.1f} fps); shapes={sorted(shapes)}"
    )
    if frames == 0:
        print("FAIL: DirectShow returned no frames.", file=sys.stderr)
        return 1
    if raw_frames != frames:
        print(
            "FAIL: DirectShow returned samples that were not raw "
            "320,000-byte YUY2.",
            file=sys.stderr,
        )
        return 1
    print("PASS: raw 400x400 YUY2 capture and HTC shutdown completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
