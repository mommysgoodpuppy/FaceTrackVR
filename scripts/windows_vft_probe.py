"""Probe direct Vive Facial Tracker capture on Windows.

Run from the repository root with the Windows project environment active:

    python scripts/windows_vft_probe.py

The probe performs the HTC extension-unit startup sequence, captures raw
400x400 YUY2 frames through OpenCV's DirectShow backend, and always attempts
to turn the emitters off before exiting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "EyeTrackApp"
sys.path.insert(0, str(APP_DIR))

import cv2  # noqa: E402

from camera_behaviors import stream_controller_for  # noqa: E402
from camera_enum import list_uvc_cameras  # noqa: E402


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

    capture = None
    enabled = False
    frames = 0
    raw_frames = 0
    shapes: set[tuple[int, ...]] = set()
    started = time.monotonic()
    try:
        print(f"Enabling HTC stream controls on DirectShow index {index}...")
        controller.enable()
        enabled = True

        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
        capture.set(cv2.CAP_PROP_FPS, 60)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            raise RuntimeError("DirectShow could not open the tracker")

        print(
            "Negotiated: "
            f"{capture.get(cv2.CAP_PROP_FRAME_WIDTH):g}x"
            f"{capture.get(cv2.CAP_PROP_FRAME_HEIGHT):g} @ "
            f"{capture.get(cv2.CAP_PROP_FPS):g} fps, "
            f"convert_rgb={capture.get(cv2.CAP_PROP_CONVERT_RGB):g}"
        )
        deadline = time.monotonic() + max(args.seconds, 0.25)
        while time.monotonic() < deadline:
            ok, frame = capture.read()
            if not ok:
                continue
            frames += 1
            shapes.add(tuple(frame.shape))
            raw_frames += int(_frame_is_raw_yuy2(frame))
    finally:
        if capture is not None:
            capture.release()
        if enabled:
            print("Disabling HTC stream controls...")
            controller.disable()

    elapsed = max(time.monotonic() - started, 1e-6)
    print(
        f"Captured {frames} frames in {elapsed:.2f}s "
        f"({frames / elapsed:.1f} fps); shapes={sorted(shapes)}"
    )
    if frames == 0:
        print("FAIL: DirectShow returned no frames.", file=sys.stderr)
        return 1
    if raw_frames != frames:
        print(
            "FAIL: frames were converted instead of raw 320,000-byte YUY2. "
            "The application needs a dedicated DirectShow capture path.",
            file=sys.stderr,
        )
        return 1
    print("PASS: raw 400x400 YUY2 capture and HTC shutdown completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
