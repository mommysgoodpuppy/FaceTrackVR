"""Mouth tracking for compatible Vive Facial Tracker models.

Self-contained module: owns its camera, its inference thread and (optionally)
its own VRCFT output client. The app only calls ``LipTracker.start(config)`` /
``.stop()`` and, when the pyVRCFT output mode is active, may hand us its shared
``UnifiedTrackingData`` frame so mouth shapes ride the existing 100 Hz sender
instead of a second OSC client.

Compatible model pipeline (weights are user-supplied, never distributed):
  VFT YUYV 400x400 (vendor XU handshake required to start streaming)
  -> 800x400 mono view (YUYV bytes re-read row-wise: luma on even cols,
     chroma bytes on odd cols — this is what HTC's model was trained on)
  -> medianBlur(5), preserving V4L2 row order
  -> stereo panes [0:400] / [400:800] -> merge -> cubic resize 100x100 -> * 1/255
  -> ONNX -> 30 raw outputs
  -> stateful postprocess -> 60 LipShape_v2 weights
  -> Unified Expressions mapping
"""

from contextlib import nullcontext
import logging
import math
import os
import sys
import threading
import time

import numpy as np
from camera_behaviors import stream_controller_for
from camera_enum import (
    is_uvc_named_source,
    parse_uvc_named_source,
    resolve_uvc_address_to_index,
)

logger = logging.getLogger(__name__)

try:
    import cv2
    import onnxruntime as ort
    from lip_post_v2 import LipPostprocessV2, LIP_SHAPE_V2
    from one_euro_filter import OneEuroFilter

    _DEPS_OK = True
except ImportError as e:  # pragma: no cover
    _DEPS_OK = False
    _DEPS_ERROR = e

class MouthCamera:
    """Local camera capture with device-specific setup when required."""

    def __init__(self, device="/dev/video2"):
        self.device = device
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._stream_controller = None
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    @staticmethod
    def _is_raw_vft_frame(frame: np.ndarray) -> bool:
        return (
            frame.dtype == np.uint8
            and frame.nbytes == 400 * 400 * 2
            and frame.shape in ((400, 400, 2), (400, 800))
        )

    def _loop(self):
        open_source = self.device
        camera_name = ""
        camera_address = str(self.device)
        camera_index = None
        if isinstance(self.device, str) and is_uvc_named_source(self.device):
            camera_name, saved_address = parse_uvc_named_source(self.device)
            resolved = resolve_uvc_address_to_index(camera_name, saved_address)
            if resolved is None:
                self.error = f"cannot find {camera_name or self.device}"
                logger.error(self.error)
                self._ready.set()
                return
            camera_index, camera_address = resolved
            if sys.platform.startswith("linux") and camera_address.startswith("/dev/"):
                open_source = camera_address
            else:
                open_source = camera_index
        elif sys.platform == "win32":
            # Older settings stored only the Windows PnP address. Resolve it
            # once so existing configurations migrate without asking the user
            # to select the tracker again.
            resolved = resolve_uvc_address_to_index("", camera_address)
            if resolved is not None:
                camera_index, camera_address = resolved
                open_source = camera_index

        self._stream_controller = stream_controller_for(
            camera_name, camera_address, device_index=camera_index
        )
        if self._stream_controller is not None:
            try:
                self._stream_controller.enable()
            except Exception as e:
                self.error = f"camera initialization failed: {e}"
                logger.error(self.error)
                try:
                    # A failed handshake can occur after some registers were
                    # written. Attempt shutdown even when enable() did not get
                    # far enough to set the controller's enabled flag.
                    self._stream_controller.set_enabled(False)
                except Exception:
                    logger.warning(
                        "Camera cleanup after failed initialization also failed",
                        exc_info=True,
                    )
                self._ready.set()
                return
        if sys.platform == "win32" and isinstance(open_source, int):
            backend = cv2.CAP_DSHOW
        elif (
            sys.platform.startswith("linux")
            and isinstance(open_source, str)
            and open_source.startswith("/dev/")
        ):
            backend = cv2.CAP_V4L2
        else:
            backend = cv2.CAP_ANY
        cap = None
        try:
            if (
                sys.platform == "win32"
                and self._stream_controller is not None
                and isinstance(open_source, int)
            ):
                from windows_dshow_capture import capture_yuy2

                def publish(frame):
                    with self._cond:
                        self._frame = frame
                        self._frame_id += 1
                        self._cond.notify_all()
                    self._ready.set()

                try:
                    capture_yuy2(open_source, self._stop, publish)
                except Exception as e:
                    self.error = f"camera capture failed: {e}"
                    logger.error(self.error)
                    self._ready.set()
                return

            cap = cv2.VideoCapture(open_source, backend)
            if self._stream_controller is not None:
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                # Windows/DirectShow names the native format YUY2; Linux/V4L2
                # exposes the byte-identical packed format as YUYV.
                fourcc = "YUY2" if sys.platform == "win32" else "YUYV"
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
                cap.set(cv2.CAP_PROP_FPS, 60)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self.error = f"cannot open {self.device}"
                logger.error(self.error)
                self._ready.set()
                return
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.005)
                    continue
                if (
                    self._stream_controller is not None
                    and not self._is_raw_vft_frame(frame)
                ):
                    self.error = (
                        "camera backend converted the Vive Facial Tracker "
                        f"frame (shape={frame.shape}, bytes={frame.nbytes}); "
                        "raw 400x400 YUY2 is required"
                    )
                    logger.error(self.error)
                    self._ready.set()
                    return
                with self._cond:
                    self._frame = frame
                    self._frame_id += 1
                    self._cond.notify_all()
                self._ready.set()
        finally:
            if cap is not None:
                cap.release()
            if self._stream_controller is not None:
                try:
                    self._stream_controller.disable()
                except Exception:
                    logger.warning(
                        "Camera shutdown control failed", exc_info=True
                    )

    def start(self):
        self._stop.clear()
        self._ready.clear()
        self.error = None
        self._thread = threading.Thread(target=self._loop, daemon=True, name="MouthCapture")
        self._thread.start()
        if not self._ready.wait(timeout=3.0):
            self.error = f"timed out starting {self.device}"
            logger.error(self.error)
            return False
        return self.error is None

    def read(self, timeout=0.05, last_id=-1):
        """Wait for a frame newer than ``last_id``; returns (frame, id)."""
        with self._cond:
            if self._frame_id <= last_id:
                self._cond.wait(timeout)
            return self._frame, self._frame_id

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._stream_controller is not None:
            try:
                self._stream_controller.disable()
            except Exception as e:
                logger.warning("Camera shutdown control failed: %s", e)

# ---------------------------------------------------------------------------
# Unified Expressions compatibility mapping.

def map_to_unified(w: dict) -> dict:
    """LipShape_v2 named weights -> Unified Expressions weights."""
    shapes = {}

    def s(name, value):
        # Zeroes must be emitted too, otherwise a previously active shape
        # remains stuck in the shared VRCFT frame.
        # Do not clamp: VRCFT's UnifiedExpressionShape.Weight is a plain float,
        # and compatibility mapping preserves sums and differences as-is.
        shapes[name] = float(value)

    s("JawOpen", w["JawOpen"] + w["MouthApeShape"])
    s("JawLeft", w["JawLeft"])
    s("JawRight", w["JawRight"])
    s("JawForward", w["JawForward"])
    s("MouthClosed", w["MouthApeShape"])

    ou = w["MouthUpperOverturn"]
    ol = w["MouthLowerOverturn"]
    s("MouthUpperUpRight", w["MouthUpperUpRight"] - ou)
    s("MouthUpperDeepenRight", w["MouthUpperUpRight"] - ou)
    s("MouthUpperUpLeft", w["MouthUpperUpLeft"] - ou)
    s("MouthUpperDeepenLeft", w["MouthUpperUpLeft"] - ou)
    s("MouthLowerDownLeft", w["MouthLowerDownLeft"] - ol)
    s("MouthLowerDownRight", w["MouthLowerDownRight"] - ol)

    pout = w["MouthPout"]
    s("LipPuckerUpperLeft", pout)
    s("LipPuckerLowerLeft", pout)
    s("LipPuckerUpperRight", pout)
    s("LipPuckerLowerRight", pout)
    s("LipFunnelUpperLeft", ou)
    s("LipFunnelUpperRight", ou)
    s("LipFunnelLowerLeft", ou)
    s("LipFunnelLowerRight", ou)
    s("LipSuckUpperLeft", w["MouthUpperInside"])
    s("LipSuckUpperRight", w["MouthUpperInside"])
    s("LipSuckLowerLeft", w["MouthLowerInside"])
    s("LipSuckLowerRight", w["MouthLowerInside"])

    s("MouthUpperLeft", w["MouthUpperLeft"])
    s("MouthUpperRight", w["MouthUpperRight"])
    s("MouthLowerLeft", w["MouthLowerLeft"])
    s("MouthLowerRight", w["MouthLowerRight"])

    s("MouthCornerPullLeft", w["MouthSmileLeft"])
    s("MouthCornerPullRight", w["MouthSmileRight"])
    s("MouthCornerSlantLeft", w["MouthSmileLeft"])
    s("MouthCornerSlantRight", w["MouthSmileRight"])
    s("MouthFrownLeft", w["MouthSadLeft"])
    s("MouthFrownRight", w["MouthSadRight"])

    s("MouthRaiserUpper", w["MouthLowerOverlay"] - w["MouthUpperInside"])
    s("MouthRaiserLower", w["MouthLowerOverlay"])

    s("CheekPuffLeft", w["CheekPuffLeft"])
    s("CheekPuffRight", w["CheekPuffRight"])
    s("CheekSuckLeft", w["CheekSuck"])
    s("CheekSuckRight", w["CheekSuck"])

    s("TongueOut", (w["TongueLongStep1"] + w["TongueLongStep2"]) / 2.0)
    s("TongueUp", w["TongueUp"])
    s("TongueDown", w["TongueDown"])
    s("TongueLeft", w["TongueLeft"])
    s("TongueRight", w["TongueRight"])
    s("TongueRoll", w["TongueRoll"])

    # Compatibility-only shapes derived from the closest model outputs.
    s("CheekSquintLeft", w["MouthSmileLeft"])
    s("CheekSquintRight", w["MouthSmileRight"])
    s("MouthDimpleLeft", w["MouthSmileLeft"])
    s("MouthDimpleRight", w["MouthSmileRight"])
    # Compatibility mode uses MouthSadRight for both stretch channels.
    s("MouthStretchLeft", w["MouthSadRight"])
    s("MouthStretchRight", w["MouthSadRight"])
    return shapes


# Shape groups and names match VRCFT's ParameterAdjustment behavior. This is
# deliberately a post-mapping operation so inference and post-processing stay
# unchanged.
_VRCFT_ADJUST_GROUPS = {
    "jaw_open": ("JawClench", "JawMandibleRaise", "JawOpen"),
    "mouth_closed": ("MouthClosed",),
    "mouth_open": (
        "MouthUpperDeepenLeft", "MouthUpperDeepenRight",
        "MouthUpperUpLeft", "MouthUpperUpRight",
        "MouthLowerDownLeft", "MouthLowerDownRight",
    ),
    "smile": (
        "MouthCornerPullLeft", "MouthCornerPullRight",
        "MouthCornerSlantLeft", "MouthCornerSlantRight",
    ),
    "frown": ("MouthFrownLeft", "MouthFrownRight"),
    "pucker": (
        "LipPuckerLowerLeft", "LipPuckerLowerRight",
        "LipPuckerUpperLeft", "LipPuckerUpperRight",
    ),
    "funnel": (
        "LipFunnelLowerLeft", "LipFunnelLowerRight",
        "LipFunnelUpperLeft", "LipFunnelUpperRight",
    ),
    "tongue_out": ("TongueOut",),
}


def apply_vrcft_parameter_adjustment(shapes: dict, ranges: dict) -> dict:
    """Apply VRCFT ParameterAdjustment ranges, without clamping."""
    adjusted = dict(shapes)
    for group, names in _VRCFT_ADJUST_GROUPS.items():
        floor, ceiling = ranges.get(group, (0.0, 1.0))
        if ceiling == floor:
            continue
        for name in names:
            if name in adjusted:
                adjusted[name] = _apply_parameter_range(
                    adjusted[name], floor, ceiling
                )
    return adjusted


def _apply_parameter_range(value: float, floor: float, ceiling: float) -> float:
    """VRCFT's unclamped System.Single parameter-range mutation."""
    if ceiling == floor:
        return float(value)
    value32 = np.float32(value)
    floor32 = np.float32(floor)
    ceiling32 = np.float32(ceiling)
    return float(
        np.float32(value32 - floor32)
        / np.float32(ceiling32 - floor32)
    )


def apply_global_response(values: dict, minimum: float, maximum: float, curve: float) -> dict:
    """Apply one dead-zone/saturation/curve to every mouth magnitude.

    Signed compact parameters retain their direction; the transform only
    reshapes absolute magnitude. Curve <1 exaggerates small motion, curve >1
    damps it, and maximum controls how early full output is reached.
    """
    span = maximum - minimum
    if span <= 0.0:
        raise ValueError("global response maximum must exceed minimum")
    out = {}
    for name, raw_value in values.items():
        value = float(raw_value)
        magnitude = max(0.0, min(1.0, (abs(value) - minimum) / span))
        shaped = magnitude ** curve
        out[name] = math.copysign(shaped, value) if value != 0.0 else 0.0
    return out


def map_to_improved_v2(
    w: dict,
    *,
    boost_jaw_open: bool = False,
    parameter_ranges: dict | None = None,
) -> dict:
    """Direct, compact v2 mouth outputs from model shapes.

    Unlike the compatibility path, this does not inflate one model channel
    into several Unified shapes and then average/subtract those shapes again.
    Signed compact channels retain useful opposition, while mutually exclusive
    opening/funnel channels are kept independent and bounded where negative
    weights have no physical meaning.
    """
    avg = lambda left, right: (left + right) * 0.5

    def ranged(group: str, value: float) -> float:
        if parameter_ranges is None:
            return float(value)
        floor, ceiling = parameter_ranges.get(group, (0.0, 1.0))
        if floor == 0.0 and ceiling == 1.0:
            return float(value)
        return _apply_parameter_range(value, floor, ceiling)

    upper_right = max(0.0, w["MouthUpperUpRight"] - w["MouthUpperOverturn"])
    upper_left = max(0.0, w["MouthUpperUpLeft"] - w["MouthUpperOverturn"])
    lower_right = max(0.0, w["MouthLowerDownRight"] - w["MouthLowerOverturn"])
    lower_left = max(0.0, w["MouthLowerDownLeft"] - w["MouthLowerOverturn"])
    return {
        "v2/MouthClosed": ranged("mouth_closed", w["MouthApeShape"]),
        # Keep the two model opening concepts independent. The compatibility
        # path sums JawOpen + MouthApeShape, which amplifies their shared noise.
        "v2/JawOpen": ranged(
            "jaw_open",
            (
                w["JawOpen"] + w["MouthApeShape"]
                if boost_jaw_open else w["JawOpen"]
            ),
        ),
        "v2/JawX": w["JawRight"] - w["JawLeft"],
        "v2/JawForward": w["JawForward"],
        "v2/MouthRaiserUpper": max(
            0.0, w["MouthLowerOverlay"] - w["MouthUpperInside"]
        ),
        "v2/MouthRaiserLower": w["MouthLowerOverlay"],
        "v2/TongueX": w["TongueRight"] - w["TongueLeft"],
        "v2/TongueY": w["TongueUp"] - w["TongueDown"],
        "v2/TongueOut": ranged(
            "tongue_out", avg(w["TongueLongStep1"], w["TongueLongStep2"])
        ),
        "v2/LipSuckUpper": w["MouthUpperInside"],
        "v2/LipSuckLower": w["MouthLowerInside"],
        "v2/LipPucker": ranged("pucker", w["MouthPout"]),
        # Preserve the lower-overturn signal instead of copying upper overturn
        # into all four funnel shapes and averaging it back out.
        "v2/LipFunnel": ranged(
            "funnel", avg(w["MouthUpperOverturn"], w["MouthLowerOverturn"])
        ),
        "v2/CheekPuffSuckRight": w["CheekPuffRight"] - w["CheekSuck"],
        "v2/CheekPuffSuckLeft": w["CheekPuffLeft"] - w["CheekSuck"],
        "v2/SmileFrownRight": (
            ranged("smile", w["MouthSmileRight"])
            - ranged("frown", w["MouthSadRight"])
        ),
        "v2/SmileFrownLeft": (
            ranged("smile", w["MouthSmileLeft"])
            - ranged("frown", w["MouthSadLeft"])
        ),
        # This avatar exposes the two stretch channels independently. The
        # model has matching per-side Sad inputs, so preserve them
        # instead of inheriting VRCFT's compatibility quirk that copies the
        # right side into both outputs.
        "v2/MouthStretchRight": w["MouthSadRight"],
        "v2/MouthStretchLeft": w["MouthSadLeft"],
        "v2/MouthX": (
            w["MouthUpperRight"] + w["MouthLowerRight"]
            - w["MouthUpperLeft"] - w["MouthLowerLeft"]
        ) * 0.5,
        "v2/MouthLowerDown": ranged(
            "mouth_open", avg(lower_right, lower_left)
        ),
        "v2/MouthUpperUpRight": ranged("mouth_open", upper_right),
        "v2/MouthUpperUpLeft": ranged("mouth_open", upper_left),
        # The model output vocabulary has no direct equivalents for these.
        "v2/MouthPress": 0.0,
        "v2/MouthTightenerRight": 0.0,
        "v2/MouthTightenerLeft": 0.0,
        "v2/NoseSneer": 0.0,
    }


class NewSmoothFilter:
    """Low-latency physiology guard for direct compact mouth parameters.

    Non-tongue expressions pass through exactly. Tongue X/Y are treated as one
    polar direction-and-extension vector, while protrusion remains an
    independent slew-limited axis.
    """

    _TONGUE_VECTOR = frozenset(("v2/TongueX", "v2/TongueY"))

    def __init__(self):
        self.values: dict[str, float] = {}
        self.last_time: float | None = None
        self.tongue_angle: float | None = None
        self.tongue_magnitude: float = 0.0
        self.tongue_out_filtered_target: float | None = None
        self.tongue_candidate_angle: float | None = None
        self.tongue_candidate_frames: int = 0

    def reset(self):
        self.values.clear()
        self.last_time = None
        self.tongue_angle = None
        self.tongue_magnitude = 0.0
        self.tongue_out_filtered_target = None
        self.tongue_candidate_angle = None
        self.tongue_candidate_frames = 0

    def __call__(
        self,
        values: dict,
        *,
        now: float | None = None,
        tongue_speed: float = 4.0,
        tongue_angle_speed: float = 360.0,
        tongue_protrusion_hz: float = 12.0,
        tongue_retracted_range: float = 0.35,
        tongue_reversal_frames: int = 2,
        deadband: float = 0.006,
    ) -> dict:
        now = time.monotonic() if now is None else float(now)
        dt = 1.0 / 60.0 if self.last_time is None else now - self.last_time
        dt = max(1.0 / 240.0, min(dt, 0.1))
        self.last_time = now
        out = {}

        # Protrusion is a physical limit on directional travel: a tongue still
        # behind the teeth cannot reach the same visible up/down extreme as a
        # fully extended tongue. Slew-limit this axis first so the directional
        # envelope itself expands and contracts continuously.
        tongue_out_name = "v2/TongueOut"
        if tongue_out_name in values:
            raw_tongue_out = float(values[tongue_out_name])
            if self.tongue_out_filtered_target is None:
                self.tongue_out_filtered_target = raw_tongue_out
            else:
                response = max(0.01, float(tongue_protrusion_hz))
                alpha = 1.0 - math.exp(-2.0 * math.pi * response * dt)
                self.tongue_out_filtered_target += alpha * (
                    raw_tongue_out - self.tongue_out_filtered_target
                )
            tongue_out_target = self.tongue_out_filtered_target
            if tongue_out_name not in self.values:
                tongue_out = tongue_out_target
            else:
                tongue_out_current = self.values[tongue_out_name]
                tongue_out_delta = tongue_out_target - tongue_out_current
                tongue_out_limit = tongue_speed * dt
                tongue_out = tongue_out_current + max(
                    -tongue_out_limit,
                    min(tongue_out_limit, tongue_out_delta),
                )
            self.values[tongue_out_name] = tongue_out
            out[tongue_out_name] = tongue_out

        # Treat directional tongue output as one physical polar control rather
        # than two unrelated classifier channels. Magnitude is extension from
        # neutral; angle is direction. A >90-degree reversal must retract near
        # neutral before changing direction, preventing Up/Down label flicker
        # from teleporting the tongue through the mouth.
        tongue_names = ("v2/TongueX", "v2/TongueY")
        if all(name in values for name in tongue_names):
            target_x = float(values[tongue_names[0]])
            target_y = float(values[tongue_names[1]])
            target_magnitude = min(1.0, math.hypot(target_x, target_y))
            target_angle = (
                math.atan2(target_y, target_x)
                if target_magnitude > deadband else self.tongue_angle
            )

            if self.tongue_angle is None:
                self.tongue_angle = target_angle if target_angle is not None else 0.0
                self.tongue_magnitude = target_magnitude
            else:
                neutral_radius = max(deadband * 2.0, 0.04)
                if self.tongue_magnitude <= neutral_radius and target_angle is not None:
                    # Direction has no physical meaning at neutral, so acquire
                    # the new angle without drawing an artificial sideways arc.
                    self.tongue_angle = target_angle
                angle_delta = 0.0 if target_angle is None else (
                    (target_angle - self.tongue_angle + math.pi) % (2.0 * math.pi)
                    - math.pi
                )
                magnitude_step = tongue_speed * dt
                reversal = (
                    abs(angle_delta) > math.pi * 0.5
                    and self.tongue_magnitude > neutral_radius
                )
                confirmed_reversal = reversal
                if reversal and tongue_reversal_frames > 1:
                    candidate_delta = (
                        math.inf
                        if self.tongue_candidate_angle is None
                        else abs(
                            (
                                target_angle
                                - self.tongue_candidate_angle
                                + math.pi
                            )
                            % (2.0 * math.pi)
                            - math.pi
                        )
                    )
                    if candidate_delta <= math.pi * 0.25:
                        self.tongue_candidate_frames += 1
                    else:
                        self.tongue_candidate_angle = target_angle
                        self.tongue_candidate_frames = 1
                    confirmed_reversal = (
                        self.tongue_candidate_frames >= tongue_reversal_frames
                    )
                elif not reversal:
                    self.tongue_candidate_angle = None
                    self.tongue_candidate_frames = 0

                if reversal and not confirmed_reversal:
                    # Hold the last physically valid pose for the single
                    # contradictory frame. No output dip is introduced.
                    pass
                elif reversal:
                    self.tongue_magnitude = max(
                        0.0, self.tongue_magnitude - magnitude_step
                    )
                else:
                    angle_step = math.radians(tongue_angle_speed) * dt
                    self.tongue_angle += max(
                        -angle_step, min(angle_step, angle_delta)
                    )
                    magnitude_delta = target_magnitude - self.tongue_magnitude
                    self.tongue_magnitude += max(
                        -magnitude_step, min(magnitude_step, magnitude_delta)
                    )

            tongue_x = math.cos(self.tongue_angle) * self.tongue_magnitude
            tongue_y = math.sin(self.tongue_angle) * self.tongue_magnitude
            if tongue_out_name in out:
                protrusion = max(0.0, min(1.0, out[tongue_out_name]))
                retracted_range = max(
                    0.0, min(1.0, float(tongue_retracted_range))
                )
                vertical_range = retracted_range + (
                    1.0 - retracted_range
                ) * protrusion
                tongue_y *= vertical_range
            if self.tongue_magnitude <= deadband:
                tongue_x = tongue_y = 0.0
            self.values[tongue_names[0]] = tongue_x
            self.values[tongue_names[1]] = tongue_y
            out[tongue_names[0]] = tongue_x
            out[tongue_names[1]] = tongue_y

        for name, raw_value in values.items():
            if name in self._TONGUE_VECTOR or name == tongue_out_name:
                continue
            target = float(raw_value)
            self.values[name] = target
            out[name] = target
        return out


class MouthBounceFilter:
    """Synthetic per-parameter spring motion with an exact damped solution.

    ``mix`` blends the spring with the untouched target, so zero is a true
    bypass. The continuous spring solution avoids frame-rate-dependent
    numerical explosions when inference timing varies.
    """

    # Compact parameters whose negative half is meaningful. Unified shapes use
    # separate directional channels and therefore have the ordinary 0..1
    # range. This list selects the physical travel limits of the spring.
    _SIGNED_PARAMS = frozenset(
        (
            "v2/JawX",
            "v2/TongueX",
            "v2/TongueY",
            "v2/CheekPuffSuck",
            "v2/CheekPuffSuckLeft",
            "v2/CheekPuffSuckRight",
            "v2/SmileFrown",
            "v2/SmileFrownLeft",
            "v2/SmileFrownRight",
            "v2/MouthX",
        )
    )

    def __init__(self):
        self.position: dict[str, float] = {}
        self.velocity: dict[str, float] = {}
        self.last_time: float | None = None

    def reset(self):
        self.position.clear()
        self.velocity.clear()
        self.last_time = None

    @staticmethod
    def _step(position, velocity, target, dt, response_hz, damping):
        omega = 2.0 * math.pi * response_hz
        error = position - target
        if damping < 1.0 - 1e-6:
            damped = omega * math.sqrt(1.0 - damping * damping)
            phase = damped * dt
            decay = math.exp(-damping * omega * dt)
            cos_phase = math.cos(phase)
            sin_phase = math.sin(phase)
            b = (velocity + damping * omega * error) / damped
            wave = error * cos_phase + b * sin_phase
            next_error = decay * wave
            next_velocity = decay * (
                -damping * omega * wave
                - error * damped * sin_phase
                + b * damped * cos_phase
            )
        else:
            # Damping=1 is the non-ringing, critically damped endpoint.
            decay = math.exp(-omega * dt)
            b = velocity + omega * error
            next_error = (error + b * dt) * decay
            next_velocity = (velocity - omega * b * dt) * decay
        return target + next_error, next_velocity

    @classmethod
    def _bounds(cls, name):
        return (-1.0, 1.0) if name in cls._SIGNED_PARAMS else (0.0, 1.0)

    @staticmethod
    def _smooth_fold_unit(value):
        """Periodic 0..1 fold with zero slope at every boundary.

        Unlike a clamp (flat lost motion) or triangle reflection (a velocity
        cusp), cosine represents travel beyond either cap as a smooth inward
        return. Position and first derivative are continuous at the turn.
        """
        return 0.5 - 0.5 * math.cos(math.pi * value)

    @classmethod
    def _visible_position(cls, name, target, spring_position):
        """Fold a spring position and anchor its equilibrium to ``target``."""
        low, high = cls._bounds(name)
        width = high - low
        target_unit = (target - low) / width
        spring_unit = (spring_position - low) / width
        folded = cls._smooth_fold_unit(spring_unit)
        folded_target = cls._smooth_fold_unit(target_unit)
        # Smoothly re-anchor the nonlinear fold so the resting spring equals
        # the exact source target.  A fractional-linear map keeps both caps,
        # has no piecewise kink at equilibrium, and remains monotonic.
        # ``cos`` loses the quadratic difference from 1.0 for sufficiently
        # small distances from either endpoint.  Check the folded value too:
        # a valid float32 model output around 1e-9 can otherwise make the
        # fractional-linear anchor divide by zero and kill LipTracker.
        if (
            target_unit <= 1e-12
            or target_unit >= 1.0 - 1e-12
            or folded_target <= 0.0
            or folded_target >= 1.0
        ):
            visible_unit = folded
        else:
            anchor = (
                target_unit * (1.0 - folded_target)
                / (folded_target * (1.0 - target_unit))
            )
            visible_unit = (
                anchor * folded / (1.0 + (anchor - 1.0) * folded)
            )
        return low + visible_unit * width

    def __call__(
        self,
        values: dict,
        *,
        now: float | None = None,
        response_hz: float = 5.0,
        damping: float = 0.55,
        mix: float = 35.0,
        exclude=frozenset(),
    ) -> dict:
        now = time.monotonic() if now is None else float(now)
        dt = 1.0 / 60.0 if self.last_time is None else now - self.last_time
        dt = max(1.0 / 240.0, min(dt, 0.1))
        self.last_time = now
        blend = max(0.0, min(100.0, float(mix))) * 0.01
        out = {}
        for name, raw_value in values.items():
            target = float(raw_value)
            if name in exclude:
                # New Smooth owns tongue physiology. Do not let a decorative
                # spring reintroduce a direction flip or impossible speed.
                self.position.pop(name, None)
                self.velocity.pop(name, None)
                out[name] = target
                continue
            low, high = self._bounds(name)
            if not low <= target <= high:
                # Preserve genuine out-of-range compatibility arithmetic and
                # re-enter the bounded spring cleanly if it returns later.
                self.position.pop(name, None)
                self.velocity.pop(name, None)
                out[name] = target
                continue
            if name not in self.position:
                self.position[name] = target
                self.velocity[name] = 0.0
                out[name] = target
                continue
            else:
                response = max(0.01, float(response_hz))
                damp = max(0.0, min(1.0, float(damping)))
                self.position[name], self.velocity[name] = self._step(
                    self.position[name], self.velocity[name], target,
                    dt, response, damp,
                )
            spring_value = self._visible_position(
                name, target, self.position[name]
            )
            out[name] = target + (spring_value - target) * blend
        return out

# ---------------------------------------------------------------------------

# Registry of the active tracker so UI widgets can reach it without the app
# threading a reference through every layer.
_active_tracker: "LipTracker | None" = None

try:
    from pyvrcft import UNIFIED_EXPRESSIONS as _UNIFIED_EXPRESSIONS

    _UNIFIED_NAMES = tuple(_UNIFIED_EXPRESSIONS)
except ImportError:  # pragma: no cover
    _UNIFIED_NAMES = ()


def get_active_tracker() -> "LipTracker | None":
    return _active_tracker




class LipTracker:
    """Self-contained Vive Facial Tracker runtime. start()/stop() only."""

    def __init__(self, config):
        self.config = config
        self.camera: MouthCamera | None = None
        self.session = None
        self._model_input_name = "input"
        self._model_output_name = None
        self.post = None
        self._lip_smoother = None
        self._new_smooth = NewSmoothFilter()
        self._mouth_bounce = MouthBounceFilter()
        self._last_output_mode = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.shared_data = None  # optional pyVRCFT shared frame
        self.shared_data_provider = None  # callable -> shared frame (lazy)
        self._own_client = None
        self.last_shapes: dict = {}
        self.presence: float = 0.0
        self.inference_hz: float = 0.0
        self._last_inference_at = None
        self._preview_lock = threading.Lock()
        self._raw_preview: np.ndarray | None = None
        self._model_preview: np.ndarray | None = None

    # -- model path ---------------------------------------------------------

    def _model_path(self) -> str:
        configured = str(
            getattr(self.config, "gui_lip_onnx_path", "") or ""
        ).strip()
        if not configured:
            raise FileNotFoundError(
                "No mouth-tracking model is configured. Select a compatible "
                "ONNX model in Mouth settings."
            )
        path = os.path.abspath(os.path.expanduser(configured))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Mouth-tracking model not found: {path}")
        return path

    def _device(self) -> str | None:
        dev = str(getattr(self.config, "gui_lip_device", "") or "").strip()
        return dev or None

    # -- lifecycle ----------------------------------------------------------

    def start(self, shared_data=None):
        if not _DEPS_OK:
            logger.warning(f"Lip tracking unavailable: {_DEPS_ERROR}")
            return False
        if self._thread is not None:
            return True
        device = self._device()
        if device is None:
            logger.info("Mouth tracking is disconnected: no camera selected")
            return False
        try:
            self.session = self._make_session(self._model_path())
        except Exception as e:
            logger.warning(f"Lip tracking disabled: {e}")
            return False
        self.post = LipPostprocessV2()
        self._reset_lip_smoother()
        self._new_smooth.reset()
        self._mouth_bounce.reset()
        self.camera = MouthCamera(device)
        if not self.camera.start():
            logger.warning("Lip tracking disabled: %s", self.camera.error)
            self.camera.stop()
            self.camera = None
            return False
        self.shared_data = shared_data
        shared = self._shared_frame()
        if shared is not None:
            try:
                shapes = shared.shapes
                for name in _UNIFIED_NAMES:
                    if name not in shapes:
                        shapes[name] = 0.0
                # The mouth mapping owns CheekSquint through MouthSmile.
                shared._lip_owns_cheek_squint = True
            except Exception:
                logger.warning("could not prepopulate shared shapes", exc_info=True)
        # Runtime OSC-mode changes restart this tracker, so only create a
        # second VRCFT client when there is no shared pyVRCFT sender.
        if shared is None and self._own_client is None:
            self._own_client = self._make_own_client()
        self._stop.clear()
        global _active_tracker
        _active_tracker = self
        self._thread = threading.Thread(target=self._loop, daemon=True, name="LipTracker")
        self._thread.start()
        logger.info("Lip tracker started (%s)", device)
        return True

    def _make_session(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_mem_pattern = False
        session = ort.InferenceSession(
            path, options, providers=["CPUExecutionProvider"]
        )
        self._model_input_name, self._model_output_name = (
            self._validate_model_contract(session)
        )
        return session

    @staticmethod
    def _validate_model_contract(session):
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                "The mouth-tracking model must have exactly one input and "
                "one output."
            )

        model_input = inputs[0]
        model_output = outputs[0]
        expected_input = [1, 2, 100, 100]
        expected_output = [1, 30]
        if model_input.type != "tensor(float)":
            raise ValueError(
                "The mouth-tracking model input must be float32 "
                f"(found {model_input.type})."
            )
        if list(model_input.shape) != expected_input:
            raise ValueError(
                "The mouth-tracking model input shape must be "
                f"{expected_input} (found {list(model_input.shape)})."
            )
        if model_output.type != "tensor(float)":
            raise ValueError(
                "The mouth-tracking model output must be float32 "
                f"(found {model_output.type})."
            )
        if list(model_output.shape) != expected_output:
            raise ValueError(
                "The mouth-tracking model output shape must be "
                f"{expected_output} (found {list(model_output.shape)})."
            )
        return model_input.name, model_output.name

    def _make_own_client(self):
        try:
            from pyvrcft import UnifiedTrackingData, VRCFTClient

            client = VRCFTClient(
                send_host=str(self.config.gui_osc_address),
                send_port=int(self.config.gui_osc_port),
                recv_port=0,  # OSCQuery-only; never fight over the receiver
            )
            self._own_client_data = UnifiedTrackingData()
            self._own_client_data.expression_tracking_active = True
            client.start()
            return client
        except Exception as e:
            logger.warning(f"Lip standalone VRCFT output unavailable: {e}")
            return None

    def recalibrate(self):
        """Reset stateful mouth post-processing and warm-up."""
        # VRCFT's optional calibration mutation is separate and must not be
        # applied unconditionally here.
        self.post = LipPostprocessV2()
        self._reset_lip_smoother()
        self._new_smooth.reset()
        self._mouth_bounce.reset()
        logger.info("Lip tracker postprocess state reset")

    def _reset_lip_smoother(self):
        """Reset the optional EyeTrackVR One-Euro stage to neutral."""
        try:
            min_cutoff = float(self.config.gui_min_cutoff)
            beta = float(self.config.gui_speed_coefficient)
        except (AttributeError, TypeError, ValueError):
            min_cutoff, beta = 0.0004, 0.9
        self._lip_smoother = OneEuroFilter(
            np.zeros(len(LIP_SHAPE_V2), dtype=np.float32),
            min_cutoff=min_cutoff,
            beta=beta,
        )

    def _apply_optional_smoothing(self, shapes):
        values = np.asarray(shapes, dtype=np.float32)
        if not bool(getattr(self.config, "gui_lip_use_etvr_smoothing", False)):
            return values
        if self._lip_smoother is None:
            self._reset_lip_smoother()
        # The main smoothing slider can change without restarting lip tracking.
        try:
            self._lip_smoother.min_cutoff[:] = float(self.config.gui_min_cutoff)
            self._lip_smoother.beta[:] = float(self.config.gui_speed_coefficient)
        except (AttributeError, TypeError, ValueError):
            pass
        return self._lip_smoother(values).astype(np.float32)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
        shared = self._shared_frame()
        if shared is not None:
            try:
                shared._lip_owns_cheek_squint = False
            except Exception:
                pass
        if self._own_client is not None:
            self._own_client.stop()
            self._own_client = None
        global _active_tracker
        if _active_tracker is self:
            _active_tracker = None
        logger.info("Lip tracker stopped")

    # -- processing ---------------------------------------------------------

    @staticmethod
    def _model_byte_plane(frame: np.ndarray) -> np.ndarray:
        if frame.shape == (400, 400, 2):
            return frame.reshape(400, 800)
        if frame.ndim == 3:
            if frame.shape[2] == 1:
                frame = frame[:, :, 0]
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape
        if width == height * 2:
            if frame.shape != (400, 800):
                frame = cv2.resize(frame, (800, 400), interpolation=cv2.INTER_AREA)
        else:
            if frame.shape != (400, 400):
                frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_AREA)
            frame = np.concatenate((frame, frame), axis=1)
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def _infer(self, camera_frame: np.ndarray) -> np.ndarray:
        b = self._model_byte_plane(camera_frame)
        frame = np.concatenate(
            (
                cv2.medianBlur(np.ascontiguousarray(b[:, :400]), 5),
                cv2.medianBlur(np.ascontiguousarray(b[:, 400:]), 5),
            ),
            axis=1,
        )
        # Copies only feed the UI. Never draw into `frame`: inference consumes
        # this exact byte plane below.
        if bool(getattr(self.config, "gui_lip_preview", True)):
            with self._preview_lock:
                # Raw two-byte camera frames and ordinary decoded frames are
                # both normalized to the model's two 400-pixel planes above.
                self._raw_preview = b.copy()
                self._model_preview = frame.copy()
        merged = cv2.merge([frame[:, :400], frame[:, 400:]])
        # Compatible models expect cubic resizing.
        resized = cv2.resize(
            merged, (100, 100), interpolation=cv2.INTER_CUBIC
        ).astype(np.float32) * (1.0 / 255.0)
        x = resized.transpose(2, 0, 1)[None]
        outputs = None if self._model_output_name is None else [self._model_output_name]
        return self.session.run(outputs, {self._model_input_name: x})[0][0]

    def _loop(self):
        last_id = -1
        while not self._stop.is_set():
            frame, last_id = self.camera.read(timeout=0.1, last_id=last_id)
            if frame is None:
                continue
            try:
                raw = self._infer(frame)
            except Exception as e:
                logger.warning(f"lip inference error: {e}")
                time.sleep(0.1)
                continue
            self.presence = float(raw[0])
            now = time.monotonic()
            if self._last_inference_at is not None:
                hz = 1.0 / max(now - self._last_inference_at, 1e-6)
                self.inference_hz = hz if self.inference_hz == 0.0 else self.inference_hz * 0.9 + hz * 0.1
            self._last_inference_at = now
            model_shapes = self.post(raw)
            named_shapes = self._apply_optional_smoothing(
                model_shapes[:len(LIP_SHAPE_V2)]
            )
            w = {name: float(named_shapes[i]) for i, name in enumerate(LIP_SHAPE_V2)}
            self.last_shapes = w
            unified = map_to_unified(w)
            unified = self._apply_optional_vrcft_adjustment(unified)
            mode = self._output_mode()
            if mode != self._last_output_mode:
                self._new_smooth.reset()
                self._mouth_bounce.reset()
                self._last_output_mode = mode
            improved = (
                map_to_improved_v2(
                    w,
                    boost_jaw_open=(
                        mode == "new_smooth"
                        and bool(
                            getattr(
                                self.config,
                                "gui_lip_smooth_boost_jaw_open",
                                False,
                            )
                        )
                    ),
                    parameter_ranges=self._parameter_adjustment_ranges(),
                )
                if mode != "vrcft" else None
            )
            if mode == "vrcft":
                unified = self._apply_optional_global_response(unified)
                unified = self._apply_optional_bounce(unified)
            else:
                improved = self._apply_optional_global_response(improved)
            if mode == "new_smooth":
                improved = self._apply_new_smooth(improved)
                improved = self._apply_optional_bounce(
                    improved,
                    exclude=frozenset(
                        ("v2/TongueX", "v2/TongueY", "v2/TongueOut")
                    ),
                )
            elif mode == "direct":
                improved = self._apply_optional_bounce(improved)
            self._emit(unified, improved)

    def _apply_optional_vrcft_adjustment(self, unified: dict) -> dict:
        ranges = self._parameter_adjustment_ranges()
        if ranges is None:
            return unified
        return apply_vrcft_parameter_adjustment(unified, ranges)

    def _parameter_adjustment_ranges(self) -> dict | None:
        if not bool(getattr(self.config, "gui_lip_vrcft_adjust", False)):
            return None
        ranges = {}
        for group in _VRCFT_ADJUST_GROUPS:
            ranges[group] = (
                float(getattr(self.config, f"gui_lip_adjust_{group}_min", 0.0)),
                float(getattr(self.config, f"gui_lip_adjust_{group}_max", 1.0)),
            )
        return ranges

    def _output_mode(self) -> str:
        mode = str(getattr(self.config, "gui_lip_output_mode", "vrcft"))
        if mode not in ("vrcft", "direct", "new_smooth"):
            return "vrcft"
        return mode

    def _apply_optional_bounce(self, values: dict, *, exclude=frozenset()) -> dict:
        if not bool(getattr(self.config, "gui_lip_bounce_enable", False)):
            self._mouth_bounce.reset()
            return values
        return self._mouth_bounce(
            values,
            response_hz=float(
                getattr(self.config, "gui_lip_bounce_response_hz", 5.0)
            ),
            damping=float(getattr(self.config, "gui_lip_bounce_damping", 0.55)),
            mix=float(getattr(self.config, "gui_lip_bounce_mix", 35.0)),
            exclude=exclude,
        )

    def _apply_optional_global_response(self, values: dict) -> dict:
        if not bool(getattr(self.config, "gui_lip_response_enable", False)):
            return values
        return apply_global_response(
            values,
            minimum=float(getattr(self.config, "gui_lip_response_min", 0.0)),
            maximum=float(getattr(self.config, "gui_lip_response_max", 1.0)),
            curve=float(getattr(self.config, "gui_lip_response_curve", 1.0)),
        )

    def _apply_new_smooth(self, values: dict) -> dict:
        stability = str(
            getattr(
                self.config, "gui_lip_smooth_tongue_stability", "balanced"
            )
        )
        presets = {
            # speed, angular speed, protrusion Hz, reversal frames, deadband
            "responsive": (5.0, 540.0, 18.0, 2, 0.004),
            "balanced": (3.5, 360.0, 12.0, 2, 0.006),
            "strong": (2.0, 240.0, 8.0, 3, 0.010),
        }
        (
            tongue_speed,
            angle_speed,
            protrusion_hz,
            reversal_frames,
            deadband,
        ) = presets.get(stability, presets["balanced"])
        return self._new_smooth(
            values,
            tongue_speed=tongue_speed,
            tongue_angle_speed=angle_speed,
            tongue_protrusion_hz=protrusion_hz,
            tongue_retracted_range=float(
                getattr(
                    self.config,
                    "gui_lip_smooth_tongue_retracted_range",
                    0.35,
                )
            ),
            tongue_reversal_frames=reversal_frames,
            deadband=deadband,
        )

    def get_preview_frame(self, debug=False):
        """Return a UI-safe BGR snapshot; never exposes inference buffers."""
        with self._preview_lock:
            source = self._model_preview if debug else self._raw_preview
            if source is None:
                return None
            image = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        if not debug:
            cv2.line(image, (400, 0), (400, image.shape[0] - 1),
                     (64, 180, 255), 1)
            return image

        cv2.line(image, (400, 0), (400, image.shape[0] - 1), (64, 180, 255), 1)
        cv2.putText(image, f"presence {self.presence:.3f}  {self._output_mode()}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(image, "model / post", (166, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 210, 255), 1, cv2.LINE_AA)
        top = sorted(self.last_shapes.items(), key=lambda item: -item[1])[:8]
        for row, (name, value) in enumerate(top):
            y = 42 + row * 20
            width = int(max(0.0, min(1.0, value)) * 150)
            cv2.rectangle(image, (8, y - 10), (158, y + 2), (45, 45, 45), -1)
            cv2.rectangle(image, (8, y - 10), (8 + width, y + 2), (80, 210, 255), -1)
            cv2.putText(image, f"{name} {value:.3f}", (166, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1, cv2.LINE_AA)

        diagnostics = None
        shared = self._shared_frame()
        client = getattr(shared, "_vrcft_client", None) if shared is not None else None
        if client is not None and hasattr(client, "get_diagnostics"):
            diagnostics = client.get_diagnostics()
        if diagnostics:
            cv2.putText(image, "resolved VRChat values", (410, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 220, 255), 1, cv2.LINE_AA)
            wire = sorted(
                diagnostics["resolved_mouth"].items(),
                key=lambda item: -abs(item[1]),
            )[:8]
            for row, (name, value) in enumerate(wire):
                y = 42 + row * 20
                short_name = name.removeprefix("v2/")
                cv2.putText(image, f"{short_name} {value:+.3f}", (410, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1, cv2.LINE_AA)
            echo = diagnostics["echo_ms"]
            echo_text = "n/a" if echo is None else f"{echo:.1f}ms"
            cv2.putText(
                image,
                f"infer {self.inference_hz:.1f}Hz  graph {diagnostics['tracking_hz']:.1f}Hz",
                (8, 374), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (180, 230, 180), 1, cv2.LINE_AA,
            )
            cv2.putText(
                image,
                f"OSC {diagnostics['send_hz']:.1f}Hz / {diagnostics['send_messages']} changed  echo {echo_text}",
                (410, 374), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (180, 230, 180), 1, cv2.LINE_AA,
            )
        return image

    def _shared_frame(self):
        if self.shared_data is not None:
            return self.shared_data
        if self.shared_data_provider is not None:
            try:
                return self.shared_data_provider()
            except Exception:
                return None
        return None

    def _emit(self, unified: dict, improved_v2: dict | None = None):
        shared = self._shared_frame()
        if shared is not None:
            lock = getattr(shared, "_tracking_lock", None)
            with lock if lock is not None else nullcontext():
                shapes = shared.shapes
                for name, value in unified.items():
                    shapes[name] = value
                shared._lip_v2_overrides = improved_v2 or {}
                shared.expression_tracking_active = True
            # Improved mode is genuinely direct: do not wait for an eye frame
            # to recompute the graph or for the periodic 100 Hz sender tick.
            immediate = getattr(shared, "_lip_immediate_sender", None)
            if improved_v2 and callable(immediate):
                immediate(improved_v2)
        elif self._own_client is not None:
            for name, value in unified.items():
                self._own_client_data.shapes[name] = value
            self._own_client_data._lip_v2_overrides = improved_v2 or {}
            if improved_v2:
                self._own_client.send_immediate(improved_v2)
            else:
                self._own_client.update_tracking(self._own_client_data)
