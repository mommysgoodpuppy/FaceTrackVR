"""
Babble-compatible OSC output (Baballonia's protocol).

Two receivers consume it:

  * Resonite's built-in Babble driver, so FaceTrackVR drives Resonite avatars
    without Baballonia, VRCFT, or a Resonite mod running. If Resonite does not
    detect a Baballonia process, force the driver with the ``-ForceBabble``
    launch argument; the receive port is configurable under Settings ->
    Devices.
  * The VRCFT-Babble VRCFaceTracking module, which is how FaceTrackVR's mouth
    reaches VRCFT in the VRCFT module output mode. Eyes keep using the ETVR
    module there, so this sender runs mouth-only (``send_eyes=False``).

Wire reference (Baballonia repo, ``src/VRCFaceTracking.Baballonia``):
  * ``BabbleOSC.cs`` — UDP receiver, float-only OSC messages, one param
    per datagram (NOT bundles: the receiver parses a single OscMessage).
  * ``BabbleVRC.cs`` — applies EyeLeftX/Y to gaze, EyeLeftLid to openness,
    EyeLeftWiden to EyeWide. Squint (``/LeftEyeBrow``) and lower
    (``/LeftEyeLower``) are commented out on both the send
    (``ParameterSenderService.cs``) and apply sides, so they are
    deliberately NOT emitted here (cf. Resonite-Issues #6821).
  * ``BabbleExpressions.cs`` — lower-face address table. Most entries are
    the UE shape name with a lowercased first letter; MouthFunnel,
    MouthPucker, MouthLeft/Right and MouthRollUpper/Lower are pre-collapsed
    keys fed from several UE shapes, so this sender collapses with max().

Like PyVRCFTSender this owns a shared ``UnifiedTrackingData`` frame so lip
tracking (``lip.py``) publishes mouth shapes into it instead of spawning
its own client. Unlike VRCFT there is no avatar discovery or send thread:
every eye or mouth frame is packed and sent immediately, mirroring
Baballonia's ~100 Hz full-set ``ParameterSenderService`` tick.
"""

from __future__ import annotations

import logging
import socket
import threading

from eye import EyeId
from osc.OSCMessage import OSCMessage
from osc.VRChatOSCSender import VRChatOSCSender

from pyvrcft import UnifiedTrackingData
from pyvrcft import osc as osc_codec
from pyvrcft.expressions import UNIFIED_EXPRESSIONS

logger = logging.getLogger(__name__)

UNIFIED_EXPRESSION_SET = frozenset(UNIFIED_EXPRESSIONS)

# UE shapes NEXT produces itself; an auxiliary model must not override them.
# (Same rule as PyVRCFTSender.)
NEXT_OWNED_SHAPES = {
    "BrowInnerUpLeft",
    "BrowInnerUpRight",
    "BrowOuterUpLeft",
    "BrowOuterUpRight",
    "CheekSquintLeft",
    "CheekSquintRight",
    "EyeSquintLeft",
    "EyeSquintRight",
}

# UE shape -> Babble OSC address, 1:1 entries from BabbleExpressions.cs.
SHAPE_ADDRESSES = {
    "CheekPuffLeft": "/cheekPuffLeft",
    "CheekPuffRight": "/cheekPuffRight",
    "CheekSuckLeft": "/cheekSuckLeft",
    "CheekSuckRight": "/cheekSuckRight",
    "JawOpen": "/jawOpen",
    "JawForward": "/jawForward",
    "JawLeft": "/jawLeft",
    "JawRight": "/jawRight",
    "NoseSneerLeft": "/noseSneerLeft",
    "NoseSneerRight": "/noseSneerRight",
    "MouthUpperUpLeft": "/mouthUpperUpLeft",
    "MouthUpperUpRight": "/mouthUpperUpRight",
    "MouthLowerDownLeft": "/mouthLowerDownLeft",
    "MouthLowerDownRight": "/mouthLowerDownRight",
    "MouthPressLeft": "/mouthPressLeft",
    "MouthPressRight": "/mouthPressRight",
    "MouthStretchLeft": "/mouthStretchLeft",
    "MouthStretchRight": "/mouthStretchRight",
    "MouthFrownLeft": "/mouthFrownLeft",
    "MouthFrownRight": "/mouthFrownRight",
    "MouthDimpleLeft": "/mouthDimpleLeft",
    "MouthDimpleRight": "/mouthDimpleRight",
    "MouthCornerPullLeft": "/mouthSmileLeft",
    "MouthCornerPullRight": "/mouthSmileRight",
    "MouthRaiserUpper": "/mouthShrugUpper",
    "MouthRaiserLower": "/mouthShrugLower",
    "MouthClosed": "/mouthClose",
    "TongueOut": "/tongueOut",
    "TongueUp": "/tongueUp",
    "TongueDown": "/tongueDown",
    "TongueLeft": "/tongueLeft",
    "TongueRight": "/tongueRight",
    "TongueRoll": "/tongueRoll",
    "TongueBendDown": "/tongueBendDown",
    "TongueCurlUp": "/tongueCurlUp",
    "TongueSquish": "/tongueSquish",
    "TongueFlat": "/tongueFlat",
    "TongueTwistLeft": "/tongueTwistLeft",
    "TongueTwistRight": "/tongueTwistRight",
}

# Pre-collapsed Baballonia pipeline keys: UE shapes -> shared address.
COLLAPSED_ADDRESSES = {
    "/mouthFunnel": (
        "LipFunnelUpperLeft",
        "LipFunnelUpperRight",
        "LipFunnelLowerLeft",
        "LipFunnelLowerRight",
    ),
    "/mouthPucker": (
        "LipPuckerUpperLeft",
        "LipPuckerUpperRight",
        "LipPuckerLowerLeft",
        "LipPuckerLowerRight",
    ),
    "/mouthLeft": ("MouthUpperLeft", "MouthLowerLeft"),
    "/mouthRight": ("MouthUpperRight", "MouthLowerRight"),
    "/mouthRollUpper": ("LipSuckUpperLeft", "LipSuckUpperRight"),
    "/mouthRollLower": ("LipSuckLowerLeft", "LipSuckLowerRight"),
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def frame_to_babble_params(
    data: UnifiedTrackingData, include_eyes: bool = True
) -> list[tuple[str, float]]:
    """Map one canonical tracking frame to (address, value) Babble params."""
    params: list[tuple[str, float]] = []
    if include_eyes:
        params += [
            ("/LeftEyeX", float(data.eye.left.gaze_x)),
            ("/LeftEyeY", float(data.eye.left.gaze_y)),
            ("/LeftEyeLid", _clamp01(data.eye.left.openness)),
            ("/LeftEyeWiden", _clamp01(data.shapes["EyeWideLeft"])),
            ("/RightEyeX", float(data.eye.right.gaze_x)),
            ("/RightEyeY", float(data.eye.right.gaze_y)),
            ("/RightEyeLid", _clamp01(data.eye.right.openness)),
            ("/RightEyeWiden", _clamp01(data.shapes["EyeWideRight"])),
        ]
    for shape, address in SHAPE_ADDRESSES.items():
        params.append((address, _clamp01(data.shapes[shape])))
    for address, shapes in COLLAPSED_ADDRESSES.items():
        params.append((address, _clamp01(max(data.shapes[s] for s in shapes))))
    return params


class BabbleSender:
    """Stateless Babble-OSC backend. Same 4-method shape as PyVRCFTSender
    (``start`` / ``stop`` / ``output_osc_info`` / ``output_eyebrow_info``)
    plus ``send_frame``, which the lip tracker calls after each mouth frame."""

    def __init__(self):
        self.sock: socket.socket | None = None
        self.host: str = "127.0.0.1"
        self.port: int = 8888
        self.data = UnifiedTrackingData()
        self._tracking_lock = threading.RLock()
        # Auxiliary producers (notably the VFT lip tracker) use the same lock
        # so one Babble output snapshot cannot contain half of two lip frames.
        self.data._tracking_lock = self._tracking_lock
        self.is_single_eye = False
        self.send_eyes = True

    def start(self, config, send_eyes: bool = True):
        """Open the UDP socket to the Babble receiver (Resonite's driver or
        the VRCFT-Babble module)."""
        self.host = str(config.gui_babble_address)
        self.port = int(config.gui_babble_port)
        self.send_eyes = send_eyes
        self.data._lip_frame_sender = self.send_frame
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logger.info(
            "Babble sender started -> %s:%s%s",
            self.host,
            self.port,
            "" if send_eyes else " (mouth only)",
        )

    def stop(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None
            self.data._lip_frame_sender = None
            logger.info("Babble sender stopped")

    def _send_frame_locked(self):
        sock = self.sock
        if sock is None:
            return
        for address, value in frame_to_babble_params(self.data, self.send_eyes):
            try:
                sock.sendto(
                    osc_codec.encode_message(address, [value]),
                    (self.host, self.port),
                )
            except OSError:
                logger.warning("Babble send failed", exc_info=True)
                break

    def send_frame(self):
        """Flush the current shared frame (called after each mouth frame)."""
        with self._tracking_lock:
            self._send_frame_locked()

    def output_osc_info(self, osc_message: OSCMessage, main_config, config):
        with self._tracking_lock:
            self._output_osc_info_locked(osc_message, main_config, config)

    def _apply_eye(self, eye, eye_info, openness):
        eye.gaze = (float(eye_info.x), float(eye_info.y))
        eye.openness = float(openness)

    def _output_osc_info_locked(self, osc_message, main_config, config):
        eye_id, eye_info = osc_message.data
        self.is_single_eye = VRChatOSCSender.get_is_single_eye(
            main_config.eye_display_id
        )

        # eye_info.blink is openness (1 = open, 0 = closed); mirror the eyelid
        # inversion setting the other modes apply via _eyelid_transformer.
        openness = (
            1.0 - eye_info.blink if config.osc_invert_eye_close else eye_info.blink
        )
        squeeze = float(eye_info.squeeze)
        lip_owns_cheeks = bool(
            getattr(self.data, "_lip_owns_cheek_squint", False)
        )

        if self.is_single_eye:
            self._apply_eye(self.data.eye.left, eye_info, openness)
            self._apply_eye(self.data.eye.right, eye_info, openness)
            self.data.shapes["EyeSquintLeft"] = squeeze
            self.data.shapes["EyeSquintRight"] = squeeze
            if not lip_owns_cheeks:
                self.data.shapes["CheekSquintLeft"] = squeeze
                self.data.shapes["CheekSquintRight"] = squeeze
        elif eye_id == EyeId.LEFT:
            self._apply_eye(self.data.eye.left, eye_info, openness)
            self.data.shapes["EyeSquintLeft"] = squeeze
            if not lip_owns_cheeks:
                self.data.shapes["CheekSquintLeft"] = squeeze
        elif eye_id == EyeId.RIGHT:
            self._apply_eye(self.data.eye.right, eye_info, openness)
            self.data.shapes["EyeSquintRight"] = squeeze
            if not lip_owns_cheeks:
                self.data.shapes["CheekSquintRight"] = squeeze

        for name, value in getattr(eye_info, "auxiliary_expressions", {}).items():
            if name in UNIFIED_EXPRESSION_SET and name not in NEXT_OWNED_SHAPES:
                self.data.shapes[name] = float(value)

        self._send_frame_locked()

    def output_eyebrow_info(self, eye_id, brow_val: float, main_config):
        with self._tracking_lock:
            self._output_eyebrow_info_locked(eye_id, brow_val, main_config)

    def _output_eyebrow_info_locked(self, eye_id, brow_val, main_config):
        is_single = VRChatOSCSender.get_is_single_eye(main_config.eye_display_id)
        brow_val = float(brow_val)
        if is_single or eye_id == EyeId.LEFT:
            self.data.shapes["BrowInnerUpLeft"] = brow_val
            self.data.shapes["BrowOuterUpLeft"] = brow_val
        if is_single or eye_id == EyeId.RIGHT:
            self.data.shapes["BrowInnerUpRight"] = brow_val
            self.data.shapes["BrowOuterUpRight"] = brow_val
        # Brows have no stable Babble address (EyeBrow/Lower are commented
        # out upstream), so no send here; the shapes stay for future use.
