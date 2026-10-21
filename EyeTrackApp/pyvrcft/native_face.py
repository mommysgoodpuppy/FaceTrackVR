"""VRChat native face tracking over OSC (``/tracking/face/v1``).

VRChat's native face tracking takes one OSC blob of exactly 102 bytes, one
unsigned byte per tracking key in the order below (decoded as ``byte / 255``).
Every message carries the full set; unsupported keys stay 0. The avatar maps
the keys itself (Avatar Descriptor -> Face Tracking) and VRChat syncs them to
remote viewers without custom synced parameters.

Reference: https://vrc-beta-docs.netlify.app/avatars/face-tracking/osc

The keys are Unified Expressions names. Everything except the eye look,
eyelid and pupil keys is a UE shape weight taken straight from
``UnifiedTrackingData.shapes``; those eye keys are derived from the per-eye
gaze/openness data and the normalized pupil value.
"""

from __future__ import annotations

ADDRESS = "/tracking/face/v1"

TRACKING_KEYS = (
    # 0-7 eye look
    "EyeLookOutRight", "EyeLookInRight", "EyeLookUpRight", "EyeLookDownRight",
    "EyeLookOutLeft", "EyeLookInLeft", "EyeLookUpLeft", "EyeLookDownLeft",
    # 8-17 eye state
    "EyeClosedRight", "EyeClosedLeft", "EyeDilationRight", "EyeDilationLeft",
    "EyeConstrictRight", "EyeConstrictLeft", "EyeSquintRight", "EyeSquintLeft",
    "EyeWideRight", "EyeWideLeft",
    # 18-35 brow, nose, cheek
    "BrowPinchRight", "BrowPinchLeft", "BrowLowererRight", "BrowLowererLeft",
    "BrowInnerUpRight", "BrowInnerUpLeft", "BrowOuterUpRight", "BrowOuterUpLeft",
    "NasalDilationRight", "NasalDilationLeft", "NasalConstrictRight", "NasalConstrictLeft",
    "CheekSquintRight", "CheekSquintLeft", "CheekPuffRight", "CheekPuffLeft",
    "CheekSuckRight", "CheekSuckLeft",
    # 36-43 jaw
    "JawOpen", "JawRight", "JawLeft", "JawForward", "JawBackward", "JawClench",
    "JawMandibleRaise", "MouthClosed",
    # 44-57 lip suck, funnel, pucker
    "LipSuckUpperRight", "LipSuckUpperLeft", "LipSuckLowerRight", "LipSuckLowerLeft",
    "LipSuckCornerRight", "LipSuckCornerLeft",
    "LipFunnelUpperRight", "LipFunnelUpperLeft", "LipFunnelLowerRight", "LipFunnelLowerLeft",
    "LipPuckerUpperRight", "LipPuckerUpperLeft", "LipPuckerLowerRight", "LipPuckerLowerLeft",
    # 58-69 mouth upper/lower
    "MouthUpperUpRight", "MouthUpperUpLeft", "MouthUpperDeepenRight", "MouthUpperDeepenLeft",
    "NoseSneerRight", "NoseSneerLeft", "MouthLowerDownRight", "MouthLowerDownLeft",
    "MouthUpperRight", "MouthUpperLeft", "MouthLowerRight", "MouthLowerLeft",
    # 70-85 mouth corners and shape
    "MouthCornerPullRight", "MouthCornerPullLeft", "MouthCornerSlantRight",
    "MouthCornerSlantLeft", "MouthFrownRight", "MouthFrownLeft",
    "MouthStretchRight", "MouthStretchLeft", "MouthDimpleRight", "MouthDimpleLeft",
    "MouthRaiserUpper", "MouthRaiserLower", "MouthPressRight", "MouthPressLeft",
    "MouthTightenerRight", "MouthTightenerLeft",
    # 86-101 tongue, throat, neck
    "TongueOut", "TongueUp", "TongueDown", "TongueRight", "TongueLeft", "TongueRoll",
    "TongueBendDown", "TongueCurlUp", "TongueSquish", "TongueFlat",
    "TongueTwistRight", "TongueTwistLeft", "SoftPalateClose", "ThroatSwallow",
    "NeckFlexRight", "NeckFlexLeft",
)
assert len(TRACKING_KEYS) == 102


def _byte(value: float) -> int:
    return int(round(min(1.0, max(0.0, float(value))) * 255.0))


def compute_native_face_keys(data, pupil: float = 0.5) -> dict:
    """Tracking key -> weight (0..1) for one frame.

    ``pupil`` is the normalized pupil value (0 constricted, 0.5 neutral,
    1 dilated), split into the separate dilation/constriction keys.
    """
    w = data.shapes
    left, right = data.eye.left, data.eye.right
    dilation = max(0.0, (pupil - 0.5) * 2.0)
    constrict = max(0.0, (0.5 - pupil) * 2.0)

    keys = {}
    for name in TRACKING_KEYS[18:]:
        keys[name] = w[name]
    keys.update({
        # Gaze is x right, y up. Looking right moves the right eye outward
        # and the left eye inward.
        "EyeLookOutRight": max(0.0, right.gaze_x),
        "EyeLookInRight": max(0.0, -right.gaze_x),
        "EyeLookUpRight": max(0.0, right.gaze_y),
        "EyeLookDownRight": max(0.0, -right.gaze_y),
        "EyeLookOutLeft": max(0.0, -left.gaze_x),
        "EyeLookInLeft": max(0.0, left.gaze_x),
        "EyeLookUpLeft": max(0.0, left.gaze_y),
        "EyeLookDownLeft": max(0.0, -left.gaze_y),
        "EyeClosedRight": 1.0 - right.openness,
        "EyeClosedLeft": 1.0 - left.openness,
        "EyeDilationRight": dilation,
        "EyeDilationLeft": dilation,
        "EyeConstrictRight": constrict,
        "EyeConstrictLeft": constrict,
        "EyeSquintRight": w["EyeSquintRight"],
        "EyeSquintLeft": w["EyeSquintLeft"],
        "EyeWideRight": w["EyeWideRight"],
        "EyeWideLeft": w["EyeWideLeft"],
    })
    return keys


def encode_native_face(data, pupil: float = 0.5) -> bytes:
    """The 102-byte ``/tracking/face/v1`` payload for one frame."""
    keys = compute_native_face_keys(data, pupil)
    return bytes(_byte(keys[name]) for name in TRACKING_KEYS)

