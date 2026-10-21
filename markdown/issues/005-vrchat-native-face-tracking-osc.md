# 005: VRChat native face tracking OSC (`/tracking/face/v1`)

Status: implemented, untested in game · Priority: normal

VRChat beta adds native face tracking over OSC:
<https://vrc-beta-docs.netlify.app/avatars/face-tracking/osc>.

> since you do your own thing you might want to look at this and switch
> dynamically between native and FT parameters when they're present in the
> OSC config

## Protocol (from the beta docs)

- Address `/tracking/face/v1`, one OSC blob of **exactly 102 bytes**, one
  `uint8` per key (VRChat decodes it as `byte / 255`). Every message must
  carry all 102 keys; unsupported keys are 0.
- Keys follow Unified Expressions order: eye look (0 to 7), eye state (8 to
  17), brow/nose/cheek (18 to 35), jaw (36 to 43), lips (44 to 57), mouth
  (58 to 85), tongue/throat/neck (86 to 101). `JawOpen` is index 36.
- The avatar needs Face Tracking configured with "Use All Tracking Keys".
  VRChat syncs it natively, so no per-key synced avatar parameters are needed.
- The page does **not** say how an app detects avatar support.

## Implementation

- `pyvrcft/native_face.py` packs `UnifiedTrackingData` into the 102-byte
  payload. Non-eye keys are UE shape weights. Gaze becomes the eight
  EyeLook keys (x right, y up; looking right is right-eye out and left-eye
  in). EyeClosed is `1 - openness`. The normalized pupil (0.5 = neutral)
  splits into EyeDilation above 0.5 and EyeConstrict below it.
- The pyVRCFT OSC codec now encodes and parses blobs (`b`).
- `VRCFTClient` sends it from the send loop as a direct, deduplicated
  message, but only when the avatar's OSC config resolved **no** VRCFT
  parameters (v2 or legacy v1). Avatars with FT parameters keep getting
  those, so they are never double-driven. An unresolved avatar gets native
  face, the same rule as the native `/tracking/eye/*` endpoints.
- Setting `gui_vrc_native_face` (default on) adds a "Native face" checkbox
  next to the OSC output modes. Only the pyVRCFT mode uses it.
- The lip tracker's standalone client (used with non-pyVRCFT output modes)
  never sends native face. The payload must include the eye keys, and that
  client has no eye data.
- Tests: `tests/test_pyvrcft_native_face.py`.

## Open questions

- Verify in the VRChat beta with an avatar that has "Use All Tracking Keys"
  enabled and no VRCFT parameters. Check gaze direction and scale (is full
  normalized gaze the right magnitude for EyeLook = 1?), lids and pupil.
- The docs give no way to detect native face tracking support. If VRChat
  exposes one later (OSCQuery or the avatar JSON), use it instead of the "no
  FT parameters" rule.
- With direct and new-smooth lip modes, native face updates at the eye-frame
  rate, not on every mouth frame.
- The docs give no send rate. The send loop runs at about 100 Hz and only
  resends when the payload changes.
