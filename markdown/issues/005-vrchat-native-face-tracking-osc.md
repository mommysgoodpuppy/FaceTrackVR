# 005: VRChat native face tracking OSC (`/tracking/face/v1`)

Status: open · Priority: normal

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

## Plan

- Add a packer in pyVRCFT: `UnifiedTrackingData` to 102 bytes, with gaze
  split into the look-in/out/up/down keys and openness turned into
  EyeClosed.
- Send it from the pyVRCFT send loop, like the existing native
  `/tracking/eye/*` endpoints.
- Choose dynamically per avatar. If the avatar's OSC config declares FT
  params (v2 or v1), keep sending those. Otherwise send native face. Mirror
  the existing `native_gaze_relevant` logic.
- Find out how VRChat exposes native-FT avatar support, for example in the
  OSCQuery tree or the avatar JSON, before relying on "no FT params" alone.
