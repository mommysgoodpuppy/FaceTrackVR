# 001: SRanipal / VRCFT v1 avatars get no mouth (and reportedly no eyes)

Status: mouth **fixed**, eyes **needs-info** · Priority: high

## Report

> the mouth tracking isnt working on sranipal avatars either through pyvrcft
> or vrcft proper. ue native avatars work right. cant get the eyes to track on
> any algo, cams work fine though.

Follow-up (relayed): "your app wasn't outputting anything with the V1
parameters". Parameter list: <https://docs.vrcft.io/docs/v4.0/category/parameters>.
Example avatar: `avtr_0107e441-baee-476f-9ecc-89e8550f787f`.

"V1" means VRCFT's legacy SRanipal-era parameters: bare names such as
`JawOpen`, `MouthApeShape`, `SmileSadRight` and `EyesX`, with no `v2/`
prefix. VRCFT still emits them (`UnifiedTracking.AllParameters_v1`).

## Findings

- **Mouth (root cause):** the lip tracker only wrote Unified Expressions
  shapes, and pyVRCFT only derived `v2/*` lip parameters from them. The legacy
  path (`compute_legacy_outputs`) covered eyes only, so an SRanipal avatar
  received no lip parameters at all. The same applied with VRCFT proper,
  because FaceTrackVR's mouth output never goes through VRCFT (see 003).
- **Eyes:** not reproduced. Against a synthetic v1 avatar, pyVRCFT sends
  `EyesX/Y`, `Left/RightEyeLid`, `EyesDilation`, the binary
  `LeftEyeLidExpandedSqueeze{1,2,Negative}` bits and `EyeTrackingActive`.

## Fix (mouth)

`pyvrcft/expressions.py` now ports VRCFT's `UnifiedSRanMapper`,
`LipShapeMerger` and `LipShapeConversion`: all 37 SRanipal shapes plus every
merged shape. They are emitted with the legacy eye params on every graph
update. The lip tracker's direct and new-smooth modes also include them in
their immediate packet. Tests: `tests/test_pyvrcft_legacy_lip.py`.

## Still to do

- Ask the reporter for a log from one run with the v1 avatar loaded. The line
  `Avatar ... N avatar params, M/K VRCFT params matched` shows whether the
  avatar resolved. Also ask which output mode and eye algorithm they used.
- "Eyes don't track on any algorithm" while the cameras work may be a
  tracking problem rather than an output problem. Check the in-app values
  before blaming OSC.
- v1 `*LidExpandedSqueeze` is driven with plain openness, not SRanipal's
  squeeze/widen packing (0.8 = neutral). Behaviour is unchanged from
  EyeTrackVR, but v1 avatars tuned for VRCFT may look slightly wide.
- Verify in game with the example avatar.
