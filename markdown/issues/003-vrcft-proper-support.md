# 003: Proper VRCFaceTracking support

Status: open · Priority: normal

Some users want to keep running VRCFaceTracking, for other trackers, its
avatar parameter UI, or modules. FaceTrackVR's default is to act as its own
VRCFT (embedded pyVRCFT).

## Current state

- The "VRCFT module" path (`osc/VRCFTModuleMessenger.py`) only forwards eye
  data to the ETVR module on port 8889.
- Mouth output always goes straight to VRChat (shared pyVRCFT frame or the lip
  tracker's own client). It never reaches VRCFT, so VRCFT cannot apply its
  own v1/v2 mapping, calibration or parameter adjustment to the mouth.
- When both run, the two apps send to VRChat's port 9000 independently and
  can fight over the same parameters.

## Direction

- A real VRCFT module, or an extension of the ETVR one, that receives eye and
  mouth Unified Expressions from FaceTrackVR. VRCFT would then own all output
  to VRChat.
- An explicit "VRCFT owns output" mode that disables every direct sender,
  including the lip tracker's own client.
- Detect a running VRCFT (it holds port 9001) and warn about double output.
