# 003: Proper VRCFaceTracking support

Status: implemented, untested with VRCFT · Priority: normal

Some users want to keep running VRCFaceTracking, for other trackers, its
avatar parameter UI, or modules. FaceTrackVR's default is to act as its own
VRCFT (embedded pyVRCFT).

## Decision

Don't write a custom VRCFT module. Eyes already work through the ETVR
module. For the mouth, use the existing **VRCFT-Babble** module (Baballonia
project, in VRCFT's module registry). It listens for Babble OSC on
`127.0.0.1:8888` and fills VRCFT's Unified Expressions shapes. Its config
defaults to `IsEyeSupported: false` and `IsFaceSupported: true`, which is
exactly the split we want.

## What landed

- In the VRCFT module output mode (`gui_use_module`), the mouth goes to
  VRCFT-Babble through the Babble sender in mouth-only mode
  (`send_eyes=False`), at the Babble address and port. Eyes still go to the
  ETVR module.
- The lip tracker's own direct-to-VRChat client is no longer used in that
  mode, so VRCFT owns all output to VRChat. That gives VRCFT's v1/v2 mapping,
  calibration and parameter adjustment, and no double output on port 9000.
- The VRCFT mode tooltip tells users to install VRCFT-Babble.
- Test: `test_osc_sender_babble_routing` in `tests/test_babble_sender.py`.

## Still to do

- Verify with VRCFT running both modules.
- Without VRCFT-Babble installed, the mouth goes nowhere in this mode, where
  it previously went straight to VRChat. Consider detecting this, or say it in
  the README.
- The mouth reaches VRCFT through Babble's reduced key set (see 002).
- The VRCFT tooltip changed only in `en.toml`. Other languages keep the old
  translated text.
