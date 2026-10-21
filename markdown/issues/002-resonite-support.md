# 002: Full Resonite support

Status: in-progress · Priority: normal

Uncommitted work-in-progress is in the `FaceTrackVR-resonite` worktree
(`/home/ellie/git/FaceTrackVR-resonite`, branch `FaceTrackVR-resonite`):
`osc/BabbleSender.py` drives Resonite's built-in Babble driver over
Baballonia's OSC protocol, with settings UI and `tests/test_babble_sender.py`.

## Notes

- The worktree's `config.py` diff **removes `gui_pupil_dilation`**, and the
  comment line about v2/v1/native auto-adapting. This looks accidental;
  restore both before committing.
- The worktree branch is based on `1ce4e7d`, so rebase it onto current
  `experimental` before continuing.
- Define "full": eyes and mouth through Babble, per-eye openness and widen,
  and the shapes Babble collapses (funnel, pucker, roll). Squint and lower lid
  are not supported by Babble on either side (Resonite-Issues #6821).
- Document the `-ForceBabble` launch argument and the receive port in the
  README.
