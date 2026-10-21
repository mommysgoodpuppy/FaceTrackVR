# 002: Full Resonite support

Status: implemented, untested in game · Priority: normal

The Babble-protocol output from the `FaceTrackVR-resonite` worktree has been
ported onto `experimental` by hand, not rebased. The worktree's uncommitted
diff also deleted unrelated lines (`gui_pupil_dilation`, the English
`osc.receive` strings, a `_TEMPLATE.toml` comment and a section header), and
those deletions were left out. The worktree is untouched and now superseded;
remove it once nothing else is needed from it.

## What landed

- `osc/BabbleSender.py` drives Resonite's built-in Babble driver with
  Baballonia's OSC protocol: one float message per datagram, sent on every
  eye or mouth frame. The address table was checked against Baballonia's
  `BabbleOSC.cs` and `BabbleExpressions.cs`.
- "Babble (Resonite)" output mode, with the Babble address and port (default
  `127.0.0.1:8888`) in the advanced OSC settings.
- The lip tracker publishes into the sender's shared frame and flushes it
  after each mouth frame (`_lip_frame_sender`), in every lip output mode.
- Tests: `tests/test_babble_sender.py`.

## Still to do

- Verify in Resonite. If the driver doesn't start, launch with `-ForceBabble`.
- Babble supports neither squint nor the lower lid, on either side
  (Resonite-Issues #6821). Brows have no Babble address.
- Shapes Babble collapses (funnel, pucker, mouth left/right, roll) lose their
  per-side detail.
- Document the setup in the README.
