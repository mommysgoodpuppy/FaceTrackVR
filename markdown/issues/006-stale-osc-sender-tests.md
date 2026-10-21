# 006: Stale legacy OSC sender tests

Status: open · Priority: low

`tests/test_osc_v1_params.py`, `test_osc_v2_params.py` and
`test_osc_native_params.py` fail on clean `experimental` (13 cases). The
sender now emits `EyesDilation` and squeeze messages that the expected lists
predate, for example `At index 3 diff: ('/avatar/parameters/EyesDilation', 0)`.
Update the expectations, after checking that the extra messages are
intended. Run tests with `PYTHONPATH=EyeTrackApp` from the repo root, as CI
does.
