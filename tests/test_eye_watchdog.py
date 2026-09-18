from eye_watchdog import EyePipelineWatchdog


def sample(watchdog, now, **overrides):
    values = {
        "active": True,
        "capture_last_frame": now,
        "capture_last_change": now,
        "capture_last_usable_contrast": now,
        "output_last_frame": now,
        "workers_alive": True,
    }
    values.update(overrides)
    return watchdog.observe(now=now, **values)


def test_watchdog_ignores_startup_then_detects_inference_stall():
    watchdog = EyePipelineWatchdog(startup_grace_s=10.0)
    assert sample(watchdog, 0.0) is None
    assert sample(watchdog, 9.9, output_last_frame=0.0) is None
    assert sample(watchdog, 10.1, output_last_frame=6.0) == (
        "capture advanced but eye inference stopped"
    )


def test_watchdog_detects_open_camera_repeating_same_image():
    watchdog = EyePipelineWatchdog(
        startup_grace_s=1.0,
        frozen_frame_timeout_s=5.0,
    )
    assert sample(watchdog, 0.0) is None
    assert sample(watchdog, 6.0, capture_last_change=0.5) == (
        "capture repeated an unchanged image"
    )


def test_watchdog_detects_camera_streaming_blank_images():
    watchdog = EyePipelineWatchdog(startup_grace_s=1.0, flat_frame_timeout_s=5.0)
    assert sample(watchdog, 0.0) is None
    assert sample(watchdog, 6.0, capture_last_usable_contrast=0.5) == (
        "capture produced blank or near-flat images"
    )


def test_watchdog_detects_dead_capture_and_worker():
    watchdog = EyePipelineWatchdog(startup_grace_s=1.0, capture_timeout_s=2.0)
    assert sample(watchdog, 0.0) is None
    assert sample(watchdog, 3.0, capture_last_frame=0.5) == (
        "capture stopped advancing"
    )

    watchdog.reset()
    assert sample(watchdog, 10.0) is None
    assert sample(watchdog, 12.0, workers_alive=False) == "an eye worker exited"


def test_watchdog_recovery_is_bounded_until_sustained_health():
    watchdog = EyePipelineWatchdog(
        startup_grace_s=1.0,
        inference_timeout_s=1.0,
        rearm_after_s=2.0,
    )
    assert sample(watchdog, 0.0) is None
    assert sample(watchdog, 2.0, output_last_frame=0.5) is not None
    assert watchdog.observe(
        now=2.5,
        active=False,
        capture_last_frame=0.0,
        capture_last_change=0.0,
        capture_last_usable_contrast=0.0,
        output_last_frame=0.0,
        workers_alive=False,
    ) is None
    assert watchdog.recovering
    assert sample(watchdog, 3.0, output_last_frame=0.5) is None

    watchdog.finish_recovery(3.0)
    assert sample(watchdog, 4.1) is None
    assert sample(watchdog, 6.2) is None
    assert sample(watchdog, 6.3, output_last_frame=4.0) is not None


def test_watchdog_resets_when_tracking_is_intentionally_stopped():
    watchdog = EyePipelineWatchdog(startup_grace_s=1.0)
    assert sample(watchdog, 0.0) is None
    assert watchdog.observe(
        now=2.0,
        active=False,
        capture_last_frame=0.0,
        capture_last_change=0.0,
        capture_last_usable_contrast=0.0,
        output_last_frame=0.0,
        workers_alive=False,
    ) is None
    assert sample(watchdog, 20.0, output_last_frame=0.0) is None


def test_watchdog_circuit_breaker_stops_restart_loop_until_manual_reset():
    watchdog = EyePipelineWatchdog(
        startup_grace_s=0.0,
        inference_timeout_s=1.0,
        rearm_after_s=0.0,
        max_recoveries=2,
        recovery_window_s=60.0,
    )

    assert sample(watchdog, 0.0) is None
    assert sample(watchdog, 2.0, output_last_frame=0.5) is not None
    watchdog.finish_recovery(2.0)
    assert sample(watchdog, 2.1) is None
    assert sample(watchdog, 4.0, output_last_frame=2.1) is not None
    watchdog.finish_recovery(4.0)
    assert sample(watchdog, 4.1) is None
    assert sample(watchdog, 6.0, output_last_frame=4.1) is not None
    assert watchdog.circuit_open
    assert not watchdog.recovering
    assert sample(watchdog, 8.0, output_last_frame=4.1) is None

    watchdog.reset()
    assert not watchdog.circuit_open
    assert sample(watchdog, 10.0) is None
    assert sample(watchdog, 12.0, output_last_frame=10.0) is not None
