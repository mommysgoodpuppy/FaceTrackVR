"""Liveness policy for the eye capture/inference pipeline.

The policy is deliberately independent of Tk and worker implementation details
so it can be tested with a synthetic clock.  The UI supplies aggregate
heartbeats for the active cameras and eye processors, then performs the actual
soft restart when ``observe`` returns a reason.
"""

from __future__ import annotations

from collections import deque


class EyePipelineWatchdog:
    """Detect a stalled eye pipeline and request bounded recovery attempts.

    A recovery is re-armed only after the complete pipeline has remained
    healthy for ``rearm_after_s``.  This prevents a disconnected or genuinely
    broken camera from causing an endless restart loop.
    """

    def __init__(
        self,
        *,
        startup_grace_s: float = 10.0,
        capture_timeout_s: float = 3.0,
        frozen_frame_timeout_s: float = 5.0,
        flat_frame_timeout_s: float = 5.0,
        inference_timeout_s: float = 3.0,
        rearm_after_s: float = 10.0,
        max_recoveries: int = 3,
        recovery_window_s: float = 600.0,
    ) -> None:
        self.startup_grace_s = startup_grace_s
        self.capture_timeout_s = capture_timeout_s
        self.frozen_frame_timeout_s = frozen_frame_timeout_s
        self.flat_frame_timeout_s = flat_frame_timeout_s
        self.inference_timeout_s = inference_timeout_s
        self.rearm_after_s = rearm_after_s
        self.max_recoveries = max_recoveries
        self.recovery_window_s = recovery_window_s
        self._active_since: float | None = None
        self._healthy_since: float | None = None
        self._armed = True
        self._recovering = False
        self._recovery_times: deque[float] = deque()
        self._circuit_open = False

    @property
    def recovering(self) -> bool:
        return self._recovering

    @property
    def circuit_open(self) -> bool:
        return self._circuit_open

    def reset(self) -> None:
        self._active_since = None
        self._healthy_since = None
        self._armed = True
        self._recovering = False
        self._recovery_times.clear()
        self._circuit_open = False

    def finish_recovery(self, now: float) -> None:
        """Begin a fresh startup grace period without immediately re-arming."""
        self._recovering = False
        self._active_since = now
        self._healthy_since = None

    def observe(
        self,
        *,
        now: float,
        active: bool,
        capture_last_frame: float,
        capture_last_change: float,
        capture_last_usable_contrast: float,
        output_last_frame: float,
        workers_alive: bool,
        suspended: bool = False,
    ) -> str | None:
        """Return a recovery reason when the current sample is unhealthy.

        Heartbeats are monotonic timestamps.  Aggregate callers should pass the
        oldest timestamp across active cameras/eyes so one stalled member is
        sufficient to make the complete pipeline unhealthy.
        """
        if self._recovering:
            return None
        if not active or suspended:
            self.reset()
            return None
        if self._active_since is None:
            self._active_since = now
            return None
        if now - self._active_since < self.startup_grace_s:
            return None

        reason = None
        if capture_last_frame <= 0.0:
            reason = "capture produced no frames"
        elif now - capture_last_frame > self.capture_timeout_s:
            reason = "capture stopped advancing"
        elif (
            capture_last_usable_contrast <= 0.0
            or now - capture_last_usable_contrast > self.flat_frame_timeout_s
        ):
            reason = "capture produced blank or near-flat images"
        elif (
            capture_last_change > 0.0
            and now - capture_last_change > self.frozen_frame_timeout_s
        ):
            reason = "capture repeated an unchanged image"
        elif not workers_alive:
            reason = "an eye worker exited"
        elif output_last_frame <= 0.0 or now - output_last_frame > self.inference_timeout_s:
            reason = "capture advanced but eye inference stopped"

        if reason is not None:
            self._healthy_since = None
            if self._armed:
                self._armed = False
                cutoff = now - self.recovery_window_s
                while self._recovery_times and self._recovery_times[0] < cutoff:
                    self._recovery_times.popleft()
                if len(self._recovery_times) >= self.max_recoveries:
                    self._circuit_open = True
                    return reason
                self._recovery_times.append(now)
                self._recovering = True
                return reason
            return None

        if not self._armed:
            if self._healthy_since is None:
                self._healthy_since = now
            if now - self._healthy_since >= self.rearm_after_s:
                self._armed = True
                self._healthy_since = None
        return None
