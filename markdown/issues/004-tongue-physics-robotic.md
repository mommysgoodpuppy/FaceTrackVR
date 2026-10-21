# 004: Tongue physics limits look robotic

Status: open · Priority: low

The tongue limits in `NewSmoothFilter` (`lip.py`) are the "physics" here:
speed and angular-speed slew limits on the polar tongue vector, plus a
reversal guard. They make tongue motion look mechanical: constant-velocity
moves and hard stops.

## Ideas

- Replace linear slew with a critically damped spring, so motion eases in
  and out instead of moving at a constant rate.
- Make the speed limit scale with distance, so small motions are not capped.
- Make the reversal guard probabilistic instead of a fixed frame count.
- Record a short clip, before and after, to judge changes.
