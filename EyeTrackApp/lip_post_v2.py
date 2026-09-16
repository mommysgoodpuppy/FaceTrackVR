#!/usr/bin/env python3
"""Stateful post-processing for compatible 30-value mouth models."""
import numpy as np

from lip_pipeline import GAIN_FRAMES, GAIN_HOLD

# Stable output order used by compatible v2 models.
LIP_SHAPE_V2 = [
    "JawRight", "JawLeft", "JawForward", "JawOpen", "MouthApeShape",
    "MouthUpperRight", "MouthUpperLeft", "MouthLowerRight", "MouthLowerLeft",
    "MouthUpperOverturn", "MouthLowerOverturn", "MouthPout",
    "MouthSmileRight", "MouthSmileLeft", "MouthSadRight", "MouthSadLeft",
    "CheekPuffRight", "CheekPuffLeft", "CheekSuck",
    "MouthUpperUpRight", "MouthUpperUpLeft",
    "MouthLowerDownRight", "MouthLowerDownLeft",
    "MouthUpperInside", "MouthLowerInside", "MouthLowerOverlay",
    "TongueLongStep1",
    "TongueLeft", "TongueRight", "TongueUp", "TongueDown", "TongueRoll",
    "TongueLongStep2",
    "TongueUpRightMorph", "TongueUpLeftMorph",
    "TongueDownRightMorph", "TongueDownLeftMorph",
]

C_DELTA = 0.2
C_DELTA_ALT = 0.55
C_RAW_ALT = 0.9
C_NORM_FLOOR = 0.502
C_PRESENCE = 0.4
C_TSTEP1_GATE = 0.1
C_JAW_TONGUE_OFF = 0.05
C_JAW_MORPH = 0.6
C_TONGUEUP_MIN = 0.3
C_MORPH_ADD = 0.12
C_TSTEP1_DEN = 0.6
C_TSTEP1_NUM = 0.4
C_TSTEP1_MULT = 1.5
C_TSTEP2_GATE = 0.05

TH_EMA_DIV = 20.5
TH_EMA_INC = 0.001
SM_EMA_DIV = 0.5
SM_EMA_INC = 0.25
WARMUP = 26


class LipPostprocessV2:
    def __init__(self):
        self.norm = [0.5] * 32
        self.norm[2] = 0.7
        self.prev_raw = [0.0] * 4
        self.th = [0.5] * 4
        self.ths = [0.5] * 4
        self.thraw = [0.0] * 4
        self.thacc = [0.5] * 4
        self.sm_state = [0.0] * 33
        self.sm_prev = [0.0] * 33
        self.sm_acc = [0.0] * 33
        self.lost_hold = 0
        self.track_hold = 0
        self.was_lost = False
        self.learn_hold = 0

    def _reset_thresholds(self):
        """Loss-reset branch values (0.5, with norm[2]=0.7)."""
        self.th = [0.5] * 4
        self.ths = [0.5] * 4
        self.norm = [0.5] * 32
        self.norm[2] = 0.7
        self.thacc = [0.5] * 4
        self.sm_state = [0.0] * 33
        self.sm_prev = [0.0] * 33
        self.sm_acc = [0.0] * 33

    def __call__(self, raw):
        """raw: 30 model floats -> 60 LipShape_v2 floats."""
        out = [0.0] * 60
        presence = float(raw[0])

        # ---- LOST branch ------------------------------------------------
        if presence < C_PRESENCE:
            if self.was_lost:
                self.lost_hold += 1
            self.was_lost = True
            self.track_hold = 0
            if self.lost_hold > 25:
                out[0:41] = [0.0] * 41
                out[37] = presence
                self._reset_thresholds()
                self.was_lost = True
                self.track_hold = 0
            return out

        # ---- TRACKED branch ---------------------------------------------
        if not self.was_lost:
            self.track_hold += 1
        self.was_lost = False
        self.lost_hold = 0
        if self.track_hold < WARMUP:
            return out

        # Threshold learning uses the current frame without a one-frame delay.
        self.prev_raw = [raw[1], raw[5], raw[6], raw[2]]

        if self.track_hold > GAIN_FRAMES:
            self.learn_hold += 1
            if self.learn_hold > GAIN_HOLD:
                self.learn_hold = 0
                for i in range(4):
                    cur = self.prev_raw[i]
                    delta = abs(cur - self.th[i])
                    if delta < C_DELTA or (i == 3 and delta < C_DELTA_ALT and cur < C_RAW_ALT):
                        self.thraw[i] = cur
                        acc = TH_EMA_INC + self.thacc[i]
                        rate = acc / (acc + TH_EMA_DIV)
                        smoothed = (cur - self.ths[i]) * rate + self.ths[i]
                        self.thacc[i] = (1.0 - rate) * acc
                        self.ths[i] = smoothed
                        self.th[i] = smoothed

        out = [0.0] * 60

        # Complementary shape pairs are split around learned neutral points.
        t = self.th[0]
        if t <= raw[1]:
            out[1] = (raw[1] - t) / (1.0 - t)
        else:
            out[0] = (t - raw[1]) / t
        # JawForward
        if self.th[3] < raw[2]:
            out[2] = raw[2] - self.th[3]
        # JawOpen / MouthApeShape passthrough
        out[3] = raw[3]
        out[4] = raw[4]
        # MouthUpperRight / MouthUpperLeft
        t = self.th[1]
        if t <= raw[5]:
            out[6] = (raw[5] - t) / (1.0 - t)
        else:
            out[5] = (t - raw[5]) / t
        # MouthLowerRight / MouthLowerLeft
        t = self.th[2]
        if t <= raw[6]:
            out[8] = (raw[6] - t) / (1.0 - t)
        else:
            out[7] = (t - raw[6]) / t

        # passthrough blocks
        out[9:17] = raw[7:15]
        out[17:25] = raw[15:23]
        out[25:32] = raw[23:30]

        # per-channel running-max normalization, 32 channels, 4 per iteration
        for i in range(32):
            v = out[i]
            if self.norm[i] < v:
                self.norm[i] = v
            m = self.norm[i]
            if i < 27 or m >= C_NORM_FLOOR:
                out[i] = v / m

        out[41] = out[26]

        # tongue interplay
        if out[31] > C_PRESENCE:               # TongueRoll
            out[26] = out[26] * C_TSTEP1_MULT
            out[27] = out[28] = out[29] = out[30] = 0.0
        # Tongue directions are mutually exclusive on each axis.
        if out[30] <= out[29]:
            out[30] = 0.0                       # TongueDown yields to Up
        else:
            out[29] = 0.0
        if out[27] <= out[28]:
            out[27] = 0.0                       # TongueLeft yields to Right
        else:
            out[28] = 0.0

        # TongueLongStep2 derivation (out[32]) + Step1 normalization
        s1 = out[26]
        if out[30] <= C_TSTEP1_GATE:            # TongueDown low
            if (C_DELTA < out[29] and out[3] < C_JAW_MORPH and C_DELTA < s1):
                out[32] = (out[29] + C_MORPH_ADD) + out[27] * C_DELTA + out[28] * C_DELTA
                out[26] = 1.0
            elif ((C_DELTA < out[27] and out[3] < C_JAW_MORPH and C_TONGUEUP_MIN < s1)
                  or (C_DELTA < out[28] and out[3] < C_JAW_MORPH and C_TONGUEUP_MIN < s1)):
                out[32] = out[29] * C_DELTA + (max(out[27], out[28] if out[28] > C_DELTA else out[27]) + C_MORPH_ADD) + out[30] * C_DELTA
                out[26] = 1.0
            elif s1 >= C_TSTEP1_NUM:
                out[32] = (s1 - C_TSTEP1_NUM) / C_TSTEP1_DEN
                out[26] = 1.0
            else:
                out[32] = 0.0
                out[26] = s1 / C_TSTEP1_NUM
        else:                                    # TongueDown high
            out[32] = (out[30] + C_DELTA) + out[27] * C_DELTA + out[28] * C_DELTA
            out[26] = 1.0
        if out[3] < C_JAW_TONGUE_OFF:            # jaw closed -> tongue off
            for i in range(26, 33):
                out[i] = 0.0
        if out[32] > C_PRESENCE:
            out[26] = 1.0

        # output EMA over 33 channels
        for i in range(33):
            v = out[i]
            acc = SM_EMA_INC + self.sm_acc[i]
            rate = acc / (acc + SM_EMA_DIV)
            new = (v - self.sm_state[i]) * rate + self.sm_state[i]
            self.sm_acc[i] = (1.0 - rate) * acc
            self.sm_state[i] = new
            self.sm_prev[i] = v
            out[i] = new

        if out[32] > C_TSTEP2_GATE:
            out[26] = 1.0

        # tongue morphs = lateral x vertical products
        out[33] = out[28] * out[29]
        out[34] = out[27] * out[29]
        out[35] = out[28] * out[30]
        out[36] = out[27] * out[30]
        return out
