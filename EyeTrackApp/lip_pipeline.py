#!/usr/bin/env python3
"""Mouth-model preprocessing, inference, and legacy post-processing.

The compatible model accepts two 400 x 400 byte planes derived from a
400 x 400 YUYV frame and produces 30 non-negative values. Frames are median
filtered, resized to 100 x 100 with cubic interpolation, converted to float32,
and scaled to the 0..1 range before inference.
"""
import numpy as np
import cv2
import onnxruntime as ort

MODEL_INPUT = 100

PAIR_THRESHOLD_DEFAULT = 0.5
PAIR_MAX = 1.0
TONGUE_TH = 0.05
TONGUE_TH2 = 0.2
TONGUE_SUP = 0.7
OUT5_SCALE = 0.5
PRESENCE_GATE = 0.4
HOLD_FRAMES = 25
GAIN_FRAMES = 150
GAIN_HOLD = 12
GAIN_DELTA = 0.2
GAIN_DELTA_ALT = 0.4
GAIN_INIT = 0.7
GAIN_DEFAULT = 1.0
SMOOTH_ADAPT = 0.55
OUT_COUNT = 27


class LipPostprocess:
    """Stateful legacy model-output normalization."""

    def __init__(self):
        self.th = [0.5] * 4
        self.reset()

    def reset(self):
        self.tracking_hold = 0
        self.lost_hold = 0
        self.was_lost = False
        self.gain_hold = 0
        self.gain = np.full(26, GAIN_DEFAULT, np.float32)
        self.gain[0] = GAIN_INIT
        self.running_max = np.zeros(26, np.float32)
        self.smooth_state = np.zeros(26, np.float32)
        self.smooth_acc = np.zeros(26, np.float32)
        self.prev_raw = np.zeros(26, np.float32)

    def __call__(self, raw):
        """raw: 30 post-ReLU model floats. Returns 27 output floats [0..1] + presence."""
        out = np.zeros(OUT_COUNT, np.float32)
        presence = float(raw[0])
        tracked = presence >= PRESENCE_GATE

        if tracked:
            if not self.was_lost:
                self.tracking_hold += 1
            self.was_lost = False
            self.lost_hold = 0

            if self.tracking_hold <= HOLD_FRAMES:
                return out

            if self.tracking_hold > GAIN_FRAMES:
                self.gain_hold += 1
                if self.gain_hold > GAIN_HOLD:
                    self._auto_gain(raw)

            r = raw[1:25]
            # complementary pairs, thresholds normalized to [0,1]
            t = self.th[0]
            if t <= r[0]:
                out[1] = (r[0] - t) / (PAIR_MAX - t)
                out[2] = 0.0
            else:
                out[2] = (t - r[0]) / t
                out[1] = 0.0
            th138 = self.th[3]
            out[0] = max(0.0, r[1] - th138)
            out[3] = r[2]
            out[4] = r[3]

            t = self.th[1]
            if t <= r[4]:
                out[24] = (r[4] - t) / (PAIR_MAX - t)
                out[25] = 0.0
            else:
                out[25] = (t - r[4]) / t
                out[24] = 0.0

            t = self.th[2]
            if t <= r[5]:
                out[7] = (r[5] - t) / (PAIR_MAX - t)
                out[8] = 0.0
            else:
                out[8] = (t - r[5]) / t
                out[7] = 0.0

            out[5] = (r[6] + r[7]) * OUT5_SCALE
            out[6] = r[8]
            out[9] = r[9]
            out[10] = r[10]
            out[11] = r[11]
            out[15] = r[21]
            out[22] = r[16]
            out[23] = r[17]
            out[20] = r[18]
            out[21] = r[19]
            out[16] = r[20]
            out[12] = 0.0
            out[17] = r[22]
            out[26] = presence

            # tongue suppression
            if r[9] > TONGUE_TH:
                out[22] -= r[9] * TONGUE_SUP
                out[23] -= r[9] * TONGUE_SUP
            if r[10] > TONGUE_TH2:
                out[20] -= r[10] * TONGUE_SUP
                out[21] -= r[10] * TONGUE_SUP

            # per-channel running-max normalization (13 pairs -> 26 channels)
            for i in range(26):
                v = out[i] if i < OUT_COUNT else 0.0
                if self.running_max[i] < v:
                    self.running_max[i] = v
                m = self.running_max[i] if self.running_max[i] > 0 else 1.0
                out[i] = out[i] / m

            # adaptive exponential smoothing (13 pairs)
            for i in range(0, 26, 2):
                out[i] = self._smooth(i, out[i])
                out[i + 1] = self._smooth(i + 1, out[i + 1])
            return out

        # lost
        if self.was_lost:
            self.lost_hold += 1
        self.was_lost = True
        self.tracking_hold = 0
        if self.lost_hold > HOLD_FRAMES:
            self.th = [PAIR_THRESHOLD_DEFAULT] * 4
            out[26] = presence
            self.reset()
            self.was_lost = True
            self.tracking_hold = 0
        return out

    def _auto_gain(self, raw):
        """Update per-channel gain after sustained tracking."""
        for idx in range(4):
            cur = float(raw[1 + idx])
            prev = self.prev_raw[idx]
            delta = cur - prev
            ok = delta < GAIN_DELTA or (idx == 3 and delta < GAIN_DELTA_ALT and cur < GAIN_DELTA_ALT)
            if ok:
                acc = self.smooth_acc[idx] + delta
                rate = acc / (acc + SMOOTH_ADAPT) if (acc + SMOOTH_ADAPT) > 0 else 0
                new = (cur - self.running_max[idx]) * rate + self.running_max[idx]
                self.gain[idx] = (1.0 - rate) * acc
                self.running_max[idx] = new
                self.prev_raw[idx] = new
        self.gain_hold = 0

    def _smooth(self, i, v):
        """Apply adaptive exponential smoothing to one channel."""
        acc = self.smooth_acc[i] + v
        rate = acc / (acc + 1e-9) if acc > 0 else 0.0
        new = (v - self.smooth_state[i]) * rate + self.smooth_state[i]
        self.smooth_acc[i] = (1.0 - rate) * acc
        self.smooth_state[i] = new
        return new


class MouthModelPipeline:
    """Full pipeline: YUYV 400x400 bytes -> 27 expression floats."""

    def __init__(self, onnx_path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            onnx_path, so, providers=["CPUExecutionProvider"])
        self.post = LipPostprocess()

    def process(self, yuyv: np.ndarray) -> np.ndarray:
        """yuyv: 400x400x2 uint8 (or 320000 bytes). Returns 27 floats."""
        raw = self.infer_raw(yuyv)
        return self.post(raw)
    def infer_raw(self, yuyv: np.ndarray, flip_vertical: bool = False) -> np.ndarray:
        """Return the 30 raw model outputs (pre-postprocess).

        ``flip_vertical`` is a diagnostic override; normal V4L2 row order is
        already the order expected by compatible models."""
        b = yuyv.reshape(400, 800)
        if flip_vertical:
            b = b[::-1]
        frame = cv2.medianBlur(np.ascontiguousarray(b), 5)
        left = frame[:, :400]
        right = frame[:, 400:]
        merged = cv2.merge([left, right])
        resized = cv2.resize(merged, (100, 100), interpolation=cv2.INTER_CUBIC)
        f32 = resized.astype(np.float32) * (1.0 / 255.0)
        x = f32.transpose(2, 0, 1)[None]
        return self.sess.run(None, {"input": x})[0][0]


if __name__ == "__main__":
    import sys
    dev = sys.argv[1] if len(sys.argv) > 1 else "/dev/video2"
    subprocess_probe(dev)
