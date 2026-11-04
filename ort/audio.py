"""Audio utilities for wire-level 24 kHz PCM16 base64 and internal 16 kHz float32."""

from __future__ import annotations

import base64

import numpy as np
import soxr

# OpenAI wire format: 24 kHz, PCM16, mono, base64 of raw samples.
WIRE_SAMPLE_RATE = 24_000
# Internal ASR / VAD sample rate; faster-whisper expects 16 kHz float32.
INTERNAL_SAMPLE_RATE = 16_000


def b64_pcm16_to_float32(b64: str) -> np.ndarray:
    """Convert wire-level base64(PCM16 @24k) to 24 kHz float32 mono in [-1, 1]."""
    raw = base64.b64decode(b64)
    if len(raw) % 2:  # Tolerate odd byte counts by truncating the tail.
        raw = raw[:-1]
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    return pcm / 32768.0


def float32_to_pcm16(samples: np.ndarray) -> np.ndarray:
    """The existing protocol's single float-to-PCM16 quantization boundary."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2")


def float32_to_b64_pcm16(samples: np.ndarray) -> str:
    """Convert float32 mono [-1, 1] to wire-level base64(PCM16). Caller ensures 24 kHz."""
    return base64.b64encode(float32_to_pcm16(samples).tobytes()).decode("ascii")


class WireResampler:
    """One stateful model-native -> 24 kHz conversion per utterance."""

    def __init__(self):
        self.rate = None
        self._converter = None
        self._finished = False

    def add(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if self._finished:
            raise ValueError("PCM resampler is already finalized")
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all() or np.any(np.abs(samples) > 1.0001):
            raise ValueError("TTS PCM must be finite normalized mono audio")
        if self.rate is None:
            if not 8000 <= sample_rate <= 192000:
                raise ValueError("Unsupported TTS sample rate")
            self.rate = sample_rate
            self._converter = soxr.ResampleStream(sample_rate, WIRE_SAMPLE_RATE, 1, dtype="float32")
        elif self.rate != sample_rate:
            raise ValueError("TTS stream changed sample rate mid-utterance")
        return self._converter.resample_chunk(samples, last=False)

    def finish(self) -> np.ndarray:
        if self._finished:
            raise ValueError("PCM resampler was finalized twice")
        self._finished = True
        if self._converter is None:
            return np.zeros(0, dtype=np.float32)
        return self._converter.resample_chunk(np.zeros(0, dtype=np.float32), last=True)


def resample(samples: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    return soxr.resample(samples.astype(np.float32), sr_in, sr_out).astype(np.float32)


def time_stretch(samples: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Pitch-preserving time-scale modification via WSOLA.

    ``speed > 1`` makes playback faster (shorter) without shifting pitch;
    ``speed <= 1`` returns the input unchanged. Used so translated speech,
    which runs ~1.5x longer than the source, tracks the speaking pace
    instead of drifting behind realtime.
    """
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    win_len = max(256, int(round(0.040 * sr)))  # ~40 ms analysis window
    if speed <= 1.0 + 1e-3 or x.size < 2 * win_len:
        return x

    syn_hop = win_len // 2  # 50% overlap; Hann OLA sums to ~unity
    ana_hop = int(round(syn_hop * speed))
    tol = syn_hop // 2  # similarity search radius (samples)
    overlap = win_len - syn_hop
    win = np.hanning(win_len).astype(np.float32)

    out_len = int(np.ceil(x.size / speed)) + win_len
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)

    out[:win_len] += x[:win_len] * win
    norm[:win_len] += win
    # The samples that would naturally follow the grain we just placed.
    natural = x[syn_hop : syn_hop + overlap].copy()

    centre = ana_hop
    frame = 1
    while True:
        sp = frame * syn_hop
        lo = max(0, centre - tol)
        hi = min(x.size - win_len, centre + tol)
        if hi <= lo or sp + win_len > out_len:
            break
        # Pick the grain whose head best matches the natural continuation
        # (normalized cross-correlation) to avoid pitch-period phase jumps.
        cands = np.lib.stride_tricks.sliding_window_view(
            x[lo : hi + overlap], overlap
        )
        scores = cands @ natural
        energy = np.sqrt(np.einsum("ij,ij->i", cands, cands)) + 1e-9
        best = lo + int(np.argmax(scores / energy))

        out[sp : sp + win_len] += x[best : best + win_len] * win
        norm[sp : sp + win_len] += win
        natural = x[best + syn_hop : best + syn_hop + overlap].copy()
        if natural.size < overlap:
            break
        centre = best + ana_hop
        frame += 1

    norm[norm < 1e-6] = 1.0
    y = out / norm
    return y[: int(round(x.size / speed))].astype(np.float32)


def to_internal(samples_24k: np.ndarray) -> np.ndarray:
    """Convert wire-level 24 kHz audio to internal 16 kHz audio."""
    return resample(samples_24k, WIRE_SAMPLE_RATE, INTERNAL_SAMPLE_RATE)


def to_wire(samples: np.ndarray, sr_in: int) -> str:
    """Convert float32 audio at any sample rate to wire-level base64(PCM16 @24k)."""
    return float32_to_b64_pcm16(resample(samples, sr_in, WIRE_SAMPLE_RATE))


def ms_to_samples(ms: int, sr: int) -> int:
    return int(round(ms * sr / 1000.0))
