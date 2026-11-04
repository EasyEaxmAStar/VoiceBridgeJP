"""Validated local WAV conversion for the upstream mono 16 kHz ASR interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import INTERNAL_SAMPLE_RATE, resample


@dataclass(frozen=True)
class WavAudio:
    samples: np.ndarray
    source_sample_rate: int
    source_channels: int
    duration_seconds: float


def load_asr_wav(path: Path | str, *, max_seconds: float = 1200, strict: bool = False) -> WavAudio:
    """Read PCM/float WAV, downmix channels, and resample to float32 at 16 kHz."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"WAV file does not exist: {path}")
    if strict and path.stat().st_size > 512 * 2**20:
        raise ValueError("Input WAV exceeds the 512 MiB decoded-workflow file limit")
    try:
        info = sf.info(path)
        if info.format not in {"WAV", "WAVEX", "RF64"}:
            raise ValueError("Input must be a WAV file; convert other formats first")
        if info.frames <= 0 or not 1 <= info.channels <= 8:
            raise ValueError("WAV must contain audio in 1–8 channels")
        if not 8000 <= info.samplerate <= 192000:
            raise ValueError("WAV sample rate must be between 8000 and 192000 Hz")
        if info.duration > max_seconds:
            raise ValueError(f"WAV exceeds the {max_seconds:g} second input limit")
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
    except sf.LibsndfileError as exc:
        raise ValueError(f"Cannot decode WAV {path}: {exc}") from exc
    if not np.isfinite(samples).all():
        raise ValueError("WAV contains NaN or infinite samples")
    if np.max(np.abs(samples)) > 1.0001:
        raise ValueError("WAV samples exceed [-1, 1]; normalize the recording first")
    if strict:
        near_peak = np.abs(samples) >= .999
        if np.mean(near_peak) >= .001 or np.any(near_peak[2:] & near_peak[1:-1] & near_peak[:-2]):
            raise ValueError("Input WAV is clipped; provide an unclipped recording")
    mono = samples.mean(axis=1, dtype=np.float32)
    if not np.any(mono):
        raise ValueError("WAV contains only silence (or channels cancel when downmixed)")
    if strict and np.sqrt(np.mean(mono.astype(np.float64)**2)) < 1e-4:
        raise ValueError("Input WAV is silent or too quiet")
    converted = np.ascontiguousarray(
        np.clip(resample(mono, rate, INTERNAL_SAMPLE_RATE), -1, 1), dtype=np.float32
    )
    return WavAudio(converted, rate, info.channels, info.duration)
