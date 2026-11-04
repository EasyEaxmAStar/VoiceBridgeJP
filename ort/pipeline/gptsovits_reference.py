"""Validate an explicit, immutable static reference before starting GPU inference."""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class StaticReference:
    path: str
    text: str
    language: str
    fingerprint: str
    duration: float
    rms: float


def validate_reference(audio: str, text: str, language: str) -> StaticReference:
    if not audio or not text.strip() or "\x00" in text:
        raise ValueError("GPT-SoVITS requires a static reference WAV and its exact transcript")
    if language not in {"ja", "en", "zh"}:
        raise ValueError("Reference language must be ja, en or zh; Japanese is recommended")
    path = Path(audio).expanduser().resolve()
    try:
        if path.stat().st_size > 32 * 2**20:
            raise ValueError("Reference WAV is too large (maximum 32 MiB)")
        raw = path.read_bytes()
        info = sf.info(io.BytesIO(raw))
        if info.format not in {"WAV", "WAVEX", "RF64"}:
            raise ValueError("Reference must be a WAV container")
        if not 3 <= info.duration <= 10 or info.channels not in {1, 2}:
            raise ValueError("Reference WAV must be 3–10 seconds, mono or stereo")
        if not 16000 <= info.samplerate <= 48000:
            raise ValueError("Reference WAV sample rate must be 16–48 kHz")
        samples, _ = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except (OSError, sf.LibsndfileError) as exc:
        raise ValueError(f"Cannot read/decode static reference WAV: {path}") from exc
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1.00001:
        raise ValueError("Reference WAV contains non-finite or out-of-range samples")
    near_peak = np.abs(samples) >= 0.999
    if np.mean(near_peak) >= 0.001 or np.any(near_peak[2:] & near_peak[1:-1] & near_peak[:-2]):
        raise ValueError("Reference WAV is clipped; choose a clean recording")
    mono = samples.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    if rms < 1e-4:
        raise ValueError("Reference WAV is silent or too quiet")
    fingerprint = hashlib.sha256(raw + json.dumps([text, language], ensure_ascii=False).encode("utf8")).hexdigest()
    return StaticReference(str(path), text, language, fingerprint, info.duration, rms)
