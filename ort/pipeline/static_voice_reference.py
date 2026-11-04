"""Local static-reference validation and in-memory speaker-conditioning cache."""

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from ..audio import resample
from .base import normalize_reference_loudness


@dataclass(frozen=True)
class StaticReferenceData:
    samples: np.ndarray
    sample_rate: int
    text: str
    fingerprint: str
    source_seconds: float


class StaticVoiceReferenceCache:
    """Hash actual WAV bytes on use; never reuse stale or invalid conditioning.

    Caller serializes access with the model's inference lock. No embedding or
    reference transcript is persisted to disk by this cache.
    """

    def __init__(self, audio: str = "", text: str = ""):
        self.audio = audio.strip()
        self.text = text
        self._data: StaticReferenceData | None = None
        self._prompt_key: tuple[str, str] | None = None
        self.prompt: object | None = None
        self.hits = 0
        self.misses = 0
        self.build_seconds = 0.0

    @property
    def configured(self) -> bool:
        # A partial/invalid setting is still configured and must fail rather
        # than silently select a rolling microphone voice.
        return bool(self.audio or self.text.strip())

    def invalidate(self) -> None:
        self._data = None
        self.prompt = None
        self._prompt_key = None

    def configure(self, audio: str, text: str) -> None:
        settings = (audio.strip(), text)
        if settings != (self.audio, self.text):
            self.audio, self.text = settings
            self.invalidate()

    def reference(self) -> StaticReferenceData:
        if not self.audio or not self.text.strip():
            self.invalidate()
            raise ValueError("Static voice cloning requires both a local reference WAV and its exact transcript")
        if "\x00" in self.text:
            self.invalidate()
            raise ValueError("Reference transcript contains a NUL character")
        path = Path(self.audio).expanduser()
        try:
            raw = path.read_bytes()
        except OSError as exc:
            self.invalidate()
            raise ValueError(f"Cannot read static reference WAV: {path}") from exc
        fingerprint = hashlib.sha256(json.dumps([
            str(path.resolve()), hashlib.sha256(raw).hexdigest(), self.text,
            "mono24k-loudness-v1",
        ], ensure_ascii=False).encode("utf8")).hexdigest()
        if self._data is not None and self._data.fingerprint == fingerprint:
            return self._data
        # Drop the old prompt before validating new bytes. A deleted, corrupted,
        # or replaced file must not result in continued use of the previous voice.
        self.invalidate()
        try:
            info = sf.info(io.BytesIO(raw))
            if info.format not in {"WAV", "WAVEX", "RF64"}:
                raise ValueError("Static reference must be a WAV file")
            if not 1 <= info.channels <= 8 or not 8000 <= info.samplerate <= 192000:
                raise ValueError("Reference WAV must have 1–8 channels at 8–192 kHz")
            if not 3 <= info.duration <= 30:
                raise ValueError("Reference WAV must contain 3–30 seconds of speech; 5–10 seconds is recommended")
            samples, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        except sf.LibsndfileError as exc:
            raise ValueError("Cannot decode the static reference WAV") from exc
        if not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1.0001:
            raise ValueError("Reference WAV must contain finite normalized samples in [-1, 1]")
        mono = samples.mean(axis=1, dtype=np.float32)
        if np.sqrt(np.mean(mono.astype(np.float64) ** 2)) < 1e-5:
            raise ValueError("Reference WAV is silent or too quiet for conditioning")
        mono = np.clip(resample(mono, rate, 24000), -1, 1)
        mono = np.ascontiguousarray(normalize_reference_loudness(mono), dtype=np.float32)
        mono.setflags(write=False)
        self._data = StaticReferenceData(mono, 24000, self.text, fingerprint, info.duration)
        return self._data

    def ensure(
        self, model_identity: str,
        build_prompt: Callable[[StaticReferenceData], object] | None,
        *, count_hit: bool = True,
    ) -> tuple[StaticReferenceData, object | None]:
        reference = self.reference()
        key = (model_identity, reference.fingerprint)
        if build_prompt is None:
            # Legacy APIs may accept raw references without exposing reusable
            # conditioning. Retain validated audio but report no prompt hits.
            self.prompt, self._prompt_key = None, None
            return reference, None
        if self._prompt_key == key and self.prompt is not None:
            self.hits += int(count_hit)
            return reference, self.prompt
        self.prompt, self._prompt_key = None, None
        self.misses += 1
        started = time.perf_counter()
        prompt = build_prompt(reference)
        self.build_seconds += time.perf_counter() - started
        if prompt is None or (isinstance(prompt, (list, tuple)) and not prompt):
            raise RuntimeError("TTS model returned no voice-clone conditioning")
        self.prompt, self._prompt_key = prompt, key
        return reference, prompt
