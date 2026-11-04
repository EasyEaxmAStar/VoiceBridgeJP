"""Rolling source-speaker reference buffer for voice-clone TTS."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from .base import VoiceReference


@dataclass
class _Chunk:
    samples: np.ndarray
    text: str


class VoiceReferenceBuffer:
    """Accumulates recent ASR input audio as a bounded TTS reference."""

    def __init__(
        self,
        sample_rate: int,
        min_seconds: float = 3.0,
        max_seconds: float = 10.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if min_seconds <= 0 or max_seconds <= 0:
            raise ValueError("reference durations must be positive")
        if min_seconds > max_seconds:
            raise ValueError("min_seconds cannot exceed max_seconds")

        self.sample_rate = int(sample_rate)
        self.min_samples = int(round(min_seconds * self.sample_rate))
        self.max_samples = int(round(max_seconds * self.sample_rate))
        self._chunks: list[_Chunk] = []
        self._total_samples = 0
        self._version = 0
        self._reference_id = uuid4().hex

    @property
    def source_duration_s(self) -> float:
        return self._total_samples / self.sample_rate

    def append(self, samples: np.ndarray, text: str) -> None:
        clean_text = " ".join((text or "").split())
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0 or not clean_text:
            return

        if audio.size > self.max_samples:
            audio = audio[-self.max_samples :]
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        self._chunks.append(_Chunk(samples=audio, text=clean_text))
        self._total_samples += audio.size
        self._trim()
        self._version += 1

    def snapshot(self) -> VoiceReference | None:
        if self._total_samples == 0:
            return None

        samples = np.concatenate([c.samples for c in self._chunks])
        source_samples = int(samples.size)

        text = " ".join(c.text for c in self._chunks if c.text).strip()
        return VoiceReference(
            samples=np.ascontiguousarray(samples, dtype=np.float32),
            sample_rate=self.sample_rate,
            text=text,
            version=self._version,
            source_samples=source_samples,
            reference_id=self._reference_id,
        )

    def _trim(self) -> None:
        overflow = self._total_samples - self.max_samples
        while overflow > 0 and self._chunks:
            first = self._chunks[0]
            if overflow >= first.samples.size:
                overflow -= first.samples.size
                self._total_samples -= first.samples.size
                self._chunks.pop(0)
                continue

            first.samples = np.ascontiguousarray(first.samples[overflow:])
            self._total_samples -= overflow
            overflow = 0
