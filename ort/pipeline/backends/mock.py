"""Deterministic mock backends for protocol tests; they load no models."""

from __future__ import annotations

import numpy as np

from ..base import TTSResult, Transcription, VoiceReference

# Fixed mapping so protocol tests can make exact assertions.
_FIXED_ZH = "你好世界"
_TABLE = {"你好世界": "hello world"}


class MockASR:
    def transcribe(
        self, audio16k: np.ndarray, language: str | None = None
    ) -> Transcription:
        # Any non-empty audio returns the fixed sentence; empty audio returns empty text.
        text = _FIXED_ZH if audio16k.size else ""
        return Transcription(text=text, language=language or "zh")


class MockTranslator:
    def translate(self, text: str, context=(), source=None, target=None) -> str:  # noqa: ANN001
        return _TABLE.get(text.strip(), f"[mock-en] {text.strip()}")


class MockTTS:
    sample_rate = 24_000

    def synthesize(  # noqa: ANN001
        self,
        text: str,
        language=None,
        voice_reference: VoiceReference | None = None,
    ) -> TTSResult:
        # Duration follows text length; output is a recognizable non-silent sine wave.
        seconds = max(0.2, min(2.0, 0.06 * max(1, len(text))))
        n = int(self.sample_rate * seconds)
        t = np.arange(n, dtype=np.float32) / self.sample_rate
        wave = 0.2 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
        return TTSResult(samples=wave, sample_rate=self.sample_rate)
