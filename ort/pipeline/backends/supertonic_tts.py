"""Candidate TTS: Supertonic-3, an ONNX/CPU backend from the supertonic package.

It has no MLX/Metal thread affinity, so it can in principle run independently
from the single inference thread. Output is about 44.1 kHz and is resampled by
the pipeline to the 24 kHz wire format.
"""

from __future__ import annotations

import threading

import numpy as np

from ...config import Config
from ...languages import DEFAULT_TARGET_LANGUAGE, SUPERTONIC_TARGET, norm

_LOCK = threading.Lock()
_FALLBACK_SR = 44_100


class SupertonicTTS:
    def __init__(self, config: Config):
        from supertonic import TTS

        self._tts = TTS(model="supertonic-3", auto_download=True)
        self._style = self._tts.get_voice_style(voice_name=config.supertonic_voice)
        self._steps = config.supertonic_steps
        self._default_lang = norm(DEFAULT_TARGET_LANGUAGE)

    def warmup(self) -> None:
        self.synthesize("warmup")

    def synthesize(
        self,
        text: str,
        language: str | None = None,
        voice_reference=None,  # noqa: ANN001
    ):
        from ..base import TTSResult

        text = text.strip()
        if not text:
            return TTSResult(np.zeros(0, dtype=np.float32), _FALLBACK_SR)

        lang = norm(language) if language else self._default_lang
        if lang not in SUPERTONIC_TARGET:
            raise ValueError(
                f"Supertonic 不支援目標語 {lang!r}；支援：{sorted(SUPERTONIC_TARGET)}"
            )

        with _LOCK:
            wav, duration = self._tts.synthesize(
                text,
                voice_style=self._style,
                total_steps=self._steps,
                lang=lang,
            )
        samples = np.asarray(wav, dtype=np.float32).reshape(-1)
        # Infer sample rate from samples / duration; the package does not expose it.
        try:
            dur = float(np.asarray(duration).reshape(-1)[0])
            sr = int(round(samples.size / dur)) if dur > 0 else _FALLBACK_SR
        except Exception:  # noqa: BLE001
            raise RuntimeError("Supertonic did not return a valid duration") from None
        if not (8_000 <= sr <= 48_000):
            raise RuntimeError(f"Supertonic returned an invalid sample rate: {sr}")
        return TTSResult(samples=samples, sample_rate=sr)
