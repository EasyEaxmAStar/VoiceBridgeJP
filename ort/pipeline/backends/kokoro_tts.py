"""Optional fast TTS: Kokoro-82M via the kokoro package.

For multilingual output, target language selects a lang_code and default voice.
Each lang_code gets a lazily created cached KPipeline. Output is 24 kHz, matching
the wire format.
"""

from __future__ import annotations

import os
import threading

import numpy as np

from ...config import Config
from ...languages import DEFAULT_TARGET_LANGUAGE, kokoro_params, norm

_KOKORO_SR = 24_000
_LOCK = threading.Lock()


class KokoroTTS:
    sample_rate = _KOKORO_SR

    def __init__(self, config: Config):
        from kokoro import KPipeline

        self._KPipeline = KPipeline
        self._default_target = norm(DEFAULT_TARGET_LANGUAGE)
        # Explicit environment voice override applies only to the default target language.
        self._voice_override = os.environ.get("ORT_KOKORO_VOICE") or None
        self._pipes: dict[str, object] = {}  # lang_code -> KPipeline

    def _pipe(self, lang_code: str):
        p = self._pipes.get(lang_code)
        if p is None:
            p = self._KPipeline(lang_code=lang_code)
            self._pipes[lang_code] = p
        return p

    def warmup(self) -> None:
        self.synthesize("warm up", self._default_target)

    def synthesize(
        self,
        text: str,
        language: str | None = None,
        voice_reference=None,  # noqa: ANN001
    ):
        from ..base import TTSResult

        text = text.strip()
        if not text:
            return TTSResult(np.zeros(0, dtype=np.float32), _KOKORO_SR)

        tgt = norm(language) if language else self._default_target
        lang_code, default_voice = kokoro_params(tgt)
        voice = (
            self._voice_override
            if (self._voice_override and tgt == self._default_target)
            else default_voice
        )

        chunks: list[np.ndarray] = []
        with _LOCK:
            pipe = self._pipe(lang_code)
            for result in pipe(text, voice=voice):
                audio = getattr(result, "audio", None)
                if audio is None and isinstance(result, (tuple, list)):
                    audio = result[-1]
                if audio is None:
                    continue
                arr = (
                    audio.detach().cpu().numpy()
                    if hasattr(audio, "detach")
                    else np.asarray(audio)
                )
                chunks.append(arr.astype(np.float32).reshape(-1))

        samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return TTSResult(samples=samples, sample_rate=_KOKORO_SR)
