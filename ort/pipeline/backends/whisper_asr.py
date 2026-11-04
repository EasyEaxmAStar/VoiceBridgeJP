"""ASR backend: faster-whisper through CTranslate2 on CPU/CUDA."""

from __future__ import annotations

import threading

import numpy as np

from ...audio import INTERNAL_SAMPLE_RATE
from ...config import Config
from ...languages import source_code
from ..base import Transcription

# CTranslate2 is not thread-safe; serialize inference with a lock.
_LOCK = threading.Lock()


class WhisperASR:
    def __init__(self, config: Config):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            config.whisper_model,
            device=config.whisper_device,
            device_index=config.whisper_device_index,
            compute_type=config.whisper_compute_type,
        )

    def warmup(self) -> None:
        self.transcribe(np.zeros(INTERNAL_SAMPLE_RATE // 2, dtype=np.float32))

    def transcribe(
        self, audio16k: np.ndarray, language: str | None = None
    ) -> Transcription:
        if audio16k.size == 0:
            return Transcription(text="", language=source_code(language))
        audio = np.ascontiguousarray(audio16k, dtype=np.float32)
        # Empty language lets Whisper auto-detect.
        src = source_code(language) or (language or "").strip().lower()
        lang = None if src in ("", "auto") else src
        with _LOCK:
            segments, info = self._model.transcribe(
                audio,
                language=lang,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            text = "".join(seg.text for seg in segments)
        detected = source_code(getattr(info, "language", None)) or source_code(language)
        return Transcription(text=text.strip(), language=detected)
