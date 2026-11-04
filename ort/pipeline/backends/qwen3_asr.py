"""Default MLX ASR: Qwen3-ASR-1.7B through mlx-audio.

The MLX profile uses mlx-community Qwen3-ASR 8-bit mirrors to keep the model
loading path fully MLX-native.
"""

from __future__ import annotations

import glob
import os
import tempfile
import threading

import numpy as np
import soundfile as sf

from ...audio import INTERNAL_SAMPLE_RATE
from ...config import Config
from ...languages import ASR_FORCE_NAME, source_code
from ..base import Transcription

_LOCK = threading.Lock()


class Qwen3ASR:
    def __init__(self, config: Config):
        from mlx_audio.stt.utils import load_model

        # Official repo ids and shortcuts map to MLX 8-bit mirrors.
        _MAP = {
            "qwen/qwen3-asr-0.6b": "mlx-community/Qwen3-ASR-0.6B-8bit",
            "qwen/qwen3-asr-1.7b": "mlx-community/Qwen3-ASR-1.7B-8bit",
            "0.6b": "mlx-community/Qwen3-ASR-0.6B-8bit",
            "1.7b": "mlx-community/Qwen3-ASR-1.7B-8bit",
        }
        repo = _MAP.get(config.qwen_asr_model.lower(), config.qwen_asr_model)
        self._repo = repo
        self._model = load_model(repo)

    def warmup(self) -> None:
        self.transcribe(np.zeros(INTERNAL_SAMPLE_RATE, dtype=np.float32))

    def transcribe(
        self, audio16k: np.ndarray, language: str | None = None
    ) -> Transcription:
        if audio16k.size == 0:
            return Transcription(text="", language=source_code(language))
        from mlx_audio.stt.generate import generate_transcription

        audio = np.ascontiguousarray(audio16k, dtype=np.float32)
        hint = self._language_hint(language)
        with _LOCK, tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, "seg.wav")
            out = os.path.join(d, "out")
            sf.write(wav, audio, INTERNAL_SAMPLE_RATE, subtype="PCM_16")
            kwargs = {"language": hint} if hint is not None else {}
            result = generate_transcription(
                model=self._model,
                audio=wav,
                output_path=out,
                format="txt",
                verbose=False,
                **kwargs,
            )
            return self._extract_transcription(result, out, d, language)

    def _language_hint(self, language: str | None) -> str | None:
        source = source_code(language)
        if source is None and language is not None:
            source = language.strip().lower()
        if source is None:
            return None
        if source in {"", "auto"}:
            return None
        if source not in ASR_FORCE_NAME:
            raise ValueError(
                f"Qwen3-ASR 不支援來源語提示 {language!r}；"
                f"支援：{sorted(ASR_FORCE_NAME)}"
            )
        return ASR_FORCE_NAME[source]

    @staticmethod
    def _result_language(result) -> str | None:
        """Resolve the detected source language.

        mlx-audio's Qwen3-ASR ``generate()`` returns ``language`` as a
        per-chunk list (one entry per audio chunk); pick the first
        non-empty entry before normalizing.
        """
        lang = getattr(result, "language", None)
        if isinstance(lang, (list, tuple)):
            lang = next((x for x in lang if x), None)
        return source_code(lang)

    @staticmethod
    def _extract_transcription(
        result, out: str, d: str, language_hint: str | None
    ) -> Transcription:
        for attr in ("text", "transcription"):
            val = getattr(result, attr, None)
            if isinstance(val, str) and val:
                return Transcription(
                    text=val.strip(),
                    language=Qwen3ASR._result_language(result)
                    or source_code(language_hint),
                )
        if isinstance(result, str) and result.strip():
            return Transcription(
                text=result.strip(), language=source_code(language_hint)
            )
        # mlx-audio writes transcript files for text output; read the declared paths.
        for path in (f"{out}.txt", out, *glob.glob(os.path.join(d, "*.txt"))):
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    return Transcription(
                        text=f.read().strip(), language=source_code(language_hint)
                    )
        return Transcription(text="", language=source_code(language_hint))
