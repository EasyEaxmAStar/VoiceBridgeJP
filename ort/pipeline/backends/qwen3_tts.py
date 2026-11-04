"""Default TTS: Qwen3-TTS-12Hz-1.7B-Base through mlx-audio.

The Base variants have no built-in speaker. Voice cloning is in-context: each
generation receives (ref_audio, ref_text), and output keeps the reference timbre
across languages. Qwen3-TTS supports 10 explicit languages; unsupported targets
use 'auto'. Output is 24 kHz, matching the wire format.

Session reference audio is accumulated from ASR input audio. An explicitly
configured ORT_QWEN3_TTS_REF_AUDIO takes precedence over that rolling reference.
"""

from __future__ import annotations

import tempfile
import threading

import numpy as np
import soundfile as sf

from ...config import Config
from ...languages import DEFAULT_TARGET_LANGUAGE, qwen3tts_lang
from ..base import VoiceReference, padded_voice_reference_samples
from ..static_voice_reference import StaticVoiceReferenceCache

_LOCK = threading.Lock()


class Qwen3TTS:
    def __init__(self, config: Config):
        self._static_cache = StaticVoiceReferenceCache(
            config.qwen3_tts_ref_audio or "", config.qwen3_tts_ref_text or ""
        )
        if self._static_cache.configured:
            self._static_cache.reference()
        from mlx_audio.tts.utils import load_model

        _MAP = {
            "0.6b": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
            "1.7b": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
            "qwen/qwen3-tts-0.6b": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
            "qwen/qwen3-tts-1.7b": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        }
        repo = _MAP.get(config.qwen3_tts_model.lower(), config.qwen3_tts_model)
        self._repo = repo
        self._model = load_model(repo)
        self.sample_rate = int(self._model.sample_rate)
        self._default_target = DEFAULT_TARGET_LANGUAGE

        ref_audio = (config.qwen3_tts_ref_audio or "").strip()
        ref_text = (config.qwen3_tts_ref_text or "").strip()
        if ref_audio:
            if not ref_text:
                raise ValueError(
                    "設定 ORT_QWEN3_TTS_REF_AUDIO 時必須一併給 ORT_QWEN3_TTS_REF_TEXT"
                    "（參照音訊的逐字稿）"
                )
            self._ref_audio, self._ref_text = ref_audio, ref_text
        else:
            self._ref_audio, self._ref_text = "", ""

    def warmup(self) -> None:
        if self._ref_audio:
            self.synthesize("warm up", self._default_target)

    def synthesize(
        self,
        text: str,
        language: str | None = None,
        voice_reference: VoiceReference | None = None,
    ):
        from ..base import TTSResult

        text = text.strip()
        if not text:
            return TTSResult(np.zeros(0, dtype=np.float32), self.sample_rate)

        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if self._static_cache.configured:
            reference = self._static_cache.reference()
            temp_dir = tempfile.TemporaryDirectory()
            ref_audio = f"{temp_dir.name}/static_reference.wav"
            sf.write(ref_audio, reference.samples, reference.sample_rate, subtype="PCM_16")
            ref_text = reference.text
        elif voice_reference is not None:
            temp_dir = tempfile.TemporaryDirectory()
            ref_audio, ref_text = self._write_reference(voice_reference, temp_dir.name)
        elif self._ref_audio:
            ref_audio, ref_text = self._ref_audio, self._ref_text
        else:
            raise ValueError("Qwen3-TTS requires a session voice reference")

        lang_code = qwen3tts_lang(language or self._default_target)
        chunks: list[np.ndarray] = []
        try:
            with _LOCK:
                for result in self._model.generate(
                    text=text,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    lang_code=lang_code,
                    verbose=False,
                ):
                    audio = getattr(result, "audio", None)
                    if audio is None:
                        continue
                    chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

        samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return TTSResult(samples=samples, sample_rate=self.sample_rate)

    def _write_reference(
        self, voice_reference: VoiceReference, directory: str
    ) -> tuple[str, str]:
        samples = padded_voice_reference_samples(voice_reference, self.sample_rate)
        wav = f"{directory}/voice_reference.wav"
        sf.write(wav, samples, self.sample_rate, subtype="PCM_16")
        return wav, voice_reference.text
