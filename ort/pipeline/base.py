"""Pipeline interfaces, data types, and backend factory."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Protocol, TypeVar, runtime_checkable

import numpy as np

from ..config import Config

log = logging.getLogger("ort.pipeline")

_B = TypeVar("_B")


def _timed(kind: str, name: str, factory: Callable[[], _B]) -> _B:
    t = time.perf_counter()
    obj = factory()
    log.info(
        "[timing] load %s=%s cold_start=%.3fs", kind, name, time.perf_counter() - t
    )
    return obj


@dataclass
class Segment:
    """An immutable committed audio span in internal 16 kHz float32 mono."""

    samples: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class Transcription:
    text: str
    language: str | None = None


@dataclass
class TTSResult:
    samples: np.ndarray  # float32 mono [-1, 1]
    sample_rate: int


@dataclass(frozen=True)
class VoiceReference:
    """Session-local source-speaker reference for voice-clone TTS."""

    samples: np.ndarray
    sample_rate: int
    text: str
    version: int
    source_samples: int
    reference_id: str

    @property
    def duration_s(self) -> float:
        return self.samples.size / max(1, self.sample_rate)

    @property
    def source_duration_s(self) -> float:
        return self.source_samples / max(1, self.sample_rate)


def normalize_reference_loudness(
    samples: np.ndarray,
    target_rms: float = 0.1,
    max_gain: float = 40.0,
) -> np.ndarray:
    """Scale speech to a stable RMS for voice-clone TTS.

    Qwen3-TTS ICL inherits the reference loudness, so a quiet microphone
    otherwise yields near-silent, degenerate ("嗚嗚啊啊") output. Gain is
    capped so a silent/noise-only reference is not amplified into a loud
    floor, and the result is peak-limited to avoid clipping.
    """
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return samples
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if rms < 1e-5:  # silence: no usable signal to normalize
        return samples
    out = samples * min(target_rms / rms, max_gain)
    peak = float(np.max(np.abs(out)))
    if peak > 0.97:
        out = out * (0.97 / peak)
    return np.ascontiguousarray(out, dtype=np.float32)


def padded_voice_reference_samples(
    voice_reference: VoiceReference,
    sample_rate: int,
    min_seconds: float = 3.0,
) -> np.ndarray:
    """Return a TTS-ready copy, normalizing loudness and padding at handoff."""

    samples = voice_reference.samples
    if voice_reference.sample_rate != sample_rate:
        from ..audio import resample

        samples = resample(samples, voice_reference.sample_rate, sample_rate)
    else:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)

    # Normalize before zero-padding so trailing silence does not dilute RMS.
    samples = normalize_reference_loudness(samples)

    min_samples = int(round(min_seconds * sample_rate))
    if samples.size < min_samples:
        samples = np.concatenate(
            [samples, np.zeros(min_samples - samples.size, dtype=np.float32)]
        )
    return np.ascontiguousarray(samples, dtype=np.float32)


@runtime_checkable
class ASRBackend(Protocol):
    def transcribe(
        self, audio16k: np.ndarray, language: str | None = None
    ) -> Transcription:
        """Transcribe 16 kHz float32 speech and return detected source language."""
        ...


@runtime_checkable
class TranslatorBackend(Protocol):
    def translate(
        self,
        text: str,
        context: "Sequence[tuple[str, str]]" = (),
        source: str | None = None,
        target: str | None = None,
    ) -> str:
        """Translate source-language text into target-language text.

        context contains committed (source, target) spans in temporal order, so
        incremental translation can remain coherent across spans. source/target
        are ISO 639-1 codes; source must be explicit, while target None means
        backend default.
        """
        ...


@runtime_checkable
class TTSBackend(Protocol):
    def synthesize(
        self,
        text: str,
        language: str | None = None,
        voice_reference: VoiceReference | None = None,
    ) -> TTSResult:
        """Synthesize target-language text; voice_reference is for voice-clone TTS."""
        ...


@dataclass
class Pipeline:
    asr: ASRBackend
    translator: TranslatorBackend
    tts: TTSBackend
    config: Config

    def warmup(self) -> None:
        """Run a small pass to trigger model loading before the first connection."""
        for obj in (self.asr, self.translator, self.tts):
            warm = getattr(obj, "warmup", None)
            if callable(warm):
                warm()


def build_pipeline(config: Config) -> Pipeline:
    """Select and construct backends from config; mock backends load no models."""
    from .backends import mock

    t_all = time.perf_counter()

    if config.asr_backend == "mock":
        asr: ASRBackend = mock.MockASR()
    elif config.asr_backend == "qwen3":
        if config.runtime == "mlx":
            from .backends.qwen3_asr import Qwen3ASR

            asr = _timed("asr", "qwen3:mlx", lambda: Qwen3ASR(config))
        else:
            from .backends.qwen3_asr_torch import Qwen3ASRTorch

            asr = _timed(
                "asr", f"qwen3:{config.runtime}", lambda: Qwen3ASRTorch(config)
            )
    elif config.asr_backend == "whisper":
        from .backends.whisper_asr import WhisperASR

        asr = _timed(
            "asr", f"whisper:{config.whisper_model}", lambda: WhisperASR(config)
        )
    else:
        raise ValueError(f"未知 ASR 後端: {config.asr_backend}")

    if config.translator_backend == "mock":
        translator: TranslatorBackend = mock.MockTranslator()
    elif config.translator_backend == "translategemma":
        if config.runtime == "mlx":
            from .backends.translategemma_mlx import TranslateGemmaMLX

            translator = _timed(
                "mt", "translategemma:mlx", lambda: TranslateGemmaMLX(config)
            )
        else:
            from .backends.translategemma_torch import TranslateGemmaTorch

            translator = _timed(
                "mt",
                f"translategemma:{config.runtime}",
                lambda: TranslateGemmaTorch(config),
            )
    elif config.translator_backend == "opusmt":
        from .backends.opus_mt import OpusMTTranslator

        translator = _timed(
            "mt", "opusmt:cpu", lambda: OpusMTTranslator(config)
        )
    else:
        raise ValueError(f"未知 Translator 後端: {config.translator_backend}")

    if config.tts_backend == "mock":
        tts: TTSBackend = mock.MockTTS()
    elif config.tts_backend == "gptsovits":
        if config.gptsovits_voice_preset:
            from .backends.personal_presets import PersonalPresetTTS
            tts=_timed("tts","gptsovits:personal-preset",lambda:PersonalPresetTTS(config))
        else:
            from .backends.gptsovits import GPTSoVITS
            tts = _timed("tts", "gptsovits:v2ProPlus", lambda: GPTSoVITS(config))
    elif config.tts_backend == "kokoro":
        from .backends.kokoro_tts import KokoroTTS

        tts = _timed("tts", "kokoro", lambda: KokoroTTS(config))
    elif config.tts_backend == "supertonic":
        from .backends.supertonic_tts import SupertonicTTS

        tts = _timed("tts", "supertonic", lambda: SupertonicTTS(config))
    elif config.tts_backend == "qwen3_tts":
        if config.runtime == "mlx":
            from .backends.qwen3_tts import Qwen3TTS

            tts = _timed("tts", "qwen3_tts:mlx", lambda: Qwen3TTS(config))
        else:
            from .backends.qwen3_tts_torch import Qwen3TTSTorch

            tts = _timed(
                "tts", f"qwen3_tts:{config.runtime}", lambda: Qwen3TTSTorch(config)
            )
    elif config.tts_backend == "say":
        from .backends.say_tts import SayTTS

        tts = _timed("tts", "say", lambda: SayTTS(config))
    else:
        raise ValueError(f"未知 TTS 後端: {config.tts_backend}")

    log.info(
        "[timing] pipeline built asr=%s mt=%s tts=%s total_cold_start=%.3fs",
        config.asr_backend,
        config.translator_backend,
        config.tts_backend,
        time.perf_counter() - t_all,
    )

    return Pipeline(asr=asr, translator=translator, tts=tts, config=config)
