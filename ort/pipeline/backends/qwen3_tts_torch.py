"""Qwen3-TTS backend for CUDA/CPU through the official qwen-tts package."""

from __future__ import annotations

import tempfile
import threading
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from ...config import Config
from ...languages import DEFAULT_TARGET_LANGUAGE, qwen3tts_lang
from ..base import VoiceReference, padded_voice_reference_samples
from ..static_voice_reference import StaticVoiceReferenceCache, StaticReferenceData

_LOCK = threading.RLock()
log = logging.getLogger("ort.tts")


def _missing_dep(name: str) -> RuntimeError:
    return RuntimeError(
        f"{name} is required for ORT_RUNTIME=cuda/cpu with qwen3_tts. "
        "Use the separate pinned TTS environment documented in README "
        "(requirements-tts-cuda.lock.txt)."
    )


def _official_repo(model: str) -> str:
    m = model.strip()
    key = m.lower()
    mapping = {
        "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "qwen/qwen3-tts-0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "qwen/qwen3-tts-1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "mlx-community/qwen3-tts-12hz-0.6b-base-8bit": (
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        ),
        "mlx-community/qwen3-tts-12hz-1.7b-base-8bit": (
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        ),
    }
    return mapping.get(key, m)


class Qwen3TTSTorch:
    def __init__(self, config: Config):
        started = time.perf_counter()
        self._static_cache = StaticVoiceReferenceCache(
            config.qwen3_tts_ref_audio or "", config.qwen3_tts_ref_text or ""
        )
        # Reject bad static settings before allocating model weights. Never
        # silently choose the microphone voice when a static reference is set.
        if self._static_cache.configured:
            self._static_cache.reference()
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise _missing_dep("torch and qwen-tts") from exc

        self._torch = torch
        self._repo = _official_repo(config.qwen3_tts_model)
        self._default_target = DEFAULT_TARGET_LANGUAGE
        dtype = self._dtype(config.torch_dtype)

        kwargs: dict[str, object] = {"device_map": self._device_map(config.device)}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if config.torch_attn not in {"", "none"}:
            kwargs["attn_implementation"] = config.torch_attn
        model_path = self._repo
        if config.qwen3_tts_offline:
            if not Path(model_path).is_dir():
                from huggingface_hub import snapshot_download

                model_path = snapshot_download(repo_id=self._repo, local_files_only=True)
            kwargs["local_files_only"] = True
        self._model = Qwen3TTSModel.from_pretrained(model_path, **kwargs)
        self.device = str(getattr(self._model, "device", kwargs["device_map"]))
        self.dtype = str(getattr(self._model, "dtype", dtype))
        self._ref_audio = self._static_cache.audio
        self._ref_text = self._static_cache.text
        self._static_voice_prompt = None
        self._voice_prompt_cache: tuple[str, int, object] | None = None
        self.synthesis_count = 0
        self.last_synthesis_seconds: float | None = None
        self.last_reference_source: str | None = None
        self._prompt_supported = callable(getattr(self._model, "create_voice_clone_prompt", None))
        with _LOCK, self._torch.inference_mode():
            if self._static_cache.configured:
                self._static_conditioning(count_hit=False)
        self._synchronize()
        self.initialization_seconds = time.perf_counter() - started
        log.info("tts_initialized model=%s seconds=%.3f static=%s conditioning_cache_supported=%s peak=%s",
                 self._repo, self.initialization_seconds, self._static_cache.configured,
                 self._prompt_supported, self._peak_memory())

    def _synchronize(self) -> None:
        cuda = getattr(self._torch, "cuda", None)
        if self.device.startswith("cuda") and cuda is not None and cuda.is_available():
            cuda.synchronize(self.device)

    def _peak_memory(self) -> dict:
        cuda = getattr(self._torch, "cuda", None)
        if self.device.startswith("cuda") and cuda is not None and cuda.is_available():
            return {"peak_allocated_mib": cuda.max_memory_allocated(self.device) / 2**20,
                    "peak_reserved_mib": cuda.max_memory_reserved(self.device) / 2**20}
        return {"peak_allocated_mib": None, "peak_reserved_mib": None}

    def _model_identity(self) -> str:
        model = getattr(self._model, "model", None)
        config = getattr(model, "config", None)
        return repr((self._repo, id(self._model), id(model), getattr(config, "_name_or_path", None),
                     getattr(config, "_commit_hash", None)))

    def _build_static_prompt(self, reference: StaticReferenceData):
        self._synchronize()
        prompt = self._model.create_voice_clone_prompt(
            ref_audio=(reference.samples.copy(), reference.sample_rate),
            ref_text=reference.text, x_vector_only_mode=False,
        )
        self._synchronize()
        return prompt

    def _static_conditioning(self, *, count_hit: bool = True):
        self._prompt_supported = callable(getattr(self._model, "create_voice_clone_prompt", None))
        reference, prompt = self._static_cache.ensure(
            self._model_identity(), self._build_static_prompt if self._prompt_supported else None,
            count_hit=count_hit,
        )
        self._static_voice_prompt = prompt
        return reference, prompt

    def set_static_reference(self, audio: str, exact_transcript: str) -> None:
        """Reconfigure/rebuild conditioning. Both empty explicitly enables rolling fallback."""
        with _LOCK, self._torch.inference_mode():
            self._static_cache.configure(audio, exact_transcript)
            self._ref_audio, self._ref_text = self._static_cache.audio, self._static_cache.text
            self._static_voice_prompt = None
            if self._static_cache.configured:
                self._static_conditioning(count_hit=False)

    def clear_static_reference(self) -> None:
        self.set_static_reference("", "")

    def metrics(self) -> dict:
        return {"initialization_seconds": self.initialization_seconds,
                "synthesis_count": self.synthesis_count,
                "last_synthesis_seconds": self.last_synthesis_seconds,
                "last_reference_source": self.last_reference_source,
                "static_configured": self._static_cache.configured,
                "conditioning_cache_supported": self._prompt_supported,
                "prompt_cache_hits": self._static_cache.hits,
                "prompt_cache_misses": self._static_cache.misses,
                "prompt_build_seconds": self._static_cache.build_seconds,
                **self._peak_memory()}

    def _device_map(self, device: str) -> str:
        if device == "cuda":
            return "cuda:0"
        return device

    def _dtype(self, name: str):  # noqa: ANN001
        if name in {"", "none"}:
            return None
        if not hasattr(self._torch, name):
            raise ValueError(f"Unsupported ORT_TORCH_DTYPE={name!r}")
        return getattr(self._torch, name)

    def warmup(self) -> None:
        if self._static_cache.configured:
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
            return TTSResult(np.zeros(0, dtype=np.float32), 24_000)

        lang = qwen3tts_lang(language or self._default_target).title()
        with _LOCK, self._torch.inference_mode():
            self._synchronize()
            started = time.perf_counter()
            if self._static_cache.configured:
                self._static_voice_prompt = None
                reference, prompt = self._static_conditioning()
                self.last_reference_source = "static"
                conditioning = {"voice_clone_prompt": prompt} if prompt is not None else {
                    "ref_audio": (reference.samples.copy(), reference.sample_rate),
                    "ref_text": reference.text, "x_vector_only_mode": False,
                }
                if prompt is None:
                    log.warning("TTS API has no conditioning-cache support; using validated static reference")
            else:
                conditioning = {"voice_clone_prompt": self._voice_prompt(voice_reference)}
                self.last_reference_source = "rolling"
            wavs, sr = self._model.generate_voice_clone(
                text=text, language=lang, **conditioning,
            )
            self._synchronize()
            self.last_synthesis_seconds = time.perf_counter() - started
            if not wavs or int(sr) <= 0:
                raise RuntimeError("TTS returned no valid audio")
            samples = np.asarray(wavs[0], dtype=np.float32)
            if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
                raise RuntimeError("TTS returned empty, non-mono, or non-finite audio")
            self.synthesis_count += 1
            log.info("tts_synthesized seconds=%.3f reference=%s cache_hits=%d cache_misses=%d peak=%s",
                     self.last_synthesis_seconds, self.last_reference_source,
                     self._static_cache.hits, self._static_cache.misses, self._peak_memory())
        return TTSResult(samples=samples, sample_rate=int(sr))

    def _voice_prompt(self, voice_reference: VoiceReference | None):  # noqa: ANN202
        if voice_reference is None:
            raise ValueError("No voice reference: set the static reference WAV and exact transcript or supply a rolling reference")

        if (
            self._voice_prompt_cache is not None
            and self._voice_prompt_cache[0] == voice_reference.reference_id
            and self._voice_prompt_cache[1] == voice_reference.version
        ):
            return self._voice_prompt_cache[2]

        with tempfile.TemporaryDirectory() as d:
            ref_audio = self._write_reference(voice_reference, d)
            with _LOCK:
                prompt = self._model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=voice_reference.text,
                    x_vector_only_mode=False,
                )
        self._voice_prompt_cache = (
            voice_reference.reference_id,
            voice_reference.version,
            prompt,
        )
        return prompt

    @staticmethod
    def _write_reference(voice_reference: VoiceReference, directory: str) -> str:
        sample_rate = 24_000
        samples = padded_voice_reference_samples(voice_reference, sample_rate)
        wav = f"{directory}/voice_reference.wav"
        sf.write(wav, samples, sample_rate, subtype="PCM_16")
        return wav
