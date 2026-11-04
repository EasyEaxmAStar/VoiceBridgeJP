"""Qwen3-ASR backend for CUDA/CPU through the official qwen-asr package."""

from __future__ import annotations

import threading
import inspect
from pathlib import Path

import numpy as np

from ...audio import INTERNAL_SAMPLE_RATE
from ...asr_vocabulary import is_internal_context_echo
from ...config import Config
from ...languages import ASR_FORCE_NAME, source_code
from ..base import Transcription

_LOCK = threading.Lock()


def _missing_dep(name: str) -> RuntimeError:
    return RuntimeError(
        f"{name} is required for ORT_RUNTIME=cuda/cpu with qwen3 ASR. "
        "For native Windows CUDA, install requirements-asr-cuda.lock.txt "
        "using the README command. Other non-MLX profiles can use "
        "`uv sync --extra nonmlx`."
    )


def _official_repo(model: str) -> str:
    key = model.strip().lower()
    mapping = {
        "0.6b": "Qwen/Qwen3-ASR-0.6B",
        "1.7b": "Qwen/Qwen3-ASR-1.7B",
        "qwen/qwen3_asr_0.6b": "Qwen/Qwen3-ASR-0.6B",
        "qwen/qwen3_asr_1.7b": "Qwen/Qwen3-ASR-1.7B",
        "qwen/qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
        "qwen/qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
        "mlx_community/qwen3_asr_0.6b_8bit": "Qwen/Qwen3-ASR-0.6B",
        "mlx_community/qwen3_asr_1.7b_8bit": "Qwen/Qwen3-ASR-1.7B",
        "mlx-community/qwen3-asr-0.6b-8bit": "Qwen/Qwen3-ASR-0.6B",
        "mlx-community/qwen3-asr-1.7b-8bit": "Qwen/Qwen3-ASR-1.7B",
    }
    return mapping.get(key, model.strip())


def _device_map(config: Config) -> str:
    raw = (config.qwen_asr_device or "").strip().lower()
    device = config.device if raw in {"", "auto"} else raw
    if device == "cuda":
        return "cuda:0"
    return device


class Qwen3ASRTorch:
    def __init__(self, config: Config):
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise _missing_dep("torch and qwen-asr") from exc

        self._torch = torch
        self._repo = _official_repo(config.qwen_asr_model)
        model_path = self._repo
        if config.qwen_asr_offline:
            if not Path(model_path).is_dir():
                from huggingface_hub import snapshot_download

                model_path = snapshot_download(
                    repo_id=self._repo, local_files_only=True
                )

        dtype = self._dtype(config.torch_dtype)
        kwargs: dict[str, object] = {
            "device_map": _device_map(config),
            "max_inference_batch_size": 1,
            "max_new_tokens": config.qwen_asr_max_new_tokens,
        }
        if dtype is not None:
            kwargs["dtype"] = dtype
        if config.torch_attn not in {"", "none"}:
            kwargs["attn_implementation"] = config.torch_attn
        if config.qwen_asr_offline:
            kwargs["local_files_only"] = True

        # The official wrapper loads its processor separately without forwarding
        # local_files_only. A local snapshot path keeps both loads offline.
        self._model = Qwen3ASRModel.from_pretrained(model_path, **kwargs)
        self.device = str(getattr(self._model, "device", kwargs["device_map"]))
        self.dtype = str(getattr(self._model, "dtype", dtype))
        self.supports_context = 'context' in inspect.signature(self._model.transcribe).parameters
        self._context_provider = None

    def set_context_provider(self, provider):
        """Install a snapshot provider; no model reload and no transcript rewriting."""
        if provider and provider()['context'] and not self.supports_context:
            raise ValueError('Installed qwen-asr does not support context')
        self._context_provider = provider

    def _dtype(self, name: str):  # noqa: ANN001
        if name in {"", "none"}:
            return None
        if not hasattr(self._torch, name):
            raise ValueError(f"Unsupported ORT_TORCH_DTYPE={name!r}")
        return getattr(self._torch, name)

    def _language_hint(self, source_language: str | None) -> str | None:
        source = source_code(source_language)
        if source is None and source_language is not None:
            source = source_language.strip().lower()
        if source is None:
            return None
        if source in {"", "auto"}:
            return None
        if source not in ASR_FORCE_NAME:
            raise ValueError(
                f"Qwen3-ASR 不支援來源語提示 {source_language!r}；"
                f"支援：{sorted(ASR_FORCE_NAME)}"
            )
        return ASR_FORCE_NAME[source]

    def warmup(self) -> None:
        self.transcribe(np.zeros(INTERNAL_SAMPLE_RATE // 2, dtype=np.float32))

    def transcribe(
        self, audio16k: np.ndarray, language: str | None = None, *, context: str | None = None, vocabulary_snapshot=None
    ) -> Transcription:
        samples = np.asarray(audio16k)
        if samples.ndim != 1:
            raise ValueError("ASR input must be mono 16 kHz audio (one-dimensional array)")
        if samples.size == 0:
            return Transcription(text="", language=source_code(language))
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        if not np.isfinite(audio).all():
            raise ValueError("ASR input contains NaN or infinite samples")
        if np.max(np.abs(audio)) > 1.0001:
            raise ValueError("ASR input must be normalized float samples in [-1, 1]")
        hint = self._language_hint(language)
        provider = getattr(self, '_context_provider', None)
        if vocabulary_snapshot is not None and context is not None:raise ValueError('Choose one ASR context source')
        snapshot = vocabulary_snapshot if vocabulary_snapshot is not None else (provider() if context is None and provider else None)
        selected = snapshot['context'] if snapshot else (context or '')
        if not isinstance(selected,str) or len(selected)>1024:
            raise ValueError('ASR context must be a string of at most1024 characters')
        # This Chinese vocabulary must not bias Japanese or automatic-language routes.
        selected = selected if hint == 'Chinese' else ''
        if selected and not self.supports_context:
            raise ValueError('Installed qwen-asr does not support context')
        kwargs = dict(context=selected) if selected else {}
        with _LOCK, self._torch.inference_mode():
            results = self._model.transcribe(
                audio=(audio, INTERNAL_SAMPLE_RATE),
                language=hint,
                **kwargs,
            )
        self.last_recognition_context = dict(applied=bool(selected), language_hint=hint,
            vocabulary_sha256=snapshot['sha256'] if snapshot else None,
            context_characters=len(selected), transcript_postprocessing='strip_only')
        if not results:
            return Transcription(text="", language=source_code(language))
        text = getattr(results[0], "text", None)
        detected = source_code(getattr(results[0], "language", None)) or source_code(
            language
        )
        if isinstance(text, str) and is_internal_context_echo(text):
            self.last_recognition_context.update(
                transcript_postprocessing='blocked_internal_context_echo',
                blocked_reason='ASR_CONTEXT_ECHO_BLOCKED',
            )
            return Transcription(text="", language=detected)
        return Transcription(
            text=text.strip() if isinstance(text, str) else "",
            language=detected,
        )
