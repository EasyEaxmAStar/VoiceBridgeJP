"""Runtime configuration for selecting MLX, CUDA, or CPU backend profiles."""

from __future__ import annotations

import os
import json
import platform
from pathlib import Path
from dataclasses import dataclass, field

_DEFAULT_PERSONAL_PROFILE = ''


def _personal_reference_default(key: str) -> str:
    # Public reference configuration only; never select a personal model profile.
    config=Path(__file__).resolve().parents[1]/'config/public_tts.local.json'
    if not config.is_file():return ''
    data=json.loads(config.read_text(encoding='utf8'))
    return str(data.get('reference_wav' if key=='wav' else 'reference_text',''))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _norm(name: str) -> str:
    return (name or "").strip().lower().replace("-", "_")


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _resolve_runtime(runtime: str, device: str) -> str:
    rt = _norm(runtime)
    dev = _norm(device)
    if rt == "auto":
        if dev.startswith("cuda"):
            return "cuda"
        if dev in {"mlx", "cuda", "cpu"}:
            return dev
        return "mlx" if _is_apple_silicon() else "cpu"
    if rt not in {"mlx", "cuda", "cpu"}:
        raise ValueError("ORT_RUNTIME must be one of: auto, mlx, cuda, cpu")
    return rt


def _resolve_device(runtime: str, device: str) -> str:
    dev = _norm(device)
    if dev not in {"", "auto"}:
        return dev
    if runtime == "mlx":
        return "mlx"
    if runtime == "cuda":
        return "cuda"
    return "cpu"


def _default_asr(runtime: str) -> str:
    return "qwen3" if runtime in {"mlx", "cuda"} else "whisper"


def _default_tts(runtime: str) -> str:
    return "gptsovits"


def _default_whisper_compute(_runtime: str, device: str) -> str:
    return "float16" if device.startswith("cuda") else "int8"


def _default_whisper_device_index(device: str) -> int:
    if not device.startswith("cuda:"):
        return 0
    try:
        return int(device.split(":", 1)[1])
    except ValueError as exc:
        raise ValueError(f"Invalid CUDA device: {device!r}") from exc


def _default_torch_dtype(runtime: str) -> str:
    return "bfloat16" if runtime == "cuda" else "float32"


@dataclass(frozen=True)
class Config:
    # Runtime selection:
    #   auto : Apple Silicon uses MLX; other platforms use CPU.
    #   mlx  : MLX / Metal（Apple Silicon）。
    #   cuda : PyTorch/CTranslate2 CUDA。
    #   cpu  : PyTorch/CTranslate2/ONNX CPU。
    runtime: str = field(default_factory=lambda: _env("ORT_RUNTIME", "auto"))
    device: str = field(default_factory=lambda: _env("ORT_DEVICE", "auto"))

    # Backend selection. auto resolves from runtime; environment variables can override.
    #   ASR        : "qwen3" | "whisper" | "mock"
    #   Translator : "translategemma" | "opusmt" | "mock"
    #   TTS        : "gptsovits" (production) | legacy backends | "mock"
    asr_backend: str = field(default_factory=lambda: _env("ORT_ASR_BACKEND", "auto"))
    translator_backend: str = field(
        default_factory=lambda: _env("ORT_TRANSLATOR_BACKEND", "auto")
    )
    tts_backend: str = field(default_factory=lambda: _env("ORT_TTS_BACKEND", "auto"))

    # --- model ids -----------------------------------------------------------
    qwen_asr_model: str = field(
        default_factory=lambda: _env("ORT_QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
    )
    qwen_asr_device: str = field(
        default_factory=lambda: _env("ORT_QWEN_ASR_DEVICE", "auto")
    )
    qwen_asr_max_new_tokens: int = field(
        default_factory=lambda: int(_env("ORT_QWEN_ASR_MAX_NEW_TOKENS", "256"))
    )
    qwen_asr_offline: bool = field(
        default_factory=lambda: _env("ORT_QWEN_ASR_OFFLINE", "0").lower()
        in {"1", "true", "yes"}
    )
    translategemma_model: str = field(
        default_factory=lambda: _env(
            "ORT_TRANSLATEGEMMA_MODEL", "mlx-community/translategemma-4b-it-4bit"
        )
    )
    translategemma_torch_model: str = field(
        default_factory=lambda: _env(
            "ORT_TRANSLATEGEMMA_TORCH_MODEL", "google/translategemma-4b-it"
        )
    )
    opusmt_model: str = field(
        default_factory=lambda: _env(
            "ORT_OPUSMT_MODEL", "Helsinki-NLP/opus-mt-tc-big-zh-ja"
        )
    )
    kokoro_lang: str = field(default_factory=lambda: _env("ORT_KOKORO_LANG", "a"))
    kokoro_voice: str = field(
        default_factory=lambda: _env("ORT_KOKORO_VOICE", "af_heart")
    )
    supertonic_voice: str = field(
        default_factory=lambda: _env("ORT_SUPERTONIC_VOICE", "F1")
    )
    supertonic_steps: int = field(
        default_factory=lambda: int(_env("ORT_SUPERTONIC_STEPS", "8"))
    )
    # Qwen3-TTS Base voice cloning. Model accepts 0.6b/1.7b shortcuts or a full repo id.
    # An explicitly selected static reference takes precedence over rolling ASR input.
    qwen3_tts_model: str = field(
        default_factory=lambda: _env("ORT_QWEN3_TTS_MODEL", "1.7b")
    )
    qwen3_tts_ref_audio: str = field(
        default_factory=lambda: _env("ORT_QWEN3_TTS_REF_AUDIO", "")
    )
    qwen3_tts_ref_text: str = field(
        default_factory=lambda: _env("ORT_QWEN3_TTS_REF_TEXT", "")
    )
    qwen3_tts_offline: bool = field(
        default_factory=lambda: _env("ORT_QWEN3_TTS_OFFLINE", "0").strip().lower()
        in {"1", "true", "yes"}
    )

    gptsovits_root: str = field(default_factory=lambda: _env("ORT_GPTSOVITS_ROOT", ".model-cache/GPT-SoVITS"))
    gptsovits_python: str = field(default_factory=lambda: _env("ORT_GPTSOVITS_PYTHON", ".venv-gptsovits/Scripts/python.exe"))
    gptsovits_manifest: str = field(default_factory=lambda: _env("ORT_GPTSOVITS_MANIFEST", "gptsovits.lock.json"))
    # Activated atomically after real personal checkpoint/reference validation.
    gptsovits_personal_profile: str = field(default_factory=lambda: _env("ORT_GPTSOVITS_PERSONAL_PROFILE", _DEFAULT_PERSONAL_PROFILE))
    gptsovits_voice_preset: str = field(default_factory=lambda: _env('ORT_VOICE_PRESET',''))
    gptsovits_preset_registry: str = field(default_factory=lambda: _env('ORT_VOICE_PRESET_REGISTRY',''))
    gptsovits_prosody_level: str = field(default_factory=lambda: _env('ORT_PROSODY_LEVEL','ORIGINAL'))
    gptsovits_seed: str = field(default_factory=lambda: _env('ORT_TTS_SEED','auto'))
    gptsovits_ref_audio: str = field(default_factory=lambda: _env("ORT_GPTSOVITS_REF_AUDIO", _personal_reference_default('wav')))
    gptsovits_ref_text: str = field(default_factory=lambda: _env("ORT_GPTSOVITS_REF_TEXT", _personal_reference_default('text')))
    gptsovits_ref_language: str = field(default_factory=lambda: _env("ORT_GPTSOVITS_REF_LANGUAGE", "ja"))
    gptsovits_pronunciation_lexicon: str = field(default_factory=lambda: _env(
        "ORT_GPTSOVITS_PRONUNCIATION_LEXICON",
        str(Path(__file__).resolve().parents[1]/"config/pronunciation.ja.production.json")))
    gptsovits_frontend_diagnostics: bool = field(default_factory=lambda: _env("ORT_GPTSOVITS_FRONTEND_DIAGNOSTICS", "0") == "1")
    # Audit-only future selection; STEP 07 never activates unverified weights.
    gptsovits_custom_gpt_path: str = field(default_factory=lambda: _env("ORT_GPTSOVITS_CUSTOM_GPT", ""))
    gptsovits_custom_sovits_path: str = field(default_factory=lambda: _env("ORT_GPTSOVITS_CUSTOM_SOVITS", ""))
    gptsovits_request_queue: int = 4
    gptsovits_output_queue: int = 8
    gptsovits_startup_timeout: float = 180.0
    gptsovits_request_timeout: float = 120.0

    # --- optional ASR ---------------------------------------------------------
    whisper_model: str = field(
        default_factory=lambda: _env("ORT_WHISPER_MODEL", "small")
    )
    whisper_device: str = field(
        default_factory=lambda: _env("ORT_WHISPER_DEVICE", "auto")
    )
    whisper_compute_type: str = field(
        default_factory=lambda: _env("ORT_WHISPER_COMPUTE", "auto")
    )
    whisper_device_index: int | str = field(
        default_factory=lambda: _env("ORT_WHISPER_DEVICE_INDEX", "auto")
    )

    # --- non-MLX torch runtime ----------------------------------------------
    torch_dtype: str = field(default_factory=lambda: _env("ORT_TORCH_DTYPE", "auto"))
    torch_attn: str = field(default_factory=lambda: _env("ORT_TORCH_ATTN", "auto"))
    torch_max_new_tokens: int = field(
        default_factory=lambda: int(_env("ORT_TORCH_MAX_NEW_TOKENS", "256"))
    )

    # Segmentation / commit parameters.
    energy_threshold: float = field(
        default_factory=lambda: float(_env("ORT_VAD_ENERGY", "0.012"))
    )
    # incremental=True: clause-level commits for low latency.
    # incremental=False: whole-utterance commits for deterministic regression anchors.
    incremental: bool = field(
        default_factory=lambda: _env("ORT_INCREMENTAL", "1") not in ("0", "false", "")
    )
    # Incremental mode: commit spoken content after an intra-clause pause this
    # long. Tuned to sentence boundaries (not every micro-pause) so committed
    # spans are coherent sentences rather than 2-3 word fragments.
    commit_silence_ms: int = field(
        default_factory=lambda: int(_env("ORT_COMMIT_SILENCE_MS", "700"))
    )
    # Incremental mode: force a commit after continuous speech reaches this cap.
    # High enough that natural sentence pauses commit first; the cap only
    # bounds runaway monologue and rarely cuts mid-word.
    max_commit_ms: int = field(
        default_factory=lambda: int(_env("ORT_MAX_COMMIT_MS", "8000"))
    )
    # Whole-utterance mode only.
    min_silence_ms: int = field(
        default_factory=lambda: int(_env("ORT_MIN_SILENCE_MS", "550"))
    )
    max_segment_ms: int = field(
        default_factory=lambda: int(_env("ORT_MAX_SEGMENT_MS", "12000"))
    )
    min_segment_ms: int = field(
        default_factory=lambda: int(_env("ORT_MIN_SEGMENT_MS", "300"))
    )

    # Translated speech is typically ~1.5x longer than the source utterance,
    # so continuous playback drifts behind realtime. Play output back slightly
    # faster (pitch-preserved time-stretch) to absorb the difference. 1.0
    # disables it; values above ~1.4 start to sound rushed.
    output_speed: float | str = field(
        default_factory=lambda: _env("ORT_OUTPUT_SPEED", "auto")
    )

    @classmethod
    def from_env(cls) -> "Config":
        return cls()

    def __post_init__(self) -> None:
        runtime = _resolve_runtime(self.runtime, self.device)
        device = _resolve_device(runtime, self.device)
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "device", device)

        if _norm(self.asr_backend) == "auto":
            object.__setattr__(self, "asr_backend", _default_asr(runtime))
        else:
            object.__setattr__(self, "asr_backend", _norm(self.asr_backend))

        if _norm(self.translator_backend) == "auto":
            object.__setattr__(self, "translator_backend", "translategemma")
        else:
            object.__setattr__(
                self, "translator_backend", _norm(self.translator_backend)
            )

        if _norm(self.tts_backend) == "auto":
            object.__setattr__(self, "tts_backend", _default_tts(runtime))
        else:
            object.__setattr__(self, "tts_backend", _norm(self.tts_backend))


        speed = (1.0 if self.tts_backend == "gptsovits" else 1.25) if self.output_speed == "auto" else float(self.output_speed)
        object.__setattr__(self, "output_speed", speed)
        if self.gptsovits_voice_preset and speed!=1.0:
            raise ValueError('Approved personal presets require native PCM and browser1.25; extra output time stretching is disabled')

        if _norm(self.whisper_device) == "auto":
            object.__setattr__(
                self, "whisper_device", "cuda" if runtime == "cuda" else "cpu"
            )
        else:
            object.__setattr__(self, "whisper_device", _norm(self.whisper_device))

        whisper_device_index = self.whisper_device_index
        if isinstance(whisper_device_index, str):
            raw_index = _norm(whisper_device_index)
            if raw_index in {"", "auto"}:
                whisper_device_index = _default_whisper_device_index(device)
            else:
                whisper_device_index = int(raw_index)
        elif whisper_device_index < 0:
            whisper_device_index = _default_whisper_device_index(device)
        object.__setattr__(self, "whisper_device_index", whisper_device_index)

        if _norm(self.whisper_compute_type) == "auto":
            object.__setattr__(
                self,
                "whisper_compute_type",
                _default_whisper_compute(runtime, self.whisper_device),
            )
        else:
            object.__setattr__(
                self, "whisper_compute_type", _norm(self.whisper_compute_type)
            )

        if _norm(self.torch_dtype) == "auto":
            object.__setattr__(self, "torch_dtype", _default_torch_dtype(runtime))
        else:
            object.__setattr__(self, "torch_dtype", _norm(self.torch_dtype))

        if _norm(self.torch_attn) == "auto":
            object.__setattr__(
                self, "torch_attn", "sdpa" if runtime == "cuda" else "eager"
            )
        else:
            object.__setattr__(self, "torch_attn", _norm(self.torch_attn))

    def mock(self) -> "Config":
        """Return a config with all mock backends for protocol tests."""
        from dataclasses import replace

        return replace(
            self, asr_backend="mock", translator_backend="mock", tts_backend="mock"
        )
