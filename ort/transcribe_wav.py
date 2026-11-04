"""Native Windows CUDA Qwen3-ASR-0.6B saved-WAV recognition and diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

from .asr_audio import load_asr_wav
from .config import Config

MODEL = "Qwen/Qwen3-ASR-0.6B"


def _normalized(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", text) if ch.isalnum())


def _recognize(args: argparse.Namespace) -> int:
    # Validate/decode before allocating a GPU model.
    audio = load_asr_wav(args.wav)
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    import torch
    from .pipeline.backends.qwen3_asr_torch import Qwen3ASRTorch

    if not args.device.startswith("cuda:"):
        raise ValueError("This command requires an explicit CUDA device, e.g. cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install the pinned CUDA wheel set; see README.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    backend = Qwen3ASRTorch(
        Config(
            runtime="cuda", device=args.device, qwen_asr_device=args.device,
            qwen_asr_model=MODEL, qwen_asr_offline=True,
            torch_attn="sdpa",
            torch_dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            qwen_asr_max_new_tokens=args.max_new_tokens,
        )
    )
    torch.cuda.synchronize(device)
    init_seconds = time.perf_counter() - started
    if not backend.device.startswith("cuda"):
        raise RuntimeError(f"ASR loaded on {backend.device}; expected CUDA")
    report = {
        "model": MODEL, "backend": "transformers", "attention": "sdpa",
        "streaming": False, "offline": True, "gpu": properties.name,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "model_device": backend.device, "model_dtype": backend.dtype,
        "gpu_total_mib": properties.total_memory / 2**20,
        "wav": str(args.wav.resolve()),
        "source_sample_rate": audio.source_sample_rate,
        "source_channels": audio.source_channels,
        "duration_seconds": audio.duration_seconds,
        "internal_sample_rate": 16000,
        "init_seconds": init_seconds,
        "init_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "runs": [],
    }
    for _ in range(args.repeat):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = backend.transcribe(audio.samples, language="zh")
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        run = {
            "text": result.text, "language": result.language,
            "inference_seconds": elapsed, "rtf": elapsed / audio.duration_seconds,
        }
        if args.expected_text is not None:
            run["expected_match"] = _normalized(result.text) == _normalized(args.expected_text)
        report["runs"].append(run)
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    report["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return int(any(not run["text"] or run.get("expected_match") is False for run in report["runs"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="download only Qwen3-ASR-0.6B and its processor")
    command = sub.add_parser("transcribe", help="offline saved-WAV recognition on CUDA, no streaming")
    command.add_argument("wav", type=Path)
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--repeat", type=int, default=1)
    command.add_argument("--max-new-tokens", type=int, default=256)
    command.add_argument("--expected-text", help="optional punctuation-insensitive regression reference")
    command.add_argument("--report", type=Path, help="optional local JSON metrics/transcript file")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            from huggingface_hub import snapshot_download

            print(snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"], max_workers=2))
            return 0
        return _recognize(args)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"Qwen ASR error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
