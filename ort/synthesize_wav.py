"""Standalone static-reference Japanese TTS. Writes WAVs; never opens audio devices."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import Config
from .pipeline.static_voice_reference import StaticVoiceReferenceCache

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def output_paths(output: Path, repeat: int) -> list[Path]:
    if output.suffix.lower() != ".wav":
        raise ValueError("Output must have a .wav extension")
    if not 1 <= repeat <= 20:
        raise ValueError("Repeat must be between 1 and 20")
    return [output if i == 0 else output.with_name(f"{output.stem}-{i + 1:02d}.wav")
            for i in range(repeat)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-wav", default=os.environ.get("ORT_QWEN3_TTS_REF_AUDIO", ""))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--reference-text", help="Exact Mandarin reference transcript")
    group.add_argument("--reference-text-file", type=Path, help="UTF-8 verbatim reference transcript")
    parser.add_argument("--text", required=True, help="Japanese text to synthesize")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--report", type=Path, help="Optional JSON timings/cache/memory report")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        transcript = (args.reference_text_file.read_text(encoding="utf-8-sig")
                      if args.reference_text_file else args.reference_text)
        if transcript is None:
            transcript = os.environ.get("ORT_QWEN3_TTS_REF_TEXT", "")
        reference = StaticVoiceReferenceCache(args.reference_wav, transcript)
        reference.reference()  # Fail before CUDA/model allocation.
        if not args.text.strip():
            raise ValueError("Japanese text must not be blank")
        outputs = output_paths(args.output, args.repeat)
        targets = outputs + ([args.report] if args.report else [])
        resolved = [p.resolve() for p in targets]
        protected = {Path(args.reference_wav).resolve()}
        if args.reference_text_file:
            protected.add(args.reference_text_file.resolve())
        if len(set(resolved)) != len(resolved) or protected.intersection(resolved):
            raise ValueError("Output/report paths must be distinct and cannot replace reference inputs")
        for path in targets:
            if path.exists() and not args.overwrite:
                raise ValueError(f"Output already exists (use --overwrite explicitly): {path}")
            if not path.parent.is_dir():
                raise ValueError(f"Output parent directory does not exist: {path.parent}")
        # Standalone verification is deliberately offline. Download the fixed
        # model separately; never silently download weights from a WAV command.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from .pipeline.backends.qwen3_tts_torch import Qwen3TTSTorch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment")
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        backend = Qwen3TTSTorch(Config(
            runtime="cuda", device="cuda:0", qwen3_tts_model=MODEL,
            qwen3_tts_offline=True, qwen3_tts_ref_audio=args.reference_wav,
            qwen3_tts_ref_text=transcript, torch_dtype="bfloat16", torch_attn="sdpa",
        ))
        rows = []
        for path in outputs:
            result = backend.synthesize(args.text, "ja")
            # Exclusive creation prevents accidental replacement even if a path
            # appears after preflight. No microphone or playback path is involved.
            with path.open("wb" if args.overwrite else "xb") as stream:
                sf.write(stream, result.samples, result.sample_rate, format="WAV", subtype="PCM_16")
            rows.append({"output": str(path.resolve()), "sample_rate": result.sample_rate,
                         "channels": 1, "subtype": "PCM_16",
                         "duration_seconds": result.samples.size / result.sample_rate,
                         "rms": float(np.sqrt(np.mean(result.samples.astype(np.float64) ** 2))),
                         "synthesis_seconds": backend.last_synthesis_seconds})
        report = {"model": MODEL, "device": backend.device, "dtype": backend.dtype,
                  "offline": True, "language": "Japanese", "outputs": rows, **backend.metrics()}
        if args.report:
            with args.report.open("w" if args.overwrite else "x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"TTS error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
