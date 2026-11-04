"""CLI for downloading and testing the offline Chinese-to-Japanese translator."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import Config
from .pipeline.backends.opus_mt import DEFAULT_OPUSMT_MODEL, OpusMTTranslator, process_memory_mb


def _download(model_id: str) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub in the OPUS-MT CPU environment") from exc
    # Keep only what the PyTorch Marian model and SentencePiece tokenizer need;
    # the repository also publishes large TensorFlow weights and benchmark data.
    path = snapshot_download(
        repo_id=model_id,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "source.spm",
            "target.spm",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
    )
    print(f"Downloaded {model_id} to {path}")
    print("Subsequent translation uses local_files_only=True (offline).")
    return 0


def _cases(path: Path) -> list[dict[str, str]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not all(
        isinstance(case, dict)
        and isinstance(case.get("zh"), str)
        and isinstance(case.get("ja_contains", []), list)
        and all(isinstance(term, str) for term in case.get("ja_contains", []))
        for case in cases
    ):
        raise ValueError(f"Fixture file must contain a JSON list of objects with string 'zh': {path}")
    return cases


def _translate(args: argparse.Namespace) -> int:
    config = Config(
        runtime="cpu",
        device="cpu",
        translator_backend="opusmt",
        opusmt_model=args.model,
        torch_max_new_tokens=args.max_new_tokens,
    )
    started = time.perf_counter()
    translator = OpusMTTranslator(config)
    load_seconds = time.perf_counter() - started
    print(f"Model: {translator.repo} (CPU, float32)")
    print(
        "Memory MiB: "
        f"RSS before load={translator.memory_before_load_mb if translator.memory_before_load_mb is not None else 'n/a'}, "
        f"RSS after load={translator.memory_after_load['rss_mb'] if translator.memory_after_load['rss_mb'] is not None else 'n/a'}, "
        f"peak RSS={translator.memory_after_load['peak_rss_mb'] if translator.memory_after_load['peak_rss_mb'] is not None else 'n/a'}, "
        f"model parameters={translator.parameter_memory_mb:.1f}"
    )
    print(f"Load time: {load_seconds:.3f}s")

    cases = _cases(args.cases) if args.cases else [{"zh": args.text}]
    failed = 0
    for index, case in enumerate(cases, 1):
        source = case["zh"]
        before = time.perf_counter()
        target = translator.translate(source, source="zh", target="ja")
        elapsed = time.perf_counter() - before
        case_failed = False
        print(f"[{index}] zh: {source}")
        print(f"[{index}] ja: {target}")
        print(f"[{index}] latency: {elapsed:.3f}s")
        if not target:
            case_failed = True
            print(f"[{index}] ERROR: model returned empty output", file=sys.stderr)
        missing = [term for term in case.get("ja_contains", []) if term not in target]
        if missing:
            case_failed = True
            print(
                f"[{index}] ERROR: missing semantic/punctuation anchors {missing!r}",
                file=sys.stderr,
            )
        elif case.get("ja_contains"):
            print(f"[{index}] fixture anchors: PASS")
        failed += int(case_failed)
    print(f"Peak process memory MiB: {process_memory_mb()['peak_rss_mb'] or 'n/a'}")
    print(f"Case summary: {len(cases) - failed}/{len(cases)} passed")
    return int(failed > 0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="download weights/tokenizer while online")
    download.add_argument("--model", default=DEFAULT_OPUSMT_MODEL)
    translate = sub.add_parser("translate", help="translate text locally with cached model files")
    translate.add_argument("--model", default=DEFAULT_OPUSMT_MODEL)
    group = translate.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Chinese source text")
    group.add_argument("--cases", type=Path, help="JSON list of objects with 'zh' source strings")
    translate.add_argument("--max-new-tokens", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _download(args.model) if args.command == "download" else _translate(args)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"OPUS-MT error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
