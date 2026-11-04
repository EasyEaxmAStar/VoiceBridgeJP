"""Reproducible cached-model ASR benchmarks; run never downloads data/models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .asr_audio import load_asr_wav
from .asr_metrics import NORMALIZATION, score_text
from .audio import resample
from .config import Config
from .pipeline.segmentation import Segmenter

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "tests/fixtures/asr_benchmark.json"


def gpu_telemetry() -> dict:
    """Sample outside timed inference; failure never changes benchmark behavior."""
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=driver_version,temperature.gpu,power.draw,clocks.current.sm,pstate",
            "--format=csv,noheader", "--id=0",
        ], capture_output=True, text=True, timeout=5, check=True)
        return {"utc": datetime.now(timezone.utc).isoformat(), "nvidia_smi": result.stdout.strip()}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"unavailable": str(exc)}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_manifest(path: Path) -> tuple[dict, Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or not manifest.get("cases"):
        raise ValueError("Manifest must have version=1 and a non-empty cases list")
    ids = [case["id"] for case in manifest["cases"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Case IDs must be unique")
    for case in manifest["cases"]:
        if "asset" in case:
            dataset_row = manifest.get("assets", {}).get(case["asset"], {}).get("hf_row")
            if dataset_row and case.get("reference") != dataset_row["reference"]:
                raise ValueError(f"Case reference differs from the published dataset row: {case['id']}")
    return manifest, (path.parent / manifest.get("root", ".")).resolve()


def fetch_assets(path: Path) -> None:
    import requests

    manifest, root = read_manifest(path)
    for name, asset in manifest.get("assets", {}).items():
        destination = root / asset["file"]
        if destination.is_file() and sha256(destination) == asset["sha256"]:
            print(f"Verified {name}", flush=True)
            continue
        if destination.exists():
            raise ValueError(f"Existing fixture hash mismatch; preserve/inspect it before fetching: {destination}")
        url = asset.get("url")
        if "hf_row" in asset:
            spec = asset["hf_row"]
            response = requests.get("https://datasets-server.huggingface.co/rows", params={
                "dataset": spec["dataset"], "config": spec["config"],
                "split": spec["split"], "offset": spec["index"], "length": 1,
            }, timeout=60)
            response.raise_for_status()
            row = response.json()["rows"][0]["row"]
            if row["transcription"] != spec["reference"]:
                raise ValueError(f"Published reference changed for {name}")
            url = row["audio"][0]["src"]
            if spec["revision"] not in url:
                raise ValueError(f"Dataset revision changed for {name}")
        if not url:
            raise ValueError(f"No download source for {name}; add the local WAV manually")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        content = response.content
        expected = asset.get("download_sha256", asset["sha256"])
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Download SHA-256 mismatch for {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if asset.get("convert_to_wav"):
            import io

            audio, rate = sf.read(io.BytesIO(content), dtype="float32")
            sf.write(destination, audio, rate, subtype="PCM_16")
            if sha256(destination) != asset["sha256"]:
                raise ValueError(f"Converted WAV hash mismatch for {name}; decoder version may differ")
        else:
            destination.write_bytes(content)
        print(f"Fetched {name}: {destination}", flush=True)


@dataclass
class Case:
    id: str
    reference: str
    punctuation_known: bool
    tags: list[str]
    audio: np.ndarray
    segments: list[np.ndarray]
    diagnostics: dict


def prepare_cases(manifest: dict, root: Path, seed: int) -> list[Case]:
    specs = {case["id"]: case for case in manifest["cases"]}
    prepared: dict[str, Case] = {}
    active: set[str] = set()

    def prepare(case_id: str) -> Case:
        if case_id in prepared:
            return prepared[case_id]
        if case_id in active:
            raise ValueError(f"Cyclic derived fixture: {case_id}")
        active.add(case_id)
        spec = specs[case_id]
        started = time.perf_counter()
        if "base" in spec:
            base = prepare(spec["base"])
            audio = base.audio.copy()
            reference = base.reference
            known = base.punctuation_known
            source = {"derived_from": base.id}
        elif "concat" in spec:
            bases = [prepare(value) for value in spec["concat"]]
            silence = np.zeros(round(spec.get("pause_s", 1) * 16000), dtype=np.float32)
            parts = []
            for base in bases:
                if parts:
                    parts.append(silence)
                parts.append(base.audio)
            audio = np.concatenate(parts)
            reference = " ".join(base.reference for base in bases)
            known = all(base.punctuation_known for base in bases)
            source = {"derived_from": spec["concat"], "inserted_pause_s": len(silence) / 16000}
        else:
            asset = manifest["assets"][spec["asset"]]
            path = root / asset["file"]
            if not path.is_file() or sha256(path) != asset["sha256"]:
                raise ValueError(f"Missing/changed fixture {path}; run the explicit fetch command")
            loaded = load_asr_wav(path)
            audio = loaded.samples
            if "start_s" in spec:
                audio = audio[round(spec["start_s"] * 16000):round(spec["end_s"] * 16000)]
            reference = spec["reference"]
            known = spec.get("punctuation_known", False)
            source = {"asset": spec["asset"], "sha256": asset["sha256"],
                      "source_rate": loaded.source_sample_rate, "source_channels": loaded.source_channels}
        if not audio.size:
            raise ValueError(f"Empty fixture: {case_id}")
        transform = spec.get("transform", {})
        kind = transform.get("kind")
        if kind == "padding":
            audio = np.pad(audio, (round(transform["before_s"] * 16000), round(transform["after_s"] * 16000)))
        elif kind == "noise":
            rng = np.random.default_rng(seed + int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16))
            noise = rng.normal(size=audio.size).astype(np.float32)
            noise *= np.sqrt(np.mean(audio.astype(np.float64) ** 2)) / max(1e-12, np.sqrt(np.mean(noise.astype(np.float64) ** 2))) / 10 ** (transform["snr_db"] / 20)
            audio = np.clip(audio + noise, -1, 1)
        elif kind == "clipping":
            audio = np.clip(audio * transform["gain"], -1, 1)
        elif kind == "resample48k":
            stereo = np.repeat(resample(audio, 16000, 48000)[:, None], 2, axis=1)
            # Simulate a PCM16 stereo recording, including quantization, then
            # the same downmix/resample order used by the WAV loader.
            pcm = np.clip(stereo * 32768, -32768, 32767).astype("<i2")
            audio = np.clip(resample((pcm.astype(np.float32) / 32768).mean(axis=1), 48000, 16000), -1, 1)
        elif kind is not None:
            raise ValueError(f"Unknown transform {kind}")
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        frames = [audio[i:i + 480] for i in range(0, len(audio), 480)]
        active_frames = [i for i, frame in enumerate(frames) if np.sqrt(np.mean(frame.astype(np.float64) ** 2)) >= 0.012]
        diagnostics = {**source, "transform": transform, "duration_s": len(audio) / 16000,
                       "peak": float(np.max(np.abs(audio))), "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
                       "clipped_sample_fraction": float(np.mean(np.abs(audio) >= 0.999)),
                       "energy_start_s": active_frames[0] * .03 if active_frames else None,
                       "energy_end_s": min(len(audio) / 16000, (active_frames[-1] + 1) * .03) if active_frames else None}
        segments = [audio]
        if spec.get("segmentation") == "default":
            segmenter = Segmenter(Config(runtime="cuda", energy_threshold=.012, incremental=True,
                                         commit_silence_ms=700, max_commit_ms=8000, min_segment_ms=300))
            output = []
            for i in range(0, len(audio), 320):
                output.extend(segmenter.add(audio[i:i + 320]))
            output.extend(segmenter.flush())
            segments = [segment.samples for segment in output]
        diagnostics.update(segment_seconds=[len(value) / 16000 for value in segments],
                           omitted_samples=len(audio) - sum(len(value) for value in segments),
                           preparation_seconds=time.perf_counter() - started)
        result = Case(case_id, reference, known, spec.get("tags", []), audio, segments, diagnostics)
        prepared[case_id] = result
        active.remove(case_id)
        return result

    return [prepare(value) for value in specs]


def summarize(rows: list[dict]) -> dict:
    # Accuracy counts one canonical result per distinct case. Repeats increase
    # latency observations, not the amount of independently labelled speech.
    first = {row["case_id"]: row for row in reversed(rows)}
    characters = sum(row["reference_characters"] for row in first.values())
    errors = sum(row["character_errors"] for row in first.values())
    latency = [row["latency_s"] for row in rows]
    rtf = [row["rtf"] for row in rows]
    punct = [row for row in first.values() if row["punctuation_errors"] is not None]
    return {"cases": len(first), "latency_observations": len(rows),
            "reference_characters": characters, "character_errors": errors, "cer": errors / characters,
            "punctuation_scored_cases": len(punct),
            "punctuation_errors": sum(row["punctuation_errors"] for row in punct),
            "p50_latency_s": float(np.percentile(latency, 50)), "p95_latency_s": float(np.percentile(latency, 95)),
            "p50_rtf": float(np.percentile(rtf, 50)), "p95_rtf": float(np.percentile(rtf, 95))}


def _run(args: argparse.Namespace) -> int:
    import torch
    from transformers.utils import logging as hf_logging
    from .pipeline.backends.qwen3_asr_torch import Qwen3ASRTorch

    manifest, root = read_manifest(args.manifest)
    cases = prepare_cases(manifest, root, args.seed)
    if args.ids:
        requested = set(args.ids.split(","))
        cases = [case for case in cases if case.id in requested]
        if requested != {case.id for case in cases}:
            raise ValueError("Unknown case ID in --ids")
    if args.repeats < 3:
        raise ValueError("At least 3 repeats are required for warm percentile estimates")
    if (args.candidate_threads is not None and args.candidate_threads < 1) or (args.candidate_max_tokens is not None and args.candidate_max_tokens < 1):
        raise ValueError("Candidate thread/token counts must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; install the existing ASR CUDA lock")
    hf_logging.set_verbosity_error()  # Logging only; leave generation behavior intact.
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    original_threads = torch.get_num_threads()
    source_files = (Path(__file__), Path(__file__).with_name("asr_metrics.py"),
                    Path(__file__).parent / "pipeline/backends/qwen3_asr_torch.py")
    source_hashes = {}
    snapshots = args.output.parent / "source_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    for source in source_files:
        digest = sha256(source)
        source_hashes[str(source.relative_to(Path(__file__).resolve().parents[1]))] = digest
        snapshot = snapshots / f"{digest}.py"
        if not snapshot.exists():
            snapshot.write_bytes(source.read_bytes())
    started = time.perf_counter()
    backend = Qwen3ASRTorch(Config(runtime="cuda", device="cuda:0", qwen_asr_device="cuda:0",
                                  qwen_asr_model="0.6b", qwen_asr_offline=True,
                                  torch_attn="sdpa", torch_dtype="bfloat16", qwen_asr_max_new_tokens=256))
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - started
    if backend._model.model.training:
        raise RuntimeError("Expected the loaded ASR model to be in eval mode")
    if backend.device != "cuda:0":
        raise RuntimeError(f"Expected cuda:0, got {backend.device}")
    report = {"schema_version": 1, "normalization": NORMALIZATION, "seed": args.seed,
              "utc": datetime.now(timezone.utc).isoformat(), "gpu_telemetry_start": gpu_telemetry(),
              "manifest_sha256": sha256(args.manifest), "model": backend._repo,
              "snapshot_revision": backend._model.model.config._name_or_path.replace("\\", "/").split("/")[-1],
              "model_device": backend.device, "model_dtype": backend.dtype,
              "gpu": torch.cuda.get_device_name(), "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "qwen-asr", "transformers", "numpy", "librosa", "soundfile", "soxr")},
              "cuda_runtime": torch.version.cuda, "default_cpu_threads": original_threads,
              "initialization_s": initialization_seconds,
              "initialization_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "generation": backend._model.model.generation_config.to_dict(),
              "vad": {"frame_ms": 30, "energy_threshold": .012, "commit_silence_ms": 700,
                      "max_commit_ms": 8000, "min_segment_ms": 300},
              "code_sha256": source_hashes, "manifest": manifest,
              "cases": {case.id: {"reference": case.reference, "tags": case.tags,
                                     "punctuation_known": case.punctuation_known, **case.diagnostics} for case in cases},
              "priming": [], "conditions": []}
    languages = ["zh", "auto"] if args.language == "both" else [args.language]

    def recognize(case: Case, language: str) -> dict:
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = [backend.transcribe(part, language="zh" if language == "zh" else None) for part in case.segments]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        text = "".join(output.text for output in outputs)
        return {"case_id": case.id, "language_mode": language, "text": text,
                "detected_languages": [output.language for output in outputs],
                "latency_s": elapsed, "rtf": elapsed / (len(case.audio) / 16000),
                **score_text(case.reference, text, punctuation_known=case.punctuation_known)}

    for case in cases:
        for language in languages:
            report["priming"].append(recognize(case, language))
        print(f"Primed {case.id}", flush=True)
    report["first_request"] = report["priming"][0]
    report["priming_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    report["priming_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    conditions = [("baseline_before", original_threads, 256)]
    if args.candidate_threads is not None:
        conditions += [("candidate_threads", args.candidate_threads, 256), ("baseline_after", original_threads, 256)]
    elif args.candidate_max_tokens is not None:
        conditions += [("candidate_max_tokens", original_threads, args.candidate_max_tokens), ("baseline_after", original_threads, 256)]
    try:
        for name, threads, max_tokens in conditions:
            telemetry_before = gpu_telemetry()
            torch.set_num_threads(threads)
            backend._model.max_new_tokens = max_tokens
            first = recognize(cases[0], languages[0])
            torch.cuda.reset_peak_memory_stats()
            rows = []
            for repetition in range(args.repeats):
                jobs = [(case, language) for case in cases for language in languages]
                random.Random(args.seed + repetition).shuffle(jobs)
                for case, language in jobs:
                    rows.append({"repeat": repetition, **recognize(case, language)})
                print(f"{name} round {repetition + 1}/{args.repeats}", flush=True)
            condition = {"name": name, "cpu_threads": threads, "max_new_tokens": max_tokens,
                         "gpu_telemetry_before": telemetry_before, "gpu_telemetry_after": gpu_telemetry(),
                         "condition_first_request": first, "runs": rows,
                         "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                         "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                         "summary": {language: summarize([row for row in rows if row["language_mode"] == language]) for language in languages}}
            report["conditions"].append(condition)
            # Save each completed phase so an interrupted experiment is reviewable.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        torch.set_num_threads(original_threads)
        backend._model.max_new_tokens = 256
    print(json.dumps({condition["name"]: condition["summary"] for condition in report["conditions"]}, ensure_ascii=False, indent=2))
    return 0


def add_recording(args: argparse.Namespace) -> None:
    load_asr_wav(args.wav)
    reference = args.reference_file.read_text(encoding="utf-8").strip()
    score_text(reference, reference)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest.exists() else {"version": 1, "root": ".", "assets": {}, "cases": []}
    if args.id in manifest["assets"] or any(case["id"] == args.id for case in manifest["cases"]):
        raise ValueError("Case ID already exists; choose another ID")
    manifest["assets"][args.id] = {"file": str(args.wav.resolve()), "sha256": sha256(args.wav), "source": "user recording; user-supplied verbatim reference"}
    manifest["cases"].append({"id": args.id, "asset": args.id, "reference": reference,
                              "punctuation_known": args.punctuation_known, "tags": args.tags.split(",")})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="explicitly download hash-verified speech fixtures only")
    fetch.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run = commands.add_parser("run", help="offline reused-model benchmark with optional A/B/A experiment")
    run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--seed", type=int, default=20261028)
    run.add_argument("--language", choices=["zh", "auto", "both"], default="both")
    run.add_argument("--ids", help="optional comma-separated case IDs")
    group = run.add_mutually_exclusive_group()
    group.add_argument("--candidate-threads", type=int)
    group.add_argument("--candidate-max-tokens", type=int)
    add = commands.add_parser("add", help="add a local user WAV and verbatim UTF-8 reference")
    add.add_argument("--manifest", type=Path, required=True)
    add.add_argument("--id", required=True)
    add.add_argument("--wav", type=Path, required=True)
    add.add_argument("--reference-file", type=Path, required=True)
    add.add_argument("--tags", default="user_recording")
    add.add_argument("--punctuation-known", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            fetch_assets(args.manifest)
        elif args.command == "add":
            add_recording(args)
        else:
            return _run(args)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"ASR benchmark error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
