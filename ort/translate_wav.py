"""Offline production Mandarin WAV -> Japanese 24 kHz PCM16 WAV and metrics."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import wave

import numpy as np
import soundfile as sf

from . import inference
from .asr_audio import load_asr_wav
from .config import Config
from .gpu_sampling import GPUSampler
from .wav_pipeline import ASR_MODEL, MT_MODEL, WavTranslationPipeline, recorded_utterances, require_production


@contextmanager
def offline_network():
    """No sockets are needed by the WAV pipeline or anonymous-pipe worker."""
    original = socket.socket.connect, socket.socket.connect_ex
    keys = {"HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1", "HF_HUB_DISABLE_TELEMETRY":"1"}
    previous = {key:os.environ.get(key) for key in keys}
    stats = {"python_socket_guard":True, "blocked_connection_attempts":0,
             "successful_python_socket_connections":0,
             "limitations":"Python socket guard and cached-only loaders; not an OS firewall or packet capture. The TTS child enforces its own socket guard."}
    def denied(*args, **kwargs):
        stats["blocked_connection_attempts"] += 1
        raise RuntimeError("Offline WAV pipeline prohibits network connections")
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    os.environ.update(keys)
    try:
        yield stats
    finally:
        socket.socket.connect, socket.socket.connect_ex = original
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def validate_output(path, expected_samples, expected_digest):
    info = sf.info(path)
    if (info.format, info.subtype, info.samplerate, info.channels, info.frames) != ("WAV", "PCM_16", 24000, 1, expected_samples):
        raise RuntimeError("Final WAV container/rate/channels/sample count is invalid")
    digest = hashlib.sha256()
    squares = 0.0
    for block in sf.blocks(path, blocksize=24000, dtype="int16"):
        digest.update(block.astype("<i2", copy=False).tobytes())
        squares += float(np.sum((block.astype(np.float64)/32768)**2))
    if digest.hexdigest() != expected_digest:
        raise RuntimeError("Final WAV read-back PCM checksum mismatch")
    rms = float(np.sqrt(squares/max(1, expected_samples)))
    if expected_samples < 2400 or rms < 1e-4:
        raise RuntimeError("Final Japanese WAV is too short or silent")
    return {"format":info.format, "subtype":info.subtype, "sample_rate":info.samplerate,
            "channels":info.channels, "samples":info.frames, "duration_seconds":info.duration,
            "rms":rms, "pcm_sha256":digest.hexdigest(), "readback_integrity":"PASS"}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-wav", type=Path, required=True)
    p.add_argument("--reference-wav", type=Path)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--reference-text")
    group.add_argument("--reference-text-file", type=Path)
    p.add_argument("--reference-language", choices=["ja"], default="ja")
    p.add_argument("--output-wav", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--vad", action="store_true", help="Split using the unchanged configured EnergyVAD/commit settings")
    p.add_argument("--repeat", type=int, default=1, help="Repeat the saved input with the same resident model instances (1-10)")
    p.add_argument("--warmup", action="store_true", help="One excluded existing pipeline warmup before processing")
    p.add_argument("--queue-capacity", type=int, default=4)
    p.add_argument("--job-timeout", type=float, default=180)
    p.add_argument('--voice-preset',choices=['PERSONAL_R06','PERSONAL_R17','PERSONAL_R18'])
    p.add_argument('--prosody-level',choices=['ORIGINAL','SUBTLE','NATURAL','EXPRESSIVE'])
    p.add_argument('--seed',help='Fixed seed integer or random; omitted uses the selected level default')
    return p


def run(args):
    started = time.perf_counter()
    report = {"status":"RUNNING", "created_utc":datetime.now(timezone.utc).isoformat(),
              "time_basis":"time.perf_counter seconds in the same parent; completion includes output validation and worker/telemetry shutdown, excludes interpreter startup/exit and the report's own serialization. Not live capture timestamps.",
              "timestamps":{"cli_start":started}, "utterances":[],
              "personal_voice_similarity":"NOT YET VERIFIED"}
    defaults = Config()
    text = args.reference_text
    if args.reference_text_file:
        text = args.reference_text_file.read_text(encoding="utf-8-sig")
    if text is None:
        text = defaults.gptsovits_ref_text
    reference = args.reference_wav or Path(defaults.gptsovits_ref_audio or ".missing-reference")
    cfg = Config(runtime="cuda", device="cuda:0", qwen_asr_device="cuda:0",
        asr_backend="qwen3", qwen_asr_model=ASR_MODEL, qwen_asr_offline=True,
        torch_dtype="bfloat16", torch_attn="sdpa", translator_backend="opusmt", opusmt_model=MT_MODEL,
        tts_backend="gptsovits", output_speed=1.0, gptsovits_ref_audio=str(reference),
        gptsovits_ref_text=text, gptsovits_ref_language=args.reference_language)
    from dataclasses import replace
    cfg=replace(cfg,gptsovits_voice_preset=getattr(args,'voice_preset',None) or defaults.gptsovits_voice_preset,
        gptsovits_prosody_level=getattr(args,'prosody_level',None) or defaults.gptsovits_prosody_level,
        gptsovits_seed=getattr(args,'seed',None) or defaults.gptsovits_seed)
    if not 1 <= args.repeat <= 10 or not 1 <= args.queue_capacity <= 32 or args.job_timeout <= 0:
        raise ValueError("Invalid repeat/queue capacity/job timeout")
    outputs = [args.output_wav.resolve(), args.report.resolve()]
    playback_page=args.output_wav.with_suffix('.playback.html') if cfg.gptsovits_voice_preset else None
    if playback_page:outputs.append(playback_page.resolve())
    inputs = {args.input_wav.resolve(), reference.resolve()}
    if args.reference_text_file:
        inputs.add(args.reference_text_file.resolve())
    if len(set(outputs)) != len(outputs) or inputs.intersection(outputs):
        raise ValueError("Output/report must be distinct from each other and every input")
    if args.output_wav.suffix.lower() != ".wav":
        raise ValueError("Output must have a .wav extension")
    if any(p.exists() for p in outputs) or not all(p.parent.is_dir() for p in outputs):
        raise ValueError("Choose new output/report files in existing directories; overwriting is disabled")
    require_production(cfg)
    engine = None
    sampler = GPUSampler()
    temporary = None
    pids = [os.getpid()]
    with args.report.open("x", encoding="utf8") as report_file:
        try:
            with offline_network() as network:
                report["offline"] = network
                report["timestamps"]["input_load_start"] = time.perf_counter()
                audio = load_asr_wav(args.input_wav, max_seconds=1200 if args.vad else 30, strict=True)
                report["timestamps"]["input_preprocessing_end"] = time.perf_counter()
                report["input"] = {"path":str(args.input_wav.resolve()), "sample_rate":audio.source_sample_rate,
                    "channels":audio.source_channels, "duration_seconds":audio.duration_seconds,
                    "internal_rate":16000, "sha256":hashlib.sha256(args.input_wav.read_bytes()).hexdigest(),
                    "strict_clipping_and_silence_validation":True}
                report["vad"] = {"enabled":args.vad, "energy_threshold":cfg.energy_threshold,
                    "incremental":cfg.incremental, "commit_silence_ms":cfg.commit_silence_ms,
                    "max_commit_ms":cfg.max_commit_ms, "min_silence_ms":cfg.min_silence_ms,
                    "max_segment_ms":cfg.max_segment_ms, "min_segment_ms":cfg.min_segment_ms,
                    "frame_ms":30, "timing":"Recording offsets, fed as fast as queue backpressure permits; not simulated microphone timing. Silence-only forced-flush tails are skipped."}
                sampler.start()
                engine = WavTranslationPipeline(cfg, capacity=args.queue_capacity, job_timeout=args.job_timeout)
                sampler.resident = True
                ready = engine.pipeline.tts.ready
                pids.append(ready["pid"])
                report["initialization"] = engine.initialization
                report["models"] = {"asr":ASR_MODEL, "asr_device":engine.pipeline.asr.device,
                    "asr_dtype":engine.pipeline.asr.dtype, "asr_attention":"sdpa", "translator":MT_MODEL,
                    "translator_device":"cpu", "tts_backend":"gptsovits", "tts_ready":ready,
                    "parent_pid":os.getpid(), "model_instances_built":1}
                report["static_reference"] = {"path":str(reference.resolve()), "language":args.reference_language,
                    "exact_transcript":text, "fingerprint":ready.get("reference_fingerprint"),
                    "sha256":hashlib.sha256(reference.read_bytes()).hexdigest(), "rolling_reference":False}
                if args.warmup:
                    t = time.perf_counter()
                    inference.run_sync(engine.pipeline.warmup)
                    report["excluded_warmup_seconds"] = time.perf_counter()-t
                with tempfile.NamedTemporaryFile(prefix="voicebridge-", suffix=".partial", dir=args.output_wav.parent, delete=False) as temp:
                    temporary = Path(temp.name)
                digest = hashlib.sha256()
                total = 0
                pending = deque()

                def consume(handle, writer):
                    nonlocal total
                    result = handle.result(timeout=args.job_timeout+1)
                    pcm = result.pop("pcm")
                    raw = pcm.astype("<i2", copy=False).tobytes()
                    if hashlib.sha256(raw).hexdigest() != result["pcm_sha256"]:
                        raise RuntimeError("Utterance PCM was modified before writing")
                    result["output_start_sample"] = total
                    writer.writeframesraw(raw)
                    digest.update(raw)
                    total += pcm.size
                    result["output_end_sample"] = total
                    result["timestamps"]["wav_append_complete"] = time.perf_counter()
                    result["request_phase"] = "warm" if args.warmup or report["utterances"] else "first_after_cold_initialization"
                    report["utterances"].append(result)
                    print(f"[{result['id']}] Chinese ASR: {result['source_text']}", flush=True)
                    print(f"[{result['id']}] Japanese translation: {result['japanese_text']}", flush=True)

                with wave.open(str(temporary), "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(24000)
                    for job in recorded_utterances(audio, cfg, vad=args.vad, repeat=args.repeat):
                        pending.append(engine.submit(job, block=True))
                        # Bound completed PCM futures too, not just waiting jobs.
                        if len(pending) >= args.queue_capacity+1:
                            consume(pending.popleft(), writer)
                    while pending:
                        consume(pending.popleft(), writer)
                report["timestamps"]["wav_finalized"] = time.perf_counter()
                if not total:
                    raise RuntimeError("No non-silent utterances were committed")
                report["output"] = validate_output(temporary, total, digest.hexdigest())
                # Same-directory hard link publishes exclusively: never overwrite a
                # destination created by somebody else during inference.
                os.link(temporary, args.output_wav)
                temporary.unlink()
                temporary = None
                report["output"]["path"] = str(args.output_wav.resolve())
                if playback_page:
                    from .preset_playback import write_player
                    ready=engine.pipeline.tts.ready if hasattr(engine,'pipeline') else report['models']['tts_ready']
                    write_player(playback_page,[args.output_wav],ready['voice_preset'])
                    report['preferred_playback']=dict(page=str(playback_page.resolve()),policy=ready['voice_preset']['playback'],
                        note='WAV remains native24k; approved perceived speed requires this browser player, not native1.25 or WSOLA')
                report["timestamps"]["output_validation_end"] = time.perf_counter()
                report["max_waiting_backlog"] = engine.max_backlog
                report["queue_capacity"] = args.queue_capacity
                report["status"] = "PASS"
        except BaseException as exc:
            report.update(status="FAIL", error={"type":type(exc).__name__, "message":str(exc)})
            raise
        finally:
            try:
                if engine:
                    sampler.resident = False
                    try:
                        engine.close()
                        report["worker_shutdown"] = "PASS"
                    except BaseException as exc:
                        report.update(status="FAIL", worker_shutdown="FAIL",
                                      cleanup_error={"type":type(exc).__name__, "message":str(exc)})
                        raise
            finally:
                sampler.resident = False
                sampler.stop()
                if temporary and temporary.exists():
                    temporary.unlink()
                report["gpu"] = sampler.report(pids)
                report["timestamps"]["pipeline_completion"] = time.perf_counter()
                report["end_to_end_seconds"] = report["timestamps"]["pipeline_completion"]-started
                if "initialization" in report:
                    report["initialization_seconds"] = report["initialization"]["seconds"]
                if "input_preprocessing_end" in report["timestamps"]:
                    report["input_loading_preprocessing_seconds"] = report["timestamps"]["input_preprocessing_end"]-report["timestamps"]["input_load_start"]
                if report["utterances"]:
                    rows = report["utterances"]
                    report["asr_process_overall_peak"] = {
                        key:max([report["initialization"]["asr_process_peak"][key]]+[row["asr_process_peak"][key] for row in rows])
                        for key in ("allocated_mib", "reserved_mib")}
                    report["processing_window_seconds"] = rows[-1]["timestamps"]["output_complete"]-rows[0]["timestamps"]["asr_start"]
                json.dump(report, report_file, ensure_ascii=False, indent=2)
                report_file.write("\n")
    print(f"Japanese WAV: {args.output_wav.resolve()}")
    print(f"Report: {args.report.resolve()}")
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        run(args)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"WAV translation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # Existing Marian's JSON reader uses the default process encoding. Preserve
    # both environments and re-enter this CLI in UTF-8 mode on native Windows.
    if os.name == "nt" and not sys.flags.utf8_mode:
        import subprocess
        raise SystemExit(subprocess.call([sys.executable, "-X", "utf8", "-m", "ort.translate_wav", *sys.argv[1:]]))
    raise SystemExit(main())
