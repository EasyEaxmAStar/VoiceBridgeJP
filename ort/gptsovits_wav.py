"""Standalone production GPT-SoVITS Japanese WAV synthesis and local metrics."""
from __future__ import annotations

import argparse
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from .config import Config
from .pipeline.base import build_pipeline
from .tts_controls import DEFAULT_SPEED


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-wav")
    parser.add_argument("--reference-text-file", type=Path)
    parser.add_argument("--reference-language", default="ja", choices=["ja", "en", "zh"])
    parser.add_argument("--text", action="append", required=True, help="Japanese text; repeat for ordered sentences")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--concurrent", type=int, default=1, help="Submit up to four sentences to the bounded queue")
    lexicon_group = parser.add_mutually_exclusive_group()
    lexicon_group.add_argument("--pronunciation-lexicon", type=Path, help="Override the production Japanese TTS lexicon")
    lexicon_group.add_argument("--no-pronunciation-lexicon",action="store_true",help="Disable TTS-only normalization")
    parser.add_argument("--frontend-diagnostics", action="store_true", help="Retain actual target phonemes in report (contains spoken text)")
    parser.add_argument("--frontend-mode", choices=["all_ja","ja"], default="all_ja")
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument('--voice-preset',choices=['PERSONAL_R06','PERSONAL_R17','PERSONAL_R18'])
    parser.add_argument('--prosody-level',choices=['ORIGINAL','SUBTLE','NATURAL','EXPRESSIVE'],default=None)
    parser.add_argument('--seed',help='Fixed integer or random; ORIGINAL defaults42, refinements default fresh per utterance')
    args = parser.parse_args(argv)
    backend = None
    try:
        defaults=Config()
        selected=args.voice_preset or defaults.gptsovits_voice_preset
        playback_page=args.output.with_suffix('.playback.html') if selected else None
        reference_wav=args.reference_wav or defaults.gptsovits_ref_audio
        reference_text=args.reference_text_file.read_text(encoding='utf-8-sig') if args.reference_text_file else defaults.gptsovits_ref_text
        if not 1 <= args.repeat <= 10 or not 1 <= args.concurrent <= 4:
            raise ValueError("repeat must be 1–10 and concurrent 1–4")
        if args.output.suffix.lower() != ".wav" or not args.output.parent.is_dir():
            raise ValueError("Choose a .wav output in an existing directory")
        texts = args.text * args.repeat
        outputs = [args.output if i == 0 else args.output.with_name(f"{args.output.stem}-{i+1:02d}.wav")
                   for i in range(len(texts))]
        targets = outputs + ([args.report] if args.report else [])
        if playback_page:targets.append(playback_page)
        protected = {Path(reference_wav).resolve()}
        if args.reference_text_file:protected.add(args.reference_text_file.resolve())
        if len({p.resolve() for p in targets}) != len(targets) or protected.intersection(p.resolve() for p in targets):
            raise ValueError("Outputs must be distinct from each other and the reference inputs")
        if any(p.exists() for p in targets):
            raise ValueError("Output/report already exists; choose new paths")
        cfg = Config(tts_backend="gptsovits", asr_backend="mock", translator_backend="mock",
                     gptsovits_ref_audio=reference_wav,
                     gptsovits_ref_text=reference_text,
                     gptsovits_ref_language=args.reference_language)
        cfg=replace(cfg,gptsovits_voice_preset=selected,
            gptsovits_prosody_level=args.prosody_level or defaults.gptsovits_prosody_level,
            gptsovits_seed=args.seed if args.seed is not None else defaults.gptsovits_seed)
        if args.pronunciation_lexicon:
            cfg = replace(cfg,gptsovits_pronunciation_lexicon=str(args.pronunciation_lexicon))
        if args.no_pronunciation_lexicon:
            cfg = replace(cfg,gptsovits_pronunciation_lexicon="")
        if args.frontend_diagnostics:
            cfg = replace(cfg,gptsovits_frontend_diagnostics=True)
        pipeline = build_pipeline(cfg)
        backend = pipeline.tts

        def run(item):
            index, text = item
            started = time.perf_counter()
            speed=args.speed if args.speed is not None else (1.0 if selected else DEFAULT_SPEED)
            chunks = list(backend.synthesize_stream(text, "ja",options={"speed_factor":speed,
                "pronunciation_enabled":not args.no_pronunciation_lexicon},frontend_mode=args.frontend_mode))
            elapsed = time.perf_counter() - started
            if not chunks:
                raise RuntimeError("Synthesis returned no PCM")
            samples = np.concatenate([x.samples for x in chunks])
            rate = chunks[0].sample_rate
            if not np.isfinite(samples).all() or np.sqrt(np.mean(samples.astype(np.float64)**2)) < 1e-4:
                raise RuntimeError("Synthesis returned silent or invalid audio")
            with outputs[index].open("xb") as stream:
                sf.write(stream, samples, rate, format="WAV", subtype="PCM_16")
            return {"index": index, "text": text, "seconds": elapsed, "audio_seconds": samples.size / rate,
                    "rtf": elapsed / (samples.size / rate), "chunks": len(chunks), "sample_rate": rate,
                    "rms": float(np.sqrt(np.mean(samples.astype(np.float64)**2))),
                    "output": str(outputs[index].resolve())}

        with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
            rows = list(executor.map(run, enumerate(texts)))
        report = {"backend": cfg.tts_backend, "ready": backend.ready, "outputs": rows,
                  "requests": list(backend.completed_metrics), "max_queue_backlog": backend.max_backlog}
        if selected:
            from .preset_playback import write_player
            write_player(playback_page,outputs,backend.ready['voice_preset'])
            report.update(playback_page=str(playback_page.resolve()),playback_policy=backend.ready['voice_preset']['playback'],
                stored_wav_speed='native1.00; preferred speed occurs only in approved browser player; no additional time stretching')
        if args.report:
            with args.report.open("x", encoding="utf8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"GPT-SoVITS error: {exc}\n")
    finally:
        if backend is not None:
            backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
