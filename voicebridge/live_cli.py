"""Interactive native live CLI; no GUI and no alternative inference backends."""
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import sys
import time

from ort.gpu_sampling import GPUSampler
from ort.tts_controls import DEFAULT_SPEED
from ort.pipeline.gptsovits_reference import validate_reference
from ort.config import Config
from .live import LiveBridge,LiveConfig,State
from .server_process import ManagedServer
from .live_loopback import LoopbackRecorder


def add_arguments(sub):
    parser = sub.add_parser("live",help="real continuous WASAPI Mandarin -> Japanese CABLE Input")
    parser.add_argument("--input-device",required=True,help="exact WASAPI microphone name or freshly enumerated index")
    parser.add_argument("--output-device",required=True,help="exact WASAPI CABLE Input name or current index")
    parser.add_argument("--reference-wav",type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--reference-text-file",type=Path)
    group.add_argument("--reference-text")
    parser.add_argument("--reference-language",choices=["ja"],default="ja")
    parser.add_argument("--source-language",choices=["zh","ja","en","auto"],default="zh")
    parser.add_argument("--tts-speed",type=float,default=DEFAULT_SPEED)
    parser.add_argument("--no-pronunciation-lexicon",action="store_false",dest="pronunciation_enabled",default=True)
    parser.add_argument("--fragment-interval",type=float,default=.3)
    parser.add_argument("--trim-digital-silence",action="store_true")
    parser.add_argument("--adaptive-speed",action="store_true")
    parser.add_argument("--mode",choices=["low-latency","full-sentence"],default="full-sentence")
    parser.add_argument("--silence-ms",type=int,help="candidate 300/450 or baseline 700; 300-2000 supported")
    parser.add_argument("--max-utterance-seconds",type=float,default=30)
    parser.add_argument("--capture-seconds",type=float,default=2)
    parser.add_argument("--playback-seconds",type=float,default=8)
    parser.add_argument("--queue-capacity",type=int,default=4)
    parser.add_argument("--input-rate",type=int)
    parser.add_argument("--output-rate",type=int)
    parser.add_argument("--input-channels",type=int,default=1)
    parser.add_argument("--connect",help="connect to an existing verified localhost server; otherwise own/start one")
    parser.add_argument("--port",type=int,default=8765)
    parser.add_argument("--duration",type=float,help="seconds after RUNNING; otherwise interactive commands")
    parser.add_argument("--metrics",type=Path)
    parser.add_argument("--loopback-device",help="optional explicit CABLE Output recording endpoint")
    parser.add_argument("--loopback-wav",type=Path,help="independent cable verification WAV; maximum 180 seconds")


async def run_live(args):
    if os.name != "nt":
        raise RuntimeError("This application requires native Windows WASAPI")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration<=0):
        raise ValueError("Duration must be positive finite seconds")
    defaults=Config()
    text = args.reference_text if args.reference_text is not None else (args.reference_text_file.read_text(encoding="utf-8-sig") if args.reference_text_file else defaults.gptsovits_ref_text)
    reference = validate_reference(str(args.reference_wav or defaults.gptsovits_ref_audio),text,args.reference_language)
    if args.metrics and (args.metrics.exists() or not args.metrics.parent.is_dir()):
        raise ValueError("Choose a new metrics path in an existing directory")
    if bool(args.loopback_device) != bool(args.loopback_wav):
        raise ValueError("Supply --loopback-device and --loopback-wav together")
    if args.loopback_wav and (args.loopback_wav.exists() or not args.loopback_wav.parent.is_dir()):
        raise ValueError("Choose a new loopback WAV path")
    if args.loopback_wav and args.metrics and args.loopback_wav.resolve()==args.metrics.resolve():
        raise ValueError("Metrics and loopback WAV must have different paths")
    if not sys.stdin.isatty() and args.duration is None:
        raise ValueError("Interactive console is unavailable; use --duration for a bounded hardware test")
    server = None if args.connect else ManagedServer(reference,port=args.port)
    url = args.connect or server.url
    config = LiveConfig(args.input_device,args.output_device,reference.path,reference.text,
        reference_language=args.reference_language,url=url,mode=args.mode,silence_ms=args.silence_ms,
        source_language=args.source_language,tts_speed=args.tts_speed,fragment_interval=args.fragment_interval,
        trim_digital_silence=args.trim_digital_silence,adaptive_speed=args.adaptive_speed,
        pronunciation_enabled=args.pronunciation_enabled,
        max_utterance_seconds=args.max_utterance_seconds,capture_seconds=args.capture_seconds,
        playback_seconds=args.playback_seconds,utterance_capacity=args.queue_capacity,
        input_rate=args.input_rate,output_rate=args.output_rate,input_channels=args.input_channels)
    bridge = LiveBridge(config,server=server)
    sampler = GPUSampler().start()
    print("STARTING: checking endpoints and loading the existing offline production models.",flush=True)
    command = ""
    last_state = None
    failed = False
    interrupted = False
    failure = None
    observer = None
    try:
        await bridge.start()
        if args.loopback_device:
            observer = LoopbackRecorder(args.loopback_device,api=bridge.api,devices=bridge.devices)
            await observer.start()
            bridge.translated_audio_observer = observer.observe_translated
        sampler.resident = True
        print(json.dumps(bridge.device_info,ensure_ascii=False,indent=2))
        print('Japanese pronunciation dictionary:',json.dumps(bridge.created.get('session',{}).get('metadata',{}).get('pronunciation_lexicon'),ensure_ascii=False),flush=True)
        print("RUNNING — synthesized Japanese only -> CABLE Input. Qwen ASR recognizes final utterances; it is not streaming ASR.",flush=True)
        print("Commands: pause, resume, stop, start, status, quit. Ctrl+C stops and flushes.",flush=True)
        deadline = time.monotonic()+args.duration if args.duration is not None else None
        while deadline is None or time.monotonic()<deadline:
            if bridge.state==State.ERROR and last_state != State.ERROR:
                failed = True
                print(f"ERROR (muted): {bridge.last_error}. Type start to explicitly restart; quit to exit.",flush=True)
                if args.duration is not None:
                    break
            last_state = bridge.state
            for identifier,row in list(bridge.traces.items()):
                if not row.get("console_reported") and row.get("japanese_text"):
                    print(f"[{identifier}] 中文: {row['source_text']}\n[{identifier}] 日本語: {row['japanese_text']}",flush=True)
                    row["console_reported"] = True
            if sys.stdin.isatty():
                import msvcrt
                while msvcrt.kbhit():
                    char = msvcrt.getwch()
                    if char=="\x03":
                        raise KeyboardInterrupt
                    if char in {"\r","\n"}:
                        value,command = command.strip().casefold(),""
                        print()
                        if value in {"quit","exit","q"}:
                            return 1 if failed else 0
                        if value=="pause": await bridge.pause()
                        elif value=="resume": await bridge.resume()
                        elif value=="stop":
                            sampler.resident = False
                            await bridge.stop()
                        elif value=="start":
                            await bridge.start()
                            sampler.resident = True
                        elif value=="status":
                            report = bridge.report()
                            print(json.dumps({k:report[k] for k in ("state","last_error","counts","maxima","latencies_seconds")},ensure_ascii=False,indent=2))
                        elif value:
                            print("Use pause, resume, stop, start, status or quit.")
                        print(bridge.state.value,flush=True)
                    elif char=="\b":
                        command = command[:-1]
                    elif char.isprintable():
                        command = (command+char)[-256:]
                        print(char,end="",flush=True)
            await asyncio.sleep(.03)
        return 1 if failed else 0
    except (KeyboardInterrupt,asyncio.CancelledError):
        interrupted = True
        return 0
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sampler.resident = False
        await bridge.stop()
        stop_returned = time.perf_counter()
        loopback = None
        try:
            if observer:
                await asyncio.sleep(.5)
                await observer.stop()
                loopback = observer.analyze(bridge.traces,args.loopback_wav,stop_returned_at=stop_returned)
        finally:
            sampler.stop()
        report = bridge.report()
        report["independent_loopback"] = loopback
        report["input_kind"] = "physical Windows WASAPI microphone"
        report["gpu"] = sampler.report()
        report["interrupted"] = interrupted
        report["command_error"] = failure
        report["result"] = "FAIL" if bridge.last_error or failure or failed else "PENDING HARDWARE VERIFICATION"
        if args.metrics:
            with args.metrics.open("x",encoding="utf8") as handle:
                json.dump(report,handle,ensure_ascii=False,indent=2)
        print(f"{bridge.state.value}: output muted, buffers invalidated, owned server closed.",flush=True)
