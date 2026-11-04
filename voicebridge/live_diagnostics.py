"""Paced recorded-input verification of the real live client/local model service.

Recording replay and software callbacks are explicitly diagnostics, never a
production capture/model fallback. --hardware-output opens only CABLE endpoints.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from ort.asr_audio import load_asr_wav
from ort.audio import resample
from ort.gpu_sampling import GPUSampler
from ort.pipeline.gptsovits_reference import validate_reference
from .audio_devices import classify_device,enumerate_devices
from .live import LiveBridge,LiveConfig,State
from .server_process import ManagedServer
from .live_loopback import LoopbackRecorder


class ReplayInput:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.active = False
    def start(self): self.active = True
    def abort(self): self.active = False
    def close(self): self.active = False


class SoftwareOutput:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.active = False
        self.latency = 0
        self.task = None
    def start(self):
        self.active = True
        self.loop = asyncio.get_running_loop()
        async def pump():
            frames = self.kwargs["blocksize"]
            period = frames/self.kwargs["samplerate"]
            origin = time.perf_counter()
            sequence = 0
            while self.active:
                output = np.zeros((frames,self.kwargs["channels"]),dtype=np.float32)
                now = time.perf_counter()
                self.kwargs["callback"](output,frames,SimpleNamespace(currentTime=now,outputBufferDacTime=now),None)
                sequence += 1
                while time.perf_counter()<origin+sequence*period:
                    await asyncio.sleep(min(.005,origin+sequence*period-time.perf_counter()))
        self.task = asyncio.create_task(pump())
    def abort(self):
        self.active = False
        if self.task:
            self.loop.call_soon_threadsafe(self.task.cancel)
    def close(self): self.abort()


class ReplayAPI:
    def __init__(self, hardware=False):
        self.input = None
        self.hardware = hardware
        if hardware:
            import sounddevice
            self.sd = sounddevice
    def check_input_settings(self, **kwargs): pass
    def check_output_settings(self, **kwargs):
        if self.hardware: self.sd.check_output_settings(**kwargs)
    def InputStream(self, **kwargs):
        self.input = ReplayInput(**kwargs)
        return self.input
    def OutputStream(self, **kwargs):
        return self.sd.OutputStream(**kwargs) if self.hardware else SoftwareOutput(**kwargs)


def replay_devices():
    return [classify_device(i,{"name":name,"max_input_channels":ins,"max_output_channels":outs,
                              "default_samplerate":48000},"Windows WASAPI")
            for i,name,ins,outs in [(21,"Microphone (recorded replay)",1,0),(16,"CABLE Input (VB-Audio Virtual Cable)",0,2)]]


async def feed_recording(bridge, api, samples16k, *, repeats=1):
    # A trailing one-second pause closes the final natural utterance. This is
    # actual wall-clock pacing; sample timestamps are never placed in the future
    # to manufacture favorable latency measurements.
    source = np.concatenate([resample(samples16k,16000,bridge.input_rate),np.zeros(bridge.input_rate,dtype=np.float32)])
    frames = bridge.input_frames
    origin = time.perf_counter()
    position = 0
    for _ in range(repeats):
        for offset in range(0,len(source),frames):
            if bridge.state != State.RUNNING:
                return
            block = np.zeros((frames,bridge.config.input_channels),dtype=np.float32)
            part = source[offset:offset+frames]
            block[:len(part)] = part[:,None]
            target = origin+position/bridge.input_rate
            while time.perf_counter()<target:
                await asyncio.sleep(min(.005,target-time.perf_counter()))
            now = time.perf_counter()
            api.input.callback(block,frames,SimpleNamespace(inputBufferAdcTime=target,currentTime=now),None)
            position += frames
    # Drain the Soxr lookahead/VAD with genuine paced silence, not a forced cut.
    for _ in range(30):
        if bridge.state != State.RUNNING: return
        target = origin+position/bridge.input_rate
        while time.perf_counter()<target: await asyncio.sleep(min(.005,target-time.perf_counter()))
        now = time.perf_counter()
        api.input.callback(np.zeros((frames,bridge.config.input_channels),dtype=np.float32),frames,
            SimpleNamespace(inputBufferAdcTime=target,currentTime=now),None)
        position += frames


async def run(args):
    if args.report.exists() or not args.report.parent.is_dir():
        raise ValueError("Choose a new report path in an existing directory")
    if not 1 <= args.repeat <= 10 or len(set(args.silences))!=len(args.silences) or any(x not in range(300,2001) for x in args.silences):
        raise ValueError("Use 1..10 repeats and distinct silence thresholds in 300..2000 ms")
    if args.hardware_output:
        for silence in args.silences:
            if args.report.with_name(f"{args.report.stem}-{silence}-loopback.wav").exists():
                raise ValueError("Choose new loopback output names")
    audio = load_asr_wav(args.input_wav,strict=True)
    text = args.reference_text_file.read_text(encoding="utf-8-sig")
    reference = validate_reference(str(args.reference_wav),text,"ja")
    server = ManagedServer(reference,port=args.port)
    sampler = GPUSampler().start()
    reports = []
    failure = None
    try:
        await server.start()
        sampler.resident = True
        for silence in args.silences:
            api = ReplayAPI(args.hardware_output)
            observer = None
            cfg = LiveConfig(args.input_device,args.output_device,reference.path,reference.text,url=server.url,
                             tts_speed=getattr(args,"tts_speed",1.2),fragment_interval=getattr(args,"fragment_interval",.3),
                             trim_digital_silence=getattr(args,"trim_digital_silence",False),
                             mode="full-sentence" if silence==700 else "low-latency",silence_ms=silence)
            bridge = LiveBridge(cfg,api=api,devices=enumerate_devices if args.hardware_output else replay_devices,
                                translated_audio_observer=None)
            feed = None
            stop_returned = None
            try:
                if args.hardware_output:
                    observer = LoopbackRecorder(args.loopback_device,api=api.sd,devices=enumerate_devices)
                    await observer.start()
                    bridge.translated_audio_observer = observer.observe_translated
                await bridge.start()
                print(f"REAL MODELS / RECORDED REPLAY: silence={silence}ms, repetitions={args.repeat}",flush=True)
                feed = asyncio.create_task(feed_recording(bridge,api,audio.samples,repeats=args.repeat))
                if args.test_stop:
                    deadline = time.monotonic()+60
                    while not any("first_output_callback_at" in x for x in bridge.traces.values()):
                        if bridge.state==State.ERROR: raise RuntimeError(bridge.last_error)
                        if time.monotonic()>deadline: raise TimeoutError("No first playback callback")
                        await asyncio.sleep(.01)
                    await asyncio.sleep(.6)
                    await bridge.stop()
                    stop_returned = time.perf_counter()
                    await asyncio.sleep(.5)
                else:
                    await feed
                    stop_returned = None
                deadline = time.monotonic()+120
                while not args.test_stop and time.monotonic()<deadline:
                    if bridge.state==State.ERROR:
                        raise RuntimeError(bridge.last_error)
                    if bridge.traces and all(row["status"]=="played" for row in bridge.traces.values()):
                        break
                    await asyncio.sleep(.02)
                if not args.test_stop and (not bridge.traces or not all(row["status"]=="played" for row in bridge.traces.values())):
                    raise TimeoutError("Translated playback did not finish")
            finally:
                if feed and not feed.done():
                    feed.cancel()
                    await asyncio.gather(feed,return_exceptions=True)
                if bridge.state != State.STOPPED:
                    await bridge.stop()
                    stop_returned = time.perf_counter()
                if observer:
                    await asyncio.sleep(.5)
                    await observer.stop()
            if observer:
                loopback = observer.analyze(bridge.traces,args.report.with_name(f"{args.report.stem}-{silence}-loopback.wav"),
                                           stop_returned_at=stop_returned)
            else:
                loopback = None
            report = bridge.report()
            report.update(input_kind="paced saved-WAV replay, not physical microphone",
                          output_kind="real WASAPI CABLE Input" if args.hardware_output else "paced software callback, not hardware")
            report["independent_loopback"] = loopback
            report["forced_stop_test"] = args.test_stop
            reports.append(report)
            print(json.dumps({"silence_ms":silence,"utterances":report["total_utterances"],
                "counts":report["counts"],"latencies_seconds":report["latencies_seconds"]},ensure_ascii=False),flush=True)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        sampler.resident = False
        server.close()
        sampler.stop()
        result = {"status":"FAIL" if failure else "COMPLETED", "failure":failure,
                  "runs":reports,"gpu":sampler.report(),"model_server_log":str(getattr(server,"log_path","")),
                  "cold_server_startup_seconds":getattr(server,"startup_seconds",None),
                  "hardware_microphone_acceptance":"PENDING", "no_alternative_models":True}
        with args.report.open("x",encoding="utf8") as handle:
            json.dump(result,handle,ensure_ascii=False,indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-wav",type=Path,required=True)
    parser.add_argument("--reference-wav",type=Path,required=True)
    parser.add_argument("--reference-text-file",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--silences",type=int,nargs="+",default=[300,450,700])
    parser.add_argument("--repeat",type=int,default=1)
    parser.add_argument("--port",type=int,default=8765)
    parser.add_argument("--hardware-output",action="store_true")
    parser.add_argument("--tts-speed",type=float,default=1.2)
    parser.add_argument("--fragment-interval",type=float,default=.3)
    parser.add_argument("--trim-digital-silence",action="store_true")
    parser.add_argument("--loopback-device",default="CABLE Output (VB-Audio Virtual Cable)")
    parser.add_argument("--test-stop",action="store_true",help="stop during first Japanese playback; keep recording the cable tail")
    parser.add_argument("--input-device",default="21",help="physical device identity used only for selection; audio source remains replay")
    parser.add_argument("--output-device",default="CABLE Input (VB-Audio Virtual Cable)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__": main()
