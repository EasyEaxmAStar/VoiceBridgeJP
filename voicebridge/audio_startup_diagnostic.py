"""Isolated native stream-start matrix. Silence only; no models or forwarding."""
import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import time


def probe(case):
    # Threaded initialization reproduces LiveBridge's lazy preflight import.
    async def initialize():
        def load():
            import sounddevice as sd
            sd.query_hostapis()
            sd.query_devices()
            return sd
        if case.startswith("main-threadquery"):
            load()
            return await asyncio.to_thread(load)
        return await asyncio.to_thread(load) if case.startswith("thread-") else load()
    sd = asyncio.run(initialize())
    import numpy as np
    result = {"case":case,"sounddevice":sd.__version__,"portaudio":sd.get_portaudio_version(),
              "hostapis":list(sd.query_hostapis()),"devices":{},"steps":[]}
    streams = []
    try:
        for index,direction in ((21,"input"),(16,"output"),(20,"input")):
            device = dict(sd.query_devices(index))
            assert "WASAPI" in sd.query_hostapis(device["hostapi"])["name"], device
            assert device[f"max_{direction}_channels"]>0
            result["devices"][index]=device
        callback = "blocking" not in case
        latency = "low" if "low" in case else "high"
        blocksize = 480 if "fixed" in case else 0
        def out(data,frames,info,status): data.fill(0)
        def inp(data,frames,info,status): pass
        kwargs = dict(device=16,samplerate=48000,channels=2,dtype="float32",latency=latency,blocksize=blocksize)
        if callback: kwargs["callback"]=out
        result["output_options"]={**kwargs,"callback":callback,"extra_settings":None}
        sd.check_output_settings(device=16,samplerate=48000,channels=2,dtype="float32")
        output = sd.OutputStream(**kwargs)
        streams.append(output)
        result["steps"].append("output constructed")
        def input_stream(index,channels):
            stream = sd.InputStream(device=index,samplerate=48000,channels=channels,dtype="float32",
                latency=latency,blocksize=480,callback=inp)
            streams.append(stream)
            result["steps"].append(f"input {index} constructed")
            return stream
        mic = input_stream(21,1) if "mic" in case or "production" in case else None
        loop = input_stream(20,2) if "loop" in case else None
        if "loop-first" in case:
            loop.start()
            result["steps"].append("loop started first")
        output.start()
        result["steps"].append("output started")
        if mic:
            mic.start()
            result["steps"].append("mic started")
        if loop and "loop-first" not in case:
            loop.start()
            result["steps"].append("loop started")
        if not callback: output.write(np.zeros((4800,2),dtype=np.float32))
        time.sleep(.2)
        result["status"]="PASS"
    except Exception as exc:
        result.update(status="FAIL",error_type=type(exc).__name__,error=str(exc),native_error_args=exc.args)
    finally:
        result["cleanup_errors"]=[]
        for stream in reversed(streams):
            for operation in ("abort","close"):
                try: getattr(stream,operation)()
                except Exception as exc: result["cleanup_errors"].append(repr(exc))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe")
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    if args.probe:
        print(json.dumps(probe(args.probe),ensure_ascii=False,default=str))
        return
    if not args.report or args.report.exists(): parser.error("Supply a new --report path")
    cases=["main-blocking-high","main-callback-high","main-callback-high-fixed",
           "main-callback-low-fixed","main-mic-high-fixed","main-mic-loop-high-fixed",
           "main-production-low-fixed","main-production-loop-first-low-fixed",
           "thread-production-low-fixed","thread-production-high-fixed",
           "main-threadquery-production-low-fixed"]
    results=[]
    for case in cases:
        try:
            run=subprocess.run([sys.executable,"-X","utf8","-m",__spec__.name,"--probe",case],
                capture_output=True,text=True,encoding="utf8",timeout=20)
            row=json.loads(run.stdout) if run.returncode==0 else {"case":case,"status":"FAIL","stderr":run.stderr,"returncode":run.returncode}
        except subprocess.TimeoutExpired:
            row={"case":case,"status":"TIMEOUT"}
        results.append(row)
        print(case,row["status"],flush=True)
    args.report.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf8")


if __name__=="__main__": main()
