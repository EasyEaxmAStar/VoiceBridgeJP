"""Reanalyze retained physical PCM/clock evidence without loading models."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import soundfile as sf
from .live_loopback import LoopbackRecorder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError("Choose a new output path")
    data=json.loads(args.report.read_text(encoding="utf8"))
    for i,run in enumerate(data.get("runs",[data])):
        old=run["independent_loopback"]
        if not old.get("expected_pcm_artifact") or not old.get("clock_trace"):
            raise ValueError("Original report lacks retained expected PCM/per-block clocks; cannot reconstruct evidence")
        audio,rate=sf.read(old["wav"],dtype="float32",always_2d=True)
        recorder=LoopbackRecorder(old["device"],api=None,devices=lambda:[])
        recorder.rate,recorder.channels=rate,audio.shape[1]
        recorder.device=SimpleNamespace(name=old["device"])
        recorder.origin,recorder.original_adc=old["clock_origin_at"],old["original_adc_clock_start"]
        recorder.blocks=[audio]
        recorder.clock_trace=old["clock_trace"]
        recorder.recording_latency=old.get("recording_latency_seconds")
        with np.load(old["expected_pcm_artifact"],allow_pickle=False) as expected:
            recorder.expected={int(k):[expected[k]] for k in expected.files}
        rows={row["id"]:row for row in run["traces"]}
        stopped=next((x["stop_returned_at"] for x in reversed(run["lifecycle"]) if x["state"]=="STOPPED"),None)
        run["independent_loopback"]=recorder.analyze(rows,args.output.with_name(f"{args.output.stem}-{i}.wav"),stop_returned_at=stopped)
        for row in rows.values():
            stamp=row.get("loopback_first_non_silent_at")
            row["reviewed_speech_end_to_loopback_seconds"]=stamp-row["context"]["speech_end_at"] if stamp is not None else None
        # Old summary was computed with the former timing algorithm.
        run["latencies_seconds"].pop("speech_end_to_loopback",None)
        values=[x["reviewed_speech_end_to_loopback_seconds"] for x in rows.values() if x["reviewed_speech_end_to_loopback_seconds"] is not None]
        run["reviewed_hardware_latency_seconds"]={"n":len(values),"p50":float(np.percentile(values,50)) if values else None,"p95":float(np.percentile(values,95)) if values else None,"min":min(values) if values else None,"max":max(values) if values else None}
    data["reanalysis_source"]=str(args.report.resolve())
    with args.output.open("x",encoding="utf8") as handle: json.dump(data,handle,ensure_ascii=False,indent=2)


if __name__=="__main__": main()
