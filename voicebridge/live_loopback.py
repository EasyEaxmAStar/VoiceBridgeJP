"""Independent CABLE Output measurement; never feeds the recognition pipeline."""
import asyncio
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from .live import select_device
from .live_buffers import CaptureRing


class LoopbackRecorder:
    def __init__(self, device, *, api, devices, max_seconds=180):
        self.api,self.devices,self.selection = api,devices,device
        self.max_seconds = max_seconds
        self.expected = {}
        self.expected_samples = 0
        self.blocks = []
        self._stream = None
        self._task = None
        self.origin = None
        self.error = None
        self.window_complete = False
        self.clock_trace = []

    async def start(self):
        device = select_device(self.devices(),self.selection,"loopback")
        self.device = device
        self.rate = int(device.default_samplerate)
        self.channels = min(2,device.max_input_channels)
        self.frames = round(self.rate*.01)
        self.ring = CaptureRing(200,self.frames,self.channels,self.rate)
        self.api.check_input_settings(device=device.index,samplerate=self.rate,channels=self.channels,dtype="float32")
        self._stream = self.api.InputStream(device=device.index,samplerate=self.rate,channels=self.channels,
            blocksize=self.frames,dtype="float32",latency="low",callback=self.ring.callback)
        try:
            self._stream.start()
            self.recording_latency = float(self._stream.latency)
        except BaseException:
            self._stream.close()
            self._stream = None
            raise
        async def drain():
            while self._stream is not None:
                if self.ring.fault:
                    self.error = self.ring.fault
                    return
                item = self.ring.pop()
                if item is None:
                    await asyncio.sleep(.002)
                    continue
                samples,adc,position,raw_adc = item
                if self.origin is None:
                    self.origin = adc-position/self.rate
                    self.original_adc = raw_adc
                if position+len(samples) > self.max_seconds*self.rate:
                    self.window_complete = True
                    await asyncio.to_thread(self._stream.abort)
                    await asyncio.to_thread(self._stream.close)
                    self._stream = None
                    return
                self.blocks.append(samples)
                self.clock_trace.append({"sample":position,"adc_mapped_at":adc,"raw_adc":raw_adc,
                    "drained_at":time.perf_counter(),"mapping_offset":adc-raw_adc if raw_adc is not None else None})
        self._task = asyncio.create_task(drain())

    def observe_translated(self, identifier, samples, rate):
        if self.window_complete:
            return  # A finite measurement window must not interrupt production.
        if self.expected_samples+samples.size > self.max_seconds*self.rate:
            self.window_complete = True
            return
        if rate != self.rate:
            raise RuntimeError("Loopback/reference device rates differ")
        self.expected_samples += samples.size
        self.expected.setdefault(identifier,[]).append(samples)

    async def stop(self):
        if self._stream:
            await asyncio.wait_for(asyncio.to_thread(self._stream.abort),3)
            await asyncio.wait_for(asyncio.to_thread(self._stream.close),3)
            self._stream = None
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task,return_exceptions=True)

    def analyze(self, traces, wav_path, *, stop_returned_at=None):
        from scipy.signal import correlate
        if self.error or not self.blocks or self.origin is None:
            return {"status":"FAIL","error":self.error or "No independent loopback samples"}
        path = Path(wav_path)
        if path.exists():
            raise ValueError("Loopback WAV already exists")
        expected_path = path.with_suffix(".expected.npz")
        if expected_path.exists():
            raise ValueError("Expected PCM artifact already exists")
        with expected_path.open("xb") as handle:
            np.savez(handle,**{str(k):np.concatenate(v) for k,v in self.expected.items()})
        recorded = np.concatenate(self.blocks)
        with path.open("xb") as handle:
            sf.write(handle,recorded,self.rate,format="WAV",subtype="PCM_16")
        info = sf.info(path)
        if (info.samplerate,info.channels,info.frames,info.subtype)!=(self.rate,self.channels,len(recorded),"PCM_16"):
            raise RuntimeError("Independent recording WAV read-back failed")
        mono = recorded.mean(axis=1,dtype=np.float32)
        rms = float(np.sqrt(np.mean(mono.astype(np.float64)**2)))
        def sample_at(timestamp):
            if not self.clock_trace:
                return round((timestamp-self.origin)*self.rate)
            clock = min(self.clock_trace,key=lambda x:abs(x["adc_mapped_at"]-timestamp))
            return clock["sample"]+round((timestamp-clock["adc_mapped_at"])*self.rate)
        def time_at(sample):
            if not self.clock_trace:
                return self.origin+sample/self.rate
            index = np.searchsorted([x["sample"] for x in self.clock_trace],sample,side="right")-1
            clock = self.clock_trace[max(0,int(index))]
            return clock["adc_mapped_at"]+(sample-clock["sample"])/self.rate
        residual = mono.copy()
        matches = []
        for row in traces.values():
            row["loopback_first_non_silent_at"] = None
            row["physical_playback_completion_at"] = None
        for identifier,pieces in self.expected.items():
            row = traces.get(identifier)
            if not row or "first_output_callback_at" not in row:
                matches.append({"utterance_id":identifier,"matched":False,"reason":"No output callback timestamp"})
                continue
            expected = np.concatenate(pieces)
            nonzero = np.flatnonzero(np.abs(expected)>1e-4)
            if not len(nonzero):
                matches.append({"utterance_id":identifier,"matched":False,"reason":"Expected output is silent"})
                continue
            first = int(nonzero[0])
            template = expected[first:first+int(.4*self.rate)]
            if len(template)<int(.1*self.rate):
                matches.append({"utterance_id":identifier,"matched":False,"reason":"Insufficient template duration"})
                continue
            approximate = sample_at(row["first_output_callback_at"])+first
            low = max(0,approximate-int(1*self.rate))
            high = min(len(mono),approximate+int(1*self.rate)+len(template))
            search = mono[low:high]
            if len(search)<len(template):
                matches.append({"utterance_id":identifier,"matched":False,"reason":"Insufficient independent recording"})
                continue
            # Float32 FFT residuals divided by near-zero energy previously
            # yielded impossible scores in silent windows after Stop.
            correlation = correlate(search.astype(np.float64),template.astype(np.float64),mode="valid",method="fft")
            energy = search.astype(np.float64)**2
            cumulative = np.r_[0,np.cumsum(energy)]
            window_energy = np.maximum(cumulative[len(template):]-cumulative[:-len(template)],0)
            norms = np.sqrt(window_energy*np.sum(template.astype(np.float64)**2))
            valid = (window_energy>len(template)*1e-8) & (norms>1e-12)
            score = np.full(len(correlation),-1.,dtype=np.float64)
            np.divide(correlation,norms,out=score,where=valid)
            score[np.abs(score)>1+1e-6] = -1
            best = int(np.argmax(score))
            strength = float(score[best])
            observed = low+best
            start = observed-first
            matched = strength >= .95 and start >= 0
            match = {"utterance_id":identifier,"normalized_correlation":strength,"matched":matched,
                     "expected_first_non_silent_sample":first,"expected_initial_silence_seconds":first/self.rate,
                     "matched_recording_sample":observed,"search_low_sample":low,"search_high_sample":high,
                     "legacy_first_block_time_at":self.origin+observed/self.rate,
                     "first_non_silent_at":time_at(observed) if matched else None}
            if matched:
                row["loopback_first_non_silent_at"] = match["first_non_silent_at"]
                count = min(len(expected),len(mono)-start)
                actual = mono[start:start+count]
                original = expected[:count]
                gain = float(np.dot(actual.astype(np.float64),original)/max(1e-12,np.dot(original.astype(np.float64),original)))
                residual[start:start+count] -= original*gain
                complete = count==len(expected) and row.get("status")=="played"
                match.update(gain=gain,matched_samples=count,output_start_sample=start,complete=complete)
                match["callback_to_matched_seconds"] = match["first_non_silent_at"]-row["first_output_callback_at"]
                match["after_initial_silence_seconds"] = match["callback_to_matched_seconds"]-first/self.rate
                detected = np.flatnonzero(np.abs(actual)>1e-4)
                match["first_detected_recording_sample"] = start+int(detected[0]) if len(detected) else None
                match["first_detected_at"] = time_at(start+int(detected[0])) if len(detected) else None
                if complete:
                    row["physical_playback_completion_at"] = time_at(start+count)
            matches.append(match)
        unexplained = float(np.sqrt(np.mean(residual.astype(np.float64)**2)))
        signal = np.flatnonzero(np.abs(mono)>1e-4)
        clock_errors = [x["adc_mapped_at"]-x["sample"]/self.rate-self.origin for x in self.clock_trace]
        post_stop = None
        if stop_returned_at is not None:
            start = max(0,sample_at(stop_returned_at))
            tail = mono[start:]
            post_stop = {"observed_seconds":len(tail)/self.rate,
                "rms":float(np.sqrt(np.mean(tail.astype(np.float64)**2))) if len(tail) else None,
                "peak":float(np.max(np.abs(tail))) if len(tail) else None}
        matching = bool(matches) and all(x["matched"] for x in matches) and rms>1e-4
        stopped_cleanly = (post_stop is not None and post_stop["observed_seconds"]>=.25
                           and post_stop["peak"] is not None and post_stop["peak"]<1e-4)
        return {"status":"PASS" if matching else "FAIL", "scope":"Japanese template arrival only; inspect full-output integrity and stop-tail results separately",
            "stop_tail_status":"PASS" if stopped_cleanly else "FAIL" if post_stop else "NOT_MEASURED",
            "full_output_integrity":"PASS" if matching and all(x.get("complete") for x in matches) and unexplained<max(1e-4,rms*.01) else "NOT_ESTABLISHED",
            "measurement_window_complete":self.window_complete,"maximum_recording_seconds":self.max_seconds,
            "device":self.device.name,"sample_rate":self.rate,"channels":self.channels,
            "duration_seconds":len(recorded)/self.rate,"rms":rms,"wav":str(path.resolve()),
            "format":"WAV PCM_16","clock_origin_at":self.origin,"original_adc_clock_start":self.original_adc,
            "matches":matches,"unexplained_signal_rms":unexplained,"post_stop":post_stop,
            "expected_pcm_artifact":str(expected_path.resolve()),"clock_trace":self.clock_trace,
            "recording_latency_seconds":getattr(self,"recording_latency",None),
            "detection_threshold":1e-4,
            "recording_first_detected_sample":int(signal[0]) if len(signal) else None,
            "recording_first_detected_at":time_at(int(signal[0])) if len(signal) else None,
            "recording_last_detected_at":time_at(int(signal[-1])) if len(signal) else None,
            "recording_signal_identity":"Threshold detection alone does not identify Japanese speech",
            "timing_validity":"UNESTABLISHED: block clock/counter disagreement exceeds 20 ms" if clock_errors and max(clock_errors)-min(clock_errors)>.02 else "Block timestamps retained; physical clock accuracy not independently calibrated",
            "clock_basis":"Per-block ADC mapped timestamp; first-block extrapolation only if clock trace absent",
            "first_block_extrapolation_error_range_seconds":[min((x["adc_mapped_at"]-x["sample"]/self.rate-self.origin for x in self.clock_trace),default=0),max((x["adc_mapped_at"]-x["sample"]/self.rate-self.origin for x in self.clock_trace),default=0)],
            "isolation_evidence":"Independent loopback correlated with only validated Japanese playback PCM; residual energy quantifies unexplained signal. This is not a guarantee about unrelated applications using the same cable.",
            "timing_limitations":"Loopback first speech sample is template-matched at the recorded ADC clock mapped to perf_counter; input speech-end is still a 30 ms VAD estimate."}
