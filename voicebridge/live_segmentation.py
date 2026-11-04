"""Timestamped wrapper around the existing 30 ms Segmenter; one VAD owner."""
from collections import deque
from dataclasses import dataclass, replace
import time
import math
import numpy as np

from ort.config import Config
from ort.pipeline.segmentation import Segmenter
from ort.pipeline.recognition import RecognitionContext


@dataclass(frozen=True)
class LiveUtterance:
    context: RecognitionContext
    samples: np.ndarray
    forced: bool = False
    boundary_cause: str = 'vad'
    boundary_requested_at: float | None = None


class TimedSegmenter:
    def __init__(self, session_id, epoch, *, mode="full-sentence", silence_ms=None,
                 max_seconds=30, first_id=1, energy=.012, native_rate=16000,
                 min_speech_ms=30,pre_roll_ms=300):
        silence = silence_ms if silence_ms is not None else (450 if mode=="low-latency" else 700)
        if mode not in {"low-latency","full-sentence"} or silence not in range(300,2001) or not .3 <= max_seconds <= 30:
            raise ValueError("Invalid live segmentation settings")
        self.config = Config(incremental=True,energy_threshold=energy,commit_silence_ms=silence,
                             max_commit_ms=round(max_seconds*1000),min_segment_ms=300)
        self.segmenter = Segmenter(self.config)
        self.session_id,self.epoch,self.next_id = session_id,epoch,first_id
        self.pre_roll = deque(maxlen=max(0,pre_roll_ms//30))
        self.min_speech_ms=min_speech_ms;self._speech_frames=0;self._pending_settings=None;self.rejected_short=0
        self.tail = np.zeros(0,dtype=np.float32)
        self.offset = 0
        self.active = False
        self.origin_at = None
        self.start_sample = 0
        self.speech_end = None
        self.native_rate = native_rate
        self.origin_adc_at = None
        self.boundary_mode='AUTO';self._pending_boundary=None;self._click=None
        self.boundary_settings=dict(mode='AUTO',button='RIGHT',scope='IN_APP_TRIGGER_AREA',debounce_ms=250)
        self.boundary_events=deque(maxlen=128)

    def configure_boundary(self,settings):
        from .features import DEFAULTS,validate
        value={**DEFAULTS,'boundary':settings};self._pending_boundary=validate(value)['boundary']
        self._apply_pending()

    def request_commit(self,requested_at=None,cause='manual_button'):
        if self.boundary_mode=='AUTO' or not self.active or not self._speech_frames or self._click:return False
        stamp=time.perf_counter() if requested_at is None else requested_at
        target=max(self.offset,math.ceil(max(0,stamp-self.origin_at)*16000/self.segmenter.frame-1e-8)*self.segmenter.frame)
        self._click=dict(requested_at=stamp,target_sample=target,cause=cause)
        return True

    def _finish(self,samples,cause,click=None):
        requested=click['requested_at'] if click else None
        valid=self._speech_frames*30>=self.min_speech_ms
        context=RecognitionContext(self.session_id,self.epoch,self.next_id,self.start_sample,self.offset,
            self.origin_at+self.start_sample/16000,self.speech_end,
            max(time.perf_counter(),self.origin_at+self.offset/16000),self.native_rate,
            round(self.start_sample*self.native_rate/16000),round(self.offset*self.native_rate/16000),
            None if self.origin_adc_at is None else self.origin_adc_at+self.start_sample/16000)
        self.boundary_events.append(dict(cause=cause,requested_at=requested,committed_at=time.perf_counter(),
            cut_sample=self.offset,cut_audio_at=self.origin_at+self.offset/16000,utterance_id=self.next_id if valid else None))
        self.active=False;self.speech_end=None;self._click=None;self.pre_roll.clear()
        if not valid:self.rejected_short+=1;return None
        self.next_id+=1
        return LiveUtterance(context,np.frombuffer(samples.tobytes(),dtype=np.float32),cause=='max_duration',cause,requested)

    def _commit_click(self):
        if not self._click or self.offset<self._click['target_sample']:return None
        click=self._click
        samples=np.concatenate(self.segmenter._pending) if self.segmenter._pending else np.zeros(0,dtype=np.float32)
        self.segmenter.reset()
        return self._finish(samples,click['cause'],click)

    def configure_next(self,settings):
        from .features import DEFAULTS,validate
        import copy
        value=copy.deepcopy(DEFAULTS);value['segmentation']=settings
        self._pending_settings=validate(value)['segmentation']

    def _apply_pending(self):
        if self.active:return
        if self._pending_boundary is not None:
            self.boundary_settings=self._pending_boundary
            self.boundary_mode=self._pending_boundary['mode'];self._pending_boundary=None
            self.segmenter.min_silence_frames=(self.config.max_commit_ms//30+1 if self.boundary_mode=='MANUAL' else max(1,self.config.commit_silence_ms//30))
        if self._pending_settings is None:return
        s=self._pending_settings
        self.config=replace(self.config,commit_silence_ms=s['silence_ms'],energy_threshold=s['energy'],max_commit_ms=round(s['max_seconds']*1000))
        self.segmenter=Segmenter(self.config);self.min_speech_ms=s['min_speech_ms']
        if self.boundary_mode=='MANUAL':self.segmenter.min_silence_frames=self.config.max_commit_ms//30+1
        self.pre_roll=deque(self.pre_roll,maxlen=s['pre_roll_ms']//30)
        self._pending_settings=None

    def feed(self, samples16k, *, origin_at=None):
        if self.origin_at is None:
            self.origin_at = origin_at if origin_at is not None else time.perf_counter()
        self.tail = np.concatenate((self.tail,np.asarray(samples16k,dtype=np.float32)))
        jobs = []
        if self._click and self.offset>=self._click['target_sample']:
            job=self._commit_click()
            if job:jobs.append(job)
        size = self.segmenter.frame
        while self.tail.size >= size:
            self._apply_pending()
            frame,self.tail = self.tail[:size],self.tail[size:]
            frame = frame.copy()
            start = self.offset
            self.offset += size
            speech = self.segmenter._rms(frame) >= self.segmenter.energy_threshold
            if not self.active:
                if not speech:
                    self.pre_roll.append((start,frame))
                    continue
                self.active = True
                self._speech_frames=0
                self.start_sample = self.pre_roll[0][0] if self.pre_roll else start
                for _,prior in self.pre_roll:
                    self.segmenter.add(prior)
                self.pre_roll.clear()
            if speech:
                self._speech_frames+=1
                self.speech_end = self.origin_at+self.offset/16000
            segments = self.segmenter.add(frame)
            for segment in segments:
                quiet_seconds = self.origin_at+self.offset/16000-self.speech_end
                limit=self.offset-self.start_sample >= self.segmenter.max_segment_samples
                if self.boundary_mode=='AUTO' and limit and quiet_seconds+1e-8 < self.segmenter.min_silence_frames*.03:
                    raise RuntimeError(f"Utterance {self.next_id} exceeded the audio budget without a safe pause; cancelled, not cut mid-word")
                job=self._finish(segment.samples,'max_duration' if limit else 'vad')
                if job:jobs.append(job)
            # VAD and mouse collision consumes the same current utterance once.
            if self.active and self._click and self.offset>=self._click['target_sample']:
                job=self._commit_click()
                if job:jobs.append(job)
        return jobs
