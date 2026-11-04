"""Dedicated bounded metadata actor; deliberately no camera frame processing."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
import threading
import time
import uuid

import numpy as np

from .camera_events import (CameraEngine, CameraEvent, CameraConfig, Scope, Producer,
                            EventKind, SourceDescriptor, SourceKind)
from .camera_ipc import PipeReader


@dataclass(frozen=True, slots=True)
class CallbackObservation:
    generation: int
    at: float
    tag: int
    count: int
    first: int
    last: int
    nonzero: int
    reason: str
    rate: int
    end_frame: int


class CameraController:
    def __init__(self, session, *, reader=None, capacity=256, registry_capacity=1024,
                 clock=time.perf_counter, start=True, enabled=True, autoload_environment=True):
        self.reader = reader if reader is not None else PipeReader()
        self.clock = clock;self.engine = CameraEngine(Scope(self.reader.owner, session, 1), clock=clock)
        self.capacity = capacity;self.registry_capacity = registry_capacity
        self.observations = deque();self.history=deque(maxlen=256);self.sources = {};self.completions={}
        self._asr_seq = self._play_seq = self._control_seq = 0
        self._lock = threading.RLock();self.stop = threading.Event();self.loss = False
        self.error = None;self.unknown_sources = 0;self.dropped = 0;self.received = 0
        self.last_reason = 'initial';self.max_delivery_seconds = 0.;self._loss_reported = False
        self.video=None;self.video_error=None
        self.trigger_mode='asr';self.onset_threshold=.012;self.onsets=deque();self._mic_id=0
        self._detector=None;self._detector_generation=None;self.last_onset=None
        self._control(EventKind.CAMERA_ENABLE if enabled else EventKind.CAMERA_DISABLE)
        video_config=os.environ.get('VOICEBRIDGE_CAMERA_VIDEO_CONFIG') if autoload_environment else None
        if video_config:
            try:
                from .camera_process import CameraVideoProcess
                self.video=CameraVideoProcess(video_config)
            except Exception as exc:self.video_error=str(exc)
        self.thread = None
        if start:
            self.thread = threading.Thread(target=self._run, name='camera-control-actor', daemon=True)
            self.thread.start()

    @property
    def environment(self):return self.reader.environment

    def set_enabled(self,enabled,*,timeout=30):
        if type(enabled) is not bool:raise ValueError('Expected explicit camera enabled flag')
        config=CameraConfig(timeout=timeout)
        with self._lock:
            self._control(EventKind.CAMERA_DISABLE)
            self._detector=None;self._detector_generation=None;self.onsets.clear()
            self.engine.config=config
            if enabled:
                snapshot=self.engine.snapshot
                if snapshot.telemetry_lost:
                    # Loss stays latched in the pure reducer. Explicit camera
                    # recovery creates a NEW locally generated camera-owner
                    # incarnation; the authenticated transport stays resident.
                    # ROUTE_RESET fences old timestamps and requires fresh PCM
                    # observations. Audio session/epoch and source registry do
                    # not change, and no caller can supply this owner token.
                    self._control(EventKind.ROUTE_RESET,new_scope=Scope(uuid.uuid4().hex,snapshot.scope.session,snapshot.scope.epoch),route=snapshot.route)
                    self._asr_seq=self._play_seq=0
                self._control(EventKind.CAMERA_ENABLE)
            if self.video:self.video.publish(self.engine.snapshot)

    def _control(self, kind, event_at=None, **values):
        self._control_seq += 1
        return self.engine.dispatch(CameraEvent(kind, self.engine.snapshot.scope, Producer.CONTROL,
                                                self._control_seq, self.clock() if event_at is None else event_at, **values))

    def microphone(self,generation,mono,rate,captured_at=None):
        """Existing capture-worker tap. No PCM retained or sent over IPC."""
        if self.trigger_mode!='microphone':return
        s=self.engine.snapshot
        if not s.enabled or s.route!='TRANSLATE' or s.scope.epoch!=generation:return
        from .camera_onset import OnsetDetector
        if self._detector_generation!=generation or self._detector is None or self._detector.threshold!=self.onset_threshold:
            self._detector=OnsetDetector(self.onset_threshold);self._detector_generation=generation
        decided=self.clock()
        # Preserve acquisition time so queued old microphone blocks cannot
        # reactivate an episode after a newer playback/CLEAN fence.
        at=decided if captured_at is None else min(decided,float(captured_at))
        if not np.isfinite(at):self.loss=True;return
        if self._detector.feed(mono,rate):
            if len(self.onsets)>=32:self.loss=True;return
            self._mic_id+=1;self.onsets.append((generation,at,self._mic_id,captured_at,decided))

    def reset(self, generation, route):
        with self._lock:
            scope = self.engine.snapshot.scope
            if generation <= scope.epoch:return
            route=getattr(route,'value',route)
            self._control(EventKind.ROUTE_RESET, new_scope=Scope(scope.owner, scope.session, generation), route=route)
            self._asr_seq = self._play_seq = 0;self.sources.clear();self.completions.clear();self._loss_reported=False
            if route=='DIRECT':self.sources[generation,-generation]=SourceDescriptor(SourceKind.DIRECT)
            elif route=='DEVICE_TEST':self.sources[generation,0]=SourceDescriptor(SourceKind.DEVICE_TEST)

    def complete(self, generation, identifier, end_frame):
        with self._lock:
            if (generation,identifier) in self.sources:self.completions[generation,identifier]=end_frame

    def fence(self, reason='lifecycle'):
        with self._lock:
            self.error = reason;self._control(EventKind.TELEMETRY_LOST)

    def register(self, context, source):
        with self._lock:
            scope = self.engine.snapshot.scope
            if (context.session_id, context.epoch) != (scope.session, scope.epoch):return
            key = context.epoch, context.utterance_id
            old = self.sources.get(key)
            if old is not None and old != source:
                self.fence('source_metadata_conflict');return
            if key not in self.sources and len(self.sources) >= self.registry_capacity:
                self.fence('source_registry_full');return
            self.sources[key] = source

    def job_failed(self, context):
        # Local private failure notification has no effect on audio behavior.
        with self._lock:
            scope = self.engine.snapshot.scope
            if (context.session_id, context.epoch) != (scope.session, scope.epoch):return
            if self.trigger_mode=='microphone':self._control(EventKind.MICROPHONE_CANCEL)
            self._asr_seq += 1
            self.engine.dispatch(CameraEvent(EventKind.JOB_FAILED, scope, Producer.ASR,
                self._asr_seq, self.clock(), job_id=context.utterance_id))
            self.sources.pop((scope.epoch,context.utterance_id),None)

    def observe(self, generation, tag, data, count, reason, rate, end_frame=0):
        """Audio callback: no actor/pipe lock or wait, bounded scalar-only SPSC.

        CPython deque operations are atomic under the GIL. Never overwrite a
        record: signal loss instead. Arrays below are bounded by the existing
        callback block and are never retained or sent to another thread.
        """
        if len(self.observations) >= self.capacity:
            self.loss = True;self.dropped += 1;return
        try:
            first = last = -1;nonzero = 0
            if count:
                indices = np.flatnonzero(np.any(data[:count] != 0, axis=1))
                nonzero = len(indices)
                if nonzero:first = int(indices[0]);last = int(indices[-1])
            self.observations.append(CallbackObservation(generation, self.clock(), int(tag), count,
                first, last, nonzero, reason, rate, end_frame))
        except Exception:
            self.loss = True;self.dropped += 1

    def _callback(self, observation):
        scope = self.engine.snapshot.scope
        if observation.generation != scope.epoch:return
        self.history.append(observation)
        self.last_reason = observation.reason
        self.received += 1
        if observation.reason in {'device_error', 'starvation', 'lock_miss', 'inactive', 'stale'}:
            self.fence('callback_' + observation.reason);return
        self._play_seq += 1
        source = self.sources.get((scope.epoch, observation.tag))
        if observation.count and source is None:
            self.unknown_sources += 1;source = SourceDescriptor()
        if observation.nonzero:
            # Future sample offsets are scheduled callback-domain bounds, not
            # a physical DAC measurement. Reducer receives at the acceptance
            # clock; first/last offsets stay in this bounded diagnostic record.
            end = observation.at + (observation.last + 1) / observation.rate
            event = CameraEvent(EventKind.PLAYBACK_STARTED, scope, Producer.PLAYBACK,
                self._play_seq, observation.at, source=source or SourceDescriptor(), until=end,
                consumed_frames=observation.count, nonzero_frames=observation.nonzero, validated_pcm=True)
        else:
            event = CameraEvent(EventKind.PLAYBACK_FINISHED, scope, Producer.PLAYBACK,
                                self._play_seq, observation.at)
        self.engine.dispatch(event)
        for key,end in tuple(self.completions.items()):
            if key[0]==scope.epoch and observation.end_frame>=end:
                self.completions.pop(key,None);self.sources.pop(key,None)

    def pump(self):
        """One bounded actor turn, also available for deterministic tests."""
        with self._lock:
            if self.loss or (self.reader.lost.is_set() and not self._loss_reported):
                self.loss = False;self._loss_reported=True
                self.fence(self.reader.error or 'callback_queue_overflow')
            # Prioritize all available callback records over delayed ASR.
            for _ in range(self.capacity):
                try:observation = self.observations.popleft()
                except IndexError:break
                self._callback(observation)
            for _ in range(128):
                try:message = self.reader.messages.get_nowait()
                except Exception:break
                if message['kind'] == 'HEARTBEAT':continue
                scope = self.engine.snapshot.scope
                if (message.get('session'), message.get('epoch')) != (scope.session, scope.epoch):continue
                if self.trigger_mode=='microphone':
                    if message['kind'] in {'JOB_FAILED','ASR_CANCELLED'}:self._control(EventKind.MICROPHONE_CANCEL)
                    continue
                key = scope.epoch, message.get('job')
                source = self.sources.get(key)
                if source is None or source.kind != SourceKind.ASR_UTTERANCE:
                    self.fence('unregistered_asr_job');continue
                self._asr_seq += 1
                self.max_delivery_seconds = max(self.max_delivery_seconds, self.clock()-message['at'])
                self.engine.dispatch(CameraEvent(EventKind(message['kind']), scope, Producer.ASR,
                    self._asr_seq, message['at'], job_id=message['job']))
            for _ in range(32):
                try:generation,at,identifier,captured_at,decided=self.onsets.popleft()
                except IndexError:break
                if self.trigger_mode=='microphone' and generation==self.engine.snapshot.scope.epoch:
                    self._control(EventKind.MICROPHONE_ONSET,event_at=at,job_id=identifier)
                    self.last_onset=dict(captured_block_at=captured_at,detected_at=decided,delivered_at=self.clock(),generation=generation)
            # Reconcile callbacks that arrived during ASR reduction before a
            # coherent decision can be observed through decision/status.
            for _ in range(self.capacity):
                try:observation=self.observations.popleft()
                except IndexError:break
                self._callback(observation)
            if self.loss:self.loss=False;self.fence('callback_queue_overflow')
            self.engine.dispatch()
            if self.video:
                try:self.video.publish(self.engine.snapshot)
                except Exception as exc:self.video_error=str(exc)

    def _run(self):
        while not self.stop.wait(.002):
            try:self.pump()
            except Exception as exc:self.fence(type(exc).__name__)

    @property
    def decision(self):
        """Future frame consumers use this coherent actor-turn snapshot."""
        with self._lock:return self.engine.snapshot

    def status(self):
        s = self.decision
        return dict(mode=s.mode.value, reason=s.reason, revision=s.revision, clean_fence=s.clean_fence,
                    video=self.video.status() if self.video else dict(enabled=False,error=self.video_error),
                    telemetry_lost=s.telemetry_lost, pipe_connected=self.reader.connected.is_set(),
                    callback_records=self.received, callback_drops=self.dropped,
                    unknown_sources=self.unknown_sources, registered_sources=len(self.sources),
                    max_asr_delivery_seconds=self.max_delivery_seconds, last_callback_reason=self.last_reason,
                    trigger_mode=self.trigger_mode,last_microphone_onset=self.last_onset)

    def close(self):
        with self._lock:self._control(EventKind.CAMERA_DISABLE)
        self.stop.set()
        if self.thread:self.thread.join(1)
        if self.video:
            try:self.video.publish(self.engine.snapshot);self.video.close()
            except Exception as exc:self.video_error=str(exc)
        self.reader.close()
