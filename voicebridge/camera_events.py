"""Pure camera control contracts. No devices, audio processing, IPC or models.

Phase 2 must supply trusted callback observations, not queue/TTS notifications.
An engine belongs to one controller lifetime; control sequence numbers survive
route resets. Playback/ASR sequences restart at 1 after each scope reset.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import math
import threading
import time
from typing import Callable


class CameraMode(StrEnum):
    OFF = 'OFF'
    CLEAN = 'CLEAN'
    GLITCHING = 'GLITCHING'


class EventKind(StrEnum):
    ASR_STARTED = 'ASR_STARTED'
    MICROPHONE_ONSET = 'MICROPHONE_ONSET'
    MICROPHONE_CANCEL = 'MICROPHONE_CANCEL'
    PLAYBACK_STARTED = 'PLAYBACK_STARTED'
    PLAYBACK_FINISHED = 'PLAYBACK_FINISHED'
    ASR_CANCELLED = 'ASR_CANCELLED'
    JOB_FAILED = 'JOB_FAILED'
    ROUTE_RESET = 'ROUTE_RESET'
    PAUSE = 'PAUSE'
    STOP = 'STOP'
    CAMERA_ENABLE = 'CAMERA_ENABLE'
    CAMERA_DISABLE = 'CAMERA_DISABLE'
    TELEMETRY_LOST = 'TELEMETRY_LOST'
    SAFETY_TIMEOUT = 'SAFETY_TIMEOUT'


class Producer(StrEnum):
    CONTROL = 'CONTROL'
    ASR = 'ASR'
    PLAYBACK = 'PLAYBACK'


class SourceKind(StrEnum):
    ASR_UTTERANCE = 'ASR_UTTERANCE'
    QUICK_REPLY = 'QUICK_REPLY'
    DIRECT = 'DIRECT'
    DEVICE_TEST = 'DEVICE_TEST'
    UNKNOWN = 'UNKNOWN'


class ShortcutID(StrEnum):
    HAI = 'quick.hai'
    GREETING_POLITE = 'quick.greeting_polite'


@dataclass(frozen=True)
class SourceDescriptor:
    kind: SourceKind = SourceKind.UNKNOWN
    shortcut_id: str | None = None

    @property
    def exempt(self) -> bool:
        # Never infer a shortcut from spoken text or accept an exemption flag.
        return self.kind == SourceKind.QUICK_REPLY and self.shortcut_id == ShortcutID.HAI


@dataclass(frozen=True)
class Scope:
    owner: str
    session: str
    epoch: int

    def __post_init__(self):
        if (not isinstance(self.owner, str) or not 1 <= len(self.owner) <= 128
                or not isinstance(self.session, str) or not 1 <= len(self.session) <= 128
                or type(self.epoch) is not int or self.epoch < 1):
            raise ValueError('Invalid owner/session/epoch')


@dataclass(frozen=True)
class CameraEvent:
    kind: EventKind
    scope: Scope
    producer: Producer
    seq: int
    at: float
    job_id: int | None = None
    source: SourceDescriptor = SourceDescriptor()
    # Callback-domain nonzero consumed sample interval ends here. Never a
    # synthesis/queue duration. FINISHED is an authoritative idle observation.
    until: float | None = None
    consumed_frames: int = 0
    nonzero_frames: int = 0
    validated_pcm: bool = False
    new_scope: Scope | None = None
    route: str | None = None
    episode_revision: int | None = None


@dataclass(frozen=True)
class CameraConfig:
    timeout: float = 30.
    telemetry_lease: float = .5
    history_seconds: float = 10.
    history_capacity: int = 2048
    pending_capacity: int = 16

    def __post_init__(self):
        if (type(self.timeout) not in {int, float} or not math.isfinite(self.timeout)
                or not 1 <= self.timeout <= 120):
            raise ValueError('Safety timeout must be 1–120 seconds')
        if (type(self.telemetry_lease) not in {int, float}
                or type(self.history_seconds) not in {int, float}
                or not 0 < self.telemetry_lease <= 1 or not math.isfinite(self.telemetry_lease)
                or not self.telemetry_lease <= self.history_seconds <= 120):
            raise ValueError('Invalid telemetry/history duration')
        if any(type(x) is not int or not 1 <= x <= 8192
               for x in (self.history_capacity, self.pending_capacity)):
            raise ValueError('Invalid bounded capacity')


@dataclass(frozen=True)
class CameraSnapshot:
    scope: Scope
    mode: CameraMode = CameraMode.OFF
    enabled: bool = False
    route: str = 'STANDBY'
    revision: int = 0
    clean_fence: int = 0
    reason: str = 'initial'
    # Fixed-size per-producer sequence and timestamp vectors.
    sequences: tuple[int, int, int] = (0, 0, 0)
    timestamps: tuple[float, float, float] = (-math.inf, -math.inf, -math.inf)
    telemetry_at: float = -math.inf
    telemetry_lost: bool = False
    intervals: tuple[tuple[float, float], ...] = ()
    last_playback_start: float = -math.inf
    job_watermark: int = 0
    microphone_watermark: int = 0
    pending: tuple[int, ...] = ()
    deadline: float | None = None
    episode_revision: int | None = None
    start_fence: float = -math.inf
    clock_at: float = -math.inf


_CHANNEL = {Producer.CONTROL: 0, Producer.ASR: 1, Producer.PLAYBACK: 2}
_ASR = {EventKind.ASR_STARTED, EventKind.ASR_CANCELLED, EventKind.JOB_FAILED}
_PLAYBACK = {EventKind.PLAYBACK_STARTED, EventKind.PLAYBACK_FINISHED}
_ROUTES = {'TRANSLATE', 'DIRECT', 'STANDBY', 'PAUSED', 'STOPPED', 'ERROR', 'DEVICE_TEST'}


def _finite(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def _clean(s: CameraSnapshot, at: float, reason: str, *, lost=False) -> CameraSnapshot:
    return replace(s, mode=CameraMode.CLEAN if s.enabled else CameraMode.OFF,
                   pending=(), deadline=None, episode_revision=None, start_fence=max(s.start_fence, at),
                   telemetry_lost=s.telemetry_lost or lost, reason=reason)


def _publish(old: CameraSnapshot, new: CameraSnapshot) -> CameraSnapshot:
    if old == new:
        return old
    revision = old.revision + 1
    return replace(new, revision=revision,
                   clean_fence=revision if new.mode != CameraMode.GLITCHING else old.clean_fence)


def reduce_event(s: CameraSnapshot, event: CameraEvent | None, now: float,
                 config: CameraConfig = CameraConfig()) -> CameraSnapshot:
    """Return a new immutable snapshot; None performs a watchdog tick.

    This is a trusted in-process contract, not an untrusted IPC decoder. A
    transport must authenticate producers and translate parse/queue failures
    into TELEMETRY_LOST. No foreign event may reset the current scope.
    """
    old = s
    if not _finite(now) or now < s.clock_at:
        return _publish(old, _clean(s, s.clock_at, 'invalid_clock', lost=True))
    s = replace(s, clock_at=now)
    if s.mode == CameraMode.GLITCHING:
        if s.deadline is not None and now >= s.deadline:
            s = _clean(s, now, 'safety_timeout')
        elif s.telemetry_lost or now - s.telemetry_at >= config.telemetry_lease:
            s = _clean(s, now, 'telemetry_expired', lost=True)
    if event is None:
        return _publish(old, s)
    if not isinstance(event, CameraEvent):
        return _publish(old, _clean(s, now, 'invalid_event', lost=True))
    if event.scope != s.scope:
        return _publish(old, s)
    if (not isinstance(event.kind, EventKind) or not isinstance(event.producer, Producer)
            or type(event.seq) is not int or event.seq < 1
            or not _finite(event.at) or event.at > now):
        return _publish(old, _clean(s, now, 'invalid_event', lost=True))
    expected = Producer.ASR if event.kind in _ASR else Producer.PLAYBACK if event.kind in _PLAYBACK else Producer.CONTROL
    if event.producer != expected:
        return _publish(old, _clean(s, now, 'wrong_producer', lost=True))
    channel = _CHANNEL[expected]
    previous = s.sequences[channel]
    if event.seq <= previous:
        return _publish(old, s)
    seqs = list(s.sequences);seqs[channel] = event.seq
    times = list(s.timestamps);times[channel] = event.at
    s = replace(s, sequences=tuple(seqs), timestamps=tuple(times))
    if (event.kind == EventKind.SAFETY_TIMEOUT
            and event.episode_revision != s.episode_revision):
        # A delayed timer belongs to its original episode, not a newer one.
        # Consume its sequence without regressing the control timestamp.
        return _publish(old, replace(s, timestamps=old.timestamps))
    if event.at < old.timestamps[channel] or now - event.at > config.history_seconds:
        return _publish(old, _clean(s, now, 'out_of_order_or_expired', lost=True))
    if event.seq != previous + 1:
        return _publish(old, _clean(s, now, 'sequence_gap', lost=True))

    kind = event.kind
    if kind == EventKind.ROUTE_RESET:
        target = event.new_scope
        if (not isinstance(target, Scope) or event.route not in _ROUTES
                or target == s.scope or (target.owner == s.scope.owner and target.epoch <= s.scope.epoch)):
            return _publish(old, _clean(s, now, 'invalid_reset', lost=True))
        s = CameraSnapshot(target, enabled=s.enabled, mode=CameraMode.CLEAN if s.enabled else CameraMode.OFF,
                           route=event.route, sequences=(event.seq, 0, 0),
                           timestamps=(event.at, -math.inf, -math.inf), start_fence=now, clock_at=now)
        s = replace(s, reason='route_reset')
    elif kind in {EventKind.CAMERA_ENABLE, EventKind.CAMERA_DISABLE}:
        s = replace(s, enabled=kind == EventKind.CAMERA_ENABLE)
        s = _clean(s, now, kind.value.lower())
    elif kind in {EventKind.PAUSE, EventKind.STOP}:
        s = replace(s, route='PAUSED' if kind == EventKind.PAUSE else 'STOPPED')
        s = _clean(s, now, kind.value.lower())
    elif kind == EventKind.TELEMETRY_LOST:
        s = _clean(s, now, 'telemetry_lost', lost=True)
    elif kind == EventKind.SAFETY_TIMEOUT:
        # Early or stale timer notifications cannot cancel a newer episode.
        if s.deadline is not None and now >= s.deadline:
            s = _clean(s, now, 'safety_timeout')
    elif kind in _PLAYBACK:
        if now - event.at >= config.telemetry_lease:
            return _publish(old, _clean(s, now, 'late_playback_telemetry', lost=True))
        spans = tuple(span for span in s.intervals if span[1] >= now - config.history_seconds)
        if kind == EventKind.PLAYBACK_FINISHED:
            spans = tuple((a, min(b, event.at)) for a, b in spans)
        s = replace(s, telemetry_at=event.at, intervals=spans)
        if kind == EventKind.PLAYBACK_STARTED:
            if (type(event.validated_pcm) is not bool or type(event.consumed_frames) is not int
                    or type(event.nonzero_frames) is not int
                    or not 0 <= event.nonzero_frames <= event.consumed_frames
                    or not isinstance(event.source, SourceDescriptor)):
                return _publish(old, _clean(s, now, 'invalid_playback', lost=True))
            if not event.validated_pcm or not event.consumed_frames or not event.nonzero_frames:
                # Silence/priming/enqueue are explicitly not playback signals.
                return _publish(old, s)
            if (not _finite(event.until)
                    or not event.at < event.until <= event.at + config.telemetry_lease):
                return _publish(old, _clean(s, now, 'invalid_playback_interval', lost=True))
            if not event.source.exempt:
                if len(spans) >= config.history_capacity:
                    return _publish(old, _clean(s, now, 'history_overflow', lost=True))
                s = replace(s, intervals=spans + ((event.at, event.until),),
                            last_playback_start=max(s.last_playback_start, event.at))
                s = _clean(s, event.at, 'non_exempt_playback')
        # FINISHED is callback idle (not producer completion) and never starts
        # glitches. Already observed intervals remain available for late ASR.
    elif kind == EventKind.MICROPHONE_CANCEL:
        s = _clean(s, now, 'microphone_cancel')
    elif kind in _ASR or kind == EventKind.MICROPHONE_ONSET:
        if type(event.job_id) is not int or event.job_id < 1:
            return _publish(old, _clean(s, now, 'invalid_job', lost=True))
        microphone=kind == EventKind.MICROPHONE_ONSET
        watermark=s.microphone_watermark if microphone else s.job_watermark
        if kind not in {EventKind.ASR_STARTED,EventKind.MICROPHONE_ONSET}:
            s = replace(s, job_watermark=max(s.job_watermark, event.job_id))
            if event.job_id in s.pending:
                s = _clean(s, now, kind.value.lower())
        elif event.job_id > watermark:
            s = replace(s, **({'microphone_watermark':event.job_id} if microphone else {'job_watermark':event.job_id}))
            playing = any(a <= event.at < b or a <= now < b for a, b in s.intervals)
            if (s.enabled and s.route == 'TRANSLATE' and not s.telemetry_lost
                    and now - s.telemetry_at < config.telemetry_lease and not playing
                    and event.at > s.start_fence and event.at > s.last_playback_start):
                if len(s.pending) >= config.pending_capacity:
                    s = _clean(s, now, 'pending_overflow', lost=True)
                else:
                    s = replace(s, mode=CameraMode.GLITCHING, pending=s.pending + (event.job_id,),
                                deadline=s.deadline if s.deadline is not None else now + config.timeout,
                                episode_revision=s.episode_revision if s.deadline is not None else old.revision + 1,
                                reason='microphone_onset' if microphone else 'asr_started')
    return _publish(old, s)


class CameraEngine:
    """Thread-safe convenience owner; reducer remains independently testable."""
    def __init__(self, scope: Scope, *, clock: Callable[[], float] = time.monotonic,
                 config: CameraConfig = CameraConfig()):
        self._snapshot = CameraSnapshot(scope)
        self.clock, self.config = clock, config
        self._lock = threading.Lock()

    @property
    def snapshot(self) -> CameraSnapshot:
        with self._lock:
            return self._snapshot

    def dispatch(self, event: CameraEvent | None = None) -> CameraSnapshot:
        with self._lock:
            self._snapshot = reduce_event(self._snapshot, event, self.clock(), self.config)
            return self._snapshot


@dataclass(frozen=True)
class CameraCommand:
    mode: CameraMode
    revision: int
    clean_fence: int
    deadline: float | None

    @classmethod
    def from_snapshot(cls, snapshot: CameraSnapshot) -> CameraCommand:
        return cls(snapshot.mode, snapshot.revision, snapshot.clean_fence, snapshot.deadline)


class CommandGate:
    """Future frame worker's revision latch, usable from a priority clean path.

    Commands must originate from one trusted engine lifetime. Recreate the gate
    on worker/owner handshake, never mix different engines' revision domains.
    """
    def __init__(self):
        self.current = CameraCommand(CameraMode.OFF, 0, 0, None)
        self._lock = threading.Lock()

    def apply(self, command: CameraCommand) -> bool:
        with self._lock:
            if command.revision <= self.current.revision or command.revision < self.current.clean_fence:
                return False
            if command.mode == CameraMode.GLITCHING and command.revision <= command.clean_fence:
                return False
            self.current = command
            return True
