"""Backend-independent recognition states; production Qwen emits FINAL only.

The incremental protocols are extension contracts, not an implemented streaming
recognizer. No provisional or stable-prefix update is eligible for translation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Iterable, Protocol

import numpy as np


class TranscriptPhase(str, Enum):
    PROVISIONAL = "provisional"
    STABLE_PREFIX = "stable_prefix"
    FINAL = "final"


@dataclass(frozen=True)
class RecognitionContext:
    session_id: str
    epoch: int
    utterance_id: int
    audio_start_sample: int
    audio_end_sample: int
    capture_start_at: float
    speech_end_at: float
    vad_commit_at: float
    native_sample_rate: int | None = None
    native_start_sample: int | None = None
    native_end_sample: int | None = None
    capture_adc_clock_start: float | None = None

    def __post_init__(self):
        if not isinstance(self.session_id, str) or not 1 <= len(self.session_id) <= 100:
            raise ValueError("Invalid recognition session ID")
        if any(type(x) is not int for x in (self.epoch,self.utterance_id,self.audio_start_sample,self.audio_end_sample)):
            raise ValueError("Recognition IDs/sample positions must be integers")
        if self.epoch < 1 or self.utterance_id < 1 or not 0 <= self.audio_start_sample < self.audio_end_sample:
            raise ValueError("Invalid recognition epoch, utterance or audio range")
        if not all(math.isfinite(x) for x in (self.capture_start_at,self.speech_end_at,self.vad_commit_at)):
            raise ValueError("Recognition timestamps must be finite")
        if not self.capture_start_at <= self.speech_end_at <= self.vad_commit_at:
            raise ValueError("Recognition timestamps are out of order")
        native = (self.native_sample_rate,self.native_start_sample,self.native_end_sample)
        if any(x is not None for x in native):
            if not all(type(x) is int for x in native) or not 8000 <= native[0] <= 192000 or not 0 <= native[1] < native[2]:
                raise ValueError("Invalid original native audio sample positions")
        if self.capture_adc_clock_start is not None and not math.isfinite(self.capture_adc_clock_start):
            raise ValueError("Original ADC timestamp is not finite")

    def wire(self):
        return asdict(self)


@dataclass(frozen=True)
class RecognitionUpdate:
    context: RecognitionContext
    text: str
    phase: TranscriptPhase
    revision: int
    language: str | None = None


class TranscriptGate:
    """Stable prefixes cannot be revised; only one final result can commit."""

    def __init__(self, context):
        self.context = context
        self.revision = 0
        self.stable = ""
        self.final = False
        self.cancelled = False

    def apply(self, update: RecognitionUpdate) -> str | None:
        if self.cancelled or self.final or update.context != self.context:
            raise ValueError("Stale/cancelled/duplicate recognition update")
        if not isinstance(update.phase, TranscriptPhase) or update.revision <= self.revision:
            raise ValueError("Recognition phase/revision is invalid or reordered")
        if not update.text.startswith(self.stable):
            raise ValueError("A confirmed transcript prefix was rewritten")
        self.revision = update.revision
        if update.phase == TranscriptPhase.STABLE_PREFIX:
            self.stable = update.text
        if update.phase != TranscriptPhase.FINAL:
            return None
        self.final = True
        return update.text

    def cancel(self):
        self.cancelled = True


class IncrementalRecognitionSession(Protocol):
    """Future model session; timestamps/IDs remain owned by the audio job."""
    def push(self, samples16k: np.ndarray, start_sample: int) -> Iterable[RecognitionUpdate]: ...
    def finish(self) -> Iterable[RecognitionUpdate]: ...
    def cancel(self) -> None: ...


class IncrementalASRBackend(Protocol):
    def open_utterance(self, context: RecognitionContext, language: str) -> IncrementalRecognitionSession: ...


def final_update(transcription, context):
    if isinstance(transcription, RecognitionUpdate):
        update = transcription
    else:
        update = RecognitionUpdate(context, transcription.text, TranscriptPhase.FINAL, 1, transcription.language)
    text = TranscriptGate(context).apply(update)
    if text is None:
        raise ValueError("An utterance ASR backend returned non-final text; translation/TTS are forbidden")
    return update
