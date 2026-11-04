"""Streaming segmentation and a minimal stable-prefix committer.

The v1 commit strategy treats utterance boundaries as stable points: after
speech followed by enough silence, or after a max segment cap, it commits the
span as an immutable Segment. Translation and TTS consume only committed
segments, avoiding constantly revised spoken output. Partial rewrite is future
work, not part of this phase.
"""

from __future__ import annotations

import numpy as np

from ..audio import INTERNAL_SAMPLE_RATE, ms_to_samples
from ..config import Config
from .base import Segment

_FRAME_MS = 30


class Segmenter:
    """Consumes 16 kHz float32 audio and emits committed Segment objects."""

    def __init__(self, config: Config, sample_rate: int = INTERNAL_SAMPLE_RATE):
        self.sr = sample_rate
        self.energy_threshold = config.energy_threshold
        self.frame = ms_to_samples(_FRAME_MS, sample_rate)
        # Incremental mode commits on short pauses with a smaller cap; whole-utterance
        # mode uses the longer legacy thresholds.
        silence_ms = (
            config.commit_silence_ms if config.incremental else config.min_silence_ms
        )
        max_ms = config.max_commit_ms if config.incremental else config.max_segment_ms
        self.min_silence_frames = max(
            1, ms_to_samples(silence_ms, sample_rate) // self.frame
        )
        self.max_segment_samples = ms_to_samples(max_ms, sample_rate)
        self.min_segment_samples = ms_to_samples(config.min_segment_ms, sample_rate)

        self._pending: list[np.ndarray] = []
        self._pending_len = 0
        self._tail: np.ndarray = np.zeros(0, dtype=np.float32)
        self._had_speech = False
        self._silence_run = 0

    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        if frame.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))

    def add(self, samples16k: np.ndarray) -> list[Segment]:
        """Feed new audio and return completed committed segments."""
        if samples16k.size:
            self._tail = np.concatenate([self._tail, samples16k.astype(np.float32)])

        out: list[Segment] = []
        while self._tail.size >= self.frame:
            frame = self._tail[: self.frame]
            self._tail = self._tail[self.frame :]
            self._pending.append(frame)
            self._pending_len += frame.size

            if self._rms(frame) >= self.energy_threshold:
                self._had_speech = True
                self._silence_run = 0
            else:
                self._silence_run += 1

            end_of_utterance = (
                self._had_speech and self._silence_run >= self.min_silence_frames
            )
            too_long = self._pending_len >= self.max_segment_samples
            if end_of_utterance or too_long:
                seg = self._take()
                if seg is not None:
                    out.append(seg)
        return out

    def flush(self) -> list[Segment]:
        """Flush remaining audio on input_audio_buffer.commit or connection end."""
        if self._tail.size:
            self._pending.append(self._tail)
            self._pending_len += self._tail.size
            self._tail = np.zeros(0, dtype=np.float32)
        seg = self._take(force=True)
        return [seg] if seg is not None else []

    def reset(self) -> None:
        self._pending.clear()
        self._pending_len = 0
        self._tail = np.zeros(0, dtype=np.float32)
        self._had_speech = False
        self._silence_run = 0

    def _take(self, force: bool = False) -> Segment | None:
        if not self._pending:
            self._reset_pending()
            return None
        audio = np.concatenate(self._pending)
        # Read _had_speech before _reset_pending(), which clears it. Getting this
        # order wrong made non-forced _take always return None, so add() never
        # emitted on clause pauses; old tests missed it because they used forced flush.
        has_speech = True if force else self._had_speech
        self._reset_pending()
        if audio.size < self.min_segment_samples or not has_speech:
            return None
        return Segment(samples=audio, sample_rate=self.sr)

    def _reset_pending(self) -> None:
        self._pending = []
        self._pending_len = 0
        self._had_speech = False
        self._silence_run = 0
