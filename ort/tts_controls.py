"""Opt-in native controls and conservative streaming digital-silence boundaries."""
import hashlib
import math
import numpy as np

DEFAULT_SPEED = 1.20


def controls(value=None):
    value = dict(value or {})
    if set(value)-{"speed_factor", "fragment_interval", "trim_digital_silence", "pronunciation_enabled"}:
        raise ValueError("Unknown TTS control")
    result = {"speed_factor": DEFAULT_SPEED, "fragment_interval": .3,
              "trim_digital_silence": False, "pronunciation_enabled": True, **value}
    for key, low, high in (("speed_factor", 1., 1.3), ("fragment_interval", .05, .3)):
        if type(result[key]) not in (int, float) or not math.isfinite(result[key]) or not low <= result[key] <= high:
            raise ValueError("Invalid native TTS " + key)
    if type(result["trim_digital_silence"]) is not bool:
        raise ValueError("trim_digital_silence must be boolean")
    if type(result["pronunciation_enabled"]) is not bool:
        raise ValueError("pronunciation_enabled must be boolean")
    return result


class BoundaryPCM:
    """Only proven exact-zero boundary audio is cut. Noise/frication retained.

    No internal pause compression, fades over phonemes, or noise-floor guessing.
    Holdback is 100ms; retain 80ms at both boundaries. Experimental opt-in.
    """
    def __init__(self, rate, enabled=False):
        self.enabled = enabled
        self.hold = round(rate*.1)
        self.margin = round(rate*.08)
        self.pending = np.zeros(0, dtype=np.int16)
        self.leading = True
        self.raw_hash = hashlib.sha256()
        self.metrics = dict(raw_samples=0, leading_removed=0, trailing_removed=0)

    def add(self, pcm, final=False):
        self.raw_hash.update(pcm.tobytes())
        self.metrics["raw_samples"] += pcm.size
        if not self.enabled:
            return pcm
        self.pending = np.concatenate((self.pending, pcm))
        if self.leading:
            nonzero = np.flatnonzero(self.pending)
            excess = max(0, (int(nonzero[0]) if nonzero.size else len(self.pending))-self.margin)
            self.pending = self.pending[excess:]
            self.metrics["leading_removed"] += excess
            if nonzero.size:
                self.leading = False
        if final:
            nonzero = np.flatnonzero(self.pending)
            end = min(len(self.pending), int(nonzero[-1])+1+self.margin) if nonzero.size else min(len(self.pending),self.margin)
            self.metrics["trailing_removed"] += len(self.pending)-end
            output, self.pending = self.pending[:end], np.zeros(0,dtype=np.int16)
            return output
        size = max(0,len(self.pending)-self.hold)
        output, self.pending = self.pending[:size], self.pending[size:]
        return output


class AdaptiveSpeed:
    """Experimental thresholds; freeze returned speed for the whole request."""
    def __init__(self):
        self.level = 0

    def choose(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid playback backlog")
        while self.level < 2 and seconds >= (3., 6.)[self.level]:
            self.level += 1
        while self.level and seconds < (2.5, 5.5)[self.level-1]:
            self.level -= 1
        return (DEFAULT_SPEED, 1.25, 1.30)[self.level]
