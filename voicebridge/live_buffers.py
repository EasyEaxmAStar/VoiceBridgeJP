"""Bounded single-producer/single-consumer rings for CPython 3.12 callbacks.

Slots are published only after copy; unread slots are never overwritten. NumPy
may release the GIL during copying, but publication counters remain unchanged
until that copy finishes. Each ring has exactly one producer and one consumer.
Epoch changes retire the entire ring rather than racing pointer resets.
"""
from __future__ import annotations
import time

import numpy as np


class CaptureRing:
    def __init__(self, blocks, block_frames, channels, sample_rate=48000):
        self.sample_rate = sample_rate
        self.data = np.empty((blocks,block_frames,channels),dtype=np.float32)
        self.times = np.empty(blocks,dtype=np.float64)
        self.positions = np.empty(blocks,dtype=np.int64)
        self.adc_clocks = np.full(blocks,np.nan,dtype=np.float64)
        self.written = self.read = self.frames_seen = self.dropped = 0
        self.max_depth = 0
        self.fault = None

    def callback(self, data, frames, info, status):
        position = self.frames_seen
        self.frames_seen += frames
        if status:
            self.fault = f"Input stream status: {status}"
        if frames != self.data.shape[1] or self.written-self.read >= len(self.data):
            self.dropped += 1
            self.fault = "Capture ring overflow or unexpected callback frame size"
            return
        slot = self.written % len(self.data)
        now = time.perf_counter()
        try:
            adc = now + float(info.inputBufferAdcTime-info.currentTime)
            raw_adc = float(info.inputBufferAdcTime)
        except (AttributeError, TypeError):
            adc = now-frames/self.sample_rate
            raw_adc = np.nan
        np.copyto(self.data[slot],data)
        self.times[slot] = adc
        self.positions[slot] = position
        self.adc_clocks[slot] = raw_adc
        self.written += 1
        self.max_depth = max(self.max_depth,self.written-self.read)

    def pop(self):
        if self.read == self.written:
            return None
        slot = self.read % len(self.data)
        raw = float(self.adc_clocks[slot])
        result = (self.data[slot].copy(),float(self.times[slot]),int(self.positions[slot]),raw if np.isfinite(raw) else None)
        self.read += 1
        return result


class PlaybackRing:
    def __init__(self, capacity_frames, channels):
        self.data = np.zeros((capacity_frames,channels),dtype=np.float32)
        self.tags = np.zeros(capacity_frames,dtype=np.int64)
        self.read = self.written = self.max_depth = 0

    @property
    def depth(self): return self.written-self.read

    def push(self, samples, utterance_id):
        count = len(samples)
        if count > len(self.data)-self.depth:
            return False
        start = self.written % len(self.data)
        first = min(count,len(self.data)-start)
        self.data[start:start+first] = samples[:first]
        self.tags[start:start+first] = utterance_id
        if first < count:
            self.data[:count-first] = samples[first:]
            self.tags[:count-first] = utterance_id
        self.written += count
        self.max_depth = max(self.max_depth,self.depth)
        return True

    def take(self, output, tag_output):
        output.fill(0)
        tag_output.fill(0)
        count = min(len(output),self.depth)
        start = self.read % len(self.data)
        first = min(count,len(self.data)-start)
        output[:first] = self.data[start:start+first]
        tag_output[:first] = self.tags[start:start+first]
        if first < count:
            output[first:count] = self.data[:count-first]
            tag_output[first:count] = self.tags[:count-first]
        self.read += count
        return count
