"""CPU-only, seeded and bounded visual effects. Clean frames are untouched."""
from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class EffectsConfig:
    regions_min: int = 2
    regions_max: int = 6
    block_min: int = 5
    block_max: int = 35
    stutter_intensity: float = 1.
    freeze_min: float = .1
    freeze_max: float = .8
    randomness: float = 1.
    continuous_mosaic: bool = False
    strong_stutter: bool = False

    def __post_init__(self):
        if type(self.continuous_mosaic) is not bool:raise ValueError('Continuous mosaic must be an explicit boolean')
        if type(self.strong_stutter) is not bool:raise ValueError('Strong stutter must be an explicit boolean')
        if not all(type(x) is int for x in (self.regions_min,self.regions_max)) or not 1<=self.regions_min<=self.regions_max<=12:
            raise ValueError('Mosaic regions must be integers within 1–12')
        import math
        if not all(type(x) is int for x in (self.block_min,self.block_max)) or not 1<=self.block_min<=self.block_max<=64:
            raise ValueError('Mosaic pixel blocks must be within1–64')
        if any(type(x) not in {int,float} or not math.isfinite(x) for x in (self.stutter_intensity,self.freeze_min,self.freeze_max,self.randomness)):
            raise ValueError('Effect parameters must be finite numbers')
        if not 0<=self.stutter_intensity<=1 or not 0<=self.randomness<=1 or not .05<=self.freeze_min<=self.freeze_max<=.8:
            raise ValueError('Stutter/randomness range0–1; freeze range0.05–0.8s')


class GlitchProcessor:
    def __init__(self, *, seed=None, config=EffectsConfig(), resize=None):
        self.rng=random.Random(seed);self.config=config;self.resize=resize
        self.reset()

    def reset(self):
        self.active=False;self.cached=None;self.regions=();self.kind='clean'
        self.next_event=0.;self.event_end=0.;self.next_unique=0.;self.freeze_until=0.
        self.events=0;self.last_unique=None
        self._mosaic_source=None;self._mosaic_pixels=None
        self._strong_pixels=None;self._strong_until=0.

    def _visible_frame(self,frame):
        """Optional full-frame mosaic floor; never used for CLEAN output.

        Random regions alone can miss the subject, and stutter-only events
        can be invisible when capture is slower than their unique-frame rate.
        This explicitly selected floor keeps even the first glitch frame and
        the random inter-event intervals visibly pixelated. Keep one bounded
        immutable-source cache; mutable caller frames must always recompute.
        """
        immutable=not frame.flags.writeable
        if immutable and frame is self._mosaic_source:return self._mosaic_pixels
        h,w=frame.shape[:2];block=self.config.block_max
        if self.resize is None:
            import cv2
            resize=lambda image,size:cv2.resize(image,size,interpolation=cv2.INTER_NEAREST)
        else:resize=self.resize
        small=resize(frame,(max(1,w//block),max(1,h//block)))
        output=resize(small,(w,h));output.setflags(write=False)
        self._mosaic_source=frame if immutable else None
        self._mosaic_pixels=output if immutable else None
        return output

    def _event(self, frame, now):
        h,w=frame.shape[:2]
        self.kind=self.rng.choice(('mosaic','stutter','both'))
        if self.config.stutter_intensity!=1 and self.kind!='mosaic' and self.rng.random()>=self.config.stutter_intensity:self.kind='mosaic'
        self.event_end=now+self._uniform(.2,1.1)
        self.next_event=self.event_end+self._uniform(.04,.45)
        regions=[]
        if self.kind in {'mosaic','both'}:
            for _ in range(self.rng.randint(self.config.regions_min,self.config.regions_max)):
                rw=self.rng.randint(max(2,w//12),max(3,w//2));rh=self.rng.randint(max(2,h//12),max(3,h//2))
                regions.append((self.rng.randrange(w-rw+1),self.rng.randrange(h-rh+1),rw,rh,self.rng.randint(self.config.block_min,self.config.block_max)))
        self.regions=tuple(regions);self.events+=1
        self.next_unique=now;self.freeze_until=0.

    def _uniform(self,a,b):
        value=self.rng.uniform(a,b)
        # Exact original draw/value/order for the stable default.
        return value if self.config.randomness==1 else (a+b)/2+(value-(a+b)/2)*self.config.randomness

    def process(self, frame, now, glitching):
        if not glitching:return self._process(frame,now,False)
        if self.config.strong_stutter:
            if self._strong_pixels is not None and now<self._strong_until:return self._strong_pixels
            # Fresh update at the selected freeze boundary, never compounded
            # with the original mild stutter's cached frame/hold interval.
            self.cached=None;self.freeze_until=0.;self.next_unique=0.
            result=self._process(frame,now,True)
            self._strong_pixels=result.copy();self._strong_pixels.setflags(write=False)
            self._strong_until=now+self._uniform(self.config.freeze_min,self.config.freeze_max)
            return self._strong_pixels
        return self._process(frame,now,True)

    def _process(self, frame, now, glitching):
        if not glitching:
            if self.active:self.reset()
            return frame
        if self.config.continuous_mosaic:frame=self._visible_frame(frame)
        if not self.active:self.active=True;self.next_event=now
        if now>=self.next_event:self._event(frame,now)
        if now>=self.event_end:
            self.cached=None
            return frame
        stutter=self.kind in {'stutter','both'}
        if stutter and self.cached is not None and now<max(self.freeze_until,self.next_unique):return self.cached
        output=frame
        if self.regions:
            if self.resize is None:
                import cv2
                resize=lambda roi,size:cv2.resize(roi,size,interpolation=cv2.INTER_NEAREST)
            else:resize=self.resize
            output=frame.copy()
            for x,y,w,h,block in self.regions:
                roi=output[y:y+h,x:x+w]
                small=resize(roi,(max(1,w//block),max(1,h//block)))
                output[y:y+h,x:x+w]=resize(small,(w,h))
        if stutter:
            self.next_unique=now+1/self._uniform(5,15)
            if self.rng.random()<.16:self.freeze_until=now+self._uniform(self.config.freeze_min,self.config.freeze_max)
            # Copy only when holding an unmodified source frame; capture owns
            # its immutable frame, but the cache should not retain old slots.
            self.cached=output.copy() if output is frame else output
        else:self.cached=None
        self.last_unique=now
        return output
