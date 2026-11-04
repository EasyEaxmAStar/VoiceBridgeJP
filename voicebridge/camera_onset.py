"""Bounded-window activity edges; invoked by capture drain, never audio callback."""
import math
import numpy as np


class OnsetDetector:
    def __init__(self,threshold=.012):
        self.threshold=threshold;self.floor=.002;self.active=False;self.quiet=0.;self.attack=0.

    def feed(self,mono,rate):
        onset=False
        for offset in range(0,len(mono),max(1,round(rate*.01))):
            block=mono[offset:offset+max(1,round(rate*.01))]
            rms=float(np.sqrt(np.mean(np.square(block,dtype=np.float64))))
            duration=len(block)/rate;high=max(self.threshold,self.floor*3)
            if not math.isfinite(rms):raise ValueError('Invalid microphone energy')
            if not self.active:
                if rms>=high:
                    self.attack+=duration
                    if self.attack>=.01-1e-9:self.active=True;onset=True;self.quiet=0.
                else:
                    self.attack=0.;self.floor=.995*self.floor+.005*min(rms,self.threshold/2)
            elif rms<high*.55:
                self.quiet+=duration
                if self.quiet>=.35:self.active=False;self.attack=0.;self.quiet=0.
            else:self.quiet=0.
        return onset
