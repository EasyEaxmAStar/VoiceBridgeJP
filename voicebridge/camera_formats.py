"""Bounded CPU NV12 experiment. Stable BGR conversion remains in pyvirtualcam."""
import numpy as np
import ctypes,os

_libc=ctypes.CDLL('msvcrt' if os.name=='nt' else None)
_memcmp=_libc.memcmp
_memcmp.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];_memcmp.restype=ctypes.c_int


def same_pixels(a,b):
    """Read-only comparison of two strongly owned frames, no image allocation.

    Content change rate is not a sensor/exposure timestamp or a guarantee that
    identical images were duplicate hardware samples.
    """
    if a.shape!=b.shape or a.dtype!=b.dtype:return False
    if not a.flags.c_contiguous or not b.flags.c_contiguous:return bool(np.array_equal(a,b))
    return _memcmp(a.ctypes.data,b.ctypes.data,a.nbytes)==0


class NV12Converter:
    def __init__(self,width,height):
        if type(width) is not int or type(height) is not int or width<=0 or height<=0 or width%2 or height%2:
            raise ValueError('NV12 requires positive even width and height')
        self.width=width;self.height=height
        self.i420=np.empty((height*3//2,width),np.uint8);self.nv12=np.empty_like(self.i420)
        self.previous=None;self.hits=0;self.conversions=0
    def convert(self,frame):
        import cv2
        if frame.dtype!=np.uint8 or frame.shape!=(self.height,self.width,3) or not frame.flags.c_contiguous:
            raise ValueError('NV12 conversion requires contiguous uint8 H×W×3 BGR')
        if not frame.flags.writeable and frame is self.previous:
            self.hits+=1;return self.nv12
        cv2.cvtColor(frame,cv2.COLOR_BGR2YUV_I420,dst=self.i420)
        size=self.width*self.height;flat=self.i420.reshape(-1);out=self.nv12.reshape(-1)
        out[:size]=flat[:size];out[size::2]=flat[size:size*5//4];out[size+1::2]=flat[size*5//4:]
        self.previous=frame if not frame.flags.writeable else None;self.conversions+=1
        return self.nv12
