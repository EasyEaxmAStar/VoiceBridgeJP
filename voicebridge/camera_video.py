"""Latest-frame capture, processing and paced output; no audio/model imports."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
import time

import numpy as np

from .camera_effects import EffectsConfig,GlitchProcessor
from .camera_events import CameraCommand,CameraMode,CommandGate


class CameraOutputError(RuntimeError):
    def __init__(self,code,message):super().__init__(message);self.code=code


def obs_channel_in_use():
    """Read-only native OBS queue existence check, no registry/device writes."""
    import ctypes,os
    if os.name!='nt':return False
    from ctypes import wintypes as W
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenFileMappingW.argtypes=[W.DWORD,W.BOOL,W.LPCWSTR];kernel.OpenFileMappingW.restype=W.HANDLE
    kernel.CloseHandle.argtypes=[W.HANDLE]
    handle=kernel.OpenFileMappingW(4,False,'OBSVirtualCamVideo')
    if handle:kernel.CloseHandle(handle);return True
    return False


def virtual_error(exc,config):
    native=str(exc)
    if 'not found' in native.casefold() or 'not installed' in native.casefold():code='MISSING'
    elif config.virtual_backend=='obs' and obs_channel_in_use():code='BUSY'
    else:code='UNAVAILABLE'
    labels={'MISSING':'未检出所选虚拟摄像头后端，请安装或修复官方OBS。',
            'BUSY':'OBS虚拟摄像头通道已被占用，请关闭其他写入程序或持有旧通道的接收程序。',
            'UNAVAILABLE':'无法打开所选虚拟摄像头，请检查后端、设备占用和权限。'}
    return CameraOutputError(code,labels[code]+' 未安装驱动或替换设备。Native: '+native)


@dataclass(frozen=True)
class VideoConfig:
    camera_identity: str
    width: int = 1280
    height: int = 720
    fps: int = 30
    capture_backend: str = 'dshow'
    virtual_backend: str = 'obs'
    virtual_device: str = 'OBS Virtual Camera'
    seed: int | None = None
    regions_min: int = 2
    regions_max: int = 6
    max_frame_age: float = .2
    reconnect_seconds: float = 1.
    processing_backend: str = 'bgr'
    block_min: int = 5
    block_max: int = 35
    stutter_intensity: float = 1.
    freeze_min: float = .1
    freeze_max: float = .8
    randomness: float = 1.
    continuous_mosaic: bool = False
    strong_stutter: bool = False
    trigger_mode: str = 'microphone'
    onset_threshold: float = .012

    def __post_init__(self):
        if self.trigger_mode not in {'microphone','asr'}:raise ValueError('Choose microphone or asr trigger')
        if type(self.onset_threshold) not in {int,float} or not math.isfinite(self.onset_threshold) or not .001<=self.onset_threshold<=.1:raise ValueError('Onset RMS threshold must be within .001–.1')
        if not isinstance(self.camera_identity,str) or not self.camera_identity or len(self.camera_identity)>1024:
            raise ValueError('Explicit physical camera identity is required; run --diagnose')
        if (self.width,self.height) not in {(640,480),(1280,720),(1920,1080)} or type(self.fps) is not int or not 15<=self.fps<=60:
            raise ValueError('Use 640x480, 1280x720 or verified1920x1080, 15–60 fps (baseline 30)')
        if self.capture_backend!='dshow':raise ValueError('Only dshow has verified device-identity binding in this release')
        if self.virtual_backend not in {'obs','unitycapture'} or not isinstance(self.virtual_device,str) or not self.virtual_device:
            raise ValueError('Select an explicit obs/unitycapture virtual device')
        if self.virtual_backend=='obs' and self.virtual_device!='OBS Virtual Camera':raise ValueError('OBS backend requires OBS Virtual Camera')
        if self.seed is not None and type(self.seed) is not int:raise ValueError('Seed must be an integer or null')
        if not .05<=self.max_frame_age<=1 or not math.isfinite(self.max_frame_age):raise ValueError('Invalid maximum frame age')
        if not .1<=self.reconnect_seconds<=10:raise ValueError('Invalid reconnect interval')
        if self.processing_backend not in {'bgr','nv12'}:raise ValueError('Choose bgr stable or nv12 experimental')
        self.effects_config()

    def effects_config(self):
        return EffectsConfig(self.regions_min,self.regions_max,self.block_min,self.block_max,
                             self.stutter_intensity,self.freeze_min,self.freeze_max,self.randomness,self.continuous_mosaic,self.strong_stutter)


@dataclass(frozen=True)
class LiveFrame:
    sequence: int
    at: float
    pixels: np.ndarray


class LatestFrame:
    """Exactly one clean capture; no video queue and no consumer backlog."""
    def __init__(self):self._frame=None;self._lock=threading.Lock()
    def put(self, frame):
        with self._lock:self._frame=frame
    def get(self):
        with self._lock:return self._frame
    def invalidate(self):self.put(None)


class OpenCVCapture:
    def __init__(self, config):
        import cv2
        from .camera_devices import resolve_camera,enumerate_cameras
        device=resolve_camera(config.camera_identity)
        # DirectShow enumeration/index binding is explicit. MSMF has a separate
        # ordering; do not guess a cross-backend numeric mapping.
        if config.capture_backend!='dshow':raise RuntimeError('MSMF identity binding is not verified. Select dshow; no backend substitution was made.')
        self.capture=cv2.VideoCapture(device.index,cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture.release();raise RuntimeError('Cannot open selected webcam: check Windows camera privacy permission and close other camera owners')
        try:
            if not any(d.index==device.index and d.identity==device.identity and not d.virtual for d in enumerate_cameras()):
                raise RuntimeError('Camera enumeration changed during open; refusing a possible device substitution')
        except Exception:
            self.capture.release();raise
        self.capture.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,config.width);self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,config.height)
        self.capture.set(cv2.CAP_PROP_FPS,config.fps)
        self.buffer_setting=self.capture.set(cv2.CAP_PROP_BUFFERSIZE,1)
        self.device=device
        self.format=dict(width=self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),height=self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                         requested_fps_readback=self.capture.get(cv2.CAP_PROP_FPS),buffer_setting_supported=self.buffer_setting)

    def read(self):return self.capture.read()
    def close(self):self.capture.release()


class VirtualOutput:
    def __init__(self, config):
        import pyvirtualcam
        self.config=config;self.error=None;self.closed=False;self.last_retry=0.
        self.requested_backend=config.processing_backend;self.active_backend=config.processing_backend
        self.fallback_reason=None;self.recovery_revision=0;self.converter=None
        self.conversion_times=deque(maxlen=1024);self.native_times=deque(maxlen=1024)
        try:
            self.camera=self._open()
        except Exception as exc:
            if self.active_backend=='nv12':
                self._fallback(type(exc).__name__)
                try:self.camera=self._open()
                except Exception as fallback:raise virtual_error(fallback,config) from fallback
            else:raise virtual_error(exc,config) from exc
    def _open(self):
        import pyvirtualcam
        c=self.config
        if self.active_backend=='nv12':
            if c.virtual_backend!='obs':raise ValueError('Experimental NV12 is verified only with OBS')
            from .camera_formats import NV12Converter
            self.converter=NV12Converter(c.width,c.height)
        else:self.converter=None
        return pyvirtualcam.Camera(width=c.width,height=c.height,fps=c.fps,
            fmt=pyvirtualcam.PixelFormat.NV12 if self.active_backend=='nv12' else pyvirtualcam.PixelFormat.BGR,
            device=c.virtual_device,backend=c.virtual_backend)
    def _fallback(self,reason):
        self.active_backend='bgr';self.converter=None;self.recovery_revision+=1
        self.fallback_reason='NV12失败，已回退同一设备的CPU BGR：'+reason
    def send(self,frame):
        if self.closed:return False
        if self.camera is None:
            if time.monotonic()-self.last_retry<1:return False
            self.last_retry=time.monotonic()
            try:
                self.camera=self._open()
            except Exception:return False
        try:
            payload=frame
            if self.converter:
                at=time.perf_counter();payload=self.converter.convert(frame);self.conversion_times.append(time.perf_counter()-at)
            at=time.perf_counter();self.camera.send(payload);self.native_times.append(time.perf_counter()-at)
            self.error=None;return True
        except Exception as exc:
            experimental=self.active_backend=='nv12'
            if experimental:self._fallback(type(exc).__name__)
            self.error='Virtual output disconnected; retrying only the selected backend/device'
            try:self.camera.close()
            except Exception:pass
            self.camera=None;self.last_retry=time.monotonic()-(1 if experimental else 0);return False
            
    def status(self):
        def percentile(values,p):return None if not values else float(np.percentile(values,p)*1000)
        return dict(requested_backend=self.requested_backend,active_backend=self.active_backend,
            ready=self.camera is not None and not self.closed,fallback_reason=self.fallback_reason,
            recovery_revision=self.recovery_revision,
            backend_state='CLOSED' if self.closed else 'READY' if self.camera is not None else 'RETRYING',
            conversion_p50_ms=percentile(self.conversion_times,50),conversion_p95_ms=percentile(self.conversion_times,95),
            native_send_p50_ms=percentile(self.native_times,50),native_send_p95_ms=percentile(self.native_times,95),
            conversion_scope='NV12 explicit CPU conversion' if self.active_backend=='nv12' else 'BGR conversion is inside native send/copy timing',
            cache_hits=self.converter.hits if self.converter else 0,error=self.error)
    def close(self):
        self.closed=True
        if self.camera is not None:self.camera.close();self.camera=None


class CaptureLoop:
    def __init__(self, config, slot, *, factory=OpenCVCapture, clock=time.perf_counter):
        self.config=config;self.slot=slot;self.factory=factory;self.clock=clock
        self.stop=threading.Event();self.thread=None;self.error=None;self.frames=0;self.reconnects=0
        self.first_frame_at=None;self.intervals=deque(maxlen=1024);self.last_frame_at=None;self.format=None
        self.read_times=deque(maxlen=1024);self.content=deque(maxlen=1024);self._previous_pixels=None
    def start(self):
        self.thread=threading.Thread(target=self._run,name='camera-latest-capture',daemon=True);self.thread.start()
    def _run(self):
        while not self.stop.is_set():
            source=None
            try:
                source=self.factory(self.config)
                self.format=getattr(source,'format',None)
                while not self.stop.is_set():
                    started=self.clock();ok,pixels=source.read();self.read_times.append(self.clock()-started)
                    if not ok:raise RuntimeError('Webcam disconnected or capture failed')
                    if pixels.dtype!=np.uint8 or pixels.shape!=(self.config.height,self.config.width,3):
                        raise RuntimeError('Webcam did not supply requested BGR size. Choose a supported 720p/1080p mode; no silent resize.')
                    pixels=np.ascontiguousarray(pixels).copy();pixels.setflags(write=False)
                    at=self.clock()
                    from .camera_formats import same_pixels
                    changed=self._previous_pixels is None or not same_pixels(pixels,self._previous_pixels)
                    self.content.append((at,changed));self._previous_pixels=pixels
                    if self.first_frame_at is None:self.first_frame_at=at
                    if self.last_frame_at is not None:self.intervals.append(at-self.last_frame_at)
                    self.last_frame_at=at
                    self.frames+=1;self.slot.put(LiveFrame(self.frames,at,pixels));self.error=None
            except Exception as exc:
                self.error=str(exc);self.slot.invalidate();self._previous_pixels=None;self.reconnects+=1
            finally:
                if source:
                    try:source.close()
                    except Exception:pass
            self.stop.wait(self.config.reconnect_seconds)
    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(1)
        self.slot.invalidate()
        self._previous_pixels=None
        return self.thread is None or not self.thread.is_alive()

    def status(self):
        content=tuple(self.content);elapsed=content[-1][0]-content[0][0] if len(content)>1 else 0
        reads=tuple(self.read_times);intervals=tuple(self.intervals)
        return dict(capture_frames=self.frames,capture_error=self.error,reconnects=self.reconnects,
            capture_observed_fps=1/float(np.mean(intervals)) if intervals else None,
            content_change_fps=sum(changed for at,changed in content[1:])/elapsed if elapsed else None,
            content_change_scope='Pixel-content changes per second, not sensor exposure identity',
            read_p50_ms=float(np.percentile(reads,50)*1000) if reads else None,
            read_p95_ms=float(np.percentile(reads,95)*1000) if reads else None)


class CommandState:
    """Reader-to-writer latest control, bounded and revision-fenced."""
    def __init__(self,clock=time.perf_counter):
        self.clock=clock;self.gate=CommandGate();self.last_received=-math.inf;self.lock=threading.Lock()
    def update(self, command):
        with self.lock:
            if self.gate.apply(command):self.last_received=self.clock();return True
            return False
    def read(self):
        with self.lock:return self.gate.current,self.last_received


class VideoPipeline:
    def __init__(self,config,slot,commands,output,*,processor=None,clock=time.perf_counter,wait=None):
        self.config=config;self.slot=slot;self.commands=commands;self.output=output;self.clock=clock
        self.processor=processor or GlitchProcessor(seed=config.seed,config=config.effects_config())
        self.stop=threading.Event();self.wait=wait or self.stop.wait
        self.final_wait=time.sleep if wait is None else wait
        self.blank=np.zeros((config.height,config.width,3),dtype=np.uint8)
        self.sent=0;self.repeated=0;self.stale=0;self.deadline_misses=0;self.mode='CLEAN';self.error=None
        self.glitch_frames=0;self.clean_recoveries=0;self.last_recovery_age_ms=None
        self.send_times=deque(maxlen=1024);self.ages=deque(maxlen=1024)
        self._fence=0;self._last_pixels=None
        self.blocked_episode=None;self.output_misses=0
        self.effect_errors=0;self.last_effect_error=None
        self._output_recovery=0

    def step(self):
        now=self.clock();command,heartbeat=self.commands.read()
        frame=self.slot.get()
        healthy=frame is not None and 0<=now-frame.at<=self.config.max_frame_age
        episode=(command.clean_fence,command.deadline)
        recovery=getattr(self.output,'recovery_revision',0)
        if recovery!=self._output_recovery:
            self._output_recovery=recovery;self.blocked_episode=episode;self.processor.reset()
        eligible=(command.mode==CameraMode.GLITCHING and now-heartbeat<.5
                and command.deadline is not None and now<command.deadline)
        if command.mode==CameraMode.GLITCHING and (not eligible or not healthy or getattr(self.output,'error',None)):
            self.blocked_episode=episode
        glitch=eligible and episode!=self.blocked_episode
        if command.clean_fence!=self._fence:
            self.processor.reset();self._fence=command.clean_fence
        if not healthy:
            self.processor.reset();pixels=self.blank;self.stale+=1
            self.error='No fresh webcam frame; output is blank, not a stale freeze';glitch=False
        else:
            self.error=None;self.ages.append(now-frame.at)
            try:pixels=self.processor.process(frame.pixels,now,glitch)
            except Exception as exc:
                self.processor.reset();self.blocked_episode=episode;glitch=False;pixels=frame.pixels
                self.effect_errors+=1;self.last_effect_error=type(exc).__name__
                self.error='Visual effect failed; sending current clean frame'
        # A clean fence may arrive while processing. Re-read the latest capture,
        # never recover with the frame that entered a freeze/effect previously.
        latest,last_received=self.commands.read()
        if (latest.clean_fence!=command.clean_fence or latest.mode!=CameraMode.GLITCHING
                or self.clock()-last_received>=.5 or latest.deadline is None or self.clock()>=latest.deadline):
            self.processor.reset();current=self.slot.get();glitch=False
            pixels=current.pixels if current is not None and 0<=self.clock()-current.at<=self.config.max_frame_age else self.blank
        previous_mode=self.mode
        self.mode='GLITCHING' if glitch else 'CLEAN'
        self.glitch_frames+=int(glitch)
        if previous_mode=='GLITCHING' and not glitch:
            self.clean_recoveries+=1
            current=self.slot.get()
            self.last_recovery_age_ms=None if current is None else (self.clock()-current.at)*1000
        self.repeated+=int(pixels is self._last_pixels);self._last_pixels=pixels
        if self.output.send(pixels) is False:
            self.output_misses+=1;self.error=getattr(self.output,'error','Virtual output unavailable')
            self.blocked_episode=episode;self.processor.reset()
        else:self.sent+=1;self.send_times.append(self.clock())

    def run(self, *, seconds=None, on_status=None):
        start=target=last_status=self.clock();period=1/self.config.fps
        try:
            while not self.stop.is_set() and (seconds is None or self.clock()-start<seconds):
                now=self.clock()
                if now<target:
                    if self.wait(target-now):break
                    continue
                self.step();target+=period
                if self.clock()>target+period:
                    self.deadline_misses+=1;target=self.clock()+period  # no unbounded catch-up bursts
                if on_status and self.clock()-last_status>=1:
                    on_status(self.status());last_status=self.clock()
        finally:
            self.processor.reset()
            self.mode='CLEAN'
            # Best effort fresh final frame before closing; a missing camera or
            # blocked native driver cannot promise receiver-side live recovery.
            current=self.slot.get()
            if current is not None and current.pixels is not self._last_pixels and 0<=self.clock()-current.at<=self.config.max_frame_age:
                if self.send_times:self.final_wait(max(0,self.send_times[-1]+period-self.clock()))
                current=self.slot.get()
                try:
                    if current is not None and 0<=self.clock()-current.at<=self.config.max_frame_age:self.output.send(current.pixels)
                except Exception:pass
            self.output.close()

    def status(self):
        intervals=np.diff(tuple(self.send_times))
        return dict(sent=self.sent,mode=self.mode,stale_frames=self.stale,repeated=self.repeated,
            continuous_mosaic=self.config.continuous_mosaic,strong_stutter=self.config.strong_stutter,
            effect_kind='strong_freeze' if self.config.strong_stutter and self.processor.active else self.processor.kind,
            glitch_frames=self.glitch_frames,clean_recoveries=self.clean_recoveries,last_recovery_age_ms=self.last_recovery_age_ms,
            deadline_misses=self.deadline_misses,error=self.error,
            output_misses=self.output_misses,
            effect_errors=self.effect_errors,last_effect_error=self.last_effect_error,
            observed_fps=float(1/np.mean(intervals)) if len(intervals) else None,
            p95_frame_age_ms=float(np.percentile(self.ages,95)*1000) if self.ages else None,
            max_buffers='latest slot=1, effect cache=1, optional mosaic source/output=1 each, strong freeze cache=1, transient/current output and fixed blank; diagnostics=1024; native driver buffers separate')
