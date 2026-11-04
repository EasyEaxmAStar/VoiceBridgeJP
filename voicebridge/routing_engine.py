"""One WASAPI output owner, persistent capture and generation-fenced local routes."""
from __future__ import annotations
import asyncio,collections,json,os,queue,threading,time,uuid,urllib.request
from dataclasses import dataclass,replace,asdict
from enum import Enum
import numpy as np
import soxr
from .live import LiveConfig,select_device,AudioBridgeError
from .live_buffers import CaptureRing,PlaybackRing
from .audio_devices import enumerate_devices
from .translation_link import TranslationLink
from .server_process import ManagedServer
from .release_voices import DEFAULT,NAMES
from ort.config import Config
from ort.pipeline.gptsovits_reference import validate_reference


class Route(str,Enum):
    STOPPED='STOPPED';STANDBY='STANDBY';TRANSLATE='TRANSLATE';DIRECT='DIRECT';PAUSED='PAUSED';ERROR='ERROR';TEST='DEVICE_TEST'


@dataclass(frozen=True)
class RouteBuffers:
    generation:int
    route:Route
    capture:CaptureRing
    playback:PlaybackRing


class RoutingEngine:
    def __init__(self,config:LiveConfig,*,api=None,devices=enumerate_devices,server=None,
                 preset=DEFAULT,fixture_input=False,link_factory=TranslationLink,output_latency='low',camera_telemetry=None):
        self.config=replace(config,tts_speed=1.,adaptive_speed=False)
        self.api,self.devices,self.server=api,devices,server
        self.preset=preset;self.fixture_input=fixture_input;self.link_factory=link_factory
        self.output_latency_setting=output_latency
        self.wasapi_auto_convert=False
        self.rendered={};self.rendered_samples=0
        self.speed=1.25;self.adaptive=False;self.session_id=uuid.uuid4().hex;self.next_id=1
        self.feature_settings=None
        self.boundary_armed=False;self._last_boundary_click=-float('inf')
        # User-read cable latency7168/48000=149.3ms, plus host/callback margin.
        self.cable_drain_seconds=.2
        self._lock=threading.Lock();self._lifecycle=asyncio.Lock();self._tasks=[];self._links=[];self._link=None;self._link_task=None
        self._input=self._output=None;self._input_token=0;self.audio_running=False;self.snapshot=None
        self.runtime_state='INITIALIZING';self.runtime=None;self.last_error=None;self.last_route=Route.TRANSLATE
        self.events=collections.deque(maxlen=2048);self.completed=collections.deque(maxlen=256)
        self.output_events=collections.deque(maxlen=8192)
        self.callback_events=queue.Queue(maxsize=8192);self.audio_observer=None
        self.counts=collections.Counter();self.input_level=self.output_level=0.;self._muting=True
        self._runtime_pids=set()
        self._faded=threading.Event();self._last_sample=None;self._fade_in=0;self._closing=False
        self._primed=False
        self.camera=None
        self.camera_telemetry=(os.environ.get('VOICEBRIDGE_CAMERA_TELEMETRY')=='1' if camera_telemetry is None else camera_telemetry)
        self.camera_error=None
        self.translation_prebuffer_seconds=1.;self._playing_tag=None;self._completed_source=set()
    def _event(self,kind,**data):self.events.append(dict(kind=kind,at=time.perf_counter(),**data))
    async def _http(self,path,body=None):
        url=self.config.url.replace('ws://','http://').split('/v1/')[0]+path
        def call():
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            request=urllib.request.Request(url,data=json.dumps(body).encode() if body is not None else None,
                headers={'Content-Type':'application/json'})
            with opener.open(request,timeout=180 if body else 5) as response:return json.load(response)
        return await asyncio.to_thread(call)
    async def initialize(self):
        if self.runtime_state=='STANDBY_READY' and self.runtime and self.runtime.get('worker_alive'):return
        self.runtime_state='LOADING_MODELS';self._event('initialize')
        try:
            if self.camera_telemetry and self.camera is None:
                try:
                    from .camera_controller import CameraController
                    self.camera=getattr(self,'camera_factory',CameraController)(self.session_id)
                except Exception as camera_error:self.camera_error=type(camera_error).__name__
            if self.api is None:
                # First initialization is on this stream-owner main thread.
                import sounddevice
                self.api=sounddevice
            if self.server is None and self.config.url=='ws://127.0.0.1:8765/v1/realtime/translations':
                ref=validate_reference(self.config.reference_wav,self.config.reference_text,'ja')
                self.server=ManagedServer(ref,module='voicebridge.hot_server',environment={'VOICEBRIDGE_RELEASE_PRESET':self.preset})
            if self.camera and self.server and self.server.process is None:
                self.server.environment.update(self.camera.environment)
            if self.server and (self.server.process is None or self.server.process.poll() is not None):await self.server.start()
            self.runtime=await self._http('/runtime')
            pid=self.runtime.get('pid')
            if pid is not None and pid not in self._runtime_pids:
                self._runtime_pids.add(pid);self.counts['model_runtime_loads']+=1
            if self.runtime['state']!='STANDBY_READY' or not self.runtime['worker_alive'] or self.runtime['warmup_runs']!=1:
                raise AudioBridgeError('Required real inference warmup has not passed')
            if self.runtime['release_preset']!=self.preset:raise AudioBridgeError('Resident production preset differs from explicit selection')
            features=self.runtime.get('ready',{}).get('production_features')
            if features:self._apply_features((await self._http('/features'))['settings'])
            self.runtime_state='STANDBY_READY';await self._open_audio();self._event('standby_ready',models=self.runtime['model_ids'])
        except BaseException as exc:
            self.runtime_state='ERROR';self.last_error=str(exc);await self._close_audio()
            if self.server:self.server.close()
            if self.camera:self.camera.close();self.camera=None
            raise
    def list_devices(self):return [asdict(d) for d in self.devices() if 'wasapi' in d.host_api.casefold()]
    def _apply_features(self,settings):
        if self.feature_settings and settings.get('boundary')!=self.feature_settings.get('boundary'):self.boundary_armed=False
        self.feature_settings=settings;self.speed=settings['speed'];self.adaptive=settings['adaptive']
        if self._link:
            self._link.segmenter.configure_next(settings['segmentation']);self._link.segmenter.configure_boundary(settings['boundary'])
    def arm_boundary(self,enabled,generation):
        if type(enabled) is not bool:raise ValueError('Expected explicit ARM/DISARM')
        b=(self.feature_settings or {}).get('boundary',{})
        if self._link:b=self._link.segmenter.boundary_settings
        if enabled and (not self.snapshot or self.snapshot.generation!=generation or self.snapshot.route!=Route.TRANSLATE or b.get('mode','AUTO')=='AUTO'):
            raise ValueError('Only an active MANUAL/HYBRID TRANSLATE session may be armed')
        self.boundary_armed=enabled;self._event('boundary_arm',armed=enabled)
    def end_utterance(self,*,generation,at=None,cause='manual_button',button=None,scope=None):
        stamp=time.perf_counter() if at is None else at;now=time.perf_counter()
        b=(self.feature_settings or {}).get('boundary',{})
        if self._link:b=self._link.segmenter.boundary_settings
        if (not self.boundary_armed or not self.snapshot or self.snapshot.generation!=generation or self.snapshot.route!=Route.TRANSLATE
            or not self._link or b.get('mode','AUTO')=='AUTO'):return dict(accepted=False,reason='inactive_or_disarmed')
        if type(stamp) not in {float,int} or not np.isfinite(stamp) or not now-5<=stamp<=now+.1:return dict(accepted=False,reason='invalid_event_clock')
        if cause not in {'manual_button','keyboard','mouse'}:raise ValueError('Unknown boundary cause')
        if self._link._jobs.full():
            self.counts['boundary_backpressure']+=1
            return dict(accepted=False,reason='bounded_job_queue_full_no_cut')
        if cause=='mouse' and (scope!=b['scope'] or b['button']=='DISABLED' or button not in {'LEFT','RIGHT'} or b['button'] not in {button,'EITHER'}):return dict(accepted=False,reason='scope_or_button')
        if stamp-self._last_boundary_click < b['debounce_ms']/1000:return dict(accepted=False,reason='debounced')
        accepted=self._link.segmenter.request_commit(stamp,cause)
        if accepted:
            self._last_boundary_click=stamp
            self._admit_jobs(self._link,self._link.segmenter.feed(np.zeros(0,dtype='float32')))
        result=dict(accepted=accepted,reason='queued_at_frame_boundary' if accepted else 'no_speech_or_already_pending',requested_at=stamp,acknowledged_at=now,generation=generation)
        self._event('boundary_request',**result);return result
    def _admit_jobs(self,link,jobs):
        for job in jobs:
            self.next_id=max(self.next_id,job.context.utterance_id+1)
            try:link.admit(job)
            except asyncio.QueueFull:
                self.counts['utterance_queue_overflows']+=1
                self._queue_warning=dict(id=job.context.utterance_id,at=time.perf_counter(),reason='waiting_queue_full')
                self._event('utterance_admission_rejected',id=job.context.utterance_id,reason='waiting_queue_full')
                # Only this not-yet-accepted utterance is rejected. Preserve
                # the active request, accepted FIFO, audio ring and generation.
    async def update_features(self,settings):
        result=await self._http('/features',settings)
        self._apply_features(result['settings']);self._event('features',sha256=result['sha256'],reload_seconds=result['reload_seconds'])
        return result
    def quick_reply(self,text,*,shortcut_id=None):
        from .quick_reply import QuickJob,PHRASES,validate_shortcut
        from ort.pipeline.recognition import RecognitionContext
        import copy
        link=self._link
        if text not in PHRASES:raise ValueError('Unknown quick reply')
        shortcut_id=validate_shortcut(text,shortcut_id)
        if self.runtime_state!='STANDBY_READY' or not self.snapshot or self.snapshot.route!=Route.TRANSLATE or not link or not link.connected:
            raise AudioBridgeError('Quick replies require an active TRANSLATE route and ready TTS')
        if link._jobs.full() or sum(isinstance(job,QuickJob) for job in link._jobs._queue)>=2:
            raise AudioBridgeError('Quick reply queue full; no audio submitted')
        now=time.perf_counter();identifier=max(self.next_id,link.segmenter.next_id)
        context=RecognitionContext(self.session_id,self.snapshot.generation,identifier,0,1,now,now,now)
        job=QuickJob(context,np.zeros(0,dtype='float32'),text,copy.deepcopy(self.feature_settings),self.preset,boundary_requested_at=now,shortcut_id=shortcut_id)
        link.admit(job);link.segmenter.next_id=identifier+1;self.next_id=identifier+1
        self._event('quick_reply_queued',text=text,id=identifier,generation=self.snapshot.generation)
        return dict(status='Queued',id=identifier,text=text,generation=self.snapshot.generation)
    async def _open_audio(self):
        if self.audio_running:return
        ds=self.devices();self.microphone=select_device(ds,self.config.input_device,'microphone')
        self.cable=select_device(ds,self.config.output_device,'playback')
        self.input_rate=int(self.config.input_rate or self.microphone.default_samplerate)
        self.input_channels=self.config.input_channels
        self.output_rate=int(self.config.output_rate or self.cable.default_samplerate)
        self.output_channels=min(2,self.cable.max_output_channels)
        self.output_frames=round(self.output_rate*.01);self._last_sample=np.zeros(self.output_channels,dtype=np.float32)
        self.api.check_output_settings(device=self.cable.index,samplerate=self.output_rate,channels=self.output_channels,dtype='float32')
        self._new_snapshot(Route.STANDBY)
        try:
            self._output=self.api.OutputStream(device=self.cable.index,samplerate=self.output_rate,channels=self.output_channels,
                dtype='float32',blocksize=self.output_frames,latency=self.output_latency_setting,callback=self._render,
                extra_settings=self.api.WasapiSettings(exclusive=False,auto_convert=self.wasapi_auto_convert))
            self.counts['output_stream_opens']+=1
            self._input=self._make_input(self.microphone,self.input_rate,self.input_channels)
            self.audio_running=True;self._output.start();self._input.start()
            self.output_latency=float(self._output.latency);self.input_latency=float(self._input.latency)
            self._tasks=[asyncio.create_task(self._capture_worker()),asyncio.create_task(self._monitor())]
        except BaseException:await self._close_audio();raise
    def _make_input(self,device,rate,channels):
        if not 1<=channels<=device.max_input_channels:raise AudioBridgeError('Unsupported microphone channels')
        self.api.check_input_settings(device=device.index,samplerate=rate,channels=channels,dtype='float32')
        self._input_token+=1;token=self._input_token
        def callback(data,frames,info,status):self._capture(data,frames,info,status,token=token)
        stream=self.api.InputStream(device=device.index,samplerate=rate,channels=channels,dtype='float32',
            blocksize=round(rate*.01),latency='low',callback=callback)
        self.counts['input_stream_opens']+=1;return stream
    def _new_snapshot(self,route):
        self.boundary_armed=False
        generation=self.snapshot.generation+1 if self.snapshot else 1
        seconds=min(.25,self.config.playback_seconds) if route==Route.DIRECT else self.config.playback_seconds
        snapshot=RouteBuffers(generation,route,CaptureRing(200,round(self.input_rate*.01),self.input_channels,self.input_rate),
            PlaybackRing(round(self.output_rate*seconds),self.output_channels))
        with self._lock:self.snapshot=snapshot
        camera=getattr(self,'camera',None)
        if camera:camera.reset(generation,route)
        return snapshot
    def _capture(self,data,frames,info,status,*,token,fixture=False):
        if token!=self._input_token or not self.audio_running:return
        self.input_level=float(np.max(np.abs(data))) if len(data) else 0.
        snapshot=self.snapshot
        self.counts['microphone_callbacks']+=1
        if self.fixture_input and not fixture:return
        if snapshot.route not in {Route.TRANSLATE,Route.DIRECT}:return
        # Snapshot/ring association is immutable. A racing callback can only
        # publish into a retired generation, which no output producer consumes.
        snapshot.capture.callback(data,frames,info,status)
    def _render(self,data,frames,info,status):
        data.fill(0)
        camera=getattr(self,'camera',None)
        camera_snapshot=self.snapshot;camera_count=0;camera_tag=0;camera_reason='inactive'
        if not self._lock.acquire(False):
            self.counts['output_lock_misses']+=1
            if camera:camera.observe(camera_snapshot.generation,0,data,0,'lock_miss',self.output_rate)
            return
        try:
            if status:
                camera_reason='device_error'
                self.counts['output_device_errors']+=1;self.last_error=f'WASAPI output: {status}';return
            if not self.audio_running:return
            snapshot=self.snapshot
            camera_snapshot=snapshot
            if self._muting or snapshot.route not in {Route.TRANSLATE,Route.DIRECT,Route.TEST}:
                camera_reason='muted'
                if not self._faded.is_set() and self._last_sample is not None:
                    n=min(frames,round(self.output_rate*.002))
                    data[:n]=np.linspace(1.,0.,n,dtype=np.float32)[:,None]*self._last_sample
                    self._faded.set()
                self._last_sample[:]=0;return
            if not snapshot.playback.depth:
                camera_reason='empty'
                self.counts['empty_output_callbacks']+=1
                if snapshot.route==Route.TRANSLATE and self._playing_tag is not None and self._playing_tag not in self._completed_source:
                    self.counts['translation_starvations']+=1
                    self.last_error='Translated audio starved after playback began; output muted rather than inserting a hidden gap'
                    camera_reason='starvation'
                elif self._playing_tag in self._completed_source:
                    self._completed_source.discard(self._playing_tag);self._playing_tag=None
                return
            first_tag=int(snapshot.playback.tags[snapshot.playback.read%len(snapshot.playback.data)])
            if first_tag!=self._playing_tag:
                self._completed_source.discard(self._playing_tag);self._playing_tag=first_tag;self._primed=False
            if not self._primed:
                budget=self.translation_prebuffer_seconds if snapshot.route==Route.TRANSLATE else .02
                if snapshot.playback.depth<round(self.output_rate*budget) and first_tag not in self._completed_source:
                    camera_reason='priming';return
                self._primed=True
            # Do not consume the next utterance before its own prebuffer gate.
            available=min(frames,snapshot.playback.depth)
            preview=np.take(snapshot.playback.tags,np.arange(snapshot.playback.read,snapshot.playback.read+available)%len(snapshot.playback.tags))
            changes=np.flatnonzero(preview!=first_tag);limit=int(changes[0]) if len(changes) else frames
            if available<frames and snapshot.route==Route.TRANSLATE and first_tag not in self._completed_source:
                self.counts['translation_starvations']+=1
                self.last_error='Incomplete translated callback block before source completion; output muted'
                camera_reason='starvation'
                return
            tags=np.zeros(frames,dtype=np.int64);count=snapshot.playback.take(data[:limit],tags[:limit])
            if not count:self.counts['empty_output_callbacks']+=1
            if count and self._fade_in:
                n=min(self._fade_in,count);data[:n]*=np.linspace(0.,1.,n,dtype=np.float32)[:,None];self._fade_in=0
            camera_count=count;camera_tag=int(tags[0]) if count else 0;camera_reason='consumed' if count else 'empty'
            self._last_sample[:]=data[-1];self.output_level=float(np.max(np.abs(data)))
            if self.fixture_input and count and self.rendered_samples<self.output_rate*180:
                self.rendered.setdefault(int(tags[0]),[]).append(data[:count].copy());self.rendered_samples+=count
            if count:
                try:self.callback_events.put_nowait(dict(generation=snapshot.generation,route=snapshot.route.value,
                    at=time.perf_counter(),samples=count,tag=int(tags[0]),level=self.output_level,
                    adc_dac_hint=float(getattr(info,'outputBufferDacTime',0)-getattr(info,'currentTime',0))))
                except queue.Full:self.counts['callback_diagnostic_drops']+=1
        finally:
            self._lock.release()
            if camera:
                camera.observe(camera_snapshot.generation,camera_tag,data,camera_count,camera_reason,self.output_rate,camera_snapshot.playback.read)
    async def enqueue(self,mono,identifier,generation):
        if not np.isfinite(mono).all():raise AudioBridgeError('Non-finite route PCM')
        data=np.repeat(mono[:,None],self.output_channels,axis=1)
        deadline=time.monotonic()+self.config.request_timeout
        while self.audio_running:
            snapshot=self.snapshot
            if snapshot.generation!=generation or snapshot.route not in {Route.DIRECT,Route.TRANSLATE,Route.TEST}:
                self.counts['stale_output_rejected']+=1;return False
            # Existing SPSC ring publishes its counter after the copy. Never
            # hold the output-owner lock while NumPy releases the GIL: doing
            # so caused a measured10ms silent hole in the physical DIRECT test.
            # A concurrent switch retires this entire ring; the callback only
            # reads the newly published snapshot, so late writes stay fenced.
            if snapshot.playback.push(data,identifier):return True
            if time.monotonic()>deadline:raise AudioBridgeError('Bounded output backpressure deadline')
            await asyncio.sleep(.002)
        return False
    async def _capture_worker(self):
        current=None;converter=None;origin=None;native_origin=None
        try:
            while self.audio_running:
                snapshot=self.snapshot
                if snapshot is not current:
                    current=snapshot;target=16000 if snapshot.route==Route.TRANSLATE else self.output_rate
                    converter=soxr.ResampleStream(self.input_rate,target,1,dtype='float32');origin=None;native_origin=None
                item=snapshot.capture.pop()
                if item is None:await asyncio.sleep(.002);continue
                data,adc,position,raw=item;mono=data.mean(axis=1,dtype=np.float32)
                if not np.isfinite(mono).all():raise AudioBridgeError('Microphone returned invalid PCM')
                if origin is None:
                    origin=adc-position/self.input_rate
                    native_origin=None if raw is None else raw-position/self.input_rate
                converted=converter.resample_chunk(mono,last=False)
                if self.snapshot is not snapshot:continue
                if snapshot.route==Route.TRANSLATE and self.camera:
                    try:self.camera.microphone(snapshot.generation,mono,self.input_rate,captured_at=adc)
                    except Exception:self.camera.fence('microphone_detector_failure')
                if snapshot.route==Route.DIRECT:
                    if self.audio_observer:self.audio_observer('DIRECT',snapshot.generation,0,mono,converted,self.output_rate)
                    await self.enqueue(converted,-snapshot.generation,snapshot.generation)
                elif snapshot.route==Route.TRANSLATE:
                    link=self._link
                    if link is None or link.epoch!=snapshot.generation:continue
                    link.segmenter.origin_adc_at=native_origin
                    self._admit_jobs(link,link.segmenter.feed(converted,origin_at=origin))
                    if link.segmenter.boundary_mode=='AUTO':self.boundary_armed=False
        except asyncio.CancelledError:raise
        except Exception as exc:await self.fail(str(exc))
    async def _mute_flush(self):
        started=time.perf_counter();self._muting=True;self._faded.clear()
        # Give the sole output callback one bounded opportunity for a2ms ramp.
        deadline=time.monotonic()+.04
        while not self._faded.is_set() and self._output and self._output.active and time.monotonic()<deadline:await asyncio.sleep(.001)
        # Drain the already submitted driver buffers while callbacks emit only
        # silence. Hardware tests evaluate this; the latency hint alone is not proof.
        if self._output and self._output.active:
            await asyncio.sleep(min(.4,max(self.cable_drain_seconds,getattr(self,'output_latency',.01)+.02)))
        # Preserve the endpoint's clock while switching routes. Only Stop/
        # shutdown closes it; callbacks keep sending silence during the drain.
        return time.perf_counter()-started
    async def activate(self,route):
        route=Route(route)
        if route not in {Route.TRANSLATE,Route.DIRECT,Route.STANDBY}:raise ValueError('Use pause/stop for inactive states')
        async with self._lifecycle:
            if self.runtime_state!='STANDBY_READY':raise AudioBridgeError('Real model warmup has not completed')
            try:
                if not self.audio_running:await self._open_audio()
                if self.snapshot.route==route:return
                await self._transition(route)
            except Exception as exc:
                await self.fail(str(exc));raise
    async def _transition(self,route):
        self.boundary_armed=False
        if self.camera:self.camera.fence('route_transition')
        started=time.perf_counter();old=self.snapshot;flush=await self._mute_flush()
        snapshot=self._new_snapshot(route)
        previous=self._link;self._link=None
        if previous:
            # No wait for CUDA here: fence output first, then asynchronously
            # disconnect/drain the previous model lease without unloading models.
            task=asyncio.create_task(previous.close_link());self._links.append(task)
            if self._link_task:self._link_task.cancel()
        self._fade_in=round(self.output_rate*.002);self._primed=False;self._playing_tag=None;self._completed_source=set()
        self._faded.set();self._muting=route not in {Route.TRANSLATE,Route.DIRECT,Route.TEST}
        if route==Route.TRANSLATE:
            self._link=self.link_factory(self,snapshot.generation)
            self._link_task=asyncio.create_task(self._link.run());self._links.append(self._link_task)
        if route in {Route.TRANSLATE,Route.DIRECT}:self.last_route=route
        self._links=[t for t in self._links if not t.done()]
        if len(self._links)>8:
            # Connecting sessions are cancellable too; cap retired network tasks.
            retired=self._links[:-2]
            for task in retired:task.cancel()
            await asyncio.gather(*retired,return_exceptions=True)
            self._links=[t for t in self._links if not t.done()]
        self._event('route',before=old.route.value,after=route.value,generation=snapshot.generation,
            command_seconds=time.perf_counter()-started,flush_seconds=flush,discarded_frames=old.playback.depth,
            capture_discarded_blocks=old.capture.written-old.capture.read)
    async def pause(self):
        async with self._lifecycle:
            if self.audio_running and self.snapshot.route!=Route.PAUSED:await self._transition(Route.PAUSED)
    async def resume(self):await self.activate(self.last_route)
    async def feed_diagnostic_wav(self,path):
        """Opt-in test source through the real capture/VAD path, never a fallback."""
        if not self.fixture_input:raise AudioBridgeError('Diagnostic input requires explicit --fixture-input at startup')
        if self.snapshot.route not in {Route.TRANSLATE,Route.DIRECT}:raise AudioBridgeError('Activate a route first')
        from types import SimpleNamespace
        from ort.asr_audio import load_asr_wav
        from ort.audio import resample
        audio=load_asr_wav(path,max_seconds=120,strict=True)
        x=resample(audio.samples,16000,self.input_rate)
        x=np.pad(x,(0,round(self.input_rate*1.1)))
        width=round(self.input_rate*.01);start=time.perf_counter();snapshot=self.snapshot
        for i in range(0,len(x),width):
            if self.snapshot is not snapshot:raise AudioBridgeError('Diagnostic source invalidated by route change')
            block=x[i:i+width]
            if len(block)<width:block=np.pad(block,(0,width-len(block)))
            stamp=time.perf_counter()
            self._capture(np.repeat(block[:,None],self.input_channels,axis=1),width,
                SimpleNamespace(currentTime=stamp,inputBufferAdcTime=stamp),None,token=self._input_token,fixture=True)
            await asyncio.sleep(max(0,start+(i+width)/self.input_rate-time.perf_counter()))
    async def test_device(self):
        async with self._lifecycle:
            if not self.audio_running:await self._open_audio()
            was_active=self.snapshot.route in {Route.TRANSLATE,Route.DIRECT}
            await self._transition(Route.TEST)
            t=np.arange(round(self.output_rate*.3))/self.output_rate
            tone=(.05*np.sin(2*np.pi*440*t)*np.minimum(1,t/.02)*np.minimum(1,(.3-t)/.02)).astype('float32')
            await self.enqueue(tone,0,self.snapshot.generation)
            deadline=time.monotonic()+2
            while self.snapshot.playback.depth:
                if time.monotonic()>deadline:raise AudioBridgeError('Device-test output timeout')
                await asyncio.sleep(.005)
            await self._transition(Route.PAUSED if was_active else Route.STANDBY)
            self._event('device_test',frequency_hz=440,seconds=.3,no_microphone_passthrough=True)
    def check_device_silent(self):
        """Production GUI diagnostic: settings verification only; no PCM/tone."""
        cable=select_device(self.devices(),self.config.output_device,'playback')
        self.api.check_output_settings(device=cable.index,samplerate=self.output_rate,channels=self.output_channels,dtype='float32')
        self._event('device_check_silent',device=cable.index,generated_samples=0)
    async def recover(self):
        async with self._lifecycle:
            await self._close_audio();self.last_error=None;self.runtime_state='INITIALIZING'
            try:healthy=(await self._http('/runtime'))['worker_alive']
            except Exception:healthy=False
            if not healthy:
                if not self.server:raise AudioBridgeError('External runtime failed; its owner must recover it')
                previous=self.server;await asyncio.to_thread(previous.close)
                if self.camera:self.camera.close();self.camera=None
                self.server=ManagedServer(previous.reference,port=previous.port,module='voicebridge.hot_server',
                    environment={'VOICEBRIDGE_RELEASE_PRESET':self.preset})
                self.counts['explicit_runtime_restarts']+=1
            await self.initialize()
    async def switch_microphone(self,selection,*,rate=None,channels=1):
        async with self._lifecycle:
            if not self.audio_running:raise AudioBridgeError('Open the audio engine before selecting a microphone')
            candidate=select_device(self.devices(),selection,'microphone');new_rate=int(rate or candidate.default_samplerate)
            self.api.check_input_settings(device=candidate.index,samplerate=new_rate,channels=channels,dtype='float32')
            old=(self.microphone,self.input_rate,self.input_channels,self.config);route=self.snapshot.route;t=time.perf_counter()
            await self._transition(Route.STANDBY);self._input_token+=1
            self._input.abort();self._input.close();self._input=None
            try:
                self.microphone,self.input_rate,self.input_channels=candidate,new_rate,channels
                self.config=replace(self.config,input_device=str(selection),input_rate=new_rate,input_channels=channels)
                self._input=self._make_input(candidate,new_rate,channels)
                self._new_snapshot(Route.STANDBY);self._input.start();await self._transition(route)
                self._event('microphone',status='COMMITTED',device=asdict(candidate),seconds=time.perf_counter()-t)
            except BaseException as exc:
                if self._input:
                    self._input.close();self._input=None
                self.microphone,self.input_rate,self.input_channels,self.config=old
                try:
                    self._input=self._make_input(old[0],old[1],old[2]);self._new_snapshot(Route.STANDBY)
                    self._input.start();await self._transition(route)
                except Exception as recovery:
                    await self.fail(f'New microphone and previous-device recovery failed: {exc}; {recovery}');raise
                self._event('microphone',status='ROLLED_BACK',error=str(exc),seconds=time.perf_counter()-t)
                raise
    async def select_voice(self,preset):
        if preset not in NAMES:raise ValueError('Unknown approved voice')
        async with self._lifecycle:
            route=self.snapshot.route if self.snapshot else Route.STANDBY
            if self.audio_running:await self._transition(Route.STANDBY)
            deadline=time.monotonic()+self.config.request_timeout+5
            while (await self._http('/runtime'))['inference_busy']:
                if time.monotonic()>deadline:raise TimeoutError('Previous session did not drain for voice switch')
                await asyncio.sleep(.02)
            self.runtime=await self._http('/voice',dict(preset=preset));self.preset=preset
            if self.audio_running:await self._transition(route)
    async def _monitor(self):
        last=0
        while self.audio_running:
            try:
                if self.snapshot.capture.fault:raise AudioBridgeError(self.snapshot.capture.fault)
                if self.last_error:raise AudioBridgeError(self.last_error)
                if not self._input or not self._input.active or not self._output or not self._output.active:
                    raise AudioBridgeError('Explicitly selected audio device stopped/disconnected')
                while not self.callback_events.empty():
                    event=self.callback_events.get_nowait();self.output_events.append(event)
                    rows=getattr(self._link,'traces',{})
                    row=rows.get(event['tag'])
                    if row is not None and row.get('route_generation')==event['generation']:
                        row.setdefault('first_output_callback_at',event['at'])
                        if event['level']>.001:row.setdefault('first_non_silent_output_callback_at',event['at'])
                for row in self.completed:
                    if (row.get('status')=='buffered' and row.get('route_generation')==self.snapshot.generation
                        and row.get('playback_end_frame',float('inf'))<=self.snapshot.playback.read):
                        row.update(status='played',callback_playback_complete_at=time.perf_counter())
                if time.monotonic()-last>1:
                    last=time.monotonic();ds=self.devices()
                    if self.microphone not in ds or self.cable not in ds:raise AudioBridgeError('Selected WASAPI endpoint identity disappeared')
                    self.runtime=await self._http('/runtime')
                    if not self.runtime.get('worker_alive') or self.runtime.get('state')!='STANDBY_READY':
                        raise AudioBridgeError('Resident AI runtime is no longer operational')
                await asyncio.sleep(.01)
            except asyncio.CancelledError:raise
            except Exception as exc:await self.fail(str(exc));return
    async def fail(self,error):
        if self.camera:self.camera.fence('audio_failure')
        self.last_error=error;self._muting=True;self.runtime_state='ERROR'
        if self.snapshot:
            self._new_snapshot(Route.ERROR)
            if self._output:self._output.abort()
        self._event('error',error=error)
    async def _close_audio(self):
        self._muting=True;self.audio_running=False;self._input_token+=1
        if self._link:await self._link.close_link();self._link=None
        if self._link_task:self._link_task.cancel()
        for task in self._tasks:task.cancel()
        if self._tasks:await asyncio.gather(*self._tasks,return_exceptions=True)
        self._tasks=[]
        errors=[]
        for stream in (self._input,self._output):
            if stream:
                for method in (stream.abort,stream.close):
                    try:method()
                    except Exception as exc:errors.append(str(exc))
        self._input=self._output=None
        if errors:self.last_error='; '.join(errors)
    async def stop(self):
        async with self._lifecycle:
            if self.audio_running:await self._transition(Route.STOPPED)
            await self._close_audio();self._event('stopped_models_retained')
    async def shutdown(self):
        await self.stop()
        for task in self._links:task.cancel()
        if self._links:await asyncio.gather(*self._links,return_exceptions=True)
        if self.server:await asyncio.to_thread(self.server.close)
        if self.camera:self.camera.close();self.camera=None
        self.runtime_state='STOPPED';self._event('shutdown')
    def status(self,*,detail=False):
        waiting=self._link._jobs if self._link else None
        oldest=waiting._queue[0] if waiting and not waiting.empty() else None
        admitted=(getattr(self._link,'traces',{}).get(oldest.context.utterance_id,{}).get('admitted_at',oldest.context.vad_commit_at) if oldest else None)
        waiting_count=waiting.qsize() if waiting else 0
        waiting_capacity=waiting.maxsize if waiting else TranslationLink.WAITING_CAPACITY
        route=self.snapshot.route.value if self.snapshot else 'STOPPED'
        state=self.runtime_state
        if state=='STANDBY_READY':state={'TRANSLATE':'ACTIVE_TRANSLATE','DIRECT':'ACTIVE_DIRECT','PAUSED':'PAUSED','ERROR':'ERROR'}.get(route,state)
        rows=list(self.completed)
        if not detail:
            if self._link:rows=sorted({r['id']:r for r in [*rows,*getattr(self._link,'traces',{}).values()]}.values(),key=lambda r:r['id'])
            keys={'id','context','source_text','japanese_text','status','request_kind','admitted_at','websocket_send_at','queue_wait_seconds','stage_times','first_validated_pcm_at',
                'first_output_callback_at','first_non_silent_output_callback_at','callback_playback_complete_at','speech_end_at','vad_commit_at','tts_metrics'}
            rows=[{k:v for k,v in r.items() if k in keys} for r in rows[-5:]]
        return dict(state=state,runtime_state=self.runtime_state,route=route,generation=self.snapshot.generation if self.snapshot else 0,
            camera_telemetry=self.camera.status() if self.camera else dict(enabled=False,error=self.camera_error),
            boundary=dict(armed=self.boundary_armed,settings=self._link.segmenter.boundary_settings if self._link else (self.feature_settings or {}).get('boundary',{}),
                desired_settings=(self.feature_settings or {}).get('boundary',{}),
                effective_mode=self._link.segmenter.boundary_mode if self._link else 'AUTO',
                recording=self._link.segmenter.active if self._link else False,
                duration=(self._link.segmenter.offset-self._link.segmenter.start_sample)/16000 if self._link and self._link.segmenter.active else 0.,
                last=self._link.segmenter.boundary_events[-1] if self._link and self._link.segmenter.boundary_events else None),
            diagnostic_fixture_input=self.fixture_input,
            raw_microphone_transmitted=route=='DIRECT',preset=self.preset,speed=self.speed,adaptive=self.adaptive,
            features={k:v for k,v in self.feature_settings.items() if k not in {'dictionary','expressions'}} if self.feature_settings else None,
            cable_drain_seconds=self.cable_drain_seconds,
            translation_prebuffer_seconds=self.translation_prebuffer_seconds,
            wasapi_output_settings=dict(exclusive=False,auto_convert=self.wasapi_auto_convert,latency=self.output_latency_setting),
            error=self.last_error,microphone=asdict(self.microphone) if hasattr(self,'microphone') else None,
            output=asdict(self.cable) if hasattr(self,'cable') else None,input_level=self.input_level,output_level=self.output_level,
            queue_seconds=self.snapshot.playback.depth/self.output_rate if self.snapshot else 0,
            waiting_jobs=waiting_count,waiting_capacity=waiting_capacity,
            oldest_waiting_age_seconds=max(0,time.perf_counter()-admitted) if admitted is not None else 0.,
            queue_nearly_full=waiting_count>=max(1,(waiting_capacity*3+3)//4),
            queue_last_rejection=getattr(self,'_queue_warning',None),
            translation_connected=self._link.connected if self._link else False,
            model_runtime=self.runtime,counts=dict(self.counts),events=list(self.events) if detail else list(self.events)[-10:],
            output_events=list(self.output_events) if detail else list(self.output_events)[-5:],
            utterances=rows)
