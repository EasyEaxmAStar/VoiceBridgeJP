"""Production live controller: independent bounded capture/VAD/WS/playback.

Only validated translated WebSocket PCM can enter the playback ring. Microphone
callbacks publish into a different capture ring and have no playback reference.
"""
from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import ipaddress
import json
import math
import queue
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import uuid

import numpy as np
import soxr

from ort.audio import float32_to_b64_pcm16, resample
from ort.pipeline.gptsovits_reference import validate_reference
from ort.tts_controls import DEFAULT_SPEED
from .audio_bridge import AudioBridgeError, pcm24k_b64_to_float32
from .audio_devices import enumerate_devices
from .live_buffers import CaptureRing, PlaybackRing
from .live_segmentation import TimedSegmenter


class State(str,Enum):
    STOPPED="STOPPED"
    STARTING="STARTING"
    RUNNING="RUNNING"
    PAUSED="PAUSED"
    STOPPING="STOPPING"
    ERROR="ERROR"


@dataclass(frozen=True)
class LiveConfig:
    input_device: str
    output_device: str
    reference_wav: str
    reference_text: str
    reference_language: str = "ja"
    url: str = "ws://127.0.0.1:8765/v1/realtime/translations"
    mode: str = "full-sentence"
    silence_ms: int | None = None
    max_utterance_seconds: float = 30
    capture_seconds: float = 2
    playback_seconds: float = 8
    utterance_capacity: int = 4
    jitter_ms: int = 60
    request_timeout: float = 120
    shutdown_timeout: float = 3
    settle_seconds: float = .1
    input_rate: int | None = None
    output_rate: int | None = None
    input_channels: int = 1
    source_language: str = "zh"
    tts_speed: float = DEFAULT_SPEED
    fragment_interval: float = .3
    trim_digital_silence: bool = False
    adaptive_speed: bool = False
    pronunciation_enabled: bool = True


def local_url(url):
    parts = urlsplit(url)
    try:
        local = parts.hostname=="localhost" or ipaddress.ip_address(parts.hostname).is_loopback
    except (ValueError,TypeError):
        local = False
    if parts.scheme != "ws" or not local or parts.username or parts.password or parts.fragment or parts.path != "/v1/realtime/translations":
        raise AudioBridgeError("Live inference requires a localhost ws:// /v1/realtime/translations URL")
    query = dict(parse_qsl(parts.query))
    query["voicebridge"] = "committed-v1"
    return urlunsplit(parts._replace(query=urlencode(query)))


def select_device(devices, selection, role):
    wasapi = [d for d in devices if "wasapi" in d.host_api.casefold()]
    text = str(selection)
    candidates = [d for d in wasapi if d.index==int(text)] if text.isdecimal() else [d for d in wasapi if d.name.casefold()==text.casefold()]
    if len(candidates) != 1:
        raise AudioBridgeError(f"Select one current WASAPI {role} by exact name or current index: {selection!r}")
    device = candidates[0]
    name = device.name.casefold()
    if role == "microphone":
        physical = any(x in name for x in ("microphone","mic ","麦克风","マイク","headset","mikrofon"))
        excluded = any(x in name for x in ("cable","virtual","stereo mix","立体声混音","loopback","monitor","desktop","steam","voicemeeter","remote"))
        if not device.is_physical_microphone or not physical or excluded:
            raise AudioBridgeError("Input must be an identifiable physical microphone, never cable/desktop/virtual capture")
    elif role == "playback":
        if not device.is_cable_input_playback:
            raise AudioBridgeError("Playback must be the explicitly selected VB-CABLE CABLE Input endpoint")
    elif role == "loopback" and not device.is_cable_output_recording:
        raise AudioBridgeError("Verification recording must be VB-CABLE CABLE Output")
    return device


class LiveBridge:
    def __init__(self, config: LiveConfig, *, api=None, devices=enumerate_devices, connect=None,
                 server=None, translated_audio_observer=None, _verify_production=True):
        self.config = config
        from ort.tts_controls import controls
        controls({"speed_factor":config.tts_speed,"fragment_interval":config.fragment_interval,
                  "trim_digital_silence":config.trim_digital_silence,"pronunciation_enabled":config.pronunciation_enabled})
        if config.source_language not in {"zh","ja","en","auto"}:
            raise ValueError("Invalid live source language")
        self.reference = validate_reference(config.reference_wav,config.reference_text,config.reference_language)
        if config.reference_language != "ja":
            raise AudioBridgeError("STEP 07 verifies Japanese static references only")
        self.url = local_url(config.url)
        if not .1 <= config.capture_seconds <= 10 or not .2 <= config.playback_seconds <= 30 or not 1 <= config.utterance_capacity <= 32:
            raise ValueError("Invalid live buffer budgets")
        if (type(config.utterance_capacity) is not int or type(config.input_channels) is not int
            or not 1 <= config.input_channels <= 8
            or not 0 <= config.jitter_ms < config.playback_seconds*1000
            or not all(math.isfinite(x) and x > 0 for x in (config.request_timeout,config.shutdown_timeout))
            or not math.isfinite(config.settle_seconds) or not 0 <= config.settle_seconds <= 2):
            raise ValueError("Invalid live timeout, channel or jitter settings")
        for rate in (config.input_rate,config.output_rate):
            if rate is not None and (type(rate) is not int or not 8000 <= rate <= 192000):
                raise ValueError("Device sample rate must be an integer from 8000 to 192000")
        self.api,self.devices,self.connect,self.server = api,devices,connect,server
        self.translated_audio_observer = translated_audio_observer
        self.verify_production = _verify_production
        self.state = State.STOPPED
        self.epoch = 0
        self.session_id = uuid.uuid4().hex
        self.next_id = 1
        self.traces = OrderedDict()
        self.total_utterances = self.evicted_traces = 0
        self.counts = {k:0 for k in ("capture_blocks","input_overflows","output_underflows","jitter_underflows",
            "queue_overflows","rejected_jobs","cancelled_jobs","cancelled_uncommitted","capture_callbacks","capture_frames",
            "reconnects","late_events","callback_events_lost")}
        self.events = []
        self.maxima = {"capture_blocks":0,"waiting_jobs":0,"playback_frames":0}
        self._tasks = []
        self._input = self._output = self._ws = None
        self._enabled = False
        self._fault = None
        self._expected_devices = None
        self.last_error = None
        self._callback_events = queue.Queue(maxsize=2048)
        self._lifecycle = asyncio.Lock()
        self._fault_task = None

    def preflight(self):
        if self.api is None:
            import sounddevice
            self.api = sounddevice
        devices = self.devices()
        mic = select_device(devices,self.config.input_device,"microphone")
        cable = select_device(devices,self.config.output_device,"playback")
        fingerprints = (mic.name,cable.name)
        if self._expected_devices is not None and self._expected_devices != fingerprints:
            raise AudioBridgeError("Device indices now identify different endpoints; explicitly select the intended devices again")
        self._expected_devices = fingerprints
        self.microphone,self.cable = mic,cable
        self.input_rate = int(self.config.input_rate or mic.default_samplerate)
        self.output_rate = int(self.config.output_rate or cable.default_samplerate)
        self.output_channels = min(2,cable.max_output_channels)
        self.input_frames,self.output_frames = round(self.input_rate*.01),round(self.output_rate*.01)
        if self.config.input_channels > mic.max_input_channels:
            raise AudioBridgeError("Microphone does not expose the requested channel count")
        self.api.check_input_settings(device=mic.index,samplerate=self.input_rate,channels=self.config.input_channels,dtype="float32")
        self.api.check_output_settings(device=cable.index,samplerate=self.output_rate,channels=self.output_channels,dtype="float32")
        return {"microphone":asdict(mic),"cable_playback":asdict(cable),"input_rate":self.input_rate,
                "output_rate":self.output_rate,"output_channels":self.output_channels}

    async def start(self):
        async with self._lifecycle:
            if self.state == State.RUNNING:
                return
            if self.state not in {State.STOPPED,State.PAUSED,State.ERROR}:
                raise AudioBridgeError(f"Cannot start while {self.state}")
            if self.state == State.ERROR:
                await self._deactivate(State.STOPPED,"explicit recovery",close_server=True)
            self.state = State.STARTING
            self.epoch += 1
            epoch = self.epoch
            self._fault = None
            self._fault_started_at = None
            self.startup_error = None
            try:
                # PortAudio/WASAPI initialization must occur on the stream-owner
                # thread, not inside the preflight executor. A cold background
                # import reproduces Pa_StartStream -9999 on native Windows.
                if self.api is None:
                    import sounddevice
                    self.api = sounddevice
                self.device_info = await asyncio.to_thread(self.preflight)
                if self.server and (not self.server.process or self.server.process.poll() is not None):
                    await self.server.start()
                if self.connect is None:
                    import websockets
                    self.connect = lambda url:websockets.connect(url,max_size=256*1024,max_queue=8,
                        open_timeout=600,close_timeout=.5,ping_interval=20,ping_timeout=20)
                if hasattr(self,"created"):
                    self.counts["reconnects"] += 1
                self._ws = await self.connect(self.url).__aenter__()
                first = await asyncio.wait_for(self._receive(),600)
                self._validate_created(first)
                self.created = first
                await self._ws.send(json.dumps({"type":"session.update","session":{"audio":{
                    "input":{"transcription":{"language":self.config.source_language}},"output":{"language":"ja"}}}}))
                updated = await asyncio.wait_for(self._receive(),5)
                if updated.get("type") != "session.updated":
                    raise AudioBridgeError(f"Language handshake failed: {updated}")
                self.capture = CaptureRing(max(2,round(self.config.capture_seconds/.01)),self.input_frames,
                                           self.config.input_channels,self.input_rate)
                self.playback = PlaybackRing(round(self.output_rate*self.config.playback_seconds),self.output_channels)
                self.segmenter = TimedSegmenter(self.session_id,epoch,mode=self.config.mode,
                    silence_ms=self.config.silence_ms,max_seconds=self.config.max_utterance_seconds,first_id=self.next_id,
                    native_rate=self.input_rate)
                self._jobs = asyncio.Queue(maxsize=self.config.utterance_capacity)
                self._inflight = None
                self._request_done = asyncio.Event()
                self._origin = None
                self._output_started = False
                self._streaming_audio = False
                tags = np.zeros(self.output_frames,dtype=np.int64)
                def input_callback(data,frames,info,status):
                    if self._enabled and self.epoch==epoch:
                        self.counts["capture_callbacks"] += 1
                        self.counts["capture_frames"] += frames
                        self.capture.callback(data,frames,info,status)
                def output_callback(data,frames,info,status):
                    data.fill(0)
                    if not self._enabled or self.epoch != epoch:
                        return
                    if frames != self.output_frames:
                        self._fault = "Output callback frame size changed"
                        return
                    if status:
                        self.counts["output_underflows"] += 1
                        self._fault = f"Output device status: {status}"
                        return
                    if not self._output_started:
                        if self.playback.depth == 0 or (self._streaming_audio and self.playback.depth < self.output_rate*self.config.jitter_ms/1000):
                            return
                        self._output_started = True
                    before = self.playback.read
                    count = self.playback.take(data,tags)
                    if count < frames and self._streaming_audio:
                        self.counts["jitter_underflows"] += 1
                    if count:
                        now = time.perf_counter()
                        try:
                            dac = now+float(info.outputBufferDacTime-info.currentTime)
                        except (AttributeError,TypeError):
                            dac = None
                        changes = np.r_[0,np.flatnonzero(np.diff(tags[:count]))+1,count]
                        for a,b in zip(changes[:-1],changes[1:]):
                            audible = np.flatnonzero(np.max(np.abs(data[a:b]),axis=1)>1e-4)
                            try:
                                self._callback_events.put_nowait((epoch,int(tags[a]),now,
                                    None if dac is None else dac+b/self.output_rate,before+int(b),
                                    before+int(a),int(a)+int(audible[0]) if len(audible) else None))
                            except queue.Full:
                                self.counts["callback_events_lost"] += 1
                                self._fault = "Callback metrics ring overflow"
                self._output = self.api.OutputStream(device=self.cable.index,samplerate=self.output_rate,
                    channels=self.output_channels,dtype="float32",blocksize=self.output_frames,latency="low",
                    callback=output_callback)
                self._input = self.api.InputStream(device=self.microphone.index,samplerate=self.input_rate,
                    channels=self.config.input_channels,dtype="float32",blocksize=self.input_frames,latency="low",
                    callback=input_callback)
                self.output_latency = float(getattr(self._output,"latency",0))
                self._enabled = True
                self.state = State.RUNNING
                self._output.start()
                self._input.start()
                self.last_error = None
                self._tasks = [asyncio.create_task(self._guard(fn(epoch)),name=name) for name,fn in (
                    ("live-vad",self._capture_worker),("live-send",self._sender),
                    ("live-receive",self._receiver),("live-monitor",self._monitor))]
                self.events.append({"state":"RUNNING","epoch":epoch,"at":time.perf_counter()})
            except BaseException as exc:
                self.startup_error = {"type":type(exc).__name__,"message":str(exc),"native_args":repr(exc.args)}
                self.last_error = str(exc)
                await self._deactivate(State.ERROR,str(exc),close_server=bool(self.server))
                raise

    def _validate_created(self, event):
        if event.get("type") != "session.created":
            raise AudioBridgeError(f"Server did not create a session: {event}")
        profile = event.get("session",{}).get("voicebridge",{})
        if profile.get('tts_voice_preset',{}):
            policy=profile['tts_voice_preset']['playback']
            if policy['method']=='browser_preserves_pitch':
                raise AudioBridgeError('This personal preset requires browser playbackRate1.25 with preservesPitch. Native CABLE playback cannot reproduce that policy; no native-speed or WSOLA substitution was made. Use its approved browser/WAV player or the unchanged existing production profile.')
        audio = event.get("session",{}).get("audio",{}).get("output",{})
        if audio.get("sample_rate") != 24000 or audio.get("format") != "pcm16" or audio.get("language") != "ja":
            raise AudioBridgeError("Server output is not the required Japanese 24 kHz PCM16 contract")
        if profile.get("version") != 1 or profile.get("input_mode") != "client_committed":
            raise AudioBridgeError("Server has not negotiated client-owned VAD; refusing double segmentation")
        if self.verify_production:
            actual = (profile.get("asr_backend"),profile.get("asr_model"),profile.get("translator_backend"),
                      profile.get("translator_model"),profile.get("tts_backend"),profile.get("tts_version"))
            expected = ("qwen3","Qwen/Qwen3-ASR-0.6B","opusmt","Helsinki-NLP/opus-mt-tc-big-zh-ja","gptsovits","v2ProPlus")
            if (actual != expected or profile.get("reference_fingerprint") != self.reference.fingerprint
                or not str(profile.get("asr_device","")).startswith("cuda") or profile.get("asr_attention")!="sdpa"):
                raise AudioBridgeError("Actual server production models/static reference differ from the selected settings")

    async def _receive(self):
        raw = await self._ws.recv()
        value = json.loads(raw)
        if not isinstance(value,dict):
            raise AudioBridgeError("WebSocket event is not an object")
        return value

    async def _guard(self, coroutine):
        try:
            await coroutine
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.state not in {State.RUNNING,State.STARTING}:
                return  # A locally requested socket close is not a worker fault.
            self._enabled = False
            self.last_error = str(exc) or type(exc).__name__
            self._fault = self.last_error
            self._fault_started_at = time.perf_counter()
            if not self._fault_task or self._fault_task.done():
                self._fault_task = asyncio.create_task(self._fail(self.last_error))

    async def _fail(self, reason):
        async with self._lifecycle:
            if self.state in {State.RUNNING,State.STARTING}:
                await self._deactivate(State.ERROR,reason,close_server=bool(self.server))

    def _trace(self, job):
        while len(self.traces) >= 512:
            _,old = self.traces.popitem(last=False)
            if old.get("status") not in {"played","cancelled","rejected"}:
                raise AudioBridgeError("Active trace budget exceeded")
            self.evicted_traces += 1
        row = {"id":job.context.utterance_id,"context":job.context.wire(),"status":"committed",
            "admitted_at":time.perf_counter(),"input_sha256":hashlib.sha256(job.samples.tobytes()).hexdigest(),
            "source_text":None,"japanese_text":None,"loopback_first_non_silent_at":None,
            "physical_playback_completion_at":None}
        self.traces[row["id"]] = row
        self.total_utterances += 1
        return row

    async def _capture_worker(self, epoch):
        converter = soxr.ResampleStream(self.input_rate,16000,1,dtype="float32")
        while self.epoch==epoch and self.state==State.RUNNING:
            item = self.capture.pop()
            if item is None:
                await asyncio.sleep(.002)
                continue
            block,adc,position,raw_adc = item
            if self._origin is None:
                self._origin = adc-position/self.input_rate
                self.segmenter.origin_adc_at = None if raw_adc is None else raw_adc-position/self.input_rate
            self.counts["capture_blocks"] += 1
            mono = block.mean(axis=1,dtype=np.float32)
            if not np.isfinite(mono).all():
                raise AudioBridgeError("Microphone returned non-finite samples")
            internal = converter.resample_chunk(mono,last=False)
            for job in self.segmenter.feed(internal,origin_at=self._origin):
                self.next_id = max(self.next_id,job.context.utterance_id+1)
                row = self._trace(job)
                try:
                    self._jobs.put_nowait(job)
                    self.maxima["waiting_jobs"] = max(self.maxima["waiting_jobs"],self._jobs.qsize())
                except asyncio.QueueFull:
                    row["status"] = "rejected"
                    self.counts["queue_overflows"] += 1
                    self.counts["rejected_jobs"] += 1
                    raise AudioBridgeError(f"Utterance queue saturated; rejected ID {row['id']}; output muted")

    async def _sender(self, epoch):
        from ort.tts_controls import AdaptiveSpeed, controls
        policy = AdaptiveSpeed()
        while self.epoch==epoch and self.state==State.RUNNING:
            job = await self._jobs.get()
            self._inflight = job
            self._request_done.clear()
            row = self.traces[job.context.utterance_id]
            row.update(status="inference",websocket_send_at=time.perf_counter())
            row["capture_callbacks_at_send"] = self.counts["capture_callbacks"]
            row["queue_wait_seconds"] = row["websocket_send_at"]-row["admitted_at"]
            from .quick_reply import QuickJob
            if isinstance(job,QuickJob):
                request=dict(type='voicebridge.quick_reply',voicebridge=job.context.wire(),
                    text=job.text,settings=job.settings,preset=job.preset)
                if job.shortcut_id is not None:request['shortcut_id']=job.shortcut_id
                await self._ws.send(json.dumps(request))
                row.update(websocket_commit_sent_at=time.perf_counter(),request_kind='QUICK_REPLY')
                await asyncio.wait_for(self._request_done.wait(),self.config.request_timeout)
                self._inflight=None;self._jobs.task_done();continue
            wire = await asyncio.to_thread(resample,job.samples,16000,24000)
            for offset in range(0,len(wire),2400):
                await self._ws.send(json.dumps({"type":"input_audio_buffer.append",
                    "audio":float32_to_b64_pcm16(wire[offset:offset+2400])}))
            backlog = self.playback.depth/self.output_rate
            options = controls({"speed_factor":max(self.config.tts_speed,policy.choose(backlog)) if self.config.adaptive_speed else self.config.tts_speed,
                "fragment_interval":self.config.fragment_interval,"trim_digital_silence":self.config.trim_digital_silence,
                "pronunciation_enabled":self.config.pronunciation_enabled})
            pending_rows = [r for r in self.traces.values() if r.get("status")=="committed"]
            row.update(playback_seconds_at_submission=backlog,tts_controls=options,
                pending_audio={"synthesized_ring_seconds":backlog,
                    "device_latency_hint_seconds":getattr(self,"output_latency",None),
                    "unsynthesized_waiting_jobs":self._jobs.qsize(),"unsynthesized_current_jobs":1,
                    "waiting_source_audio_seconds":sum((r["context"]["audio_end_sample"]-r["context"]["audio_start_sample"])/16000 for r in pending_rows),
                    "unsynthesized_japanese_seconds":None})
            await self._ws.send(json.dumps({"type":"input_audio_buffer.commit","voicebridge":job.context.wire(),"tts_options":options}))
            row["websocket_commit_sent_at"] = time.perf_counter()
            await asyncio.wait_for(self._request_done.wait(),self.config.request_timeout)
            self._inflight = None
            self._jobs.task_done()

    async def _receiver(self, epoch):
        current = None
        converter = None
        sequence = 0
        samples = 0
        finalized = False
        while self.epoch==epoch and self.state==State.RUNNING:
            event = await self._receive()
            if self.epoch != epoch:
                self.counts["late_events"] += 1
                return
            kind = event.get("type")
            if kind == "error":
                handler=getattr(self,'_handle_server_error',None)
                if handler and handler(event,epoch):
                    current=None;converter=None;sequence=samples=0;finalized=False
                    continue
                raise AudioBridgeError(f"Production server error: {event.get('error')}")
            if kind not in {"session.input_transcript.delta","session.output_transcript.delta",
                            "session.output_transcript.done","session.output_audio.delta","session.output_audio.done"}:
                raise AudioBridgeError(f"Unexpected live event: {kind}")
            metadata = event.get("voicebridge",{})
            context = metadata.get("context",{})
            job = self._inflight
            if job is None or context != job.context.wire():
                raise AudioBridgeError("Stale, reordered or uncorrelated live response")
            identifier = job.context.utterance_id
            row = self.traces[identifier]
            row.setdefault("stage_times",{}).update(metadata.get("stage_times",{}))
            if current != identifier:
                current = identifier
                converter = soxr.ResampleStream(24000,self.output_rate,1,dtype="float32")
                sequence = samples = 0
                finalized = False
            if finalized:
                raise AudioBridgeError("Duplicate response after utterance completion")
            if kind == "session.input_transcript.delta":
                if metadata.get("transcript_phase") != "final" or row["source_text"] is not None:
                    raise AudioBridgeError("Provisional/duplicate ASR text cannot enter translation/playback")
                row["source_text"] = event.get("delta")
            elif kind == "session.output_transcript.delta":
                from .quick_reply import QuickJob
                if (not row["source_text"] and not isinstance(job,QuickJob)) or row["japanese_text"] is not None:
                    raise AudioBridgeError("Japanese translation arrived before final ASR or was duplicated")
                row["japanese_text"] = event.get("delta")
            elif kind == "session.output_transcript.done":
                if event.get("transcript") != row["japanese_text"] or not row["japanese_text"]:
                    raise AudioBridgeError("Japanese transcript completion mismatch")
                row["translation_confirmed"] = True
            elif kind == "session.output_audio.delta":
                if not row.get("translation_confirmed") or metadata.get("chunk_sequence") != sequence:
                    raise AudioBridgeError("Untranslated or reordered PCM event; output refused")
                mono = pcm24k_b64_to_float32(event.get("delta",""))
                raw = base64.b64decode(event["delta"],validate=True)
                if (metadata.get("sample_rate"),metadata.get("channels"),metadata.get("pcm_format"),metadata.get("samples")) != (24000,1,"s16le",mono.size):
                    raise AudioBridgeError("Translated PCM sample rate/channels/length metadata mismatch")
                if hashlib.sha256(raw).hexdigest() != metadata.get("pcm_sha256") or (raw[:4]==b"RIFF" and raw[8:12]==b"WAVE"):
                    raise AudioBridgeError("Translated PCM checksum mismatch or WAV header in raw PCM")
                sequence += 1
                samples += mono.size
                row.setdefault("first_validated_pcm_at",time.perf_counter())
                self._streaming_audio = True
                await self._enqueue_playback(converter.resample_chunk(mono,last=False),identifier,epoch)
            elif kind == "session.output_audio.done":
                if not sequence or metadata.get("chunk_count") != sequence:
                    raise AudioBridgeError("Missing, duplicated or mismatched PCM completion")
                await self._enqueue_playback(converter.resample_chunk(np.zeros(0,dtype=np.float32),last=True),identifier,epoch)
                row.update(status="buffered",server_audio_done_at=time.perf_counter(),wire_samples=samples,
                           playback_end_frame=self.playback.written,tts_metrics=metadata.get("tts_metrics",{}),
                           capture_callbacks_at_done=self.counts["capture_callbacks"])
                await self._complete_output(identifier,epoch)
                finalized = True
                self._streaming_audio = False
                self._request_done.set()

    async def _complete_output(self, identifier, epoch):
        """Optional application processing after strict protocol completion."""
        return

    async def _enqueue_playback(self, mono, identifier, epoch):
        if not mono.size:
            return
        samples = np.repeat(mono[:,None],self.output_channels,axis=1)
        # One wire event is bounded at 256 KiB; do not grow this buffer on full.
        if len(samples) > len(self.playback.data):
            raise AudioBridgeError("Translated chunk exceeds playback memory budget")
        deadline = time.monotonic()+self.config.request_timeout
        while self.epoch==epoch and self.state==State.RUNNING:
            if self.playback.push(samples,identifier):
                if self.translated_audio_observer:
                    self.translated_audio_observer(identifier,mono.copy(),self.output_rate)
                return
            if time.monotonic() >= deadline:
                raise AudioBridgeError("Playback backpressure deadline exceeded")
            await asyncio.sleep(.005)

    async def _monitor(self, epoch):
        last_check = time.monotonic()
        while self.epoch==epoch and self.state==State.RUNNING:
            if self.capture.fault or self._fault:
                raise AudioBridgeError(self.capture.fault or self._fault)
            if not self._input.active or not self._output.active:
                raise AudioBridgeError("Selected audio device disconnected/stopped; explicit recovery required")
            self.maxima["capture_blocks"] = max(self.maxima["capture_blocks"],self.capture.max_depth)
            self.maxima["playback_frames"] = max(self.maxima["playback_frames"],self.playback.max_depth)
            while True:
                try:
                    event_epoch,identifier,accepted,dac_end,end_frame,start_frame,audible_offset = self._callback_events.get_nowait()
                except queue.Empty:
                    break
                row = self.traces.get(identifier)
                if event_epoch != epoch or row is None:
                    self.counts["late_events"] += 1
                    continue
                row.setdefault("first_output_callback_at",accepted)
                row.setdefault("first_output_frame",start_frame)
                if audible_offset is not None:
                    row.setdefault("first_non_silent_output_callback_at",accepted)
                    row.setdefault("first_non_silent_output_sample_offset",audible_offset)
                row.update(last_output_callback_at=accepted,scheduled_dac_end_at=dac_end,
                           last_callback_frame=end_frame)
            for row in self.traces.values():
                if row.get("status")=="buffered" and row.get("last_callback_frame",-1)>=row["playback_end_frame"]:
                    row["status"] = "played"
                    row["callback_playback_complete_at"] = row["last_output_callback_at"]
            if time.monotonic()-last_check >= 1:
                last_check = time.monotonic()
                devices = await asyncio.to_thread(self.devices)
                for original in (self.microphone,self.cable):
                    current = next((d for d in devices if d.index==original.index),None)
                    if current != original:
                        raise AudioBridgeError("Selected endpoint disappeared or its format/identity changed")
            await asyncio.sleep(.005)

    async def pause(self):
        async with self._lifecycle:
            if self.state == State.RUNNING:
                await self._deactivate(State.PAUSED,"pause",close_server=False)

    async def resume(self):
        if self.state != State.PAUSED:
            raise AudioBridgeError("Resume requires PAUSED")
        await self.start()

    async def stop(self):
        async with self._lifecycle:
            if self.state==State.STOPPED and not self._tasks and not self._input and not self._output and not self._ws:
                if self.server:
                    await asyncio.to_thread(self.server.close)
                return
            await self._deactivate(State.STOPPED,"stop",close_server=True)

    async def _deactivate(self, final_state, reason, *, close_server):
        requested = time.perf_counter()
        self._enabled = False
        old_epoch = self.epoch
        old_capture = getattr(self,"capture",None)
        discarded_blocks = (old_capture.written-old_capture.read+old_capture.dropped) if old_capture else 0
        if old_capture:
            self.counts["input_overflows"] += old_capture.dropped
            self.maxima["capture_blocks"] = max(self.maxima["capture_blocks"],old_capture.max_depth)
        affected = [row["id"] for row in self.traces.values() if row["context"]["epoch"]==old_epoch and row["status"] not in {"played","cancelled","rejected"}]
        pending = getattr(self,"segmenter",None)
        if pending is not None and pending.active:
            affected.append(pending.next_id)
            self.next_id = max(self.next_id,pending.next_id+1)
            self.counts["cancelled_uncommitted"] += 1
            pending.active = False
        self.epoch += 1
        self.state = State.STOPPING
        errors = []
        for stream in (self._output,self._input):
            if stream:
                for operation in (stream.abort,stream.close):
                    try:
                        await asyncio.wait_for(asyncio.to_thread(operation),self.config.shutdown_timeout)
                    except Exception as exc:
                        errors.append(str(exc))
        self._input = self._output = None
        if self._ws:
            try:
                await asyncio.wait_for(self._ws.close(),self.config.shutdown_timeout)
            except Exception as exc:
                errors.append(str(exc))
            self._ws = None
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks,return_exceptions=True)
        self._tasks.clear()
        for row in self.traces.values():
            if row["context"]["epoch"] == old_epoch and row["status"] not in {"played","cancelled","rejected"}:
                row.update(status="cancelled",cancelled_at=time.perf_counter(),cancellation_reason=reason)
                self.counts["cancelled_jobs"] += 1
        if hasattr(self,"_jobs"):
            while not self._jobs.empty():
                self._jobs.get_nowait()
        # Retire, never reset rings that an old callback might still reference.
        if hasattr(self,"playback"):
            self.playback = PlaybackRing(len(self.playback.data),self.output_channels)
        if hasattr(self,"capture"):
            self.capture = CaptureRing(len(self.capture.data),self.input_frames,self.config.input_channels,self.input_rate)
        while not self._callback_events.empty():
            self._callback_events.get_nowait()
        if close_server and self.server:
            await asyncio.to_thread(self.server.close)
        await asyncio.sleep(self.config.settle_seconds)
        self.state = State.ERROR if errors else final_state
        if errors:
            self.last_error = "; ".join(errors)
        self.events.append({"state":self.state.value,"epoch":old_epoch,"reason":reason,
            "affected_utterance_ids":affected,"capture_discarded_blocks":discarded_blocks,
            "capture_discarded_seconds":discarded_blocks*.01,
            "fault_to_mute_seconds":requested-self._fault_started_at if getattr(self,"_fault_started_at",None) else None,
            "stop_requested_at":requested,"stop_returned_at":time.perf_counter(),"stream_errors":errors})
        while len(self.events)>256:
            self.events.pop(0)

    def report(self):
        rows = list(self.traces.values())
        values = {"speech_end_to_vad_commit":[],"vad_commit_to_first_pcm":[],"speech_end_to_output_callback":[],
                  "speech_end_to_first_pcm":[],"speech_end_to_loopback":[],"queue_wait":[],
                  "asr":[],"translation":[],"tts_first_parent_pcm":[],"tts_complete":[],
                  "processing_to_first_pcm":[],"queue_inclusive_processing":[]}
        for row in rows:
            context = row["context"]
            stages = row.get("stage_times",{})
            pairs = [("speech_end_to_vad_commit",context.get("vad_commit_at"),context.get("speech_end_at")),
                ("vad_commit_to_first_pcm",row.get("first_validated_pcm_at"),context.get("vad_commit_at")),
                ("speech_end_to_output_callback",row.get("first_output_callback_at"),context.get("speech_end_at")),
                ("speech_end_to_first_pcm",row.get("first_validated_pcm_at"),context.get("speech_end_at")),
                ("speech_end_to_loopback",row.get("loopback_first_non_silent_at"),context.get("speech_end_at")),
                ("asr",stages.get("asr_end"),stages.get("asr_start")),
                ("translation",stages.get("mt_end"),stages.get("mt_start")),
                ("tts_first_parent_pcm",row.get("first_validated_pcm_at"),stages.get("tts_start")),
                ("tts_complete",stages.get("tts_end"),stages.get("tts_start")),
                ("processing_to_first_pcm",row.get("first_validated_pcm_at"),stages.get("asr_start")),
                ("queue_inclusive_processing",row.get("server_audio_done_at"),row.get("admitted_at"))]
            for key,end,start in pairs:
                if end is not None and start is not None:
                    values[key].append(end-start)
            if "queue_wait_seconds" in row:
                values["queue_wait"].append(row["queue_wait_seconds"])
            if row.get("wire_samples"):
                row["output_audio_seconds"] = row["wire_samples"]/24000
                if stages.get("tts_end") is not None and stages.get("tts_start") is not None:
                    row["tts_rtf"] = (stages["tts_end"]-stages["tts_start"])/row["output_audio_seconds"]
        summaries = {key:{"n":len(v),"min":min(v) if v else None,"max":max(v) if v else None,
            "p50":float(np.percentile(v,50)) if v else None,"p95":float(np.percentile(v,95)) if v else None}
            for key,v in values.items()}
        return {"state":self.state.value,"last_error":self.last_error,"session_id":self.session_id,
            "startup_error":getattr(self,"startup_error",None),
            "total_utterances":self.total_utterances,"traces_retained":len(rows),"traces_evicted":self.evicted_traces,
            "counts":self.counts.copy(),"maxima":self.maxima.copy(),"latencies_seconds":summaries,
            "traces":rows,"lifecycle":self.events.copy(),"config":{k:v for k,v in asdict(self.config).items() if k!="reference_text"},
            "device_info":getattr(self,"device_info",None),"server_profile":getattr(self,"created",{}),
            "output_reported_latency_seconds":getattr(self,"output_latency",None),
            "asr_recognition":"final utterance Qwen3-ASR; incremental streaming is not implemented",
            "physical_timing":"ADC frame-mapped speech-end estimate at 30 ms resolution; callback acceptance is measured; DAC timing is scheduled, not independently observed. Loopback is null unless independently measured.",
            "hardware_acceptance":"PENDING unless a physical microphone + independent translated cable recording is verified"}
