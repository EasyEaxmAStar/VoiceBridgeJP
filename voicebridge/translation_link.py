"""Reuse the existing final-only VAD/WS protocol validator without owning audio devices."""
import asyncio,hashlib,json,time
import numpy as np
import soxr
from .live import LiveBridge,State,AudioBridgeError,local_url
from .live_segmentation import TimedSegmenter
from .audio_processing import TempoStream


class TranslationLink(LiveBridge):
    WAITING_CAPACITY=16
    def __init__(self,engine,generation):
        super().__init__(engine.config,_verify_production=True)
        self.engine=engine;self.epoch=generation;self.state=State.RUNNING
        self.session_id=engine.session_id;self.next_id=engine.next_id
        self.output_rate,self.output_channels=engine.output_rate,engine.output_channels
        self.playback=engine.snapshot.playback
        self._jobs=asyncio.Queue(maxsize=self.WAITING_CAPACITY);self._inflight=None
        self._request_done=asyncio.Event();self._ws=None;self._parts={};self._tasks=[];self._origin=None
        self.segmenter=TimedSegmenter(self.session_id,generation,mode=self.config.mode,silence_ms=self.config.silence_ms,
            max_seconds=self.config.max_utterance_seconds,first_id=engine.next_id,native_rate=engine.input_rate)
        if getattr(engine,'feature_settings',None):self.segmenter.configure_next(engine.feature_settings['segmentation'])
        if getattr(engine,'feature_settings',None):self.segmenter.configure_boundary(engine.feature_settings['boundary'])
        self.speed_by_id={};self._converters={};self._request_rates={};self.connected=False
        self.processors={}
    def _handle_server_error(self,event,epoch):
        error_type=event.get('error',{}).get('type')
        if error_type not in {'translation_rejected','utterance_failed_private'}:return False
        job=self._inflight;metadata=event.get('voicebridge',{})
        if job is None or metadata.get('context')!=job.context.wire() or self.engine.snapshot.generation!=epoch:
            raise AudioBridgeError('Uncorrelated translation rejection')
        identifier=job.context.utterance_id;row=self.traces[identifier]
        if identifier in self.processors or row.get('first_validated_pcm_at'):
            raise AudioBridgeError('Translation rejected after PCM had begun')
        row.update(status='translation_rejected' if error_type=='translation_rejected' else 'technical_error',stage_times=metadata.get('stage_times',{}),warning='UNRESOLVED_PRIVATE：本句仅本地诊断，无输出音频')
        row['policy_outcome']=row['stage_times'].get('translation_policy',{}).get('outcome','UNRESOLVED_PRIVATE' if error_type=='translation_rejected' else 'TECHNICAL_ERROR')
        if row['policy_outcome']=='FILLER_SKIPPED':row['status']='filler_skipped';self.engine.counts['filler_skipped']+=1
        self.engine.completed.append(row);self.engine.counts['unsafe_translation_rejections']+=1
        camera=getattr(self.engine,'camera',None)
        if camera:camera.job_failed(job.context)
        self.speed_by_id.pop(identifier,None);self._request_done.set()
        return True
    async def run(self):
        import websockets
        try:
            self._ws=await websockets.connect(self.url,max_size=256*1024,max_queue=8,open_timeout=180,close_timeout=.5)
            created=await asyncio.wait_for(self._receive(),180)
            actual=created['session']['voicebridge'];expected=self.engine.runtime['ready']
            voice=actual['tts_voice_preset']
            if (voice['profile_sha256']!=expected['voice_preset']['profile_sha256'] or
                voice['prosody_level']!=expected['voice_preset']['prosody_level'] or
                actual['tts_worker_pid']!=expected['pid']):
                raise AudioBridgeError('Actual personal voice/level/worker differs from selected release')
            policy=voice['playback']
            if policy['method']!='ffmpeg_atempo':raise AudioBridgeError('Release requires explicit reversible atempo policy')
            # Validate all original model/reference identities using the existing
            # production check; only the separately acknowledged playback differs.
            check=json.loads(json.dumps(created));check['session']['voicebridge'].pop('tts_voice_preset')
            self._validate_created(check);self.created=created
            await self._ws.send(json.dumps(dict(type='session.update',session=dict(voicebridge_playback=policy,
                audio=dict(input=dict(transcription=dict(language='zh')),output=dict(language='ja'))))))
            if (await self._receive())['type']!='session.updated':raise AudioBridgeError('Release playback handshake failed')
            if self.engine.snapshot.generation!=self.epoch:return
            self.connected=True
            self._tasks=[asyncio.create_task(self._sender(self.epoch)),asyncio.create_task(self._receiver(self.epoch))]
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:raise
        except Exception as exc:
            if self.engine.snapshot.generation==self.epoch:await self.engine.fail(str(exc))
        finally:
            self.connected=False
            for task in self._tasks:task.cancel()
            if self._tasks:await asyncio.gather(*self._tasks,return_exceptions=True)
            for processor in list(self.processors.values()):await processor.cancel()
            self.processors.clear()
            if self._ws:await self._ws.close()
    def admit(self,job):
        # Check before publishing metadata/traces; rejection never admits a
        # partial job or retains its audio. No await between check and put.
        if self._jobs.full():raise asyncio.QueueFull
        camera=getattr(self.engine,'camera',None)
        if camera:
            from .quick_reply import QuickJob,validate_shortcut
            from .camera_events import SourceDescriptor,SourceKind
            if isinstance(job,QuickJob):validate_shortcut(job.text,job.shortcut_id)
            camera.register(job.context,SourceDescriptor(SourceKind.QUICK_REPLY,job.shortcut_id)
                if isinstance(job,QuickJob) else SourceDescriptor(SourceKind.ASR_UTTERANCE))
        row=self._trace(job)
        speed=self.engine.speed
        if self.engine.adaptive:
            from ort.tts_controls import AdaptiveSpeed
            if not hasattr(self,'_speed_policy'):self._speed_policy=AdaptiveSpeed()
            speed=max(speed,self._speed_policy.choose(self.engine.snapshot.playback.depth/self.output_rate))
        self.speed_by_id[job.context.utterance_id]=speed
        row.update(output_speed_frozen=speed,route_generation=self.epoch,boundary_cause=job.boundary_cause,boundary_requested_at=job.boundary_requested_at)
        self._jobs.put_nowait(job)
    async def _enqueue_playback(self,mono,identifier,epoch):
        # Existing receiver validates rate, sequence, sample count, transcript
        # and checksum for each chunk. Final completion is checked separately
        # before _complete_output; playback can already contain valid chunks.
        if len(mono):
            if identifier not in self.processors:
                async def sink(samples):
                    if not await self.engine.enqueue(samples,identifier,epoch):raise asyncio.CancelledError
                processor=TempoStream(self.output_rate,self.speed_by_id[identifier],sink)
                self.processors[identifier]=processor;await processor.start()
            await self.processors[identifier].add(mono)
    async def _complete_output(self,identifier,epoch):
        job=self._inflight
        if job is None:raise AudioBridgeError('Missing completed utterance')
        processor=self.processors.pop(identifier,None)
        if processor is None:raise AudioBridgeError('No validated PCM to process')
        try:original,processed,metrics=await processor.finish()
        finally:await processor.cancel()
        if self.engine.snapshot.generation!=epoch:return
        self.engine._completed_source.add(identifier)
        self.speed_by_id.pop(identifier)
        row=self.traces[identifier];row.update(processing=metrics,processed_pcm_at=time.perf_counter())
        from .quick_reply import QuickJob
        import os
        evidence=os.environ.get('VOICEBRIDGE_QUICK_EVIDENCE')
        if evidence and isinstance(job,QuickJob):
            from pathlib import Path
            import soundfile as sf
            folder=Path(evidence);folder.mkdir(parents=True,exist_ok=True)
            await asyncio.to_thread(sf.write,folder/f'quick-{identifier}.wav',processed,self.output_rate,subtype='PCM_16')
            (folder/f'quick-{identifier}.json').write_text(json.dumps(dict(text=job.text,preset=job.preset,settings=job.settings,
                tts=row.get('tts_metrics'),processing=metrics),ensure_ascii=False,indent=2),encoding='utf8')
        if self.engine.audio_observer:self.engine.audio_observer('TRANSLATE',epoch,identifier,original,processed,self.output_rate)
        row.update(playback_end_frame=self.engine.snapshot.playback.written,status='buffered')
        camera=getattr(self.engine,'camera',None)
        if camera:camera.complete(epoch,identifier,row['playback_end_frame'])
        self.engine.completed.append(row)
    async def close_link(self):
        self.state=State.STOPPING
        for row in self.traces.values():
            if row.get('status') not in {'played','translation_rejected','technical_error'}:row.update(status='cancelled',policy_outcome='CANCELLED/STALE',cancelled_at=time.perf_counter())
        finishing=getattr(self,'_finishing',None)
        if finishing:finishing.cancel();await asyncio.gather(finishing,return_exceptions=True)
        for task in self._tasks:task.cancel()
        for processor in list(self.processors.values()):await processor.cancel()
        self.processors.clear()
        if self._ws:await self._ws.close()
        while not self._jobs.empty():self._jobs.get_nowait()
        self.speed_by_id.clear()

