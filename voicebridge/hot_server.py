"""One owned localhost real-model runtime. No silent warmup and no idle inference."""
import argparse,asyncio,copy,json,os,socket,time,ipaddress
from pathlib import Path

def install_qwen_lifespan(app,qwen,snapshot):
    from contextlib import asynccontextmanager
    original=app.router.lifespan_context
    app.state.voicebridge_qwen_owner=qwen
    @asynccontextmanager
    async def lifespan(application):
        async with original(application):
            qwen.task=asyncio.create_task(qwen.start(snapshot(),getattr(getattr(app.state,'pipeline',None),'voicebridge_gpu',app.state.live_lock)))
            try:yield
            finally:
                try:await app.state.voicebridge_qwen_owner.close()
                finally:
                    publisher=getattr(getattr(app.state,'pipeline',None),'voicebridge_camera_publisher',None)
                    if publisher:publisher.close()
    app.router.lifespan_context=lifespan


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--instance',required=True)
    p.add_argument('--gate',type=Path,required=True);args=p.parse_args()
    deadline=time.monotonic()+10
    while not args.gate.exists():
        if time.monotonic()>deadline:raise TimeoutError('Parent ownership gate missing')
        time.sleep(.01)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    original,original_ex=socket.socket.connect,socket.socket.connect_ex
    def guarded(fn):
        def call(stream,address):
            try:allowed=isinstance(address,tuple) and (address[0]=='localhost' or ipaddress.ip_address(address[0]).is_loopback)
            except ValueError:allowed=False
            if not allowed:raise RuntimeError('Offline runtime permits loopback only')
            return fn(stream,address)
        return call
    socket.socket.connect,socket.socket.connect_ex=guarded(original),guarded(original_ex)
    from ort import inference
    from ort.server import create_app
    from ort.wav_pipeline import load_production
    from ort.asr_audio import load_asr_wav
    from .release_voices import ROOT,DEFAULT,NAMES,configuration,ReleaseTTS
    import numpy as np
    import psutil,torch,uvicorn
    name=os.environ.get('VOICEBRIDGE_RELEASE_PRESET',DEFAULT);cfg=configuration(name)
    status=dict(state='INITIALIZING',model_loads=0,worker_restarts=0,warmup_runs=0,pid=os.getpid(),
        release_preset=name,phases=[],transmitting=False)
    status_path=ROOT/'.model-cache/hot-runtime'/f'{args.instance}.json';status_path.parent.mkdir(exist_ok=True)
    started=time.perf_counter();pipeline=None
    def publish(state):
        status['state']=state;status['phases'].append(dict(state=state,at=time.perf_counter()))
        status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf8')
    def initialize():
        nonlocal pipeline
        publish('LOADING_MODELS');t=time.perf_counter();pipeline=load_production(cfg)
        if os.environ.get('VOICEBRIDGE_CAMERA_PIPE'):
            from .camera_ipc import ASRPublisher
            pipeline.voicebridge_camera_publisher=ASRPublisher()
        from .production_policy import ProductionPolicy
        policy=ProductionPolicy()
        from ort.asr_vocabulary import NameVocabulary,with_dictionary
        pipeline.voicebridge_asr_names=NameVocabulary()
        policy.asr_names=pipeline.voicebridge_asr_names
        pipeline.voicebridge_asr_context=lambda snapshot:with_dictionary(pipeline.voicebridge_asr_names.snapshot(),snapshot['dictionary'],snapshot['sha256'])
        pipeline.asr.set_context_provider(lambda:pipeline.voicebridge_asr_context(policy.snapshot()))
        pipeline.voicebridge_policy=policy
        pipeline.tts=ReleaseTTS(pipeline.tts,name,policy=policy);status.update(model_loads=1,initialization_seconds=time.perf_counter()-t)
        publish('WARMING_UP');t=time.perf_counter();torch.cuda.synchronize(0)
        fixture=load_asr_wav(Path(cfg.gptsovits_ref_audio),strict=True)
        a=time.perf_counter();source=pipeline.asr.transcribe(fixture.samples,'ja');torch.cuda.synchronize(0)
        if not source.text.strip():raise RuntimeError('Real ASR speech-fixture warmup failed')
        asr_seconds=time.perf_counter()-a
        a=time.perf_counter();translated=pipeline.translator.translate('今天下午三点开会。',(), 'zh','ja')
        if not translated or not any('\u3040'<=c<='\u9fff' for c in translated):raise RuntimeError('Real CPU MT warmup failed')
        mt_seconds=time.perf_counter()-a;a=time.perf_counter()
        chunks=list(pipeline.tts.synthesize_stream('こんにちは。今日はよろしくお願いします。','ja',sampling={'seed':42}))
        x=np.concatenate([c.samples for c in chunks]) if chunks else np.zeros(0)
        if not len(x) or not np.isfinite(x).all() or np.sqrt(np.mean(x.astype(float)**2))<1e-4:raise RuntimeError('Real personal TTS warmup failed')
        torch.cuda.synchronize(0)
        status.update(warmup_runs=1,warmup_seconds=time.perf_counter()-t,cold_start_seconds=time.perf_counter()-started,
            warmup=dict(asr_seconds=asr_seconds,asr_text=source.text,mt_seconds=mt_seconds,mt_text=translated,
                tts_seconds=time.perf_counter()-a,tts_samples=len(x),audio_discarded=True),
            ready=copy.deepcopy(pipeline.tts.ready),model_ids=dict(asr=id(pipeline.asr),mt=id(pipeline.translator),tts_pid=pipeline.tts.ready['pid']))
        publish('STANDBY_READY');return pipeline
    try:inference.run_sync(initialize)
    except BaseException as exc:
        status.update(error=f'{type(exc).__name__}: {exc}');publish('ERROR')
        if pipeline:
            pipeline.tts.close()
            publisher=getattr(pipeline,'voicebridge_camera_publisher',None)
            if publisher:publisher.close()
        raise
    app=create_app(cfg);app.state.pipeline=pipeline;app.state.voicebridge_instance=args.instance
    pipeline.voicebridge_gpu=asyncio.Lock()
    from .warm_qwen import WarmQwen
    qwen=WarmQwen(ROOT/'.model-cache/hot-runtime'/args.instance)
    pipeline.voicebridge_qwen=qwen
    # create_app already owns a lifespan. FastAPI ignores on_event hooks when
    # a lifespan is supplied, so compose it rather than registering dead hooks.
    install_qwen_lifespan(app,qwen,pipeline.voicebridge_policy.snapshot)
    from fastapi import HTTPException
    @app.get('/runtime')
    async def runtime():
        process=psutil.Process();result=copy.deepcopy(status)
        result.update(server_alive=True,worker_alive=pipeline.tts._process.poll() is None,
            qwen=qwen.status(),ASR_READY=True,OPUS_READY=True,TTS_READY=pipeline.tts._process.poll() is None,
            hybrid_ready=qwen.state=='QWEN_READY',startup_to_opus_ready_seconds=status['cold_start_seconds'],
            parent_rss_mib=process.memory_info().rss/2**20,threads=process.num_threads(),
            asr_allocated_mib=torch.cuda.memory_allocated(0)/2**20,asr_reserved_mib=torch.cuda.memory_reserved(0)/2**20,
            inference_busy=app.state.live_lock.locked(),ready=pipeline.tts.ready)
        if not result['worker_alive']:result.update(state='ERROR',error='Resident personal worker exited')
        result['qwen']['total_model_loads']=status.get('qwen_previous_loads',0)+qwen.worker.loads
        result['qwen']['explicit_restarts']=status.get('qwen_restarts',0)
        return result
    @app.post('/qwen-recover')
    async def recover_qwen():
        nonlocal qwen
        if app.state.live_lock.locked():raise HTTPException(409,'Pause/Stop before recovering Qwen')
        if qwen.status()['state']!='QWEN_ERROR':raise HTTPException(409,'Qwen recovery is only for a failed worker')
        status['qwen_previous_loads']=status.get('qwen_previous_loads',0)+qwen.worker.loads
        await qwen.close();qwen=WarmQwen(ROOT/'.model-cache/hot-runtime'/args.instance/str(time.time_ns()))
        pipeline.voicebridge_qwen=qwen;app.state.voicebridge_qwen_owner=qwen
        status['qwen_restarts']=status.get('qwen_restarts',0)+1
        qwen.task=asyncio.create_task(qwen.start(pipeline.voicebridge_policy.snapshot(),pipeline.voicebridge_gpu))
        return qwen.status()
    @app.post('/translation-backend')
    async def translation_backend(body:dict):
        if body.get('mode') not in {'OPUS_FAST','QWEN_FALLBACK'}:raise HTTPException(400,'Unknown translation mode')
        if app.state.live_lock.locked():raise HTTPException(409,'Pause/Stop before changing translation backend')
        if body['mode']=='QWEN_FALLBACK' and qwen.state!='QWEN_READY':raise HTTPException(409,'Qwen warmup not ready; OPUS remains available')
        qwen.mode=body['mode'];return qwen.status()
    @app.post('/qwen-preview')
    async def qwen_preview(body:dict):
        text=body.get('text')
        if not isinstance(text,str) or not 1<=len(text)<=500:raise HTTPException(400,'Invalid source text')
        if app.state.live_lock.locked():raise HTTPException(409,'Active session owns GPU work')
        async with app.state.live_lock:
            snap=pipeline.voicebridge_policy.snapshot()
            style=body.get('prompt_style','expanded')
            if style not in {'expanded','minimal'}:raise HTTPException(400,'Unknown prompt style')
            snap={**snap,'qwen_prompt_style':style,'qwen_measure_breakdown':body.get('profile') is True,'qwen_token_budget':body.get('token_budget',160)}
            try:
                async with pipeline.voicebridge_gpu:text,audit=await qwen.translate(text,snap)
                return dict(text=text,audit=audit)
            except Exception as exc:return dict(rejected=True,status='FILLER_SKIPPED' if str(exc)=='FILLER_SKIPPED' else 'QWEN_TECHNICAL_FAILURE',error=str(exc))
    @app.post('/voice')
    async def voice(body:dict):
        if body.get('preset') not in NAMES:raise HTTPException(400,'Unknown approved release preset')
        from .release_voices import validate_variation
        try:validate_variation(body['preset'],pipeline.voicebridge_policy.snapshot()['settings'])
        except ValueError as exc:raise HTTPException(400,str(exc))
        if app.state.live_lock.locked():raise HTTPException(409,'Wait for the previous utterance/session to drain')
        # Share the same model lease as committed-v1. No routing-mode event is added.
        async with app.state.live_lock:
            old_pid=pipeline.tts.ready['pid']
            try:
                async with pipeline.voicebridge_gpu:await inference.run(pipeline.tts.switch,body['preset'])
            except Exception as exc:raise HTTPException(500,str(exc))
            if pipeline.tts.ready['pid']!=old_pid:status['worker_restarts']+=1
        status.update(release_preset=body['preset'],ready=pipeline.tts.ready)
        status['model_ids']['tts_pid']=pipeline.tts.ready['pid'];publish('STANDBY_READY')
        return await runtime()
    @app.get('/features')
    async def features():return pipeline.voicebridge_policy.status()
    @app.get('/asr-names')
    async def asr_names():
        return {**pipeline.voicebridge_asr_names.snapshot(), 'supported':pipeline.asr.supports_context,
                'effective':pipeline.voicebridge_asr_context(pipeline.voicebridge_policy.snapshot()),
                'last_recognition':getattr(pipeline.asr,'last_recognition_context',None),
                'acceptance':'Optional context; independent name/negative recordings still required for reliability'}
    @app.post('/asr-names')
    async def save_asr_names(body:dict):
        from ort.asr_vocabulary import context_for
        try:
            if context_for(body) and not pipeline.asr.supports_context:
                raise ValueError('Installed qwen-asr has no supported context API')
            from ort.asr_vocabulary import with_dictionary
            snapshot=pipeline.voicebridge_policy.snapshot()
            with_dictionary(dict(settings=body,sha256='pending'),snapshot['dictionary'],snapshot['sha256'])
            await asyncio.to_thread(pipeline.voicebridge_asr_names.save,body)
            return await asr_names()
        except (ValueError,TypeError,KeyError) as exc:raise HTTPException(400,str(exc))
    @app.post('/asr-name-test')
    async def asr_name_test(body:dict):
        if app.state.live_lock.locked():raise HTTPException(409,'Stop TRANSLATE before testing recognition')
        async with app.state.live_lock:
            snapshot=pipeline.voicebridge_asr_context(pipeline.voicebridge_policy.snapshot())
            def recognize():
                loaded=load_asr_wav(body['path'],strict=True,max_seconds=15)
                torch.cuda.synchronize(0);started=time.perf_counter()
                result=pipeline.asr.transcribe(loaded.samples,'zh',vocabulary_snapshot=snapshot)
                torch.cuda.synchronize(0)
                return dict(text=result.text,seconds=time.perf_counter()-started,
                    context_enabled=snapshot['settings']['enabled'],context_sha256=snapshot['sha256'],
                    context=snapshot['context'],audio_transmitted=False)
            try:
                async with pipeline.voicebridge_gpu:return await inference.run(recognize)
            except (ValueError,TypeError,KeyError,OSError) as exc:raise HTTPException(400,str(exc))
    @app.post('/features')
    async def save_features(body:dict):
        try:
            from .release_voices import validate_variation
            validate_variation(pipeline.tts.name,body)
            # Text snapshots may change while a sentence is running. The active
            # sentence retains its captured snapshot and reference conditioning.
            result=await asyncio.to_thread(pipeline.voicebridge_policy.save,body)
            return {**pipeline.voicebridge_policy.status(),**result}
        except (ValueError,TypeError,KeyError) as exc:raise HTTPException(400,str(exc))
    @app.post('/translation-preview')
    async def translation_preview(body:dict):
        if app.state.live_lock.locked():raise HTTPException(409,'Please wait for the active translation session to stop')
        text=body.get('text','')
        if not isinstance(text,str) or not 1<=len(text)<=500:raise HTTPException(400,'Provide 1–500 characters')
        async with app.state.live_lock:
            from .polite_japanese import TranslationRejected
            try:
                result,audit=await inference.run(pipeline.voicebridge_policy.translate_model,pipeline.translator,text,pipeline.voicebridge_policy.snapshot())
                return dict(text=result,audit=audit)
            except TranslationRejected as exc:return dict(rejected=True,audit=exc.audit)
    @app.post('/dictionary-preview')
    async def dictionary_preview(body:dict):
        try:
            text=body.get('text','')
            if not isinstance(text,str) or len(text)>500:raise ValueError('Provide up to500 characters of Japanese display text')
            from .user_dictionary import UserDictionary
            def preview():
                dictionary=UserDictionary(body['dictionary'],pipeline.voicebridge_policy.store.base)
                spoken,audit=dictionary.apply(text);return dict(spoken=spoken,audit=audit)
            return await asyncio.to_thread(preview)
        except (ValueError,TypeError,KeyError) as exc:raise HTTPException(400,str(exc))
    @app.post('/pronunciation-test')
    async def pronunciation_test(body:dict):
        if app.state.live_lock.locked():raise HTTPException(409,'Stop TRANSLATE before testing pronunciation')
        text=body.get('text','')
        if not isinstance(text,str) or not 1<=len(text)<=200:raise HTTPException(400,'Provide 1–200 characters')
        async with app.state.live_lock:
            snapshot=pipeline.voicebridge_policy.snapshot()
            if 'dictionary' in body:
                from .user_dictionary import UserDictionary
                dictionary=await asyncio.to_thread(UserDictionary,body['dictionary'],pipeline.voicebridge_policy.store.base)
                import hashlib
                snapshot={**snapshot,'dictionary':dictionary,'settings':{**snapshot['settings'],'dictionary':dictionary.data},
                    'sha256':hashlib.sha256((snapshot['sha256']+'|preview|'+dictionary.fingerprint).encode()).hexdigest()}
            def generate():
                import soundfile as sf
                # Test audio is a local WAV only, never injected into live output.
                parts=list(pipeline.tts.synthesize_stream(text,'ja',policy_snapshot=snapshot))
                audio=np.concatenate([c.samples for c in parts]);rate=parts[0].sample_rate
                from .audio_processing import pitch_preserving
                audio,processing=pitch_preserving(audio,rate,snapshot['settings']['speed'])
                folder=ROOT/'.model-cache/dictionary-previews';folder.mkdir(exist_ok=True)
                target=folder/f'{time.time_ns()}.wav';sf.write(target,audio,rate,subtype='PCM_16')
                return dict(wav=str(target),metrics=copy.deepcopy(pipeline.tts.last_metrics),processing=processing)
            async with pipeline.voicebridge_gpu:return await inference.run(generate)
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=args.port,log_level='info',ws='websockets',timeout_graceful_shutdown=3))
    @app.post('/runtime/shutdown')
    async def shutdown(body:dict):
        if body.get('instance')!=args.instance:raise HTTPException(403,'This runtime is owned by another launcher')
        server.should_exit=True
        return {'shutting_down':True}
    server.run()


if __name__=='__main__':main()
