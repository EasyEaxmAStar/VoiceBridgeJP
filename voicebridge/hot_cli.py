"""Persistent production audio runtime CLI and bounded line-based GUI control pipe."""
import argparse,asyncio,json,os,sys,time
from pathlib import Path
from .routing_engine import RoutingEngine,Route
from .release_voices import DEFAULT,NAMES
from .live import LiveConfig
from .server_process import ManagedServer
from ort.config import Config
from ort.pipeline.gptsovits_reference import validate_reference


def emit(kind,**data):print(json.dumps(dict(type=kind,**data),ensure_ascii=False),flush=True)


async def main_async(args):
    if args.parent_gate:
        deadline=time.monotonic()+15
        while not args.parent_gate.exists():
            if time.monotonic()>deadline:raise RuntimeError('GUI process ownership gate timed out')
            await asyncio.sleep(.02)
    cfg=Config();reference=validate_reference(cfg.gptsovits_ref_audio,cfg.gptsovits_ref_text,'ja')
    server=None if args.connect else ManagedServer(reference,port=args.port,module='voicebridge.hot_server',
        environment={'VOICEBRIDGE_RELEASE_PRESET':args.preset})
    engine=RoutingEngine(LiveConfig(args.input_device,args.output_device,reference.path,reference.text,
        url=args.connect or server.url),server=server,preset=args.preset,fixture_input=args.fixture_input)
    camera_service=None
    if args.json_lines:
        try:
            from .camera_service import CameraService
            camera_service=CameraService(engine);camera_service.prepare();camera_service._refresh()
        except Exception as exc:emit('camera_error',message='摄像头控制不可用（音频继续）：'+str(exc))
    from ort.gpu_sampling import GPUSampler
    gpu=GPUSampler(interval_ms=1000).start()
    def current_status(*,detail=False):
        status=engine.status(detail=detail)
        if camera_service:status['camera']=camera_service.status()
        return status
    running=True;diagnostic_task=None
    async def diagnostic_source(path):
        try:await engine.feed_diagnostic_wav(path)
        except asyncio.CancelledError:raise
        except Exception as exc:emit('diagnostic_source_ended',message=str(exc))
    async def ticker():
        polls=0
        while running:
            polls+=1
            if polls%5==0 and engine.runtime_state=='STANDBY_READY' and not engine.audio_running:
                try:engine.runtime=await engine._http('/runtime')
                except Exception:pass
            if engine.runtime_state in {'INITIALIZING','LOADING_MODELS','WARMING_UP'} and server:
                path=server.root/'.model-cache/hot-runtime'/f'{server.token}.json'
                try:
                    phase=json.loads(path.read_text(encoding='utf8'))['state']
                    if phase in {'LOADING_MODELS','WARMING_UP'}:engine.runtime_state=phase
                except (OSError,ValueError,KeyError):pass
            status=current_status()
            with gpu._lock:status['gpu']=dict(gpu.samples[-1]) if gpu.samples else None
            emit('status',status=status);await asyncio.sleep(.2)
    tick=asyncio.create_task(ticker());code=0
    try:
        try:await engine.initialize()
        except Exception as exc:emit('error',message=f'Initialization failed: {exc}')
        if camera_service:
            try:await camera_service.start()
            except Exception as exc:emit('camera_error',message=str(exc))
        while True:
            if sys.stdin.isatty():
                import msvcrt
                text=''
                while True:
                    if msvcrt.kbhit():
                        char=msvcrt.getwch()
                        if char=='\x03':raise KeyboardInterrupt
                        if char in '\r\n':break
                        text=(text+char)[-8192:]
                    await asyncio.sleep(.01)
                line=text
            else:line=await asyncio.to_thread(sys.stdin.readline,512*1024+1)
            if not line:break
            try:
                value=json.loads(line) if line.lstrip().startswith('{') else dict(command=line.strip().lower())
                if not isinstance(value,dict) or len(line)>512*1024:raise ValueError('Expected one bounded command object')
                command=value.get('command')
                if command in {'quit','shutdown'}:break
                if command in {'translate','direct'}:await engine.activate(command.upper())
                elif command in {'start','activate'}:await engine.activate(value.get('route','TRANSLATE'))
                elif command=='pause':await engine.pause()
                elif command=='resume':await engine.resume()
                elif command=='stop':await engine.stop()
                elif command=='recover':await engine.recover()
                elif command in {'devices','refresh'}:emit('devices',devices=engine.list_devices())
                elif command=='camera_get':
                    if camera_service is None:raise ValueError('Camera control requires the GUI JSON controller')
                    emit('camera_settings',**camera_service.payload())
                elif command=='camera_devices':
                    if camera_service is None:raise ValueError('Camera service unavailable')
                    emit('camera_devices',devices=await camera_service.devices_list())
                elif command=='camera_save':
                    if camera_service is None:raise ValueError('Camera service unavailable')
                    emit('camera_settings',**await camera_service.apply(value['settings'],confirm_nv12=value.get('confirm_nv12',False)))
                elif command=='camera_disable':
                    if camera_service is None:raise ValueError('Camera service unavailable')
                    emit('camera_settings',**await camera_service.disable())
                elif command=='camera_retry':
                    if camera_service is None:raise ValueError('Camera service unavailable')
                    emit('camera_settings',**await camera_service.retry())
                elif command=='camera_probe1080':
                    if camera_service is None:raise ValueError('Camera service unavailable')
                    emit('camera_settings',**await camera_service.verify_1080(value['identity']))
                elif command=='microphone':await engine.switch_microphone(value['selection'],rate=value.get('rate'),channels=value.get('channels',1))
                elif command=='voice':await engine.select_voice(value['preset'])
                elif command=='feature_get':emit('features',**await engine._http('/features'))
                elif command=='quick_reply':emit('quick_reply',**engine.quick_reply(value['text'],shortcut_id=value.get('shortcut_id')))
                elif command=='translation_backend':emit('translation_backend',**await engine._http('/translation-backend',dict(mode=value['mode'])))
                elif command=='qwen_recover':emit('qwen_recover',**await engine._http('/qwen-recover',{}))
                elif command=='asr_names_get':emit('asr_names',**await engine._http('/asr-names'))
                elif command=='asr_name_test':emit('asr_name_test',**await engine._http('/asr-name-test',dict(path=value['path'])))
                elif command=='asr_names_save':emit('asr_names',**await engine._http('/asr-names',value['settings']))
                elif command=='feature_save':emit('features',**await engine.update_features(value['settings']))
                elif command=='boundary_arm':engine.arm_boundary(value['enabled'],value['generation'])
                elif command=='boundary_commit':emit('boundary_commit',**engine.end_utterance(**{k:v for k,v in value.items() if k!='command'}))
                elif command=='dictionary_preview':emit('dictionary_preview',**await engine._http('/dictionary-preview',dict(text=value['text'],dictionary=value['dictionary'])))
                elif command=='pronunciation_test':emit('pronunciation_test',**await engine._http('/pronunciation-test',{k:value[k] for k in ['text','dictionary'] if k in value}))
                elif command=='translation_preview':emit('translation_preview',**await engine._http('/translation-preview',dict(text=value['text'])))
                elif command=='speed':
                    speed=float(value['value'])
                    if not .75<=speed<=1.5:raise ValueError('Speed range is.75–1.50')
                    if engine.feature_settings:await engine.update_features({**engine.feature_settings,'speed':speed})
                    else:engine.speed=speed
                elif command=='adaptive':
                    if type(value.get('enabled')) is not bool:raise ValueError('Expected boolean adaptive option')
                    if engine.feature_settings:await engine.update_features({**engine.feature_settings,'adaptive':value['enabled']})
                    else:engine.adaptive=value['enabled']
                elif command=='test':engine.check_device_silent()
                elif command=='diagnostic_wav':
                    if diagnostic_task and not diagnostic_task.done():raise ValueError('A bounded diagnostic source is already active')
                    diagnostic_task=asyncio.create_task(diagnostic_source(value['path']))
                elif command!='status':raise ValueError('Commands: translate,direct,pause,resume,stop,recover,devices,microphone,voice,speed,adaptive,test,status,shutdown')
                emit('command_done',command=command,status=current_status())
            except Exception as exc:emit('error',message=str(exc),status=current_status())
    except (KeyboardInterrupt,asyncio.CancelledError):pass
    finally:
        if diagnostic_task:diagnostic_task.cancel();await asyncio.gather(diagnostic_task,return_exceptions=True)
        running=False;tick.cancel();await asyncio.gather(tick,return_exceptions=True)
        if camera_service:
            try:await camera_service.close()
            except Exception as exc:emit('camera_error',message='摄像头关闭失败，继续释放音频与模型：'+str(exc))
        await engine.shutdown();gpu.stop();report=current_status(detail=True);report['gpu']=gpu.report()
        if args.report:
            with args.report.open('x',encoding='utf8') as out:json.dump(report,out,ensure_ascii=False,indent=2)
        emit('shutdown',status=report)
    return code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    from .public_profile import microphone_default
    p.add_argument('--input-device',default=microphone_default())
    p.add_argument('--output-device',default='CABLE Input (VB-Audio Virtual Cable)')
    p.add_argument('--preset',choices=NAMES,default=DEFAULT);p.add_argument('--connect');p.add_argument('--port',type=int,default=8765)
    p.add_argument('--report',type=Path);p.add_argument('--json-lines',action='store_true')
    p.add_argument('--parent-gate',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--fixture-input',action='store_true',help='Diagnostic only: ignore physical microphone; explicitly supplied WAVs only')
    args=p.parse_args()
    if args.report and (args.report.exists() or not args.report.parent.is_dir()):p.error('Choose a new report file in an existing directory')
    return asyncio.run(main_async(args))


if __name__=='__main__':raise SystemExit(main())
