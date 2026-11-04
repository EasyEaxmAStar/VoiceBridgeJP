"""GUI-controller camera lifecycle, isolated from audio/model ownership."""
from __future__ import annotations
import asyncio,copy,json,os,subprocess,time,uuid
from pathlib import Path

from .camera_controller import CameraController
from .camera_process import CameraVideoProcess
from .camera_settings import CameraPreferences,ROOT,atomic_json,validate


def enumerate_video(python=None):
    """Slow native enumeration runs in a controller thread, never Qt/audio."""
    python=Path(python or ROOT/'.venv-camera/Scripts/python.exe')
    if not python.is_file():raise RuntimeError('缺少 .venv-camera；请恢复已锁定的摄像头环境，不要安装全部AI依赖')
    result=subprocess.run([str(python),'-X','utf8','-m','voicebridge.camera_worker','--diagnose'],
        cwd=ROOT,capture_output=True,text=True,encoding='utf8',timeout=12,creationflags=subprocess.CREATE_NO_WINDOW)
    if len(result.stdout)>65536 or len(result.stderr)>65536:raise RuntimeError('摄像头枚举输出超限')
    if result.returncode:raise RuntimeError('无法枚举摄像头：'+result.stderr[-500:])
    data=next((json.loads(line) for line in reversed(result.stdout.splitlines()) if line.startswith('{')),None)
    if not data or 'cameras' not in data:raise RuntimeError('摄像头诊断返回无效数据')
    for device in data['cameras']:
        name=device['name'].casefold()
        device['infrared']=('infrared' in name or ' ir ' in ' '+name+' ' or '红外' in name or '赤外' in name)
    return data


class CameraService:
    def __init__(self,engine,*,preferences=None,controller_factory=CameraController,video_factory=CameraVideoProcess,directory=None):
        self.engine=engine;self.preferences=preferences or CameraPreferences()
        self.settings=copy.deepcopy(self.preferences.effective)
        self.controller_factory=controller_factory;self.video_factory=video_factory
        self.directory=Path(directory or ROOT/'.model-cache/camera-runtime'/uuid.uuid4().hex)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.lock=asyncio.Lock();self.monitor=None;self.closed=False;self.confirmed=False
        self.error=self.preferences.error;self.notice=None;self.controller=None;self.suspended=False
        self.virtual_failure=None
        self.last_recovery=0;self.revision=0;self.cached={};self.devices=None;self.verified_modes=set()
        self.mode_path=self.preferences.path.with_name(self.preferences.path.stem+'.verified-modes.json')
        try:
            modes=json.loads(self.mode_path.read_text(encoding='utf8'))
            if modes.get('schema')==1 and isinstance(modes.get('identities'),list) and len(modes['identities'])<=64:
                self.verified_modes={x for x in modes['identities'] if isinstance(x,str) and 0<len(x)<=1024}
        except (OSError,ValueError,AttributeError):pass
        self.engine.camera_telemetry=True
        self.engine.camera_factory=self._make_controller

    def _make_controller(self,session):
        return self.controller_factory(session,enabled=False,autoload_environment=False)

    def prepare(self):
        # Must precede owned server startup so its ASR publisher inherits this
        # same authenticated pipe. Does not open a webcam or load any model.
        if self.engine.camera is None:self.engine.camera=self._make_controller(self.engine.session_id)
        self.controller=self.engine.camera

    async def start(self):
        self.prepare()
        if self.settings['enabled']:
            try:await self.apply(self.settings,persist=False)
            except Exception as exc:self.error=str(exc)
        self._refresh()
        if self.monitor is None:self.monitor=asyncio.create_task(self._watch())

    async def devices_list(self):
        self.devices=await asyncio.to_thread(enumerate_video)
        return self.devices

    async def _detach(self):
        camera=self.engine.camera
        if camera is None:return
        camera.set_enabled(False,timeout=self.settings['safety_timeout'])
        with camera._lock:
            video=camera.video;camera.video=None
        if video:await asyncio.to_thread(video.close)

    async def apply(self,value,*,persist=True,confirm_nv12=False,_probe=False):
        candidate=validate(value)
        if type(confirm_nv12) is not bool:raise ValueError('实验确认必须为布尔值')
        if persist and self.preferences.locked:raise ValueError('启动环境覆盖正生效；清除环境变量并重启后才能保存。可使用“停止摄像头”。')
        # Validate before touching a healthy worker or the saved file.
        if candidate['enabled'] and candidate['video']['width']==1920 and candidate['video']['camera_identity'] not in self.verified_modes and not _probe:
            raise ValueError('此设备1080p尚未验证。请先运行摄像头模式诊断；当前配置保持不变。')
        async with self.lock:
            if self.closed:raise RuntimeError('Camera service is shutting down')
            previous=copy.deepcopy(self.settings)
            if self.engine.camera is None:self.prepare()
            await self._detach()
            self.settings=candidate;self.confirmed=confirm_nv12 or self.confirmed;self.suspended=False
            self.error=None;self.notice=None;self.virtual_failure=None
            video=None
            try:
                if candidate['enabled']:
                    config=copy.deepcopy(candidate['video'])
                    if config['processing_backend']=='nv12' and not self.confirmed:
                        config['processing_backend']='bgr';self.notice='NV12待本次启动重新确认；当前使用CPU BGR稳定模式。'
                    atomic_json(self.directory/'video-config.json',config)
                    creation=asyncio.create_task(asyncio.to_thread(self.video_factory,self.directory/'video-config.json'))
                    try:video=await asyncio.shield(creation)
                    except asyncio.CancelledError:
                        # Native process creation cannot be interrupted safely.
                        # Drain it and close exactly the child it created.
                        pending=await creation
                        await asyncio.to_thread(pending.close)
                        raise
                    if self.closed:
                        await asyncio.to_thread(video.close);raise RuntimeError('Camera service closed during startup')
                    camera=self.engine.camera
                    camera.trigger_mode=config['trigger_mode'];camera.onset_threshold=config['onset_threshold']
                    with camera._lock:camera.video=video
                    camera.set_enabled(True,timeout=candidate['safety_timeout'])
                    self.controller=camera;self.last_recovery=0
                if persist:self.preferences.save(candidate)
                self.revision+=1;self._refresh()
            except BaseException as exc:
                if video:
                    camera=self.engine.camera
                    if camera:
                        camera.set_enabled(False)
                        with camera._lock:camera.video=None
                    await asyncio.to_thread(video.close)
                self.settings=previous;self.error=str(exc);self.revision+=1;self._refresh()
                raise
            return self.payload()

    async def disable(self):
        # Safety stop works even with explicit environment overrides. Persisted
        # configuration is untouched; GUI clearly displays session suspension.
        async with self.lock:
            await self._detach();self.suspended=True;self.error=None;self.virtual_failure=None
            self.notice='本次会话摄像头已停止；已保存的设置及启动环境未改动。'
            self.revision+=1;self._refresh()
        return self.payload()

    async def retry(self):
        return await self.apply(self.settings,persist=False)

    async def verify_1080(self,identity):
        devices=await self.devices_list()
        if not any(d['identity']==identity and not d['virtual'] and not d.get('infrared') for d in devices['cameras']):
            raise ValueError('请选择当前明确可用的物理摄像头；不会验证替代设备')
        previous=copy.deepcopy(self.settings);suspended=self.suspended
        probe=copy.deepcopy(previous);probe['enabled']=True
        probe['video'].update(camera_identity=identity,width=1920,height=1080,processing_backend='bgr')
        success=False
        try:
            await self.apply(probe,persist=False,_probe=True)
            end=time.monotonic()+8
            while time.monotonic()<end:
                video=self.engine.camera.video
                if video is None or not video.status()['alive']:break
                status=video.last_status
                if status.get('capture_frames',0)>=10 and not status.get('capture_error'):
                    success=True;self.verified_modes.add(identity)
                    atomic_json(self.mode_path,dict(schema=1,identities=sorted(self.verified_modes)))
                    break
                await asyncio.sleep(.1)
        finally:
            await self.apply(previous,persist=False)
            if suspended:await self.disable()
        if not success:raise ValueError('此设备1080p未通过真实采集验证；已恢复原摄像头配置，未静默缩放。')
        self.notice='此设备已实际交付至少10张1920×1080帧；可选择1080p。未承诺30FPS。'
        self._refresh();return self.payload()

    def _refresh(self):
        camera=self.engine.camera
        state=camera.status() if camera else {}
        video=state.get('video',{});last=video.get('last',{});output=last.get('output',{})
        active=bool(camera and camera.video and not self.suspended)
        self.cached=dict(revision=self.revision,enabled=active,mode=last.get('mode',state.get('mode','OFF')) if active else 'DISABLED',
            controller_mode=state.get('mode','OFF'),
            settings=copy.deepcopy(self.settings),source=self.preferences.source,environment_locked=self.preferences.locked,
            requested_backend=self.settings['video']['processing_backend'],active_backend=output.get('active_backend'),
            nv12_confirmation_required=self.settings['video']['processing_backend']=='nv12' and not self.confirmed,
            error=self.error or video.get('error') or last.get('capture_error') or last.get('error'),
            notice=self.notice or output.get('fallback_reason'),video_alive=video.get('alive',False),
            capture_ready=bool(last.get('capture_frames',0) and not last.get('capture_error') and not last.get('error')),
            virtual_ready=output.get('ready',False),stats=copy.deepcopy(last),
            virtual_state='CLOSED' if self.closed else self.virtual_failure or (last.get('error_code') or output.get('backend_state','INITIALIZING') if active else 'DISABLED'),
            verified_1080=self.settings['video']['camera_identity'] in self.verified_modes,
            telemetry_connected=state.get('pipe_connected',False),video_pid=video.get('pid'))

    def payload(self):return dict(camera=copy.deepcopy(self.cached))
    def status(self):return copy.deepcopy(self.cached)

    async def _watch(self):
        try:
            while not self.closed:
                await asyncio.sleep(.5)
                try:await self.watch_turn()
                except Exception as exc:
                    self.error='Camera monitor: '+type(exc).__name__;self._refresh()
        except asyncio.CancelledError:raise

    async def watch_turn(self):
        camera=self.engine.camera
        if camera is not self.controller and camera is not None:
            self.controller=camera
            if self.settings['enabled'] and not self.suspended:
                try:await self.apply(self.settings,persist=False)
                except Exception as exc:self.error=str(exc)
        if camera and camera.video:
            state=camera.video.status();last=state.get('last',{});recovery=last.get('output',{}).get('recovery_revision',0)
            if recovery>self.last_recovery:
                self.last_recovery=recovery;camera.set_enabled(True,timeout=self.settings['safety_timeout'])
                self.notice=last['output'].get('fallback_reason');self.revision+=1
            if not state.get('alive'):
                failed=camera.video
                async with self.lock:
                    if camera.video is failed:
                        await self._detach()
                        self.virtual_failure=last.get('error_code','WORKER_EXITED')
                        self.error=(state.get('error') or last.get('error') or '摄像头进程已退出')+'；已释放视频拥有者，音频模型保持原状态，请点击重试。';self.revision+=1
        self._refresh()

    async def close(self):
        self.closed=True
        if self.monitor:
            self.monitor.cancel();await asyncio.gather(self.monitor,return_exceptions=True);self.monitor=None
        async with self.lock:await self._detach()
        self._refresh()
