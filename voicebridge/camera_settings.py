"""Versioned camera-only preferences. Never writes voice/audio dictionaries."""
from __future__ import annotations
from dataclasses import asdict
import copy,json,os,tempfile
from pathlib import Path

from .camera_video import VideoConfig

ROOT=Path(__file__).resolve().parents[1]
PRESETS={
 'LOW':dict(regions_min=1,regions_max=3,block_min=3,block_max=14,stutter_intensity=.35,freeze_min=.1,freeze_max=.25,randomness=1.),
 'MEDIUM':dict(regions_min=2,regions_max=6,block_min=5,block_max=35,stutter_intensity=1.,freeze_min=.1,freeze_max=.8,randomness=1.),
 'HIGH':dict(regions_min=4,regions_max=9,block_min=8,block_max=50,stutter_intensity=1.,freeze_min=.2,freeze_max=.8,randomness=1.),
}


def defaults():
    video=asdict(VideoConfig('_unselected_'));video['camera_identity']=''
    return dict(schema=1,enabled=False,intensity='MEDIUM',safety_timeout=30.,video=video)


def validate(value):
    if not isinstance(value,dict) or set(value)!={'schema','enabled','intensity','safety_timeout','video'}:
        raise ValueError('摄像头配置字段无效')
    if type(value['schema']) is not int or value['schema']!=1:raise ValueError('不支持的摄像头配置版本')
    if type(value['enabled']) is not bool:raise ValueError('启用选项必须为布尔值')
    if value['intensity'] not in {*PRESETS,'CUSTOM'}:raise ValueError('无效的效果强度')
    from .camera_events import CameraConfig
    CameraConfig(timeout=value['safety_timeout'])
    if not isinstance(value['video'],dict):raise ValueError('无效的视频配置')
    video=copy.deepcopy(value['video']);identity=video.get('camera_identity','')
    if not isinstance(identity,str) or len(identity)>1024:raise ValueError('无效的摄像头身份')
    if value['enabled'] and not identity:raise ValueError('请明确选择物理摄像头，不会自动替换设备')
    config=VideoConfig(**{**video,'camera_identity':identity or '_unselected_'})
    result=copy.deepcopy(value);result['video']=asdict(config);result['video']['camera_identity']=identity
    return result


def migrate(value):
    if isinstance(value,dict) and value.get('schema')==1:return validate(value)
    # Legacy flat VideoConfig is a profile, never implicit permission to open.
    config=VideoConfig(**value)
    result=defaults();result['video']=asdict(config)
    if any(result['video'][k]!=v for k,v in PRESETS['MEDIUM'].items()):result['intensity']='CUSTOM'
    return validate(result)


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf8',dir=path.parent,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name);json.dump(value,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path);temporary=None
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


class CameraPreferences:
    def __init__(self,path=None,*,environment=None,legacy=None):
        self.environment=dict(os.environ if environment is None else environment)
        self.path=Path(path or self.environment.get('VOICEBRIDGE_CAMERA_PREFERENCES',ROOT/'config/camera_preferences.json'))
        self.legacy=Path(legacy or ROOT/'config/camera_glitch.local.json')
        self.error=None;self.locked=False;self.source='safe defaults'
        self.saved=defaults()
        profile=self.environment.get('VOICEBRIDGE_CAMERA_VIDEO_CONFIG')
        flag=self.environment.get('VOICEBRIDGE_CAMERA_TELEMETRY')
        self.locked=bool(profile or flag not in {None,'1'})
        try:
            if self.path.is_file():self.saved=migrate(json.loads(self.path.read_text(encoding='utf8')));self.source='GUI preferences'
            elif self.legacy.is_file():self.saved=migrate(json.loads(self.legacy.read_text(encoding='utf8')));self.source='existing profile (disabled until explicitly enabled)'
            self.effective=copy.deepcopy(self.saved)
            if flag not in {None,'0','1'}:raise ValueError('VOICEBRIDGE_CAMERA_TELEMETRY 必须为0或1')
            if profile:
                self.effective['video']=asdict(VideoConfig(**json.loads(Path(profile).read_text(encoding='utf8'))))
                self.locked=True;self.source='startup environment profile'
                if flag=='1':self.effective['enabled']=True
            if flag=='0':self.effective['enabled']=False;self.locked=True;self.source='startup environment disabled'
            self.effective=validate(self.effective)
        except (OSError,ValueError,TypeError) as exc:
            self.error=str(exc);self.effective=defaults();self.source='invalid configuration; safe disabled'
    def save(self,value):
        result=validate(value)
        if self.locked:raise ValueError('启动环境覆盖正生效；设置需清除对应环境变量后重启。可随时使用“停止摄像头”。')
        atomic_json(self.path,result);self.saved=copy.deepcopy(result);self.effective=copy.deepcopy(result);self.error=None;self.source='GUI preferences'
        return result
