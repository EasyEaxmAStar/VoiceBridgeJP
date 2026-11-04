"""Official pretrained GPT-SoVITS selection. No private presets or checkpoints."""
import copy,hashlib,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAMES=('OFFICIAL_V2PROPLUS',)
DEFAULT=NAMES[0]
def selection(name=DEFAULT):
    if name not in NAMES:raise ValueError('Only official v2ProPlus is available')
    return dict(base_preset=DEFAULT,level='BASE',sampling={},seed_allowlist=[42])
def configuration(name=DEFAULT):
    selection(name)
    from ort.config import Config
    return Config(runtime='cuda',device='cuda:0',asr_backend='qwen3',qwen_asr_device='cuda:0',qwen_asr_model='Qwen/Qwen3-ASR-0.6B',qwen_asr_offline=True,torch_dtype='bfloat16',torch_attn='sdpa',translator_backend='opusmt',tts_backend='gptsovits',output_speed=1.,gptsovits_personal_profile='',gptsovits_voice_preset='',gptsovits_custom_gpt_path='',gptsovits_custom_sovits_path='')
def validate_variation(name,settings):
    selection(name)
    if settings.get('variation','STABLE')!='STABLE':raise ValueError('Public edition supports STABLE only; private seed studies are not distributed')
    seed=settings.get('seed_lock')
    if seed is not None and (type(seed) is not int or not 0<=seed<2**31):raise ValueError('Invalid seed')
class ReleaseTTS:
    def __init__(self,backend,name=DEFAULT,policy=None):
        self.backend=backend;self.name=name;self.policy=policy;self.entry=selection(name);self._lock=threading.RLock()
    def __getattr__(self,name):return getattr(self.backend,name)
    @property
    def ready(self):
        value=copy.deepcopy(self.backend.ready)
        value['release_preset']=self.name
        value['voice_preset']=dict(name=self.name,profile_sha256=hashlib.sha256((ROOT/'gptsovits.lock.json').read_bytes()).hexdigest(),prosody_level='BASE',playback=dict(method='ffmpeg_atempo',rate=1.25,preserves_pitch=True,native_speed_factor=1.,postprocessing='per_utterance'),speaker_stable=False,subjective_refinement='NOT_EVALUATED')
        value['verified_seed_allowlist']=[]
        if self.policy:value['production_features']=self.policy.status(include_settings=False)
        return value
    def synthesize_stream(self,*args,sampling=None,policy_snapshot=None,**kwargs):
        snapshot=policy_snapshot or (self.policy.snapshot() if self.policy else None)
        settings=snapshot['settings'] if snapshot else dict(variation='STABLE',seed_lock=None)
        validate_variation(self.name,settings)
        if snapshot:kwargs['pronunciation_snapshot']=snapshot['dictionary']
        options=dict(kwargs.pop('options',{}) or {});options['speed_factor']=1.
        seed=settings.get('seed_lock')
        native={**dict(sampling or {})}
        if seed is not None:native['seed']=seed
        with self._lock:yield from self.backend.synthesize_stream(*args,sampling=native,options=options,**kwargs)
    def synthesize(self,*args,**kwargs):
        import numpy as np
        from ort.pipeline.base import TTSResult
        chunks=list(self.synthesize_stream(*args,**kwargs))
        if not chunks:raise RuntimeError('No official TTS PCM')
        return TTSResult(np.concatenate([c.samples for c in chunks]),chunks[0].sample_rate)
    def switch(self,name):selection(name)
    def close(self):
        try:self.backend.close()
        finally:
            if self.policy:self.policy.close()
