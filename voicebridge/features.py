"""Atomic local preferences and immutable-by-copy utterance snapshots."""
import copy, hashlib, json, os, tempfile, threading, time
from pathlib import Path
from ort.pronunciation import PronunciationLexicon
from .user_dictionary import UserDictionary,validate_dictionary
from .variation import LEVELS

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'config/user_preferences.json'
DEFAULTS=dict(schema=2,variation='STABLE',seed_lock=None,speed=1.25,adaptive=False,
    segmentation=dict(silence_ms=700,min_speech_ms=30,max_seconds=30.,energy=.012,pre_roll_ms=300,smart_mandarin=False),
    boundary=dict(mode='AUTO',button='RIGHT',scope='IN_APP_TRIGGER_AREA',debounce_ms=250),
    formality='STRICT_BUSINESS_POLITE',dictionary=dict(schema=1,entries=[]),expressions={})


def validate(value):
    if isinstance(value,dict) and value.get('schema')==1 and set(value)==set(DEFAULTS)-{'boundary'}:
        value=copy.deepcopy(value);value.update(schema=2,boundary=copy.deepcopy(DEFAULTS['boundary']))
    if not isinstance(value,dict) or set(value)!=set(DEFAULTS):raise ValueError('Invalid preferences fields')
    v=copy.deepcopy(value)
    if v['schema']!=2 or v['variation'] not in LEVELS or v['formality'] not in {'STRICT_BUSINESS_POLITE','OFF'}:raise ValueError('Invalid variation/style policy')
    b=v['boundary']
    if (not isinstance(b,dict) or set(b)!=set(DEFAULTS['boundary']) or b['mode'] not in {'AUTO','MANUAL','HYBRID'}
        or b['button'] not in {'LEFT','RIGHT','EITHER','DISABLED'} or b['scope'] not in {'IN_APP_TRIGGER_AREA','GLOBAL_MOUSE'}
        or type(b['debounce_ms']) is not int or not 100<=b['debounce_ms']<=1000):raise ValueError('Invalid speech boundary settings')
    if v['seed_lock'] is not None and (type(v['seed_lock']) is not int or not 0<=v['seed_lock']<2**31):raise ValueError('Seed lock needs an integer in [0,2147483647]')
    if type(v['adaptive']) is not bool or type(v['speed']) not in {float,int} or not .75<=v['speed']<=1.5:raise ValueError('Invalid playback speed')
    s=v['segmentation']
    if not isinstance(s,dict) or set(s)!=set(DEFAULTS['segmentation']):raise ValueError('Invalid segmentation fields')
    ranges=dict(silence_ms=(300,1200),min_speech_ms=(30,1000),max_seconds=(3,30),energy=(.001,.1),pre_roll_ms=(0,600))
    for key,(lo,hi) in ranges.items():
        if type(s[key]) not in {int,float} or not lo<=s[key]<=hi:raise ValueError(f'{key} must be {lo}–{hi}')
    if any(type(s[k]) is not int for k in ('silence_ms','min_speech_ms','pre_roll_ms')) or type(s['smart_mandarin']) is not bool:raise ValueError('Invalid segmentation units')
    if s['max_seconds']*1000<s['silence_ms']+s['pre_roll_ms']+s['min_speech_ms']:raise ValueError('Utterance budget is too short')
    v['dictionary']=validate_dictionary(v['dictionary'])
    if not isinstance(v['expressions'],dict) or len(v['expressions'])>256 or any(not isinstance(k,str) or not isinstance(x,str) or not 1<=len(k)<=300 or not 1<=len(x)<=500 or any(ord(c)<32 for c in k+x) for k,x in v['expressions'].items()):raise ValueError('Invalid exact polite-expression dictionary')
    from .polite_japanese import source_key
    normalized=[source_key(k) for k in v['expressions']]
    if any(not k for k in normalized) or len(set(normalized))!=len(normalized):raise ValueError('Ambiguous normalized preferred expression')
    return v


class FeatureStore:
    def __init__(self,path=None):
        self.path=Path(path or os.environ.get('VOICEBRIDGE_PREFERENCES',PATH));self._lock=threading.RLock();self.base=PronunciationLexicon.load(ROOT/'config/pronunciation.ja.production.json')
        value=json.loads(self.path.read_text(encoding='utf8')) if self.path.exists() else copy.deepcopy(DEFAULTS)
        self._value=validate(value);self._compile()
    def _compile(self):
        self._dictionary=UserDictionary(self._value['dictionary'],self.base)
        self.sha=hashlib.sha256(json.dumps(self._value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    def snapshot(self):
        with self._lock:return dict(settings=copy.deepcopy(self._value),dictionary=self._dictionary,sha256=self.sha)
    def save(self,value):
        start=time.perf_counter();candidate=validate(value);dictionary=UserDictionary(candidate['dictionary'],self.base)
        content=json.dumps(candidate,ensure_ascii=False,indent=2)
        with self._lock:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            fd,temp=tempfile.mkstemp(prefix=self.path.name+'.',suffix='.partial',dir=self.path.parent)
            try:
                with os.fdopen(fd,'w',encoding='utf8') as f:f.write(content);f.flush();os.fsync(f.fileno())
                os.replace(temp,self.path)
            finally:
                if os.path.exists(temp):os.unlink(temp)
            self._value=candidate;self._dictionary=dictionary
            self.sha=hashlib.sha256(json.dumps(candidate,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        return dict(sha256=self.sha,reload_seconds=time.perf_counter()-start)
