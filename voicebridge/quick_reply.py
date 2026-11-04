"""Exact final Japanese utterances, never a translation or recording fallback."""
import copy,hashlib,json
from dataclasses import dataclass
from .features import validate
from .user_dictionary import UserDictionary

PHRASES=('はい','はい、よろしくお願いいたします。')
SHORTCUTS={'quick.hai':PHRASES[0], 'quick.greeting_polite':PHRASES[1]}

def validate_shortcut(text,shortcut_id=None):
    if text not in PHRASES:raise ValueError('Unknown quick reply')
    if not isinstance(shortcut_id,str) or shortcut_id not in SHORTCUTS:return None
    if SHORTCUTS[shortcut_id]!=text:
        raise ValueError('Quick reply ID/text mismatch')
    return shortcut_id

@dataclass(frozen=True)
class QuickJob:
    context:object
    samples:object
    text:str
    settings:dict
    preset:str
    boundary_cause:str='quick_reply'
    boundary_requested_at:float|None=None
    shortcut_id:str|None=None

def quick_snapshot(settings,policy):
    value=validate(settings)
    from .release_voices import validate_variation
    # The release TTS validates the active seed policy again at synthesis.
    dictionary=UserDictionary(value['dictionary'],policy.store.base)
    sha=hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return dict(settings=copy.deepcopy(value),dictionary=dictionary,sha256=sha)
