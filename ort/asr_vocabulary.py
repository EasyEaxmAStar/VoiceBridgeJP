"""Bounded recognition-context vocabulary, independent of Japanese TTS lexicons."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import threading
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(schema=1, enabled=False, entries=[])


def is_internal_context_echo(text: str) -> bool:
    """Reject our internal prompt scaffold, never a name or its homophones alone.

    Reject the whole contaminated result: deleting the prompt cannot establish
    that any remaining words were actually spoken. Works even after hints are
    disabled, and across punctuation/case/spacing variants of the scaffold.
    """
    normalized = unicodedata.normalize('NFKC', text).casefold()
    compact = ''.join(c for c in normalized if c.isalnum())
    return 'importantchinesepropername' in compact


def validate(value):
    if not isinstance(value, dict) or set(value) != {'schema','enabled','entries'} or type(value['schema']) is not int or value['schema'] != 1 or type(value['enabled']) is not bool:
        raise ValueError('ASR name vocabulary requires schema1, enabled boolean and entries')
    result = copy.deepcopy(value)
    if not isinstance(result['entries'],list) or len(result['entries']) > 16:
        raise ValueError('At most16 ASR name entries allowed')
    seen = set()
    for entry in result['entries']:
        if not isinstance(entry,dict) or set(entry) != {'text','pinyin','type','enabled'}:
            raise ValueError('Name entry requires text, pinyin, type, enabled')
        text, pinyin = entry['text'], entry['pinyin']
        if not isinstance(text,str) or not re.fullmatch(r'[\u3400-\u9fff]{1,8}(?:·[\u3400-\u9fff]{1,8})?',text):
            raise ValueError('Chinese person name must contain1–17 Chinese characters/middle dot')
        if not isinstance(pinyin,str) or len(pinyin)>80 or pinyin!=pinyin.strip():
            raise ValueError('Provide bounded Mandarin pinyin')
        if any(not (c in "12345 '-" or c.isalpha() and 'LATIN' in unicodedata.name(c,'')) for c in pinyin):
            raise ValueError('Pinyin allows Latin letters/tone marks, spaces, apostrophe and tones1–5')
        if entry['type'] != 'person_name' or type(entry['enabled']) is not bool or text in seen:
            raise ValueError('Invalid or duplicate ASR person name')
        seen.add(text)
    return result


def context_for(value):
    value = validate(value)
    if not value['enabled']:
        return ''
    lines = [f"Important Chinese proper name: {e['text']}."+(f" Pronunciation: {e['pinyin']}." if e['pinyin'] else '')
             for e in value['entries'] if e['enabled']]
    context = '\n'.join(lines)
    if len(context) > 1024:
        raise ValueError('ASR vocabulary context exceeds1024 characters')
    return context


def with_dictionary(original,dictionary,settings_sha256=None):
    """Keep the explicit ASR master switch and dedicated-name overrides intact."""
    value=copy.deepcopy(original['settings']);derived=0
    if value['enabled']:
        names={e['text'] for e in value['entries']}
        for entry in dictionary.asr_entries():
            if entry['text'] not in names:value['entries'].append(entry);names.add(entry['text']);derived+=1
    context=context_for(value)
    digest=hashlib.sha256((original['sha256']+'|'+(settings_sha256 or dictionary.fingerprint)).encode()).hexdigest()
    return dict(settings=value,context=context,sha256=digest,derived_names=derived,
                original_vocabulary_sha256=original['sha256'],dictionary_sha256=dictionary.fingerprint)


class NameVocabulary:
    def __init__(self,path=None):
        self.path=Path(path or os.environ.get('VOICEBRIDGE_ASR_VOCABULARY',ROOT/'config/asr_names.json'))
        self._lock=threading.Lock()
        self._set(self._read())

    def _read(self):
        if not self.path.exists():return DEFAULTS
        if self.path.stat().st_size>32768:raise ValueError('ASR vocabulary file exceeds32KiB')
        return json.loads(self.path.read_text(encoding='utf-8-sig'))

    def _set(self,value):
        self._value=validate(value);self._context=context_for(self._value)
        self._sha=hashlib.sha256(json.dumps(self._value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

    def snapshot(self):
        with self._lock:return dict(settings=copy.deepcopy(self._value),context=self._context,sha256=self._sha)

    def reload(self):
        candidate=validate(self._read());context_for(candidate)
        with self._lock:self._set(candidate)
        return self.snapshot()

    def save(self,value):
        candidate=validate(value);context_for(candidate)
        content=json.dumps(candidate,ensure_ascii=False,indent=2)
        with self._lock:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            fd,temp=tempfile.mkstemp(prefix=self.path.name+'.',suffix='.partial',dir=self.path.parent)
            try:
                with os.fdopen(fd,'w',encoding='utf8') as stream:
                    stream.write(content);stream.flush();os.fsync(stream.fileno())
                os.replace(temp,self.path)
            finally:
                if os.path.exists(temp):os.unlink(temp)
            self._set(candidate)
        return self.snapshot()
