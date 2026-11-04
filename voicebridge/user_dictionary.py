"""Validated user overlay. Display terminology and spoken readings stay separate."""
import copy, hashlib, json, re, time,threading
from ort.pronunciation import PronunciationLexicon, _fold, _unique


def _validate_english(data):
    if not isinstance(data,dict) or set(data)!={'schema','entries'} or data['schema']!=1 or not isinstance(data['entries'],list) or len(data['entries'])>512:
        raise ValueError('Dictionary needs schema 1 and up to 512 entries')
    result=copy.deepcopy(data);seen={}
    for entry in result['entries']:
        if not isinstance(entry,dict) or set(entry)-{'source','japanese','reading','aliases','case_sensitive','enabled','priority','match_mode','notes'}:
            raise ValueError('Unknown dictionary field')
        for key,default in dict(japanese=entry.get('source'),aliases=[],case_sensitive=False,enabled=True,priority=100,match_mode='word',notes='').items():entry.setdefault(key,default)
        if type(entry['enabled']) is not bool or type(entry['case_sensitive']) is not bool or type(entry['priority']) is not int or not -1000<=entry['priority']<=1000:
            raise ValueError('Invalid dictionary switches/priority')
        if entry['match_mode'] not in {'word','exact'} or not isinstance(entry['aliases'],list) or len(entry['aliases'])>16:
            raise ValueError('Match mode must be word or exact; up to 16 aliases')
        if any(not isinstance(entry[k],str) or len(entry[k])>200 or any(ord(c)<32 for c in entry[k]) for k in ('japanese','notes')):
            raise ValueError('Invalid Japanese display text or notes')
        if not entry['japanese'].strip():raise ValueError('Japanese display text is required')
        for term in [entry.get('source'),*entry['aliases']]:
            PronunciationLexicon(json.dumps(dict(schema_version=1,entries=[dict(term=term,reading=entry.get('reading'))]),ensure_ascii=False).encode())
            if not re.search('[A-Za-z]',term):raise ValueError('User terms must contain an English letter')
            identity=(_fold(term),entry['priority'])
            prior=seen.get(identity,[])
            if entry['enabled'] and any(not entry['case_sensitive'] or not case or old==term for old,case in prior):raise ValueError('Ambiguous equal-priority user term: '+term)
            if entry['enabled']:seen.setdefault(identity,[]).append((term,entry['case_sensitive']))
    return result


def validate_dictionary(data):
    if isinstance(data,dict) and type(data.get('schema')) is int and data['schema']==1:return _validate_english(data)
    from .dictionary_schema import validate_unified
    return validate_unified(data,_validate_english)


_TOKENIZER=None
_TOKENIZER_LOCK=threading.Lock()
def chinese_boundaries(text):
    global _TOKENIZER
    with _TOKENIZER_LOCK:
        if _TOKENIZER is None:
            try:import jieba
            except ImportError as exc:raise ValueError('Chinese terminology needs requirements-dictionary.txt; no model reload required') from exc
            jieba.setLogLevel(40);_TOKENIZER=jieba.Tokenizer();_TOKENIZER.initialize()
    return {position for _,start,end in _TOKENIZER.tokenize(text,HMM=False) for position in [start,end]}


def cjk(char):return bool(char) and '\u3400'<=char<='\u9fff'


def person_boundary(text,start,end):
    # Unknown Han names may be split into individual characters by Jieba.
    # Do not interpret a suffix of an unlisted longer name as this person.
    left=('我叫','名字是','我是','联系','给','通知','由','和','与','感谢','找','请问','告诉','根据','谢谢','认识','请联系')
    right=('的','今天','明天','已经','会','将','在','负责','说','同意','参加','确认','先生','女士','同学','老师','和','与')
    return (not cjk(text[start-1] if start else '') or text[:start].endswith(left)) and (not cjk(text[end] if end<len(text) else '') or text[end:].startswith(right))


class UserDictionary:
    def __init__(self,data,base):
        from .dictionary_schema import canonical
        self.data=validate_dictionary(data);self.base=base;self.rules=[];self.source_rules=[]
        self.entries=[canonical(e) for e in self.data['entries']]
        for i,entry in enumerate(self.entries):
            if not entry['enabled']:continue
            identifier=f'user:{i}:{entry["source_term"]}'
            def lexicon(term):return PronunciationLexicon(json.dumps(dict(schema_version=1,entries=[dict(term=term,reading=entry['reading_kana'] or 'カ')]),ensure_ascii=False).encode())
            if not entry['legacy'] and entry['entry_type']!='direct_output':
                for term in dict.fromkeys([entry['source_term'],*entry['aliases']]):self.source_rules.append((entry,lexicon(term),identifier))
            if entry['reading_kana']:
                terms=[entry['source_term'],*entry['aliases'],entry['target_ja']] if entry['source_language']=='en' else [entry['target_ja']]
                for term in dict.fromkeys(terms):self.rules.append((entry,lexicon(term),identifier))
        self.fingerprint=hashlib.sha256(json.dumps(self.data,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        self.metadata=dict(sha256=self.fingerprint,base_sha256=base.fingerprint,rule_count=len(self.rules)+len(base.entries),source_rule_count=len(self.source_rules),path='user overlay + preserved production lexicon')
        if any(e['enabled'] and e['source_language']=='zh' for e in self.entries):chinese_boundaries('')

    def _matches(self,text,base):
        candidates=[]
        if base:
            candidates=[dict(**r,rule_id='base:'+r['term'],priority=-2000,user=False) for r in self.base.apply(text)[1]['rules']]
        for entry,lex,identifier in self.rules:
            for rule in lex.apply(text)[1]['rules']:
                if entry['case_sensitive'] and rule['matched']!=rule['term']:continue
                if entry['legacy'] and entry['match_mode']=='exact' and rule['matched']!=text:continue
                if not entry['legacy'] and (cjk(text[rule['start']-1] if rule['start'] else '') or cjk(text[rule['end']] if rule['end']<len(text) else '')):continue
                candidates.append(dict(**rule,rule_id=identifier,priority=entry['priority'],user=True,japanese=entry['target_ja'],source_language=entry['source_language'],legacy=entry['legacy']))
        # Explicit overrides precede base. Within a layer: longest span, then
        # priority, then stable rule id. Equal-term/priority ambiguity is rejected.
        selected=[]
        for rule in sorted(candidates,key=lambda r:(-int(r['user']),-(r['end']-r['start']),-r['priority'],r['rule_id'])):
            if any(rule['start']<r['end'] and rule['end']>r['start'] for r in selected):continue
            selected.append(rule)
        return sorted(selected,key=lambda r:r['start'])

    def terminology(self,text):
        return self._render(text,[r for r in self._matches(text,False) if r['source_language']=='en'],'japanese')

    def source_matches(self,text,language='zh'):
        if language!='zh':return []
        boundaries=chinese_boundaries(text) if any(e['source_language']=='zh' for e,_,_ in self.source_rules) else set()
        candidates=[]
        for entry,lex,identifier in self.source_rules:
            for rule in lex.apply(text)[1]['rules']:
                if entry['case_sensitive'] and rule['matched']!=rule['term']:continue
                if entry['match_mode']=='exact' and text.strip('。！？!?，, \u3000')!=rule['matched']:continue
                if entry['source_language']=='zh' and (rule['start'] not in boundaries or rule['end'] not in boundaries):continue
                if entry['entry_type']=='person_name' and not person_boundary(text,rule['start'],rule['end']):continue
                candidates.append(dict(**rule,rule_id=identifier,priority=entry['priority'],target_ja=entry['target_ja'],source_language=entry['source_language'],entry_type=entry['entry_type']))
        selected=[]
        for rule in sorted(candidates,key=lambda r:(-(r['end']-r['start']),-r['priority'],r['rule_id'])):
            if not any(rule['start']<r['end'] and rule['end']>r['start'] for r in selected):selected.append(rule)
        return sorted(selected,key=lambda r:r['start'])

    def asr_entries(self):
        selected={}
        for entry in sorted(self.entries,key=lambda e:-e['priority']):
            if entry['enabled'] and entry['source_language']=='zh' and entry['entry_type']=='person_name':
                selected.setdefault(entry['source_term'],dict(text=entry['source_term'],pinyin=entry['pinyin'],type='person_name',enabled=True))
        return list(selected.values())

    def direct_output(self,text):
        import unicodedata
        key=lambda x:unicodedata.normalize('NFKC',x).strip().rstrip('。.')
        matches=[(e,i) for i,e in enumerate(self.entries) if e['enabled'] and e['entry_type']=='direct_output'
                 and any((key(text)==key(t) if e['case_sensitive'] else key(text).casefold()==key(t).casefold()) for t in [e['source_term'],*e['aliases']])]
        if not matches:return None
        e,i=sorted(matches,key=lambda pair:(-pair[0]['priority'],pair[1]))[0]
        return dict(text=e['target_ja'],rule_id=f'user:{i}:{e["source_term"]}')

    @staticmethod
    def _render(text,rules,field):
        parts=[];last=0
        for r in rules:parts.extend([text[last:r['start']],r[field]]);last=r['end']
        parts.append(text[last:]);return ''.join(parts)

    def apply(self,text):
        start=time.perf_counter();rules=self._matches(text,True);spoken=self._render(text,rules,'reading')
        return spoken,dict(original_tts_text=text,spoken_tts_text=spoken,rules=rules,lexicon_sha256=self.fingerprint,
            base_sha256=self.base.fingerprint,processing_seconds=time.perf_counter()-start)


def load_json(content):
    if len(content)>512*1024:raise ValueError('User dictionary exceeds 512 KiB')
    data=json.loads(content.decode('utf-8-sig'),object_pairs_hook=_unique)
    if isinstance(data,list):data=dict(schema=2,entries=data)
    if isinstance(data,dict) and 'source_term' in data:data=dict(schema=2,entries=[data])
    return validate_dictionary(data)


def load_csv(content):
    from .dictionary_schema import import_csv
    return validate_dictionary(import_csv(content))
