"""Additive unified dictionary schema and bounded JSON/CSV interchange."""
import copy,csv,io,json,re
from ort.pronunciation import PronunciationLexicon,_fold
from ort.asr_vocabulary import validate as validate_names

FIELDS=['source_language','source_term','target_ja','reading_kana','pinyin','entry_type','enabled','priority','aliases','case_sensitive','match_mode','notes']
TYPES=('person_name','technical_term','organization','general_vocabulary','direct_output')


def canonical(entry):
    if 'source_term' in entry:return {**entry,'legacy':False}
    return dict(source_language='en',source_term=entry['source'],target_ja=entry['japanese'],reading_kana=entry['reading'],
        pinyin='',entry_type='technical_term',enabled=entry['enabled'],priority=entry['priority'],aliases=entry['aliases'],
        case_sensitive=entry['case_sensitive'],match_mode=entry['match_mode'],notes=entry['notes'],legacy=True)


def validate_unified(data,legacy_validator):
    if not isinstance(data,dict) or set(data)!={'schema','entries'} or type(data['schema']) is not int or data['schema']!=2 or not isinstance(data['entries'],list) or len(data['entries'])>512:
        raise ValueError('Unified dictionary needs schema2 and up to512 entries')
    result=copy.deepcopy(data)
    for i,entry in enumerate(result['entries']):
        if isinstance(entry,dict) and 'source' in entry:
            result['entries'][i]=legacy_validator(dict(schema=1,entries=[entry]))['entries'][0];continue
        if not isinstance(entry,dict) or set(entry)-set(FIELDS)-{'type'}:raise ValueError('Unknown unified dictionary field')
        if 'type' in entry:
            if 'entry_type' in entry and entry['entry_type']!=entry['type']:raise ValueError('Conflicting dictionary entry type')
            entry['entry_type']=entry.pop('type')
        for key,value in dict(reading_kana='',pinyin='',entry_type='general_vocabulary',enabled=True,priority=100,aliases=[],case_sensitive=False,match_mode='word',notes='').items():entry.setdefault(key,value)
        if set(entry)!=set(FIELDS) or entry['source_language'] not in {'zh','en'} or entry['entry_type'] not in TYPES:raise ValueError('Choose zh/en, source term, Japanese form and supported type')
        if type(entry['enabled']) is not bool or type(entry['case_sensitive']) is not bool or type(entry['priority']) is not int or not -1000<=entry['priority']<=1000:raise ValueError('Invalid dictionary switches/priority')
        if entry['match_mode'] not in {'word','exact'} or not isinstance(entry['aliases'],list) or len(entry['aliases'])>16:raise ValueError('Match mode word/exact; up to16 aliases')
        if entry['entry_type']=='direct_output' and entry['match_mode']!='exact':raise ValueError('Direct output requires whole-utterance exact mode')
        if any(not isinstance(entry[k],str) or len(entry[k])>200 or any(ord(c)<32 for c in entry[k]) for k in ['source_term','target_ja','reading_kana','pinyin','notes']):raise ValueError('Invalid dictionary text')
        if not entry['target_ja'].strip() or len(entry['target_ja'])>80 or entry['target_ja']!=entry['target_ja'].strip() or re.search(r'ZXQ[A-Z]+QXZ',entry['target_ja']):raise ValueError('Japanese written form required, up to80 characters without internal markers')
        for term in [entry['source_term'],*entry['aliases']]:
            PronunciationLexicon(json.dumps(dict(schema_version=1,entries=[dict(term=term,reading=entry['reading_kana'] or 'カ')]),ensure_ascii=False).encode())
            if entry['source_language']=='en' and not re.search('[A-Za-z]',term):raise ValueError('English source needs Latin letters')
            if entry['source_language']=='zh' and not re.search('[\u3400-\u9fff]',term) and (term==entry['source_term'] or not re.search('[A-Za-z]',term)):raise ValueError('Chinese source needs Chinese characters; aliases may be English')
            if entry['source_language']=='zh' and len(term)==1 and entry['match_mode']!='exact':raise ValueError('Single-character Chinese entries require exact mode')
        if entry['pinyin']:
            if entry['source_language']!='zh' or entry['entry_type']!='person_name':raise ValueError('Pinyin ASR hints are for Chinese person names')
        if entry['source_language']=='zh' and entry['entry_type']=='person_name':
            validate_names(dict(schema=1,enabled=True,entries=[dict(text=entry['source_term'],pinyin=entry['pinyin'],type='person_name',enabled=True)]))
    seen={};spoken={}
    for raw in result['entries']:
        e=canonical(raw)
        if not e['enabled']:continue
        for term in [e['source_term'],*e['aliases']]:
            key=(e['source_language'],_fold(term),e['priority']);prior=seen.setdefault(key,[])
            if any(not e['case_sensitive'] or not sensitive or old==term for old,sensitive in prior):raise ValueError('Ambiguous equal-priority source term: '+term)
            prior.append((term,e['case_sensitive']))
        if e['reading_kana']:
            key=(_fold(e['target_ja']),e['priority'])
            if key in spoken and spoken[key]!=e['reading_kana']:raise ValueError('Conflicting equal-priority Japanese pronunciation')
            spoken[key]=e['reading_kana']
    return result


def import_csv(content):
    if len(content)>512*1024:raise ValueError('Dictionary CSV exceeds512KiB')
    reader=csv.DictReader(io.StringIO(content.decode('utf-8-sig')))
    if not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames) or set(reader.fieldnames)-set(FIELDS)-{'legacy_english'}:raise ValueError('Invalid dictionary CSV columns')
    entries=[]
    for row in reader:
        if len(entries)>=512 or None in row:raise ValueError('Invalid/oversized dictionary CSV')
        e={k:v for k,v in row.items() if v not in ('',None)}
        for key in ['enabled','case_sensitive']:
            if key in e:
                if e[key].lower() not in {'true','false'}:raise ValueError('CSV switches must be true/false')
                e[key]=e[key].lower()=='true'
        if 'priority' in e:e['priority']=int(e['priority'])
        if 'aliases' in e:e['aliases']=json.loads(e['aliases'])
        legacy=e.pop('legacy_english','false').lower()
        if legacy not in {'true','false'}:raise ValueError('Invalid legacy CSV flag')
        if legacy=='true':
            if e.get('source_language')!='en' or e.get('pinyin') or e.get('entry_type')!='technical_term':raise ValueError('Invalid legacy English CSV record')
            e=dict(source=e['source_term'],japanese=e['target_ja'],reading=e['reading_kana'],**{k:e[k] for k in ['aliases','case_sensitive','enabled','priority','match_mode','notes'] if k in e})
        entries.append(e)
    return dict(schema=1 if entries and all('source' in e for e in entries) else 2,entries=entries)


def export_csv(data):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=FIELDS+['legacy_english']);writer.writeheader()
    for raw in data['entries']:
        e=canonical(raw);e['legacy_english']=str(e.pop('legacy')).lower();e['aliases']=json.dumps(e['aliases'],ensure_ascii=False)
        e['enabled']=str(e['enabled']).lower();e['case_sensitive']=str(e['case_sensitive']).lower();writer.writerow(e)
    return stream.getvalue().encode('utf-8-sig')
