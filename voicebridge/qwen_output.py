"""User-approved continuity policy: heuristics warn; technical failures block."""
import json,re,unicodedata
from pathlib import Path
from .hybrid_translation import semantic_sentinels

class QwenTechnicalFailure(ValueError):pass
class FillerSkipped(ValueError):pass

def filler_key(text):return unicodedata.normalize('NFKC',text).strip(' \t\r\n。.!！?？、,…')

def load_fillers(path=None):
    value=json.loads(Path(path or Path(__file__).resolve().parents[1]/'config/qwen_fillers.json').read_text(encoding='utf8'))
    entries=value.get('standalone')
    if value.get('schema')!=1 or not isinstance(entries,list) or len(entries)>64 or any(not isinstance(v,str) or not 1<=len(v)<=20 for v in entries):raise ValueError('Invalid standalone filler configuration')
    protected={'はい','いいえ','ええ','好的','明白了','ありがとうございます','承知しました'}
    result={filler_key(v) for v in entries}
    if result&protected:raise ValueError('Meaningful acknowledgments cannot be configured as fillers')
    return result

def accept_qwen(source,text,snapshot,*,fillers=None):
    if not isinstance(text,str) or not text.strip() or len(text)>3000:raise QwenTechnicalFailure('Empty or oversized Qwen output')
    try:text.encode('utf8',errors='strict')
    except UnicodeError as exc:raise QwenTechnicalFailure('Invalid Qwen encoding') from exc
    if any(unicodedata.category(c)=='Cc' and c not in '\n\t' for c in text) or re.search(r'<think>|```',text):raise QwenTechnicalFailure('Malformed Qwen text framing')
    try:structured=json.loads(text)
    except json.JSONDecodeError:structured=None
    if isinstance(structured,(dict,list)):raise QwenTechnicalFailure('Structured data is not a text utterance')
    fillers=load_fillers() if fillers is None else fillers
    if filler_key(source) in fillers or filler_key(text) in fillers:raise FillerSkipped('FILLER_SKIPPED')
    warnings=[]
    if not re.search(r'[\u3040-\u30ff\u3400-\u9fff]',text):warnings.append('Japanese_language_unverified_ascii_term_or_other_language')
    try:
        if not semantic_sentinels(source,text,snapshot['dictionary'],recovery=True):warnings.append('source_heuristics_unresolved')
    except Exception as exc:warnings.append('audit_unavailable:'+type(exc).__name__)
    # This is intentionally NOT semantic verification or a style veto.
    return text,dict(outcome='QWEN_ACCEPTED_WITH_AUDIT_WARNINGS' if warnings else 'QWEN_ACCEPTED',
        accepted=True,semantic_verification='UNVERIFIED',audit_only_warnings=warnings,
        source_text=source,japanese_text=text,settings_sha256=snapshot['sha256'])
