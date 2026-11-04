"""Text-only production policy sharing the existing resident model pipeline."""
from collections import deque
import threading
from .features import FeatureStore
from .japanese_morph import Morphology
from .polite_japanese import PoliteJapanese, TranslationRejected
from .polite_rewrite import BusinessRewrite
from .terminology import TerminologyError


class ProductionPolicy:
    def __init__(self,store=None,morph=None):
        self.store=store or FeatureStore();self.morph=morph or Morphology()
        self.polite=PoliteJapanese(self.morph);self.rewriter=BusinessRewrite(self.polite);self.audit=deque(maxlen=256);self.rejections=0;self._lock=threading.Lock()
    def snapshot(self):return self.store.snapshot()
    def direct(self,source,snapshot):
        result=snapshot['dictionary'].direct_output(source)
        from .polite_japanese import source_key
        if not result:
            for original,target in snapshot['settings']['expressions'].items():
                if source_key(original)==source_key(source) and bool(set(original)&set('?？'))==bool(set(source)&set('?？')):
                    result=dict(text=target,rule_id='preferred_expression:'+original);break
        if result:
            valid,reason=self.polite.validate(result['text'])
            if not valid:raise TranslationRejected(dict(source_text=source,accepted=False,reason=reason,outcome='UNRESOLVED_PRIVATE'))
        return result
    def translate(self,source,candidate,snapshot):
        direct=self.direct(source,snapshot)
        if direct:
            audit=dict(source_text=source,baseline_translation=candidate,japanese_text=direct['text'],accepted=True,
                outcome='DICTIONARY_DIRECT',rule_id=direct['rule_id'],settings_sha256=snapshot['sha256'],seconds=0.)
            with self._lock:self.audit.append(audit)
            return direct['text'],audit
        plan=snapshot.get('translation_terms')
        try:
            text,audit=self.rewriter.apply(source,candidate,snapshot)
            if plan and plan.active:
                plan.validate(text);audit['terminology']=plan.audit()
        except TerminologyError as exc:
            audit=dict(source_text=source,baseline_translation=candidate,accepted=False,reason='terminology_unverified',
                details=str(exc),settings_sha256=snapshot['sha256'],outcome='UNRESOLVED_PRIVATE',terminology=plan.audit() if plan else None)
            with self._lock:self.audit.append(audit);self.rejections+=1
            raise TranslationRejected(audit) from exc
        except TranslationRejected as exc:
            with self._lock:self.audit.append(exc.audit);self.rejections+=1
            raise
        except Exception as exc:
            audit=dict(source_text=source,baseline_translation=candidate,accepted=False,reason='validator_unavailable',
                error_type=type(exc).__name__,settings_sha256=snapshot['sha256'],outcome='TECHNICAL_ERROR')
            with self._lock:self.audit.append(audit);self.rejections+=1
            raise TranslationRejected(audit) from exc
        with self._lock:self.audit.append(audit)
        return text,audit
    def translate_model(self,translator,source,snapshot):
        """Text preview uses exactly the same protected source projection as WS."""
        if self.direct(source,snapshot):return self.translate(source,'',snapshot)
        from .terminology import TranslationTerms
        plan=TranslationTerms(source,snapshot['dictionary'],'zh')
        candidate=translator.translate(plan.model_input,(),'zh','ja')
        if plan.active:
            try:candidate=plan.restore(candidate);candidate=plan.reviewed_candidate() or candidate
            except TerminologyError as exc:
                audit=dict(source_text=source,baseline_translation=candidate,accepted=False,reason='terminology_unverified',details=str(exc),outcome='UNRESOLVED_PRIVATE',terminology=plan.audit())
                with self._lock:self.audit.append(audit);self.rejections+=1
                raise TranslationRejected(audit) from exc
            snapshot={**snapshot,'translation_terms':plan,'prefer_terminology_candidate':True}
        return self.translate(source,candidate,snapshot)
    def save(self,value):
        if getattr(self,'asr_names',None):
            from .user_dictionary import UserDictionary
            from ort.asr_vocabulary import with_dictionary
            with_dictionary(self.asr_names.snapshot(),UserDictionary(value['dictionary'],self.store.base))
        # User preferred expressions must themselves pass the same runtime gate.
        for target in value.get('expressions',{}).values():
            valid,_=self.polite.validate(target)
            if not valid:raise ValueError('Preferred expression did not pass professional Japanese validation')
        return self.store.save(value)
    def status(self,include_settings=True):
        snapshot=self.snapshot()
        settings=snapshot['settings']
        if not include_settings:settings={k:v for k,v in settings.items() if k not in {'dictionary','expressions'}}
        return dict(settings=settings,sha256=snapshot['sha256'],rejections=self.rejections,
            morphology_pid=getattr(getattr(self.morph,'process',None),'pid',None))
    def close(self):
        close=getattr(self.morph,'close',None)
        if close:close()
