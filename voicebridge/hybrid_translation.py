"""Opt-in translation experiment. Returns text once, never owns audio output."""
import asyncio,itertools,re,time
from collections import Counter

MODES=('FAST_ONLY','SEQUENTIAL_FALLBACK','HYBRID_SPECULATIVE')

def risk_reasons(source,dictionary):
    patterns={'negation':r'不|没|未|禁止','conditional':r'如果|假如|只要|除非','causal':r'因为|所以|由于',
              'fragment':r'虽然|但是|的话[。…]*$','numbers':r'\d','technical':r'[A-Za-z]{2,}'}
    reasons=[name for name,pattern in patterns.items() if re.search(pattern,source)]
    if dictionary.source_matches(source):reasons.append('dictionary_terms')
    return reasons

def semantic_sentinels(source,target,dictionary,*,recovery=False):
    """Necessary checks, never proof of translation correctness."""
    if not isinstance(target,str) or not target.strip() or len(target)>max(120,len(source)*6):return False
    if not re.search('[\u3040-\u30ff\u3400-\u9fff]',target):return False
    if re.search(r'<think>|```|翻訳に失敗|翻訳できません|translation failed',target,re.I):return False
    pieces=[s.strip() for s in re.split('[。！？!?\n]+',target) if s.strip()]
    if any(pieces.count(s)>=3 for s in pieces):return False
    for n in re.findall(r'\d+(?:\.\d+)?',source):
        if not re.search(r'(?<!\d)'+re.escape(n)+r'(?!\d)',target):return False
    if re.search(r'没有|不能|不要|不是|不使用|不同意|无法|未完成',source) and not re.search(r'ない|なく|ません|未|不可能|できず',target):
        if not (recovery and source.strip('。 .')=='不是' and target.strip('。 .') in {'いいえ','いいえ、違います'}):return False
    if re.search(r'如果|假如|除非',source) and not re.search(r'なら|れば|たら|場合|限り',target):return False
    if recovery and source.startswith(('因为','由于')) and not re.search(r'ため|ので|から|理由',target):return False
    if recovery and source.startswith('虽然') and not re.search(r'ものの|ても|かかわらず|ですが|けれど|けど|が[、,]',target):return False
    modality=r'可能|かもしれ|おそらく|不明|確か|分かりません'
    if recovery:modality+=r'|わかりません|確信できません|確実.*ません'
    if re.search(r'可能|也许|不确定',source) and not re.search(modality,target):return False
    return all(r['target_ja'] in target for r in dictionary.source_matches(source))

class HybridTranslator:
    def __init__(self,opus,qwen,policy,*,mode='FAST_ONLY',timeout=8.,capacity=4,audio_busy=lambda:False,selective_speculation=False):
        if mode not in MODES:raise ValueError('Unknown translation strategy')
        self.opus,self.qwen,self.policy=opus,qwen,policy;self.mode=mode;self.timeout=timeout
        self.capacity=capacity;self.audio_busy=audio_busy;self.active=0;self.epoch=0;self.ids=itertools.count(1)
        self.selective_speculation=selective_speculation
        self.tasks=set();self.counts=Counter()
    def cancel(self):
        self.epoch+=1
        for task in list(self.tasks):task.cancel()
    async def recover_final(self,source,snapshot,*,fillers=None):
        """Shared one-request sequential recovery; never a second content veto."""
        from .qwen_output import accept_qwen,load_fillers,filler_key,FillerSkipped
        fillers=load_fillers() if fillers is None else fillers
        if filler_key(source) in fillers:raise FillerSkipped('FILLER_SKIPPED')
        self.counts['qwen_requests']+=1
        text=await self.qwen(source,snapshot)
        return accept_qwen(source,text,snapshot,fillers=fillers)
    async def translate(self,source,snapshot):
        if self.active>=self.capacity:raise RuntimeError('Translation queue capacity exceeded')
        self.active+=1;epoch=self.epoch;identifier=next(self.ids);started=time.perf_counter();tasks=[]
        risk=risk_reasons(source,snapshot['dictionary']);qtask=None;decisions=[]
        async def quality():
            if snapshot.get('qwen_final_policy',False):return await self.recover_final(source,snapshot)
            self.counts['qwen_requests']+=1
            text=await self.qwen(source,snapshot)
            semantic=semantic_sentinels(source,text,snapshot['dictionary'],recovery=snapshot.get('recovery_policy',False))
            decision=dict(backend='qwen',candidate=text,level='YELLOW' if semantic else 'RED',reason='style_review' if semantic else 'source_sentinels_unresolved');decisions.append(decision)
            if not semantic and snapshot.get('recovery_policy',False):
                self.counts['qwen_requests']+=1
                text=await self.qwen(source,{**snapshot,'translation_repair':True})
                semantic=semantic_sentinels(source,text,snapshot['dictionary'],recovery=True)
                decision=dict(backend='qwen_repair',candidate=text,level='YELLOW' if semantic else 'RED',reason='bounded_source_grounded_retry');decisions.append(decision)
            if not semantic:raise ValueError('Qwen semantic checks unresolved')
            try:result=self.policy.translate(source,text,{**snapshot,'prefer_terminology_candidate':True})
            except Exception as exc:decision['audit']=getattr(exc,'audit',None);raise
            decision.update(level='GREEN',selected_text=result[0]);return result
        def spawn(coro):
            task=asyncio.create_task(coro);tasks.append(task);self.tasks.add(task)
            task.add_done_callback(self.tasks.discard);return task
        try:
            direct=self.policy.direct(source,snapshot)
            if direct:result=self.policy.translate(source,'',snapshot);route='DICTIONARY_DIRECT'
            else:
                async with asyncio.timeout(self.timeout):
                    speculate=bool(risk)
                    if self.selective_speculation:
                        # Preserve all semantic checks, but do not run a GPU
                        # model merely because a common short negative exists.
                        speculate=('conditional' in risk or 'fragment' in risk or
                            ('negation' in risk and ('causal' in risk or 'numbers' in risk)) or
                            ('technical' in risk and 'numbers' in risk))
                    if self.mode=='HYBRID_SPECULATIVE' and speculate and not self.audio_busy():qtask=spawn(quality())
                    otask=spawn(self.opus(source,snapshot))
                    if qtask:
                        done,_=await asyncio.wait([otask,qtask],return_when=asyncio.FIRST_COMPLETED)
                        if qtask in done and not qtask.cancelled() and qtask.exception() is None:
                            result=qtask.result();route='PARALLEL_QWEN'
                            if epoch!=self.epoch:raise asyncio.CancelledError('Stale generation')
                            self.counts[route]+=1
                            return result[0],dict(id=identifier,generation=epoch,route=route,risk=risk,
                                seconds=time.perf_counter()-started,policy=result[1],counts=dict(self.counts),decisions=decisions)
                    # Low-risk OPUS is the fast path; risky OPUS must also meet
                    # independent source checks, not just style acceptance.
                    try:
                        result=await otask
                        decisions.append(dict(backend='opus',candidate=result[0],level='GREEN',audit=result[1]))
                        if risk and self.mode!='FAST_ONLY' and not semantic_sentinels(source,result[0],snapshot['dictionary'],recovery=snapshot.get('recovery_policy',False)):
                            decisions[-1].update(level='RED',reason='source_sentinels_unresolved');raise ValueError('OPUS semantic checks unresolved')
                        route='OPUS_FAST'
                    except asyncio.CancelledError:raise
                    except Exception as exc:
                        if not decisions or decisions[-1].get('backend')!='opus':decisions.append(dict(backend='opus',level='YELLOW',audit=getattr(exc,'audit',None),reason=str(exc)))
                        if self.mode=='FAST_ONLY':raise
                        if qtask is None:qtask=spawn(quality());route='SEQUENTIAL_QWEN'
                        else:route='PARALLEL_QWEN'
                        result=await qtask
            if epoch!=self.epoch:raise asyncio.CancelledError('Stale translation generation')
            self.counts[route]+=1
            return result[0],dict(id=identifier,generation=epoch,route=route,risk=risk,
                seconds=time.perf_counter()-started,policy=result[1],counts=dict(self.counts),decisions=decisions)
        except BaseException as exc:
            exc.translation_decisions=decisions
            raise
        finally:
            for task in tasks:
                if not task.done():task.cancel()
            # Retrieve all exceptions; cancelled workers implement their own
            # bounded teardown, with no callback able to commit audio.
            if tasks:await asyncio.gather(*tasks,return_exceptions=True)
            self.active-=1
