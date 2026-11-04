"""Conservative, offline business-Japanese gate. Unresolved output is not spoken.

Exact utterance mappings are reviewed translations, not suffix replacement.
Morphological checks distinguish embedded plain forms from sentence predicates.
This gate is deliberately incomplete; it cannot prove universal grammaticality.
"""
import re, time, unicodedata

CURATED={
 '你好':'こんにちは。','谢谢':'ありがとうございます。','谢谢你':'ありがとうございます。',
 '嗯知道了':'はい、承知しました。','好的我明白了':'はい、承知しました。','明白了':'承知しました。','好的':'はい、承知しました。',
 '是的':'はい、そのとおりです。','不是':'いいえ、違います。','不行':'対応できません。',
 '我不知道':'存じません。','不知道':'存じません。','不可能':'不可能です。',
 '等一下':'少々お待ちください。','稍等一下':'少々お待ちください。',
 '把这个发给我':'こちらをお送りください。','请把这个发给我':'こちらをお送りください。',
 '对不起':'申し訳ありません。','不好意思':'恐れ入ります。','我不同意':'賛成できません。',
 '我不能接受':'受け入れることはできません。','我拒绝':'お断りいたします。',
 '请再说一遍':'もう一度お話しいただけますか。','请问会议几点开始':'会議は何時からですか。',
 '很高兴认识您':'お会いできてうれしく思います。','今天请多关照':'本日はよろしくお願いいたします。',
 '目前的系统采用Python和MySQL':'現在のシステムはPythonとMySQLを使用しています。',
 '今天下午三点开会':'本日午後三時に会議を行います。',
 '我觉得这个方案可以但是成本可能比较高':'この案はよいと思いますが、費用が高くなる可能性があります。',
 '如果明天有时间的话我们可以先测试一下':'明日お時間がありましたら、先にテストしてみることができます。',
 '这个项目最重要的不是速度而是稳定性':'このプロジェクトで最も重要なのは、速度ではなく安定性です。',
 '先确认一下数据库的连接然后检查接口':'まずデータベースの接続を確認してから、インターフェースを確認してください。',
 '我刚才说的是Azure不是其他云平台':'先ほど申し上げたのはAzureで、ほかのクラウドプラットフォームではありません。',
}
EXACT_JA={'うん、分かった。':'はい、承知しました。','分かった。':'承知しました。',
 'ちょっと待て。':'少々お待ちください。','これを送って。':'こちらをお送りください。',
 '知らない。':'存じません。','ありがとう。':'ありがとうございます。','無理だ。':'対応できません。'}

def source_key(text):
    normalized=unicodedata.normalize('NFKC',text)
    return ''.join(c for i,c in enumerate(normalized) if not c.isspace() and
        (c not in '，。！？、,.!?' or c=='.' and 0<i<len(normalized)-1 and normalized[i-1].isdigit() and normalized[i+1].isdigit()))

class TranslationRejected(ValueError):
    def __init__(self,audit):super().__init__('译文尚未通过商务敬体校验，本句未播放');self.audit=audit


class PoliteJapanese:
    def __init__(self,morph):self.morph=morph
    def validate(self,text,source=None,*,recovery=False):
        tokens=self.morph(text)
        if not tokens:return False,'empty_morphology'
        # Proper substring boundaries come from OpenJTalk, not a substring blacklist.
        for i,token in enumerate(tokens):
            if token['string'] in {'お前','てめえ','貴様','うん','ううん'}:return False,'casual_or_rude_token'
            if token['string']=='ありがとう' and (i+1==len(tokens) or tokens[i+1]['orig'] not in {'ござる','ございます'}):
                return False,'casual_thanks_without_polite_auxiliary'
            if '命令' in token['cform'] and not (token['orig']=='くださる' and token['string']=='ください'):
                return False,'unresolved_imperative'
        sentences=re.split(r'[。！？!?\n]+',text)
        for sentence in sentences:
            sentence=sentence.strip(' \u3000「」『』"、,')
            if not sentence:continue
            # Professional fixed expressions may have no predicate. All other
            # sentences need a token-aligned polite terminal construction.
            if sentence in {'はい','いいえ','こんにちは','おはようございます','こんばんは','恐れ入ります'}:continue
            if recovery and sentence=='ええと' and source_key(source or '')=='呃':continue
            local=[t for t in self.morph(sentence) if t['pos']!='記号']
            if not local:return False,'missing_predicate'
            while local and local[-1]['pos']=='助詞' and local[-1]['string'] in {'か','ね','よ'}:local.pop()
            if not local:return False,'casual_fragment'
            joined=''.join(t['string'] for t in local)
            polite=any(joined.endswith(x) for x in ('です','ます','ません','ました','でした','ませんでした','でしょう','ございます','ください'))
            if not polite:
                # A committed connective is not a standalone casual assertion.
                # Require an explicit matching Chinese construction; no future
                # audio wait, invented clause or general suffix rewriting.
                key=source_key(source or '')
                connective=(key.startswith('如果') and joined.endswith(('れば','たら','場合'))
                    or key.startswith('因为') and joined.endswith(('ため','ので'))
                    or key.startswith('虽然') and joined.endswith(('ものの','ても')))
                if recovery:
                    connective=connective or (key.startswith(('如果','假如')) and joined.endswith(('場合に','なら','ならば','でしたら'))
                        or key.startswith(('因为','由于')) and joined.endswith(('ために','から','からです','ので'))
                        or key.startswith('虽然') and joined.endswith(('にもかかわらず','けれども','ですが','ものの')))
                if not (connective and len([s for s in sentences if s.strip()])==1):
                    return False,'unresolved_plain_predicate_or_fragment'
            # A polite ending cannot license an earlier standalone plain finite
            # clause separated by a comma. Embedded modifiers remain valid.
            for clause in re.split('[、,]',sentence)[:-1]:
                ct=[t for t in self.morph(clause) if t['pos']!='記号']
                if ct and ct[-1]['pos'] in {'動詞','形容詞','助動詞'} and ct[-1]['cform']=='基本形' and ct[-1]['orig'] not in {'です','ます'}:
                    return False,'plain_finite_clause'
        return True,'morphology_and_polite_terminal'
    def apply(self,source,candidate,snapshot):
        start=time.perf_counter();settings=snapshot['settings'];key=source_key(source)
        preferred={source_key(k):v for k,v in settings['expressions'].items()}
        translated=candidate
        if settings['formality']=='STRICT_BUSINESS_POLITE':
            translated=preferred.get(key,candidate if snapshot.get('prefer_terminology_candidate') else CURATED.get(key,candidate))
            translated=EXACT_JA.get(translated,translated)
        translated=snapshot['dictionary'].terminology(translated)
        transformations=[]
        if settings['formality']=='STRICT_BUSINESS_POLITE':
            pieces=re.split(r'([。！？!?]+)',translated)
            for i in range(0,len(pieces),2):
                part=pieces[i]
                if not part.strip():continue
                tokens=[t for t in self.morph(part) if t['pos']!='記号']
                if (snapshot.get('recovery_policy',False) and len(tokens)>=3
                    and tokens[-3]['pos']=='名詞' and tokens[-2]['string']=='で'
                    and tokens[-2]['orig']=='だ' and tokens[-2]['pos']=='助動詞'
                    and tokens[-1]['string']=='ある' and tokens[-1]['pos']=='助動詞'
                    and part.endswith('である')):
                    pieces[i]=part[:-3]+'です';transformations.append('verified_nominal_dearu_copula')
                # Only a positively identified nominal copula. No general verb
                # conjugation, noun-fragment guessing or negative rewriting.
                if (len(tokens)>=2 and tokens[-1]['string']=='だ' and tokens[-1]['orig']=='だ'
                    and tokens[-1]['pos']=='助動詞' and tokens[-2]['pos']=='名詞'
                    and not any(t['string'] in {'何','なん','誰','お前'} for t in tokens)):
                    pieces[i]=part[:-1]+'です';transformations.append('morphological_nominal_copula')
            translated=''.join(pieces)
        audit=dict(source_text=source,baseline_translation=candidate,japanese_text=translated,
            policy=settings['formality'],settings_sha256=snapshot['sha256'],curated=key in CURATED or key in preferred,transformations=transformations)
        if settings['formality']=='STRICT_BUSINESS_POLITE':
            valid,reason=self.validate(translated,source=source if snapshot.get('allow_connective_fragments',False) or snapshot.get('recovery_policy',False) else None,recovery=snapshot.get('recovery_policy',False))
        else:valid,reason=True,'explicit_policy_off'
        audit.update(accepted=valid,reason=reason,seconds=time.perf_counter()-start)
        if snapshot.get('recovery_policy',False):audit['decision_level']='GREEN' if valid else 'YELLOW'
        if not valid:raise TranslationRejected(audit)
        return translated,audit
