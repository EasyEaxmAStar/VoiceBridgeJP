"""At most two verified source-conditioned repairs; never generate error speech.

No prompt is sent to OPUS: it is not an instruction-following model. The second
pass composes only already-finalized, individually reviewed source clauses.
Unknown/incomplete content remains private instead of guessing intent.
"""
import re,time
from .polite_japanese import CURATED,TranslationRejected,source_key

SOURCE_REPAIRS={
    '不':'いいえ。', '没办法':'仕方がありません。', '我不想去':'行きたくありません。',
    '你给我滚':'ここから出ていってください。', '你是谁':'どなたですか。',
    '我没有同意':'同意していません。', '我同意':'同意します。',
    '好的谢谢今天辛苦了':'はい、ありがとうございます。本日はお疲れさまでした。',
    '我不去':'行きません。','我不能去':'行くことはできません。','我还没有同意':'まだ同意していません。',
    '我不同意这个决定':'この決定に同意しません。','这不可能':'それは不可能です。',
    '我知道了':'承知しました。','我不认识他':'その方とは面識がありません。',
    '先确认连接':'まず接続を確認してください。','然后检查API':'次にAPIを確認してください。',
    '我先确认连接':'まず接続を確認します。','我然后检查API':'次にAPIを確認します。',
    '我先确认数据库的连接':'まずデータベースの接続を確認します。',
    '请确认Python3.12的版本':'Python 3.12のバージョンを確認してください。',
    '现在不要重启MySQL':'今はMySQLを再起動しないでください。',
    '请不要删除PostgreSQL的数据':'PostgreSQLのデータを削除しないでください。',
    '换一首歌':'別の曲に替えてください。',
    '呃不是':'いいえ、違います。','呃不是这个意思':'いいえ、そのような意味ではありません。',
}

class BusinessRewrite:
    def __init__(self,gate):self.gate=gate
    def apply(self,source,candidate,snapshot):
        started=time.perf_counter()
        explicit={source_key(k) for k in snapshot['settings']['expressions']}
        try:
            text,audit=self.gate.apply(source,candidate,snapshot)
            reviewed=SOURCE_REPAIRS.get(source_key(source))
            if reviewed and source_key(source) not in explicit and text!=snapshot['dictionary'].terminology(reviewed):
                corrected=snapshot['dictionary'].terminology(reviewed)
                valid,reason=self.gate.validate(corrected)
                if not valid:raise TranslationRejected({**audit,'reason':reason})
                audit.update(japanese_text=corrected,semantic_source_override=True)
                text=corrected
            audit.update(outcome='ORIGINAL_POLITE' if text==candidate else 'REWRITTEN_POLITE',initial_policy_accepted=True,repair_attempts=[],rewrite_seconds=0.)
            return text,audit
        except TranslationRejected as exc:audit=exc.audit
        attempts=[];key=source_key(source)
        # Exact reviewed source semantics preserve negation/aspect and intent.
        target=SOURCE_REPAIRS.get(key)
        proposals=[('reviewed_whole_source',target)]
        # Bounded second pass: no new context, no unbounded retry, no nominal
        # fragment -> guessed statement. All component intents must be known.
        clauses=[x.strip() for x in re.split('[。！？!?]+',source) if x.strip()]
        known={**CURATED,**SOURCE_REPAIRS}
        composed=(''.join(known[source_key(c)] for c in clauses)
            if 2<=len(clauses)<=4 and all(source_key(c) in known for c in clauses) else None)
        proposals.append(('reviewed_final_clause_composition',composed))
        for rule,proposal in proposals:
            if not proposal:
                attempts.append(dict(rule=rule,accepted=False,reason='source_intent_not_verified'));continue
            text=snapshot['dictionary'].terminology(proposal)
            valid,reason=self.gate.validate(text)
            attempts.append(dict(rule=rule,accepted=valid,reason=reason,candidate=text,semantic_evidence='exact reviewed Chinese intent; no inferred future context'))
            if valid:
                audit.update(japanese_text=text,accepted=True,outcome='REWRITTEN_POLITE',initial_policy_accepted=False,
                    reason='verified_source_repair',repair_attempts=attempts,rewrite_seconds=time.perf_counter()-started,seconds=time.perf_counter()-started)
                return text,audit
        audit.update(accepted=False,outcome='UNRESOLVED_PRIVATE',initial_policy_accepted=False,repair_attempts=attempts,
            rewrite_seconds=time.perf_counter()-started,seconds=time.perf_counter()-started)
        raise TranslationRejected(audit)
