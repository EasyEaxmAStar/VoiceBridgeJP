"""Source-anchored noun terminology; original ASR is never edited.

Only dictionary-selected, token-boundary spans are protected inside a separate
MT input. Missing/duplicated markers fail closed, not a best-effort replacement.
"""
import copy,re,unicodedata


class TerminologyError(ValueError):pass


class TranslationTerms:
    def __init__(self,source,dictionary,language='zh'):
        self.source=source;self.rules=copy.deepcopy(dictionary.source_matches(source,language))
        if len(self.rules)>32:raise TerminologyError('At most32 terminology occurrences per utterance')
        if self.rules and re.search(r'ZXQ[A-Z]+QXZ',source):raise TerminologyError('Source collides with protected terminology marker')
        self.markers={};parts=[];last=0
        for i,rule in enumerate(self.rules):
            marker='ZXQ'+chr(65+i//26)+chr(65+i%26)+'QXZ'
            self.markers[marker]=rule['target_ja'];rule['marker']=marker
            parts.extend([source[last:rule['start']],marker]);last=rule['end']
        parts.append(source[last:]);self.model_input=''.join(parts);self.raw_candidate=None;self.restored_candidate=None
    @property
    def active(self):return bool(self.rules)
    def restore(self,candidate):
        self.raw_candidate=candidate
        if not self.active:self.restored_candidate=candidate;return candidate
        found=re.findall(r'ZXQ[A-Z]+QXZ',candidate)
        if sorted(found)!=sorted(self.markers) or any(candidate.count(marker)!=1 for marker in self.markers):
            raise TerminologyError('MT omitted, duplicated or altered a protected term; no speech generated')
        result=re.sub(r'ZXQ[A-Z]+QXZ',lambda match:self.markers[match.group()],candidate)
        self.restored_candidate=result;return result
    def validate(self,japanese):
        # No unapproved paraphrase substitutes for a required Japanese noun.
        for target in self.markers.values():
            required=sum(value==target for value in self.markers.values())
            if japanese.count(target)<required:raise TerminologyError('Polite output lost a required Japanese term')
        source=self.source
        for rule in reversed(self.rules):source=source[:rule['start']]+' '+source[rule['end']:]
        # Conservative checks reject unresolved polarity/numeric changes. They
        # never invent missing content, conjugate verbs or prove general meaning.
        without_positive=source
        for phrase in ['不错','不得不','不仅','不但','差不多']:without_positive=without_positive.replace(phrase,'')
        if re.search(r'不是|不要|不能|不会|没有|不使用|不采用|不支持|别|未确认|禁止',without_positive):
            if not re.search(r'ない|なく|ません|禁止|未確認|不可能|非対応|できず',japanese):raise TerminologyError('Negative source polarity not verified in Japanese')
        for number in re.findall(r'\d+(?:[.:]\d+)*',unicodedata.normalize('NFKC',source)):
            if not re.search(r'(?<!\d)'+re.escape(number)+r'(?!\d)',unicodedata.normalize('NFKC',japanese)):
                raise TerminologyError('Numeric source content not verified in Japanese')
    def reviewed_contrast(self):
        if len(self.markers)!=2:return None
        a,b=list(self.markers)
        stripped=re.sub(r'[\s，。！？、,.!?]','',self.model_input)
        if stripped=='这不是'+a+'而是'+b:
            return 'これは'+self.markers[a]+'ではなく、'+self.markers[b]+'です。'
        return None
    def reviewed_candidate(self):
        contrast=self.reviewed_contrast()
        if contrast:return contrast
        if not self.markers:return None
        stripped=re.sub(r'[\s。！？!?]','',self.model_input)
        token='(?:'+'|'.join(map(re.escape,self.markers))+')'
        match=re.fullmatch(r'(我|我们)(不使用|使用)('+token+r'(?:和'+token+r'){0,3})',stripped)
        if match and all(not re.search(r'[。！？!?]',target) for target in self.markers.values()):
            terms=[self.markers[m] for m in re.findall(r'ZXQ[A-Z]+QXZ',match[3])]
            return ('私は' if match[1]=='我' else '私たちは')+'と'.join(terms)+('を使用していません。' if match[2]=='不使用' else 'を使用しています。')
        return None
    def audit(self):
        return dict(active=self.active,source_text=self.source,model_input=self.model_input,
            raw_model_translation=self.raw_candidate,restored_candidate=self.restored_candidate,
            rules=copy.deepcopy(self.rules),method='token_boundary_protected_nouns',reviewed_contrast=bool(self.reviewed_contrast()),reviewed_source_template=bool(self.reviewed_candidate()))
