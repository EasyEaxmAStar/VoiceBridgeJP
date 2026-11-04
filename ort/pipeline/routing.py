"""Conservative utterance routing; ASR language is evidence, not confidence."""
import re


def japanese_route(text, source):
    kana = bool(re.search(r"[\u3040-\u30ff]", text))
    han = bool(re.search(r"[\u3400-\u9fff]", text))
    if source == "ja":
        # Known Mandarin grammatical material contradicts Japanese ASR's single
        # language label. Do not invent a reliable span classifier from scripts.
        if re.search(r"我们|这个|关于|我认为|我想|的系统|目前|需要|请问", text):
            raise ValueError("Uncertain Mandarin/Japanese mixture; explicit review required")
        return "ja_direct", "ASR Japanese; original text preserved (no confidence score)"
    if source == "zh" and not kana:
        return "zh_mt", "ASR Chinese; Latin identifiers remain in the same utterance"
    if source == "en" and not kana and not han:
        return "en_mt", "ASR English with no contradictory CJK script evidence"
    raise ValueError("Uncertain or unsupported language mixture; no translation fallback")
