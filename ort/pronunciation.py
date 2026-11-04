"""Bounded, deterministic Japanese TTS-only pronunciation lexicon.

NFKC/casefold is used for matching, never for rewriting untouched text. Edits
are made once against the original string. URL/path/code spans are protected.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata as ud


def _fold(value):
    return ud.normalize("NFKC", value).casefold()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate pronunciation JSON key: {key}")
        result[key] = value
    return result


def _identifier(char):
    return bool(char) and (char in {"_", "-"} or char.isdecimal() or
        "LATIN" in ud.name(char, "") or ud.category(char).startswith("M"))


_PROTECTED = re.compile(
    r"`[^`]*`|(?:[a-z][a-z0-9+.-]*://|www\.)[^\s<>\u3000]+|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"(?:[A-Za-z]:[\\/]|\\\\|[./~]*[/\\])[\w./\\~:+-]+|"
    r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)+|"
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+(?:\([^\n]*?\))?|"
    r"[A-Za-z_]\w*\([^\n]*?\)|"
    r"[A-Za-z_]\w*\.(?:py|sql|json|exe|md|txt)\b", re.I)


class PronunciationLexicon:
    def __init__(self, content: bytes):
        if len(content) > 128 * 1024:
            raise ValueError("Pronunciation lexicon exceeds 128 KiB")
        data = json.loads(content.decode("utf-8-sig"), object_pairs_hook=_unique)
        if (not isinstance(data, dict) or set(data) != {"schema_version", "entries"}
                or type(data["schema_version"]) is not int or data["schema_version"] != 1
                or not isinstance(data["entries"], list) or len(data["entries"]) > 512):
            raise ValueError("Expected pronunciation schema_version 1 and up to 512 entries")
        entries = []
        keys = set()
        for rule in data["entries"]:
            if not isinstance(rule,dict) or set(rule) != {"term", "reading"}:
                raise ValueError("Each pronunciation entry needs exactly term and reading")
            term, reading = rule["term"], rule["reading"]
            if (not isinstance(term,str) or not isinstance(reading,str) or
                    not 1 <= len(term) <= 80 or not 1 <= len(reading) <= 120 or
                    term != term.strip() or reading != reading.strip() or
                    any(ud.category(c).startswith("C") for c in term+reading)):
                raise ValueError("Invalid pronunciation term/reading")
            key = _fold(term)
            # Readings are deliberately phonetic Japanese, not arbitrary prose
            # or recursively evaluated templates. Unknown words stay unchanged.
            if not re.fullmatch(r"[\u3041-\u3096\u30a1-\u30fa\u30fc\u30fb ]+",reading):
                raise ValueError("Pronunciation reading must be Japanese kana")
            if key in keys:
                raise ValueError("Ambiguous normalized pronunciation term: " + term)
            keys.add(key)
            entries.append((key, term, reading))
        self.entries = sorted(entries,key=lambda x:(-len(x[0]),x[0]))
        self.fingerprint = hashlib.sha256(content).hexdigest()
        self.path = None

    @property
    def metadata(self):
        return dict(path=self.path,rule_count=len(self.entries),sha256=self.fingerprint)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as stream:
            lexicon = cls(stream.read(128*1024+1))
        lexicon.path = str(Path(path).resolve())
        return lexicon

    def apply(self, text):
        started = time.perf_counter()
        if not isinstance(text,str) or len(text) > 20000:
            raise ValueError("Pronunciation input must be text of at most 20000 characters")
        # Per-character mapping preserves exact original offsets even for width
        # variants / casefold expansions. Multi-codepoint composed rules that
        # cannot be matched safely are left unchanged, never guessed.
        folded, starts, ends = [], [], []
        for i,char in enumerate(text):
            part = _fold(char)
            folded.append(part)
            starts.extend([i]*len(part)); ends.extend([i+1]*len(part))
        folded = "".join(folded)
        protected = [False]*len(folded)
        for match in _PROTECTED.finditer(folded):
            protected[match.start():match.end()] = [True]*(match.end()-match.start())
        applied, pieces = [], []
        cursor = last = 0
        while cursor < len(folded):
            found = None
            if not protected[cursor] and (cursor==0 or starts[cursor]!=starts[cursor-1]):
                for key, term, reading in self.entries:
                    end = cursor+len(key)
                    if (folded.startswith(key,cursor) and not any(protected[cursor:end])
                            and (end==len(folded) or ends[end-1]!=ends[end])
                            and not _identifier(folded[cursor-1] if cursor else "")
                            and not _identifier(folded[end] if end<len(folded) else "")):
                        found = (end,term,reading)
                        break
            if found:
                end,term,reading = found
                start_original, end_original = starts[cursor],ends[end-1]
                pieces.extend([text[last:start_original],reading])
                applied.append(dict(term=term,matched=text[start_original:end_original],reading=reading,
                                    start=start_original,end=end_original))
                last = end_original; cursor = end
            else:
                cursor += 1
        pieces.append(text[last:])
        return "".join(pieces), {"original_tts_text":text,"spoken_tts_text":"".join(pieces),
            "rules":applied,"lexicon_sha256":self.fingerprint,
            "processing_seconds":time.perf_counter()-started}
