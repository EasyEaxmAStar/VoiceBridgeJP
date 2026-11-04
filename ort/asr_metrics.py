"""Deterministic Mandarin/mixed-text CER and separately aligned punctuation."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

NORMALIZATION = "NFKC + casefold + Unicode alphanumeric; no script/numeral/homophone folding"
_PUNCTUATION = {"。": ".", "“": '"', "”": '"', "‘": "'", "’": "'"}


def normalized_text(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def edit_distance(reference: Sequence, hypothesis: Sequence) -> int:
    row = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        new = [i]
        for j, right in enumerate(hypothesis, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = new
    return row[-1]


def _boundary_alignment(reference: str, hypothesis: str) -> dict[int, int]:
    """Align hypothesis punctuation anchors through content-character edits."""
    table = [list(range(len(hypothesis) + 1))]
    for i, left in enumerate(reference, 1):
        row = [i]
        for j, right in enumerate(hypothesis, 1):
            row.append(min(row[-1] + 1, table[-1][j] + 1,
                           table[-1][j - 1] + (left != right)))
        table.append(row)
    i, j = len(reference), len(hypothesis)
    anchors = {j: i}
    while i or j:
        if i and j and table[i][j] == table[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            i, j = i - 1, j - 1
        elif j and table[i][j] == table[i][j - 1] + 1:
            j -= 1
        else:
            i -= 1
        anchors.setdefault(j, i)
    return anchors


def _punctuation(text: str) -> list[tuple[int, str]]:
    anchors = []
    position = 0
    for char in unicodedata.normalize("NFKC", text).casefold():
        if char.isalnum():
            position += 1
        elif unicodedata.category(char).startswith("P"):
            anchors.append((position, _PUNCTUATION.get(char, char)))
    return anchors


def score_text(reference: str, hypothesis: str, *, punctuation_known: bool = False) -> dict:
    ref, hyp = normalized_text(reference), normalized_text(hypothesis)
    if not ref:
        raise ValueError("Reference must contain real alphanumeric transcript characters")
    errors = edit_distance(ref, hyp)
    result = {"reference_characters": len(ref), "character_errors": errors,
              "cer": errors / len(ref), "punctuation_errors": None,
              "reference_punctuation": None}
    if punctuation_known:
        alignment = _boundary_alignment(ref, hyp)
        ref_punct = _punctuation(reference)
        hyp_punct = [(alignment[position], mark) for position, mark in _punctuation(hypothesis)]
        result.update(punctuation_errors=edit_distance(ref_punct, hyp_punct),
                      reference_punctuation=len(ref_punct))
    return result
