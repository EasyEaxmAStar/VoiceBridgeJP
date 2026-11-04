"""Language support tables and normalization helpers.

Supported end-to-end coverage is the intersection of the active model stages:
- Input/source speech depends on ASR. Qwen3-ASR supports 30 languages plus
  Chinese dialects; faster-whisper is available as a broader optional ASR.
- Translation uses TranslateGemma-4b-it, which supports 55 ISO 639-1 languages.
- Output/target speech depends on TTS. Qwen3-TTS supports 10 languages, Kokoro
  supports 8, Supertonic-3 supports 31, and macOS say depends on installed voices.

Available speech-to-speech pairs are ASR source coverage x active TTS targets.
"""

from __future__ import annotations

DEFAULT_TARGET_LANGUAGE = "en"

# ISO 639-1 -> display name for languages used by this project.
LANG_NAMES: dict[str, str] = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
}

# Qwen3-ASR's 30 spoken languages; it also supports Chinese dialects.
QWEN3_ASR_SOURCE = {
    "zh",
    "en",
    "yue",
    "ar",
    "de",
    "fr",
    "es",
    "pt",
    "id",
    "it",
    "ko",
    "ru",
    "th",
    "vi",
    "ja",
    "tr",
    "hi",
    "ms",
    "nl",
    "sv",
    "da",
    "fi",
    "pl",
    "cs",
    "fil",
    "fa",
    "el",
    "hu",
    "mk",
    "ro",
}

# Qwen3-ASR accepts language names, or None for auto-detection.
ASR_FORCE_NAME = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}

# Kokoro-82M: target ISO code -> (lang_code letter, default voice).
KOKORO_TARGET = {
    "en": ("a", "af_heart"),  # American English
    "en-GB": ("b", "bf_alice"),  # British English
    "ja": ("j", "jf_alpha"),
    "zh": ("z", "zf_xiaobei"),
    "es": ("e", "ef_dora"),
    "fr": ("f", "ff_siwis"),
    "hi": ("h", "hf_alpha"),
    "it": ("i", "if_sara"),
    "pt": ("p", "pf_dora"),  # Brazilian Portuguese
}

# Supertonic-3 target languages; pass the ISO 639-1 code directly as lang.
SUPERTONIC_TARGET = {
    "en",
    "ko",
    "ja",
    "ar",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "es",
    "et",
    "fi",
    "fr",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "lt",
    "lv",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "tr",
    "uk",
    "vi",
}

# Qwen3-TTS Base voice cloning: target ISO code -> Qwen3-TTS language name.
# Unsupported languages use 'auto', letting the model infer from text with lower diction quality.
QWEN3TTS_TARGET = {
    "zh": "chinese",
    "en": "english",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "pt": "portuguese",
    "ru": "russian",
    "es": "spanish",
}


def norm(code: str) -> str:
    """Normalize a language code: lowercase, underscores to hyphens, keep en-GB."""
    c = (code or "").strip().replace("_", "-")
    if c.lower() == "en-gb":
        return "en-GB"
    return c.split("-")[0].lower()


_LANG_NAME_TO_CODE = {name.lower(): code for code, name in LANG_NAMES.items()}
_LANG_NAME_TO_CODE.update(
    {name.lower(): code for code, name in ASR_FORCE_NAME.items()}
)


def source_code(language: str | None) -> str | None:
    """Return an ISO-ish source language code from an ASR language label."""
    if not language:
        return None
    raw = language.strip()
    code = norm(raw)
    if code in LANG_NAMES or code in ASR_FORCE_NAME:
        return code
    return _LANG_NAME_TO_CODE.get(raw.lower())


def tts_targets(tts_backend: str) -> set[str]:
    """Return target languages supported by the active TTS backend."""
    if tts_backend == "gptsovits":
        return {"ja"}
    if tts_backend == "kokoro":
        return {norm(k) for k in KOKORO_TARGET}
    if tts_backend == "supertonic":
        return set(SUPERTONIC_TARGET)
    if tts_backend == "qwen3_tts":
        return set(QWEN3TTS_TARGET)
    # say and mock are unrestricted here; say depends on system voices.
    return set(LANG_NAMES)


def supported_target(tts_backend: str, target: str) -> bool:
    return norm(target) in tts_targets(tts_backend) or tts_backend in ("say", "mock")


def kokoro_params(target: str) -> tuple[str, str]:
    """Map a target language to Kokoro (lang_code, default voice), or raise."""
    key = "en-GB" if norm(target) == "en-GB" else norm(target)
    if key not in KOKORO_TARGET:
        raise ValueError(
            f"Kokoro 不支援目標語 {target!r}；支援："
            f"{sorted(norm(k) for k in KOKORO_TARGET)}"
        )
    return KOKORO_TARGET[key]


def qwen3tts_lang(target: str | None) -> str:
    """Map a target language to a Qwen3-TTS language name; unsupported targets use auto."""
    return QWEN3TTS_TARGET.get(norm(target or ""), "auto")


def display(code: str) -> str:
    return LANG_NAMES.get(norm(code), norm(code))
