"""Pluggable translation pipeline: VAD -> ASR -> committer -> Translator -> TTS."""

from .base import (
    ASRBackend,
    Pipeline,
    Segment,
    TranslatorBackend,
    TTSBackend,
    TTSResult,
    build_pipeline,
)

__all__ = [
    "ASRBackend",
    "TranslatorBackend",
    "TTSBackend",
    "TTSResult",
    "Segment",
    "Pipeline",
    "build_pipeline",
]
