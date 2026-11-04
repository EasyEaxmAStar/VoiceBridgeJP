"""TTS backend: macOS `say` -> AIFF -> ffmpeg -> float32.

This is deterministic, uses no model dependencies, and produces real audio.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from ...config import Config

_TTS_SAMPLE_RATE = 24_000


class SayTTS:
    sample_rate = _TTS_SAMPLE_RATE

    def __init__(self, config: Config):
        self._voice = os.environ.get("ORT_SAY_VOICE", "")  # Empty means system default.

    def warmup(self) -> None:
        self.synthesize("warmup")

    def synthesize(  # noqa: ANN001
        self,
        text: str,
        language=None,
        voice_reference=None,
    ):
        from ..base import TTSResult

        text = text.strip()
        if not text:
            return TTSResult(np.zeros(0, dtype=np.float32), _TTS_SAMPLE_RATE)

        with tempfile.TemporaryDirectory() as d:
            aiff = os.path.join(d, "tts.aiff")
            say_cmd = ["say", "-o", aiff]
            if self._voice:
                say_cmd += ["-v", self._voice]
            say_cmd += ["--", text]
            subprocess.run(say_cmd, check=True, capture_output=True)

            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    aiff,
                    "-f",
                    "f32le",
                    "-ac",
                    "1",
                    "-ar",
                    str(_TTS_SAMPLE_RATE),
                    "pipe:1",
                ],
                check=True,
                capture_output=True,
            )
        samples = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
        return TTSResult(samples=samples, sample_rate=_TTS_SAMPLE_RATE)
