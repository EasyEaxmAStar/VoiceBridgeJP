"""Strict pinned-upstream PCM boundary; never reinterpret error audio as speech."""
import json

import numpy as np


def validate_pcm(rate, samples, expected_rate, position):
    metadata = {"chunk": position, "rate": str(rate), "type": type(samples).__name__}
    if isinstance(samples, np.ndarray):
        metadata.update(dtype=str(samples.dtype), shape=list(samples.shape), bytes=samples.nbytes)
    if (not isinstance(rate, (int, np.integer)) or isinstance(rate, bool) or rate != expected_rate
            or not isinstance(samples, np.ndarray) or samples.dtype != np.int16 or samples.ndim != 1):
        # TTS.run's except branch yields precisely this placeholder before its
        # model-reinitialization/re-raise path. Never resample/emit the sentinel.
        if (rate == 16000 and isinstance(samples, np.ndarray) and samples.dtype == np.int16
                and samples.shape == (16000,) and not np.any(samples) and expected_rate != 16000):
            raise RuntimeError("GPT-SoVITS upstream error sentinel (16 kHz silence); "
                               "original exception is in worker.log: " + json.dumps(metadata))
        raise RuntimeError("Upstream returned unsupported PCM format: " + json.dumps(metadata))
    return samples
