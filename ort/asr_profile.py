"""Diagnostic stage timing for the installed Qwen wrapper; no package patching."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .asr_benchmark import DEFAULT_MANIFEST, prepare_cases, read_manifest
from .asr_metrics import score_text
from .config import Config
from .pipeline.backends.qwen3_asr_torch import Qwen3ASRTorch


def main() -> None:
    import torch
    from transformers.utils import logging
    from qwen_asr.inference.utils import float_range_normalize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    logging.set_verbosity_error()
    manifest, root = read_manifest(DEFAULT_MANIFEST)
    cases = [case for case in prepare_cases(manifest, root, 20261028) if case.id in {"news", "meeting", "mixed"}]
    backend = Qwen3ASRTorch(Config(runtime="cuda", device="cuda:0", qwen_asr_device="cuda:0",
                                  qwen_asr_model="0.6b", qwen_asr_offline=True,
                                  torch_attn="sdpa", torch_dtype="bfloat16", qwen_asr_max_new_tokens=256))
    wrapper = backend._model
    original_processor, original_generate = wrapper.processor, wrapper.model.generate
    observations = []
    current = {}

    def timed(key, function, *positional, **keyword):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = function(*positional, **keyword)
        torch.cuda.synchronize()
        current.setdefault(key, []).append(time.perf_counter() - start)
        return result

    class Inputs(Mapping):
        def __init__(self, batch):
            self.batch = batch

        def __getitem__(self, key):
            return self.batch[key]

        def __iter__(self):
            return iter(self.batch)

        def __len__(self):
            return len(self.batch)

        def to(self, *positional, **keyword):
            current.setdefault("transfer_inputs", []).append({
                key: {"dtype": str(value.dtype), "device": str(value.device),
                      "bytes": value.numel() * value.element_size()}
                for key, value in self.batch.items() if isinstance(value, torch.Tensor)
            })
            timed("to_seconds", self.batch.to, *positional, **keyword)
            return self

    class Processor:
        def __getattr__(self, name):
            return getattr(original_processor, name)

        def __call__(self, *positional, **keyword):
            return Inputs(timed("processor_seconds", original_processor, *positional, **keyword))

        def batch_decode(self, *positional, **keyword):
            return timed("decode_seconds", original_processor.batch_decode, *positional, **keyword)

    def generate(*positional, **keyword):
        current["inference_mode_enabled"] = torch.is_inference_mode_enabled()
        current["grad_enabled"] = torch.is_grad_enabled()
        result = timed("generate_seconds", original_generate, *positional, **keyword)
        current["generated_tokens"] = result.sequences.shape[-1] - keyword["input_ids"].shape[-1]
        return result

    # Shape/model priming is not part of the diagnostic observations.
    for case in cases:
        backend.transcribe(case.audio, language="zh")
    wrapper.processor = Processor()
    wrapper.model.generate = generate
    try:
        for case in cases:
            for iteration in range(args.repeats):
                current = {"case_id": case.id, "iteration": iteration}
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = backend.transcribe(case.audio, language="zh")
                torch.cuda.synchronize()
                current.update(total_seconds=time.perf_counter() - start,
                               text=result.text, **score_text(case.reference, result.text))
                observations.append(current)
    finally:
        wrapper.processor = original_processor
        wrapper.model.generate = original_generate
    copy_audit = {}
    for case in cases:
        start = time.perf_counter()
        for _ in range(100):
            float_range_normalize(case.audio)
        copy_audit[case.id] = {
            "official_numpy_normalization_mean_s": (time.perf_counter() - start) / 100,
            "waveform_bytes": case.audio.nbytes,
            "adapter_contiguous_conversion_shares_memory": bool(np.shares_memory(case.audio, np.ascontiguousarray(case.audio, dtype=np.float32))),
        }
    report = {"note": "Extra stage synchronizations add overhead. These diagnostic times are not the uninstrumented benchmark latency. No runtime method remains replaced after profiling.",
              "model_eval": not wrapper.model.training, "generation": wrapper.model.generation_config.to_dict(),
              "cpu_threads": torch.get_num_threads(), "numpy_copy_audit": copy_audit,
              "observations": observations}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    for case in cases:
        rows = [row for row in observations if row["case_id"] == case.id]
        print(case.id, {key: round(float(np.mean([sum(row[key]) if isinstance(row[key], list) else row[key] for row in rows])), 6)
                        for key in ("total_seconds", "processor_seconds", "to_seconds", "generate_seconds", "decode_seconds")})


if __name__ == "__main__":
    main()
