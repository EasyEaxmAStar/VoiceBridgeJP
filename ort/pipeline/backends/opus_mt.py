"""CPU-only OPUS-MT zh-to-ja translator using the direct Transformers API."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Sequence

from ...config import Config

DEFAULT_OPUSMT_MODEL = "Helsinki-NLP/opus-mt-tc-big-zh-ja"
_LOCK = threading.Lock()


def process_memory_mb() -> dict[str, float | None]:
    """Return process RSS and peak working set in MiB when available."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if ok:
            return {
                "rss_mb": counters.WorkingSetSize / (1024 * 1024),
                "peak_rss_mb": counters.PeakWorkingSetSize / (1024 * 1024),
            }
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB; macOS reports bytes.
        peak_mb = rss / (1024 if os.name != "nt" else 1024 * 1024)
        return {"rss_mb": None, "peak_rss_mb": peak_mb}
    except (ImportError, OSError, AttributeError):
        return {"rss_mb": None, "peak_rss_mb": None}


class OpusMTTranslator:
    """Marian seq2seq translator. Model weights and generation are CPU float32."""

    def __init__(self, config: Config):
        try:
            import torch
            from huggingface_hub import snapshot_download
            from sentencepiece import SentencePieceProcessor
            from transformers import AutoModelForSeq2SeqLM, MarianTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "OPUS-MT requires the pinned CPU environment; see "
                "requirements-opusmt-cpu.txt and README setup instructions"
            ) from exc

        self._torch = torch
        self.repo = config.opusmt_model
        self._max_new_tokens = config.torch_max_new_tokens
        self._load_started = time.perf_counter()
        self._rss_before_load = process_memory_mb()["rss_mb"]
        try:
            snapshot = Path(
                snapshot_download(
                    repo_id=self.repo,
                    local_files_only=True,
                    allow_patterns=[
                        "config.json",
                        "generation_config.json",
                        "model.safetensors",
                        "source.spm",
                        "target.spm",
                        "special_tokens_map.json",
                        "tokenizer_config.json",
                        "vocab.json",
                    ],
                )
            )
        except OSError as exc:
            raise RuntimeError(
                f"OPUS-MT model {self.repo!r} is not cached locally. "
                "Run `python -m ort.translate_text download` once while online, "
                "then inference will work offline."
            ) from exc
        # The upstream conversion labels target vocab.json as a shared
        # tokenizer vocab even though source SentencePiece has Chinese pieces
        # absent from it. Build the source id mapping from source.spm and let
        # MarianTokenizer use separate vocabularies, otherwise many Chinese
        # source tokens become <unk> before inference.
        revision = snapshot.name
        project_root = Path(__file__).resolve().parents[3]
        cache_root = Path(
            os.environ.get(
                "ORT_CACHE_DIR", str(project_root / ".model-cache")
            )
        ) / "opusmt-source-vocabs" / hashlib.sha256(
            f"{self.repo}@{revision}".encode("utf-8")
        ).hexdigest()[:20]
        cache_root.mkdir(parents=True, exist_ok=True)
        source_vocab = cache_root / "source_vocab.json"
        if not source_vocab.exists():
            sp = SentencePieceProcessor(model_file=str(snapshot / "source.spm"))
            vocab = {
                sp.id_to_piece(index): index for index in range(sp.get_piece_size())
            }
            vocab["<pad>"] = max(vocab.values()) + 1
            source_vocab.write_text(
                json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
            )
        self._tokenizer = MarianTokenizer(
            source_spm=str(snapshot / "source.spm"),
            target_spm=str(snapshot / "target.spm"),
            vocab=str(source_vocab),
            target_vocab_file=str(snapshot / "vocab.json"),
            source_lang="zh",
            target_lang="ja",
            separate_vocabs=True,
            unk_token="<unk>",
            eos_token="</s>",
            pad_token="<pad>",
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            str(snapshot),
            dtype=torch.float32,
            local_files_only=True,
        )
        self._model.to(torch.device("cpu"))
        self._model.eval()
        self.load_seconds = time.perf_counter() - self._load_started
        self.memory_after_load = process_memory_mb()
        self.memory_before_load_mb = self._rss_before_load
        self.parameter_memory_mb = sum(
            parameter.numel() * parameter.element_size()
            for parameter in self._model.parameters()
        ) / (1024 * 1024)

    def warmup(self) -> None:
        self.translate("你好", source="zh", target="ja")

    def translate_many(self,texts):
        """Opt-in final-sentence batch, same resident CPU model and decoding."""
        if not texts or len(texts)>16 or any(not isinstance(x,str) or not x.strip() for x in texts):
            raise ValueError('Expected 1–16 nonempty Mandarin sentences')
        with _LOCK,self._torch.inference_mode():
            encoded=self._tokenizer(texts,return_tensors='pt',padding=True,truncation=True,max_length=512)
            encoded={name:t.to('cpu') for name,t in encoded.items()}
            generated=self._model.generate(**encoded,do_sample=False,max_new_tokens=self._max_new_tokens)
            return [self._tokenizer.decode(row,skip_special_tokens=True).strip() for row in generated]

    def translate(
        self,
        text: str,
        context: Sequence[tuple[str, str]] = (),
        source: str | None = None,
        target: str | None = None,
    ) -> str:
        text = text.strip()
        if not text:
            return ""
        if source not in (None, "zh") or target not in (None, "ja"):
            raise ValueError("OPUS-MT zh-ja only supports source='zh' and target='ja'")

        # Marian translation is sentence-based and has no chat-style context
        # channel. Accept the shared interface's context argument, but do not
        # prepend earlier utterances to the user's actual source text.
        del context
        with _LOCK, self._torch.inference_mode():
            encoded = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            encoded = {name: tensor.to("cpu") for name, tensor in encoded.items()}
            generated = self._model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self._max_new_tokens,
            )
            return self._tokenizer.decode(
                generated[0], skip_special_tokens=True
            ).strip()
