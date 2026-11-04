"""TranslateGemma backend for CUDA/CPU through Hugging Face Transformers."""

from __future__ import annotations

import threading

from ...config import Config
from ...languages import DEFAULT_TARGET_LANGUAGE

_LOCK = threading.Lock()


def _missing_dep(name: str) -> RuntimeError:
    return RuntimeError(
        f"{name} is required for ORT_RUNTIME=cuda/cpu. "
        "Install non-MLX dependencies with `uv sync --extra nonmlx`."
    )


class TranslateGemmaTorch:
    def __init__(self, config: Config):
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise _missing_dep("torch and transformers") from exc

        self._torch = torch
        self._repo = config.translategemma_torch_model
        self._tgt = DEFAULT_TARGET_LANGUAGE
        self._max_new_tokens = config.torch_max_new_tokens

        dtype = self._dtype(config.torch_dtype)
        kwargs: dict[str, object] = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if config.torch_attn not in {"", "none"}:
            kwargs["attn_implementation"] = config.torch_attn

        self._processor = AutoProcessor.from_pretrained(self._repo)
        self._model = AutoModelForImageTextToText.from_pretrained(self._repo, **kwargs)
        self._device = self._torch.device(config.device)
        self._model.to(self._device)
        self._model.eval()

    def _dtype(self, name: str):  # noqa: ANN001
        if name in {"", "none"}:
            return None
        if not hasattr(self._torch, name):
            raise ValueError(f"Unsupported ORT_TORCH_DTYPE={name!r}")
        return getattr(self._torch, name)

    def warmup(self) -> None:
        self.translate("你好", source="zh", target=self._tgt)

    def _user_turn(self, text: str, src: str, tgt: str) -> dict:
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": src,
                    "target_lang_code": tgt,
                    "text": text,
                }
            ],
        }

    def translate(self, text: str, context=(), source=None, target=None) -> str:  # noqa: ANN001
        text = text.strip()
        if not text:
            return ""

        if source is None:
            raise ValueError("TranslateGemma requires source_lang_code")
        src = source
        tgt = target or self._tgt
        messages: list[dict] = []
        for csrc, ctgt in context:
            csrc, ctgt = (csrc or "").strip(), (ctgt or "").strip()
            if csrc and ctgt:
                messages.append(self._user_turn(csrc, src, tgt))
                messages.append({"role": "assistant", "content": ctgt})
        messages.append(self._user_turn(text, src, tgt))

        with _LOCK, self._torch.inference_mode():
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._device)
            input_len = inputs["input_ids"].shape[-1]
            generated = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self._max_new_tokens,
            )[0][input_len:]
            return self._processor.decode(generated, skip_special_tokens=True).strip()
