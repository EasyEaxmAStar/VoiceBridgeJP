"""Default MLX translator: TranslateGemma-4b-it through mlx-lm.

The original google/translategemma-4b-it model is manually gated, so the default
MLX path uses the non-gated mlx-community/translategemma-4b-it-4bit mirror.

Implementation notes:
- Load with mlx-lm, not mlx-vlm. mlx-vlm 0.5.0 fails on this Gemma3 multimodal
  config; mlx-lm reads text_config and works.
- TranslateGemma's chat template requires structured content:
  `{type:'text', source_lang_code, target_lang_code, text}`.
"""

from __future__ import annotations

import threading

from ...config import Config
from ...languages import DEFAULT_TARGET_LANGUAGE

_LOCK = threading.Lock()
_STOP_TOKENS = ("<end_of_turn>", "<eos>", "<start_of_turn>")


class TranslateGemmaMLX:
    def __init__(self, config: Config):
        from mlx_lm import load

        self._repo = config.translategemma_model
        self._tgt = DEFAULT_TARGET_LANGUAGE
        self._model, self._tokenizer = load(self._repo)

        # Performance fix: TranslateGemma ends turns with <end_of_turn>, while
        # mlx-lm stops only on <eos>(id=1) by default. Without this, generation
        # runs to max_tokens and wastes most translation latency.
        try:
            eot = self._tokenizer._tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if isinstance(eot, int) and eot >= 0:
                cur = set(getattr(self._tokenizer, "eos_token_ids", set()) or set())
                self._tokenizer.eos_token_ids = cur | {eot}
        except Exception:  # noqa: BLE001
            pass  # Still works; _clean trims control tokens, just more slowly.

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
        from mlx_lm import generate

        if source is None:
            raise ValueError("TranslateGemma requires source_lang_code")
        src = source
        tgt = target or self._tgt
        # Feed committed (source -> target) spans as prior user/assistant turns
        # so this span continues coherently without retranslating played output.
        messages: list[dict] = []
        for czh, cen in context:
            czh, cen = (czh or "").strip(), (cen or "").strip()
            if czh and cen:
                messages.append(self._user_turn(czh, src, tgt))
                messages.append({"role": "assistant", "content": cen})
        messages.append(self._user_turn(text, src, tgt))
        with _LOCK:
            prompt = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            raw = generate(self._model, self._tokenizer, prompt, max_tokens=256)
        return self._clean(raw)

    @staticmethod
    def _clean(text: str) -> str:
        for tok in _STOP_TOKENS:
            idx = text.find(tok)
            if idx != -1:
                text = text[:idx]
        text = text.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”":
            text = text[1:-1].strip()
        return text
