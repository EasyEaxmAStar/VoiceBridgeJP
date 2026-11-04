"""Pinned CPU English->Japanese Marian model, local artifacts only."""
import time

REPO = "Helsinki-NLP/opus-mt-en-jap"
REVISION = "a863894cdd2b80f3bc1c5966734aee9ffec207d1"


class EnglishMT:
    def __init__(self):
        from huggingface_hub import snapshot_download
        from transformers import MarianTokenizer, AutoModelForSeq2SeqLM
        import torch
        self.torch = torch
        start = time.perf_counter()
        path = snapshot_download(REPO, revision=REVISION, local_files_only=True)
        self.tokenizer = MarianTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True,
                                                        dtype=torch.float32).to("cpu").eval()
        self.load_seconds = time.perf_counter()-start

    def translate(self, text):
        with self.torch.inference_mode():
            encoded = self.tokenizer(text, return_tensors="pt", truncation=False)
            if encoded.input_ids.shape[-1] > 500:
                raise ValueError("English MT input exceeds safe model context")
            result = self.model.generate(**encoded, do_sample=False, max_new_tokens=512)
            return self.tokenizer.decode(result[0], skip_special_tokens=True).strip()
