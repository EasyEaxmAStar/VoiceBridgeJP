"""Explicit first-run model provisioning; no audio or personal data is supplied."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--asr',action='store_true');p.add_argument('--opus',action='store_true');p.add_argument('--tts',action='store_true');a=p.parse_args()
    if not any((a.asr,a.opus,a.tts)):p.error('Choose --asr, --opus and/or --tts after reading THIRD_PARTY_NOTICES.md')
    from huggingface_hub import snapshot_download,hf_hub_download
    root=Path(__file__).resolve().parents[1]
    if a.asr:snapshot_download('Qwen/Qwen3-ASR-0.6B')
    if a.opus:snapshot_download('Helsinki-NLP/opus-mt-tc-big-zh-ja')
    if a.tts:
        manifest=json.loads((root/'gptsovits.lock.json').read_text(encoding='utf8'))
        weights=root/'.model-cache/GPT-SoVITS/GPT_SoVITS/pretrained_models'
        if not (root/'.model-cache/GPT-SoVITS/.git').exists():raise RuntimeError('Clone the pinned GPT-SoVITS source first; see README.md')
        for artifact in manifest['artifacts']:
            hf_hub_download(manifest['model_repository'],artifact['path'],revision=manifest['model_revision'],local_dir=weights)
    print('Requested downloads complete; runtime still verifies the GPT-SoVITS manifest hashes.')

if __name__=='__main__':main()
