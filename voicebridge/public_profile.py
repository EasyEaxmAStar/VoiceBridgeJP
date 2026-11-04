"""Local public-edition preferences; no developer-specific defaults."""
import json
import os
from pathlib import Path


def _setting(key, default):
    path=Path(__file__).resolve().parents[1]/'config/public_tts.local.json'
    return json.loads(path.read_text(encoding='utf8')).get(key,default) if path.is_file() else default


def microphone_default():
    return os.environ.get('VOICEBRIDGE_INPUT_DEVICE') or _setting('microphone','')


def output_default():
    return os.environ.get('VOICEBRIDGE_OUTPUT_DEVICE') or _setting('output_device','CABLE Input (VB-Audio Virtual Cable)')


def input_channels_default():
    return int(os.environ.get('VOICEBRIDGE_INPUT_CHANNELS') or _setting('input_channels',1))
