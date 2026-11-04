"""Local public-edition preferences; no developer-specific defaults."""
import json,os
from pathlib import Path

def microphone_default():
    if os.environ.get('VOICEBRIDGE_INPUT_DEVICE'):return os.environ['VOICEBRIDGE_INPUT_DEVICE']
    path=Path(__file__).resolve().parents[1]/'config/public_tts.local.json'
    return json.loads(path.read_text(encoding='utf8')).get('microphone','') if path.is_file() else ''
