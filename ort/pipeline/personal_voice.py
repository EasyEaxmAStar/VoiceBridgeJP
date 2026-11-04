"""Hash-pinned local personal v2ProPlus pair; never substitutes a base voice."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def load_personal_profile(path, *, verify_models=True, base_manifest=None):
    profile_path=Path(path).expanduser().resolve()
    try:
        data=json.loads(profile_path.read_text(encoding='utf8'))
    except (OSError,ValueError) as exc:
        raise ValueError(f'Personal voice profile missing/invalid; generic fallback is disabled: {profile_path}') from exc
    if data.get('schema')!=1 or data.get('speaker')!='voicebridge_personal' or data.get('architecture')!='v2ProPlus':
        raise ValueError('Unsupported personal voice profile identity/architecture')
    if data.get('training_completed') is not True:raise ValueError('Personal GPT and SoVITS training must be completed')
    if base_manifest and digest(base_manifest)!=data.get('base_manifest_sha256'):
        raise ValueError('Personal voice base manifest changed; explicit revalidation required')
    for key in ('gpt','sovits','reference_wav','reference_text_file'):
        entry=data.get(key,{})
        if not isinstance(entry,dict) or not isinstance(entry.get('path'),str) or not isinstance(entry.get('sha256'),str):
            raise ValueError('Malformed personal artifact metadata: '+key)
        target=Path(entry.get('path','')).expanduser()
        if not target.is_absolute() or not target.is_file():raise ValueError('Personal voice artifact missing: '+key)
        if verify_models or key.startswith('reference_'):
            if target.stat().st_size!=entry.get('bytes') or digest(target)!=entry.get('sha256'):
                raise ValueError('Personal voice artifact hash/size changed: '+key)
    if data.get('reference_language')!='ja' or data.get('output_language')!='ja':
        raise ValueError('Personal voice reference/output must be Japanese')
    data=dict(data,profile_path=str(profile_path),profile_sha256=digest(profile_path))
    data['reference_text']=Path(data['reference_text_file']['path']).read_text(encoding='utf-8-sig')
    return data


def personal_metadata(profile):
    return dict(speaker=profile['speaker'],architecture=profile['architecture'],
        profile_path=profile['profile_path'],profile_sha256=profile['profile_sha256'],
        gpt=profile['gpt'],sovits=profile['sovits'],reference_wav=profile['reference_wav'],
        source_recording_sha256=profile['source_recording_sha256'],fine_tuned=True)


def check_personal_reference(profile, audio, text, language):
    if not audio or not text or not Path(audio).is_file():
        raise ValueError('Personal voice static reference is missing; no fallback')
    if language!='ja' or digest(audio)!=profile['reference_wav']['sha256'] or text.strip()!=profile['reference_text'].strip():
        raise ValueError('Personal voice requires its verified own-recording reference and exact transcript; no other-speaker fallback')
