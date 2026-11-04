"""Hash-verified, explicitly selected personal voices; no permanent default."""
import copy
import json
from pathlib import Path
import re
from dataclasses import replace
from .pipeline.personal_voice import load_personal_profile,digest
from .tts_sampling import sampling_parameters
from .tts_generation import generation_parameters

PRESETS=('PERSONAL_R06','PERSONAL_R17','PERSONAL_R18')
LEVELS=('ORIGINAL','SUBTLE','NATURAL','EXPRESSIVE')
DEFAULT_REGISTRY=str(Path(__file__).resolve().parents[1]/'.model-cache/personal-voice/final-prosody-selection/voice-presets.json')


def join_japanese_spaces(text):
    jp=r'\u3040-\u30ff\u3400-\u9fff\u3005'
    return re.sub(r'(?<=['+jp+r'])[ \u3000]+(?=['+jp+r'])','',text)


def resolve_preset(config):
    name=config.gptsovits_voice_preset
    if name not in PRESETS:raise ValueError('Unknown personal voice preset; no fallback')
    path=Path(config.gptsovits_preset_registry).resolve()
    data=json.loads(path.read_text(encoding='utf8'))
    if data.get('schema')!=1 or data.get('default_preset') is not None:raise ValueError('Personal preset registry must not select a permanent default')
    entry=copy.deepcopy(data['presets'][name])
    if entry['name']!=name or entry['original_candidate']!=name.removeprefix('PERSONAL_'):raise ValueError('Preset candidate identity mismatch')
    source=entry['approved_source']
    if digest(source['path'])!=source['sha256']:raise ValueError('Approved original configuration changed')
    original=json.loads(Path(source['path']).read_text(encoding='utf8'))
    if original['candidate']!=entry['original_candidate']:raise ValueError('Approved source label differs')
    expected=original['configuration']
    if (generation_parameters(expected['generation'])!=generation_parameters(entry['generation']) or
        expected['join_spaces']!=entry['join_japanese_spaces'] or expected['interval']!=entry['fragment_interval'] or
        sampling_parameters(expected['sampling'])!=sampling_parameters(entry['levels']['ORIGINAL']['sampling'])):
        raise ValueError('Preset ORIGINAL differs from its immutable approved source')
    profile=load_personal_profile(entry['profile'],base_manifest=config.gptsovits_manifest)
    if profile['profile_sha256']!=entry['profile_sha256']:raise ValueError('Approved preset profile changed')
    for field in ['gpt','sovits','reference_wav','reference_text_file']:
        if profile[field]!=entry[field]:raise ValueError('Preset artifact identity differs: '+field)
    if digest(entry['lexicon_path'])!=entry['lexicon_sha256']:raise ValueError('Approved pronunciation lexicon changed')
    if config.gptsovits_prosody_level not in LEVELS:raise ValueError('Unknown prosody level')
    for value in entry['levels'].values():sampling_parameters(value['sampling'])
    generation_parameters(entry['generation'])
    if entry['playback']!=dict(method='browser_preserves_pitch',rate=1.25,preserves_pitch=True,native_speed_factor=1.0,postprocessing='none'):
        raise ValueError('Preferred playback policy changed; explicit revalidation required')
    if config.gptsovits_ref_audio and digest(config.gptsovits_ref_audio)!=entry['reference_wav']['sha256']:
        raise ValueError('Preset requires its approved reference; supplied reference differs')
    if config.gptsovits_ref_text and config.gptsovits_ref_text.strip()!=profile['reference_text'].strip():
        raise ValueError('Preset requires its exact approved reference transcript')
    return entry,replace(config,gptsovits_personal_profile=profile['profile_path'],
        gptsovits_ref_audio=entry['reference_wav']['path'],gptsovits_ref_text=profile['reference_text'])


def preset_metadata(entry,level):
    return dict(name=entry['name'],candidate=entry['original_candidate'],prosody_level=level,
        profile_sha256=entry['profile_sha256'],playback=entry['playback'],speaker_stable=True,
        subjective_refinement='PENDING user listening' if level!='ORIGINAL' else 'User approved original candidate')


def selected_sampling(entry,level,seed='auto'):
    values=copy.deepcopy(entry['levels'][level]['sampling'])
    values['seed']=42 if level=='ORIGINAL' else -1
    if seed not in ('auto',None):values['seed']=-1 if seed=='random' else int(seed)
    return sampling_parameters(values)
