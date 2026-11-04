"""Exact approved browser-rate companion player; native WAVs are unchanged."""
import html,json
from pathlib import Path


def write_player(destination,outputs,preset):
    destination=Path(destination)
    if destination.exists():raise ValueError('Playback page already exists')
    cards=''.join(f'<p>{html.escape(Path(p).name)}</p><audio controls preload="metadata" src="{html.escape(Path(p).resolve().as_uri())}"></audio>' for p in outputs)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>已选个人音色播放</title><style>body{font:18px system-ui;max-width:850px;margin:40px auto}audio{width:100%;margin:12px 0}</style><h1>'''+html.escape(preset['name'])+'''</h1><p>保留获认可的原生1.00 WAV，并以浏览器1.25、保音高播放。没有对存储WAV另做拉伸、移调或响度处理。原生VB-CABLE播放端无法直接等同此策略。</p>'''+cards+'''<script>document.querySelectorAll('audio').forEach(a=>{a.defaultPlaybackRate=1.25;a.playbackRate=1.25;a.preservesPitch=true;a.addEventListener('play',()=>{document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause()});a.playbackRate=1.25;a.preservesPitch=true})});</script></html>'''
    with destination.open('x',encoding='utf8') as stream:stream.write(page)
