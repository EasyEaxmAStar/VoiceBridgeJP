"""Explicit resident preset switching with preflight and unchanged old worker on failure."""
from dataclasses import replace
import threading
import numpy as np
from .gptsovits import GPTSoVITS,TTSWorkerError
from ..base import TTSResult
from ...voice_presets import resolve_preset,LEVELS,preset_metadata


class PresetBusy(TTSWorkerError):pass


class PersonalPresetTTS:
    def __init__(self,config,*,factory=GPTSoVITS):
        self.config=config;self._factory=factory;self._lock=threading.Lock();self._active=0;self._switching=False
        self._backend=factory(config);self.switch_history=[]

    def __getattr__(self,name):
        if name.startswith('__'):raise AttributeError(name)
        return getattr(self._backend,name)

    def synthesize_stream(self,*args,**kwargs):
        with self._lock:
            if self._switching:raise PresetBusy('Preset switch is in progress; no audio submitted')
            current=self._backend;self._active+=1
        try:yield from current.synthesize_stream(*args,**kwargs)
        finally:
            with self._lock:self._active-=1

    def synthesize(self,*args,**kwargs):
        chunks=list(self.synthesize_stream(*args,**kwargs))
        return TTSResult(np.concatenate([c.samples for c in chunks]),chunks[0].sample_rate) if chunks else TTSResult(np.zeros(0,dtype='float32'),32000)

    def set_prosody_level(self,level):
        if level not in LEVELS:raise ValueError('Unknown prosody level')
        with self._lock:
            if self._active or self._switching:raise PresetBusy('Wait for queued utterances before changing prosody')
            self.config=replace(self.config,gptsovits_prosody_level=level)
            self._backend.config=replace(self._backend.config,gptsovits_prosody_level=level)
            self._backend.ready['voice_preset']=preset_metadata(self._backend.preset,level)

    def switch_voice_preset(self,name,*,level='ORIGINAL'):
        with self._lock:
            if self._active or self._switching:raise PresetBusy('Wait for queued utterances before switching voice')
            self._switching=True;old=self._backend;old_config=self.config
        candidate=None
        try:
            selected=replace(old_config,gptsovits_voice_preset=name,gptsovits_prosody_level=level)
            entry,bound=resolve_preset(selected)
            same=entry['profile_sha256']==old.personal['profile_sha256']
            if same:
                candidate=old
                chunks=list(old.synthesize_stream('音声の切り替えを確認します。','ja',
                    options={'speed_factor':1.,'fragment_interval':entry['fragment_interval']},
                    generation=entry['generation'],sampling=entry['levels'][level]['sampling']))
            else:
                candidate=self._factory(selected)
                chunks=list(candidate.synthesize_stream('音声の切り替えを確認します。','ja'))
            if not chunks or not all(c.sample_rate==32000 and np.isfinite(c.samples).all() for c in chunks):raise RuntimeError('Invalid preset preflight PCM')
            if np.sqrt(np.mean(np.concatenate([c.samples for c in chunks]).astype(float)**2))<1e-4:raise RuntimeError('Preset preflight is silent')
        except BaseException as exc:
            cleanup_error=None
            if candidate is not None and candidate is not old:
                try:candidate.close()
                except Exception as cleanup:cleanup_error=str(cleanup)
            # A failed preflight using the existing worker can poison its IPC.
            # Recover the previous configuration explicitly; never replay the failed text.
            recovered=False
            if candidate is old and (old._failure or old._process.poll() is not None):
                try:old.restart();recovered=True
                except Exception as recovery:
                    self.switch_history.append(dict(preset=name,status='FAILED_CLOSED',error=str(exc),recovery_error=str(recovery)))
                    raise TTSWorkerError('Preset preflight and previous personal worker recovery both failed') from recovery
            self.switch_history.append(dict(preset=name,status='ROLLED_BACK',error=str(exc),old_pid=old.ready['pid'],
                old_worker_restarted=recovered,candidate_cleanup_error=cleanup_error))
            raise
        else:
            if same:
                old.preset=entry;old.config=bound;old.ready['voice_preset']=preset_metadata(entry,level)
            with self._lock:self._backend=candidate;self.config=selected
            event=dict(preset=name,level=level,status='COMMITTED',models_reused=same,pid=candidate.ready['pid'])
            self.switch_history.append(event)
            if candidate is not old:
                try:old.close()
                except Exception as cleanup:
                    # Publication already happened. Report the cleanup failure truthfully,
                    # retaining the verified new worker rather than claiming rollback.
                    event.update(status='COMMITTED_CLEANUP_ERROR',cleanup_error=str(cleanup))
                    raise TTSWorkerError('New personal preset is active, but previous worker cleanup failed') from cleanup
            return self.ready
        finally:
            with self._lock:self._switching=False

    def warmup(self):self.synthesize('こんにちは。','ja')
    def restart(self):
        with self._lock:
            if self._active or self._switching:raise PresetBusy('Cannot restart while speaking')
        self._backend.restart()
    def close(self):self._backend.close()
