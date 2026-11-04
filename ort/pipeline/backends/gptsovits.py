"""Production static-reference GPT-SoVITS adapter with bounded resident-worker IPC."""
from __future__ import annotations

import base64
import contextlib
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import queue
import secrets
import subprocess
import tempfile
import threading
import time

import numpy as np

from ...config import Config
from ..base import TTSResult
from ..gptsovits_reference import validate_reference


class TTSWorkerError(RuntimeError):
    pass


class TTSQueueFull(TTSWorkerError):
    pass


@dataclass
class _Job:
    id: int
    text: str
    output: queue.Queue
    submitted: float = field(default_factory=time.perf_counter)
    cancelled: threading.Event = field(default_factory=threading.Event)
    options: dict = field(default_factory=dict)
    pronunciation: dict = field(default_factory=dict)
    lexicon_metadata: dict = field(default_factory=dict)
    frontend_mode: str = "all_ja"
    sampling: dict = field(default_factory=dict)
    generation: dict = field(default_factory=dict)
    voice_preset: dict = field(default_factory=dict)


class GPTSoVITS:
    """One serialized GPU worker. Rolling references are never used.

    synthesize preserves the upstream whole-WAV interface. synthesize_stream is
    an optional extension consumed by the session on its inference thread.
    """

    def __init__(self, config: Config):
        startup_started = time.perf_counter()
        self.preset=None
        if config.gptsovits_voice_preset:
            from ...voice_presets import resolve_preset
            self.preset,config=resolve_preset(config)
        self.config = config
        from ..personal_voice import load_personal_profile, check_personal_reference
        self.personal = load_personal_profile(config.gptsovits_personal_profile,base_manifest=config.gptsovits_manifest) if config.gptsovits_personal_profile else None
        if self.personal:
            check_personal_reference(self.personal,config.gptsovits_ref_audio,config.gptsovits_ref_text,config.gptsovits_ref_language)
        from ...pronunciation import PronunciationLexicon
        self.lexicon = PronunciationLexicon.load(config.gptsovits_pronunciation_lexicon) if config.gptsovits_pronunciation_lexicon else None
        self.lexicon_metadata = dict(enabled=bool(self.lexicon),**(self.lexicon.metadata if self.lexicon else
            dict(path=None,rule_count=0,sha256=None)))
        from ..checkpoint_selection import custom_checkpoint_intent
        intent = custom_checkpoint_intent(config.gptsovits_custom_gpt_path,config.gptsovits_custom_sovits_path)
        if intent:
            raise TTSWorkerError("Custom checkpoints are audit-only in STEP 07; official verification is unchanged. Intent: " + json.dumps(intent))
        self.reference = validate_reference(config.gptsovits_ref_audio, config.gptsovits_ref_text,
                                            config.gptsovits_ref_language)
        if not 1 <= config.gptsovits_request_queue <= 32 or not 1 <= config.gptsovits_output_queue <= 64:
            raise ValueError("TTS queue capacities are outside supported bounds")
        python = Path(config.gptsovits_python).resolve()
        vendor = Path(config.gptsovits_root).resolve()
        manifest = Path(config.gptsovits_manifest).resolve()
        if not python.is_file() or not vendor.is_dir() or not manifest.is_file():
            raise TTSWorkerError("GPT-SoVITS environment/source/models are missing; follow README_WINDOWS_GPTSOVITS.md")
        cache = manifest.parent / ".model-cache/gptsovits-workers"
        cache.mkdir(parents=True, exist_ok=True)
        self._directory = Path(tempfile.mkdtemp(prefix="worker-", dir=cache))
        self._closed = threading.Event()
        self._requests = queue.Queue(maxsize=config.gptsovits_request_queue)
        self._frames = queue.Queue(maxsize=config.gptsovits_output_queue)
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._sequence = 0
        self._random_seed_base=secrets.randbits(31)
        self._recent_random_seeds=deque(maxlen=128)
        self._active = None
        self._failure = None
        self.last_metrics = {}
        self.completed_metrics = deque(maxlen=32)
        self.max_backlog = 0
        settings = {"root": str(vendor), "manifest": str(manifest),
                    "reference_wav": self.reference.path, "reference_text": self.reference.text,
                    "reference_language": self.reference.language,
                    "frontend_diagnostics": config.gptsovits_frontend_diagnostics}
        if self.personal:
            settings.update(personal_profile=self.personal['profile_path'],personal_profile_sha256=self.personal['profile_sha256'])
        settings_path = self._directory / "config.json"
        settings_path.write_text(json.dumps(settings, ensure_ascii=False), encoding="utf8")
        self.log_path = self._directory / "worker.log"
        self._log = self.log_path.open("wb")
        worker = Path(__file__).resolve().parents[2] / "gptsovits_worker.py"
        try:
            self._process = subprocess.Popen([str(python), "-u", str(worker), "--config", str(settings_path)],
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                             text=True, encoding="utf8", bufsize=1,
                                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            self.close()
            raise TTSWorkerError("Could not start GPT-SoVITS worker") from exc
        self._reader = threading.Thread(target=self._read_frames, daemon=True, name="tts-pcm-reader")
        self._reader.start()
        try:
            ready = self._receive(config.gptsovits_startup_timeout)
            if ready.get("type") != "ready" or ready.get("version") != "v2ProPlus":
                raise TTSWorkerError(ready.get("message", "Unexpected TTS startup response"))
            if ready.get("reference_fingerprint") != self.reference.fingerprint:
                raise TTSWorkerError("Worker reference differs from the selected static reference")
            if self.personal:
                from ..personal_voice import personal_metadata
                if ready.get('personal_voice')!=personal_metadata(self.personal):
                    raise TTSWorkerError('Worker did not load the selected personal GPT/SoVITS pair; fallback rejected')
            self.ready = ready
            if self.preset:
                from ...voice_presets import preset_metadata
                self.ready['voice_preset']=preset_metadata(self.preset,config.gptsovits_prosody_level)
            self.ready['pronunciation_lexicon'] = self.lexicon_metadata.copy()
            self.ready["worker_startup_wall_seconds"] = time.perf_counter() - startup_started
        except BaseException:
            self.close()
            raise
        self._dispatcher = threading.Thread(target=self._dispatch, daemon=True, name="tts-requests")
        self._dispatcher.start()

    def _read_frames(self):
        try:
            while not self._closed.is_set():
                line = self._process.stdout.readline(16 * 2**20)
                if not line:
                    raise TTSWorkerError("GPT-SoVITS worker exited or its IPC pipe closed")
                if not line.endswith("\n"):
                    raise TTSWorkerError("TTS IPC frame exceeds the bounded frame limit")
                value = json.loads(line)
                self._put(self._frames, value)
        except Exception as exc:
            self._failure = TTSWorkerError(str(exc))

    def _put(self, target, value, cancel=None):
        while not self._closed.is_set() and not (cancel and cancel.is_set()):
            try:
                target.put(value, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _receive(self, timeout):
        deadline = time.monotonic() + timeout
        while not self._closed.is_set():
            try:
                return self._frames.get(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                if self._failure:
                    raise self._failure
                if self._process.poll() is not None:
                    raise TTSWorkerError(f"TTS worker exited with code {self._process.returncode}")
                if time.monotonic() >= deadline:
                    raise TTSWorkerError("TTS worker response timed out")
        raise TTSWorkerError("TTS worker is closed")

    def _write(self, value):
        with self._write_lock:
            if self._closed.is_set():
                raise TTSWorkerError("TTS worker is closed")
            try:
                self._process.stdin.write(json.dumps(value, ensure_ascii=True) + "\n")
                self._process.stdin.flush()
            except (OSError, ValueError) as exc:
                raise TTSWorkerError("TTS IPC write failed") from exc

    def _dispatch(self):
        while not self._closed.is_set():
            try:
                job = self._requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if job.cancelled.is_set():
                self._requests.task_done()
                continue
            if self._failure:
                self._put(job.output, self._failure, job.cancelled)
                self._requests.task_done()
                continue
            self._active = job
            start = time.perf_counter()
            try:
                self._write({"type": "synthesize", "id": job.id, "text": job.text, "language": "ja", "options": job.options,
                             "text_language": job.frontend_mode,"sampling":job.sampling,"generation":job.generation})
                count = total = 0
                digest = hashlib.sha256()
                first = None
                last = None
                deadline = time.monotonic() + self.config.gptsovits_request_timeout
                while True:
                    if time.monotonic() >= deadline:
                        raise TTSWorkerError("TTS worker response timed out")
                    frame = self._receive(max(0.001, deadline - time.monotonic()))
                    if frame.get("id") != job.id:
                        raise TTSWorkerError("TTS response belongs to another utterance")
                    if frame.get("type") == "error":
                        raise TTSWorkerError(frame.get("message", "TTS synthesis failed"))
                    if frame.get("type") == "done":
                        if frame.get("chunks") != count or frame.get("samples") != total or frame.get("sha256") != digest.hexdigest():
                            raise TTSWorkerError("TTS final PCM integrity/order mismatch")
                        self.last_metrics = {**frame, "queue_wait_seconds": start - job.submitted,
                                             "pronunciation": job.pronunciation,
                                             "pronunciation_lexicon": job.lexicon_metadata,
                                             "submitted_at": job.submitted,
                                             "first_received_pcm_at": job.submitted + first if first is not None else None,
                                             "last_received_pcm_at": last,
                                             "completed_at": time.perf_counter(),
                                             "first_received_pcm_seconds": first,
                                             "complete_seconds": time.perf_counter() - job.submitted,
                                             "max_queue_backlog": self.max_backlog}
                        if self.personal:
                            self.last_metrics['personal_voice']=self.ready['personal_voice']
                        if frame.get('sampling') is not None and frame['sampling']!=job.sampling:
                            raise TTSWorkerError('Worker sampling differs from frozen utterance settings')
                        self.last_metrics.update(sampling=job.sampling.copy(),sampling_acknowledged=frame.get('sampling') is not None)
                        if frame.get('generation') is not None and frame['generation']!=job.generation:
                            raise TTSWorkerError('Worker generation differs from frozen utterance settings')
                        self.last_metrics.update(generation=job.generation.copy(),generation_acknowledged=frame.get('generation') is not None)
                        if job.voice_preset:self.last_metrics['voice_preset']=job.voice_preset
                        self.completed_metrics.append(self.last_metrics.copy())
                        self._put(job.output, None, job.cancelled)
                        break
                    if frame.get("type") != "chunk" or frame.get("sequence") != count:
                        raise TTSWorkerError("TTS chunk sequence is missing, duplicated or out of order")
                    pcm = base64.b64decode(frame["pcm"], validate=True)
                    rate = frame.get("sample_rate")
                    if rate != self.ready["sample_rate"] or not pcm or len(pcm) % 2 or len(pcm) != 2 * frame.get("samples", -1):
                        raise TTSWorkerError("Invalid raw PCM chunk length/sample rate")
                    if hashlib.sha256(pcm).hexdigest() != frame.get("sha256"):
                        raise TTSWorkerError("TTS PCM chunk checksum mismatch")
                    digest.update(pcm)
                    total += len(pcm) // 2
                    count += 1
                    if first is None:
                        first = time.perf_counter() - job.submitted
                    last = time.perf_counter()
                    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                    self._put(job.output, TTSResult(samples, rate), job.cancelled)
            except Exception as exc:
                self._put(job.output, TTSWorkerError(str(exc)), job.cancelled)
                # On a timeout/broken stream, do not let late PCM contaminate the
                # next utterance. Explicit close/restart is required.
                if not isinstance(exc, TTSWorkerError) or "synthesis failed" not in str(exc):
                    self._failure = TTSWorkerError(str(exc))
            finally:
                self._active = None
                self._requests.task_done()

    def synthesize_stream(self, text, language=None, voice_reference=None, *, options=None,
                          apply_pronunciation=None, frontend_mode="all_ja",sampling=None,generation=None,
                          pronunciation_snapshot=None):
        from ...tts_controls import controls
        from ...tts_sampling import sampling_parameters
        if self.preset:
            from ...voice_presets import selected_sampling
            sampling={**selected_sampling(self.preset,self.config.gptsovits_prosody_level,self.config.gptsovits_seed),**dict(sampling or {})}
            generation={**self.preset['generation'],**dict(generation or {})}
            options={"speed_factor":1.0,"fragment_interval":self.preset['fragment_interval'],**dict(options or {})}
        sampling=sampling_parameters(sampling)
        from ...tts_generation import generation_parameters
        generation=generation_parameters(generation)
        options = controls(options)
        if self.preset and options['speed_factor']!=1.0:
            raise ValueError('Approved preset native speed is1.00 plus browser1.25; native-speed substitution is disabled')
        if apply_pronunciation is None:
            apply_pronunciation = options["pronunciation_enabled"]
        if type(apply_pronunciation) is not bool or frontend_mode not in {"all_ja","ja"}:
            raise ValueError("Invalid pronunciation switch or Japanese frontend mode")
        options["pronunciation_enabled"] = apply_pronunciation
        if language not in {None, "ja"}:
            raise ValueError("GPT-SoVITS production output is Japanese (ja)")
        if not text.strip():
            return
        if self._closed.is_set() or self._failure or self._process.poll() is not None:
            raise TTSWorkerError(str(self._failure or "TTS worker is closed or has crashed"))
        if self.personal:
            from ..personal_voice import digest
            if digest(self.personal['profile_path'])!=self.personal['profile_sha256']:
                raise ValueError('Personal voice selection changed; restart worker to reload checkpoints/conditioning')
        current = validate_reference(self.reference.path, self.reference.text, self.reference.language)
        if current.fingerprint != self.reference.fingerprint:
            raise ValueError("Static reference changed; restart the worker with the selected reference")
        original_text = text
        with self._state_lock:
            lexicon = pronunciation_snapshot if pronunciation_snapshot is not None else self.lexicon
            if lexicon and apply_pronunciation:
                text, pronunciation = lexicon.apply(text)
            else:
                pronunciation = dict(original_tts_text=original_text,spoken_tts_text=text,rules=[],
                                     lexicon_sha256=None,processing_seconds=0.)
            if self.preset and self.preset['join_japanese_spaces']:
                from ...voice_presets import join_japanese_spaces
                text=join_japanese_spaces(text)
                pronunciation={**pronunciation,'spoken_tts_text':text,'japanese_space_join':True}
            self._sequence += 1
            if sampling['seed']==-1:
                for _ in range(16):
                    candidate=secrets.randbits(31)
                    if candidate not in self._recent_random_seeds:break
                else:raise TTSWorkerError('Could not allocate a fresh utterance seed')
                sampling['seed']=candidate;self._recent_random_seeds.append(candidate)
            job = _Job(self._sequence, text, queue.Queue(maxsize=self.config.gptsovits_output_queue))
            job.sampling=sampling.copy()
            job.generation=generation.copy()
            if self.preset:
                from ...voice_presets import preset_metadata
                job.voice_preset=preset_metadata(self.preset,self.config.gptsovits_prosody_level)
            job.options = options
            job.lexicon_metadata = {**(lexicon.metadata if lexicon else self.lexicon_metadata),'enabled':bool(lexicon and apply_pronunciation)}
            job.pronunciation = pronunciation
            job.frontend_mode = frontend_mode
            try:
                self._requests.put_nowait(job)
            except queue.Full as exc:
                raise TTSQueueFull("TTS request queue is full; sentence was not submitted") from exc
            self.max_backlog = max(self.max_backlog, self._requests.qsize())
        completed = False
        try:
            while not self._closed.is_set():
                try:
                    item = job.output.get(timeout=0.1)
                except queue.Empty:
                    if self._failure:
                        raise self._failure
                    continue
                if item is None:
                    completed = True
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
            raise TTSWorkerError("TTS shutdown interrupted synthesis")
        finally:
            job.cancelled.set()
            if not completed and self._active is job and not self._closed.is_set():
                self._write({"type": "cancel", "id": job.id})

    def synthesize(self, text, language=None, voice_reference=None):
        chunks = list(self.synthesize_stream(text, language, voice_reference))
        if not chunks:
            return TTSResult(np.zeros(0, dtype=np.float32), self.ready["sample_rate"])
        return TTSResult(np.concatenate([chunk.samples for chunk in chunks]), chunks[0].sample_rate)

    def warmup(self):
        self.synthesize("こんにちは。", "ja")

    def restart(self):
        """Explicit crash recovery; never replay a failed sentence silently."""
        config = self.config
        self.close()
        self.__init__(config)

    def close(self):
        if not hasattr(self, "_closed") or self._closed.is_set():
            return
        process = getattr(self, "_process", None)
        if process is not None:
            with contextlib.suppress(Exception):
                self._write({"type": "shutdown"})
        self._closed.set()
        if process is not None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            for stream in (process.stdin, process.stdout):
                if stream:
                    with contextlib.suppress(OSError, ValueError):
                        stream.close()
        for name in ("_reader", "_dispatcher"):
            thread = getattr(self, name, None)
            if thread and thread is not threading.current_thread():
                thread.join(timeout=2)
        if hasattr(self, "_log"):
            self._log.close()
        # Keep logs for diagnosis, but erase the private reference configuration.
        if hasattr(self, "_directory"):
            for name in ("config.json", "config.upstream.yaml"):
                path = self._directory / name
                if path.exists():
                    path.unlink()
