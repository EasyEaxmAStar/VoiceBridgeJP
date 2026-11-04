"""Reusable ordered saved-WAV utterance jobs on the existing production stack."""
from __future__ import annotations

from concurrent.futures import CancelledError, Future, TimeoutError
from dataclasses import dataclass, field
import hashlib
import queue
import threading
import time

import numpy as np
import soundfile as sf

from . import inference
from .audio import INTERNAL_SAMPLE_RATE, WIRE_SAMPLE_RATE, WireResampler, float32_to_pcm16
from .config import Config
from .pipeline.base import build_pipeline
from .pipeline.gptsovits_reference import validate_reference
from .pipeline.segmentation import Segmenter

ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
MT_MODEL = "Helsinki-NLP/opus-mt-tc-big-zh-ja"


def require_production(config):
    if (config.runtime, config.asr_backend, config.translator_backend, config.tts_backend) != ("cuda", "qwen3", "opusmt", "gptsovits"):
        raise ValueError("WAV translation requires CUDA Qwen3 ASR, CPU OPUS-MT and GPT-SoVITS; fallbacks are disabled")
    if config.gptsovits_personal_profile or config.gptsovits_voice_preset:
        raise ValueError('Public edition requires official pretrained checkpoints without private presets')
    if config.qwen_asr_model != ASR_MODEL or config.opusmt_model != MT_MODEL:
        raise ValueError("Only the verified 0.6B ASR and OPUS-MT zh-ja model identities are accepted")
    if not config.qwen_asr_offline or config.torch_dtype != "bfloat16" or config.torch_attn != "sdpa":
        raise ValueError("Production ASR must use cached-only CUDA BF16/SDPA")
    validate_reference(config.gptsovits_ref_audio, config.gptsovits_ref_text, config.gptsovits_ref_language)


def load_production(config):
    """One factory call; no independent model implementation or fallback path."""
    import torch
    from .pipeline.backends.qwen3_asr_torch import Qwen3ASRTorch
    from .pipeline.backends.opus_mt import OpusMTTranslator
    from .pipeline.backends.gptsovits import GPTSoVITS
    from .pipeline.backends.personal_presets import PersonalPresetTTS

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The verified CUDA BF16 ASR environment is required")
    pipeline = build_pipeline(config)
    try:
        if not isinstance(pipeline.asr, Qwen3ASRTorch) or not pipeline.asr.device.startswith("cuda"):
            raise RuntimeError("Actual ASR backend is not CUDA Qwen3-ASR")
        if not isinstance(pipeline.translator, OpusMTTranslator) or next(pipeline.translator._model.parameters()).device.type != "cpu":
            raise RuntimeError("Actual translator is not CPU OPUS-MT")
        actual_tts=pipeline.tts._backend if isinstance(pipeline.tts,PersonalPresetTTS) else pipeline.tts
        if not isinstance(actual_tts, GPTSoVITS) or pipeline.tts.ready["version"] != "v2ProPlus":
            raise RuntimeError("Actual TTS backend is not GPT-SoVITS v2 ProPlus")
        return pipeline
    except BaseException:
        close = getattr(pipeline.tts, "close", None)
        if close:
            close()
        raise


class CUDAMemory:
    def __init__(self):
        import torch
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; no CPU/mock fallback is permitted")
        torch.cuda.set_device(0)

    def reset(self):
        self.torch.cuda.synchronize(0)
        self.torch.cuda.reset_peak_memory_stats(0)

    def sync(self):
        self.torch.cuda.synchronize(0)

    def peak(self):
        self.sync()
        return {"allocated_mib":self.torch.cuda.max_memory_allocated(0)/2**20,
                "reserved_mib":self.torch.cuda.max_memory_reserved(0)/2**20}


@dataclass(frozen=True)
class UtteranceJob:
    id: str
    samples: np.ndarray = field(repr=False)
    origin: str
    vad_commit_audio_seconds: float
    committed_at: float = field(default_factory=time.perf_counter)

    def __post_init__(self):
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1.0001:
            raise ValueError("Utterance input must be nonempty finite normalized mono 16 kHz PCM")
        # A bytes-backed array cannot be made writable by setflags().
        object.__setattr__(self, "samples", np.frombuffer(samples.tobytes(), dtype=np.float32))


def recorded_utterances(audio, config, *, vad=False, repeat=1):
    """Commit times are recording offsets; feeding is not paced as live audio."""
    sequence = 0
    for repetition in range(repeat):
        if not vad:
            sequence += 1
            yield UtteranceJob(f"utt-{sequence:06d}", audio.samples, f"repeat-{repetition+1}", audio.duration_seconds)
            continue
        segmenter = Segmenter(config)
        for offset in range(0, audio.samples.size, segmenter.frame):
            frame = audio.samples[offset:offset+segmenter.frame]
            for segment in segmenter.add(frame):
                if np.sqrt(np.mean(segment.samples.astype(np.float64)**2)) < 1e-4:
                    continue
                sequence += 1
                yield UtteranceJob(f"utt-{sequence:06d}", segment.samples, f"repeat-{repetition+1}",
                                   (offset+frame.size)/INTERNAL_SAMPLE_RATE)
        for segment in segmenter.flush():
            # Upstream forced flush can return a silence-only trailing span.
            if np.sqrt(np.mean(segment.samples.astype(np.float64)**2)) < 1e-4:
                continue
            sequence += 1
            yield UtteranceJob(f"utt-{sequence:06d}", segment.samples, f"repeat-{repetition+1}", audio.duration_seconds)


@dataclass
class JobHandle:
    job: UtteranceJob
    future: Future = field(default_factory=Future)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    submitted_at: float = field(default_factory=time.perf_counter)
    enqueued_at: float = 0.0

    def cancel(self):
        self.cancel_event.set()

    def result(self, timeout=None):
        try:
            return self.future.result(timeout)
        except TimeoutError:
            self.cancel()
            raise


class WavTranslationPipeline:
    """One resident factory pipeline; bounded FIFO admission and one consumer.

    ASR/MT cancellation is cooperative at stage boundaries. TTS cancellation
    uses the existing generator/worker cancellation and drain mechanism.
    """

    def __init__(self, config: Config, *, capacity=4, job_timeout=180, _factory=load_production, _memory=None):
        require_production(config)
        if not 1 <= capacity <= 32 or job_timeout <= 0:
            raise ValueError("Invalid utterance queue capacity/timeout")
        self.config = config
        self.job_timeout = job_timeout
        started = time.perf_counter()
        self.memory = _memory or CUDAMemory()
        self.memory.reset()
        runtime_ready = time.perf_counter()
        self.pipeline = inference.run_sync(_factory, config)
        try:
            self.initialization = {"started_at":started, "ended_at":time.perf_counter(),
                                   "runtime_preparation_seconds":runtime_ready-started,
                                   "asr_process_peak":self.memory.peak()}
        except BaseException:
            self.pipeline.tts.close()
            raise
        self.initialization["seconds"] = self.initialization["ended_at"]-started
        self.initialization["factory_loading_seconds"] = self.initialization["ended_at"]-runtime_ready
        self._queue = queue.Queue(maxsize=capacity)
        self._stop = threading.Event()
        self._admission = threading.Lock()
        self._seen = set()
        self._active = None
        self._thread = threading.Thread(target=self._consume, daemon=True, name="wav-utterances")
        self.max_backlog = 0
        self._thread.start()

    def submit(self, job, *, block=False):
        handle = JobHandle(job)
        while not self._stop.is_set():
            with self._admission:
                if job.id in self._seen:
                    raise ValueError(f"Duplicate utterance ID: {job.id}")
                handle.enqueued_at = time.perf_counter()
                try:
                    self._queue.put_nowait(handle)
                    self._seen.add(job.id)
                    self.max_backlog = max(self.max_backlog, self._queue.qsize())
                    return handle
                except queue.Full:
                    if not block:
                        raise
            if time.perf_counter()-handle.submitted_at >= self.config.gptsovits_request_timeout:
                raise TimeoutError("Utterance queue admission timed out")
            self._stop.wait(.01)
        raise RuntimeError("WAV pipeline is closed")

    def _consume(self):
        while not self._stop.is_set():
            try:
                handle = self._queue.get(timeout=.1)
            except queue.Empty:
                continue
            self._active = handle
            try:
                result = inference.run_sync(self._run_job, handle)
                handle.future.set_result(result)
            except BaseException as exc:
                handle.future.set_exception(exc)
            finally:
                self._active = None
                self._queue.task_done()

    def _check(self, handle):
        if handle.cancel_event.is_set() or self._stop.is_set():
            raise CancelledError(f"Cancelled {handle.job.id}")
        if time.perf_counter()-handle.submitted_at > self.job_timeout:
            raise TimeoutError(f"Utterance deadline exceeded: {handle.job.id}")

    def _run_job(self, handle):
        self._check(handle)
        job = handle.job
        times = {"vad_commit_at":job.committed_at, "submission_at":handle.submitted_at,
                 "enqueued_at":handle.enqueued_at, "asr_start":time.perf_counter()}
        self.memory.reset()
        transcription = self.pipeline.asr.transcribe(job.samples, language="zh")
        self.memory.sync()
        times["asr_end"] = time.perf_counter()
        asr_memory = self.memory.peak()
        self._check(handle)
        if not transcription.text.strip():
            raise RuntimeError(f"ASR returned empty text for {job.id}")
        times["mt_start"] = time.perf_counter()
        japanese = self.pipeline.translator.translate(transcription.text, source="zh", target="ja")
        times["mt_end"] = time.perf_counter()
        self._check(handle)
        if not japanese.strip():
            raise RuntimeError(f"Translator returned empty text for {job.id}")
        times["tts_call_start"] = time.perf_counter()
        converter = WireResampler()
        native_sizes = []
        pcm_parts = []
        stream = self.pipeline.tts.synthesize_stream(japanese, "ja", None)
        try:
            for chunk in stream:
                self._check(handle)
                if chunk.sample_rate != 32000 or chunk.samples.size == 0:
                    raise RuntimeError("GPT-SoVITS must return playable native 32 kHz mono PCM")
                received = time.perf_counter()
                times.setdefault("tts_first_consumed_pcm", received)
                times["tts_last_consumed_pcm"] = received
                native_sizes.append(int(chunk.samples.size))
                pcm_parts.append(float32_to_pcm16(converter.add(chunk.samples, chunk.sample_rate)))
            pcm_parts.append(float32_to_pcm16(converter.finish()))
        finally:
            stream.close()
        self._check(handle)
        times["tts_end"] = time.perf_counter()
        metrics = self.pipeline.tts.last_metrics.copy()
        if not native_sizes or not metrics.get("reference_features_reused") or not metrics.get("speaker_embedding_reused"):
            raise RuntimeError("GPT-SoVITS did not confirm static conditioning reuse/playable chunks")
        if metrics["samples"] != sum(native_sizes) or metrics["chunks"] != len(native_sizes):
            raise RuntimeError("Native TTS sample/chunk accounting mismatch")
        times.update(tts_submission=metrics["submitted_at"], tts_first_parent_pcm=metrics["first_received_pcm_at"],
                     tts_last_parent_pcm=metrics["last_received_pcm_at"])
        pcm = np.concatenate(pcm_parts)
        if pcm.size != round(sum(native_sizes)*WIRE_SAMPLE_RATE/32000):
            raise RuntimeError("Stateful resampling changed the required sample count")
        rms = float(np.sqrt(np.mean((pcm.astype(np.float64)/32768)**2)))
        if rms < 1e-4:
            raise RuntimeError("GPT-SoVITS produced silent output")
        times["output_complete"] = time.perf_counter()
        return {"id":job.id, "origin":job.origin, "source_text":transcription.text, "japanese_text":japanese,
                "source_language_hint":"zh", "input_seconds":job.samples.size/16000,
                "input_sha256":hashlib.sha256(job.samples.tobytes()).hexdigest(),
                "vad_commit_audio_seconds":job.vad_commit_audio_seconds,
                "timestamps":times, "queue_wait_seconds":times["asr_start"]-handle.enqueued_at,
                "enqueue_wait_seconds":handle.enqueued_at-handle.submitted_at,
                "asr_seconds":times["asr_end"]-times["asr_start"], "mt_seconds":times["mt_end"]-times["mt_start"],
                "tts_first_pcm_seconds":times["tts_first_parent_pcm"]-times["tts_submission"],
                "tts_seconds":times["tts_end"]-times["tts_call_start"],
                "execution_seconds":times["output_complete"]-times["asr_start"],
                "submission_to_output_seconds":times["output_complete"]-handle.submitted_at,
                "audio_seconds":pcm.size/24000, "rms":rms, "native_chunk_samples":native_sizes,
                "tts_rtf":(times["tts_end"]-times["tts_call_start"])/(pcm.size/24000),
                "backlog_at_completion":self._queue.qsize(),
                "output_samples":int(pcm.size), "pcm_sha256":hashlib.sha256(pcm.tobytes()).hexdigest(),
                "asr_process_peak":asr_memory, "tts_metrics":metrics, "pcm":pcm}

    def close(self):
        if self._stop.is_set():
            return
        self._stop.set()
        if self._active:
            self._active.cancel()
        while True:
            try:
                handle = self._queue.get_nowait()
                handle.cancel()
                handle.future.set_exception(CancelledError("Pipeline shutdown"))
                self._queue.task_done()
            except queue.Empty:
                break
        # Closing the existing TTS adapter wakes an active IPC read promptly.
        self.pipeline.tts.close()
        self._thread.join(timeout=self.config.gptsovits_request_timeout+5)
        if self._thread.is_alive():
            raise RuntimeError("ASR/MT did not reach a cooperative shutdown boundary")
