"""Per-connection session state machine for wire events and pipeline output."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import time
from typing import Any, Awaitable, Callable

import numpy as np

from . import inference
from . import protocol as P
from .audio import (
    WIRE_SAMPLE_RATE,
    WireResampler,
    b64_pcm16_to_float32,
    float32_to_b64_pcm16,
    ms_to_samples,
    resample,
    time_stretch,
    to_internal,
)
from .config import Config
from .asr_vocabulary import is_internal_context_echo
from .languages import DEFAULT_TARGET_LANGUAGE, source_code
from .pipeline.base import Pipeline, Segment
from .pipeline.segmentation import Segmenter
from .pipeline.voice_reference import VoiceReferenceBuffer
from .pipeline.recognition import RecognitionContext, final_update

log = logging.getLogger("ort.session")

SendFn = Callable[[dict[str, Any]], Awaitable[None]]

# Output audio delta size: 100 ms at 24 kHz.
_AUDIO_CHUNK = ms_to_samples(100, WIRE_SAMPLE_RATE)


AudioSink = Callable[[np.ndarray, int], Awaitable[None]]


class TranslationSession:
    def __init__(
        self,
        send: SendFn,
        pipeline: Pipeline,
        config: Config,
        model: str,
        audio_sink: AudioSink | None = None,
        committed_input: bool = False,
    ) -> None:
        self._transport_send = send
        self._committed_input = committed_input
        self._aborted = False
        self._committed_audio = []
        self._committed_samples = 0
        self._recognition_context = None
        self._live_times = {}
        self._live_audio_sequence = 0
        self._last_live_id = 0
        self._live_identity = None
        self._pipeline = pipeline
        # WebRTC supplies audio_sink and receives translated speech on an outbound
        # audio track. WebSocket sessions keep output_audio.delta(base64) events.
        self._audio_sink = audio_sink
        self._config = config
        # Target language is session-level. Source language is detected by ASR;
        # session.update may supply an input transcription language hint.
        self._source_hint: str | None = None
        self._tgt_lang = "ja" if config.tts_backend == "gptsovits" else DEFAULT_TARGET_LANGUAGE
        self._output_speed = config.output_speed
        self._segmenter = Segmenter(config)
        self._voice_ref = VoiceReferenceBuffer(sample_rate=self._segmenter.sr)
        self._seg_no = 0
        # Committed (source, target) spans used as bounded translation context.
        self._ctx: list[tuple[str, str]] = []
        self._ctx_max = 4
        # Ordered buffer: committed spans enter the queue and are drained in order
        # within the handling turn. The event loop can still buffer more audio
        # while inference is awaited.
        self._queue: asyncio.Queue[Segment] = asyncio.Queue()
        self._session_obj: dict[str, Any] = {
            "id": P.new_event_id("sess"),
            "object": "realtime.translation_session",
            "model": model,
            "audio": {
                "input": {
                    "format": "pcm16",
                    "sample_rate": WIRE_SAMPLE_RATE,
                    "transcription": {"language": None},
                },
                "output": {
                    "format": "pcm16",
                    "sample_rate": WIRE_SAMPLE_RATE,
                    "language": self._tgt_lang,
                },
            },
        }
        lexicon_metadata = getattr(pipeline.tts,'lexicon_metadata',None)
        self._preset_playback=getattr(pipeline.tts,'ready',{}).get('voice_preset')
        self._preset_playback_ack=False
        if lexicon_metadata is not None:
            self._session_obj['metadata'] = {'pronunciation_lexicon':lexicon_metadata.copy()}
        if self._preset_playback:
            self._session_obj.setdefault('metadata',{})['voice_preset']=self._preset_playback
        if committed_input:
            ready = getattr(pipeline.tts,"ready",{})
            self._session_obj["voicebridge"] = {
                "version":1,"input_mode":"client_committed","asr_mode":"utterance_final",
                "asr_backend":config.asr_backend,"asr_model":getattr(pipeline.asr,"_repo",config.qwen_asr_model),
                "asr_device":getattr(pipeline.asr,"device",None),"asr_attention":config.torch_attn,
                "translator_backend":config.translator_backend,"translator_model":getattr(pipeline.translator,"repo",config.opusmt_model),
                "tts_backend":config.tts_backend,"tts_version":ready.get("version"),
                "reference_fingerprint":ready.get("reference_fingerprint"),
                "tts_checkpoint_identity":ready.get("checkpoint_identity"),
                "tts_worker_pid":ready.get("pid"),
                "tts_personal_voice":ready.get("personal_voice"),
                "tts_voice_preset":ready.get("voice_preset"),
                "pronunciation_lexicon":lexicon_metadata,
                "server_monotonic":time.perf_counter(),"clock":"Windows system-wide perf_counter"}

    async def _send(self, event):
        if self._aborted:
            return
        if self._recognition_context is not None:
            extra = {"context":self._recognition_context.wire(),"stage_times":self._live_times.copy(),
                     "transcript_phase":"final"}
            if event["type"] == P.SERVER_OUTPUT_AUDIO_DELTA:
                raw = base64.b64decode(event["delta"],validate=True)
                extra["chunk_sequence"] = self._live_audio_sequence
                extra.update(sample_rate=24000,channels=1,pcm_format="s16le",samples=len(raw)//2,
                             pcm_sha256=hashlib.sha256(raw).hexdigest())
                self._live_audio_sequence += 1
                self._live_times.setdefault("first_server_pcm_at",time.perf_counter())
            if event["type"] == P.SERVER_OUTPUT_AUDIO_DONE:
                self._live_times["tts_end"] = time.perf_counter()
                extra["chunk_count"] = self._live_audio_sequence
                extra["tts_metrics"] = getattr(self._pipeline.tts,"last_metrics",{}).copy()
                extra["stage_times"] = self._live_times.copy()
            event = {**event,"voicebridge":extra}
        await self._transport_send(event)

    def abort(self):
        """Invalidate output immediately; native calls drain without replay."""
        self._aborted = True
        from voicebridge.camera_ipc import publish_optional
        if getattr(self,'_live_times',{}).get('request_kind')!='QUICK_REPLY':
            publish_optional(self._pipeline, 'ASR_CANCELLED', getattr(self,'_recognition_context',None), time.perf_counter())
        pending=getattr(self,'_qwen_pending',None)
        if pending and not pending.done():pending.cancel()
        self._committed_audio.clear()
        self._committed_samples = 0

    async def _qwen_fallback(self,quality,source):
        self._live_times['qwen_request_at']=time.perf_counter()
        self._qwen_pending=asyncio.create_task(quality.translate(source,self._feature_snapshot))
        try:
            result=await self._qwen_pending
            self._live_times.update(qwen_completed_at=time.perf_counter(),mt_end=time.perf_counter(),translation_backend='QWEN_FALLBACK')
            return result
        finally:self._qwen_pending=None

    async def _committed_commit(self, message):
        try:
            from .tts_controls import controls
            self._request_tts_options = controls(message.get("tts_options")) if "tts_options" in message else None
            context = RecognitionContext(**message.get("voicebridge",{}))
            identity = context.session_id,context.epoch
            if context.utterance_id <= self._last_live_id or self._live_identity not in {None,identity}:
                raise ValueError("Duplicate/reordered utterance or changed session generation")
            if self._source_hint not in {None, "zh", "ja", "en"} or self._tgt_lang != "ja":
                raise ValueError("The committed production profile requires zh/ja/en/auto -> ja")
            if not self._committed_audio:
                raise ValueError("Committed utterance has no PCM")
            wire = np.concatenate(self._committed_audio)
            self._committed_audio.clear()
            self._committed_samples = 0
            internal = to_internal(wire)
            if abs(len(internal)-(context.audio_end_sample-context.audio_start_sample)) > 1:
                raise ValueError("Utterance metadata and PCM sample range disagree")
            if np.sqrt(np.mean(internal.astype(np.float64)**2)) < 1e-4:
                raise ValueError("Silence-only committed utterance")
        except (ValueError,TypeError) as exc:
            await self._send(P.error_event(str(exc)))
            return
        self._last_live_id = context.utterance_id
        self._live_identity = identity
        self._recognition_context = context
        self._live_times = {"server_admission_at":time.perf_counter()}
        self._live_audio_sequence = 0
        try:
            immutable = np.frombuffer(internal.tobytes(),dtype=np.float32)
            await self._process(Segment(immutable,16000))
        finally:
            self._recognition_context = None

    def _recognize_live(self, segment):
        torch = getattr(self._pipeline.asr,"_torch",None)
        cuda = torch.cuda if torch is not None and self._config.runtime=="cuda" else None
        if cuda:
            cuda.synchronize(0)
            cuda.reset_peak_memory_stats(0)
        self._live_times["asr_start"] = time.perf_counter()
        from voicebridge.camera_ipc import publish_optional
        publish_optional(self._pipeline, 'ASR_STARTED', self._recognition_context, self._live_times['asr_start'])
        names=getattr(self,'_asr_names_snapshot',None)
        try:
            result = self._pipeline.asr.transcribe(segment.samples,self._source_hint,**({'vocabulary_snapshot':names} if names else {}))
        except Exception:
            publish_optional(self._pipeline, 'JOB_FAILED', self._recognition_context, time.perf_counter())
            raise
        if hasattr(self._pipeline.asr,'last_recognition_context'):
            self._live_times['asr_name_context']=dict(self._pipeline.asr.last_recognition_context)
        if cuda:
            cuda.synchronize(0)
            self._live_times["asr_allocated_mib"] = cuda.max_memory_allocated(0)/2**20
            self._live_times["asr_reserved_mib"] = cuda.max_memory_reserved(0)/2**20
        self._live_times["asr_end"] = time.perf_counter()
        return final_update(result,self._recognition_context)

    def _translate_live(self, text, context, source, target):
        self._live_times["mt_start"] = time.perf_counter()
        self._live_times['opus_start']=self._live_times['mt_start']
        policy=getattr(self._pipeline,'voicebridge_policy',None)
        snapshot=getattr(self,'_feature_snapshot',None)
        if callable(getattr(policy,'direct',None)) and snapshot and source=='zh' and target=='ja':
            direct=policy.direct(text,snapshot)
            if direct:
                self._live_times.update(translation_backend='DICTIONARY_DIRECT',mt_end=time.perf_counter())
                return direct['text']
        from .pipeline.routing import japanese_route
        route, reason = japanese_route(text, source)
        self._live_times.update(language_route=route, route_reason=reason, detected_language=source)
        if route == "ja_direct":
            result = text
        elif route == "en_mt":
            import os
            if os.environ.get("ORT_EXPERIMENTAL_EN_MT") != "1":
                raise ValueError("English MT candidate failed semantic acceptance; experimental opt-in required")
            if not hasattr(self._pipeline, "_english_mt"):
                from .pipeline.backends.english_mt import EnglishMT
                self._pipeline._english_mt = EnglishMT()
            result = self._pipeline._english_mt.translate(text)
        else:
            settings=getattr(self,'_feature_snapshot',None)
            plan=None
            if settings and source=='zh':
                from voicebridge.terminology import TranslationTerms
                plan=TranslationTerms(text,settings['dictionary'],source)
                if plan.active:
                    text=plan.model_input
                    self._feature_snapshot={**settings,'translation_terms':plan,'prefer_terminology_candidate':True}
            smart=bool(settings and settings['settings']['segmentation']['smart_mandarin'])
            if smart and source=='zh':
                from voicebridge.mandarin import sentences
                groups=sentences(text)
                self._live_times.update(mandarin_groups=groups,mandarin_group_count=len(groups),segmentation_added_wait_seconds=0.)
            else:groups=[text]
            if len(groups)>1 and len(groups)<=16 and callable(getattr(self._pipeline.translator,'translate_many',None)):
                result=''.join(self._pipeline.translator.translate_many(groups))
            else:result = self._pipeline.translator.translate(text,context,source,target)
            if plan and plan.active:
                try:result=plan.restore(result);result=plan.reviewed_candidate() or result
                finally:self._live_times['translation_terminology']=plan.audit()
        self._live_times["mt_end"] = time.perf_counter()
        self._live_times['opus_end']=self._live_times['mt_end']
        return result

    # --- Lifecycle -----------------------------------------------------------
    async def start(self) -> None:
        await self._send(
            P.server_event(P.SERVER_SESSION_CREATED, session=self._session_obj)
        )

    async def _drain(self) -> None:
        """Process all currently queued committed spans in order.

        Events are sent by the route task inside the receive -> handle turn. This
        matches uvicorn and Starlette TestClient behavior; detached task sends
        are not pumped by TestClient and were observed to deadlock. Latency gains
        come from the incremental committer splitting long utterances into small
        committed spans, not from task-level inference parallelism.
        """
        while True:
            try:
                seg = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await self._process(seg)
            except Exception:  # noqa: BLE001
                log.exception("處理 span 失敗（續做下一個）")

    async def handle(self, message: dict[str, Any]) -> None:
        raw_type = message.get("type")
        if not isinstance(raw_type, str):
            await self._send(P.error_event("事件缺少字串型別 'type'"))
            return
        norm = P.normalize_inbound_type(raw_type)
        if norm == P.C_SESSION_UPDATE:
            await self._on_session_update(message)
        elif norm == P.C_INPUT_AUDIO_APPEND:
            await self._on_audio_append(message)
        elif norm == P.C_INPUT_AUDIO_COMMIT:
            if self._committed_input:
                await self._committed_commit(message)
            else:
                await self._on_commit()
        elif raw_type=='voicebridge.quick_reply' and self._committed_input:
            await self._quick_reply(message)
        elif norm == P.C_INPUT_AUDIO_CLEAR:
            self._segmenter.reset()
            self._committed_audio.clear()
            self._committed_samples = 0
        else:
            await self._send(P.error_event(f"不支援的事件型別: {raw_type}"))

    async def _quick_reply(self,message):
        return await self._gpu_owned_call(self._quick_reply_owned,message)

    async def _gpu_owned_call(self,fn,*args):
        coordinator=getattr(self._pipeline,'voicebridge_gpu',None)
        if coordinator is None:return await fn(*args)
        waited=time.perf_counter();await asyncio.wait_for(coordinator.acquire(),self._config.gptsovits_request_timeout)
        self._gpu_wait_seconds=time.perf_counter()-waited
        if hasattr(self,'_live_times'):self._live_times['gpu_scheduling_wait_seconds']=self._gpu_wait_seconds
        async def owned():
            try:return await fn(*args)
            finally:coordinator.release()
        task=asyncio.create_task(owned())
        try:return await asyncio.shield(task)
        except asyncio.CancelledError:
            # A Python cancellation cannot stop a native CUDA call. Keep the
            # stage lease until it actually drains; output is fenced at once.
            self.abort()
            task.add_done_callback(lambda done:done.exception() if not done.cancelled() else None)
            raise

    async def _quick_reply_owned(self,message):
        """Explicit final Japanese, same PCM emitter and resident TTS lease."""
        from voicebridge.quick_reply import PHRASES,quick_snapshot
        try:
            text=message['text']
            from voicebridge.quick_reply import validate_shortcut
            validate_shortcut(text,message.get('shortcut_id'))
            if text not in PHRASES or not self._preset_playback_ack or self._tgt_lang!='ja':raise ValueError('Invalid quick reply, language or playback handshake')
            context=RecognitionContext(**message['voicebridge']);identity=context.session_id,context.epoch
            if context.utterance_id<=self._last_live_id or self._live_identity not in {None,identity}:raise ValueError('Stale quick reply identity')
            expected=message['preset']
            if expected!=self._pipeline.tts.ready['release_preset']:raise ValueError('Quick reply preset changed before synthesis')
            self._feature_snapshot=quick_snapshot(message['settings'],self._pipeline.voicebridge_policy)
        except (KeyError,ValueError,TypeError) as exc:
            await self._send(P.error_event(str(exc),code='utterance_failed_private'));return
        self._last_live_id=context.utterance_id;self._live_identity=identity;self._recognition_context=context
        self._live_times=dict(server_admission_at=time.perf_counter(),request_kind='QUICK_REPLY',translation_backend='DIRECT_TTS',gpu_scheduling_wait_seconds=getattr(self,'_gpu_wait_seconds',0.))
        self._live_audio_sequence=0;self._request_tts_options=None
        try:
            await self._send(P.server_event(P.SERVER_OUTPUT_TRANSCRIPT_DELTA,delta=text))
            await self._send(P.server_event(P.SERVER_OUTPUT_TRANSCRIPT_DONE,transcript=text))
            if self._aborted:return
            self._live_times['tts_start']=time.perf_counter()
            await self._stream_tts(self._pipeline.tts.synthesize_stream,text,None,time.perf_counter())
        except Exception as exc:await self._send(P.error_event('Quick reply synthesis unavailable; local diagnostic only',code='utterance_failed_private'))
        finally:self._recognition_context=None

    # --- Event handling ------------------------------------------------------
    async def _on_session_update(self, message: dict[str, Any]) -> None:
        patch = message.get("session")
        if self._preset_playback and isinstance(patch,dict) and 'voicebridge_playback' in patch:
            self._preset_playback_ack=patch['voicebridge_playback']==self._preset_playback['playback']
            if not self._preset_playback_ack:
                await self._send(P.error_event('Selected personal preset requires its approved browser1.25/preservesPitch playback policy'))
                return
        if isinstance(patch, dict):
            audio = patch.get("audio", {})
            input_lang = (
                audio.get("input", {}).get("transcription", {}).get("language")
                if isinstance(audio, dict)
                else None
            )
            if isinstance(input_lang, str):
                from . import languages as L

                raw_hint = L.norm(input_lang)
                hint = (
                    None
                    if raw_hint in {"", "auto"}
                    else L.source_code(input_lang) or raw_hint
                )
                self._source_hint = hint
                transcription_cfg = self._session_obj["audio"]["input"][
                    "transcription"
                ]
                transcription_cfg["language"] = hint

            output_lang = (
                audio.get("output", {}).get("language")
                if isinstance(audio, dict)
                else None
            )
            if isinstance(output_lang, str) and output_lang:
                from . import languages as L

                if L.supported_target(self._config.tts_backend, output_lang):
                    self._tgt_lang = L.norm(output_lang)
                    self._session_obj["audio"]["output"]["language"] = self._tgt_lang
                else:
                    # The active TTS backend does not support this target;
                    # keep the old setting.
                    await self._send(
                        P.error_event(
                            f"目標語 {output_lang!r} 不在 {self._config.tts_backend} "
                            f"支援範圍：{sorted(L.tts_targets(self._config.tts_backend))}"
                        )
                    )
        await self._send(
            P.server_event(P.SERVER_SESSION_UPDATED, session=self._session_obj)
        )

    async def _on_audio_append(self, message: dict[str, Any]) -> None:
        b64 = message.get("audio")
        if not isinstance(b64, str) or not b64:
            await self._send(P.error_event("input_audio_buffer.append 缺少 'audio'"))
            return
        try:
            if self._committed_input:
                raw = base64.b64decode(b64,validate=True)
                if not raw or len(raw)%2:
                    raise ValueError("Invalid committed PCM16 byte boundary")
            wire = b64_pcm16_to_float32(b64)
        except Exception as exc:  # noqa: BLE001
            await self._send(P.error_event(f"音訊解碼失敗: {exc}"))
            return
        await self.feed_audio(wire)

    async def feed_audio(self, wire_24k: np.ndarray) -> None:
        """Feed wire-sample-rate 24 kHz float32 audio, segment it, and process in order.

        WebSocket calls this after base64 decoding; WebRTC calls it after Opus
        decoding and resampling.
        """
        if self._committed_input:
            if self._committed_samples+wire_24k.size > 31*WIRE_SAMPLE_RATE:
                raise ValueError("Committed utterance exceeds the bounded 31 second server buffer")
            self._committed_audio.append(wire_24k.copy())
            self._committed_samples += wire_24k.size
            return
        internal = to_internal(wire_24k)
        for seg in self._segmenter.add(internal):
            self._queue.put_nowait(seg)
        await self._drain()

    async def commit(self) -> None:
        """Force-flush remaining audio and process it, including WebRTC track end."""
        await self._on_commit()

    async def _on_commit(self) -> None:
        # commit is an explicit boundary: flush remaining audio and drain the
        # queue so "commit -> receive result" remains predictable for clients.
        for seg in self._segmenter.flush():
            self._queue.put_nowait(seg)
        await self._drain()

    # --- Pipeline driver -----------------------------------------------------
    async def _process(self, segment: Segment) -> None:
        return await self._gpu_owned_call(self._process_owned,segment)

    async def _process_owned(self,segment:Segment)->None:
        policy=getattr(self._pipeline,'voicebridge_policy',None)
        self._feature_snapshot=policy.snapshot() if policy else None
        if getattr(self,'_preset_playback',None) and not self._preset_playback_ack:
            await self._send(P.error_event('Personal preset playback policy is not acknowledged; refusing wrong-speed output'))
            return
        self._seg_no += 1
        seg_no = self._seg_no
        # t_commit approximates when the span enters the pipeline; perceived latency starts here.
        t_commit = time.perf_counter()
        seg_dur = segment.samples.size / max(1, segment.sample_rate)

        try:
            t = time.perf_counter()
            name_provider=getattr(self._pipeline,'voicebridge_asr_context',None)
            self._asr_names_snapshot=name_provider(self._feature_snapshot) if name_provider else None
            if self._committed_input:
                transcription = await inference.run(self._recognize_live,segment)
            else:
                transcription = await inference.run(
                    self._pipeline.asr.transcribe, segment.samples, self._source_hint
                )
            src_text = transcription.text
            src_lang = source_code(transcription.language) or source_code(
                self._source_hint
            )
            asr_s = time.perf_counter() - t
        except Exception as exc:  # noqa: BLE001
            log.exception("ASR 失敗")
            if policy and self._committed_input:
                self._live_times['translation_policy']=dict(outcome='TECHNICAL_ERROR',reason='asr_failure',error_type=type(exc).__name__)
                await self._send(P.error_event('ASR unavailable for this utterance; local diagnostic only',code='utterance_failed_private'))
                return
            await self._send(P.error_event(f"ASR 失敗: {exc}", code="server_error"))
            return
        # Independent of translation/style policy: internal prompt text must
        # never enter MT, rolling references, Qwen recovery, or TTS.
        recognition_audit = getattr(self._pipeline.asr, 'last_recognition_context', {}) or {}
        if is_internal_context_echo(src_text) or (
            not src_text and recognition_audit.get('blocked_reason') == 'ASR_CONTEXT_ECHO_BLOCKED'
        ):
            self._live_times['translation_policy'] = dict(
                outcome='TECHNICAL_ERROR', reason='ASR_CONTEXT_ECHO_BLOCKED'
            )
            log.warning('ASR_CONTEXT_ECHO_BLOCKED seg#%d; no translation or audio', seg_no)
            await self._send(P.error_event(
                'ASR_CONTEXT_ECHO_BLOCKED: internal recognition hint suppressed; local diagnostic only',
                code='utterance_failed_private',
            ))
            return
        if not src_text:
            if self._committed_input:
                await self._send(P.error_event("ASR returned empty final text",code='utterance_failed_private' if policy else 'server_error'))
            log.info(
                "[timing] seg#%d dur=%.2fs asr=%.3fs -> empty ASR, skipped",
                seg_no,
                seg_dur,
                asr_s,
            )
            return
        if self._aborted:
            return
        if not src_lang:
            await self._send(
                P.error_event("ASR 未回傳來源語言，無法建立翻譯 source_lang_code")
            )
            return
        if self._committed_input:
            voice_reference = None
        else:
            self._voice_ref.append(segment.samples, src_text)
            voice_reference = self._voice_ref.snapshot()
        await self._send(
            P.server_event(P.SERVER_INPUT_TRANSCRIPT_DELTA, delta=src_text)
        )

        self._qwen_final=False
        try:
            t = time.perf_counter()
            ctx = tuple(self._ctx[-self._ctx_max :])
            tgt_text = await inference.run(
                self._translate_live if self._committed_input else self._pipeline.translator.translate,
                src_text,
                ctx,
                src_lang,
                self._tgt_lang,
            )
            mt_s = time.perf_counter() - t
        except Exception as exc:  # noqa: BLE001
            log.exception("翻譯失敗")
            quality=getattr(self._pipeline,'voicebridge_qwen',None)
            if quality and quality.mode=='QWEN_FALLBACK' and src_lang=='zh':
                try:
                    tgt_text,audit=await self._qwen_fallback(quality,src_text)
                    self._qwen_final=True;self._live_times['translation_policy']=audit
                except Exception as fallback_error:
                    self._live_times['translation_policy']=dict(outcome='FILLER_SKIPPED' if str(fallback_error)=='FILLER_SKIPPED' else 'QWEN_TECHNICAL_FAILURE',reason=str(fallback_error))
                    await self._send(P.error_event('No audio for this utterance; local diagnostic only',code='utterance_failed_private'));return
            elif policy and self._committed_input:
                from voicebridge.terminology import TerminologyError
                term_error=isinstance(exc,TerminologyError)
                self._live_times['translation_policy']=dict(outcome='UNRESOLVED_PRIVATE' if term_error else 'TECHNICAL_ERROR',reason='terminology_unverified' if term_error else 'mt_failure',error_type=type(exc).__name__,details=str(exc) if term_error else None)
                await self._send(P.error_event('术语未完整保留，本句未播放；请检查词典' if term_error else 'MT unavailable for this utterance; local diagnostic only',code='translation_rejected' if term_error else 'utterance_failed_private'))
                return
            else:
                await self._send(P.error_event(f"翻譯失敗: {exc}", code="server_error"))
                return
        if self._aborted:
            return
        if not tgt_text:
            if self._committed_input:
                await self._send(P.error_event("Translator returned empty text",code='utterance_failed_private' if policy else 'server_error'))
            return
        if policy and self._tgt_lang=='ja' and not self._qwen_final:
            from voicebridge.polite_japanese import TranslationRejected
            try:
                tgt_text,audit=await inference.run(policy.translate,src_text,tgt_text,self._feature_snapshot)
                quality=getattr(self._pipeline,'voicebridge_qwen',None)
                # Source sentinels assess model translations, not the user's
                # validated whole-utterance override (which may spell a name
                # in kana instead of the separate terminology rule's kanji).
                if quality and quality.mode=='QWEN_FALLBACK' and audit.get('outcome')!='DICTIONARY_DIRECT':
                    from voicebridge.hybrid_translation import semantic_sentinels
                    if not semantic_sentinels(src_text,tgt_text,self._feature_snapshot['dictionary'],recovery=True):
                        raise TranslationRejected(dict(reason='OPUS_source_heuristics_unresolved',japanese_text=tgt_text))
            except TranslationRejected as exc:
                quality=getattr(self._pipeline,'voicebridge_qwen',None)
                if quality and quality.mode=='QWEN_FALLBACK' and src_lang=='zh':
                    try:tgt_text,audit=await self._qwen_fallback(quality,src_text)
                    except Exception as fallback_error:
                        self._live_times['translation_policy']=dict(outcome='FILLER_SKIPPED' if str(fallback_error)=='FILLER_SKIPPED' else 'QWEN_TECHNICAL_FAILURE',reason=str(fallback_error))
                        await self._send(P.error_event('No audio for this utterance; local diagnostic only',code='utterance_failed_private'));return
                else:
                    if self._committed_input:self._live_times['translation_policy']=exc.audit
                    await self._send(P.error_event(str(exc),code='translation_rejected'))
                    return
            except Exception:
                quality=getattr(self._pipeline,'voicebridge_qwen',None)
                if quality and quality.mode=='QWEN_FALLBACK' and src_lang=='zh':
                    try:tgt_text,audit=await self._qwen_fallback(quality,src_text)
                    except Exception as fallback_error:
                        self._live_times['translation_policy']=dict(outcome='QWEN_TECHNICAL_FAILURE',reason=str(fallback_error))
                        await self._send(P.error_event('No audio for this utterance; local diagnostic only',code='utterance_failed_private'));return
                else:
                    await self._send(P.error_event('日语校验暂不可用，本句未播放',code='translation_rejected'));return
            if self._committed_input:self._live_times['translation_policy']=audit
        # Accumulate bounded cross-span context for coherent continuation.
        if self._aborted:return
        self._ctx.append((src_text, tgt_text))
        if len(self._ctx) > self._ctx_max:
            self._ctx = self._ctx[-self._ctx_max :]
        await self._send(
            P.server_event(P.SERVER_OUTPUT_TRANSCRIPT_DELTA, delta=tgt_text)
        )
        await self._send(
            P.server_event(P.SERVER_OUTPUT_TRANSCRIPT_DONE, transcript=tgt_text)
        )

        if self._aborted:return

        try:
            t = time.perf_counter()
            if self._committed_input:
                self._live_times["tts_start"] = t
            stream = getattr(self._pipeline.tts, "synthesize_stream", None)
            if callable(stream) and self._output_speed == 1.0:
                first_audio_s, out_dur = await self._stream_tts(stream, tgt_text, voice_reference, t_commit)
                log.info("[timing] seg#%d streaming_tts=%.3fs first_audio=%.3fs audio=%.3fs",
                         seg_no, time.perf_counter() - t, first_audio_s, out_dur)
                return
            tts = await inference.run(
                self._pipeline.tts.synthesize,
                tgt_text,
                self._tgt_lang,
                voice_reference,
            )
            tts_s = time.perf_counter() - t
        except Exception as exc:  # noqa: BLE001
            log.exception("TTS 失敗")
            await self._send(P.error_event(f"TTS 失敗: {exc}", code="server_error"))
            return

        # Translated speech runs longer than the source; speed it up
        # (pitch-preserved) so continuous output tracks the speaking pace.
        samples = tts.samples
        if self._output_speed > 1.0 + 1e-3 and samples.size:
            samples = await inference.run(
                time_stretch, samples, tts.sample_rate, self._output_speed
            )

        first_audio_s = await self._emit_audio(samples, tts.sample_rate, t_commit)
        out_dur = samples.size / max(1, tts.sample_rate)
        total_s = time.perf_counter() - t_commit
        # RTF = processing time / audio duration; values below 1 are faster than real time.
        log.info(
            "[timing] seg#%d dur=%.2fs | asr=%.3fs rtf=%.2f src=%s chars=%d | "
            "mt=%.3fs tgt=%s chars=%d | tts=%.3fs rtf=%.2f out=%.2fs | "
            "first_audio=%.3fs total=%.3fs",
            seg_no,
            seg_dur,
            asr_s,
            asr_s / seg_dur if seg_dur else 0.0,
            src_lang,
            len(src_text),
            mt_s,
            self._tgt_lang,
            len(tgt_text),
            tts_s,
            tts_s / out_dur if out_dur else 0.0,
            out_dur,
            first_audio_s,
            total_s,
        )

    async def _stream_tts(self, stream, text, reference, t_commit):
        options = getattr(self,"_request_tts_options",None)
        kwargs={'options':options} if options and self._config.tts_backend=='gptsovits' else {}
        if getattr(self,'_feature_snapshot',None) is not None:kwargs['policy_snapshot']=self._feature_snapshot
        iterator = stream(text,self._tgt_lang,reference,**kwargs)
        converter = WireResampler()
        first = None
        emitted = 0
        try:
            while True:
                chunk = await inference.run(_next_tts_chunk, iterator)
                if chunk is None:
                    break
                wire = converter.add(chunk.samples, chunk.sample_rate)
                if wire.size:
                    received = await self._emit_audio(wire, WIRE_SAMPLE_RATE, t_commit, done=False)
                    first = received if first is None else first
                    emitted += wire.size
            wire = converter.finish()
            if wire.size:
                received = await self._emit_audio(wire, WIRE_SAMPLE_RATE, t_commit, done=False)
                first = received if first is None else first
                emitted += wire.size
        except BaseException:
            if emitted:
                await self._send(P.server_event(P.SERVER_OUTPUT_AUDIO_DONE))
            raise
        finally:
            await inference.run(iterator.close)
        await self._send(P.server_event(P.SERVER_OUTPUT_AUDIO_DONE))
        return first if first is not None else time.perf_counter() - t_commit, emitted / WIRE_SAMPLE_RATE

    async def _emit_audio(self, samples: np.ndarray, sr: int, t_commit: float, done: bool = True) -> float:
        """Emit audio deltas and return seconds from commit to first audio delta."""
        if samples.size == 0:
            if done:
                await self._send(P.server_event(P.SERVER_OUTPUT_AUDIO_DONE))
            return time.perf_counter() - t_commit
        wire = resample(samples, sr, WIRE_SAMPLE_RATE)

        if self._audio_sink is not None:
            # WebRTC sends the whole translated span to the outbound track, which
            # handles 20 ms frame extraction and pacing.
            await self._audio_sink(wire, WIRE_SAMPLE_RATE)
            first_audio_s = time.perf_counter() - t_commit
            if done:
                await self._send(P.server_event(P.SERVER_OUTPUT_AUDIO_DONE))
            return first_audio_s

        first_audio_s = -1.0
        for start in range(0, wire.size, _AUDIO_CHUNK):
            chunk = wire[start : start + _AUDIO_CHUNK]
            await self._send(
                P.server_event(
                    P.SERVER_OUTPUT_AUDIO_DELTA,
                    delta=float32_to_b64_pcm16(chunk),
                )
            )
            if first_audio_s < 0:
                first_audio_s = time.perf_counter() - t_commit
        if done:
            await self._send(P.server_event(P.SERVER_OUTPUT_AUDIO_DONE))
        return first_audio_s


def _next_tts_chunk(iterator):
    return next(iterator, None)
