# Wire Protocol

Warm-Qwen/quick-reply additive local committed-v1 event:
`{"type":"voicebridge.quick_reply","text":"はい","preset":"PERSONAL_R17_NATURAL","settings":{...},"voicebridge":{...RecognitionContext...}}`.
Only `はい` and `はい、よろしくお願いいたします。` are accepted. The same selected-personal-playback handshake is mandatory; the preset and validated settings snapshot are frozen at admission. This control job carries **no input PCM and no ASR transcript**. Its `RecognitionContext` uses the reserved control range0..1 and command-admission timestamps for identity/order, not recorded speech samples. Responses reuse the existing output transcript, ordered24kHz mono PCM16 delta/checksum and completion events. `stage_times.request_kind=QUICK_REPLY`, backend `DIRECT_TTS`; ASR/MT stages do not run. Client output ownership, route generation and FIFO fencing remain mandatory; DIRECT/PAUSED/STOPPED reject admission locally.

Qwen Fallback is opt-in on the owned hot runtime (`POST /translation-backend`, mode `QWEN_FALLBACK`; default `OPUS_FAST`). Successful Qwen text bypasses BusinessRewrite/politeness/source-heuristic vetoes. Source heuristics only populate private `audit_only_warnings`; `semantic_verification=UNVERIFIED` even with no warning. Explicit whole-utterance fillers are configured in `config/qwen_fillers.json`. Empty/malformed, unterminated/truncated, timeout, crash, canceled and stale outputs never reach TTS. This policy does not certify translation fidelity. No application failure notice or beep is synthesized. `/runtime` reports QWEN_LOADING/WARMING/READY/ERROR separately from ASR/OPUS/TTS readiness; `/qwen-recover` explicitly restarts only a failed Qwen owner after the session lease drains.

STEP07.8 additive committed-profile option: a commit may include top-level
`tts_options` with `speed_factor` (1.0–1.3), `fragment_interval` (0.05–0.3),
`trim_digital_silence` (boolean). These are frozen per utterance; unknown keys or
invalid values fail explicitly. No event names or PCM formats changed. Source
language may be `zh`, `ja`, `en` or `auto`; final Japanese can bypass MT but never
FIFO/TTS. English remains experimental after a failed semantic acceptance test.

STEP07.8.2: omitted native controls default to speed1.20, fragment_interval0.30,
trim_digital_silence=false. Additive `pronunciation_enabled` boolean defaults to
true and can disable the adapter's TTS-only lexicon per request; raw transcripts
are unaffected. Explicit speed1.00 remains supported. Production dictionary
contains only user-accepted Python/MySQL readings. Adaptive is client-side,
off by default; frozen tiers1.20/1.25/1.30 use measured ring duration with3/6s up
and2.5/5.5s down thresholds. Pending unsynthesized jobs are reported separately.

VoiceBridge JP STEP 05.6 production profile: `gptsovits` enforces its configured
static reference and defaults output to Japanese. Its resident worker yields
native PCM chunks, resampled continuously to the existing 24 kHz wire format.
There is one audio-done event per utterance and no new event names. Rolling
reference descriptions below apply to the preserved legacy backends.

Goal: be **WebSocket wire-compatible with the OpenAI Realtime Translation API** (`/v1/realtime/translations`), so an OpenAI SDK/client only has to swap the base URL to talk to the local server.

Source: <https://developers.openai.com/api/docs/guides/realtime-translation> (captured 2026-05).

## Connection

```
ws://<host>/v1/realtime/translations?model=gpt-realtime-translate
```

- Locally, `Authorization` is not validated by default (added later for cloud mode).
- The server emits `session.created` immediately after the connection is established.

## Audio format

- Wire: **24 kHz, PCM16, mono, base64-of-raw** (`audio` / `delta` fields).
- Internally ASR/VAD use 16 kHz float32; 24k↔16k resampling is handled at the boundary in `ort/audio.py`.

## Event-prefix compatibility strategy

The official docs are inconsistent about the input event prefix (`input_audio_buffer.append` vs `session.input_audio_buffer.append`). This project:

- **Output (server→client)**: uses `session.`-prefixed translation events as the canonical names.
- **Input (client→server)**: accepts both the prefixed and unprefixed aliases, normalizes, then dispatches (see `_INBOUND_ALIASES` in `ort/protocol.py`).

## Client → server

| Wire type (accepted aliases) | Internal canonical type | Key fields |
|---|---|---|
| `session.update` | `session.update` | `session.audio.input.transcription.language`, `session.audio.output.language` |
| `input_audio_buffer.append` / `session.input_audio_buffer.append` | append | `audio` (base64 PCM16 @24k) |
| `input_audio_buffer.commit` / `session.…commit` | commit | — (force-close the current buffer) |
| `input_audio_buffer.clear` / `session.…clear` | clear | — (clear the buffer) |

> Note: OpenAI translation sessions recommend "stream continuously, including inter-utterance silence". This server auto-commits at utterance boundaries via VAD; `input_audio_buffer.commit` is an optional forced flush.

## Server → client

| type | fields | description |
|---|---|---|
| `session.created` | `session` | connection established, includes the session object |
| `session.updated` | `session` | response to `session.update` (echoes the configured language) |
| `session.input_transcript.delta` | `delta` | source-language recognized text |
| `session.output_transcript.delta` | `delta` | translated text |
| `session.output_transcript.done` | `transcript` | end of that translated span |
| `session.output_audio.delta` | `delta` | translated speech chunk (base64 PCM16 @24k, ~100 ms/chunk) |
| `session.output_audio.done` | — | end of that speech span |
| `error` | `error{type,message}` | error |

All server events carry an `event_id`.

## Internal pipeline

```
append(24k PCM16) → resample 16k → Segmenter (EnergyVAD, commits at clause/utterance boundaries)
  → runtime-selected ASR (source text, auto-detected language) → input_transcript.delta
  → rolling voice reference (real ASR audio + transcript; cap at 10 s)
  → runtime-selected Translator (source → target) → output_transcript.delta (+done)
  → runtime-selected TTS / voice clone (pad reference copy to 3 s if needed) → output_audio.delta (+done)
```

Languages are **multilingual**: the source is detected by the ASR for each committed span. Clients may provide an ASR hint with `session.audio.input.transcription.language`, and the target defaults to `en` unless switched at runtime, within the supported range, by `session.audio.output.language`. Both session-level language fields are echoed in `session.updated`. See `ort/languages.py` for the support tables.

## Known limitations / future work

- Legacy commit strategy: **incremental committer (default)** — configured `commit_silence_ms` 700 ms commits, or continuous speech reaching `max_commit_ms` 8000 ms force-commits; `ORT_INCREMENTAL=0` switches back to whole-utterance mode. Text-level partial rewrite (overlap re-translation) is not implemented. The opt-in live path below has different explicit safety-budget semantics.
- Voice reference: each session starts with a 0 s reference buffer. Every non-empty ASR span appends its source audio and transcript; the newest 10 s of real source audio are retained. Padding is not stored in the buffer: when Qwen3-TTS is called, the backend creates a handoff copy and right-pads that copy to 3 s only if the real reference is shorter.
- Processing model: committed spans go into an ordered queue and are drained in order by the route task within the receive→handle turn. **No detached background task** — Starlette's `WebSocketTestSession` does not pump a background task's sends (verified to deadlock). MLX inference is serialized through a dedicated Apple Silicon thread; CUDA and CPU profiles use their backend-native execution paths. The event loop can still buffer incoming audio while awaiting inference.
- **Both WebSocket and WebRTC are implemented and verified**:
  - WS: `/v1/realtime/translations` (events + base64 PCM16).
  - WebRTC (aiortc): `POST /v1/realtime/translations/client_secrets` for a short-lived credential; `POST /v1/realtime/translations/calls` (body = SDP offer, returns SDP answer). The uplink Opus track is decoded/resampled into the same pipeline; events go over the `oai-events` DataChannel; translated speech goes out on an outbound Opus track.
  - A real WebRTC client's microphone track **stays continuous** (a pause = silence; the incremental committer commits at pauses); a track that ends by raising would trigger an RTCP BYE + SCTP Abort and tear down the connection.
- Pluggable backends (`ort/pipeline/backends`) are selected by `ORT_RUNTIME`:
  - `mlx`: Qwen3-ASR-1.7B (mlx-audio) → TranslateGemma-4b-it (mlx-lm) → Qwen3-TTS-12Hz-1.7B-Base (mlx-audio, reference-audio voice cloning). This is the verified Apple Silicon profile.
  - `cuda`: Qwen3-ASR-1.7B (`qwen-asr`, Transformers/Torch) → TranslateGemma-4b-it (Transformers/Torch) → Qwen3-TTS-12Hz-1.7B-Base (`qwen-tts`, reference-audio voice cloning).
  - `cpu`: Whisper/faster-whisper (CTranslate2 CPU) → TranslateGemma-4b-it (Transformers/Torch) → Supertonic-3 (ONNX).
- Optional ASR/TTS backends include `whisper`, `kokoro`, `supertonic`, `say`, `qwen3_tts`, and mock backends. MLX/Metal thread affinity still applies to MLX model init and inference in `ort/inference.py`.

## VoiceBridge live opt-in: committed-v1

`ws://127.0.0.1:8765/v1/realtime/translations?voicebridge=committed-v1`
retains every event name and the existing 24 kHz mono raw PCM16 payload. It is an
explicit local extension; clients without this query keep legacy behavior.
`session.created.session.voicebridge` negotiates version 1, input mode
`client_committed`, ASR mode `utterance_final`, actual model/worker/checkpoint
identities and static-reference fingerprint. The live client rejects a mismatch.

The client exclusively owns the 30 ms VAD and sends one immutable utterance as
normal append events followed by normal commit. The commit additionally includes
`voicebridge` = the recognition context: `session_id`, cancellation `epoch`,
monotonic `utterance_id`, `audio_start_sample`/`audio_end_sample` at 16 kHz,
`capture_start_at`, `speech_end_at`, `vad_commit_at`, and optional original
`native_sample_rate`, `native_start_sample`, `native_end_sample`,
`capture_adc_clock_start`. Windows system-wide perf_counter timestamps refer to
the same host; original ADC clock values are preserved separately. PCM duration
must match the sample range. Source/target must explicitly be zh/ja. The server
collects at most 31 s of wire PCM and performs no second VAD or rolling reference
update on this path.

Existing output transcript/audio events additionally carry `voicebridge.context`
and `stage_times`. Transcript phase is `final`; current Qwen recognition emits
no provisional/stable-prefix events. Future backend updates distinguish those
states internally, but provisional text cannot reach translation or synthesis.
PCM deltas include zero-based `chunk_sequence`, `sample_rate:24000`, `channels:1`,
`pcm_format:s16le`, decoded `samples` and `pcm_sha256`. Audio done includes
`chunk_count` and actual `tts_metrics`. One translated transcript completion
precedes audio; one audio completion closes each utterance. A WAV header is not
valid raw PCM. The client validates IDs/epoch/order/format/hash before playback.

The route uses a bounded incoming event queue of eight and an exclusive live
model lease, with receive/disconnect handling independent of inference. This is
opt-in; the legacy TestClient-compatible route keeps its existing dispatch.
Disconnect immediately invalidates sends; native work drains before the lease
is released. A drain deadline failure requires explicit server restart. Client
Stop/Pause uses connection closure and a new cancellation epoch, not a new wire
event. See [live lifecycle and hardware verification](README_WINDOWS_LIVE.md).

### STEP08.4 pronunciation metadata (additive)

When GPT-SoVITS is actually loaded, `session.created.session.metadata` includes
`pronunciation_lexicon={enabled,path,rule_count,sha256}` for the loaded adapter
snapshot. Committed-v1 also includes it under `session.voicebridge`.
TTS startup `ready.pronunciation_lexicon` and completed per-request
`tts_metrics.pronunciation_lexicon` carry the same path/count/SHA; per-request
`enabled` reflects that request's explicit normalization switch. A loaded
dictionary may remain described while a request is disabled. The existing
`pronunciation` audit retains exact original/spoken text and actually applied
rules; no ASR/translated-text edits or PCM/event-name changes occur.

Production default is the exact383-rule STEP08 dictionary, SHA
`1a9caeac95afc778c60e2054ae836f2cddcdb1d63bfa2b0f0195d44620b558ca`.
Enabling is user-approved with mixed linguistic evidence; it is not a claim
that383 readings passed human listening. Malformed required files fail before
worker startup; override/disable and explicit accepted-two rollback remain.

### Optional approved personal presets and playback policy

Explicit `ORT_VOICE_PRESET=PERSONAL_R06|PERSONAL_R17|PERSONAL_R18` selects a
hash-verified personal pair. `ORT_PROSODY_LEVEL` defaults to `ORIGINAL`;
`SUBTLE|NATURAL|EXPRESSIVE` are experimental, awaiting listening. No preset is
selected by default. Selected sessions advertise `metadata.voice_preset`, also
`voicebridge.tts_voice_preset` on committed-v1. Per-request TTS metrics freeze
the preset, level, sampling values and actual seed.

These three approved listening configurations store native-speed1.00 PCM. Their
preferred playback is browser `playbackRate=1.25`, `preservesPitch=true`, without
postprocessing. A capable client must acknowledge the exact advertised object:

```json
{"type":"session.update","session":{"voicebridge_playback":{"method":"browser_preserves_pitch","rate":1.25,"preserves_pitch":true,"native_speed_factor":1.0,"postprocessing":"none"}}}
```

Without a matching acknowledgement, processing rejects the utterance before
ASR/TTS. Event names and 24kHz mono PCM16 payloads do not change. Acknowledgement
is the client's assertion; the server cannot prove its physical playback rate.
The existing native CABLE live client explicitly rejects these optional presets:
its renderer has no equivalent browser algorithm. Legacy production sessions
without a preset retain existing behavior and require no acknowledgement.

### Native desktop release client

The new `voicebridge.hot_server` explicitly selects one of four approved
PERSONAL_R17_NATURAL/EXPRESSIVE or PERSONAL_R18_NATURAL/EXPRESSIVE aliases.
It advertises the reversible native-client playback policy `ffmpeg_atempo`,
rate1.25, preserves_pitch=true, native_speed_factor1.0,
postprocessing=reversible_per_utterance, browser_equivalence=NOT_VERIFIED.
The desktop client acknowledges this exact policy and uses one continuous
atempo filter per utterance; explicit speed adjustments are frozen per job.
Original worker TTS metrics retain the approved source/browser listening
configuration; actual postprocessing is separately recorded in the engine's
`processing` metrics. No claim of browser acoustic equivalence is implied.

All existing event names and24kHz monoPCM16 payloads remain unchanged. Routing
commands use a separate local controller pipe, not invented WebSocket events.
The old native live CLI retains its strict browser-policy rejection. DIRECT
bypasses the server entirely and is explicitly activated, never startup default.
