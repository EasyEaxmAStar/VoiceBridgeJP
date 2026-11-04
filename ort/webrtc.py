"""WebRTC transport peer built with aiortc.

OpenAI Realtime Translation WebRTC flow:
  POST /v1/realtime/translations/client_secrets -> obtain short-lived credential
  POST /v1/realtime/translations/calls          -> send SDP offer, receive SDP answer

Media: the client microphone sends an Opus track upstream, which is decoded,
resampled, and fed into the existing TranslationSession. Events use the
"oai-events" DataChannel; translated speech is sent as an outbound Opus track
instead of base64 audio events.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from fractions import Fraction
from typing import Any, Awaitable, Callable

import av
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from .audio import resample
from .config import Config
from .languages import DEFAULT_TARGET_LANGUAGE
from .session import TranslationSession

log = logging.getLogger("ort.webrtc")

_OUT_RATE = 48_000  # Standard Opus sample rate.
_FRAME = 960  # 20ms @ 48kHz
_FRAME_S = _FRAME / _OUT_RATE


class TranslatedAudioTrack(MediaStreamTrack):
    """Outbound audio track that sends translated speech in real-time 20 ms frames."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        # Do not use self._lock: aiortc's MediaStreamTrack base class already uses
        # that attribute internally. push/recv do not run truly concurrently on the
        # same event loop when the critical sections do not await, so no lock is needed.
        self._buf = np.zeros(0, dtype=np.float32)
        self._samples = 0
        self._t0: float | None = None

    async def push(self, samples: np.ndarray, sr: int) -> None:
        data = resample(samples.astype(np.float32), sr, _OUT_RATE)
        self._buf = np.concatenate([self._buf, data])

    async def recv(self) -> av.AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError
        if self._t0 is None:
            self._t0 = time.perf_counter()
        idx = self._samples // _FRAME
        target = self._t0 + idx * _FRAME_S
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

        if self._buf.size >= _FRAME:
            chunk = self._buf[:_FRAME]
            self._buf = self._buf[_FRAME:]
        else:
            chunk = np.zeros(_FRAME, dtype=np.float32)
            self._buf = np.zeros(0, dtype=np.float32)

        pcm = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
        frame.sample_rate = _OUT_RATE
        frame.pts = self._samples
        frame.time_base = Fraction(1, _OUT_RATE)
        self._samples += _FRAME
        return frame


def _make_event_sender(get_channel: Callable[[], Any], pending: list[dict]):
    async def send(event: dict) -> None:
        ch = get_channel()
        state = None if ch is None else ch.readyState
        if ch is not None and state == "open":
            ch.send(json.dumps(event))
            log.debug("event→dc %s", event.get("type"))
        else:
            pending.append(event)  # Channel not open yet; buffer until open.
            log.debug("event 緩衝 %s (ch_state=%s)", event.get("type"), state)

    return send


async def _consume_incoming(
    track: MediaStreamTrack, session: TranslationSession
) -> None:
    """Consume the inbound audio track: decode Opus, resample to 24 kHz mono, feed pipeline."""
    resampler = av.AudioResampler(format="s16", layout="mono", rate=24_000)
    frames = 0
    log.info("incoming audio track 開始消費")
    try:
        while True:
            frame = await track.recv()
            frames += 1
            for r in resampler.resample(frame):
                arr = r.to_ndarray().reshape(-1).astype(np.float32) / 32768.0
                if arr.size:
                    await session.feed_audio(arr)
    except (MediaStreamError, asyncio.CancelledError):
        pass
    except Exception:  # noqa: BLE001
        log.exception("incoming audio 消費失敗")
    finally:
        log.info("incoming audio 結束（frames=%d）→ flush", frames)
        try:
            await session.commit()
        except Exception:  # noqa: BLE001
            log.exception("commit 失敗")


def add_webrtc_routes(
    app: FastAPI,
    cfg: Config,
    ensure_pipeline: Callable[[], Awaitable[Any]],
) -> None:
    pcs: set[RTCPeerConnection] = set()
    app.state.webrtc_pcs = pcs

    @app.post("/v1/realtime/translations/client_secrets")
    async def client_secrets() -> JSONResponse:  # noqa: ANN202
        return JSONResponse(
            {
                "value": "ek_" + secrets.token_hex(16),
                "expires_at": int(time.time()) + 600,
                "session": {
                    "object": "realtime.translation_session",
                    "model": "gpt-realtime-translate",
                    "audio": {
                        "input": {"transcription": {"language": None}},
                        "output": {"language": DEFAULT_TARGET_LANGUAGE},
                    },
                },
            }
        )

    @app.post("/v1/realtime/translations/calls")
    async def calls(request: Request) -> Response:  # noqa: ANN202
        raw = (await request.body()).decode("utf-8", "replace").strip()
        ctype = request.headers.get("content-type", "")
        if "application/json" in ctype:
            payload = json.loads(raw)
            offer_sdp = payload.get("sdp", "")
        else:  # application/sdp, the OpenAI default.
            offer_sdp = raw
        if not offer_sdp:
            return Response("缺少 SDP offer", status_code=400)

        pipeline = await ensure_pipeline()
        model = request.query_params.get("model", "gpt-realtime-translate")

        pc = RTCPeerConnection()
        pcs.add(pc)
        out_track = TranslatedAudioTrack()
        pc.addTrack(out_track)

        channel_box: dict[str, Any] = {"ch": None}
        pending: list[dict] = []
        send = _make_event_sender(lambda: channel_box["ch"], pending)
        session = TranslationSession(
            send, pipeline, cfg, model, audio_sink=out_track.push
        )

        async def flush_pending() -> None:
            ch = channel_box["ch"]
            if ch is None or ch.readyState != "open":
                return
            for ev in pending:
                ch.send(json.dumps(ev))
            pending.clear()

        @pc.on("datachannel")
        def on_datachannel(channel) -> None:  # noqa: ANN001
            log.info(
                "on datachannel label=%s state=%s", channel.label, channel.readyState
            )
            channel_box["ch"] = channel

            @channel.on("open")
            def _on_open() -> None:  # noqa: ANN202
                log.info("server dc OPEN")
                asyncio.ensure_future(flush_pending())

            @channel.on("close")
            def _on_close() -> None:  # noqa: ANN202
                log.info("server dc CLOSE")

            if (
                channel.readyState == "open"
            ):  # May already be open when the handler runs.
                asyncio.ensure_future(flush_pending())

            @channel.on("message")
            def _on_message(msg) -> None:  # noqa: ANN001
                # Client events such as session.update use the same DataChannel.
                try:
                    data = json.loads(msg)
                except Exception:  # noqa: BLE001
                    return
                if isinstance(data, dict):
                    asyncio.ensure_future(session.handle(data))

        @pc.on("track")
        def on_track(track) -> None:  # noqa: ANN001
            log.info("收到遠端 track kind=%s", track.kind)
            if track.kind == "audio":
                asyncio.ensure_future(_consume_incoming(track, session))

        @pc.on("connectionstatechange")
        async def on_state() -> None:  # noqa: ANN202
            log.info("server pc connection=%s", pc.connectionState)
            # Release only on failed/closed. disconnected can be temporary.
            if pc.connectionState in ("failed", "closed"):
                pcs.discard(pc)
                await pc.close()

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer_sdp, type="offer")
        )
        await session.start()  # Send session.created, buffered until channel open.
        answer = await pc.createAnswer()
        await pc.setLocalDescription(
            answer
        )  # aiortc is non-trickle; ICE completes first.

        return Response(content=pc.localDescription.sdp, media_type="application/sdp")
