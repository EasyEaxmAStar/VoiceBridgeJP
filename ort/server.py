"""FastAPI app for the /v1/realtime/translations WebSocket endpoint.

The model pipeline is cached as a process-level singleton and reused across
connections, avoiding model reloads per connection.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from . import inference
from .config import Config
from .pipeline.base import Pipeline, build_pipeline
from .session import TranslationSession

log = logging.getLogger("ort.server")


def create_app(config: Config | None = None, warmup: bool = False, pipeline_factory=None) -> FastAPI:
    cfg = config or Config.from_env()
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            pipeline = app.state.pipeline
            close = getattr(pipeline.tts, "close", None) if pipeline is not None else None
            if callable(close):
                await inference.run(close)

    app = FastAPI(title="open-realtime-translate", lifespan=lifespan)
    app.state.config = cfg
    app.state.pipeline = None  # Built lazily.
    app.state.pipeline_lock = asyncio.Lock()
    app.state.live_lock = asyncio.Lock()
    app.state.live_failure = None

    def _build() -> Pipeline:
        log.info(
            "建立 pipeline asr=%s translator=%s tts=%s",
            cfg.asr_backend,
            cfg.translator_backend,
            cfg.tts_backend,
        )
        return (pipeline_factory or build_pipeline)(cfg)

    def get_pipeline() -> Pipeline:
        """Synchronously get the pipeline for CLI / non-event-loop contexts."""
        if app.state.pipeline is None:
            # Model construction must share the same dedicated thread as inference.
            app.state.pipeline = inference.run_sync(_build)
        return app.state.pipeline

    async def ensure_pipeline() -> Pipeline:
        """Event-loop path: build on the dedicated inference thread without blocking."""
        if app.state.pipeline is None:
            async with app.state.pipeline_lock:
                if app.state.pipeline is None:
                    app.state.pipeline = await inference.run(_build)
        return app.state.pipeline

    app.state.get_pipeline = get_pipeline
    app.state.ensure_pipeline = ensure_pipeline

    if warmup:
        pipeline = get_pipeline()
        inference.run_sync(pipeline.warmup)

    # WebRTC transport peers for /v1/realtime/translations/{client_secrets,calls}.
    from .webrtc import add_webrtc_routes

    add_webrtc_routes(app, cfg, ensure_pipeline)

    @app.get("/health")
    async def health() -> dict[str, str]:  # noqa: ANN202
        result = {"status":"ok"}
        token = getattr(app.state,"voicebridge_instance",None)
        if token:
            result["voicebridge_instance"] = token
        return result

    @app.websocket("/v1/realtime/translations")
    async def realtime_translations(ws: WebSocket) -> None:  # noqa: ANN202
        await ws.accept()
        if ws.query_params.get("voicebridge") == "committed-v1":
            await committed_connection(ws)
            return
        model = ws.query_params.get("model", "gpt-realtime-translate")
        pipeline = await ensure_pipeline()

        async def send(event: dict) -> None:
            await ws.send_text(json.dumps(event))

        session = TranslationSession(send, pipeline, cfg, model)
        await session.start()
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    from . import protocol as P

                    await send(P.error_event("訊息非合法 JSON"))
                    continue
                if not isinstance(message, dict):
                    from . import protocol as P

                    await send(P.error_event("事件必須是 JSON 物件"))
                    continue
                await session.handle(message)
        except WebSocketDisconnect:
            # Translation sessions have no explicit finish event. Clients stream
            # continuously and then close; already-sent deltas are the result.
            pass
        finally:
            session.abort()

    async def committed_connection(ws):
        """Opt-in live path: bounded receive queue, one exclusive model lease.

        Legacy routes retain their synchronous TestClient-compatible driver.
        On disconnect native calls drain; all further sends are invalidated.
        """
        session = None
        consumer = None
        receive = None
        acquired = False
        incoming = asyncio.Queue(maxsize=8)
        try:
            if app.state.live_failure:
                raise RuntimeError(app.state.live_failure)
            await asyncio.wait_for(app.state.live_lock.acquire(),cfg.gptsovits_request_timeout+5)
            acquired = True
            pipeline = await ensure_pipeline()
            async def send(event):
                await ws.send_text(json.dumps(event))
            session = TranslationSession(send,pipeline,cfg,"gpt-realtime-translate",committed_input=True)
            await session.start()
            async def consume():
                while not session._aborted:
                    message = await incoming.get()
                    try:
                        await session.handle(message)
                    finally:
                        incoming.task_done()
            consumer = asyncio.create_task(consume(),name="voicebridge-committed-inference")
            while True:
                receive = asyncio.create_task(ws.receive_text())
                done,_ = await asyncio.wait({receive,consumer},return_when=asyncio.FIRST_COMPLETED)
                if consumer in done:
                    receive.cancel()
                    await asyncio.gather(receive,return_exceptions=True)
                    consumer.result()
                    break
                raw = receive.result()
                if len(raw) > 256*1024:
                    raise ValueError("Live input event exceeds 256 KiB")
                message = json.loads(raw)
                if not isinstance(message,dict):
                    raise ValueError("Live event must be an object")
                incoming.put_nowait(message)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            with suppress(Exception):
                await ws.send_text(json.dumps({"type":"error","event_id":"live_error",
                    "error":{"type":"server_error","message":str(exc)}}))
                await ws.close(code=1011)
        finally:
            if receive and not receive.done():
                receive.cancel()
                await asyncio.gather(receive,return_exceptions=True)
            if session:
                session.abort()
            if consumer:
                # Wait for a running native stage/TTS generator to drain before
                # releasing the shared-model lease. An idle queue wait cancels.
                if session and session._recognition_context is None:
                    consumer.cancel()
                else:
                    while not incoming.empty():
                        incoming.get_nowait()
                    try:
                        await asyncio.wait_for(asyncio.shield(consumer),cfg.gptsovits_request_timeout+5)
                    except asyncio.TimeoutError:
                        app.state.live_failure = "Previous native job did not drain; explicitly restart the server"
                        consumer.cancel()
                await asyncio.gather(consumer,return_exceptions=True)
            if acquired:
                app.state.live_lock.release()

    return app


app = create_app()
