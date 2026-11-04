"""Asynchronous microphone-to-local-WebSocket audio bridge.

Microphone samples only travel to the local translation server. Playback accepts
only server ``session.output_audio.delta`` events and requires a VB-CABLE CABLE
Input output endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import queue
import sys
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ort.audio import WIRE_SAMPLE_RATE, resample
from .audio_devices import AudioDevice, AudioDeviceError, enumerate_devices


class AudioBridgeError(RuntimeError):
    """Actionable bridge configuration, protocol, or device error."""


@dataclass(frozen=True)
class BridgeConfig:
    url: str
    input_device: int
    output_device: int
    input_sample_rate: int | None = None
    output_sample_rate: int | None = None
    input_queue_blocks: int = 32
    output_queue_chunks: int = 32
    shutdown_timeout_s: float = 3.0
    source_language: str = "zh"
    target_language: str = "ja"


def _json(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False, separators=(",", ":"))


def pcm24k_b64_to_float32(payload: str) -> np.ndarray:
    """Decode a strict wire-format base64(PCM16 mono @ 24 kHz) event chunk."""
    try:
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise AudioBridgeError("Server returned malformed base64 PCM audio") from exc
    if len(raw) == 0 or len(raw) % 2:
        raise AudioBridgeError("Server PCM audio must contain complete non-empty int16 samples")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def capture_block_to_wire(samples: np.ndarray, input_rate: int) -> str:
    """Convert one captured mono float block to base64(PCM16 mono @ 24 kHz)."""
    from ort.audio import float32_to_b64_pcm16

    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1, dtype=np.float32)
    mono = mono.reshape(-1)
    wire = resample(mono, input_rate, WIRE_SAMPLE_RATE)
    if wire.size == 0:
        return ""
    return float32_to_b64_pcm16(wire)


def prepare_playback(
    pcm24k: np.ndarray, output_rate: int, output_channels: int
) -> np.ndarray:
    """Resample server output and duplicate mono audio to the selected endpoint."""
    mono = resample(np.asarray(pcm24k, dtype=np.float32).reshape(-1), WIRE_SAMPLE_RATE, output_rate)
    if output_channels == 1:
        return mono.reshape(-1, 1)
    return np.repeat(mono[:, None], output_channels, axis=1)


class AudioBridge:
    """Run one native input stream, local WebSocket session, and cable output."""

    def __init__(
        self,
        config: BridgeConfig,
        *,
        device_provider: Callable[[], list[AudioDevice]] = enumerate_devices,
        connect_factory: Callable[[str], Any] | None = None,
        input_stream_factory: Callable[..., Any] | None = None,
        output_stream_factory: Callable[..., Any] | None = None,
        check_input_settings: Callable[..., Any] | None = None,
        check_output_settings: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.device_provider = device_provider
        self.connect_factory = connect_factory
        self.input_stream_factory = input_stream_factory
        self.output_stream_factory = output_stream_factory
        self.check_input_settings = check_input_settings
        self.check_output_settings = check_output_settings
        self.input_dropped_blocks = 0
        self.output_dropped_chunks = 0
        self.input_status_errors = 0
        self.last_input_status = ""

    def _resolve_devices(self) -> tuple[AudioDevice, AudioDevice]:
        try:
            devices = self.device_provider()
        except AudioDeviceError as exc:
            raise AudioBridgeError(str(exc)) from exc
        microphone = next((d for d in devices if d.index == self.config.input_device), None)
        cable = next((d for d in devices if d.index == self.config.output_device), None)
        if microphone is None or microphone.max_input_channels <= 0 or not microphone.is_physical_microphone:
            raise AudioBridgeError(f"Input device {self.config.input_device} is not an available physical microphone")
        if cable is None or not cable.is_cable_input_playback or cable.max_output_channels <= 0:
            raise AudioBridgeError(
                f"Output device {self.config.output_device} is not VB-CABLE CABLE Input; refusing audio playback"
            )
        return microphone, cable

    def _load_audio_api(self) -> None:
        if (
            self.connect_factory is not None
            and self.input_stream_factory is not None
            and self.output_stream_factory is not None
            and self.check_input_settings is not None
            and self.check_output_settings is not None
        ):
            return
        try:
            import sounddevice as sd
            import websockets
        except ImportError as exc:
            raise AudioBridgeError("Install the project audio and WebSocket dependencies before running the bridge") from exc
        self.connect_factory = self.connect_factory or websockets.connect
        self.input_stream_factory = self.input_stream_factory or sd.InputStream
        self.output_stream_factory = self.output_stream_factory or sd.OutputStream
        self.check_input_settings = self.check_input_settings or sd.check_input_settings
        self.check_output_settings = self.check_output_settings or sd.check_output_settings

    async def run(self, stop_event: asyncio.Event) -> None:
        """Start capture and playback until stop, disconnect, or a stream error."""
        microphone, cable = self._resolve_devices()
        self._load_audio_api()
        input_rate = int(self.config.input_sample_rate or microphone.default_samplerate)
        output_rate = int(self.config.output_sample_rate or cable.default_samplerate)
        input_channels = 1
        output_channels = min(2, cable.max_output_channels)
        assert self.check_input_settings and self.check_output_settings
        try:
            self.check_input_settings(
                device=microphone.index, samplerate=input_rate, channels=input_channels, dtype="float32"
            )
            self.check_output_settings(
                device=cable.index, samplerate=output_rate, channels=output_channels, dtype="float32"
            )
        except Exception as exc:
            raise AudioBridgeError(f"Selected microphone or CABLE Input does not support the requested audio format: {exc}") from exc

        raw_input: queue.Queue[np.ndarray] = queue.Queue(maxsize=max(1, self.config.input_queue_blocks))
        input_ready = asyncio.Event()
        input_closed = asyncio.Event()
        output_audio: asyncio.Queue[np.ndarray | None] = asyncio.Queue(
            maxsize=max(1, self.config.output_queue_chunks)
        )
        loop = asyncio.get_running_loop()
        commit_sent = False
        server_done = asyncio.Event()

        def capture_callback(indata: np.ndarray, _frames: int, _time_info: Any, status: Any) -> None:
            # Keep callback work bounded: copy a block and enqueue it. No resampling,
            # encoding, network calls, inference, or playback happens on this thread.
            if status:
                self.input_status_errors += 1
                self.last_input_status = str(status)
            block = np.array(indata, dtype=np.float32, copy=True)
            try:
                raw_input.put_nowait(block)
            except queue.Full:
                try:
                    raw_input.get_nowait()  # Drop stale input rather than growing latency.
                except queue.Empty:
                    pass
                try:
                    raw_input.put_nowait(block)
                    self.input_dropped_blocks += 1
                except queue.Full:
                    self.input_dropped_blocks += 1
            loop.call_soon_threadsafe(input_ready.set)

        capture = None
        try:
            async with self.connect_factory(self.config.url) as websocket:
                first = await self._receive_json(websocket)
                if first.get("type") != "session.created":
                    raise AudioBridgeError(f"Expected session.created, received {first.get('type')!r}")
                await websocket.send(
                    _json(
                        {
                            "type": "session.update",
                            "session": {
                                "audio": {
                                    "input": {"transcription": {"language": self.config.source_language}},
                                    "output": {"language": self.config.target_language},
                                }
                            },
                        }
                    )
                )
                capture = self.input_stream_factory(
                    device=microphone.index,
                    samplerate=input_rate,
                    channels=input_channels,
                    dtype="float32",
                    callback=capture_callback,
                )
                capture.start()

                async def send_audio() -> None:
                    nonlocal commit_sent
                    while not input_closed.is_set() or not raw_input.empty():
                        try:
                            block = raw_input.get_nowait()
                        except queue.Empty:
                            input_ready.clear()
                            if input_closed.is_set() and raw_input.empty():
                                break
                            try:
                                await asyncio.wait_for(input_ready.wait(), timeout=0.1)
                            except TimeoutError:
                                pass
                            continue
                        payload = capture_block_to_wire(block, input_rate)
                        if payload:
                            await websocket.send(_json({"type": "input_audio_buffer.append", "audio": payload}))
                    commit_sent = True
                    await websocket.send(_json({"type": "input_audio_buffer.commit"}))

                async def receive_events() -> None:
                    while True:
                        event = await self._receive_json(websocket)
                        event_type = event.get("type")
                        if event_type == "error":
                            detail = event.get("error", {})
                            raise AudioBridgeError(f"Translation server error: {detail.get('message', detail)}")
                        if event_type == "session.output_audio.delta":
                            encoded = event.get("delta")
                            if not isinstance(encoded, str):
                                raise AudioBridgeError("Server output_audio.delta is missing its base64 delta")
                            mono = pcm24k_b64_to_float32(encoded)
                            playback = prepare_playback(mono, output_rate, output_channels)
                            if output_audio.full():
                                try:
                                    output_audio.get_nowait()  # Keep the newest translated audio bounded.
                                    self.output_dropped_chunks += 1
                                except asyncio.QueueEmpty:
                                    pass
                            output_audio.put_nowait(playback)
                        elif event_type == "session.output_audio.done" and commit_sent:
                            server_done.set()

                async def play_audio() -> None:
                    assert self.output_stream_factory
                    with self.output_stream_factory(
                        device=cable.index,
                        samplerate=output_rate,
                        channels=output_channels,
                        dtype="float32",
                    ) as stream:
                        while True:
                            item = await output_audio.get()
                            if item is None:
                                return
                            await asyncio.to_thread(stream.write, item)

                receiver_task = asyncio.create_task(receive_events(), name="voicebridge-websocket-receiver")
                playback_task = asyncio.create_task(play_audio(), name="voicebridge-cable-playback")
                stop_task = asyncio.create_task(stop_event.wait(), name="voicebridge-stop")
                try:
                    done, _ = await asyncio.wait(
                        {stop_task, receiver_task, playback_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if stop_task not in done:
                        for task in done:
                            task.result()
                        raise AudioBridgeError("Audio bridge worker stopped unexpectedly")
                    await asyncio.to_thread(capture.stop)
                    capture.close()
                    capture = None
                    input_closed.set()
                    loop.call_soon_threadsafe(input_ready.set)
                    await send_audio()
                    try:
                        await asyncio.wait_for(server_done.wait(), timeout=self.config.shutdown_timeout_s)
                    except TimeoutError:
                        pass  # Silence/no VAD segment is a valid stop condition.
                    await output_audio.put(None)
                    await playback_task
                finally:
                    stop_task.cancel()
                    if not receiver_task.done():
                        receiver_task.cancel()
                    if not playback_task.done():
                        playback_task.cancel()
                    await asyncio.gather(stop_task, receiver_task, playback_task, return_exceptions=True)
        except AudioBridgeError:
            raise
        except Exception as exc:
            raise AudioBridgeError(f"Audio bridge failed: {exc}") from exc
        finally:
            if capture is not None:
                try:
                    await asyncio.to_thread(capture.stop)
                except Exception:
                    pass
                try:
                    capture.close()
                except Exception:
                    pass

    @staticmethod
    async def _receive_json(websocket: Any) -> dict[str, Any]:
        try:
            raw = await websocket.recv()
            event = json.loads(raw)
        except Exception as exc:
            raise AudioBridgeError(f"WebSocket receive failed or returned invalid JSON: {exc}") from exc
        if not isinstance(event, dict):
            raise AudioBridgeError("WebSocket event must be a JSON object")
        return event


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VoiceBridge JP local microphone-to-VB-CABLE bridge")
    sub = parser.add_subparsers(dest="command", required=True)
    from .live_cli import add_arguments
    add_arguments(sub)
    run = sub.add_parser("run", help="bridge microphone audio to a local translation server")
    smoke = sub.add_parser("smoke", help="run a timed bridge smoke test against mock backends")
    for command in (run, smoke):
        command.add_argument("--url", default="ws://127.0.0.1:8000/v1/realtime/translations")
        command.add_argument("--input-device", required=True, type=int, help="physical microphone index from audio_devices list")
        command.add_argument("--output-device", required=True, type=int, help="CABLE Input playback index from audio_devices list")
        command.add_argument("--input-rate", type=int)
        command.add_argument("--output-rate", type=int)
    run.add_argument("--duration", type=float, help="optional timed run; otherwise stop with Ctrl+C")
    smoke.add_argument("--duration", type=float, default=5.0)
    return parser


async def _run_cli(args: argparse.Namespace) -> None:
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        raise AudioBridgeError("Duration must be a positive number of seconds")
    stop_event = asyncio.Event()
    bridge = AudioBridge(
        BridgeConfig(
            url=args.url,
            input_device=args.input_device,
            output_device=args.output_device,
            input_sample_rate=args.input_rate,
            output_sample_rate=args.output_rate,
        )
    )
    bridge_task = asyncio.create_task(bridge.run(stop_event))
    try:
        if args.duration is None:
            await stop_event.wait()
        else:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=args.duration)
            except TimeoutError:
                stop_event.set()
        await bridge_task
    except asyncio.CancelledError:
        stop_event.set()
        await bridge_task
        raise


def main(argv: list[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    try:
        if args.command == "live":
            from .live_cli import run_live
            return asyncio.run(run_live(args))
        asyncio.run(_run_cli(args))
        return 0
    except KeyboardInterrupt:
        return 0
    except (AudioBridgeError,ValueError,OSError,RuntimeError) as exc:
        print(f"Audio bridge error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
