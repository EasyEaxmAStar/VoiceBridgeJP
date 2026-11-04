"""Windows audio device discovery and explicitly selected playback tests."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: int
    is_physical_microphone: bool
    is_cable_input_playback: bool
    is_cable_output_recording: bool


class AudioDeviceError(RuntimeError):
    """Actionable audio-device or playback error."""


def _cable_direction(name: str) -> tuple[bool, bool]:
    normalized = " ".join(name.casefold().replace("-", " ").split())
    has_cable = "cable" in normalized and ("vb audio" in normalized or "virtual cable" in normalized)
    if not has_cable:
        return False, False
    return "input" in normalized, "output" in normalized


def classify_device(index: int, info: dict[str, Any], host_api: str) -> AudioDevice:
    name = str(info.get("name", "Unknown audio device"))
    inputs = int(info.get("max_input_channels", 0))
    outputs = int(info.get("max_output_channels", 0))
    cable_in, cable_out = _cable_direction(name)
    normalized = name.casefold()
    generic_capture = any(
        token in normalized
        for token in (
            "sound mapper",
            "primary sound driver",
            "default capture",
            "capture driver",
            "捕获驱动程序",
        )
    )
    return AudioDevice(
        index=index,
        name=name,
        host_api=host_api,
        max_input_channels=inputs,
        max_output_channels=outputs,
        default_samplerate=int(float(info.get("default_samplerate", 0))),
        is_physical_microphone=inputs > 0 and not cable_out and not generic_capture,
        is_cable_input_playback=outputs > 0 and cable_in,
        is_cable_output_recording=inputs > 0 and cable_out,
    )


def enumerate_devices(
    *, query_devices: Callable[[], Sequence[dict[str, Any]]] | None = None,
    query_hostapis: Callable[[], Sequence[dict[str, Any]]] | None = None,
) -> list[AudioDevice]:
    """Enumerate PortAudio endpoints and preserve their native host API names."""
    if query_devices is None or query_hostapis is None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioDeviceError(
                "Audio support is missing. Install the project environment dependencies first."
            ) from exc
        query_devices = query_devices or sd.query_devices
        query_hostapis = query_hostapis or sd.query_hostapis
    try:
        hostapis = query_hostapis()
        devices = query_devices()
    except Exception as exc:
        raise AudioDeviceError(f"Could not enumerate Windows audio devices: {exc}") from exc

    result: list[AudioDevice] = []
    for index, info in enumerate(devices):
        api_index = int(info.get("hostapi", -1))
        api_name = str(hostapis[api_index]["name"]) if 0 <= api_index < len(hostapis) else "Unknown host API"
        result.append(classify_device(index, info, api_name))
    return result


def supported_sample_rates(
    device: AudioDevice,
    *,
    check_output_settings: Callable[..., Any] | None = None,
    candidates: Sequence[int] = (48000, 44100, 24000, 22050, 16000),
) -> list[int]:
    """Probe rates with PortAudio; defaults advertised rates are not guarantees."""
    channels = min(2, device.max_output_channels)
    if channels <= 0:
        return []
    if check_output_settings is None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioDeviceError("sounddevice is required to probe output sample rates") from exc
        check_output_settings = sd.check_output_settings
    supported: list[int] = []
    for rate in candidates:
        try:
            check_output_settings(device=device.index, channels=channels, dtype="float32", samplerate=rate)
        except Exception:
            continue
        supported.append(int(rate))
    return supported


def generate_test_tone(
    sample_rate: int, seconds: float, frequency: float, channels: int
):
    """Return a constant-frequency sine at the requested rate and channels."""
    import numpy as np

    frames = max(1, round(sample_rate * seconds))
    times = np.arange(frames, dtype=np.float64) / sample_rate
    mono = (0.12 * np.sin(2 * np.pi * frequency * times)).astype(np.float32)
    return np.repeat(mono[:, None], channels, axis=1)


def measure_dominant_frequency(samples: Any, sample_rate: int) -> float:
    """Estimate the strongest non-silent frequency in recorded loopback audio."""
    import numpy as np

    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        raise AudioDeviceError("The loopback recording contained no audio")
    threshold = max(float(np.max(np.abs(audio))) * 0.01, 1e-7)
    active = np.flatnonzero(np.abs(audio) >= threshold)
    if active.size:
        audio = audio[active[0] : active[-1] + 1]
    if audio.size < 16:
        raise AudioDeviceError("The loopback recording was too short to measure")
    audio = audio - np.mean(audio)
    windowed = audio * np.hanning(audio.size)
    fft_size = 1 << int(math.ceil(math.log2(max(4096, audio.size * 8))))
    spectrum = np.abs(np.fft.rfft(windowed, n=fft_size))
    frequencies = np.fft.rfftfreq(fft_size, d=1 / sample_rate)
    spectrum[0] = 0
    return float(frequencies[int(np.argmax(spectrum))])


def play_test_tone(
    device: AudioDevice,
    *,
    seconds: float = 0.5,
    frequency: float = 440.0,
    sample_rate: int | None = None,
    loopback_device: AudioDevice | None = None,
    check_output_settings: Callable[..., Any] | None = None,
    check_input_settings: Callable[..., Any] | None = None,
    output_stream_factory: Callable[..., Any] | None = None,
    playrec_function: Callable[..., Any] | None = None,
) -> tuple[int, float | None]:
    """Play a brief sine tone on exactly the selected output. Returns used rate."""
    if device.max_output_channels <= 0:
        raise AudioDeviceError(f"Selected device is not an output endpoint: {device.name}")
    if not math.isfinite(seconds) or seconds <= 0 or seconds > 3:
        raise AudioDeviceError("Tone duration must be greater than 0 and at most 3 seconds")
    if not math.isfinite(frequency) or frequency < 50 or frequency > 4000:
        raise AudioDeviceError("Tone frequency must be between 50 and 4000 Hz")
    if check_output_settings is None or output_stream_factory is None or (
        loopback_device is not None and (check_input_settings is None or playrec_function is None)
    ):
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioDeviceError("sounddevice is required for playback") from exc
        check_output_settings = check_output_settings or sd.check_output_settings
        check_input_settings = check_input_settings or sd.check_input_settings
        output_stream_factory = output_stream_factory or sd.OutputStream
        playrec_function = playrec_function or sd.playrec

    channels = min(2, device.max_output_channels)
    rate = int(sample_rate or device.default_samplerate)
    try:
        check_output_settings(device=device.index, channels=channels, dtype="float32", samplerate=rate)
    except Exception as exc:
        available = supported_sample_rates(device, check_output_settings=check_output_settings)
        options = ", ".join(map(str, available)) if available else "none detected"
        raise AudioDeviceError(
            f"Device {device.index} ({device.name}) does not support {rate} Hz {channels}-channel output; "
            f"probed rates: {options}. Choose a supported rate explicitly."
        ) from exc

    tone = generate_test_tone(rate, seconds, frequency, channels)
    measured_frequency: float | None = None
    try:
        if loopback_device is not None:
            if loopback_device.max_input_channels <= 0:
                raise AudioDeviceError(f"Loopback device is not a recording endpoint: {loopback_device.name}")
            check_input_settings(
                device=loopback_device.index,
                samplerate=rate,
                channels=min(2, loopback_device.max_input_channels),
                dtype="float32",
            )
            captured = playrec_function(
                tone,
                samplerate=rate,
                channels=channels,
                dtype="float32",
                device=(loopback_device.index, device.index),
                blocking=True,
            )
            measured_frequency = measure_dominant_frequency(captured, rate)
            if abs(measured_frequency - frequency) > max(5.0, frequency * 0.01):
                raise AudioDeviceError(
                    f"Sent {frequency:g} Hz to {device.name}, but {loopback_device.name} "
                    f"measured {measured_frequency:.1f} Hz. Check the selected endpoints and sample rate."
                )
        else:
            with output_stream_factory(
                device=device.index,
                samplerate=rate,
                channels=channels,
                dtype="float32",
            ) as stream:
                stream.write(tone)
    except Exception as exc:
        if isinstance(exc, AudioDeviceError):
            raise
        message = str(exc)
        if "permission" in message.casefold() or "access" in message.casefold():
            raise AudioDeviceError(f"Windows denied access to {device.name}: {message}") from exc
        raise AudioDeviceError(f"Playback failed on selected device {device.index} ({device.name}): {message}") from exc
    return rate, measured_frequency


def _role(device: AudioDevice) -> str:
    roles: list[str] = []
    if device.is_physical_microphone:
        roles.append("recording microphone")
    if device.is_cable_input_playback:
        roles.append("VB-CABLE playback target (CABLE Input)")
    if device.is_cable_output_recording:
        roles.append("meeting-app recording endpoint (CABLE Output)")
    if not roles:
        if device.max_input_channels:
            roles.append("recording input")
        if device.max_output_channels:
            roles.append("playback output")
    return ", ".join(roles) or "unavailable"


def _print_devices(devices: Sequence[AudioDevice]) -> None:
    print("Windows audio devices (select output index explicitly for a tone test):")
    for device in devices:
        print(
            f"[{device.index}] {device.name} | host API: {device.host_api} | "
            f"in: {device.max_input_channels}ch | out: {device.max_output_channels}ch | "
            f"default: {device.default_samplerate} Hz | role: {_role(device)}"
        )
    if not any(d.is_physical_microphone for d in devices):
        print("No physical recording microphone detected.")
    if not any(d.is_cable_input_playback for d in devices):
        print("VB-CABLE CABLE Input playback endpoint not detected; no output will be substituted.")
    if not any(d.is_cable_output_recording for d in devices):
        print("VB-CABLE CABLE Output recording endpoint not detected.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List and test Windows audio devices for VoiceBridge JP")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list Windows audio endpoints and their roles")
    test_parser = subparsers.add_parser("test", help="play a brief synthetic tone to one explicitly selected output")
    test_parser.add_argument("--device", type=int, required=True, help="exact playback device index from the list")
    test_parser.add_argument("--seconds", type=float, default=0.5, help="tone duration (0.01 to 3 seconds)")
    test_parser.add_argument("--frequency", type=float, default=440.0, help="tone frequency in Hz")
    test_parser.add_argument("--sample-rate", type=int, help="requested sample rate; must be accepted by device")
    args = parser.parse_args(argv)
    try:
        devices = enumerate_devices()
        if args.command == "list":
            _print_devices(devices)
            return 0
        selected = next((device for device in devices if device.index == args.device), None)
        if selected is None:
            raise AudioDeviceError(f"Device index {args.device} is unavailable or disconnected; run 'list' again")
        if selected.max_output_channels <= 0:
            raise AudioDeviceError(f"Device {args.device} is an input-only recording endpoint")
        loopback = None
        if selected.is_cable_input_playback:
            loopback = next(
                (
                    device for device in devices
                    if device.is_cable_output_recording and device.host_api == selected.host_api
                ),
                next((device for device in devices if device.is_cable_output_recording), None),
            )
        actual_rate, measured_frequency = play_test_tone(
            selected,
            seconds=args.seconds,
            frequency=args.frequency,
            sample_rate=args.sample_rate,
            loopback_device=loopback,
        )
        message = f"Sent {args.frequency:g} Hz sine for {args.seconds:g} s to [{selected.index}] {selected.name} at {actual_rate} Hz"
        if measured_frequency is not None:
            message += f"; CABLE Output loopback measured {measured_frequency:.1f} Hz"
        else:
            message += "; loopback was not measured"
        print(message + ".")
        return 0
    except AudioDeviceError as exc:
        print(f"Audio device error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
