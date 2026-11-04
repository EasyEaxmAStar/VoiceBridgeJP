"""OpenAI Realtime Translation wire-protocol constants and normalization.

PROTOCOL.md is the source of truth. The official docs are inconsistent about
event prefixes, so server output uses `session.`-prefixed translation events as
canonical names while input accepts unprefixed aliases.
"""

from __future__ import annotations

import secrets
from typing import Any

# --- Server -> client events emitted by this implementation -----------------
SERVER_SESSION_CREATED = "session.created"
SERVER_SESSION_UPDATED = "session.updated"
SERVER_INPUT_TRANSCRIPT_DELTA = "session.input_transcript.delta"
SERVER_OUTPUT_TRANSCRIPT_DELTA = "session.output_transcript.delta"
SERVER_OUTPUT_TRANSCRIPT_DONE = "session.output_transcript.done"
SERVER_OUTPUT_AUDIO_DELTA = "session.output_audio.delta"
SERVER_OUTPUT_AUDIO_DONE = "session.output_audio.done"
SERVER_ERROR = "error"

# --- Internally normalized client -> server events --------------------------
C_SESSION_UPDATE = "session.update"
C_INPUT_AUDIO_APPEND = "input_audio_buffer.append"
C_INPUT_AUDIO_COMMIT = "input_audio_buffer.commit"
C_INPUT_AUDIO_CLEAR = "input_audio_buffer.clear"

# Inbound aliases -> internal canonical types. Covers GA Realtime and translation docs.
_INBOUND_ALIASES: dict[str, str] = {
    "session.update": C_SESSION_UPDATE,
    "transcription_session.update": C_SESSION_UPDATE,
    "input_audio_buffer.append": C_INPUT_AUDIO_APPEND,
    "session.input_audio_buffer.append": C_INPUT_AUDIO_APPEND,
    "input_audio_buffer.commit": C_INPUT_AUDIO_COMMIT,
    "session.input_audio_buffer.commit": C_INPUT_AUDIO_COMMIT,
    "input_audio_buffer.clear": C_INPUT_AUDIO_CLEAR,
    "session.input_audio_buffer.clear": C_INPUT_AUDIO_CLEAR,
}


def normalize_inbound_type(raw_type: str) -> str | None:
    """Normalize inbound event type to the internal type; unknown types return None."""
    return _INBOUND_ALIASES.get(raw_type)


def new_event_id(prefix: str = "evt") -> str:
    return f"{prefix}_{secrets.token_hex(10)}"


def server_event(event_type: str, **fields: Any) -> dict[str, Any]:
    """Build a server event with an event_id."""
    return {"event_id": new_event_id(), "type": event_type, **fields}


def error_event(
    message: str, code: str = "invalid_request_error", event_id: str | None = None
) -> dict[str, Any]:
    err: dict[str, Any] = {"type": code, "message": message}
    if event_id is not None:
        err["event_id"] = event_id
    return server_event(SERVER_ERROR, error=err)
