# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Transcription of speech while it is being spoken.

audio.cpp takes raw PCM on a chunked request body and answers on the same connection with the words
it has heard so far (POST /v1/audio/transcriptions/live, for a model its server runs in streaming
mode). One LiveTranscript is one such request: feed() it audio as it is captured, finish() when the
speaker stops, and read events() for the text.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
from typing import Iterator, Optional
from urllib.parse import urlencode

_LIVE_PATH = "/v1/audio/transcriptions/live"
_CONNECT_TIMEOUT_SECONDS = 5.0
# The runtime gives up on a body that goes quiet for 30 s; nothing useful arrives after that.
_IO_TIMEOUT_SECONDS = 45.0
_ERROR_BODY_BYTES = 4096


class LiveTranscriptError(RuntimeError):
    """The runtime refused the session or ended it without a transcript."""


def family_streams(family: Optional[str]) -> bool:
    """Whether the installed runtime transcribes this family live: its own spec says so."""
    from core.inference.audio_cpp_models import runtime_spec

    spec = runtime_spec(family or "")
    if not spec:
        return False
    return "asr" in (spec.get("tasks") or ()) and "streaming" in (spec.get("modes") or ())


class LiveTranscript:
    """One utterance on one connection. feed() and finish() belong to one thread, events() to another."""

    def __init__(
        self,
        port: int,
        model_id: str,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> None:
        query = {
            "model": model_id,
            "sample_rate": str(int(sample_rate)),
            "channels": "1",
            "sample_format": "s16le",
        }
        if language:
            query["language"] = language
        self._closed = threading.Event()
        self._finished = False
        self._send_lock = threading.Lock()
        try:
            # A raw socket, not http.client: its connection object drops the socket once the
            # reply's headers are read, and the audio still has to follow them.
            self._sock = socket.create_connection(
                ("127.0.0.1", port), timeout = _CONNECT_TIMEOUT_SECONDS
            )
            self._sock.settimeout(_IO_TIMEOUT_SECONDS)
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock.sendall(
                (
                    f"POST {_LIVE_PATH}?{urlencode(query)} HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{port}\r\n"
                    "Content-Type: application/octet-stream\r\n"
                    "Transfer-Encoding: chunked\r\n\r\n"
                ).encode("ascii")
            )
        except OSError as exc:
            raise LiveTranscriptError("The audio runtime did not take a live session.") from exc

    def feed(self, pcm: bytes) -> None:
        """Mono 16-bit little-endian samples at the session's rate, in the order they were heard."""
        if not pcm:
            return
        self._send(b"%x\r\n" % len(pcm) + pcm + b"\r\n")

    def finish(self) -> None:
        """The speaker stopped: the runtime transcribes what is left and ends the reply."""
        with self._send_lock:
            if self._finished:
                return
            self._finished = True
        self._send(b"0\r\n\r\n", ending = True)

    def _send(self, data: bytes, ending: bool = False) -> None:
        with self._send_lock:
            if self._closed.is_set() or (self._finished and not ending):
                raise LiveTranscriptError("The live session has ended.")
            try:
                self._sock.sendall(data)
            except OSError as exc:
                raise LiveTranscriptError("The audio runtime closed the live session.") from exc

    def events(self) -> Iterator[dict]:
        """{"type": "delta", "text": new words}, then one {"type": "done", "text": all of it}."""
        response = http.client.HTTPResponse(self._sock)
        try:
            response.begin()
            if response.status != 200:
                detail = response.read(_ERROR_BODY_BYTES).decode("utf-8", "replace").strip()
                raise LiveTranscriptError(
                    _runtime_message(detail) or f"The audio runtime answered {response.status}."
                )
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                kind = event.get("type") if isinstance(event, dict) else None
                if kind == "transcript.text.delta":
                    text = str(event.get("delta") or "")
                    if text:
                        yield {"type": "delta", "text": text}
                elif kind == "transcript.text.done":
                    yield {"type": "done", "text": str(event.get("text") or "")}
                elif kind == "error":
                    raise LiveTranscriptError(
                        _runtime_message(data) or "The audio runtime could not transcribe this."
                    )
            if not self._closed.is_set():
                raise LiveTranscriptError("The audio runtime ended the session early.")
        except (OSError, http.client.HTTPException) as exc:
            if self._closed.is_set():
                return
            raise LiveTranscriptError("The audio runtime closed the live session.") from exc
        finally:
            response.close()

    def close(self) -> None:
        """Ends the session now. Without finish() first the runtime discards what it heard."""
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()


def _runtime_message(body: str) -> str:
    try:
        error = json.loads(body).get("error")
    except (ValueError, AttributeError):
        return ""
    if isinstance(error, dict):
        return str(error.get("message") or "")
    return str(error or "")
