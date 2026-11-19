# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import http.client
import json
import socket
import threading
from typing import Iterator, Optional
from urllib.parse import urlencode

_LIVE_PATH = "/v1/audio/transcriptions/live"
_CONNECT_TIMEOUT_SECONDS = 5.0
# The runtime drops a body that goes quiet for 30 s.
_IO_TIMEOUT_SECONDS = 45.0
_ERROR_BODY_BYTES = 4096


class LiveTranscriptError(RuntimeError):
    """The runtime refused the session or ended it without a transcript."""


def family_streams(family: Optional[str]) -> bool:
    from core.inference.audio_cpp_models import runtime_spec

    spec = runtime_spec(family or "")
    if not spec:
        return False
    return "asr" in (spec.get("tasks") or ()) and "streaming" in (spec.get("modes") or ())


class LiveTranscript:
    """One utterance per connection. feed() and finish() on one thread; events() on another."""

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
            # Raw socket: http.client drops the stream after the headers.
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
        if not pcm:
            return
        self._send(b"%x\r\n" % len(pcm) + pcm + b"\r\n")

    def finish(self) -> None:
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
        """Ends now. Call finish() first, or the runtime discards the audio."""
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
