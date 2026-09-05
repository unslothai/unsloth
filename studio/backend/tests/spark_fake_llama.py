# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A fake llama-server for the two-Spark router tests: asyncio, loopback, SSE.

Speaks just enough HTTP/1.1 for ``httpx``: one request per connection, ``/health``,
``/slots``, ``/props`` and a streaming ``/v1/chat/completions`` that emits ``chunks``
SSE frames tagged with the server's name, so a test can tell which backend served a
request and in what order the frames arrived. ``die_after`` aborts the socket after
that many frames (a process dying mid-stream); ``hold`` parks every generation on an
event so admission and queueing can be observed; ``health_ok`` flips ``/health``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple


class FakeLlama:
    def __init__(
        self,
        name: str,
        *,
        chunks: int = 4,
        delay: float = 0.005,
        die_after: Optional[int] = None,
        hold: Optional[asyncio.Event] = None,
        health_ok: bool = True,
    ):
        self.name = name
        self.chunks = chunks
        self.delay = delay
        self.die_after = die_after
        self.hold = hold
        self.health_ok = health_ok
        self.served: List[Tuple[str, Dict[str, Any], Dict[str, str]]] = []
        self.in_flight = 0
        self.port: Optional[int] = None
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self, port: int = 0) -> "FakeLlama":
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", port)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
            try:
                await asyncio.wait_for(server.wait_closed(), timeout = 2.0)
            except (asyncio.TimeoutError, Exception):
                pass

    @property
    def generation_count(self) -> int:
        return sum(1 for path, _b, _h in self.served if path.startswith("/v1/chat/completions"))

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin-1").split("\r\n")
            method, path, _version = lines[0].split(" ", 2)
            headers: Dict[str, str] = {}
            for line in lines[1:]:
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()
            body_bytes = b""
            length = int(headers.get("content-length", "0") or 0)
            if length:
                body_bytes = await reader.readexactly(length)
            body: Dict[str, Any] = {}
            if body_bytes:
                try:
                    body = json.loads(body_bytes)
                except ValueError:
                    body = {"_raw": body_bytes.decode("utf-8", "replace")}
            self.served.append((path, body, headers))
            if path == "/health":
                await self._json(
                    writer,
                    200 if self.health_ok else 503,
                    {"status": "ok" if self.health_ok else "loading"},
                )
            elif path == "/slots":
                await self._json(writer, 200, [{"id": 0, "is_processing": self.in_flight > 0}])
            elif path == "/props":
                await self._json(writer, 200, {"served_by": self.name})
            elif method == "POST" and path.startswith("/v1/chat/completions"):
                await self._stream(writer)
            else:
                await self._json(writer, 404, {"error": {"message": "not found"}})
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _json(self, writer: asyncio.StreamWriter, status: int, payload: Any) -> None:
        data = json.dumps(payload).encode("utf-8")
        writer.write(
            (
                f"HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n"
            ).encode("latin-1")
            + data
        )
        await writer.drain()

    async def _stream(self, writer: asyncio.StreamWriter) -> None:
        self.in_flight += 1
        try:
            if self.hold is not None:
                await self.hold.wait()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()
            for index in range(self.chunks):
                if self.die_after is not None and index == self.die_after:
                    writer.transport.abort()
                    return
                frame = (
                    "data: "
                    + json.dumps({"choices": [{"delta": {"content": f"{self.name}-{index}"}}]})
                    + "\n\n"
                )
                await self._chunk(writer, frame.encode("utf-8"))
                await asyncio.sleep(self.delay)
            await self._chunk(writer, b"data: [DONE]\n\n")
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        finally:
            self.in_flight -= 1

    @staticmethod
    async def _chunk(writer: asyncio.StreamWriter, data: bytes) -> None:
        writer.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")
        await writer.drain()


def sse_contents(text: str) -> List[str]:
    """The ``content`` of every data frame in an SSE body, in order; errors as ``error:``."""
    out: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            out.append("[DONE]")
            continue
        try:
            data = json.loads(payload)
        except ValueError:
            continue
        if "error" in data:
            out.append("error:" + str(data["error"].get("message", "")))
            continue
        for choice in data.get("choices", []):
            content = choice.get("delta", {}).get("content")
            if content:
                out.append(content)
    return out
