# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Control frames: a 4-byte big-endian length, then one UTF-8 JSON object. Pairing and the agent's
operation selection use these; once an RPC stream is opened the connection carries raw bytes only."""

from __future__ import annotations

import asyncio
import json
import struct
from typing import Any, Dict

MAX_FRAME = 64 * 1024
_HEADER = struct.Struct(">I")


class FrameError(ValueError):
    """A peer sent something that is not a well-formed frame."""


def encode(obj: Dict[str, Any]) -> bytes:
    body = json.dumps(obj, separators = (",", ":"), sort_keys = True).encode("utf-8")
    if len(body) > MAX_FRAME:
        raise FrameError(f"frame of {len(body)} bytes exceeds {MAX_FRAME}")
    return _HEADER.pack(len(body)) + body


def decode(body: bytes) -> Dict[str, Any]:
    try:
        obj = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise FrameError(f"not JSON: {exc}") from None
    if not isinstance(obj, dict):
        raise FrameError("a frame must be a JSON object")
    return obj


async def read_frame(reader: asyncio.StreamReader, timeout: float = 10.0) -> Dict[str, Any]:
    try:
        header = await asyncio.wait_for(reader.readexactly(_HEADER.size), timeout)
        (length,) = _HEADER.unpack(header)
        if length > MAX_FRAME:
            raise FrameError(f"frame of {length} bytes exceeds {MAX_FRAME}")
        body = await asyncio.wait_for(reader.readexactly(length), timeout)
    except asyncio.IncompleteReadError:
        raise FrameError("connection closed mid-frame") from None
    except asyncio.TimeoutError:
        raise FrameError("timed out waiting for a frame") from None
    return decode(body)


async def write_frame(writer: asyncio.StreamWriter, obj: Dict[str, Any]) -> None:
    writer.write(encode(obj))
    await writer.drain()


def expect(obj: Dict[str, Any], kind: str, **fields: type) -> Dict[str, Any]:
    """``obj`` if it is a ``kind`` frame whose listed fields have the listed types, else FrameError."""
    if obj.get("type") != kind:
        raise FrameError(f"expected {kind}, got {obj.get('type')!r}")
    for name, typ in fields.items():
        value = obj.get(name)
        # bool is an int subclass: an int field must not accept True.
        if not isinstance(value, typ) or (typ is int and isinstance(value, bool)):
            raise FrameError(f"{kind}.{name} must be {typ.__name__}")
    return obj
