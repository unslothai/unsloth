# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stems out of a saved separation answer, decoded in chunks from an mmap.

Not ``json.loads``: ~42 MB of base64 per stem for a 180 s track would be held three times over.
"""

from __future__ import annotations

import binascii
import json
import mmap
import re
import struct
from pathlib import Path
from typing import Any, Optional

# Keys whose string value can carry base64 audio, matching ``_audio_candidates`` in the backend.
_AUDIO_KEYS = ("audio", "wav", "data", "audio_base64", "b64_json")
_AUDIO_KEY_RE = re.compile(rb'(?<!\\)"(' + "|".join(_AUDIO_KEYS).encode() + rb')"\s*:\s*"')
_PLACEHOLDER = "\u0000span"
_ID_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_-]")
_B64_ALPHABET = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
# Everything else is dropped before decoding: JSON's ``\\/`` escape, line breaks, padding spaces.
_B64_DROP = bytes(b for b in range(256) if b not in _B64_ALPHABET)
NO_STEMS = "The audio runtime returned no stems."


class SeparationOutputError(RuntimeError):
    """The answer has no stem Studio can decode."""

    def __init__(self, detail: str = NO_STEMS):
        super().__init__(detail)


def _string_end(mm: mmap.mmap, start: int) -> int:
    """Index of the unescaped quote that closes the JSON string starting at ``start``."""
    pos = start
    while True:
        end = mm.find(b'"', pos)
        if end < 0:
            raise SeparationOutputError()
        slashes = 0
        while end - 1 - slashes >= start and mm[end - 1 - slashes] == 0x5C:
            slashes += 1
        if slashes % 2 == 0:
            return end
        pos = end + 1


def _audio_spans(mm: mmap.mmap) -> list[tuple[int, int]]:
    spans = []
    pos = 0
    while True:
        match = _AUDIO_KEY_RE.search(mm, pos)
        if match is None:
            return spans
        start = match.end()
        end = _string_end(mm, start)
        spans.append((start, end))
        pos = end + 1


def _redacted(mm: mmap.mmap, spans: list[tuple[int, int]]) -> Any:
    """The document with every audio string replaced by ``"\\u0000span<k>"``, parsed."""
    parts = []
    prev = 0
    for index, (start, end) in enumerate(spans):
        parts.append(mm[prev:start])
        parts.append(f"\\u0000span{index}".encode("ascii"))
        prev = end
    parts.append(mm[prev:])
    try:
        return json.loads(b"".join(parts).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SeparationOutputError() from exc


def _find_outputs(node: Any) -> Optional[list]:
    if isinstance(node, dict):
        outputs = node.get("named_audio_outputs")
        if isinstance(outputs, list):
            return outputs
        for value in node.values():
            found = _find_outputs(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_outputs(item)
            if found is not None:
                return found
    return None


def _span_index(item: dict) -> Optional[int]:
    for key in _AUDIO_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.startswith(_PLACEHOLDER):
            try:
                return int(value[len(_PLACEHOLDER) :])
            except ValueError:
                return None
    return None


def safe_stem_id(raw: Any, taken: set[str]) -> str:
    """A file-safe id (``[A-Za-z0-9_-]{1,64}``), unique among ``taken`` by a ``_2`` style suffix."""
    base = _ID_UNSAFE_RE.sub("_", str(raw or "").strip())[:64] or "stem"
    stem_id = base
    n = 2
    while stem_id in taken:
        suffix = f"_{n}"
        stem_id = base[: 64 - len(suffix)] + suffix
        n += 1
    taken.add(stem_id)
    return stem_id


def _decode_span(mm: mmap.mmap, start: int, end: int, dest: Path, chunk: int) -> None:
    head = mm[start : min(end, start + 128)]
    if head.startswith(b"data:"):
        comma = head.find(b",")
        if comma >= 0:
            start += comma + 1
    chunk = max(4, chunk - chunk % 4)
    carry = b""
    with open(dest, "wb") as out:
        pos = start
        while pos < end:
            stop = min(end, pos + chunk)
            raw = mm[pos:stop]
            text = raw.translate(None, _B64_DROP)
            del raw
            if carry:
                text = carry + text
            usable = len(text) - len(text) % 4
            if usable:
                out.write(binascii.a2b_base64(memoryview(text)[:usable]))
            carry = text[usable:]
            del text
            pos = stop
        if carry.strip(b"="):
            out.write(binascii.a2b_base64(carry + b"=" * (-len(carry) % 4)))


def riff_info(path: Path) -> dict[str, Any]:
    """``sample_rate``, ``channels``, ``frames`` from a RIFF/WAVE header; ValueError otherwise."""
    with open(path, "rb") as f:
        header = f.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError("not a RIFF/WAVE file")
        size = path.stat().st_size
        fmt = None
        while True:
            chunk_head = f.read(8)
            if len(chunk_head) < 8:
                raise ValueError("no data chunk")
            name, length = chunk_head[:4], struct.unpack("<I", chunk_head[4:])[0]
            if name == b"fmt ":
                body = f.read(length)
                if len(body) < 16:
                    raise ValueError("short fmt chunk")
                audio_format, channels, rate, _byte_rate, block_align, _bits = struct.unpack(
                    "<HHIIHH", body[:16]
                )
                if audio_format not in (1, 3, 0xFFFE) or not channels or not rate:
                    raise ValueError("unsupported WAV encoding")
                fmt = (channels, rate, block_align or 1)
                if length % 2:
                    f.seek(1, 1)
            elif name == b"data":
                if fmt is None:
                    raise ValueError("data before fmt")
                # A streamed writer leaves the size at 0 or 0xFFFFFFFF; the file length is the truth.
                available = size - f.tell()
                if length == 0 or length > available:
                    length = available
                channels, rate, block_align = fmt
                return {"sample_rate": rate, "channels": channels, "frames": length // block_align}
            else:
                f.seek(length + (length % 2), 1)


def extract_named_outputs(
    json_path: Path | str,
    out_dir: Path | str,
    *,
    chunk: int = 4 << 20,
) -> list[dict[str, Any]]:
    """Decode ``named_audio_outputs[*]`` into ``out_dir/<id>.wav``, all or none.

    Top-level ``audio`` is ignored: in a batch answer it repeats the first named output."""
    json_path = Path(json_path)
    out_dir = Path(out_dir)
    try:
        size = json_path.stat().st_size
    except OSError as exc:
        raise SeparationOutputError() from exc
    if size == 0:
        raise SeparationOutputError()
    written: list[Path] = []
    try:
        with open(json_path, "rb") as f, mmap.mmap(f.fileno(), 0, access = mmap.ACCESS_READ) as mm:
            spans = _audio_spans(mm)
            outputs = _find_outputs(_redacted(mm, spans))
            if not outputs:
                raise SeparationOutputError()
            taken: set[str] = set()
            results = []
            for item in outputs:
                if not isinstance(item, dict) or item.get("id") in (None, ""):
                    raise SeparationOutputError()
                index = _span_index(item)
                if index is None or index >= len(spans):
                    raise SeparationOutputError()
                stem_id = safe_stem_id(item["id"], taken)
                dest = out_dir / f"{stem_id}.wav"
                written.append(dest)
                start, end = spans[index]
                try:
                    _decode_span(mm, start, end, dest, chunk)
                    info = riff_info(dest)
                except (binascii.Error, ValueError) as exc:
                    raise SeparationOutputError() from exc
                results.append(
                    {
                        "id": stem_id,
                        "path": str(dest),
                        "sample_rate": info["sample_rate"],
                        "channels": info["channels"],
                        "duration_s": round(info["frames"] / info["sample_rate"], 3),
                    }
                )
            return results
    except BaseException:
        for path in written:
            try:
                path.unlink(missing_ok = True)
            except OSError:
                pass
        raise
