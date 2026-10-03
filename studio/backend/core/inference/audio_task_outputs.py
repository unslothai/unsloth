# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every WAV a ``/v1/tasks/run`` response carries, in order.

A task answers with raw audio, or JSON holding base64 WAVs: one ``audio``, or a list of
``named_audio_outputs`` (Stable Audio's batch variations, a separator's stems). When both are
present the top-level ``audio`` repeats the first named one, so it is not counted twice (S3).
Pure: no I/O and no runtime imports, so the route and the worker can share it.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import wave
from typing import Any

_EMPTY = "The audio runtime returned no audio for this request."


def _decode(value: Any) -> bytes | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        data = base64.b64decode(value.split(",", 1)[-1], validate = False)
    except (binascii.Error, ValueError):
        return None
    return data if data[:4] == b"RIFF" else None


def task_outputs(content_type: str, data: bytes) -> list[tuple[str, bytes]]:
    """``[(id, wav_bytes), ...]`` from a ``/v1/tasks/run`` reply; RuntimeError when it has none."""
    if (content_type or "").startswith("audio/") or data[:4] == b"RIFF":
        return [("audio", data)]
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("The audio runtime returned an unreadable response.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(_EMPTY)
    named = payload.get("named_audio_outputs")
    if isinstance(named, list) and named:
        outputs: list[tuple[str, bytes]] = []
        for index, item in enumerate(named):
            if not isinstance(item, dict):
                continue
            wav = _decode(item.get("audio"))
            if wav is not None:
                outputs.append((str(item.get("id") or f"audio_{index}"), wav))
        if outputs:
            return outputs
    wav = _decode(payload.get("audio"))
    if wav is not None:
        return [("audio", wav)]
    raise RuntimeError(_EMPTY)


def wav_header(wav_bytes: bytes) -> tuple[int, float]:
    """``(sample_rate, duration_s)`` read from the WAV itself; a reply may omit its rate."""
    with wave.open(io.BytesIO(wav_bytes)) as w:
        rate = int(w.getframerate())
        frames = int(w.getnframes())
    return rate, (round(frames / rate, 3) if rate else 0.0)
