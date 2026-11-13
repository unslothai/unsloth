# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Cross-rank values via the c10d store: JSON, never pickle (TCPStore is unauthenticated).
Non-plain values arrive as None."""

import json
import os


def _plain(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, (list, tuple)):
        items = [_plain(v) for v in value]
        return {"t": items} if isinstance(value, tuple) else items
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {"d": {k: _plain(v) for k, v in value.items()}}
    return None


def _unplain(value):
    if isinstance(value, list):
        return [_unplain(v) for v in value]
    if isinstance(value, dict):
        if isinstance(value.get("t"), list):
            return tuple(_unplain(v) for v in value["t"])
        inner = value.get("d")
        return {k: _unplain(v) for k, v in inner.items()} if isinstance(inner, dict) else None
    return value


def encode(value) -> bytes:
    return json.dumps(_plain(value)).encode("utf-8")


def decode(data):
    try:
        return _unplain(json.loads(bytes(data).decode("utf-8")))
    except UnicodeDecodeError as exc:
        raise ValueError(str(exc)) from None
