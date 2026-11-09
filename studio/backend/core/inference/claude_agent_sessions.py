# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One saved Claude Code session per Studio chat, and the fingerprints that tell what it has seen."""

import contextlib
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from core.inference.claude_agent import _THINK, _sdk, _text, image_url, workdir
from core.inference.mcp_images import (
    DETACHED_IMAGE_TURN_TEXT,
    IMAGE_TURN_TEXT,
    is_synthetic_image_turn,
)

_SPACE = re.compile(r"\s+")
# Studio drops the oldest turns to fit; at least this many must still line up to resume.
_MIN_TAIL = 2
_busy: set[str] = set()


def claim(thread_id: str) -> bool:
    if thread_id in _busy:
        return False
    _busy.add(thread_id)
    return True


def release(thread_id: str) -> None:
    _busy.discard(thread_id)


def _path(thread_id: str) -> Path:
    return workdir() / "sessions" / f"{hashlib.sha256(thread_id.encode()).hexdigest()[:40]}.json"


def load(thread_id: str) -> Optional[dict[str, Any]]:
    try:
        record = json.loads(_path(thread_id).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    checkpoints = record.get("checkpoints") if isinstance(record, dict) else None
    if not isinstance(record.get("claude_session_id"), str) or not isinstance(checkpoints, list):
        return None
    if not all(isinstance(c, dict) and isinstance(c.get("hash"), str) for c in checkpoints):
        return None
    return record


def save(thread_id: str, record: dict[str, Any]) -> None:
    path = _path(thread_id)
    path.parent.mkdir(mode = 0o700, parents = True, exist_ok = True)
    fd, tmp = tempfile.mkstemp(dir = path.parent, prefix = ".record-")
    try:
        with os.fdopen(fd, "w", encoding = "utf-8") as handle:
            json.dump(record, handle)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok = True)
        raise


def forget(thread_id: str) -> None:
    record = load(thread_id)
    _path(thread_id).unlink(missing_ok = True)
    sdk = _sdk()
    if record is not None and sdk is not None:
        with contextlib.suppress(Exception):
            sdk.delete_session(record["claude_session_id"], directory = str(workdir()))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys = True).encode()).hexdigest()


def visible(text: str) -> str:
    return _SPACE.sub("", _THINK.sub("", text))


def _user_parts(message: dict[str, Any]) -> list[str]:
    content = message.get("content")
    parts = []
    for part in content if isinstance(content, list) else [{"type": "text", "text": content}]:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text") if isinstance(part.get("text"), str) else ""
            # Tool images Studio folded into the next user turn are not the user's.
            if text.startswith((IMAGE_TURN_TEXT, DETACHED_IMAGE_TURN_TEXT)):
                break
            if visible(text):
                parts.append(visible(text))
        elif (url := image_url(part)) is not None:
            parts.append("image:" + hashlib.sha256(url.encode()).hexdigest())
    return parts


def reply_hash(text: str, tools: list[str]) -> str:
    return _digest(["assistant", visible(text), tools])


def user_hash(message: dict[str, Any]) -> str:
    return _digest(["user", _user_parts(message)])


def groups(turns: list[dict[str, Any]]) -> list[tuple[str, int, str]]:
    # Each user message, then everything up to the next one as one reply. Tool results and
    # whitespace are left out: Studio sends both back in its own shape, and they cannot be edited.
    spans: list[list[Any]] = []
    for index, message in enumerate(turns):
        if message.get("role") == "user" and not is_synthetic_image_turn(message):
            spans.append(["user", index, [message]])
        elif spans and spans[-1][0] == "assistant":
            spans[-1][2].append(message)
        else:
            spans.append(["assistant", index, [message]])
    out = []
    for role, start, members in spans:
        if role == "user":
            out.append((role, start, user_hash(members[0])))
            continue
        text = "".join(_text(m.get("content")) for m in members if m.get("role") == "assistant")
        tools = [
            (call.get("function") or {}).get("name") or ""
            for m in members
            for call in m.get("tool_calls") or []
            if isinstance(call, dict)
        ]
        out.append((role, start, reply_hash(text, tools)))
    return out


def decide(record: Optional[dict[str, Any]], hashes: list[str]) -> tuple[str, int, int]:
    # (resume | fork | fresh, saved checkpoints kept, first group Claude has not seen)
    saved = [c["hash"] for c in (record or {}).get("checkpoints") or []]
    if not saved or not hashes:
        return "fresh", 0, 0
    matched = 0
    while matched < min(len(saved), len(hashes)) and saved[matched] == hashes[matched]:
        matched += 1
    if matched == len(saved):
        return "resume", matched, matched
    if matched:
        # Edited, regenerated or deleted: branch after the last reply both still share.
        for keep in range(matched, 0, -1):
            if record["checkpoints"][keep - 1].get("uuid"):
                return "fork", keep, keep
        return "fresh", 0, 0
    # Studio dropped the oldest turns to fit; Claude's session still has them.
    for start in range(len(hashes)):
        for at in (i for i in range(len(saved)) if saved[i] == hashes[start]):
            length = len(saved) - at
            if length >= _MIN_TAIL and hashes[start : start + length] == saved[at:]:
                return "resume", len(saved), start + length
    return "fresh", 0, 0
