# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Codex text: message events or completed items, not response_item (injected context)."""

from __future__ import annotations

import base64
import binascii
import json
import re
from itertools import islice
from pathlib import Path

from core.external_import import (
    Source,
    SourceProject,
    Transcript,
    clean_text,
    display_name,
    file_times_ms,
    first_user_text,
    iso_ms,
    read_jsonl,
    session_id_of,
    stable_id,
    title_from,
    tool_call,
)
from utils.device_sync_settings import sniff_image_type

_UNKNOWN_FOLDER = "unknown-folder"


_META: dict[Path, dict] = {}
# {home: {rollout id: path}} for fork bases, built once per list_projects run.
_BY_ROLLOUT_ID: dict[Path, dict[str, Path]] = {}


def _meta(path: Path) -> dict:
    """The rollout's session_meta payload ({} when the file cannot say), cached per path."""
    if path not in _META:
        try:
            first = next(islice(read_jsonl(path), 1), {})
        except (OSError, ImportError):
            first = {}
        payload = first.get("payload") if first.get("type") == "session_meta" else None
        payload = payload if isinstance(payload, dict) else {}
        # Only what is read below: the payload also carries the full base instructions.
        _META[path] = {
            k: payload[k]
            for k in ("id", "cwd", "source", "history_base", "history_mode")
            if k in payload
        }
    return _META[path]


def _rollouts(home: Path) -> list[Path]:
    # archived_sessions holds threads archived in Codex; a rollout in both counts once.
    found: dict[str, Path] = {}
    for root in (home / "archived_sessions", home / "sessions"):
        if root.is_dir():
            for path in root.rglob("rollout-*.jsonl*"):
                if path.name.endswith((".jsonl", ".jsonl.zst")):
                    found[session_id_of(path)] = path
    return sorted(found.values(), key = lambda p: p.name)


def _rollout_id(path: Path) -> str:
    """``rollout-<YYYY-MM-DDThh-mm-ss>-<thread id>[_<rollout id>]``: a revert names its own rollout."""
    return session_id_of(path)[len("rollout-YYYY-MM-DDThh-mm-ss-") :].rsplit("_", 1)[-1]


def _rollout_path(home: Path, rollout: str) -> Path | None:
    """The first rollout (by name) with this id; rescans if it is new or moved since the last scan."""
    found = _BY_ROLLOUT_ID.get(home)
    path = found.get(rollout) if found is not None else None
    if path is None or not path.is_file():
        found = {}
        for candidate in _rollouts(home):
            found.setdefault(_rollout_id(candidate), candidate)
        _BY_ROLLOUT_ID[home] = found
        path = found.get(rollout)
    return path


def _thread_id(path: Path) -> str:
    return str(_meta(path).get("id") or session_id_of(path))


def _by_thread(home: Path) -> dict[str, list[Path]]:
    """Rollouts per Codex thread, oldest first: a revert writes a new file under the same id."""
    threads: dict[str, list[Path]] = {}
    for path in _rollouts(home):
        threads.setdefault(_thread_id(path), []).append(path)
    return threads


def list_projects(home: Path) -> list[SourceProject]:
    _META.clear()
    _BY_ROLLOUT_ID.clear()
    by_cwd: dict[str, list[Path]] = {}
    for paths in _by_thread(home).values():
        path = paths[-1]  # the thread's current state
        meta = _meta(path)
        # Subagent threads repeat work their parent already shows.
        source = meta.get("source")
        if isinstance(source, dict) and "subagent" in source:
            continue
        # The raw cwd is the identity (hashed into the project id); "a-b" and "a/b" stay apart.
        by_cwd.setdefault(str(meta.get("cwd") or _UNKNOWN_FOLDER), []).append(path)
    return [
        SourceProject(cwd, display_name(re.sub(r"[^A-Za-z0-9]+", "-", cwd)), paths)
        for cwd, paths in by_cwd.items()
    ]


def _records(path: Path) -> list[tuple[str, int, dict]]:
    """A fork's prefix comes from the rollout named by ``history_base.thread_id``, not session_meta.id."""
    own = list(read_jsonl(path))
    origin = session_id_of(path)
    tagged = [(origin, record.get("ordinal", index), record) for index, record in enumerate(own)]
    base = _meta(path).get("history_base")
    if not isinstance(base, dict):
        return tagged
    home = next(
        (p.parent for p in path.parents if p.name in ("sessions", "archived_sessions")), None
    )
    rollout = str(base.get("thread_id"))
    base_path = _rollout_path(home, rollout) if home else None
    if base_path is None or base_path == path:
        return tagged  # base rollout deleted: the tail is all there is
    cut = base.get("end_ordinal_exclusive")
    prefix = [t for t in _records(base_path) if not isinstance(cut, int) or t[1] < cut]
    return prefix + tagged


def _mime(data: str) -> str | None:
    try:
        return sniff_image_type(base64.b64decode(data[:16]))
    except (binascii.Error, ValueError):
        return None


def _image(url) -> dict | None:
    head, _, data = str(url or "").partition(";base64,")
    mime = head.removeprefix("data:")
    # view_image sends the file's bytes as application/octet-stream.
    mime = mime if mime.startswith("image/") else _mime(data)
    return {"data": data, "mimeType": mime} if head.startswith("data:") and data and mime else None


def _output(output) -> str | dict:
    if not isinstance(output, list):
        return clean_text(output) if isinstance(output, str) else ""
    items = [item for item in output if isinstance(item, dict)]
    text = clean_text("\n\n".join(item.get("text") or "" for item in items))
    images = [
        image
        for item in items
        if item.get("type") == "input_image" and (image := _image(item.get("image_url")))
    ]
    return {"text": text, "images": images} if images else text


def _call(payload: dict, message_id: str) -> dict:
    raw = payload.get("arguments") if payload["type"] == "function_call" else payload.get("input")
    try:
        args = json.loads(raw) if payload["type"] == "function_call" else {"input": raw}
    except (TypeError, ValueError):
        args = {"input": raw}
    return tool_call(
        str(payload.get("call_id") or message_id),
        {"name": payload.get("name"), "input": args},
    )


def _turn(text: str, images: list[str]) -> list[dict]:
    text = clean_text(text)
    return [{"type": "image", "image": url} for url in images] + (
        [{"type": "text", "text": text}] if text else []
    )


def read_transcript(path: Path, thread_id: str, session_id: str) -> Transcript:
    file_created, file_updated = file_times_ms(path)
    revision = session_id_of(path)
    messages: list[dict] = []
    inherited = 0
    open_calls: dict[str, dict] = {}
    reread: set[str] = set()
    images: list[str] = []
    pictured: set[str] = set()
    made: set[str] = set()
    ends = [0]
    drawn = [0]
    for position, (origin, ordinal, line) in enumerate(_records(path)):
        payload = line.get("payload")
        if not isinstance(payload, dict):
            continue
        kind = (line.get("type"), payload.get("type"))
        # Thread + origin file + ordinal: a revert's inherited messages keep their ids, and a fork
        # (another thread) gets its own, since message ids are unique across threads.
        message_id = stable_id("codex", session_id, origin, str(ordinal), length = 16)
        item = payload.get("item") if kind == ("event_msg", "item_completed") else None
        if kind == ("response_item", "message") and payload.get("role") == "user":
            # The image bytes are only here; the turn's event that follows names just a path.
            images = [
                str(part["image_url"])
                for part in payload.get("content") or []
                if isinstance(part, dict)
                and part.get("type") == "input_image"
                and str(part.get("image_url") or "").startswith("data:image/")
            ]
            continue
        if kind in (("event_msg", "user_message"), ("event_msg", "agent_message")):
            role = "user" if payload["type"] == "user_message" else "assistant"
            attached = payload.get("images") or payload.get("local_images")
            parts = _turn(str(payload.get("message") or ""), images if attached else [])
        elif isinstance(item, dict) and item.get("type") in ("UserMessage", "AgentMessage"):
            role = "user" if item["type"] == "UserMessage" else "assistant"
            content = [part for part in item.get("content") or [] if isinstance(part, dict)]
            text = "".join(str(part.get("text") or "") for part in content)
            attached = any(part.get("type") in ("image", "local_image") for part in content)
            parts = _turn(text, images if attached else [])
        elif kind in (("response_item", "function_call"), ("response_item", "custom_tool_call")):
            role, parts = "assistant", [_call(payload, message_id)]
            open_calls[parts[0]["toolCallId"]] = parts[0]
        elif kind in (
            ("response_item", "function_call_output"),
            ("response_item", "custom_tool_call_output"),
        ):
            call = open_calls.pop(str(payload.get("call_id")), None)
            result = _output(payload.get("output"))
            if call is not None:
                # An empty output is still a finished call; Studio replays "" differently from none.
                call["result"] = result
            continue
        elif kind == ("response_item", "image_generation_call") and payload.get("result"):
            image = str(payload["result"])
            role, parts = (
                "assistant",
                [
                    {
                        **tool_call(
                            str(payload.get("id") or message_id),
                            {
                                "name": "image_generation",
                                "input": {
                                    "prompt": str(payload.get("revised_prompt") or ""),
                                    "_server_tool": True,
                                },
                            },
                        ),
                        "result": {"image_b64": image, "image_mime": _mime(image) or "image/png"},
                    }
                ],
            )
            made.add(message_id)
        else:
            continue
        if not parts:
            continue
        if item is not None:
            reread.add(message_id)
        timestamp = iso_ms(line.get("timestamp"))
        inherited += origin != revision
        messages.append(
            {
                "id": message_id,
                "threadId": thread_id,
                "parentId": messages[-1]["id"] if messages else None,
                "role": role,
                "content": parts,
                "createdAt": timestamp if timestamp is not None else file_created + position,
                "metadata": {"importedFrom": "codex", "codexSessionId": session_id},
            }
        )
        images = []
        if parts[-1]["type"] == "image" or message_id in made:
            pictured.add(message_id)
        else:
            ends.append(len(messages))
        if message_id not in made:
            drawn.append(len(messages))
    # Keyed on the file's own mode: a deleted base rollout must not drop the tag.
    if reread or _meta(path).get("history_mode") == "paginated":
        revision = f"{revision}:items"
    base = revision.partition(":")[0]
    # An ":images" mark predates generated images.
    earlier = {revision: ends, f"{base}:images": drawn} if made else {revision: ends}
    return Transcript(
        session_id = session_id,
        thread_id = thread_id,
        title = title_from(first_user_text(messages), "Codex session"),
        created_at_ms = messages[0]["createdAt"] if messages else file_created,
        updated_at_ms = max(messages[-1]["createdAt"] if messages else 0, file_updated),
        messages = messages,
        revision = f"{base}:generated" if made else f"{base}:images" if pictured else revision,
        inherited = inherited,
        reread = frozenset(reread | pictured),
        earlier = earlier if pictured else {},
    )


SOURCE = Source(
    key = "codex",
    label = "Codex",
    home_env = "CODEX_HOME",
    default_home = ".codex",
    list_projects = list_projects,
    read_transcript = read_transcript,
    session_key = _thread_id,
    # Counts subagent rollouts too: the probe only decides whether to show the row.
    count_sessions = lambda home: len(_rollouts(home)),
)
