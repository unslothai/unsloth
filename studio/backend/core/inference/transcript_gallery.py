# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account-scoped transcript history; optional keys are validated on every read."""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.inference import gallery_flags
from utils.account_context import is_owner_context
from utils.paths import ensure_account_dir, ensure_dir, studio_root
from utils.paths.storage_roots import account_path

_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_BASE_KEYS = ("id", "title", "text", "model", "duration", "language", "created_at", "archived")
_MAX_SEGMENTS = 20_000
_MAX_WORDS = 100_000
_MAX_SPEAKERS = 64
SPEAKER_NAME_MAX = 40
_SOURCE_KINDS = ("input", "clip", "voice")
# The Audio page's input, clip and voice ids (audio_gallery's pattern).
_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class TranscriptPatchError(ValueError):
    """A speaker rename the record cannot take (an unknown id, a name too long)."""


def gallery_dir() -> Path:
    if is_owner_context():
        return ensure_dir(studio_root() / "transcripts")
    return ensure_account_dir(account_path("transcripts"))


def _seconds(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _spans(items, limit: int, text_key: str, speakers: bool) -> list[dict] | None:
    if not isinstance(items, list) or len(items) > limit:
        return None
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        start, end = _seconds(item.get("start")), _seconds(item.get("end"))
        text = item.get(text_key)
        if start is None or end is None or start > end or not isinstance(text, str):
            continue
        span = {"start": start, "end": end, text_key: text}
        speaker = item.get("speaker")
        if speakers and isinstance(speaker, str) and 0 < len(speaker) <= 64:
            span["speaker"] = speaker
        out.append(span)
    return out


def _sanitize_details(record: dict) -> dict:
    """``record`` with each optional key kept only when well formed; base keys untouched."""
    clean = {key: value for key, value in record.items() if key in _BASE_KEYS}
    segments = _spans(record.get("segments"), _MAX_SEGMENTS, "text", speakers = True)
    if segments:
        clean["segments"] = segments
    words = _spans(record.get("words"), _MAX_WORDS, "word", speakers = False)
    if words:
        clean["words"] = words
    speakers = record.get("speakers")
    if isinstance(speakers, list) and len(speakers) <= _MAX_SPEAKERS:
        kept = [
            {"id": s["id"], "label": s["label"]}
            for s in speakers
            if isinstance(s, dict)
            and isinstance(s.get("id"), str)
            and 0 < len(s["id"]) <= 64
            and isinstance(s.get("label"), str)
            and len(s["label"]) <= 64
        ]
        if kept:
            clean["speakers"] = list({s["id"]: s for s in kept}.values())
    known = {s["id"] for s in clean.get("speakers", ())}
    names = record.get("speaker_names")
    if isinstance(names, dict):
        kept_names = {
            sid: name.strip()
            for sid, name in names.items()
            if sid in known
            and isinstance(name, str)
            and name.strip()
            and len(name.strip()) <= SPEAKER_NAME_MAX
        }
        if kept_names:
            clean["speaker_names"] = kept_names
    source = record.get("source")
    if (
        isinstance(source, dict)
        and source.get("kind") in _SOURCE_KINDS
        and isinstance(source.get("id"), str)
        and _SOURCE_ID_RE.fullmatch(source["id"])
        and isinstance(source.get("name"), str)
    ):
        clean["source"] = {"kind": source["kind"], "id": source["id"], "name": source["name"][:255]}
    if record.get("timestamps") is True and "segments" in clean:
        clean["timestamps"] = True
    return clean


def summary(record: dict) -> dict:
    row = {key: record.get(key) for key in _BASE_KEYS}
    row["segment_count"] = len(record.get("segments") or ())
    row["has_words"] = bool(record.get("words"))
    for key in ("speakers", "speaker_names", "source"):
        if key in record:
            row[key] = record[key]
    return row


def _write(directory: Path, record: dict) -> None:
    transcript_id = record["id"]
    staged = directory / f".{transcript_id}.tmp"
    try:
        staged.write_text(json.dumps(record, ensure_ascii = False), encoding = "utf-8")
        os.replace(staged, directory / f"{transcript_id}.json")
    finally:
        staged.unlink(missing_ok = True)


def save(result: dict, title: str) -> dict:
    transcript_id = uuid.uuid4().hex
    record = {
        "id": transcript_id,
        "title": title.replace("\\", "/").rsplit("/", 1)[-1][:255] or "Recording",
        "text": result["text"],
        "model": result["model"],
        "duration": result.get("duration"),
        "language": result.get("language"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archived": False,
    }
    details = {
        key: result[key]
        for key in ("segments", "words", "speakers", "speaker_names", "source", "timestamps")
        if key in result
    }
    if details:
        record = _sanitize_details({**record, **details})
    _write(gallery_dir(), record)
    return record


def _read(directory: Path, transcript_id: str) -> dict | None:
    if not _ID_RE.fullmatch(transcript_id):
        return None
    path = directory / f"{transcript_id}.json"
    if path.is_symlink():
        return None
    try:
        record = json.loads(path.read_text(encoding = "utf-8"))
        if not isinstance(record, dict) or record.get("id") != transcript_id:
            return None
        if not all(
            isinstance(record.get(key), str) for key in ("text", "model", "title", "created_at")
        ):
            return None
        if any(key not in _BASE_KEYS for key in record):
            return _sanitize_details(record)
        return record
    except (OSError, ValueError):
        return None


def get(transcript_id: str) -> dict | None:
    directory = gallery_dir()
    record = _read(directory, transcript_id)
    if record is not None:
        record["archived"] = gallery_flags.is_archived(gallery_flags.read(directory), transcript_id)
    return record


def list_transcripts(
    limit: int = 50,
    before: str | None = None,
    archived: bool = False,
) -> dict:
    directory = gallery_dir()
    flags = gallery_flags.read(directory)
    records = []
    for path in directory.glob("*.json"):
        record = _read(directory, path.stem)
        if record is None or gallery_flags.is_archived(flags, path.stem) != archived:
            continue
        cursor = f"{record['created_at']}|{record['id']}"
        if before is not None and cursor >= before:
            continue
        record["archived"] = archived
        records.append(summary(record))
    records.sort(key = lambda record: (record["created_at"], record["id"]), reverse = True)
    visible = records[:limit]
    return {
        "transcripts": visible,
        "next_cursor": (
            f"{visible[-1]['created_at']}|{visible[-1]['id']}" if len(records) > limit else None
        ),
    }


def set_archived(transcript_id: str, archived: bool) -> dict | None:
    # No require_file_lock, as audio_gallery.set_flags: only clear(), which DELETES on a
    # flag, has to fail closed when the lock is unavailable.
    directory = gallery_dir()
    with gallery_flags.exclusive(directory):
        record = _read(directory, transcript_id)
        if record is None:
            return None
        gallery_flags.set_flags_locked(directory, transcript_id, archived = archived)
        record["archived"] = archived
        return record


def set_speaker_names(transcript_id: str, names: dict) -> dict | None:
    """None or "" unnames; raises ``TranscriptPatchError`` on unknown ids or overlong names."""
    directory = gallery_dir()
    with gallery_flags.exclusive(directory):
        record = _read(directory, transcript_id)
        if record is None:
            return None
        known = {s["id"] for s in record.get("speakers", ())}
        merged = dict(record.get("speaker_names") or {})
        for speaker_id, name in names.items():
            if speaker_id not in known:
                raise TranscriptPatchError(f"This transcript has no speaker '{speaker_id}'.")
            if name is not None and not isinstance(name, str):
                raise TranscriptPatchError("A speaker name must be text.")
            name = (name or "").strip()
            if len(name) > SPEAKER_NAME_MAX:
                raise TranscriptPatchError(
                    f"Speaker names can be at most {SPEAKER_NAME_MAX} characters."
                )
            if name:
                merged[speaker_id] = name
            else:
                merged.pop(speaker_id, None)
        if merged:
            record["speaker_names"] = merged
        else:
            record.pop("speaker_names", None)
        stored = {key: value for key, value in record.items() if key != "archived"}
        stored["archived"] = False
        _write(directory, stored)
        record["archived"] = gallery_flags.is_archived(gallery_flags.read(directory), transcript_id)
        return record


def delete(transcript_id: str) -> bool:
    directory = gallery_dir()
    with gallery_flags.exclusive(directory):
        if _read(directory, transcript_id) is None:
            return False
        (directory / f"{transcript_id}.json").unlink()
        gallery_flags.forget_locked(directory, [transcript_id])
    return True


def clear() -> int:
    directory = gallery_dir()
    with gallery_flags.exclusive(directory, require_file_lock = True):
        flags = gallery_flags.read_trusted(directory)
        removed = []
        for path in directory.glob("*.json"):
            if gallery_flags.is_archived(flags, path.stem) or _read(directory, path.stem) is None:
                continue
            path.unlink()
            removed.append(path.stem)
        gallery_flags.forget_locked(directory, removed)
    return len(removed)
