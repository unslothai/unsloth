# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account-scoped transcript history, stored independently of model residency."""

from __future__ import annotations

import json
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


def gallery_dir() -> Path:
    if is_owner_context():
        return ensure_dir(studio_root() / "transcripts")
    return ensure_account_dir(account_path("transcripts"))


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
    directory = gallery_dir()
    staged = directory / f".{transcript_id}.tmp"
    try:
        staged.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        os.replace(staged, directory / f"{transcript_id}.json")
    finally:
        staged.unlink(missing_ok=True)
    return record


def _read(directory: Path, transcript_id: str) -> dict | None:
    if not _ID_RE.fullmatch(transcript_id):
        return None
    path = directory / f"{transcript_id}.json"
    if path.is_symlink():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("id") != transcript_id:
            return None
        if not all(
            isinstance(record.get(key), str) for key in ("text", "model", "title", "created_at")
        ):
            return None
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
        records.append(record)
    records.sort(key=lambda record: (record["created_at"], record["id"]), reverse=True)
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
        gallery_flags.set_flags_locked(directory, transcript_id, archived=archived)
        record["archived"] = archived
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
    with gallery_flags.exclusive(directory, require_file_lock=True):
        flags = gallery_flags.read_trusted(directory)
        removed = []
        for path in directory.glob("*.json"):
            if gallery_flags.is_archived(flags, path.stem) or _read(directory, path.stem) is None:
                continue
            path.unlink()
            removed.append(path.stem)
        gallery_flags.forget_locked(directory, removed)
    return len(removed)
