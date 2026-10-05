# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Saved voices: ``{id}.wav`` (24 kHz mono, at most 30 s) plus ``{id}.json`` under the account's
``<gallery_dir>/voices``, kept until deleted. As in the gallery the sidecar is written last and is
what makes a pair a voice, so a lone WAV is never listed, served or deleted."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from core.inference import audio_gallery
from core.inference.audio_inputs import (
    REFERENCE_MAX_SECONDS,
    REFERENCE_RATE,
    AudioInputError,
    _valid_id,
    _write_json,
    inputs_dir,
    transcode,
)
from loggers import get_logger

logger = get_logger(__name__)

MAX_VOICES = 200
NAME_MAX = 80
TRANSCRIPT_MAX = 4000
LANGUAGE_MAX = 64
_REQUIRED = ("name", "duration_s", "sample_rate", "created_at")


def voices_dir() -> Path:
    directory = audio_gallery.gallery_dir() / "voices"
    directory.mkdir(parents = True, exist_ok = True)
    return directory


def _sidecar(voice_id: str) -> Path:
    return voices_dir() / f"{voice_id}.json"


def _read(voice_id: str) -> Optional[dict[str, Any]]:
    if not _valid_id(voice_id):
        return None
    try:
        meta = json.loads(_sidecar(voice_id).read_text(encoding = "utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    if not isinstance(meta, dict) or any(k not in meta for k in _REQUIRED):
        return None
    return meta


def _record(voice_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": voice_id,
        "name": meta["name"],
        "transcript": meta.get("transcript"),
        "language": meta.get("language"),
        "duration_s": meta["duration_s"],
        "sample_rate": meta["sample_rate"],
        "created_at": meta["created_at"],
        "url": f"/api/inference/audio/voices/{voice_id}/file",
    }


def _clean_text(value: Optional[str], limit: int) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text[:limit] if text else None


def _clean_name(value: Optional[str]) -> str:
    text = " ".join(str(value or "").split())[:NAME_MAX]
    if not text:
        raise AudioInputError(400, "Give the voice a name.")
    return text


def _ids() -> list[str]:
    try:
        return [p.stem for p in voices_dir().glob("*.json") if _read(p.stem) is not None]
    except OSError:
        return []


def create(src_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Save the first 30 s of ``src_path`` as a voice; 400 past 200 voices."""
    name = _clean_name(meta.get("name"))
    if len(_ids()) >= MAX_VOICES:
        raise AudioInputError(
            400, f"You have {MAX_VOICES} saved voices. Delete one to save another."
        )
    voice_id = uuid.uuid4().hex
    wav_path = voices_dir() / f"{voice_id}.wav"
    try:
        info = transcode(
            src_path,
            wav_path,
            rate = REFERENCE_RATE,
            mono = True,
            max_seconds = REFERENCE_MAX_SECONDS,
            cut = True,
        )
        record_meta = {
            "name": name,
            "transcript": _clean_text(meta.get("transcript"), TRANSCRIPT_MAX),
            "language": _clean_text(meta.get("language"), LANGUAGE_MAX),
            "duration_s": info["duration_s"],
            "sample_rate": info["sample_rate"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "created_at_epoch": time.time(),
        }
        _write_json(_sidecar(voice_id), record_meta)
    except BaseException:
        wav_path.unlink(missing_ok = True)
        raise
    return _record(voice_id, record_meta)


def list_voices() -> list[dict[str, Any]]:
    """Every saved voice in this account, newest first."""
    entries = []
    for voice_id in _ids():
        meta = _read(voice_id)
        if meta is None or not (voices_dir() / f"{voice_id}.wav").is_file():
            continue
        created = float(meta.get("created_at_epoch") or 0.0)
        entries.append((created, meta["created_at"], voice_id, meta))
    entries.sort(key = lambda e: (e[0], e[1], e[2]), reverse = True)
    return [_record(voice_id, meta) for _e, _c, voice_id, meta in entries]


def get(voice_id: str) -> Optional[dict[str, Any]]:
    meta = _read(voice_id)
    if meta is None or voice_path(voice_id) is None:
        return None
    return _record(voice_id, meta)


def voice_path(voice_id: str) -> Optional[Path]:
    """The WAV of a saved voice in this account; None if unknown, unsafe or another account's."""
    if _read(voice_id) is None:
        return None
    directory = voices_dir()
    path = directory / f"{voice_id}.wav"
    try:
        path.resolve().relative_to(directory.resolve())
    except (OSError, ValueError):
        return None
    return path if path.is_file() else None


def update(voice_id: str, patch: dict[str, Any]) -> Optional[dict[str, Any]]:
    meta = _read(voice_id)
    if meta is None or voice_path(voice_id) is None:
        return None
    if "name" in patch and patch["name"] is not None:
        meta["name"] = _clean_name(patch["name"])
    if "transcript" in patch:
        meta["transcript"] = _clean_text(patch["transcript"], TRANSCRIPT_MAX)
    if "language" in patch:
        meta["language"] = _clean_text(patch["language"], LANGUAGE_MAX)
    _write_json(_sidecar(voice_id), meta)
    return _record(voice_id, meta)


def delete(voice_id: str) -> bool:
    """Remove an owned voice; WAV first, as the gallery does."""
    path = voice_path(voice_id)
    if path is None:
        return False
    try:
        path.unlink()
    except OSError as exc:
        logger.warning("audio_voices.delete_failed: %s", exc)
        return False
    try:
        _sidecar(voice_id).unlink(missing_ok = True)
    except OSError:
        pass
    try:
        for copy in inputs_dir().glob(f"v-{voice_id}.*.wav"):
            copy.unlink(missing_ok = True)
    except OSError:
        pass
    return True
