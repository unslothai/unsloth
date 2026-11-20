# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Characters, personas and character chats (a thread plus ``chat_cast`` and ``chat_scene``).

Avatars are files next to studio.db, one per character or persona.
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Optional

from core.characters.cards import EDITABLE_TEXT_FIELDS, camel
from storage.studio_db import connection as _connect
from storage.studio_db import (
    delete_chat_threads,
    now_ms,
    sync_chat_messages,
    upsert_chat_thread,
)
from utils.image_files import atomic_write_bytes
from utils.paths.storage_roots import account_path, ensure_account_dir

MAX_CHARACTERS = 5000
MAX_PERSONAS = 200
MAX_CAST = 16
MAX_ID_CHARS = 64


class CharacterLimitReached(Exception):
    """An account's character or persona limit, or a chat's cast limit."""


class CharacterNotFound(LookupError):
    pass


class PersonaNotFound(LookupError):
    pass


class CastAllMuted(ValueError):
    """A group change that would leave every member muted."""


def _loads(value: Optional[str], fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii = False)


def _avatar_path(kind: str, item_id: str) -> Path:
    if not re.fullmatch(rf"[A-Za-z0-9_-]{{1,{MAX_ID_CHARS}}}", item_id):
        raise ValueError("bad id")
    return account_path(kind) / f"{item_id}.img"


def write_avatar(kind: str, item_id: str, data: bytes) -> None:
    path = _avatar_path(kind, item_id)
    # Only writes make the folder, and never for a deleted account.
    ensure_account_dir(path.parent)
    atomic_write_bytes(path, data)


def read_avatar(kind: str, item_id: str) -> Optional[bytes]:
    path = _avatar_path(kind, item_id)
    return path.read_bytes() if path.is_file() else None


def delete_avatar(kind: str, item_id: str) -> None:
    _avatar_path(kind, item_id).unlink(missing_ok = True)


_CHARACTER_COLUMNS = {camel(column): column for column in (*EDITABLE_TEXT_FIELDS, "talkativeness")}
_CHARACTER_JSON_COLUMNS = {
    camel(field): (f"{field}_json", fallback)
    for field, fallback in (("alternate_greetings", []), ("tags", []), ("extensions", {}))
}
_CHARACTER_NULLABLE_JSON = {
    camel(field): f"{field}_json" for field in ("depth_prompt", "character_book")
}


def _character_from_row(row: sqlite3.Row, *, full: bool = True) -> dict:
    character: dict[str, Any] = {"id": row["id"]}
    character.update({key: row[column] for key, column in _CHARACTER_COLUMNS.items()})
    for key, (column, fallback) in _CHARACTER_JSON_COLUMNS.items():
        character[key] = _loads(row[column], fallback)
    for key, column in _CHARACTER_NULLABLE_JSON.items():
        character[key] = _loads(row[column], None)
    character.update(
        {
            "favorite": bool(row["favorite"]),
            "hasAvatar": bool(row["avatar_type"]),
            "avatarType": row["avatar_type"],
            "avatarRev": row["avatar_rev"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
            "lastChatAt": row["last_active"] or None,
            "chatCount": row["chat_count"],
        }
    )
    if full:
        character["card"] = _loads(row["card_json"], None)
    return character


_ACTIVITY_COLUMNS = (
    "(SELECT COUNT(*) FROM chat_cast c JOIN chat_threads t ON t.id = c.thread_id"
    " WHERE c.character_id = characters.id AND t.archived = 0) AS chat_count,"
    # Recency follows the character's chats, so continuing an old chat moves it up.
    " MAX(COALESCE(last_chat_at, 0), COALESCE((SELECT MAX(COALESCE(t.updated_at, t.created_at))"
    " FROM chat_cast c JOIN chat_threads t ON t.id = c.thread_id"
    " WHERE c.character_id = characters.id AND t.archived = 0), 0)) AS last_active"
)
# Every column but card_json, which only a single read or an export needs.
_LIST_COLUMNS = ", ".join(
    (
        "id",
        *_CHARACTER_COLUMNS.values(),
        *(column for column, _fallback in _CHARACTER_JSON_COLUMNS.values()),
        *_CHARACTER_NULLABLE_JSON.values(),
        "favorite",
        "avatar_type",
        "avatar_rev",
        "created_at",
        "updated_at",
        "last_chat_at",
    )
)


def list_characters() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT {_LIST_COLUMNS}, {_ACTIVITY_COLUMNS} FROM characters"
            " ORDER BY COALESCE(NULLIF(last_active, 0), updated_at) DESC, name"
        ).fetchall()
    return [_character_from_row(row, full = False) for row in rows]


def get_character(character_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            f"SELECT *, {_ACTIVITY_COLUMNS} FROM characters WHERE id = ?", (character_id,)
        ).fetchone()
    return _character_from_row(row) if row else None


def get_characters(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT *, {_ACTIVITY_COLUMNS} FROM characters WHERE id IN ({placeholders})", ids
        ).fetchall()
    by_id = {row["id"]: _character_from_row(row) for row in rows}
    return [by_id[i] for i in ids if i in by_id]


def _character_values(fields: dict) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, column in _CHARACTER_COLUMNS.items():
        if key in fields and fields[key] is not None:
            values[column] = fields[key]
    for key, (column, _fallback) in _CHARACTER_JSON_COLUMNS.items():
        if key in fields and fields[key] is not None:
            values[column] = _dumps(fields[key])
    for key, column in _CHARACTER_NULLABLE_JSON.items():
        if key in fields:
            values[column] = _dumps(fields[key]) if fields[key] else None
    if "favorite" in fields and fields["favorite"] is not None:
        values["favorite"] = 1 if fields["favorite"] else 0
    return values


def create_character(
    fields: dict,
    *,
    card: Optional[dict] = None,
    avatar: Optional[bytes] = None,
    avatar_type: Optional[str] = None,
    starter_id: Optional[str] = None,
) -> Optional[dict]:
    """A new character; with *starter_id*, only if the account never received that starter (None otherwise)."""
    character_id = uuid.uuid4().hex
    now = now_ms()
    values = _character_values(fields)
    values.update(
        {
            "id": character_id,
            "card_json": _dumps(card) if card is not None else None,
            "avatar_type": avatar_type if avatar else None,
            "avatar_rev": 1 if avatar else 0,
            "created_at": now,
            "updated_at": now,
        }
    )
    if avatar:
        write_avatar("characters", character_id, avatar)
    try:
        with _connect("BEGIN IMMEDIATE") as conn:
            if starter_id is not None:
                # Claimed in the same transaction as the insert, so two loads cannot seed twice.
                claimed = conn.execute(
                    "INSERT OR IGNORE INTO character_starters (starter_id, seeded_at) VALUES (?, ?)",
                    (starter_id, now),
                ).rowcount
                if not claimed:
                    if avatar:
                        delete_avatar("characters", character_id)
                    return None
            (count,) = conn.execute("SELECT COUNT(*) FROM characters").fetchone()
            if count >= MAX_CHARACTERS:
                raise CharacterLimitReached(
                    f"An account can hold up to {MAX_CHARACTERS} characters."
                )
            columns = ", ".join(values)
            conn.execute(
                f"INSERT INTO characters ({columns}) VALUES ({', '.join('?' for _ in values)})",
                list(values.values()),
            )
    except BaseException:
        if avatar:
            delete_avatar("characters", character_id)
        raise
    return get_character(character_id)  # type: ignore[return-value]


def seeded_starters() -> set[str]:
    with _connect() as conn:
        return {row[0] for row in conn.execute("SELECT starter_id FROM character_starters")}


def update_character(character_id: str, fields: dict) -> Optional[dict]:
    values = _character_values(fields)
    values["updated_at"] = now_ms()
    assignments = ", ".join(f"{column} = ?" for column in values)
    with _connect("BEGIN IMMEDIATE") as conn:
        cursor = conn.execute(
            f"UPDATE characters SET {assignments} WHERE id = ?",
            [*values.values(), character_id],
        )
        if cursor.rowcount == 0:
            return None
    return get_character(character_id)


def _set_avatar(table: str, item_id: str, data: Optional[bytes], media_type: Optional[str]) -> bool:
    # File and row change under one write lock, so a racing upload or delete cannot leave
    # one upload's bytes under another's type, or a file with no row.
    with _connect("BEGIN IMMEDIATE") as conn:
        if conn.execute(f"SELECT 1 FROM {table} WHERE id = ?", (item_id,)).fetchone() is None:
            return False
        if data:
            write_avatar(table, item_id, data)
        else:
            delete_avatar(table, item_id)
        conn.execute(
            f"UPDATE {table} SET avatar_type = ?, avatar_rev = avatar_rev + 1, updated_at = ? WHERE id = ?",
            (media_type if data else None, now_ms(), item_id),
        )
    return True


def set_character_avatar(
    character_id: str, data: Optional[bytes], media_type: Optional[str]
) -> Optional[dict]:
    if not _set_avatar("characters", character_id, data, media_type):
        return None
    return get_character(character_id)


def delete_character(character_id: str) -> bool:
    with _connect("BEGIN IMMEDIATE") as conn:
        threads = [
            row[0]
            for row in conn.execute(
                "SELECT thread_id FROM chat_cast WHERE character_id = ?", (character_id,)
            )
        ]
        cursor = conn.execute("DELETE FROM characters WHERE id = ?", (character_id,))
        deleted = cursor.rowcount > 0
        # The cascade must not leave a group with every member muted.
        for thread_id in threads:
            rows = conn.execute(
                "SELECT character_id, muted FROM chat_cast WHERE thread_id = ? ORDER BY position, character_id",
                (thread_id,),
            ).fetchall()
            if len(rows) > 1 and all(row["muted"] for row in rows):
                conn.execute(
                    "UPDATE chat_cast SET muted = 0 WHERE thread_id = ? AND character_id = ?",
                    (thread_id, rows[0]["character_id"]),
                )
    if deleted:
        delete_avatar("characters", character_id)
    return deleted


def _persona_from_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "hasAvatar": bool(row["avatar_type"]),
        "avatarType": row["avatar_type"],
        "avatarRev": row["avatar_rev"],
        "isDefault": bool(row["is_default"]),
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def list_personas() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM personas ORDER BY is_default DESC, name").fetchall()
    return [_persona_from_row(row) for row in rows]


def get_persona(persona_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM personas WHERE id = ?", (persona_id,)).fetchone()
    return _persona_from_row(row) if row else None


def create_persona(
    name: str,
    description: str = "",
    is_default: bool = False,
) -> dict:
    persona_id = uuid.uuid4().hex
    now = now_ms()
    with _connect("BEGIN IMMEDIATE") as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM personas").fetchone()
        if count >= MAX_PERSONAS:
            raise CharacterLimitReached(f"An account can hold up to {MAX_PERSONAS} personas.")
        # The first persona is the default, so a new chat always has someone to speak as.
        if count == 0:
            is_default = True
        if is_default:
            conn.execute("UPDATE personas SET is_default = 0")
        conn.execute(
            "INSERT INTO personas (id, name, description, is_default, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (persona_id, name, description, 1 if is_default else 0, now, now),
        )
    return get_persona(persona_id)  # type: ignore[return-value]


def update_persona(persona_id: str, fields: dict) -> Optional[dict]:
    values: dict[str, Any] = {}
    for key in ("name", "description"):
        if fields.get(key) is not None:
            values[key] = fields[key]
    values["updated_at"] = now_ms()
    with _connect("BEGIN IMMEDIATE") as conn:
        if conn.execute("SELECT 1 FROM personas WHERE id = ?", (persona_id,)).fetchone() is None:
            return None
        if fields.get("isDefault"):
            conn.execute("UPDATE personas SET is_default = 0")
            values["is_default"] = 1
        assignments = ", ".join(f"{column} = ?" for column in values)
        conn.execute(
            f"UPDATE personas SET {assignments} WHERE id = ?", [*values.values(), persona_id]
        )
    return get_persona(persona_id)


def set_persona_avatar(
    persona_id: str, data: Optional[bytes], media_type: Optional[str]
) -> Optional[dict]:
    if not _set_avatar("personas", persona_id, data, media_type):
        return None
    return get_persona(persona_id)


def delete_persona(persona_id: str) -> bool:
    with _connect("BEGIN IMMEDIATE") as conn:
        row = conn.execute("SELECT is_default FROM personas WHERE id = ?", (persona_id,)).fetchone()
        if row is None:
            return False
        if row["is_default"]:
            successor = conn.execute(
                "SELECT id FROM personas WHERE id != ? ORDER BY created_at LIMIT 1", (persona_id,)
            ).fetchone()
        else:
            successor = conn.execute("SELECT id FROM personas WHERE is_default = 1").fetchone()
        successor_id = successor["id"] if successor else None
        # Its chats move to the default here: the foreign key's SET NULL would mean the account instead.
        conn.execute(
            "UPDATE chat_scene SET persona_id = ?, updated_at = ? WHERE persona_id = ?",
            (successor_id, now_ms(), persona_id),
        )
        conn.execute("DELETE FROM personas WHERE id = ?", (persona_id,))
        if row["is_default"] and successor_id:
            conn.execute("UPDATE personas SET is_default = 1 WHERE id = ?", (successor_id,))
    delete_avatar("personas", persona_id)
    return True


def create_character_chat(
    *,
    thread: dict,
    character_ids: list[str],
    persona_id: Optional[str],
    settings: Optional[dict],
    messages: list[dict],
) -> None:
    """Thread, cast, scene and opening messages, all or nothing."""
    now = thread["createdAt"]
    upsert_chat_thread(thread)
    try:
        with _connect("BEGIN IMMEDIATE") as conn:
            # Checked again under the write lock: a delete since the caller's read would trip a foreign key.
            placeholders = ",".join("?" for _ in character_ids)
            (known,) = conn.execute(
                f"SELECT COUNT(*) FROM characters WHERE id IN ({placeholders})", character_ids
            ).fetchone()
            if known != len(character_ids):
                raise CharacterNotFound("character")
            if (
                persona_id
                and conn.execute("SELECT 1 FROM personas WHERE id = ?", (persona_id,)).fetchone()
                is None
            ):
                raise PersonaNotFound(persona_id)
            for position, character_id in enumerate(character_ids):
                conn.execute(
                    "INSERT INTO chat_cast (thread_id, character_id, position, muted) VALUES (?, ?, ?, 0)",
                    (thread["id"], character_id, position),
                )
            conn.executemany(
                "UPDATE characters SET last_chat_at = ? WHERE id = ?",
                [(now, character_id) for character_id in character_ids],
            )
            conn.execute(
                "INSERT INTO chat_scene (thread_id, persona_id, settings_json, updated_at) VALUES (?, ?, ?, ?)",
                (thread["id"], persona_id, _dumps(settings or {}), now),
            )
        sync_chat_messages(thread["id"], messages)
    except BaseException:
        delete_chat_threads([thread["id"]])
        raise


def _scene_from_rows(
    scene: Optional[sqlite3.Row], cast_rows: list[sqlite3.Row], thread_id: str
) -> Optional[dict]:
    if scene is None and not cast_rows:
        return None
    return {
        "threadId": thread_id,
        "personaId": scene["persona_id"] if scene else None,
        "settings": _loads(scene["settings_json"], {}) if scene else {},
        "cast": [
            {
                "characterId": row["character_id"],
                "position": row["position"],
                "muted": bool(row["muted"]),
            }
            for row in cast_rows
        ],
    }


def get_chat_scene(thread_id: str) -> Optional[dict]:
    with _connect() as conn:
        scene = conn.execute(
            "SELECT * FROM chat_scene WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        cast_rows = conn.execute(
            "SELECT * FROM chat_cast WHERE thread_id = ? ORDER BY position, character_id",
            (thread_id,),
        ).fetchall()
    return _scene_from_rows(scene, cast_rows, thread_id)


def _merge_settings(stored: dict, patch: dict) -> dict:
    """``patch`` over ``stored``: None clears a key, and a nested object (the author's note) merges field by field."""
    merged = dict(stored)
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def update_chat_scene(
    thread_id: str,
    *,
    add: Optional[list[str]] = None,
    remove: Optional[list[str]] = None,
    mute: Optional[dict[str, bool]] = None,
    persona_id: Optional[str] = None,
    persona_set: bool = False,
    settings: Optional[dict] = None,
) -> Optional[dict]:
    """Applies one edit to the stored scene, so edits from several tabs or queued writes compose."""
    with _connect("BEGIN IMMEDIATE") as conn:
        if conn.execute("SELECT 1 FROM chat_threads WHERE id = ?", (thread_id,)).fetchone() is None:
            return None
        scene = conn.execute(
            "SELECT * FROM chat_scene WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        # Only adding a cast makes a plain chat a character chat.
        if scene is None and not add:
            return None
        if (
            persona_set
            and persona_id
            and conn.execute("SELECT 1 FROM personas WHERE id = ?", (persona_id,)).fetchone()
            is None
        ):
            raise PersonaNotFound(persona_id)
        now = now_ms()
        stored = _loads(scene["settings_json"], {}) if scene else {}
        conn.execute(
            "INSERT OR REPLACE INTO chat_scene (thread_id, persona_id, settings_json, updated_at)"
            " VALUES (?, ?, ?, ?)",
            (
                thread_id,
                persona_id if persona_set else (scene["persona_id"] if scene else None),
                _dumps(_merge_settings(stored, settings) if settings else stored),
                now,
            ),
        )
        if add or remove or mute:
            rows = conn.execute(
                "SELECT character_id, muted FROM chat_cast WHERE thread_id = ? ORDER BY position, character_id",
                (thread_id,),
            ).fetchall()
            cast = {row["character_id"]: bool(row["muted"]) for row in rows}
            for character_id in remove or []:
                cast.pop(character_id, None)
            added = [i for i in dict.fromkeys(add or []) if i not in cast]
            if added:
                placeholders = ",".join("?" for _ in added)
                known = {
                    row[0]
                    for row in conn.execute(
                        f"SELECT id FROM characters WHERE id IN ({placeholders})", added
                    )
                }
                if any(i not in known for i in added):
                    raise CharacterNotFound("character")
                cast.update((i, False) for i in added)
            # A member another tab removed has nothing left to mute.
            cast.update({i: bool(v) for i, v in (mute or {}).items() if i in cast})
            if len(cast) > MAX_CAST:
                raise CharacterLimitReached(f"A chat can have up to {MAX_CAST} characters.")
            if len(cast) > 1 and all(cast.values()):
                raise CastAllMuted("A group chat needs at least one character who is not muted.")
            conn.execute("DELETE FROM chat_cast WHERE thread_id = ?", (thread_id,))
            for position, (character_id, muted) in enumerate(cast.items()):
                conn.execute(
                    "INSERT INTO chat_cast (thread_id, character_id, position, muted) VALUES (?, ?, ?, ?)",
                    (thread_id, character_id, position, 1 if muted else 0),
                )
            if added:
                conn.executemany(
                    "UPDATE characters SET last_chat_at = ? WHERE id = ?", [(now, i) for i in added]
                )
    return get_chat_scene(thread_id)


def chat_index() -> dict[str, list[str]]:
    """Every character chat's cast, in order; a chat whose characters were all deleted maps to []."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT s.thread_id AS thread_id, c.character_id AS character_id FROM chat_scene s"
            " LEFT JOIN chat_cast c ON c.thread_id = s.thread_id ORDER BY s.thread_id, c.position"
        ).fetchall()
    index: dict[str, list[str]] = {}
    for row in rows:
        members = index.setdefault(row["thread_id"], [])
        if row["character_id"]:
            members.append(row["character_id"])
    return index


def chats_for_character(character_id: str, limit: int = 200) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT t.id, t.title, t.created_at, t.updated_at, t.archived,
                   (SELECT COUNT(*) FROM chat_cast c2 WHERE c2.thread_id = t.id) AS cast_size,
                   (SELECT COUNT(*) FROM chat_messages m WHERE m.thread_id = t.id
                      AND COALESCE(json_extract(m.metadata_json, '$.roleplay.kind'), '')
                          NOT IN ('opening', 'turn')) AS message_count
            FROM chat_cast c JOIN chat_threads t ON t.id = c.thread_id
            WHERE c.character_id = ?
            ORDER BY COALESCE(t.updated_at, t.created_at) DESC
            LIMIT ?
            """,
            (character_id, limit),
        ).fetchall()
    return [
        {
            "threadId": row["id"],
            "title": row["title"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"] or row["created_at"],
            "archived": bool(row["archived"]),
            "castSize": row["cast_size"],
            "messageCount": row["message_count"],
        }
        for row in rows
    ]
