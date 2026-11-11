# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Characters, personas and character chats in the account's studio.db.

A character chat is an ordinary chat thread plus a cast (``chat_cast``) and a scene
(``chat_scene``: the persona the user plays and per-chat settings). Portraits live as files
next to the database, one per character or persona, written atomically.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from storage.studio_db import (
    delete_chat_threads,
    get_connection,
    upsert_chat_message,
    upsert_chat_thread,
)

MAX_CHARACTERS = 5000
MAX_PERSONAS = 200
MAX_CAST = 16
MAX_AVATAR_BYTES = 20 * 1024 * 1024

# Tools stay off in a character chat unless the user turns them on there.
CHARACTER_CHAT_SETTINGS = {
    "toolsEnabled": False,
    "codeToolsEnabled": False,
    "imageToolsEnabled": False,
    "webFetchToolsEnabled": False,
    "deepResearchEnabled": False,
    "artifactsEnabled": False,
    "mcpEnabledForChat": False,
    "ragEnabled": False,
}
OPENING_TEXT = "[Start a new chat]"


class LimitReached(Exception):
    pass


@contextmanager
def _connect(begin: Optional[str] = None) -> Iterator[sqlite3.Connection]:
    conn = get_connection()
    try:
        if begin:
            conn.execute(begin)
        yield conn
        if conn.in_transaction:
            conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def now_ms() -> int:
    return int(time.time() * 1000)


def _loads(value: Optional[str], fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii = False)


def _image_dir(kind: str) -> Path:
    from utils.paths.storage_roots import account_path

    path = account_path(kind)
    path.mkdir(parents = True, exist_ok = True)
    return path


def _image_path(kind: str, item_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", item_id):
        raise ValueError("bad id")
    return _image_dir(kind) / f"{item_id}.img"


def write_image(kind: str, item_id: str, data: bytes) -> None:
    path = _image_path(kind, item_id)
    fd, tmp = tempfile.mkstemp(dir = path.parent, prefix = ".upload-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok = True)
        raise


def read_image(kind: str, item_id: str) -> Optional[bytes]:
    path = _image_path(kind, item_id)
    return path.read_bytes() if path.is_file() else None


def delete_image(kind: str, item_id: str) -> None:
    _image_path(kind, item_id).unlink(missing_ok = True)


_CHARACTER_COLUMNS = {
    "name": "name",
    "description": "description",
    "personality": "personality",
    "scenario": "scenario",
    "firstMes": "first_mes",
    "mesExample": "mes_example",
    "systemPrompt": "system_prompt",
    "postHistoryInstructions": "post_history_instructions",
    "creatorNotes": "creator_notes",
    "creator": "creator",
    "characterVersion": "character_version",
    "talkativeness": "talkativeness",
}
_CHARACTER_JSON_COLUMNS = {
    "alternateGreetings": ("alternate_greetings_json", []),
    "tags": ("tags_json", []),
    "extensions": ("extensions_json", {}),
}
_CHARACTER_NULLABLE_JSON = {
    "depthPrompt": "depth_prompt_json",
    "characterBook": "character_book_json",
}


def _character_from_row(row: sqlite3.Row, *, full: bool = True) -> dict:
    character = {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "personality": row["personality"],
        "scenario": row["scenario"],
        "firstMes": row["first_mes"],
        "alternateGreetings": _loads(row["alternate_greetings_json"], []),
        "mesExample": row["mes_example"],
        "systemPrompt": row["system_prompt"],
        "postHistoryInstructions": row["post_history_instructions"],
        "creatorNotes": row["creator_notes"],
        "creator": row["creator"],
        "characterVersion": row["character_version"],
        "tags": _loads(row["tags_json"], []),
        "talkativeness": row["talkativeness"],
        "favorite": bool(row["favorite"]),
        "depthPrompt": _loads(row["depth_prompt_json"], None),
        "characterBook": _loads(row["character_book_json"], None),
        "extensions": _loads(row["extensions_json"], {}),
        "hasAvatar": bool(row["avatar_type"]),
        "avatarType": row["avatar_type"],
        "avatarRev": row["avatar_rev"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "lastChatAt": (row["last_active"] or None)
        if "last_active" in row.keys()
        else row["last_chat_at"],
        "chatCount": row["chat_count"] if "chat_count" in row.keys() else 0,
    }
    if full:
        character["card"] = _loads(row["card_json"], None)
    return character


_CHAT_COUNT = (
    "(SELECT COUNT(*) FROM chat_cast c JOIN chat_threads t ON t.id = c.thread_id"
    " WHERE c.character_id = characters.id AND t.archived = 0) AS chat_count,"
    # Recency follows the character's chats, so continuing an old chat moves it up.
    " MAX(COALESCE(last_chat_at, 0), COALESCE((SELECT MAX(COALESCE(t.updated_at, t.created_at))"
    " FROM chat_cast c JOIN chat_threads t ON t.id = c.thread_id"
    " WHERE c.character_id = characters.id), 0)) AS last_active"
)


def list_characters() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT *, {_CHAT_COUNT} FROM characters"
            " ORDER BY COALESCE(NULLIF(last_active, 0), updated_at) DESC, name"
        ).fetchall()
    return [_character_from_row(row, full = False) for row in rows]


def get_character(character_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            f"SELECT *, {_CHAT_COUNT} FROM characters WHERE id = ?", (character_id,)
        ).fetchone()
    return _character_from_row(row) if row else None


def get_characters(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT *, {_CHAT_COUNT} FROM characters WHERE id IN ({placeholders})", ids
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
        write_image("characters", character_id, avatar)
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
                        delete_image("characters", character_id)
                    return None
            (count,) = conn.execute("SELECT COUNT(*) FROM characters").fetchone()
            if count >= MAX_CHARACTERS:
                raise LimitReached(f"An account can hold up to {MAX_CHARACTERS} characters.")
            columns = ", ".join(values)
            conn.execute(
                f"INSERT INTO characters ({columns}) VALUES ({', '.join('?' for _ in values)})",
                list(values.values()),
            )
    except BaseException:
        if avatar:
            delete_image("characters", character_id)
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


def set_character_avatar(
    character_id: str, data: Optional[bytes], media_type: Optional[str]
) -> Optional[dict]:
    with _connect() as conn:
        if (
            conn.execute("SELECT 1 FROM characters WHERE id = ?", (character_id,)).fetchone()
            is None
        ):
            return None
    if data:
        write_image("characters", character_id, data)
    else:
        delete_image("characters", character_id)
    with _connect("BEGIN IMMEDIATE") as conn:
        conn.execute(
            "UPDATE characters SET avatar_type = ?, avatar_rev = avatar_rev + 1, updated_at = ? WHERE id = ?",
            (media_type if data else None, now_ms(), character_id),
        )
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
        delete_image("characters", character_id)
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
            raise LimitReached(f"An account can hold up to {MAX_PERSONAS} personas.")
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
    if get_persona(persona_id) is None:
        return None
    if data:
        write_image("personas", persona_id, data)
    else:
        delete_image("personas", persona_id)
    with _connect("BEGIN IMMEDIATE") as conn:
        conn.execute(
            "UPDATE personas SET avatar_type = ?, avatar_rev = avatar_rev + 1, updated_at = ? WHERE id = ?",
            (media_type if data else None, now_ms(), persona_id),
        )
    return get_persona(persona_id)


def delete_persona(persona_id: str) -> bool:
    with _connect("BEGIN IMMEDIATE") as conn:
        row = conn.execute("SELECT is_default FROM personas WHERE id = ?", (persona_id,)).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM personas WHERE id = ?", (persona_id,))
        if row["is_default"]:
            conn.execute(
                "UPDATE personas SET is_default = 1 WHERE id ="
                " (SELECT id FROM personas ORDER BY created_at LIMIT 1)"
            )
    delete_image("personas", persona_id)
    return True


_MACRO_PATTERNS = (
    (re.compile(r"\{\{\s*char\s*\}\}|<BOT>", re.IGNORECASE), "char"),
    (re.compile(r"\{\{\s*user\s*\}\}|<USER>", re.IGNORECASE), "user"),
)


def fill_names(text: str, char: str, user: str) -> str:
    for pattern, which in _MACRO_PATTERNS:
        text = pattern.sub((char if which == "char" else user).replace("\\", "\\\\"), text)
    return text


def _text_content(text: str) -> list[dict]:
    return [{"type": "text", "text": text}]


def chat_title(names: list[str]) -> str:
    if not names:
        return "Character chat"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} & {names[1]}"
    return f"{', '.join(names[:-1])} & {names[-1]}"


def create_character_chat(
    *,
    character_ids: list[str],
    persona_id: Optional[str],
    user_name: str,
    project_id: Optional[str] = None,
    settings: Optional[dict] = None,
    title: Optional[str] = None,
    greetings: Optional[dict[str, list[str]]] = None,
) -> dict:
    """Makes the thread, its cast and scene, then the opening turn with every member's greetings.

    Greetings hang under a hidden opening user turn so alternate greetings are ordinary sibling
    replies (and the history starts with a user turn, which strict chat templates require).
    """
    requested = list(dict.fromkeys(character_ids))
    characters = get_characters(requested)
    # Every requested member must exist: a group must not quietly become a smaller chat.
    if not characters or len(characters) != len(requested):
        raise LookupError("character")
    if len(characters) > MAX_CAST:
        raise LimitReached(f"A chat can have up to {MAX_CAST} characters.")
    persona = get_persona(persona_id) if persona_id else None
    speaker_name = (persona or {}).get("name") or user_name or "User"
    thread_id = str(uuid.uuid4())
    now = now_ms()
    upsert_chat_thread(
        {
            "id": thread_id,
            "title": title or chat_title([c["name"] for c in characters]),
            "modelType": "base",
            "modelId": "",
            "projectId": project_id,
            "createdAt": now,
            "updatedAt": now,
            "settings": dict(CHARACTER_CHAT_SETTINGS),
        }
    )
    greeting_ids: list[str] = []
    try:
        with _connect("BEGIN IMMEDIATE") as conn:
            for position, character in enumerate(characters):
                conn.execute(
                    "INSERT INTO chat_cast (thread_id, character_id, position, muted) VALUES (?, ?, ?, 0)",
                    (thread_id, character["id"], position),
                )
                conn.execute(
                    "UPDATE characters SET last_chat_at = ? WHERE id = ?", (now, character["id"])
                )
            conn.execute(
                "INSERT INTO chat_scene (thread_id, persona_id, settings_json, updated_at) VALUES (?, ?, ?, ?)",
                (thread_id, persona["id"] if persona else None, _dumps(settings or {}), now),
            )
        created = now
        parent_id = str(uuid.uuid4())
        upsert_chat_message(
            {
                "id": parent_id,
                "threadId": thread_id,
                "parentId": None,
                "role": "user",
                "content": _text_content(OPENING_TEXT),
                "metadata": {"roleplay": {"kind": "opening"}},
                "createdAt": created,
            }
        )
        group = len(characters) > 1
        for index, character in enumerate(characters):
            source = [character["firstMes"], *character["alternateGreetings"]]
            resolved = (greetings or {}).get(character["id"])
            if resolved is None or len(resolved) != len(source):
                resolved = [fill_names(g, character["name"], speaker_name) for g in source]
            texts = [g for g in resolved if g and g.strip()]
            if group and index > 0:
                # A hidden turn marker keeps each member's greeting answering a user turn.
                marker_id = str(uuid.uuid4())
                created += 1
                upsert_chat_message(
                    {
                        "id": marker_id,
                        "threadId": thread_id,
                        "parentId": parent_id,
                        "role": "user",
                        "content": _text_content(f"[{character['name']}'s turn]"),
                        "metadata": {"roleplay": {"kind": "turn", "speakerId": character["id"]}},
                        "createdAt": created,
                    }
                )
                parent_id = marker_id
            if not texts:
                continue
            first_id = None
            # Alternates first: with no saved branch choice a chat opens on its newest row, which
            # should be the main greeting, here and on every other device.
            order = [*range(1, len(texts)), 0]
            for greeting_index in order:
                greeting = texts[greeting_index]
                message_id = str(uuid.uuid4())
                created += 1
                upsert_chat_message(
                    {
                        "id": message_id,
                        "threadId": thread_id,
                        "parentId": parent_id,
                        "role": "assistant",
                        "content": _text_content(greeting),
                        "metadata": {
                            "roleplay": {
                                "kind": "reply",
                                "speakerId": character["id"],
                                "speakerName": character["name"],
                                "greeting": greeting_index,
                            }
                        },
                        "createdAt": created,
                    }
                )
                if greeting_index == 0:
                    first_id = message_id
            greeting_ids.append(first_id)  # type: ignore[arg-type]
            parent_id = first_id  # type: ignore[assignment]
    except BaseException:
        delete_chat_threads([thread_id])
        raise
    return {"threadId": thread_id, "greetingIds": greeting_ids, "headId": parent_id}


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


class NoSpeakerLeft(ValueError):
    """A group change that would leave every member muted."""


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
                    raise LookupError("character")
                cast.update((i, False) for i in added)
            # A member another tab removed has nothing left to mute.
            cast.update({i: bool(v) for i, v in (mute or {}).items() if i in cast})
            if len(cast) > MAX_CAST:
                raise LimitReached(f"A chat can have up to {MAX_CAST} characters.")
            if len(cast) > 1 and all(cast.values()):
                raise NoSpeakerLeft("A group chat needs at least one character who is not muted.")
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
