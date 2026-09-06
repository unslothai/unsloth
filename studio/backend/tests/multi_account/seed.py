# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Frozen pre-account data. Seeding uses sqlite and files, never product storage APIs."""

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

LEGACY_REF = "cc0cdab40e"
PASSWORD = "legacy-owner-password"
SALT = "0123456789abcdef0123456789abcdef"
JWT_SECRET = "legacy-jwt-secret-that-is-longer-than-32-bytes"
OLD_AUTH_COLUMNS = (
    "id",
    "username",
    "password_salt",
    "password_hash",
    "jwt_secret",
    "must_change_password",
)
OLD_AUTH_SCHEMA = """
CREATE TABLE auth_user (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_salt TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    jwt_secret TEXT NOT NULL,
    must_change_password INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE app_secrets (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
THREAD_ID = "legacy-thread"
MESSAGE_ID = "legacy-message"
SENTINEL = "Owner's saved conversation: café / 日本語"


def seed_studio_db(path: Path, *, populated: bool = True) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(Path(__file__).with_name("legacy_studio_schema.sql").read_text())
        if populated:
            conn.execute(
                "INSERT INTO chat_threads (id,title,model_type,model_id,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?)",
                (THREAD_ID, SENTINEL, "base", "legacy/model", 1000, 1000),
            )
            conn.execute(
                "INSERT INTO chat_messages (id,thread_id,role,content_json,created_at) "
                "VALUES (?,?,?,?,?)",
                (
                    MESSAGE_ID,
                    THREAD_ID,
                    "user",
                    json.dumps([{"type": "text", "text": SENTINEL}]),
                    1000,
                ),
            )
            conn.execute(
                "INSERT INTO chat_settings VALUES (?,?,?)", ("theme", '"dark"', "2026-01-01")
            )
            conn.execute(
                "INSERT INTO app_settings VALUES (?,?,?)",
                ("legacy-owner-setting", '{"preserve":true}', "2026-01-01"),
            )
            # Fixed AES-GCM vector, generated once with key bytes(range(32)), nonce bytes(range(12)),
            # AAD b'unsloth-studio-credential\0hf_token\0default', plaintext b'hf_legacy_private'.
            conn.execute(
                "INSERT INTO credential_secrets VALUES (?,?,?,?,?,?,?)",
                (
                    "hf_token",
                    "default",
                    1,
                    bytes(range(12)),
                    bytes.fromhex(CREDENTIAL_HEX),
                    "2026-01-01",
                    "2026-01-01",
                ),
            )
        conn.execute("INSERT OR REPLACE INTO chat_attachment_inventory_state VALUES (1,1,0,1000)")
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


# Filled with a frozen ciphertext, rather than invoking any encryption/storage API when seeding.
CREDENTIAL_HEX = "2f648977a082a378f41ee7f9d89f1919e62d7f5cd10fc5977ef1f485c413ea1920"


def seed_legacy_install(home: Path) -> dict[str, bytes]:
    auth_path = home / "auth" / "auth.db"
    auth_path.parent.mkdir(parents = True)
    with closing(sqlite3.connect(auth_path)) as conn:
        conn.executescript(OLD_AUTH_SCHEMA)
        digest = hashlib.pbkdf2_hmac("sha256", PASSWORD.encode(), SALT.encode(), 100_000).hex()
        conn.execute(
            "INSERT INTO auth_user VALUES (1,?,?,?,?,0)",
            ("unsloth", SALT, digest, JWT_SECRET),
        )
        conn.execute(
            "INSERT INTO app_secrets VALUES (?,?)",
            ("credential_encryption_key_v1", bytes(range(32)).hex()),
        )
        conn.commit()
    seed_studio_db(home / "studio.db")
    rag_path = home / "rag" / "rag.db"
    rag_path.parent.mkdir(parents = True)
    with closing(sqlite3.connect(rag_path)) as conn:
        conn.execute(
            "CREATE TABLE knowledge_bases (id TEXT PRIMARY KEY, name TEXT NOT NULL, "
            "description TEXT, embedding_model TEXT, created_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO knowledge_bases VALUES ('legacy-kb',?,NULL,NULL,'2026-01-01')", (SENTINEL,)
        )
        conn.commit()
    for leaf in (
        "outputs/my-finetune/checkpoint-100/adapter_model.safetensors",
        "sandbox/legacy-thread/workspace/notes.txt",
        "rag/documents/legacy-document/original.pdf",
        "assets/uploads/legacy-dataset.jsonl",
    ):
        path = home / leaf
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"\x00legacy\r\n" + leaf.encode() + b"\xff")
    return {
        str(path.relative_to(home)): path.read_bytes() for path in home.rglob("*") if path.is_file()
    }


def old_auth_row(path: Path) -> tuple:
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute(
            f"SELECT {','.join(OLD_AUTH_COLUMNS)} FROM auth_user WHERE username = ?", ("unsloth",)
        ).fetchone()
