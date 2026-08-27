# Copyright 2026-present the Unforgettable contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""B-store schema. Vec0 is reserved for later; FTS5 is required."""

from __future__ import annotations

import sqlite3


def _require_fts5(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _unforgettable_fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS _unforgettable_fts5_probe")
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            "SQLite is missing FTS5. Unforgettable needs a CPython build with FTS5."
        ) from exc


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    _require_fts5(conn)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS namespaces (
            id TEXT NOT NULL PRIMARY KEY,
            name TEXT NOT NULL,
            admission TEXT NOT NULL DEFAULT 'auto',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS records (
            id TEXT NOT NULL PRIMARY KEY,
            namespace_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            provenance TEXT NOT NULL,
            confidence REAL,
            supersedes_id TEXT,
            source_episode_id TEXT,
            contact_tag TEXT,
            speaker TEXT NOT NULL DEFAULT 'model',
            speaker_label TEXT,
            warrant TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (namespace_id) REFERENCES namespaces(id),
            FOREIGN KEY (supersedes_id) REFERENCES records(id)
        );
        CREATE INDEX IF NOT EXISTS idx_records_namespace ON records(namespace_id);
        CREATE INDEX IF NOT EXISTS idx_records_status ON records(status);
        CREATE INDEX IF NOT EXISTS idx_records_kind ON records(kind);
        CREATE INDEX IF NOT EXISTS idx_records_provenance ON records(provenance);
        CREATE INDEX IF NOT EXISTS idx_records_speaker ON records(speaker);

        CREATE TABLE IF NOT EXISTS admissions_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id TEXT,
            decision TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        -- FTS5 over title+body. record_id is stored so we can join back.
        -- sqlite-vec / a records_vec table is a later addition, not a Phase 1 dep.
        CREATE VIRTUAL TABLE IF NOT EXISTS record_fts USING fts5(
            title,
            body,
            record_id UNINDEXED
        );

        CREATE TABLE IF NOT EXISTS rollouts (
            id TEXT NOT NULL PRIMARY KEY,
            episode_id TEXT NOT NULL,
            contact TEXT NOT NULL,
            outcome TEXT NOT NULL,
            summary TEXT NOT NULL,
            source_record_id TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_rollouts_episode ON rollouts(episode_id);

        CREATE TABLE IF NOT EXISTS compiled (
            source_record_id TEXT NOT NULL PRIMARY KEY,
            explicit INTEGER NOT NULL DEFAULT 0,
            compiled_at TEXT NOT NULL,
            FOREIGN KEY (source_record_id) REFERENCES records(id)
        );

        CREATE TABLE IF NOT EXISTS compiled_blocked (
            source_record_id TEXT NOT NULL PRIMARY KEY
        );

        CREATE TABLE IF NOT EXISTS retrieve_uses (
            id TEXT NOT NULL PRIMARY KEY,
            episode_id TEXT NOT NULL,
            record_id TEXT NOT NULL,
            contact TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_retrieve_uses_record ON retrieve_uses(record_id);
        CREATE INDEX IF NOT EXISTS idx_retrieve_uses_episode ON retrieve_uses(episode_id);

        CREATE TABLE IF NOT EXISTS inject_stats (
            id TEXT NOT NULL PRIMARY KEY,
            episode_id TEXT NOT NULL,
            contact TEXT NOT NULL,
            standing_chars INTEGER NOT NULL,
            retrieve_chars INTEGER NOT NULL,
            trajectory_chars INTEGER NOT NULL,
            total_chars INTEGER NOT NULL,
            compiled_ids TEXT NOT NULL,
            retrieved_ids TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_inject_stats_episode ON inject_stats(episode_id);

        CREATE TABLE IF NOT EXISTS packs (
            id TEXT NOT NULL PRIMARY KEY,
            created_at TEXT NOT NULL,
            n_train INTEGER NOT NULL,
            n_holdout INTEGER NOT NULL,
            include_sim INTEGER NOT NULL DEFAULT 0,
            report TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pack_items (
            id TEXT NOT NULL PRIMARY KEY,
            pack_id TEXT NOT NULL,
            role TEXT NOT NULL,
            source TEXT NOT NULL,
            source_id TEXT NOT NULL,
            episode_id TEXT,
            kind TEXT NOT NULL,
            provenance TEXT NOT NULL,
            contact TEXT NOT NULL DEFAULT 'world',
            messages TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (pack_id) REFERENCES packs(id)
        );
        CREATE INDEX IF NOT EXISTS idx_pack_items_pack ON pack_items(pack_id);

        CREATE TABLE IF NOT EXISTS adapters (
            id TEXT NOT NULL PRIMARY KEY,
            pack_id TEXT NOT NULL,
            status TEXT NOT NULL,
            backend TEXT NOT NULL,
            base_model TEXT NOT NULL,
            recipe TEXT NOT NULL,
            path TEXT NOT NULL,
            gguf_path TEXT,
            metrics TEXT,
            created_at TEXT NOT NULL,
            promoted_at TEXT,
            FOREIGN KEY (pack_id) REFERENCES packs(id)
        );
        CREATE INDEX IF NOT EXISTS idx_adapters_status ON adapters(status);
        """
    )
    _add_missing_columns(conn)


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(records)")}
    extras = {
        "confidence": "REAL",
        "supersedes_id": "TEXT",
        "source_episode_id": "TEXT",
        "contact_tag": "TEXT",
        "speaker": "TEXT NOT NULL DEFAULT 'model'",
        "speaker_label": "TEXT",
        "warrant": "TEXT",
    }
    added = set()
    for name, decl in extras.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE records ADD COLUMN {name} {decl}")
            added.add(name)
    if "speaker" in added:
        conn.execute(
            """
            UPDATE records SET speaker = CASE
                WHEN kind = 'directive' THEN 'user'
                WHEN provenance = 'world' THEN 'world'
                WHEN provenance = 'sim' THEN 'sim'
                WHEN provenance = 'mixed' THEN 'world'
                WHEN provenance = 'human' THEN 'user'
                ELSE 'model'
            END
            """
        )
    adapter_cols = {row[1] for row in conn.execute("PRAGMA table_info(adapters)")}
    if adapter_cols and "gguf_path" not in adapter_cols:
        conn.execute("ALTER TABLE adapters ADD COLUMN gguf_path TEXT")
