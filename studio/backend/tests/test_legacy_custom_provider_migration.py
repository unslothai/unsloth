# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Legacy rows migrate by trusted endpoint and label, never a lookalike host."""

import sqlite3

from storage import providers_db


def test_legacy_rows_migrate_without_promoting_spoofed_hosts(tmp_path, monkeypatch):
    db = tmp_path / "studio.db"
    rows = [
        ("openai", "OpenAI", "https://api.openai.com/v1", "openai"),
        ("azure", "Custom", "https://team.services.ai.azure.com/openai/v1", "openai"),
        (
            "spoof",
            "Custom",
            "https://team.services.ai.azure.com.attacker.example/openai/v1",
            "custom",
        ),
        ("gateway", "Custom", "https://gateway.example/v1", "custom"),
        ("vllm", "vLLM", "http://localhost:8000/v1", "vllm"),
    ]
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE llm_providers (id TEXT PRIMARY KEY, provider_type TEXT, "
            "display_name TEXT, base_url TEXT, is_enabled INTEGER DEFAULT 1, "
            "created_at TEXT, updated_at TEXT)"
        )
        conn.executemany(
            "INSERT INTO llm_providers "
            "(id, provider_type, display_name, base_url, created_at, updated_at) "
            "VALUES (?, 'openai', ?, ?, 'created', 'unchanged')",
            [(key, label, url) for key, label, url, _ in rows],
        )
    monkeypatch.setattr(providers_db, "studio_db_path", lambda: db)
    monkeypatch.setattr(providers_db, "ensure_dir", lambda _path: None)
    providers_db.reset_schema_state_for_tests()
    migrated = {row["id"]: row for row in providers_db.list_providers()}
    assert {key: migrated[key]["provider_type"] for key, _, _, _ in rows} == {
        key: expected for key, _, _, expected in rows
    }
    assert all(row["api_type"] == "chat_completions" for row in migrated.values())
    assert all(row["updated_at"] == "unchanged" for row in migrated.values())
