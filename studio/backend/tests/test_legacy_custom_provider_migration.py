# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sqlite3

from storage import providers_db


def test_legacy_openai_compatible_rows_migrate_only_from_builtin_labels(tmp_path, monkeypatch):
    db = tmp_path / "studio.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE llm_providers (
                id TEXT PRIMARY KEY,
                provider_type TEXT NOT NULL,
                display_name TEXT NOT NULL,
                base_url TEXT NOT NULL,
                is_enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        rows = [
            ("openai", "openai", " OpenAI ", "https://api.openai.com/v1///"),
            ("renamed-openai", "openai", "My OpenAI Key", "https://api.openai.com/v1"),
            ("preset-named-openai", "openai", "vLLM", "https://api.openai.com/v1"),
            (
                "azure-openai",
                "openai",
                "Team Azure",
                "https://team.openai.azure.com/openai/v1",
            ),
            ("url-custom", "openai", "OpenAI", " https://gateway.example/v1/ "),
            ("renamed-custom", "openai", "My Gateway", "http://localhost:9000/v1"),
            ("default-custom", "openai", "Custom", "https://gateway.example/v1"),
            ("official-named-custom", "openai", "Custom", "https://api.openai.com/v1"),
            ("empty-name", "openai", "", "https://api.openai.com/v1"),
            ("ambiguous", "openai", "My Gateway", ""),
            ("llama", "openai", " LLAMA.CPP ", "http://localhost:8080/v1"),
            ("vllm", "openai", "vLLM", "http://localhost:8000/v1"),
            ("preset-without-url", "openai", "Ollama", ""),
            ("anthropic", "anthropic", "Anthropic", "https://api.anthropic.com"),
        ]
        conn.executemany(
            "INSERT INTO llm_providers "
            "(id, provider_type, display_name, base_url, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'created', 'unchanged')",
            rows,
        )

    monkeypatch.setattr(providers_db, "studio_db_path", lambda: db)
    monkeypatch.setattr(providers_db, "ensure_dir", lambda _path: None)
    providers_db.reset_schema_state_for_tests()

    migrated = {row["id"]: row for row in providers_db.list_providers()}
    assert {provider_id: row["provider_type"] for provider_id, row in migrated.items()} == {
        "openai": "openai",
        "renamed-openai": "openai",
        "preset-named-openai": "openai",
        "azure-openai": "openai",
        "url-custom": "openai",
        "renamed-custom": "openai",
        "default-custom": "custom",
        "official-named-custom": "openai",
        "empty-name": "openai",
        "ambiguous": "openai",
        "llama": "llama_cpp",
        "vllm": "vllm",
        "preset-without-url": "ollama",
        "anthropic": "anthropic",
    }
    assert migrated["default-custom"]["api_type"] == "chat_completions"
    assert all(row["updated_at"] == "unchanged" for row in migrated.values())

    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO llm_providers "
            "(id, provider_type, display_name, base_url, api_type, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 'created', 'unchanged')",
            (
                "post-upgrade-openai",
                "openai",
                "Work OpenAI",
                "https://api.openai.com/v1",
                "responses",
            ),
        )

    providers_db.reset_schema_state_for_tests()
    restarted = {
        row["id"]: row["provider_type"] for row in providers_db.list_providers()
    }
    assert restarted["post-upgrade-openai"] == "openai"
    assert {provider_id: restarted[provider_id] for provider_id in migrated} == {
        provider_id: row["provider_type"] for provider_id, row in migrated.items()
    }
