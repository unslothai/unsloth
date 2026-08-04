# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for provider model persistence (unslothai/unsloth#7281)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import storage.providers_db as providers_db


@pytest.fixture()
def isolated_providers_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "studio.db"
    monkeypatch.setattr(providers_db, "studio_db_path", lambda: db_path)
    monkeypatch.setattr(providers_db, "ensure_dir", lambda _path: None)
    providers_db._schema_ready = False
    yield db_path
    providers_db._schema_ready = False


def test_create_and_list_provider_models(isolated_providers_db: Path):
    providers_db.create_provider(
        id = "ollama1",
        provider_type = "ollama",
        display_name = "Home Ollama",
        base_url = "http://127.0.0.1:11434",
        models = ["llama3.2", "qwen2.5"],
        available_models = ["llama3.2", "qwen2.5", "mistral"],
    )

    row = providers_db.get_provider("ollama1")
    assert row is not None
    assert row["models"] == ["llama3.2", "qwen2.5"]
    assert row["available_models"] == ["llama3.2", "qwen2.5", "mistral"]
    assert row["studio_tool_execution"] == 0

    listed = providers_db.list_providers()
    assert len(listed) == 1
    assert listed[0]["models"] == ["llama3.2", "qwen2.5"]


def test_update_provider_models(isolated_providers_db: Path):
    providers_db.create_provider(
        id = "vllm1",
        provider_type = "vllm",
        display_name = "Remote vLLM",
        base_url = "http://studio-host:8000/v1",
        models = ["meta-llama/Llama-3.2-1B-Instruct"],
        available_models = ["meta-llama/Llama-3.2-1B-Instruct"],
        studio_tool_execution = True,
    )

    assert providers_db.update_provider(
        id = "vllm1",
        models = ["meta-llama/Llama-3.2-3B-Instruct"],
        available_models = [
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
        ],
        studio_tool_execution = False,
    )

    row = providers_db.get_provider("vllm1")
    assert row is not None
    assert row["models"] == ["meta-llama/Llama-3.2-3B-Instruct"]
    assert row["available_models"] == [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]
    assert row["studio_tool_execution"] == 0


def test_schema_migration_disables_tools_for_existing_connections(isolated_providers_db: Path):
    conn = sqlite3.connect(str(isolated_providers_db))
    conn.execute(
        """
        CREATE TABLE llm_providers (
            id TEXT NOT NULL PRIMARY KEY,
            provider_type TEXT NOT NULL,
            display_name TEXT NOT NULL,
            base_url TEXT NOT NULL,
            is_enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """INSERT INTO llm_providers
        (id, provider_type, display_name, base_url, created_at, updated_at)
        VALUES ('existing', 'ollama', 'Existing Ollama', 'http://localhost:11434/v1', 'now', 'now')"""
    )
    conn.commit()
    conn.close()

    row = providers_db.get_provider("existing")

    assert row is not None
    assert row["studio_tool_execution"] == 0
