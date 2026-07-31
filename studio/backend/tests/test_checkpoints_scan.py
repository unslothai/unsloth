# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import sqlite3
import sys
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
sys.modules.setdefault("structlog", _types.ModuleType("structlog"))

from utils.models import checkpoints as checkpoints_module
from utils.training_runs import build_default_output_dir_name


def _make_history_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _setup_training_runs_table(db_path: Path) -> None:
    conn = _make_history_connection(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE training_runs (
                id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                output_dir TEXT,
                started_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _make_outputs_dir(tmp_path, monkeypatch) -> Path:
    studio_home = tmp_path / "studio-home"
    outputs_dir = studio_home / "outputs"
    outputs_dir.mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    return outputs_dir


def test_scan_checkpoints_uses_output_dir_history_for_base_model(tmp_path, monkeypatch):
    outputs_dir = _make_outputs_dir(tmp_path, monkeypatch)
    run_dir = outputs_dir / "custom-run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}")

    db_path = tmp_path / "studio.db"
    _setup_training_runs_table(db_path)
    conn = _make_history_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO training_runs (id, model_name, config_json, output_dir, started_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "run-1",
                "unsloth/Llama-3.2-3B-Instruct",
                "{}",
                str(run_dir.resolve()),
                "2026-04-09T00:00:00Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(
        checkpoints_module,
        "get_connection",
        lambda: _make_history_connection(db_path),
    )

    models = checkpoints_module.scan_checkpoints(outputs_dir = str(outputs_dir))

    assert models[0][2]["base_model"] == "unsloth/Llama-3.2-3B-Instruct"


def test_scan_checkpoints_matches_project_suffixed_default_dir_against_history(
    tmp_path, monkeypatch
):
    outputs_dir = _make_outputs_dir(tmp_path, monkeypatch)
    run_name = build_default_output_dir_name(
        "unsloth/Llama-3.2-3B-Instruct",
        "Customer Support",
        timestamp = 1771227800,
    )
    run_dir = outputs_dir / run_name
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}")

    db_path = tmp_path / "studio.db"
    _setup_training_runs_table(db_path)
    conn = _make_history_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO training_runs (id, model_name, config_json, output_dir, started_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "run-2",
                "unsloth/Llama-3.2-3B-Instruct",
                json.dumps({"project_name": "Customer Support"}),
                None,
                "2026-04-09T00:00:00Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(
        checkpoints_module,
        "get_connection",
        lambda: _make_history_connection(db_path),
    )

    models = checkpoints_module.scan_checkpoints(outputs_dir = str(outputs_dir))

    assert models[0][2]["base_model"] == "unsloth/Llama-3.2-3B-Instruct"


def test_scan_checkpoints_strips_project_suffix_without_history(tmp_path, monkeypatch):
    outputs_dir = _make_outputs_dir(tmp_path, monkeypatch)
    run_name = build_default_output_dir_name(
        "unsloth/Llama-3.2-3B-Instruct",
        "Customer Support",
        timestamp = 1771227800,
    )
    run_dir = outputs_dir / run_name
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}")

    db_path = tmp_path / "studio.db"
    _setup_training_runs_table(db_path)
    monkeypatch.setattr(
        checkpoints_module,
        "get_connection",
        lambda: _make_history_connection(db_path),
    )

    models = checkpoints_module.scan_checkpoints(outputs_dir = str(outputs_dir))

    assert models[0][2]["base_model"] == "unsloth/Llama-3.2-3B-Instruct"


def test_scan_checkpoints_preserves_project_marker_in_model_without_history(tmp_path, monkeypatch):
    outputs_dir = _make_outputs_dir(tmp_path, monkeypatch)
    run_name = build_default_output_dir_name(
        "org/foo__project-bar",
        timestamp = 1771227800,
    )
    run_dir = outputs_dir / run_name
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}")

    db_path = tmp_path / "studio.db"
    _setup_training_runs_table(db_path)
    monkeypatch.setattr(
        checkpoints_module,
        "get_connection",
        lambda: _make_history_connection(db_path),
    )

    models = checkpoints_module.scan_checkpoints(outputs_dir = str(outputs_dir))

    assert models[0][2]["base_model"] == "org/foo__project-bar"


def test_scan_checkpoints_preserves_legacy_folder_name_fallback(tmp_path, monkeypatch):
    outputs_dir = _make_outputs_dir(tmp_path, monkeypatch)
    run_dir = outputs_dir / "unsloth_Llama-3.2-3B-Instruct_1771227800"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}")

    db_path = tmp_path / "studio.db"
    _setup_training_runs_table(db_path)
    monkeypatch.setattr(
        checkpoints_module,
        "get_connection",
        lambda: _make_history_connection(db_path),
    )

    models = checkpoints_module.scan_checkpoints(outputs_dir = str(outputs_dir))

    assert models[0][2]["base_model"] == "unsloth/Llama-3.2-3B-Instruct"


def test_scan_checkpoints_prefers_exact_history_match_over_newer_suffix(tmp_path, monkeypatch):
    outputs_dir = _make_outputs_dir(tmp_path, monkeypatch)
    run_dir = outputs_dir / "unsloth_Test_1771227800"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}")

    copied_dir = tmp_path / "copied" / run_dir.name
    copied_dir.mkdir(parents = True)

    db_path = tmp_path / "studio.db"
    _setup_training_runs_table(db_path)
    conn = _make_history_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO training_runs (id, model_name, config_json, output_dir, started_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "run-exact",
                "correct/base",
                "{}",
                str(run_dir.resolve()),
                "2026-04-09T00:00:00Z",
            ),
        )
        conn.execute(
            """
            INSERT INTO training_runs (id, model_name, config_json, output_dir, started_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "run-suffix",
                "wrong/base",
                "{}",
                str(copied_dir.resolve()),
                "2026-04-10T00:00:00Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(
        checkpoints_module,
        "get_connection",
        lambda: _make_history_connection(db_path),
    )

    models = checkpoints_module.scan_checkpoints(outputs_dir = str(outputs_dir))

    assert models[0][2]["base_model"] == "correct/base"
