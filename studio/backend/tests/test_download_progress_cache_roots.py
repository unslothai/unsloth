# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
import types
from pathlib import Path


class _DummyLogger:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


if "structlog" not in sys.modules:
    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

try:
    __import__("loggers")
except Exception:
    sys.modules.pop("loggers", None)
    loggers_stub = types.ModuleType("loggers")
    loggers_stub.get_logger = lambda *args, **kwargs: _DummyLogger()
    sys.modules["loggers"] = loggers_stub

from utils import hf_cache_scan


def _repo_dir(root: Path, repo_type: str, repo_id: str) -> Path:
    repo_dir = root / f"{repo_type}s--{repo_id.replace('/', '--')}"
    (repo_dir / "blobs").mkdir(parents = True)
    return repo_dir


def _patch_cache_roots(monkeypatch, active_root: Path, roots: list[Path]) -> None:
    monkeypatch.setattr(
        hf_cache_scan,
        "_hf_cache_root",
        lambda create = False: active_root,
    )
    monkeypatch.setattr(
        hf_cache_scan,
        "_hf_cache_roots",
        lambda: [active_root, *roots],
    )


def test_preferred_repo_cache_dirs_uses_active_root_when_present(
    tmp_path,
    monkeypatch,
):
    repo_id = "Org/Model"
    active = tmp_path / "active"
    legacy = tmp_path / "legacy"
    active.mkdir()
    legacy.mkdir()
    _patch_cache_roots(monkeypatch, active, [legacy])

    active_repo = _repo_dir(active, "model", repo_id)
    _repo_dir(legacy, "model", repo_id)

    result = hf_cache_scan.preferred_repo_cache_dirs("model", repo_id)

    assert result == [active_repo]


def test_preferred_repo_cache_dirs_falls_back_to_all_roots(
    tmp_path,
    monkeypatch,
):
    repo_id = "Org/Model"
    active = tmp_path / "active"
    legacy_a = tmp_path / "legacy-a"
    legacy_b = tmp_path / "legacy-b"
    active.mkdir()
    legacy_a.mkdir()
    legacy_b.mkdir()
    _patch_cache_roots(monkeypatch, active, [legacy_a, legacy_b])

    legacy_a_repo = _repo_dir(legacy_a, "model", repo_id)
    legacy_b_repo = _repo_dir(legacy_b, "model", repo_id)

    result = hf_cache_scan.preferred_repo_cache_dirs("model", repo_id)

    assert result == [legacy_a_repo, legacy_b_repo]


def test_preferred_repo_cache_dirs_can_force_active_root_before_creation(
    tmp_path,
    monkeypatch,
):
    repo_id = "Org/Model"
    active = tmp_path / "active"
    legacy = tmp_path / "legacy"
    active.mkdir()
    legacy.mkdir()
    _patch_cache_roots(monkeypatch, active, [legacy])

    _repo_dir(legacy, "model", repo_id)

    result = hf_cache_scan.preferred_repo_cache_dirs(
        "model",
        repo_id,
        force_active = True,
    )

    assert result == [active / "models--Org--Model"]


def test_select_best_cache_progress_does_not_sum_duplicate_roots():
    selected = hf_cache_scan.select_best_cache_progress([
        (60, 0, "legacy-a"),
        (60, 0, "legacy-b"),
    ])

    assert selected == (60, 0, "legacy-a")
