# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path
import sys
import types

# Keep this test runnable in lightweight environments where optional logging
# deps are not installed.
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

from utils.paths.path_utils import (
    resolve_cached_repo_id_case,
    get_cache_case_resolution_stats,
    reset_cache_case_resolution_state,
)
import utils.paths.path_utils as path_utils


def _mk_cache_repo(cache_root: Path, repo_id: str) -> Path:
    repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
    repo_dir.mkdir(parents = True, exist_ok = True)
    return repo_dir


def test_resolve_cached_repo_id_case_exact_hit(tmp_path, monkeypatch):
    reset_cache_case_resolution_state()
    _mk_cache_repo(tmp_path, "Org/Model")
    monkeypatch.setattr(path_utils, "_hf_hub_cache_dir", lambda: tmp_path)

    resolved = resolve_cached_repo_id_case("Org/Model")

    assert resolved == "Org/Model"
    stats = get_cache_case_resolution_stats()
    assert stats["calls"] == 1
    assert stats["exact_hits"] == 1
    assert stats["variant_hits"] == 0


def test_resolve_cached_repo_id_case_variant_hit(tmp_path, monkeypatch):
    reset_cache_case_resolution_state()
    _mk_cache_repo(tmp_path, "Org/Model")
    monkeypatch.setattr(path_utils, "_hf_hub_cache_dir", lambda: tmp_path)

    resolved = resolve_cached_repo_id_case("org/model")

    assert resolved == "Org/Model"
    stats = get_cache_case_resolution_stats()
    assert stats["variant_hits"] == 1
    assert stats["tie_breaks"] == 0


def test_resolve_cached_repo_id_case_tie_break_deterministic(tmp_path, monkeypatch):
    reset_cache_case_resolution_state()
    _mk_cache_repo(tmp_path, "Org/Model")
    _mk_cache_repo(tmp_path, "org/model")
    monkeypatch.setattr(path_utils, "_hf_hub_cache_dir", lambda: tmp_path)

    resolved = resolve_cached_repo_id_case("oRg/mOdEl")

    # Deterministic rule: lexical sort of candidate repo ids.
    assert resolved == "Org/Model"
    stats = get_cache_case_resolution_stats()
    assert stats["variant_hits"] == 1
    assert stats["tie_breaks"] == 1


def test_resolve_cached_repo_id_case_no_cache_fallback(tmp_path, monkeypatch):
    reset_cache_case_resolution_state()
    monkeypatch.setattr(path_utils, "_hf_hub_cache_dir", lambda: tmp_path)

    resolved = resolve_cached_repo_id_case("Org/Missing")

    assert resolved == "Org/Missing"
    stats = get_cache_case_resolution_stats()
    assert stats["fallbacks"] == 1
    assert stats["variant_hits"] == 0
    assert stats["exact_hits"] == 0


def test_resolve_cached_repo_id_case_memoization(tmp_path, monkeypatch):
    reset_cache_case_resolution_state()
    _mk_cache_repo(tmp_path, "Org/Model")
    monkeypatch.setattr(path_utils, "_hf_hub_cache_dir", lambda: tmp_path)

    first = resolve_cached_repo_id_case("org/model")
    second = resolve_cached_repo_id_case("org/model")

    assert first == "Org/Model"
    assert second == "Org/Model"
    stats = get_cache_case_resolution_stats()
    assert stats["calls"] == 2
    assert stats["variant_hits"] == 1
    assert stats["memo_hits"] == 1


def test_resolve_cached_repo_id_case_late_cache_population(tmp_path, monkeypatch):
    """Regression guard: memoized fallback should not hide a later cache variant."""
    reset_cache_case_resolution_state()
    monkeypatch.setattr(path_utils, "_hf_hub_cache_dir", lambda: tmp_path)

    first = resolve_cached_repo_id_case("org/model")
    assert first == "org/model"

    # Simulate cache being populated after first miss (e.g. another code path/download).
    _mk_cache_repo(tmp_path, "Org/Model")

    second = resolve_cached_repo_id_case("org/model")

    # Desired behavior: second lookup should pick up the now-existing variant.
    assert second == "Org/Model"
