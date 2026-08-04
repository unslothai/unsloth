# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from pathlib import Path
import sys
import types as _types

import pytest


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

from utils.models.checkpoints import (
    list_preview_targets,
    preview_ref,
    resolve_preview_checkpoint,
)


def _make_run(outputs: Path) -> tuple[Path, Path]:
    run = outputs / "unsloth_SmolLM-135M_1775412608"
    run.mkdir(parents = True)
    (run / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "HuggingFaceTB/SmolLM-135M"})
    )
    ckpt = run / "checkpoint-60"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "HuggingFaceTB/SmolLM-135M"})
    )
    return run, ckpt


def _point_outputs_root_at(monkeypatch, outputs: Path) -> None:
    from utils.paths import storage_roots as _sr
    from utils.models import checkpoints as _ckpt

    monkeypatch.setattr(_sr, "outputs_root", lambda: outputs)
    # checkpoints imported outputs_root by name; patch that alias too (preview_ref uses it).
    monkeypatch.setattr(_ckpt, "outputs_root", lambda: outputs)


def test_resolve_main_adapter_and_checkpoint(tmp_path: Path, monkeypatch):
    outputs = tmp_path / "outputs"
    run, ckpt = _make_run(outputs)
    _point_outputs_root_at(monkeypatch, outputs)

    assert resolve_preview_checkpoint(run.name) == run
    assert resolve_preview_checkpoint(run.name, "checkpoint-60") == ckpt


def test_resolve_missing_raises_not_found(tmp_path: Path, monkeypatch):
    outputs = tmp_path / "outputs"
    _make_run(outputs)
    _point_outputs_root_at(monkeypatch, outputs)

    with pytest.raises(FileNotFoundError):
        resolve_preview_checkpoint("does-not-exist")
    (outputs / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_preview_checkpoint("empty")


def test_resolve_rejects_traversal(tmp_path: Path, monkeypatch):
    outputs = tmp_path / "outputs"
    _make_run(outputs)
    _point_outputs_root_at(monkeypatch, outputs)

    with pytest.raises(ValueError):
        resolve_preview_checkpoint("..", "etc")


def test_list_preview_targets_flattens_with_latest_flag(tmp_path: Path, monkeypatch):
    outputs = tmp_path / "outputs"
    run, _ = _make_run(outputs)
    _point_outputs_root_at(monkeypatch, outputs)

    targets = list_preview_targets(str(outputs))
    by_ref = {t["ref"]: t for t in targets}

    assert by_ref[run.name]["is_latest"] is True
    assert by_ref[run.name]["checkpoint"] is None
    assert by_ref[f"{run.name}/checkpoint-60"]["is_latest"] is False
    assert by_ref[f"{run.name}/checkpoint-60"]["checkpoint"] == "checkpoint-60"
    assert all(t["base_model"] == "HuggingFaceTB/SmolLM-135M" for t in targets)


def test_preview_ref_flat_run_is_basename(tmp_path: Path, monkeypatch):
    outputs = tmp_path / "outputs"
    run, _ = _make_run(outputs)
    _point_outputs_root_at(monkeypatch, outputs)

    assert preview_ref(str(run)) == run.name


def test_preview_ref_preserves_one_level_nesting(tmp_path: Path, monkeypatch):
    outputs = tmp_path / "outputs"
    _point_outputs_root_at(monkeypatch, outputs)
    nested = outputs / "experiments" / "run1"
    nested.mkdir(parents = True)
    (nested / "adapter_config.json").write_text("{}")

    # /p route supports run/checkpoint, so a single level of nesting survives.
    assert preview_ref(str(nested)) == "experiments/run1"


def test_preview_ref_none_for_unpreviewable_or_too_deep(tmp_path: Path, monkeypatch):
    outputs = tmp_path / "outputs"
    _point_outputs_root_at(monkeypatch, outputs)

    # Missing / no model artifact -> not previewable.
    assert preview_ref(None) is None
    empty = outputs / "empty"
    empty.mkdir(parents = True)
    assert preview_ref(str(empty)) is None

    # Too deep for the two-segment /p route -> no dead link.
    deep = outputs / "a" / "b" / "run"
    deep.mkdir(parents = True)
    (deep / "adapter_config.json").write_text("{}")
    assert preview_ref(str(deep)) is None

    # Outside outputs_root -> None.
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "adapter_config.json").write_text("{}")
    assert preview_ref(str(outside)) is None
