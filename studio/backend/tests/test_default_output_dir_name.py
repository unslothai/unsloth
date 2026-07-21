# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto-generated training output dir names stay inside outputs_root.

Regression for local-model training: a model loaded by absolute path (e.g.
``G:\\modelsAI\\...\\gemma-4-12B-it`` on a non-system drive) used to seed the
default run dir with that full path, so ``resolve_output_dir`` raised
``path escapes root`` because the result was not under ``<studio>/outputs``.
"""

import importlib.util
from pathlib import Path

import pytest


_BACKEND_DIR = Path(__file__).resolve().parent.parent


def _load_storage_roots():
    path = _BACKEND_DIR / "utils/paths/storage_roots.py"
    spec = importlib.util.spec_from_file_location("storage_roots_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repo_id_keeps_namespace():
    sr = _load_storage_roots()
    assert sr.default_run_dir_name("unsloth/gemma-3-4b") == "unsloth_gemma-3-4b"
    assert sr.default_run_dir_name("gemma-3-4b") == "gemma-3-4b"


def test_local_paths_collapse_to_basename():
    sr = _load_storage_roots()
    assert (
        sr.default_run_dir_name(r"G:\modelsAI\gguf\test\gemma-4-12B-it")
        == "gemma-4-12B-it"
    )
    assert sr.default_run_dir_name("/data/models/gemma-3-4b") == "gemma-3-4b"
    assert sr.default_run_dir_name("~/models/gemma-3-4b") == "gemma-3-4b"
    assert sr.default_run_dir_name("C:/Users/me/models/gemma-3-4b") == "gemma-3-4b"


def test_empty_falls_back_to_model():
    sr = _load_storage_roots()
    assert sr.default_run_dir_name("") == "model"
    assert sr.default_run_dir_name("   ") == "model"


def test_very_long_name_is_capped():
    sr = _load_storage_roots()
    name = sr.default_run_dir_name("a" * 500)
    assert 0 < len(name) <= 200


def test_derived_name_resolves_under_outputs_root(tmp_path, monkeypatch):
    sr = _load_storage_roots()
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(sr, "outputs_root", lambda: outputs)

    name = sr.default_run_dir_name(r"G:\modelsAI\gguf\test\gemma-4-12B-it")
    resolved = sr.resolve_output_dir(f"{name}_1781327234")
    assert resolved == outputs / "gemma-4-12B-it_1781327234"
    # No escape: the absolute G: source no longer leaks into the output path.
    assert "modelsAI" not in str(resolved)
