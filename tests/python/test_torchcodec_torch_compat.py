# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""torch / torchcodec ABI guardrails (unslothai/unsloth#7225)."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
IMPORT_FIXES_PATH = REPO_ROOT / "unsloth" / "import_fixes.py"


def _load_import_fixes_module():
    spec = importlib.util.spec_from_file_location(
        "unsloth_import_fixes_under_test",
        IMPORT_FIXES_PATH,
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pyproject_declares_audio_extra_with_torchcodec_pin():
    text = PYPROJECT.read_text(encoding = "utf-8")
    assert "audio = [" in text
    assert "torchcodec>=0.10.0,<0.11.0" in text


def test_torchcodec_matrix_matches_notebook_validator():
    from scripts import notebook_validator as nv
    fixes = _load_import_fixes_module()
    assert fixes._TORCH_TORCHCODEC_MINORS == nv.TORCH_TORCHCODEC


def test_torch210_rejects_torchcodec_011(monkeypatch):
    import importlib.metadata
    import torch

    fixes = _load_import_fixes_module()
    monkeypatch.setattr(torch, "__version__", "2.10.0+cu128", raising = False)
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _name: "0.11.0",
    )

    hint = fixes._torchcodec_version_mismatch_hint()
    assert hint is not None
    assert "torchcodec 0.11.0" in hint
    assert "unsloth[audio]" in hint


def test_torch210_accepts_torchcodec_010(monkeypatch):
    import importlib.metadata
    import torch

    fixes = _load_import_fixes_module()
    monkeypatch.setattr(torch, "__version__", "2.10.0+cu128", raising = False)
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _name: "0.10.0+cu128",
    )

    assert fixes._torchcodec_version_mismatch_hint() is None
