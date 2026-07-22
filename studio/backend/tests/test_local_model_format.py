# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for local GGUF ``model_format`` classification (PR #6364 follow-up).

Suffixless GGUF folders (custom folders / LM Studio) carry no ``-GGUF`` name
hint, so the scanners must surface ``model_format = "gguf"`` for the UI to route
them through the GGUF load path. The rule, shared by ``_dir_model_format`` and
``_scan_models_dir``: a directory is GGUF-format when it holds ``.gguf`` files
and no non-GGUF weights (``.safetensors`` / ``.bin``); a stray ``config.json``
must not disqualify it.

No GPU/network: only file names and sizes are inspected.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# Keep runnable without optional logging deps (mirrors the sibling tests).
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.models as models_route


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"\0")
    return path


def test_dir_model_format_gguf_only(tmp_path):
    d = tmp_path / "model"
    _touch(d / "model-Q4_K_M.gguf")
    assert models_route._dir_model_format(d) == "gguf"


def test_dir_model_format_gguf_with_config_is_still_gguf(tmp_path):
    # A config.json alongside the .gguf must not flip it to non-GGUF.
    d = tmp_path / "model"
    _touch(d / "config.json")
    _touch(d / "model-Q4_K_M.gguf")
    assert models_route._dir_model_format(d) == "gguf"


def test_dir_model_format_mixed_weights_is_not_gguf(tmp_path):
    # Real safetensors weights present -> not a GGUF folder.
    d = tmp_path / "model"
    _touch(d / "model.safetensors")
    _touch(d / "model-Q4_K_M.gguf")
    assert models_route._dir_model_format(d) is None


def test_dir_model_format_no_gguf(tmp_path):
    d = tmp_path / "model"
    _touch(d / "config.json")
    _touch(d / "model.safetensors")
    assert models_route._dir_model_format(d) is None


def test_dir_model_format_ignores_tokenizer_bin(tmp_path):
    # A companion tokenizer.bin is not a weight file, so a GGUF folder shipping
    # one is still GGUF (not misread as a plain .bin checkpoint).
    d = tmp_path / "model"
    _touch(d / "tokenizer.bin")
    _touch(d / "model-Q4_K_M.gguf")
    assert models_route._dir_model_format(d) == "gguf"


def test_dir_model_format_weight_bin_is_not_gguf(tmp_path):
    # A real PyTorch weight .bin alongside a .gguf means mixed weights -> None.
    d = tmp_path / "model"
    _touch(d / "pytorch_model.bin")
    _touch(d / "model-Q4_K_M.gguf")
    assert models_route._dir_model_format(d) is None


def test_scan_models_dir_classifies_gguf_with_config(tmp_path):
    root = tmp_path / "models"
    # GGUF repo that also ships a config.json (the regression case).
    _touch(root / "gguf_repo" / "config.json")
    _touch(root / "gguf_repo" / "model-Q4_K_M.gguf")
    # A plain safetensors checkpoint stays non-GGUF.
    _touch(root / "st_repo" / "config.json")
    _touch(root / "st_repo" / "model.safetensors")
    # A standalone .gguf file is GGUF.
    _touch(root / "loose.gguf")

    fmt = {Path(m.path).name: m.model_format for m in models_route._scan_models_dir(root)}

    assert fmt["gguf_repo"] == "gguf"
    assert fmt["st_repo"] is None
    assert fmt["loose.gguf"] == "gguf"


def test_scan_models_dir_classifies_root_gguf_with_config(tmp_path):
    # Custom scan folders can point directly at a GGUF repo, not only at a
    # parent directory that contains model repos.
    root = tmp_path / "SuffixlessRepo"
    _touch(root / "config.json")
    _touch(root / "model-Q4_K_M.gguf")

    [row] = models_route._scan_models_dir(root)

    assert row.path == str(root)
    assert row.model_format == "gguf"
