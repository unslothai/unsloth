# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Skips MTP drafter GGUFs and takes the largest snapshot, so a partial revision cannot undercount."""

from __future__ import annotations

import sys
import types
from pathlib import Path

if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.models as models_route


def _write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"\0" * size)
    return path


def test_resolves_quant_from_parent_directory_layout(tmp_path):
    root = tmp_path / "repo"
    f = _write(root / "BF16" / "model.gguf", 1234)

    path, total = models_route._resolve_quant_gguf(str(root), "BF16", is_local = True)

    assert path == str(f)
    assert total == 1234


def test_skips_mtp_drafter_for_main_weights(tmp_path):
    # Main Q8_0 weights next to a same-quant MTP drafter that sorts first by name.
    root = tmp_path / "repo"
    main = _write(root / "model-Q8_0.gguf", 100)
    _write(root / "MTP" / "model-Q8_0-MTP.gguf", 50)

    path, total = models_route._resolve_quant_gguf(str(root), "Q8_0", is_local = True)

    assert path == str(main)
    assert total == 100


def test_skips_dspark_drafter_for_main_weights(tmp_path):
    root = tmp_path / "repo"
    main = _write(root / "model-Q8_0.gguf", 100)
    _write(root / "dspark" / "dspark-model-Q8_0.gguf", 50)

    path, total = models_route._resolve_quant_gguf(str(root), "Q8_0", is_local = True)

    assert path == str(main)
    assert total == 100


def test_prefers_the_complete_snapshot(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    snaps = cache / "models--org--repo" / "snapshots"
    _write(snaps / "aaaa" / "model-Q4_K_M.gguf", 10)
    complete_first = _write(snaps / "bbbb" / "model-00001-of-00002-Q4_K_M.gguf", 30)
    _write(snaps / "bbbb" / "model-00002-of-00002-Q4_K_M.gguf", 40)

    monkeypatch.setattr(
        "utils.hf_cache_settings.known_hf_hub_caches",
        lambda: [cache],
    )

    path, total = models_route._resolve_quant_gguf("org/repo", "Q4_K_M", is_local = False)

    assert total == 70
    assert path == str(complete_first)


def test_returns_none_when_quant_absent(tmp_path):
    root = tmp_path / "repo"
    _write(root / "model-Q4_K_M.gguf", 100)

    path, total = models_route._resolve_quant_gguf(str(root), "Q8_0", is_local = True)

    assert path is None
    assert total == 0
