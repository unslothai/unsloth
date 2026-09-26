# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A {run}/{checkpoint}/ GGUF export is keyed by a main GGUF, never by an mmproj or imatrix beside it."""

from pathlib import Path

import pytest

from utils.account_context import OWNER, run_as
from utils.models import model_config
from utils.paths import storage_roots


@pytest.fixture(autouse = True)
def studio(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))


def _checkpoint(run: str, *files: str) -> Path:
    checkpoint = run_as(OWNER, storage_roots.exports_root) / run / "checkpoint-1"
    checkpoint.mkdir(parents = True)
    for name in files:
        (checkpoint / name).write_bytes(b"GGUF")
    return checkpoint


def test_two_level_gguf_export_skips_companions():
    kept = _checkpoint("with-main", "mmproj-F16.gguf", "imatrix.gguf", "model.Q4_K_M.gguf")
    _checkpoint("companions-only", "mmproj-F16.gguf", "imatrix_unsloth.gguf")

    exported = run_as(OWNER, model_config.scan_exported_models)

    assert [(name, path, kind) for name, path, kind, _base in exported] == [
        ("with-main / checkpoint-1", str(kept / "model.Q4_K_M.gguf"), "gguf"),
    ]
