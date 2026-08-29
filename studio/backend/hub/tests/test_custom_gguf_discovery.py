# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
from pathlib import Path

import pytest

from hub.services.models import local_inventory
from hub.utils import gguf


def _write_gguf(path: Path, size: int = 4) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"G" * size)
    return path


def _custom_rows(*roots: Path):
    rows = []
    for root in roots:
        rows.extend(
            local_inventory._promote_to_custom_source(row)
            for row in local_inventory._scan_custom_folder(root)
        )
    return local_inventory._dedupe_local_models(rows)


def _symlink_dir(alias: Path, target: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks unavailable")
    try:
        alias.symlink_to(target, target_is_directory = True)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")


def test_checkpoint_family_matches_quantized_and_split_variants():
    assert gguf.gguf_checkpoint_family("model-a-Q4_K_M.gguf") == "model-a"
    assert gguf.gguf_checkpoint_family("model-a-Q8_0.gguf") == "model-a"
    assert gguf.gguf_checkpoint_family("model-Q4_K_M-00001-of-00002.gguf") == "model"
    assert gguf.gguf_checkpoint_family("model-Q4_K_M-00002-of-00002.gguf") == "model"


def test_checkpoint_family_keeps_distinct_models_separate():
    assert gguf.gguf_checkpoint_family("model-a-Q4_K_M.gguf") == "model-a"
    assert gguf.gguf_checkpoint_family("model-b-Q4_K_M.gguf") == "model-b"
    assert gguf.gguf_checkpoint_family("alpha.gguf") == "alpha"
    assert gguf.gguf_checkpoint_family("beta.gguf") == "beta"


def test_checkpoint_family_normalizes_quant_directories_and_windows_separators():
    assert gguf.gguf_checkpoint_family("Q4_K_M/model.gguf") == "model"
    assert gguf.gguf_checkpoint_family(r"Q8_0\model.gguf") == "model"
    assert gguf.gguf_checkpoint_family("Q4_K_M.gguf") is None


def test_same_checkpoint_quantizations_use_one_grouped_row(tmp_path):
    root = tmp_path / "root"
    model = root / "model-a"
    _write_gguf(model / "model-a-Q4_K_M.gguf")
    _write_gguf(model / "model-a-Q8_0.gguf")

    assert [Path(row.path) for row in _custom_rows(root)] == [model]


def test_distinct_checkpoints_sharing_a_quant_use_file_rows(tmp_path):
    root = tmp_path / "root"
    holder = root / "publisher"
    model_a = _write_gguf(holder / "model-a-Q4_K_M.gguf")
    model_b = _write_gguf(holder / "model-b-Q4_K_M.gguf")

    rows = _custom_rows(root)

    assert {Path(row.path) for row in rows} == {model_a, model_b}
    assert all(not row.capabilities.requires_variant for row in rows)


def test_unqualified_distinct_checkpoints_use_file_rows(tmp_path):
    root = tmp_path / "root"
    holder = root / "holder"
    alpha = _write_gguf(holder / "alpha.gguf")
    beta = _write_gguf(holder / "beta.gguf")

    assert {Path(row.path) for row in _custom_rows(root)} == {alpha, beta}


def test_overlapping_parent_and_model_roots_keep_the_grouped_row(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    _write_gguf(model / "model-Q4_K_M.gguf")

    assert [Path(row.path) for row in _custom_rows(root, model)] == [model]


def test_symlinked_parent_and_real_model_roots_keep_one_row(tmp_path):
    real_model = tmp_path / "outside" / "model"
    _write_gguf(real_model / "model-Q4_K_M.gguf")
    _write_gguf(real_model / "model-Q8_0.gguf")
    alias_root = tmp_path / "aliases"
    alias_root.mkdir()
    alias_model = alias_root / "model"
    _symlink_dir(alias_model, real_model)

    rows = _custom_rows(alias_root, real_model)

    assert len(rows) == 1
    assert Path(rows[0].path).resolve() == real_model.resolve()


def test_symlinked_and_real_parent_roots_dedupe_group_rows(tmp_path):
    real_root = tmp_path / "real"
    real_model = real_root / "model"
    _write_gguf(real_model / "model-Q4_K_M.gguf")
    _write_gguf(real_model / "model-Q8_0.gguf")
    alias_root = tmp_path / "aliases"
    alias_root.mkdir()
    _symlink_dir(alias_root / "model", real_model)

    rows = _custom_rows(real_root, alias_root)

    assert len(rows) == 1
    assert Path(rows[0].path).resolve() == real_model.resolve()
