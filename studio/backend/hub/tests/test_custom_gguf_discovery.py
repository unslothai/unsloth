# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hub.services.models import local_inventory
from hub.utils import gguf
from utils.models.model_config import detect_gguf_model, _find_local_gguf_by_variant


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


def _symlink_file(alias: Path, target: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks unavailable")
    try:
        alias.symlink_to(target)
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
    assert gguf.gguf_checkpoint_family("Model-A-Q4_K_M.gguf") == "Model-A"


def test_checkpoint_family_normalizes_quant_directories_and_windows_separators():
    assert gguf.gguf_checkpoint_family("Q4_K_M/model.gguf") == "model"
    assert gguf.gguf_checkpoint_family(r"Q8_0\model.gguf") == "model"
    assert gguf.gguf_checkpoint_family("Q8_0/model-a-Q8_0.gguf") == "model-a"
    assert gguf.gguf_checkpoint_family("Q4_K_M.gguf") is None


def test_same_checkpoint_quantizations_use_one_grouped_row(tmp_path):
    root = tmp_path / "root"
    model = root / "model-a"
    q4 = _write_gguf(model / "model-a-Q4_K_M.gguf", 8)
    q8 = _write_gguf(model / "model-a-Q8_0.gguf", 16)

    assert [Path(row.path) for row in _custom_rows(root)] == [model]
    variants, _ = gguf.list_local_gguf_variants(str(model), model_root = str(root))
    assert {variant.quant for variant in variants} == {"Q4_K_M", "Q8_0"}
    assert Path(detect_gguf_model(str(model), model_root = str(root))) == q8
    assert Path(_find_local_gguf_by_variant(str(model), "Q4_K_M", model_root = str(root))) == q4


def test_distinct_checkpoints_sharing_a_quant_use_file_rows(tmp_path):
    root = tmp_path / "root"
    holder = root / "publisher"
    model_a = _write_gguf(holder / "model-a-Q4_K_M.gguf")
    model_b = _write_gguf(holder / "model-b-Q4_K_M.gguf")

    rows = _custom_rows(root)

    assert {Path(row.path) for row in rows} == {model_a, model_b}
    assert all(not row.capabilities.requires_variant for row in rows)


@pytest.mark.parametrize(
    ("first_name", "second_name"),
    [
        ("model-a-Q4_K_M.gguf", "model.a-Q4_K_M.gguf"),
        ("Model-A-Q4_K_M.gguf", "model-a-Q4_K_M.gguf"),
        ("model-a-Q4_K_M.gguf", "model.a-Q8_0.gguf"),
        ("Model-A-Q4_K_M.gguf", "model-a-Q8_0.gguf"),
    ],
)
def test_checkpoint_names_preserve_exact_identity(tmp_path, first_name, second_name):
    root = tmp_path / "root"
    holder = root / "holder"
    first = _write_gguf(holder / first_name)
    second = _write_gguf(holder / second_name)
    if first.samefile(second):
        pytest.skip("filesystem does not preserve the distinct names")

    rows = _custom_rows(root)

    assert {Path(row.path) for row in rows} == {first, second}


def test_unqualified_distinct_checkpoints_use_file_rows(tmp_path):
    root = tmp_path / "root"
    holder = root / "holder"
    alpha = _write_gguf(holder / "alpha.gguf")
    beta = _write_gguf(holder / "beta.gguf")

    assert {Path(row.path) for row in _custom_rows(root)} == {alpha, beta}


def test_bare_quant_filenames_use_the_parent_model_group(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    q4 = _write_gguf(model / "Q4_K_M.gguf")
    q8 = _write_gguf(model / "Q8_0.gguf")

    assert [Path(row.path) for row in _custom_rows(root)] == [model]
    variants, _ = gguf.list_local_gguf_variants(str(model), model_root = str(root))
    assert {Path(variant.filename).name for variant in variants} == {q4.name, q8.name}


def test_bare_quant_shards_use_the_parent_model_group(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    _write_gguf(model / "Q4_K_M-00001-of-00002.gguf")
    _write_gguf(model / "Q4_K_M-00002-of-00002.gguf")

    rows = _custom_rows(root)

    assert [Path(row.path) for row in rows] == [model]


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


def test_loose_file_symlink_dedupes_by_physical_identity(tmp_path):
    root = tmp_path / "root"
    target = _write_gguf(root / "model-Q4_K_M.gguf")
    alias = root / "latest.gguf"
    _symlink_file(alias, target)

    rows = _custom_rows(root)

    assert len(rows) == 1
    assert Path(rows[0].path).samefile(target)


def test_loose_symlink_into_grouped_model_does_not_add_a_row(tmp_path):
    grouped_root = tmp_path / "grouped-root"
    model = grouped_root / "model"
    target = _write_gguf(model / "model-Q4_K_M.gguf")
    loose_root = tmp_path / "loose-root"
    loose_root.mkdir()
    _symlink_file(loose_root / "model-Q4_K_M.gguf", target)

    rows = _custom_rows(grouped_root, loose_root)

    assert [Path(row.path) for row in rows] == [model]


def test_renamed_loose_symlink_into_grouped_model_does_not_add_a_row(tmp_path):
    grouped_root = tmp_path / "grouped-root"
    model = grouped_root / "model"
    target = _write_gguf(model / "model-Q4_K_M.gguf")
    loose_root = tmp_path / "loose-root"
    loose_root.mkdir()
    _symlink_file(loose_root / "latest.gguf", target)

    rows = _custom_rows(grouped_root, loose_root)

    assert [Path(row.path) for row in rows] == [model]


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


def test_split_gguf_shards_stay_grouped(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    _write_gguf(model / "model-Q4_K_M-00001-of-00002.gguf")
    _write_gguf(model / "model-Q4_K_M-00002-of-00002.gguf")

    assert [Path(row.path) for row in _custom_rows(root)] == [model]


def test_pre_quant_split_markers_stay_grouped(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    _write_gguf(model / "model-00001-of-00002-Q4_K_M.gguf")
    _write_gguf(model / "model-00002-of-00002-Q4_K_M.gguf")

    assert [Path(row.path) for row in _custom_rows(root)] == [model]


def test_minimax_h3_partitions_stay_grouped(tmp_path):
    root = tmp_path / "root"
    model = root / "minimax-h3"
    _write_gguf(model / "minimax_h3_fl2va-Q4_K_M.gguf")
    _write_gguf(model / "minimax_h3_ref2va-Q4_K_M.gguf")

    assert [Path(row.path) for row in _custom_rows(root)] == [model]


def test_symlinked_variant_stays_grouped_and_loadable(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    target = _write_gguf(tmp_path / "outside" / "model-Q4_K_M.gguf")
    model.mkdir(parents = True)
    _symlink_file(model / target.name, target)

    rows = _custom_rows(root)

    assert [Path(row.path) for row in rows] == [model]
    assert Path(detect_gguf_model(str(model), model_root = str(root))).samefile(target)


def test_loose_gguf_models_stay_separate(tmp_path):
    root = tmp_path / "root"
    alpha = _write_gguf(root / "alpha-Q4_K_M.gguf")
    beta = _write_gguf(root / "beta-Q8_0.gguf")

    assert {Path(row.path) for row in _custom_rows(root)} == {alpha, beta}


def test_direct_custom_model_root_keeps_loose_variants(tmp_path):
    root = tmp_path / "direct-model-root"
    q4 = _write_gguf(root / "model-Q4_K_M.gguf")
    q8 = _write_gguf(root / "model-Q8_0.gguf")

    assert {Path(row.path) for row in _custom_rows(root)} == {q4, q8}


def test_direct_custom_model_root_dedupes_split_shards(tmp_path):
    root = tmp_path / "direct-model-root"
    first = _write_gguf(root / "Q4_K_M-00001-of-00002.gguf", 11)
    _write_gguf(root / "Q4_K_M-00002-of-00002.gguf", 7)

    rows = _custom_rows(root)

    assert len(rows) == 1
    assert Path(rows[0].path) == first
    assert rows[0].size_bytes == 18
    assert Path(detect_gguf_model(rows[0].path, model_root = str(root))) == first


def test_direct_split_prefix_matching_follows_the_loader(tmp_path):
    root = tmp_path / "direct-model-root"
    first = _write_gguf(root / "Model-Q4_K_M-00001-of-00002.gguf", 11)
    _write_gguf(root / "model-Q4_K_M-00002-of-00002.gguf", 7)

    rows = _custom_rows(root)

    assert [(Path(row.path), row.size_bytes) for row in rows] == [(first, 18)]


def test_incomplete_direct_split_is_not_collapsed(tmp_path):
    root = tmp_path / "direct-model-root"
    first = _write_gguf(root / "model-Q4_K_M-00001-of-00003.gguf")
    third = _write_gguf(root / "model-Q4_K_M-00003-of-00003.gguf")

    rows = _custom_rows(root)

    assert {Path(row.path) for row in rows} == {first, third}


def test_grouped_split_reports_parts_without_counting_companions(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    _write_gguf(model / "model-F16-00001-of-00002.gguf", 11)
    _write_gguf(model / "model-F16-00002-of-00002.gguf", 7)
    _write_gguf(model / "model-mmproj-F16.gguf", 3)
    _write_gguf(model / "mtp-model.gguf", 2)

    rows = _custom_rows(root)
    variants, _has_vision = gguf.list_local_gguf_variants(str(model), model_root = str(root))

    assert len(rows) == 1
    assert [(variant.quant, variant.shard_count) for variant in variants] == [("F16", 2)]


def test_mixed_split_and_ordinary_models_stay_separate(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    _write_gguf(model / "alpha-F16-00001-of-00002.gguf")
    _write_gguf(model / "alpha-F16-00002-of-00002.gguf")
    _write_gguf(model / "beta-Q8_0.gguf")

    rows = _custom_rows(root)
    variants, _has_vision = gguf.list_local_gguf_variants(str(model), model_root = str(root))

    assert len(rows) == 2
    split_row = next(row for row in rows if Path(row.path).name.startswith("alpha-F16-00001"))
    ordinary_row = next(row for row in rows if Path(row.path).name == "beta-Q8_0.gguf")
    assert Path(split_row.path).name == "alpha-F16-00001-of-00002.gguf"
    assert Path(ordinary_row.path).name == "beta-Q8_0.gguf"
    assert {(variant.quant, variant.shard_count) for variant in variants} == {
        ("F16", 2),
        ("Q8_0", 0),
    }


def test_two_digit_split_like_names_stay_separate(tmp_path):
    root = tmp_path / "root"
    holder = root / "holder"
    names = ("model-Q4_K_M-01-of-02.gguf", "model-Q4_K_M-02-of-02.gguf")
    expected = {_write_gguf(holder / name) for name in names}

    rows = _custom_rows(root)

    assert {Path(row.path) for row in rows} == expected


def test_nested_independent_gguf_models_do_not_share_a_parent_group(tmp_path):
    root = tmp_path / "root"
    publisher = root / "publisher"
    loose = _write_gguf(publisher / "model-a-Q4_K_M.gguf")
    nested = publisher / "model-b"
    _write_gguf(nested / "model-b-Q8_0.gguf")

    rows = _custom_rows(root)

    assert {Path(row.path) for row in rows} == {loose, nested}
    variants, _ = gguf.list_local_gguf_variants(str(nested), model_root = str(root))
    assert [variant.quant for variant in variants] == ["Q8_0"]
    assert _find_local_gguf_by_variant(str(nested), "Q4_K_M", model_root = str(root)) is None


def test_nested_quant_directory_stays_in_the_parent_group(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    _write_gguf(model / "model-a-Q4_K_M.gguf")
    _write_gguf(model / "Q8_0" / "model-a-Q8_0.gguf")

    rows = _custom_rows(root)

    assert [Path(row.path) for row in rows] == [model]
    variants, _ = gguf.list_local_gguf_variants(str(model), model_root = str(root))
    assert {variant.quant for variant in variants} == {"Q4_K_M", "Q8_0"}


def test_overlapping_nested_quant_root_keeps_the_parent_group(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    _write_gguf(model / "model-a-Q4_K_M.gguf")
    quant_dir = model / "Q8_0"
    _write_gguf(quant_dir / "model-a-Q8_0.gguf")
    (quant_dir / "config.json").write_text("{}", encoding = "utf-8")

    rows = _custom_rows(root, quant_dir)

    assert [Path(row.path) for row in rows] == [model]


def test_quant_named_independent_model_root_stays_selectable(tmp_path):
    root = tmp_path / "root"
    parent = root / "parent"
    _write_gguf(parent / "model-a-Q4_K_M.gguf")
    (parent / "config.json").write_text("{}", encoding = "utf-8")
    child = parent / "Q8_0"
    _write_gguf(child / "model-b-Q8_0.gguf")
    (child / "config.json").write_text("{}", encoding = "utf-8")

    rows = _custom_rows(root, child)

    assert {Path(row.path) for row in rows} == {parent, child}


def test_symlinked_nested_quant_root_keeps_the_real_parent_group(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    _write_gguf(model / "model-a-Q4_K_M.gguf")
    quant_dir = model / "Q8_0"
    _write_gguf(quant_dir / "model-a-Q8_0.gguf")
    (quant_dir / "config.json").write_text("{}", encoding = "utf-8")
    alias_model = tmp_path / "alias-model"
    _symlink_dir(alias_model, model)

    rows = _custom_rows(root, alias_model / "Q8_0")

    assert [Path(row.path) for row in rows] == [model]


def test_overlapping_nested_independent_model_root_stays_selectable(tmp_path):
    root = tmp_path / "root"
    parent = root / "parent"
    _write_gguf(parent / "parent-Q4_K_M.gguf")
    (parent / "config.json").write_text("{}", encoding = "utf-8")
    child = parent / "child"
    _write_gguf(child / "child-Q8_0.gguf")
    (child / "config.json").write_text("{}", encoding = "utf-8")

    rows = _custom_rows(root, child)

    assert {Path(row.path) for row in rows} == {parent, child}


def test_overlapping_nested_independent_loose_root_stays_selectable(tmp_path):
    root = tmp_path / "root"
    parent = root / "parent"
    _write_gguf(parent / "parent-Q4_K_M.gguf")
    (parent / "config.json").write_text("{}", encoding = "utf-8")
    child = parent / "child"
    loose = _write_gguf(child / "child-Q8_0.gguf")

    rows = _custom_rows(root, child)

    assert {Path(row.path) for row in rows} == {parent, loose}


def test_aliased_nested_independent_loose_root_stays_selectable(tmp_path):
    root = tmp_path / "root"
    parent = root / "parent"
    _write_gguf(parent / "parent-Q4_K_M.gguf")
    (parent / "config.json").write_text("{}", encoding = "utf-8")
    child = parent / "child"
    loose = _write_gguf(child / "child-Q8_0.gguf")
    alias_child = tmp_path / "alias-child"
    _symlink_dir(alias_child, child)

    rows = _custom_rows(root, alias_child)

    assert {Path(row.path).resolve() for row in rows} == {parent.resolve(), loose.resolve()}


def test_mixed_nested_roots_dedupe_only_the_matching_quant_group(tmp_path):
    root = tmp_path / "root"
    parent = root / "parent"
    _write_gguf(parent / "model-a-Q4_K_M.gguf")
    (parent / "config.json").write_text("{}", encoding = "utf-8")
    quant_dir = parent / "Q8_0"
    _write_gguf(quant_dir / "model-a-Q8_0.gguf")
    (quant_dir / "config.json").write_text("{}", encoding = "utf-8")
    child = parent / "child"
    _write_gguf(child / "child-Q4_K_M.gguf")
    (child / "config.json").write_text("{}", encoding = "utf-8")

    rows = _custom_rows(root, quant_dir, child)

    assert {Path(row.path) for row in rows} == {parent, child}


def test_big_endian_companion_does_not_leave_a_duplicate_loose_row(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    _write_gguf(model / "model-Q4_K_M.gguf")
    _write_gguf(model / "model-Q4_K_M-be.gguf")

    rows = _custom_rows(root, model)

    assert [Path(row.path) for row in rows] == [model]


def test_mtp_companion_does_not_leave_a_duplicate_loose_row(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    _write_gguf(model / "model-Q4_K_M.gguf")
    _write_gguf(model / "MTP" / "model-Q8_0.gguf")

    rows = _custom_rows(root, model)

    assert [Path(row.path) for row in rows] == [model]


def test_nested_symlinked_variant_directory_stays_selectable(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    q4 = _write_gguf(model / "model-a-Q4_K_M.gguf")
    q8_dir = tmp_path / "outside" / "Q8_0"
    q8 = _write_gguf(q8_dir / "model-a-Q8_0.gguf")
    alias = model / "Q8_0"
    _symlink_dir(alias, q8_dir)

    rows = _custom_rows(root)

    assert {Path(row.path) for row in rows} == {q4, alias}
    variants, _ = gguf.list_local_gguf_variants(str(alias), model_root = str(root))
    assert [variant.quant for variant in variants] == ["Q8_0"]
    assert Path(_find_local_gguf_by_variant(str(alias), "Q8_0", model_root = str(root))).samefile(q8)


def test_windows_junction_child_is_suppressed_when_parent_walk_sees_it(tmp_path, monkeypatch):
    model = tmp_path / "root" / "model"
    q4 = _write_gguf(model / "model-a-Q4_K_M.gguf")
    outside = tmp_path / "outside" / "Q8_0"
    q8 = _write_gguf(outside / "model-a-Q8_0.gguf")
    child = model / "Q8_0"
    _symlink_dir(child, outside)
    linked_q8 = child / q8.name
    parent_row = SimpleNamespace(model_format = "gguf", partial = False, path = str(model))
    child_row = SimpleNamespace(model_format = "gguf", partial = False, path = str(child))

    monkeypatch.setattr(
        gguf,
        "iter_gguf_files",
        lambda directory, recursive = False: iter(
            [q4, linked_q8] if directory == model else [linked_q8]
        ),
    )

    assert gguf.suppress_grouped_gguf_file_rows([parent_row, child_row]) == [parent_row]


def test_windows_junction_independent_child_stays_selectable(tmp_path, monkeypatch):
    model = tmp_path / "root" / "model"
    q4 = _write_gguf(model / "model-a-Q4_K_M.gguf")
    outside = tmp_path / "outside" / "child"
    q8 = _write_gguf(outside / "child-Q8_0.gguf")
    child = model / "child"
    _symlink_dir(child, outside)
    linked_q8 = child / q8.name
    parent_row = SimpleNamespace(model_format = "gguf", partial = False, path = str(model))
    child_row = SimpleNamespace(model_format = "gguf", partial = False, path = str(child))

    monkeypatch.setattr(
        gguf,
        "iter_gguf_files",
        lambda directory, recursive = False: iter(
            [q4, linked_q8] if directory == model else [linked_q8]
        ),
    )

    assert gguf.suppress_grouped_gguf_file_rows([parent_row, child_row]) == [
        parent_row,
        child_row,
    ]


def test_lmstudio_publisher_model_layout_keeps_its_model_id(tmp_path):
    root = tmp_path / "lmstudio"
    model = root / "publisher" / "model"
    _write_gguf(model / "model-Q4_K_M.gguf")
    _write_gguf(model / "model-Q8_0.gguf")

    rows = local_inventory._scan_custom_folder(root)

    assert [(Path(row.path), row.model_id) for row in rows] == [(model, "publisher/model")]


def test_mixed_gguf_and_safetensors_keep_one_row_per_format(tmp_path):
    root = tmp_path / "root"
    model = root / "hybrid"
    _write_gguf(model / "hybrid-Q4_K_M.gguf")
    _write_gguf(model / "model.safetensors")
    (model / "config.json").write_text(json.dumps({"model_type": "llama"}))

    rows = _custom_rows(root)

    assert {(Path(row.path), row.model_format) for row in rows} == {
        (model, "gguf"),
        (model, "safetensors"),
    }


def test_incomplete_custom_models_stay_hidden(tmp_path):
    root = tmp_path / "root"
    partial = root / "partial"
    partial.mkdir(parents = True)
    (partial / "config.json").write_text("{}")
    _write_gguf(partial / "model.gguf.incomplete")
    complete = _write_gguf(root / "complete.gguf")

    assert [Path(row.path) for row in _custom_rows(root)] == [complete]


def test_mixed_case_gguf_suffixes_stay_grouped(tmp_path):
    root = tmp_path / "root"
    model = root / "model"
    _write_gguf(model / "model-Q4_K_M.GGUF")
    _write_gguf(model / "model-Q8_0.GguF")

    assert [Path(row.path) for row in _custom_rows(root)] == [model]


def test_symlinked_model_directory_stays_grouped(tmp_path):
    real_model = tmp_path / "outside" / "model"
    _write_gguf(real_model / "model-Q4_K_M.gguf")
    _write_gguf(real_model / "model-Q8_0.gguf")
    root = tmp_path / "root"
    root.mkdir()
    alias = root / "model-alias"
    _symlink_dir(alias, real_model)

    assert [Path(row.path) for row in _custom_rows(root)] == [alias]


def test_physical_identity_preserves_native_posix_names(tmp_path):
    if os.sep == "\\":
        pytest.skip("names are not distinct on Windows")
    backslash_root = tmp_path / "model\\one"
    nested_root = tmp_path / "model" / "one"
    trailing_root = tmp_path / "model "
    plain_root = tmp_path / "model"
    paths = {
        _write_gguf(backslash_root / "model-Q4_K_M.gguf"),
        _write_gguf(nested_root / "model-Q4_K_M.gguf"),
        _write_gguf(trailing_root / "model-Q4_K_M.gguf"),
        _write_gguf(plain_root / "model-Q4_K_M.gguf"),
    }

    discovered = [
        local_inventory._promote_to_custom_source(
            local_inventory._local_model_info(
                scan_path = path,
                load_path = path,
                source = "lmstudio",
                model_format = "gguf",
            )
        )
        for path in paths
    ]
    rows = local_inventory._dedupe_local_models(discovered)

    assert {Path(row.path) for row in rows} == paths
