# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF picker rows describe one checkpoint each.

A repo can publish several checkpoints under one set of quant labels
(``unsloth/LTX-2.3-GGUF`` ships 63 GGUFs as ``ltx-2.3-22b-dev-*`` at the root,
``distilled/ltx-2.3-22b-distilled-*`` and ``distilled-1.1/...``). Keying a row on
the quant token alone folded those into one row per quant, hid two of the three
checkpoints and advertised the sum of all three as the row's size.

Splitting a genuinely split GGUF is the opposite mistake, so both are pinned here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.inference.llama_cpp import _gguf_files_for_variant
from hub.utils.gguf import (
    GgufVariantInfo,
    _apply_gguf_display_labels,
    extract_quant_label,
    gguf_variant_family,
    gguf_variant_key,
    group_gguf_variant_files,
    list_local_gguf_variants,
)
from hub.utils.gguf_plan import build_gguf_variant_plans, is_main_gguf_variant_path
from hub.utils.inventory_scan import complete_snapshot_variants
from utils.models.model_config import (
    _extract_quant_label,
    _find_local_gguf_by_variant,
    _gguf_variant_key,
)

LTX_FILES = [
    ("ltx-2.3-22b-dev-Q6_K.gguf", 17_770_000_000),
    ("distilled/ltx-2.3-22b-distilled-Q6_K.gguf", 17_770_000_000),
    ("distilled-1.1/ltx-2.3-22b-distilled-1.1-Q6_K.gguf", 17_770_000_000),
]

# A real split GGUF: one variant, several shards, which must stay one row.
SHARDED_FILES = [
    ("BF16/DeepSeek-R1-BF16-00001-of-00003.gguf", 40),
    ("BF16/DeepSeek-R1-BF16-00002-of-00003.gguf", 40),
    ("BF16/DeepSeek-R1-BF16-00003-of-00003.gguf", 20),
]


class _Sibling:
    def __init__(self, rfilename: str, size: int) -> None:
        self.rfilename = rfilename
        self.size = size
        self.lfs = {"sha256": f"hash-of-{rfilename}"}


# --------------------------------------------------------------------------------------
# The key itself
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        # The shape almost every repo uses keeps the historical key, so every stored
        # pin, manifest and marker keeps resolving.
        ("Wan2.2-TI2V-5B-Q8_0.gguf", "Q8_0"),
        ("gemma-3-4b-it-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
        ("BF16/DeepSeek-R1-BF16-00001-of-00003.gguf", "BF16"),
        ("MXFP4_MOE/model-MXFP4_MOE-00001-of-00003.gguf", "MXFP4_MOE"),
        # A <model>-<QUANT>/ directory names the same quant, so it adds nothing.
        ("DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf", "Q4_K_M"),
        ("Llama-3.3-70B-Instruct-F16/Llama-3.3-70B-Instruct-F16-00001-of-00004.gguf", "F16"),
        # A directory naming something else is another checkpoint, and qualifies.
        ("ltx-2.3-22b-dev-Q6_K.gguf", "Q6_K"),
        (
            "distilled/ltx-2.3-22b-distilled-Q6_K.gguf",
            "distilled/ltx-2.3-22b-distilled-Q6_K",
        ),
        (
            "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q6_K.gguf",
            "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q6_K",
        ),
        # A quant-named directory never overrides the basename, and a suffix after the
        # quant does not qualify it: both are established spellings.
        ("Q8_0/model-Q4_K_M.gguf", "Q4_K_M"),
        ("BF16/gemma-4-12b-it-Q8_0-MTP-001-of-002.gguf", "Q8_0"),
        # No quant anywhere: unchanged fallback.
        ("weights/model.gguf", "weights/model"),
    ],
)
def test_gguf_variant_key(path, expected):
    assert gguf_variant_key(path) == expected


def test_key_is_unchanged_for_conventional_layouts():
    """The persisted variant identity must not churn where there is no collision."""
    for path in (
        "Wan2.2-TI2V-5B-Q4_K_M.gguf",
        "Qwen3-8B-UD-Q8_K_XL.gguf",
        "BF16/DeepSeek-R1-BF16-00002-of-00003.gguf",
        "DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf",
    ):
        assert gguf_variant_key(path) == extract_quant_label(path)


def test_shards_of_one_gguf_share_a_key_and_a_family():
    keys = {gguf_variant_key(path) for path, _ in SHARDED_FILES}
    families = {gguf_variant_family(path) for path, _ in SHARDED_FILES}
    assert keys == {"BF16"}
    assert len(families) == 1


def test_sibling_checkpoints_do_not_share_a_family():
    assert len({gguf_variant_family(path) for path, _ in LTX_FILES}) == 3


def test_loader_mirror_agrees_with_the_hub_key():
    """``model_config._gguf_variant_key`` is a copy; drift means a row that misloads."""
    corpus = [path for path, _ in (*LTX_FILES, *SHARDED_FILES)] + [
        "Wan2.2-TI2V-5B-Q8_0.gguf",
        "gemma-3-4b-it-UD-Q4_K_XL.gguf",
        "MXFP4_MOE/model-MXFP4_MOE-00001-of-00003.gguf",
        "DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf",
        "Llama-3.3-70B-Instruct-Q6_K/Llama-3.3-70B-Instruct-Q6_K-00001-of-00002.gguf",
        "Q6_K/Llama-3.3-70B-Instruct-Q6_K-00001-of-00002.gguf",
        "Q8_0/model-Q4_K_M.gguf",
        "BF16/gemma-4-12b-it-Q8_0-MTP-001-of-002.gguf",
        "weights/model.gguf",
        "BF16/foo.gguf",
    ]
    assert [_gguf_variant_key(p) for p in corpus] == [gguf_variant_key(p) for p in corpus]


# --------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------


def test_each_checkpoint_gets_its_own_row_at_its_own_size():
    rows = group_gguf_variant_files(LTX_FILES)
    assert len(rows) == 3
    assert {size for _, size in rows.values()} == {17_770_000_000}
    assert sorted(filename for filename, _ in rows.values()) == sorted(p for p, _ in LTX_FILES)


def test_shards_stay_one_row_summing_to_the_whole_file():
    rows = group_gguf_variant_files(SHARDED_FILES)
    assert list(rows) == ["BF16"]
    assert rows["BF16"] == ("BF16/DeepSeek-R1-BF16-00001-of-00003.gguf", 100)


def test_one_quant_shipped_twice_is_not_advertised_twice():
    """``unsloth/QwQ-32B-GGUF`` carries the same BF16 as ``-BF16`` and ``.BF16``."""
    duplicated = [
        ("BF16/QwQ-32B-BF16-00001-of-00002.gguf", 50),
        ("BF16/QwQ-32B-BF16-00002-of-00002.gguf", 15),
        ("BF16/QwQ-32B.BF16-00001-of-00002.gguf", 50),
        ("BF16/QwQ-32B.BF16-00002-of-00002.gguf", 15),
    ]
    rows = group_gguf_variant_files(duplicated)
    assert list(rows) == ["BF16"]
    assert rows["BF16"] == ("BF16/QwQ-32B-BF16-00001-of-00002.gguf", 65)


def test_local_lister_matches_the_remote_shape(tmp_path):
    snapshot = tmp_path / "snap"
    for path, _ in LTX_FILES:
        target = snapshot / path
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"x" * 512)
    (snapshot / "config.json").write_text("{}")

    variants, _ = list_local_gguf_variants(str(snapshot))
    assert sorted(v.quant for v in variants) == sorted(
        gguf_variant_key(path) for path, _ in LTX_FILES
    )
    assert all(v.size_bytes == 512 for v in variants)


def test_qualified_rows_carry_a_readable_label():
    variants = [
        GgufVariantInfo(filename = path, quant = gguf_variant_key(path), size_bytes = 1)
        for path, _ in LTX_FILES
    ]
    _apply_gguf_display_labels(variants)
    labels = {v.quant: v.display_label for v in variants}
    assert labels["Q6_K"] is None
    assert labels["distilled/ltx-2.3-22b-distilled-Q6_K"] == "Q6_K · distilled"
    assert labels["distilled-1.1/ltx-2.3-22b-distilled-1.1-Q6_K"] == "Q6_K · distilled-1.1"


# --------------------------------------------------------------------------------------
# The row must be downloadable, loadable and complete-able under its own key
# --------------------------------------------------------------------------------------


def test_every_row_key_matches_exactly_its_own_file():
    for path, _ in LTX_FILES:
        key = gguf_variant_key(path)
        owned = [other for other, _ in LTX_FILES if is_main_gguf_variant_path(other, key)]
        assert owned == [path], key


def test_download_plan_fetches_one_checkpoint_not_three():
    plans = build_gguf_variant_plans([_Sibling(path, size) for path, size in LTX_FILES])
    assert len(plans) == 3
    for path, size in LTX_FILES:
        plan = plans[gguf_variant_key(path).lower()]
        assert plan.main_filenames == frozenset({path})
        assert plan.main_size_bytes == size


def test_download_plan_keeps_every_shard_of_a_split_gguf():
    plans = build_gguf_variant_plans([_Sibling(path, size) for path, size in SHARDED_FILES])
    assert list(plans) == ["bf16"]
    assert plans["bf16"].main_filenames == frozenset(path for path, _ in SHARDED_FILES)
    assert plans["bf16"].main_size_bytes == 100


def test_download_plan_drops_a_duplicate_copy_of_one_quant():
    """Fetching both copies doubles the download and never reads as complete."""
    siblings = [
        _Sibling("Q6_K/model-Q6_K-00001-of-00002.gguf", 50),
        _Sibling("Q6_K/model-Q6_K-00002-of-00002.gguf", 8),
        _Sibling("model-Q6_K/model-Q6_K-00001-of-00002.gguf", 50),
        _Sibling("model-Q6_K/model-Q6_K-00002-of-00002.gguf", 8),
    ]
    plan = build_gguf_variant_plans(siblings)["q6_k"]
    assert plan.main_size_bytes == 58
    assert {path.split("/")[0] for path in plan.main_filenames} == {"Q6_K"}


def test_readiness_agrees_with_the_rows(tmp_path):
    """A finished download must read as complete under the same key it was listed by."""
    snapshot = tmp_path / "snap"
    for path, _ in LTX_FILES:
        target = snapshot / path
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"x" * 512)
    variants, _ = list_local_gguf_variants(str(snapshot))
    assert {v.quant for v in variants} <= complete_snapshot_variants(str(snapshot))


def test_a_chosen_row_loads_its_own_checkpoint(tmp_path):
    snapshot = tmp_path / "snap"
    for path, _ in LTX_FILES:
        target = snapshot / path
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"x" * 512)
    (snapshot / "config.json").write_text("{}")

    for path, _ in LTX_FILES:
        key = gguf_variant_key(path)
        assert _find_local_gguf_by_variant(str(snapshot), key) == str(snapshot / path)
        assert _gguf_files_for_variant([p for p, _ in LTX_FILES], key) == [path]


def test_llama_cpp_still_hands_over_every_shard():
    paths = [path for path, _ in SHARDED_FILES]
    assert _gguf_files_for_variant(paths, "bf16") == paths


def test_the_bare_label_still_resolves_for_an_ordinary_repo(tmp_path):
    """A pin stored before this change keeps working where nothing collided."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "Wan2.2-TI2V-5B-Q6_K.gguf").write_bytes(b"x" * 64)
    (snapshot / "config.json").write_text("{}")
    assert _extract_quant_label("Wan2.2-TI2V-5B-Q6_K.gguf") == "Q6_K"
    assert _find_local_gguf_by_variant(str(snapshot), "Q6_K") == str(
        snapshot / "Wan2.2-TI2V-5B-Q6_K.gguf"
    )


def test_delete_removes_only_the_chosen_checkpoint(tmp_path):
    from routes.models import _delete_gguf_variant_files

    snapshot = tmp_path / "snap"
    for path, _ in LTX_FILES:
        target = snapshot / path
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"x" * 512)

    count, _ = _delete_gguf_variant_files(snapshot, "distilled/ltx-2.3-22b-distilled-Q6_K")
    assert count == 1
    assert not (snapshot / "distilled/ltx-2.3-22b-distilled-Q6_K.gguf").exists()
    assert (snapshot / "ltx-2.3-22b-dev-Q6_K.gguf").exists()
    assert (snapshot / "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q6_K.gguf").exists()


def test_delete_removes_every_shard_of_one_variant(tmp_path):
    from routes.models import _delete_gguf_variant_files

    snapshot = tmp_path / "snap"
    for path, _ in SHARDED_FILES:
        target = snapshot / path
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"x" * 8)

    count, _ = _delete_gguf_variant_files(snapshot, "BF16")
    assert count == 3
    assert not list((snapshot / "BF16").glob("*.gguf"))


def test_a_local_subset_keys_the_same_as_the_whole_repo(tmp_path):
    """The key is a pure function of the path, so a half-downloaded snapshot and the
    remote listing cannot disagree and strand a finished download as incomplete."""
    whole = group_gguf_variant_files(LTX_FILES)
    for path, size in LTX_FILES:
        subset = group_gguf_variant_files([(path, size)])
        assert list(subset) == [gguf_variant_key(path)]
        assert subset[gguf_variant_key(path)] == whole[gguf_variant_key(path)]


def test_paths_are_keyed_the_same_on_windows_separators():
    assert gguf_variant_key("distilled\\ltx-2.3-22b-distilled-Q6_K.gguf") == gguf_variant_key(
        "distilled/ltx-2.3-22b-distilled-Q6_K.gguf"
    )


def test_row_filenames_are_real_paths_under_the_snapshot(tmp_path):
    """Every advertised row must point at a file that exists, for all three checkpoints."""
    snapshot = tmp_path / "snap"
    for path, _ in LTX_FILES:
        target = snapshot / path
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"x" * 512)
    variants, _ = list_local_gguf_variants(str(snapshot))
    assert len(variants) == 3
    for variant in variants:
        assert (snapshot / variant.filename).is_file()
    assert Path(sorted(v.filename for v in variants)[-1]).name.startswith("ltx-2.3-22b-dev")
