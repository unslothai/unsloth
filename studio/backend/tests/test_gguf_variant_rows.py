# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rows key on the checkpoint, not the quant token alone, so same-quant checkpoints stay separate."""

from __future__ import annotations

import re
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
    is_qualified_gguf_variant_key,
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

SHARDED_FILES = [
    ("BF16/DeepSeek-R1-BF16-00001-of-00003.gguf", 40),
    ("BF16/DeepSeek-R1-BF16-00002-of-00003.gguf", 40),
    ("BF16/DeepSeek-R1-BF16-00003-of-00003.gguf", 20),
]

H3_FILES = [
    "minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf",
    "minimax_h3_fl2va_pruned-Q4_K.gguf",
    "minimax_h3_ref2va_pruned-Q4_K.gguf",
]


class _Sibling:
    def __init__(self, rfilename: str, size: int) -> None:
        self.rfilename = rfilename
        self.size = size
        self.lfs = {"sha256": f"hash-of-{rfilename}"}


@pytest.mark.parametrize(
    "path,expected",
    [
        # Common shape keeps the historical key so stored pins, manifests and markers still resolve.
        ("Wan2.2-TI2V-5B-Q8_0.gguf", "Q8_0"),
        ("gemma-3-4b-it-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
        ("BF16/DeepSeek-R1-BF16-00001-of-00003.gguf", "BF16"),
        ("MXFP4_MOE/model-MXFP4_MOE-00001-of-00003.gguf", "MXFP4_MOE"),
        ("DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf", "Q4_K_M"),
        ("Llama-3.3-70B-Instruct-F16/Llama-3.3-70B-Instruct-F16-00001-of-00004.gguf", "F16"),
        ("ltx-2.3-22b-dev-Q6_K.gguf", "Q6_K"),
        (
            "distilled/ltx-2.3-22b-distilled-Q6_K.gguf",
            "distilled/ltx-2.3-22b-distilled-Q6_K",
        ),
        (
            "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q6_K.gguf",
            "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q6_K",
        ),
        ("Q8_0/model-Q4_K_M.gguf", "Q4_K_M"),
        ("BF16/gemma-4-12b-it-Q8_0-MTP-001-of-002.gguf", "Q8_0"),
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


def test_packed_and_grouped_quants_keep_their_own_key_in_both_copies():
    # prism-ml/Ternary-Bonsai-*-gguf ships all three beside each other.
    files = {
        "Ternary-Bonsai-8B-PQ2_0.gguf": "PQ2_0",
        "Ternary-Bonsai-8B-Q2_0.gguf": "Q2_0",
        "Ternary-Bonsai-8B-Q2_0_g64.gguf": "Q2_0_g64",
        # prism-ml/Ternary-Bonsai-27B-gguf spells its group-64 pack without the "_0".
        "Ternary-Bonsai-27B-Q2_g64.gguf": "Q2_g64",
        "Ternary-Bonsai-2-27B-PTQ1_0.gguf": "PTQ1_0",
    }
    for path, key in files.items():
        assert gguf_variant_key(path) == key
        assert _gguf_variant_key(path) == key


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


def test_h3_partitions_and_builds_at_one_quant_are_separate_rows():
    files = [
        ("minimax_h3_fl2va-Q4_K_M.gguf", 18),
        ("minimax_h3_fl2va_pruned-Q4_K_M.gguf", 11),
        ("minimax_h3_ref2va-Q4_K_M.gguf", 18),
        ("minimax_h3_ref2va_pruned-Q4_K_M.gguf", 11),
        ("minimax_h3_ref2va_pruned-Q2_K_M.gguf", 6),
    ]
    rows = group_gguf_variant_files(files)
    assert list(rows) == [
        "minimax_h3_fl2va-Q4_K_M",
        "minimax_h3_fl2va_pruned-Q4_K_M",
        "minimax_h3_ref2va-Q4_K_M",
        "minimax_h3_ref2va_pruned-Q4_K_M",
        "minimax_h3_ref2va_pruned-Q2_K_M",
    ]
    assert [filename for filename, _size in rows.values()] == [
        filename for filename, _size in files
    ]


def test_h3_variant_keys_match_the_loader_copy():
    for filename in (
        "minimax_h3_fl2va-Q4_K_M.gguf",
        "minimax_h3_fl2va_pruned-Q4_K_M.gguf",
        "minimax_h3_ref2va-Q4_K_M.gguf",
        "minimax_h3_ref2va_pruned-Q2_K_M.gguf",
        "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
    ):
        assert _gguf_variant_key(filename) == gguf_variant_key(filename)


def test_h3_root_stems_are_qualified_keys():
    qualified = gguf_variant_key(H3_FILES[0])
    assert qualified == "minimax_h3_fl2va_pruned-UD-Q2_K_XL"
    assert is_qualified_gguf_variant_key(qualified)
    assert not is_qualified_gguf_variant_key("UD-Q2_K_XL")


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


def test_the_route_model_carries_display_label_to_the_picker():
    """The route's GgufVariantDetail must declare display_label, or pydantic silently drops it."""
    from hub.schemas.inventory import GgufVariantDetail as HubDetail
    from models.models import GgufVariantDetail as RouteDetail

    assert "display_label" in HubDetail.model_fields
    assert "display_label" in RouteDetail.model_fields, (
        "the route's own response model dropped display_label; a qualified row will render "
        "as its relative path"
    )

    row = RouteDetail(
        filename = "distilled/ltx-2.3-22b-distilled-Q6_K.gguf",
        quant = "distilled/ltx-2.3-22b-distilled-Q6_K",
        display_label = "Q6_K · distilled",
        size_bytes = 17_774_906_400,
    )
    assert row.model_dump()["display_label"] == "Q6_K · distilled"


def test_an_unqualified_row_still_reports_no_display_label():
    """The display label is only for keys a picker cannot show; a plain quant token gets none."""
    from models.models import GgufVariantDetail as RouteDetail

    row = RouteDetail(filename = "model-Q4_K_M.gguf", quant = "Q4_K_M", size_bytes = 1)
    assert row.model_dump()["display_label"] is None


def test_a_discarded_shard_family_leaves_the_download_targets_too():
    """Discarding a shard family must also drop its target_filenames, required_hashes and download size."""
    from hub.utils.download_manifest import ExpectedFile
    from hub.utils.gguf_plan import plan_from_expected_files

    expected = [
        ExpectedFile(path = "BF16/QwQ-32B-BF16-00001-of-00002.gguf", size = 50, sha256 = "a1"),
        ExpectedFile(path = "BF16/QwQ-32B-BF16-00002-of-00002.gguf", size = 15, sha256 = "a2"),
        ExpectedFile(path = "BF16/QwQ-32B.BF16-00001-of-00002.gguf", size = 50, sha256 = "b1"),
        ExpectedFile(path = "BF16/QwQ-32B.BF16-00002-of-00002.gguf", size = 15, sha256 = "b2"),
        ExpectedFile(path = "mmproj-F16.gguf", size = 5, sha256 = "c1"),
    ]
    plan = plan_from_expected_files("bf16", expected)

    assert plan.main_filenames == {
        "BF16/QwQ-32B-BF16-00001-of-00002.gguf",
        "BF16/QwQ-32B-BF16-00002-of-00002.gguf",
    }
    assert set(plan.target_filenames) == plan.main_filenames | {"mmproj-F16.gguf"}
    assert plan.required_hashes == frozenset({"a1", "a2", "c1"})
    assert plan.download_size_bytes == 70
    assert plan.main_size_bytes == 65


def test_a_genuine_split_keeps_every_shard_in_the_plan():
    """The narrowing must stay blind to shard count: one family is one checkpoint."""
    from hub.utils.download_manifest import ExpectedFile
    from hub.utils.gguf_plan import plan_from_expected_files

    expected = [
        ExpectedFile(path = f"BF16/DeepSeek-R1-BF16-0000{n}-of-00003.gguf", size = 10, sha256 = f"h{n}")
        for n in (1, 2, 3)
    ]
    plan = plan_from_expected_files("bf16", expected)
    assert len(plan.main_filenames) == 3
    assert set(plan.target_filenames) == plan.main_filenames
    assert plan.download_size_bytes == 30


def test_a_quant_shipped_whole_and_split_plans_only_the_set_the_loader_opens():
    """Qwen/Qwen2.5-Coder-7B-Instruct-GGUF ships Q4_K_M both as one file and as two shards."""
    from routes.models import _one_shard_family_of
    from utils.models.model_config import _group_gguf_variant_files

    split = [
        ("qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf", 3_993_201_376),
        ("qwen2.5-coder-7b-instruct-q4_k_m-00002-of-00002.gguf", 689_872_288),
    ]
    files = [*split, ("qwen2.5-coder-7b-instruct-q4_k_m.gguf", 4_683_073_536)]
    opened = _gguf_files_for_variant([path for path, _ in files], "Q4_K_M")
    assert opened[: len(split)] == [path for path, _ in split]

    plan = build_gguf_variant_plans([_Sibling(path, size) for path, size in files])["q4_k_m"]
    assert set(plan.target_filenames) == plan.main_filenames == {path for path, _ in split}
    assert plan.download_size_bytes == plan.main_size_bytes == 4_683_073_664

    row = (split[0][0], 4_683_073_664)
    assert group_gguf_variant_files(files) == {"q4_k_m": row}
    assert _group_gguf_variant_files([(p, "q4_k_m", s) for p, s in files]) == {"q4_k_m": row}
    assert [e[0] for e in _one_shard_family_of([(p, None, s) for p, s in files])] == [
        path for path, _ in split
    ]


def test_a_qualified_key_is_an_explicit_checkpoint_request():
    """A qualified key is an explicit checkpoint request, so a missing one must not fall back to another."""
    from core.inference.openai_auto_download import looks_like_quant

    assert looks_like_quant("distilled/ltx-2.3-22b-distilled-Q6_K") is True
    assert looks_like_quant("distilled-1.1/ltx-2.3-22b-distilled-1.1-Q6_K") is True
    assert looks_like_quant("Q6_K") is True
    assert looks_like_quant("IQ4_XS-3.53bpw") is True
    assert looks_like_quant("latest") is False
    assert looks_like_quant("8b") is False
    assert looks_like_quant(None) is False


def test_the_exact_key_wins_over_the_bare_quant_label():
    """A bare quant label matches every checkpoint carrying that quant, so the exact key must win."""
    from routes.models import _main_variant_rank

    # Pass the request verbatim: folding it would strip a qualified key's path punctuation.
    assert _main_variant_rank("ltx-2.3-22b-dev-Q6_K.gguf", "Q6_K") == 0
    assert _main_variant_rank("distilled/ltx-2.3-22b-distilled-Q6_K.gguf", "Q6_K") == 1
    assert _main_variant_rank("ltx-2.3-22b-dev-Q4_K_M.gguf", "Q6_K") is None
    qualified = "distilled/ltx-2.3-22b-distilled-Q6_K"
    assert _main_variant_rank("distilled/ltx-2.3-22b-distilled-Q6_K.gguf", qualified) == 0
    assert _main_variant_rank("ltx-2.3-22b-dev-Q6_K.gguf", qualified) is None
    assert _main_variant_rank("model-UD-Q4_K_XL.gguf", "udq4kxl") == 0
    assert _main_variant_rank("exp-a/model-Q6_K.gguf", "expa/model-Q6_K") is None
    assert _main_variant_rank("exp-a/model-Q6_K.gguf", "exp-a/model-q6_k") == 0
    h3 = "minimax_h3_ref2va_pruned-Q4_K"
    assert _main_variant_rank(f"{h3}.gguf", h3) == 0
    assert _main_variant_rank(f"{h3}.gguf", "Q4_K") is None
    assert _main_variant_rank("minimax_h3_fl2va_pruned-Q4_K.gguf", "Q4_K") is None
    assert _main_variant_rank(f"{h3}.gguf", "minimax-h3-ref2va-pruned-Q4_K") is None


def test_two_checkpoints_in_one_directory_get_distinguishable_labels():
    """Two rows labelled ``Q6_K · experiments`` are two rows a user cannot tell apart."""
    paths = ("experiments/model-a-Q6_K.gguf", "experiments/model-b-Q6_K.gguf")
    variants = [
        GgufVariantInfo(filename = path, quant = gguf_variant_key(path), size_bytes = 1) for path in paths
    ]
    _apply_gguf_display_labels(variants)
    labels = [v.display_label for v in variants]
    assert labels == ["Q6_K · experiments/model-a-Q6_K", "Q6_K · experiments/model-b-Q6_K"]
    assert len(set(labels)) == 2


def test_the_model_config_listers_advertise_the_qualified_keys(tmp_path):
    """Listers must advertise qualified keys; otherwise /v1, VRAM preflight and picker sizing miss them."""
    from utils.models.model_config import list_local_gguf_variants

    snapshot = tmp_path / "snap"
    for path, _size in LTX_FILES:
        target = snapshot / path
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"x" * 512)
    (snapshot / "config.json").write_text("{}")

    variants, _has_vision = list_local_gguf_variants(str(snapshot))
    assert sorted(v.quant for v in variants) == sorted(
        gguf_variant_key(path) for path, _ in LTX_FILES
    )
    for variant in variants:
        resolved = _find_local_gguf_by_variant(str(snapshot), variant.quant)
        assert resolved is not None, variant.quant
        assert _gguf_variant_key(Path(resolved).relative_to(snapshot).as_posix()) == variant.quant


def test_the_model_config_lister_advertises_h3_root_stem_keys(tmp_path):
    """H3 qualifies partitions in the root filename rather than a directory. The generic
    lister must still advertise the key that the Hub worker plans and the loader resolves."""
    from utils.models.model_config import list_local_gguf_variants

    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    for name in H3_FILES:
        (snapshot / name).write_bytes(b"x" * 64)
    (snapshot / "config.json").write_text("{}")

    variants, _ = list_local_gguf_variants(str(snapshot))
    assert {variant.quant for variant in variants} == {gguf_variant_key(name) for name in H3_FILES}
    for variant in variants:
        assert _find_local_gguf_by_variant(str(snapshot), variant.quant) is not None


def test_h3_variants_require_the_advertised_full_stem():
    """Root-level partition keys stay canonical instead of creating a second state identity."""
    from types import SimpleNamespace

    from core.inference.openai_auto_download import _gguf_variants, _match_variant
    from hub.utils.gguf_plan import plan_for_variant

    plans = build_gguf_variant_plans([_Sibling(name, 64) for name in H3_FILES])
    variants = _gguf_variants([SimpleNamespace(rfilename = name, size = 64) for name in H3_FILES])

    ud_key = gguf_variant_key(H3_FILES[0]).lower()
    assert {key.lower() for key in variants} == set(plans)
    assert plan_for_variant(plans, H3_FILES[0][:-5]) is plans[ud_key]
    assert _match_variant(H3_FILES[0][:-5], variants).lower() == ud_key
    assert plan_for_variant(plans, "UD-Q2_K_XL") is None
    assert _match_variant("UD-Q2_K_XL", variants) is None
    assert plan_for_variant(plans, "Q4_K") is None
    assert _match_variant("Q4_K", variants) is None


def test_h3_auto_download_excludes_the_bundled_text_encoders():
    """The Qwen GGUFs are load-time companions, never selectable H3 transformers."""
    from types import SimpleNamespace

    from core.inference.openai_auto_download import _gguf_variants

    siblings = [
        SimpleNamespace(rfilename = name, size = 64)
        for name in [
            *H3_FILES,
            "qwen3vl_32b_minimax_h3-Q2_K_M.gguf",
            "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
        ]
    ]
    variants = _gguf_variants(siblings, "unsloth/MiniMax-H3-GGUF")
    assert set(variants) == {gguf_variant_key(name) for name in H3_FILES}
    assert "Q2_K_M" not in variants
    assert "Q4_K_M" not in variants


def test_the_ordinary_repo_keeps_its_bare_labels(tmp_path):
    """The qualified key is only for repos that need it. Every ordinary repo must list exactly
    the labels it listed before, or every stored pin and every /v1 model id breaks at once."""
    from utils.models.model_config import list_local_gguf_variants

    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    for name in ("model-Q4_K_M.gguf", "model-Q6_K.gguf", "BF16/model-BF16-00001-of-00002.gguf"):
        target = snapshot / name
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"x" * 64)
    (snapshot / "config.json").write_text("{}")
    variants, _ = list_local_gguf_variants(str(snapshot))
    assert sorted(v.quant for v in variants) == ["BF16", "Q4_K_M", "Q6_K"]


def test_the_auto_download_map_is_keyed_like_the_plan():
    """``_match_variant`` looks the request up in this map and the worker then looks the plan up
    by the same string, so a map keyed on the bare label turned every qualified row into a 404
    instead of dispatching its download."""
    from types import SimpleNamespace

    from core.inference.openai_auto_download import _gguf_variants

    siblings = [SimpleNamespace(rfilename = path, size = size) for path, size in LTX_FILES]
    sizes = _gguf_variants(siblings)
    assert set(sizes) == {gguf_variant_key(path) for path, _ in LTX_FILES}
    assert "distilled/ltx-2.3-22b-distilled-Q6_K" in sizes
    assert all(size > 0 for size in sizes.values())


def test_an_unknown_layout_row_keeps_the_label_it_always_had(tmp_path):
    """Files without a recognised quant keep their old label; renaming them would break stored pins."""
    from utils.models.model_config import list_local_gguf_variants

    snapshot = tmp_path / "snap"
    (snapshot / "BF16").mkdir(parents = True)
    (snapshot / "Qwen3.6-27B-MTP-001-of-002.gguf").write_bytes(b"x" * 100)
    (snapshot / "BF16" / "gemma-4-12b-it-Q8_0-001-of-002.gguf").write_bytes(b"x" * 40)
    (snapshot / "config.json").write_text("{}")

    variants, _ = list_local_gguf_variants(str(snapshot))
    assert sorted(v.quant for v in variants) == ["MTP", "Q8_0"]


def test_a_shared_container_directory_still_answers_its_bare_quant():
    """A shared container directory must still answer the bare quant, or stored pins miss the plan map."""
    from types import SimpleNamespace

    from hub.utils.gguf_plan import build_gguf_variant_plans, plan_for_variant

    siblings = [
        SimpleNamespace(rfilename = f"weights/model-{q}.gguf", size = 10, lfs = None)
        for q in ("Q4_K_M", "Q6_K")
    ]
    plans = build_gguf_variant_plans(siblings)
    assert set(plans) == {"weights/model-q4_k_m", "weights/model-q6_k"}
    assert plan_for_variant(plans, "Q4_K_M") is plans["weights/model-q4_k_m"]
    assert plan_for_variant(plans, "q6_k") is plans["weights/model-q6_k"]
    assert plan_for_variant(plans, "weights/model-q6_k") is plans["weights/model-q6_k"]
    assert plan_for_variant(plans, "Q8_0") is None
    assert plan_for_variant(plans, "") is None


def test_an_ambiguous_bare_quant_gets_no_fallback():
    """Where a repo genuinely holds several checkpoints at one quant, the bare name does not name
    one of them, and guessing is the collapse this PR exists to undo."""
    from types import SimpleNamespace

    from hub.utils.gguf_plan import build_gguf_variant_plans, plan_for_variant

    siblings = [
        SimpleNamespace(rfilename = path, size = 10, lfs = None)
        for path in (
            "distilled/ltx-2.3-22b-distilled-Q6_K.gguf",
            "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q6_K.gguf",
        )
    ]
    plans = build_gguf_variant_plans(siblings)
    assert len(plans) == 2
    assert plan_for_variant(plans, "Q6_K") is None
    assert plan_for_variant(plans, "distilled/ltx-2.3-22b-distilled-q6_k") is not None


def test_a_bare_local_id_keeps_the_checkpoint_a_plain_load_takes():
    """Plain loads take the repo root via non-recursive detect_gguf_model; /v1 must not pick a sibling."""
    from core.inference.openai_auto_download import preferred_quant

    quants = (
        "Q6_K",
        "distilled/ltx-2.3-22b-distilled-Q6_K",
        "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q6_K",
    )
    unqualified = tuple(q for q in quants if "/" not in q)
    assert preferred_quant(unqualified) == "Q6_K"


def test_bpw_precisions_stay_separately_selectable(tmp_path):
    """bpw variants stay separate rows; keying by the token extractor would merge them under one quant."""
    from utils.models.model_config import list_local_gguf_variants

    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    for bpw in ("3.53", "3.97", "4.19"):
        (snapshot / f"Qwen3.6-IQ4_XS-{bpw}bpw.gguf").write_bytes(b"x" * 64)
    (snapshot / "config.json").write_text("{}")
    variants, _ = list_local_gguf_variants(str(snapshot))
    assert sorted(v.quant for v in variants) == [
        "IQ4_XS-3.53bpw",
        "IQ4_XS-3.97bpw",
        "IQ4_XS-4.19bpw",
    ]


def test_a_parent_only_quant_survives_the_endian_filter():
    """_is_big_endian_gguf_path reads a quant token, so a qualified key drops parent-only quants."""
    from types import SimpleNamespace

    from hub.utils.gguf_plan import build_gguf_variant_plans

    siblings = [
        SimpleNamespace(rfilename = "distilled/Q4_K_M/foo.gguf", size = 10, lfs = None),
    ]
    plans = build_gguf_variant_plans(siblings)
    assert plans, "the only file in the repo was filtered out of every plan"
    key = next(iter(plans))
    assert plans[key].main_filenames == {"distilled/Q4_K_M/foo.gguf"}


def test_admission_resolves_a_bare_alias_instead_of_404ing():
    """Admission checks its own size map, so the alias fallback must be in both, or a bare quant 404s."""
    from core.inference.openai_auto_download import _match_variant

    variants = {"weights/model-Q4_K_M": 10, "weights/model-Q6_K": 20}
    assert _match_variant("Q4_K_M", variants) == "weights/model-Q4_K_M"
    assert _match_variant("weights/model-q6_k", variants) == "weights/model-Q6_K"
    assert _match_variant("Q8_0", variants) is None
    # With two checkpoints at one quant the bare name is ambiguous and must miss.
    ambiguous = {"distilled/m-Q6_K": 10, "distilled-1.1/m-Q6_K": 20}
    assert _match_variant("Q6_K", ambiguous) is None


def test_a_parent_only_quant_survives_the_main_file_predicate():
    """is_main_gguf_variant_path runs the endian test too. Handed the qualified key it could not
    see the parent-only quant, so the plan came back with no main files and an interrupted
    download had no hashes to resume against."""
    from hub.utils.gguf_plan import is_main_gguf_variant_path

    path = "distilled/Q4_K_M/foo.gguf"
    assert is_main_gguf_variant_path(path, gguf_variant_key(path))


def test_a_local_load_keeps_the_qualified_identity_it_was_asked_for(tmp_path):
    """A local load keeps the qualified variant identity, so /status and the delete guard match the key."""
    from utils.models.model_config import ModelConfig

    snapshot = tmp_path / "snap"
    for path, _size in LTX_FILES:
        target = snapshot / path
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"x" * 512)
    (snapshot / "config.json").write_text("{}")

    wanted = "distilled/ltx-2.3-22b-distilled-Q6_K"
    cfg = ModelConfig.from_identifier(str(snapshot), gguf_variant = wanted)
    assert cfg is not None and cfg.is_gguf and cfg.is_local
    assert cfg.gguf_variant == wanted
    assert "distilled/" in Path(cfg.gguf_file).as_posix()
    bare = ModelConfig.from_identifier(str(snapshot), gguf_variant = "Q6_K")
    assert bare is not None and bare.gguf_variant == "Q6_K"


def _cache_repo(tmp_path: Path, repo_id: str, names: list[str]):
    """An HF cache repo: snapshot symlinks pointing at blobs, as the deletion scan reads it."""
    from types import SimpleNamespace

    repo_dir = tmp_path / f"models--{repo_id.replace('/', '--')}"
    snap = repo_dir / "snapshots" / "rev1"
    blobs = repo_dir / "blobs"
    snap.mkdir(parents = True)
    blobs.mkdir(parents = True)
    files = []
    for index, name in enumerate(names):
        blob = blobs / f"sha{index}"
        blob.write_bytes(b"x" * 64)
        link = snap / name
        link.parent.mkdir(parents = True, exist_ok = True)
        link.symlink_to(blob)
        files.append(
            SimpleNamespace(
                file_name = Path(name).name,
                file_path = str(link),
                blob_path = str(blob),
                size_on_disk = 64,
            )
        )
    return SimpleNamespace(
        repo_id = repo_id,
        repo_type = "model",
        repo_path = repo_dir,
        revisions = [SimpleNamespace(files = files, snapshot_path = str(snap))],
    ), snap


def test_deleting_a_container_variant_accepts_the_bare_quant_the_download_admits(tmp_path):
    """Container deletes must accept the bare quant downloads admit, or weights stay on disk."""
    from hub.services.models.deletion import _delete_gguf_variant_from_repos
    from hub.utils.gguf_plan import build_gguf_variant_plans, plan_for_variant

    repo, snap = _cache_repo(tmp_path, "org/Model-GGUF", ["weights/model-Q4_K_M.gguf"])
    # Keep in step with the download side, which already resolves the bare name.
    plans = build_gguf_variant_plans([_Sibling("weights/model-Q4_K_M.gguf", 64)])
    assert set(plans) == {"weights/model-q4_k_m"}
    assert plan_for_variant(plans, "Q4_K_M") is not None

    _delete_gguf_variant_from_repos("org/Model-GGUF", "Q4_K_M", [repo], None, root = tmp_path)
    assert not (snap / "weights" / "model-Q4_K_M.gguf").is_symlink()


def test_deleting_an_h3_variant_requires_the_full_stem(tmp_path):
    from fastapi import HTTPException
    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    repo, snap = _cache_repo(tmp_path, "unsloth/MiniMax-H3-GGUF", H3_FILES)
    _delete_gguf_variant_from_repos(
        repo.repo_id,
        H3_FILES[0][:-5],
        [repo],
        None,
        root = tmp_path,
    )
    assert not (snap / H3_FILES[0]).is_symlink()

    with pytest.raises(HTTPException) as exc_info:
        _delete_gguf_variant_from_repos(repo.repo_id, "UD-Q2_K_XL", [repo], None, root = tmp_path)
    assert exc_info.value.status_code == 404
    assert (snap / H3_FILES[1]).is_symlink()
    assert (snap / H3_FILES[2]).is_symlink()


def test_an_ambiguous_bare_quant_deletes_nothing(tmp_path):
    """Two checkpoints at one quant: the bare name genuinely does not name one of them, and
    deleting the wrong one is unrecoverable. Same rule ``plan_for_variant`` applies."""
    from fastapi import HTTPException

    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    repo, snap = _cache_repo(
        tmp_path,
        "org/Model-GGUF",
        ["distilled/model-Q4_K_M.gguf", "distilled-1.1/model-Q4_K_M.gguf"],
    )
    with pytest.raises(HTTPException) as excinfo:
        _delete_gguf_variant_from_repos("org/Model-GGUF", "Q4_K_M", [repo], None, root = tmp_path)
    assert excinfo.value.status_code == 404
    assert (snap / "distilled" / "model-Q4_K_M.gguf").is_symlink()
    assert (snap / "distilled-1.1" / "model-Q4_K_M.gguf").is_symlink()

    _delete_gguf_variant_from_repos(
        "org/Model-GGUF", "distilled/model-Q4_K_M", [repo], None, root = tmp_path
    )
    assert not (snap / "distilled" / "model-Q4_K_M.gguf").is_symlink()
    assert (snap / "distilled-1.1" / "model-Q4_K_M.gguf").is_symlink()


def test_a_bare_quant_that_is_its_own_key_still_deletes_only_itself(tmp_path):
    """Root key is the bare quant, so the alias fallback must not reach a qualified sibling."""
    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    repo, snap = _cache_repo(
        tmp_path, "org/Model-GGUF", ["model-Q4_K_M.gguf", "distilled/model-Q4_K_M.gguf"]
    )
    _delete_gguf_variant_from_repos("org/Model-GGUF", "Q4_K_M", [repo], None, root = tmp_path)
    assert not (snap / "model-Q4_K_M.gguf").is_symlink()
    assert (snap / "distilled" / "model-Q4_K_M.gguf").is_symlink()


def test_a_qualified_key_whose_basename_ends_in_be_still_resolves_for_loading():
    """Endian check reads only the quant token, so a qualified key whose basename ends in be still loads."""
    files = [
        "distilled/Q4_K_M/foo-be.gguf",
        "other/Q4_K_M/bar.gguf",
        "stories260K.gguf",
        "stories260K-be.gguf",
    ]
    assert _gguf_files_for_variant(files, "distilled/Q4_K_M/foo-be") == [
        "distilled/Q4_K_M/foo-be.gguf"
    ]
    assert _gguf_files_for_variant(files, "other/Q4_K_M/bar") == ["other/Q4_K_M/bar.gguf"]
    assert _gguf_files_for_variant(files, "stories260K") == ["stories260K.gguf"]
    assert is_main_gguf_variant_path(
        "distilled/Q4_K_M/foo-be.gguf", gguf_variant_key("distilled/Q4_K_M/foo-be.gguf")
    )


def test_a_bpw_build_keeps_its_own_identity_everywhere():
    """A bpw build keeps its own identity in the lister, plan, auto-download map and delete predicate."""
    from utils.models.model_config import _extract_quant_label as loader_label

    a = "model-IQ4_XS-3.53bpw.gguf"
    b = "model-IQ4_XS-3.97bpw.gguf"
    for path in (a, b):
        assert gguf_variant_key(path) == loader_label(path)
        assert _gguf_variant_key(path) == gguf_variant_key(path)
    assert gguf_variant_key(a) != gguf_variant_key(b)
    assert gguf_variant_key("model-IQ4_XS-3.53bpw-00001-of-00002.gguf") == gguf_variant_key(a)
    grouped = group_gguf_variant_files([(a, 10), (b, 20)])
    assert grouped == {"IQ4_XS-3.53bpw": (a, 10), "IQ4_XS-3.97bpw": (b, 20)}
    # Plain quant keys are unchanged so stored pins still resolve.
    assert gguf_variant_key("model-Q4_K_M.gguf") == "Q4_K_M"
    assert gguf_variant_key("distilled/model-Q6_K.gguf") == "distilled/model-Q6_K"


def test_a_bpw_key_needs_no_scope_label():
    """A bpw key already reads as a label, so display must not append the filename to it."""
    variants = [
        GgufVariantInfo(filename = "model-IQ4_XS-3.53bpw.gguf", quant = "IQ4_XS-3.53bpw", size_bytes = 1),
        GgufVariantInfo(filename = "model-IQ4_XS-3.97bpw.gguf", quant = "IQ4_XS-3.97bpw", size_bytes = 2),
        GgufVariantInfo(
            filename = "distilled/model-Q6_K.gguf", quant = "distilled/model-Q6_K", size_bytes = 3
        ),
    ]
    _apply_gguf_display_labels(variants)
    assert variants[0].display_label is None
    assert variants[1].display_label is None
    assert variants[2].display_label == "Q6_K · distilled"


def test_a_bare_auto_download_stays_on_the_root_checkpoint():
    """preferred_quant is order-sensitive, so once the map carried a key per checkpoint a bare
    org/repo could pick distilled/model-Q6_K over the root model-Q6_K -- while the same id
    resolves to the root locally, i.e. one model id serving two different sets of weights."""
    from core.inference.openai_auto_download import _match_variant

    both = {"distilled/model-Q6_K": 1, "Q6_K": 2}
    assert _match_variant(None, both) == "Q6_K"
    assert _match_variant(None, {"Q6_K": 2, "distilled/model-Q6_K": 1}) == "Q6_K"
    assert _match_variant("distilled/model-Q6_K", both) == "distilled/model-Q6_K"
    assert _match_variant(None, {"distilled/model-Q6_K": 1}) == "distilled/model-Q6_K"


def test_the_bare_alias_keeps_the_bpw_and_survives_a_dotted_stem():
    """Bare-alias fallbacks must keep the bpw modifier and not re-stem a key at the dot in 3.53bpw."""
    from hub.utils.gguf import bare_quant_alias

    assert bare_quant_alias("weights/model-IQ4_XS-3.53bpw") == "IQ4_XS-3.53bpw"
    assert bare_quant_alias("weights/model-Q4_K_M") == "Q4_K_M"
    assert bare_quant_alias("distilled/ltx-2.3-22b-distilled-Q6_K") == "Q6_K"


def test_a_bpw_container_build_resolves_by_its_bare_spelling(tmp_path):
    """A shared container qualifies the key, so the bare bpw spelling must keep resolving."""
    from hub.services.models.deletion import _delete_gguf_variant_from_repos
    from hub.utils.gguf_plan import build_gguf_variant_plans, plan_for_variant

    plans = build_gguf_variant_plans([_Sibling("weights/model-IQ4_XS-3.53bpw.gguf", 64)])
    assert set(plans) == {"weights/model-iq4_xs-3.53bpw"}
    assert plan_for_variant(plans, "IQ4_XS-3.53bpw") is not None
    assert plan_for_variant(plans, "IQ4_XS") is None

    repo, snap = _cache_repo(tmp_path, "org/Model-GGUF", ["weights/model-IQ4_XS-3.53bpw.gguf"])
    _delete_gguf_variant_from_repos("org/Model-GGUF", "IQ4_XS-3.53bpw", [repo], None, root = tmp_path)
    assert not (snap / "weights" / "model-IQ4_XS-3.53bpw.gguf").is_symlink()


def test_the_inference_lister_sizes_one_shard_family_not_both_copies():
    """Shard copies of one quant must be sized once, or the VRAM guard refuses a fitting load."""
    from utils.models.model_config import _group_gguf_variant_files

    entries = [
        ("QwQ-32B-BF16-00001-of-00002.gguf", "BF16", 10),
        ("QwQ-32B-BF16-00002-of-00002.gguf", "BF16", 10),
        ("QwQ-32B.BF16-00001-of-00002.gguf", "BF16", 10),
        ("QwQ-32B.BF16-00002-of-00002.gguf", "BF16", 10),
    ]
    grouped = _group_gguf_variant_files(entries)
    assert grouped == {"BF16": ("QwQ-32B-BF16-00001-of-00002.gguf", 20)}
    # Must equal the hub lister's grouping so the two cannot drift.
    assert grouped == group_gguf_variant_files([(n, s) for n, _q, s in entries])
    split = [("m-00001-of-00003.gguf", "Q4_K_M", 5)] * 1 + [
        ("m-00002-of-00003.gguf", "Q4_K_M", 5),
        ("m-00003-of-00003.gguf", "Q4_K_M", 5),
    ]
    assert _group_gguf_variant_files(split)["Q4_K_M"][1] == 15


def test_a_bpw_modifier_in_the_quant_directory_still_makes_two_keys():
    """A bpw modifier in a quant directory still yields two keys, not a shared bare IQ4_XS key."""
    a = "IQ4_XS-3.53bpw/model.gguf"
    b = "IQ4_XS-3.97bpw/model.gguf"
    assert gguf_variant_key(a) == "IQ4_XS-3.53bpw"
    assert gguf_variant_key(b) == "IQ4_XS-3.97bpw"
    assert _gguf_variant_key(a) == gguf_variant_key(a)
    assert _gguf_variant_key(b) == gguf_variant_key(b)
    assert group_gguf_variant_files([(a, 10), (b, 20)]) == {
        "IQ4_XS-3.53bpw": (a, 10),
        "IQ4_XS-3.97bpw": (b, 20),
    }
    assert gguf_variant_key("Q4_K_M/model.gguf") == "Q4_K_M"
    assert gguf_variant_key("Q6_K/model-3.5bpw.gguf") == "Q6_K-3.5bpw"


def test_the_remote_default_variant_prefers_the_root_checkpoint():
    """pick_best_gguf must default to the root checkpoint, not an equal-named file in a subdirectory."""
    import inspect

    from hub.services.models import gguf_variants as service

    source = inspect.getsource(service)
    assert "default_gguf_scope(variants, repo_id, key = lambda v: v.quant)" in source
    assert "pick_best_gguf(_default_variant_candidates(variants, repo_id))" in source


def test_the_load_guard_sees_the_alias_the_delete_accepts():
    """Load guard must treat a bare quant and its qualified row as one checkpoint, as delete does."""
    from routes.models import _variant_names_same_checkpoint as same

    assert same("Q4_K_M", "weights/model-Q4_K_M")
    assert same("weights/model-Q4_K_M", "Q4_K_M")
    assert same("Q4_K_M", "Q4_K_M")
    assert same("Q6_K", "minimax_h3_fl2va_pruned-Q6_K")
    assert not same("minimax_h3_fl2va_pruned-Q6_K", "minimax_h3_ref2va_pruned-Q6_K")
    assert not same("Q4_K_M", "Q6_K")
    assert not same("distilled/model-Q6_K", "other/model-Q6_K")
    assert not same(None, "Q4_K_M")
    assert not same("Q4_K_M", "")


def test_every_branch_derives_the_default_from_the_root_rows():
    """Every /gguf-variants branch must derive the default from the root rows, not its own branch."""
    import inspect
    import types

    from hub.services.models import gguf_variants as service

    rows = [
        types.SimpleNamespace(filename = "distilled/model-Q6_K.gguf", quant = "distilled/model-Q6_K"),
        types.SimpleNamespace(filename = "model-Q6_K.gguf", quant = "Q6_K"),
    ]
    assert service._default_variant_candidates(rows) == ["model-Q6_K.gguf"]
    # empty root candidates fall back to all rows.
    assert service._default_variant_candidates(rows[:1]) == ["distilled/model-Q6_K.gguf"]
    # guards every response branch against selecting from raw filenames.
    source = inspect.getsource(service)
    # accepts formatter line wrapping.
    assert len(re.findall(r"pick_best_gguf\(\s*_default_variant_candidates\(", source)) == 1
    assert len(re.findall(r"default_variant = _default_variant_key\(", source)) == 5
    assert "pick_best_gguf(filenames)" not in source
    # custom folders use the selected row key because stem keys need not identify the file (#13926).
    assert "gguf_variant_key(best) if best" not in source


def test_the_kv_cache_estimate_sizes_one_shard_family():
    """only one duplicate BF16 shard family contributes to the kv-cache estimate."""
    from routes.models import _one_shard_family_of

    duplicated = [
        ("QwQ-32B-BF16-00001-of-00002.gguf", None, 10),
        ("QwQ-32B-BF16-00002-of-00002.gguf", None, 10),
        ("QwQ-32B.BF16-00001-of-00002.gguf", None, 10),
        ("QwQ-32B.BF16-00002-of-00002.gguf", None, 10),
    ]
    kept = _one_shard_family_of(duplicated)
    assert [e[0] for e in kept] == [
        "QwQ-32B-BF16-00001-of-00002.gguf",
        "QwQ-32B-BF16-00002-of-00002.gguf",
    ]
    assert sum(e[2] for e in kept) == 20
    split = [(f"m-0000{i}-of-00003.gguf", None, 5) for i in (1, 2, 3)]
    assert _one_shard_family_of(split) == split


def test_the_remote_load_path_auto_selects_the_root_checkpoint():
    """ModelConfig.from_identifier must select the root checkpoint for a bare repo id, like other paths."""
    import inspect

    from utils.models import model_config as mc

    source = inspect.getsource(mc.ModelConfig.from_identifier)
    # The shared scope (root rows, or a curated repo's default subdirectory) keyed on the bare id.
    assert "for v in default_gguf_scope(" in source
    assert "identifier," in source
    # The filter reads the ADVERTISED identity, the same one the lister assigns each row.
    assert "key = lambda v: _qualified_variant_name(v.filename, v.quant)" in source
