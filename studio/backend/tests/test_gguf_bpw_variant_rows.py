# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A base quant published at several bit widths is several variants.

``byteshape/Llama-3.1-8B-Instruct-GGUF`` ships 18 GGUFs at four base quants, telling
them apart with a bits-per-weight modifier the loader's label has always kept and the
hub's dropped (``Llama-3.1-8B-Instruct-IQ3_S-2.54bpw.gguf`` through ``-3.31bpw``).
Every file sits in the repo root, so there is no directory to qualify the key by, and
keying on the bare token folded 18 checkpoints into 4 rows: 14 of them unselectable,
each row advertising one file's size while the download plan fetched a different one.

The modifier therefore belongs to the key. Repos without one must key exactly as
before, and a genuinely split GGUF must still collapse its shards into one row, so
both are pinned here alongside.
"""

from __future__ import annotations

import pytest

from core.inference.llama_cpp import _gguf_files_for_variant
from hub.utils.gguf import (
    GgufVariantInfo,
    _apply_gguf_display_labels,
    extract_quant_label,
    extract_quant_token,
    gguf_variant_family,
    gguf_variant_key,
    group_gguf_variant_files,
    list_local_gguf_variants,
    quant_token_with_bpw,
)
from hub.utils.gguf_plan import build_gguf_variant_plans, is_main_gguf_variant_path
from hub.utils.inventory_scan import complete_snapshot_variants
from utils.models.model_config import (
    _extract_quant_label,
    _find_local_gguf_by_variant,
    _gguf_variant_key,
)

# Live listing of byteshape/Llama-3.1-8B-Instruct-GGUF, one base quant's worth.
BPW_FILES = [
    ("Llama-3.1-8B-Instruct-IQ4_XS-3.57bpw.gguf", 3_587_540_000),
    ("Llama-3.1-8B-Instruct-IQ4_XS-3.94bpw.gguf", 3_958_211_616),
    ("Llama-3.1-8B-Instruct-IQ4_XS-4.05bpw.gguf", 4_069_393_440),
]

# One bpw variant, split across shards, which must stay one row.
BPW_SHARDED_FILES = [
    ("IQ4_XS-3.57bpw/model-IQ4_XS-3.57bpw-00001-of-00002.gguf", 40),
    ("IQ4_XS-3.57bpw/model-IQ4_XS-3.57bpw-00002-of-00002.gguf", 25),
]


class _Sibling:
    def __init__(self, rfilename: str, size: int) -> None:
        self.rfilename = rfilename
        self.size = size
        self.lfs = {"sha256": f"hash-of-{rfilename}"}


def _materialize(root, files):
    for path, _ in files:
        target = root / path
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"x" * 512)
    (root / "config.json").write_text("{}")
    return root


# --------------------------------------------------------------------------------------
# The key itself
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        # The modifier trailing the quant token is part of the token.
        ("Llama-3.1-8B-Instruct-IQ3_S-2.54bpw.gguf", "IQ3_S-2.54bpw"),
        ("Llama-3.1-8B-Instruct-IQ3_S-3.31bpw.gguf", "IQ3_S-3.31bpw"),
        ("Qwen-Image-2512-Q6_K-6.61bpw.gguf", "Q6_K-6.61bpw"),
        # Integer widths and an uppercase spelling both occur on the Hub.
        ("Step-3.5-Flash-IQ3_S-3BPW.gguf", "IQ3_S-3BPW"),
        # Shard suffix is stripped first, so every shard keys alike.
        ("model-IQ4_XS-3.57bpw-00001-of-00002.gguf", "IQ4_XS-3.57bpw"),
        # The modifier can live on the directory instead, when the basename has no quant.
        ("IQ4_XS-3.53bpw/model.gguf", "IQ4_XS-3.53bpw"),
        # A modifier that does not trail the token is not part of it: it belongs to
        # something else in the name and cannot be relied on to identify the file.
        ("flux1-dev-Q8_0-fp32-08.577bpw.gguf", "Q8_0"),
        # No modifier at all: the historical key, unchanged.
        ("Wan2.2-TI2V-5B-Q8_0.gguf", "Q8_0"),
        ("BF16/DeepSeek-R1-BF16-00001-of-00003.gguf", "BF16"),
    ],
)
def test_bpw_variant_key(path, expected):
    assert gguf_variant_key(path) == expected


def test_key_is_unchanged_where_there_is_no_modifier():
    """The persisted variant identity must not churn for the repos that have no bpw
    modifier, which is all but a handful on the whole Hub."""
    for path in (
        "Wan2.2-TI2V-5B-Q4_K_M.gguf",
        "Qwen3-8B-UD-Q8_K_XL.gguf",
        "BF16/DeepSeek-R1-BF16-00002-of-00003.gguf",
        "DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf",
        "MXFP4_MOE/model-MXFP4_MOE-00001-of-00003.gguf",
    ):
        assert gguf_variant_key(path) == extract_quant_label(path)
        assert quant_token_with_bpw(path) == extract_quant_token(path)


def test_a_bare_bpw_name_is_not_truncated_at_its_decimal_point():
    """Folder names and stored variant strings arrive without a ``.gguf`` extension, and
    cutting at the last dot would leave ``IQ4_XS-3``, silently dropping the modifier and
    keying the folder as the bare quant again."""
    assert quant_token_with_bpw("IQ4_XS-3.53bpw") == "IQ4_XS-3.53bpw"
    assert quant_token_with_bpw("Llama-3.1-8B-Instruct-IQ4_XS-3.57bpw") == "IQ4_XS-3.57bpw"
    assert quant_token_with_bpw("Q6_K") == "Q6_K"


def test_an_interrupted_bpw_folder_is_labelled_like_its_row(tmp_path):
    """An empty ``<quant>/`` folder marks a variant partial. Labelled by the bare token it
    would mark a row that no longer exists, and the real one would read as absent."""
    from hub.utils.gguf import list_empty_gguf_variant_dirs

    snapshot = tmp_path / "models--acme--m" / "snapshots" / "rev"
    (snapshot / "IQ4_XS-3.57bpw").mkdir(parents = True)
    assert list_empty_gguf_variant_dirs("acme/m", root = tmp_path) == {"IQ4_XS-3.57bpw"}


def test_files_at_one_base_quant_do_not_share_a_key():
    assert len({gguf_variant_key(path) for path, _ in BPW_FILES}) == len(BPW_FILES)
    assert {extract_quant_token(path) for path, _ in BPW_FILES} == {"IQ4_XS"}


def test_shards_of_one_bpw_variant_share_a_key_and_a_family():
    assert {gguf_variant_key(path) for path, _ in BPW_SHARDED_FILES} == {"IQ4_XS-3.57bpw"}
    assert len({gguf_variant_family(path) for path, _ in BPW_SHARDED_FILES}) == 1


def test_loader_mirror_agrees_on_bpw_labels():
    """``model_config._gguf_variant_token`` is a copy of ``quant_token_with_bpw``; drift
    means a row keyed one way and loaded another. The loader's own label already carried
    the modifier, which is the divergence this closes."""
    corpus = [path for path, _ in (*BPW_FILES, *BPW_SHARDED_FILES)] + [
        "Llama-3.1-8B-Instruct-IQ3_S-2.54bpw.gguf",
        "Step-3.5-Flash-IQ3_S-3BPW.gguf",
        "IQ4_XS-3.53bpw/model.gguf",
        "flux1-dev-Q8_0-fp32-08.577bpw.gguf",
        "Reflection-70b-PreciseQuant-6bpw.gguf",
        "Wan2.2-TI2V-5B-Q8_0.gguf",
        "distilled/ltx-2.3-22b-distilled-Q6_K.gguf",
        # Bare names, as a <quant>/ folder and a stored variant string arrive.
        "IQ4_XS-3.53bpw",
        "Llama-3.1-8B-Instruct-IQ4_XS-3.57bpw",
        "Q6_K",
    ]
    assert [_gguf_variant_key(p) for p in corpus] == [gguf_variant_key(p) for p in corpus]
    for path, _ in BPW_FILES:
        assert _extract_quant_label(path) == gguf_variant_key(path)


# --------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------


def test_each_bit_width_gets_its_own_row_at_its_own_size():
    rows = group_gguf_variant_files(BPW_FILES)
    assert len(rows) == 3
    assert {key: size for key, (_, size) in rows.items()} == {
        "IQ4_XS-3.57bpw": 3_587_540_000,
        "IQ4_XS-3.94bpw": 3_958_211_616,
        "IQ4_XS-4.05bpw": 4_069_393_440,
    }


def test_bpw_shards_stay_one_row_summing_to_the_whole_file():
    rows = group_gguf_variant_files(BPW_SHARDED_FILES)
    assert list(rows) == ["IQ4_XS-3.57bpw"]
    assert rows["IQ4_XS-3.57bpw"] == (BPW_SHARDED_FILES[0][0], 65)


def test_local_lister_matches_the_remote_shape(tmp_path):
    snapshot = _materialize(tmp_path / "snap", BPW_FILES)
    variants, _ = list_local_gguf_variants(str(snapshot))
    assert sorted(v.quant for v in variants) == sorted(
        gguf_variant_key(path) for path, _ in BPW_FILES
    )
    assert all(v.size_bytes == 512 for v in variants)


def test_a_bpw_row_reads_as_its_own_quant_and_needs_no_scope_suffix():
    """``IQ4_XS-3.57bpw`` is the label byteshape publishes and the loader reports, so it
    is showable as it stands. Only a key qualified by a path needs a second name."""
    variants = [
        GgufVariantInfo(filename = path, quant = gguf_variant_key(path), size_bytes = 1)
        for path, _ in BPW_FILES
    ]
    _apply_gguf_display_labels(variants)
    assert {v.display_label for v in variants} == {None}


# --------------------------------------------------------------------------------------
# The row must be downloadable, loadable and complete-able under its own key
# --------------------------------------------------------------------------------------


def test_every_row_key_matches_exactly_its_own_file():
    for path, _ in BPW_FILES:
        key = gguf_variant_key(path)
        owned = [other for other, _ in BPW_FILES if is_main_gguf_variant_path(other, key)]
        assert owned == [path], key


def test_download_plan_fetches_one_bit_width_not_three():
    plans = build_gguf_variant_plans([_Sibling(path, size) for path, size in BPW_FILES])
    assert len(plans) == 3
    for path, size in BPW_FILES:
        plan = plans[gguf_variant_key(path).lower()]
        assert plan.main_filenames == frozenset({path})
        assert plan.main_size_bytes == size


def test_download_plan_keeps_every_shard_of_a_split_bpw_variant():
    plans = build_gguf_variant_plans([_Sibling(p, s) for p, s in BPW_SHARDED_FILES])
    assert list(plans) == ["iq4_xs-3.57bpw"]
    assert plans["iq4_xs-3.57bpw"].main_filenames == frozenset(p for p, _ in BPW_SHARDED_FILES)
    assert plans["iq4_xs-3.57bpw"].main_size_bytes == 65


def test_readiness_agrees_with_the_rows(tmp_path):
    """A finished download must read as complete under the same key it was listed by."""
    snapshot = _materialize(tmp_path / "snap", BPW_FILES)
    variants, _ = list_local_gguf_variants(str(snapshot))
    assert {v.quant for v in variants} <= complete_snapshot_variants(str(snapshot))


def test_a_chosen_row_loads_its_own_bit_width(tmp_path):
    snapshot = _materialize(tmp_path / "snap", BPW_FILES)
    for path, _ in BPW_FILES:
        key = gguf_variant_key(path)
        assert _find_local_gguf_by_variant(str(snapshot), key) == str(snapshot / path)
        assert _gguf_files_for_variant([p for p, _ in BPW_FILES], key) == [path]


def test_llama_cpp_hands_over_every_shard_of_one_bit_width():
    paths = [path for path, _ in BPW_SHARDED_FILES]
    assert _gguf_files_for_variant(paths, "iq4_xs-3.57bpw") == paths


def test_delete_removes_only_the_chosen_bit_width(tmp_path):
    from routes.models import _delete_gguf_variant_files

    snapshot = _materialize(tmp_path / "snap", BPW_FILES)
    count, _ = _delete_gguf_variant_files(snapshot, "IQ4_XS-3.94bpw")
    assert count == 1
    assert not (snapshot / "Llama-3.1-8B-Instruct-IQ4_XS-3.94bpw.gguf").exists()
    assert (snapshot / "Llama-3.1-8B-Instruct-IQ4_XS-3.57bpw.gguf").exists()
    assert (snapshot / "Llama-3.1-8B-Instruct-IQ4_XS-4.05bpw.gguf").exists()


def test_empty_folder_cleanup_takes_only_the_variants_own_folder(tmp_path):
    """Cleanup after a delete removes the emptied ``<quant>/`` folder. A bpw key owns
    ``IQ4_XS-3.57bpw/`` and nothing else; reducing it to its bare token would take a
    sibling row's ``IQ4_XS/`` with it."""
    from hub.services.models.deletion import _remove_empty_variant_dirs

    snapshot = tmp_path / "repo" / "snapshots" / "rev"
    for folder in ("IQ4_XS", "IQ4_XS-3.57bpw", "model-IQ4_XS-3.57bpw", "IQ4_XS-3.94bpw"):
        (snapshot / folder).mkdir(parents = True)

    class _Repo:
        repo_path = str(tmp_path / "repo")

    # Both spellings of this variant's own folder go, as they do for a bare quant.
    removed, failures = _remove_empty_variant_dirs([_Repo()], "IQ4_XS-3.57bpw")
    assert (removed, failures) == (2, [])
    assert not (snapshot / "IQ4_XS-3.57bpw").exists()
    assert not (snapshot / "model-IQ4_XS-3.57bpw").exists()
    assert (snapshot / "IQ4_XS").is_dir()
    assert (snapshot / "IQ4_XS-3.94bpw").is_dir()


def test_empty_folder_cleanup_still_takes_a_bare_quant_folder(tmp_path):
    """The unqualified path is unchanged: a plain ``Q6_K`` variant still clears
    ``Q6_K/`` and the ``<model>-Q6_K/`` spelling of the same quant."""
    from hub.services.models.deletion import _remove_empty_variant_dirs

    snapshot = tmp_path / "repo" / "snapshots" / "rev"
    for folder in ("Q6_K", "model-Q6_K", "Q4_K_M"):
        (snapshot / folder).mkdir(parents = True)

    class _Repo:
        repo_path = str(tmp_path / "repo")

    removed, failures = _remove_empty_variant_dirs([_Repo()], "Q6_K")
    assert (removed, failures) == (2, [])
    assert (snapshot / "Q4_K_M").is_dir()


def test_a_local_subset_keys_the_same_as_the_whole_repo(tmp_path):
    """The key is a pure function of the path, so a half-downloaded snapshot and the
    remote listing cannot disagree and strand a finished download as incomplete."""
    whole = group_gguf_variant_files(BPW_FILES)
    for path, size in BPW_FILES:
        subset = group_gguf_variant_files([(path, size)])
        assert list(subset) == [gguf_variant_key(path)]
        assert subset[gguf_variant_key(path)] == whole[gguf_variant_key(path)]


def test_paths_are_keyed_the_same_on_windows_separators():
    assert gguf_variant_key("IQ4_XS-3.53bpw\\model.gguf") == gguf_variant_key(
        "IQ4_XS-3.53bpw/model.gguf"
    )
