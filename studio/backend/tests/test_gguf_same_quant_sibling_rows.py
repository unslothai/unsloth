# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two builds of one quant in one directory are two rows.

The GGUF naming convention puts the weight encoding last but for an optional type field, so a
build tag past the quant token names a SECOND checkpoint at that quant. Repos publish them in
bulk: ``ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`` ships an ``-mtp`` speculative-decoding copy beside
every quant, and ``bartowski/gemma-2-9b-it-GGUF`` ships ``-fp16``/``-f32``/``-Q8`` embedding
builds beside the plain ones.

Keying both on the bare token gave the pair one row, and because ``-`` sorts before ``.`` the
survivor was the TAGGED file: the row said ``Q4_K_M`` and fetched ``Q4_K_M-fp16``. The other
build was not merely unlabelled, it was unreachable.

The key stays a pure function of the path -- a cache scan sees fewer files than the remote
listing, and a key that read the file SET would disagree between them and strand a finished
download as incomplete -- so the plain build keeps its historical bare key and only the tagged
sibling is qualified. The resume and lookup paths are pinned here alongside for that reason.
"""

from __future__ import annotations

import pytest

from core.inference.llama_cpp import _gguf_files_for_variant
from hub.utils.gguf import (
    GgufVariantInfo,
    _apply_gguf_display_labels,
    extract_quant_label,
    gguf_variant_key,
    group_gguf_variant_files,
    list_local_gguf_variants,
)
from hub.utils.gguf_plan import build_gguf_variant_plans, plan_for_variant
from hub.utils.inventory_scan import complete_snapshot_variants
from utils.models.model_config import _find_local_gguf_by_variant, _gguf_variant_key

# Live listing of ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF, two quants' worth.
MTP_FILES = [
    ("Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf", 12_100_000_000),
    ("Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf", 11_750_000_000),
    ("Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf", 10_400_000_000),
    ("Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf", 10_050_000_000),
]

# Live listing of bartowski/gemma-2-9b-it-GGUF, the colliding quants.
EMBEDDING_PRECISION_FILES = [
    ("gemma-2-9b-it-Q4_K_M-fp16.gguf", 6_300_000_000),
    ("gemma-2-9b-it-Q4_K_M.gguf", 5_760_000_000),
    ("gemma-2-9b-it-Q8_0.gguf", 9_820_000_000),
    ("gemma-2-9b-it-Q8_0_L.gguf", 10_680_000_000),
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
        # The plain build keeps the historical bare key, so every stored pin resolves.
        ("Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf", "IQ3_S"),
        ("gemma-2-9b-it-Q4_K_M.gguf", "Q4_K_M"),
        # A build tag past the token identifies the file instead.
        ("Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf", "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp"),
        ("gemma-2-9b-it-Q4_K_M-fp16.gguf", "gemma-2-9b-it-Q4_K_M-fp16"),
        ("gemma-2-9b-it-Q8_0_L.gguf", "gemma-2-9b-it-Q8_0_L"),
        ("Meta-Llama-3.1-8B-Instruct-Q4_0_4_4.gguf", "Meta-Llama-3.1-8B-Instruct-Q4_0_4_4"),
        # Shard suffix is stripped before the token is read, so every shard keys alike.
        ("model-Q4_K_M-mtp-00001-of-00003.gguf", "model-Q4_K_M-mtp"),
        ("model-Q4_K_M-00002-of-00003.gguf", "Q4_K_M"),
        # MaziyarPanahi's non-canonical split leaves ``.gguf`` twice; it still closes at its token.
        ("Llama-3.3-70B-Instruct.Q6_K.gguf-00001-of-00006.gguf", "Q6_K"),
        # TheBloke's dot separator, and a leading imatrix marker, are not build tags.
        ("llama-2-7b-chat.Q4_K_M.gguf", "Q4_K_M"),
        ("Melody1437-26B-A4B.i1-Q4_K_M.gguf", "Q4_K_M"),
        # A parent directory named the quant: the basename is the build's own name and the
        # directory rule decides, exactly as before.
        ("Q4_K_M/model.gguf", "Q4_K_M"),
        ("Q6_K/model-3.5bpw.gguf", "Q6_K-3.5bpw"),
        # The bpw modifier is part of the token, not a tag trailing it.
        ("Llama-3.1-8B-Instruct-IQ4_XS-3.57bpw.gguf", "IQ4_XS-3.57bpw"),
    ],
)
def test_variant_key(path, expected):
    assert gguf_variant_key(path) == expected


def test_the_loaders_copy_agrees():
    """``model_config._gguf_variant_key`` is a mirror; drift means a row that misloads."""
    for path, _ in MTP_FILES + EMBEDDING_PRECISION_FILES:
        assert _gguf_variant_key(path) == gguf_variant_key(path)


# --------------------------------------------------------------------------------------
# The rows
# --------------------------------------------------------------------------------------


def test_every_mtp_sibling_is_its_own_row():
    rows = group_gguf_variant_files(MTP_FILES)
    assert len(rows) == len(MTP_FILES)
    assert {filename for filename, _ in rows.values()} == {path for path, _ in MTP_FILES}
    assert rows["IQ3_S"] == ("Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf", 11_750_000_000)
    assert rows["Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp"] == (
        "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
        12_100_000_000,
    )


def test_the_plain_build_is_reachable_beside_its_embedding_precision_sibling():
    """The row a user reads as ``Q4_K_M`` must fetch ``Q4_K_M``, not the ``-fp16`` build that
    sorts before it."""
    rows = group_gguf_variant_files(EMBEDDING_PRECISION_FILES)
    assert len(rows) == len(EMBEDDING_PRECISION_FILES)
    assert rows["Q4_K_M"][0] == "gemma-2-9b-it-Q4_K_M.gguf"
    assert rows["Q8_0"][0] == "gemma-2-9b-it-Q8_0.gguf"


def test_a_local_folder_lists_both_builds(tmp_path):
    snapshot = _materialize(tmp_path / "snap", MTP_FILES)
    variants, _ = list_local_gguf_variants(str(snapshot))
    assert {variant.filename for variant in variants} == {path for path, _ in MTP_FILES}


def test_the_tagged_row_says_which_build_it_is():
    """Both rows show the same quant, so the tagged one prints what tells them apart while the
    plain one keeps the label it has always had."""
    rows = group_gguf_variant_files(MTP_FILES)
    variants = [
        GgufVariantInfo(filename = filename, quant = key, size_bytes = size)
        for key, (filename, size) in rows.items()
    ]
    _apply_gguf_display_labels(variants)
    labels = {variant.quant: variant.display_label for variant in variants}
    assert labels["IQ3_S"] is None
    assert labels["Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp"] == "IQ3_S · Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp"


def test_a_lone_tagged_build_keeps_its_bare_label():
    """The key cannot read the listing, so it qualifies a tagged name whether or not a plain
    sibling exists (``google/gemma-4-31B-it-qat-q4_0-gguf`` ships one build). No second row
    carries the token there, so nothing needs telling apart and the row reads as it always has."""
    path = "gemma-4-31B_q4_0-it.gguf"
    variants = [
        GgufVariantInfo(filename = path, quant = gguf_variant_key(path), size_bytes = 17)
    ]
    _apply_gguf_display_labels(variants)
    assert variants[0].display_label == "q4_0"


def test_a_shared_quant_still_collapses_its_own_shards():
    """A genuinely split GGUF is ONE build, and both of these are."""
    sharded = [
        ("model-Q4_K_M-00001-of-00002.gguf", 10),
        ("model-Q4_K_M-00002-of-00002.gguf", 15),
        ("model-Q4_K_M-mtp-00001-of-00002.gguf", 20),
        ("model-Q4_K_M-mtp-00002-of-00002.gguf", 25),
    ]
    rows = group_gguf_variant_files(sharded)
    assert rows == {
        "Q4_K_M": ("model-Q4_K_M-00001-of-00002.gguf", 25),
        "model-Q4_K_M-mtp": ("model-Q4_K_M-mtp-00001-of-00002.gguf", 45),
    }


# --------------------------------------------------------------------------------------
# The download plan and the resume path
# --------------------------------------------------------------------------------------


def test_each_row_plans_only_its_own_weights():
    plans = build_gguf_variant_plans([_Sibling(path, size) for path, size in MTP_FILES])
    assert set(plans) == {gguf_variant_key(path).lower() for path, _ in MTP_FILES}
    for path, size in MTP_FILES:
        plan = plans[gguf_variant_key(path).lower()]
        assert plan.main_filenames == frozenset({path})
        assert plan.main_size_bytes == size


def test_a_half_downloaded_snapshot_keys_the_same_as_the_remote_listing():
    """The key is a pure function of the path. A cache scan that sees one file of the pair must
    key it exactly as the full listing did, or a finished download reads as incomplete."""
    whole = group_gguf_variant_files(MTP_FILES)
    for path, size in MTP_FILES:
        subset = group_gguf_variant_files([(path, size)])
        assert list(subset) == [gguf_variant_key(path)]
        assert subset[gguf_variant_key(path)] == whole[gguf_variant_key(path)]


def test_a_finished_download_reads_as_complete(tmp_path):
    """Each build downloaded on its own, as the worker fetches it: the row it was listed under
    must still be the row the resume scan reports complete."""
    for path, size in MTP_FILES:
        snapshot = _materialize(tmp_path / gguf_variant_key(path), [(path, size)])
        variants, _ = list_local_gguf_variants(str(snapshot))
        assert [variant.quant for variant in variants] == [gguf_variant_key(path)]
        assert gguf_variant_key(path) in complete_snapshot_variants(str(snapshot))


def test_a_chosen_row_loads_its_own_build(tmp_path):
    snapshot = _materialize(tmp_path / "snap", MTP_FILES)
    paths = [path for path, _ in MTP_FILES]
    for path, _ in MTP_FILES:
        key = gguf_variant_key(path)
        assert _find_local_gguf_by_variant(str(snapshot), key) == str(snapshot / path)
        assert _gguf_files_for_variant(paths, key.lower()) == [path]


def test_a_legacy_bare_pin_still_resolves_when_only_the_tagged_build_exists():
    """Qualifying a root-level name re-keys the single-build repos that never collided
    (``google/gemma-4-31B-it-qat-q4_0-gguf`` ships one ``gemma-4-31B_q4_0-it.gguf``). The bare
    spelling every stored pin uses has to keep resolving there, exactly as it does for a
    directory-qualified key."""
    plans = build_gguf_variant_plans([_Sibling("gemma-4-31B_q4_0-it.gguf", 17)])
    assert set(plans) == {"gemma-4-31b_q4_0-it"}
    plan = plan_for_variant(plans, "q4_0")
    assert plan is not None
    assert plan.main_filenames == frozenset({"gemma-4-31B_q4_0-it.gguf"})


def test_a_bare_pin_refuses_to_pick_between_two_builds():
    """With both builds published the bare token names neither: serving one would download a
    checkpoint the user did not ask for, which is the bug this file exists for."""
    plans = build_gguf_variant_plans(
        [_Sibling(path, size) for path, size in MTP_FILES if "IQ3_S" in path]
    )
    assert plan_for_variant(plans, "iq3_s") is plans["iq3_s"]
    assert plan_for_variant(plans, "iq3_s").main_filenames == frozenset(
        {"Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"}
    )
    tagged = build_gguf_variant_plans(
        [_Sibling("model-IQ3_S-mtp.gguf", 1), _Sibling("model-IQ3_S-fp16.gguf", 2)]
    )
    assert plan_for_variant(tagged, "iq3_s") is None


def test_delete_removes_only_the_chosen_build(tmp_path):
    from routes.models import _delete_gguf_variant_files

    snapshot = _materialize(tmp_path / "snap", MTP_FILES)
    count, _ = _delete_gguf_variant_files(snapshot, "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp")
    assert count == 1
    assert not (snapshot / "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf").exists()
    assert (snapshot / "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf").exists()


def test_the_quant_label_is_unchanged_for_both_builds():
    """Only the KEY gains the tag. The label feeding the preference order, the endian check and
    the VRAM estimate still reads the token, so nothing downstream sees a new quant."""
    for path, _ in MTP_FILES:
        assert extract_quant_label(path) in ("IQ3_S", "IQ3_XXS")


def test_paths_are_keyed_the_same_on_windows_separators():
    assert gguf_variant_key("sub\\model-Q4_K_M-mtp.gguf") == gguf_variant_key(
        "sub/model-Q4_K_M-mtp.gguf"
    )
