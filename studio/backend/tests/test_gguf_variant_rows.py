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


def test_the_route_model_carries_display_label_to_the_picker():
    """The route builds ``models.models.GgufVariantDetail``, not the hub twin. When only the
    twin declared ``display_label`` pydantic dropped the kwarg without a word, so every
    qualified row reached the picker labelled with its whole relative path. Both models
    have to declare it or the qualified label silently never ships."""
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
    """The label is only for keys a picker cannot show. A plain quant token must not grow
    one, or every ordinary row gains a redundant second name."""
    from models.models import GgufVariantDetail as RouteDetail

    row = RouteDetail(filename = "model-Q4_K_M.gguf", quant = "Q4_K_M", size_bytes = 1)
    assert row.model_dump()["display_label"] is None


# --------------------------------------------------------------------------------------
# A row must not reach outside its own checkpoint
# --------------------------------------------------------------------------------------


def test_a_discarded_shard_family_leaves_the_download_targets_too():
    """Narrowing only ``main_files`` left the copy in ``target_filenames`` /
    ``required_hashes`` / ``download_size_bytes``. The worker fetched both copies, reclaim
    then deleted the unchosen one (it is absent from the narrowed ``main_hashes``), and the
    manifest still required it -- so the finished job reported partial and re-downloaded a
    multi-gigabyte checkpoint on every retry."""
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
    # Everything the worker fetches and the manifest checks agrees with that choice.
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


def test_a_qualified_key_is_an_explicit_checkpoint_request():
    """``resolve_local_gguf`` falls back to the first local variant for a ``:tag`` that
    names no quant, which is right for ``:latest`` and wrong for one of our own rows. The
    full-match rejected every slash-qualified key, so asking for an absent checkpoint was
    answered by whichever one happened to be downloaded, under the requested model id."""
    from core.inference.openai_auto_download import looks_like_quant

    assert looks_like_quant("distilled/ltx-2.3-22b-distilled-Q6_K") is True
    assert looks_like_quant("distilled-1.1/ltx-2.3-22b-distilled-1.1-Q6_K") is True
    # Unchanged for the spellings it already judged.
    assert looks_like_quant("Q6_K") is True
    assert looks_like_quant("IQ4_XS-3.53bpw") is True
    assert looks_like_quant("latest") is False
    assert looks_like_quant("8b") is False
    assert looks_like_quant(None) is False


def test_the_exact_key_wins_over_the_bare_quant_label():
    """A bare label names every checkpoint carrying that quant, so the repo-root ``Q6_K``
    row matched the qualified files too and then took whichever sorted first -- reporting
    the sum of every checkpoint as its size and revealing another one's file."""
    from routes.models import _main_variant_rank

    # The request VERBATIM: folding it up front would strip a qualified key's path punctuation.
    assert _main_variant_rank("ltx-2.3-22b-dev-Q6_K.gguf", "Q6_K") == 0
    assert _main_variant_rank("distilled/ltx-2.3-22b-distilled-Q6_K.gguf", "Q6_K") == 1
    assert _main_variant_rank("ltx-2.3-22b-dev-Q4_K_M.gguf", "Q6_K") is None
    # And the qualified row is exact under its own key, while the root file is not a match.
    qualified = "distilled/ltx-2.3-22b-distilled-Q6_K"
    assert _main_variant_rank("distilled/ltx-2.3-22b-distilled-Q6_K.gguf", qualified) == 0
    assert _main_variant_rank("ltx-2.3-22b-dev-Q6_K.gguf", qualified) is None
    # The bare spelling keeps its hyphen/underscore folding; a qualified one keeps its path.
    assert _main_variant_rank("model-UD-Q4_K_XL.gguf", "udq4kxl") == 0
    assert _main_variant_rank("exp-a/model-Q6_K.gguf", "expa/model-Q6_K") is None
    assert _main_variant_rank("exp-a/model-Q6_K.gguf", "exp-a/model-q6_k") == 0


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
    """Three consumers read their variant identities from these two listers rather than from the
    hub copy: the /v1 local index (``local_model_resolver`` builds ``entry.variants`` here), the
    remote VRAM preflight (which looks for a matching ``v.quant`` to get ``main_bytes``), and the
    picker's own remote sizing. While they grouped on the bare label the qualified rows were
    invisible to all three, and a slash-qualified suffix is now an explicit variant, so the miss
    is a 404 rather than a fallback onto some other checkpoint."""
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
    # Every advertised key resolves back to its OWN file, which is what the row promises.
    for variant in variants:
        resolved = _find_local_gguf_by_variant(str(snapshot), variant.quant)
        assert resolved is not None, variant.quant
        assert _gguf_variant_key(Path(resolved).relative_to(snapshot).as_posix()) == variant.quant


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
    # Each row is sized for its own checkpoint, not the sum of every checkpoint at that quant.
    assert all(size > 0 for size in sizes.values())


def test_an_unknown_layout_row_keeps_the_label_it_always_had(tmp_path):
    """The qualified key is for several checkpoints sharing one quant, nothing else. A file with
    no recognised quant token has always been listed here under this module's label (the last
    hyphenated segment) rather than the whole stem, and renaming those rows would break every pin
    that holds one -- for no benefit, since there is no ambiguity to resolve."""
    from utils.models.model_config import list_local_gguf_variants

    snapshot = tmp_path / "snap"
    (snapshot / "BF16").mkdir(parents = True)
    (snapshot / "Qwen3.6-27B-MTP-001-of-002.gguf").write_bytes(b"x" * 100)
    (snapshot / "BF16" / "gemma-4-12b-it-Q8_0-001-of-002.gguf").write_bytes(b"x" * 40)
    (snapshot / "config.json").write_text("{}")

    variants, _ = list_local_gguf_variants(str(snapshot))
    assert sorted(v.quant for v in variants) == ["MTP", "Q8_0"]


def test_a_shared_container_directory_still_answers_its_bare_quant():
    """A repo that files every variant under one container (``weights/model-Q4_K_M.gguf``)
    qualifies every key, because the key is a pure function of the path and cannot know the
    directory disambiguates nothing. Every stored pin and every explicit repo:Q4_K_M then missed
    the plan map, and the worker exited with 'No GGUF shards matching variant'."""
    from types import SimpleNamespace

    from hub.utils.gguf_plan import build_gguf_variant_plans, plan_for_variant

    siblings = [
        SimpleNamespace(rfilename = f"weights/model-{q}.gguf", size = 10, lfs = None)
        for q in ("Q4_K_M", "Q6_K")
    ]
    plans = build_gguf_variant_plans(siblings)
    # The rows are still one per checkpoint, keyed on the path.
    assert set(plans) == {"weights/model-q4_k_m", "weights/model-q6_k"}
    # ... and the bare pin still resolves, because it names exactly one of them.
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
    # The exact key still resolves, which is what the advertised row asks for.
    assert plan_for_variant(plans, "distilled/ltx-2.3-22b-distilled-q6_k") is not None


def test_a_bare_local_id_keeps_the_checkpoint_a_plain_load_takes():
    """A plain local load resolves through non-recursive detect_gguf_model and always takes the
    repo root; the /v1 index ranks on the key text and would hand a bare id an equally good
    ``distilled/...`` row that sorts earlier. One id, two answers, different weights."""
    from core.inference.openai_auto_download import preferred_quant

    quants = (
        "Q6_K",
        "distilled/ltx-2.3-22b-distilled-Q6_K",
        "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q6_K",
    )
    unqualified = tuple(q for q in quants if "/" not in q)
    assert preferred_quant(unqualified) == "Q6_K"


def test_bpw_precisions_stay_separately_selectable(tmp_path):
    """_extract_quant_label deliberately keeps the bpw modifier so byteshape's IQ4_XS at 3.53,
    3.97 and 4.19 stay three rows. The token extractor drops it, so routing these listers through
    the key merged all three under IQ4_XS and an explicit request for the formerly advertised
    spelling missed."""
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
    """_is_big_endian_gguf_path reads a quant TOKEN so it can tell a parent-only quant from a
    big-endian build. Handing it the qualified key made it misread the path and drop the file
    from every plan, so the advertised row could never be downloaded."""
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
    """Admission rejects against its own size map before the worker plan is ever consulted, so
    the container-directory fallback has to exist on both. Without it a legacy org/repo:Q4_K_M
    got a 404 and the worker fix was unreachable."""
    from core.inference.openai_auto_download import _match_variant

    variants = {"weights/model-Q4_K_M": 10, "weights/model-Q6_K": 20}
    assert _match_variant("Q4_K_M", variants) == "weights/model-Q4_K_M"
    assert _match_variant("weights/model-q6_k", variants) == "weights/model-Q6_K"
    assert _match_variant("Q8_0", variants) is None
    # Ambiguous stays a miss: with two checkpoints at one quant the bare name names neither.
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
    """_find_local_gguf_by_variant picks the right file, but the returned config dropped the
    variant, so the load intent carried none and llama.cpp recorded the bare label off the
    filename. /status then named the root row for a qualified checkpoint, and the deletion guard
    compared that bare label against the selected key and let the delete through."""
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
    # A bare request still records the bare identity it asked for.
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
    """A repo filing its sole Q4_K_M under a shared container qualifies that key, so every stored
    pin and every explicit ``repo:Q4_K_M`` names it by quant alone -- which ``plan_for_variant``
    admits for the download. Matching only the qualified key on delete answered "not found" and
    left the weights on disk."""
    from hub.services.models.deletion import _delete_gguf_variant_from_repos
    from hub.utils.gguf_plan import build_gguf_variant_plans, plan_for_variant

    repo, snap = _cache_repo(tmp_path, "org/Model-GGUF", ["weights/model-Q4_K_M.gguf"])
    # The download side already resolves the bare name here; keep the two in step.
    plans = build_gguf_variant_plans([_Sibling("weights/model-Q4_K_M.gguf", 64)])
    assert set(plans) == {"weights/model-q4_k_m"}
    assert plan_for_variant(plans, "Q4_K_M") is not None

    _delete_gguf_variant_from_repos("org/Model-GGUF", "Q4_K_M", [repo], None, root = tmp_path)
    assert not (snap / "weights" / "model-Q4_K_M.gguf").is_symlink()


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

    # The qualified key still deletes exactly its own checkpoint.
    _delete_gguf_variant_from_repos(
        "org/Model-GGUF", "distilled/model-Q4_K_M", [repo], None, root = tmp_path
    )
    assert not (snap / "distilled" / "model-Q4_K_M.gguf").is_symlink()
    assert (snap / "distilled-1.1" / "model-Q4_K_M.gguf").is_symlink()


def test_a_bare_quant_that_is_its_own_key_still_deletes_only_itself(tmp_path):
    """The overwhelmingly common shape: the root file's key IS the bare quant, so the alias
    fallback must not fire and must not reach a qualified sibling."""
    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    repo, snap = _cache_repo(
        tmp_path, "org/Model-GGUF", ["model-Q4_K_M.gguf", "distilled/model-Q4_K_M.gguf"]
    )
    _delete_gguf_variant_from_repos("org/Model-GGUF", "Q4_K_M", [repo], None, root = tmp_path)
    assert not (snap / "model-Q4_K_M.gguf").is_symlink()
    assert (snap / "distilled" / "model-Q4_K_M.gguf").is_symlink()


def test_a_qualified_key_whose_basename_ends_in_be_still_resolves_for_loading():
    """The endian predicate reads a quant TOKEN -- whether the quant came from the parent
    directory only. Handed the path-qualified key instead, it cannot find that string in the
    basename or the parent and reads distilled/Q4_K_M/foo-be.gguf as a big-endian build, dropping
    the one file the key owns: the row is advertised and downloadable but never loadable."""
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
    # A genuinely big-endian build is still dropped: the -be file never joins the row.
    assert _gguf_files_for_variant(files, "stories260K") == ["stories260K.gguf"]
    # ... and the plan and the lister agree the qualified file is a normal parent-quant one.
    assert is_main_gguf_variant_path(
        "distilled/Q4_K_M/foo-be.gguf", gguf_variant_key("distilled/Q4_K_M/foo-be.gguf")
    )


def test_a_bpw_build_keeps_its_own_identity_everywhere():
    """Two builds of one base quant at different bits-per-weight are two checkpoints. The loader's
    label always kept the modifier; the variant key dropped it, so the local export lister
    advertised IQ4_XS-3.53bpw while the plan, the auto-download map and the delete predicate all
    said IQ4_XS -- the advertised name 404s and the collapsed one unlinks BOTH builds."""
    from utils.models.model_config import _extract_quant_label as loader_label

    a = "model-IQ4_XS-3.53bpw.gguf"
    b = "model-IQ4_XS-3.97bpw.gguf"
    for path in (a, b):
        assert gguf_variant_key(path) == loader_label(path)
        assert _gguf_variant_key(path) == gguf_variant_key(path)
    assert gguf_variant_key(a) != gguf_variant_key(b)
    # Shards of ONE bpw build still share a key.
    assert gguf_variant_key("model-IQ4_XS-3.53bpw-00001-of-00002.gguf") == gguf_variant_key(a)
    # Two rows, each sized on its own family rather than the sum of both.
    grouped = group_gguf_variant_files([(a, 10), (b, 20)])
    assert grouped == {"IQ4_XS-3.53bpw": (a, 10), "IQ4_XS-3.97bpw": (b, 20)}
    # A plain quant is untouched, so every stored pin still resolves.
    assert gguf_variant_key("model-Q4_K_M.gguf") == "Q4_K_M"
    assert gguf_variant_key("distilled/model-Q6_K.gguf") == "distilled/model-Q6_K"


def test_a_bpw_key_needs_no_scope_label():
    """It reads as a label already, so the display pass must not append the whole filename to it
    the way it does for a path-qualified key."""
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
    # An explicit qualified ask still resolves to the sibling.
    assert _match_variant("distilled/model-Q6_K", both) == "distilled/model-Q6_K"
    # A repo with nothing at the root falls back to the whole set rather than refusing.
    assert _match_variant(None, {"distilled/model-Q6_K": 1}) == "distilled/model-Q6_K"
