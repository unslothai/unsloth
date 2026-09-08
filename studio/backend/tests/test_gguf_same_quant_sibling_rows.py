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
from hub.utils.gguf_plan import build_gguf_variant_plans, plan_for_variant, plan_from_expected_files
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
    variants = [GgufVariantInfo(filename = path, quant = gguf_variant_key(path), size_bytes = 17)]
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


# --------------------------------------------------------------------------------------
# The separator the tag uses, and the paths that resolve the legacy bare spelling
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        # A DOT separates the tag just as often as a dash, and it is not an extension.
        ("model-Q4_K_M.fp16.gguf", "model-Q4_K_M.fp16"),
        ("Qwen3.8-27B-GSQ-RCO-IQ3_S.mtp.gguf", "Qwen3.8-27B-GSQ-RCO-IQ3_S.mtp"),
        # Only the extension itself is ignored, however many times it repeats.
        ("llama-2-7b-chat.Q4_K_M.gguf", "Q4_K_M"),
        ("Llama-3.3-70B-Instruct.Q6_K.gguf-00001-of-00006.gguf", "Q6_K"),
    ],
)
def test_a_dotted_build_tag_is_part_of_the_identity(path, expected):
    """Stripping every dotted component as though it were an extension collapsed
    ``model-Q4_K_M.fp16.gguf`` back onto bare ``Q4_K_M`` -- the very collision this file exists
    to prevent, reached through the other separator."""
    assert gguf_variant_key(path) == expected
    assert _gguf_variant_key(path) == expected


def test_the_dotted_pair_is_two_rows():
    rows = group_gguf_variant_files([("model-Q4_K_M.fp16.gguf", 2), ("model-Q4_K_M.gguf", 1)])
    assert rows == {
        "Q4_K_M": ("model-Q4_K_M.gguf", 1),
        "model-Q4_K_M.fp16": ("model-Q4_K_M.fp16.gguf", 2),
    }


def test_the_load_guard_sees_the_alias_a_root_stem_delete_accepts():
    """Deletion resolves a bare ``q4_0`` onto the lone tagged build's qualified key. The guard
    compared the two strings literally, so a bare request unlinked the build that was LOADED."""
    from hub.services.models.deletion import _loaded_repo_variant_blocks_delete

    loaded = "gemma-4-31B_q4_0-it"
    assert _loaded_repo_variant_blocks_delete("repo", "repo", "q4_0", loaded) is True
    assert _loaded_repo_variant_blocks_delete("repo", "repo", loaded, loaded) is True
    # A different quant is still free to go.
    assert _loaded_repo_variant_blocks_delete("repo", "repo", "Q8_0", loaded) is False


def test_a_cached_lone_tagged_build_resolves_under_its_legacy_bare_pin(tmp_path, monkeypatch):
    """The download path accepts ``q4_0`` for the qualified row, so cached-path lookup has to as
    well; returning None there reported a downloaded model as not_downloaded and skipped every
    header-derived fact (VRAM estimate, embedding/diffusion detection)."""
    from hub.utils import gguf as gguf_module

    snapshot = _materialize(tmp_path / "snap", [("gemma-4-31B_q4_0-it.gguf", 17)])
    monkeypatch.setattr(gguf_module, "iter_snapshots_preferring_whole", lambda *a, **k: [snapshot])
    resolved = gguf_module.resolve_local_gguf_path("google/gemma-4-31B-it-qat-q4_0-gguf", "q4_0")
    assert resolved == str(snapshot / "gemma-4-31B_q4_0-it.gguf")


def test_an_ambiguous_bare_spelling_resolves_to_nothing():
    """Two tagged builds at one quant: the bare token names neither, so every caller fails closed
    rather than picking the one that sorts first."""
    from hub.utils.gguf import resolve_variant_alias

    keys = ["model-IQ3_S-mtp", "model-IQ3_S-fp16"]
    assert resolve_variant_alias(keys, "iq3_s") is None
    assert resolve_variant_alias(keys, "model-IQ3_S-mtp") == "model-IQ3_S-mtp"


def test_an_interrupted_bare_alias_download_still_resumes():
    """The worker writes the manifest under the spelling the request used. Rebuilding the plan
    from that manifest matched no main file once the key was qualified, so resume aborted on top
    of blobs it had already fetched."""
    from hub.utils.download_manifest import ExpectedFile

    expected = [ExpectedFile(path = "gemma-4-31B_q4_0-it.gguf", size = 17, sha256 = "h1")]
    plan = plan_from_expected_files("q4_0", expected)
    assert plan.main_filenames == frozenset({"gemma-4-31B_q4_0-it.gguf"})
    assert plan.main_size_bytes == 17


def test_a_companion_cannot_shadow_the_build_a_bare_pin_names():
    """An mmproj is named for its OWN precision and keys like any other file
    (``Qwen3.8-27B-mmproj-hybrid-Q8_0-F16.gguf`` -> ``Q8_0``, live on
    ``NikiKrutan/Kiwen1.1-27B-MTP-GGUF``). Resolving the bare pin over every GGUF in the manifest
    let that exact match win and hand the caller a plan with no main file at all."""
    from hub.utils.download_manifest import ExpectedFile

    expected = [
        ExpectedFile(path = "model-Q8_0-mtp.gguf", size = 10, sha256 = "main"),
        ExpectedFile(path = "mmproj-Q8_0.gguf", size = 2, sha256 = "mmproj"),
    ]
    plan = plan_from_expected_files("q8_0", expected)
    assert plan.main_filenames == frozenset({"model-Q8_0-mtp.gguf"})


def test_the_main_candidate_predicate_stays_in_lockstep():
    """``is_main_gguf_variant_path`` is the candidate test plus the key comparison; if the two
    drift, the resolver sees a different file set than the filter that follows it."""
    from hub.utils.gguf_plan import is_main_gguf_candidate, is_main_gguf_variant_path
    for path in ("model-Q8_0-mtp.gguf", "mmproj-Q8_0.gguf", "imatrix-model.gguf", "notes.txt"):
        key = gguf_variant_key(path)
        assert is_main_gguf_variant_path(path, key) == (
            is_main_gguf_candidate(path) and gguf_variant_key(path).lower() == key.lower()
        )


def test_the_load_guard_compares_the_two_spellings_symmetrically():
    """A build can be LOADED through the legacy bare pin and deleted through its advertised
    qualified row, or the reverse. A one-directional check covered only the second, so the first
    unlinked a model that was resident."""
    from hub.services.models.deletion import _loaded_repo_variant_blocks_delete

    bare, qualified = "q4_0", "gemma-4-31B_q4_0-it"
    assert _loaded_repo_variant_blocks_delete("r", "r", qualified, bare) is True
    assert _loaded_repo_variant_blocks_delete("r", "r", bare, qualified) is True
    assert _loaded_repo_variant_blocks_delete("r", "r", "Q8_0", qualified) is False


def test_the_loaders_refuse_a_bare_pin_that_names_two_builds(tmp_path):
    """``plan_for_variant`` refuses an ambiguous bare quant, but the loader fallbacks matched on
    the LABEL and took the first file by name -- so a stale pin ran a different checkpoint once
    both qualified rows had been downloaded."""
    snapshot = _materialize(
        tmp_path / "two", [("model-Q4_K_M-mtp.gguf", 1), ("model-Q4_K_M-fp16.gguf", 2)]
    )
    assert _find_local_gguf_by_variant(str(snapshot), "Q4_K_M") is None
    assert _gguf_files_for_variant(["model-Q4_K_M-mtp.gguf", "model-Q4_K_M-fp16.gguf"], "q4_k_m") == []


def test_a_lone_tagged_build_still_loads_under_its_legacy_bare_pin(tmp_path):
    """The refusal above must not cost the single-build repos their bare spelling."""
    snapshot = _materialize(tmp_path / "one", [("gemma-4-31B_q4_0-it.gguf", 17)])
    assert _find_local_gguf_by_variant(str(snapshot), "q4_0") == str(
        snapshot / "gemma-4-31B_q4_0-it.gguf"
    )
    assert _gguf_files_for_variant(["gemma-4-31B_q4_0-it.gguf"], "q4_0") == [
        "gemma-4-31B_q4_0-it.gguf"
    ]


def test_a_foreign_provider_tag_is_not_one_of_our_rows():
    """``looks_like_quant`` gates a Hub 404 between a refusal and falling through to another
    server. An Ollama tag mints the same shape as a lone tagged build's key, so the root-stem
    test is only offered where the caller actually holds the repo's listing."""
    from core.inference.openai_auto_download import looks_like_quant

    for tag in ("8b-instruct-q4_0", "8b-instruct-q4_0-fp16", "70b-instruct-q8_0-f16"):
        assert looks_like_quant(tag) is False
    assert looks_like_quant("gemma-4-31B_q4_0-it", allow_root_stem = True) is True
    # The shapes that never needed the listing are unaffected.
    assert looks_like_quant("Q4_K_M") is True
    assert looks_like_quant("distilled/model-Q6_K") is True
