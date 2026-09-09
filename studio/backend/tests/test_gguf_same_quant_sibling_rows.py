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
    assert (
        _gguf_files_for_variant(["model-Q4_K_M-mtp.gguf", "model-Q4_K_M-fp16.gguf"], "q4_k_m") == []
    )


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
    test is switched OFF on the three paths where the Hub returned no listing -- and stays on
    everywhere else, since a caller holding the listing (or the loaded model's own identity) has
    to keep reading a qualified root stem as a real quant."""
    from core.inference.openai_auto_download import looks_like_quant

    for tag in ("8b-instruct-q4_0", "8b-instruct-q4_0-fp16", "70b-instruct-q8_0-f16"):
        assert looks_like_quant(tag, allow_root_stem = False) is False
    assert looks_like_quant("gemma-4-31B_q4_0-it") is True
    # Neither the plain nor the path-qualified shape ever depended on the root-stem test.
    for shape in ("Q4_K_M", "distilled/model-Q6_K"):
        assert looks_like_quant(shape) is True
        assert looks_like_quant(shape, allow_root_stem = False) is True


def test_route_side_resolution_refuses_an_ambiguous_bare_alias(tmp_path):
    """``_resolve_quant_gguf`` feeds the KV-cache estimate and the cached-path/reveal endpoints.
    Ranking both tagged builds as the legacy label and taking the first by name priced and
    revealed a checkpoint the caller never asked for, while ``plan_for_variant`` refused it."""
    from routes.models import _resolve_quant_gguf

    both = _materialize(
        tmp_path / "both", [("model-Q4_K_M-mtp.gguf", 1), ("model-Q4_K_M-fp16.gguf", 2)]
    )
    assert _resolve_quant_gguf(str(both), "Q4_K_M", True) == (None, 0)

    # The unambiguous cases keep working: a lone tagged build, and a build's own shards.
    lone = _materialize(tmp_path / "lone", [("gemma-4-31B_q4_0-it.gguf", 17)])
    path, _ = _resolve_quant_gguf(str(lone), "q4_0", True)
    assert path == str(lone / "gemma-4-31B_q4_0-it.gguf")

    sharded = _materialize(
        tmp_path / "sharded",
        [("m-Q4_K_M-00001-of-00002.gguf", 1), ("m-Q4_K_M-00002-of-00002.gguf", 2)],
    )
    path, total = _resolve_quant_gguf(str(sharded), "Q4_K_M", True)
    assert path == str(sharded / "m-Q4_K_M-00001-of-00002.gguf")
    assert total == 1024


def test_snapshot_ordering_reconciles_the_alias(monkeypatch, tmp_path):
    """A lone tagged build is stored under its qualified key, so a legacy bare pin matched no
    snapshot's complete set: every revision sorted as torn and the newest HALF download won,
    handing metadata a shard the loader would never open."""
    from hub.utils import gguf as gguf_module
    from hub.utils import inventory_scan

    torn, whole = tmp_path / "newer", tmp_path / "older"
    torn.mkdir()
    whole.mkdir()
    monkeypatch.setattr(gguf_module, "iter_hf_cache_snapshots", lambda *a, **k: [torn, whole])
    monkeypatch.setattr(
        inventory_scan,
        "complete_snapshot_variants",
        lambda p: set() if str(p) == str(torn) else {"gemma-4-31B_q4_0-it"},
    )
    assert gguf_module.iter_snapshots_preferring_whole("repo", "q4_0") == [whole, torn]


def test_the_local_index_keeps_the_bare_alias_for_a_lone_tagged_build():
    """A persisted ``repo:q4_0`` reaches the auto-switch index through the legacy-alias table.
    ``_qualified_variant_name`` now returns the qualified key itself, so the table recorded no
    bare spelling and the pin could no longer switch to the checkpoint on disk."""
    import types

    from core.inference.local_model_resolver import _legacy_variant_aliases

    def row(quant, filename):
        return types.SimpleNamespace(quant = quant, filename = filename)

    lone = dict(_legacy_variant_aliases([row("gemma-4-31B_q4_0-it", "gemma-4-31B_q4_0-it.gguf")]))
    assert lone["q4_0"] == "gemma-4-31B_q4_0-it"

    # Ambiguous, and already-owned, both stay unaliased.
    two = dict(
        _legacy_variant_aliases(
            [row("m-Q4_K_M-mtp", "m-Q4_K_M-mtp.gguf"), row("m-Q4_K_M-fp16", "m-Q4_K_M-fp16.gguf")]
        )
    )
    assert "q4_k_m" not in two
    plain = dict(
        _legacy_variant_aliases(
            [row("Q4_K_M", "m-Q4_K_M.gguf"), row("m-Q4_K_M-mtp", "m-Q4_K_M-mtp.gguf")]
        )
    )
    assert "q4_k_m" not in plain


def test_a_saved_recipe_still_recognises_the_loaded_build():
    """A recipe saved with the legacy bare variant, against a build now loaded under its
    qualified key: compared literally the recipe refused the checkpoint that was loaded."""
    from hub.utils.gguf import variant_spellings_may_name_one_build

    assert variant_spellings_may_name_one_build("q4_0", "gemma-4-31B_q4_0-it") is True
    assert variant_spellings_may_name_one_build("gemma-4-31B_q4_0-it", "q4_0") is True
    assert variant_spellings_may_name_one_build("Q8_0", "gemma-4-31B_q4_0-it") is False


# --------------------------------------------------------------------------------------
# What a BARE org/repo means, once one quant has two root builds
# --------------------------------------------------------------------------------------


def test_the_default_for_a_bare_repo_id_does_not_depend_on_listing_order():
    """``preferred_quant`` ranks on the quant TEXT, so two root builds of one quant tie and the
    winner falls out of input order. The remote map is in Hub listing order while the local and
    picker listings are size-sorted, so a bare ``org/repo`` could mean the plain build from one
    resolver and the tagged build from the other -- and change weights after a download."""
    from core.inference.openai_auto_download import preferred_quant
    from hub.utils.gguf import collapse_same_quant_root_builds

    keys = ["Hy3-Q4_K_M-mtp", "Q4_K_M", "Hy3-IQ1_M-mtp", "IQ1_M"]
    forward = preferred_quant(collapse_same_quant_root_builds(keys))
    backward = preferred_quant(collapse_same_quant_root_builds(list(reversed(keys))))
    assert forward == backward == "Q4_K_M"


def test_the_collapse_prefers_the_plain_build_and_is_deterministic_without_one():
    from hub.utils.gguf import collapse_same_quant_root_builds

    # A plain row owns the bare quant, so it is what the unqualified id means. The tagged name
    # here sorts BEFORE the bare one, so a winner picked by plain lexicographic order would be
    # the tagged build -- which is the live shape (``Hy3-Q4_K_M-mtp`` beside ``Q4_K_M``) and the
    # reason this cannot be asserted with a pair whose case ordering hides it.
    assert sorted(["Hy3-Q4_K_M-mtp", "Q4_K_M"])[0] == "Hy3-Q4_K_M-mtp"
    assert collapse_same_quant_root_builds(["Hy3-Q4_K_M-mtp", "Q4_K_M"]) == ["Q4_K_M"]
    assert collapse_same_quant_root_builds(["Q4_K_M", "Hy3-Q4_K_M-mtp"]) == ["Q4_K_M"]
    # No plain row: still one deterministic winner, whichever order they arrive in.
    both = ["m-Q4_K_M-mtp", "m-Q4_K_M-fp16"]
    assert collapse_same_quant_root_builds(both) == collapse_same_quant_root_builds(both[::-1])
    # Different quants are never collapsed together, and a path-qualified key passes through.
    assert set(collapse_same_quant_root_builds(["Q4_K_M", "Q8_0"])) == {"Q4_K_M", "Q8_0"}
    assert collapse_same_quant_root_builds(["distilled/m-Q6_K"]) == ["distilled/m-Q6_K"]


def test_the_collapse_only_decides_the_default_not_what_is_advertised():
    """Every build stays individually selectable; this narrows the ranking input alone."""
    rows = group_gguf_variant_files(MTP_FILES)
    assert len(rows) == len(MTP_FILES)


def test_a_resident_build_is_not_confused_with_the_row_the_resolver_chose():
    """``variant_spellings_may_name_one_build`` is deliberately loose and is only safe where a
    false match FAILS CLOSED, as in the delete guard. Where a repo publishes a plain row beside a
    tagged one the two spellings name different checkpoints, so the already-serving and recipe
    checks resolve against the inventory instead of comparing the strings loosely."""
    from hub.utils.gguf import resolve_variant_alias, variant_spellings_may_name_one_build

    inventory = ["Q4_K_M", "model-Q4_K_M-mtp"]
    # Loose: would call the resident tagged build a match for the plain row.
    assert variant_spellings_may_name_one_build("model-Q4_K_M-mtp", "Q4_K_M") is True
    # Inventory-aware: the plain row owns the bare quant, so it resolves to itself.
    assert resolve_variant_alias(inventory, "Q4_K_M") == "Q4_K_M"
    # And with no plain row the bare spelling still reaches the one tagged build.
    assert resolve_variant_alias(["gemma-4-31B_q4_0-it"], "q4_0") == "gemma-4-31B_q4_0-it"


def test_an_active_download_blocks_the_delete_that_now_reaches_its_build():
    """The delete resolves the legacy bare quant onto the qualified key, so the in-flight guard
    has to accept the same alias.

    ``_variant_keys_to_delete`` unlinks ``model-Q4_K_M-mtp``'s files for a request spelled
    ``Q4_K_M``. A user who started that download before the split has the job registered under
    the bare spelling and the picker now offers the qualified row, so the two sides of the guard
    hold different spellings of ONE build. Comparing them literally let the delete through while
    the worker was still writing the blobs.
    """
    from hub.utils.download_registry import DownloadRegistry

    repo = "org/repo"
    for job_variant, delete_variant in (
        ("q4_k_m", "model-q4_k_m-mtp"),
        ("model-q4_k_m-mtp", "q4_k_m"),
        ("q4_k_m", "q4_k_m"),
    ):
        registry = DownloadRegistry()
        key = f"{repo}::{job_variant}"
        claimed, why = registry.claim(
            key, "http", repo_type = "model", repo_id = repo, variant = job_variant
        )
        assert claimed, why
        assert registry.begin_delete(repo, delete_variant) is False
        # And the load-side probe sees the job under either spelling.
        assert registry.has_active_variant(repo, delete_variant) is True
    # A genuinely different quant still downloads and deletes concurrently.
    registry = DownloadRegistry()
    registry.claim(f"{repo}::q4_k_m", "http", repo_type = "model", repo_id = repo, variant = "q4_k_m")
    assert registry.begin_delete(repo, "q8_0") is True
    assert registry.has_active_variant(repo, "q8_0") is False


def test_a_whole_snapshot_job_still_blocks_every_variant_delete():
    """The alias compare must not lose the None cases: a variantless job writes the whole
    snapshot, so it conflicts with any delete, and a whole-repo delete conflicts with any job."""
    from hub.utils.download_registry import DownloadRegistry

    repo = "org/repo"
    registry = DownloadRegistry()
    registry.claim(repo, "http", repo_type = "model", repo_id = repo)
    assert registry.begin_delete(repo, "q4_k_m") is False
    assert registry.has_active_variant(repo, "q4_k_m") is False
    assert registry.has_active_variant(repo, None) is True

    registry = DownloadRegistry()
    registry.claim(f"{repo}::q4_k_m", "http", repo_type = "model", repo_id = repo, variant = "q4_k_m")
    assert registry.begin_delete(repo, None) is False


def test_every_resolver_agrees_what_a_bare_repo_id_means():
    """The collapse is the ranking input for all three answers to "which build is the default",
    so the picker service has to apply it too. It is size-sorted where the remote map is in Hub
    listing order, so without it the two disagree exactly when a repo publishes a plain build
    beside a tagged one -- and the id would change weights once the repo was downloaded."""
    from core.inference.openai_auto_download import _match_variant
    from hub.services.models.gguf_variants import _default_variant_candidates
    from hub.utils.gguf import pick_best_gguf

    class _Row:
        def __init__(self, filename):
            self.filename = filename
            self.quant = gguf_variant_key(filename)
            self.size_bytes = 100

    files = ["Hy3-Q4_K_M.gguf", "Hy3-Q4_K_M-mtp.gguf"]
    for order in (files, files[::-1]):
        rows = [_Row(f) for f in order]
        assert pick_best_gguf(_default_variant_candidates(rows)) == "Hy3-Q4_K_M.gguf"
        assert _match_variant(None, {row.quant: row.size_bytes for row in rows}) == "Q4_K_M"


def test_the_collapse_reads_a_one_shot_iterable_once():
    """It is annotated ``Iterable[str]`` and reads its argument twice, so a generator emptied
    the result silently -- a default of nothing rather than the wrong build, but only because
    every caller happens to hand it a list today."""
    from hub.utils.gguf import collapse_same_quant_root_builds

    keys = ["Q4_K_M", "model-Q4_K_M-mtp"]
    assert collapse_same_quant_root_builds(key for key in keys) == ["Q4_K_M"]
    assert collapse_same_quant_root_builds(iter(keys)) == collapse_same_quant_root_builds(keys)


def test_a_windows_spelled_request_matches_the_key_it_names():
    """A key is always minted with forward slashes, but a request can arrive carrying the host's
    separator. ``collapse_same_quant_root_builds`` and ``_main_variant_rank`` already fold it;
    the shared resolver and the delete guard have to agree, or one Windows install resolves a
    directory-qualified pin and another refuses it."""
    from hub.utils.gguf import resolve_variant_alias, variant_spellings_may_name_one_build

    keys = ["weights/model-Q4_K_M", "Q6_K"]
    assert resolve_variant_alias(keys, "weights/model-Q4_K_M") == "weights/model-Q4_K_M"
    assert resolve_variant_alias(keys, "weights\\model-Q4_K_M") == "weights/model-Q4_K_M"
    # And a listing that itself arrived with backslashes still answers a posix-spelled pin.
    assert resolve_variant_alias(["weights\\model-Q4_K_M"], "weights/model-Q4_K_M") == (
        "weights\\model-Q4_K_M"
    )
    assert variant_spellings_may_name_one_build("weights\\model-Q6_K", "weights/model-Q6_K")


def test_the_requirement_lookup_accepts_what_the_plan_lookup_accepts():
    """Requirements ARE the plans the worker fetches, so the two lookups have to agree.

    A literal dict get missed the legacy bare spelling of a lone tagged build, so the download
    reported no expected size and no expected files and the poller fell back to a byte tally
    over the shared blobs directory -- which cannot tell this quant's bytes from a sibling's.
    """
    from hub.services.models.gguf_variants import _build_gguf_variant_requirements

    class _Sibling:
        def __init__(self, rfilename):
            self.rfilename = rfilename
            self.size = 1000
            self.lfs = None

    requirements = _build_gguf_variant_requirements(
        [_Sibling(f) for f in ("model-Q4_K_M-mtp.gguf", "model-Q6_K.gguf")]
    )
    assert sorted(requirements) == ["model-q4_k_m-mtp", "q6_k"]
    for pin in ("Q4_K_M", "model-Q4_K_M-mtp"):
        assert plan_for_variant(requirements, pin) is not None, pin
    # Two builds at one quant: the bare spelling names neither, here as everywhere else.
    ambiguous = _build_gguf_variant_requirements(
        [_Sibling(f) for f in ("m-Q4_K_M-mtp.gguf", "m-Q4_K_M-fp16.gguf")]
    )
    assert plan_for_variant(ambiguous, "Q4_K_M") is None


def test_a_bare_pin_still_names_the_sole_root_build():
    """A path-qualified key never owned the bare quant, so it must not contest one.

    Re-keying a tagged ROOT build put it in the alias list beside
    ``distilled/model-Q4_K_M``, and the pair made the pin ambiguous. On main that pin matched
    the root build EXACTLY, so it resolved; here it stopped resolving at all.
    """
    from hub.utils.gguf import resolve_variant_alias

    keys = ["model-Q4_K_M-mtp", "distilled/model-Q4_K_M"]
    assert resolve_variant_alias(keys, "Q4_K_M") == "model-Q4_K_M-mtp"
    # Two ROOT builds still tie, and still refuse.
    assert resolve_variant_alias(["m-Q4_K_M-mtp", "m-Q4_K_M-fp16"], "Q4_K_M") is None
    # Nothing at the root: the sole path-qualified key answers, as it did before.
    assert resolve_variant_alias(["distilled/model-Q4_K_M"], "Q4_K_M") == ("distilled/model-Q4_K_M")
    # An exact key always wins over the alias tier.
    assert resolve_variant_alias(["Q4_K_M", "distilled/model-Q4_K_M"], "Q4_K_M") == "Q4_K_M"


def test_a_delete_reservation_holds_against_the_other_spelling():
    """``begin_delete`` records the request's spelling and ``claim`` compared it literally, so a
    worker spawned during the delete window wrote blobs the delete was already unlinking. This is
    the reciprocal of the active-job guard: both ends of the window have to see one build."""
    from hub.utils.download_registry import DownloadRegistry

    repo = "org/repo"
    for delete_variant, claim_variant in (
        ("q4_k_m", "model-q4_k_m-mtp"),
        ("model-q4_k_m-mtp", "q4_k_m"),
    ):
        registry = DownloadRegistry()
        assert registry.begin_delete(repo, delete_variant) is True
        ok, why = registry.claim(
            f"{repo}::{claim_variant}",
            "http",
            repo_type = "model",
            repo_id = repo,
            variant = claim_variant,
        )
        assert ok is False and why == "deleting", (delete_variant, claim_variant, why)
    # A genuinely different quant still downloads while another is deleted.
    registry = DownloadRegistry()
    assert registry.begin_delete(repo, "q4_k_m") is True
    ok, _why = registry.claim(
        f"{repo}::q8_0", "http", repo_type = "model", repo_id = repo, variant = "q8_0"
    )
    assert ok is True


def test_two_spellings_of_one_build_do_not_run_as_sibling_quants():
    """Sibling quants download concurrently because each worker purges only its own main blobs.
    Two spellings of ONE build re-resolve to the SAME blobs, so the second worker rewrites what
    the first is writing; that is a conflict, not a sibling."""
    from hub.utils.download_registry import DownloadRegistry

    repo = "org/repo"
    registry = DownloadRegistry()
    ok, _ = registry.claim(
        f"{repo}::q4_k_m", "http", repo_type = "model", repo_id = repo, variant = "q4_k_m"
    )
    assert ok
    ok, why = registry.claim(
        f"{repo}::model-q4_k_m-mtp",
        "http",
        repo_type = "model",
        repo_id = repo,
        variant = "model-q4_k_m-mtp",
    )
    assert ok is False and why == "running", why
    # A real sibling quant is still admitted concurrently.
    ok, _ = registry.claim(f"{repo}::q8_0", "http", repo_type = "model", repo_id = repo, variant = "q8_0")
    assert ok is True


def test_the_requirement_cache_answers_the_spelling_it_was_asked():
    """The fetch caches each plan under its own key, so a legacy bare spelling missed on every
    call and re-ran model_info. The download-progress endpoint asks once per poll."""
    from hub.services.models import gguf_variants as service

    class _Sibling:
        def __init__(self, rfilename):
            self.rfilename = rfilename
            self.size = 1000
            self.lfs = None

    siblings = [_Sibling("model-Q4_K_M-mtp.gguf"), _Sibling("model-Q6_K.gguf")]
    calls = {"n": 0}
    real_fetch = service._fetch_gguf_variant_requirements

    def _counting_fetch(
        repo_id,
        hf_token = None,
        *,
        _siblings = siblings,
        **kwargs,
    ):
        calls["n"] += 1
        return real_fetch(repo_id, hf_token, siblings = _siblings)

    service._VARIANT_REQUIREMENT_CACHE.clear()
    service._fetch_gguf_variant_requirements = _counting_fetch
    try:
        first = service.gguf_variant_requirements("org/repo", "Q4_K_M")
        second = service.gguf_variant_requirements("org/repo", "Q4_K_M")
    finally:
        service._fetch_gguf_variant_requirements = real_fetch
    assert first is not None and second is not None
    assert calls["n"] == 1, f"the bare spelling re-fetched {calls['n']} times"


def test_two_bit_widths_are_two_precisions_not_two_builds():
    """``extract_quant_token`` drops the bpw modifier, so the collapse grouped
    ``IQ4_XS-3.53bpw`` with ``IQ4_XS-4.05bpw`` and kept the lexicographically first. The listers
    sort larger first, so a bare repo id that meant 4.05bpw silently dropped to 3.53bpw."""
    from core.inference.openai_auto_download import preferred_quant
    from hub.utils.gguf import collapse_same_quant_root_builds

    widths = ["IQ4_XS-4.05bpw", "IQ4_XS-3.53bpw"]
    assert collapse_same_quant_root_builds(widths) == widths
    assert preferred_quant(collapse_same_quant_root_builds(widths)) == "IQ4_XS-4.05bpw"
    # A tagged build at the SAME bit width is still a second build of that precision.
    same_width = ["IQ4_XS-3.53bpw", "m-IQ4_XS-3.53bpw-mtp"]
    assert collapse_same_quant_root_builds(same_width) == ["IQ4_XS-3.53bpw"]
    # And a plain quant with no modifier is unaffected.
    assert collapse_same_quant_root_builds(["Q4_K_M", "m-Q4_K_M-mtp"]) == ["Q4_K_M"]


def test_a_qualified_delete_purges_the_bare_state_too():
    """State is written under the spelling the DOWNLOAD used. A build fetched through the legacy
    bare quant and deleted through its advertised row took the exact-key return, so the bare
    manifest and marker were never purged and an offline refresh rebuilt a partial row."""
    from hub.services.models.deletion import _state_spellings_for_delete

    class _Repo:
        def __init__(self, names):
            self._names = names

    def _matches(target_repo, _pred):
        return [(None, None, name) for name in target_repo._names]

    import hub.services.models.deletion as deletion

    real = deletion._repo_file_matches
    deletion._repo_file_matches = _matches
    try:
        lone = _Repo(["model-Q4_K_M-mtp.gguf", "model-Q6_K.gguf"])
        # Qualified request: the bare spelling its download may have used is purged as well.
        assert _state_spellings_for_delete(lone, "model-q4_k_m-mtp") == {
            "model-q4_k_m-mtp",
            "q4_k_m",
        }
        # Bare request: still resolves forward onto the qualified key.
        assert _state_spellings_for_delete(lone, "q4_k_m") == {"q4_k_m", "model-q4_k_m-mtp"}
        # A plain sibling OWNS the bare spelling, so it is another build's state; leave it.
        shared = _Repo(["model-Q4_K_M.gguf", "model-Q4_K_M-mtp.gguf"])
        assert _state_spellings_for_delete(shared, "model-q4_k_m-mtp") == {"model-q4_k_m-mtp"}
    finally:
        deletion._repo_file_matches = real


def test_the_media_index_answers_the_legacy_spelling_too(tmp_path):
    """An image or video request persisted as ``repo:Q4_K_M`` has to reach a lone tagged build,
    the way the chat index and both loaders do. The media index registered only the exact key,
    so the same reference the GGUF paths accept was rejected as unavailable here."""
    from core.inference import media_model_index as mmi

    for name in ("model-Q4_K_M-mtp.gguf", "model-Q6_K.gguf"):
        (tmp_path / name).write_bytes(b"GGUF" + b"\0" * 64)

    # The loader probe reads real headers; this test is about which NAMES get indexed.
    real_can_open = mmi._loader_can_open
    mmi._loader_can_open = lambda *a, **k: True
    index: dict = {}
    try:
        assert mmi._add_gguf_picks(index, None, ("repo",), tmp_path, tmp_path) is True
    finally:
        mmi._loader_can_open = real_can_open

    assert index["repo:model-q4_k_m-mtp"].gguf_filename == "model-Q4_K_M-mtp.gguf"
    # The legacy bare spelling, which nothing else in the repo owns.
    assert index["repo:q4_k_m"].gguf_filename == "model-Q4_K_M-mtp.gguf"
    assert index["repo:q6_k"].gguf_filename == "model-Q6_K.gguf"


def test_the_media_index_leaves_a_contested_bare_spelling_alone(tmp_path):
    """A plain row owning the bare quant means the alias is that row's, not the tagged one's."""
    from core.inference import media_model_index as mmi

    for name in ("model-Q4_K_M.gguf", "model-Q4_K_M-mtp.gguf"):
        (tmp_path / name).write_bytes(b"GGUF" + b"\0" * 64)

    real_can_open = mmi._loader_can_open
    mmi._loader_can_open = lambda *a, **k: True
    index: dict = {}
    try:
        mmi._add_gguf_picks(index, None, ("repo",), tmp_path, tmp_path)
    finally:
        mmi._loader_can_open = real_can_open

    assert index["repo:q4_k_m"].gguf_filename == "model-Q4_K_M.gguf"
    assert index["repo:model-q4_k_m-mtp"].gguf_filename == "model-Q4_K_M-mtp.gguf"
    # And the bare id itself means the plain build, as it does in every other resolver.
    assert index["repo"].gguf_filename == "model-Q4_K_M.gguf"


def test_a_quant_named_directory_still_contests_the_bare_spelling():
    """Root precedence is the lister's rule, not the presence of a slash.

    A quant-named parent adds no identity, so ``Q4_K_M/model-Q4_K_M-fp16.gguf`` is a second
    build AT the root, not a distilled checkpoint. Testing for a slash let the root build win a
    contest it should have lost, and an existing pin got one of two checkpoints rather than a
    refusal.
    """
    from hub.utils.gguf import resolve_variant_alias

    contested = [
        gguf_variant_key("model-Q4_K_M-mtp.gguf"),
        gguf_variant_key("Q4_K_M/model-Q4_K_M-fp16.gguf"),
    ]
    assert contested == ["model-Q4_K_M-mtp", "Q4_K_M/model-Q4_K_M-fp16"]
    assert resolve_variant_alias(contested, "Q4_K_M") is None
    # A directory that DOES name another checkpoint stays subordinate, so the root build wins.
    assert resolve_variant_alias(["model-Q4_K_M-mtp", "distilled/model-Q4_K_M"], "Q4_K_M") == (
        "model-Q4_K_M-mtp"
    )


def test_the_auto_download_lookup_shares_the_root_precedence_rule():
    """Deciding the alias independently meant this rejected a pin the root build owns, and the
    download it gates never ran even though the plan lookup would have resolved it."""
    from core.inference.openai_auto_download import _bare_quant_alias
    from hub.utils.gguf import resolve_variant_alias

    for keys in (
        ["model-Q4_K_M-mtp", "distilled/model-Q4_K_M"],
        ["model-Q4_K_M-mtp", "model-Q4_K_M-fp16"],
        ["Q4_K_M", "model-Q4_K_M-mtp"],
        ["gemma-4-31B_q4_0-it"],
    ):
        lowered = {key.lower(): key for key in keys}
        expected = resolve_variant_alias(lowered.keys(), "Q4_K_M" if "Q4" in keys[0] else "q4_0")
        wanted = "Q4_K_M" if "Q4" in keys[0] else "q4_0"
        assert _bare_quant_alias(wanted, lowered) == (
            lowered.get(expected) if expected is not None else None
        ), keys


def test_a_far_bit_width_annotation_is_not_the_quant_s_own():
    """``quant_token_with_bpw`` reads only an ADJACENT modifier, and the group identity has to
    agree with it. Searching the whole key read ``08.577bpw`` as Q8_0's bit width, so the build
    never grouped with the plain Q8_0 it is a second copy of and the bare id stayed
    order-dependent."""
    from hub.utils.gguf import _quant_group_identity, collapse_same_quant_root_builds
    from hub.utils.gguf import quant_token_with_bpw

    far = "flux1-dev-Q8_0-fp32-08.577bpw"
    assert quant_token_with_bpw(far) == "Q8_0"
    assert _quant_group_identity(far) == "q8_0"
    assert collapse_same_quant_root_builds(["Q8_0", far]) == ["Q8_0"]
    # An adjacent modifier is still the quant's own, mid-name or not.
    assert _quant_group_identity("m-IQ4_XS-3.53bpw-mtp") == "iq4_xs-3.53bpw"
    assert collapse_same_quant_root_builds(["IQ4_XS-3.53bpw", "m-IQ4_XS-3.53bpw-mtp"]) == [
        "IQ4_XS-3.53bpw"
    ]


def test_an_unopenable_plain_row_keeps_its_own_bare_spelling(tmp_path):
    """Alias ownership is decided over every published row. Scoring only the openable ones let a
    plain build the loader refuses vanish from the contest, so ``repo:q4_k_m`` was registered for
    its tagged sibling and an explicit request for the plain row ran different weights."""
    from core.inference import media_model_index as mmi

    for name in ("model-Q4_K_M.gguf", "model-Q4_K_M-mtp.gguf"):
        (tmp_path / name).write_bytes(b"GGUF" + b"\0" * 64)

    real_can_open = mmi._loader_can_open
    # the plain build is present but unopenable; only the tagged sibling loads
    mmi._loader_can_open = lambda _path, filename: filename != "model-Q4_K_M.gguf"
    index: dict = {}
    try:
        mmi._add_gguf_picks(index, None, ("repo",), tmp_path, tmp_path)
    finally:
        mmi._loader_can_open = real_can_open

    assert index["repo:model-q4_k_m-mtp"].gguf_filename == "model-Q4_K_M-mtp.gguf"
    # The bare spelling belongs to the plain row, which is not loadable, so nothing answers it.
    assert "repo:q4_k_m" not in index


def test_a_partial_siblings_manifest_survives_a_qualified_delete(tmp_path):
    """An interrupted plain-build download has a bare-spelled manifest and an incomplete blob but
    no snapshot file, so it is invisible to the snapshot scan. Claiming the bare spelling on that
    evidence purged the partial download's resume state and orphaned its bytes."""
    import hub.services.models.deletion as deletion

    class _Repo:
        def __init__(self, names):
            self._names = names

    def _matches(target_repo, _pred):
        return [(None, None, name) for name in target_repo._names]

    class _Expected:
        def __init__(self, path):
            self.path = path

    class _Manifest:
        def __init__(self, paths):
            self.expected_files = tuple(_Expected(p) for p in paths)

    real_matches = deletion._repo_file_matches
    real_read = deletion.download_manifest.read_manifest
    deletion._repo_file_matches = _matches
    lone = _Repo(["model-Q4_K_M-mtp.gguf"])
    try:
        # A bare manifest naming the PLAIN build: a different checkpoint, still downloading.
        deletion.download_manifest.read_manifest = lambda *a, **k: _Manifest(["model-Q4_K_M.gguf"])
        assert deletion._state_spellings_for_delete(
            lone, "model-q4_k_m-mtp", "org/repo", tmp_path
        ) == {"model-q4_k_m-mtp"}
        # A bare manifest naming THIS build: it is this download's own state, so it goes.
        deletion.download_manifest.read_manifest = lambda *a, **k: _Manifest(
            ["model-Q4_K_M-mtp.gguf"]
        )
        assert deletion._state_spellings_for_delete(
            lone, "model-q4_k_m-mtp", "org/repo", tmp_path
        ) == {"model-q4_k_m-mtp", "q4_k_m"}
        # No manifest at all: nothing to protect, so the reverse alias is still purged.
        deletion.download_manifest.read_manifest = lambda *a, **k: None
        assert deletion._state_spellings_for_delete(
            lone, "model-q4_k_m-mtp", "org/repo", tmp_path
        ) == {"model-q4_k_m-mtp", "q4_k_m"}

        # An unreadable manifest fails CLOSED: a stale marker is cheaper than a lost resume.
        def _raise(*a, **k):
            raise OSError("unreadable")

        deletion.download_manifest.read_manifest = _raise
        assert deletion._state_spellings_for_delete(
            lone, "model-q4_k_m-mtp", "org/repo", tmp_path
        ) == {"model-q4_k_m-mtp"}
    finally:
        deletion._repo_file_matches = real_matches
        deletion.download_manifest.read_manifest = real_read
