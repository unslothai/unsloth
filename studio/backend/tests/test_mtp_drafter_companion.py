# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Separate-file drafter contracts: MTP (Gemma 4), DSpark and DFlash.

Pins: the drafter-path predicate and its two layering mirrors, Gemma
effective-size extraction, companion classification in variant plans
(including resume from pre-fix manifests where the drafter leaked into a
quant's main files), and local drafter detection / self-pairing rejection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from hub.utils.download_manifest import ExpectedFile
from hub.utils.gguf import (
    is_mtp_drafter_path,
    list_local_gguf_variants as list_hub_local_gguf_variants,
)
from hub.utils.gguf_plan import (
    build_gguf_variant_plans,
    plan_from_expected_files,
    preferred_mtp_sibling,
)
from utils.models.model_config import (
    ModelConfig,
    _is_mtp_drafter,
    _local_gguf_companion_search_root,
    detect_gguf_model,
    detect_dflash_file,
    detect_dspark_file,
    detect_mtp_file,
    extract_model_size_b,
    list_local_gguf_variants,
)
from utils.native_path_leases import native_gguf_companion_parent_allowed


# ── Predicate + layering mirrors ─────────────────────────────────────

DRAFTER_CASES = [
    ("mtp-gemma-4-12b-it.gguf", True),
    ("MTP/gemma-4-12b-it-Q8_0-MTP.gguf", True),
    # New-scheme MTP/ copies carry the mtp- basename prefix too.
    ("MTP/mtp-gemma-4-E4B-it-BF16.gguf", True),
    ("foo/MTP/bar.gguf", True),
    ("gemma-4-12b-it-Q8_0.gguf", False),
    # Baked-in Qwen MTP repos: the head is inside the main GGUF, the file
    # IS the model -- must never be classified as a companion.
    ("Qwen3.6-27B-MTP-Q4_K_M.gguf", False),
    ("prompt-mtp-test.gguf", False),
    ("smtp/model.gguf", False),
    ("mtp-readme.txt", False),
    # DSpark drafters (DeepSeek V4 Flash). Their BF16/Q8_0 tokens make them the
    # two smallest, most pickable entries in a repo whose real quants are 87 GB+.
    ("dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf", True),
    ("dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf", True),
    ("DSPARK/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf", True),
    ("dspark/whatever.gguf", True),
    ("dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf", True),
    # Local scans can hand the predicate a Windows path.
    ("dspark\\dspark-DeepSeek-V4-Flash-0731-BF16.gguf", True),
    # Same drafter under its general.architecture name; the prefix carries it,
    # e.g. ggml-org/Qwen3.6-27B-GGUF ships one at the repo root.
    ("dflash/dflash-model-Q8_0.gguf", True),
    ("dflash-model.gguf", True),
    ("dflash-Qwen3.6-27B-BF16.gguf", True),
    # ...but dflash is a family name, so the DIRECTORY is not a drafter marker:
    # no published repo uses a dflash/ companion folder, while users do name a
    # local folder after the family they downloaded.
    ("dflash/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf", False),
    ("foo/dflash/bar.gguf", False),
    # Real Hub filenames where dflash/dspark is the family name: each IS the model.
    ("Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf", False),
    ("qwen36-35b-a3b-dflash-Q8_0.gguf", False),
    ("laguna-xs21-dflash-q4.gguf", False),
    ("xdspark/model.gguf", False),
    ("dspark/README.md", False),
]


@pytest.mark.parametrize("path,expected", DRAFTER_CASES)
def test_drafter_predicate_and_mirrors_agree(path, expected):
    from core.inference.llama_cpp import _is_companion_gguf_path

    assert is_mtp_drafter_path(path) is expected
    assert _is_mtp_drafter(path) is expected
    # The core mirror bundles mmproj; none of these inputs are mmproj, so
    # it must agree with the canonical predicate.
    assert _is_companion_gguf_path(path) is expected


# ── Gemma effective-size extraction ──────────────────────────────────


@pytest.mark.parametrize(
    "model_id,size_b",
    [
        ("unsloth/gemma-4-E2B-it-GGUF", 2.0),
        ("unsloth/gemma-4-E4B-it", 4.0),
        ("unsloth/gemma-3n-E4B-it", 4.0),
        # MoE active params beat effective and total notation.
        ("unsloth/Qwen3.5-35B-A3B", 3.0),
        ("unsloth/gemma-4-12b-it-GGUF", 12.0),
        ("unsloth/Qwen3.5-9B-MTP-GGUF", 9.0),
        ("no-size-here", None),
    ],
)
def test_extract_model_size_b(model_id, size_b):
    assert extract_model_size_b(model_id) == size_b


# ── Variant plan companion classification ────────────────────────────


def _sib(name: str, size: int, sha: str):
    return SimpleNamespace(rfilename = name, size = size, lfs = {"sha256": sha})


GEMMA_SIBLINGS = [
    _sib("gemma-4-12b-it-Q4_K_M.gguf", 4_000, "main-q4"),
    _sib("gemma-4-12b-it-Q8_0.gguf", 8_000, "main-q8"),
    _sib("mtp-gemma-4-12b-it.gguf", 100, "drafter"),
    _sib("MTP/gemma-4-12b-it-Q8_0-MTP.gguf", 100, "mtp-sub-q8"),
    _sib("MTP/gemma-4-12b-it-BF16-MTP.gguf", 200, "mtp-sub-bf16"),
    _sib("mmproj-F16.gguf", 500, "mmproj"),
]


def test_variant_plans_carry_drafter_as_companion():
    plans = build_gguf_variant_plans(GEMMA_SIBLINGS)

    # No phantom quants from the drafter's Q8_0 label or the MTP/ copies.
    assert set(plans) == {"q4_k_m", "q8_0"}
    for plan in plans.values():
        assert "mtp-gemma-4-12b-it.gguf" in plan.target_filenames
        assert not any("MTP/" in name for name in plan.target_filenames)
        assert "drafter" in plan.companion_hashes
        assert "drafter" not in plan.main_hashes
        assert plan.mmproj_filenames == frozenset({"mmproj-F16.gguf"})

    q4 = plans["q4_k_m"]
    assert q4.main_filenames == frozenset({"gemma-4-12b-it-Q4_K_M.gguf"})
    assert q4.main_size_bytes == 4_000
    # Download size = main + mmproj + drafter.
    assert q4.download_size_bytes == 4_600


def test_baked_in_repo_plans_unchanged():
    plans = build_gguf_variant_plans([_sib("Qwen3.6-27B-MTP-Q4_K_M.gguf", 4_000, "q4")])
    assert plans["q4_k_m"].target_filenames == ("Qwen3.6-27B-MTP-Q4_K_M.gguf",)


def test_old_manifest_resume_reclassifies_drafter():
    # Pre-fix manifests could leak the drafter into a quant's expected
    # files; resume must classify it as a companion, not a main shard.
    old = [
        ExpectedFile(path = "gemma-4-12b-it-Q8_0.gguf", size = 8_000, sha256 = "main-q8"),
        ExpectedFile(path = "mtp-gemma-4-12b-it.gguf", size = 100, sha256 = "drafter"),
    ]
    plan = plan_from_expected_files("Q8_0", old)
    assert plan.main_hashes == frozenset({"main-q8"})
    assert plan.companion_hashes == frozenset({"drafter"})
    assert plan.mmproj_filenames == frozenset()


# ── Local detection / self-pairing ───────────────────────────────────


def test_detect_mtp_file_finds_root_sibling(tmp_path):
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"x")
    (tmp_path / "mtp-model.gguf").write_bytes(b"x")
    (tmp_path / "MTP").mkdir()
    (tmp_path / "MTP" / "model-Q8_0-MTP.gguf").write_bytes(b"x")

    found = detect_mtp_file(str(tmp_path / "model-Q4_K_M.gguf"))
    assert found is not None
    assert found.endswith("mtp-model.gguf")


def test_detect_mtp_file_none_without_sibling(tmp_path):
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"x")
    assert detect_mtp_file(str(tmp_path / "model-Q4_K_M.gguf")) is None


def test_detect_dspark_file_prefers_matching_q8_sidecar(tmp_path):
    weight = tmp_path / "DeepSeek-V4-Flash-0731-UD-Q4_K_M.gguf"
    weight.write_bytes(b"target")
    folder = tmp_path / "dspark"
    folder.mkdir()
    q8 = folder / "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
    q8.write_bytes(b"q8")
    (folder / "dspark-Other-Model-Q8_0.gguf").write_bytes(b"foreign")

    assert detect_dspark_file(str(weight)) == str(q8.resolve())


def test_detect_dspark_file_accepts_the_suffix_naming_scheme(tmp_path):
    """``<model>-dspark.gguf`` pairs too, the same second scheme MTP accepts."""
    weight = tmp_path / "model-Q4_K_M.gguf"
    weight.write_bytes(b"target")
    folder = tmp_path / "dspark"
    folder.mkdir()
    sidecar = folder / "model-Q8_0-dspark.gguf"
    sidecar.write_bytes(b"draft")

    assert detect_dspark_file(str(weight)) == str(sidecar.resolve())


def test_detect_dspark_file_rejects_a_weight_copy_in_the_dspark_folder(tmp_path):
    """A published drafter NAME is required even inside ``dspark/``: a weight
    copy parked there would otherwise launch as --model-draft."""
    weight = tmp_path / "model-Q4_K_M.gguf"
    weight.write_bytes(b"target")
    folder = tmp_path / "dspark"
    folder.mkdir()
    (folder / "model-Q8_0.gguf").write_bytes(b"a full weight, not a sidecar")

    assert detect_dspark_file(str(weight)) is None


def test_detect_dspark_file_does_not_cross_attach_a_sibling_family(tmp_path):
    """A folder holding a model and its "-Lite" sibling must not pair across
    them: the family has to prefix the weight at a non-alphanumeric boundary."""
    base = tmp_path / "DeepSeek-V4-Flash-Q4_K_M.gguf"
    base.write_bytes(b"target")
    lite_sidecar = tmp_path / "dspark-DeepSeek-V4-Flash-Lite-Q8_0.gguf"
    lite_sidecar.write_bytes(b"foreign")

    assert detect_dspark_file(str(base)) is None

    own = tmp_path / "dspark-DeepSeek-V4-Flash-Q8_0.gguf"
    own.write_bytes(b"mine")
    assert detect_dspark_file(str(base)) == str(own.resolve())


def test_detect_dspark_file_skips_an_incomplete_split_set(tmp_path):
    """A partial split would fail llama-server's draft startup and disable
    speculation, so a complete lower-precision copy wins instead."""
    weight = tmp_path / "model-Q4_K_M.gguf"
    weight.write_bytes(b"target")
    (tmp_path / "dspark-model-Q8_0-00001-of-00002.gguf").write_bytes(b"x")
    complete = tmp_path / "dspark-model-BF16.gguf"
    complete.write_bytes(b"x")

    assert detect_dspark_file(str(weight)) == str(complete.resolve())


def test_detect_dspark_file_sums_shards_so_a_split_cannot_outrank_a_smaller_copy(tmp_path):
    """Candidates are collapsed to shard 1, so size must be summed across the
    set or a large split copy outranks a smaller single file at equal precision."""
    weight = tmp_path / "model-Q4_K_M.gguf"
    weight.write_bytes(b"target")
    folder = tmp_path / "dspark"
    folder.mkdir()
    for shard in (1, 2):
        (folder / f"dspark-model-Q8_0-0000{shard}-of-00002.gguf").write_bytes(b"x" * 4000)
    single = tmp_path / "dspark-model-Q8_0.gguf"
    single.write_bytes(b"x" * 5000)

    assert detect_dspark_file(str(weight)) == str(single.resolve())


def test_detect_dspark_file_keeps_a_split_sidecar_on_its_snapshot_path(tmp_path):
    """llama-server opens the sibling shards implicitly, and the blob target of
    a cache symlink has no sibling shard names, so a split must NOT resolve."""
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / "abc"
    blobs.mkdir()
    snapshot.mkdir(parents = True)
    weight = snapshot / "model-Q4_K_M.gguf"
    weight.write_bytes(b"target")

    first = snapshot / "dspark-model-Q8_0-00001-of-00002.gguf"
    second = snapshot / "dspark-model-Q8_0-00002-of-00002.gguf"
    (blobs / "sha_1").write_bytes(b"d" * 4096)
    (blobs / "sha_2").write_bytes(b"d")
    try:
        first.symlink_to(blobs / "sha_1")
        second.symlink_to(blobs / "sha_2")
    except OSError:
        pytest.skip("symlinks unavailable")

    found = detect_dspark_file(str(weight), str(snapshot))
    assert found == str(first)
    assert (Path(found).parent / second.name).exists()


def test_detect_gguf_model_rejects_drafter_file(tmp_path):
    drafter = tmp_path / "mtp-model.gguf"
    drafter.write_bytes(b"x")
    assert detect_gguf_model(str(drafter)) is None


def test_detect_gguf_model_dir_skips_companions(tmp_path):
    main = tmp_path / "model-Q4_K_M.gguf"
    main.write_bytes(b"xxxx")
    # Companions are larger so a size-sorted pick would wrongly win.
    (tmp_path / "mtp-model.gguf").write_bytes(b"x" * 64)
    (tmp_path / "mmproj-F16.gguf").write_bytes(b"x" * 128)

    assert detect_gguf_model(str(tmp_path)) == str(main.resolve())


def test_detect_mtp_file_pairs_by_weight_name(tmp_path):
    # Multi-model folder: each weight must get its own drafter, never the
    # first-sorted foreign one.
    (tmp_path / "gemma-4-12b-it-Q4_K_M.gguf").write_bytes(b"x")
    (tmp_path / "gemma-4-31B-it-Q4_K_M.gguf").write_bytes(b"x")
    (tmp_path / "mtp-gemma-4-12b-it.gguf").write_bytes(b"x")
    (tmp_path / "mtp-gemma-4-31B-it.gguf").write_bytes(b"x")

    found = detect_mtp_file(str(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf"))
    assert found is not None and found.endswith("mtp-gemma-4-31B-it.gguf")


def test_detect_mtp_file_skips_foreign_drafter(tmp_path):
    (tmp_path / "qwen3-8b-Q4_K_M.gguf").write_bytes(b"x")
    (tmp_path / "mtp-gemma-4-12b-it.gguf").write_bytes(b"x")
    assert detect_mtp_file(str(tmp_path / "qwen3-8b-Q4_K_M.gguf")) is None


def test_detect_mtp_file_qat_prefix_layout(tmp_path):
    # unsloth's qat repo: drafter stem omits the -qat suffix but prefixes
    # the weight name (mtp-gemma-4-12B-it.gguf / gemma-4-12B-it-qat-Q4_0.gguf).
    (tmp_path / "gemma-4-12B-it-qat-Q4_0.gguf").write_bytes(b"x")
    (tmp_path / "mtp-gemma-4-12B-it.gguf").write_bytes(b"x")
    found = detect_mtp_file(str(tmp_path / "gemma-4-12B-it-qat-Q4_0.gguf"))
    assert found is not None and found.endswith("mtp-gemma-4-12B-it.gguf")


def test_detect_mtp_file_search_root(tmp_path):
    # Weight in a quant subdir, drafter at the granted directory root.
    sub = tmp_path / "Q4_K_M"
    sub.mkdir()
    (sub / "gemma-4-12b-it-Q4_K_M.gguf").write_bytes(b"x")
    (tmp_path / "mtp-gemma-4-12b-it.gguf").write_bytes(b"x")
    found = detect_mtp_file(str(sub / "gemma-4-12b-it-Q4_K_M.gguf"), search_root = str(tmp_path))
    assert found is not None and found.endswith("mtp-gemma-4-12b-it.gguf")


def test_quant_directory_selection_finds_repo_root_mtp(tmp_path):
    quant_dir = tmp_path / "Q4_0"
    quant_dir.mkdir()
    weight = quant_dir / "gemma-4-E4B-it-qat-Q4_0.gguf"
    weight.write_bytes(b"x")
    mtp_dir = tmp_path / "MTP"
    mtp_dir.mkdir()
    drafter = mtp_dir / "mtp-gemma-4-E4B-it-Q4_0.gguf"
    drafter.write_bytes(b"x")

    search_root = _local_gguf_companion_search_root(str(quant_dir), str(weight))
    assert Path(search_root).resolve() == tmp_path.resolve()
    config = ModelConfig.from_identifier(str(quant_dir))
    assert config.is_local
    assert config.gguf_file == str(weight.resolve())
    assert config.gguf_mtp_file == str(drafter.resolve())


def test_bare_relative_gguf_directory_is_local_source(tmp_path, monkeypatch):
    model_dir = tmp_path / "outputs" / "gemma"
    model_dir.mkdir(parents = True)
    weight = model_dir / "gemma-4-E4B-it-qat-Q4_0.gguf"
    weight.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)

    config = ModelConfig.from_identifier("outputs/gemma")
    assert config.is_local
    assert config.gguf_file == str(weight.resolve())


def test_detect_mtp_file_falls_back_to_new_scheme_subdir(tmp_path):
    weight = tmp_path / "gemma-4-E4B-it-qat-Q4_0.gguf"
    weight.write_bytes(b"x")
    sub = tmp_path / "MTP"
    sub.mkdir()
    (sub / "mtp-gemma-4-E4B-it-BF16.gguf").write_bytes(b"x")
    q4 = sub / "mtp-gemma-4-E4B-it-Q4_0.gguf"
    q4.write_bytes(b"x")

    found = detect_mtp_file(str(weight))
    assert found == str(q4.resolve())


def test_detect_mtp_file_falls_back_to_old_scheme_subdir(tmp_path):
    weight = tmp_path / "gemma-4-12b-it-Q4_K_M.gguf"
    weight.write_bytes(b"x")
    sub = tmp_path / "MTP"
    sub.mkdir()
    drafter = sub / "gemma-4-12b-it-Q8_0-MTP.gguf"
    drafter.write_bytes(b"x")

    found = detect_mtp_file(str(weight))
    assert found == str(drafter.resolve())


def test_detect_mtp_file_root_still_wins_over_subdir(tmp_path):
    weight = tmp_path / "gemma-4-E4B-it-qat-Q4_0.gguf"
    weight.write_bytes(b"x")
    root = tmp_path / "mtp-gemma-4-E4B-it.gguf"
    root.write_bytes(b"x")
    sub = tmp_path / "MTP"
    sub.mkdir()
    (sub / "mtp-gemma-4-E4B-it-Q4_0.gguf").write_bytes(b"x")

    assert detect_mtp_file(str(weight)) == str(root.resolve())


def test_detect_mtp_file_subdir_skips_foreign_drafter(tmp_path):
    weight = tmp_path / "gemma-4-E4B-it-qat-Q4_0.gguf"
    weight.write_bytes(b"x")
    sub = tmp_path / "MTP"
    sub.mkdir()
    (sub / "mtp-gemma-4-12b-it-Q4_0.gguf").write_bytes(b"x")

    assert detect_mtp_file(str(weight)) is None


@pytest.mark.parametrize(
    "companion_path",
    ["mtp-gemma-4-E4B-it-Q4_0.gguf", "MTP/mtp-gemma-4-E4B-it-Q4_0.gguf"],
)
def test_detect_mtp_file_requires_model_name_boundary(tmp_path, companion_path):
    weight = tmp_path / "gemma-4-E4B-item-qat-Q4_0.gguf"
    weight.write_bytes(b"x")
    companion = tmp_path / companion_path
    companion.parent.mkdir(parents = True, exist_ok = True)
    companion.write_bytes(b"x")

    assert detect_mtp_file(str(weight)) is None


def test_detect_mtp_file_accepts_case_variant_subdir(tmp_path):
    weight = tmp_path / "gemma-4-E4B-it-qat-Q4_0.gguf"
    weight.write_bytes(b"x")
    sub = tmp_path / "mtp"
    sub.mkdir()
    drafter = sub / "mtp-gemma-4-E4B-it-Q4_0.gguf"
    drafter.write_bytes(b"x")

    assert detect_mtp_file(str(weight)) == str(drafter.resolve())


def test_native_companion_parent_accepts_root_and_mtp_subdir(tmp_path):
    weight = tmp_path / "gemma-4-E4B-it-qat-Q4_0.gguf"
    weight.write_bytes(b"x")
    root_drafter = tmp_path / "mtp-gemma-4-E4B-it.gguf"
    root_drafter.write_bytes(b"x")
    sub = tmp_path / "MtP"
    sub.mkdir()
    nested_drafter = sub / "mtp-gemma-4-E4B-it-Q4_0.gguf"
    nested_drafter.write_bytes(b"x")

    assert native_gguf_companion_parent_allowed(root_drafter, weight)
    assert native_gguf_companion_parent_allowed(nested_drafter, weight, allowed_subdirs = ("mtp",))


def test_native_companion_parent_rejects_other_nested_directory(tmp_path):
    weight = tmp_path / "gemma-4-E4B-it-qat-Q4_0.gguf"
    weight.write_bytes(b"x")
    sub = tmp_path / "other"
    sub.mkdir()
    drafter = sub / "mtp-gemma-4-E4B-it-Q4_0.gguf"
    drafter.write_bytes(b"x")

    assert not native_gguf_companion_parent_allowed(drafter, weight)


def test_native_companion_parent_rejects_mtp_symlink_escape(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weight = model_dir / "gemma-4-E4B-it-qat-Q4_0.gguf"
    weight.write_bytes(b"x")
    outside = tmp_path / "outside"
    outside.mkdir()
    drafter = outside / "mtp-gemma-4-E4B-it-Q4_0.gguf"
    drafter.write_bytes(b"x")
    try:
        (model_dir / "MTP").symlink_to(outside, target_is_directory = True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    assert not native_gguf_companion_parent_allowed(
        model_dir / "MTP" / drafter.name, weight, allowed_subdirs = ("mtp",)
    )


# ── Reload dedup includes the drafter ────────────────────────────────


def _loaded_backend(weight, drafter_path):
    from core.inference.llama_cpp import LlamaCppBackend

    b = LlamaCppBackend()
    # Shape matches atexit cleanup expectations (terminate/wait/kill).
    b._process = SimpleNamespace(
        poll = lambda: None,
        terminate = lambda: None,
        wait = lambda timeout = None: 0,
        kill = lambda: None,
    )
    b._healthy = True
    b._model_identifier = "local-gemma"
    b._gguf_path = str(weight)
    b._hf_variant = None
    b._requested_n_ctx = 4096
    b._cache_type_kv = None
    b._requested_spec_mode = "auto"
    b._speculative_type = "draft-mtp" if drafter_path else "default"
    b._spec_draft_n_max = None
    b._chat_template_override = None
    b._extra_args = None
    b._mtp_draft_path = drafter_path
    return b


def _target_state_kwargs(weight, mtp_draft_path):
    return dict(
        model_identifier = "local-gemma",
        hf_variant = None,
        n_ctx = 4096,
        cache_type_kv = None,
        speculative_type = "auto",
        spec_draft_n_max = None,
        chat_template_override = None,
        extra_args = None,
        is_vision = False,
        gguf_path = str(weight),
        mtp_draft_path = mtp_draft_path,
    )


def test_already_in_target_state_bounces_on_new_drafter(tmp_path):
    from core.inference.llama_cpp import GgufLoadIntent

    weight = tmp_path / "gemma-4-12b-it-Q4_K_M.gguf"
    weight.write_bytes(b"x")
    drafter = tmp_path / "mtp-gemma-4-12b-it.gguf"
    drafter.write_bytes(b"x")

    # Loaded without a drafter; one now exists on disk -> must reload.
    b = _loaded_backend(weight, None)
    assert not b.adopt_load_intent_if_matched(
        GgufLoadIntent(**_target_state_kwargs(weight, str(drafter)))
    )
    # Same drafter as launched -> still deduped.
    b = _loaded_backend(weight, str(drafter))
    assert b.adopt_load_intent_if_matched(
        GgufLoadIntent(**_target_state_kwargs(weight, str(drafter)))
    )
    intent = dict(
        model_identifier = "local-gemma",
        n_ctx = 4096,
        mtp_draft_path = str(drafter),
        compare_mtp_draft = True,
    )
    assert b.adopt_load_intent_if_matched(GgufLoadIntent(**intent))
    intent["mtp_draft_path"] = None
    assert not b.adopt_load_intent_if_matched(GgufLoadIntent(**intent))


def test_detect_gguf_model_rejects_mtp_subdir_copy(tmp_path):
    # Direct selection of an MTP/ copy: the basename alone has no mtp-
    # prefix, so rejection relies on the parent dir name.
    sub = tmp_path / "MTP"
    sub.mkdir()
    copy = sub / "gemma-4-12b-it-BF16-MTP.gguf"
    copy.write_bytes(b"x")
    deep = sub / "BF16" / copy.name
    deep.parent.mkdir()
    deep.write_bytes(b"x")
    assert detect_gguf_model(str(copy)) is None
    # Selecting the MTP dir itself must not surface the copies as models.
    assert detect_gguf_model(str(sub)) is None
    assert list_local_gguf_variants(str(tmp_path))[0] == []
    assert list_hub_local_gguf_variants(str(tmp_path))[0] == []


def test_registered_mtp_root_keeps_descendant_models_and_excludes_companions(tmp_path, monkeypatch):
    root = tmp_path / "MTP"
    nested = root / "BF16"
    nested.mkdir(parents = True)
    main = root / "Qwen3.6-27B-MTP-001-of-002.gguf"
    terminal = root / "gemma-4-12b-it-Q8_0-MTP.gguf"
    prefixed = root / "mtp-gemma-4-12b-it-Q8_0.gguf"
    nested_model = nested / "gemma-4-12b-it-Q8_0-MTP-001-of-002.gguf"
    for file, size in ((main, 100), (terminal, 20), (prefixed, 30), (nested_model, 40)):
        file.write_bytes(b"x" * size)
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: [{"path": str(root)}])

    assert detect_gguf_model(str(main)) == detect_gguf_model(str(root)) == str(main.resolve())
    assert all(detect_gguf_model(str(file)) is None for file in (terminal, prefixed))
    assert detect_gguf_model(str(nested_model)) == str(nested_model.resolve())
    assert [(v.quant, v.filename) for v in list_local_gguf_variants(str(root))[0]] == [
        ("MTP", main.name),
        ("Q8_0", f"BF16/{nested_model.name}"),
    ]
    hub_variants = list_hub_local_gguf_variants(str(root))[0]
    assert (hub_variants[0].quant, hub_variants[0].filename) == ("Qwen3.6-27B-MTP", main.name)
    assert (hub_variants[-1].quant, hub_variants[-1].filename) == (
        "Q8_0",
        f"BF16/{nested_model.name}",
    )
    config = ModelConfig.from_identifier(str(root), gguf_variant = hub_variants[0].quant)
    assert config and config.is_gguf and config.gguf_file == str(main.resolve())


# ── Root drafter wins over new-scheme MTP/ copies ────────────────────
# The MTP/ copies were renamed to share the mtp- basename prefix (e.g.
# MTP/mtp-gemma-4-E4B-it-BF16.gguf). Auto-fetch/load must still resolve the
# small repo-root drafter, not a sort-first MTP/ copy (uppercase precedes
# lowercase, so the subdir path would otherwise win).

NEW_SCHEME_SIBLINGS = [
    _sib("gemma-4-12b-it-Q4_K_M.gguf", 4_000, "main-q4"),
    _sib("gemma-4-12b-it-Q8_0.gguf", 8_000, "main-q8"),
    _sib("mtp-gemma-4-12b-it.gguf", 100, "drafter"),
    _sib("MTP/mtp-gemma-4-12b-it-Q8_0.gguf", 100, "mtp-sub-q8"),
    _sib("MTP/mtp-gemma-4-12b-it-BF16.gguf", 200, "mtp-sub-bf16"),
    _sib("mmproj-F16.gguf", 500, "mmproj"),
]


def test_preferred_mtp_sibling_prefers_root_over_new_scheme_copies():
    picked = preferred_mtp_sibling(NEW_SCHEME_SIBLINGS)
    assert picked is not None and picked.rfilename == "mtp-gemma-4-12b-it.gguf"


def test_variant_plans_new_scheme_uses_root_drafter():
    plans = build_gguf_variant_plans(NEW_SCHEME_SIBLINGS)
    assert set(plans) == {"q4_k_m", "q8_0"}
    for plan in plans.values():
        assert "mtp-gemma-4-12b-it.gguf" in plan.target_filenames
        assert not any("MTP/" in name for name in plan.target_filenames)
        assert "drafter" in plan.companion_hashes
    # Download size = main + mmproj + root drafter (not the 200-byte BF16 copy).
    assert plans["q4_k_m"].download_size_bytes == 4_600


def test_download_mtp_prefers_root_over_new_scheme_copies(monkeypatch):
    # _pick_mtp is nested; capture it via the companion-download seam.
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)  # online: skip reuse probe
    captured = {}

    def _fake_companion(
        *,
        hf_repo,
        hf_token,
        pick,
        label,
        cancel_event = None,
        near_path = None,
    ):
        captured["pick"] = pick
        return None

    b = LlamaCppBackend()
    b._download_companion_gguf = _fake_companion
    b._download_mtp(hf_repo = "unsloth/gemma-4-E4B-it-qat-mobile-GGUF")

    repo_files = [
        "MTP/mtp-gemma-4-E4B-it-BF16.gguf",
        "MTP/mtp-gemma-4-E4B-it-Q4_0.gguf",
        "MTP/mtp-gemma-4-E4B-it-Q8_0.gguf",
        "gemma-4-E4B-it-qat-UD-Q2_K_XL.gguf",
        "mmproj-F16.gguf",
        "mtp-gemma-4-E4B-it.gguf",
    ]
    assert captured["pick"](repo_files) == "mtp-gemma-4-E4B-it.gguf"


# ── Reuse an on-disk drafter offline; fetch fresh online ─────────────


def _seed_snapshot(tmp_path, names):
    snap = tmp_path / "snap"
    for rel in names:
        f = snap / rel
        f.parent.mkdir(parents = True, exist_ok = True)
        f.write_bytes(b"x")
    return snap


def test_download_mtp_reuses_cached_root_drafter_offline(tmp_path, monkeypatch):
    import utils.models.model_config as mc
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    snap = _seed_snapshot(
        tmp_path,
        [
            "gemma-4-E4B-it-qat-UD-Q2_K_XL.gguf",
            "mtp-gemma-4-E4B-it.gguf",
            "MTP/mtp-gemma-4-E4B-it-BF16.gguf",
            "mmproj-F16.gguf",
        ],
    )
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: [snap])

    got = LlamaCppBackend()._download_mtp(hf_repo = "unsloth/gemma-4-E4B-it-qat-mobile-GGUF")
    assert got is not None and Path(got).name == "mtp-gemma-4-E4B-it.gguf"


def test_download_mtp_reuses_cached_subdir_copy_when_no_root_offline(tmp_path, monkeypatch):
    # Pre-fix build may have fetched only the MTP/ copy; reuse it offline.
    import utils.models.model_config as mc
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    snap = _seed_snapshot(
        tmp_path,
        [
            "gemma-4-E4B-it-qat-UD-Q2_K_XL.gguf",
            "MTP/mtp-gemma-4-E4B-it-BF16.gguf",
        ],
    )
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: [snap])

    got = LlamaCppBackend()._download_mtp(hf_repo = "unsloth/gemma-4-E4B-it-qat-mobile-GGUF")
    assert got is not None and Path(got).name == "mtp-gemma-4-E4B-it-BF16.gguf"


def test_download_mtp_prefers_root_across_snapshots_offline(tmp_path, monkeypatch):
    # A newer partial snapshot holds only the MTP/ copy; an older one has the
    # root. Must still return the small root, not the large subdir copy.
    import utils.models.model_config as mc
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    snap_partial = _seed_snapshot(tmp_path / "new", ["MTP/mtp-gemma-4-E4B-it-BF16.gguf"])
    snap_full = _seed_snapshot(tmp_path / "old", ["mtp-gemma-4-E4B-it.gguf"])
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: [snap_partial, snap_full])

    got = LlamaCppBackend()._download_mtp(hf_repo = "unsloth/gemma-4-E4B-it-qat-mobile-GGUF")
    assert got is not None and Path(got).name == "mtp-gemma-4-E4B-it.gguf"


def test_download_mtp_reuse_follows_snapshot_order_offline(tmp_path, monkeypatch):
    # Two snapshots both hold a root drafter; newest-first order must win so a
    # fresh main GGUF is not paired with a stale drafter revision.
    import utils.models.model_config as mc
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    newest = _seed_snapshot(tmp_path / "newest", ["mtp-gemma-4-E4B-it.gguf"])
    oldest = _seed_snapshot(tmp_path / "oldest", ["mtp-gemma-4-E4B-it.gguf"])
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: [newest, oldest])

    got = LlamaCppBackend()._download_mtp(hf_repo = "unsloth/gemma-4-E4B-it-qat-mobile-GGUF")
    assert got is not None and Path(got).parent.parent.name == "newest"


def test_download_mtp_prefers_main_snapshot_offline(tmp_path, monkeypatch):
    import utils.models.model_config as mc
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    snapshots = tmp_path / "models--unsloth--gemma" / "snapshots"
    old = snapshots / "old"
    new = snapshots / "new"
    old.mkdir(parents = True)
    new.mkdir(parents = True)
    main = old / "gemma-UD-Q4_K_XL.gguf"
    old_drafter = old / "mtp-gemma.gguf"
    new_drafter = new / "mtp-gemma.gguf"
    main.write_bytes(b"main")
    old_drafter.write_bytes(b"old")
    new_drafter.write_bytes(b"new")
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda _repo: [new, old])

    got = LlamaCppBackend()._download_mtp(
        hf_repo = "unsloth/gemma-GGUF",
        near_path = str(main),
    )

    assert got == str(old_drafter)


def test_download_mtp_online_skips_cache_reuse(tmp_path, monkeypatch):
    # Online, do not reuse a cached copy: go to the download path so a changed
    # drafter is refetched (hf_hub_download checks the current revision).
    import utils.models.model_config as mc
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    snap = _seed_snapshot(tmp_path, ["mtp-gemma-4-E4B-it.gguf"])
    monkeypatch.setattr(mc, "_iter_hf_cache_snapshots", lambda repo: [snap])

    reached = {}

    def _fake_companion(
        *,
        hf_repo,
        hf_token,
        pick,
        label,
        cancel_event = None,
        near_path = None,
    ):
        reached["hit"] = True
        return None

    b = LlamaCppBackend()
    b._download_companion_gguf = _fake_companion
    assert b._download_mtp(hf_repo = "unsloth/gemma-4-E4B-it-qat-mobile-GGUF") is None
    assert reached.get("hit") is True


# ── DSpark sidecar fetch is gated on the binary that would launch it ──


def _dspark_download_probe(
    monkeypatch,
    *,
    supports_dspark,
    cached = None,
):
    """Run _download_dspark against a stubbed capability probe and an optionally
    cached sidecar; report whether the ~11 GB fetch (and even the repo listing)
    was reached."""
    import core.inference.llama_cpp as llama_cpp_module
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: {"supports_dspark": supports_dspark}),
    )
    monkeypatch.setattr(
        llama_cpp_module, "_companion_snapshot_sibling", lambda near_path, pick: cached
    )
    reached = {}

    def _fake_companion(
        *,
        hf_repo,
        hf_token,
        pick,
        label,
        cancel_event = None,
        near_path = None,
        outcome = None,
    ):
        reached["hit"] = True
        if outcome is not None:
            outcome["listed"] = True
        return "/cache/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"

    b = LlamaCppBackend()
    b._download_companion_gguf = _fake_companion
    got = b._download_dspark(
        hf_repo = "unsloth/DeepSeek-V4-Flash-0731-GGUF",
        near_path = "/cache/snap/DeepSeek-V4-Flash-0731-Q4_K_M.gguf",
        binary = "/fake/llama-server",
    )
    return got, reached.get("hit", False)


def test_download_dspark_skips_the_fetch_when_the_binary_cannot_run_it(monkeypatch):
    """The sidecar is ~11 GB and _build_speculative_flags drops DSpark outright on
    a binary without --spec-type draft-dspark (every prebuilt in the known-broken
    window included), so the capability must be checked BEFORE the download."""
    got, reached = _dspark_download_probe(monkeypatch, supports_dspark = False)
    assert got is None
    assert reached is False


def test_download_dspark_fetches_when_the_binary_supports_it(monkeypatch):
    got, reached = _dspark_download_probe(monkeypatch, supports_dspark = True)
    assert got == "/cache/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
    assert reached is True


def test_download_dspark_records_whether_the_repo_publishes_a_sidecar(monkeypatch):
    """The reuse check retries a failed fetch but must never retry a repo that
    ships none, so the two "returned None" cases have to stay distinguishable."""
    import core.inference.llama_cpp as llama_cpp_module
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: {"supports_dspark": True}),
    )
    monkeypatch.setattr(
        llama_cpp_module, "_companion_snapshot_sibling", lambda near_path, pick: None
    )

    def _companion(listed):
        def _fake(
            *,
            hf_repo,
            hf_token,
            pick,
            label,
            cancel_event = None,
            near_path = None,
            outcome = None,
        ):
            if outcome is not None:
                outcome["listed"] = listed
            return None

        return _fake

    for listed, expect_absent in ((False, True), (True, False)):
        b = LlamaCppBackend()
        b._download_companion_gguf = _companion(listed)
        assert b._download_dspark(hf_repo = "org/repo", binary = "/fake/llama-server") is None
        assert b._dspark_sidecar_absent is expect_absent


def test_an_unreachable_hub_is_not_recorded_as_a_missing_sidecar(monkeypatch, tmp_path):
    """A listing that never completed says nothing about the repo. Recording it as
    a definitive absence would suppress the reuse check's retry, so DSpark would
    never be fetched once connectivity returned."""
    import core.inference.llama_cpp as llama_cpp_module
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setattr(
        llama_cpp_module, "_companion_snapshot_sibling", lambda near_path, pick: None
    )
    monkeypatch.setattr(llama_cpp_module, "_hub_download_in_flight", lambda repo: False)
    monkeypatch.setattr(
        llama_cpp_module, "_hub_cache_dir_for_snapshot_path", lambda p: str(tmp_path)
    )

    def _explode(repo, token = None):
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr("huggingface_hub.list_repo_files", _explode)
    monkeypatch.setattr("utils.models.model_config._iter_hf_cache_snapshots", lambda *a, **k: [])

    outcome: dict = {}
    b = LlamaCppBackend()
    b._cancel_event.clear()
    assert (
        b._download_companion_gguf(
            hf_repo = "org/repo",
            hf_token = None,
            pick = lambda names: None,
            label = "DSpark drafter",
            outcome = outcome,
        )
        is None
    )
    assert "listed" not in outcome


def test_download_dspark_still_reports_a_cached_sidecar_it_cannot_run(monkeypatch):
    """Skipping the fetch must not hide a sidecar already on disk: the route
    rediscovers it on every Apply, so answering None would leave the reuse check
    comparing it against the launched None and reloading the same drafter-free
    server each time. _build_speculative_flags re-checks and still falls back."""
    cached = "/cache/snap/dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
    got, reached = _dspark_download_probe(monkeypatch, supports_dspark = False, cached = cached)
    assert got == cached
    assert reached is False


@pytest.mark.parametrize("shape", ["dangling", "directory"])
def test_detect_dspark_file_skips_a_sidecar_it_cannot_open(tmp_path, shape):
    """A dangling snapshot symlink or a directory named like a sidecar must read
    as "no sidecar", not as a drafter: handing llama-server a --model-draft it
    cannot open fails the whole load instead of falling back to no speculation.
    detect_mtp_file has always guarded this."""
    import os

    weight = tmp_path / "model-Q4_K_M.gguf"
    weight.write_bytes(b"target")
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    if shape == "dangling":
        os.symlink(tmp_path / "missing_blob", sidecar)
    else:
        sidecar.mkdir()

    assert detect_dspark_file(str(weight)) is None
    # Same shape, same answer for MTP: the two must not diverge.
    mtp = tmp_path / "mtp-model.gguf"
    if shape == "dangling":
        os.symlink(tmp_path / "missing_blob", mtp)
    else:
        mtp.mkdir()
    assert detect_mtp_file(str(weight)) is None


@pytest.mark.parametrize("kind", ["dspark", "mtp"])
def test_a_release_specific_sidecar_outranks_the_base_family(tmp_path, kind):
    """Both prefix-match a 0731 weight, and only one is really its drafter. The
    prefix rule cannot be tightened to equality -- mtp-gemma-4-12B-it.gguf really
    does ship beside gemma-4-12B-it-qat-*.gguf -- so ranking settles it."""
    weight = tmp_path / "DeepSeek-V4-Flash-0731-UD-Q4_K_XL.gguf"
    weight.write_bytes(b"target")
    base = tmp_path / f"{kind}-DeepSeek-V4-Flash-Q8_0.gguf"
    exact = tmp_path / f"{kind}-DeepSeek-V4-Flash-0731-Q8_0.gguf"
    base.write_bytes(b"x" * 10)
    exact.write_bytes(b"x" * 4000)  # deliberately the larger, so size cannot decide

    detect = detect_dspark_file if kind == "dspark" else detect_mtp_file
    found = detect(str(weight))
    assert found is not None and Path(found).name == exact.name


def test_root_mtp_candidates_are_ranked_not_taken_in_directory_order(tmp_path):
    """The root branch returns the first match it walks, so specificity has to be
    applied before the walk: mtp-model.gguf sorts ahead of mtp-model_v2-Q8_0.gguf
    yet only the second names this weight's family."""
    weight = tmp_path / "model_v2-Q4_K_M.gguf"
    weight.write_bytes(b"target")
    (tmp_path / "mtp-model.gguf").write_bytes(b"base")
    exact = tmp_path / "mtp-model_v2-Q8_0.gguf"
    exact.write_bytes(b"exact")

    found = detect_mtp_file(str(weight))
    assert found is not None and Path(found).name == exact.name


def test_root_mtp_ranking_spans_the_search_root(tmp_path):
    """Same ordering hazard across the two scanned directories: the base sits in
    the model's own folder and the exact match only in the search root."""
    home = tmp_path / "home"
    root = tmp_path / "root"
    home.mkdir()
    root.mkdir()
    weight = home / "model_v2-Q4_K_M.gguf"
    weight.write_bytes(b"target")
    (home / "mtp-model.gguf").write_bytes(b"base")
    exact = root / "mtp-model_v2-Q8_0.gguf"
    exact.write_bytes(b"exact")

    found = detect_mtp_file(str(weight), search_root = str(root))
    assert found is not None and Path(found).name == exact.name


def test_a_more_specific_subdir_drafter_beats_a_base_family_root_one(tmp_path):
    """Specificity has to be compared across layouts, not only within one: the
    root file names a different family, so it is not this weight's drafter at all
    and root preference must not save it."""
    weight = tmp_path / "model_v2-Q4_K_M.gguf"
    weight.write_bytes(b"target")
    (tmp_path / "mtp-model.gguf").write_bytes(b"base")
    (tmp_path / "MTP").mkdir()
    exact = tmp_path / "MTP" / "mtp-model_v2-Q8_0.gguf"
    exact.write_bytes(b"exact")

    found = detect_mtp_file(str(weight))
    assert found is not None and Path(found).name == exact.name


def test_root_still_wins_when_both_layouts_name_the_same_family(tmp_path):
    """Negative control: equal specificity is every published layout, and there
    the root copy keeps its long-standing preference."""
    weight = tmp_path / "model-Q4_K_M.gguf"
    weight.write_bytes(b"target")
    root = tmp_path / "mtp-model.gguf"
    root.write_bytes(b"root")
    (tmp_path / "MTP").mkdir()
    (tmp_path / "MTP" / "model-Q8_0-MTP.gguf").write_bytes(b"sub")

    found = detect_mtp_file(str(weight))
    assert found is not None and Path(found).name == root.name


def test_an_unusable_candidate_does_not_win_the_tier_comparison(tmp_path):
    """Tier order is decided by specificity, so an unusable candidate that
    outranks everything would put its tier first and then be skipped, handing
    back a less specific sibling instead of the usable copy next door."""
    import os

    weight = tmp_path / "model_v2_release-Q4_K_M.gguf"
    weight.write_bytes(b"target")
    # Most specific, but a dangling link: it must not speak for the root tier.
    os.symlink(tmp_path / "missing", tmp_path / "mtp-model_v2_release.gguf")
    (tmp_path / "mtp-model.gguf").write_bytes(b"base")
    (tmp_path / "MTP").mkdir()
    usable = tmp_path / "MTP" / "mtp-model_v2-Q8_0.gguf"
    usable.write_bytes(b"mid")

    found = detect_mtp_file(str(weight))
    assert found is not None and Path(found).name == usable.name


def test_a_rejected_candidate_does_not_win_the_tier_comparison(tmp_path):
    """Same shape as the unusable case, via accept: a native grant can reject the
    most specific root file, which is then skipped at emission. It must not have
    spoken for its tier, or a base-family sibling goes out ahead of the more
    specific accepted copy under MTP/."""
    weight = tmp_path / "model_v2_release-Q4_K_M.gguf"
    weight.write_bytes(b"target")
    rejected = tmp_path / "mtp-model_v2_release.gguf"
    rejected.write_bytes(b"out-of-grant")
    (tmp_path / "mtp-model.gguf").write_bytes(b"base")
    (tmp_path / "MTP").mkdir()
    accepted = tmp_path / "MTP" / "mtp-model_v2-Q8_0.gguf"
    accepted.write_bytes(b"mid")

    found = detect_mtp_file(str(weight), accept = lambda candidate: rejected.name not in candidate)
    assert found is not None and Path(found).name == accepted.name


def test_a_qat_weight_still_pairs_with_its_base_family_drafter(tmp_path):
    """Negative control for the ranking above: unsloth/gemma-4-12B-it-qat-GGUF
    ships mtp-gemma-4-12B-it.gguf, so the prefix rule has to keep working when no
    more specific sidecar exists."""
    weight = tmp_path / "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
    weight.write_bytes(b"target")
    drafter = tmp_path / "mtp-gemma-4-12B-it.gguf"
    drafter.write_bytes(b"draft")

    found = detect_mtp_file(str(weight))
    assert found is not None and Path(found).name == drafter.name


def test_detect_mtp_file_returns_first_shard_of_split_subdir_drafter(tmp_path):
    """llama-server takes shard 1 as the model path, so a split MTP/ copy must
    not resolve to whichever shard happens to be smallest."""
    weight = tmp_path / "model-Q4_0.gguf"
    weight.write_bytes(b"x")
    sub = tmp_path / "MTP"
    sub.mkdir()
    first = sub / "mtp-model-Q4_0-00001-of-00002.gguf"
    first.write_bytes(b"x" * 4096)
    (sub / "mtp-model-Q4_0-00002-of-00002.gguf").write_bytes(b"x")

    assert detect_mtp_file(str(weight)) == str(first.resolve())


def test_detect_mtp_file_skip_root_ignores_root_drafter(tmp_path):
    """skip_root is how a native load recovers when the root drafter is out
    of bounds for its grant."""
    quant_dir = tmp_path / "Q4_0"
    quant_dir.mkdir()
    weight = quant_dir / "model.gguf"
    weight.write_bytes(b"x")
    (tmp_path / "mtp-model.gguf").write_bytes(b"x")
    sub = tmp_path / "MTP"
    sub.mkdir()
    subdir_copy = sub / "mtp-model-Q4_0.gguf"
    subdir_copy.write_bytes(b"x")

    assert detect_mtp_file(str(weight), str(tmp_path)) == str(
        (tmp_path / "mtp-model.gguf").resolve()
    )
    assert detect_mtp_file(str(weight), str(tmp_path), skip_root = True) == str(subdir_copy.resolve())


def test_detect_mtp_file_rejects_weight_copy_inside_mtp_dir(tmp_path):
    """Everything under MTP/ counts as a drafter for menu exclusion, but only
    a published drafter name may be launched as --model-draft."""
    weight = tmp_path / "gemma-4-E4B-it-qat-Q4_0.gguf"
    weight.write_bytes(b"x")
    sub = tmp_path / "MTP"
    sub.mkdir()
    (sub / "gemma-4-E4B-it-qat-Q4_0.gguf").write_bytes(b"x")

    assert detect_mtp_file(str(weight)) is None


def test_detect_mtp_file_pairs_k_quant_subdir_drafter(tmp_path):
    """Pairing must use the full quant vocabulary, not just Q<d>_<d>/BF16/F16."""
    weight = tmp_path / "gemma-4-12b-it-Q4_0.gguf"
    weight.write_bytes(b"x")
    sub = tmp_path / "MTP"
    sub.mkdir()
    drafter = sub / "mtp-gemma-4-12b-it-UD-Q4_K_XL.gguf"
    drafter.write_bytes(b"x")

    assert detect_mtp_file(str(weight)) == str(drafter.resolve())


def test_detect_mtp_file_keeps_snapshot_path_for_sharded_subdir_drafter(tmp_path):
    """A split copy stored as HF snapshot symlinks must launch from the
    snapshot path: the blob target has no sibling shard names."""
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / "abc"
    sub = snapshot / "MTP"
    blobs.mkdir(parents = True)
    sub.mkdir(parents = True)

    (blobs / "sha_weight").write_bytes(b"w")
    weight = snapshot / "model-Q4_0.gguf"
    try:
        weight.symlink_to(blobs / "sha_weight")
    except OSError:
        pytest.skip("symlinks unavailable")

    first = sub / "mtp-model-Q4_0-00001-of-00002.gguf"
    second = sub / "mtp-model-Q4_0-00002-of-00002.gguf"
    (blobs / "sha_1").write_bytes(b"d" * 4096)
    (blobs / "sha_2").write_bytes(b"d")
    first.symlink_to(blobs / "sha_1")
    second.symlink_to(blobs / "sha_2")

    found = detect_mtp_file(str(weight), str(snapshot))
    assert found == str(first)
    assert (Path(found).parent / second.name).exists()


def test_detect_mtp_file_pairs_sharded_old_scheme_subdir_drafter(tmp_path):
    """An old-scheme split copy is <model>-Q8_0-MTP-00001-of-00002.gguf, whose
    stem does not end in -mtp until the shard suffix comes off."""
    weight = tmp_path / "model-Q4_0.gguf"
    weight.write_bytes(b"x")
    sub = tmp_path / "MTP"
    sub.mkdir()
    first = sub / "model-Q8_0-MTP-00001-of-00002.gguf"
    first.write_bytes(b"x" * 4096)
    (sub / "model-Q8_0-MTP-00002-of-00002.gguf").write_bytes(b"x")

    assert detect_mtp_file(str(weight)) == str(first)


def test_detect_mtp_file_keeps_snapshot_path_for_sharded_root_drafter(tmp_path):
    """The root branch needs the same shard handling as the MTP/ branch."""
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / "abc"
    blobs.mkdir(parents = True)
    snapshot.mkdir(parents = True)

    (blobs / "sha_weight").write_bytes(b"w")
    weight = snapshot / "model-Q4_0.gguf"
    try:
        weight.symlink_to(blobs / "sha_weight")
    except OSError:
        pytest.skip("symlinks unavailable")

    first = snapshot / "mtp-model-Q4_0-00001-of-00002.gguf"
    second = snapshot / "mtp-model-Q4_0-00002-of-00002.gguf"
    (blobs / "sha_1").write_bytes(b"d" * 4096)
    (blobs / "sha_2").write_bytes(b"d")
    first.symlink_to(blobs / "sha_1")
    second.symlink_to(blobs / "sha_2")

    found = detect_mtp_file(str(weight), str(snapshot))
    assert found == str(first)
    assert (Path(found).parent / second.name).exists()


def test_detect_mtp_file_pairs_bpw_qualified_subdir_drafter(tmp_path):
    """_extract_quant_label supports bpw-qualified names, so pairing must too."""
    weight = tmp_path / "model-Q4_0.gguf"
    weight.write_bytes(b"x")
    sub = tmp_path / "MTP"
    sub.mkdir()
    drafter = sub / "mtp-model-IQ4_XS-3.53bpw.gguf"
    drafter.write_bytes(b"x")

    assert detect_mtp_file(str(weight)) == str(drafter.resolve())


def test_detect_mtp_file_skips_incomplete_split_drafter(tmp_path):
    """An incomplete shard set fails llama-server's draft startup, so a
    complete copy must win rather than MTP being disabled."""
    weight = tmp_path / "model-Q4_0.gguf"
    weight.write_bytes(b"x")
    sub = tmp_path / "MTP"
    sub.mkdir()
    # Declares two shards but ships only the first.
    (sub / "mtp-model-Q4_0-00001-of-00002.gguf").write_bytes(b"x" * 50)
    complete = sub / "mtp-model-BF16.gguf"
    complete.write_bytes(b"x" * 100)

    assert detect_mtp_file(str(weight)) == str(complete.resolve())


def test_detect_mtp_file_ranks_split_drafter_by_total_size(tmp_path):
    """Candidates collapse to shard 1, so a split copy must be summed or it
    outranks a smaller single file."""
    weight = tmp_path / "model-Q4_0.gguf"
    weight.write_bytes(b"x")
    sub = tmp_path / "MTP"
    sub.mkdir()
    (sub / "mtp-model-Q8_0-00001-of-00002.gguf").write_bytes(b"x" * 90)
    (sub / "mtp-model-Q8_0-00002-of-00002.gguf").write_bytes(b"x" * 90)
    smaller = sub / "mtp-model-BF16.gguf"
    smaller.write_bytes(b"x" * 100)

    assert detect_mtp_file(str(weight)) == str(smaller.resolve())


def test_companion_search_root_promotes_bpw_quant_directory(tmp_path):
    """A bpw-qualified quant directory must resolve to the repository root, or
    the repo-root MTP/ copy is never in scope for it."""
    quant_dir = tmp_path / "IQ4_XS-3.53bpw"
    quant_dir.mkdir()
    weight = quant_dir / "model.gguf"
    weight.write_bytes(b"x")
    sub = tmp_path / "MTP"
    sub.mkdir()
    drafter = sub / "mtp-model.gguf"
    drafter.write_bytes(b"x")

    # Directory selection and the file inside it agree on the root.
    assert _local_gguf_companion_search_root(str(quant_dir), str(weight)) == str(tmp_path)
    assert _local_gguf_companion_search_root(str(weight), str(weight)) == str(tmp_path)
    assert detect_mtp_file(str(weight), str(tmp_path)) == str(drafter.resolve())


def test_companion_search_root_keeps_non_quant_directories(tmp_path):
    """Sharing the quant vocabulary must not widen what gets promoted."""
    for name in ("DeepSeek-V3-UD-Q2_K_XL", "outputs", "Q4_0-extra", "Q4_0bpw"):
        directory = tmp_path / name
        directory.mkdir()
        weight = directory / "model.gguf"
        weight.write_bytes(b"x")
        assert _local_gguf_companion_search_root(str(directory), str(weight)) == str(directory)


# ── DSpark drafters (DeepSeek V4 Flash) ──────────────────────────────

DEEPSEEK_SIBLINGS = [
    _sib("UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00002.gguf", 9_000, "q4-1"),
    _sib("UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00002-of-00002.gguf", 8_000, "q4-2"),
    _sib("UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00002.gguf", 5_000, "iq1-1"),
    _sib("dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf", 1_100, "dspark-bf16"),
    _sib("dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf", 1_000, "dspark-q8"),
]


def test_dspark_drafters_are_not_quants_and_are_not_auto_fetched():
    plans = build_gguf_variant_plans(DEEPSEEK_SIBLINGS)

    # The drafters carry BF16/Q8_0 tokens; neither may become a quant. They were
    # also the two smallest entries, so the fit heuristic used to promote them in
    # a repo whose real quants are 87 GB+.
    assert set(plans) == {"ud-q4_k_xl", "ud-iq1_s"}

    # DSpark is opt-in and ~11 GB per file, so unlike the root mtp-*.gguf it must
    # not be folded into every plan.
    for plan in plans.values():
        assert not any(name.startswith("dspark/") for name in plan.target_filenames)
        assert plan.companion_hashes == frozenset()

    q4 = plans["ud-q4_k_xl"]
    assert q4.main_size_bytes == 17_000
    assert q4.download_size_bytes == 17_000


def test_a_root_dflash_drafter_is_not_a_quant():
    """ggml-org/Qwen3.6-27B-GGUF ships a 3 GB dflash- drafter beside the real
    54 GB BF16; merging them hands llama.cpp the drafter as the model."""
    plans = build_gguf_variant_plans(
        [
            _sib("Qwen3.6-27B-BF16.gguf", 54_000, "bf16"),
            _sib("dflash-Qwen3.6-27B-BF16.gguf", 3_000, "dflash"),
        ]
    )
    assert set(plans) == {"bf16"}
    assert plans["bf16"].main_filenames == frozenset({"Qwen3.6-27B-BF16.gguf"})
    assert plans["bf16"].main_size_bytes == 54_000


def test_gemma_mtp_is_still_auto_downloaded():
    """Filtering only removes drafters from the quant list; the root mtp-*.gguf
    is still fetched with every variant so MTP speculative decoding works."""
    plans = build_gguf_variant_plans(GEMMA_SIBLINGS)
    for plan in plans.values():
        assert "mtp-gemma-4-12b-it.gguf" in plan.target_filenames


def test_a_cached_dspark_drafter_is_never_launched_as_an_mtp_drafter(tmp_path, monkeypatch):
    """_cached_repo_mtp_drafter uses the predicate inversely, to pick a drafter to
    launch with --spec-type draft-mtp. DSpark needs draft-dspark, so broadening
    that predicate must not widen what is launched."""
    import core.inference.llama_cpp as llama_cpp_module

    def _snapshot(files):
        snap = tmp_path / f"snap{len(list(tmp_path.iterdir()))}"
        for rel in files:
            path = snap / rel
            path.parent.mkdir(parents = True, exist_ok = True)
            path.write_bytes(b"x")
        return snap

    dspark_only = _snapshot(["dspark/dspark-model-Q8_0.gguf", "model-Q4_K_M.gguf"])
    with_mtp = _snapshot(["mtp-model.gguf", "model-Q4_K_M.gguf"])

    backend = llama_cpp_module.LlamaCppBackend.__new__(llama_cpp_module.LlamaCppBackend)
    snapshots: list[Path] = []
    monkeypatch.setattr(
        "utils.models.model_config._iter_hf_cache_snapshots",
        lambda *a, **k: list(snapshots),
    )

    snapshots[:] = [dspark_only]
    assert backend._cached_repo_mtp_drafter("some/repo") is None

    snapshots[:] = [with_mtp, dspark_only]
    found = backend._cached_repo_mtp_drafter("some/repo")
    assert found is not None and found.endswith("mtp-model.gguf")


def test_cached_dspark_lookup_prefers_q8_and_excludes_dflash(tmp_path, monkeypatch):
    import core.inference.llama_cpp as llama_cpp_module

    snap = tmp_path / "snapshot"
    for rel in (
        "dspark/dspark-model-BF16.gguf",
        "dspark/dspark-model-Q8_0.gguf",
        "dflash-model-Q8_0.gguf",
    ):
        path = snap / rel
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"x")
    monkeypatch.setattr(
        "utils.models.model_config._iter_hf_cache_snapshots",
        lambda *a, **k: [snap],
    )
    backend = llama_cpp_module.LlamaCppBackend.__new__(llama_cpp_module.LlamaCppBackend)

    # `as_posix()`, because the lookup returns an OS-native path: on Windows it
    # comes back with backslashes and the literal below never matched, so this
    # was the one red in an otherwise green cross-platform run.
    assert (
        Path(backend._cached_repo_dspark_drafter("some/repo"))
        .as_posix()
        .endswith("dspark/dspark-model-Q8_0.gguf")
    )


# ── Deletion: only auto-fetched companions are reclaimed ─────────────


def _cache_repo(tmp_path: Path, repo_id: str, names: list[str]):
    """An HF cache repo: snapshot symlinks pointing at blobs."""
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
                # Basename, like huggingface_hub (file_path.name) and our own
                # recovery scan (entry.name); the directory only reaches the
                # predicates via _repo_file_matches' snapshot-relative rebuild.
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


def test_deleting_one_of_several_variants_keeps_the_dspark_drafter(tmp_path):
    """A sibling quant still uses it, so only the last variant may reclaim it."""
    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    repo, snap = _cache_repo(
        tmp_path,
        "unsloth/DeepSeek-V4-Flash-0731-GGUF",
        [
            "model-Q4_K_M.gguf",
            "model-Q8_0.gguf",
            "dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf",
        ],
    )
    _delete_gguf_variant_from_repos(
        "unsloth/DeepSeek-V4-Flash-0731-GGUF", "Q4_K_M", [repo], None, root = tmp_path
    )

    assert not (snap / "model-Q4_K_M.gguf").is_symlink()
    assert (snap / "dspark" / "dspark-DeepSeek-V4-Flash-0731-BF16.gguf").is_symlink()


def test_deleting_the_last_variant_reclaims_an_opt_in_dspark_drafter(tmp_path):
    """Unsloth downloads the sidecar itself once the user opts in, and companion
    filtering keeps it out of the variant menu, so leaving it behind is an
    invisible ~11 GB allocation. Nothing can launch it with no main GGUF left."""
    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    repo, snap = _cache_repo(
        tmp_path,
        "unsloth/DeepSeek-V4-Flash-0731-GGUF",
        ["model-Q4_K_M.gguf", "dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf"],
    )
    _delete_gguf_variant_from_repos(
        "unsloth/DeepSeek-V4-Flash-0731-GGUF", "Q4_K_M", [repo], None, root = tmp_path
    )

    assert not (snap / "model-Q4_K_M.gguf").is_symlink()
    assert not (snap / "dspark" / "dspark-DeepSeek-V4-Flash-0731-BF16.gguf").is_symlink()


def test_a_suffix_scheme_sidecar_is_not_mistaken_for_a_quant(tmp_path):
    """The second published naming scheme, <model>-dspark.gguf. Its basename
    carries no drafter marker, only its dspark/ parent does, so a predicate fed
    the bare file_name read it as a real Q8_0 variant: deleting the genuine Q8_0
    would take the sidecar the Q4_K_M still needs, and no companion could ever be
    reclaimed because a main GGUF always appeared to remain."""
    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    repo, snap = _cache_repo(
        tmp_path,
        "unsloth/DeepSeek-V4-Flash-0731-GGUF",
        [
            "model-Q4_K_M.gguf",
            "model-Q8_0.gguf",
            "dspark/DeepSeek-V4-Flash-0731-Q8_0-dspark.gguf",
        ],
    )
    _delete_gguf_variant_from_repos(
        "unsloth/DeepSeek-V4-Flash-0731-GGUF", "Q8_0", [repo], None, root = tmp_path
    )

    assert not (snap / "model-Q8_0.gguf").is_symlink()
    assert (snap / "model-Q4_K_M.gguf").is_symlink()
    assert (snap / "dspark" / "DeepSeek-V4-Flash-0731-Q8_0-dspark.gguf").is_symlink()

    # ...and it still goes with the last variant.
    _delete_gguf_variant_from_repos(
        "unsloth/DeepSeek-V4-Flash-0731-GGUF", "Q4_K_M", [repo], None, root = tmp_path
    )
    assert not (snap / "dspark" / "DeepSeek-V4-Flash-0731-Q8_0-dspark.gguf").is_symlink()


def test_deleting_the_last_variant_keeps_a_dflash_weight(tmp_path):
    """Negative control: dflash is a family name a user picks for real weights,
    and Unsloth never fetches it as a companion, so it is not reclaimable."""
    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    repo, snap = _cache_repo(
        tmp_path, "org/Model-GGUF", ["model-Q4_K_M.gguf", "dflash-model-Q8_0.gguf"]
    )
    _delete_gguf_variant_from_repos("org/Model-GGUF", "Q4_K_M", [repo], None, root = tmp_path)

    assert not (snap / "model-Q4_K_M.gguf").is_symlink()
    assert (snap / "dflash-model-Q8_0.gguf").is_symlink()


def test_deleting_the_last_variant_still_reclaims_mtp_and_mmproj(tmp_path):
    """Positive control: those ARE fetched with every variant, so they still go."""
    from hub.services.models.deletion import _delete_gguf_variant_from_repos

    repo, snap = _cache_repo(
        tmp_path, "org/Model-GGUF", ["model-Q4_K_M.gguf", "mtp-model.gguf", "mmproj-F16.gguf"]
    )
    _delete_gguf_variant_from_repos("org/Model-GGUF", "Q4_K_M", [repo], None, root = tmp_path)

    assert not (snap / "model-Q4_K_M.gguf").is_symlink()
    assert not (snap / "mtp-model.gguf").is_symlink()
    assert not (snap / "mmproj-F16.gguf").is_symlink()


# ── DFlash: predicate, discovery, capability gate and emission ───────
#
# DFlash is the third separate-file drafter kind. It differs from DSpark in two
# ways that these tests pin:
#   * it is ON under Auto rather than opt-in, because the published sidecar is
#     ~1.5 GiB and ships in the model's own GGUF repo (DSpark's is ~11 GB);
#   * its published filename (``dflash-kquant.gguf``) names no model family, so
#     discovery confirms the header's ``general.architecture`` instead of
#     pairing on the filename.


DFLASH_PREDICATE_CASES = [
    # The published sidecar, and the family-named scheme ggml-org uses.
    ("dflash-kquant.gguf", True),
    ("dflash-Qwen3.6-27B-BF16.gguf", True),
    ("dflash-draft-3.6-q8_0.gguf", True),
    ("DFLASH-Qwen3.6-27B-Q8_0.gguf", True),
    # Adversarial: dflash is also a family a publisher puts on real weights.
    ("Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf", False),
    ("qwen35-4b-dflash-Q8_0.gguf", False),
    ("laguna-s-2.1-dflash-Q4_K_M.gguf", False),
    # A user's own dflash/ folder holds whatever they downloaded, so unlike
    # dspark/ and MTP/ the DIRECTORY is not a drafter marker (_DRAFTER_DIR_KINDS).
    ("dflash/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf", False),
    ("foo/dflash/bar.gguf", False),
    # The other kinds must not leak into this one: each needs its own --spec-type.
    ("dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf", False),
    ("mtp-gemma-4-12b-it.gguf", False),
    ("dflash-notes.txt", False),
]


@pytest.mark.parametrize("path,expected", DFLASH_PREDICATE_CASES)
def test_is_dflash_drafter_path(path, expected):
    from core.inference.llama_cpp import (
        _is_dflash_drafter_path,
        _is_dspark_drafter_path,
        _is_mtp_only_drafter_path,
    )
    assert _is_dflash_drafter_path(path) is expected
    if expected:
        # The three kinds partition: a DFlash sidecar launched as MTP or DSpark
        # would get a --spec-type its architecture cannot serve.
        assert _is_dspark_drafter_path(path) is False
        assert _is_mtp_only_drafter_path(path) is False


@pytest.mark.parametrize(
    "value,expected",
    [
        ("dflash", "dflash"),
        ("DFlash", "dflash"),
        ("  dflash ", "dflash"),
        ("draft-dflash", "dflash"),
        ("DRAFT-DFLASH", "dflash"),
    ],
)
def test_canonicalize_spec_mode_accepts_dflash(value, expected):
    from core.inference.llama_cpp import _canonicalize_spec_mode
    assert _canonicalize_spec_mode(value) == expected


# ── Capability probe ─────────────────────────────────────────────────

_NEEDS_BASH = pytest.mark.skipif(
    sys.platform == "win32",
    reason = "fake llama-server is a bash stub; Windows has no direct executor",
)


def _fake_llama_server(path: Path, help_text: str) -> Path:
    path.write_text(f"#!/usr/bin/env bash\ncat <<'EOF'\n{help_text}\nEOF\n")
    path.chmod(0o755)
    return path


@_NEEDS_BASH
@pytest.mark.parametrize(
    "spec_line,expected",
    [
        ("--spec-type none,draft-mtp,draft-dflash,draft-dspark,ngram-mod", True),
        # A published prebuilt that predates the arch: emitting draft-dflash
        # would abort the launch instead of falling back.
        ("--spec-type none,draft-mtp,draft-dspark,ngram-mod", False),
        # Word boundaries, so a longer token cannot be read as support.
        ("--spec-type none,draft-dflash2,ngram-mod", False),
        ("--spec-type none,xdraft-dflash,ngram-mod", False),
    ],
)
def test_probe_server_capabilities_reports_dflash(tmp_path, spec_line, expected):
    from core.inference.llama_cpp import LlamaCppBackend

    fake = _fake_llama_server(tmp_path / "llama-server", spec_line)
    LlamaCppBackend._capability_cache.clear()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["supports_dflash"] is expected
    # DSpark's answer is read from the same block and must not move.
    assert caps["supports_dspark"] is ("draft-dspark" in spec_line)


def test_missing_binary_reports_no_dflash():
    """The not-found dict is returned before any parsing, so it has to carry the
    key: a caller reading it with .get() would otherwise treat "absent" as False
    only by luck, and _estimate_gguf_required_gb default-denies on it."""
    from core.inference.llama_cpp import LlamaCppBackend

    caps = LlamaCppBackend.probe_server_capabilities("/nonexistent/llama-server")
    assert caps["found"] is False
    assert caps["supports_dflash"] is False


# ── Emission ─────────────────────────────────────────────────────────


def _spec_backend(
    monkeypatch,
    *,
    supports_dflash = True,
    supports_dspark = True,
):
    from core.inference.llama_cpp import LlamaCppBackend

    caps = {
        "found": True,
        "mtp_token": "draft-mtp",
        "supports_mtp": True,
        "supports_dspark": supports_dspark,
        "supports_dflash": supports_dflash,
        "mtp_probe_inconclusive": False,
        "ngram_mod_flavor": "new",
        "supports_ngram_mod": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: caps),
    )
    backend = LlamaCppBackend()
    backend._nextn_predict_layers = None
    return backend


def _spec_flags(backend, **kwargs):
    base = dict(
        speculative_type = None,
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "unsloth/Muse-Glimmer-30B-GGUF",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    base.update(kwargs)
    return backend._build_speculative_flags(**base)


def test_auto_launches_dflash_when_a_sidecar_is_present(monkeypatch):
    """The headline behaviour: unlike DSpark, DFlash needs no opt-in. On a
    B200 with the published 1.52 GiB sidecar this is 1.21x-1.36x decode over
    spec-off at n_max=2, at 61-77% draft acceptance."""
    backend = _spec_backend(monkeypatch)
    flags = _spec_flags(backend, speculative_type = "auto", dflash_draft_path = "/m/dflash-kquant.gguf")

    assert flags == [
        "--model-draft",
        "/m/dflash-kquant.gguf",
        "--spec-type",
        "draft-dflash",
        "--spec-draft-n-max",
        "2",
    ]
    assert backend._speculative_type == "draft-dflash"
    assert backend._spec_drafter_kind == "dflash"
    assert backend._spec_fallback_reason is None


def test_auto_uses_the_cpu_draft_depth_off_gpu(monkeypatch):
    backend = _spec_backend(monkeypatch)
    flags = _spec_flags(backend, speculative_type = "auto", dflash_draft_path = "/m/d.gguf", gpus = False)
    assert flags[-2:] == ["--spec-draft-n-max", "3"]


def test_a_user_draft_depth_override_reaches_dflash(monkeypatch):
    backend = _spec_backend(monkeypatch)
    flags = _spec_flags(
        backend, speculative_type = "dflash", dflash_draft_path = "/m/d.gguf", spec_draft_n_max = 6
    )
    assert flags[-2:] == ["--spec-draft-n-max", "6"]
    assert backend._spec_draft_n_max == 6


def test_auto_does_not_emit_dflash_when_the_binary_cannot_run_it(monkeypatch):
    """A --spec-type the binary does not know aborts the launch, so the sidecar
    being on disk is not enough. Published prebuilts predate the arch."""
    backend = _spec_backend(monkeypatch, supports_dflash = False)
    flags = _spec_flags(backend, speculative_type = "auto", dflash_draft_path = "/m/dflash-kquant.gguf")

    assert "draft-dflash" not in flags
    assert "--model-draft" not in flags
    assert flags == ["--spec-default"]
    assert backend._speculative_type == "default"


def test_forced_dflash_without_the_capability_falls_back(monkeypatch):
    backend = _spec_backend(monkeypatch, supports_dflash = False)
    flags = _spec_flags(backend, speculative_type = "dflash", dflash_draft_path = "/m/d.gguf")

    assert flags == ["--spec-default"]
    assert backend._speculative_type == "default"
    assert backend._spec_fallback_reason == "binary_no_mtp"


def test_forced_dflash_without_a_sidecar_falls_back(monkeypatch):
    backend = _spec_backend(monkeypatch)
    flags = _spec_flags(backend, speculative_type = "dflash", dflash_draft_path = None)

    assert flags == ["--spec-default"]
    assert backend._spec_fallback_reason == "drafter_not_found"
    assert backend._spec_drafter_kind == "dflash"


def test_dspark_keeps_first_refusal_when_a_repo_ships_both(monkeypatch):
    """Mirrors llama.cpp's own downloader, which ranks dspark ahead of dflash.
    In practice a repo ships one kind or neither; this pins that adding DFlash
    did not reorder the existing choice."""
    backend = _spec_backend(monkeypatch)
    flags = _spec_flags(
        backend,
        speculative_type = "auto",
        dspark_draft_path = "/m/dspark-x-Q8_0.gguf",
        dflash_draft_path = "/m/dflash-kquant.gguf",
    )
    assert "draft-dspark" in flags
    assert "draft-dflash" not in flags
    assert backend._spec_drafter_kind == "dspark"


def test_dspark_emission_is_unchanged(monkeypatch):
    """Regression guard on the behaviour this PR must not touch."""
    backend = _spec_backend(monkeypatch)
    flags = _spec_flags(
        backend, speculative_type = "dspark", dspark_draft_path = "/m/dspark-x-Q8_0.gguf"
    )
    assert flags == [
        "--model-draft",
        "/m/dspark-x-Q8_0.gguf",
        "--spec-type",
        "draft-dspark",
        "--spec-draft-n-max",
        "3",
    ]
    assert backend._speculative_type == "draft-dspark"


def test_a_dflash_sidecar_alone_does_not_change_the_mtp_or_off_paths(monkeypatch):
    """Forced modes stay forced: a sidecar sitting on disk must not promote."""
    backend = _spec_backend(monkeypatch)
    assert _spec_flags(backend, speculative_type = "off", dflash_draft_path = "/m/d.gguf") == []
    flags = _spec_flags(backend, speculative_type = "ngram", dflash_draft_path = "/m/d.gguf")
    assert "draft-dflash" not in flags
    assert flags[:2] == ["--spec-type", "ngram-mod"]


# ── Local discovery ──────────────────────────────────────────────────


def _write_gguf(path: Path, architecture: str) -> Path:
    """A real GGUF header carrying one general.architecture string, which is
    what detect_dflash_file confirms."""
    import struct

    key = b"general.architecture"
    value = architecture.encode()
    blob = struct.pack("<IIQQ", 0x46554747, 3, 0, 1)
    blob += struct.pack("<Q", len(key)) + key
    blob += struct.pack("<I", 8) + struct.pack("<Q", len(value)) + value
    path.write_bytes(blob)
    return path


def test_detect_dflash_file_finds_the_unpaired_published_sidecar(tmp_path):
    """The published sidecar is dflash-kquant.gguf, which names no model family,
    so the filename pairing DSpark uses would reject the one file this exists to
    find. The header settles it instead."""
    weight = _write_gguf(tmp_path / "Muse-Glimmer-30B-UD-Q4_K_XL.gguf", "muse-glimmer")
    sidecar = _write_gguf(tmp_path / "dflash-kquant.gguf", "dflash")

    assert detect_dflash_file(str(weight)) == str(sidecar.resolve())


def test_detect_dflash_file_rejects_a_real_model_that_is_merely_named_dflash(tmp_path):
    """Two layers have to hold: the prefix rule (this file does not start with
    dflash-) and the header check behind it."""
    weight = _write_gguf(tmp_path / "Qwen3.6-27B-Q4_K_M.gguf", "qwen3")
    _write_gguf(tmp_path / "Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf", "qwen3moe")

    assert detect_dflash_file(str(weight)) is None


def test_detect_dflash_file_rejects_a_dflash_prefixed_file_of_another_architecture(tmp_path):
    """The prefix alone is not enough: someone may name real weights that way,
    and --model-draft on a full model is a silent 15 GB allocation."""
    weight = _write_gguf(tmp_path / "model-Q4_K_M.gguf", "llama")
    _write_gguf(tmp_path / "dflash-something-Q8_0.gguf", "llama")

    assert detect_dflash_file(str(weight)) is None


def test_detect_dflash_file_ignores_a_dflash_directory(tmp_path):
    """dflash/ is a folder name a user picks for the family they downloaded, so
    unlike dspark/ it is never scanned. Reaching in would hand llama-server a
    real weight as --model-draft."""
    weight = _write_gguf(tmp_path / "model-Q4_K_M.gguf", "llama")
    sub = tmp_path / "dflash"
    sub.mkdir()
    _write_gguf(sub / "dflash-kquant.gguf", "dflash")

    assert detect_dflash_file(str(weight)) is None


def test_detect_dflash_file_prefers_the_sidecar_that_names_this_weight(tmp_path):
    """Both schemes are published. In a multi-model folder the family-named one
    wins for the weight it names, so a foreign sidecar cannot attach first."""
    weight = _write_gguf(tmp_path / "Qwen3.6-27B-Q4_K_M.gguf", "qwen3")
    _write_gguf(tmp_path / "dflash-kquant.gguf", "dflash")
    paired = _write_gguf(tmp_path / "dflash-Qwen3.6-27B-Q8_0.gguf", "dflash")

    assert detect_dflash_file(str(weight)) == str(paired.resolve())


@pytest.mark.parametrize("shape", ["dangling", "directory"])
def test_detect_dflash_file_skips_a_sidecar_it_cannot_open(tmp_path, shape):
    """Same guard as detect_dspark_file: a --model-draft llama-server cannot open
    fails the whole load rather than falling back to no speculation."""
    import os

    weight = _write_gguf(tmp_path / "model-Q4_K_M.gguf", "llama")
    sidecar = tmp_path / "dflash-kquant.gguf"
    if shape == "dangling":
        os.symlink(tmp_path / "missing_blob", sidecar)
    else:
        sidecar.mkdir()

    assert detect_dflash_file(str(weight)) is None


def test_model_config_reports_a_local_dflash_sidecar(tmp_path):
    """The field the load intent and the VRAM guard both read."""
    _write_gguf(tmp_path / "Muse-Glimmer-30B-UD-Q4_K_XL.gguf", "muse-glimmer")
    sidecar = _write_gguf(tmp_path / "dflash-kquant.gguf", "dflash")

    config = ModelConfig.from_identifier(str(tmp_path / "Muse-Glimmer-30B-UD-Q4_K_XL.gguf"))
    assert config.is_gguf is True
    assert config.gguf_dflash_file == str(sidecar.resolve())
    assert config.gguf_dspark_file is None


# ── Download gating ──────────────────────────────────────────────────


def _dflash_download_probe(
    tmp_path,
    monkeypatch,
    *,
    supports_dflash,
    cached = None,
):
    import core.inference.llama_cpp as llama_cpp_module
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: {"supports_dflash": supports_dflash}),
    )
    monkeypatch.setattr(
        llama_cpp_module, "_companion_snapshot_sibling", lambda near_path, pick: cached
    )
    reached = {}

    def _fake_companion(
        *,
        hf_repo,
        hf_token,
        pick,
        label,
        cancel_event = None,
        near_path = None,
        outcome = None,
        on_transient_failure = None,
    ):
        reached["hit"] = True
        reached["picked"] = pick(
            [
                "Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
                "mmproj-Muse-Glimmer-30B-Q8_0.gguf",
                "dflash-kquant.gguf",
            ]
        )
        if outcome is not None:
            outcome["listed"] = True
        # A real file: the fetch is only accepted once its header says dflash.
        return str(_write_gguf(tmp_path / "dflash-kquant.gguf", "dflash"))

    b = LlamaCppBackend()
    b._download_companion_gguf = _fake_companion
    got = b._download_dflash(
        hf_repo = "unsloth/Muse-Glimmer-30B-GGUF",
        near_path = "/cache/snap/Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
        binary = "/fake/llama-server",
    )
    return got, reached


def test_download_dflash_fetches_when_the_binary_supports_it(tmp_path, monkeypatch):
    got, reached = _dflash_download_probe(tmp_path, monkeypatch, supports_dflash = True)
    assert got == str(tmp_path / "dflash-kquant.gguf")
    assert reached["hit"] is True
    # The picker must select the sidecar, not the weight or the projector.
    assert reached["picked"] == "dflash-kquant.gguf"


def test_download_dflash_skips_the_fetch_when_the_binary_cannot_run_it(tmp_path, monkeypatch):
    """Same gate as DSpark: _build_speculative_flags drops DFlash outright on a
    binary without --spec-type draft-dflash, so the file would never be opened."""
    got, reached = _dflash_download_probe(tmp_path, monkeypatch, supports_dflash = False)
    assert got is None
    assert reached.get("hit", False) is False


def test_download_dflash_still_reports_a_cached_sidecar_it_cannot_run(tmp_path, monkeypatch):
    """The route rediscovers it on every Apply, so answering None would compare
    it against a launched None and reload the same server each time.

    A real file on disk, since the reuse now confirms the header says dflash
    before handing the path back."""
    cached = str(_write_gguf(tmp_path / "dflash-kquant.gguf", "dflash"))
    got, reached = _dflash_download_probe(
        tmp_path, monkeypatch, supports_dflash = False, cached = cached
    )
    assert got == cached
    assert reached.get("hit", False) is False


def test_download_dflash_records_whether_the_repo_publishes_a_sidecar(monkeypatch):
    """Most repos publish none, and retrying that on every Apply would relaunch
    an identical server forever; a failed fetch must still be retried."""
    import core.inference.llama_cpp as llama_cpp_module
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: {"supports_dflash": True}),
    )
    monkeypatch.setattr(
        llama_cpp_module, "_companion_snapshot_sibling", lambda near_path, pick: None
    )

    def _companion(listed):
        def _fake(
            *,
            hf_repo,
            hf_token,
            pick,
            label,
            cancel_event = None,
            near_path = None,
            outcome = None,
            on_transient_failure = None,
        ):
            if outcome is not None:
                outcome["listed"] = listed
            return None

        return _fake

    for listed, expect_absent in ((False, True), (True, False)):
        b = LlamaCppBackend()
        b._download_companion_gguf = _companion(listed)
        assert b._download_dflash(hf_repo = "org/repo", binary = "/fake/llama-server") is None
        assert b._dflash_sidecar_absent is expect_absent


def test_a_cached_dflash_drafter_is_never_launched_as_an_mtp_drafter(tmp_path, monkeypatch):
    """Each kind needs its own --spec-type, so the offline MTP reuse scan must
    not pick up a DFlash sidecar sitting in the same snapshot."""
    from core.inference.llama_cpp import LlamaCppBackend

    snap = tmp_path / "snapshots" / "abc"
    snap.mkdir(parents = True)
    # Real headers: the cached lookup confirms general.architecture before it
    # hands a path to --model-draft.
    for name in ("dflash-kquant.gguf", "model-Q4_K_M.gguf"):
        _write_gguf(snap / name, "dflash" if name.startswith("dflash-") else "llama")
    monkeypatch.setattr(
        "utils.models.model_config._iter_hf_cache_snapshots", lambda *a, **k: [snap]
    )

    b = LlamaCppBackend()
    assert b._cached_repo_mtp_drafter("org/repo") is None
    assert b._cached_repo_dspark_drafter("org/repo") is None
    assert b._cached_repo_dflash_drafter("org/repo") == str(snap / "dflash-kquant.gguf")


# ── Remote sidecars pair with the selected weight ────────────────────
#
# detect_dflash_file already refuses a sidecar named after a NEIGHBOURING
# weight, so a multi-family folder cannot attach a foreign drafter locally. The
# download and the offline cache reuse ranked by precision and name alone, so
# dflash-model-A-Q8_0.gguf beat the generic dflash-kquant.gguf and model B was
# launched with model A's drafter. All three paths now share
# dflash_repo_preference_key.

_MULTI_FAMILY_LISTING = [
    "model-A-Q4_K_M.gguf",
    "model-B-Q4_K_M.gguf",
    "dflash-model-A-Q8_0.gguf",
    "dflash-kquant.gguf",
]


def _dflash_download_pick(monkeypatch, *, listing, near_path):
    """The sidecar _download_dflash's picker chooses out of a repo listing."""
    import core.inference.llama_cpp as llama_cpp_module
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: {"supports_dflash": True}),
    )
    monkeypatch.setattr(
        llama_cpp_module, "_companion_snapshot_sibling", lambda near_path, pick: None
    )
    picked = {}

    def _fake_companion(
        *,
        hf_repo,
        hf_token,
        pick,
        label,
        cancel_event = None,
        near_path = None,
        outcome = None,
        on_transient_failure = None,
    ):
        picked["name"] = pick(listing)
        return None

    b = LlamaCppBackend()
    b._download_companion_gguf = _fake_companion
    b._download_dflash(
        hf_repo = "org/repo",
        near_path = near_path,
        binary = "/fake/llama-server",
    )
    return picked.get("name")


def test_download_dflash_skips_a_sidecar_named_after_another_weight(monkeypatch):
    """Model B must get the generic sidecar, not the higher-precision one that
    names model A."""
    assert (
        _dflash_download_pick(
            monkeypatch,
            listing = _MULTI_FAMILY_LISTING,
            near_path = "/cache/snap/model-B-Q4_K_M.gguf",
        )
        == "dflash-kquant.gguf"
    )


def test_download_dflash_takes_the_sidecar_naming_this_weight(monkeypatch):
    """The other direction: model A's own sidecar still wins over the generic
    one, as it does in detect_dflash_file."""
    assert (
        _dflash_download_pick(
            monkeypatch,
            listing = _MULTI_FAMILY_LISTING,
            near_path = "/cache/snap/model-A-Q4_K_M.gguf",
        )
        == "dflash-model-A-Q8_0.gguf"
    )


@pytest.mark.parametrize("sidecar", ["dflash-kquant.gguf", "dflash-bf16.gguf"])
def test_download_dflash_keeps_the_single_published_sidecar(monkeypatch, sidecar):
    """The shipped unsloth/Muse-Glimmer-30B-GGUF layout. Its sidecar's stem is a
    precision token, not a family, so nothing may treat "names no weight here"
    as a rejection."""
    assert (
        _dflash_download_pick(
            monkeypatch,
            listing = [
                "Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
                "mmproj-Muse-Glimmer-30B-Q8_0.gguf",
                sidecar,
            ],
            near_path = "/cache/snap/Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
        )
        == sidecar
    )


def test_cached_dflash_lookup_pairs_with_the_selected_weight(tmp_path, monkeypatch):
    """The offline reuse must reach the same file the download would have
    fetched, or a reload swaps drafters as soon as the cache is warm."""
    from core.inference.llama_cpp import LlamaCppBackend

    snap = tmp_path / "snapshots" / "abc"
    snap.mkdir(parents = True)
    for name in _MULTI_FAMILY_LISTING:
        _write_gguf(snap / name, "dflash" if name.startswith("dflash-") else "llama")
    monkeypatch.setattr(
        "utils.models.model_config._iter_hf_cache_snapshots", lambda *a, **k: [snap]
    )

    b = LlamaCppBackend()
    for weight, expected in (
        ("model-B-Q4_K_M.gguf", "dflash-kquant.gguf"),
        ("model-A-Q4_K_M.gguf", "dflash-model-A-Q8_0.gguf"),
    ):
        assert b._cached_repo_dflash_drafter("org/repo", near_path = str(snap / weight)) == str(
            snap / expected
        )
    # No weight in hand: precision order, exactly as before.
    assert b._cached_repo_dflash_drafter("org/repo") == str(snap / "dflash-model-A-Q8_0.gguf")


def test_cached_dflash_lookup_keeps_the_single_published_sidecar(tmp_path, monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend

    snap = tmp_path / "snapshots" / "abc"
    snap.mkdir(parents = True)
    for name in ("Muse-Glimmer-30B-UD-Q4_K_XL.gguf", "dflash-kquant.gguf"):
        _write_gguf(snap / name, "dflash" if name.startswith("dflash-") else "muse-glimmer")
    monkeypatch.setattr(
        "utils.models.model_config._iter_hf_cache_snapshots", lambda *a, **k: [snap]
    )

    b = LlamaCppBackend()
    assert b._cached_repo_dflash_drafter(
        "org/repo", near_path = str(snap / "Muse-Glimmer-30B-UD-Q4_K_XL.gguf")
    ) == str(snap / "dflash-kquant.gguf")


def test_local_and_remote_dflash_pairing_agree(tmp_path):
    """One rule, three call sites: the local scan, the download picker and the
    cache lookup all go through dflash_repo_preference_key /
    _drafter_names_other_weight."""
    from utils.models.drafters import dflash_repo_preference_key

    others = ["model-A-Q4_K_M.gguf", "model-B-Q4_K_M.gguf"]
    ranked = sorted(
        ("dflash-model-A-Q8_0.gguf", "dflash-kquant.gguf"),
        key = lambda name: dflash_repo_preference_key(name, "model-B-Q4_K_M.gguf", others),
    )
    assert ranked[0] == "dflash-kquant.gguf"

    for name in _MULTI_FAMILY_LISTING:
        _write_gguf(tmp_path / name, "dflash" if name.startswith("dflash-") else "llama")
    assert detect_dflash_file(str(tmp_path / "model-B-Q4_K_M.gguf")) == str(
        (tmp_path / "dflash-kquant.gguf").resolve()
    )


# ── Reclaim: deliberately unchanged ──────────────────────────────────


def test_dflash_stays_unreclaimable_even_though_auto_now_launches_it(tmp_path):
    """Auto downloading a sidecar does not make the name safe to delete by.
    Reclaiming wrongly destroys weights a user chose (whole repos publish
    nothing but root-level dflash-*.gguf), while not reclaiming leaves ~1.5 GiB,
    an order of magnitude under the ~11 GB DSpark case the rule was written for.
    The positive control below shows the reclaim itself still works."""
    from hub.services.models.deletion import _delete_gguf_variant_from_repos
    from hub.utils.gguf import is_reclaimable_drafter_path

    assert is_reclaimable_drafter_path("dflash-kquant.gguf") is False
    assert is_reclaimable_drafter_path("dspark-model-Q8_0.gguf") is True

    repo, snap = _cache_repo(
        tmp_path,
        "org/Model-GGUF",
        ["model-Q4_K_M.gguf", "dflash-kquant.gguf", "dspark-model-Q8_0.gguf"],
    )
    _delete_gguf_variant_from_repos("org/Model-GGUF", "Q4_K_M", [repo], None, root = tmp_path)

    assert not (snap / "model-Q4_K_M.gguf").is_symlink()
    assert (snap / "dflash-kquant.gguf").is_symlink()
    assert not (snap / "dspark-model-Q8_0.gguf").is_symlink()


def test_detect_dflash_file_skips_a_sidecar_named_for_another_weight(tmp_path):
    """A multi-model folder must not attach a foreign drafter.

    _drafter_matches_weight is False both for a sidecar naming no family and for
    one naming a DIFFERENT family, so ranking alone bucketed them together and
    precision could float the foreign one to the top: loading model B beside
    dflash-model-A-Q8_0.gguf and the generic dflash-kquant.gguf launched model
    A's drafter for model B. Both files carry a real dflash header, so the
    architecture check behind the ranking cannot catch this one.
    """
    weight = _write_gguf(tmp_path / "Muse-Glimmer-30B-UD-Q4_K_XL.gguf", "muse-glimmer")
    _write_gguf(tmp_path / "Qwen3.6-27B-Q4_K_M.gguf", "qwen3")
    foreign = _write_gguf(tmp_path / "dflash-Qwen3.6-27B-Q8_0.gguf", "dflash")
    generic = _write_gguf(tmp_path / "dflash-kquant.gguf", "dflash")

    assert foreign.exists()
    assert detect_dflash_file(str(weight)) == str(generic.resolve())


def test_detect_dflash_file_still_prefers_a_sidecar_that_names_this_weight(tmp_path):
    """The skip above must not cost the paired case: a sidecar naming THIS
    weight's family still wins over the generic one."""
    weight = _write_gguf(tmp_path / "Muse-Glimmer-30B-UD-Q4_K_XL.gguf", "muse-glimmer")
    paired = _write_gguf(tmp_path / "dflash-Muse-Glimmer-30B-Q8_0.gguf", "dflash")
    _write_gguf(tmp_path / "dflash-kquant.gguf", "dflash")

    assert detect_dflash_file(str(weight)) == str(paired.resolve())


def test_detect_dflash_file_ignores_the_suffix_form_the_picker_cannot_hide(tmp_path):
    """Discovery and the quant picker have to agree on what a sidecar is.

    The shared companion predicates know DFlash by the dflash- prefix only, so a
    <model>-dflash.gguf accepted here would be a drafter for discovery and at the
    same time a selectable Q8_0 main model in the picker, and choosing that
    variant would hand llama-server the drafter as the target. Detection gives
    the form up rather than teaching the predicate a suffix that would hide a
    real model merely named DFlash.
    """
    from core.inference.llama_cpp import _is_companion_gguf_path

    weight = _write_gguf(tmp_path / "Muse-Glimmer-30B-UD-Q4_K_XL.gguf", "muse-glimmer")
    suffix_form = tmp_path / "Muse-Glimmer-30B-Q8_0-dflash.gguf"
    _write_gguf(suffix_form, "dflash")

    assert detect_dflash_file(str(weight)) is None
    # The invariant behind the choice: what discovery accepts, the picker hides.
    assert _is_companion_gguf_path(suffix_form.name) is False
    assert _is_companion_gguf_path("dflash-kquant.gguf") is True


def test_dflash_prefix_form_is_still_found_beside_the_weight(tmp_path):
    """Dropping the suffix form must not cost the published sidecar."""
    weight = _write_gguf(tmp_path / "Muse-Glimmer-30B-UD-Q4_K_XL.gguf", "muse-glimmer")
    sidecar = _write_gguf(tmp_path / "dflash-kquant.gguf", "dflash")

    assert detect_dflash_file(str(weight)) == str(sidecar.resolve())


def test_detect_dflash_file_validates_a_candidate_before_reading_its_header(tmp_path, monkeypatch):
    """A native grant answers through ``accept``, and its answer has to arrive
    before the file is opened.

    A dflash-*.gguf inside a leased directory can be a symlink whose target sits
    outside the lease. Parsing the header first opened that target, and no later
    rejection takes a read back, so the order is: resolve, ask accept, then read.
    """
    import os

    import utils.models.model_config as mc

    leased = tmp_path / "leased"
    leased.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    weight = _write_gguf(leased / "Muse-Glimmer-30B-UD-Q4_K_XL.gguf", "muse-glimmer")
    target = _write_gguf(outside / "dflash-kquant.gguf", "dflash")
    os.symlink(target, leased / "dflash-kquant.gguf")

    reads: list[str] = []
    real_read = mc.read_gguf_general_metadata

    def _recording_read(path, *args, **kwargs):
        reads.append(str(path))
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(mc, "read_gguf_general_metadata", _recording_read)

    def _inside_the_lease(launch: str) -> bool:
        # accept is handed the resolved launch path, not the candidate.
        return leased in Path(launch).parents

    assert detect_dflash_file(str(weight), accept = _inside_the_lease) is None
    assert reads == []  # the out-of-grant target was never opened


def test_detect_dflash_file_still_checks_the_header_of_an_accepted_candidate(tmp_path):
    """The reorder must not cost the architecture check every other caller relies
    on: an accepted candidate that is not a dflash model is still dropped."""
    weight = _write_gguf(tmp_path / "model-Q4_K_M.gguf", "llama")
    _write_gguf(tmp_path / "dflash-something-Q8_0.gguf", "llama")

    assert detect_dflash_file(str(weight), accept = lambda launch: True) is None


# ── Remote candidates are validated by header, not by name ───────────
#
# _is_dflash_drafter_path is a dflash- FILENAME test, and deliberately only the
# prefix form (widening it would let one file be both a drafter and a selectable
# main model in the quant picker). detect_dflash_file backs that name test with
# the architecture in the GGUF header; the download and the cache reuse did not,
# so a repo holding an ordinary weight called dflash-*.gguf had it downloaded in
# full and handed to llama-server as --model-draft, which falls back at startup
# after the bytes are already spent. Same helper on every path.


def _dflash_repo_download(
    tmp_path,
    monkeypatch,
    *,
    listing,
    sibling = None,
):
    """Drive _download_dflash over a repo listing whose files exist in tmp_path."""
    import core.inference.llama_cpp as llama_cpp_module
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: {"supports_dflash": True}),
    )
    monkeypatch.setattr(
        llama_cpp_module, "_companion_snapshot_sibling", lambda near_path, pick: sibling
    )
    fetched: list[str] = []

    def _fake_companion(
        *,
        hf_repo,
        hf_token,
        pick,
        label,
        cancel_event = None,
        near_path = None,
        outcome = None,
        on_transient_failure = None,
    ):
        target = pick(listing)
        if outcome is not None:
            outcome["listed"] = target is not None
        if target is None:
            return None
        fetched.append(target)
        return str(tmp_path / target)

    b = LlamaCppBackend()
    b._download_companion_gguf = _fake_companion
    got = b._download_dflash(
        hf_repo = "org/repo",
        near_path = str(tmp_path / "model-Q4_K_M.gguf"),
        binary = "/fake/llama-server",
    )
    return b, got, fetched


def test_download_dflash_falls_through_a_candidate_that_is_not_a_dflash_model(
    tmp_path, monkeypatch
):
    """The impostor outranks the real sidecar on both name rules (it pairs with
    this weight, and Q8_0 beats an unmarked precision), so the fetch reaches it
    first. Its header is what disqualifies it, and only after the fetch, so the
    search has to move on to the next candidate instead of returning None."""
    _write_gguf(tmp_path / "model-Q4_K_M.gguf", "llama")
    _write_gguf(tmp_path / "dflash-model-Q8_0.gguf", "llama")
    sidecar = _write_gguf(tmp_path / "dflash-kquant.gguf", "dflash")

    b, got, fetched = _dflash_repo_download(
        tmp_path,
        monkeypatch,
        listing = ["model-Q4_K_M.gguf", "dflash-model-Q8_0.gguf", "dflash-kquant.gguf"],
    )

    assert got == str(sidecar)
    assert fetched == ["dflash-model-Q8_0.gguf", "dflash-kquant.gguf"]
    assert b._dflash_sidecar_absent is False


def test_download_dflash_reports_no_sidecar_when_every_candidate_is_a_weight(tmp_path, monkeypatch):
    """A repo whose only dflash-*.gguf is an ordinary model publishes no sidecar,
    so the absence is recorded and the next Apply does not re-list forever."""
    _write_gguf(tmp_path / "model-Q4_K_M.gguf", "llama")
    _write_gguf(tmp_path / "dflash-model-Q8_0.gguf", "llama")

    b, got, fetched = _dflash_repo_download(
        tmp_path,
        monkeypatch,
        listing = ["model-Q4_K_M.gguf", "dflash-model-Q8_0.gguf"],
    )

    assert got is None
    assert fetched == ["dflash-model-Q8_0.gguf"]  # tried once, never re-picked
    assert b._dflash_sidecar_absent is True


def test_download_dflash_validates_the_snapshot_sibling_it_reuses(tmp_path, monkeypatch):
    """The reuse path never downloads anything, but it hands the same file to
    --model-draft, so it applies the same header rule and keeps scanning."""
    from core.inference.llama_cpp import LlamaCppBackend

    snap = tmp_path / "models--org--repo" / "snapshots" / "abc"
    snap.mkdir(parents = True)
    _write_gguf(snap / "model-Q4_K_M.gguf", "llama")
    _write_gguf(snap / "dflash-model-Q8_0.gguf", "llama")
    sidecar = _write_gguf(snap / "dflash-kquant.gguf", "dflash")

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: {"supports_dflash": True}),
    )

    b = LlamaCppBackend()
    b._download_companion_gguf = lambda **kwargs: pytest.fail("the reuse must not download")
    assert b._download_dflash(
        hf_repo = "org/repo",
        near_path = str(snap / "model-Q4_K_M.gguf"),
        binary = "/fake/llama-server",
    ) == str(sidecar)


def test_cached_dflash_lookup_skips_a_prefixed_file_of_another_architecture(tmp_path, monkeypatch):
    """The offline cache lookup ranks by name too, so a cached weight named like
    a sidecar would be launched as the drafter with nothing left to catch it."""
    from core.inference.llama_cpp import LlamaCppBackend

    snap = tmp_path / "snapshots" / "abc"
    snap.mkdir(parents = True)
    _write_gguf(snap / "model-Q4_K_M.gguf", "llama")
    _write_gguf(snap / "dflash-model-Q8_0.gguf", "llama")
    _write_gguf(snap / "dflash-kquant.gguf", "dflash")
    monkeypatch.setattr(
        "utils.models.model_config._iter_hf_cache_snapshots", lambda *a, **k: [snap]
    )

    b = LlamaCppBackend()
    assert b._cached_repo_dflash_drafter(
        "org/repo", near_path = str(snap / "model-Q4_K_M.gguf")
    ) == str(snap / "dflash-kquant.gguf")


def test_local_and_remote_dflash_architecture_checks_agree(tmp_path):
    """One rule, one place: detect_dflash_file and the remote paths both ask
    is_dflash_architecture, so neither can start trusting the name alone."""
    from utils.models.drafters import is_dflash_architecture

    weight = _write_gguf(tmp_path / "model-Q4_K_M.gguf", "llama")
    impostor = _write_gguf(tmp_path / "dflash-model-Q8_0.gguf", "llama")
    sidecar = _write_gguf(tmp_path / "dflash-kquant.gguf", "dflash")

    assert is_dflash_architecture(str(impostor)) is False
    assert is_dflash_architecture(str(sidecar)) is True
    assert is_dflash_architecture(str(tmp_path / "missing.gguf")) is False
    assert detect_dflash_file(str(weight)) == str(sidecar.resolve())


# ── Auto only stands down on DFlash for a DSpark it can launch ───────
#
# _download_dspark reports an already-cached sidecar even when the binary has no
# usable --spec-type draft-dspark (so the route's reuse check does not reload the
# same server on every Apply), and the promotion refuses that path. The DFlash
# fetch read the bare path as "DSpark won" and stood down, so a repo shipping
# both companions left a DFlash-capable binary with NO drafter at all.


class _StopAfterDownloads(Exception):
    """Ends the load once Phase 2 is done, which is all these tests observe."""


def _dflash_fetch_during_auto_load(monkeypatch, *, supports_dspark, supports_dflash, dspark_cached):
    """Whether an Auto load fetches the DFlash sidecar, and what it resolves to.

    Drives the real load path: the suppression lives inline in load_model's
    download phase, so nothing short of running it can pin the interaction.
    """
    import core.inference.llama_cpp as llama_cpp_module
    from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend

    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(
            lambda cls, binary = None: {
                "found": True,
                "supports_dspark": supports_dspark,
                "supports_dflash": supports_dflash,
            }
        ),
    )
    monkeypatch.setattr(llama_cpp_module, "_resolve_repo_id_casing", lambda repo: repo)
    monkeypatch.setattr(
        llama_cpp_module,
        "_hf_offline_if_unreachable",
        lambda: __import__("contextlib").nullcontext(),
    )

    backend = LlamaCppBackend()
    seen: dict = {"dflash_fetched": False}
    monkeypatch.setattr(backend, "_find_llama_server_binary", lambda **_kwargs: "/bin/llama")
    monkeypatch.setattr(backend, "_is_vulkan_backend", lambda _binary = None: False)
    monkeypatch.setattr(backend, "_get_gpu_memory", lambda _binary = None, **_kw: [(0, 4096, 8192)])
    monkeypatch.setattr(backend, "_gguf_path_is_diffusion", lambda *_args: False)
    monkeypatch.setattr(backend, "_kill_process", lambda: None)
    monkeypatch.setattr(
        backend, "_download_gguf", lambda **_kwargs: "/cache/snap/model-Q4_K_M.gguf"
    )
    monkeypatch.setattr(backend, "_download_mtp", lambda **_kwargs: None)
    # Exactly what _download_dspark does for a cached sidecar on a binary that
    # cannot run it: the path comes back regardless of the capability.
    monkeypatch.setattr(backend, "_download_dspark", lambda **_kwargs: dspark_cached)

    def _fetch_dflash(**_kwargs):
        seen["dflash_fetched"] = True
        return "/cache/snap/dflash-kquant.gguf"

    monkeypatch.setattr(backend, "_download_dflash", _fetch_dflash)

    def _stop(*_args, **_kwargs):
        raise _StopAfterDownloads

    # The first call past the download phase; the resolved drafter is already
    # settled by then.
    monkeypatch.setattr(backend, "_read_gguf_metadata", _stop)

    with pytest.raises(_StopAfterDownloads):
        backend.load_model(
            GgufLoadIntent(
                hf_repo = "org/repo",
                hf_variant = "Q4_K_M",
                model_identifier = "org/repo",
                speculative_type = "auto",
            )
        )
    return seen


def test_auto_still_fetches_dflash_when_the_binary_cannot_run_dspark(monkeypatch):
    """The regression: a cached DSpark sidecar this binary cannot launch is not
    a reason to skip the drafter it CAN launch."""
    seen = _dflash_fetch_during_auto_load(
        monkeypatch,
        supports_dspark = False,
        supports_dflash = True,
        dspark_cached = "/cache/snap/dspark-model-Q8_0.gguf",
    )
    assert seen["dflash_fetched"] is True


def test_auto_stands_down_on_dflash_for_a_dspark_it_can_launch(monkeypatch):
    """Unchanged where the stand-down was right: DSpark takes first refusal, so
    the ~1.5 GiB DFlash fetch would buy a file the load never opens."""
    seen = _dflash_fetch_during_auto_load(
        monkeypatch,
        supports_dspark = True,
        supports_dflash = True,
        dspark_cached = "/cache/snap/dspark-model-Q8_0.gguf",
    )
    assert seen["dflash_fetched"] is False


def test_auto_fetches_dflash_when_the_repo_ships_no_dspark_sidecar(monkeypatch):
    """Positive control: nothing about the DSpark capability gates a repo that
    publishes only the DFlash companion."""
    seen = _dflash_fetch_during_auto_load(
        monkeypatch,
        supports_dspark = True,
        supports_dflash = True,
        dspark_cached = None,
    )
    assert seen["dflash_fetched"] is True


# ── The caller's boundary reaches discovery, not just the rescan ──────
#
# ModelConfig.from_identifier runs the local companion scan, and the DFlash scan
# opens a candidate's header to confirm the architecture. A native grant covers
# one directory, so a dflash-*.gguf inside it can be a symlink whose target sits
# outside the lease. The load route rejects that afterwards, which cannot undo a
# read, so the boundary has to travel INTO the scan.


def test_from_identifier_hands_the_boundary_to_every_drafter_kind(tmp_path, monkeypatch):
    """All three kinds, not only the one that reads a header: they are the same
    discovery, and a kind that skipped the check would hand the load route a
    sidecar it has to reject a second time."""
    import utils.models.model_config as mc

    weight = _write_gguf(tmp_path / "model-Q4_K_M.gguf", "llama")
    seen: dict[str, tuple] = {}

    def _recorder(kind):
        def _detect(
            path,
            search_root = None,
            accept = None,
            **kwargs,
        ):
            seen[kind] = (path, search_root, accept)
            return None

        return _detect

    for kind, name in (
        ("mtp", "detect_mtp_file"),
        ("dspark", "detect_dspark_file"),
        ("dflash", "detect_dflash_file"),
    ):
        monkeypatch.setattr(mc, name, _recorder(kind))

    calls: list[tuple[str, str, str, str]] = []

    def _accept(candidate, gguf_file, kind, search_root):
        calls.append((candidate, gguf_file, kind, search_root))
        return False

    config = ModelConfig.from_identifier(str(weight), drafter_accept = _accept)

    assert config is not None
    assert set(seen) == {"mtp", "dspark", "dflash"}
    for kind, (path, search_root, accept) in seen.items():
        assert path == str(weight)
        assert accept is not None, kind
        # Bound to this load's file, this kind and this search root, so the three
        # closures cannot be swapped for one another.
        assert accept("/candidate.gguf") is False
        assert calls[-1] == ("/candidate.gguf", str(weight), kind, search_root)


def test_from_identifier_without_a_boundary_scans_exactly_as_before(tmp_path, monkeypatch):
    """Every caller that has no lease to impose passes nothing, and must see the
    same candidates in the same order."""
    import utils.models.model_config as mc

    weight = _write_gguf(tmp_path / "model-Q4_K_M.gguf", "llama")
    accepts: list = []

    def _detect(
        path,
        search_root = None,
        accept = None,
        **kwargs,
    ):
        accepts.append(accept)
        return None

    for name in ("detect_mtp_file", "detect_dspark_file", "detect_dflash_file"):
        monkeypatch.setattr(mc, name, _detect)

    ModelConfig.from_identifier(str(weight))
    assert accepts == [None, None, None]


def test_from_identifier_never_reads_a_sidecar_outside_the_boundary(tmp_path, monkeypatch):
    """End to end: the escaping symlink's target is never opened, and the config
    reports no DFlash sidecar rather than one the load route would reject."""
    import os

    # Patch the module detect_dflash_file resolves the name in, not the one that
    # used to re-export it: a patch on the re-exporting module never intercepts,
    # so `reads == []` below would hold whether or not the boundary works.
    import utils.models.drafters.dflash as dflash_mod

    leased = tmp_path / "leased"
    leased.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    weight = _write_gguf(leased / "model-Q4_K_M.gguf", "llama")
    target = _write_gguf(outside / "dflash-kquant.gguf", "dflash")
    os.symlink(target, leased / "dflash-kquant.gguf")

    reads: list[str] = []
    real_check = dflash_mod.is_dflash_architecture

    def _recording_check(path, *args, **kwargs):
        reads.append(str(path))
        return real_check(path, *args, **kwargs)

    monkeypatch.setattr(dflash_mod, "is_dflash_architecture", _recording_check)

    def _inside_the_lease(candidate, gguf_file, kind, search_root):
        return Path(search_root) in Path(candidate).parents

    config = ModelConfig.from_identifier(str(weight), drafter_accept = _inside_the_lease)

    assert config is not None
    assert config.gguf_dflash_file is None
    assert reads == []  # the out-of-lease target's header was never read


# ── Remote DFlash discovery is root level only, like the local scan ───
#
# The local contract is a root-level dflash- file (detect_dflash_file never
# offers a nested one, since dflash/ is a family name a user picks for real
# weights). The remote paths matched the basename in any nested directory, so an
# ordinary quants/dflash-*.gguf weight became a candidate -- and the header can
# only be read once the bytes are here, so the whole weight downloaded before the
# rejection.


@pytest.mark.parametrize(
    "path,expected",
    [
        ("dflash-kquant.gguf", True),
        ("dflash-model-Q8_0.gguf", True),
        ("quants/dflash-kquant.gguf", False),
        ("dflash/dflash-kquant.gguf", False),
        (r"quants\dflash-kquant.gguf", False),
        ("model-Q4_K_M.gguf", False),
        ("model-dflash-Q8_0.gguf", False),  # prefix-only naming rule, unchanged
    ],
)
def test_is_root_dflash_drafter_path(path, expected):
    from core.inference.llama_cpp import _is_root_dflash_drafter_path
    assert _is_root_dflash_drafter_path(path) is expected


def test_the_basename_predicate_keeps_its_own_semantics():
    """The root check is a separate predicate on purpose: _is_dflash_drafter_path
    is shared with callers that only ever have a bare filename."""
    from core.inference.llama_cpp import _is_dflash_drafter_path

    assert _is_dflash_drafter_path("quants/dflash-kquant.gguf") is True
    assert _is_dflash_drafter_path("dflash-kquant.gguf") is True


def test_download_dflash_never_fetches_a_nested_dflash_named_weight(tmp_path, monkeypatch):
    """The regression: the nested file is an ordinary weight the local scan would
    never offer, and picking it spent its entire download before the header check
    could turn it away."""
    _write_gguf(tmp_path / "model-Q4_K_M.gguf", "llama")

    b, got, fetched = _dflash_repo_download(
        tmp_path,
        monkeypatch,
        listing = ["model-Q4_K_M.gguf", "quants/dflash-model-Q8_0.gguf"],
    )

    assert got is None
    assert fetched == []  # nothing was paid for
    assert b._dflash_sidecar_absent is True


def test_download_dflash_still_takes_the_root_sidecar_beside_a_nested_one(tmp_path, monkeypatch):
    """Positive control: the nested name is skipped, not the whole repo."""
    _write_gguf(tmp_path / "model-Q4_K_M.gguf", "llama")
    sidecar = _write_gguf(tmp_path / "dflash-kquant.gguf", "dflash")

    b, got, fetched = _dflash_repo_download(
        tmp_path,
        monkeypatch,
        listing = ["model-Q4_K_M.gguf", "quants/dflash-model-Q8_0.gguf", "dflash-kquant.gguf"],
    )

    assert got == str(sidecar)
    assert fetched == ["dflash-kquant.gguf"]


def test_cached_dflash_lookup_ignores_a_nested_dflash_named_weight(tmp_path, monkeypatch):
    """Same rule on the offline cache scan, which hands its answer straight to
    --model-draft with no download to be rejected first."""
    from core.inference.llama_cpp import LlamaCppBackend

    snap = tmp_path / "snapshots" / "abc"
    (snap / "quants").mkdir(parents = True)
    _write_gguf(snap / "model-Q4_K_M.gguf", "llama")
    _write_gguf(snap / "quants" / "dflash-model-Q8_0.gguf", "dflash")
    monkeypatch.setattr(
        "utils.models.model_config._iter_hf_cache_snapshots", lambda *a, **k: [snap]
    )

    b = LlamaCppBackend()
    assert (
        b._cached_repo_dflash_drafter("org/repo", near_path = str(snap / "model-Q4_K_M.gguf")) is None
    )


# ── A split companion is fetched as a whole set ──────────────────────
#
# llama-server resolves a split drafter's sibling shards from the first shard's
# directory, so fetching only the picked shard left a drafter whose header reads
# fine and which the server then cannot open: the load fell back to no
# speculation with nothing to show for the download. The main-model downloader
# already resolves its shards with _gguf_extra_shards; the companion path reuses
# it rather than growing a second rule.


def _split_companion_download(tmp_path, monkeypatch, listing):
    import core.inference.llama_cpp as llama_cpp_module
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setattr("huggingface_hub.list_repo_files", lambda repo, token = None: list(listing))
    monkeypatch.setattr(llama_cpp_module, "_hub_download_in_flight", lambda hf_repo: False)
    fetched: list[str] = []

    def _fake_download(
        repo,
        filename,
        token,
        *,
        cancel_event = None,
        cache_dir = None,
    ):
        fetched.append(filename)
        path = tmp_path / filename
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr(llama_cpp_module, "hf_hub_download_with_xet_fallback", _fake_download)

    def _pick(names):
        return next((n for n in sorted(names) if Path(n).name.startswith("dflash-")), None)

    got = LlamaCppBackend()._download_companion_gguf(
        hf_repo = "org/repo",
        hf_token = None,
        pick = _pick,
        label = "DFlash drafter",
    )
    return got, fetched


def test_download_companion_gguf_fetches_every_shard_of_a_split_sidecar(tmp_path, monkeypatch):
    got, fetched = _split_companion_download(
        tmp_path,
        monkeypatch,
        [
            "model-Q4_K_M.gguf",
            "dflash-kquant-00001-of-00002.gguf",
            "dflash-kquant-00002-of-00002.gguf",
        ],
    )

    # The launch path is still shard 1, which is what llama-server is given.
    assert got == str(tmp_path / "dflash-kquant-00001-of-00002.gguf")
    assert fetched == ["dflash-kquant-00001-of-00002.gguf", "dflash-kquant-00002-of-00002.gguf"]


def test_download_companion_gguf_leaves_a_single_file_sidecar_alone(tmp_path, monkeypatch):
    """Every companion published today is one file, so the split handling must be
    a no-op for them: exactly one fetch, same path back."""
    got, fetched = _split_companion_download(
        tmp_path, monkeypatch, ["model-Q4_K_M.gguf", "dflash-kquant.gguf"]
    )

    assert got == str(tmp_path / "dflash-kquant.gguf")
    assert fetched == ["dflash-kquant.gguf"]


def test_companion_snapshot_reuse_skips_an_incomplete_split_sidecar(tmp_path):
    """The reuse scan is the other half: a snapshot holding shard 1 alone is not
    a reuse. Reporting it would skip the download that completes the set and
    leave the load with a drafter llama-server cannot open."""
    from core.inference.llama_cpp import _companion_snapshot_sibling

    snap = tmp_path / "models--org--repo" / "snapshots" / "abc"
    snap.mkdir(parents = True)
    (snap / "model-Q4_K_M.gguf").write_bytes(b"x")
    first = snap / "dflash-kquant-00001-of-00002.gguf"
    first.write_bytes(b"y")

    def _pick(names):
        return next((n for n in sorted(names) if Path(n).name.startswith("dflash-")), None)

    near = str(snap / "model-Q4_K_M.gguf")
    assert _companion_snapshot_sibling(near, _pick) is None

    (snap / "dflash-kquant-00002-of-00002.gguf").write_bytes(b"z")
    assert _companion_snapshot_sibling(near, _pick) == str(first)


def test_offline_companion_cache_hit_skips_an_incomplete_split(tmp_path, monkeypatch):
    """The offline cache lookup is the third way a shard can reach --model-draft,
    and offline there is no fetch left to complete the set."""
    import core.inference.llama_cpp as llama_cpp_module
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(llama_cpp_module, "_hub_download_in_flight", lambda hf_repo: False)
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files",
        lambda repo, token = None: ["dflash-kquant-00001-of-00002.gguf"],
    )
    first = tmp_path / "dflash-kquant-00001-of-00002.gguf"
    first.write_bytes(b"x")
    monkeypatch.setattr(llama_cpp_module, "_cached_hf_snapshot_file", lambda *a, **k: str(first))

    def _offline_fetch(*_args, **_kwargs):
        # What the Hub raises offline, which the caller swallows to None.
        raise RuntimeError("offline mode is enabled")

    monkeypatch.setattr(llama_cpp_module, "hf_hub_download_with_xet_fallback", _offline_fetch)

    def _pick(names):
        return next((n for n in sorted(names) if Path(n).name.startswith("dflash-")), None)

    b = LlamaCppBackend()
    # The half set is not reported as a cache hit, so the load ends with no
    # drafter rather than one llama-server cannot open.
    assert (
        b._download_companion_gguf(
            hf_repo = "org/repo", hf_token = None, pick = _pick, label = "DFlash drafter"
        )
        is None
    )

    (tmp_path / "dflash-kquant-00002-of-00002.gguf").write_bytes(b"y")
    assert b._download_companion_gguf(
        hf_repo = "org/repo", hf_token = None, pick = _pick, label = "DFlash drafter"
    ) == str(first)


def test_cached_dflash_lookup_skips_an_incomplete_split_set(tmp_path, monkeypatch):
    """The fourth way a shard can reach --model-draft: _download_dflash's offline
    fallback hands this lookup's answer back as the drafter with no fetch left to
    complete the set, so a lone shard reads as a valid header and llama-server
    then cannot open the siblings it resolves from that directory. The load drops
    speculation silently."""
    from core.inference.llama_cpp import LlamaCppBackend

    snap = tmp_path / "snapshots" / "abc"
    snap.mkdir(parents = True)
    weight = _write_gguf(snap / "model-Q4_K_M.gguf", "llama")
    first = _write_gguf(snap / "dflash-kquant-00001-of-00002.gguf", "dflash")
    monkeypatch.setattr(
        "utils.models.model_config._iter_hf_cache_snapshots", lambda *a, **k: [snap]
    )

    b = LlamaCppBackend()
    assert b._cached_repo_dflash_drafter("org/repo", near_path = str(weight)) is None

    _write_gguf(snap / "dflash-kquant-00002-of-00002.gguf", "dflash")
    assert b._cached_repo_dflash_drafter("org/repo", near_path = str(weight)) == str(first)


def test_cached_dflash_lookup_falls_through_from_a_half_split_to_a_whole_one(tmp_path, monkeypatch):
    """Skipped, not fatal, exactly as the header check is: the half set merely
    ranks first, and the snapshot still holds a sidecar that can be launched."""
    from core.inference.llama_cpp import LlamaCppBackend

    snap = tmp_path / "snapshots" / "abc"
    snap.mkdir(parents = True)
    weight = _write_gguf(snap / "model-Q4_K_M.gguf", "llama")
    # Q8_0 outranks BF16, so the incomplete set is the candidate tried first.
    _write_gguf(snap / "dflash-kquant-Q8_0-00001-of-00002.gguf", "dflash")
    whole = _write_gguf(snap / "dflash-kquant-BF16.gguf", "dflash")
    monkeypatch.setattr(
        "utils.models.model_config._iter_hf_cache_snapshots", lambda *a, **k: [snap]
    )

    assert LlamaCppBackend()._cached_repo_dflash_drafter("org/repo", near_path = str(weight)) == str(
        whole
    )


# ── A fetch that dropped is worth one more Apply ─────────────────────
#
# _dflash_sidecar_absent answers "this repo publishes none", which is permanent and
# must never be retried. The other None -- the Hub going away mid-fetch -- is invisible
# under Auto: no promotion happened, so no fallback reason was recorded, and the next
# Apply reuses a server that has no drafter for a repo that does publish one.


def _dflash_hub_download(tmp_path, monkeypatch, *, listing, fetch):
    """Drive _download_dflash through the real _download_companion_gguf, with only
    the two Hub calls stubbed out."""
    import core.inference.llama_cpp as llama_cpp_module
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: {"supports_dflash": True}),
    )
    monkeypatch.setattr(llama_cpp_module, "_hub_download_in_flight", lambda hf_repo: False)
    monkeypatch.setattr("huggingface_hub.list_repo_files", listing)
    monkeypatch.setattr(llama_cpp_module, "hf_hub_download_with_xet_fallback", fetch)
    # The local cache is not part of what is under test, and an unstubbed scan would
    # answer from whatever this machine happens to have downloaded.
    monkeypatch.setattr("utils.models.model_config._iter_hf_cache_snapshots", lambda *a, **k: [])

    b = LlamaCppBackend()
    got = b._download_dflash(hf_repo = "org/repo", binary = "/fake/llama-server")
    return b, got


def _never_fetched(*_args, **_kwargs):
    raise AssertionError("nothing should be downloaded")


def test_download_dflash_asks_again_after_a_listing_that_never_answered(tmp_path, monkeypatch):
    """An unreachable Hub says nothing about the repo, so recording it as "publishes
    none" would strand the model without the sidecar it does publish."""

    def _listing(repo, token = None):
        raise ConnectionError("hub unreachable")

    b, got = _dflash_hub_download(tmp_path, monkeypatch, listing = _listing, fetch = _never_fetched)

    assert got is None
    assert b._dflash_sidecar_absent is False
    assert b._dflash_retry_needed is True


def test_download_dflash_asks_again_after_a_download_that_dropped(tmp_path, monkeypatch):
    """The listing named the file, so this repo definitely publishes one: the bytes
    are all that is missing."""

    def _fetch(
        repo,
        filename,
        token,
        *,
        cancel_event = None,
        cache_dir = None,
    ):
        raise ConnectionError("connection reset")

    b, got = _dflash_hub_download(
        tmp_path,
        monkeypatch,
        listing = lambda repo, token = None: ["model-Q4_K_M.gguf", "dflash-kquant.gguf"],
        fetch = _fetch,
    )

    assert got is None
    assert b._dflash_sidecar_absent is False
    assert b._dflash_retry_needed is True


def test_download_dflash_does_not_ask_again_after_a_permanent_hub_error(tmp_path, monkeypatch):
    """A repo that is gone, gated, or being asked about offline is answered for good;
    retrying it on every Apply would relaunch an identical server forever."""

    class RepositoryNotFoundError(Exception):
        pass

    def _listing(repo, token = None):
        raise RepositoryNotFoundError("404")

    b, got = _dflash_hub_download(tmp_path, monkeypatch, listing = _listing, fetch = _never_fetched)

    assert got is None
    assert b._dflash_retry_needed is False


def test_download_dflash_does_not_ask_again_when_this_machine_is_the_problem(tmp_path, monkeypatch):
    """A full disk or an unwritable cache stays that way, and the retry costs a full
    unload plus another ~1.5 GiB attempt. Classified on errno, since the Hub client
    raises OSError subclasses for network trouble too."""
    import errno

    def _fetch(
        repo,
        filename,
        token,
        *,
        cancel_event = None,
        cache_dir = None,
    ):
        raise OSError(errno.ENOSPC, "No space left on device")

    b, got = _dflash_hub_download(
        tmp_path,
        monkeypatch,
        listing = lambda repo, token = None: ["model-Q4_K_M.gguf", "dflash-kquant.gguf"],
        fetch = _fetch,
    )

    assert got is None
    assert b._dflash_retry_needed is False


def test_download_dflash_treats_a_header_rejection_as_settled(tmp_path, monkeypatch):
    """A candidate whose header does not say dflash is permanently not a sidecar: the
    search falls through to the next one, and if that was the last one the repo
    publishes none. Reading it as a dropped fetch would reload on every Apply for a
    file that can never be launched."""

    def _fetch(
        repo,
        filename,
        token,
        *,
        cancel_event = None,
        cache_dir = None,
    ):
        # An ordinary weight that merely carries the sidecar's naming.
        return str(_write_gguf(tmp_path / filename, "llama"))

    b, got = _dflash_hub_download(
        tmp_path,
        monkeypatch,
        listing = lambda repo, token = None: ["model-Q4_K_M.gguf", "dflash-model-Q8_0.gguf"],
        fetch = _fetch,
    )

    assert got is None
    assert b._dflash_sidecar_absent is True
    assert b._dflash_retry_needed is False


def test_download_dflash_reaches_the_real_sidecar_behind_an_impostor(tmp_path, monkeypatch):
    """The other half of the same rule: the rejection only removes that one name from
    the pool, so the sidecar ranked behind it is still fetched and nothing is flagged
    for a retry."""

    def _fetch(
        repo,
        filename,
        token,
        *,
        cancel_event = None,
        cache_dir = None,
    ):
        arch = "dflash" if filename == "dflash-kquant.gguf" else "llama"
        return str(_write_gguf(tmp_path / filename, arch))

    b, got = _dflash_hub_download(
        tmp_path,
        monkeypatch,
        listing = lambda repo, token = None: [
            "model-Q4_K_M.gguf",
            "dflash-model-Q8_0.gguf",
            "dflash-kquant.gguf",
        ],
        fetch = _fetch,
    )

    assert got == str(tmp_path / "dflash-kquant.gguf")
    assert b._dflash_retry_needed is False


# ── The DFlash sidecar in the download plan ──────────────────────────
#
# The plan has to promise exactly what the loader will open: every shard of it,
# paired with the weight family the plan keeps, and never a whole model that
# merely carries the prefix.


def test_variant_plans_carry_every_shard_of_a_split_dflash_sidecar():
    """The loader refuses a companion whose split set is incomplete, so planning
    shard 1 alone reports the variant complete and then loses DFlash."""
    plans = build_gguf_variant_plans(
        [
            _sib("model-Q4_K_M.gguf", 15_000, "main"),
            _sib("dflash-kquant-00001-of-00002.gguf", 800, "d1"),
            _sib("dflash-kquant-00002-of-00002.gguf", 800, "d2"),
        ]
    )
    targets = set(plans["q4_k_m"].target_filenames)
    assert "dflash-kquant-00001-of-00002.gguf" in targets
    assert "dflash-kquant-00002-of-00002.gguf" in targets


def test_variant_plans_pair_dflash_with_the_weight_family_they_keep():
    """_one_shard_family keeps the lexicographically first family, so ranking the
    sidecar against the listing's first weight pairs the discarded one."""
    plans = build_gguf_variant_plans(
        [
            # Listing order puts the discarded family first.
            _sib("QwQ-32B.BF16-00001-of-00002.gguf", 30_000, "b1"),
            _sib("QwQ-32B.BF16-00002-of-00002.gguf", 30_000, "b2"),
            _sib("QwQ-32B-BF16-00001-of-00002.gguf", 30_000, "a1"),
            _sib("QwQ-32B-BF16-00002-of-00002.gguf", 30_000, "a2"),
            _sib("dflash-QwQ-32B-BF16-Q8_0.gguf", 2_000, "da"),
            _sib("dflash-QwQ-32B.BF16-Q8_0.gguf", 2_000, "db"),
        ]
    )
    plan = plans["bf16"]
    assert plan.main_filenames == frozenset(
        {"QwQ-32B-BF16-00001-of-00002.gguf", "QwQ-32B-BF16-00002-of-00002.gguf"}
    )
    assert "dflash-QwQ-32B-BF16-Q8_0.gguf" in plan.target_filenames
    assert "dflash-QwQ-32B.BF16-Q8_0.gguf" not in plan.target_filenames


def test_variant_plans_skip_a_dflash_prefixed_file_too_big_to_be_a_drafter():
    """dflash- is a prefix real weights carry (Lucebox/Qwen3.6-27B-DFlash-GGUF) and a
    listing cannot read the architecture, so size is the only bound available: a
    drafter is a few layers of its target and cannot outweigh it."""
    plans = build_gguf_variant_plans(
        [
            _sib("Qwen3.6-27B-Q4_K_M.gguf", 15_000, "main"),
            _sib("dflash-Qwen3.6-27B-BF16.gguf", 54_000, "impostor"),
        ]
    )
    plan = plans["q4_k_m"]
    assert "dflash-Qwen3.6-27B-BF16.gguf" not in plan.target_filenames
    assert plan.download_size_bytes == 15_000


def test_variant_plans_still_carry_the_published_dflash_sidecar():
    """The Muse-Glimmer shape the feature ships for stays planned."""
    plans = build_gguf_variant_plans(
        [
            _sib("Muse-Glimmer-30B-UD-Q4_K_XL.gguf", 15_878, "main"),
            _sib("mmproj-kquant.gguf", 1_400, "mmproj"),
            _sib("dflash-kquant.gguf", 1_631, "dflash"),
        ]
    )
    assert "dflash-kquant.gguf" in plans["ud-q4_k_xl"].target_filenames


def test_download_dflash_skips_a_root_weight_too_big_to_be_a_drafter(monkeypatch, tmp_path):
    """The runtime picker needs the bound the plan has: a root dflash-*.gguf that is
    an ordinary weight passes the filename test, and the header can only answer once
    the whole object is on disk."""
    from core.inference.llama_cpp import LlamaCppBackend

    weight = tmp_path / "Qwen3.6-27B-Q4_K_M.gguf"
    weight.write_bytes(b"x" * 4_000)
    monkeypatch.setattr(
        LlamaCppBackend,
        "_remote_root_gguf_sizes",
        staticmethod(
            lambda repo, token = None: {
                "dflash-Qwen3.6-27B-BF16.gguf": 40_000,
                "dflash-kquant.gguf": 500,
            }
        ),
    )
    picked = _dflash_download_pick(
        monkeypatch,
        listing = ["Qwen3.6-27B-Q4_K_M.gguf", "dflash-Qwen3.6-27B-BF16.gguf", "dflash-kquant.gguf"],
        near_path = str(weight),
    )
    assert picked == "dflash-kquant.gguf"


def test_download_companion_refuses_a_listing_missing_part_of_a_split_set(monkeypatch):
    """The snapshot and cache paths both refuse half a split companion; the download
    path returned shard 1 and handed llama-server a set it cannot open."""
    import core.inference.llama_cpp as llama_cpp_module
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.setattr(
        llama_cpp_module, "_companion_snapshot_sibling", lambda near_path, pick: None
    )
    # Patched at the source: _download_companion_gguf imports list_repo_files inside
    # its own body, so a module attribute on llama_cpp is never consulted.
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "list_repo_files",
        lambda repo, token = None: ["dflash-kquant-00001-of-00002.gguf"],
    )
    downloads: list = []
    monkeypatch.setattr(
        llama_cpp_module,
        "hf_hub_download_with_xet_fallback",
        lambda *a, **k: downloads.append(a) or "/tmp/x.gguf",
        raising = False,
    )
    b = LlamaCppBackend()
    got = b._download_companion_gguf(
        hf_repo = "org/repo",
        hf_token = None,
        pick = lambda files: next(iter(files), None),
        label = "DFlash drafter",
    )
    assert got is None
    assert downloads == []


def test_variant_plans_skip_a_half_published_split_sidecar():
    """Planning a set the listing only half carries reports the download complete and
    then loses DFlash, since the loader refuses the partial set."""
    plans = build_gguf_variant_plans(
        [
            _sib("model-Q4_K_M.gguf", 15_000, "main"),
            _sib("dflash-kquant-00001-of-00002.gguf", 800, "d1"),
        ]
    )
    assert not [f for f in plans["q4_k_m"].target_filenames if f.startswith("dflash-")]


def test_variant_plans_fall_through_to_a_usable_sidecar_behind_an_oversized_one():
    """Both plan rules filter before the ranking, so the impostor at the top of the
    order steps aside instead of taking the whole plan down with it."""
    plans = build_gguf_variant_plans(
        [
            _sib("model-B-Q4_K_M.gguf", 15_000, "main"),
            # Ranks first (names this weight) but is a whole model.
            _sib("dflash-model-B-BF16.gguf", 54_000, "impostor"),
            _sib("dflash-kquant.gguf", 900, "real"),
        ]
    )
    targets = plans["q4_k_m"].target_filenames
    assert "dflash-kquant.gguf" in targets
    assert "dflash-model-B-BF16.gguf" not in targets


def test_download_dflash_sums_a_split_family_before_the_size_bound(monkeypatch, tmp_path):
    """Each shard of a split ordinary weight can sit under the target while the set
    does not, so bounding the picked shard alone still fetched the whole thing."""
    from core.inference.llama_cpp import LlamaCppBackend

    weight = tmp_path / "model-Q4_K_M.gguf"
    weight.write_bytes(b"x" * 10_000)
    monkeypatch.setattr(
        LlamaCppBackend,
        "_remote_root_gguf_sizes",
        staticmethod(
            lambda repo, token = None: {
                "dflash-big-00001-of-00002.gguf": 6_000,
                "dflash-big-00002-of-00002.gguf": 6_000,
                "dflash-kquant.gguf": 500,
            }
        ),
    )
    picked = _dflash_download_pick(
        monkeypatch,
        listing = [
            "model-Q4_K_M.gguf",
            "dflash-big-00001-of-00002.gguf",
            "dflash-big-00002-of-00002.gguf",
            "dflash-kquant.gguf",
        ],
        near_path = str(weight),
    )
    assert picked == "dflash-kquant.gguf"


def test_download_companion_records_an_incomplete_listing_as_settled():
    """The completeness rejection lands after outcome["listed"] was set true, so
    without this the caller reads a settled answer as one worth retrying forever."""
    import core.inference.llama_cpp as llama_cpp_module
    import huggingface_hub
    from core.inference.llama_cpp import LlamaCppBackend

    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    try:
        mp.delenv("HF_HUB_OFFLINE", raising = False)
        mp.setattr(llama_cpp_module, "_companion_snapshot_sibling", lambda near_path, pick: None)
        mp.setattr(
            huggingface_hub,
            "list_repo_files",
            lambda repo, token = None: ["dspark-kquant-00001-of-00002.gguf"],
        )
        outcome: dict = {}
        got = LlamaCppBackend()._download_companion_gguf(
            hf_repo = "org/repo",
            hf_token = None,
            pick = lambda files: next(iter(files), None),
            label = "DSpark drafter",
            outcome = outcome,
        )
    finally:
        mp.undo()
    assert got is None
    assert outcome.get("listed") is False


def test_download_dflash_reaches_the_complete_family_behind_an_incomplete_one(
    monkeypatch, tmp_path
):
    """A shard from a half-published set makes _download_companion_gguf answer None,
    which ends the loop, so the complete sidecar behind it was never reached."""
    from core.inference.llama_cpp import LlamaCppBackend

    weight = tmp_path / "model-B-Q4_K_M.gguf"
    weight.write_bytes(b"x" * 10_000)
    monkeypatch.setattr(
        LlamaCppBackend, "_remote_root_gguf_sizes", staticmethod(lambda repo, token = None: {})
    )
    picked = _dflash_download_pick(
        monkeypatch,
        listing = [
            "model-B-Q4_K_M.gguf",
            # Ranks first by naming this weight, but the set is missing shard 2.
            "dflash-model-B-00001-of-00002.gguf",
            "dflash-kquant.gguf",
        ],
        near_path = str(weight),
    )
    assert picked == "dflash-kquant.gguf"


def test_split_completeness_reads_shard_indices_not_a_shard_count():
    """A listing caught mid-publication can hold 00001-of-00002 beside a stray
    00003-of-00002. Two files, so a count calls the set whole, while shard 2 is
    still missing and llama-server cannot open it: the picker then ranks a family
    it cannot load and the guard bills the training job for it."""
    from utils.models.drafters import split_listing_is_complete

    names = ["model-00001-of-00002.gguf", "model-00003-of-00002.gguf"]
    assert not split_listing_is_complete(names, names[0])
    # A duplicate listing of one shard is the same trap without the odd index.
    dupes = ["dir/model-00001-of-00002.gguf", "dir/model-00001-of-00002.gguf"]
    assert not split_listing_is_complete(dupes, dupes[0])
    whole = ["model-00001-of-00002.gguf", "model-00002-of-00002.gguf"]
    assert split_listing_is_complete(whole, whole[0])


def test_split_completeness_is_scoped_to_the_files_own_directory():
    """A repo laid out by quant can hold half of one broken set beside half of
    another. Matching basenames alone calls both of them one whole set, and the
    fetch then cannot load either."""
    from utils.models.drafters import split_listing_is_complete

    names = ["Q4/model-00001-of-00002.gguf", "Q8/model-00002-of-00002.gguf"]
    assert not split_listing_is_complete(names, names[0])
    assert not split_listing_is_complete(names, names[1])
    whole = ["Q4/model-00001-of-00002.gguf", "Q4/model-00002-of-00002.gguf"]
    assert split_listing_is_complete(whole, whole[0])
