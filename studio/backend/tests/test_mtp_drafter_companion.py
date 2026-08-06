# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Separate-file drafter contracts: MTP (Gemma 4) and DSpark (DeepSeek V4 Flash).

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

    assert backend._cached_repo_dspark_drafter("some/repo").endswith(
        "dspark/dspark-model-Q8_0.gguf"
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
    """Studio downloads the sidecar itself once the user opts in, and companion
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
    and Studio never fetches it as a companion, so it is not reclaimable."""
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
