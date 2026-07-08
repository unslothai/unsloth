# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Separate-file MTP drafter (Gemma 4) contracts.

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
from hub.utils.gguf import is_mtp_drafter_path
from hub.utils.gguf_plan import build_gguf_variant_plans, plan_from_expected_files
from utils.models.model_config import (
    _is_mtp_drafter,
    detect_gguf_model,
    detect_mtp_file,
    extract_model_size_b,
)


# ── Predicate + layering mirrors ─────────────────────────────────────

DRAFTER_CASES = [
    ("mtp-gemma-4-12b-it.gguf", True),
    ("MTP/gemma-4-12b-it-Q8_0-MTP.gguf", True),
    ("foo/MTP/bar.gguf", True),
    ("gemma-4-12b-it-Q8_0.gguf", False),
    # Baked-in Qwen MTP repos: the head is inside the main GGUF, the file
    # IS the model -- must never be classified as a companion.
    ("Qwen3.6-27B-MTP-Q4_K_M.gguf", False),
    ("prompt-mtp-test.gguf", False),
    ("smtp/model.gguf", False),
    ("mtp-readme.txt", False),
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


DFLASH_SIBLINGS = [
    _sib("Qwen3-4B-Q4_K_M.gguf", 4_000, "main-q4"),
    _sib("Qwen3-4B-DFlash-bf16.gguf", 1_200, "dflash-bf16"),
    _sib("Qwen3-4B-DFlash-q8_0.gguf", 575, "dflash-q8"),
]


def test_variant_plans_carry_dflash_drafter_preferring_quant():
    plans = build_gguf_variant_plans(DFLASH_SIBLINGS)

    # The -DFlash- files are companions, never selectable quants (no phantom q8_0).
    assert set(plans) == {"q4_k_m"}
    plan = plans["q4_k_m"]
    # The quantized drafter is fetched with the variant; the oversized bf16 is not.
    assert "Qwen3-4B-DFlash-q8_0.gguf" in plan.target_filenames
    assert "Qwen3-4B-DFlash-bf16.gguf" not in plan.target_filenames
    assert "dflash-q8" in plan.companion_hashes
    assert "dflash-q8" not in plan.main_hashes
    # Download size = main + quantized drafter (not the bf16).
    assert plan.download_size_bytes == 4_575


def test_variant_plans_skip_dflash_for_vision_repos():
    # A vision repo suppresses DFlash at load, so the download plan must not
    # fetch the drafter with every variant.
    plans = build_gguf_variant_plans(
        [
            _sib("Qwen3-VL-4B-Q4_K_M.gguf", 4_000, "main"),
            _sib("mmproj-F16.gguf", 500, "mmproj"),
            _sib("Qwen3-VL-4B-DFlash-q8_0.gguf", 575, "dflash"),
        ]
    )
    plan = plans["q4_k_m"]
    assert not any("dflash" in name.lower() for name in plan.target_filenames)
    assert plan.mmproj_filenames == frozenset({"mmproj-F16.gguf"})


def test_variant_plans_prefer_quant_over_fp16_dflash():
    # fp16 is full precision even though extract_quant_label doesn't tag it, so
    # the quantized drafter must still win the download pick.
    plans = build_gguf_variant_plans(
        [
            _sib("Qwen3-4B-Q4_K_M.gguf", 4_000, "main-q4"),
            _sib("Qwen3-4B-DFlash-fp16.gguf", 1_200, "dflash-fp16"),
            _sib("Qwen3-4B-DFlash-q8_0.gguf", 575, "dflash-q8"),
        ]
    )
    plan = plans["q4_k_m"]
    assert "Qwen3-4B-DFlash-q8_0.gguf" in plan.target_filenames
    assert "Qwen3-4B-DFlash-fp16.gguf" not in plan.target_filenames


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
    weight = tmp_path / "gemma-4-12b-it-Q4_K_M.gguf"
    weight.write_bytes(b"x")
    drafter = tmp_path / "mtp-gemma-4-12b-it.gguf"
    drafter.write_bytes(b"x")

    # Loaded without a drafter; one now exists on disk -> must reload.
    b = _loaded_backend(weight, None)
    assert not b._already_in_target_state(**_target_state_kwargs(weight, str(drafter)))
    # Same drafter as launched -> still deduped.
    b = _loaded_backend(weight, str(drafter))
    assert b._already_in_target_state(**_target_state_kwargs(weight, str(drafter)))


def test_detect_gguf_model_rejects_mtp_subdir_copy(tmp_path):
    # Direct selection of an MTP/ copy: the basename alone has no mtp-
    # prefix, so rejection relies on the parent dir name.
    sub = tmp_path / "MTP"
    sub.mkdir()
    copy = sub / "gemma-4-12b-it-BF16-MTP.gguf"
    copy.write_bytes(b"x")
    assert detect_gguf_model(str(copy)) is None
    # Selecting the MTP dir itself must not surface the copies as models.
    assert detect_gguf_model(str(sub)) is None
