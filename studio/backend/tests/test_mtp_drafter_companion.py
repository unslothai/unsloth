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
    detect_mtp_file,
    extract_model_size_b,
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
    assert native_gguf_companion_parent_allowed(nested_drafter, weight, allow_mtp_subdir = True)


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
        model_dir / "MTP" / drafter.name, weight, allow_mtp_subdir = True
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
