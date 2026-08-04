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
    # Same drafter published under its general.architecture name -- the prefix
    # carries it, e.g. ggml-org/Qwen3.6-27B-GGUF ships the drafter at the root.
    ("dflash/dflash-model-Q8_0.gguf", True),
    ("dflash-model.gguf", True),
    ("dflash-Qwen3.6-27B-BF16.gguf", True),
    ("dflash-Qwen3.6-27B-Q8_0.gguf", True),
    # ...but dflash is a family name, so the DIRECTORY is not a drafter marker.
    # No published repo uses a dflash/ companion folder, while users do name a
    # local folder after the family they downloaded.
    ("dflash/Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf", False),
    ("DFlash/Laguna-S-2.1-DFlash-Q5_K_M.gguf", False),
    ("foo/dflash/bar.gguf", False),
    # Real Hub filenames where dflash/dspark is the family name: each IS the model.
    ("Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf", False),
    ("qwen36-35b-a3b-dflash-Q8_0.gguf", False),
    ("laguna-xs21-dflash-q4.gguf", False),
    ("mimo-v25-pro-dflash-draft-bf16-rope5m.gguf", False),
    ("xdspark/model.gguf", False),
    ("sdflash/model.gguf", False),
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


DEEPSEEK_SIBLINGS = [
    _sib("UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00002.gguf", 9_000, "q4-1"),
    _sib("UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00002-of-00002.gguf", 8_000, "q4-2"),
    _sib("UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00002.gguf", 5_000, "iq1-1"),
    _sib("UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00002-of-00002.gguf", 4_000, "iq1-2"),
    _sib("dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf", 1_100, "dspark-bf16"),
    _sib("dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf", 1_000, "dspark-q8"),
]


def test_dspark_drafters_are_not_quants_and_are_not_auto_fetched():
    plans = build_gguf_variant_plans(DEEPSEEK_SIBLINGS)

    # The drafters carry BF16/Q8_0 tokens; neither may become a quant.
    assert set(plans) == {"ud-q4_k_xl", "ud-iq1_s"}

    # DSpark is opt-in and ~11 GB per file, so unlike the root mtp-*.gguf it
    # must not be folded into every plan.
    for plan in plans.values():
        assert not any(name.startswith("dspark/") for name in plan.target_filenames)
        assert plan.companion_hashes == frozenset()

    q4 = plans["ud-q4_k_xl"]
    assert q4.main_size_bytes == 17_000
    assert q4.download_size_bytes == 17_000


def test_a_repo_of_only_drafters_still_lists_them():
    # A companion needs something to accompany. mradermacher puts the repo name
    # on every file, so all 11 quants of DFlash-Qwen3.5-27B-Uncensored-GGUF
    # carry the dflash- prefix; filtering them would empty a real 27B model.
    mradermacher = [
        _sib("DFlash-Qwen3.5-27B-Uncensored.Q4_K_M.gguf", 16_000, "q4"),
        _sib("DFlash-Qwen3.5-27B-Uncensored.Q8_0.gguf", 28_000, "q8"),
        _sib("DFlash-Qwen3.5-27B-Uncensored.mmproj-f16.gguf", 900, "mmproj"),
    ]
    assert set(build_gguf_variant_plans(mradermacher)) == {"q4_k_m", "q8_0"}

    # A reprieved file still has to look like a quant, so a drafter whose name
    # carries no quant token stays a companion.
    assert build_gguf_variant_plans([_sib("dspark-drafter-blob.gguf", 6_900, "d1")]) == {}


def test_a_drafter_beside_non_gguf_weights_is_still_a_companion():
    # Weights in another format are still a main model, so the drafter has
    # something to accompany and must not be advertised as a GGUF variant.
    siblings = [
        _sib("model-00001-of-00001.safetensors", 64_000, "st"),
        _sib("mtp-model-Q8_0.gguf", 100, "d"),
    ]
    assert build_gguf_variant_plans(siblings) == {}


def test_a_reprieved_root_drafter_is_not_also_a_companion():
    # The plan builder would otherwise fold the same file in twice, doubling
    # main_size_bytes and duplicating target_filenames.
    plans = build_gguf_variant_plans([_sib("dflash-model-Q8_0.gguf", 1_000, "d")])
    plan = plans["q8_0"]
    assert plan.target_filenames == ("dflash-model-Q8_0.gguf",)
    assert plan.main_size_bytes == 1_000
    assert plan.download_size_bytes == 1_000


MRADERMACHER = [
    "DFlash-Qwen3.5-27B-Uncensored.Q4_K_M.gguf",
    "DFlash-Qwen3.5-27B-Uncensored.Q8_0.gguf",
]
GGML_ORG = [
    "Qwen3.6-27B-BF16.gguf",
    "Qwen3.6-27B-Q8_0.gguf",
    "dflash-Qwen3.6-27B-BF16.gguf",
    "mtp-Qwen3.6-27B-Q8_0.gguf",
]


def test_every_whole_repo_consumer_agrees_on_a_reprieved_repo():
    # The reprieve is only useful if the loader and the auto-download admission
    # resolve what the picker advertises.
    from core.inference.llama_cpp import _gguf_files_for_variant
    from core.inference.openai_auto_download import _gguf_variants

    assert _gguf_files_for_variant(MRADERMACHER, "Q4_K_M") == [MRADERMACHER[0]]
    assert sorted(_gguf_variants(_sib(n, 1_000, n) for n in MRADERMACHER)) == ["Q4_K_M", "Q8_0"]

    # ...and a repo that does have main weights still hides its drafters.
    assert _gguf_files_for_variant(GGML_ORG, "BF16") == ["Qwen3.6-27B-BF16.gguf"]
    assert sorted(_gguf_variants(_sib(n, 1_000, n) for n in GGML_ORG)) == ["BF16", "Q8_0"]


def test_only_a_whole_snapshot_gets_the_reprieve_locally(tmp_path):
    # An HF cache snapshot is the same listing the remote path sees; an
    # arbitrary folder is not, so it keeps the stricter per-path filter.
    for name in MRADERMACHER:
        (tmp_path / name).write_bytes(b"x" * 16)

    whole = list_hub_local_gguf_variants(str(tmp_path), whole_repo = True)[0]
    assert sorted(v.quant for v in whole) == ["Q4_K_M", "Q8_0"]
    assert list_hub_local_gguf_variants(str(tmp_path))[0] == []


def test_a_directory_named_drafter_is_never_reprieved():
    # A snapshot holding only MTP/ or dspark/ is a half-downloaded repo, not a
    # drafter-only one, so the publisher's layout still wins.
    assert build_gguf_variant_plans([_sib("MTP/mtp-gemma-4-12b-it.gguf", 100, "d")]) == {}
    assert build_gguf_variant_plans([_sib("dspark/dspark-model-BF16.gguf", 100, "d")]) == {}


def test_drafters_beside_a_main_model_are_still_excluded():
    # ggml-org/Qwen3.6-27B-GGUF: the guard must not fire here, or the 3 GB
    # dflash BF16 merges into the real 54 GB one.
    ggml_org = [
        _sib("Qwen3.6-27B-BF16.gguf", 53_810, "bf16"),
        _sib("Qwen3.6-27B-Q8_0.gguf", 28_600, "q8"),
        _sib("dflash-Qwen3.6-27B-BF16.gguf", 3_470, "dflash-bf16"),
        _sib("mtp-Qwen3.6-27B-Q8_0.gguf", 3_160, "mtp-q8"),
    ]
    plans = build_gguf_variant_plans(ggml_org)
    assert set(plans) == {"bf16", "q8_0"}
    assert plans["bf16"].main_filenames == frozenset({"Qwen3.6-27B-BF16.gguf"})
    assert plans["bf16"].main_size_bytes == 53_810


def test_dspark_only_repo_has_no_preferred_drafter_to_download():
    # preferred_mtp_sibling picks what is fetched with every variant, and a
    # DSpark repo ships no launchable drafter.
    assert preferred_mtp_sibling(DEEPSEEK_SIBLINGS) is None


def test_cached_drafter_lookup_ignores_dspark(tmp_path, monkeypatch):
    # _cached_repo_mtp_drafter feeds --spec-type draft-mtp, so a cached DSpark
    # file must not surface there even though it is a companion for selection.
    from core.inference import llama_cpp as llama_cpp_module

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


def test_local_dflash_folder_keeps_its_models(tmp_path):
    # DFlash is a family name, not a companion folder: no published repo ships a
    # dflash/ dir, but a user does name the folder after what they downloaded
    # (z-lab/Qwen3.6-35B-A3B-DFlash, poolside/Laguna-S-2.1-DFlash, ...).
    # Matching the directory would hide the weights from every detection path.
    sub = tmp_path / "dflash"
    sub.mkdir()
    main = sub / "Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf"
    main.write_bytes(b"x" * 100)
    drafter = sub / "dflash-Qwen3.6-27B-BF16.gguf"  # ggml-org/Qwen3.6-27B-GGUF
    drafter.write_bytes(b"x" * 10)

    assert detect_gguf_model(str(main)) == str(main.resolve())
    assert detect_gguf_model(str(sub)) == str(main.resolve())
    assert detect_gguf_model(str(drafter)) is None
    for lister in (list_local_gguf_variants, list_hub_local_gguf_variants):
        assert [v.filename for v in lister(str(sub))[0]] == [main.name]


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


# ── Whole-repo scope for the drafter reprieve ────────────────────────
# The reprieve applies to WHOLE-REPO listings only: remote sibling lists and HF
# cache snapshots. An arbitrary local folder never gets it (a folder holding
# only a companion is a half-downloaded repo), which the tests above pin.


def _snapshot(root: Path, repo: str, names: dict[str, int]) -> Path:
    """A one-revision HF cache snapshot for *repo* holding *names* -> size."""
    snap = root / f"models--{repo.replace('/', '--')}" / "snapshots" / "rev1"
    for rel, size in names.items():
        file = snap / rel
        file.parent.mkdir(parents = True, exist_ok = True)
        file.write_bytes(b"x" * size)
    return snap


def _use_cache(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = root),
    )


# 1. Offline detection must resolve what the online path resolves.
def test_hf_cache_detection_reprieves_a_drafter_named_repo(tmp_path, monkeypatch):
    """A cache snapshot IS a whole-repo listing. Without the reprieve, offline or
    after a Hub failure a drafter-prefixed repo returned no GGUF at all."""
    from utils.models.model_config import _detect_gguf_from_hf_cache

    repo = "mradermacher/DFlash-Qwen3.5-27B-Uncensored-GGUF"
    _snapshot(
        tmp_path,
        repo,
        {
            "DFlash-Qwen3.5-27B-Uncensored.Q4_K_M.gguf": 16,
            "DFlash-Qwen3.5-27B-Uncensored.Q8_0.gguf": 32,
        },
    )
    _use_cache(monkeypatch, tmp_path)

    assert _detect_gguf_from_hf_cache(repo) == "DFlash-Qwen3.5-27B-Uncensored.Q4_K_M.gguf"


def test_hf_cache_detection_still_rejects_a_directory_named_drafter(tmp_path, monkeypatch):
    """Negative control: a ``dspark/`` companion is never reprieved."""
    from utils.models.model_config import _detect_gguf_from_hf_cache

    repo = "unsloth/DeepSeek-V4-Flash-0731-GGUF"
    _snapshot(tmp_path, repo, {"dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf": 16})
    _use_cache(monkeypatch, tmp_path)

    assert _detect_gguf_from_hf_cache(repo) is None


# 2. The loader mirror must read the quant token exactly as the canonical does.
@pytest.mark.parametrize(
    "listing,expected",
    [
        # Quant named by the PARENT directory, not the basename.
        (["Q4_K_M/dflash-model.gguf"], set()),
        # No family is named MTP, so mtp- is never reprieved whatever names it.
        (["BF16/mtp-model.gguf"], {"BF16/mtp-model.gguf"}),
        # Still a drafter: nothing anywhere names a quant.
        (["dflash-model.gguf"], {"dflash-model.gguf"}),
        # Directory-named drafters are never reprieved, quant token or not.
        (["Q4_K_M/dspark/drafter.gguf"], {"Q4_K_M/dspark/drafter.gguf"}),
    ],
)
def test_drafter_reprieve_mirrors_agree_on_parent_quant_tokens(listing, expected):
    from core.inference.llama_cpp import _drafter_paths_in as _core_drafter_paths_in
    from hub.utils.gguf import drafter_paths_in
    from utils.models.model_config import _drafter_paths_in as _utils_drafter_paths_in

    assert set(drafter_paths_in(listing)) == expected
    assert set(_utils_drafter_paths_in(listing)) == expected
    assert set(_core_drafter_paths_in(listing)) == expected


def test_variant_file_lookup_finds_a_parent_quant_reprieved_weight():
    """The picker advertises ``Q4_K_M`` for it, so the loader must resolve it too
    or the load fails with no main file."""
    from core.inference.llama_cpp import _gguf_files_for_variant
    assert _gguf_files_for_variant(["Q4_K_M/dflash-model.gguf"], "Q4_K_M") == [
        "Q4_K_M/dflash-model.gguf"
    ]


# 3. The reprieve decision needs the FULL snapshot listing, not just the GGUFs.
def test_whole_repo_reprieve_sees_non_gguf_weights(tmp_path):
    """A safetensors weight is something to accompany, so the drafter beside it
    stays a companion instead of becoming a phantom Q8_0 quant."""
    (tmp_path / "model.safetensors").write_bytes(b"x" * 4096)
    (tmp_path / "config.json").write_bytes(b"{}")
    (tmp_path / "mtp-model-Q8_0.gguf").write_bytes(b"x" * 64)

    assert list_hub_local_gguf_variants(str(tmp_path), whole_repo = True)[0] == []


def test_whole_repo_reprieve_still_applies_without_a_main_weight(tmp_path):
    """Negative control: with nothing to accompany, the quant-labelled drafter IS
    the snapshot's weight."""
    (tmp_path / "dflash-model-Q8_0.gguf").write_bytes(b"x" * 64)

    variants = list_hub_local_gguf_variants(str(tmp_path), whole_repo = True)[0]
    assert [(v.quant, v.filename) for v in variants] == [("Q8_0", "dflash-model-Q8_0.gguf")]
    # ...but an arbitrary folder gets no reprieve, so the same tree lists nothing.
    assert list_hub_local_gguf_variants(str(tmp_path))[0] == []


# 4. A fully downloaded drafter-named repo must be a usable, chattable row.
def test_cache_inventory_admits_a_fully_downloaded_drafter_named_repo(tmp_path):
    """Every file carries the family prefix (mradermacher names files after the
    repo), so the context-free predicate left the snapshot with no variants."""
    from hub.services.models import cache_inventory, common
    from hub.utils import inventory_scan

    names = {
        "DFlash-Qwen3.5-27B-Uncensored.Q4_K_M.gguf": 32,
        "DFlash-Qwen3.5-27B-Uncensored.Q8_0.gguf": 64,
    }
    snap = _snapshot(tmp_path, "mradermacher/DFlash-Qwen3.5-27B-Uncensored-GGUF", names)

    assert inventory_scan._completed_gguf_variants(snap) == {"Q4_K_M", "Q8_0"}
    assert inventory_scan.snapshot_has_complete_variants(str(snap)) is True
    rows = common._classify_local_path(snap, "hf_cache", model_id = "org/DFlash-GGUF")
    assert [row.model_format for row in rows] == ["gguf"]

    revision = SimpleNamespace(
        commit_hash = "rev1",
        snapshot_path = str(snap),
        files = [
            SimpleNamespace(
                file_path = str(snap / rel),
                file_name = rel,
                size_on_disk = size,
                blob_path = None,
                blob_last_modified = 1.0,
            )
            for rel, size in names.items()
        ],
    )
    repo_info = SimpleNamespace(repo_path = None, revisions = [revision])
    assert cache_inventory._repo_has_gguf_files(repo_info) is True


def test_arbitrary_folder_scans_keep_the_context_free_rule(tmp_path):
    """The scoping rule: only snapshots are whole-repo listings. A scanned folder
    holding one companion is a half-downloaded repo, not a model."""
    from hub.services.models import common

    (tmp_path / "mtp-model-Q8_0.gguf").write_bytes(b"x" * 64)

    assert common._main_gguf_files(tmp_path) == []
    assert [row.model_format for row in common._classify_local_path(tmp_path, "models_dir")] == [
        "unknown"
    ]


# 5. The low-disk fallback must not strip every candidate.
def test_smallest_fitting_variant_uses_the_listing_aware_drafter_set(monkeypatch):
    """In a drafter-named repo the context-free predicate removed every candidate
    and the original disk-space error was raised over a quant that fits."""
    import huggingface_hub

    from core.inference.llama_cpp import LlamaCppBackend

    sizes = {
        "DFlash-Qwen3.5-27B-Uncensored.Q4_K_M.gguf": 100,
        "DFlash-Qwen3.5-27B-Uncensored.Q8_0.gguf": 900,
    }
    monkeypatch.setattr(
        huggingface_hub, "list_repo_files", lambda repo, token = None: list(sizes), raising = False
    )
    monkeypatch.setattr(
        huggingface_hub,
        "get_paths_info",
        lambda repo, paths, token = None: [
            SimpleNamespace(path = path, size = sizes[path]) for path in paths
        ],
        raising = False,
    )

    assert LlamaCppBackend._find_smallest_fitting_variant("org/DFlash-GGUF", 500) == (
        "DFlash-Qwen3.5-27B-Uncensored.Q4_K_M.gguf",
        100,
        [],
    )


def test_smallest_fitting_variant_still_skips_real_companions(monkeypatch):
    """Negative control: beside a real weight the drafter is a companion again."""
    import huggingface_hub

    from core.inference.llama_cpp import LlamaCppBackend

    sizes = {
        "model-Q4_K_M.gguf": 400,
        "mtp-model-Q8_0.gguf": 100,
        "mmproj-F16.gguf": 50,
    }
    monkeypatch.setattr(
        huggingface_hub, "list_repo_files", lambda repo, token = None: list(sizes), raising = False
    )
    monkeypatch.setattr(
        huggingface_hub,
        "get_paths_info",
        lambda repo, paths, token = None: [
            SimpleNamespace(path = path, size = sizes[path]) for path in paths
        ],
        raising = False,
    )

    assert LlamaCppBackend._find_smallest_fitting_variant("org/Model-GGUF", 500) == (
        "model-Q4_K_M.gguf",
        400,
        [],
    )


# 6. Only a kind that also names a model family can be reprieved. This is what
# keeps a drafter from becoming its own --model-draft: MTP is the one kind
# Studio launches, and an mtp- file is never promoted to a main weight.
MTP_ONLY_SIBLINGS = [
    _sib("mtp-model-Q8_0.gguf", 64, "drafter"),
    SimpleNamespace(rfilename = "README.md", size = 1, lfs = None),
]


def test_an_mtp_drafter_is_never_reprieved_into_its_own_model():
    """Reprieving this would advertise it as Q8_0 while drafter discovery still
    picked it, launching -m X --model-draft X."""
    from hub.utils.gguf import drafter_paths_in, is_reprievable_drafter_path

    assert not is_reprievable_drafter_path("mtp-model-Q8_0.gguf")
    assert drafter_paths_in([s.rfilename for s in MTP_ONLY_SIBLINGS]) == frozenset(
        {"mtp-model-Q8_0.gguf"}
    )
    # So it is a companion, never a variant, and the two can never collide.
    assert build_gguf_variant_plans(MTP_ONLY_SIBLINGS) == {}
    assert preferred_mtp_sibling(MTP_ONLY_SIBLINGS) is MTP_ONLY_SIBLINGS[0]


def test_a_reprieved_drafter_is_not_an_mtp_drafter_candidate(tmp_path, monkeypatch):
    """The kinds that CAN be reprieved are not the kind Studio launches, so the
    offline reuse path finds no drafter to pair a dflash weight with."""
    from core.inference.llama_cpp import LlamaCppBackend

    _snapshot(tmp_path, "org/DFlash-Model-GGUF", {"dflash-model-Q8_0.gguf": 64})
    _use_cache(monkeypatch, tmp_path)

    backend = LlamaCppBackend()
    assert backend._cached_repo_mtp_drafter("org/DFlash-Model-GGUF") is None


def test_launch_refuses_a_drafter_equal_to_the_model(tmp_path):
    """Last line of defence, whatever resolved the path (extras, a stale cache):
    llama-server must never be handed -m X --model-draft X."""
    from core.inference.llama_cpp import LlamaCppBackend

    model = tmp_path / "mtp-model-Q8_0.gguf"
    model.write_bytes(b"x" * 64)
    backend = LlamaCppBackend()

    assert (
        backend._resolve_launch_mtp_path(mtp_draft_path = str(model), model_path = str(model)) is None
    )
    flags = backend._build_speculative_flags(
        speculative_type = "mtp",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "org/MTP-Model-GGUF",
        model_path = str(model),
        gpus = False,
        binary = None,
        mtp_draft_path = str(model),
    )
    assert "--model-draft" not in flags


def test_launch_keeps_a_genuine_separate_drafter(tmp_path):
    """Negative control: a real companion beside a different model still loads."""
    from core.inference.llama_cpp import LlamaCppBackend

    model = tmp_path / "gemma-4-12b-it-Q4_K_M.gguf"
    model.write_bytes(b"x" * 64)
    drafter = tmp_path / "mtp-gemma-4-12b-it.gguf"
    drafter.write_bytes(b"x" * 8)
    backend = LlamaCppBackend()

    assert backend._resolve_launch_mtp_path(
        mtp_draft_path = str(drafter), model_path = str(model)
    ) == str(drafter)
