# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for the diffusion auto-policy decision layer.

Covers the per-family footprint estimator (bf16-resident component sizes x per-scheme
factors, transient vs steady, base-repo overrides), the dense-quant candidate resolution
(with the quant selector / prequant probe monkeypatched, no torch), and the resolved
provenance record. The loader-side ordering fix is exercised through the planner: the
regression case is a GGUF whose file-size plan forces offload while the candidate's
estimate fits resident."""

from __future__ import annotations

from types import SimpleNamespace

import core.inference.diffusion_auto_policy as ap
from core.inference.diffusion_auto_policy import (
    DenseQuantEstimate,
    build_resolved_record,
    estimate_dense_quant,
    family_bf16_components_gb,
    resolve_dense_quant_candidate,
)
from core.inference.diffusion_memory import (
    OFFLOAD_NONE,
    DeviceMemory,
    plan_diffusion_memory,
)


def _fam(name = "z-image"):
    return SimpleNamespace(name = name)


# ── the per-family table ──────────────────────────────────────────────────────
def test_family_table_covers_the_dit_families():
    for name in ("flux.1", "flux.2-klein", "flux.2-dev", "qwen-image", "z-image", "krea-2"):
        comps = family_bf16_components_gb(_fam(name))
        assert comps is not None, f"{name} missing from the bf16 component table"
        transformer, text_encoders, vae = comps
        assert transformer > 1.0 and text_encoders > 0.0 and vae > 0.0


def test_family_table_unknown_family_returns_none():
    assert family_bf16_components_gb(_fam("not-a-family")) is None


def test_base_repo_override_wins_over_the_family_default():
    # flux.2-klein's family default is the 4B base; loading the 9B GGUF passes the 9B base repo, whose transformer is over twice the size.
    default = family_bf16_components_gb(_fam("flux.2-klein"))
    nine_b = family_bf16_components_gb(
        _fam("flux.2-klein"), base_repo = "black-forest-labs/FLUX.2-klein-9B"
    )
    assert nine_b is not None and default is not None
    assert nine_b[0] > 2 * default[0]


# ── the estimator ─────────────────────────────────────────────────────────────
def test_estimate_int8_steady_is_roughly_half_bf16():
    est = estimate_dense_quant(_fam("z-image"), "int8")
    assert est is not None
    bf16_mib = 12.3 * ap._MIB_PER_GB
    assert 0.5 * bf16_mib < est.steady_transformer_mib < 0.6 * bf16_mib
    # On-the-fly quantisation transiently materialises the dense bf16 transformer.
    assert est.transient_transformer_mib == int(bf16_mib)
    assert est.prequant is False


def test_estimate_prequant_transient_equals_steady():
    # A pre-quantized checkpoint loads via the meta device: dense bf16 never lands on the GPU, so the build peak IS the quantised size.
    est = estimate_dense_quant(_fam("z-image"), "int8", prequant_available = True)
    assert est is not None
    assert est.transient_transformer_mib == est.steady_transformer_mib
    assert est.prequant is True


def test_estimate_nvfp4_is_smaller_than_int8():
    int8 = estimate_dense_quant(_fam("flux.1"), "int8")
    nvfp4 = estimate_dense_quant(_fam("flux.1"), "nvfp4")
    assert int8 is not None and nvfp4 is not None
    assert nvfp4.steady_transformer_mib < int8.steady_transformer_mib


def test_estimate_unknown_family_or_scheme_returns_none():
    assert estimate_dense_quant(_fam("not-a-family"), "int8") is None
    assert estimate_dense_quant(_fam("z-image"), "q4_k") is None


# ── candidate resolution (selector + prequant probe stubbed) ─────────────────
def _patch_selector(
    monkeypatch,
    *,
    supported = True,
    scheme = "int8",
    prequant = None,
):
    import core.inference.diffusion_transformer_quant as tq

    monkeypatch.setattr(tq, "dense_transformer_supported", lambda target: supported)
    monkeypatch.setattr(
        tq, "select_transformer_quant_scheme", lambda target, req, family = None: scheme
    )
    import core.inference.diffusion_prequant as pq

    monkeypatch.setattr(
        pq,
        "resolve_prequant_source",
        lambda fam, s, path_override = None, base_repo = None: prequant,
    )
    # Neutralize the cache-disk gate by default so resolution tests do not depend on runner free space; the disk-gate tests re-patch it.
    monkeypatch.setattr(ap, "_hf_cache_free_mib", lambda: None)


def test_candidate_resolves_for_a_supported_request(monkeypatch):
    _patch_selector(monkeypatch, scheme = "int8")
    est = resolve_dense_quant_candidate(fam = _fam("z-image"), target = object(), requested = "auto")
    assert isinstance(est, DenseQuantEstimate)
    assert est.scheme == "int8"
    assert est.transient_transformer_mib > est.steady_transformer_mib


def test_candidate_none_when_request_is_off(monkeypatch):
    _patch_selector(monkeypatch)
    for off in (None, "", "none", "off"):
        assert resolve_dense_quant_candidate(fam = _fam(), target = object(), requested = off) is None


def test_candidate_none_when_device_unsupported(monkeypatch):
    _patch_selector(monkeypatch, supported = False)
    assert resolve_dense_quant_candidate(fam = _fam(), target = object(), requested = "auto") is None


def test_candidate_none_when_no_scheme_resolves(monkeypatch):
    _patch_selector(monkeypatch, scheme = None)
    assert resolve_dense_quant_candidate(fam = _fam(), target = object(), requested = "auto") is None


def test_candidate_disk_gate_skips_when_cache_disk_low(monkeypatch):
    # The dense artifact may be a multi-GB download, so a nearly-full model-cache disk drops the candidate and the loader keeps the GGUF build.
    import core.inference.diffusion_auto_policy as ap

    _patch_selector(monkeypatch, scheme = "int8")
    monkeypatch.setattr(ap, "_hf_cache_free_mib", lambda: 1024)
    assert (
        resolve_dense_quant_candidate(fam = _fam("z-image"), target = object(), requested = "auto")
        is None
    )


def test_candidate_disk_gate_unprobeable_disk_passes(monkeypatch):
    # Disk probing must never sink the candidate: unprobeable (None) passes through.
    import core.inference.diffusion_auto_policy as ap

    _patch_selector(monkeypatch, scheme = "int8")
    monkeypatch.setattr(ap, "_hf_cache_free_mib", lambda: None)
    est = resolve_dense_quant_candidate(fam = _fam("z-image"), target = object(), requested = "auto")
    assert isinstance(est, DenseQuantEstimate)


def test_disk_gate_sizes_fp32_families_by_their_real_download(monkeypatch):
    # The size table is bf16-RESIDENT but the disk gate is about bytes landing in the HF cache. Z-Image publishes fp32 shards
    # (23,479 MiB vs 11,730 MiB resident), so gating on the resident figure let the check pass and the download fill the disk.
    import core.inference.diffusion_auto_policy as ap

    _patch_selector(monkeypatch, scheme = "int8")
    est = ap.estimate_dense_quant(_fam("z-image"), "int8")
    assert est.transient_transformer_mib == 11_730  # resident: unchanged
    assert est.download_transformer_mib == 23_460  # download: measured 23,479 MiB
    # Free space that clears the old (resident-based) bar but not the real download is refused.
    monkeypatch.setattr(ap, "_hf_cache_free_mib", lambda: 11_730 + 10 * 1024 + 512)
    assert (
        resolve_dense_quant_candidate(fam = _fam("z-image"), target = object(), requested = "auto")
        is None
    )
    monkeypatch.setattr(ap, "_hf_cache_free_mib", lambda: 23_460 + 10 * 1024 + 512)
    assert isinstance(
        resolve_dense_quant_candidate(fam = _fam("z-image"), target = object(), requested = "auto"),
        DenseQuantEstimate,
    )


def test_disk_gate_does_not_overcharge_a_family_published_below_bf16(monkeypatch):
    # The correction runs both ways: Ideogram 4 ships fp8 (17,718 MiB measured) and doubles on the way to bf16, so charging the
    # resident 35,477 MiB of disk would refuse a candidate the disk easily holds.
    import core.inference.diffusion_auto_policy as ap

    _patch_selector(monkeypatch, scheme = "int8")
    est = ap.estimate_dense_quant(_fam("ideogram-4"), "int8")
    assert est.transient_transformer_mib == 35_476
    assert est.download_transformer_mib == 17_738
    monkeypatch.setattr(ap, "_hf_cache_free_mib", lambda: 17_738 + 10 * 1024 + 512)
    assert isinstance(
        resolve_dense_quant_candidate(fam = _fam("ideogram-4"), target = object(), requested = "auto"),
        DenseQuantEstimate,
    )


def test_disk_gate_matches_download_for_bf16_published_families(monkeypatch):
    # Families that publish bf16 download what they occupy (measured 0.99-1.07x), so the two numbers stay equal and need no factor.
    import core.inference.diffusion_auto_policy as ap
    for name in ("flux.1", "flux.2-dev", "qwen-image", "krea-2", "hidream-i1"):
        est = ap.estimate_dense_quant(_fam(name), "int8")
        assert est.download_transformer_mib == est.transient_transformer_mib, name


def test_candidate_none_for_an_unlisted_family(monkeypatch):
    # No size entry means no basis to re-plan; the loader keeps today's resident-only gate.
    _patch_selector(monkeypatch)
    assert (
        resolve_dense_quant_candidate(fam = _fam("not-a-family"), target = object(), requested = "auto")
        is None
    )


def test_candidate_uses_prequant_transient_when_available(monkeypatch):
    # A hosted-repo prequant source is available without a local-path check.
    _patch_selector(monkeypatch, prequant = SimpleNamespace(kind = "repo", location = "org/int8"))
    est = resolve_dense_quant_candidate(fam = _fam("z-image"), target = object(), requested = "int8")
    assert est is not None and est.prequant is True
    assert est.transient_transformer_mib == est.steady_transformer_mib


def test_disk_gate_reserves_the_checkpoint_not_the_dense_shards_for_a_hosted_prequant(monkeypatch):
    # The gate branch the dense-sizing tests never reach: with a prequant source the loader downloads the small quantised
    # checkpoint, so free space between the steady and dense sizes must pass with the shortcut and be refused without it.
    _patch_selector(
        monkeypatch,
        scheme = "int8",
        prequant = SimpleNamespace(kind = "repo", location = "org/int8"),
    )
    est = ap.estimate_dense_quant(_fam("z-image"), "int8", prequant_available = True)
    assert est.steady_transformer_mib == 6_451  # the quantised checkpoint
    assert est.download_transformer_mib == 23_460  # the fp32 shards it replaces
    monkeypatch.setattr(ap, "_hf_cache_free_mib", lambda: 6_451 + 10 * 1024 + 512)
    gated = resolve_dense_quant_candidate(fam = _fam("z-image"), target = object(), requested = "int8")
    assert isinstance(gated, DenseQuantEstimate) and gated.prequant is True
    # force_dense (a LoRA bake) skips the shortcut, so the SAME disk must refuse the candidate.
    assert (
        resolve_dense_quant_candidate(
            fam = _fam("z-image"), target = object(), requested = "int8", force_dense = True
        )
        is None
    )


# ── the ordering-fix regression, at the planner level ─────────────────────────
def _cuda_target():
    return SimpleNamespace(device = "cuda", supports_model_cpu_offload = True)


def test_quant_candidate_fits_resident_where_gguf_plan_offloads():
    # The ordering fix on a 32 GiB consumer card: the user picked a LARGE (BF16) GGUF so the file-size plan forces offload, but
    # the dense-quant candidate is far smaller and re-planning against it keeps everything resident.
    memory = DeviceMemory("cuda", "cuda", "discrete_vram", 30000, 32768)
    z_bf16_gguf_mib = int(12.3 * ap._MIB_PER_GB * 1.05)  # BF16 GGUF resident estimate
    companions_mib = 2600  # fp8-quantised text encoders + VAE
    gguf_plan = plan_diffusion_memory(
        target = _cuda_target(),
        device_memory = memory,
        model_dense_mib = z_bf16_gguf_mib + companions_mib,
        companion_dense_mib = companions_mib,
        runtime_headroom_mib = 6963,
    )
    assert gguf_plan.offload_policy != OFFLOAD_NONE

    est = estimate_dense_quant(_fam("z-image"), "int8", prequant_available = True)
    assert est is not None
    assert est.transient_transformer_mib < z_bf16_gguf_mib / 1.8
    quant_plan = plan_diffusion_memory(
        target = _cuda_target(),
        device_memory = memory,
        model_dense_mib = est.transient_transformer_mib + companions_mib,
        companion_dense_mib = companions_mib,
        runtime_headroom_mib = 6963,
    )
    assert quant_plan.offload_policy == OFFLOAD_NONE


# ── the resolved provenance record ────────────────────────────────────────────
def test_resolved_record_marks_auto_and_explicit():
    record = build_resolved_record(
        {
            "speed_mode": (None, "default", "per-kind default"),
            "transformer_quant": ("auto", "fp8", "auto ladder"),
            "attention_backend": ("cudnn", "_native_cudnn", "requested"),
            "memory_mode": ("", "none", "planned"),
            "cpu_offload": (True, True, "legacy flag"),
        }
    )
    assert record["speed_mode"]["source"] == "auto"
    assert record["transformer_quant"]["source"] == "auto"  # "auto" delegates to backend
    assert record["attention_backend"]["source"] == "explicit"
    assert record["memory_mode"]["source"] == "auto"  # blank string delegates
    assert record["cpu_offload"]["source"] == "explicit"
    assert record["transformer_quant"]["value"] == "fp8"
    assert all("reason" in v for v in record.values())
