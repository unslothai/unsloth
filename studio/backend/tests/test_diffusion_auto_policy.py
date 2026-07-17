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
    # flux.2-klein's family default is the 4B base; loading the 9B GGUF passes the 9B
    # base repo, whose transformer is more than twice the size.
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
    # A pre-quantized checkpoint loads via the meta device: dense bf16 never lands on
    # the GPU, so the build peak IS the quantised size.
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
    # Neutralize the cache-disk gate by default so resolution tests are independent of the
    # runner's free space (a small CI disk otherwise drops the candidate). The two disk-gate
    # tests re-patch this after calling the helper to exercise the gate explicitly.
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
    # The dense artifact may be a multi-GB download; a nearly-full model-cache disk
    # drops the candidate (the loader then keeps the GGUF build).
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


def test_candidate_none_for_an_unlisted_family(monkeypatch):
    # No size entry -> no basis to re-plan; the loader keeps today's resident-only gate.
    _patch_selector(monkeypatch)
    assert (
        resolve_dense_quant_candidate(fam = _fam("not-a-family"), target = object(), requested = "auto")
        is None
    )


def test_candidate_uses_prequant_transient_when_available(monkeypatch):
    # A hosted-repo prequant source (kind="repo") is available without a local-path check.
    _patch_selector(monkeypatch, prequant = SimpleNamespace(kind = "repo", location = "org/int8"))
    est = resolve_dense_quant_candidate(fam = _fam("z-image"), target = object(), requested = "int8")
    assert est is not None and est.prequant is True
    assert est.transient_transformer_mib == est.steady_transformer_mib


# ── the ordering-fix regression, at the planner level ─────────────────────────
def _cuda_target():
    return SimpleNamespace(device = "cuda", supports_model_cpu_offload = True)


def test_quant_candidate_fits_resident_where_gguf_plan_offloads():
    # The ordering-fix mechanism, on a 32 GiB consumer card (RTX 5090 class): the user
    # picked a LARGE GGUF (the BF16 file), so the file-size plan forces offload -- but
    # the dense-quant candidate is far smaller (int8 prequant of z-image: the transient
    # IS the quantised size), and re-planning against the candidate keeps everything
    # resident. Before the fix the loader never attempted the fast path here.
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
