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

import pytest

from types import SimpleNamespace

import core.inference.diffusion_auto_policy as ap


@pytest.fixture(autouse = True)
def _assume_the_restricted_load_is_available(monkeypatch):
    """Policy/planning tests, not a check on whether this host's torchao imports.

    Without this, a machine with no (or a skewed) torchao turns every hosted-prequant decision
    below into "keep the dense weights". The capability is covered in test_diffusion_prequant.py."""
    import core.inference.diffusion_prequant as _pq
    monkeypatch.setattr(_pq, "restricted_prequant_load_supported", lambda scheme = None: True)


from core.inference.diffusion_auto_policy import (
    DenseQuantEstimate,
    build_resolved_record,
    estimate_dense_quant,
    family_bf16_components_gb,
    precision_fallback_allowed,
    precision_refusal_message,
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


def test_zimage_base_downloads_bf16_while_turbo_downloads_fp32():
    # The z-image family factor exists because the distilled Turbo publishes fp32 shards. The
    # undistilled base ships bf16 and downloads exactly what it occupies, so charging it the
    # family factor makes the free-disk gate demand twice the real size.
    assert ap.hub_download_factor(_fam("z-image")) == 2.0
    assert ap.hub_download_factor(_fam("z-image"), "Tongyi-MAI/Z-Image-Turbo") == 2.0
    assert ap.hub_download_factor(_fam("z-image"), "Tongyi-MAI/Z-Image") == 1.0
    # A family with no factor at all still defaults to 1.0, base repo or not.
    assert ap.hub_download_factor(_fam("flux.2-klein"), "black-forest-labs/FLUX.2-klein-4B") == 1.0
    # A base repo arrives however the user typed it, so the override cannot be case-sensitive.
    assert ap.hub_download_factor(_fam("z-image"), "  tongyi-mai/Z-IMAGE ") == 1.0

    turbo = estimate_dense_quant(_fam("z-image"), "int8", base_repo = "Tongyi-MAI/Z-Image-Turbo")
    base = estimate_dense_quant(_fam("z-image"), "int8", base_repo = "Tongyi-MAI/Z-Image")
    assert turbo is not None and base is not None
    # Same architecture, so the resident footprint is identical and only the download differs.
    assert base.steady_transformer_mib == turbo.steady_transformer_mib
    assert base.download_transformer_mib * 2 == turbo.download_transformer_mib


def test_base_repo_override_wins_over_the_family_default():
    # flux.2-klein's family default is the 4B base; loading the 9B GGUF passes the 9B base repo, whose transformer is over twice the size.
    default = family_bf16_components_gb(_fam("flux.2-klein"))
    nine_b = family_bf16_components_gb(
        _fam("flux.2-klein"), base_repo = "black-forest-labs/FLUX.2-klein-9B"
    )
    assert nine_b is not None and default is not None
    assert nine_b[0] > 2 * default[0]

    base_nine_b = family_bf16_components_gb(
        _fam("flux.2-klein"), base_repo = "unsloth/FLUX.2-klein-base-9B"
    )
    assert base_nine_b == nine_b


def test_klein_base_9b_is_sized_like_the_9b_not_the_4b():
    # klein-base-9B is the undistilled 9B (18.2 GB transformer + the Qwen3-8B encoder), and it is
    # the variant upstream points fine-tuning at, so it needs the same override as klein-9B.
    # Without it the base 9B is planned as a 4B and every size-driven decision under-reserves.
    default = family_bf16_components_gb(_fam("flux.2-klein"))
    nine_b = family_bf16_components_gb(
        _fam("flux.2-klein"), base_repo = "black-forest-labs/FLUX.2-klein-9B"
    )
    base_9b = family_bf16_components_gb(
        _fam("flux.2-klein"), base_repo = "black-forest-labs/FLUX.2-klein-base-9B"
    )
    assert base_9b == nine_b
    assert base_9b is not None and default is not None and base_9b[0] > 2 * default[0]
    # The unsloth mirror is what Unsloth actually loads, and canonical_base has to route it here too.
    assert (
        family_bf16_components_gb(_fam("flux.2-klein"), base_repo = "unsloth/FLUX.2-klein-base-9B")
        == base_9b
    )
    # Same case-insensitivity the trust gate applies: a typed-in base must not fall back to the 4B.
    assert (
        family_bf16_components_gb(
            _fam("flux.2-klein"), base_repo = " BLACK-FOREST-LABS/flux.2-klein-base-9b "
        )
        == base_9b
    )


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


def test_candidate_disk_gate_spares_an_already_cached_prequant(monkeypatch):
    """A cached checkpoint downloads nothing, so the space gate has no claim on it.

    The gate's own comment said a cached re-download is a no-op, but it ran regardless. That
    discarded exactly the candidate the auto retry exists to find: the retry only ever proposes a
    rung whose checkpoint is already cached, so on a low-disk or moved-cache install every retry
    fell back to the GGUF despite a resident-fit local artifact.
    """
    import core.inference.diffusion_auto_policy as ap
    import core.inference.diffusion_prequant as pq

    _patch_selector(monkeypatch, scheme = "int8")
    monkeypatch.setattr(
        pq, "usable_prequant_source", lambda *a, **k: type("S", (), {"kind": "repo"})()
    )
    monkeypatch.setattr(pq, "prequant_checkpoint_cached", lambda *a, **k: True)
    # Far too little space for the checkpoint, which is precisely the case being excused.
    monkeypatch.setattr(ap, "_hf_cache_free_mib", lambda: 1024)
    est = resolve_dense_quant_candidate(fam = _fam("z-image"), target = object(), requested = "auto")
    assert isinstance(est, DenseQuantEstimate)
    assert est.prequant is True

    # Uncached on the same low disk still trips the gate: this excuses a cached artifact, not
    # every prequant.
    monkeypatch.setattr(pq, "prequant_checkpoint_cached", lambda *a, **k: False)
    assert (
        resolve_dense_quant_candidate(fam = _fam("z-image"), target = object(), requested = "auto")
        is None
    )


def test_a_local_override_is_never_gated_on_disk_space(monkeypatch):
    """A local path override downloads nothing, so the space gate has no claim on it.

    prequant_checkpoint_cached only answers for hosted repos (_cached_in_root returns None for any
    other kind), so probing a path source there reports False and re-applies the gate to a file
    already on disk. The retry treats a local override as costing no bytes; this has to agree, or a
    low-disk host drops the local rung to GGUF.
    """
    import core.inference.diffusion_auto_policy as ap
    import core.inference.diffusion_prequant as pq

    _patch_selector(monkeypatch, scheme = "int8")
    monkeypatch.setattr(
        pq, "usable_prequant_source", lambda *a, **k: type("S", (), {"kind": "path"})()
    )
    # Would say "not cached" for a path source, exactly as the real one does.
    monkeypatch.setattr(pq, "prequant_checkpoint_cached", lambda *a, **k: False)
    monkeypatch.setattr(ap, "_hf_cache_free_mib", lambda: 1024)
    est = resolve_dense_quant_candidate(fam = _fam("z-image"), target = object(), requested = "auto")
    assert isinstance(est, DenseQuantEstimate)


def test_the_cached_probe_is_pinned_to_the_active_cache_root(monkeypatch):
    """Unpinned, cached_checkpoint_path reads only huggingface_hub's import-time constant.

    Unsloth's cache folder is a setting, so after it changes the retry proves the checkpoint cached
    in the LIVE root while an unpinned probe here still calls it uncached and re-applies the gate,
    defeating the moved-cache retry this excuse exists for. The retry and the loader both pin the
    active root; so must this.
    """
    import core.inference.diffusion_auto_policy as ap
    import core.inference.diffusion_prequant as pq
    import utils.hf_cache_settings as cache_settings

    seen: list = []

    _patch_selector(monkeypatch, scheme = "int8")
    monkeypatch.setattr(
        pq, "usable_prequant_source", lambda *a, **k: type("S", (), {"kind": "repo"})()
    )
    monkeypatch.setattr(cache_settings, "active_hf_hub_cache", lambda: "/live-root")
    monkeypatch.setattr(
        pq,
        "prequant_checkpoint_cached",
        lambda _src, **kw: (seen.append(kw.get("cache_dir")), True)[1],
    )
    monkeypatch.setattr(ap, "_hf_cache_free_mib", lambda: 1024)
    resolve_dense_quant_candidate(fam = _fam("z-image"), target = object(), requested = "auto")
    assert seen == ["/live-root"], seen


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
    # The correction runs both ways: Ideogram 4 ships fp8 (17,718 MiB measured) and doubles on the way to bf16, so charging the resident 35,477 MiB would refuse a fine candidate.
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


def test_resolved_record_keeps_the_request_beside_the_engaged_value():
    # P1-2: a declined explicit precision must not collapse into a bare source="explicit" that
    # renders no badge while the dropdown still advertises the ask.
    record = build_resolved_record(
        {
            "transformer_quant": ("fp8", "off", "the dense build did not fit", "fell_back"),
            "text_encoder_quant": ("int8", "fp8", "no keep-bf16 schedule"),
            "memory_mode": ("low_vram", "sequential", "planned"),
        }
    )
    assert record["transformer_quant"]["requested"] == "fp8"
    assert record["transformer_quant"]["value"] == "off"
    assert record["transformer_quant"]["status"] == "fell_back"
    # Derived without the call site classifying it: request and engaged value disagree.
    assert record["text_encoder_quant"]["status"] == "fell_back"
    # ...but only where the two share a vocabulary. memory_mode requests a MODE and engages an
    # offload POLICY, so an honored request must not be reported as a fallback.
    assert record["memory_mode"]["status"] == "applied"


def test_resolved_record_treats_an_honored_off_request_as_applied():
    # "none"/"off"/"" all ask for no quant, which an "off" engagement satisfies.
    record = build_resolved_record(
        {
            "transformer_quant": ("none", "off", "GGUF loaded"),
            "text_encoder_quant": ("fp8", "fp8", "cast in place"),
            "cpu_offload": (True, True, "legacy flag"),
        }
    )
    assert record["transformer_quant"]["status"] == "applied"
    assert record["text_encoder_quant"]["status"] == "applied"
    assert record["cpu_offload"]["status"] == "applied"


def test_resolved_record_auto_never_reports_a_fallback():
    # An auto request delegates the choice, so a decline is the ladder working: no ask to betray.
    record = build_resolved_record({"transformer_quant": (None, "off", "no CUDA")})
    assert record["transformer_quant"]["source"] == "auto"
    assert record["transformer_quant"]["requested"] is None
    assert record["transformer_quant"]["status"] == "applied"


def test_precision_fallback_escape_hatch(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DIFFUSION_ALLOW_PRECISION_FALLBACK", raising = False)
    assert precision_fallback_allowed() is False
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ALLOW_PRECISION_FALLBACK", "1")
    assert precision_fallback_allowed() is True
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ALLOW_PRECISION_FALLBACK", "0")
    assert precision_fallback_allowed() is False


def test_the_refusal_only_offers_auto_where_auto_exists():
    # transformer_quant has an "auto" mode, so pointing the user at it is the right advice.
    msg = precision_refusal_message(
        "transformer_quant",
        "nvfp4",
        "this GPU has no fp4 tensor cores",
        off_label = "Off to run the checkpoint as-is",
    )
    assert "Choose Auto" in msg and msg.endswith("or Off to run the checkpoint as-is.")

    # text_encoder_quant does not: both request models restrict it to fp8 / fp8_dynamic / int8 /
    # nvfp4, so a user who followed "Choose Auto" here got a 422 from request validation. The
    # remedy has to name the thing that actually works.
    te = precision_refusal_message(
        "text_encoder_quant",
        "int8",
        "this device does not have the tensor cores that backend needs",
        off_label = "leave it unset to keep the dense bf16 encoder",
        auto_available = False,
    )
    assert "Auto" not in te
    assert te.endswith("Leave it unset to keep the dense bf16 encoder.")
