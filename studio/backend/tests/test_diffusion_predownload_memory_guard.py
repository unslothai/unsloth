# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic tests for the pre-download unified-memory guard from issue #9130.

Hub sizes recorded on 2026-08-24 are aggregated by component directory.
"""

from __future__ import annotations

import types

import pytest

from core.inference import diffusion as diffusion_mod
from core.inference.diffusion import DiffusionBackend
from core.inference.diffusion_device import DiffusionDeviceTarget
from core.inference.diffusion_families import detect_family_for_pick
from core.inference.diffusion_memory import DeviceMemory

MIB = 1024 * 1024

REPORTER_POOL_MIB = 104424
STRIX_HALO_POOL_MIB = 64 * 1024


def _files(**dirs: int) -> list:
    return [(f"{name}/model.safetensors", mib * MIB) for name, mib in dirs.items()]


# unsloth/Qwen-Image-2512 @ b96dde7f, 57.70 GB total.
QWEN_IMAGE_2512 = _files(transformer = 38966, text_encoder = 15812, vae = 240, tokenizer = 10)

FLUX2_DEV = _files(transformer = 61461, text_encoder = 45798, vae = 321, tokenizer = 16)

Z_IMAGE_TURBO = _files(transformer = 23479, text_encoder = 7672, vae = 160, tokenizer = 15)

LUMINA_2 = _files(transformer = 9956, text_encoder = 9973, vae = 320, tokenizer = 21)

IDEOGRAM_4_FP8 = _files(
    transformer = 8859,
    unconditional_transformer = 8859,
    text_encoder = 8373,
    vae = 160,
    tokenizer = 11,
)

IDEOGRAM_4_NF4 = _files(
    transformer = 4980,
    unconditional_transformer = 4980,
    text_encoder = 5230,
    vae = 160,
    tokenizer = 11,
)


LUMINA_2_MINUS_DENSE_TE = _files(transformer = 9956, vae = 320, tokenizer = 21)
LUMINA_HOSTED_TE_MIB = 3056
LUMINA_HOSTED_TE = {
    "text_encoder": (
        "unsloth/Lumina-Image-2.0-fp8-te",
        [("text_encoder_fp8.pt", LUMINA_HOSTED_TE_MIB * MIB)],
    )
}


def _target(*, device = "cuda", ordinal = None) -> DiffusionDeviceTarget:
    return DiffusionDeviceTarget(
        device = device,
        dtype = "bfloat16",
        backend = device,
        vendor = "amd",
        supports_model_cpu_offload = True,
        supports_default_torch_compile = False,
        supports_pinned_transfer = True,
        ordinal = ordinal,
    )


def _family(name = "qwen-image", base_repo = "Qwen/Qwen-Image"):
    return types.SimpleNamespace(name = name, base_repo = base_repo)


def _flux2(name = "flux.2-dev", base_repo = "black-forest-labs/FLUX.2-dev"):
    return _family(name = name, base_repo = base_repo)


def _real_family(repo_id = "unsloth/FLUX.2-dev"):
    fam = detect_family_for_pick(repo_id, None, None)
    assert fam is not None
    return fam


def _backend(
    monkeypatch,
    *,
    memory_kind = "unified_memory",
    total_mib = REPORTER_POOL_MIB,
    free_mib = None,
    device = "cuda",
):
    backend = DiffusionBackend()
    target = _target(device = device)
    monkeypatch.setattr(backend, "_target_for_ordinal", lambda *_a, **_k: target)
    snapshot = DeviceMemory(
        device,
        device,
        memory_kind,
        free_mib if free_mib is not None else total_mib,
        total_mib,
    )
    monkeypatch.setattr(diffusion_mod, "snapshot_device_memory", lambda _t: snapshot)
    return backend


def _verdict(backend, fam, repo, base, files):
    return backend.declared_footprint_shortfall(
        fam, repo, base, kind = "pipeline", declared_files = files
    )


def _flux_verdict(backend):
    return _verdict(
        backend, _flux2(), "unsloth/FLUX.2-dev", "black-forest-labs/FLUX.2-dev", FLUX2_DEV
    )


def test_reporter_pool_accepts_named_qwen_but_rejects_flux(monkeypatch):
    backend = _backend(monkeypatch)
    assert _flux_verdict(backend) is not None
    assert (
        _verdict(
            backend,
            _family(),
            "unsloth/Qwen-Image-2512",
            "unsloth/Qwen-Image-2512",
            QWEN_IMAGE_2512,
        )
        is None
    )


def test_a_pipeline_that_cannot_fit_is_refused_from_metadata_alone(monkeypatch):
    backend = _backend(monkeypatch, total_mib = STRIX_HALO_POOL_MIB)
    message = _flux_verdict(backend)
    assert message is not None
    assert "flux.2-dev" in message
    assert "about 107 GB of memory for its weights" in message
    assert "about 51 GB is usable" in message
    assert "currently free" not in message
    assert "UNSLOTH_DIFFUSION_ALLOW_OVERSIZED_LOAD=1" in message


def test_a_mixed_precision_repo_is_sized_component_by_component(monkeypatch):
    fam = _family(name = "z-image", base_repo = "Tongyi-MAI/Z-Image-Turbo")
    backend = _backend(monkeypatch, total_mib = 24 * 1024)
    message = _verdict(
        backend, fam, "unsloth/Z-Image-Turbo", "Tongyi-MAI/Z-Image-Turbo", Z_IMAGE_TURBO
    )
    assert message is not None
    assert "about 21 GB of memory for its weights" in message


def test_a_raw_fp8_repo_converts_its_encoder_as_well_as_its_denoisers(monkeypatch):
    fam = _family(name = "ideogram-4", base_repo = "ideogram-ai/ideogram-4-fp8")
    tight = _backend(monkeypatch, total_mib = STRIX_HALO_POOL_MIB)
    message = _verdict(
        tight, fam, "unsloth/ideogram-4-fp8", "ideogram-ai/ideogram-4-fp8", IDEOGRAM_4_FP8
    )
    assert message is not None
    assert "about 53 GB of memory for its weights" in message


def test_an_fp32_repo_halves_its_companions_too(monkeypatch):
    fam = _family(name = "lumina-2", base_repo = "Alpha-VLLM/Lumina-Image-2.0")
    backend = _backend(monkeypatch, total_mib = 16 * 1024)
    assert (
        _verdict(backend, fam, "unsloth/Lumina-Image-2.0", "Alpha-VLLM/Lumina-Image-2.0", LUMINA_2)
        is None
    )
    tiny = _backend(monkeypatch, total_mib = 12 * 1024)
    assert (
        _verdict(tiny, fam, "unsloth/Lumina-Image-2.0", "Alpha-VLLM/Lumina-Image-2.0", LUMINA_2)
        is not None
    )


def test_an_nf4_sibling_of_the_same_family_is_not_inflated(monkeypatch):
    fam = _family(name = "ideogram-4", base_repo = "ideogram-ai/ideogram-4-fp8")
    backend = _backend(monkeypatch, total_mib = 32 * 1024)
    for base in ("ideogram-ai/ideogram-4-nf4-diffusers", "ideogram-ai/ideogram-4-nf4"):
        name = base.split("/")[1]
        assert _verdict(backend, fam, f"unsloth/{name}", base, IDEOGRAM_4_NF4) is None, base
    tight = _backend(monkeypatch, total_mib = 32 * 1024)
    assert (
        _verdict(tight, fam, "unsloth/ideogram-4-fp8", "ideogram-ai/ideogram-4-fp8", IDEOGRAM_4_FP8)
        is not None
    )


def test_a_hosted_pre_cast_encoder_is_not_converted_again(monkeypatch):
    fam = _family(name = "lumina-2", base_repo = "Alpha-VLLM/Lumina-Image-2.0")
    backend = _backend(monkeypatch, total_mib = 12 * 1024)
    message = backend.declared_footprint_shortfall(
        fam,
        "unsloth/Lumina-Image-2.0",
        "Alpha-VLLM/Lumina-Image-2.0",
        kind = "pipeline",
        declared_files = LUMINA_2_MINUS_DENSE_TE,
        prequant_bytes = LUMINA_HOSTED_TE_MIB * MIB,
    )
    assert message is not None
    assert "about 10 GB of memory for its weights" in message


def test_a_bf16_repo_is_counted_as_it_downloads():
    from core.inference.diffusion_auto_policy import resident_bytes_from_declared
    assert resident_bytes_from_declared(_family(), "Qwen/Qwen-Image", QWEN_IMAGE_2512) == sum(
        size for _name, size in QWEN_IMAGE_2512
    )


@pytest.mark.parametrize(
    ("family", "base"),
    [
        pytest.param("z-image", "unsloth/custom-z-image-bf16", id = "z-image"),
        pytest.param("lumina-2", "unsloth/custom-lumina-bf16", id = "lumina"),
    ],
)
def test_a_custom_pipeline_does_not_inherit_family_storage_precision(family, base):
    from core.inference.diffusion_auto_policy import resident_bytes_from_declared

    files = _files(transformer = 20_000, text_encoder = 8_000)
    assert resident_bytes_from_declared(_family(name = family, base_repo = base), base, files) == sum(
        size for _name, size in files
    )


def test_only_a_compatible_prequant_can_replace_dense_encoder_shards(monkeypatch):
    from core.inference import diffusion_te_prequant as te_prequant

    source = te_prequant.TePrequantSource(
        kind = "repo", location = "unsloth/Qwen-Image-FP8", filename = "encoder.pt"
    )
    monkeypatch.setattr(
        te_prequant,
        "te_prequant_sources",
        lambda *_a, **_k: {"text_encoder": source},
    )
    monkeypatch.setattr(
        te_prequant,
        "te_prequant_hub_files",
        lambda sources, *_a, **_k: {component: [("encoder.pt", 8 * MIB)] for component in sources},
    )
    monkeypatch.setattr(diffusion_mod, "resolve_diffusion_device_target", lambda *_a, **_k: _target())
    fam = _family(name = "qwen-image", base_repo = "Qwen/Qwen-Image")

    assert (
        DiffusionBackend._te_prequant_plan_files(
            fam, "fp8", None, base_repo = "unsloth/custom-qwen-image"
        )
        == {}
    )
    assert "text_encoder" in DiffusionBackend._te_prequant_plan_files(
        fam, "fp8", None, base_repo = "Qwen/Qwen-Image"
    )


def test_hidream_te4_uses_its_standalone_base_for_prequant_compatibility(monkeypatch):
    from core.inference import diffusion_te_prequant as te_prequant

    seen: dict = {}
    source = te_prequant.TePrequantSource(
        kind = "repo", location = "unsloth/HiDream-I1-Full-FP8", filename = "te4.pt"
    )

    def _sources(*_a, **kwargs):
        seen["components"] = tuple(kwargs["components"])
        return {"text_encoder_4": source}

    monkeypatch.setattr(te_prequant, "te_prequant_sources", _sources)
    monkeypatch.setattr(
        te_prequant,
        "te_prequant_hub_files",
        lambda sources, *_a, **_k: {component: [("te4.pt", 8 * MIB)] for component in sources},
    )
    monkeypatch.setattr(diffusion_mod, "resolve_diffusion_device_target", lambda *_a, **_k: _target())
    fam = _family(name = "hidream-i1", base_repo = "HiDream-ai/HiDream-I1-Full")

    planned = DiffusionBackend._te_prequant_plan_files(
        fam, "fp8", None, base_repo = "HiDream-ai/HiDream-I1-Dev"
    )
    assert "text_encoder_4" in seen["components"]
    assert "text_encoder_4" in planned


def test_hidream_dense_te4_is_counted_when_no_prequant_is_selected():
    from core.inference.diffusion import _predownload_encoder_bytes
    from core.inference.diffusion_hidream import HIDREAM_LLAMA_BF16_BYTES

    fam = _family(name = "hidream-i1", base_repo = "HiDream-ai/HiDream-I1-Full")
    assert _predownload_encoder_bytes(fam, {}) == HIDREAM_LLAMA_BF16_BYTES
    assert _predownload_encoder_bytes(fam, {}, pipeline_declared = False) == 0
    hosted = {"text_encoder_4": ("unsloth/HiDream-I1-Full-FP8", [("te4.pt", 8 * MIB)])}
    assert _predownload_encoder_bytes(fam, hosted) == 8 * MIB


def test_discrete_vram_is_never_refused(monkeypatch):
    backend = _backend(monkeypatch, memory_kind = "discrete_vram", total_mib = 24 * 1024)
    assert _flux_verdict(backend) is None


def test_the_verdict_is_capacity_not_the_free_reading(monkeypatch):
    backend = _backend(monkeypatch, free_mib = 4 * 1024)
    assert (
        _verdict(
            backend,
            _family(),
            "unsloth/Qwen-Image-2512",
            "unsloth/Qwen-Image-2512",
            QWEN_IMAGE_2512,
        )
        is None
    )


@pytest.mark.parametrize("kind", ["gguf", "single_file"])
def test_only_a_pipeline_pick_is_judged(monkeypatch, kind):
    backend = _backend(monkeypatch, total_mib = STRIX_HALO_POOL_MIB)
    assert (
        backend.declared_footprint_shortfall(
            _flux2(),
            "unsloth/FLUX.2-dev",
            "black-forest-labs/FLUX.2-dev",
            kind = kind,
            declared_files = FLUX2_DEV,
        )
        is None
    )


@pytest.mark.parametrize("files", [None, [], [("model_index.json", 0)]])
def test_no_sizes_is_no_verdict(monkeypatch, files):
    backend = _backend(monkeypatch, total_mib = STRIX_HALO_POOL_MIB)
    assert (
        _verdict(backend, _flux2(), "unsloth/FLUX.2-dev", "black-forest-labs/FLUX.2-dev", files)
        is None
    )


def test_an_unreadable_device_never_refuses(monkeypatch):
    backend = _backend(monkeypatch, total_mib = None)
    assert _flux_verdict(backend) is None


def test_a_probe_that_raises_never_refuses(monkeypatch):
    backend = _backend(monkeypatch, total_mib = STRIX_HALO_POOL_MIB)

    def _boom(_target):
        raise RuntimeError("no CUDA here")

    monkeypatch.setattr(diffusion_mod, "snapshot_device_memory", _boom)
    assert _flux_verdict(backend) is None


def test_the_escape_hatch_still_opens_it(monkeypatch):
    backend = _backend(monkeypatch, total_mib = STRIX_HALO_POOL_MIB)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ALLOW_OVERSIZED_LOAD", "1")
    assert _flux_verdict(backend) is None


def _stub_estimate(monkeypatch, files):
    def _estimate(*_a, **kwargs):
        out = kwargs.get("file_sizes_out")
        if out is not None:
            out["unsloth/FLUX.2-dev"] = {name: size for name, size in files}
        return sum(size for _name, size in files), []

    monkeypatch.setattr(DiffusionBackend, "_estimate_download_bytes", staticmethod(_estimate))


def _stub_pick(
    monkeypatch,
    files,
    *,
    family = None,
    te_prequant = None,
):
    fam = family if family is not None else _real_family()
    monkeypatch.setattr(diffusion_mod, "detect_family_for_pick", lambda *_a, **_k: fam)
    monkeypatch.setattr(
        DiffusionBackend, "_te_prequant_plan_files", lambda *_a, **_k: te_prequant or {}
    )
    monkeypatch.setattr(diffusion_mod, "prefer_ungated_mirror", lambda base, *_a, **_k: base)
    monkeypatch.setattr(diffusion_mod, "_assert_base_repo_accessible", lambda *_a, **_k: None)
    _stub_estimate(monkeypatch, files)


def _staged_backend(
    monkeypatch,
    *,
    files,
    calls,
    total_mib = STRIX_HALO_POOL_MIB,
    te_prequant = None,
    family = None,
):
    backend = _backend(monkeypatch, total_mib = total_mib)
    backend._load_token = 1
    backend._loading = diffusion_mod._LoadingState(
        repo_id = "unsloth/FLUX.2-dev", base_repo = "unsloth/FLUX.2-dev"
    )
    _stub_pick(monkeypatch, files, family = family, te_prequant = te_prequant)
    monkeypatch.setattr(diffusion_mod, "assert_flux2_pick_compatible", lambda *_a, **_k: None)
    monkeypatch.setattr(diffusion_mod, "assert_pick_is_not_speech", lambda *_a, **_k: None)
    monkeypatch.setattr(diffusion_mod, "_local_base_transformer_present", lambda *_a, **_k: False)

    def _prefetch(self, *_a, **_k):
        calls.append("prefetch")
        return None

    monkeypatch.setattr(DiffusionBackend, "_prefetch_files", _prefetch)
    monkeypatch.setattr(backend, "load_pipeline", lambda **_k: calls.append("load"))
    return backend


@pytest.mark.parametrize(
    ("total_mib", "expected_calls", "rejected"),
    [
        pytest.param(STRIX_HALO_POOL_MIB, [], True, id = "refused"),
        pytest.param(256 * 1024, ["prefetch", "load"], False, id = "fits"),
    ],
)
def test_run_load_applies_the_guard_before_prefetch(
    monkeypatch, total_mib, expected_calls, rejected
):
    calls: list = []
    backend = _staged_backend(monkeypatch, files = FLUX2_DEV, calls = calls, total_mib = total_mib)

    backend._run_load(repo_id = "unsloth/FLUX.2-dev", model_kind = "pipeline", _load_token = 1)

    assert calls == expected_calls
    if rejected:
        assert "usable on this device" in backend._loading.error
    else:
        assert backend._loading is None


def test_an_offline_load_never_probes_the_device(monkeypatch):
    calls: list = []
    backend = _staged_backend(monkeypatch, files = [], calls = calls)

    def _probe(_target):
        raise AssertionError("the offline path opened a device probe")

    monkeypatch.setattr(diffusion_mod, "snapshot_device_memory", _probe)

    backend._run_load(
        repo_id = "unsloth/FLUX.2-dev",
        model_kind = "pipeline",
        local_files_only = True,
        _load_token = 1,
    )

    assert backend._loading is None, getattr(backend._loading, "error", None)
    assert calls == ["prefetch", "load"]


def _plan_backend(
    monkeypatch,
    *,
    files,
    mismatch = None,
    total_mib = STRIX_HALO_POOL_MIB,
    te_prequant = None,
    family = None,
):
    backend = _backend(monkeypatch, total_mib = total_mib)
    _stub_pick(monkeypatch, files, family = family, te_prequant = te_prequant)
    monkeypatch.setattr(diffusion_mod, "flux2_pick_mismatch", lambda *_a, **_k: mismatch)
    monkeypatch.setattr(diffusion_mod, "speech_pick_refusal", lambda *_a, **_k: None)
    monkeypatch.setattr(DiffusionBackend, "_dit_prequant_plan_source", lambda *_a, **_k: None)
    return backend


@pytest.mark.parametrize(
    ("total_mib", "rejected"),
    [
        pytest.param(STRIX_HALO_POOL_MIB, True, id = "refused"),
        pytest.param(256 * 1024, False, id = "fits"),
    ],
)
def test_download_plan_reports_only_real_shortfalls(monkeypatch, total_mib, rejected):
    backend = _plan_backend(monkeypatch, files = FLUX2_DEV, total_mib = total_mib)
    plan = backend.download_plan("unsloth/FLUX.2-dev", model_kind = "pipeline")
    reason = plan["incompatible_reason"]
    assert (reason is not None) is rejected
    if rejected:
        assert "usable on this device" in reason


def test_download_plan_keeps_an_earlier_refusal(monkeypatch):
    backend = _plan_backend(monkeypatch, files = FLUX2_DEV, mismatch = "wrong base size")
    plan = backend.download_plan("unsloth/FLUX.2-dev", model_kind = "pipeline")
    assert plan["incompatible_reason"] == "wrong base size"


def test_download_plan_skips_the_device_probe_when_asked(monkeypatch):
    backend = _plan_backend(monkeypatch, files = FLUX2_DEV)

    def _probe(_target):
        raise AssertionError("the plan probed the device with allow_device_probe cleared")

    monkeypatch.setattr(diffusion_mod, "snapshot_device_memory", _probe)
    plan = backend.download_plan(
        "unsloth/FLUX.2-dev", model_kind = "pipeline", allow_device_probe = False
    )
    assert plan["incompatible_reason"] is None


def test_both_entry_points_size_a_hosted_encoder_the_same(monkeypatch):
    fam = _real_family("unsloth/Lumina-Image-2.0")

    calls: list = []
    staged = _staged_backend(
        monkeypatch,
        files = LUMINA_2_MINUS_DENSE_TE,
        calls = calls,
        total_mib = 12 * 1024,
        te_prequant = LUMINA_HOSTED_TE,
        family = fam,
    )
    staged._run_load(repo_id = "unsloth/Lumina-Image-2.0", model_kind = "pipeline", _load_token = 1)
    assert calls == [], "nothing may be staged once the verdict is in"
    assert "usable on this device" in staged._loading.error

    planner = _plan_backend(
        monkeypatch,
        files = LUMINA_2_MINUS_DENSE_TE,
        total_mib = 12 * 1024,
        te_prequant = LUMINA_HOSTED_TE,
        family = fam,
    )
    plan = planner.download_plan("unsloth/Lumina-Image-2.0", model_kind = "pipeline")
    assert plan["incompatible_reason"] is not None
    assert plan["incompatible_reason"] == staged._loading.error


def test_both_entry_points_count_hidreams_external_dense_te4(monkeypatch):
    fam = _real_family("HiDream-ai/HiDream-I1-Full")
    declared = _files(transformer = 47_000)

    calls: list = []
    staged = _staged_backend(
        monkeypatch,
        files = declared,
        calls = calls,
        total_mib = STRIX_HALO_POOL_MIB,
        family = fam,
    )
    staged._run_load(repo_id = "HiDream-ai/HiDream-I1-Full", model_kind = "pipeline", _load_token = 1)
    assert calls == []
    assert "usable on this device" in staged._loading.error

    planner = _plan_backend(
        monkeypatch,
        files = declared,
        total_mib = STRIX_HALO_POOL_MIB,
        family = fam,
    )
    plan = planner.download_plan("HiDream-ai/HiDream-I1-Full", model_kind = "pipeline")
    assert plan["incompatible_reason"] == staged._loading.error
