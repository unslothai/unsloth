# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic tests for the pre-download unified-memory guard from issue #9130.

Hub sizes recorded on 2026-08-24 are aggregated by component directory.
"""

from __future__ import annotations

import json
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


def _target(
    *,
    device = "cuda",
    ordinal = None,
    dtype = "bfloat16",
) -> DiffusionDeviceTarget:
    return DiffusionDeviceTarget(
        device = device,
        dtype = dtype,
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
    dtype = "bfloat16",
):
    backend = DiffusionBackend()
    target = _target(device = device, dtype = dtype)
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


def test_a_pipeline_that_cannot_fit_is_refused_from_metadata_alone(monkeypatch):
    backend = _backend(monkeypatch, total_mib = STRIX_HALO_POOL_MIB)
    message = _flux_verdict(backend)
    assert message is not None
    assert "flux.2-dev" in message
    assert "about 107 GB of memory for its weights" in message
    assert "about 51 GB is usable" in message
    assert "currently free" not in message
    assert "UNSLOTH_DIFFUSION_ALLOW_OVERSIZED_LOAD=1" in message


@pytest.mark.parametrize(
    ("base", "files", "prequant_bytes", "dtype_scale", "expected"),
    [
        pytest.param(
            "Tongyi-MAI/Z-Image-Turbo",
            Z_IMAGE_TURBO,
            0,
            1.0,
            23_479 * MIB // 2 + (7_672 + 160 + 15) * MIB,
            id = "fp32-denoiser",
        ),
        pytest.param(
            "ideogram-ai/ideogram-4-fp8",
            IDEOGRAM_4_FP8,
            0,
            1.0,
            2 * sum(size for _name, size in IDEOGRAM_4_FP8),
            id = "raw-fp8-pipeline",
        ),
        pytest.param(
            "Alpha-VLLM/Lumina-Image-2.0",
            LUMINA_2,
            0,
            1.0,
            sum(size for _name, size in LUMINA_2) // 2,
            id = "fp32-pipeline",
        ),
        pytest.param(
            "ideogram-ai/ideogram-4-nf4-diffusers",
            IDEOGRAM_4_NF4,
            0,
            1.0,
            sum(size for _name, size in IDEOGRAM_4_NF4),
            id = "custom-precision-sibling",
        ),
        pytest.param(
            "Alpha-VLLM/Lumina-Image-2.0",
            LUMINA_2_MINUS_DENSE_TE,
            LUMINA_HOSTED_TE_MIB * MIB,
            1.0,
            sum(size for _name, size in LUMINA_2_MINUS_DENSE_TE) // 2 + LUMINA_HOSTED_TE_MIB * MIB,
            id = "hosted-prequant",
        ),
        pytest.param(
            "Alpha-VLLM/Lumina-Image-2.0",
            LUMINA_2_MINUS_DENSE_TE,
            LUMINA_HOSTED_TE_MIB * MIB,
            2.0,
            sum(size for _name, size in LUMINA_2_MINUS_DENSE_TE) + LUMINA_HOSTED_TE_MIB * MIB,
            id = "float32-with-hosted-prequant",
        ),
    ],
)
def test_declared_sizes_are_converted_to_their_resident_precision(
    base, files, prequant_bytes, dtype_scale, expected
):
    from core.inference.diffusion_auto_policy import resident_bytes_from_declared
    assert (
        resident_bytes_from_declared(
            base,
            files,
            prequant_bytes = prequant_bytes,
            dtype_scale = dtype_scale,
        )
        == expected
    )


# stabilityai/stable-diffusion-xl-base-1.0, 12.9 GB: unet, both text encoders and the vae are
# all stored F32 in the DEFAULT variant (headers read 2026-08-25), and the loader skips the fp16
# twins, so the download is twice the bf16 residency.
SDXL_BASE = [
    ("unet/diffusion_pytorch_model.safetensors", 9794 * MIB),
    ("text_encoder/model.safetensors", 469 * MIB),
    ("text_encoder_2/model.safetensors", 2650 * MIB),
    ("vae/diffusion_pytorch_model.safetensors", 319 * MIB),
]


def test_an_fp32_stored_pipeline_is_not_priced_at_its_download_size(monkeypatch):
    """SDXL is the one U-Net family, so its denoiser sits in ``unet/`` and lands in the
    companion bucket rather than the denoiser one. Both factors therefore have to halve, or
    a 6.5 GB bf16 pipeline prices as 12.9 GB and is refused on every 16 GB pool."""
    from core.inference.diffusion_auto_policy import resident_bytes_from_declared

    declared = sum(size for _name, size in SDXL_BASE)
    for base in ("stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/sdxl-turbo"):
        assert resident_bytes_from_declared(base, SDXL_BASE) == declared // 2, base

    fam = _family(name = "sdxl", base_repo = "stabilityai/stable-diffusion-xl-base-1.0")
    for pool_mib in (12 * 1024, 16 * 1024, 24 * 1024):
        backend = _backend(monkeypatch, total_mib = pool_mib)
        assert (
            _verdict(
                backend, fam, "stabilityai/stable-diffusion-xl-base-1.0", fam.base_repo, SDXL_BASE
            )
            is None
        ), pool_mib


def test_float32_target_rejects_lumina_that_bf16_accepts(monkeypatch):
    fam = _family(name = "lumina-2", base_repo = "Alpha-VLLM/Lumina-Image-2.0")
    bf16 = _backend(monkeypatch, total_mib = 16 * 1024)
    assert _verdict(bf16, fam, "unsloth/Lumina-Image-2.0", fam.base_repo, LUMINA_2) is None

    fp32 = _backend(monkeypatch, total_mib = 16 * 1024, device = "mps", dtype = "float32")
    message = _verdict(fp32, fam, "unsloth/Lumina-Image-2.0", fam.base_repo, LUMINA_2)
    assert message is not None
    assert "about 22 GB of memory for its weights" in message


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
    monkeypatch.setattr(
        diffusion_mod, "resolve_diffusion_device_target", lambda *_a, **_k: _target()
    )
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
    monkeypatch.setattr(
        diffusion_mod, "resolve_diffusion_device_target", lambda *_a, **_k: _target()
    )
    fam = _family(name = "hidream-i1", base_repo = "HiDream-ai/HiDream-I1-Full")

    planned = DiffusionBackend._te_prequant_plan_files(
        fam, "fp8", None, base_repo = "HiDream-ai/HiDream-I1-Dev"
    )
    assert "text_encoder_4" in seen["components"]
    assert "text_encoder_4" in planned


def test_hidream_dense_te4_is_counted_when_no_prequant_is_selected():
    from core.inference.diffusion import (
        _prequant_plan_bytes,
        _predownload_encoder_bf16_bytes,
    )
    from core.inference.diffusion_hidream import HIDREAM_LLAMA_BF16_BYTES

    fam = _family(name = "hidream-i1", base_repo = "HiDream-ai/HiDream-I1-Full")
    assert _predownload_encoder_bf16_bytes(fam, {}) == HIDREAM_LLAMA_BF16_BYTES
    assert _predownload_encoder_bf16_bytes(fam, {}, pipeline_declared = False) == 0
    hosted = {"text_encoder_4": ("unsloth/HiDream-I1-Full-FP8", [("te4.pt", 8 * MIB)])}
    assert _prequant_plan_bytes(hosted) == 8 * MIB
    assert _predownload_encoder_bf16_bytes(fam, hosted) == 0


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
def test_no_sizes_does_not_resolve_a_target(monkeypatch, files):
    backend = DiffusionBackend()

    def _resolve(*_a, **_k):
        raise AssertionError("zero metadata must not resolve a device")

    monkeypatch.setattr(backend, "_target_for_ordinal", _resolve)
    assert (
        backend.declared_footprint_shortfall(
            _flux2(),
            "unsloth/FLUX.2-dev",
            "black-forest-labs/FLUX.2-dev",
            kind = "pipeline",
            declared_files = files,
        )
        is None
    )


def _stub_pipeline_hub(monkeypatch, tmp_path, repos):
    manifests = {}
    for index, (repo, (payload, _files, _sha)) in enumerate(repos.items()):
        manifest = tmp_path / f"model-index-{index}.json"
        manifest.write_text(json.dumps(payload), encoding = "utf-8")
        manifests[repo] = manifest
    info_calls: list = []
    download_calls: list = []

    class _Api:
        def model_info(self, repo_id, **_kwargs):
            info_calls.append(repo_id)
            _payload, files, sha = repos[repo_id]
            siblings = [types.SimpleNamespace(rfilename = name, size = size) for name, size in files]
            return types.SimpleNamespace(siblings = siblings, sha = sha)

    def fake_download(repo_id, filename, **kwargs):
        download_calls.append((repo_id, filename, kwargs.get("revision")))
        return str(manifests[repo_id])

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api())
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    return info_calls, download_calls


def test_pipeline_estimate_uses_selected_components_and_default_variant(monkeypatch, tmp_path):
    repo = "unsloth/custom-pipeline"
    payload = {
        "_class_name": "FluxPipeline",
        "_ignore_files": ["transformer/ignored.safetensors"],
        "transformer": ["diffusers", "FluxTransformer2DModel"],
        "unused": [None, None],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }
    files = [
        ("model_index.json", 100),
        ("transformer/config.json", 200),
        ("transformer/diffusion_pytorch_model.safetensors", 20 * MIB),
        ("transformer/diffusion_pytorch_model.fp8.safetensors", 10 * MIB),
        ("transformer/diffusion_pytorch_model.fp8-00001-of-00002.safetensors", 10 * MIB),
        ("transformer/ignored.safetensors", 9 * MIB),
        ("unused/diffusion_pytorch_model.safetensors", 30 * MIB),
        ("scheduler/scheduler_config.json", 300),
    ]
    _stub_pipeline_hub(monkeypatch, tmp_path, {repo: (payload, files, "a" * 40)})
    staged: dict = {}
    resident: list = []
    total, files = DiffusionBackend._estimate_download_bytes(
        repo,
        None,
        repo,
        None,
        kind = "pipeline",
        file_sizes_out = staged,
        resident_file_sizes_out = resident,
    )

    expected = {
        "model_index.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
    }
    assert set(files) == expected
    assert set(staged[repo]) == expected
    assert resident == [("transformer/diffusion_pytorch_model.safetensors", 20 * MIB)]
    assert total == 20 * MIB + 600


def test_gated_pipeline_manifest_is_read_from_the_fetch_mirror(monkeypatch, tmp_path):
    upstream = "black-forest-labs/FLUX.2-dev"
    mirror = "unsloth/FLUX.2-dev"
    payload = {"transformer": ["diffusers", "Flux2Transformer2DModel"]}
    files = [
        ("model_index.json", 100),
        ("transformer/diffusion_pytorch_model.safetensors", 20 * MIB),
    ]
    info_calls, download_calls = _stub_pipeline_hub(
        monkeypatch, tmp_path, {mirror: (payload, files, "b" * 40)}
    )
    monkeypatch.setattr(
        diffusion_mod,
        "prefer_ungated_mirror",
        lambda repo_id, *_a, **_k: mirror if repo_id == upstream else repo_id,
    )
    resident: list = []
    revisions: dict = {}
    fetch_repos: dict = {}
    _total, _files = DiffusionBackend._estimate_download_bytes(
        upstream,
        None,
        upstream,
        None,
        kind = "pipeline",
        resident_file_sizes_out = resident,
        revisions_out = revisions,
        fetch_repos_out = fetch_repos,
    )

    assert info_calls == [mirror]
    assert download_calls == [(mirror, "model_index.json", "b" * 40)]
    assert resident == [("transformer/diffusion_pytorch_model.safetensors", 20 * MIB)]
    assert revisions == {mirror: "b" * 40}
    assert fetch_repos == {upstream: mirror}


def test_pipeline_listing_restarts_when_the_exact_scope_selects_the_mirror(monkeypatch, tmp_path):
    upstream = "black-forest-labs/FLUX.2-dev"
    mirror = "unsloth/FLUX.2-dev"
    payload = {"transformer": ["diffusers", "Flux2Transformer2DModel"]}
    listings = {
        upstream: (
            payload,
            [
                ("model_index.json", 100),
                ("transformer/diffusion_pytorch_model.safetensors", 20 * MIB),
            ],
            "a" * 40,
        ),
        mirror: (
            {**payload, "vae": ["diffusers", "AutoencoderKL"]},
            [
                ("model_index.json", 101),
                ("transformer/diffusion_pytorch_model.safetensors", 21 * MIB),
                ("vae/diffusion_pytorch_model.safetensors", 3 * MIB),
            ],
            "b" * 40,
        ),
    }
    info_calls, download_calls = _stub_pipeline_hub(monkeypatch, tmp_path, listings)

    def fake_prefer(
        repo_id,
        *_args,
        files = None,
        **_kwargs,
    ):
        assert repo_id == upstream
        return upstream if tuple(files or ()) == ("model_index.json",) else mirror

    monkeypatch.setattr(diffusion_mod, "prefer_ungated_mirror", fake_prefer)
    resident: list = []
    revisions: dict = {}
    fetch_repos: dict = {}
    total, files = DiffusionBackend._estimate_download_bytes(
        upstream,
        None,
        upstream,
        None,
        kind = "pipeline",
        resident_file_sizes_out = resident,
        revisions_out = revisions,
        fetch_repos_out = fetch_repos,
    )

    assert info_calls == [upstream, mirror]
    assert download_calls == [
        (upstream, "model_index.json", "a" * 40),
        (mirror, "model_index.json", "b" * 40),
    ]
    assert set(files) == {
        "model_index.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    }
    assert total == 24 * MIB + 101
    assert resident == [
        ("transformer/diffusion_pytorch_model.safetensors", 21 * MIB),
        ("vae/diffusion_pytorch_model.safetensors", 3 * MIB),
    ]
    assert revisions == {mirror: "b" * 40}
    assert fetch_repos == {upstream: mirror}


@pytest.mark.parametrize(
    ("family", "included"),
    [
        pytest.param("krea-2", "transformer", id = "krea"),
        pytest.param("ideogram-4", "unconditional_transformer", id = "ideogram"),
    ],
)
def test_explicit_pipeline_assemblers_use_their_fixed_component_sets(
    monkeypatch, tmp_path, family, included
):
    from core.inference.diffusion import _explicit_pipeline_components

    repo = f"unsloth/{family}"
    files = [
        ("model_index.json", 100),
        (f"{included}/diffusion_pytorch_model.safetensors", 20 * MIB),
        ("unused/diffusion_pytorch_model.safetensors", 30 * MIB),
    ]
    _stub_pipeline_hub(monkeypatch, tmp_path, {repo: ({"_class_name": "Pipeline"}, files, None)})
    staged: dict = {}
    resident: list = []
    _total, files = DiffusionBackend._estimate_download_bytes(
        repo,
        None,
        repo,
        None,
        kind = "pipeline",
        pipeline_components = _explicit_pipeline_components(_family(name = family)),
        file_sizes_out = staged,
        resident_file_sizes_out = resident,
    )
    assert files == ["model_index.json", f"{included}/diffusion_pytorch_model.safetensors"]
    assert resident == [(f"{included}/diffusion_pytorch_model.safetensors", 20 * MIB)]


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
        file_sizes = kwargs.get("file_sizes_out")
        if file_sizes is not None:
            file_sizes["unsloth/FLUX.2-dev"] = {name: size for name, size in files}
        resident_sizes = kwargs.get("resident_file_sizes_out")
        if resident_sizes is not None:
            resident_sizes.extend(files)
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


def test_download_plan_keeps_an_earlier_refusal(monkeypatch):
    backend = _plan_backend(monkeypatch, files = FLUX2_DEV, mismatch = "wrong base size")
    plan = backend.download_plan("unsloth/FLUX.2-dev", model_kind = "pipeline")
    assert plan["incompatible_reason"] == "wrong base size"


def test_download_plan_skips_the_device_probe_when_asked(monkeypatch):
    backend = _plan_backend(monkeypatch, files = FLUX2_DEV)

    def _probe(*_args, **_kwargs):
        raise AssertionError("the plan probed the device with allow_device_probe cleared")

    monkeypatch.setattr(diffusion_mod, "snapshot_device_memory", _probe)
    monkeypatch.setattr(DiffusionBackend, "_te_prequant_plan_files", _probe)
    monkeypatch.setattr(DiffusionBackend, "_dit_prequant_plan_source", _probe)
    plan = backend.download_plan(
        "unsloth/FLUX.2-dev",
        model_kind = "pipeline",
        text_encoder_quant = "fp8",
        allow_device_probe = False,
    )
    assert plan["incompatible_reason"] is None


@pytest.mark.parametrize(
    ("repo", "files", "total_mib", "te_prequant", "expected_calls"),
    [
        pytest.param(
            "unsloth/FLUX.2-dev",
            FLUX2_DEV,
            STRIX_HALO_POOL_MIB,
            None,
            [],
            id = "refused",
        ),
        pytest.param(
            "unsloth/FLUX.2-dev",
            FLUX2_DEV,
            256 * 1024,
            None,
            ["prefetch", "load"],
            id = "fits",
        ),
        pytest.param(
            "unsloth/Lumina-Image-2.0",
            LUMINA_2_MINUS_DENSE_TE,
            12 * 1024,
            LUMINA_HOSTED_TE,
            [],
            id = "hosted-encoder",
        ),
        pytest.param(
            "HiDream-ai/HiDream-I1-Full",
            _files(transformer = 47_000),
            STRIX_HALO_POOL_MIB,
            None,
            [],
            id = "external-dense-encoder",
        ),
    ],
)
def test_load_and_plan_apply_the_same_memory_guard(
    monkeypatch, repo, files, total_mib, te_prequant, expected_calls
):
    fam = _real_family(repo)
    calls: list = []
    staged = _staged_backend(
        monkeypatch,
        files = files,
        calls = calls,
        total_mib = total_mib,
        te_prequant = te_prequant,
        family = fam,
    )
    staged._run_load(repo_id = repo, model_kind = "pipeline", _load_token = 1)
    assert calls == expected_calls

    planner = _plan_backend(
        monkeypatch,
        files = files,
        total_mib = total_mib,
        te_prequant = te_prequant,
        family = fam,
    )
    plan = planner.download_plan(repo, model_kind = "pipeline")
    reason = plan["incompatible_reason"]
    if expected_calls:
        assert staged._loading is None
        assert reason is None
    else:
        assert "usable on this device" in staged._loading.error
        assert reason == staged._loading.error
