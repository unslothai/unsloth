# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic tests for seeding an official image PIPELINE pick with its hosted pre-quantized denoiser.

An official ``kind == "pipeline"`` pick with the precision left to us now defaults to the hosted
FP8 / INT8 checkpoint the family table names, the way MiniMax-H3's video loader already does: the
artifact REPLACES the released ``transformer/`` shards, so it is planned before a byte moves, the
shards are dropped from the pull, and the quantised denoiser is handed to ``from_pretrained``
instead of being built dense and rewritten in place.

torch and diffusers are stubbed via ``sys.modules`` and the Hub is a fixture, so the plan, the pin
and the load are all exercised without CUDA, torchao or a real download.
"""

from __future__ import annotations

import contextlib
import json
import sys
import types

import pytest

from core.inference import diffusion as dmod
from core.inference import diffusion_prequant as pqmod
from core.inference import diffusion_transformer_quant as tqmod
from core.inference.diffusion import DiffusionBackend
from core.inference.diffusion_auto_policy import DenseQuantEstimate
from core.inference.diffusion_denoiser_prequant import PIPELINE_SEED_DECLINED
from core.inference.diffusion_device import DiffusionDeviceTarget
from core.inference.diffusion_families import detect_family_for_pick
from core.inference.diffusion_memory import DeviceMemory

MIB = 1024 * 1024

Z_IMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"
PREQUANT_REPO = "unsloth/Z-Image-Turbo-FP8"
PREQUANT_FILE = "Z-Image-Turbo-FP8.pt"
PREQUANT_BYTES = 6000 * MIB

# Tongyi-MAI/Z-Image-Turbo's component layout; the denoiser is the two 11 GiB shards.
Z_IMAGE_INDEX = {
    "_class_name": "ZImagePipeline",
    "transformer": ["diffusers", "ZImageTransformer2DModel"],
    "text_encoder": ["transformers", "Qwen3Model"],
    "vae": ["diffusers", "AutoencoderKL"],
    "tokenizer": ["transformers", "Qwen2Tokenizer"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
}
DENOISER_SHARDS = [
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
]
Z_IMAGE_FILES = [
    ("model_index.json", 100),
    ("transformer/config.json", 200),
    (DENOISER_SHARDS[0], 11_000 * MIB),
    (DENOISER_SHARDS[1], 11_000 * MIB),
    ("text_encoder/model.safetensors", 7_672 * MIB),
    ("vae/diffusion_pytorch_model.safetensors", 160 * MIB),
    ("tokenizer/tokenizer_config.json", 300),
    ("scheduler/scheduler_config.json", 300),
]
DENOISER_BYTES = 22_000 * MIB
ALL_BYTES = sum(size for _name, size in Z_IMAGE_FILES)


def _family():
    fam = detect_family_for_pick(Z_IMAGE_REPO, None, None)
    assert fam is not None and fam.name == "z-image"
    return fam


def _target(*, device = "cuda", ordinal = None) -> DiffusionDeviceTarget:
    return DiffusionDeviceTarget(
        device = device,
        dtype = "bfloat16",
        backend = device,
        vendor = "nvidia",
        supports_model_cpu_offload = True,
        supports_default_torch_compile = True,
        supports_pinned_transfer = True,
        ordinal = ordinal,
    )


def _estimate(prequant = True) -> DenseQuantEstimate:
    return DenseQuantEstimate(
        scheme = "fp8",
        steady_transformer_mib = 11_500,
        transient_transformer_mib = 11_500 if prequant else 23_000,
        companions_mib = 7_500,
        prequant = prequant,
        download_transformer_mib = 23_000,
        text_encoders_mib = 7_320,
    )


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """The pipeline repo plus the hosted checkpoint's repo, as ``model_info`` sees them."""
    repos = {
        Z_IMAGE_REPO: Z_IMAGE_FILES,
        PREQUANT_REPO: [(PREQUANT_FILE, PREQUANT_BYTES), ("README.md", 10)],
    }
    manifest = tmp_path / "model_index.json"
    manifest.write_text(json.dumps(Z_IMAGE_INDEX), encoding = "utf-8")
    info_calls: list = []

    class _Api:
        def __init__(self, *_a, **_k) -> None:
            pass

        def model_info(self, repo_id, **_kwargs):
            info_calls.append(repo_id)
            return types.SimpleNamespace(
                siblings = [
                    types.SimpleNamespace(rfilename = name, size = size)
                    for name, size in repos[repo_id]
                ],
                sha = "c" * 40,
            )

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    # The mirror table maps this base onto unsloth/Z-Image-Turbo; the listing is the same either
    # way and the swap is not what these tests are about.
    monkeypatch.setattr(dmod, "prefer_ungated_mirror", lambda base, *_a, **_k: base)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda repo_id, filename, **kw: str(manifest)
    )
    return info_calls


def _estimate_bytes(**overrides):
    kwargs = {
        "kind": "pipeline",
        "pipeline_components": None,
        "single_file_is_pipeline": False,
        **overrides,
    }
    return DiffusionBackend._estimate_download_bytes(
        Z_IMAGE_REPO, None, Z_IMAGE_REPO, None, **kwargs
    )


# ── the download plan ────────────────────────────────────────────────────────────
def test_the_plan_drops_the_released_denoiser_shards_and_keeps_its_config(hub):
    """The hosted checkpoint replaces the weights, not the config: the seeding loader meta-inits
    the denoiser from ``transformer/config.json``, so dropping it would send a fully staged load
    back to the Hub."""
    skipped: list[str] = []
    resident: list = []
    total, files = _estimate_bytes(
        skip_transformer_weights = True,
        skipped_files_out = skipped,
        resident_file_sizes_out = resident,
    )
    assert "transformer/config.json" in files
    assert not [f for f in files if f in DENOISER_SHARDS]
    assert skipped == DENOISER_SHARDS
    assert total == ALL_BYTES - DENOISER_BYTES
    # Nothing the pipeline materialises from those shards is declared resident either.
    assert not [name for name, _size in resident if name.startswith("transformer/")]


def test_the_plan_keeps_the_shards_when_nothing_is_seeded(hub):
    skipped: list[str] = []
    total, files = _estimate_bytes(skipped_files_out = skipped)
    assert all(shard in files for shard in DENOISER_SHARDS)
    assert skipped == []
    assert total == ALL_BYTES


def _plan_backend(
    monkeypatch,
    *,
    planned,
    mismatch = None,
):
    backend = DiffusionBackend()
    monkeypatch.setattr(backend, "_target_for_ordinal", lambda *_a, **_k: _target())
    monkeypatch.setattr(dmod, "detect_family_for_pick", lambda *_a, **_k: _family())
    monkeypatch.setattr(dmod, "prefer_ungated_mirror", lambda base, *_a, **_k: base)
    monkeypatch.setattr(dmod, "_assert_base_repo_accessible", lambda *_a, **_k: None)
    monkeypatch.setattr(DiffusionBackend, "_te_prequant_plan_files", lambda *_a, **_k: {})
    monkeypatch.setattr(dmod, "flux2_pick_mismatch", lambda *_a, **_k: mismatch)
    monkeypatch.setattr(dmod, "speech_pick_refusal", lambda *_a, **_k: None)
    monkeypatch.setattr(
        DiffusionBackend, "_pipeline_planned_denoiser_scheme", lambda *_a, **_k: planned
    )
    # Nothing on this host is cached, so every planned file is a real entry.
    monkeypatch.setattr(
        DiffusionBackend, "_files_already_cached", staticmethod(lambda *_a, **_k: set())
    )
    monkeypatch.setattr(
        DiffusionBackend, "_hub_file_is_cached", staticmethod(lambda *_a, **_k: False)
    )
    monkeypatch.setattr(DiffusionBackend, "declared_footprint_shortfall", lambda *_a, **_k: None)
    return backend


def test_the_plan_counts_and_stages_the_hosted_checkpoint(monkeypatch, hub):
    """The shards are gone from the pull, so the artifact that replaces them is real footprint the
    plan would otherwise never report, and a file the manager must stage rather than let the load
    fetch inline."""
    backend = _plan_backend(monkeypatch, planned = "fp8")
    plan = backend.download_plan(Z_IMAGE_REPO, model_kind = "pipeline")

    entries = {entry["repo_id"]: entry for entry in plan["entries"]}
    assert entries[PREQUANT_REPO]["files"] == [PREQUANT_FILE]
    assert entries[PREQUANT_REPO]["bytes"] == PREQUANT_BYTES
    assert not [f for f in entries[Z_IMAGE_REPO]["files"] if f in DENOISER_SHARDS]
    assert "transformer/config.json" in entries[Z_IMAGE_REPO]["files"]
    assert plan["required_bytes"] == ALL_BYTES - DENOISER_BYTES + PREQUANT_BYTES
    # The pipeline repo is still the selected model; the checkpoint is its companion.
    assert entries[Z_IMAGE_REPO]["checkpoint"] is True
    assert entries[PREQUANT_REPO]["checkpoint"] is False


def test_an_unplanned_pipeline_pick_plans_exactly_as_before(monkeypatch, hub):
    """No hosted artifact for the resolved scheme: the released shards stay in the pull and the
    loader quantises them in place, which is the behaviour this change falls through to."""
    backend = _plan_backend(monkeypatch, planned = None)
    plan = backend.download_plan(Z_IMAGE_REPO, model_kind = "pipeline")

    entries = {entry["repo_id"]: entry for entry in plan["entries"]}
    assert PREQUANT_REPO not in entries
    assert all(shard in entries[Z_IMAGE_REPO]["files"] for shard in DENOISER_SHARDS)
    assert plan["required_bytes"] == ALL_BYTES


def test_a_declined_plan_keeps_the_released_shards(monkeypatch, hub):
    backend = _plan_backend(monkeypatch, planned = PIPELINE_SEED_DECLINED)
    plan = backend.download_plan(Z_IMAGE_REPO, model_kind = "pipeline")

    entries = {entry["repo_id"]: entry for entry in plan["entries"]}
    assert PREQUANT_REPO not in entries
    assert all(shard in entries[Z_IMAGE_REPO]["files"] for shard in DENOISER_SHARDS)


def test_the_artifact_carries_the_denoiser_share_of_the_unified_memory_verdict(monkeypatch, hub):
    """The shards are out of the declared set, so without the artifact bytes the refusal would
    price this pipeline at its companions alone."""
    backend = _plan_backend(monkeypatch, planned = "fp8")
    seen: dict = {}

    def _shortfall(_self, *_a, **kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(DiffusionBackend, "declared_footprint_shortfall", _shortfall)
    backend.download_plan(Z_IMAGE_REPO, model_kind = "pipeline")
    assert seen["prequant_bytes"] == PREQUANT_BYTES
    assert not [name for name, _s in seen["declared_files"] if name.startswith("transformer/")]


def test_a_gguf_pick_still_never_downloads_a_second_denoiser(monkeypatch):
    """Unchanged: for a GGUF pick the hosted checkpoint is a SECOND denoiser beside one already on
    disk, so an auto quant only ever uses a cached one. The pipeline rule is the opposite because
    there the artifact REPLACES the download."""
    backend = DiffusionBackend()
    monkeypatch.setattr(backend, "_target_for_ordinal", lambda *_a, **_k: _target())
    monkeypatch.setattr(dmod, "_uncached_prequant_repo", lambda *_a, **_k: PREQUANT_REPO)

    def _never(*_a, **_k):
        raise AssertionError("a GGUF pick sized an uncached hosted checkpoint")

    monkeypatch.setattr(DiffusionBackend, "_prequant_source_hub_entry", staticmethod(_never))
    assert (
        backend._dit_prequant_plan_source(
            _family(), "gguf", None, {"base_repo": Z_IMAGE_REPO, "transformer_quant": None}
        )
        is None
    )


def test_a_planned_pipeline_pick_is_sized_from_the_hub(monkeypatch, hub):
    backend = DiffusionBackend()
    monkeypatch.setattr(backend, "_target_for_ordinal", lambda *_a, **_k: _target())
    monkeypatch.setattr(pqmod, "restricted_prequant_load_supported", lambda _scheme: True)
    assert backend._dit_prequant_plan_source(
        _family(),
        "pipeline",
        None,
        {"base_repo": Z_IMAGE_REPO, "_pipeline_prequant_planned": "fp8"},
    ) == (PREQUANT_REPO, PREQUANT_FILE, PREQUANT_BYTES)


def test_a_pipeline_pick_with_no_planned_scheme_sizes_nothing(monkeypatch, hub):
    backend = DiffusionBackend()
    for planned in (None, PIPELINE_SEED_DECLINED):
        assert (
            backend._dit_prequant_plan_source(
                _family(),
                "pipeline",
                None,
                {"base_repo": Z_IMAGE_REPO, "_pipeline_prequant_planned": planned},
            )
            is None
        )


# ── the plan-time settle ─────────────────────────────────────────────────────────
def _settle_backend(
    monkeypatch,
    *,
    offload = "none",
    scheme = "fp8",
    candidate = None,
):
    backend = DiffusionBackend()
    monkeypatch.setattr(backend, "_target_for_ordinal", lambda *_a, **_k: _target())
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda _t: True)
    monkeypatch.setattr(dmod, "_pipeline_quant_uncompilable_reason", lambda *_a, **_k: None)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: scheme
    )
    monkeypatch.setattr(pqmod, "restricted_prequant_load_supported", lambda _scheme: True)
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **_k: _estimate() if candidate is None else candidate,
    )
    monkeypatch.setattr(
        dmod,
        "snapshot_device_memory",
        lambda _t: DeviceMemory("cuda", "cuda", "dedicated", 1_000, 180_000),
    )
    monkeypatch.setattr(
        DiffusionBackend,
        "_plan_memory",
        lambda *_a, **_k: types.SimpleNamespace(offload_policy = offload),
    )
    return backend


def _settle(backend, **overrides):
    kwargs = {
        "base": Z_IMAGE_REPO,
        "kind": "pipeline",
        "transformer_quant": None,
        "speed_mode": None,
        "repo_id": Z_IMAGE_REPO,
        **overrides,
    }
    return backend._pipeline_planned_denoiser_scheme(_family(), **kwargs)


def test_an_unset_precision_defaults_to_the_hosted_checkpoint(monkeypatch):
    assert _settle(_settle_backend(monkeypatch)) == "fp8"


def test_an_ampere_host_defaults_to_the_hosted_int8_checkpoint(monkeypatch):
    """Same family, same table: the ladder picks the scheme, and z-image hosts both."""
    assert _settle(_settle_backend(monkeypatch, scheme = "int8")) == "int8"


@pytest.mark.parametrize("quant", ["none", "off"])
def test_precision_off_keeps_the_released_weights(monkeypatch, quant):
    """The hosted checkpoint re-rolls the sample, so a request NOT to quantise is honoured."""
    assert _settle(_settle_backend(monkeypatch), transformer_quant = quant) is None


def test_speed_off_keeps_the_released_weights(monkeypatch):
    """Speed=off is a bit-exact contract and a quantised denoiser is not bit-exact."""
    assert _settle(_settle_backend(monkeypatch), speed_mode = "off") is None


def test_a_baked_lora_keeps_the_dense_path(monkeypatch):
    """Adapters attach to dense Linears before torchao converts them, so a bake needs the released
    weights. An all-zero list bakes nothing and is unaffected."""
    backend = _settle_backend(monkeypatch)
    assert _settle(backend, loras = [("adapter", 0.8)]) is None
    assert _settle(backend, loras = [("adapter", 0.0)]) == "fp8"


@pytest.mark.parametrize("mode", ["balanced", "low_vram"])
def test_a_definite_offload_request_keeps_the_released_weights(monkeypatch, mode):
    assert _settle(_settle_backend(monkeypatch), memory_mode = mode) is None


def test_an_artifact_sized_plan_that_still_offloads_declines(monkeypatch):
    """Offload hooks move modules with Module.to(), which torchao tensors reject, so the decline is
    pinned rather than left for the loader to rediscover after the shards are gone."""
    backend = _settle_backend(monkeypatch, offload = "sequential")
    assert _settle(backend) == PIPELINE_SEED_DECLINED


def test_a_family_with_no_hosted_artifact_falls_through(monkeypatch):
    """nvfp4 has no hosted z-image checkpoint, so the loader's in-memory quantise handles it."""
    assert _settle(_settle_backend(monkeypatch, scheme = "nvfp4")) is None


def test_a_base_with_no_hosted_artifact_falls_through(monkeypatch):
    """Both hosted checkpoints are baked from the distilled Turbo transformer, so the undistilled
    base is in ``prequant_excluded_bases`` and must quantise its own weights."""
    assert _settle(_settle_backend(monkeypatch), base = "Tongyi-MAI/Z-Image") is None


def test_a_gguf_pick_is_never_settled_here(monkeypatch):
    assert _settle(_settle_backend(monkeypatch), kind = "gguf") is None


@pytest.mark.parametrize("family", ["krea-2", "ideogram-4"])
def test_a_per_component_assembler_is_never_seeded(monkeypatch, family):
    """Their loaders never see ``pipe_kwargs``, so a seed would be dropped AFTER the plan had
    already left their released shards out of the pull."""
    backend = _settle_backend(monkeypatch)
    fam = types.SimpleNamespace(name = family, base_repo = "x/y")
    assert (
        backend._pipeline_planned_denoiser_scheme(
            fam,
            base = "x/y",
            kind = "pipeline",
            transformer_quant = None,
            speed_mode = None,
        )
        is None
    )


def test_an_uncompilable_host_keeps_the_released_weights(monkeypatch):
    """A torchao denoiser that is never compiled is ~30x slower than the bf16 it replaced, and the
    loader declines such a pipeline anyway; planning around a seed it will refuse would drop shards
    for nothing."""
    backend = _settle_backend(monkeypatch)
    monkeypatch.setattr(dmod, "_pipeline_quant_uncompilable_reason", lambda *_a, **_k: "no triton")
    assert _settle(backend) is None


def test_an_unanswerable_probe_keeps_the_released_weights(monkeypatch):
    backend = _settle_backend(monkeypatch)

    def _boom(*_a, **_k):
        raise RuntimeError("driver went away")

    monkeypatch.setattr(dmod, "select_transformer_quant_scheme", _boom)
    assert _settle(backend) is None


# ── the pin from plan to load ────────────────────────────────────────────────────
def _run_load_backend(
    monkeypatch,
    *,
    planned,
    verified = True,
):
    backend = DiffusionBackend()
    backend._load_token = 1
    backend._loading = dmod._LoadingState(repo_id = Z_IMAGE_REPO, base_repo = Z_IMAGE_REPO)
    monkeypatch.setattr(dmod, "detect_family_for_pick", lambda *_a, **_k: _family())
    monkeypatch.setattr(dmod, "prefer_ungated_mirror", lambda base, *_a, **_k: base)
    monkeypatch.setattr(dmod, "_assert_base_repo_accessible", lambda *_a, **_k: None)
    monkeypatch.setattr(dmod, "assert_flux2_pick_compatible", lambda *_a, **_k: None)
    monkeypatch.setattr(dmod, "assert_pick_is_not_speech", lambda *_a, **_k: None)
    monkeypatch.setattr(dmod, "_local_base_transformer_present", lambda *_a, **_k: False)
    monkeypatch.setattr(DiffusionBackend, "_te_prequant_plan_files", lambda *_a, **_k: {})
    monkeypatch.setattr(DiffusionBackend, "declared_footprint_shortfall", lambda *_a, **_k: None)
    monkeypatch.setattr(
        DiffusionBackend, "_pipeline_planned_denoiser_scheme", lambda *_a, **_k: planned
    )
    monkeypatch.setattr(
        DiffusionBackend,
        "_dit_prequant_plan_source",
        lambda *_a, **_k: (PREQUANT_REPO, PREQUANT_FILE, PREQUANT_BYTES)
        if verified and planned not in (None, PIPELINE_SEED_DECLINED)
        else None,
    )
    fetched: list = []
    monkeypatch.setattr(
        DiffusionBackend,
        "_fetch_denoiser_prequant",
        lambda _self, entry, *_a, **_k: fetched.append(entry),
    )
    monkeypatch.setattr(DiffusionBackend, "_prefetch_files", lambda *_a, **_k: None)
    seen: dict = {}
    monkeypatch.setattr(backend, "load_pipeline", lambda **kwargs: seen.update(kwargs))
    return backend, seen, fetched


def test_the_load_is_pinned_to_the_plan_that_scoped_the_pull(monkeypatch, hub):
    backend, seen, fetched = _run_load_backend(monkeypatch, planned = "fp8")
    backend._run_load(repo_id = Z_IMAGE_REPO, model_kind = "pipeline", _load_token = 1)

    assert backend._loading is None, getattr(backend._loading, "error", None)
    assert seen["_pipeline_prequant_planned"] == "fp8"
    assert seen["_pipeline_prequant_skipped"] == tuple(DENOISER_SHARDS)
    # Staged under this load's cancel event rather than inline under the load lock.
    assert fetched == [(PREQUANT_REPO, PREQUANT_FILE, PREQUANT_BYTES)]
    # ...and claimed, so a delete cannot yank it mid-fetch.
    assert PREQUANT_REPO in backend.loading_repo_ids() or backend._loading is None


def test_an_artifact_that_does_not_resolve_keeps_the_released_shards(monkeypatch, hub):
    """The skip flag is never re-checked, so a checkpoint that did not resolve on the Hub has to
    leave the loader's bf16 fallback something to open."""
    backend, seen, fetched = _run_load_backend(monkeypatch, planned = "fp8", verified = False)
    backend._run_load(repo_id = Z_IMAGE_REPO, model_kind = "pipeline", _load_token = 1)

    assert seen["_pipeline_prequant_planned"] is None
    assert seen["_pipeline_prequant_skipped"] == ()
    assert fetched == []


def test_the_decline_is_pinned_across_plan_and_load(monkeypatch, hub):
    """The pull kept the released shards on this decline, so the loader must not re-take the
    decision against post-eviction free memory and fetch the artifact inline."""
    backend, seen, _fetched = _run_load_backend(monkeypatch, planned = PIPELINE_SEED_DECLINED)
    backend._run_load(repo_id = Z_IMAGE_REPO, model_kind = "pipeline", _load_token = 1)

    assert seen["_pipeline_prequant_planned"] == PIPELINE_SEED_DECLINED
    assert seen["_pipeline_prequant_skipped"] == ()


def test_an_offline_load_never_probes_the_hub_for_an_artifact(monkeypatch):
    """Nothing may be downloaded, so the settle and its ``model_info`` both stand down."""
    backend, seen, fetched = _run_load_backend(monkeypatch, planned = "fp8")

    def _never(*_a, **_k):
        raise AssertionError("the offline path settled a hosted checkpoint")

    monkeypatch.setattr(DiffusionBackend, "_pipeline_planned_denoiser_scheme", _never)
    backend._run_load(
        repo_id = Z_IMAGE_REPO, model_kind = "pipeline", local_files_only = True, _load_token = 1
    )
    assert seen["_pipeline_prequant_planned"] is None
    assert fetched == []


# ── the load ─────────────────────────────────────────────────────────────────────
class _FakePipe:
    def __init__(self) -> None:
        self.transformer = object()

    def to(self, *_a, **_k):
        return self


class _FakePipeline:
    last: dict = {}

    @classmethod
    def from_pretrained(cls, base, **kwargs):
        _FakePipeline.last = {"base": base, **kwargs}
        return _FakePipe()


class _FakeTransformer:
    pass


@pytest.fixture
def fake_runtime(monkeypatch):
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.cuda = types.SimpleNamespace(is_available = lambda: False)
    torch.backends = types.SimpleNamespace(mps = None)
    torch.inference_mode = lambda: contextlib.nullcontext()

    diffusers = types.ModuleType("diffusers")
    diffusers.ZImagePipeline = _FakePipeline
    diffusers.ZImageTransformer2DModel = _FakeTransformer
    diffusers.GGUFQuantizationConfig = lambda compute_dtype = None: ("quant", compute_dtype)

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setattr(dmod, "clear_gpu_cache", lambda: None)
    _FakePipeline.last = {}
    yield


def _load_backend(
    monkeypatch,
    *,
    seeded = True,
    offload = "none",
):
    """A backend whose pipeline assembly, memory plan and quantiser are all observable."""
    backend = DiffusionBackend()
    monkeypatch.setattr(backend, "_target_for_ordinal", lambda *_a, **_k: _target())
    monkeypatch.setattr(dmod, "apply_diffusion_device_ordinal", lambda _t: None)
    monkeypatch.setattr(dmod, "prefer_ungated_mirror", lambda base, *_a, **_k: base)
    monkeypatch.setattr(dmod, "apply_memory_plan", lambda *_a, **_k: ("none", False))
    monkeypatch.setattr(dmod, "raise_on_unified_memory_shortfall", lambda *_a, **_k: None)
    monkeypatch.setattr(DiffusionBackend, "_resident_sized_plan", lambda _s, plan, *_a, **_k: plan)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda _t: True)
    monkeypatch.setattr(tqmod, "dense_transformer_supported", lambda _t: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
    )
    monkeypatch.setattr(dmod, "dense_quant_blocker", lambda _pipe: None)
    monkeypatch.setattr(dmod, "_pipeline_quant_uncompilable_reason", lambda *_a, **_k: None)
    monkeypatch.setattr(dmod, "stored_denoiser_precision", lambda *_a, **_k: None)
    monkeypatch.setattr(dmod, "denoiser_modules", lambda pipe: [("transformer", object())])
    monkeypatch.setattr(pqmod, "restricted_prequant_load_supported", lambda _scheme: True)

    plans: list = []

    def _plan_memory(_self, *_a, **kwargs):
        plans.append(kwargs)
        policy = offload if kwargs.get("transformer_resident_override_mib") is not None else "none"
        return types.SimpleNamespace(
            offload_policy = policy, requested_mode = None, estimates = {}, reasons = ()
        )

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", _plan_memory)

    quantised: list = []

    def _quantize(pipe, target, **kwargs):
        quantised.append(kwargs.get("mode"))
        return "fp8"

    monkeypatch.setattr(dmod, "quantize_transformer", _quantize)

    seeds: list = []

    def _load_prequantized(_cls, base, source, **kwargs):
        seeds.append({"base": base, "source": source, **kwargs})
        return _FakeTransformer() if seeded else None

    monkeypatch.setattr(pqmod, "load_prequantized_transformer", _load_prequantized)

    restored: list = []
    monkeypatch.setattr(
        DiffusionBackend,
        "_prefetch_files",
        lambda _self, repo_id, gguf, base, files, *_a, **_k: restored.append(tuple(files)),
    )
    return backend, types.SimpleNamespace(
        plans = plans, quantised = quantised, seeds = seeds, restored = restored
    )


def _load(backend, **overrides):
    kwargs = {
        "model_kind": "pipeline",
        "_pipeline_prequant_planned": "fp8",
        "_pipeline_prequant_skipped": tuple(DENOISER_SHARDS),
        **overrides,
    }
    return backend.load_pipeline(Z_IMAGE_REPO, **kwargs)


def test_a_seeded_denoiser_engages_the_quant_without_a_second_conversion(fake_runtime, monkeypatch):
    """The pipeline is ASSEMBLED around the quantised denoiser, so the in-memory rewrite that
    would otherwise build it dense first is a no-op for this load."""
    backend, spy = _load_backend(monkeypatch)
    status = _load(backend)

    assert isinstance(_FakePipeline.last["transformer"], _FakeTransformer)
    assert status["transformer_quant"] == "fp8"
    assert spy.quantised == []
    assert spy.restored == []
    seed = spy.seeds[0]
    assert seed["source"].location == PREQUANT_REPO
    assert seed["source"].filename == PREQUANT_FILE
    assert seed["scheme"] == "fp8"
    assert seed["min_features"] == dmod.DEFAULT_MIN_LINEAR_FEATURES
    assert seed["cache_dir"] == dmod.hub_cache_dir()


def test_the_resolved_record_names_the_hosted_file(fake_runtime, monkeypatch):
    """The scheme alone cannot tell the hosted artifact from a runtime quantise of the released
    weights, and the two do not render the same image."""
    backend, _spy = _load_backend(monkeypatch)
    resolved = _load(backend)["resolved"]["transformer_quant"]

    assert resolved["value"] == "fp8"
    assert resolved["artifact"] == f"prequant:{PREQUANT_REPO}/{PREQUANT_FILE}"
    assert PREQUANT_FILE in resolved["reason"]
    # `source` still says who chose the precision, which is what renders the "Auto: FP8" badge.
    assert resolved["source"] == "auto"


def test_a_seed_that_does_not_land_replans_and_tops_up_the_shards(fake_runtime, monkeypatch):
    """Seeding is best-effort, the plan was priced on it landing, and from_pretrained cannot
    re-fetch a dropped shard from a local snapshot dir."""
    backend, spy = _load_backend(monkeypatch, seeded = False)
    status = _load(backend)

    assert "transformer" not in _FakePipeline.last
    assert spy.restored and set(DENOISER_SHARDS) <= set(spy.restored[0])
    # ...and the released weights are quantised in place instead, which is the fallback.
    assert spy.quantised == ["auto"]
    assert status["transformer_quant"] == "fp8"
    assert status["resolved"]["transformer_quant"].get("artifact") is None


def test_an_artifact_sized_plan_that_offloads_at_load_time_drops_the_seed(
    fake_runtime, monkeypatch
):
    """The plan settled this against CAPACITY while the previous pipeline was resident; live free
    memory can be smaller, and torchao tensors cannot ride an offload rotation."""
    backend, spy = _load_backend(monkeypatch, offload = "sequential")
    _load(backend)

    assert spy.seeds == []
    assert spy.restored and set(DENOISER_SHARDS) <= set(spy.restored[0])


def test_a_declined_plan_never_seeds_at_the_load(fake_runtime, monkeypatch):
    backend, spy = _load_backend(monkeypatch)
    # A decline scopes the pull exactly as it was, so the loader is handed no skipped shards either.
    _load(
        backend,
        _pipeline_prequant_planned = PIPELINE_SEED_DECLINED,
        _pipeline_prequant_skipped = (),
    )

    assert spy.seeds == []
    assert spy.restored == []
    assert spy.quantised == ["auto"]


def test_a_direct_load_with_no_plan_phase_keeps_todays_behaviour(fake_runtime, monkeypatch):
    """A direct call has no plan to be pinned to, so it may not take a decision the pull was never
    scoped on: the released shards are on disk and the in-memory quantise is the right path."""
    backend, spy = _load_backend(monkeypatch)
    _load(backend, _pipeline_prequant_planned = None, _pipeline_prequant_skipped = ())

    assert spy.seeds == []
    assert spy.quantised == ["auto"]


def test_an_explicit_scheme_with_a_hosted_artifact_is_seeded_too(monkeypatch):
    """The default is what changes; an explicit request was already honoured and still is."""
    backend = _settle_backend(monkeypatch)
    assert _settle(backend, transformer_quant = "fp8") == "fp8"


def test_an_explicit_scheme_with_no_artifact_falls_to_the_in_memory_path(fake_runtime, monkeypatch):
    """Nothing to seed, so the released weights are quantised in place and an explicit scheme that
    cannot be honoured there still fails closed, exactly as before."""
    backend, spy = _load_backend(monkeypatch)
    status = _load(
        backend,
        transformer_quant = "fp8",
        _pipeline_prequant_planned = None,
        _pipeline_prequant_skipped = (),
    )
    assert spy.seeds == []
    assert spy.quantised == ["fp8"]
    assert status["transformer_quant"] == "fp8"

    monkeypatch.setattr(dmod, "quantize_transformer", lambda *_a, **_k: None)
    monkeypatch.setattr(dmod, "transformer_is_quantised", lambda _m: False)
    with pytest.raises(RuntimeError, match = "transformer_quant='fp8'"):
        _load(
            backend,
            transformer_quant = "fp8",
            _pipeline_prequant_planned = None,
            _pipeline_prequant_skipped = (),
        )
