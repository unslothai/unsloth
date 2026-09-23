# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic tests for seeding an image PIPELINE pick with its hosted pre-quantized denoiser, which
REPLACES the released ``transformer/`` shards. torch, diffusers and the Hub are stubbed."""

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
PREQUANT_FILE = "Z-Image-Turbo-FP8.safetensors"
PREQUANT_BYTES = 6000 * MIB

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


def test_the_plan_drops_the_released_denoiser_shards_and_keeps_its_config(hub):
    """The pull drops the denoiser shards but keeps the config the seeding loader meta-inits from."""
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
    monkeypatch.setattr(
        DiffusionBackend, "_files_already_cached", staticmethod(lambda *_a, **_k: set())
    )
    monkeypatch.setattr(
        DiffusionBackend, "_hub_file_is_cached", staticmethod(lambda *_a, **_k: False)
    )
    monkeypatch.setattr(DiffusionBackend, "declared_footprint_shortfall", lambda *_a, **_k: None)
    return backend


def test_the_plan_counts_and_stages_the_hosted_checkpoint(monkeypatch, hub):
    backend = _plan_backend(monkeypatch, planned = "fp8")
    plan = backend.download_plan(Z_IMAGE_REPO, model_kind = "pipeline")

    entries = {entry["repo_id"]: entry for entry in plan["entries"]}
    assert entries[PREQUANT_REPO]["files"] == [PREQUANT_FILE]
    assert entries[PREQUANT_REPO]["bytes"] == PREQUANT_BYTES
    assert not [f for f in entries[Z_IMAGE_REPO]["files"] if f in DENOISER_SHARDS]
    assert "transformer/config.json" in entries[Z_IMAGE_REPO]["files"]
    assert plan["required_bytes"] == ALL_BYTES - DENOISER_BYTES + PREQUANT_BYTES
    assert entries[Z_IMAGE_REPO]["checkpoint"] is True
    assert entries[PREQUANT_REPO]["checkpoint"] is False


def test_an_unplanned_pipeline_pick_plans_exactly_as_before(monkeypatch, hub):
    """With no hosted artifact the released shards stay in the pull and the plan is unchanged."""
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
    """The unified-memory refusal prices the denoiser from the artifact, not the dropped shards."""
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
    """A GGUF pick never sizes an uncached hosted checkpoint: there it is a SECOND denoiser."""
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
    monkeypatch.setattr(
        pqmod, "restricted_prequant_load_supported", lambda _scheme, filename = None: True
    )
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
    monkeypatch.setattr(
        pqmod, "restricted_prequant_load_supported", lambda _scheme, filename = None: True
    )
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
    assert _settle(_settle_backend(monkeypatch, scheme = "int8")) == "int8"


@pytest.mark.parametrize("quant", ["none", "off"])
def test_precision_off_keeps_the_released_weights(monkeypatch, quant):
    assert _settle(_settle_backend(monkeypatch), transformer_quant = quant) is None


def test_speed_off_keeps_the_released_weights(monkeypatch):
    """Speed=off keeps the released weights: it is a bit-exact contract."""
    assert _settle(_settle_backend(monkeypatch), speed_mode = "off") is None


def test_an_explicit_scheme_under_speed_off_still_seeds(monkeypatch):
    """Speed=off only silences an AUTO precision. An explicit scheme is still quantized and upgrades
    the speed to `default`, so it must seed the hosted checkpoint rather than pull the bf16 shards."""
    backend = _settle_backend(monkeypatch)
    assert _settle(backend, transformer_quant = "fp8", speed_mode = "off") == "fp8"


def test_a_baked_lora_keeps_the_dense_path(monkeypatch):
    """A LoRA bake keeps the dense path; an all-zero list bakes nothing and is unaffected."""
    backend = _settle_backend(monkeypatch)
    assert _settle(backend, loras = [("adapter", 0.8)]) is None
    assert _settle(backend, loras = [("adapter", 0.0)]) == "fp8"


@pytest.mark.parametrize("mode", ["balanced", "low_vram"])
def test_a_definite_offload_request_keeps_the_released_weights(monkeypatch, mode):
    assert _settle(_settle_backend(monkeypatch), memory_mode = mode) is None


def test_an_artifact_sized_plan_that_still_offloads_declines(monkeypatch):
    """An artifact-sized plan that still offloads pins a decline: torchao rejects offload hooks."""
    backend = _settle_backend(monkeypatch, offload = "sequential")
    assert _settle(backend) == PIPELINE_SEED_DECLINED


def _settle_backend_walking(monkeypatch, *, artifacts: tuple, candidates: tuple):
    """A backend whose memory verdict depends on the rung: an int8-sized plan (31 GB) offloads, an
    fp8-sized one (19 GB) stays resident. ``artifacts`` names the schemes with a hosted file,
    ``candidates`` the auto ladder."""
    from core.inference import diffusion_transformer_quant as tq

    backend = _settle_backend(monkeypatch, scheme = candidates[0])
    monkeypatch.setattr(tq, "auto_scheme_candidates", lambda target, family = None: candidates)
    monkeypatch.setattr(
        dmod,
        "denoiser_prequant_source",
        lambda fam, scheme, **_k: ("unsloth/Qwen-Image-FP8", f"{scheme}.pt")
        if scheme in artifacts
        else None,
    )
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **k: DenseQuantEstimate(
            scheme = k["requested"],
            steady_transformer_mib = 31_000 if k["requested"] == "int8" else 19_000,
            transient_transformer_mib = 0,
            companions_mib = 7_500,
            prequant = True,
            download_transformer_mib = 0,
            text_encoders_mib = 7_320,
        ),
    )
    monkeypatch.setattr(
        DiffusionBackend,
        "_plan_memory",
        lambda *_a, **k: types.SimpleNamespace(
            offload_policy = "sequential"
            if k["transformer_resident_override_mib"] >= 31_000
            else "none"
        ),
    )
    return backend


def test_an_artifact_too_large_for_the_card_yields_to_the_next_hosted_rung(monkeypatch):
    """int8 leads the ladder, but Qwen-Image's int8 file is 12 GB larger than its fp8 one: where the
    int8-sized plan offloads, auto seeds the fp8 artifact instead of pinning a decline."""
    backend = _settle_backend_walking(
        monkeypatch, artifacts = ("int8", "fp8"), candidates = ("int8", "fp8")
    )
    assert _settle(backend) == "fp8"


def test_a_walk_with_no_resident_rung_declines(monkeypatch):
    """Every hosted rung offloads: the decline is pinned so plan and load agree."""
    backend = _settle_backend_walking(monkeypatch, artifacts = ("int8",), candidates = ("int8", "fp8"))
    assert _settle(backend) == PIPELINE_SEED_DECLINED


def test_an_explicit_scheme_is_never_swapped_for_a_lower_rung(monkeypatch):
    """An explicit int8 that offloads declines; auto's walk is not offered to an explicit request."""
    backend = _settle_backend_walking(
        monkeypatch, artifacts = ("int8", "fp8"), candidates = ("int8", "fp8")
    )
    assert _settle(backend, transformer_quant = "int8") == PIPELINE_SEED_DECLINED


def test_a_family_with_no_hosted_artifact_falls_through(monkeypatch):
    """A scheme with no hosted artifact falls through to the in-memory quantise."""
    assert _settle(_settle_backend(monkeypatch, scheme = "nvfp4")) is None


def test_a_base_with_no_hosted_artifact_falls_through(monkeypatch):
    """An excluded base (the undistilled Z-Image) quantises its own weights."""
    assert _settle(_settle_backend(monkeypatch), base = "Tongyi-MAI/Z-Image") is None


def test_a_forced_fp8_accumulate_the_artifact_cannot_bake_keeps_the_released_weights(monkeypatch):
    """The hosted fp8 checkpoints bake fast accumulate, and ``_validate_checkpoint`` refuses a baked
    value differing from a FORCED one, so seeding drops the shards for a checkpoint the load rejects."""
    backend = _settle_backend(monkeypatch)
    assert _settle(backend, transformer_quant = "fp8", fast_accum = False) is None
    assert _settle(backend, transformer_quant = "fp8", fast_accum = True) == "fp8"
    assert _settle(backend, transformer_quant = "fp8") == "fp8"


def test_a_forced_accumulate_never_blocks_a_scheme_that_bakes_none(monkeypatch):
    """Only fp8 records ``fast_accum``, so the int8 artifact is seeded whatever the caller forces."""
    backend = _settle_backend(monkeypatch, scheme = "int8")
    assert _settle(backend, transformer_quant = "int8", fast_accum = False) == "int8"


def test_a_gguf_pick_is_never_settled_here(monkeypatch):
    assert _settle(_settle_backend(monkeypatch), kind = "gguf") is None


@pytest.mark.parametrize("family", ["krea-2", "ideogram-4"])
def test_a_per_component_assembler_is_never_seeded(monkeypatch, family):
    """A per-component assembler is never seeded: it never sees ``pipe_kwargs``."""
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
    """An uncompilable host keeps the released weights, since the loader would refuse the seed."""
    backend = _settle_backend(monkeypatch)
    monkeypatch.setattr(dmod, "_pipeline_quant_uncompilable_reason", lambda *_a, **_k: "no triton")
    assert _settle(backend) is None


def test_an_unanswerable_probe_keeps_the_released_weights(monkeypatch):
    backend = _settle_backend(monkeypatch)

    def _boom(*_a, **_k):
        raise RuntimeError("driver went away")

    monkeypatch.setattr(dmod, "select_transformer_quant_scheme", _boom)
    assert _settle(backend) is None


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
    assert fetched == [(PREQUANT_REPO, PREQUANT_FILE, PREQUANT_BYTES)]
    assert PREQUANT_REPO in backend.loading_repo_ids() or backend._loading is None


def test_an_artifact_that_does_not_resolve_keeps_the_released_shards(monkeypatch, hub):
    """An artifact that does not resolve on the Hub keeps the released shards in the pull."""
    backend, seen, fetched = _run_load_backend(monkeypatch, planned = "fp8", verified = False)
    backend._run_load(repo_id = Z_IMAGE_REPO, model_kind = "pipeline", _load_token = 1)

    assert seen["_pipeline_prequant_planned"] is None
    assert seen["_pipeline_prequant_skipped"] == ()
    assert fetched == []


def test_a_local_checkpoint_is_still_seeded_on_an_online_load(monkeypatch, hub):
    """An operator's own checkpoint has no Hub entry, so the plan drops no shards for it; the seed
    must still survive the pull, or the load quantises bf16 in memory and ignores the given file."""
    backend, seen, fetched = _run_load_backend(monkeypatch, planned = "fp8", verified = False)
    monkeypatch.setattr(dmod, "denoiser_prequant_cached", lambda *_a, **_k: True)
    backend._run_load(
        repo_id = Z_IMAGE_REPO,
        model_kind = "pipeline",
        transformer_prequant_path = "/models/z-image-fp8.pt",
        _load_token = 1,
    )

    assert seen["_pipeline_prequant_planned"] == "fp8"
    assert seen["_pipeline_prequant_skipped"] == ()
    assert fetched == []


def test_a_local_path_the_loader_would_refuse_keeps_the_released_shards(monkeypatch, hub):
    """The seed rides on the artifact being there: a path that does not resolve keeps the bf16
    shards rather than pinning a seed the load cannot take."""
    backend, seen, _fetched = _run_load_backend(monkeypatch, planned = "fp8", verified = False)
    monkeypatch.setattr(dmod, "denoiser_prequant_cached", lambda *_a, **_k: False)
    backend._run_load(
        repo_id = Z_IMAGE_REPO,
        model_kind = "pipeline",
        transformer_prequant_path = "/models/missing.pt",
        _load_token = 1,
    )

    assert seen["_pipeline_prequant_planned"] is None


def test_the_decline_is_pinned_across_plan_and_load(monkeypatch, hub):
    """A decline is pinned from plan to load, so the loader never re-takes it and fetches inline."""
    backend, seen, _fetched = _run_load_backend(monkeypatch, planned = PIPELINE_SEED_DECLINED)
    backend._run_load(repo_id = Z_IMAGE_REPO, model_kind = "pipeline", _load_token = 1)

    assert seen["_pipeline_prequant_planned"] == PIPELINE_SEED_DECLINED
    assert seen["_pipeline_prequant_skipped"] == ()


def test_an_offline_load_never_probes_the_hub_for_an_artifact(monkeypatch):
    """An offline load answers from the cache alone, so it never probes the Hub."""
    backend, seen, fetched = _run_load_backend(monkeypatch, planned = "fp8")

    def _never(*_a, **_k):
        raise AssertionError("the offline path asked the Hub about a checkpoint")

    monkeypatch.setattr(DiffusionBackend, "_dit_prequant_plan_source", _never)
    monkeypatch.setattr(dmod, "denoiser_prequant_cached", lambda *_a, **_k: False)
    backend._run_load(
        repo_id = Z_IMAGE_REPO, model_kind = "pipeline", local_files_only = True, _load_token = 1
    )
    assert seen["_pipeline_prequant_planned"] is None
    assert fetched == []


def test_an_offline_reload_seeds_from_the_cached_artifact(monkeypatch):
    """The cache the first load built is reusable: an API reload seeds instead of assembling bf16
    from a snapshot whose released denoiser shards that load deliberately left out."""
    backend, seen, fetched = _run_load_backend(monkeypatch, planned = "fp8")

    def _never(*_a, **_k):
        raise AssertionError("the offline path asked the Hub about a checkpoint")

    monkeypatch.setattr(DiffusionBackend, "_dit_prequant_plan_source", _never)
    monkeypatch.setattr(dmod, "denoiser_prequant_cached", lambda *_a, **_k: True)
    backend._run_load(
        repo_id = Z_IMAGE_REPO, model_kind = "pipeline", local_files_only = True, _load_token = 1
    )
    assert seen["_pipeline_prequant_planned"] == "fp8"
    assert seen["_pipeline_prequant_skipped"] == ()
    assert fetched == []


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
    monkeypatch.setattr(
        pqmod, "restricted_prequant_load_supported", lambda _scheme, filename = None: True
    )

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
    """``resolved`` names the hosted FILE, which renders differently from a runtime quantise."""
    backend, _spy = _load_backend(monkeypatch)
    resolved = _load(backend)["resolved"]["transformer_quant"]

    assert resolved["value"] == "fp8"
    assert resolved["artifact"] == f"prequant:{PREQUANT_REPO}/{PREQUANT_FILE}"
    assert PREQUANT_FILE in resolved["reason"]
    # `source` must stay "auto"/"explicit": the frontend branches on it.
    assert resolved["source"] == "auto"


def test_a_seed_that_does_not_land_replans_and_tops_up_the_shards(fake_runtime, monkeypatch):
    """A failed seed re-plans at bf16 and restores shards from_pretrained cannot re-fetch."""
    backend, spy = _load_backend(monkeypatch, seeded = False)
    status = _load(backend)

    assert "transformer" not in _FakePipeline.last
    assert spy.restored and set(DENOISER_SHARDS) <= set(spy.restored[0])
    assert spy.quantised == ["auto"]
    assert status["transformer_quant"] == "fp8"
    assert status["resolved"]["transformer_quant"].get("artifact") is None


def test_a_top_up_that_spans_cache_roots_assembles_from_the_hub_id(fake_runtime, monkeypatch):
    """A cache-folder change can leave the manifest in the old root and the restored shards in the
    live one. No snapshot then holds both, so assembly must fall back to the hub id: from_pretrained
    would treat the staged directory as terminal with no transformer in it."""
    backend, _spy = _load_backend(monkeypatch, seeded = False)
    monkeypatch.setattr(DiffusionBackend, "_prefetch_files", lambda *_a, **_k: None)
    _load(backend, _base_local_dir = "/old/root/snapshots/abc")

    assert _FakePipeline.last["base"] == Z_IMAGE_REPO


def test_a_top_up_never_promotes_a_snapshot_the_staging_did_not_hand_back(
    fake_runtime, monkeypatch
):
    """The top-up stages the shards only. A snapshot the staging never returned holds them without
    the companions, so it must not become the directory the pipeline assembles from."""
    backend, _spy = _load_backend(monkeypatch, seeded = False)
    monkeypatch.setattr(
        DiffusionBackend, "_prefetch_files", lambda *_a, **_k: "/live/root/snapshots/abc"
    )
    _load(backend, _base_local_dir = None)

    assert _FakePipeline.last["base"] == Z_IMAGE_REPO


def test_a_single_root_top_up_keeps_assembling_from_the_staged_snapshot(fake_runtime, monkeypatch):
    """The ordinary case: the shards land in the staged snapshot, which keeps from_pretrained off
    the hub."""
    backend, _spy = _load_backend(monkeypatch, seeded = False)
    monkeypatch.setattr(
        DiffusionBackend, "_prefetch_files", lambda *_a, **_k: "/live/root/snapshots/abc"
    )
    _load(backend, _base_local_dir = "/live/root/snapshots/abc")

    assert _FakePipeline.last["base"] == "/live/root/snapshots/abc"


def test_an_artifact_sized_plan_that_offloads_at_load_time_drops_the_seed(
    fake_runtime, monkeypatch
):
    """A plan that offloads at load time drops the seed: torchao tensors reject offload hooks."""
    backend, spy = _load_backend(monkeypatch, offload = "sequential")
    _load(backend)

    assert spy.seeds == []
    assert spy.restored and set(DENOISER_SHARDS) <= set(spy.restored[0])


def test_the_artifact_label_names_the_file_that_really_loaded():
    """A repo serving only its fallback filename is labelled with the fallback, not the primary."""
    # _resolve_checkpoint_path falls back when the primary name is absent, so labelling from
    # source.filename would publish provenance for a file nobody fetched.
    import types

    from core.inference.diffusion_denoiser_prequant import prequant_artifact_label

    source = types.SimpleNamespace(
        kind = "repo", location = "unsloth/Z-Image-Turbo-FP8", filename = "Z-Image-Turbo-FP8.pt"
    )
    assert prequant_artifact_label(source) == (
        "prequant:unsloth/Z-Image-Turbo-FP8/Z-Image-Turbo-FP8.pt"
    )
    loaded = types.SimpleNamespace(_unsloth_prequant_path = "/cache/blobs/Z-Image-Turbo-fp8.pt")
    assert prequant_artifact_label(source, loaded) == (
        "prequant:unsloth/Z-Image-Turbo-FP8/Z-Image-Turbo-fp8.pt"
    )
    local = types.SimpleNamespace(kind = "path", location = "/models/mine.pt", filename = None)
    assert prequant_artifact_label(local, loaded) == "prequant:/models/mine.pt"


def test_a_dropped_seed_replans_once_the_dense_shards_are_back(fake_runtime, monkeypatch):
    """The plan the dense load runs on is taken AFTER the skipped transformer shards are restored."""
    # A pipeline plan prices CACHED bytes, so one taken while transformer/ was skipped saw companions
    # only: left in place it reads 'none' and the load keeps the bf16 denoiser resident.
    backend, spy = _load_backend(monkeypatch, offload = "sequential")
    _load(backend)

    assert spy.seeds == []
    assert spy.restored and set(DENOISER_SHARDS) <= set(spy.restored[0])
    assert len(spy.plans) == 3
    assert spy.plans[-1].get("transformer_resident_override_mib") is None


def test_nothing_skipped_takes_no_extra_plan(fake_runtime, monkeypatch):
    """A load that skipped no shards re-plans nothing: there is no under-counted plan to redo."""
    backend, spy = _load_backend(monkeypatch, offload = "sequential")
    _load(backend, _pipeline_prequant_skipped = ())

    assert spy.restored == []
    assert len(spy.plans) == 2


def test_a_declined_plan_never_seeds_at_the_load(fake_runtime, monkeypatch):
    backend, spy = _load_backend(monkeypatch)
    _load(
        backend,
        _pipeline_prequant_planned = PIPELINE_SEED_DECLINED,
        _pipeline_prequant_skipped = (),
    )

    assert spy.seeds == []
    assert spy.restored == []
    assert spy.quantised == ["auto"]


def test_a_direct_load_with_no_plan_phase_keeps_todays_behaviour(fake_runtime, monkeypatch):
    """A direct call with no plan phase never seeds a decision the pull was not scoped on."""
    backend, spy = _load_backend(monkeypatch)
    _load(backend, _pipeline_prequant_planned = None, _pipeline_prequant_skipped = ())

    assert spy.seeds == []
    assert spy.quantised == ["auto"]


def test_an_explicit_scheme_with_a_hosted_artifact_is_seeded_too(monkeypatch):
    backend = _settle_backend(monkeypatch)
    assert _settle(backend, transformer_quant = "fp8") == "fp8"


def test_an_explicit_scheme_with_no_artifact_falls_to_the_in_memory_path(fake_runtime, monkeypatch):
    """With nothing to seed, an explicit scheme quantises in place and still fails closed."""
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


@pytest.mark.parametrize(
    "family, repo",
    [
        ("z-image", "unsloth/Z-Image-Turbo-FP8"),
        ("flux.1-schnell", "unsloth/FLUX.1-schnell-FP8"),
        ("qwen-image", "unsloth/Qwen-Image-FP8"),
    ],
)
def test_the_rebuilt_fp8_artifacts_are_listed_for_both_schemes(family, repo):
    """The hosted repos carry an fp8 and an int8 denoiser file, so an fp8 host seeds instead of quantising in memory."""
    from core.inference.diffusion_families import detect_family, family_prequant_repo

    fam = detect_family(family)
    assert fam is not None
    for scheme in ("fp8", "int8"):
        assert family_prequant_repo(fam, scheme) == repo
