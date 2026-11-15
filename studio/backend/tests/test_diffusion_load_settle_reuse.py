# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import types

import pytest

from core.inference import diffusion as dmod
from core.inference import diffusion_memory as mem
from core.inference.diffusion_memory import DeviceMemory
from .test_diffusion_backend import fake_runtime, _stub_pipeline_dense_quant

CUDA = types.SimpleNamespace(device = "cuda", backend = "cuda", supports_model_cpu_offload = True)
FAMILY = types.SimpleNamespace(name = "z-image", base_repo = "Tongyi-MAI/Z-Image-Turbo")


def _mock_cleanup(monkeypatch):
    import torch
    from core.inference import diffusion_cuda_graph

    monkeypatch.setattr(diffusion_cuda_graph, "synchronize_outside_capture", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None, raising = False)
    mps = getattr(torch, "mps", None)
    if mps is not None:
        monkeypatch.setattr(mps, "empty_cache", lambda: None, raising = False)


@pytest.fixture
def card(monkeypatch):
    state = types.SimpleNamespace(
        free = 12_000, total = 24_000, reads = 0, sleeps = 0, kind = "discrete_vram"
    )

    def snapshot(target):
        state.reads += 1
        return DeviceMemory(target.backend, target.device, state.kind, state.free, state.total)

    def sleep(_s):
        state.sleeps += 1

    _mock_cleanup(monkeypatch)
    monkeypatch.setattr(mem, "snapshot_device_memory", snapshot)
    monkeypatch.setattr("time.sleep", sleep)
    monkeypatch.setattr(dmod, "_calibrated_activation", lambda *a, **k: None)
    monkeypatch.setattr(dmod, "estimate_image_runtime_mib", lambda **k: 1000)
    monkeypatch.setenv("UNSLOTH_SETTLE_DELAY_S", "1")
    return state


def _replan(
    backend = None,
    target = CUDA,
    family = FAMILY,
    mode = None,
    cpu_offload = False,
    **overrides,
):
    kwargs = dict(
        kind = "pipeline",
        transformer_resident_override_mib = 3000,
        companion_override_mib = 1000,
        text_encoder_override_mib = 800,
    )
    kwargs.update(overrides)
    return (backend or dmod.DiffusionBackend())._plan_memory(
        target, None, "unused", family, mode, cpu_offload, **kwargs
    )


def test_known_resident_replan_uses_one_fresh_read(card):
    result = _replan()
    assert dmod._plan_proves_resident(result)
    assert not result.vae_tiling and not result.vae_slicing
    assert result.device_memory.free_mib == 12_000
    assert 12_000 < card.total - max(2048, int(card.total * 0.10))
    assert (card.reads, card.sleeps) == (1, 0)


def test_each_resident_replan_reads_current_memory(card):
    backend = dmod.DiffusionBackend()
    first = _replan(backend)
    card.free = 14_000
    second = _replan(backend)
    assert first.device_memory.free_mib == 12_000
    assert second.device_memory.free_mib == 14_000
    assert (card.reads, card.sleeps) == (2, 0)


def test_initial_known_pipeline_plan_keeps_settling(card, tmp_path, monkeypatch):
    backend = dmod.DiffusionBackend()
    monkeypatch.setattr(backend, "_local_dir_weight_bytes", lambda *a, **k: 4000 * 2**20)
    monkeypatch.setattr(backend, "_cache_bytes", lambda *a, **k: 0)
    monkeypatch.setattr(backend, "_companion_cache_bytes", lambda *a, **k: 1000 * 2**20)
    monkeypatch.setattr(backend, "_text_encoder_cache_bytes", lambda *a, **k: 800 * 2**20)
    result = _replan(
        backend,
        transformer_resident_override_mib = None,
        repo_id = str(tmp_path),
        base_local_dir = str(tmp_path),
    )
    assert result.estimates["model_dense_mib"] == 4000
    assert dmod._plan_proves_resident(result)
    assert (card.reads, card.sleeps) == (3, 2)


def test_active_multigpu_selection_keeps_settling(card, monkeypatch):
    from core.inference.diffusion_multigpu import (
        CFG_PARALLEL_ENV,
        MULTIGPU_ENV,
        MultiGpuSelection,
        current_selection,
        selection_scope,
    )

    monkeypatch.setenv(MULTIGPU_ENV, "auto")
    monkeypatch.setenv(CFG_PARALLEL_ENV, "auto")
    selection = MultiGpuSelection(primary = 0, secondaries = (1,), automatic = True)
    target = types.SimpleNamespace(**vars(CUDA), ordinal = 0)
    with selection_scope(selection):
        assert current_selection() is selection
        result = _replan(target = target)
    assert current_selection() is None
    assert dmod._plan_proves_resident(result)
    assert (card.reads, card.sleeps) == (3, 2)


def test_explicit_cpu_offload_keeps_settling(card):
    result = _replan(cpu_offload = True)
    assert result.offload_policy == mem.OFFLOAD_MODEL
    assert result.estimates["explicit_cpu_offload"] == 1
    assert (card.reads, card.sleeps) == (3, 2)


def test_supplied_device_memory_bypasses_sampling(card):
    supplied = DeviceMemory("cuda", "cuda", "discrete_vram", 14_000, card.total)
    card.free = 7000
    result = _replan(device_memory_override = supplied)
    assert result.device_memory is supplied
    assert dmod._plan_proves_resident(result)
    assert (card.reads, card.sleeps) == (0, 0)


@pytest.mark.parametrize("mode", ["fast", "balanced", "low_vram"])
def test_explicit_memory_modes_keep_settling(card, mode):
    _replan(mode = mode)
    assert (card.reads, card.sleeps) == (3, 2)


@pytest.mark.parametrize("field", ["companion_override_mib", "text_encoder_override_mib"])
def test_unknown_component_sizes_keep_settling(card, field):
    _replan(**{field: None})
    assert (card.reads, card.sleeps) == (3, 2)


@pytest.mark.parametrize("field", ["free", "total"])
def test_unknown_memory_keeps_settling(card, field):
    setattr(card, field, None)
    _replan()
    assert (card.reads, card.sleeps) == (3, 2)


@pytest.mark.parametrize("backend,kind", [("rocm", "discrete_vram"), ("cuda", "unified_memory")])
def test_other_cuda_memory_kinds_keep_settling(card, backend, kind):
    card.kind = kind
    target = types.SimpleNamespace(device = "cuda", backend = backend, supports_model_cpu_offload = True)
    _replan(target = target)
    assert (card.reads, card.sleeps) == (3, 2)


@pytest.mark.parametrize("device", ["mps", "xpu", "cpu"])
def test_other_devices_read_fresh_without_cuda_waits(card, device):
    target = types.SimpleNamespace(device = device, backend = device, supports_model_cpu_offload = False)
    _replan(target = target)
    card.free = 14_000
    result = _replan(target = target)
    assert result.device_memory.free_mib == 14_000
    assert (card.reads, card.sleeps) == (2, 0)


def test_calibration_probe_stays_after_sampling(card, monkeypatch):
    observed = []
    monkeypatch.setattr(dmod, "_calibrated_activation", lambda *a, **k: observed.append(card.reads))
    _replan()
    assert observed == [1]
    card.free = 8000
    card.reads = card.sleeps = 0
    _replan()
    assert observed == [1, 3]


def test_flat_residency_does_not_override_calibrated_demotion(card, monkeypatch):
    card.free, card.total = 17_400, 32_000
    family = types.SimpleNamespace(name = "qwen-image-2.1", base_repo = "Qwen/Qwen-Image-2.1")
    activation = mem.calibrated_image_activation(family.name, max_speed = True)
    monkeypatch.setattr(dmod, "estimate_image_runtime_mib", lambda **k: 6963)
    monkeypatch.setattr(dmod, "_calibrated_activation", lambda *a, **k: activation)

    def sleep(_s):
        card.sleeps += 1
        card.free = 18_000

    monkeypatch.setattr("time.sleep", sleep)
    result = _replan(
        family = family,
        transformer_resident_override_mib = 2000,
        companion_override_mib = 2000,
        text_encoder_override_mib = 1000,
    )
    assert result.offload_policy == mem.OFFLOAD_NONE
    assert result.device_memory.free_mib == 18_000
    assert (card.reads, card.sleeps) == (3, 2)


def test_constrained_replan_observes_delayed_release(card, monkeypatch):
    card.free = 10_000
    monkeypatch.setattr(dmod, "estimate_image_runtime_mib", lambda **k: 6000)

    def sleep(_s):
        card.sleeps += 1
        card.free = 20_000

    monkeypatch.setattr("time.sleep", sleep)
    result = _replan(
        transformer_resident_override_mib = 3000,
        companion_override_mib = 8000,
        text_encoder_override_mib = 6000,
    )
    assert result.offload_policy == mem.OFFLOAD_GROUP
    assert result.device_memory.free_mib == 20_000
    assert (card.reads, card.sleeps) == (3, 2)


def test_original_sampler_still_waits_without_a_planner_proof(card):
    first = mem.settled_snapshot_device_memory(CUDA)
    second = mem.settled_snapshot_device_memory(CUDA)
    assert first == second
    assert (card.reads, card.sleeps) == (6, 4)


def test_pipeline_quantized_resident_replan_stops_after_its_fresh_read(
    fake_runtime, card, tmp_path, monkeypatch
):
    card.free, card.total = 26_000, 48_000
    monkeypatch.setenv(mem.GROUP_OFFLOAD_PIN_ENV, "1")
    monkeypatch.setattr(mem, "_installed_diffusers_version", lambda: (0, 39))
    monkeypatch.setattr(mem, "_installed_torchao_version", lambda: (0, 18))
    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    backend = dmod.DiffusionBackend()
    quantizations = _stub_pipeline_dense_quant(backend, monkeypatch)
    monkeypatch.setattr(backend, "_local_dir_weight_bytes", lambda *a, **k: 22_000 * 2**20)
    monkeypatch.setattr(backend, "_cache_bytes", lambda *a, **k: 0)
    monkeypatch.setattr(backend, "_companion_cache_bytes", lambda *a, **k: 8000 * 2**20)
    monkeypatch.setattr(backend, "_text_encoder_cache_bytes", lambda *a, **k: 7800 * 2**20)
    plans, accepted = [], []
    real_plan = dmod.DiffusionBackend._plan_memory
    real_sample = dmod.settled_snapshot_device_memory

    def plan(self, *args, **kwargs):
        result = real_plan(self, *args, **kwargs)
        plans.append(result)
        return result

    def sample(target, *args, **kwargs):
        predicate = kwargs.get("can_stop")
        if predicate is not None:

            def record(memory):
                result = predicate(memory)
                accepted.append(result)
                return result

            kwargs["can_stop"] = record
        return real_sample(target, *args, **kwargs)

    monkeypatch.setattr(dmod.DiffusionBackend, "_plan_memory", plan)
    monkeypatch.setattr(dmod, "settled_snapshot_device_memory", sample)
    monkeypatch.setattr(
        dmod, "apply_memory_plan", lambda pipe, plan, **kwargs: (plan.offload_policy, False)
    )
    status = backend.load_pipeline(
        "Tongyi-MAI/Z-Image-Turbo", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert len(quantizations) == 1
    assert len(plans) >= 2
    assert plans[0].offload_policy != mem.OFFLOAD_NONE
    assert dmod._plan_proves_resident(plans[-1])
    assert accepted == [True]
    assert (card.reads, card.sleeps) == (4, 2)
    assert status["transformer_quant"] == "fp8"
    assert status["offload_policy"] == mem.OFFLOAD_NONE
    assert status["resolved"]["transformer_quant"]["status"] == "applied"
    assert status["speed_mode"] == "default"


@pytest.mark.parametrize("enabled", [None, "0", "false", "1"])
def test_delayed_release_preserves_resident_denoiser(fake_runtime, tmp_path, monkeypatch, enabled):
    if enabled is None:
        monkeypatch.delenv("UNSLOTH_DIFFUSION_REUSE_SETTLED_MEMORY", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_REUSE_SETTLED_MEMORY", enabled)
    monkeypatch.setenv("UNSLOTH_SETTLE_DELAY_S", "1")
    monkeypatch.setenv(mem.GROUP_OFFLOAD_PIN_ENV, "1")
    monkeypatch.setattr(mem, "_installed_diffusers_version", lambda: (0, 39))
    monkeypatch.setattr(mem, "_installed_torchao_version", lambda: (0, 18))
    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    _mock_cleanup(monkeypatch)
    backend = dmod.DiffusionBackend()
    quantizations = _stub_pipeline_dense_quant(backend, monkeypatch)
    clock = [0]
    plans = []
    real_plan = dmod.DiffusionBackend._plan_memory

    def sleep(delay):
        clock[0] += delay

    def snapshot(target):
        return DeviceMemory(
            "cuda", "cuda", "discrete_vram", 10_000 if clock[0] < 3 else 20_000, 24_000
        )

    def plan(self, *args, **kwargs):
        result = real_plan(self, *args, **kwargs)
        plans.append(result)
        return result

    monkeypatch.setattr("time.sleep", sleep)
    monkeypatch.setattr(mem, "snapshot_device_memory", snapshot)
    monkeypatch.setattr(dmod.DiffusionBackend, "_plan_memory", plan)
    monkeypatch.setattr(
        dmod, "apply_memory_plan", lambda pipe, plan, **kwargs: (plan.offload_policy, False)
    )
    status = backend.load_pipeline(
        "Tongyi-MAI/Z-Image-Turbo", model_kind = "pipeline", _base_local_dir = str(tmp_path)
    )
    assert len(quantizations) == 1
    assert len(plans) == 2
    assert [p.device_memory.free_mib for p in plans] == [10_000, 20_000]
    assert status["transformer_quant"] == "fp8"
    assert status["offload_policy"] == "group"
