# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A torchao int8 / fp8 denoiser stays quantised when the image plan offloads it."""

from __future__ import annotations

import types
from dataclasses import replace

import pytest

from core.inference import diffusion as dmod
from core.inference import diffusion_memory as mem
from core.inference.diffusion_denoiser_prequant import PIPELINE_SEED_DECLINED
from core.inference.diffusion_memory import (
    OFFLOAD_GROUP,
    OFFLOAD_MODEL,
    OFFLOAD_NONE,
    OFFLOAD_SEQUENTIAL,
    OFFLOAD_STREAMING,
    DeviceMemory,
    plan_diffusion_memory,
    torchao_offload_plan,
    torchao_survives_plan,
)

T017, T018, T016 = (0, 17), (0, 18), (0, 16)


@pytest.fixture(autouse = True)
def _roomy_host(monkeypatch):
    monkeypatch.setattr(mem, "_pin_budget_mib", lambda: 10**7)
    monkeypatch.setattr(mem, "_pinned_memory_capped", lambda: False)
    monkeypatch.delenv(mem.GROUP_OFFLOAD_PIN_ENV, raising = False)
    # Studio's diffusers pin; a runner without diffusers would otherwise refuse every streamed tier.
    monkeypatch.setattr(mem, "_installed_diffusers_version", lambda: (0, 40))


def _plan(
    policy,
    *,
    stream_transformer = True,
    fits_model_offload = True,
):
    device = DeviceMemory("cuda", "cuda:0", "discrete_vram", 15_000, 16_384)
    plan = plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = device,
        model_dense_mib = 12_000,
        runtime_headroom_mib = 2_000,
        companion_dense_mib = 4_000,
        text_encoder_dense_mib = 3_800,
    )
    estimates = dict(plan.estimates)
    if not fits_model_offload:
        estimates["model_dense_mib"] = 40_000
    return replace(
        plan,
        offload_policy = policy,
        stream_transformer = stream_transformer,
        estimates = estimates,
    )


PLACEMENTS = {
    "resident": _plan(OFFLOAD_NONE),
    "group_dit_resident": _plan(OFFLOAD_GROUP, stream_transformer = False),
    "group_dit_streamed": _plan(OFFLOAD_GROUP),
    "model_fits": _plan(OFFLOAD_MODEL),
    "model_too_big": _plan(OFFLOAD_MODEL, fits_model_offload = False),
    "streaming": _plan(OFFLOAD_STREAMING),
    "sequential": _plan(OFFLOAD_SEQUENTIAL),
}

# placement -> scheme -> torchao version -> (survives, policy it runs on)
SURVIVAL = {
    "resident": {
        s: {v: (True, OFFLOAD_NONE) for v in (T016, T017, T018)} for s in ("int8", "fp8", "nvfp4")
    },
    "group_dit_resident": {
        s: {v: (True, OFFLOAD_GROUP) for v in (T016, T017, T018)} for s in ("int8", "fp8", "nvfp4")
    },
    "group_dit_streamed": {
        # 0.17 streams int8 through its pinnable Int8Tensor (UNSLOTH_DIFFUSION_INT8_STREAM_TORCHAO17)
        "int8": {T016: (False, None), T017: (True, OFFLOAD_GROUP), T018: (True, OFFLOAD_GROUP)},
        "fp8": {T016: (False, None), T017: (True, OFFLOAD_GROUP), T018: (True, OFFLOAD_GROUP)},
        "nvfp4": {T016: (False, None), T017: (False, None), T018: (False, None)},
    },
    "model_fits": {
        s: {v: (True, OFFLOAD_MODEL) for v in (T016, T017, T018)} for s in ("int8", "fp8", "nvfp4")
    },
    "model_too_big": {
        "int8": {
            T016: (False, None),
            T017: (True, OFFLOAD_STREAMING),
            T018: (True, OFFLOAD_STREAMING),
        },
        "fp8": {
            T016: (False, None),
            T017: (True, OFFLOAD_STREAMING),
            T018: (True, OFFLOAD_STREAMING),
        },
        "nvfp4": {T016: (False, None), T017: (False, None), T018: (False, None)},
    },
    "streaming": {
        "int8": {
            T016: (False, None),
            T017: (True, OFFLOAD_STREAMING),
            T018: (True, OFFLOAD_STREAMING),
        },
        "fp8": {
            T016: (False, None),
            T017: (True, OFFLOAD_STREAMING),
            T018: (True, OFFLOAD_STREAMING),
        },
        "nvfp4": {T016: (False, None), T017: (False, None), T018: (False, None)},
    },
    "sequential": {
        s: {v: (False, None) for v in (T016, T017, T018)} for s in ("int8", "fp8", "nvfp4")
    },
}
CASES = [
    (placement, scheme, version, *expected)
    for placement, by_scheme in SURVIVAL.items()
    for scheme, by_version in by_scheme.items()
    for version, expected in by_version.items()
]


@pytest.mark.parametrize("placement, scheme, version, survives, policy", CASES)
def test_survival_table(monkeypatch, placement, scheme, version, survives, policy):
    # the table is about versions, not about which torchao this test host happens to have
    monkeypatch.setattr(mem, "_int8_tensor_pinnable", lambda: True, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_STREAM_TORCHAO17", raising = False)
    plan = PLACEMENTS[placement]
    assert torchao_survives_plan(plan, scheme, torchao_version = version) is survives
    placed = torchao_offload_plan(plan, scheme, torchao_version = version)
    assert (placed.offload_policy if placed is not None else None) == policy


@pytest.mark.parametrize(
    "diffusers_version, streams",
    [((0, 36), False), ((0, 37), False), ((0, 38), True), (None, False)],
)
def test_diffusers_before_the_torchao_swap_keeps_the_resident_rule(
    monkeypatch, diffusers_version, streams
):
    monkeypatch.setattr(mem, "_installed_diffusers_version", lambda: diffusers_version)
    assert (
        torchao_survives_plan(PLACEMENTS["group_dit_streamed"], "int8", torchao_version = T018)
        is streams
    )
    assert torchao_survives_plan(PLACEMENTS["group_dit_resident"], "int8", torchao_version = T018)


def test_swap_retry_collects_once_then_gives_up(monkeypatch):
    import sys

    calls: list = []
    failures = {"n": 1}

    def _swap(param, source):
        calls.append(param)
        if failures["n"] > 0:
            failures["n"] -= 1
            raise RuntimeError("Cannot swap t1 because it has weakref associated with it")

    go = types.SimpleNamespace(_swap_torchao_tensor = _swap)
    hooks = types.ModuleType("diffusers.hooks")
    hooks.group_offloading = go
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)
    monkeypatch.setitem(sys.modules, "diffusers.hooks.group_offloading", go)
    assert mem.install_group_offload_torchao_swap_retry()
    assert not mem.install_group_offload_torchao_swap_retry()
    go._swap_torchao_tensor("p", "s")
    assert calls == ["p", "p"]
    failures["n"] = 2
    with pytest.raises(RuntimeError, match = "weakref"):
        go._swap_torchao_tensor("p", "s")


@pytest.mark.parametrize(
    "capped, budget, env, streams",
    [
        (False, 10**7, None, True),
        (True, 10**7, None, False),
        (False, 7_999, None, False),
        (False, None, None, False),
        (False, 10**7, "0", False),
        (True, 0, "1", True),
    ],
)
def test_a_streamed_torchao_denoiser_needs_its_pin(monkeypatch, capped, budget, env, streams):
    monkeypatch.setattr(mem, "_pinned_memory_capped", lambda: capped)
    monkeypatch.setattr(mem, "_pin_budget_mib", lambda: budget)
    if env is not None:
        monkeypatch.setenv(mem.GROUP_OFFLOAD_PIN_ENV, env)
    for name in ("group_dit_streamed", "streaming", "model_too_big"):
        assert (
            torchao_survives_plan(PLACEMENTS[name], "int8", torchao_version = T018) is streams
        ), name
    assert torchao_survives_plan(PLACEMENTS["group_dit_resident"], "int8", torchao_version = T018)
    assert torchao_survives_plan(PLACEMENTS["model_fits"], "int8", torchao_version = T018)


def test_no_torchao_keeps_the_resident_rule():
    assert not torchao_survives_plan(PLACEMENTS["group_dit_streamed"], "int8", torchao_version = None)
    assert torchao_survives_plan(PLACEMENTS["group_dit_resident"], "int8", torchao_version = None)


def test_never_moves_still_means_resident_only():
    keeps = mem.plan_keeps_transformer_resident
    assert keeps(PLACEMENTS["resident"]) and keeps(PLACEMENTS["group_dit_resident"])
    for name in ("group_dit_streamed", "model_fits", "streaming"):
        assert not keeps(PLACEMENTS[name]), name


class _Weight:
    def __init__(
        self,
        cls_name: str,
        module: str = "torchao.quantization.fake",
    ):
        self.data = None
        self.__class__ = type(cls_name, (_Weight,), {"__module__": module})


class _Module:
    def __init__(
        self,
        *class_names: str,
        plain: bool = False,
    ):
        self._params = [_Weight(name) for name in class_names]
        if plain:
            self._params.append(_Weight("Parameter", module = "torch.nn.parameter"))

    def parameters(self):
        return iter(self._params)


STREAM_KW = {
    "onload_device": "cuda",
    "use_stream": True,
    "non_blocking": True,
    "record_stream": True,
}
NO_STREAM = {"onload_device": "cuda", "use_stream": False}


@pytest.mark.parametrize(
    "classes, low_cpu_mem_usage, pin_budget, expected",
    [
        (("Int8Tensor",), False, 10**7, "as is"),
        (("Float8Tensor",), False, 10**7, "as is"),
        (("Int8Tensor", "Float8Tensor"), False, 10**7, "as is"),
        (("Int8Tensor",), True, 10**7, "pinned"),
        (("Float8Tensor",), True, 10**7, "pinned"),
        (("Int8Tensor",), True, None, "no stream"),
        # torchao 0.17 default int8: no aten.is_pinned of its own; streams once Studio registers the pin ops
        (("LinearActivationQuantizedTensor",), False, 10**7, "as is"),
        (("LinearActivationQuantizedTensor",), False, 10**7, "no stream, no pin ops"),
    ],
)
def test_group_offload_kwargs_per_weight_class(
    monkeypatch, classes, low_cpu_mem_usage, pin_budget, expected
):
    monkeypatch.setattr(mem, "_pin_budget_mib", lambda: pin_budget)
    # whether this host's torchao ships the v1 classes must not decide the table
    pin_ops = expected != "no stream, no pin ops"
    monkeypatch.setattr(mem, "install_torchao_v1_int8_pin_ops", lambda: pin_ops, raising = False)
    kwargs = {**STREAM_KW, "low_cpu_mem_usage": low_cpu_mem_usage}
    out = mem._torchao_group_offload_kwargs(_Module(*classes, plain = True), kwargs)
    if expected == "as is":
        assert out is kwargs
    elif expected == "pinned":
        assert out == {**STREAM_KW, "low_cpu_mem_usage": False}
    else:
        assert out == NO_STREAM


def test_group_offload_kwargs_leave_dense_and_stream_free_modules_alone():
    dense = {**STREAM_KW, "low_cpu_mem_usage": True}
    assert mem._torchao_group_offload_kwargs(_Module(plain = True), dense) is dense
    assert mem._torchao_group_offload_kwargs(_Module("Int8Tensor"), NO_STREAM) is NO_STREAM


@pytest.mark.parametrize(
    "policy, transformer_quant, text_encoder_quant, no_grad",
    [
        (OFFLOAD_NONE, "int8", None, False),
        (OFFLOAD_NONE, "int8", "fp8", False),
        (OFFLOAD_MODEL, "int8", None, True),
        (OFFLOAD_GROUP, "int8", None, True),
        (OFFLOAD_GROUP, "fp8", None, True),
        (OFFLOAD_STREAMING, "int8", None, True),
        (OFFLOAD_GROUP, None, "fp8", True),
        (OFFLOAD_GROUP, None, None, False),
        (OFFLOAD_MODEL, None, None, False),
    ],
)
def test_render_grad_mode(policy, transformer_quant, text_encoder_quant, no_grad):
    state = types.SimpleNamespace(
        offload_policy = policy,
        transformer_quant = transformer_quant,
        text_encoder_quant = text_encoder_quant,
    )
    assert dmod._torchao_render_needs_no_grad(state) is no_grad


def test_render_uses_the_widened_switch():
    import inspect

    source = inspect.getsource(dmod.DiffusionBackend)
    assert "if _torchao_render_needs_no_grad(state)" in source
    assert "and state.offload_policy == OFFLOAD_MODEL" not in source


def _estimate(steady):
    return types.SimpleNamespace(steady_transformer_mib = steady)


@pytest.mark.parametrize(
    "placement, estimate_mib, largest_te_mib, version, policy, declined",
    [
        ("resident", 6_000, 4_000, T018, OFFLOAD_NONE, False),
        ("group_dit_streamed", 6_000, 4_000, T018, OFFLOAD_GROUP, False),
        ("group_dit_streamed", 6_000, 4_000, T016, OFFLOAD_GROUP, True),
        ("model_fits", 6_000, 4_000, T018, OFFLOAD_MODEL, False),
        # quantised denoiser or encoder too big to onload whole: stream instead of falling back to bf16
        ("model_fits", 60_000, 4_000, T018, OFFLOAD_STREAMING, False),
        ("model_fits", 6_000, 60_000, T018, OFFLOAD_STREAMING, False),
        ("model_fits", 60_000, 4_000, T016, OFFLOAD_MODEL, True),
        ("sequential", 6_000, 4_000, T018, OFFLOAD_SEQUENTIAL, True),
    ],
)
def test_inplace_quant_placement(
    monkeypatch, placement, estimate_mib, largest_te_mib, version, policy, declined
):
    monkeypatch.setattr(mem, "_installed_torchao_version", lambda: version)
    plan, why = dmod._inplace_torchao_placement(
        PLACEMENTS[placement], "int8", _estimate(estimate_mib), lambda: largest_te_mib
    )
    assert plan.offload_policy == policy
    assert (why is not None) is declined


@pytest.mark.parametrize("estimate_mib, largest_te_mib", [(60_000, 4_000), (6_000, 60_000)])
def test_inplace_quant_declines_streaming_it_cannot_pin(monkeypatch, estimate_mib, largest_te_mib):
    """Unpinnable (Windows, capped host RAM) streaming runs without a copy stream, ~35x slower: decline it."""
    monkeypatch.setattr(mem, "_installed_torchao_version", lambda: T018)
    monkeypatch.setattr(mem, "_pin_budget_mib", lambda: 0)
    plan, why = dmod._inplace_torchao_placement(
        PLACEMENTS["model_fits"], "int8", _estimate(estimate_mib), lambda: largest_te_mib
    )
    assert plan.offload_policy == OFFLOAD_MODEL
    assert why is not None


def test_inplace_quant_no_longer_calls_group_offload_wrong():
    import inspect
    source = inspect.getsource(dmod)
    assert "Group offload is WRONG for torchao" not in source


GIB = 1024


@pytest.fixture
def planner(monkeypatch):
    import torch

    from core.inference import diffusion_auto_policy as ap
    from core.inference import diffusion_transformer_quant as tq

    monkeypatch.setattr(mem, "_installed_torchao_version", lambda: T018)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: (12, 0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _supported(
        scheme,
        device = None,
        unproven_ok = False,
    ):
        return scheme in ("int8", "fp8")

    monkeypatch.setattr(tq, "_scheme_supported", _supported)
    monkeypatch.setattr(tq, "_is_consumer_gpu", lambda *_a, **_k: True)
    monkeypatch.setattr(ap, "_hf_cache_free_mib", lambda: 10**7)
    card: dict = {}
    for module in (dmod, mem):
        for fn in (
            "snapshot_device_memory",
            "settled_snapshot_device_memory",
            "reclaimable_snapshot_device_memory",
        ):
            if hasattr(module, fn):
                monkeypatch.setattr(module, fn, lambda *_a, **_k: card["memory"])
    target = types.SimpleNamespace(
        device = "cuda",
        dtype = torch.bfloat16,
        backend = "cuda",
        supports_model_cpu_offload = True,
        supports_default_torch_compile = True,
        ordinal = None,
        capability = (12, 0),
    )
    backend = dmod.DiffusionBackend.__new__(dmod.DiffusionBackend)
    backend._resolve_device_target = lambda *_a, **_k: target
    backend._target_for_ordinal = lambda *_a, **_k: target
    backend._cache_bytes = lambda *_a, **_k: 0
    backend._released_transformer_cached = lambda *_a, **_k: False

    def pick(family_name: str, base: str, gib: int):
        from core.inference.diffusion_families import _FAMILIES
        from core.inference.diffusion_transformer_quant import select_transformer_quant_scheme

        fam = next(f for f in _FAMILIES if f.name == family_name)
        card["memory"] = DeviceMemory("cuda", "cuda", "discrete_vram", gib * GIB - 600, gib * GIB)
        seed = backend._pipeline_planned_denoiser_scheme(
            fam,
            base = base,
            kind = "pipeline",
            transformer_quant = None,
            speed_mode = None,
            repo_id = base,
            fetch_base = base,
        )
        if seed not in (None, PIPELINE_SEED_DECLINED):
            seeded = backend._seeded_pipeline_plan(
                seed,
                target,
                base,
                fam,
                None,
                False,
                repo_id = base,
                base_local_dir = None,
                fetch_base = base,
            )
            placed = torchao_offload_plan(seeded, seed)
            assert placed is not None
            return f"hosted {seed}", placed.offload_policy
        plan = backend._bf16_table_plan(
            target, fam, base, None, False, kind = "pipeline", repo_id = base, fetch_base = base
        )
        if dmod._auto_quant_eager_reason(fam, plan, None, "pipeline") is not None:
            return "bf16", plan.offload_policy
        scheme = select_transformer_quant_scheme(target, "auto", family = fam.name, base_repo = base)
        estimate = ap.estimate_dense_quant(fam, scheme, base_repo = base)
        if plan.offload_policy != OFFLOAD_NONE:
            replanned = backend._plan_memory(
                target,
                None,
                base,
                fam,
                None,
                False,
                kind = "pipeline",
                repo_id = base,
                fetch_base = base,
                transformer_resident_override_mib = estimate.steady_transformer_mib,
                **backend._candidate_companion_overrides(estimate, fam, base, target, None),
            )
            placed = torchao_offload_plan(replanned, scheme)
            if placed is not None:
                plan = placed
        largest_te = int(ap.family_bf16_components_gb(fam, base)[1] * 1000**3 / 1024**2)
        plan, why = dmod._inplace_torchao_placement(plan, scheme, estimate, lambda: largest_te)
        return ("bf16" if why else f"on-the-fly {scheme}"), plan.offload_policy

    return pick


FAMILY_TABLE = {
    # Krea 2 seeds its hosted int8 denoiser like the generic pipeline families; same offload tiers as on the fly.
    ("krea-2", "krea/Krea-2-Turbo"): {
        8: ("hosted int8", OFFLOAD_STREAMING),
        12: ("hosted int8", OFFLOAD_GROUP),
        16: ("hosted int8", OFFLOAD_GROUP),
        24: ("hosted int8", OFFLOAD_GROUP),
        32: ("hosted int8", OFFLOAD_GROUP),
    },
    ("lumina-2", "Alpha-VLLM/Lumina-Image-2.0"): {
        8: ("hosted int8", OFFLOAD_STREAMING),
        12: ("hosted int8", OFFLOAD_STREAMING),
        16: ("hosted int8", OFFLOAD_GROUP),
        24: ("hosted int8", OFFLOAD_NONE),
        # measured: bf16 while it fits resident
        32: ("bf16", OFFLOAD_NONE),
    },
    ("hunyuanimage-2.1", "hunyuanvideo-community/HunyuanImage-2.1-Diffusers"): {
        8: ("hosted int8", OFFLOAD_STREAMING),
        12: ("hosted int8", OFFLOAD_STREAMING),
        16: ("hosted int8", OFFLOAD_GROUP),
        24: ("hosted int8", OFFLOAD_GROUP),
        32: ("hosted int8", OFFLOAD_GROUP),
    },
    ("hidream-i1", "HiDream-ai/HiDream-I1-Full"): {
        8: ("hosted int8", OFFLOAD_STREAMING),
        12: ("hosted int8", OFFLOAD_STREAMING),
        16: ("hosted int8", OFFLOAD_GROUP),
        24: ("hosted int8", OFFLOAD_GROUP),
        32: ("hosted int8", OFFLOAD_GROUP),
    },
    ("ideogram-4", "ideogram-ai/ideogram-4-fp8"): {
        8: ("on-the-fly int8", OFFLOAD_STREAMING),
        12: ("on-the-fly int8", OFFLOAD_STREAMING),
        16: ("on-the-fly int8", OFFLOAD_GROUP),
        24: ("on-the-fly int8", OFFLOAD_GROUP),
        32: ("on-the-fly int8", OFFLOAD_GROUP),
    },
}


@pytest.mark.parametrize(
    "family, base, gib, expected",
    [
        (family, base, gib, expected)
        for (family, base), by_gib in FAMILY_TABLE.items()
        for gib, expected in by_gib.items()
    ],
)
def test_auto_keeps_int8_on_the_offload_tier(planner, family, base, gib, expected):
    assert planner(family, base, gib) == expected


# 0.17 now streams int8 (pinnable Int8Tensor), so it takes the same hosted int8 as 0.18 instead of falling to fp8
@pytest.mark.parametrize("version, expected", [(T016, "bf16"), (T017, "hosted int8")])
def test_older_torchao_streamed_seeds(planner, monkeypatch, version, expected):
    monkeypatch.setattr(mem, "_installed_torchao_version", lambda: version)
    monkeypatch.setattr(mem, "_int8_tensor_pinnable", lambda: True, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_STREAM_TORCHAO17", raising = False)
    precision, _policy = planner(
        "hunyuanimage-2.1", "hunyuanvideo-community/HunyuanImage-2.1-Diffusers", 16
    )
    assert precision == expected


def _gpu_stack():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    pytest.importorskip("torchao")
    hooks = pytest.importorskip("diffusers.hooks")
    return torch, hooks


@pytest.mark.parametrize("scheme", ["int8", "fp8"])
@pytest.mark.parametrize("low_cpu_mem_usage", [False, True])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_quantised_blocks_survive_group_offload_under_no_grad(
    scheme, low_cpu_mem_usage, requires_grad
):
    import copy
    import inspect

    torch, hooks = _gpu_stack()
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Int8DynamicActivationInt8WeightConfig,
        PerRow,
        quantize_,
    )

    if scheme == "fp8" and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("fp8 needs sm_89+")
    torch.manual_seed(0)
    blocks = torch.nn.Sequential(
        *[
            torch.nn.Sequential(
                torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)
            )
            for _ in range(3)
        ]
    ).to(torch.bfloat16)
    config = (
        Int8DynamicActivationInt8WeightConfig
        if scheme == "int8"
        else lambda: Float8DynamicActivationFloat8WeightConfig(granularity = PerRow())
    )
    offloaded = copy.deepcopy(blocks)
    quantize_(offloaded, config())
    # Same CPU-quantised weights: fp8 quantised on CPU can round an ulp away from CUDA (torch 2.12).
    resident = copy.deepcopy(offloaded).cuda()
    for param in offloaded.parameters():
        param.requires_grad_(requires_grad)
    params = inspect.signature(hooks.apply_group_offloading).parameters
    kwargs = {
        "onload_device": torch.device("cuda"),
        "offload_device": torch.device("cpu"),
        "offload_type": "block_level",
        "num_blocks_per_group": 1,
        "use_stream": True,
    }
    for name, value in (
        ("non_blocking", True),
        ("record_stream", True),
        ("low_cpu_mem_usage", low_cpu_mem_usage),
    ):
        if name in params:
            kwargs[name] = value
    hooks.apply_group_offloading(offloaded, **mem._torchao_group_offload_kwargs(offloaded, kwargs))
    x = torch.randn(2, 16, 256, dtype = torch.bfloat16, device = "cuda")
    with torch.no_grad():
        want = resident(x)
        for _ in range(3):
            assert torch.equal(offloaded(x), want)
    if type(next(offloaded.parameters())).__name__ == "LinearActivationQuantizedTensor":
        return  # torchao <= 0.17 v1 int8 (pin-op shim): its onload is a plain re-wrap, which inference_mode allows
    with pytest.raises(Exception), torch.inference_mode():
        offloaded(x)


@pytest.mark.parametrize("offload_graphs", [True, False])
@pytest.mark.parametrize(
    "offload_active, denoiser_offloaded, expected",
    [(False, None, False), (True, None, True), (True, False, False), (True, True, True)],
)
def test_graph_gate_follows_the_denoiser(
    monkeypatch, offload_active, denoiser_offloaded, expected, offload_graphs
):
    """With offload graphs on, a moved denoiser is judged on everything else and armed after placement; with the
    kill switch, a moved denoiser is refused as before."""
    from core.inference import diffusion_cuda_graph as dcg
    from core.inference import diffusion_speed

    if offload_graphs:
        monkeypatch.delenv(dcg.OFFLOAD_CUDA_GRAPH_ENV, raising = False)
        expected = False
    else:
        monkeypatch.setenv(dcg.OFFLOAD_CUDA_GRAPH_ENV, "0")

    seen: list = []

    def _eligible(target, **kwargs):
        seen.append(kwargs["offload_active"])
        return False, "stub"

    monkeypatch.setattr(dcg, "graph_eligible", _eligible)
    pipe = types.SimpleNamespace(components = {})
    target = types.SimpleNamespace(device = "cpu", backend = "cpu", dtype = None)
    kwargs = {} if denoiser_offloaded is None else {"denoiser_offloaded": denoiser_offloaded}
    diffusion_speed.apply_speed_optims(
        pipe,
        target,
        is_gguf = False,
        family = types.SimpleNamespace(name = "z-image"),
        speed_mode = "default",
        offload_active = offload_active,
        **kwargs,
    )
    assert seen == [expected]


class _Hooked:
    def __init__(
        self,
        keys = (),
        hf_hook = None,
    ):
        self._diffusers_hook = types.SimpleNamespace(hooks = {k: object() for k in keys})
        self._hf_hook = hf_hook


@pytest.mark.parametrize(
    "transformer, hooked",
    [
        (_Hooked(), False),
        (_Hooked(("group_offloading", "lazy_prefetch_group_offloading")), True),
        (_Hooked(hf_hook = object()), True),
        (_Hooked(("first_block_cache",)), False),
    ],
)
def test_denoiser_hooked(transformer, hooked):
    assert dmod._denoiser_hooked(types.SimpleNamespace(transformer = transformer)) is hooked


def _int8_linear(rows = 1024, cols = 1024):
    torch = pytest.importorskip("torch")
    quant = pytest.importorskip("torchao.quantization")
    linear = torch.nn.Linear(cols, rows, bias = False).to(torch.bfloat16)
    try:
        quant.quantize_(linear, quant.Int8DynamicActivationInt8WeightConfig(version = 2))
    except Exception as exc:  # noqa: BLE001 - torchao without the v2 int8 tensor
        pytest.skip(f"int8 v2 quantisation unavailable: {exc}")
    if type(linear.weight).__name__ != "Int8Tensor":
        pytest.skip("this torchao does not produce Int8Tensor")
    return linear


def test_host_size_counts_packed_torchao_storage_not_the_logical_bf16_size():
    linear = _int8_linear(2048, 2048)
    # 2048 x 2048 int8 = 4 MiB of qdata plus per-row scales; the logical bf16 size would be 8 MiB.
    assert mem._module_host_mib(linear) < 8
    assert mem._module_host_mib(linear) >= 4


def test_stream_kept_when_the_packed_weights_fit_the_pin_budget(monkeypatch):
    linear = _int8_linear(2048, 2048)
    monkeypatch.setattr(mem, "_pinned_memory_capped", lambda: False)
    monkeypatch.setattr(mem, "install_group_offload_torchao_swap_retry", lambda: None)
    # Between the packed (~4 MiB) and logical bf16 (8 MiB) sizes.
    monkeypatch.setattr(mem, "_pin_budget_mib", lambda: 6)
    kwargs = mem._torchao_group_offload_kwargs(
        linear, {"use_stream": True, "low_cpu_mem_usage": True}
    )
    assert kwargs["use_stream"] is True
    assert kwargs["low_cpu_mem_usage"] is False


def test_storage_size_recurses_through_nested_wrappers():
    class Inner:
        def __init__(self, n):
            self._n = n

        def numel(self):
            return self._n

        def element_size(self):
            return 1

    class Wrapper:
        def __init__(self, *inner):
            self._inner = inner
            for i, t in enumerate(inner):
                setattr(self, f"t{i}", t)

        def __tensor_flatten__(self):
            return [f"t{i}" for i in range(len(self._inner))], None

        def numel(self):
            return 10**9

        def element_size(self):
            return 2

    nested = Wrapper(Wrapper(Inner(100), Inner(4)), Inner(7))
    assert mem._storage_nbytes(nested) == [100, 4, 7]


@pytest.mark.parametrize(
    "env, capped, budget, keeps_stream",
    [
        ("1", True, None, True),
        ("1", False, 0, True),
        ("0", False, 10**7, False),
        ("", False, 10**7, True),
    ],
)
def test_runtime_pinning_honours_the_same_override_as_the_planner(
    monkeypatch, env, capped, budget, keeps_stream
):
    linear = _int8_linear(256, 256)
    monkeypatch.setenv(mem.GROUP_OFFLOAD_PIN_ENV, env)
    monkeypatch.setattr(mem, "_pinned_memory_capped", lambda: capped)
    monkeypatch.setattr(mem, "_pin_budget_mib", lambda: budget)
    monkeypatch.setattr(mem, "install_group_offload_torchao_swap_retry", lambda: None)
    kwargs = mem._torchao_group_offload_kwargs(
        linear, {"use_stream": True, "low_cpu_mem_usage": True}
    )
    assert kwargs["use_stream"] is keeps_stream


def test_several_torchao_denoisers_share_one_pin_budget(monkeypatch):
    """Ideogram 4 streams two torchao transformers: each fits the budget alone, together they do not."""
    first, second = _int8_linear(2048, 2048), _int8_linear(2048, 2048)
    monkeypatch.delenv(mem.GROUP_OFFLOAD_PIN_ENV, raising = False)
    monkeypatch.setattr(mem, "_pinned_memory_capped", lambda: False)
    monkeypatch.setattr(mem, "install_group_offload_torchao_swap_retry", lambda: None)
    monkeypatch.setattr(mem, "_pin_budget_mib", lambda: 6)
    base = {"use_stream": True, "low_cpu_mem_usage": True}
    pinned = [0]
    kept = [mem._torchao_group_offload_kwargs(m, dict(base), pinned) for m in (first, second)]
    assert kept[0]["low_cpu_mem_usage"] is False
    assert kept[1].get("low_cpu_mem_usage") is not False
    assert pinned[0] <= 6


def test_every_group_offload_site_shares_the_pin_total():
    import inspect

    import re

    source = inspect.getsource(mem)
    calls = re.findall(r"(?<!def )_torchao_group_offload_kwargs\(([^)]*)\)", source)
    assert calls and all("pinned_mib" in args for args in calls)


@pytest.mark.parametrize("fits_whole, raises", [(False, True), (True, False)])
def test_failed_group_setup_never_onloads_an_oversized_quantised_transformer(
    monkeypatch, fits_whole, raises
):
    """Streaming was the only placement the quantised transformer fits; model offload would OOM on first onload."""
    calls = []
    pipe = types.SimpleNamespace(
        enable_model_cpu_offload = lambda device = None: calls.append("model_offload")
    )
    monkeypatch.setattr(mem, "_apply_group_offload", lambda *a, **k: False)
    monkeypatch.setattr(mem, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    monkeypatch.setattr(mem, "_model_offload_fits_quantised", lambda plan: fits_whole)
    monkeypatch.setattr(mem, "keep_cpu_weights_on_offload", lambda *a, **k: 0)
    monkeypatch.setattr(mem, "_enable_vae_saver", lambda *a, **k: False)
    plan = _plan(OFFLOAD_GROUP)
    if raises:
        with pytest.raises(RuntimeError, match = "does not fit the GPU whole"):
            mem.apply_memory_plan(pipe, plan, device = "cuda")
        assert "model_offload" not in calls
    else:
        effective, _ = mem.apply_memory_plan(pipe, plan, device = "cuda")
        assert effective == OFFLOAD_MODEL


def test_pipeline_host_mib_counts_only_host_resident_weights():
    torch = pytest.importorskip("torch")
    cpu = torch.nn.Linear(1024, 1024, bias = False)  # 4 MiB fp32
    pipe = types.SimpleNamespace(components = {"transformer": cpu, "scheduler": object()})
    assert mem.pipeline_host_mib(pipe) == 4
    assert mem.pipeline_host_mib(None) == 0
    if torch.cuda.is_available():
        pipe.components["transformer"] = cpu.to("cuda")
        assert mem.pipeline_host_mib(pipe) == 0


def test_reclaimable_host_ram_widens_the_pin_budget(monkeypatch):
    monkeypatch.setattr(mem, "_pin_budget_mib", lambda: 4_000)
    plan = types.SimpleNamespace(
        estimates = {"model_dense_mib": 20_000, "companion_dense_mib": 8_000}
    )
    assert not mem._torchao_stream_pinnable(plan)
    assert mem._torchao_stream_pinnable(plan, 8_000)
