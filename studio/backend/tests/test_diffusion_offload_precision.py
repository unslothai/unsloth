# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A torchao int8 / fp8 denoiser stays quantised when the image plan offloads it.

Which placement each weight class survives (``torchao_offload_plan``), the group-offload kwargs a torchao module is
given, the render's no_grad switch, and the auto precision the real planner picks per family at 8-32 GiB."""

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
    """Host RAM that pins any streamed denoiser, whatever the machine running the tests has."""
    monkeypatch.setattr(mem, "_pin_budget_mib", lambda: 10 ** 7)
    monkeypatch.setattr(mem, "_pinned_memory_capped", lambda: False)
    monkeypatch.delenv(mem.GROUP_OFFLOAD_PIN_ENV, raising = False)


def _plan(policy, *, stream_transformer = True, fits_model_offload = True):
    """A real MemoryPlan with the requested placement; ``fits_model_offload`` sizes the whole-module check."""
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
    "resident": {s: {v: (True, OFFLOAD_NONE) for v in (T016, T017, T018)} for s in ("int8", "fp8", "nvfp4")},
    "group_dit_resident": {
        s: {v: (True, OFFLOAD_GROUP) for v in (T016, T017, T018)} for s in ("int8", "fp8", "nvfp4")
    },
    "group_dit_streamed": {
        # 0.17's v1 int8 survives only stream-free, measured 14x slower than streaming bf16
        "int8": {T016: (False, None), T017: (False, None), T018: (True, OFFLOAD_GROUP)},
        "fp8": {T016: (False, None), T017: (True, OFFLOAD_GROUP), T018: (True, OFFLOAD_GROUP)},
        "nvfp4": {T016: (False, None), T017: (False, None), T018: (False, None)},
    },
    "model_fits": {s: {v: (True, OFFLOAD_MODEL) for v in (T016, T017, T018)} for s in ("int8", "fp8", "nvfp4")},
    "model_too_big": {
        "int8": {T016: (False, None), T017: (False, None), T018: (True, OFFLOAD_STREAMING)},
        "fp8": {T016: (False, None), T017: (True, OFFLOAD_STREAMING), T018: (True, OFFLOAD_STREAMING)},
        "nvfp4": {T016: (False, None), T017: (False, None), T018: (False, None)},
    },
    "streaming": {
        "int8": {T016: (False, None), T017: (False, None), T018: (True, OFFLOAD_STREAMING)},
        "fp8": {T016: (False, None), T017: (True, OFFLOAD_STREAMING), T018: (True, OFFLOAD_STREAMING)},
        "nvfp4": {T016: (False, None), T017: (False, None), T018: (False, None)},
    },
    "sequential": {s: {v: (False, None) for v in (T016, T017, T018)} for s in ("int8", "fp8", "nvfp4")},
}
CASES = [
    (placement, scheme, version, *expected)
    for placement, by_scheme in SURVIVAL.items()
    for scheme, by_version in by_scheme.items()
    for version, expected in by_version.items()
]


@pytest.mark.parametrize("placement, scheme, version, survives, policy", CASES)
def test_survival_table(placement, scheme, version, survives, policy):
    plan = PLACEMENTS[placement]
    assert torchao_survives_plan(plan, scheme, torchao_version = version) is survives
    placed = torchao_offload_plan(plan, scheme, torchao_version = version)
    assert (placed.offload_policy if placed is not None else None) == policy


@pytest.mark.parametrize("diffusers_version, streams", [((0, 36), False), ((0, 37), False), ((0, 38), True), (None, False)])
def test_diffusers_before_the_torchao_swap_keeps_the_resident_rule(monkeypatch, diffusers_version, streams):
    """diffusers < 0.38 moved only a torchao weight's wrapper under group offload."""
    monkeypatch.setattr(mem, "_installed_diffusers_version", lambda: diffusers_version)
    assert torchao_survives_plan(PLACEMENTS["group_dit_streamed"], "int8", torchao_version = T018) is streams
    assert torchao_survives_plan(PLACEMENTS["group_dit_resident"], "int8", torchao_version = T018)


def test_swap_retry_collects_once_then_gives_up(monkeypatch):
    """A weakref left by uncollected garbage is retried after gc; any other failure, or a live weakref, raises."""
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
        (False, 10 ** 7, None, True),
        # Windows / WSL cap pinned memory near 1 GiB
        (True, 10 ** 7, None, False),
        # the quantised denoiser (8000 MiB here) does not fit the pinnable host RAM
        (False, 7_999, None, False),
        (False, None, None, False),
        (False, 10 ** 7, "0", False),
        (True, 0, "1", True),
    ],
)
def test_a_streamed_torchao_denoiser_needs_its_pin(monkeypatch, capped, budget, env, streams):
    """Lazy pinning refuses torchao and the stream-free fallback copies back every step, so a streamed torchao tier
    needs the up-front pin; a resident one does not."""
    monkeypatch.setattr(mem, "_pinned_memory_capped", lambda: capped)
    monkeypatch.setattr(mem, "_pin_budget_mib", lambda: budget)
    if env is not None:
        monkeypatch.setenv(mem.GROUP_OFFLOAD_PIN_ENV, env)
    for name in ("group_dit_streamed", "streaming", "model_too_big"):
        assert torchao_survives_plan(PLACEMENTS[name], "int8", torchao_version = T018) is streams, name
    assert torchao_survives_plan(PLACEMENTS["group_dit_resident"], "int8", torchao_version = T018)
    assert torchao_survives_plan(PLACEMENTS["model_fits"], "int8", torchao_version = T018)


def test_no_torchao_keeps_the_resident_rule():
    assert not torchao_survives_plan(PLACEMENTS["group_dit_streamed"], "int8", torchao_version = None)
    assert torchao_survives_plan(PLACEMENTS["group_dit_resident"], "int8", torchao_version = None)


def test_never_moves_still_means_resident_only():
    """The old predicate is unchanged for callers that mean "the denoiser never moves" (the GGUF gates)."""
    keeps = mem.plan_keeps_transformer_resident
    assert keeps(PLACEMENTS["resident"]) and keeps(PLACEMENTS["group_dit_resident"])
    for name in ("group_dit_streamed", "model_fits", "streaming"):
        assert not keeps(PLACEMENTS[name]), name


class _Weight:
    def __init__(self, cls_name: str, module: str = "torchao.quantization.fake"):
        self.data = None
        self.__class__ = type(cls_name, (_Weight,), {"__module__": module})


class _Module:
    def __init__(self, *class_names: str, plain: bool = False):
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
        (("Int8Tensor",), False, 10 ** 7, "as is"),
        (("Float8Tensor",), False, 10 ** 7, "as is"),
        (("Int8Tensor", "Float8Tensor"), False, 10 ** 7, "as is"),
        # lazy pinning refuses every torchao subclass: pin up front where host RAM allows
        (("Int8Tensor",), True, 10 ** 7, "pinned"),
        (("Float8Tensor",), True, 10 ** 7, "pinned"),
        (("Int8Tensor",), True, None, "no stream"),
        # torchao 0.17 default int8: no aten.is_pinned, so no copy stream
        (("LinearActivationQuantizedTensor",), False, 10 ** 7, "no stream"),
    ],
)
def test_group_offload_kwargs_per_weight_class(monkeypatch, classes, low_cpu_mem_usage, pin_budget, expected):
    monkeypatch.setattr(mem, "_pin_budget_mib", lambda: pin_budget)
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
        # GGUF or dense bf16 under offload keeps inference_mode
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
    """The render context asks the helper, not the model-offload-only condition #11558 shipped."""
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
        # whole module fits: keep model offload
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


def test_inplace_quant_no_longer_calls_group_offload_wrong():
    import inspect

    source = inspect.getsource(dmod)
    assert "Group offload is WRONG for torchao" not in source


# --- auto precision per family, real planner, spoofed card -------------------------------------------------------

GIB = 1024


@pytest.fixture
def planner(monkeypatch):
    """The real image planner on a spoofed sm_120 card with nothing cached (a fresh install)."""
    import torch

    from core.inference import diffusion_auto_policy as ap
    from core.inference import diffusion_transformer_quant as tq

    monkeypatch.setattr(mem, "_installed_torchao_version", lambda: T018)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: (12, 0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _supported(scheme, device = None, unproven_ok = False):
        return scheme in ("int8", "fp8")

    monkeypatch.setattr(tq, "_scheme_supported", _supported)
    monkeypatch.setattr(tq, "_is_consumer_gpu", lambda *_a, **_k: True)
    monkeypatch.setattr(ap, "_hf_cache_free_mib", lambda: 10 ** 7)
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
                seed, target, base, fam, None, False, repo_id = base, base_local_dir = None, fetch_base = base
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
        largest_te = int(ap.family_bf16_components_gb(fam, base)[1] * 1000 ** 3 / 1024 ** 2)
        plan, why = dmod._inplace_torchao_placement(plan, scheme, estimate, lambda: largest_te)
        return ("bf16" if why else f"on-the-fly {scheme}"), plan.offload_policy

    return pick


FAMILY_TABLE = {
    ("krea-2", "krea/Krea-2-Turbo"): {
        8: ("on-the-fly int8", OFFLOAD_STREAMING),
        12: ("on-the-fly int8", OFFLOAD_GROUP),
        16: ("on-the-fly int8", OFFLOAD_GROUP),
        24: ("on-the-fly int8", OFFLOAD_GROUP),
        32: ("on-the-fly int8", OFFLOAD_GROUP),
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
    """Every cell the audit found falling back to released bf16 because the tier offloads now keeps int8 with the
    offload tier; only Lumina's measured keep-bf16-while-resident cell stays bf16."""
    assert planner(family, base, gib) == expected


@pytest.mark.parametrize("version, expected", [(T016, "bf16"), (T017, "hosted fp8")])
def test_older_torchao_streamed_seeds(planner, monkeypatch, version, expected):
    """0.16 has no measured streaming path and keeps the released weights, as before. On 0.17 the int8 rung (v1,
    stream-free only) yields to the hosted fp8 one, which streams."""
    monkeypatch.setattr(mem, "_installed_torchao_version", lambda: version)
    precision, _policy = planner("hunyuanimage-2.1", "hunyuanvideo-community/HunyuanImage-2.1-Diffusers", 16)
    assert precision == expected


# --- real torchao weights through diffusers group offload (GPU) ---------------------------------------------------


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
def test_quantised_blocks_survive_group_offload_under_no_grad(scheme, low_cpu_mem_usage, requires_grad):
    """Studio's stream kwargs, passed through ``_torchao_group_offload_kwargs``, run a quantised block stack bit-exact
    vs resident under no_grad; inference_mode is what the render switch avoids. ``requires_grad`` is how a hosted
    checkpoint's weights load (``quantize_`` leaves them frozen), which diffusers' swap_tensors cannot move."""
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
        *[torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)) for _ in range(3)]
    ).to(torch.bfloat16)
    config = (
        Int8DynamicActivationInt8WeightConfig
        if scheme == "int8"
        else lambda: Float8DynamicActivationFloat8WeightConfig(granularity = PerRow())
    )
    resident = copy.deepcopy(blocks).cuda()
    quantize_(resident, config())
    offloaded = copy.deepcopy(blocks)
    quantize_(offloaded, config())
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
    for name, value in (("non_blocking", True), ("record_stream", True), ("low_cpu_mem_usage", low_cpu_mem_usage)):
        if name in params:
            kwargs[name] = value
    hooks.apply_group_offloading(offloaded, **mem._torchao_group_offload_kwargs(offloaded, kwargs))
    x = torch.randn(2, 16, 256, dtype = torch.bfloat16, device = "cuda")
    with torch.no_grad():
        want = resident(x)
        for _ in range(3):
            assert torch.equal(offloaded(x), want)
    with pytest.raises(Exception), torch.inference_mode():
        offloaded(x)


# --- CUDA graphs on the resident-denoiser tier ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "offload_active, denoiser_offloaded, expected",
    [(False, None, False), (True, None, True), (True, False, False), (True, True, True)],
)
def test_graph_gate_follows_the_denoiser(monkeypatch, offload_active, denoiser_offloaded, expected):
    """Graphs are refused for a plan that moves the denoiser, not for one that only streams the text encoders."""
    from core.inference import diffusion_cuda_graph as dcg
    from core.inference import diffusion_speed

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
    def __init__(self, keys = (), hf_hook = None):
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
