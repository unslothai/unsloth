# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Multi-GPU diffusion layer (core/inference/diffusion_multigpu.py), CPU only.

Cross-card behaviour is exercised with fake devices: the planner reads budgets through injected callables, placement
and the boundary through a patched card predicate, and CFG-parallel with a replica on the same (CPU) device, which
checks the routing, the prefetch / constant caches and every fallback without a second card."""

from __future__ import annotations

import contextlib
import dataclasses
import threading
from collections import OrderedDict

import pytest
import torch

from core.inference import diffusion_memory as dm
from core.inference import diffusion_multigpu as mg


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(mg.MULTIGPU_ENV, raising = False)
    monkeypatch.delenv(mg.CFG_PARALLEL_ENV, raising = False)
    yield


# ---------------------------------------------------------------------------------------------------- selection


def test_one_card_or_the_kill_switch_means_no_selection(monkeypatch):
    assert mg.make_selection(0, [0]) is None
    assert mg.make_selection(None, [0, 1]) is None
    assert mg.make_selection(1, [0, 1]) == mg.MultiGpuSelection(primary = 1, secondaries = (0,))
    monkeypatch.setenv(mg.MULTIGPU_ENV, "0")
    assert mg.make_selection(1, [0, 1]) is None
    assert mg.cfg_parallel_mode() == "off"


def test_selection_is_scoped_to_the_load_thread():
    sel = mg.make_selection(0, [0, 1])
    seen = {}
    with mg.selection_scope(sel):
        assert mg.current_selection() == sel
        t = threading.Thread(target = lambda: seen.setdefault("other", mg.current_selection()))
        t.start()
        t.join()
    assert mg.current_selection() is None
    # A plain thread does not inherit the load's selection: a route preflight plans for one card.
    assert seen["other"] is None


def test_resolve_selected_cuda_ordinals_translates_the_mask(monkeypatch):
    import utils.hardware.hardware as hw
    from core.inference import diffusion_device as dd

    # CUDA_VISIBLE_DEVICES=4,5 on an 8-card box: physical 4 / 5 are torch 0 / 1.
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": [4, 5], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(hw, "get_physical_gpu_count", lambda: 8)
    assert dd.resolve_selected_cuda_ordinals([5, 4]) == [0, 1]
    assert dd.resolve_selected_cuda_ordinals([]) == []
    with pytest.raises(ValueError):
        dd.resolve_selected_cuda_ordinals([0])


# ---------------------------------------------------------------------------------------------------- planner


def _split(**overrides):
    kwargs = dict(
        single_card_policy = "group",
        model_dense_mib = 20000,
        companion_dense_mib = 9000,
        runtime_headroom_mib = 2000,
        base_overhead_mib = 1000,
        primary_limit_mib = 14000,
        secondary_limit_mib = lambda o: 12750,
        selection = mg.MultiGpuSelection(primary = 0, secondaries = (1,)),
    )
    kwargs.update(overrides)
    return mg.plan_companion_split(**kwargs)


def test_split_when_the_denoiser_alone_fits_the_primary():
    split = _split()
    assert split is not None and split.secondary_ordinal == 1
    assert split.primary_required_mib == 11000 + 2000 + 1000
    assert split.secondary_required_mib == 9000 + 3000


@pytest.mark.parametrize(
    "overrides",
    [
        dict(primary_limit_mib = 13000),  # denoiser + headroom do not fit the primary either
        dict(secondary_limit_mib = lambda o: 11999),  # companions + headroom = 12000
        dict(secondary_limit_mib = lambda o: None),  # unreadable secondary
        dict(single_card_policy = "none"),  # one card already holds everything
        dict(companion_dense_mib = None),  # unknown split
        dict(selection = None),
    ],
)
def test_no_split_keeps_single_card_behaviour(overrides):
    assert _split(**overrides) is None


def test_text_encoders_alone_when_the_decode_needs_the_runtime_room():
    # Video-like: the secondary cannot also hold the decode headroom, but the encoders alone fit it, and the denoiser +
    # VAE fit the primary.
    kwargs = dict(
        companion_dense_mib = 11500,
        text_encoder_dense_mib = 11000,
        model_dense_mib = 16000,
        runtime_headroom_mib = 4000,
        primary_limit_mib = 12000,
        secondary_limit_mib = lambda o: 13600,
    )
    split = _split(**kwargs)
    assert split is not None and split.components == mg.TEXT_ENCODERS_ONLY
    assert split.secondary_required_mib == 11000 + mg.TEXT_ENCODER_HEADROOM_MIB + 1000
    assert split.primary_required_mib == 5000 + 4000 + 1000
    # without the encoder share the planner cannot price it and keeps one card
    assert _split(**{**kwargs, "text_encoder_dense_mib": None}) is None


def test_measured_phases_price_each_card_for_what_it_runs():
    # klein-like on two 15 GB cards: the flat 8 GiB estimate refuses, the measured phases fit.
    kwargs = dict(
        model_dense_mib = 7630 + 7500 + 160,
        companion_dense_mib = 7660,
        text_encoder_dense_mib = 7500,
        runtime_headroom_mib = 8192,
        base_overhead_mib = 2048,
        primary_limit_mib = 14400,
        secondary_limit_mib = lambda o: 12495,
    )
    assert _split(**kwargs) is None
    split = _split(**kwargs, phase_headroom_mib = (1516, 1160, 2645))
    assert split is not None and split.components is None
    assert split.primary_required_mib == 7630 + 1160 + 2048
    assert split.secondary_required_mib == 7660 + 2645 + 2048


def test_force_splits_even_when_one_card_fits(monkeypatch):
    monkeypatch.setenv(mg.MULTIGPU_ENV, "force")
    assert _split(single_card_policy = "none") is not None


def test_split_picks_the_secondary_with_the_most_room():
    split = _split(
        selection = mg.MultiGpuSelection(primary = 0, secondaries = (1, 2)),
        secondary_limit_mib = lambda o: {1: 12750, 2: 34000}[o],
    )
    assert split.secondary_ordinal == 2


@dataclasses.dataclass
class _Target:
    supports_model_cpu_offload: bool = True
    ordinal: int = 0
    device: str = "cuda"
    backend: str = "cuda"


def _memory(free = 24000, total = 24576):
    return dm.DeviceMemory(
        backend = "cuda", device = "cuda", memory_kind = "discrete_vram", free_mib = free, total_mib = total
    )


def _plan(**kw):
    args = dict(
        target = _Target(),
        device_memory = _memory(),
        model_dense_mib = 20000,
        companion_dense_mib = 9000,
        text_encoder_dense_mib = 8500,
        runtime_headroom_mib = 2000,
    )
    args.update(kw)
    return dm.plan_diffusion_memory(**args)


def test_planner_unchanged_without_a_selection():
    plan = _plan()
    assert plan.offload_policy != dm.OFFLOAD_NONE
    assert plan.secondary_ordinal is None


def test_planner_splits_under_a_selection(monkeypatch):
    monkeypatch.setattr(mg, "secondary_resident_limit_mib", lambda o, mode: 13600)
    single = _plan()
    with mg.selection_scope(mg.make_selection(0, [0, 1])):
        plan = _plan()
        # explicit offload modes keep their meaning
        balanced = _plan(requested_mode = "balanced")
        legacy = _plan(explicit_offload = True)
        # the target must be the primary of the selection
        other = _plan(target = _Target(ordinal = 1))
    assert plan.offload_policy == dm.OFFLOAD_NONE and plan.secondary_ordinal == 1
    assert plan.vae_tiling is False and plan.vae_slicing is False
    assert plan.as_public_dict()["secondary_ordinal"] == 1
    assert "multi-GPU" in plan.reasons[-1]
    for p in (balanced, legacy, other):
        assert p.secondary_ordinal is None
    assert balanced.offload_policy == dm.OFFLOAD_GROUP
    assert other.offload_policy == single.offload_policy


def test_planner_split_declined_when_the_secondary_is_small(monkeypatch):
    monkeypatch.setattr(mg, "secondary_resident_limit_mib", lambda o, mode: 3400)
    single = _plan()
    with mg.selection_scope(mg.make_selection(0, [0, 1])):
        plan = _plan()
    assert plan == single


def test_planner_split_off_on_unified_memory(monkeypatch):
    monkeypatch.setattr(mg, "secondary_resident_limit_mib", lambda o, mode: 13600)
    unified = dm.DeviceMemory(
        backend = "cuda", device = "cuda", memory_kind = "unified_memory", free_mib = 16000, total_mib = 16384
    )
    with mg.selection_scope(mg.make_selection(0, [0, 1])):
        plan = _plan(device_memory = unified)
    assert plan.secondary_ordinal is None


# ---------------------------------------------------------------------------------------------------- assignment


def test_assignment_prefers_moving_the_companions():
    sizes = {
        "transformer": (10000, "dit"),
        "text_encoder": (8000, "text_encoder"),
        "vae": (300, "other"),
    }
    moved = mg.choose_component_assignment(
        sizes,
        headroom_mib = 2000,
        overhead_mib = 1000,
        primary_limit_mib = 13500,
        secondary_limit_mib = 13000,
    )
    assert moved == ("text_encoder", "vae")


def test_assignment_can_move_only_the_text_encoders():
    sizes = {
        "transformer": (9000, "dit"),
        "text_encoder": (11000, "text_encoder"),
        "vae": (1400, "other"),
    }
    moved = mg.choose_component_assignment(
        sizes,
        headroom_mib = 4000,
        overhead_mib = 1000,
        primary_limit_mib = 16000,
        secondary_limit_mib = 13500,
    )
    assert moved == ("text_encoder",)


def test_place_split_resolves_text_encoders_only(monkeypatch):
    class Pipe:
        def __init__(self):
            self.transformer = torch.nn.Linear(2, 2)
            self.text_encoder = torch.nn.Linear(2, 2)
            self.text_encoder_2 = torch.nn.Linear(2, 2)
            self.vae = _Vae()

        @property
        def components(self):
            return dict(vars(self))

    monkeypatch.setattr(mg, "install_execution_device_patch", lambda: True)
    pipe = Pipe()
    real_device = torch.device
    monkeypatch.setattr(
        torch, "device", lambda *a: real_device("cpu") if a and a[0] == "cuda" else real_device(*a)
    )
    try:
        mg.place_split(pipe, primary = "cpu", secondary_ordinal = 1, names = mg.TEXT_ENCODERS_ONLY)
    finally:
        monkeypatch.setattr(torch, "device", real_device)
    assert mg.placement_status(pipe)["moved"] == ["text_encoder", "text_encoder_2"]
    assert "decode" not in pipe.vae.__dict__


def test_assignment_puts_the_second_expert_on_the_secondary():
    sizes = {
        "transformer": (12000, "dit"),
        "transformer_2": (12000, "dit"),
        "text_encoder": (5000, "text_encoder"),
        "vae": (500, "other"),
    }
    moved = mg.choose_component_assignment(
        sizes,
        headroom_mib = 2000,
        overhead_mib = 1000,
        primary_limit_mib = 21000,
        secondary_limit_mib = 16000,
    )
    assert moved == ("transformer_2",)
    # tighter primary: the second expert takes the companions along
    moved = mg.choose_component_assignment(
        sizes,
        headroom_mib = 2000,
        overhead_mib = 1000,
        primary_limit_mib = 15500,
        secondary_limit_mib = 21000,
    )
    assert moved == ("transformer_2", "text_encoder", "vae")
    assert (
        mg.choose_component_assignment(
            sizes,
            headroom_mib = 2000,
            overhead_mib = 1000,
            primary_limit_mib = 9000,
            secondary_limit_mib = 9000,
        )
        is None
    )


def test_refine_plan_for_multigpu(monkeypatch):
    plan = _plan()
    assert plan.offload_policy != dm.OFFLOAD_NONE
    sizes = {
        "transformer": (9000, "dit"),
        "text_encoder": (8000, "text_encoder"),
        "vae": (300, "other"),
    }
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: sizes)
    monkeypatch.setattr(mg, "secondary_resident_limit_mib", lambda o, mode: 12750)
    monkeypatch.setattr(mg, "best_secondary", lambda selection, probes = None: 1)
    assert mg.refine_plan_for_multigpu(object(), plan) is plan  # no selection
    with mg.selection_scope(mg.make_selection(0, [0, 1])):
        refined = mg.refine_plan_for_multigpu(object(), plan)
    assert refined.offload_policy == dm.OFFLOAD_NONE
    assert refined.secondary_ordinal == 1
    assert refined.secondary_components == ("text_encoder", "vae")


# ---------------------------------------------------------------------------------------------------- tensors across


class _Holder:
    def __init__(self, t):
        self.parameters = t
        self.mean = t * 2
        self.deterministic = False


@dataclasses.dataclass
class _DC:
    sample: torch.Tensor
    note: str = "x"


def test_move_to_device_keeps_container_types():
    t = torch.ones(2)
    Named = __import__("collections").namedtuple("Named", "a b")
    obj = {
        "tuple": (t, 1),
        "list": [t],
        "named": Named(t, "s"),
        "ordered": OrderedDict(x = t),
        "dc": _DC(t),
        "holder": _Holder(t),
        "plain": "keep",
    }
    out = mg.move_to_device(obj, torch.device("meta"))
    assert out["tuple"][0].device.type == "meta" and out["tuple"][1] == 1
    assert out["list"][0].device.type == "meta"
    assert (
        type(out["named"]) is Named
        and out["named"].a.device.type == "meta"
        and out["named"].b == "s"
    )
    assert isinstance(out["ordered"], OrderedDict) and out["ordered"]["x"].device.type == "meta"
    assert isinstance(out["dc"], _DC) and out["dc"].sample.device.type == "meta"
    assert isinstance(out["holder"], _Holder) and out["holder"].mean.device.type == "meta"
    assert out["holder"].deterministic is False and obj["holder"].parameters.device.type == "cpu"
    assert out["plain"] == "keep"
    # Already there: the same object comes back (no copy).
    assert mg.move_to_device(t, torch.device("cpu")) is t


def test_model_output_moves_with_its_attributes():
    from diffusers.utils import BaseOutput

    @dataclasses.dataclass
    class Out(BaseOutput):
        sample: torch.Tensor = None

    out = mg.move_to_device(Out(sample = torch.ones(1)), torch.device("meta"))
    assert out.sample.device.type == "meta" and out[0].device.type == "meta"


# ---------------------------------------------------------------------------------------------------- boundary


class _Vae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        self.seen = []

    def forward(self, x):
        self.seen.append(("forward", x.device))
        return self.lin(x)

    def decode(
        self,
        z,
        return_dict = False,
    ):
        self.seen.append(("decode", z.device))
        return (self.lin(z),)


def test_boundary_moves_inputs_in_and_outputs_home(monkeypatch):
    vae = _Vae()
    home = torch.device("meta")  # stands for the primary card
    moves = []

    def fake_move(
        obj,
        device,
        _depth = 0,
    ):
        moves.append(device)
        return obj

    # the module "sits on another card": every call crosses
    monkeypatch.setattr(mg, "_crosses", lambda device, h: True)
    monkeypatch.setattr(mg, "move_to_device", fake_move)
    wrapped = mg.install_boundary(vae, home)
    assert set(wrapped) == {"forward", "decode"}
    vae.decode(torch.ones(1, 2))
    # inputs to the module's own card, the result back home
    assert moves == [torch.device("cpu"), home]
    assert vae.seen[-1] == ("decode", torch.device("cpu"))
    vae(torch.ones(1, 2))
    assert moves[2:] == [torch.device("cpu"), home]
    # idempotent, and removable
    assert mg.install_boundary(vae, home) == []
    mg.remove_boundary(vae)
    assert "decode" not in vae.__dict__ and "forward" not in vae.__dict__


def test_boundary_is_inert_on_the_same_card():
    vae = _Vae()
    mg.install_boundary(vae, torch.device("cpu"))
    x = torch.ones(1, 2)
    assert torch.equal(vae(x), vae.lin(x))


def test_execution_device_reports_the_primary_of_a_split_pipeline():
    from diffusers import DiffusionPipeline

    class Pipe(DiffusionPipeline):
        def __init__(self, vae, transformer):
            super().__init__()
            self.register_modules(vae = vae, transformer = transformer)

    class Mod(torch.nn.Linear):
        @property
        def device(self):
            return self.weight.device

    vae = Mod(1, 1).to("meta")  # stands for the secondary card
    transformer = Mod(1, 1)
    pipe = Pipe(vae, transformer)
    assert mg.install_execution_device_patch()
    before = pipe.device
    vae._unsloth_split_home = torch.device("meta")
    assert pipe.device == torch.device("meta")
    del vae._unsloth_split_home
    transformer._unsloth_split_home = torch.device("cpu")
    assert pipe._execution_device == torch.device("cpu")
    assert pipe.device == torch.device("cpu")
    del transformer._unsloth_split_home
    assert pipe.device == before


def test_place_split_and_fallback(monkeypatch):
    class Pipe:
        def __init__(self):
            self.transformer = torch.nn.Linear(2, 2)
            self.text_encoder = torch.nn.Linear(2, 2)
            self.vae = _Vae()
            self.scheduler = object()

        @property
        def components(self):
            return {
                "transformer": self.transformer,
                "text_encoder": self.text_encoder,
                "vae": self.vae,
                "scheduler": self.scheduler,
            }

        def to(self, device):
            for m in (self.transformer, self.text_encoder, self.vae):
                m.to(device)

    pipe = Pipe()
    assert mg.companion_names(pipe) == ("text_encoder", "vae")
    moved_to = {}
    real_device = torch.device

    def fake_device(*args):
        # "cuda", 1 -> the CPU in this test; the placement records the requested names
        if args and args[0] == "cuda":
            return real_device("cpu")
        return real_device(*args)

    monkeypatch.setattr(mg, "install_execution_device_patch", lambda: True)
    monkeypatch.setattr(torch, "device", fake_device)
    try:
        placement = mg.place_split(pipe, primary = "cpu", secondary_ordinal = 1)
    finally:
        monkeypatch.setattr(torch, "device", real_device)
    assert set(placement) == {"transformer", "text_encoder", "vae"}
    assert pipe.transformer._unsloth_split_home == torch.device("cpu")
    assert "decode" in pipe.vae.__dict__
    status = mg.placement_status(pipe)
    assert status["moved"] == ["text_encoder", "vae"]
    mg.undo_split(pipe)
    assert (
        "decode" not in pipe.vae.__dict__ and "_unsloth_split_home" not in pipe.transformer.__dict__
    )
    assert mg.placement_status(pipe) is None
    del moved_to

    # A failing placement undoes itself and reports False, so apply_memory_plan offloads on one card.
    @dataclasses.dataclass
    class P:
        secondary_ordinal: int = 1
        secondary_components: tuple = None

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(mg, "place_split", boom)
    assert mg.apply_split_plan(pipe, P(), placement = "cpu") is False


def test_apply_memory_plan_falls_back_to_model_offload(monkeypatch):
    calls = []

    class Pipe:
        components: dict = {}

        def enable_model_cpu_offload(self, device = None):
            calls.append(("offload", device))

    plan = dataclasses.replace(_plan(), offload_policy = dm.OFFLOAD_NONE, secondary_ordinal = 1)
    monkeypatch.setattr(mg, "apply_split_plan", lambda pipe, plan, placement, logger = None: False)
    monkeypatch.setattr(dm, "keep_cpu_weights_on_offload", lambda pipe, logger = None: 0)
    policy, _tiling = dm.apply_memory_plan(Pipe(), plan, device = "cuda", placement_device = "cuda:0")
    assert policy == dm.OFFLOAD_MODEL and calls == [("offload", "cuda:0")]

    placed = []
    monkeypatch.setattr(
        mg, "apply_split_plan", lambda pipe, plan, placement, logger = None: placed.append(1) or True
    )
    policy, _tiling = dm.apply_memory_plan(Pipe(), plan, device = "cuda", placement_device = "cuda:0")
    assert policy == dm.OFFLOAD_NONE and placed == [1]


# ---------------------------------------------------------------------------------------------------- CFG-parallel


class _Dit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.calls = 0

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        return_dict = False,
    ):
        self.calls += 1
        return (self.proj(hidden_states) * timestep + encoder_hidden_states.sum(),)


def _router(dit, **kw):
    replicas = []

    def factory():
        r = mg.build_replica(dit, torch.device("cpu"))
        replicas.append(r)
        return r

    router = mg.CfgParallelRouter(
        dit, primary = "cpu", secondary = "cpu", replica_factory = factory, **kw
    )
    dit.forward = router
    return router, replicas


def _cfg_loop(
    dit,
    latents,
    steps,
    branches = 2,
    scale = 4.0,
):
    cond = torch.randn(3, 4, generator = torch.Generator().manual_seed(1))
    uncond = torch.randn(3, 4, generator = torch.Generator().manual_seed(2))
    extra = torch.randn(3, 4, generator = torch.Generator().manual_seed(3))
    for i in range(steps):
        t = torch.tensor(1.0 - i / steps)
        outs = [
            dit(hidden_states = latents, timestep = t, encoder_hidden_states = e)[0]
            for e in (cond, uncond, extra)[:branches]
        ]
        if branches == 1:
            pred = outs[0]
        else:
            pred = outs[1] + scale * (outs[0] - outs[1])
            if branches == 3:
                pred = pred + 0.5 * (outs[0] - outs[2])
        latents = latents - 0.1 * pred
    return latents


@pytest.mark.parametrize("branches", [2, 3])
def test_cfg_parallel_is_exact_and_routes_odd_branches(branches):
    torch.manual_seed(0)
    dit = _Dit()
    latents = torch.randn(2, 4)
    reference = _cfg_loop(dit, latents.clone(), steps = 5, branches = branches)
    router, replicas = _router(dit)
    out = _cfg_loop(dit, latents.clone(), steps = 5, branches = branches)
    assert torch.equal(out, reference)
    assert len(replicas) == 1 and replicas[0] is not dit
    # step 0 builds the replica (its odd branch ran on the primary), steps 1-4 run the odd branch on the replica
    assert router.stats["steps_parallel"] == 4
    assert router.stats["replica_calls"] == 4 * (branches // 2)
    assert router.describe()["disabled_reason"] is None
    # the replica's weights are its own tensors with equal values
    assert replicas[0].proj.weight is not dit.proj.weight
    assert torch.equal(replicas[0].proj.weight, dit.proj.weight)


def test_guidance_distilled_model_never_builds_a_replica():
    dit = _Dit()
    router, replicas = _router(dit)
    _cfg_loop(dit, torch.randn(2, 4), steps = 6, branches = 1)
    assert replicas == [] and router.replica is None
    assert router.stats["replica_calls"] == 0


def test_replica_failure_recomputes_on_the_primary():
    torch.manual_seed(0)
    dit = _Dit()
    latents = torch.randn(2, 4)
    reference = _cfg_loop(dit, latents.clone(), steps = 4)
    router, replicas = _router(dit)

    class Broken(torch.nn.Module):
        def forward(self, *a, **k):
            raise RuntimeError("replica kernel failed")

    router.replica_factory = lambda: Broken()
    out = _cfg_loop(dit, latents.clone(), steps = 4)
    assert torch.equal(out, reference)
    assert "replica call failed" in router.describe()["disabled_reason"]
    assert router.replica is None


def test_replica_declined_when_a_submodule_hook_captures_the_source():
    dit = _Dit()
    source = dit.proj
    dit.proj.register_forward_hook(lambda m, i, o: o + source.bias)  # closes over the source module
    router, replicas = _router(dit)
    _cfg_loop(dit, torch.randn(2, 4), steps = 3)
    assert replicas == []
    # refused for hooks is retryable: the odd branch stays on the primary until the hook is gone
    assert router.describe()["disabled_reason"] is None
    assert "proj._forward_hooks" in router.describe()["blocked_by"]


def test_portable_hooks_and_root_hooks_are_accepted():
    torch.manual_seed(0)
    dit = _Dit()
    scale = 0.5
    hook = lambda m, i, o: o * scale  # noqa: E731 - captures only a float: shared safely
    hook.__module__ = mg.__name__  # one of Studio's own hooks
    dit.proj.register_forward_hook(hook)
    calls = []
    dit.register_forward_pre_hook(
        lambda m, a: calls.append(1)
    )  # root: runs on the source around every branch
    assert mg._hooks_present(dit) is None
    latents = torch.randn(2, 4)
    reference = _cfg_loop(dit, latents.clone(), steps = 4)
    n_ref = len(calls)
    router, replicas = _router(dit)
    out = _cfg_loop(dit, latents.clone(), steps = 4)
    assert torch.equal(out, reference)
    assert router.stats["replica_calls"] == 3
    # the root hook ran exactly once per denoiser call, routed or not (stripped from the copy)
    assert len(calls) == 2 * n_ref
    assert not replicas[0]._forward_pre_hooks


def test_closure_forward_override_is_declined_but_bound_methods_are_rebound():
    import types

    dit = _Dit()
    source = dit.proj
    dit.proj.forward = lambda x: source.weight @ x  # a closure would keep reading the source module
    assert "instance override" in mg._hooks_present(dit)
    dit2 = _Dit()
    dit2.proj.forward = types.MethodType(
        lambda self, x: torch.nn.functional.linear(x, self.weight), dit2.proj
    )
    assert mg._hooks_present(dit2) is None
    replica = mg.build_replica(dit2, torch.device("cpu"))
    assert replica.proj.forward.__self__ is replica.proj


def test_replica_reapplies_recorded_compile(monkeypatch):
    mg.install_compile_recorder()
    dit = _Dit()
    dit.proj.compile(backend = "eager")
    assert dit.proj.__dict__["_unsloth_compile_args"] == ((), {"backend": "eager"})
    replica = mg.build_replica(dit, torch.device("cpu"))
    assert replica._unsloth_replica_compiled == 1
    assert replica.proj._compiled_call_impl is not None
    assert replica.proj._compiled_call_impl is not dit.proj._compiled_call_impl
    x = torch.randn(2, 4)
    assert torch.equal(replica.proj(x), dit.proj(x))


def test_compiled_without_recorded_arguments_is_refused():
    dit = _Dit()
    dit.proj._compiled_call_impl = lambda *a, **k: None
    with pytest.raises(RuntimeError, match = "recorded compile"):
        mg.build_replica(dit, torch.device("cpu"))


def test_worker_propagates_grad_mode_and_context_variables():
    var = __import__("contextvars").ContextVar("probe", default = "unset")
    worker = mg._ReplicaWorker(torch.device("cpu"))
    try:
        var.set("caller")
        with torch.inference_mode():
            seen = worker.call(
                lambda: (
                    torch.is_inference_mode_enabled(),
                    var.get(),
                    threading.current_thread().name,
                )
            )
        assert seen[0] is True and seen[1] == "caller" and seen[2].startswith("unsloth-cfg-replica")
        with pytest.raises(ValueError):
            worker.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
    finally:
        worker.close()


def test_branch_tensors_are_prefetched_fresh_every_step():
    torch.manual_seed(0)
    dit = _Dit()
    latents = torch.randn(2, 4)
    reference = _cfg_loop(dit, latents.clone(), steps = 4)
    router, _replicas = _router(dit)
    copies = []
    real_prefetch = router._prefetch

    def spy(args, kwargs):
        real_prefetch(args, kwargs)
        copies.append(len(router.carry))

    router._prefetch = spy
    out = _cfg_loop(dit, latents.clone(), steps = 4)
    assert torch.equal(out, reference)
    # Prefetch runs before every primary branch from step 1. Step 1 has no carry yet (step 0 built the replica on its
    # odd branch); from step 2 the previous step's odd-branch tensors except its latent (timestep, uncond embeds).
    assert copies == [0, 2, 2]


def test_cfg_parallel_decline_reasons(monkeypatch):
    class Pipe:
        transformer = _Dit()

    sel = mg.MultiGpuSelection(primary = 0, secondaries = (1,))
    assert (
        mg.cfg_parallel_decline(Pipe(), offload_policy = "none", selection = None)
        == "one card selected"
    )
    assert "offloaded" in mg.cfg_parallel_decline(Pipe(), offload_policy = "model", selection = sel)
    # a group plan that left the denoiser resident (encoders streamed) qualifies; a hooked denoiser does not
    assert "CUDA" in mg.cfg_parallel_decline(Pipe(), offload_policy = "group", selection = sel)
    hooked = Pipe()
    hooked.transformer = _Dit()
    source = hooked.transformer.proj
    hooked.transformer.proj.register_forward_pre_hook(lambda m, a: source)
    assert "hooked" in mg.cfg_parallel_decline(hooked, offload_policy = "group", selection = sel)
    assert "CUDA" in mg.cfg_parallel_decline(Pipe(), offload_policy = "none", selection = sel)

    class Unet:
        transformer = None

    assert "U-Net" in mg.cfg_parallel_decline(Unet(), offload_policy = "none", selection = sel)
    monkeypatch.setenv(mg.CFG_PARALLEL_ENV, "0")
    assert mg.CFG_PARALLEL_ENV in mg.cfg_parallel_decline(
        Pipe(), offload_policy = "none", selection = sel
    )


def test_status_reports_a_declined_cfg_parallel():
    class Pipe:
        transformer = _Dit()

    pipe = Pipe()
    with mg.selection_scope(mg.make_selection(0, [0, 1])):
        reason = mg.install_cfg_parallel(pipe, offload_policy = "model")
    assert "offloaded" in reason
    assert mg.placement_status(pipe) == {"cfg_parallel": {"armed": False, "declined": reason}}
    other = Pipe()
    assert mg.install_cfg_parallel(other, offload_policy = "none") == "one card selected"
    assert mg.placement_status(other) is None  # one card: nothing to report


def test_replica_is_rebuilt_when_the_weights_change():
    torch.manual_seed(0)
    dit = _Dit()
    latents = torch.randn(2, 4)
    router, replicas = _router(dit)
    _cfg_loop(dit, latents.clone(), steps = 3)
    assert len(replicas) == 1
    with torch.no_grad():
        dit.proj.weight.mul_(0.5)  # an in-place fuse, as a LoRA merge does
    reference_router = dit.__dict__.pop("forward")
    reference = _cfg_loop(dit, latents.clone(), steps = 4)
    dit.forward = reference_router
    out = _cfg_loop(dit, latents.clone(), steps = 4)
    assert torch.equal(out, reference)
    assert router.stats["replica_rebuilds"] == 1 and len(replicas) == 2
    assert torch.equal(replicas[1].proj.weight, dit.proj.weight)


def test_adapter_scale_changes_the_fingerprint():
    dit = _Dit()
    before = mg.weights_fingerprint(dit)
    dit.proj.scaling = {"default": 1.0}
    with_adapter = mg.weights_fingerprint(dit)
    dit.proj.scaling = {"default": 0.5}
    assert len({before, with_adapter, mg.weights_fingerprint(dit)}) == 3


def test_with_load_selection_scopes_and_pops_the_ordinals():
    seen = {}

    class Backend:
        @mg.with_load_selection
        def _run_load(self, **kwargs):
            """body"""
            seen["kwargs"] = dict(kwargs)
            seen["selection"] = mg.current_selection()

    Backend()._run_load(gpu_ordinal = 1, _gpu_ordinals = [0, 1], repo_id = "x")
    assert seen["kwargs"] == {"gpu_ordinal": 1, "repo_id": "x"}
    assert seen["selection"] == mg.MultiGpuSelection(primary = 1, secondaries = (0,))
    assert mg.current_selection() is None
    Backend()._run_load(gpu_ordinal = 1, repo_id = "x")
    assert seen["selection"] is None
    # source inspection still reaches the body
    import inspect

    assert '"""body"""' in inspect.getsource(Backend._run_load)


def test_portability_looks_inside_captured_containers():
    holder = {"m": torch.nn.Linear(1, 1)}
    assert not mg._portable(lambda m, a: holder)
    plain = {"scale": 2.0}
    # even scalar-only containers are shared mutable state between the source and the replica
    assert not mg._portable(lambda m, a: plain)
    frozen = ("scale", 2.0)
    assert mg._portable(lambda m, a: frozen)


class _BlockDit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = torch.nn.Linear(4, 4)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        return_dict = False,
    ):
        x = self.proj_in(hidden_states) * timestep
        for block in self.blocks:
            x = block(x) + encoder_hidden_states.sum()
        return (x,)


def test_replica_drops_group_offload_and_stays_exact():
    from diffusers.hooks import apply_group_offloading

    torch.manual_seed(0)
    dit = _BlockDit()
    latents = torch.randn(2, 4)
    reference = _cfg_loop(dit, latents.clone(), steps = 4)
    apply_group_offloading(
        dit,
        onload_device = torch.device("cpu"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
    )
    assert mg._hooks_present(dit) is None
    assert mg._registry_kind(dit) == "offload"
    router, replicas = _router(dit)
    out = _cfg_loop(dit, latents.clone(), steps = 4)
    assert torch.equal(out, reference)
    assert router.stats["replica_calls"] == 3
    replica = replicas[0]
    for sub in replica.modules():
        assert "_diffusers_hook" not in sub.__dict__
        assert "forward" not in sub.__dict__


def test_step_cache_registry_is_declined():
    class Registry:
        hooks = {"first_block_cache": object()}

    dit = _Dit()
    dit._diffusers_hook = Registry()
    assert "first_block_cache" in mg._hooks_present(dit)


def test_offload_torch_hooks_are_stripped_from_the_replica():
    dit = _Dit()
    source = dit.proj
    calls = []

    def prefetch_begin(module, args):  # captures the source, as a prefetcher does
        calls.append(source)

    prefetch_begin.__module__ = "core.inference.diffusion_offload_prefetch"
    dit.proj.register_forward_pre_hook(prefetch_begin)
    assert mg._hooks_present(dit) is None
    replica = mg.build_replica(dit, torch.device("cpu"))
    assert not replica.proj._forward_pre_hooks
    replica(
        hidden_states = torch.ones(1, 4),
        timestep = torch.tensor(1.0),
        encoder_hidden_states = torch.ones(1, 4),
    )
    assert calls == []


def test_predicted_branch_hits_and_stays_exact():
    torch.manual_seed(0)
    dit = _Dit()
    latents = torch.randn(2, 4)
    reference = _cfg_loop(dit, latents.clone(), steps = 6)
    router, _replicas = _router(dit)
    out = _cfg_loop(dit, latents.clone(), steps = 6)
    assert torch.equal(out, reference)
    # step 0 builds, step 1 learns nothing new (template from step 0), steps 1-5 launch predictions
    assert router.stats["predicted_hits"] == 5 and router.stats["predicted_misses"] == 0
    assert router.stats["replica_calls"] == 5


def test_mispredicted_branch_is_recomputed_exactly():
    torch.manual_seed(0)
    dit = _Dit()
    latents = torch.randn(2, 4)
    uncond = torch.randn(3, 4)

    def loop(steps):
        x = latents.clone()
        cond = torch.randn(3, 4, generator = torch.Generator().manual_seed(1))
        for i in range(steps):
            t = torch.tensor(1.0 - i / steps)
            a = dit(hidden_states = x, timestep = t, encoder_hidden_states = cond)[0]
            # a fresh negative tensor every step (equal values, new object): the prediction cannot be trusted
            b = dit(hidden_states = x, timestep = t, encoder_hidden_states = uncond.clone())[0]
            x = x - 0.1 * (b + 4.0 * (a - b))
        return x

    reference = loop(5)
    router, _replicas = _router(dit)
    out = loop(5)
    assert torch.equal(out, reference)
    assert router.stats["predicted_hits"] == 0 and router.stats["predicted_misses"] == 4
    assert router.stats["replica_calls"] == 4  # still on the replica, just not overlapped


def test_inline_recomputed_scalars_are_predicted_by_value():
    torch.manual_seed(0)
    dit = _Dit()
    latents = torch.randn(2, 4)
    cond = torch.randn(3, 4)
    uncond = torch.randn(3, 4)

    def loop(steps):
        x = latents.clone()
        for i in range(steps):
            t = torch.tensor(1000.0 * (1.0 - i / steps))
            a = dit(hidden_states = x, timestep = t / 1000, encoder_hidden_states = cond)[
                0
            ]  # fresh object per call
            b = dit(hidden_states = x, timestep = t / 1000, encoder_hidden_states = uncond)[0]
            x = x - 0.1 * (b + 4.0 * (a - b))
        return x

    reference = loop(6)
    router, _replicas = _router(dit)
    out = loop(6)
    assert torch.equal(out, reference)
    assert router.stats["predicted_hits"] == 5 and router.stats["predicted_misses"] == 0


def test_bitwise_equal_distinguishes_signed_zero():
    assert mg._bitwise_equal(torch.tensor([0.0]), torch.tensor([0.0]))
    assert mg._bitwise_equal(
        torch.tensor(0.5), torch.tensor(1.0) / 2
    )  # 0-dim, as an inline timestep / 1000
    assert not mg._bitwise_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not mg._bitwise_equal(torch.tensor([1.0]), torch.tensor([1.0], dtype = torch.float64))


def test_failed_split_applies_the_single_card_plan(monkeypatch):
    calls = []

    class Pipe:
        components: dict = {}

        def enable_model_cpu_offload(self, device = None):
            calls.append(("model", device))

    monkeypatch.setattr(mg, "secondary_resident_limit_mib", lambda o, mode: 13600)
    with mg.selection_scope(mg.make_selection(0, [0, 1])):
        plan = _plan()
    assert plan.secondary_ordinal == 1
    assert plan.fallback_plan is not None and plan.fallback_plan.offload_policy == dm.OFFLOAD_GROUP
    monkeypatch.setattr(mg, "apply_split_plan", lambda pipe, plan, placement, logger = None: False)
    monkeypatch.setattr(
        dm,
        "_apply_group_offload",
        lambda pipe, placement, logger, **kw: calls.append(("group", placement)) or True,
    )
    policy, _tiling = dm.apply_memory_plan(Pipe(), plan, device = "cuda", placement_device = "cuda:0")
    # the group policy one card would have used, not a blind whole-module offload
    assert policy == dm.OFFLOAD_GROUP and calls == [("group", "cuda:0")]


def test_post_load_split_keeps_the_original_plan_as_fallback(monkeypatch):
    plan = _plan()
    sizes = {
        "transformer": (9000, "dit"),
        "text_encoder": (8000, "text_encoder"),
        "vae": (300, "other"),
    }
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: sizes)
    monkeypatch.setattr(mg, "secondary_resident_limit_mib", lambda o, mode: 12750)
    monkeypatch.setattr(mg, "best_secondary", lambda selection, probes = None: 1)
    with mg.selection_scope(mg.make_selection(0, [0, 1])):
        refined = mg.refine_plan_for_multigpu(object(), plan)
    assert refined.fallback_plan is plan


def test_hook_with_a_mutable_closure_counter_is_declined():
    # The two branches run concurrently, and deepcopy keeps the hook function (and its closure) shared.
    dit = _Dit()
    counter = [0]

    def count(module, args):
        counter[0] += 1

    dit.proj.register_forward_pre_hook(count)
    assert "proj._forward_pre_hooks" in mg._hooks_present(dit)
    router, replicas = _router(dit)
    _cfg_loop(dit, torch.randn(2, 4), steps = 3)
    assert replicas == [] and "hooks" in router.describe()["blocked_by"]


def test_replica_is_dropped_when_the_source_gains_a_cache_hook():
    # transformer_cache=auto installs a stateful step cache on the SOURCE at generate time, after the replica exists
    torch.manual_seed(0)
    dit = _Dit()
    router, replicas = _router(dit)
    _cfg_loop(dit, torch.randn(2, 4), steps = 3)
    assert len(replicas) == 1

    class Registry:
        hooks = {"fbc_leader": object()}

    dit._diffusers_hook = Registry()
    _cfg_loop(dit, torch.randn(2, 4), steps = 2)
    assert router.replica is None
    assert "fbc_leader" in router.describe()["disabled_reason"]


def test_post_load_split_keeps_vae_tiling(monkeypatch):
    plan = dataclasses.replace(_plan(), vae_tiling = True, vae_slicing = True)
    sizes = {
        "transformer": (9000, "dit"),
        "text_encoder": (8000, "text_encoder"),
        "vae": (300, "other"),
    }
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: sizes)
    monkeypatch.setattr(mg, "secondary_resident_limit_mib", lambda o, mode: 12750)
    monkeypatch.setattr(mg, "best_secondary", lambda selection, probes = None: 1)
    with mg.selection_scope(mg.make_selection(0, [0, 1])):
        refined = mg.refine_plan_for_multigpu(object(), plan)
    assert refined.secondary_ordinal == 1
    # the fit may have been priced with the tiled decode peak: keep the saver it assumed
    assert refined.vae_tiling is True and refined.vae_slicing is True


def test_replica_keeps_a_bound_root_forward_override():
    import types

    torch.manual_seed(0)
    dit = _Dit()

    def doubled(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        return_dict = False,
    ):
        return (2 * type(self).forward(self, hidden_states, timestep, encoder_hidden_states)[0],)

    dit.forward = types.MethodType(doubled, dit)
    latents = torch.randn(2, 4)
    reference = _cfg_loop(dit, latents.clone(), steps = 4)
    router, replicas = _router(dit)
    out = _cfg_loop(dit, latents.clone(), steps = 4)
    assert torch.equal(out, reference)
    assert router.stats["replica_calls"] == 3
    assert replicas[0].forward.__self__ is replicas[0]


def test_in_place_edit_after_launch_rejects_the_prediction():
    torch.manual_seed(0)
    dit = _Dit()
    latents = torch.randn(2, 4)
    cond = torch.randn(3, 4)
    uncond = torch.randn(3, 4)

    def loop(steps):
        x = latents.clone()
        neg = uncond.clone()
        for i in range(steps):
            t = torch.tensor(1.0 - i / steps)
            a = dit(hidden_states = x, timestep = t, encoder_hidden_states = cond)[0]
            neg.mul_(1.01)  # same object, new values, after the prediction was launched
            b = dit(hidden_states = x, timestep = t, encoder_hidden_states = neg)[0]
            x = x - 0.1 * (b + 4.0 * (a - b))
        return x

    with torch.inference_mode():  # inference tensors carry no version counter
        reference = loop(5)
        router, _replicas = _router(dit)
        out = loop(5)
    assert torch.equal(out, reference)
    assert router.stats["predicted_hits"] == 0 and router.stats["predicted_misses"] == 4


def test_deferred_speed_profile_invalidates_the_replica(monkeypatch):
    # The 3rd generation compiles / patches the source denoiser; a replica built during the eager images must go.
    import types as _types

    from core.inference.diffusion import DiffusionBackend

    torch.manual_seed(0)
    dit = _Dit()
    router, replicas = _router(dit)

    class Pipe:
        transformer = dit

    pipe = Pipe()
    mg._CFG_ROUTERS[pipe] = router
    _cfg_loop(dit, torch.randn(2, 4), steps = 3)
    assert router.replica is not None
    state = _types.SimpleNamespace(
        pipe = pipe,
        speed_deferred = True,
        deferred_static_plan = None,
        resolved = {},
        transformer_cache = None,
    )
    fake_self = _types.SimpleNamespace(_engage_deferred_profile = lambda state: None)
    DiffusionBackend._engage_deferred_speed(fake_self, state)
    assert router.replica is None and router.disabled_reason is None
    # rebuilt from the mutated source on the next guided step
    _cfg_loop(dit, torch.randn(2, 4), steps = 2)
    assert len(replicas) == 2


class _FlagLinear(torch.nn.Linear):
    def forward(self, x):
        out = super().forward(x)
        return -out if getattr(self, "_unsloth_flip", False) else out


class _FlagDit(_Dit):
    def __init__(self):
        super().__init__()
        self.proj = _FlagLinear(4, 4)


def _flag_pre_hook(module, args, kwargs):
    # like the HunyuanVideo trim hook: per-call state written onto a SUBMODULE of the source
    # varies per call: by branch (negative embeds) and over the trajectory (late steps)
    module.proj._unsloth_flip = bool(kwargs["encoder_hidden_states"].sum() < 0) != bool(
        kwargs["timestep"] < 0.5
    )


def _flag_post_hook(module, args, output):
    module.proj.__dict__.pop("_unsloth_flip", None)


def test_root_hook_state_on_submodules_reaches_the_replica():
    torch.manual_seed(0)
    dit = _FlagDit()
    dit.register_forward_pre_hook(_flag_pre_hook, with_kwargs = True)
    dit.register_forward_hook(_flag_post_hook)
    latents = torch.randn(2, 4)
    cond = torch.ones(3, 4)
    neg = -torch.ones(3, 4)  # the uncond branch flips the flag the cond branch cleared

    def loop(steps):
        x = latents.clone()
        for i in range(steps):
            t = torch.tensor(1.0 - i / steps)
            a = dit(hidden_states = x, timestep = t, encoder_hidden_states = cond)[0]
            b = dit(hidden_states = x, timestep = t, encoder_hidden_states = neg)[0]
            x = x - 0.1 * (b + 4.0 * (a - b))
        return x

    reference = loop(5)
    router, _replicas = _router(dit)
    out = loop(5)
    assert torch.equal(out, reference)
    assert router.stats["replica_calls"] == 4


def test_runtime_controlnet_streams_beside_a_split_pipeline(monkeypatch):
    # A split reports policy "none", but its fit priced only the base pipeline: a ControlNet added at generate time
    # must stream as the single-card offload plan would have, not land whole on the primary.
    import types as _types

    from core.inference.diffusion import _controlnet_streams

    class Pipe:
        pass

    pipe = Pipe()
    state = _types.SimpleNamespace(pipe = pipe, offload_policy = "none", calibrated_placement = False)
    assert _controlnet_streams(state) is False  # one card, resident: unchanged
    mg._PLACEMENTS[pipe] = {"moved": ["text_encoder", "vae"]}
    assert _controlnet_streams(state) is True
    assert _controlnet_streams(_types.SimpleNamespace(pipe = Pipe(), offload_policy = "group")) is True


def test_generate_guard_checks_the_vae_card(monkeypatch):
    # The guard reads the primary's free memory; with the VAE on the secondary, the decode must be checked there.
    import types as _types

    from core.inference import diffusion as D

    seen = []

    @contextlib.contextmanager
    def scope(ordinal):
        seen.append(("scope", ordinal))
        yield

    def shortfall(**kw):
        seen.append(("guard", kw["device_memory"], kw.get("vae_tile_side")))
        return _types.SimpleNamespace(action = D.ACTIVATION_TILE, overridden = False)

    monkeypatch.setattr(D, "diffusion_device_scope", scope)
    monkeypatch.setattr(D, "reclaimable_snapshot_device_memory", lambda target: "card1-free")
    monkeypatch.setattr(D, "raise_on_image_activation_shortfall", shortfall)
    monkeypatch.setattr(D, "engage_vae_tiling", lambda pipe, logger = None: ("restore", True))

    class Vae(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

    pipe = _types.SimpleNamespace(vae = Vae())
    target = _types.SimpleNamespace(ordinal = 0)
    monkeypatch.setattr(mg, "_module_device", lambda m: torch.device("cuda", 1))
    restore, tiled = D._guard_vae_card(
        pipe, target, {"device_memory": "card0-free", "vae_tile_side": 512}, None, None
    )
    assert ("scope", 1) in seen and ("guard", "card1-free", 512) in seen
    assert (restore, tiled) == ("restore", True)
    # the VAE on the primary (or no split): nothing extra
    monkeypatch.setattr(mg, "_module_device", lambda m: torch.device("cuda", 0))
    seen.clear()
    assert D._guard_vae_card(pipe, target, {"device_memory": "card0-free"}, None, None) == (
        None,
        False,
    )
    assert seen == []


def test_secondary_of_lower_compute_capability_takes_no_component(monkeypatch):
    caps = {0: (8, 9), 1: (7, 5), 2: (8, 9), 3: (9, 0)}
    monkeypatch.setattr(mg, "_capability", lambda o: caps.get(int(o)))
    # an Ada primary resolves bf16 / FP8 for itself: the Turing card is dropped, equal or newer cards stay
    assert mg.make_selection(0, [0, 1, 2, 3]) == mg.MultiGpuSelection(primary = 0, secondaries = (2, 3))
    assert mg.make_selection(0, [0, 1]) is None
    # an older primary may use a newer secondary; an unreadable capability keeps today's behaviour
    assert mg.make_selection(1, [1, 0]) == mg.MultiGpuSelection(primary = 1, secondaries = (0,))
    assert mg.make_selection(0, [0, 7]) == mg.MultiGpuSelection(primary = 0, secondaries = (7,))


class _Cache:
    """A prefix-KV style cache: a value holder whose layer tensors the denoiser reads and writes."""

    def __init__(self):
        self.keys = [torch.ones(4)]
        self.writers: list = []


class _CacheDit(_Dit):
    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        kv_cache = None,
        return_dict = False,
    ):
        if kv_cache is not None:
            kv_cache.writers.append(threading.current_thread())
            hidden_states = hidden_states + kv_cache.keys[0]
        return super().forward(hidden_states, timestep, encoder_hidden_states)


def test_branch_with_a_cache_object_stays_on_the_primary():
    torch.manual_seed(0)
    dit = _CacheDit()
    cond, uncond = torch.randn(3, 4), torch.randn(3, 4)

    def loop(latents):
        cache = _Cache()
        for i in range(4):
            t = torch.tensor(1.0 - i / 4)
            outs = [
                dit(hidden_states = latents, timestep = t, encoder_hidden_states = e, kv_cache = cache)[0]
                for e in (cond, uncond)
            ]
            latents = latents - 0.1 * (outs[1] + 4.0 * (outs[0] - outs[1]))
        return latents, cache

    latents = torch.randn(2, 4)
    reference, _ = loop(latents.clone())
    router, replicas = _router(dit)
    out, cache = loop(latents.clone())
    assert torch.equal(out, reference)
    # the mutable cache is never handed to the replica thread, and no replica is paid for
    assert set(cache.writers) == {threading.current_thread()}
    assert replicas == [] and router.stats["replica_calls"] == 0
    assert router.stats["opaque_primary_calls"] == 4
    assert router.describe()["disabled_reason"] is None
    # plain containers still route to the replica
    assert not mg._holds_opaque_tensors(((torch.ones(1),), {"a": [torch.ones(1)], "b": None}))
    assert mg._holds_opaque_tensors({"kv_cache": _Cache()})


def test_split_video_request_is_guarded_on_every_card():
    from core.inference import video as vid

    class Pipe:
        def __init__(self):
            self.components = {
                "transformer": torch.nn.Linear(1024, 1024),  # 4 MiB
                "text_encoder": torch.nn.Linear(1024, 512),  # 2 MiB
                "vae": torch.nn.Linear(512, 512),  # 1 MiB
            }

    pipe = Pipe()
    mg._PLACEMENTS[pipe] = {
        "placement": {"transformer": "cuda:0", "text_encoder": "cuda:1", "vae": "cuda:1"},
        "moved": ["text_encoder", "vae"],
    }
    cards = vid._video_split_card_floors(pipe)
    assert cards == ((0, 4, False), (1, 3, True))
    big = dict(width = 1280, height = 720, frames = 121)
    clip = vid.estimate_video_runtime_mib(width = 1280, height = 720, num_frames = 121)
    roomy = {0: 1 << 20, 1: 1 << 20}
    assert (
        vid.video_split_shortfall_message(cards, family = "f", available_mib = roomy.get, **big) is None
    )
    # the decode card is priced for the decoded clip; the denoiser card only for its weights + base headroom
    tight = {0: 4 + dm.DEFAULT_BASE_OVERHEAD_MIB, 1: 3 + dm.DEFAULT_BASE_OVERHEAD_MIB}
    message = vid.video_split_shortfall_message(cards, family = "f", available_mib = tight.get, **big)
    assert clip > 0 and message is not None and "cuda:1" in message
    tight[0] -= 1
    message = vid.video_split_shortfall_message(cards, family = "f", available_mib = tight.get, **big)
    assert "cuda:0" in message
    del mg._PLACEMENTS[pipe]
    assert vid._video_split_card_floors(pipe) is None


def test_primary_tile_verdict_does_not_tile_a_vae_decoding_elsewhere(monkeypatch):
    import types as _types

    from core.inference import diffusion as D

    engaged = []
    monkeypatch.setattr(D, "diffusion_device_scope", lambda o: contextlib.nullcontext())
    monkeypatch.setattr(D, "reclaimable_snapshot_device_memory", lambda target: "card1-free")
    # the VAE's card decodes untiled; only the smaller primary asks for tiling
    monkeypatch.setattr(
        D,
        "raise_on_image_activation_shortfall",
        lambda **kw: _types.SimpleNamespace(action = "ok", overridden = False),
    )
    monkeypatch.setattr(
        D, "engage_vae_tiling", lambda pipe, logger = None: engaged.append(1) or ("restore", True)
    )
    pipe = _types.SimpleNamespace(vae = torch.nn.Linear(1, 1))
    target = _types.SimpleNamespace(ordinal = 0)
    primary_verdict = _types.SimpleNamespace(action = D.ACTIVATION_TILE, overridden = False)
    kwargs = {"device_memory": "card0-free", "vae_tile_side": 512, "vae_sliced": False}
    monkeypatch.setattr(mg, "_module_device", lambda m: torch.device("cuda", 1))
    assert D._tile_for_verdict(pipe, primary_verdict, target, kwargs, 1, None, None) is None
    assert engaged == []
    # one card: the primary's tile verdict still tiles
    monkeypatch.setattr(mg, "_module_device", lambda m: torch.device("cuda", 0))
    assert D._tile_for_verdict(pipe, primary_verdict, target, kwargs, 1, None, None) == "restore"
    assert engaged == [1]


def test_secondary_free_memory_reads_the_trusted_probe(monkeypatch):
    import utils.hardware as hw

    seen = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (80 << 30, 80 << 30))
    monkeypatch.setattr(
        hw,
        "trusted_mem_get_info",
        lambda device = None, module = None: seen.append(device) or (1 << 30, 80 << 30),
    )
    # Windows ROCm over-reports raw free memory; the capped figure is what budgets the secondary
    assert mg._free_total_mib(1) == (1024, 80 * 1024)
    assert seen == [1]


_GLOBAL_HOOK_CALLS = []


def _global_state_hook(module, inputs, output):
    _GLOBAL_HOOK_CALLS.append(1)  # module-global state: invisible to the closure check
    return output * (1.0 + len(_GLOBAL_HOOK_CALLS) % 2)


def test_foreign_submodule_hook_declines_the_replica():
    dit = _Dit()
    dit.proj.register_forward_hook(_global_state_hook)
    assert _global_state_hook.__closure__ is None
    assert mg._hooks_present(dit) == "proj._forward_hooks"
    router, replicas = _router(dit)
    _cfg_loop(dit, torch.randn(2, 4), steps = 3)
    assert replicas == [] and router.stats["replica_calls"] == 0
    # the same function from a Studio module is shared as before
    _global_state_hook.__module__, saved = mg.__name__, _global_state_hook.__module__
    try:
        assert mg._hooks_present(_Dit()) is None
        hooked = _Dit()
        hooked.proj.register_forward_hook(_global_state_hook)
        assert mg._hooks_present(hooked) is None
    finally:
        _global_state_hook.__module__ = saved


def test_force_mode_refines_a_resident_plan(monkeypatch):
    plan = dataclasses.replace(_plan(), offload_policy = dm.OFFLOAD_NONE)
    sizes = {
        "transformer": (9000, "dit"),
        "text_encoder": (8000, "text_encoder"),
        "vae": (300, "other"),
    }
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: sizes)
    monkeypatch.setattr(mg, "secondary_resident_limit_mib", lambda o, mode: 12750)
    monkeypatch.setattr(mg, "best_secondary", lambda selection, probes = None: 1)
    with mg.selection_scope(mg.make_selection(0, [0, 1])):
        assert (
            mg.refine_plan_for_multigpu(object(), plan) is plan
        )  # auto: a resident plan stays on one card
        monkeypatch.setenv(mg.MULTIGPU_ENV, "force")
        refined = mg.refine_plan_for_multigpu(object(), plan)
    assert refined.secondary_ordinal == 1 and refined.offload_policy == dm.OFFLOAD_NONE
    assert refined.secondary_components == ("text_encoder", "vae")
    assert refined.fallback_plan is plan


def test_ltx25_sidecar_encoder_keeps_the_single_card_plan(monkeypatch):
    import types as _types

    from core.inference import video as vid

    single = _plan()
    split = dataclasses.replace(
        single, offload_policy = dm.OFFLOAD_NONE, secondary_ordinal = 1, fallback_plan = single
    )
    ltx25 = _types.SimpleNamespace(name = "ltx-2.5")
    wan = _types.SimpleNamespace(name = "wan2.2")
    # the planner priced the sidecar encoder as a movable companion: LTX-2.5 falls back to one card
    assert vid._video_single_card_for_sidecar(ltx25, split) is single
    assert vid._video_single_card_for_sidecar(wan, split) is split
    assert vid._video_single_card_for_sidecar(ltx25, single) is single
    # and the post-load refinement (whose loaded sizes omit the sidecar) never splits it
    refined = []
    monkeypatch.setattr(
        vid, "refine_plan_for_multigpu", lambda pipe, plan, logger = None: refined.append(1) or split
    )
    assert vid._video_refine_for_multigpu(ltx25, object(), single) is single
    assert vid._video_refine_for_multigpu(wan, object(), single) is split and refined == [1]


# ---------------------------------------------------------------------------------------------------- CFG-batch / pricing


class _BatchDit(torch.nn.Module):
    """Per-sample math (no reduction across the batch), as a DiT's is."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self._log = [0]  # the call counter is test bookkeeping, not model state the stateful-forward check should see
        self.rows = []

    @property
    def calls(self):
        return self._log[0]

    @calls.setter
    def calls(self, value):
        self._log[0] = value

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        return_dict = False,
    ):
        self.calls += 1
        self.rows.append(hidden_states.shape[0])
        return (self.proj(hidden_states) * timestep + encoder_hidden_states.tanh(),)


def _batch_loop(
    dit,
    latents,
    steps,
    cond,
    uncond,
    scale = 4.0,
    edit = None,
):
    for i in range(steps):
        t = torch.tensor(1.0 - i / steps)
        c = dit(hidden_states = latents, timestep = t, encoder_hidden_states = cond)[0]
        if edit is not None:
            edit(i, uncond)
        u = dit(hidden_states = latents, timestep = t, encoder_hidden_states = uncond)[0]
        latents = latents - 0.1 * (u + scale * (c - u))
    return latents


def _embeds(rows = 2, length = 4):
    g = torch.Generator().manual_seed(1)
    return torch.randn(rows, length, generator = g), torch.randn(rows, length, generator = g)


def test_cfg_batch_runs_one_forward_per_step_and_matches():
    torch.manual_seed(0)
    dit = _BatchDit()
    latents = torch.randn(2, 4)
    cond, uncond = _embeds()
    reference = _batch_loop(dit, latents.clone(), 5, cond, uncond)
    dit.calls, dit.rows = 0, []
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    out = _batch_loop(dit, latents.clone(), 5, cond, uncond)
    torch.testing.assert_close(out, reference, rtol = 1e-5, atol = 1e-6)
    assert dit.calls == 2 + 4 and dit.rows[2:] == [4, 4, 4, 4]
    assert router.stats["batched_steps"] == 4 and router.stats["batched_hits"] == 4
    assert router.describe()["strategy"] == "cfg_batch"


def test_cfg_batch_rejects_an_odd_branch_edited_in_place():
    torch.manual_seed(0)
    dit = _BatchDit()
    latents = torch.randn(2, 4)
    cond, uncond = _embeds()

    def edit(i, u):
        if i == 2:
            u.add_(1.0)

    reference = _batch_loop(dit, latents.clone(), 4, cond, uncond.clone(), edit = edit)
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    out = _batch_loop(dit, latents.clone(), 4, cond, uncond.clone(), edit = edit)
    torch.testing.assert_close(out, reference, rtol = 1e-5, atol = 1e-6)
    assert router.stats["batched_misses"] == 1


def test_cfg_batch_skips_branches_of_different_shapes():
    torch.manual_seed(0)

    class LenDit(_BatchDit):
        def forward(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states,
            return_dict = False,
        ):
            self.calls += 1
            return (
                self.proj(hidden_states) * timestep + encoder_hidden_states.mean(1, keepdim = True),
            )

    dit = LenDit()
    latents = torch.randn(2, 4)
    cond, _ = _embeds(length = 4)
    _, uncond = _embeds(length = 6)
    reference = _batch_loop(dit, latents.clone(), 4, cond, uncond)
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    out = _batch_loop(dit, latents.clone(), 4, cond, uncond)
    assert torch.equal(out, reference)
    assert router.stats["batched_steps"] == 0 and router.stats["unbatched_steps"] == 3
    assert " vs " in router.describe()["last_unbatched_reason"]


def test_cfg_batch_guidance_distilled_never_batches():
    dit = _BatchDit()
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    latents = torch.randn(2, 4)
    cond, _ = _embeds()
    for i in range(4):
        latents = (
            latents
            - dit(hidden_states = latents, timestep = torch.tensor(0.5), encoder_hidden_states = cond)[0]
        )
    assert router.stats["batched_steps"] == 0 and dit.calls == 4


def test_cfg_batch_out_of_memory_falls_back(monkeypatch):
    torch.manual_seed(0)
    dit = _BatchDit()
    latents = torch.randn(2, 4)
    cond, uncond = _embeds()
    reference = _batch_loop(dit, latents.clone(), 4, cond, uncond)
    router = mg.CfgBatchRouter(dit)
    inner = router.orig_forward

    def oom_on_batch(*a, **k):
        if k["hidden_states"].shape[0] == 4:
            raise torch.OutOfMemoryError("fake")
        return inner(*a, **k)

    oom_on_batch.__module__ = mg.__name__  # stands in for a Studio forward patch under the router
    router.orig_forward = oom_on_batch
    dit.forward = router
    out = _batch_loop(dit, latents.clone(), 4, cond, uncond)
    torch.testing.assert_close(out, reference, rtol = 0, atol = 0)
    assert "out of memory" in router.describe()["disabled_reason"]


def test_cfg_batch_decline_rules(monkeypatch):
    class Flux2Transformer2DModel(_BatchDit):  # a listed class name from diffusers
        pass

    Flux2Transformer2DModel.__module__ = "diffusers.models.transformers.transformer_flux2"
    Flux2Transformer2DModel.__qualname__ = "Flux2Transformer2DModel"

    class Pipe:
        def __init__(self):
            self.transformer = Flux2Transformer2DModel()

    assert "resident" in mg.cfg_batch_decline(Pipe(), offload_policy = "none")
    assert "resident" in mg.cfg_batch_decline(Pipe(), offload_policy = "group")
    monkeypatch.setenv(mg.CFG_BATCH_ENV, "force")
    assert mg.cfg_batch_decline(Pipe(), offload_policy = "none") is None
    monkeypatch.setenv(mg.CFG_BATCH_ENV, "0")
    assert mg.CFG_BATCH_ENV in mg.cfg_batch_decline(Pipe(), offload_policy = "group")
    monkeypatch.delenv(mg.CFG_BATCH_ENV)

    class Unet:
        transformer = None

    assert "U-Net" in mg.cfg_batch_decline(Unet(), offload_policy = "group")


def test_price_cfg_parallel_declines_a_slow_link():
    fast = mg.price_cfg_parallel(
        branch_s = 0.5, in_bytes = 64 << 20, out_bytes = 32 << 20, link_in_gib_s = 20.0, link_out_gib_s = 20.0
    )
    assert (
        fast["strategy"] == "cfg_parallel" and fast["cfg_parallel_step_s"] < fast["single_step_s"]
    )
    slow = mg.price_cfg_parallel(
        branch_s = 0.05, in_bytes = 64 << 20, out_bytes = 32 << 20, link_in_gib_s = 1.0, link_out_gib_s = 1.0
    )
    assert slow["strategy"] == "single"
    unknown = mg.price_cfg_parallel(
        branch_s = 0.05,
        in_bytes = 64 << 20,
        out_bytes = 32 << 20,
        link_in_gib_s = None,
        link_out_gib_s = None,
    )
    assert unknown["strategy"] == "cfg_parallel"


def _probe(o, free, links):
    return mg.DeviceProbe(
        ordinal = o,
        name = "card",
        total_mib = 16000,
        free_mib = free,
        peers = tuple((b, True, g) for b, g in links.items()),
    )


def test_rank_secondaries_prefers_the_fast_link_then_free_memory(monkeypatch):
    monkeypatch.delenv(mg.LINK_ENV, raising = False)
    probes = {
        0: _probe(0, 1000, {1: 20.0, 2: 2.0, 3: 18.0}),
        1: _probe(1, 9000, {0: 20.0}),
        2: _probe(2, 15000, {0: 2.0}),
        3: _probe(3, 12000, {0: 19.0}),
    }
    sel = mg.MultiGpuSelection(primary = 0, secondaries = (1, 2, 3))
    assert mg.best_secondary(sel, probes) == 3
    assert mg.rank_secondaries(0, [1, 2, 3], lambda o: probes[o].free_mib, probes) == [3, 1, 2]
    blind = {o: _probe(o, p.free_mib, {}) for o, p in probes.items()}
    assert mg.best_secondary(sel, blind) == 2


def test_cfg_parallel_router_priced_out_keeps_one_card(monkeypatch):
    torch.manual_seed(0)
    dit = _Dit()
    latents = torch.randn(2, 4)
    reference = _cfg_loop(dit, latents.clone(), steps = 4)
    router, replicas = _router(dit)
    monkeypatch.setattr(router, "_should_price", lambda: True)
    monkeypatch.setattr(router, "_timed_primary", lambda a, k: (router.orig_forward(*a, **k), 1e-4))
    monkeypatch.setattr(mg, "link_gib_s", lambda src, dst, probes = None: 1e-9)
    router.primary = router.secondary = torch.device("cpu")
    out = _cfg_loop(dit, latents.clone(), steps = 4)
    assert torch.equal(out, reference)
    assert replicas == [] and "priced out" in router.describe()["disabled_reason"]
    assert router.describe()["pricing"]["strategy"] == "single"


def test_price_cfg_batch_declines_a_compute_bound_call():
    # T4x2 FLUX.2-klein-base: 2.13 s per call, its 7.7 GB stream at 9.2 GiB/s takes 0.84 s -> compute-bound
    t4 = mg.price_cfg_batch(call_s = 2.13, upload_s = 0.84)
    assert t4["strategy"] == "single"
    # upload-bound (a fast card on a narrow link): one upload serves both branches
    assert mg.price_cfg_batch(call_s = 0.2, upload_s = 0.18)["strategy"] == "cfg_batch"
    assert mg.price_cfg_batch(call_s = 0.2, upload_s = None)["strategy"] == "cfg_batch"


def test_cfg_batch_measured_no_gain_stops_after_the_first_batched_step():
    torch.manual_seed(0)
    dit = _BatchDit()
    latents = torch.randn(2, 4)
    cond, uncond = _embeds()
    reference = _batch_loop(dit, latents.clone(), 5, cond, uncond)
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    router.plain_times = [0.1, 0.1]
    router._check_batched(0.5)
    assert router.disabled_pending is None
    router._check_batched(0.19)  # fastest 0.19 > 0.92 * 0.2: no gain
    assert router.verify_batched == 0
    out = _batch_loop(dit, latents.clone(), 5, cond, uncond)
    torch.testing.assert_close(out, reference, rtol = 1e-5, atol = 1e-6)
    # the verdict was reached before this loop: it applies at the first step boundary, before any batched call
    assert router.stats["batched_steps"] == 0
    assert "measured no gain" in router.describe()["disabled_reason"]


def test_cfg_batch_falls_back_when_the_batched_call_rejects_its_inputs():
    torch.manual_seed(0)
    dit = _BatchDit()
    latents = torch.randn(2, 4)
    cond, uncond = _embeds()
    reference = _batch_loop(dit, latents.clone(), 4, cond, uncond)
    router = mg.CfgBatchRouter(dit)
    inner = router.orig_forward

    def fixed_batch(*a, **k):
        if k["hidden_states"].shape[0] != 2:
            raise ValueError("this model only runs batch 2")
        return inner(*a, **k)

    fixed_batch.__module__ = mg.__name__  # stands in for a Studio forward patch under the router
    router.orig_forward = fixed_batch
    dit.forward = router
    out = _batch_loop(dit, latents.clone(), 4, cond, uncond)
    torch.testing.assert_close(out, reference, rtol = 0, atol = 0)
    assert "ValueError" in router.describe()["disabled_reason"]


def test_cfg_batch_refuses_an_opaque_mutable_argument():
    torch.manual_seed(0)

    @dataclasses.dataclass
    class Holder:
        state: torch.Tensor

    class HolderDit(_BatchDit):
        def forward(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states,
            holder = None,
            return_dict = False,
        ):
            self.calls += 1
            holder.state.add_(1.0)  # per-call mutation of shared state
            return (
                self.proj(hidden_states) * timestep + encoder_hidden_states.tanh() + holder.state,
            )

    def loop(dit, latents, steps, cond, uncond):
        holder = Holder(torch.zeros(1))
        for i in range(steps):
            t = torch.tensor(1.0 - i / steps)
            c = dit(hidden_states = latents, timestep = t, encoder_hidden_states = cond, holder = holder)[0]
            u = dit(hidden_states = latents, timestep = t, encoder_hidden_states = uncond, holder = holder)[
                0
            ]
            latents = latents - 0.1 * (u + 4.0 * (c - u))
        return latents

    dit = HolderDit()
    latents = torch.randn(2, 4)
    cond, uncond = _embeds()
    reference = loop(dit, latents.clone(), 4, cond, uncond)
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    out = loop(dit, latents.clone(), 4, cond, uncond)
    assert torch.equal(out, reference)
    assert router.stats["batched_steps"] == 0
    assert "opaque" in router.describe()["last_unbatched_reason"]


def test_cfg_batch_stops_timing_a_single_branch_loop(monkeypatch):
    dit = _BatchDit()
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    timed = []
    monkeypatch.setattr(router, "_can_time", lambda latent: True)
    monkeypatch.setattr(router, "_timed_plain", lambda a, k: timed.append(1) or router._plain(a, k))
    latents = torch.randn(2, 4)
    cond, _ = _embeds()
    for _ in range(5):
        latents = (
            latents
            - dit(hidden_states = latents, timestep = torch.tensor(0.5), encoder_hidden_states = cond)[0]
        )
    assert len(timed) == 1  # the first step only: no second branch ever came


def test_cfg_parallel_replica_card_is_chosen_among_those_that_fit(monkeypatch):
    class Pipe:
        transformer = _Dit()

    pipe = Pipe()
    # the tiny DiT's replica needs only the 1024 MiB margin
    probes = {
        0: _probe(0, 20000, {1: 20.0, 2: 2.0}),
        1: _probe(1, 500, {0: 20.0}),  # fastest link, cannot hold the replica
        2: _probe(2, 9000, {0: 2.0}),
    }
    monkeypatch.setattr(mg, "probe_devices", lambda ordinals, measure = True: probes)
    monkeypatch.setattr(mg, "cfg_parallel_decline", lambda pipe, **k: None)
    monkeypatch.setattr(mg, "_module_device", lambda m: torch.device("cuda", 0))
    monkeypatch.delenv(mg.LINK_ENV, raising = False)
    with mg.selection_scope(mg.MultiGpuSelection(primary = 0, secondaries = (1, 2))):
        reason = mg._install_cfg_parallel_only(pipe, offload_policy = "none")
    assert reason is None, reason
    assert mg._CFG_ROUTERS[pipe].secondary == torch.device("cuda", 2)


def test_refine_tries_the_next_card_when_the_fast_link_one_cannot_hold_a_share(monkeypatch):
    plan = _plan()
    sizes = {
        "transformer": (9000, "dit"),
        "text_encoder": (8000, "text_encoder"),
        "vae": (300, "other"),
    }
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: sizes)
    # cuda:1 has the fast link but holds nothing; cuda:2 is slower and fits the companions
    monkeypatch.setattr(mg, "secondary_resident_limit_mib", lambda o, mode: {1: 3000, 2: 12750}[o])
    monkeypatch.setattr(mg, "best_secondary", lambda selection, probes = None: 1)
    monkeypatch.setattr(
        mg, "rank_secondaries", lambda primary, cands, free_of, probes = None: sorted(cands)
    )
    monkeypatch.setattr(mg, "probe_devices", lambda ordinals, measure = True: {})
    with mg.selection_scope(mg.MultiGpuSelection(primary = 0, secondaries = (1, 2))):
        refined = mg.refine_plan_for_multigpu(object(), plan)
    assert refined.offload_policy == dm.OFFLOAD_NONE
    assert refined.secondary_ordinal == 2
    assert refined.secondary_components == ("text_encoder", "vae")


def test_cfg_batch_merge_failure_runs_the_branch_plain(monkeypatch):
    torch.manual_seed(0)
    dit = _BatchDit()
    latents = torch.randn(2, 4)
    cond, uncond = _embeds()
    reference = _batch_loop(dit, latents.clone(), 4, cond, uncond)
    router = mg.CfgBatchRouter(dit)
    dit.forward = router

    def boom(first, second, batch):
        raise RuntimeError("cat failed")

    monkeypatch.setattr(mg, "_batch_leaves", boom)
    out = _batch_loop(dit, latents.clone(), 4, cond, uncond)
    torch.testing.assert_close(out, reference, rtol = 0, atol = 0)
    assert "RuntimeError" in router.describe()["disabled_reason"]


def test_cfg_batch_single_branch_request_after_a_guided_one_stops_batching():
    torch.manual_seed(0)
    dit = _BatchDit()
    cond, uncond = _embeds()
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    _batch_loop(dit, torch.randn(2, 4), 3, cond, uncond)
    before = router.stats["batched_steps"]
    latents = torch.randn(2, 4)
    for _ in range(5):
        latents = (
            latents
            - dit(hidden_states = latents, timestep = torch.tensor(0.5), encoder_hidden_states = cond)[0]
        )
    # at most the first step of the new request is speculated; none after the single-branch step is seen
    assert router.stats["batched_steps"] - before <= 1
    assert router.template is None


def test_cfg_batch_refuses_a_foreign_submodule_hook():
    torch.manual_seed(0)
    dit = _BatchDit()
    seen = []
    dit.proj.register_forward_hook(
        lambda m, i, o: seen.append(o.shape[0])
    )  # stateful, not Studio's
    cond, uncond = _embeds()
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    _batch_loop(dit, torch.randn(2, 4), 4, cond, uncond)
    assert router.stats["batched_steps"] == 0
    assert "hook" in router.describe()["last_unbatched_reason"]


def test_cfg_batch_retimes_both_branches_when_guidance_resumes(monkeypatch):
    dit = _BatchDit()
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    step = [0]
    priced = []
    monkeypatch.setattr(router, "_can_time", lambda latent: True)

    def timed(a, k):
        router.plain_times.append(step[0])
        return router._plain(a, k)

    monkeypatch.setattr(router, "_timed_plain", timed)
    monkeypatch.setattr(router, "_price", lambda: priced.append(list(router.plain_times)))
    cond, uncond = _embeds()
    latents = torch.randn(2, 4)
    for i in range(2):  # single-branch steps first
        step[0] = i
        latents = (
            latents
            - dit(hidden_states = latents, timestep = torch.tensor(0.5), encoder_hidden_states = cond)[0]
        )
    for i in range(2, 5):  # then guidance
        step[0] = i
        c = dit(hidden_states = latents, timestep = torch.tensor(0.5), encoder_hidden_states = cond)[0]
        u = dit(hidden_states = latents, timestep = torch.tensor(0.5), encoder_hidden_states = uncond)[0]
        latents = latents - 0.1 * (u + 4.0 * (c - u))
        if priced:
            break
    assert (
        priced and len(priced[0]) == 2 and priced[0][0] == priced[0][1]
    )  # both from one guided step


def test_cfg_batch_refuses_a_stateful_root_pre_hook():
    torch.manual_seed(0)
    dit = _BatchDit()
    calls = [0]

    def count(module, args, kwargs):  # per-call state: scales each branch by its call index
        calls[0] += 1
        kwargs = dict(kwargs)
        kwargs["timestep"] = kwargs["timestep"] * (1.0 + 0.01 * calls[0])
        return args, kwargs

    dit.register_forward_pre_hook(count, with_kwargs = True)
    cond, uncond = _embeds()
    latents = torch.randn(2, 4)
    reference = _batch_loop(dit, latents.clone(), 4, cond, uncond)
    calls[0] = 0
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    out = _batch_loop(dit, latents.clone(), 4, cond, uncond)
    assert torch.equal(out, reference)
    assert router.stats["batched_steps"] == 0
    assert "_forward_pre_hooks" in router.describe()["last_unbatched_reason"]


def test_cfg_batch_accepts_a_studio_root_hook():
    torch.manual_seed(0)
    dit = _BatchDit()
    hook = lambda m, a: None  # noqa: E731 - stateless, one of Studio's own
    hook.__module__ = mg.__name__
    dit.register_forward_pre_hook(hook)
    cond, uncond = _embeds()
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    _batch_loop(dit, torch.randn(2, 4), 4, cond, uncond)
    assert router.stats["batched_steps"] == 3


def test_cfg_batch_declines_a_pre_existing_outer_forward_layer(monkeypatch):
    class Outer:
        _unsloth_outer_forward = (
            True  # e.g. the static step skip: per-branch state keyed on separate calls
        )

        def __init__(self, module):
            self.inner = type(module).forward.__get__(module)
            self.calls = 0

        def __call__(self, *a, **k):
            self.calls += 1
            return self.inner(*a, **k)

    class Pipe:
        def __init__(self):
            self.transformer = _BatchDit()

    monkeypatch.setenv(mg.CFG_BATCH_ENV, "force")
    pipe = Pipe()
    pipe.transformer.forward = Outer(pipe.transformer)
    reason = mg.install_cfg_batch(pipe, offload_policy = "none")
    assert reason is not None and "outer forward" in reason
    plain = Pipe()
    assert (
        mg.install_cfg_batch(plain, offload_policy = "none") is None
    )  # the class forward alone still arms


def test_cfg_batch_output_rebuild_failure_runs_the_branch_plain(monkeypatch):
    torch.manual_seed(0)
    dit = _BatchDit()
    latents = torch.randn(2, 4)
    cond, uncond = _embeds()
    reference = _batch_loop(dit, latents.clone(), 4, cond, uncond)
    router = mg.CfgBatchRouter(dit)
    dit.forward = router

    def bad_split(
        out,
        batch,
        depth = 0,
    ):
        raise TypeError("init=False field")

    monkeypatch.setattr(mg, "_split_output", bad_split)
    out = _batch_loop(dit, latents.clone(), 4, cond, uncond)
    torch.testing.assert_close(out, reference, rtol = 0, atol = 0)
    assert "TypeError" in router.describe()["disabled_reason"]


def test_cfg_batch_pending_slowdown_applies_on_a_prediction_miss():
    torch.manual_seed(0)
    dit = _BatchDit()
    cond, uncond = _embeds()
    router = mg.CfgBatchRouter(dit)
    dit.forward = router

    def edit(i, u):
        if i == 1:
            router.disabled_pending = "measured no gain"  # set by the first batched call's timing
            u.add_(1.0)  # and this step's odd branch misses its prediction

    _batch_loop(dit, torch.randn(2, 4), 4, cond, uncond.clone(), edit = edit)
    assert router.describe()["disabled_reason"] == "measured no gain"
    assert router.stats["batched_steps"] == 1


def test_cfg_batch_headroom_is_the_denoise_phase_when_calibrated():
    # FLUX.1-like: decode peaks far above denoise; the batched call only adds denoise activations
    act = dm.CalibratedImageActivation(1_000, 892, 2_666, 600, 2_000)
    plan = dm.plan_diffusion_memory(
        target = _Target(),
        device_memory = _memory(),
        model_dense_mib = 20000,
        companion_dense_mib = 9000,
        text_encoder_dense_mib = 8500,
        runtime_headroom_mib = 8192,
        calibrated_activation = act,
    )
    assert mg.plan_denoise_headroom_mib(plan) == 892
    # without a per-phase figure the conservative whole-request headroom stays
    assert mg.plan_denoise_headroom_mib(_plan()) == _plan().estimates["runtime_headroom_mib"]


def _offload_style_wrapper(module, inner):
    """An accelerate / diffusers style offload wrapper: functools.update_wrapper(partial(new_forward, module), inner)."""
    import functools

    def new_forward(mod, *a, **k):
        return inner(*a, **k)

    new_forward.__module__ = "accelerate.hooks"
    return functools.update_wrapper(functools.partial(new_forward, module), inner)


def test_cfg_batch_sees_an_outer_layer_under_an_offload_wrapper(monkeypatch):
    class Outer:
        _unsloth_outer_forward = True

        def __init__(self, module):
            self.inner = type(module).forward.__get__(module)

        def __call__(self, *a, **k):
            return self.inner(*a, **k)

    class Pipe:
        def __init__(self):
            self.transformer = _BatchDit()

    monkeypatch.setenv(mg.CFG_BATCH_ENV, "force")
    hidden = Pipe()
    hidden.transformer.forward = _offload_style_wrapper(
        hidden.transformer, Outer(hidden.transformer)
    )
    reason = mg.install_cfg_batch(hidden, offload_policy = "none")
    assert reason is not None and "outer forward" in reason
    # the same wrapper straight over the class forward still arms
    plain = Pipe()
    plain.transformer.forward = _offload_style_wrapper(
        plain.transformer, type(plain.transformer).forward.__get__(plain.transformer)
    )
    assert mg.install_cfg_batch(plain, offload_policy = "none") is None


@pytest.mark.parametrize("make_state", [lambda: [0], lambda: {}])
def test_cfg_batch_refuses_a_mutable_container_argument(make_state):
    torch.manual_seed(0)

    class StateDit(_BatchDit):
        def forward(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states,
            state = None,
            return_dict = False,
        ):
            self.calls += 1
            if isinstance(state, list):
                state[0] += 1
            else:
                state[len(state)] = 1
            return (self.proj(hidden_states) * timestep + encoder_hidden_states.tanh(),)

    def loop(dit, latents, steps, cond, uncond, state):
        for i in range(steps):
            t = torch.tensor(1.0 - i / steps)
            c = dit(hidden_states = latents, timestep = t, encoder_hidden_states = cond, state = state)[0]
            u = dit(hidden_states = latents, timestep = t, encoder_hidden_states = uncond, state = state)[0]
            latents = latents - 0.1 * (u + 4.0 * (c - u))
        return latents

    dit = StateDit()
    cond, uncond = _embeds()
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    state = make_state()
    loop(dit, torch.randn(2, 4), 4, cond, uncond, state)
    assert router.stats["batched_steps"] == 0
    assert (
        state[0] if isinstance(state, list) else len(state)
    ) == 8  # every per-branch call reached the caller's state
    assert "mutable" in router.describe()["last_unbatched_reason"]


def test_cfg_batch_still_batches_none_and_tuple_arguments():
    torch.manual_seed(0)

    class KwDit(_BatchDit):
        def forward(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states,
            joint_attention_kwargs = None,
            guidance = None,
            ids = (),
            return_dict = False,
        ):
            self.calls += 1
            return (self.proj(hidden_states) * timestep + encoder_hidden_states.tanh(),)

    dit = KwDit()
    cond, uncond = _embeds()
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    latents = torch.randn(2, 4)
    for i in range(4):
        t = torch.tensor(1.0 - i / 4)
        kw = dict(timestep = t, joint_attention_kwargs = None, guidance = None, ids = (1, 2))
        c = dit(hidden_states = latents, encoder_hidden_states = cond, **kw)[0]
        u = dit(hidden_states = latents, encoder_hidden_states = uncond, **kw)[0]
        latents = latents - 0.1 * (u + 4.0 * (c - u))
    assert router.stats["batched_steps"] == 3


def test_cfg_batch_passes_an_empty_attention_kwargs_dict_and_checks_it():
    torch.manual_seed(0)

    class KwDit(_BatchDit):
        def forward(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states,
            joint_attention_kwargs = None,
            return_dict = False,
        ):
            self.calls += 1
            return (self.proj(hidden_states) * timestep + encoder_hidden_states.tanh(),)

    dit = KwDit()
    cond, uncond = _embeds()
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    kwargs_dict = {}  # FLUX.1 passes joint_attention_kwargs = {}
    latents = torch.randn(2, 4)
    for i in range(4):
        t = torch.tensor(1.0 - i / 4)
        c = dit(
            hidden_states = latents,
            timestep = t,
            encoder_hidden_states = cond,
            joint_attention_kwargs = kwargs_dict,
        )[0]
        u = dit(
            hidden_states = latents,
            timestep = t,
            encoder_hidden_states = uncond,
            joint_attention_kwargs = kwargs_dict,
        )[0]
        latents = latents - 0.1 * (u + 4.0 * (c - u))
    assert router.stats["batched_steps"] == 3


def test_cfg_batch_speculation_never_writes_the_callers_empty_dict():
    torch.manual_seed(0)

    class WritingDit(_BatchDit):
        def forward(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states,
            joint_attention_kwargs = None,
            return_dict = False,
        ):
            self.calls += 1
            out = (
                self.proj(hidden_states) * timestep
                + encoder_hidden_states.tanh()
                + len(joint_attention_kwargs)
            )
            if hidden_states.shape[0] == 4:  # writes only inside the doubled call
                joint_attention_kwargs["seen"] = True
            return (out,)

    def loop(dit, latents, kw):
        for i in range(4):
            t = torch.tensor(1.0 - i / 4)
            c = dit(
                hidden_states = latents,
                timestep = t,
                encoder_hidden_states = cond,
                joint_attention_kwargs = kw,
            )[0]
            u = dit(
                hidden_states = latents,
                timestep = t,
                encoder_hidden_states = uncond,
                joint_attention_kwargs = kw,
            )[0]
            latents = latents - 0.1 * (u + 4.0 * (c - u))
        return latents

    cond, uncond = _embeds()
    latents = torch.randn(2, 4)
    dit = WritingDit()
    reference = loop(dit, latents.clone(), {})
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    caller = {}
    out = loop(dit, latents.clone(), caller)
    assert caller == {}
    torch.testing.assert_close(out, reference, rtol = 0, atol = 0)
    assert "dict" in router.describe()["disabled_reason"]


def test_cfg_batch_in_place_write_to_an_input_never_reaches_the_caller():
    torch.manual_seed(0)

    class InPlaceDit(_BatchDit):
        def forward(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states,
            return_dict = False,
        ):
            self.calls += 1
            out = self.proj(hidden_states) * timestep + encoder_hidden_states.tanh()
            encoder_hidden_states.mul_(
                0.5
            )  # writes its input in place: each branch must see its own write
            timestep.add_(0.0)  # a shared, passed-through tensor written in place as well
            return (out,)

    def loop(dit, latents, cond, uncond):
        for i in range(4):
            t = torch.tensor(1.0 - i / 4)
            c = dit(hidden_states = latents, timestep = t, encoder_hidden_states = cond)[0]
            u = dit(hidden_states = latents, timestep = t, encoder_hidden_states = uncond)[0]
            latents = latents - 0.1 * (u + 4.0 * (c - u))
        return latents

    latents = torch.randn(2, 4)
    cond, uncond = _embeds()
    dit = InPlaceDit()
    ref_cond, ref_uncond = cond.clone(), uncond.clone()
    reference = loop(dit, latents.clone(), ref_cond, ref_uncond)
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    c2, u2 = cond.clone(), uncond.clone()
    out = loop(dit, latents.clone(), c2, u2)
    torch.testing.assert_close(out, reference, rtol = 0, atol = 0)
    assert torch.equal(c2, ref_cond) and torch.equal(
        u2, ref_uncond
    )  # every per-branch write reached the caller
    assert "in place" in router.describe()["disabled_reason"]


def test_price_cfg_parallel_uses_both_measured_branches():
    # a cheap even branch and an expensive odd one: parallel runs the odd one alone, barely faster than both
    lopsided = mg.price_cfg_parallel(
        branch_s = 1.0,
        primary_s = 0.1,
        in_bytes = 1 << 20,
        out_bytes = 1 << 20,
        link_in_gib_s = 9.0,
        link_out_gib_s = 9.0,
    )
    assert lopsided["strategy"] == "single"
    # the recorded T4x2 numbers (2.38 s branches, 0.001 s copies) still pick CFG-parallel
    t4 = mg.price_cfg_parallel(
        branch_s = 2.376,
        primary_s = 2.38,
        in_bytes = 8 << 20,
        out_bytes = 1 << 20,
        link_in_gib_s = 9.14,
        link_out_gib_s = 9.13,
    )
    assert t4["strategy"] == "cfg_parallel"
    # the forced 0.02 GiB/s link with a short branch stays priced out
    slow = mg.price_cfg_parallel(
        branch_s = 0.08,
        primary_s = 0.08,
        in_bytes = 5 << 20,
        out_bytes = 1 << 20,
        link_in_gib_s = 0.02,
        link_out_gib_s = 0.02,
    )
    assert slow["strategy"] == "single"


def test_cfg_parallel_router_times_both_branches_of_the_first_guided_step(monkeypatch):
    dit = _Dit()
    router, replicas = _router(dit)
    times = iter([0.1, 1.0])  # even branch, odd branch
    monkeypatch.setattr(router, "_should_price", lambda: True)
    monkeypatch.setattr(
        router, "_timed_primary", lambda a, k: (router.orig_forward(*a, **k), next(times))
    )
    monkeypatch.setattr(mg, "link_gib_s", lambda src, dst, probes = None: 9.0)
    router.primary = router.secondary = torch.device("cpu")
    _cfg_loop(dit, torch.randn(2, 4), steps = 3)
    assert router.pricing["primary_branch_s"] == 0.1 and router.pricing["branch_s"] == 1.0
    assert router.pricing["strategy"] == "single" and replicas == []


class _RouterPipe:
    def __init__(self, dit):
        self.transformer = dit


def _distilled(dit, steps = 3):
    latents = torch.randn(2, 4)
    cond, _ = _embeds()
    for _ in range(steps):
        latents = latents - dit(hidden_states = latents, timestep = torch.tensor(0.5), encoder_hidden_states = cond)[0]


def test_request_scope_retimes_the_even_branch_after_a_distilled_request(monkeypatch):
    dit = _Dit()
    router, replicas = _router(dit)
    pipe = _RouterPipe(dit)
    times = []
    monkeypatch.setattr(router, "_should_price", lambda: True)
    monkeypatch.setattr(
        router, "_timed_primary", lambda a, k: (router.orig_forward(*a, **k), times.append(1) or 0.5)
    )
    monkeypatch.setattr(mg, "link_gib_s", lambda src, dst, probes = None: 9.0)
    router.primary = router.secondary = torch.device("cpu")
    with mg.cfg_request_scope(pipe, 1024, 1024):
        _distilled(dit)
    times.clear()
    with mg.cfg_request_scope(pipe, 1024, 1024):
        _cfg_loop(dit, torch.randn(2, 4), steps = 2)
    assert router.pricing is not None and router.pricing["primary_branch_s"] == 0.5
    assert len(times) == 2  # even and odd branch of the first guided step


def test_invalidation_reprices_a_priced_out_router(monkeypatch):
    dit = _Dit()
    router, replicas = _router(dit)
    pipe = _RouterPipe(dit)
    mg._CFG_ROUTERS[pipe] = router
    router.pricing = {"strategy": "single"}
    router.disabled_reason = "priced out: one card takes 1s"
    mg.invalidate_cfg_replica(pipe, "compiled")
    assert router.pricing is None and router.disabled_reason is None
    batch_dit = _BatchDit()
    batch = mg.CfgBatchRouter(batch_dit)
    batch_pipe = _RouterPipe(batch_dit)
    mg._CFG_BATCH[batch_pipe] = batch
    batch.pricing = {"strategy": "single"}
    batch.disabled_reason = "priced out: a call takes 1s"
    mg.invalidate_cfg_replica(batch_pipe, "compiled")
    assert batch.pricing is None and batch.disabled_reason is None


def test_cfg_batch_refuses_a_stateful_forward_and_restores_its_state():
    torch.manual_seed(0)

    class CountingDit(_BatchDit):
        def __init__(self):
            super().__init__()
            self.register_buffer("seen", torch.zeros(()))
            self.ticks = 0

        def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict = False):
            self.calls += 1
            self.seen.add_(1.0)
            self.ticks += 1
            return (self.proj(hidden_states) * timestep * self.seen + encoder_hidden_states.tanh(),)

    cond, uncond = _embeds()
    latents = torch.randn(2, 4)
    ref = CountingDit()
    reference = _batch_loop(ref, latents.clone(), 4, cond, uncond)
    dit = CountingDit()
    dit.load_state_dict(ref.state_dict())
    dit.seen.zero_()
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    out = _batch_loop(dit, latents.clone(), 4, cond, uncond)
    torch.testing.assert_close(out, reference, rtol = 0, atol = 0)
    assert float(dit.seen) == 8.0 and dit.ticks == 8  # once per branch, as two plain calls
    assert "state" in router.describe()["disabled_reason"]


def test_request_scope_stops_speculation_at_a_single_branch_request():
    torch.manual_seed(0)
    dit = _BatchDit()
    cond, uncond = _embeds()
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    pipe = _RouterPipe(dit)
    with mg.cfg_request_scope(pipe, 64, 64):
        _batch_loop(dit, torch.randn(2, 4), 3, cond, uncond)
    before = router.stats["batched_steps"]
    with mg.cfg_request_scope(pipe, 64, 64):
        _distilled(dit, steps = 4)
    assert router.stats["batched_steps"] == before  # not even the first step is speculated


def test_retained_cfg_inputs_are_released_at_decode():
    torch.manual_seed(0)
    dit = _BatchDit()
    cond, uncond = _embeds()
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    pipe = _RouterPipe(dit)
    with mg.cfg_request_scope(pipe, 64, 64):
        _batch_loop(dit, torch.randn(2, 4), 3, cond, uncond)
        assert router.template is not None
        mg.release_cfg_inputs(pipe)  # the decoder was entered
        assert router.template is None and router.b0_leaves == {} and router.kept is None
        assert router.prev_latent is None


def test_cfg_batch_headroom_scales_with_the_requested_canvas():
    dit = _BatchDit()
    router = mg.CfgBatchRouter(dit, headroom_mib = 892, headroom_reference_px = 1024 * 1024)
    dit.forward = router
    pipe = _RouterPipe(dit)
    with mg.cfg_request_scope(pipe, 2048, 2048):
        assert router.needed_mib() == 892 * 4
    with mg.cfg_request_scope(pipe, 512, 512):
        assert router.needed_mib() == 892 // 4
    assert router.needed_mib() == 892  # outside a request: the calibrated figure


def test_cfg_batch_catches_a_forward_that_turns_stateful_later():
    torch.manual_seed(0)

    class LateCounter(_BatchDit):
        def __init__(self):
            super().__init__()
            self.register_buffer("seen", torch.zeros(()))

        def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict = False):
            self.calls += 1
            if float(timestep) < 0.5:  # stateless for the early steps, then advances a buffer every call
                self.seen.add_(1.0)
            return (self.proj(hidden_states) * timestep * (1 + self.seen) + encoder_hidden_states.tanh(),)

    cond, uncond = _embeds()
    latents = torch.randn(2, 4)
    ref = LateCounter()
    reference = _batch_loop(ref, latents.clone(), 6, cond, uncond)
    dit = LateCounter()
    dit.load_state_dict(ref.state_dict())
    dit.seen.zero_()
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    out = _batch_loop(dit, latents.clone(), 6, cond, uncond)
    torch.testing.assert_close(out, reference, rtol = 0, atol = 0)
    assert float(dit.seen) == float(ref.seen)
    assert "state" in router.describe()["disabled_reason"]


def test_replica_refused_for_hooks_is_retried_once_they_are_gone():
    torch.manual_seed(0)
    dit = _Dit()
    router, replicas = _router(dit)
    source = dit.proj
    handle = dit.proj.register_forward_pre_hook(lambda m, a: source and None)  # a recording hook, e.g. bg compile
    _cfg_loop(dit, torch.randn(2, 4), steps = 2)
    assert replicas == [] and router.describe()["disabled_reason"] is None
    assert router.describe()["blocked_by"] is not None
    handle.remove()
    _cfg_loop(dit, torch.randn(2, 4), steps = 3)
    assert len(replicas) == 1 and router.stats["replica_calls"] > 0


def test_request_scope_finds_a_router_under_a_later_outer_layer():
    torch.manual_seed(0)
    dit = _BatchDit()
    pipe = _RouterPipe(dit)

    class Pipe:
        transformer = dit

    import os as _os

    _os.environ[mg.CFG_BATCH_ENV] = "force"
    try:
        assert mg.install_cfg_batch(pipe, offload_policy = "none") is None
    finally:
        _os.environ.pop(mg.CFG_BATCH_ENV, None)
    router = dit.__dict__["forward"]

    class Skip:  # the deferred static step skip takes the forward slot, keeping the router in .inner
        _unsloth_outer_forward = True

        def __init__(self, inner):
            self.inner = inner

        def __call__(self, *a, **k):
            return self.inner(*a, **k)

    dit.forward = Skip(router)
    router.template = ("x", {})
    mg.release_cfg_inputs(pipe)
    assert router.template is None
    with mg.cfg_request_scope(pipe, 2048, 2048):
        assert router.request_px == 2048 * 2048


def _listed_dit_class():
    class FluxTransformer2DModel(_BatchDit):
        pass

    FluxTransformer2DModel.__module__ = "diffusers.models.transformers.transformer_flux"
    FluxTransformer2DModel.__qualname__ = "FluxTransformer2DModel"
    return FluxTransformer2DModel


def test_cfg_batch_declines_a_denoiser_class_off_the_stateless_list(monkeypatch):
    class Pipe:
        def __init__(self, dit):
            self.transformer = dit

    reason = mg.cfg_batch_decline(Pipe(_BatchDit()), offload_policy = "group")
    assert reason is not None and "stateless-forward list" in reason and mg.CFG_BATCH_ENV in reason
    # a listed diffusers class passes the list (and is then judged on its offload as before)
    assert "resident" in mg.cfg_batch_decline(Pipe(_listed_dit_class()()), offload_policy = "group")

    # through a compile wrapper: the class behind _orig_mod decides
    class Compiled(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self._orig_mod = inner

    assert mg.cfg_batch_class_listed(Compiled(_listed_dit_class()()))[0]
    assert not mg.cfg_batch_class_listed(Compiled(_BatchDit()))[0]
    # the opt-in lets an unlisted class batch
    monkeypatch.setenv(mg.CFG_BATCH_ENV, "force")
    assert mg.cfg_batch_decline(Pipe(_BatchDit()), offload_policy = "none") is None


def test_cfg_batch_restores_state_when_the_batched_call_raises():
    torch.manual_seed(0)

    class RaisingDit(_BatchDit):
        def __init__(self):
            super().__init__()
            self.register_buffer("seen", torch.zeros(()))

        def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict = False):
            self.calls += 1
            self.seen.add_(1.0)  # advanced before the doubled batch is refused
            if hidden_states.shape[0] == 4:
                raise ValueError("fixed batch")
            return (self.proj(hidden_states) * timestep * self.seen + encoder_hidden_states.tanh(),)

    cond, uncond = _embeds()
    latents = torch.randn(2, 4)
    ref = RaisingDit()
    reference = _batch_loop(ref, latents.clone(), 4, cond, uncond)
    dit = RaisingDit()
    dit.load_state_dict(ref.state_dict())
    dit.seen.zero_()
    router = mg.CfgBatchRouter(dit)
    dit.forward = router
    out = _batch_loop(dit, latents.clone(), 4, cond, uncond)
    torch.testing.assert_close(out, reference, rtol = 0, atol = 0)
    assert float(dit.seen) == float(ref.seen)
    assert "ValueError" in router.describe()["disabled_reason"]


def test_request_scope_gets_the_effective_canvas_in_both_generate_paths():
    import ast
    from pathlib import Path

    root = Path(mg.__file__).parent

    def scope_args(name):
        tree = ast.parse((root / name).read_text())
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "cfg_request_scope"
        ]
        assert len(calls) == 1, name
        return [ast.unparse(a) for a in calls[0].args[1:]]

    # image: the size the forward runs at (img2img / inpaint take it from the input image), not the call kwargs
    assert scope_args("diffusion.py") == ["guard_width", "guard_height"]
    # video: the resolved request canvas (frames do not count)
    assert scope_args("video.py") == ["request_shape.get('width')", "request_shape.get('height')"]
