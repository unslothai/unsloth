# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Multi-GPU diffusion layer (core/inference/diffusion_multigpu.py), CPU only.

Cross-card behaviour is exercised with fake devices: the planner reads budgets through injected callables, placement
and the boundary through a patched card predicate, and CFG-parallel with a replica on the same (CPU) device, which
checks the routing, the prefetch / constant caches and every fallback without a second card."""

from __future__ import annotations

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
        secondary_budget_mib = lambda o: 15000,
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
        dict(secondary_budget_mib = lambda o: 12000),  # 12000 * 0.85 < 12000
        dict(secondary_budget_mib = lambda o: None),  # unreadable secondary
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
        secondary_budget_mib = lambda o: 16000,
    )
    split = _split(**kwargs)
    assert split is not None and split.components == mg.TEXT_ENCODERS_ONLY
    assert split.secondary_required_mib == 11000 + mg.TEXT_ENCODER_HEADROOM_MIB + 1000
    assert split.primary_required_mib == 5000 + 4000 + 1000
    # without the encoder share the planner cannot price it and keeps one card
    assert _split(**{**kwargs, "text_encoder_dense_mib": None}) is None


def test_force_splits_even_when_one_card_fits(monkeypatch):
    monkeypatch.setenv(mg.MULTIGPU_ENV, "force")
    assert _split(single_card_policy = "none") is not None


def test_split_picks_the_secondary_with_the_most_room():
    split = _split(
        selection = mg.MultiGpuSelection(primary = 0, secondaries = (1, 2)),
        secondary_budget_mib = lambda o: {1: 15000, 2: 40000}[o],
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
    monkeypatch.setattr(mg, "secondary_safe_budget_mib", lambda o: 16000)
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
    monkeypatch.setattr(mg, "secondary_safe_budget_mib", lambda o: 4000)
    single = _plan()
    with mg.selection_scope(mg.make_selection(0, [0, 1])):
        plan = _plan()
    assert plan == single


def test_planner_split_off_on_unified_memory(monkeypatch):
    monkeypatch.setattr(mg, "secondary_safe_budget_mib", lambda o: 16000)
    unified = dm.DeviceMemory(
        backend = "cuda", device = "cuda", memory_kind = "unified_memory", free_mib = 16000, total_mib = 16384
    )
    with mg.selection_scope(mg.make_selection(0, [0, 1])):
        plan = _plan(device_memory = unified)
    assert plan.secondary_ordinal is None


# ---------------------------------------------------------------------------------------------------- assignment


def test_assignment_prefers_moving_the_companions():
    sizes = {"transformer": (10000, "dit"), "text_encoder": (8000, "text_encoder"), "vae": (300, "other")}
    moved = mg.choose_component_assignment(
        sizes, headroom_mib = 2000, overhead_mib = 1000, primary_limit_mib = 13500, secondary_limit_mib = 13000
    )
    assert moved == ("text_encoder", "vae")


def test_assignment_can_move_only_the_text_encoders():
    sizes = {"transformer": (9000, "dit"), "text_encoder": (11000, "text_encoder"), "vae": (1400, "other")}
    moved = mg.choose_component_assignment(
        sizes, headroom_mib = 4000, overhead_mib = 1000, primary_limit_mib = 16000, secondary_limit_mib = 13500
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
    monkeypatch.setattr(torch, "device", lambda *a: real_device("cpu") if a and a[0] == "cuda" else real_device(*a))
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
        sizes, headroom_mib = 2000, overhead_mib = 1000, primary_limit_mib = 21000, secondary_limit_mib = 16000
    )
    assert moved == ("transformer_2",)
    # tighter primary: the second expert takes the companions along
    moved = mg.choose_component_assignment(
        sizes, headroom_mib = 2000, overhead_mib = 1000, primary_limit_mib = 15500, secondary_limit_mib = 21000
    )
    assert moved == ("transformer_2", "text_encoder", "vae")
    assert (
        mg.choose_component_assignment(
            sizes, headroom_mib = 2000, overhead_mib = 1000, primary_limit_mib = 9000, secondary_limit_mib = 9000
        )
        is None
    )


def test_refine_plan_for_multigpu(monkeypatch):
    plan = _plan()
    assert plan.offload_policy != dm.OFFLOAD_NONE
    sizes = {"transformer": (9000, "dit"), "text_encoder": (8000, "text_encoder"), "vae": (300, "other")}
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: sizes)
    monkeypatch.setattr(mg, "secondary_safe_budget_mib", lambda o: 15000)
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
    assert type(out["named"]) is Named and out["named"].a.device.type == "meta" and out["named"].b == "s"
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

    def decode(self, z, return_dict = False):
        self.seen.append(("decode", z.device))
        return (self.lin(z),)


def test_boundary_moves_inputs_in_and_outputs_home(monkeypatch):
    vae = _Vae()
    home = torch.device("meta")  # stands for the primary card
    moves = []

    def fake_move(obj, device, _depth = 0):
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
    assert "decode" not in pipe.vae.__dict__ and "_unsloth_split_home" not in pipe.transformer.__dict__
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
    monkeypatch.setattr(mg, "apply_split_plan", lambda pipe, plan, placement, logger = None: placed.append(1) or True)
    policy, _tiling = dm.apply_memory_plan(Pipe(), plan, device = "cuda", placement_device = "cuda:0")
    assert policy == dm.OFFLOAD_NONE and placed == [1]


# ---------------------------------------------------------------------------------------------------- CFG-parallel


class _Dit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.calls = 0

    def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict = False):
        self.calls += 1
        return (self.proj(hidden_states) * timestep + encoder_hidden_states.sum(),)


def _router(dit, **kw):
    replicas = []

    def factory():
        r = mg.build_replica(dit, torch.device("cpu"))
        replicas.append(r)
        return r

    router = mg.CfgParallelRouter(dit, primary = "cpu", secondary = "cpu", replica_factory = factory, **kw)
    dit.forward = router
    return router, replicas


def _cfg_loop(dit, latents, steps, branches = 2, scale = 4.0):
    cond = torch.randn(3, 4, generator = torch.Generator().manual_seed(1))
    uncond = torch.randn(3, 4, generator = torch.Generator().manual_seed(2))
    extra = torch.randn(3, 4, generator = torch.Generator().manual_seed(3))
    for i in range(steps):
        t = torch.tensor(1.0 - i / steps)
        outs = [dit(hidden_states = latents, timestep = t, encoder_hidden_states = e)[0] for e in (cond, uncond, extra)[:branches]]
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


def test_replica_declined_when_the_denoiser_carries_hooks():
    dit = _Dit()
    dit.proj.register_forward_hook(lambda m, i, o: o)
    router, replicas = _router(dit)
    _cfg_loop(dit, torch.randn(2, 4), steps = 3)
    assert replicas == []
    assert "hooks" in router.describe()["disabled_reason"]


def test_closure_forward_override_is_declined_but_bound_methods_are_rebound():
    import types

    dit = _Dit()
    dit.proj.forward = lambda x: x  # a closure would keep calling the source module
    assert "instance override" in mg._hooks_present(dit)
    dit2 = _Dit()
    dit2.proj.forward = types.MethodType(lambda self, x: torch.nn.functional.linear(x, self.weight), dit2.proj)
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
            seen = worker.call(lambda: (torch.is_inference_mode_enabled(), var.get(), threading.current_thread().name))
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
    assert mg.cfg_parallel_decline(Pipe(), offload_policy = "none", selection = None) == "one card selected"
    assert "offloaded" in mg.cfg_parallel_decline(Pipe(), offload_policy = "group", selection = sel)
    assert "CUDA" in mg.cfg_parallel_decline(Pipe(), offload_policy = "none", selection = sel)

    class Unet:
        transformer = None

    assert "U-Net" in mg.cfg_parallel_decline(Unet(), offload_policy = "none", selection = sel)
    monkeypatch.setenv(mg.CFG_PARALLEL_ENV, "0")
    assert mg.CFG_PARALLEL_ENV in mg.cfg_parallel_decline(Pipe(), offload_policy = "none", selection = sel)
