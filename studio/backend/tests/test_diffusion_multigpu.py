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


def _from_diffusers(module: str, name: str):
    # Not importorskip: importing diffusers can raise RuntimeError (bitsandbytes without CUDA), an environment fact.
    # Only the import is skippable; a symbol missing from an importable diffusers is a failure.
    import importlib
    try:
        loaded = importlib.import_module(module)
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"diffusers is not importable here: {type(exc).__name__}")
    return getattr(loaded, name)


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(mg.MULTIGPU_ENV, raising = False)
    monkeypatch.delenv(mg.CFG_PARALLEL_ENV, raising = False)
    # Keep make_selection's real-host filters from collapsing fake [0, 1] selections on a one-GPU host.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(mg, "_capability", lambda o: None)
    monkeypatch.setattr(mg, "_rocm_torch", lambda: False)
    yield


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
    assert seen["other"] is None


def test_resolve_selected_cuda_ordinals_translates_the_mask(monkeypatch):
    import utils.hardware.hardware as hw
    from core.inference import diffusion_device as dd

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
        dict(secondary_limit_mib = lambda o: None),
        dict(single_card_policy = "none"),
        dict(companion_dense_mib = None),
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
    is_cuda_torch_device: bool = True


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
        balanced = _plan(requested_mode = "balanced")
        legacy = _plan(explicit_offload = True)
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
    assert mg.refine_plan_for_multigpu(object(), plan) is plan
    with mg.selection_scope(mg.make_selection(0, [0, 1])):
        refined = mg.refine_plan_for_multigpu(object(), plan)
    assert refined.offload_policy == dm.OFFLOAD_NONE
    assert refined.secondary_ordinal == 1
    assert refined.secondary_components == ("text_encoder", "vae")


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
    assert mg.move_to_device(t, torch.device("cpu")) is t


def test_model_output_moves_with_its_attributes():
    BaseOutput = _from_diffusers("diffusers.utils", "BaseOutput")

    @dataclasses.dataclass
    class Out(BaseOutput):
        sample: torch.Tensor = None

    out = mg.move_to_device(Out(sample = torch.ones(1)), torch.device("meta"))
    assert out.sample.device.type == "meta" and out[0].device.type == "meta"


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
    home = torch.device("meta")
    moves = []

    def fake_move(
        obj,
        device,
        _depth = 0,
    ):
        moves.append(device)
        return obj

    monkeypatch.setattr(mg, "_crosses", lambda device, h: True)
    monkeypatch.setattr(mg, "move_to_device", fake_move)
    wrapped = mg.install_boundary(vae, home)
    assert set(wrapped) == {"forward", "decode"}
    vae.decode(torch.ones(1, 2))
    assert moves == [torch.device("cpu"), home]
    assert vae.seen[-1] == ("decode", torch.device("cpu"))
    vae(torch.ones(1, 2))
    assert moves[2:] == [torch.device("cpu"), home]
    assert mg.install_boundary(vae, home) == []
    mg.remove_boundary(vae)
    assert "decode" not in vae.__dict__ and "forward" not in vae.__dict__


def test_boundary_is_inert_on_the_same_card():
    vae = _Vae()
    mg.install_boundary(vae, torch.device("cpu"))
    x = torch.ones(1, 2)
    assert torch.equal(vae(x), vae.lin(x))


def test_execution_device_reports_the_primary_of_a_split_pipeline(monkeypatch):
    monkeypatch.setattr(mg, "_SPLIT_EVER_PLACED", True)
    DiffusionPipeline = _from_diffusers("diffusers", "DiffusionPipeline")

    class Pipe(DiffusionPipeline):
        def __init__(self, vae, transformer):
            super().__init__()
            self.register_modules(vae = vae, transformer = transformer)

    class Mod(torch.nn.Linear):
        @property
        def device(self):
            return self.weight.device

    vae = Mod(1, 1).to("meta")
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
    pipe = Pipe()
    policy, _tiling = dm.apply_memory_plan(pipe, plan, device = "cuda", placement_device = "cuda:0")
    assert policy == dm.OFFLOAD_MODEL and calls == [("offload", "cuda:0")]
    applied = dm.applied_plan(pipe, plan)
    assert applied.offload_policy == dm.OFFLOAD_MODEL and applied.secondary_ordinal is None

    placed = []
    monkeypatch.setattr(
        mg, "apply_split_plan", lambda pipe, plan, placement, logger = None: placed.append(1) or True
    )
    policy, _tiling = dm.apply_memory_plan(Pipe(), plan, device = "cuda", placement_device = "cuda:0")
    assert policy == dm.OFFLOAD_NONE and placed == [1]


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
    assert router.stats["steps_parallel"] == 4
    assert router.stats["replica_calls"] == 4 * (branches // 2)
    assert router.describe()["disabled_reason"] is None
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
    dit.proj.register_forward_hook(lambda m, i, o: o + source.bias)
    router, replicas = _router(dit)
    _cfg_loop(dit, torch.randn(2, 4), steps = 3)
    assert replicas == []
    assert "proj._forward_hooks" in router.describe()["disabled_reason"]


def test_portable_hooks_and_root_hooks_are_accepted():
    torch.manual_seed(0)
    dit = _Dit()
    scale = 0.5
    hook = lambda m, i, o: o * scale  # noqa: E731 - captures only a float: shared safely
    hook.__module__ = mg.__name__
    dit.proj.register_forward_hook(hook)
    calls = []
    root_hook = lambda m, a: calls.append(1)  # noqa: E731 - root: runs on the source around every branch
    root_hook.__module__ = mg.__name__
    dit.register_forward_pre_hook(root_hook)
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
    assert mg.placement_status(other) is None


def test_replica_is_rebuilt_when_the_weights_change():
    torch.manual_seed(0)
    dit = _Dit()
    latents = torch.randn(2, 4)
    router, replicas = _router(dit)
    _cfg_loop(dit, latents.clone(), steps = 3)
    assert len(replicas) == 1
    with torch.no_grad():
        dit.proj.weight.mul_(0.5)
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
    apply_group_offloading = _from_diffusers("diffusers.hooks", "apply_group_offloading")

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

    def prefetch_begin(module, args):
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
    # the replica's first call (it may compile) runs unpredicted, after the primary branch
    assert router.stats["predicted_hits"] == 4 and router.stats["predicted_misses"] == 0
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
    assert router.stats["predicted_hits"] == 0 and router.stats["predicted_misses"] == 3
    assert router.stats["replica_calls"] == 4


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
            a = dit(hidden_states = x, timestep = t / 1000, encoder_hidden_states = cond)[0]
            b = dit(hidden_states = x, timestep = t / 1000, encoder_hidden_states = uncond)[0]
            x = x - 0.1 * (b + 4.0 * (a - b))
        return x

    reference = loop(6)
    router, _replicas = _router(dit)
    out = loop(6)
    assert torch.equal(out, reference)
    assert router.stats["predicted_hits"] == 4 and router.stats["predicted_misses"] == 0


def test_bitwise_equal_on_an_expanded_scalar():
    t = torch.tensor(941).expand(1)
    assert t.stride() == (0,)
    assert mg._bitwise_equal(t, t.clone())
    assert mg._bitwise_equal(t, torch.tensor(941).expand(1))
    assert not mg._bitwise_equal(t, torch.tensor([942]))
    assert mg._bitwise_equal(torch.tensor(float("nan")).expand(1), torch.tensor([float("nan")]))
    assert not mg._bitwise_equal(torch.tensor(-0.0).expand(1), torch.tensor([0.0]))
    assert mg._unchanged_since_launch([(t, t.clone())])
    assert not mg._unchanged_since_launch([(torch.tensor(942).expand(1), t.clone())])


def test_bitwise_equal_distinguishes_signed_zero():
    assert mg._bitwise_equal(torch.tensor([0.0]), torch.tensor([0.0]))
    assert mg._bitwise_equal(torch.tensor(0.5), torch.tensor(1.0) / 2)
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
    pipe = Pipe()
    policy, _tiling = dm.apply_memory_plan(pipe, plan, device = "cuda", placement_device = "cuda:0")
    assert policy == dm.OFFLOAD_GROUP and calls == [("group", "cuda:0")]
    assert dm.applied_plan(pipe, plan) is plan.fallback_plan


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
    dit = _Dit()
    counter = [0]

    def count(module, args):
        counter[0] += 1

    dit.proj.register_forward_pre_hook(count)
    assert "proj._forward_pre_hooks" in mg._hooks_present(dit)
    router, replicas = _router(dit)
    _cfg_loop(dit, torch.randn(2, 4), steps = 3)
    assert replicas == [] and "hooks" in router.describe()["disabled_reason"]


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

    with torch.inference_mode():
        reference = loop(5)
        router, _replicas = _router(dit)
        out = loop(5)
    assert torch.equal(out, reference)
    assert router.stats["predicted_hits"] == 0 and router.stats["predicted_misses"] == 3
    router.prefetched = {id(uncond): (uncond, uncond)}
    router.pending = {
        "leaves": [],
        "by_value": set(),
        "hook_state": None,
        "snapshots": [],
        "job": None,
    }
    assert router._take_prediction((), {"x": torch.zeros(1)}) is None
    assert router.prefetched == {}


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
    _cfg_loop(dit, torch.randn(2, 4), steps = 2)
    assert len(replicas) == 2
    assert router.replica is not None

    def broken_profile(state):
        raise RuntimeError("compile failed")

    with pytest.raises(RuntimeError, match = "compile failed"):
        DiffusionBackend._engage_deferred_speed(
            _types.SimpleNamespace(_engage_deferred_profile = broken_profile), state
        )
    assert router.replica is None


class _FlagLinear(torch.nn.Linear):
    def forward(self, x):
        out = super().forward(x)
        return -out if getattr(self, "_unsloth_flip", False) else out


class _FlagDit(_Dit):
    def __init__(self):
        super().__init__()
        self.proj = _FlagLinear(4, 4)


def _flag_pre_hook(module, args, kwargs):
    module.proj._unsloth_flip = bool(kwargs["encoder_hidden_states"].sum() < 0) != bool(
        kwargs["timestep"] < 0.5
    )


def _flag_post_hook(module, args, output):
    module.proj.__dict__.pop("_unsloth_flip", None)


_flag_pre_hook.__module__ = mg.__name__
_flag_post_hook.__module__ = mg.__name__


def _plain_state_pre_hook(module, args, kwargs):
    # a foreign root hook keeping per-call state in an ordinary attribute, which the replica would never see
    module.proj.flag = bool(kwargs["encoder_hidden_states"].sum() < 0)


def test_foreign_root_hook_declines_the_replica():
    dit = _FlagDit()
    dit.register_forward_pre_hook(_plain_state_pre_hook, with_kwargs = True)
    assert "a hook outside Unsloth" in mg._hooks_present(dit)
    router, replicas = _router(dit)
    _cfg_loop(dit, torch.randn(2, 4), steps = 3)
    assert replicas == [] and router.stats["replica_calls"] == 0


def test_root_hook_state_on_submodules_reaches_the_replica():
    torch.manual_seed(0)
    dit = _FlagDit()
    dit.register_forward_pre_hook(_flag_pre_hook, with_kwargs = True)
    dit.register_forward_hook(_flag_post_hook)
    latents = torch.randn(2, 4)
    cond = torch.ones(3, 4)
    neg = -torch.ones(3, 4)

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
    target = _types.SimpleNamespace(ordinal = 0, is_cuda_torch_device = True)
    monkeypatch.setattr(mg, "is_split", lambda p: True)
    monkeypatch.setattr(mg, "_module_device", lambda m: torch.device("cuda", 1))
    restore, tiled = D._guard_vae_card(
        pipe, target, {"device_memory": "card0-free", "vae_tile_side": 512}, None, None
    )
    assert ("scope", 1) in seen and ("guard", "card1-free", 512) in seen
    assert (restore, tiled) == ("restore", True)
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
    assert mg.make_selection(0, [0, 1, 2, 3]) == mg.MultiGpuSelection(primary = 0, secondaries = (2, 3))
    assert mg.make_selection(0, [0, 1]) is None
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
    assert not mg._holds_opaque_tensors(((torch.ones(1),), {"a": [torch.ones(1)], "b": None}))
    assert mg._holds_opaque_tensors({"kv_cache": _Cache()})


def test_split_video_request_is_guarded_on_every_card():
    from core.inference import video as vid

    class Pipe:
        def __init__(self):
            self.components = {
                "transformer": torch.nn.Linear(1024, 1024),
                "text_encoder": torch.nn.Linear(1024, 512),
                "vae": torch.nn.Linear(512, 512),
            }

    pipe = Pipe()
    mg._PLACEMENTS[pipe] = {
        "placement": {"transformer": "cuda:0", "text_encoder": "cuda:1", "vae": "cuda:1"},
        "moved": ["text_encoder", "vae"],
    }
    cards = vid._video_split_card_floors(pipe)
    assert cards == ((0, 4, False, True, True), (1, 3, True, False, False))
    big = dict(width = 1280, height = 720, frames = 121)
    clip = vid.estimate_video_runtime_mib(width = 1280, height = 720, num_frames = 121)
    roomy = {0: 1 << 20, 1: 1 << 20}
    assert (
        vid.video_split_shortfall_message(cards, family = "f", available_mib = roomy.get, **big) is None
    )
    # the decode card is priced for the decoded clip; the denoiser card for its weights, the base headroom and the
    # decoded frames copied back to it
    frames_mib = -(-1280 * 720 * 121 * 3 * 4 // (1 << 20))
    home_mib = max(frames_mib, vid._VIDEO_DENOISE_ACTIVATION_MIB)
    tight = {0: 4 + home_mib + dm.DEFAULT_BASE_OVERHEAD_MIB, 1: 3 + dm.DEFAULT_BASE_OVERHEAD_MIB}
    message = vid.video_split_shortfall_message(cards, family = "f", available_mib = tight.get, **big)
    assert clip > 0 and message is not None and "cuda:1" in message
    tight[1] = 1 << 20
    assert (
        vid.video_split_shortfall_message(cards, family = "f", available_mib = tight.get, **big) is None
    )
    tight[0] -= 1
    message = vid.video_split_shortfall_message(cards, family = "f", available_mib = tight.get, **big)
    assert "cuda:0" in message
    tight[0] = 4 + dm.DEFAULT_BASE_OVERHEAD_MIB
    assert vid.video_split_shortfall_message(cards, family = "f", available_mib = tight.get) is None
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
    target = _types.SimpleNamespace(ordinal = 0, is_cuda_torch_device = True)
    monkeypatch.setattr(mg, "is_split", lambda p: True)
    primary_verdict = _types.SimpleNamespace(action = D.ACTIVATION_TILE, overridden = False)
    kwargs = {"device_memory": "card0-free", "vae_tile_side": 512, "vae_sliced": False}
    monkeypatch.setattr(mg, "_module_device", lambda m: torch.device("cuda", 1))
    assert D._tile_for_verdict(pipe, primary_verdict, target, kwargs, 1, None, None) is None
    assert engaged == []
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
    assert mg._free_total_mib(1) == (1024, 80 * 1024)
    assert seen == [1]


_GLOBAL_HOOK_CALLS = []


def _global_state_hook(module, inputs, output):
    _GLOBAL_HOOK_CALLS.append(1)
    return output * (1.0 + len(_GLOBAL_HOOK_CALLS) % 2)


def test_foreign_submodule_hook_declines_the_replica():
    dit = _Dit()
    dit.proj.register_forward_hook(_global_state_hook)
    assert _global_state_hook.__closure__ is None
    assert mg._hooks_present(dit) == "proj._forward_hooks"
    router, replicas = _router(dit)
    _cfg_loop(dit, torch.randn(2, 4), steps = 3)
    assert replicas == [] and router.stats["replica_calls"] == 0
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
        assert mg.refine_plan_for_multigpu(object(), plan) is plan
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
    assert vid._video_single_card_for_sidecar(ltx25, split) is single
    assert vid._video_single_card_for_sidecar(wan, split) is split
    assert vid._video_single_card_for_sidecar(ltx25, single) is single
    refined = []
    monkeypatch.setattr(
        vid, "refine_plan_for_multigpu", lambda pipe, plan, logger = None: refined.append(1) or split
    )
    assert vid._video_refine_for_multigpu(ltx25, object(), single) is single
    assert vid._video_refine_for_multigpu(wan, object(), single) is split and refined == [1]


def _fake_cuda_host(
    monkeypatch,
    *,
    count,
    ranking_possible = True,
    current = 0,
):
    from core.inference import diffusion_device as dd

    monkeypatch.setattr(dd, "auto_cuda_ranking_possible", lambda: ranking_possible)
    monkeypatch.setattr(
        dd, "resolve_diffusion_device_target", lambda **kw: _Target(ordinal = kw.get("ordinal"))
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    monkeypatch.setattr(mg, "_capability", lambda o: (9, 0))
    from core.inference import gpu_residency

    monkeypatch.setattr(gpu_residency, "current_policy", lambda: "off")


def test_automatic_selection_uses_every_visible_card(monkeypatch):
    _fake_cuda_host(monkeypatch, count = 3)
    assert mg.automatic_selection(None) == mg.MultiGpuSelection(
        primary = 0, secondaries = (1, 2), automatic = True
    )
    assert mg.automatic_selection(2) == mg.MultiGpuSelection(
        primary = 2, secondaries = (0, 1), automatic = True
    )
    monkeypatch.setenv(mg.MULTIGPU_ENV, "0")
    assert mg.automatic_selection(None) is None


def test_automatic_selection_is_inert_on_ordinary_hosts(monkeypatch):
    # one GPU, ROCm, XPU, MPS and CPU all fail the ranking gate; a gated host never touches torch.cuda
    _fake_cuda_host(monkeypatch, count = 2, ranking_possible = False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: pytest.fail("probed a gated host"))
    assert mg.automatic_selection(None) is None
    _fake_cuda_host(monkeypatch, count = 1)
    assert mg.automatic_selection(None) is None


def test_with_load_selection_resolves_the_automatic_marker(monkeypatch):
    _fake_cuda_host(monkeypatch, count = 2)
    seen = {}

    class Backend:
        @mg.with_load_selection
        def _run_load(self, **kwargs):
            seen["kwargs"] = dict(kwargs)
            seen["selection"] = mg.current_selection()

    Backend()._run_load(gpu_ordinal = None, _gpu_ordinals = mg.AUTOMATIC_ORDINALS, repo_id = "x")
    assert seen["kwargs"] == {"gpu_ordinal": None, "repo_id": "x"}
    assert seen["selection"] == mg.MultiGpuSelection(primary = 0, secondaries = (1,), automatic = True)


def test_automatic_load_splits_only_when_one_card_would_offload(monkeypatch):
    monkeypatch.setattr(mg, "secondary_resident_limit_mib", lambda o, mode: 13600)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    selection = mg.MultiGpuSelection(primary = 0, secondaries = (1,), automatic = True)
    target = _Target(ordinal = None)
    single = _plan(target = target)
    with mg.selection_scope(selection):
        plan = _plan(target = target)
        fits = _plan(
            target = target,
            model_dense_mib = 4000,
            companion_dense_mib = 2000,
            text_encoder_dense_mib = 1500,
        )
    assert single.offload_policy != dm.OFFLOAD_NONE and single.secondary_ordinal is None
    assert plan.offload_policy == dm.OFFLOAD_NONE and plan.secondary_ordinal == 1
    assert fits.secondary_ordinal is None


def test_automatic_placement_does_not_arm_cfg_parallel(monkeypatch):
    class Pipe:
        transformer = _Dit()

    automatic = mg.MultiGpuSelection(primary = 0, secondaries = (1,), automatic = True)
    assert "automatic" in mg.cfg_parallel_decline(
        Pipe(), offload_policy = "none", selection = automatic
    )
    monkeypatch.setenv(mg.CFG_PARALLEL_ENV, "force")
    assert "CUDA" in mg.cfg_parallel_decline(Pipe(), offload_policy = "none", selection = automatic)


@pytest.mark.parametrize("module_file", ["diffusion.py", "video.py"])
def test_begin_load_marks_an_unpicked_load_automatic(module_file):
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "core" / "inference" / module_file
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    begin_loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "begin_load"
    ]
    assert len(begin_loads) == 1
    assigned = [
        node
        for node in ast.walk(begin_loads[0])
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "gpu_ordinals" for t in node.targets)
    ]
    assert len(assigned) == 1
    value = assigned[0].value
    assert isinstance(value, ast.IfExp) and isinstance(value.orelse, ast.IfExp)
    assert ast.unparse(value.orelse.body) == "AUTOMATIC_ORDINALS"
    assert ast.unparse(value.orelse.test) == "not gpu_ids"
    assert ast.unparse(value.orelse.orelse) == "[]"


def test_automatic_selection_leaves_module_compile_alone(monkeypatch):
    def original(self, *args, **kwargs):
        return self

    monkeypatch.setattr(torch.nn.Module, "compile", original)
    automatic = mg.MultiGpuSelection(primary = 0, secondaries = (1,), automatic = True)
    with mg.selection_scope(automatic):
        assert torch.nn.Module.compile is original
    with mg.selection_scope(mg.MultiGpuSelection(primary = 0, secondaries = (1,))):
        assert getattr(torch.nn.Module.compile, "_unsloth_recorder", False)


def test_automatic_secondaries_are_sized_without_a_cuda_context(monkeypatch):
    from core.inference import diffusion_device as dd

    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda *a: pytest.fail("opened a context on a secondary")
    )
    monkeypatch.setattr(
        mg, "probe_devices", lambda *a, **k: pytest.fail("probed every card through torch")
    )
    readings = {1: (9000, 15360), 2: (14000, 15360)}
    monkeypatch.setattr(dd, "smi_free_total_mib", lambda o: readings.get(o, (None, None)))
    automatic = mg.MultiGpuSelection(primary = 0, secondaries = (1, 2), automatic = True)
    with mg.selection_scope(automatic):
        assert mg.best_secondary(automatic) == 2
        memory = mg.secondary_device_memory(2)
        assert (memory.free_mib, memory.total_mib) == (14000, 15360)
        assert mg.secondary_device_memory(3) is None


def test_smi_reading_keys_ordinals_through_the_mask(monkeypatch):
    import utils.hardware.hardware as hw
    from core.inference import diffusion_device as dd

    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "_cuda_order_matches_smi", lambda: True)
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [4, 5])
    monkeypatch.setattr(hw, "_integrated_cuda_rows", lambda visible: ({}, ""))
    monkeypatch.setattr(hw, "_nvidia_mig_gpu_indices_or_none", lambda: set())
    monkeypatch.setattr(hw, "_get_parent_visible_gpu_spec", lambda: {"numeric_ids": [4, 5]})
    rows = [
        {"index": 4, "vram_total_gb": 24.0, "vram_used_gb": 20.0},
        {"index": 5, "vram_total_gb": 24.0, "vram_used_gb": 1.0},
    ]
    monkeypatch.setattr(hw, "_smi_visible_utilization", lambda spec: {"devices": rows})
    assert dd.smi_free_total_mib(1) == (23 * 1024, 24 * 1024)
    assert dd.smi_free_total_mib(2) == (None, None)
    monkeypatch.setattr(hw, "_nvidia_mig_gpu_indices_or_none", lambda: {5})
    assert dd.smi_free_total_mib(1) == (None, None)
    monkeypatch.setattr(hw, "_nvidia_mig_gpu_indices_or_none", lambda: set())
    monkeypatch.setattr(hw, "_cuda_order_matches_smi", lambda: False)
    assert dd.smi_free_total_mib(1) == (None, None)


def test_unsplit_automatic_load_has_no_other_vae_card(monkeypatch):
    import types as _types

    from core.inference import diffusion as D

    monkeypatch.setattr(mg, "_module_device", lambda m: torch.device("cuda", 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    pipe = _types.SimpleNamespace(vae = torch.nn.Linear(1, 1))
    automatic = _types.SimpleNamespace(ordinal = None, is_cuda_torch_device = True)
    assert D._vae_on_other_card(pipe, automatic) is None
    monkeypatch.setattr(mg, "is_split", lambda p: True)
    assert D._vae_on_other_card(pipe, automatic) is None
    monkeypatch.setattr(mg, "_module_device", lambda m: torch.device("cuda", 1))
    assert D._vae_on_other_card(pipe, automatic) == torch.device("cuda", 1)


def test_secondary_vae_guard_refuses_a_batch_slicing_did_not_engage(monkeypatch):
    import types as _types

    from core.inference import diffusion as D

    calls = []

    def shortfall(**kw):
        calls.append(kw.get("vae_sliced"))
        if kw.get("vae_sliced") is False:
            raise ValueError("does not fit unsliced")
        return _types.SimpleNamespace(action = D.ACTIVATION_TILE, overridden = False)

    restored = []
    monkeypatch.setattr(D, "diffusion_device_scope", lambda o: contextlib.nullcontext())
    monkeypatch.setattr(D, "reclaimable_snapshot_device_memory", lambda target: "card1-free")
    monkeypatch.setattr(D, "raise_on_image_activation_shortfall", shortfall)
    monkeypatch.setattr(
        D, "engage_vae_tiling", lambda pipe, logger = None: (lambda: restored.append(1), True)
    )
    monkeypatch.setattr(D, "vae_is_sliced", lambda vae: False)
    monkeypatch.setattr(mg, "is_split", lambda p: True)
    monkeypatch.setattr(mg, "_module_device", lambda m: torch.device("cuda", 1))
    pipe = _types.SimpleNamespace(vae = torch.nn.Linear(1, 1))
    target = _types.SimpleNamespace(ordinal = 0, is_cuda_torch_device = True)
    with pytest.raises(ValueError):
        D._guard_vae_card(pipe, target, {"batch_size": 2, "vae_sliced": True}, None, None)
    assert restored == [1] and calls == [True, False]


@pytest.mark.parametrize("vae_card", [0, 1])
def test_a_broken_recheck_keeps_the_tiling_it_engaged(monkeypatch, vae_card):
    """A re-check that raises something other than the refusal fails open, as generate() always did: the decode
    stays tiled (the guard said only a tiled decode fits) and the handle comes back for the caller to restore."""
    import types as _types

    from core.inference import diffusion as D

    def shortfall(**kw):
        if kw.get("vae_sliced") is False:
            raise RuntimeError("probe broke")
        return _types.SimpleNamespace(action = D.ACTIVATION_TILE, overridden = False)

    restored = []
    restore = lambda: restored.append(1)  # noqa: E731
    monkeypatch.setattr(D, "diffusion_device_scope", lambda o: contextlib.nullcontext())
    monkeypatch.setattr(D, "reclaimable_snapshot_device_memory", lambda target: "free")
    monkeypatch.setattr(D, "raise_on_image_activation_shortfall", shortfall)
    monkeypatch.setattr(D, "engage_vae_tiling", lambda pipe, logger = None: (restore, True))
    monkeypatch.setattr(D, "vae_is_sliced", lambda vae: False)
    monkeypatch.setattr(mg, "is_split", lambda p: True)
    monkeypatch.setattr(mg, "_module_device", lambda m: torch.device("cuda", vae_card))
    pipe = _types.SimpleNamespace(vae = torch.nn.Linear(1, 1))
    target = _types.SimpleNamespace(ordinal = 0, is_cuda_torch_device = True)
    kwargs = {"batch_size": 2, "vae_sliced": True, "vae_tile_side": 512, "device_memory": "free"}
    verdict = _types.SimpleNamespace(action = D.ACTIVATION_TILE, overridden = False)
    assert D._tile_for_verdict(pipe, verdict, target, kwargs, 2, None, None) is restore
    assert restored == []


def test_no_split_ever_placed_skips_the_component_walk(monkeypatch):
    reads = []

    class Pipe:
        @property
        def components(self):
            reads.append(1)
            return {}

    monkeypatch.setattr(mg, "_SPLIT_EVER_PLACED", False)
    assert mg._split_home(Pipe()) is None and reads == []
    monkeypatch.setattr(mg, "_SPLIT_EVER_PLACED", True)
    assert mg._split_home(Pipe()) is None and reads == [1]


def test_a_card_torch_cannot_address_is_not_a_secondary(monkeypatch):
    """nvidia-smi can list two cards while this process's torch addresses one: no second card, no selection."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(mg, "_capability", lambda o: (9, 0))
    assert mg.make_selection(0, [0, 1]) is None
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert mg.make_selection(0, [0, 1]) == mg.MultiGpuSelection(primary = 0, secondaries = (1,))


def test_two_discrete_rocm_cards_split_like_cuda(monkeypatch):
    """2x 7900 XTX / MI300: a HIP torch with two addressable discrete cards plans the same split CUDA does."""
    monkeypatch.setattr(torch.version, "hip", "7.13.0", raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(mg, "_capability", lambda o: (11, 0))  # gfx1100, as ROCm reports it
    monkeypatch.setattr(mg, "_rocm_torch", lambda: True)
    monkeypatch.setattr(mg, "_rocm_bf16", lambda o: True)
    monkeypatch.setattr(mg, "secondary_resident_limit_mib", lambda o, mode: 13600)
    selection = mg.make_selection(0, [0, 1])
    assert selection == mg.MultiGpuSelection(primary = 0, secondaries = (1,))
    with mg.selection_scope(selection):
        plan = _plan()
    assert plan.offload_policy == dm.OFFLOAD_NONE and plan.secondary_ordinal == 1


def test_refinement_keeps_an_explicit_cpu_offload(monkeypatch):
    sizes = {
        "transformer": (9000, "dit"),
        "text_encoder": (8000, "text_encoder"),
        "vae": (300, "other"),
    }
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: sizes)
    monkeypatch.setattr(mg, "secondary_resident_limit_mib", lambda o, mode: 12750)
    monkeypatch.setattr(mg, "best_secondary", lambda selection, probes = None: 1)
    legacy = _plan(explicit_offload = True)
    assert legacy.estimates.get("explicit_cpu_offload")
    moded = _plan(explicit_offload = True, requested_mode = "auto")
    assert not moded.estimates.get("explicit_cpu_offload")
    with mg.selection_scope(mg.make_selection(0, [0, 1])):
        assert mg.refine_plan_for_multigpu(object(), legacy) is legacy
        assert mg.refine_plan_for_multigpu(object(), moded).secondary_ordinal == 1


def test_render_end_releases_the_branch_tensors():
    torch.manual_seed(0)
    dit = _Dit()
    latents = torch.randn(2, 4)
    router, replicas = _router(dit)

    class Pipe:
        transformer = dit

    pipe = Pipe()
    mg._CFG_ROUTERS[pipe] = router
    _cfg_loop(dit, latents.clone(), steps = 3)
    assert router.template is not None and router.prev_latent is not None
    mg.end_cfg_render(pipe)
    assert router.template is None and router.prev_latent is None and router.pending is None
    assert router.cfg_seen is False
    assert router.prefetched == {} and router.carry == [] and router.b0_leaves == {}
    assert router.replica is not None
    reference_router = dit.__dict__.pop("forward")
    reference = _cfg_loop(dit, latents.clone(), steps = 3)
    dit.forward = reference_router
    assert torch.equal(_cfg_loop(dit, latents.clone(), steps = 3), reference)
    assert len(replicas) == 1
    mg.end_cfg_render(object())


def test_workflow_pipeline_of_a_split_still_guards_the_vae_card(monkeypatch):
    monkeypatch.setattr(mg, "_SPLIT_EVER_PLACED", True)
    import types as _types

    from core.inference import diffusion as D

    dit = torch.nn.Linear(1, 1)
    dit._unsloth_split_home = torch.device("cuda", 0)
    sibling = _types.SimpleNamespace(vae = torch.nn.Linear(1, 1), components = {"transformer": dit})
    target = _types.SimpleNamespace(ordinal = 0, is_cuda_torch_device = True)
    monkeypatch.setattr(mg, "_module_device", lambda m: torch.device("cuda", 1))
    assert not mg.is_split(sibling)
    assert D._vae_on_other_card(sibling, target) == torch.device("cuda", 1)


def test_cfg_replica_avoids_the_card_a_split_decodes_on(monkeypatch):
    class Pipe:
        transformer = _Dit()

    probes = {
        o: mg.DeviceProbe(ordinal = o, name = "B200", total_mib = 180000, free_mib = 170000)
        for o in (0, 1, 2)
    }
    monkeypatch.setattr(mg, "cfg_parallel_decline", lambda pipe, **kw: None)
    monkeypatch.setattr(mg, "_module_device", lambda m: torch.device("cuda", 0))
    monkeypatch.setattr(
        mg, "probe_devices", lambda ordinals, **kw: {o: probes[o] for o in ordinals}
    )
    pipe = Pipe()
    mg._PLACEMENTS[pipe] = {"placement": {"transformer": "cuda:0", "vae": "cuda:1"}}
    try:
        with mg.selection_scope(mg.MultiGpuSelection(primary = 0, secondaries = (1,))):
            reason = mg.install_cfg_parallel(pipe, offload_policy = "none")
        assert reason is not None and "holds the split's components" in reason
        te_only = Pipe()
        mg._PLACEMENTS[te_only] = {
            "placement": {"transformer": "cuda:0", "vae": "cuda:0", "text_encoder": "cuda:1"}
        }
        with mg.selection_scope(mg.MultiGpuSelection(primary = 0, secondaries = (1,))):
            assert "holds the split's components" in mg.install_cfg_parallel(
                te_only, offload_policy = "none"
            )
        other = Pipe()
        mg._PLACEMENTS[other] = {"placement": {"transformer": "cuda:0", "vae": "cuda:1"}}
        with mg.selection_scope(mg.MultiGpuSelection(primary = 0, secondaries = (1, 2))):
            assert mg.install_cfg_parallel(other, offload_policy = "none") is None
        assert str(mg._CFG_ROUTERS[other].secondary) == "cuda:2"
    finally:
        mg.uninstall_cfg_parallel(pipe)
        mg.uninstall_cfg_parallel(other)


def test_background_compile_declines_cfg_parallel(monkeypatch):
    class Pipe:
        transformer = _Dit()

    sel = mg.MultiGpuSelection(primary = 0, secondaries = (1,))
    monkeypatch.setenv("UNSLOTH_DIFFUSION_BG_COMPILE", "1")
    assert "background compile" in mg.cfg_parallel_decline(
        Pipe(), offload_policy = "none", selection = sel
    )
    monkeypatch.delenv("UNSLOTH_DIFFUSION_BG_COMPILE")
    assert "background compile" not in mg.cfg_parallel_decline(
        Pipe(), offload_policy = "none", selection = sel
    )


def test_a_displaced_router_is_reported_declined():
    dit = _Dit()
    router, _replicas = _router(dit)

    class Pipe:
        transformer = dit

    pipe = Pipe()
    mg._CFG_ROUTERS[pipe] = router
    assert mg.drop_displaced_cfg_router(pipe, "graphs") is False
    graphed = lambda *a, **k: None  # noqa: E731
    dit.__dict__["forward"] = graphed
    assert mg.drop_displaced_cfg_router(pipe, "graphs") is True
    assert dit.__dict__["forward"] is graphed
    assert mg.placement_status(pipe) == {"cfg_parallel": {"armed": False, "declined": "graphs"}}
    assert mg.drop_displaced_cfg_router(pipe, "graphs") is False


def test_kept_models_keep_every_media_load_on_one_card(monkeypatch):
    from core.inference import gpu_residency

    _fake_cuda_host(monkeypatch, count = 2)
    seen = {}

    class Backend:
        @mg.with_load_selection
        def _run_load(self, **kwargs):
            seen["selection"] = mg.current_selection()

    Backend()._run_load(gpu_ordinal = None, _gpu_ordinals = mg.AUTOMATIC_ORDINALS)
    assert seen["selection"] is not None
    Backend()._run_load(gpu_ordinal = 0, _gpu_ordinals = [0, 1])
    assert seen["selection"] is not None
    for mode in ("fit", "always"):
        monkeypatch.setattr(gpu_residency, "current_policy", lambda mode = mode: mode)
        Backend()._run_load(gpu_ordinal = None, _gpu_ordinals = mg.AUTOMATIC_ORDINALS)
        assert seen["selection"] is None
        Backend()._run_load(gpu_ordinal = 0, _gpu_ordinals = [0, 1])
        assert seen["selection"] is None


def test_status_models_keep_the_multi_gpu_report():
    from models.inference import DiffusionStatusResponse, VideoStatusResponse

    report = {"primary": "cuda:0", "secondary": "cuda:1", "moved": ["text_encoder"]}
    assert (
        DiffusionStatusResponse(loaded = True, multi_gpu = report).model_dump()["multi_gpu"] == report
    )
    assert VideoStatusResponse(loaded = True, multi_gpu = report).model_dump()["multi_gpu"] == report


@pytest.mark.parametrize("module_file", ["video.py"])
def test_video_split_fallback_recomputes_the_floor(module_file):
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "core" / "inference" / module_file).read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(source)
    hits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "secondary_ordinal" in ast.unparse(node.test)
        and "plan is not planned" in ast.unparse(node.test)
        and any(
            isinstance(b, ast.Assign) and getattr(b.targets[0], "id", None) == "vram_floor_mib"
            for b in node.body
        )
    ]
    assert len(hits) == 1


def test_a_split_refuses_a_controlnet_it_cannot_stream():
    # The resident fallback would put the whole ControlNet beside a denoiser one card could not hold with it.
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "core" / "inference" / "diffusion.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    tests = [
        ast.unparse(node.test)
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "_offload_controlnet_module" in ast.unparse(node.test)
    ]
    assert len(tests) == 1
    branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "_offload_controlnet_module" in ast.unparse(node.test)
    )
    refusal = branch.orelse[0]
    assert (
        isinstance(refusal, ast.If)
        and ast.unparse(refusal.test) == "calibrated or is_split(state.pipe)"
    )
    assert any(isinstance(n, ast.Raise) for n in ast.walk(refusal))


def test_replica_rebuild_counts_the_allocator_cache_as_free(monkeypatch):
    import types as _types

    dit = _Dit()
    router, _replicas = _router(dit)
    router.secondary = torch.device("cuda", 1)
    monkeypatch.setattr(mg, "_free_total_mib", lambda o: (100, 15360))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d = None: 4096 << 20)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d = None: 0)
    monkeypatch.setattr(mg, "_device_context", lambda d: contextlib.nullcontext())
    monkeypatch.setattr(
        mg, "_ReplicaWorker", lambda device: _types.SimpleNamespace(close = lambda: None)
    )
    router._build_replica()
    assert router.replica is not None and router.disabled_reason is None
    router.replica = None
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d = None: 0)
    router._build_replica()
    assert router.replica is None and "MiB free" in (router.disabled_reason or "")


def test_a_two_expert_video_split_prices_the_denoise_phase_on_both_cards():
    from core.inference import video as vid

    class Pipe:
        def __init__(self):
            self.components = {
                "transformer": torch.nn.Linear(1024, 1024),
                "transformer_2": torch.nn.Linear(1024, 1024),
                "vae": torch.nn.Linear(512, 512),
            }

    pipe = Pipe()
    mg._PLACEMENTS[pipe] = {
        "placement": {"transformer": "cuda:0", "vae": "cuda:0", "transformer_2": "cuda:1"},
        "moved": ["transformer_2"],
    }
    try:
        cards = vid._video_split_card_floors(pipe)
        assert cards == ((0, 5, True, True, True), (1, 4, False, False, True))
        shape = dict(width = 832, height = 480, frames = 25)
        roomy = {0: 1 << 20, 1: 1 << 20}
        assert (
            vid.video_split_shortfall_message(cards, family = "f", available_mib = roomy.get, **shape)
            is None
        )
        # the expert card has room for its weights but not the denoise activations
        expert_short = {0: 1 << 20, 1: 4 + dm.DEFAULT_BASE_OVERHEAD_MIB + 100}
        message = vid.video_split_shortfall_message(
            cards, family = "f", available_mib = expert_short.get, **shape
        )
        assert message is not None and "cuda:1" in message
        primary_short = {0: 5 + dm.DEFAULT_BASE_OVERHEAD_MIB + 100, 1: 1 << 20}
        message = vid.video_split_shortfall_message(
            cards, family = "f", available_mib = primary_short.get, **shape
        )
        assert message is not None and "cuda:0" in message
    finally:
        del mg._PLACEMENTS[pipe]


def test_untiled_decode_probes_the_card_the_vae_lives_on(monkeypatch):
    import types as _types

    from core.inference import video_vae_untiled as vu

    vae = torch.nn.Linear(1, 1)
    latent = _types.SimpleNamespace(device = torch.device("cuda", 0))
    monkeypatch.setattr(
        torch.nn.Linear,
        "parameters",
        lambda self, recurse = True: iter([_types.SimpleNamespace(device = torch.device("cuda", 1))]),
    )
    assert vu._decode_device(vae, latent) == torch.device("cuda", 1)
    # same card, or an offloaded VAE whose weights sit on the host: the latent's card, as before
    monkeypatch.setattr(
        torch.nn.Linear,
        "parameters",
        lambda self, recurse = True: iter([_types.SimpleNamespace(device = torch.device("cuda", 0))]),
    )
    assert vu._decode_device(vae, latent) == torch.device("cuda", 0)
    monkeypatch.setattr(
        torch.nn.Linear,
        "parameters",
        lambda self, recurse = True: iter([_types.SimpleNamespace(device = torch.device("cpu"))]),
    )
    assert vu._decode_device(vae, latent) == torch.device("cuda", 0)


def test_untiled_decode_sizes_the_card_the_vae_decodes_on(monkeypatch):
    import types as _types

    from core.inference import video_vae_untiled as vu

    family = next(iter({**vu._BYTES_PER_LATENT_PIXEL, **vu._BYTES_PER_FRAME_LATENT_PIXEL}))
    probed = []
    monkeypatch.setattr(vu, "_free_bytes", lambda device: probed.append(device))

    class Vae:
        use_tiling = True

        def decode(self, z, *args, **kwargs):
            return "tiled"

        def parameters(self):
            return iter([_types.SimpleNamespace(device = torch.device("cuda", 1))])

    pipe = _types.SimpleNamespace(vae = Vae())
    assert vu.install_untiled_decode(pipe, family)
    latent = _types.SimpleNamespace(device = torch.device("cuda", 0), shape = (1, 16, 3, 8, 8))
    assert pipe.vae.decode(latent) == "tiled"
    assert probed == [torch.device("cuda", 1)]


@pytest.mark.parametrize("module_name", ["diffusion", "video"])
def test_begin_load_hands_the_picked_cards_to_the_load_thread(monkeypatch, module_name):
    import importlib
    import types as _types

    backend_mod = importlib.import_module(f"core.inference.{module_name}")
    backend_cls = (
        backend_mod.DiffusionBackend if module_name == "diffusion" else backend_mod.VideoBackend
    )
    seen = {}

    def first_load_step(*args, **kwargs):
        seen["selection"] = mg.current_selection()
        raise RuntimeError("stop after the selection is read")

    class InlineThread:
        def __init__(
            self,
            target,
            kwargs,
            daemon = None,
        ):
            self.target, self.kwargs = target, kwargs

        def start(self):
            self.target(**self.kwargs)

    monkeypatch.setattr(backend_mod, "account_thread", InlineThread)
    monkeypatch.setattr(
        backend_mod,
        "resolve_diffusion_device_target",
        lambda *a, **k: _types.SimpleNamespace(device = "cuda"),
    )
    monkeypatch.setattr(backend_mod, "resolve_selected_cuda_ordinal", lambda ids: 0)
    monkeypatch.setattr(backend_mod, "resolve_selected_cuda_ordinals", lambda ids: [0, 1])
    monkeypatch.setattr(mg, "_capability", lambda ordinal: None)
    monkeypatch.setattr(mg, "models_kept_loaded", lambda: False)
    fam = _types.SimpleNamespace(name = "ltx-2", base_repo = "base/repo", single_file_is_pipeline = False)
    monkeypatch.setattr(backend_cls, "validate_load_request", lambda self, *a, **k: fam)
    if module_name == "diffusion":
        monkeypatch.setattr(backend_cls, "assert_precision_available", lambda self, *a, **k: None)
        monkeypatch.setattr(backend_mod, "detect_family_for_pick", first_load_step)
    else:
        from core.inference import video_minimax_h3

        monkeypatch.setattr(backend_mod, "assert_video_precision_available", lambda *a, **k: None)
        monkeypatch.setattr(backend_mod, "_ltx23_prequant_serves_on_card", lambda *a, **k: False)
        monkeypatch.setattr(video_minimax_h3, "is_h3_native", lambda *a, **k: False)
        monkeypatch.setattr(backend_mod, "_detect_load_family", first_load_step)

    backend_cls().begin_load("org/model", gpu_ids = [0, 1])
    assert seen["selection"] == mg.MultiGpuSelection(primary = 0, secondaries = (1,))
    # and only for that load
    assert mg.current_selection() is None

    # the kill switch: the extra cards are never even resolved (the single pick is, as before the multi-GPU layer)
    def never(ids):
        raise AssertionError("resolved several cards under UNSLOTH_DIFFUSION_MULTIGPU=0")

    monkeypatch.setenv(mg.MULTIGPU_ENV, "0")
    monkeypatch.setattr(backend_mod, "resolve_selected_cuda_ordinals", never)
    seen.clear()
    backend_cls().begin_load("org/model", gpu_ids = [0, 1])
    assert seen["selection"] is None


def test_fingerprint_sees_buffers_and_the_peft_adapter_selection():
    dit = _Dit()
    dit.proj.register_buffer("w_scale", torch.ones(4))
    base = mg.weights_fingerprint(dit)
    dit.proj.w_scale.mul_(2.0)
    edited = mg.weights_fingerprint(dit)
    assert edited != base
    dit.proj.w_scale = torch.ones(4)
    assert mg.weights_fingerprint(dit) not in (base, edited)
    before = mg.weights_fingerprint(dit)
    dit.proj._active_adapter = ["a"]
    with_a = mg.weights_fingerprint(dit)
    dit.proj._active_adapter = ["b"]
    assert len({before, with_a, mg.weights_fingerprint(dit)}) == 3


def test_replica_shares_the_nvfp4_step_controller():
    from core.inference.diffusion_nvfp4_protect import NVFP4StepController

    dit = _Dit()
    ctl = NVFP4StepController("0")
    dit.proj.protect = ctl
    replica = mg.build_replica(dit, torch.device("cpu"))
    assert replica.proj.protect is ctl


def test_boundary_and_router_removal_restore_an_instance_override():
    import types

    dit = _Dit()

    def own_forward(self, *args, **kwargs):
        return ("own",)

    override = types.MethodType(own_forward, dit)
    dit.forward = override
    mg.install_boundary(dit, "cpu")
    assert dit.__dict__["forward"] is not override
    mg.remove_boundary(dit)
    assert dit.__dict__["forward"] is override
    plain = _Dit()
    mg.install_boundary(plain, "cpu")
    mg.remove_boundary(plain)
    assert "forward" not in plain.__dict__

    class Pipe:
        pass

    pipe = Pipe()
    router = mg.CfgParallelRouter(dit, primary = "cpu", secondary = "cpu")
    dit.forward = router
    mg._CFG_ROUTERS[pipe] = router
    mg.uninstall_cfg_parallel(pipe)
    assert dit.__dict__["forward"] is override


def test_uninstalled_router_left_inside_a_wrapper_is_a_passthrough():
    torch.manual_seed(0)
    dit = _Dit()
    latents = torch.randn(2, 4)
    reference = _cfg_loop(dit, latents.clone(), steps = 4)
    router, replicas = _router(dit)

    class Skip:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, *args, **kwargs):
            return self.inner(*args, **kwargs)

    dit.forward = Skip(router)

    class Pipe:
        pass

    pipe = Pipe()
    mg._CFG_ROUTERS[pipe] = router
    assert mg.drop_displaced_cfg_router(pipe, "replaced")
    out = _cfg_loop(dit, latents.clone(), steps = 4)
    assert torch.equal(out, reference)
    assert replicas == [] and router.replica is None and router.prev_latent is None
    assert router.stats["replica_calls"] == 0


def test_router_stays_on_the_source_for_a_forced_eager_render_and_the_compile_thread():
    from core.inference import diffusion_bg_compile as bg

    torch.manual_seed(0)
    dit = _Dit()
    latents = torch.randn(2, 4)
    reference = _cfg_loop(dit, latents.clone(), steps = 4)
    router, replicas = _router(dit)
    for var in (bg._FORCE_EAGER, bg._NO_CAPTURE):
        token = var.set(True)
        try:
            out = _cfg_loop(dit, latents.clone(), steps = 4)
        finally:
            var.reset(token)
        assert torch.equal(out, reference)
        assert replicas == [] and router.stats["replica_calls"] == 0 and router.prev_latent is None
    out = _cfg_loop(dit, latents.clone(), steps = 4)
    assert torch.equal(out, reference) and router.stats["replica_calls"] == 3


def test_replica_blocks_get_their_own_compile_guard():
    from core.inference.diffusion_speed import guard_compiled_blocks

    mg.install_compile_recorder()
    dit = _Dit()
    dit.proj.compile(backend = "eager")  # lazy: nothing is traced here
    assert guard_compiled_blocks(dit) == 1
    source_guard = dit._unsloth_compile_guard
    replica = mg.build_replica(dit, torch.device("cpu"))
    wrapped = replica.proj._compiled_call_impl
    guard = getattr(wrapped, "_unsloth_compile_guard", None)
    assert guard is not None and guard is not source_guard
    assert replica.__dict__.get("_unsloth_compile_guard") is guard
