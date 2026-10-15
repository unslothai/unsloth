# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fault injection over the VAE OOM ladder: an OOM or a generic error at EVERY mutation point of the recovery and of
its restore, on a fake pipeline (a resident component, resident offload groups, CUDA graphs, a VAE with an
instance-bound ``_decode`` override and a later ``decode`` wrapper, a sliced batch decode). After each decode
(returned or raised) the system must be consistent: every module on its card or with a working deferred return that
resolves the card; resident groups all re-pinned or all streamed with a deferred re-pin; the wrapper stack exact after
uninstall; later slices on the recovered path; the surfaced error the decode's own OOM when nothing recovered.

CPU only: a float64 tensor stands in for one on the card, float32 for one on the host."""

import itertools
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import decode_layers as L  # noqa: E402
from core.inference import diffusion_cuda_graph as G  # noqa: E402
from core.inference import diffusion_memory as M  # noqa: E402
from core.inference import diffusion_vae_oom as O  # noqa: E402

CARD, HOST = torch.float64, torch.float32


class FakeOOM(RuntimeError):
    def __init__(self, where = ""):
        super().__init__(f"CUDA out of memory. Tried to allocate 2.00 GiB ({where})")


class Injected(ValueError):
    pass


class Device:
    type = "cuda"


DEV = Device()


class World:
    """The injector: counts mutation points; raises at the ``at``-th one (``kind`` "oom" or "error")."""

    def __init__(self, at = None, kind = "oom"):
        self.at, self.kind, self.n, self.points, self.fired = at, kind, 0, [], None

    def __call__(self, point):
        self.points.append(point)
        self.n += 1
        if self.at is not None and self.n == self.at:
            self.fired = point
            raise FakeOOM(point) if self.kind == "oom" else Injected(point)


def _move(module, dtype, world, tag):
    """Parameter by parameter, a mutation point before each: a fault leaves the module part moved."""
    for i, p in enumerate(list(module.parameters())):
        world(f"{tag}:param{i}")
        p.data = p.data.to(dtype)


class Group:
    def __init__(self):
        self._unsloth_resident = True


class Vae(torch.nn.Module):
    spatial_compression_ratio = 8

    def __init__(self, budget):
        super().__init__()
        self.use_tiling = False
        self.tile_sample_min_height = 256
        self.tile_sample_stride_height = 192
        self.a = torch.nn.Linear(64, 8, bias = False)
        self.b = torch.nn.Linear(64, 8, bias = False)
        self.budget = budget
        self.fake_hooks = False
        self.slices = []  # (slice, tiled, streamed) per successful slice decode

    def enable_tiling(self):
        self.use_tiling = True

    def _decode(self, z, return_dict = True):
        streamed = self.fake_hooks
        need = {False: 9, True: 5}[bool(self.use_tiling)] - (3 if streamed else 0)
        self.budget.world(f"decode:slice{self.budget.slice}:{'tiled' if self.use_tiling else 'untiled'}"
                          f"{':streamed' if streamed else ''}")  # fmt: skip
        if self.budget.used() + need > self.budget.cap:
            raise FakeOOM("decode")
        self.slices.append((self.budget.slice, bool(self.use_tiling), streamed))
        return (z + (0.5 if self.use_tiling else 0.0),)

    def decode(self, z, return_dict = True):
        out = []
        for i in range(z.shape[0]):  # diffusers with use_slicing: one _decode per sample
            self.budget.slice = i
            out.append(self._decode(z[i:i + 1])[0])
        return (torch.cat(out),)


class Pipe:
    pass


class Budget:
    """Units of card memory: the transformer 4, each resident group 1, the VAE weights 3, plus the decode's need."""

    def __init__(self, cap, world):
        self.cap, self.world, self.slice = cap, world, 0
        self.transformer = self.groups = self.vae = None

    def used(self):
        own = getattr(self.transformer, "_fake_groups", None)
        if own is not None:  # torchao transformer, hooked: its 4 units live in its two groups
            u = sum(2 for g in own if g._unsloth_resident)
        else:
            u = 4 if all(p.dtype == CARD for p in self.transformer.parameters()) else 0
        u += sum(1 for g in self.groups if g._unsloth_resident and g not in (own or ()))
        u += 0 if self.vae.fake_hooks else 3
        return u


def build(cap, world, monkeypatch, tiled = False, variant = "base"):
    budget = Budget(cap, world)
    transformer = torch.nn.Linear(8, 8, bias = False).to(CARD)
    text_encoder = torch.nn.Linear(8, 8, bias = False).to(CARD)
    text_encoder._diffusers_hook = types.SimpleNamespace(hooks = {"group_offloading": 1})
    text_encoder._unsloth_resident_room = 1
    groups = [Group() for _ in range(3)]
    vae = Vae(budget).to(CARD)
    vae.use_tiling = tiled
    budget.transformer, budget.groups, budget.vae = transformer, groups, vae
    pipe = Pipe()  # weak-referenceable, as a DiffusionPipeline is
    pipe.components = {"transformer": transformer, "text_encoder": text_encoder, "vae": vae}
    pipe.vae, pipe.transformer, pipe._unsloth_cuda_graphs = vae, transformer, ("graph",)

    def release(pipe_, need, logger = None, reason = ""):
        # diffusion_memory.release_resident_groups's contract: bookkeeping before each demotion; a failure part way
        # is caught and the restore for what was released is returned. Its re-pin swallows its own failures (as
        # _keep_groups_resident does); ``strict`` reports groups still streamed.
        released = []

        def restore(strict = False):
            try:
                for g in released:
                    world("groups:repin")
                    g._unsloth_resident = True
            except Exception:  # noqa: BLE001
                pass
            if strict and any(not g._unsloth_resident for g in released):
                raise RuntimeError("re-pinning resident offload groups left some streamed")

        try:
            for g in reversed(groups):
                if g._unsloth_resident:
                    released.append(g)
                    world("groups:release")
                    g._unsloth_resident = False
        except Exception:  # noqa: BLE001
            return restore if released else None
        return restore if released else None

    def apply_offload(v, device):
        world("vae_hooks:add")
        v.fake_hooks = True
        _move(v, HOST, world, "vae_out")

    def remove_offload(v, device):
        world("vae_hooks:remove")
        v.fake_hooks = False
        _move(v, CARD, world, "vae_back")

    def park(v, device):
        v.fake_hooks = False
        _move(v, HOST, world, "vae_park")
        world("vae_hooks:readd")
        v.fake_hooks = True

    def hook_resident(pipe_, device, logger = None):
        # diffusion_memory.hook_resident_denoiser: block offload hooks, every group resident; on failure it undoes
        world("torchao:hook")
        transformer._fake_groups = [Group(), Group()]
        groups.extend(transformer._fake_groups)
        transformer._unsloth_resident_room = 1
        transformer._diffusers_hook = types.SimpleNamespace(hooks = {"group_offloading": 1})
        return True

    monkeypatch.setattr(M, "release_resident_groups", release)
    monkeypatch.setattr(M, "hook_resident_denoiser", hook_resident)
    monkeypatch.setattr(O, "_holds_torchao", lambda m: variant == "torchao" and m is transformer)
    monkeypatch.setattr(G, "reset_all", lambda handles: world("graphs:reset"))
    monkeypatch.setattr(O, "_to_host", lambda m: _move(m, HOST, world, "out"))
    monkeypatch.setattr(O, "_return_to", lambda m, d: _move(m, CARD, world, "back"))
    monkeypatch.setattr(O, "_apply_vae_offload", apply_offload)
    monkeypatch.setattr(O, "_remove_vae_offload", remove_offload)
    monkeypatch.setattr(O, "_park_vae", park)
    monkeypatch.setattr(O, "_whole_on", lambda m, d: all(p.dtype == CARD for p in m.parameters()))
    monkeypatch.setattr(O, "_hooked", lambda m: bool(getattr(m, "fake_hooks", False)) or (
        getattr(m, "_diffusers_hook", None) is not None))  # fmt: skip
    monkeypatch.setattr(O, "_host_free_mib", lambda: 1 if variant == "host_short" else 1 << 20)
    monkeypatch.setattr(O, "_empty_cache", lambda d: None)
    monkeypatch.setattr(O, "_device_of", lambda z: DEV)
    monkeypatch.setattr(O, "_release_cache", lambda z, vae = None: None)

    override = types.MethodType(lambda self, z, *a, **k: Vae._decode(self, z, *a, **k), vae)
    vae._decode = override
    before = {k: vae.__dict__.get(k, "MISSING") for k in ("_decode", "decode")}
    assert O.install(vae, pipe = pipe)
    scope = vae.__dict__["decode"]

    def later(*a, **k):  # a later wrapper (untiled gate / ROCm bf16) capturing the scope layer
        return scope(*a, **k)

    later.__wrapped__ = scope
    vae.decode = later
    return pipe, budget, before, override, later


STEPS = {(False, False): 0, (True, False): 1, (True, True): 2}


def check(pipe, budget, before, override, later, world, outcome):
    vae, transformer, text_encoder = pipe.vae, pipe.components["transformer"], pipe.components["text_encoder"]
    ctx = f"fault {world.kind} at {world.fired!r}, cap {budget.cap}, outcome {outcome!r}"
    # surfaced error: the decode's own OOM when nothing recovered; an injected error only from a decode attempt
    if isinstance(outcome, BaseException):
        if isinstance(outcome, Injected):
            assert world.fired and world.fired.startswith("decode:"), ctx
        else:
            assert isinstance(outcome, FakeOOM), ctx
    # later slices stay on the recovered path (never back to a cheaper mode than an earlier slice used)
    modes = [STEPS[(t, s)] for _, t, s in vae.slices]
    assert modes == sorted(modes), (ctx, vae.slices)
    world.at = None  # memory frees up: everything deferred must now find its way back
    budget.cap = 10**6
    # the transformer: on the card, or a working deferred return resolving the card (a torchao one hooked by
    # step (a) is placed by its groups, checked below)
    deferred = transformer.__dict__.get(O._DEFERRED_ATTR)
    if getattr(transformer, "_fake_groups", None) is not None:
        pass
    elif deferred is None:
        assert all(p.dtype == CARD for p in transformer.parameters()), ctx
    else:
        assert deferred.execution_device is DEV and transformer.__dict__.get("_hf_hook") is deferred, ctx
        transformer(torch.zeros(1, 8, dtype = CARD))
        assert all(p.dtype == CARD for p in transformer.parameters()), ctx
        assert O._DEFERRED_ATTR not in transformer.__dict__ and not transformer._forward_pre_hooks, ctx
    # resident groups: all re-pinned, or all streamed (the flat plan) with a deferred re-pin
    resident = [g._unsloth_resident for g in budget.groups]
    if not all(resident):
        assert not any(resident), (ctx, resident)
        anchors = [m for m in (transformer, text_encoder) if O._DEFERRED_ATTR in m.__dict__]
        assert anchors, ctx
        for m in anchors:
            m(torch.zeros(1, 8, dtype = CARD))
        assert all(g._unsloth_resident for g in budget.groups), ctx
    # the VAE: on the card unhooked, or streamed whole from the host with a deferred return
    vdef = vae.__dict__.get(O._DEFERRED_ATTR)
    if vdef is None:
        assert not vae.fake_hooks and all(p.dtype == CARD for p in vae.parameters()), ctx
    else:
        assert vae.fake_hooks and all(p.dtype == HOST for p in vae.parameters()), ctx
        assert O.settle_deferred(vae), ctx
        assert not vae.fake_hooks and all(p.dtype == CARD for p in vae.parameters()), ctx
    assert O._SESSION_ATTR not in vae.__dict__, ctx
    # the wrapper stack: uninstall leaves exactly what was there plus the later wrapper (a pass-through below it)
    O.uninstall(vae)
    assert vae.__dict__.get("_decode") is override and vae.__dict__.get("decode") is later, ctx
    assert L.LAYERS_ATTR not in vae.__dict__ and O._RECORD_ATTR not in vae.__dict__, ctx
    vae.slices.clear()
    assert vae.decode(torch.zeros(2, 4, 1, 2, 2))[0].shape[0] == 2 and len(vae.slices) == 2, ctx


def _run(cap, at, kind, monkeypatch, tiled = False, verify = True, variant = "base"):
    world = World(at, kind)
    with monkeypatch.context() as mp:
        pipe, budget, before, override, later = build(cap, world, mp, tiled, variant)
        mp.setattr(O, "_FAULT", world)
        try:
            out = pipe.vae.decode(torch.zeros(3, 4, 1, 2, 2))
            outcome = out[0].shape
        except (FakeOOM, Injected) as exc:
            outcome = exc
        world.decode_points = world.n
        if verify:
            check(pipe, budget, before, override, later, world, outcome)
    return world


# caps: fits; (a) recovers; (b) recovers; (c) recovers; nothing recovers
CAPS = (20, 13, 8, 2, 1)
# a plain resident transformer; a hookless resident torchao one; host RAM too short to take anything
VARIANTS = ("base", "torchao", "host_short")


def _cases():
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    cases = []
    try:
        for variant, cap, tiled in itertools.product(VARIANTS, CAPS, (False, True)):
            n = _run(cap, None, "oom", mp, tiled, verify = False, variant = variant).decode_points
            cases += [
                (variant, cap, tiled, at, kind)
                for at, kind in itertools.product(range(1, n + 1), ("oom", "error"))
            ]
    finally:
        mp.undo()
    return cases


CASES = _cases()


def test_fault_injection_covers_every_mutation_point():
    assert len(CASES) >= 900, len(CASES)
    kinds = {p.split(":")[0] for v, cap, tiled in itertools.product(VARIANTS, CAPS, (False, True))
             for p in _points(cap, tiled, v)}  # fmt: skip
    for point in ("graphs", "groups", "out", "back", "vae_hooks", "vae_out", "vae_back", "decode",
                  "graphs_reset", "resident_groups", "moved", "vae_offload", "torchao", "torchao_hooks"):  # fmt: skip
        assert point in kinds, (point, kinds)


def _points(cap, tiled, variant = "base"):
    mp = pytest.MonkeyPatch()
    try:
        world = _run(cap, None, "oom", mp, tiled, verify = False, variant = variant)
    finally:
        mp.undo()
    return world.points[: world.decode_points]


@pytest.mark.parametrize("variant,cap,tiled,at,kind", CASES)
def test_fault_injected_recovery_stays_consistent(variant, cap, tiled, at, kind, monkeypatch):
    _run(cap, at, kind, monkeypatch, tiled, variant = variant)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("cap", CAPS)
def test_unfaulted_ladder_reaches_its_step(cap, variant, monkeypatch):
    world = World()
    with monkeypatch.context() as mp:
        pipe, budget, *_ = build(cap, world, mp, variant = variant)
        try:
            pipe.vae.decode(torch.zeros(3, 4, 1, 2, 2))
            raised = None
        except FakeOOM as exc:
            raised = exc
        stats = pipe.vae._unsloth_vae_oom_stats
        expect = {20: None, 13: "freed", 8: "tiled", 2: "offload", 1: None}[cap]
        if variant == "host_short":  # nothing can go to the host: no transformer move, no streamed VAE
            expect = {20: None, 13: "tiled", 8: None, 2: None, 1: None}[cap]
            if cap in (8, 2):
                assert raised is not None
                return
        assert stats.get("recovered_by") == expect, stats
        assert (raised is not None) == (cap == 1)
        if cap != 20:
            assert len({(t, s) for _, t, s in pipe.vae.slices}) <= 1  # one path for the whole batch
