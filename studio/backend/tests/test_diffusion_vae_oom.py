# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""diffusion_vae_oom: the offload-first OOM ladder (CPU, fake OOMs and a fake card) and the structural fused
GroupNorm(+SiLU) match."""

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_vae_oom as O  # noqa: E402
from core.inference import diffusion_vae_fused as F  # noqa: E402
from core.inference import decode_layers as L  # noqa: E402


_REAL_RETURN_TO = O._return_to


class _FakeOOM(RuntimeError):
    def __init__(self):
        super().__init__("CUDA out of memory. Tried to allocate 2.00 GiB")


_CARD = torch.float64  # a float64 tensor stands in for one on the decode's card; float32 is on the host


class _Budget:
    """A fake card: what is resident there (pipeline modules, the VAE weights) plus the decode's need must fit."""

    def __init__(self, cap):
        self.cap = cap

    def resident(self, modules):
        return sum(
            int(t.numel()) * t.element_size()
            for m in modules
            for t in [*m.parameters(), *m.buffers()]
            if t.dtype == _CARD
        )


class _VaeLike(torch.nn.Module):
    """Stock-diffusers-shaped tiling attrs; a decode needs ``untiled_need`` bytes untiled, ``tiled_need`` tiled, on top
    of everything resident on the fake card (the VAE's own weights included)."""

    spatial_compression_ratio = 8

    def __init__(self, budget, pipe_modules, untiled_need, tiled_need, weight_bytes = 4096):
        super().__init__()
        self.use_tiling = False
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192
        self.conv = torch.nn.Linear(weight_bytes // 8, 1, bias = False)
        self.budget, self.pipe_modules = budget, pipe_modules
        self.untiled_need, self.tiled_need = untiled_need, tiled_need
        self.seen = []
        self.cleared = 0

    def enable_tiling(self):
        self.use_tiling = True

    def clear_cache(self):
        self.cleared += 1

    def _decode(self, z, return_dict = True):
        self.seen.append(
            (self.use_tiling, dict(O.tile_attrs(self)), self.conv.weight.dtype == _CARD)
        )
        need = self.tiled_need if self.use_tiling else self.untiled_need
        if self.budget.resident([*self.pipe_modules, self]) + need > self.budget.cap:
            raise _FakeOOM()
        # deterministic, and tiled differs from untiled so a test can tell which ran
        return (z.repeat_interleave(8, -1).repeat_interleave(8, -2) + (0.5 if self.use_tiling else 0.0),)


class _Pipe:
    def __init__(self, **components):
        self.components = components
        for k, v in components.items():
            setattr(self, k, v)


def _on_card(nbytes):
    m = torch.nn.Linear(nbytes // 8, 1, bias = False)
    return m.to(_CARD)


@pytest.fixture(autouse = True)
def _fake_card(monkeypatch):
    """The ladder only frees a CUDA card: let the meta device stand in for one, and host RAM be plentiful."""
    monkeypatch.setattr(O, "_device_of", lambda z: _FakeDevice())
    monkeypatch.setattr(O, "_host_free_mib", lambda: 1 << 20)
    monkeypatch.setattr(O, "_empty_cache", lambda device: None)
    yield


class _FakeDevice:
    type = "cuda"



def _whole_on(module, device):
    tensors = [*module.parameters(), *module.buffers()]
    return bool(tensors) and all(t.dtype == _CARD for t in tensors)


@pytest.fixture(autouse = True)
def _fake_moves(monkeypatch):
    monkeypatch.setattr(O, "_whole_on", _whole_on)
    monkeypatch.setattr(O, "_return_to", lambda module, device: module.to(_CARD))
    monkeypatch.setattr(O, "_to_host", lambda module: module.to(torch.float32))
    yield


def _setup(cap, untiled_need, tiled_need, other = 8192, tiled = False, can_free = True):
    budget = _Budget(cap)
    transformer = _on_card(other) if can_free else torch.nn.Linear(4, 1)
    text_encoder = _on_card(other // 2) if can_free else torch.nn.Linear(4, 1)
    vae = _VaeLike(budget, [transformer, text_encoder], untiled_need, tiled_need)
    vae.to(_CARD)
    vae.use_tiling = tiled
    pipe = _Pipe(transformer = transformer, text_encoder = text_encoder, vae = vae, tokenizer = object())
    assert O.install(vae, pipe = pipe)
    return vae, pipe


def _z():
    return torch.ones(1, 4, 1, 8, 8)


def _devices(pipe):
    return {n: next(m.parameters()).dtype for n, m in pipe.components.items() if isinstance(m, torch.nn.Module)}


def test_decode_that_fits_is_untouched():
    vae, pipe = _setup(cap = 1 << 20, untiled_need = 100, tiled_need = 10)
    before = _devices(pipe)
    out = vae._decode(_z())
    assert vae.seen == [(False, O.tile_attrs(vae), True)]
    assert torch.equal(out[0], _z().repeat_interleave(8, -1).repeat_interleave(8, -2))
    assert _devices(pipe) == before
    assert vae._unsloth_vae_oom_stats["oom"] == 0


def test_step_a_frees_the_pipeline_and_retries_the_same_untiled_decode():
    """Fail-before: untiled needs more than is free beside the transformer and text encoder; with them on the host it
    fits. The retry is the SAME untiled decode (bit-identical), not a tiled one, and both come back."""
    # resident: 8192 + 4096 + 4096 (vae) = 16384; untiled 10000 fits only once 12288 is freed
    vae, pipe = _setup(cap = 16384 + 1000, untiled_need = 10000, tiled_need = 500)
    before = _devices(pipe)
    out = vae._decode(_z())
    assert [s[0] for s in vae.seen] == [False, False]  # untiled stays untiled
    assert torch.equal(out[0], _z().repeat_interleave(8, -1).repeat_interleave(8, -2))
    stats = vae._unsloth_vae_oom_stats
    assert stats["recovered_by"] == "freed" and stats["tiled_retry"] == 0 and stats["offload_retry"] == 0
    assert _devices(pipe) == before  # restored to the card
    assert O._SESSION_ATTR not in vae.__dict__


def test_step_b_stock_tiles_when_freeing_is_not_enough():
    vae, pipe = _setup(cap = 16384 + 1000, untiled_need = 10**9, tiled_need = 2000)
    stock = O.tile_attrs(vae)
    out = vae._decode(_z())
    assert [s[0] for s in vae.seen] == [False, False, True]
    assert all(s[1] == stock for s in vae.seen)  # never below the stock tiles
    assert vae._unsloth_vae_oom_stats["recovered_by"] == "tiled"
    assert torch.equal(out[0], _z().repeat_interleave(8, -1).repeat_interleave(8, -2) + 0.5)
    assert vae.use_tiling is False and _devices(pipe)["transformer"] == _CARD


def test_step_c_offloads_the_vae_at_stock_tiles(monkeypatch):
    """Fail-before: even tiled, the VAE's own weights leave no room; streaming them from the host does."""
    offloads = []

    monkeypatch.setattr(O, "_apply_vae_offload", lambda vae, device: (offloads.append(1), vae.conv.to(torch.float32)))
    monkeypatch.setattr(O, "_undo_vae_offload", lambda vae, device: vae.conv.to(_CARD))
    vae, pipe = _setup(cap = 4096 + 100, untiled_need = 10**9, tiled_need = 3000, tiled = True)
    stock = O.tile_attrs(vae)
    vae._decode(_z())
    # tiled caller: (a) same tiled decode, no separate (b), then (c) tiled with the VAE off the card
    assert vae.seen == [(True, stock, True), (True, stock, True), (True, stock, False)]
    assert vae._unsloth_vae_oom_stats["recovered_by"] == "offload" and offloads == [1]
    assert vae.conv.weight.dtype == _CARD  # the VAE is back on the card
    assert _devices(pipe)["text_encoder"] == _CARD


def test_streamed_decode_that_breaks_fails_on_the_oom(monkeypatch):
    """A non-OOM error inside the streamed (step c) decode, e.g. an offload hook the environment cannot run, must not
    replace the out-of-memory the render already had; the VAE still comes back."""
    monkeypatch.setattr(O, "_apply_vae_offload", lambda vae, device: None)
    monkeypatch.setattr(O, "_undo_vae_offload", lambda vae, device: None)
    vae, pipe = _setup(cap = 1, untiled_need = 10**9, tiled_need = 10**9, tiled = True)
    real = L.find(vae, "_decode", O._RETRY_LAYER).inner

    def inner(z, *a, **k):
        holder = vae.__dict__.get(O._SESSION_ATTR)
        if holder is not None and holder["session"] is not None and holder["session"].vae_offloaded:
            raise NameError("send_to_device")
        return real(z, *a, **k)

    O.uninstall(vae)
    vae._decode = inner
    O.install(vae)
    with pytest.raises(RuntimeError, match = "even with the rest of the pipeline"):
        vae._decode(_z())


def test_every_step_fails_raises_and_restores(monkeypatch):
    monkeypatch.setattr(O.DecodeSession, "offload_vae", lambda self: False)
    vae, pipe = _setup(cap = 1, untiled_need = 10**9, tiled_need = 10**9)
    before = _devices(pipe)
    with pytest.raises(RuntimeError, match = "even with the rest of the pipeline"):
        vae._decode(_z())
    assert _devices(pipe) == before
    assert all(s[1] == O.tile_attrs(vae) for s in vae.seen)
    assert vae.use_tiling is False and vae.cleared >= 2  # the causal cache is cleared between attempts


def test_nothing_to_free_skips_the_identical_retry(monkeypatch):
    monkeypatch.setattr(O.DecodeSession, "offload_vae", lambda self: False)
    vae, pipe = _setup(cap = 4096 + 600, untiled_need = 10**9, tiled_need = 500, can_free = False)
    vae._decode(_z())
    assert [s[0] for s in vae.seen] == [False, True]
    assert vae._unsloth_vae_oom_stats["freed_retry"] == 0


def test_offload_hooked_and_shared_components_stay(monkeypatch):
    """Components an offload mechanism places (accelerate / group offload) are left to it; the VAE never moves in (a)."""
    vae, pipe = _setup(cap = 16384 + 1000, untiled_need = 10**9, tiled_need = 10**9)
    pipe.components["transformer"]._hf_hook = object()
    monkeypatch.setattr(O.DecodeSession, "offload_vae", lambda self: False)
    moved = []
    real = torch.nn.Module.to

    def spy(self, *a, **k):
        if a and a[0] == torch.float32:
            moved.append(self)
        return real(self, *a, **k)

    monkeypatch.setattr(torch.nn.Module, "to", spy)
    with pytest.raises(RuntimeError):
        vae._decode(_z())
    assert moved == [pipe.components["text_encoder"]]


def test_host_ram_short_keeps_component(monkeypatch):
    monkeypatch.setattr(O, "_host_free_mib", lambda: 1)
    session = O.DecodeSession(None, _FakeDevice())
    pipe = _Pipe(transformer = _on_card(1 << 20))
    session.free_others(pipe)
    assert session.moved == []


def test_restore_that_runs_out_of_memory_returns_on_next_forward(monkeypatch):
    m = _on_card(64)
    session = O.DecodeSession(None, _FakeDevice())
    session.log.append(session._component_change("transformer", m))
    m.to(torch.float32)
    calls = []

    def flaky(module, device):
        calls.append(1)
        module.to(torch.float16)  # a partial move
        if len(calls) <= 2:
            raise _FakeOOM()
        module.to(_CARD)

    monkeypatch.setattr(O, "_return_to", flaky)
    session.restore()
    assert session.log == []
    assert next(m.parameters()).dtype == torch.float32  # whole on the host, not half moved
    # Item 4282775926: the card is still full at the next forward: the OOM surfaces and the way back STAYS
    with pytest.raises(RuntimeError, match = "still on the host.*out of memory"):
        m(torch.zeros(1, 8, dtype = _CARD))
    assert next(m.parameters()).dtype == torch.float32 and len(m._forward_pre_hooks) == 1
    assert isinstance(m.__dict__.get("_hf_hook"), O.DeferredReturn)
    m(torch.zeros(1, 8, dtype = _CARD))  # memory freed: it returns and the hooks go
    assert next(m.parameters()).dtype == _CARD and len(m._forward_pre_hooks) == 0
    assert "_hf_hook" not in m.__dict__ and O._DEFERRED_ATTR not in m.__dict__


def _tiny_vae():
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    return diffusers.AutoencoderKL(
        block_out_channels = (8, 16), down_block_types = ("DownEncoderBlock2D",) * 2,
        up_block_types = ("UpDecoderBlock2D",) * 2, latent_channels = 4, norm_num_groups = 4, layers_per_block = 1,
    ).eval()  # fmt: skip


def test_vae_restore_oom_keeps_offload_hooks_and_retries(monkeypatch):
    """Item 4282775917: the VAE's weights cannot all return after a streamed decode. Before: hooks removed, weights
    stranded on the host. After: weights whole on the host WITH the offload hooks back (it still decodes, streamed),
    and its next decode brings it back once memory allows."""
    vae = _tiny_vae()
    z = torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        ref = vae.decode(z).sample
    cpu = torch.device("cpu")
    monkeypatch.setattr(O, "_to_host", lambda m: m.to("cpu", torch.float32))
    full = [True]

    def ret(module, device):
        if full[0]:
            module.decoder.conv_in.to(torch.float64)  # part of it moved, then the card is full
            raise _FakeOOM()
        _REAL_RETURN_TO(module, device)

    monkeypatch.setattr(O, "_return_to", ret)
    session = O.DecodeSession(vae, cpu)
    O._apply_vae_offload(vae, cpu)
    session.log.append(O.Change("vae_offload", lambda: O._remove_vae_offload(vae, cpu), lambda: O._park_vae(vae, cpu),
                                lambda retry, exc: O._defer(vae, cpu, "vae_offload", retry, False, vae)))
    session.restore()
    monkeypatch.undo()
    assert session.log == []
    assert O._hooked(vae), "the weights are on the host: the offload hooks must stay"
    assert all(p.dtype == torch.float32 for p in vae.parameters())
    with torch.no_grad():
        assert torch.equal(vae.decode(z).sample, ref)  # still decodes (streamed), same output
    assert O.settle_deferred(vae)  # memory back: it returns
    assert not O._hooked(vae) and O._DEFERRED_ATTR not in vae.__dict__ and "_hf_hook" not in vae.__dict__
    with torch.no_grad():
        assert torch.equal(vae.decode(z).sample, ref)


def test_execution_device_resolves_the_card_while_a_component_waits_on_the_host(monkeypatch):
    """A real diffusers pipeline whose first component could not return: ``_execution_device`` must still be the
    card, and the next call must run (the component comes back before its forward). The meta device stands in for
    the host, the CPU for the card."""
    diffusers = pytest.importorskip("diffusers")

    class TinyPipe(diffusers.DiffusionPipeline):
        def __init__(self, unet, vae):
            super().__init__()
            self.register_modules(unet = unet, vae = vae)

        def __call__(self):
            x = torch.ones(1, 4, 8, 8, device = self._execution_device)
            return self.vae.decode(self.unet(x, 1).sample).sample

    torch.manual_seed(0)
    unet = diffusers.UNet2DModel(
        sample_size = 8, in_channels = 4, out_channels = 4, block_out_channels = (8, 16), layers_per_block = 1,
        norm_num_groups = 4, down_block_types = ("DownBlock2D", "DownBlock2D"), up_block_types = ("UpBlock2D",) * 2,
    ).eval()  # fmt: skip
    pipe = TinyPipe(unet, _tiny_vae())
    with torch.no_grad():
        ref = pipe()
    saved = {k: v.clone() for k, v in unet.state_dict().items()}
    card = torch.device("cpu")
    monkeypatch.setattr(O, "_to_host", lambda m: m.to("meta"))
    oom = [True]

    def ret(module, device):
        if oom[0]:
            raise _FakeOOM()
        module.to_empty(device = device)
        module.load_state_dict(saved)

    monkeypatch.setattr(O, "_return_to", ret)
    session = O.DecodeSession(pipe.vae, card)
    unet.to("meta")
    session.log.append(session._component_change("unet", unet))
    session.restore()
    assert next(unet.parameters()).device.type == "meta"
    assert pipe._execution_device == card  # fail-before: meta
    oom[0] = False
    with torch.no_grad():
        out = pipe()
    assert torch.equal(out, ref) and next(unet.parameters()).device == card
    assert "_hf_hook" not in unet.__dict__ and len(unet._forward_pre_hooks) == 0


def test_uninstall_removes_compiled_guard_and_settles_deferred():
    """Item 4282775938: uninstall puts the compiled decode back (before: the guard stayed and kept retrying) and
    settles every deferred return the wrapper registered."""
    vae, pipe = _setup(cap = 1 << 20, untiled_need = 1, tiled_need = 1)
    O.uninstall(vae)

    def compiled(z, return_dict = True):
        return ("compiled",)

    compiled._unsloth_compiled_decode = True
    compiled._unsloth_eager_decode = lambda *a, **k: ("eager",)
    vae.decode = compiled
    assert O.install(vae, pipe = pipe)
    assert vae.__dict__["decode"] is not compiled and vae.__dict__["decode"]._unsloth_vae_oom_guard
    m = pipe.components["text_encoder"]
    O._defer(m, _FakeDevice(), "moved:text_encoder", lambda: None, True, vae)
    assert O._DEFERRED_ATTR in m.__dict__
    O.uninstall(vae)
    assert vae.__dict__["decode"] is compiled
    assert O._DEFERRED_ATTR not in m.__dict__ and "_hf_hook" not in m.__dict__
    assert O._RECORD_ATTR not in vae.__dict__ and "_decode" not in vae.__dict__
    assert O.install(vae, pipe = pipe) and vae.__dict__["decode"]._unsloth_vae_oom_guard  # reinstall wraps again


def test_uninstall_keeps_a_deferred_return_that_still_cannot_run():
    vae, pipe = _setup(cap = 1 << 20, untiled_need = 1, tiled_need = 1)
    m = pipe.components["text_encoder"]
    O._defer(m, _FakeDevice(), "moved:text_encoder", lambda: (_ for _ in ()).throw(_FakeOOM()), True, vae)
    O.uninstall(vae)
    assert O._DEFERRED_ATTR in m.__dict__  # its only way back to the card


def test_step_a_drops_graphs_and_repins_resident_groups(monkeypatch):
    from core.inference import diffusion_cuda_graph as G
    from core.inference import diffusion_memory as M

    events = []
    monkeypatch.setattr(G, "reset_all", lambda handles: events.append(("graphs", tuple(handles))))
    monkeypatch.setattr(
        M, "release_resident_groups",
        lambda pipe, need, logger = None, reason = "": (events.append(("release", need)), lambda: events.append(("repin",)))[1],
    )  # fmt: skip
    vae, pipe = _setup(cap = 16384 + 1000, untiled_need = 10000, tiled_need = 500)
    pipe._unsloth_cuda_graphs = ("h",)
    pipe.components["text_encoder"]._unsloth_resident_room = 1  # holds resident offload groups
    vae._decode(_z())
    assert events[0] == ("graphs", ("h",)) and events[1][0] == "release" and events[-1] == ("repin",)


def test_untiled_gate_keeps_its_own_fallback():
    vae, pipe = _setup(cap = 16384 + 1000, untiled_need = 10000, tiled_need = 500)

    def gate(*a, **k):
        return None

    gate._unsloth_untiled_decode = True

    def bf16_decode(*a, **k):  # the ROCm bf16 decode exposes the gate only through __wrapped__
        return gate(*a, **k)

    bf16_decode.__wrapped__ = gate
    for outer in (gate, bf16_decode):
        vae.seen.clear()
        vae.decode = outer
        # Item 4283572976: the ladder sees the gate's untiled OOM first: the card is freed and untiled retried
        vae._decode(_z())
        assert [s[0] for s in vae.seen] == [False, False]
        assert vae._unsloth_vae_oom_stats["recovered_by"] == "freed"
    # nothing to free (or still out of memory): the gate's own tiled fallback gets the original OOM
    vae, pipe = _setup(cap = 4096 + 600, untiled_need = 10**9, tiled_need = 500, can_free = False)
    vae.decode = gate
    with pytest.raises(_FakeOOM):
        vae._decode(_z())
    assert [s[0] for s in vae.seen] == [False]


def test_kill_switch(monkeypatch):
    vae, pipe = _setup(cap = 16384 + 1000, untiled_need = 10000, tiled_need = 500)
    monkeypatch.setenv(O.OOM_RETRY_ENV, "0")
    with pytest.raises(RuntimeError, match = "out of memory"):
        vae._decode(_z())  # installed, switched off at call time: the OOM surfaces as on main
    assert len(vae.seen) == 1
    other = _VaeLike(_Budget(1), [], 1, 1)
    assert O.install(other) is False and "_decode" not in other.__dict__


def test_non_oom_propagates():
    vae, pipe = _setup(cap = 1 << 20, untiled_need = 1, tiled_need = 1)
    O.uninstall(vae)
    vae._decode = types.MethodType(lambda self, z: (_ for _ in ()).throw(ValueError("x")), vae)
    O.install(vae)
    with pytest.raises(ValueError):
        vae._decode(_z())


def test_uninstall_restores_class_decode_and_reinstall_keeps_pipe():
    vae, pipe = _setup(cap = 1 << 20, untiled_need = 1, tiled_need = 1)
    assert "_decode" in vae.__dict__
    O.uninstall(vae)
    assert "_decode" not in vae.__dict__
    assert O.install(vae) and O._pipe_of(vae) is pipe


def test_no_halving_api_left():
    for name in ("halved", "MIN_TILE_LATENTS", "MIN_PSNR_DB", "oom_halving_active"):
        assert not hasattr(O, name)


def test_real_vae_offload_hooks_round_trip(monkeypatch):
    """Step (c)'s real mechanism on a tiny diffusers VAE (CPU onload stands in for the card): leaf-level hooks decode
    bit-identically and are fully removed afterwards, weights back where they were."""
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    vae = diffusers.AutoencoderKL(
        block_out_channels = (8, 16), down_block_types = ("DownEncoderBlock2D",) * 2,
        up_block_types = ("UpDecoderBlock2D",) * 2, latent_channels = 4, norm_num_groups = 4, layers_per_block = 1,
    ).eval()  # fmt: skip
    z = torch.randn(1, 4, 8, 8)
    monkeypatch.setattr(O, "_return_to", _REAL_RETURN_TO)
    with torch.no_grad():
        ref = vae.decode(z).sample
    cpu = torch.device("cpu")
    O._apply_vae_offload(vae, cpu)
    assert O._hooked(vae)
    with torch.no_grad():
        out = vae.decode(z).sample
    assert torch.equal(out, ref)
    O._undo_vae_offload(vae, cpu)
    assert not O._hooked(vae)
    with torch.no_grad():
        assert torch.equal(vae.decode(z).sample, ref)


def test_compiled_decode_oom_frees_then_retries_compiled():
    """Step (a) under a compiled decode retries the compiled callable itself (same kernels), not the eager path."""
    vae, pipe = _setup(cap = 16384 + 1000, untiled_need = 10000, tiled_need = 500)
    calls = []

    def eager(z, return_dict = True):
        calls.append("eager")
        return vae._decode(z, return_dict)

    def compiled(z, return_dict = True):
        calls.append("compiled")
        if vae.budget.resident([*vae.pipe_modules, vae]) + vae.untiled_need > vae.budget.cap:
            try:
                raise _FakeOOM()
            except _FakeOOM as oom:
                raise RuntimeError("compiled graph failed") from oom
        return ("ok",)

    compiled._unsloth_compiled_decode = True
    compiled._unsloth_eager_decode = eager
    vae.decode = compiled
    O.uninstall(vae)
    assert O.install(vae)
    before = _devices(pipe)
    assert vae.decode(_z()) == ("ok",)
    assert calls == ["compiled", "compiled"]
    assert vae._unsloth_vae_oom_stats["recovered_by"] == "freed_compiled" and _devices(pipe) == before


def test_compiled_decode_oom_falls_to_the_eager_ladder_sharing_the_session():
    vae, pipe = _setup(cap = 16384 + 1000, untiled_need = 10**9, tiled_need = 500)
    calls = []

    def eager(z, return_dict = True):
        calls.append("eager")
        assert O._SESSION_ATTR in vae.__dict__  # the eager ladder reuses the compiled guard's session
        return vae._decode(z, return_dict)

    def compiled(z, return_dict = True):
        calls.append("compiled")
        raise _FakeOOM()

    compiled._unsloth_compiled_decode = True
    compiled._unsloth_eager_decode = eager
    vae.decode = compiled
    O.uninstall(vae)
    assert O.install(vae)
    before = _devices(pipe)
    out = vae.decode(_z())
    assert calls == ["compiled", "compiled", "eager"]
    assert vae._unsloth_vae_oom_stats["recovered_by"] == "tiled" and out[0].shape[-1] == 64
    assert _devices(pipe) == before and O._SESSION_ATTR not in vae.__dict__


def test_compiled_guard_absent_without_compile():
    vae, pipe = _setup(cap = 1 << 20, untiled_need = 1, tiled_need = 1)
    assert "decode" not in vae.__dict__


def test_is_oom_walks_the_cause_chain():
    class OutOfMemoryError(RuntimeError):
        pass

    try:
        try:
            raise OutOfMemoryError("CUDA out of memory")
        except OutOfMemoryError as inner:
            raise RuntimeError("backend compiler failed") from inner
    except RuntimeError as exc:
        wrapped = exc
    assert O.is_oom(wrapped)
    assert not O.is_oom(RuntimeError("backend compiler failed"))


def test_speed_layer_compiled_guard_exposes_eager():
    from core.inference import diffusion_speed as S

    eager = lambda *a, **k: "eager"  # noqa: E731
    guarded = S._guard_compiled_decode(types.SimpleNamespace(), lambda *a, **k: "compiled", eager, None)
    assert guarded._unsloth_compiled_decode is True and guarded._unsloth_eager_decode is eager


# -- structural fused GroupNorm(+SiLU) --------------------------------------------------------------------------


class _Block(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.act = torch.nn.SiLU()
        self.norm1 = torch.nn.GroupNorm(4, c)
        self.conv = torch.nn.Conv2d(c, c, 3, padding = 1)
        self.norm2 = torch.nn.GroupNorm(4, c)

    def forward(self, x):
        h = self.act(self.norm1(x))  # paired
        h = self.conv(h)
        h2 = self.norm2(h)  # NOT paired: its output also feeds the residual
        return x + self.act(h2) + h2


class _Dec(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = _Block(c)
        self.norm_out = torch.nn.GroupNorm(4, c)
        self.out_act = torch.nn.SiLU()
        self.attn_norm = torch.nn.GroupNorm(4, c)

    def forward(self, x):
        x = self.block(x)
        x = x + self.attn_norm(x)  # GroupNorm without SiLU
        return self.out_act(self.norm_out(x))


class AutoencoderKLSomethingNew(torch.nn.Module):
    def __init__(self, c = 8):
        super().__init__()
        self.decoder = _Dec(c)

    def _decode(
        self,
        z,
        return_dict = True,
    ):
        return (self.decoder(z),)


def test_dataflow_trace_pairs_only_exclusive_group_norm_silu():
    vae = AutoencoderKLSomethingNew()
    norms = F._plain_group_norms(vae.decoder)
    silus = F._plain_silus(vae.decoder)
    calls, consumed, _, info = F._trace_group_norm_silu(
        norms, silus, lambda: vae._decode(torch.randn(1, 8, 6, 6))
    )
    names = {id(m): n for n, m in vae.decoder.named_modules()}
    got = {names[id(m)]: (c, k) for m, c, k in zip(norms, calls, consumed)}
    assert got["block.norm1"] == (1, 1)
    assert got["norm_out"] == (1, 1)
    assert got["attn_norm"] == (1, 0)
    # Item 4283572932: norm2's output reaches the SiLU AND the residual: not exclusively consumed (before: (1, 1),
    # fused with its SiLU, which changed what the residual read)
    assert got["block.norm2"] == (1, 0)
    shared = {names[id(m)]: k for m, k in zip(norms, info["shared"])}
    assert shared["block.norm2"] >= 1 and shared["block.norm1"] == 0 and shared["norm_out"] == 0


def _install_on_cpu(monkeypatch, vae):
    monkeypatch.setattr(F, "runtime_ok", lambda: True)
    monkeypatch.delenv(F.STRUCTURAL_ENV, raising = False)
    n = F.install_structural(vae)
    assert n == 4
    state = vae._unsloth_vae_structural
    # calibrate as on the first CUDA decode (CPU tensors take the reference GroupNorm, so it is exact here)
    z = torch.randn(1, 8, 6, 6)
    inner = L.find(vae, "_decode", F._STRUCTURAL_LAYER).inner
    with torch.no_grad():
        calls, consumed, ref, _ = F._trace_group_norm_silu(
            F._plain_group_norms(vae.decoder), F._plain_silus(vae.decoder), lambda: inner(z)[0]
        )
    return state, z, ref


def test_structural_patch_is_exact_when_pairing_is_right(monkeypatch):
    vae = AutoencoderKLSomethingNew()
    state, z, ref = _install_on_cpu(monkeypatch, vae)
    norms = F._plain_group_norms(vae.decoder)
    silus = F._plain_silus(vae.decoder)
    names = {id(m): n for n, m in vae.decoder.named_modules()}
    paired = [
        names[id(m)] in ("block.norm1", "norm_out") or (False if names[id(m)] else None)
        for m in norms
    ]
    F._structural_patch(norms, paired, silus, state, None)
    with torch.no_grad():
        out = L.find(vae, "_decode", F._STRUCTURAL_LAYER).inner(z)[0]
    assert torch.allclose(out, ref, atol = 1e-6)
    assert state.calls == 4
    F._structural_unpatch(norms, silus)
    assert not any("forward" in m.__dict__ for m in (*norms, *silus))


def test_structural_probe_rejects_a_wrong_pairing(monkeypatch):
    """norm2 looks paired to the trace but its output also feeds a residual: the probe check rejects the patch."""
    vae = AutoencoderKLSomethingNew()
    monkeypatch.setattr(F, "runtime_ok", lambda: True)
    F.install_structural(vae)
    state = vae._unsloth_vae_structural
    z = torch.randn(1, 8, 6, 6) * 3
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(F, "_gn_fusable", lambda x, norm: False)
    with torch.no_grad():
        vae._decode(z)
    assert state.status == "rejected"
    assert not any("forward" in m.__dict__ for m in vae.decoder.modules())


def test_structural_skips_measured_classes_and_kill_switch(monkeypatch):
    class AutoencoderKL(AutoencoderKLSomethingNew):
        pass

    assert F.structural_candidate(AutoencoderKL()) is False
    assert F.structural_candidate(AutoencoderKLSomethingNew()) is True
    monkeypatch.setenv(F.STRUCTURAL_ENV, "0")
    assert F.structural_candidate(AutoencoderKLSomethingNew()) is False
    monkeypatch.delenv(F.STRUCTURAL_ENV)
    vae = AutoencoderKLSomethingNew()
    vae._unsloth_decode_blocks_bypassed = True
    assert F.structural_candidate(vae) is False


def _cuda_triton() -> bool:
    return torch.cuda.is_available() and not getattr(torch.version, "hip", None) and F.runtime_ok()


needs_cuda = pytest.mark.skipif(not _cuda_triton(), reason = "needs NVIDIA CUDA + Triton")


def _psnr(a, b):
    mse = (a.float() - b.float()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(4.0 / mse)).item()


def _tiny(name):
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    if name == "AutoencoderKLHunyuanImage":
        return diffusers.AutoencoderKLHunyuanImage(
            in_channels = 3, out_channels = 3, latent_channels = 8, block_out_channels = (32, 64, 64),
            layers_per_block = 1, spatial_compression_ratio = 4, sample_size = 64,
        ), (1, 8, 12, 10)  # fmt: skip
    if name == "AutoencoderKLHunyuanVideo":
        return diffusers.AutoencoderKLHunyuanVideo(
            in_channels = 3, out_channels = 3, latent_channels = 4,
            down_block_types = ("HunyuanVideoDownBlock3D",) * 3, up_block_types = ("HunyuanVideoUpBlock3D",) * 3,
            block_out_channels = (32, 32, 32), layers_per_block = 1, norm_num_groups = 8,
            spatial_compression_ratio = 4, temporal_compression_ratio = 4,
        ), (1, 4, 2, 10, 12)  # fmt: skip
    raise AssertionError(name)


@needs_cuda
@pytest.mark.parametrize("name", ["AutoencoderKLHunyuanImage", "AutoencoderKLHunyuanVideo"])
def test_structural_engages_on_uncovered_class(name):
    vae, shape = _tiny(name)
    vae = vae.cuda().to(torch.bfloat16).eval()
    z = torch.randn(*shape, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        ref = vae.decode(z, return_dict = False)[0].float()
        assert F.will_install(vae) and F.install(vae) > 0
        out = vae.decode(z, return_dict = False)[0].float()
        state = vae._unsloth_vae_structural
        assert state.status == "ok", vars(state)
        assert state.gn_silu > 0 and state.calls > 0  # engaged on the real call
        assert _psnr(out, ref) > 35, _psnr(out, ref)
        F.uninstall(vae)
        assert "_decode" not in vae.__dict__
        assert torch.equal(vae.decode(z, return_dict = False)[0].float(), ref)



class AutoencoderKLTileableNew(AutoencoderKLSomethingNew):
    spatial_compression_ratio = 8

    def __init__(self, c = 8):
        super().__init__(c)
        self.use_tiling = False
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192
        self.decoder.proj3d = torch.nn.Conv3d(
            c, c, 3
        )  # a 3-D conv the channels_last_3d relayout touches

    def enable_tiling(self):
        self.use_tiling = True


def _layers(vae):
    fn, out = vae.__dict__.get("_decode"), []
    while fn is not None and len(out) < 8:
        out.append(fn)
        fn = getattr(fn, "__wrapped__", None)
    return out


def test_uninstall_unwraps_structural_layer_under_the_oom_wrapper(monkeypatch):
    """Item 4: with the OOM retry on top, uninstall must still remove the structural calibrator underneath (and keep
    the retry). Before the fix the calibrator stayed and its saved state was dropped."""
    monkeypatch.setattr(F, "runtime_ok", lambda: True)
    vae = AutoencoderKLTileableNew()
    assert F.install_structural(vae) > 0
    assert O.install(vae)
    assert any(getattr(f, "_unsloth_vae_fused", False) for f in _layers(vae))
    F.uninstall(vae)
    layers = _layers(vae)
    assert not any(getattr(f, "_unsloth_vae_fused", False) for f in layers), layers
    assert getattr(vae.__dict__.get("_decode"), "_unsloth_vae_oom_retry", False)
    assert L.find(vae, "_decode", F._STRUCTURAL_LAYER) is None
    with torch.no_grad():
        out = vae._decode(torch.randn(1, 8, 6, 6))
    assert out[0].shape == (1, 8, 6, 6)


def test_rejected_structural_fusion_restores_contiguous_layout(monkeypatch):
    """Item 5: the speed layer relays 3-D decode convs to channels_last_3d because fusion is planned; a rejected
    calibration must put them back. Before the fix the weights stayed channels_last_3d."""
    from core.inference import diffusion_speed as S

    monkeypatch.delenv(S.VAE_CHANNELS_LAST_3D_ENV, raising = False)
    monkeypatch.setattr(F, "runtime_ok", lambda: True)
    vae = AutoencoderKLTileableNew()
    w = vae.decoder.proj3d.weight
    assert S._vae_channels_last_3d(vae, None, fused = True)
    assert w.is_contiguous(memory_format = torch.channels_last_3d) and not w.is_contiguous()
    F.install_structural(vae)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(F, "_gn_fusable", lambda x, norm: False)
    with torch.no_grad():
        vae._decode(torch.randn(1, 8, 6, 6) * 3)
    assert vae._unsloth_vae_structural.status == "rejected"
    assert vae.decoder.proj3d.weight.is_contiguous()



def test_uninstall_restores_contiguous_layout(monkeypatch):
    """Round 3 item 2: uninstall undoes the channels_last_3d relayout recorded for the fused passes."""
    from core.inference import diffusion_speed as S

    monkeypatch.delenv(S.VAE_CHANNELS_LAST_3D_ENV, raising = False)
    monkeypatch.setattr(F, "runtime_ok", lambda: True)
    vae = AutoencoderKLTileableNew()
    assert S._vae_channels_last_3d(vae, None, fused = True)
    F.install_structural(vae)
    assert not vae.decoder.proj3d.weight.is_contiguous()
    F.uninstall(vae)
    assert vae.decoder.proj3d.weight.is_contiguous()
    assert "_unsloth_channels_last_modules" not in vae.__dict__




class _CleanDec(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm_out = torch.nn.GroupNorm(4, c)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        return self.act(self.norm_out(x))


class AutoencoderKLCleanNew(torch.nn.Module):
    def __init__(self, c = 8):
        super().__init__()
        self.decoder = _CleanDec(c)

    def _decode(
        self,
        z,
        return_dict = True,
    ):
        return (self.decoder(z),)


def test_structural_rejected_when_every_norm_falls_back(monkeypatch):
    """Round 5 item 3: a correct pairing whose norms all take group_norm_act's stock fallback fused nothing, so the
    calibration must reject it (before: counted as fused and accepted)."""
    monkeypatch.setattr(F, "runtime_ok", lambda: True)
    vae = AutoencoderKLCleanNew()
    F.install_structural(vae)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(F, "_gn_fusable", lambda x, norm: False)
    with torch.no_grad():
        vae._decode(torch.randn(1, 8, 6, 6))
    state = vae._unsloth_vae_structural
    assert state.status == "rejected" and state.kernel_calls == 0
    assert not any("forward" in m.__dict__ for m in vae.decoder.modules())


def test_kernel_calls_skip_the_temporal_stride_fallback(monkeypatch):
    """A 5-D norm that passes _gn_fusable but whose T axis cannot be flattened takes the stock reference inside
    group_norm_act; calibration must not count it as a kernel run."""
    monkeypatch.setattr(F, "_gn_fusable", lambda x, norm: True)
    norm = torch.nn.GroupNorm(4, 8)
    state = types.SimpleNamespace(kernel_calls = 0, calls = 0)
    fast = F._fast_gn_maybe_act(norm, True, state)
    x = (
        torch.randn(1, 8, 3, 4, 5).transpose(2, 3).contiguous().transpose(2, 3)
    )  # T stride != H * H stride
    assert x.stride(2) != x.shape[3] * x.stride(3)
    with torch.no_grad():
        y = fast(x)
    assert state.calls == 1 and state.kernel_calls == 0
    assert torch.allclose(y, torch.nn.functional.silu(norm(x)), atol = 1e-5)


# -- PR review round 7 regressions ------------------------------------------------------------------------------


class _CausalNew(AutoencoderKLCleanNew):
    """A causal decoder whose output depends on a persistent cache filled by earlier decodes."""

    def __init__(self, c = 8):
        super().__init__(c)
        self.cache = None

    def clear_cache(self):
        self.cache = None

    def _decode(self, z, return_dict = True):
        out = self.decoder(z)
        if self.cache is not None:
            out = out + self.cache.mean()  # stale frames leak into the output
        self.cache = z.detach()
        return (out,)


def test_structural_probe_does_not_leak_cache_into_the_real_decode(monkeypatch):
    """Round 7 item 1: the calibration's probe decodes must not leave cache entries the real decode reads."""
    monkeypatch.setattr(F, "runtime_ok", lambda: True)
    ref_vae = _CausalNew()
    torch.manual_seed(0)
    z = torch.randn(1, 8, 6, 6) + 2.0
    with torch.no_grad():
        ref = ref_vae._decode(z)[0]
    vae = _CausalNew()
    vae.load_state_dict(ref_vae.state_dict())
    F.install_structural(vae)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    with torch.no_grad():
        out = vae._decode(z)[0]
    assert torch.allclose(out, ref, atol = 1e-6), (out - ref).abs().max()




# -- PR review round 9 ------------------------------------------------------------------------------------------


def test_required_deferred_return_raises_on_any_failure():
    """Item 4283031561: a required return that fails with a non-OOM error raised nothing before, so the forward ran
    with its weights on the host. Now it raises a clear error, and the way back stays for a later retry."""
    m = torch.nn.Linear(4, 4)
    O._defer(m, _FakeDevice(), "moved:transformer", lambda: (_ for _ in ()).throw(ValueError("no such device")),
             True, None)  # fmt: skip
    with pytest.raises(RuntimeError, match = "transformer.*still on the host.*ValueError: no such device"):
        m(torch.zeros(1, 4))
    assert O._DEFERRED_ATTR in m.__dict__ and len(m._forward_pre_hooks) == 1
    # not required (the VAE streams through its hooks): it runs as it is and retries next time
    v = torch.nn.Linear(4, 4)
    O._defer(v, _FakeDevice(), "vae_offload", lambda: (_ for _ in ()).throw(ValueError("x")), False, None)
    v(torch.zeros(1, 4))
    assert O._DEFERRED_ATTR in v.__dict__


def _compiled_vae():
    vae, pipe = _setup(cap = 1 << 20, untiled_need = 1, tiled_need = 1)
    O.uninstall(vae)
    calls = []

    def compiled(z, return_dict = True):
        calls.append("compiled")
        raise _FakeOOM()

    compiled._unsloth_compiled_decode = True
    compiled._unsloth_eager_decode = lambda *a, **k: ("eager",)
    vae.decode = compiled
    assert O.install(vae, pipe = pipe)
    return vae, pipe, compiled, calls


def test_uninstall_neutralises_a_compiled_guard_under_a_later_wrapper():
    """Item 4283031573: video.py wraps vae.decode (untiled gate, ROCm bf16) after the guard, capturing it. Before:
    uninstall saw a different outermost decode, dropped its record, and the guard kept retrying. Now the buried guard
    is a pass-through: an OOM out of the compiled decode surfaces, with no freeing and no eager retry."""
    vae, pipe, compiled, calls = _compiled_vae()
    guard = vae.__dict__["decode"]

    def later(*a, **k):  # captures the guard, as the untiled gate does
        return guard(*a, **k)

    later.__wrapped__ = guard
    vae.decode = later
    O.uninstall(vae)
    assert vae.__dict__["decode"] is later and later.__wrapped__ is compiled
    with pytest.raises(_FakeOOM):
        vae.decode(_z())
    assert calls == ["compiled"] and _devices(pipe)["transformer"] == _CARD


def test_uninstall_restores_compiled_decode_when_outermost():
    vae, pipe, compiled, calls = _compiled_vae()
    O.uninstall(vae)
    assert vae.__dict__["decode"] is compiled


@pytest.mark.parametrize("which", ["oom", "fused"])
def test_uninstall_keeps_an_instance_bound_decode_override(monkeypatch, which):
    """Item 4283031579: a VAE whose ``_decode`` was already an instance-bound override (types.MethodType) got the
    class method back on uninstall before (the override was popped). Now exactly the override comes back."""
    if which == "fused":
        monkeypatch.setattr(F, "runtime_ok", lambda: True)
        monkeypatch.delenv(F.STRUCTURAL_ENV, raising = False)
        vae = AutoencoderKLSomethingNew()
    else:
        vae, pipe = _setup(cap = 1 << 20, untiled_need = 1, tiled_need = 1)
        O.uninstall(vae)

    def override(self, z, *a, **k):
        return type(self)._decode(self, z, *a, **k)

    bound = types.MethodType(override, vae)
    vae._decode = bound
    if which == "fused":
        assert F.install_structural(vae) > 0
        F.uninstall(vae)
    else:
        assert O.install(vae)
        O.uninstall(vae)
    assert vae.__dict__.get("_decode") is bound


def test_uninstall_restores_the_class_decode_when_there_was_no_override():
    vae, pipe = _setup(cap = 1 << 20, untiled_need = 1, tiled_need = 1)
    O.uninstall(vae)
    assert "_decode" not in vae.__dict__ and "decode" not in vae.__dict__ and L.LAYERS_ATTR not in vae.__dict__


class _SlicedVae(_VaeLike):
    def decode(self, z, return_dict = True):
        # diffusers with use_slicing: one _decode per batch sample
        return (torch.cat([self._decode(z[i:i + 1])[0] for i in range(z.shape[0])]),)


def test_one_session_spans_a_sliced_batch_decode(monkeypatch):
    """Item 4283031587: with slicing, every slice's _decode used to recover on its own: move the pipeline off,
    restore it, then move it off again for the next slice. Now the public decode opens one session for the batch:
    the pipeline moves off once and comes back once, after the last slice."""
    moves = []
    monkeypatch.setattr(O, "_to_host", lambda m: (moves.append("off"), m.to(torch.float32)))
    monkeypatch.setattr(O, "_return_to", lambda m, d: (moves.append("on"), m.to(_CARD)))
    budget = _Budget(16384 + 1000)
    transformer, text_encoder = _on_card(8192), _on_card(4096)
    vae = _SlicedVae(budget, [transformer, text_encoder], 10000, 500)
    vae.to(_CARD)
    pipe = _Pipe(transformer = transformer, text_encoder = text_encoder, vae = vae)
    assert O.install(vae, pipe = pipe)
    out = vae.decode(torch.ones(3, 4, 1, 8, 8))[0]
    assert out.shape[0] == 3
    assert moves == ["off", "off", "on", "on"]  # once for the whole batch
    assert [s[0] for s in vae.seen] == [False, False, False, False]  # slice 1 failed once; 2 and 3 fit freed
    assert _devices(pipe)["transformer"] == _CARD and O._SESSION_ATTR not in vae.__dict__


# -- PR review round 10 -----------------------------------------------------------------------------------------


def test_real_release_resident_groups_keeps_the_restore_of_a_partial_release(monkeypatch):
    """Item 4283219211: diffusion_memory.release_resident_groups caught a failure part way and returned None, so the
    groups it had already demoted were never re-pinned. It now returns the restore for them (the failing one too)."""
    from core.inference import diffusion_memory as M

    class G:
        def __init__(self):
            self._unsloth_resident = True
            self._unsloth_resident_bytes = 1 << 20

    groups = [G(), G(), G()]
    module = torch.nn.Linear(2, 2)
    module._unsloth_resident_room = 1
    module._unsloth_resident_device = "cpu"
    pipe = types.SimpleNamespace(components = {"transformer": module})
    demoted = []

    def release(group):
        if len(demoted) == 1:
            raise RuntimeError("copy stream failed")
        group._unsloth_resident = False
        demoted.append(group)

    repinned = []
    monkeypatch.setattr(M, "_offload_groups", lambda m: groups)
    monkeypatch.setattr(M, "_release_group", release)
    monkeypatch.setattr(M, "_keep_groups_resident", lambda m, room, dev, logger, only: repinned.append(set(only)))
    restore = M.release_resident_groups(pipe, 1 << 30)
    assert restore is not None and len(demoted) == 1
    restore()
    assert repinned == [{id(groups[2]), id(groups[1])}]


def test_unwrapping_an_outer_layer_collapses_a_removed_inner_layer():
    """Item 4283219222: inner layer removed while the outer one wraps it, then the outer removed: the slot must go
    back to what the inner one had replaced (here: nothing, the class method), not to the dead inner dispatch."""
    class C:
        def f(self):
            return "class"

    c = C()
    inner = L.wrap(c, "f", "inner", lambda fn, layer: lambda: "inner:" + fn())
    outer = L.wrap(c, "f", "outer", lambda fn, layer: lambda: "outer:" + fn())
    assert c.f() == "outer:inner:class"
    assert L.unwrap(inner) is False and c.f() == "outer:class"
    assert L.unwrap(outer) is True
    assert "f" not in c.__dict__ and c.f() == "class"
    bound = types.MethodType(lambda self: "bound", c)
    c.f = bound
    inner = L.wrap(c, "f", "inner", lambda fn, layer: lambda: "inner:" + fn())
    outer = L.wrap(c, "f", "outer", lambda fn, layer: lambda: "outer:" + fn())
    L.unwrap(inner)
    L.unwrap(outer)
    assert c.__dict__["f"] is bound


def test_component_whose_move_and_rollback_both_fail_gets_a_deferred_return(monkeypatch):
    """Item 4283219189: the move to the host failed part way and moving it back failed too; the component was left
    split across devices with nothing recorded. Now it is settled whole on the host with a deferred return."""
    m = _on_card(64)
    calls = []

    def to_host(module):
        calls.append("host")
        if len(calls) == 1:
            next(module.parameters()).data = next(module.parameters()).data.to(torch.float32)
            raise _FakeOOM()
        module.to(torch.float32)

    def back(module, device):
        raise _FakeOOM()

    monkeypatch.setattr(O, "_to_host", to_host)
    monkeypatch.setattr(O, "_return_to", back)
    session = O.DecodeSession(None, _FakeDevice())
    assert session.apply(lambda: O._to_host(m), session._component_change("transformer", m)) is False
    assert session.log == [] and all(p.dtype == torch.float32 for p in m.parameters())
    assert isinstance(m.__dict__.get(O._DEFERRED_ATTR), O.DeferredReturn)


# -- PR review round 11 -----------------------------------------------------------------------------------------


def test_trace_refuses_activation_fusion_for_hooked_or_inplace_reused_norms():
    """Item 4283572932, class: exclusivity is proven on the trace. A hook on the norm, or an in-place reuse of its
    output outside the SiLU, keeps the norm unpaired (GroupNorm-only fusion)."""
    class Blk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.GroupNorm(4, 8)
            self.act = torch.nn.SiLU()

        def forward(self, x):
            h = self.norm(x)
            y = self.act(h)
            h.mul_(2)  # in-place reuse after the SiLU read it
            return y + h

    blk = Blk()
    calls, consumed, _, info = F._trace_group_norm_silu([blk.norm], [blk.act], lambda: blk(torch.randn(1, 8, 4, 4)))
    assert calls == [1] and consumed == [0] and info["shared"][0] >= 1
    clean = torch.nn.Sequential(torch.nn.GroupNorm(4, 8), torch.nn.SiLU())
    calls, consumed, _, _ = F._trace_group_norm_silu([clean[0]], [clean[1]], lambda: clean(torch.randn(1, 8, 4, 4)))
    assert consumed == [1]
    h = clean[0].register_forward_hook(lambda m, i, o: None)
    calls, consumed, _, _ = F._trace_group_norm_silu([clean[0]], [clean[1]], lambda: clean(torch.randn(1, 8, 4, 4)))
    h.remove()
    assert consumed == [0]


def _fake_kernel(monkeypatch, wrong_site = None, kernel_for = None, during = None):
    """group_norm_act on CPU: the reference, flagged as a kernel run; ``wrong_site`` gets a local error."""
    def fake(x, norm, act = True, in_bias = None, on_kernel = None, **_):
        y = F.group_norm_reference(x, norm, act, in_bias)
        if norm is wrong_site:
            y = y.clone()
            y[..., 0, 0] += 0.5  # one corner of one site: invisible in a mean, large at the site
        if during is not None and kernel_for is not None and norm not in kernel_for:
            during()  # another thread completes a kernel while this (stock-fallback) call is in flight
        if kernel_for is None or norm in kernel_for:
            if on_kernel is not None:
                on_kernel()
            legacy = F.__dict__.get("_GN_KERNEL_RUNS")  # the module-global counter older code read
            if legacy is not None:
                legacy[0] += 1
        return y

    monkeypatch.setattr(F, "group_norm_act", fake)
    monkeypatch.setattr(F, "runtime_ok", lambda: True)
    monkeypatch.delenv(F.STRUCTURAL_ENV, raising = False)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))


def test_structural_calibration_checks_every_fused_site(monkeypatch):
    """Class ask: the calibration compares each fused site against stock on the input it saw, not only the final crop.
    A site that is wrong in one corner is rejected even when the crop error stays small."""
    vae = AutoencoderKLCleanNew()
    norms = F._plain_group_norms(vae.decoder)
    _fake_kernel(monkeypatch, wrong_site = norms[0])
    F.install_structural(vae)
    with torch.no_grad():
        vae._decode(torch.randn(1, 8, 6, 6))
    state = vae._unsloth_vae_structural
    assert state.status == "rejected" and state.site_max_abs >= 0.4
    ok = AutoencoderKLCleanNew()
    _fake_kernel(monkeypatch)
    F.install_structural(ok)
    with torch.no_grad():
        ok._decode(torch.randn(1, 8, 6, 6))
    assert ok._unsloth_vae_structural.status == "ok" and ok._unsloth_vae_structural.site_max_abs == 0.0


def test_structural_kernel_accounting_is_local_to_the_calibration(monkeypatch):
    """Item 4283572986: another VAE's kernel run (another thread) during this VAE's calibration used to bump the
    shared counter and make an all-fallback calibration look fused. Now only this calibration's own runs count."""
    vae = AutoencoderKLCleanNew()
    other_norm = torch.nn.GroupNorm(4, 8)
    other = F._fast_gn_maybe_act(other_norm, True, types.SimpleNamespace(kernel_calls = 0, calls = 0))
    import threading

    def concurrent():
        t = threading.Thread(target = lambda: other(torch.randn(1, 8, 4, 4)))
        t.start()
        t.join()

    # this VAE's norms all take the stock fallback; another decode's kernel completes meanwhile
    _fake_kernel(monkeypatch, kernel_for = {other_norm}, during = concurrent)
    F.install_structural(vae)
    with torch.no_grad():
        vae._decode(torch.randn(1, 8, 6, 6))
    assert vae._unsloth_vae_structural.status == "rejected"


def test_strict_restore_reports_an_incomplete_repin(monkeypatch):
    """Item 4283572942: the re-pin swallows its own failures; the VAE path restores strictly and gets an error while
    any released group is still streamed, so its settle / defer path runs and the restore is kept."""
    from core.inference import diffusion_memory as M

    class Grp:
        def __init__(self):
            self._unsloth_resident = True
            self._unsloth_resident_bytes = 1 << 20

    groups = [Grp(), Grp()]
    module = torch.nn.Linear(2, 2)
    module._unsloth_resident_room = 1
    module._unsloth_resident_device = "cpu"
    pipe = types.SimpleNamespace(components = {"transformer": module})
    monkeypatch.setattr(M, "_offload_groups", lambda m: groups)
    monkeypatch.setattr(M, "_release_group", lambda g: setattr(g, "_unsloth_resident", False))
    pinned = [0]

    def keep(m, room, dev, logger, only):  # copies one group, then fails and swallows it (as the real one does)
        groups[0]._unsloth_resident = True
        pinned[0] += 1

    monkeypatch.setattr(M, "_keep_groups_resident", keep)
    restore = M.release_resident_groups(pipe, 1 << 30)
    restore()  # the lenient form other callers use: unchanged
    with pytest.raises(RuntimeError, match = "left 1 streamed"):
        restore(strict = True)
    got = [restore]
    session = O.DecodeSession(None, _FakeDevice())
    released = []
    monkeypatch.setattr(M, "release_resident_groups", lambda *a, **k: (released.append(1), restore)[1])
    anchor = module
    pipe2 = types.SimpleNamespace(components = {"transformer": anchor})
    assert session._release_groups(pipe2) == 1
    session.restore()
    assert session.log == [] and O._DEFERRED_ATTR in anchor.__dict__  # kept for a retry, not dropped
    del got


def test_step_a_streams_a_resident_torchao_transformer(monkeypatch):
    """Item 4283572958: a hookless resident torchao transformer was skipped by step (a). It now gets Studio's own
    block offload hooks (hook_resident_denoiser, as an oversized request does) and its groups stream."""
    from core.inference import diffusion_memory as M

    vae, pipe = _setup(cap = 16384 + 1000, untiled_need = 10000, tiled_need = 500)
    transformer = pipe.components["transformer"]
    pipe.transformer = transformer
    monkeypatch.setattr(O, "_holds_torchao", lambda m: m is transformer)
    events = []

    def hook(p, device, logger = None):
        events.append("hook")
        transformer._unsloth_resident_room = 1  # hooked, every group resident
        return True

    def release(p, need, logger = None, reason = ""):
        events.append("release")
        transformer.to(torch.float32)  # its groups stream
        return lambda strict = False: (events.append("repin"), transformer.to(_CARD))

    monkeypatch.setattr(M, "hook_resident_denoiser", hook)
    monkeypatch.setattr(M, "release_resident_groups", release)
    vae._decode(_z())
    assert events == ["hook", "release", "repin"]
    assert vae._unsloth_vae_oom_stats["recovered_by"] == "freed"


def test_vae_offload_respects_host_ram(monkeypatch):
    """Item 4283572966: step (c) creates host copies of the VAE weights; with too little host RAM it is skipped."""
    applied = []
    monkeypatch.setattr(O, "_apply_vae_offload", lambda v, d: applied.append(1))
    monkeypatch.setattr(O, "_host_free_mib", lambda: 1)
    session = O.DecodeSession(_VaeLike(_Budget(1), [], 1, 1), _FakeDevice())
    assert session.offload_vae() is False and applied == []
    monkeypatch.setattr(O, "_host_free_mib", lambda: 1 << 20)
    assert session.offload_vae() is True and applied == [1]


def test_wide_tiles_free_the_card_before_falling_back_to_stock_tiles(monkeypatch):
    """Item 4283572976: Studio's wide tiles caught their own OOM and fell back to the smaller stock tiles before the
    OOM retry could offload anything. Now the retry frees the card first and the wide tiles decode."""
    diffusers = pytest.importorskip("diffusers")
    from core.inference import diffusion_vae_tiling as vt

    monkeypatch.delenv(vt.WIDE_TILES_ENV, raising = False)
    monkeypatch.setenv(vt.MAX_TILE_ENV, "1")
    monkeypatch.setattr(vt, "floor_shortfall", lambda vae, z: None)
    torch.manual_seed(0)
    vae = diffusers.AutoencoderKLQwenImage(
        base_dim = 8, z_dim = 4, dim_mult = [1, 2, 4, 4], num_res_blocks = 1, attn_scales = [],
        temperal_downsample = [False, True, True], latents_mean = [0.0] * 4, latents_std = [1.0] * 4,
    ).eval()  # fmt: skip
    vae.enable_tiling()
    assert vt.install(vae)
    z = torch.randn(1, 4, 1, 48, 48, generator = torch.Generator().manual_seed(1))
    freed, fallbacks = [], []
    monkeypatch.setattr(vt, "_log_stock_fallback", lambda vae, reason: fallbacks.append(reason))
    monkeypatch.setattr(O, "_device_of", lambda z: _FakeDevice())
    monkeypatch.setattr(O.DecodeSession, "free_others", lambda self, pipe: (freed.append(1), setattr(self, "freed", True), 1)[2])
    real = vae.decoder.forward

    def decoder(x, *a, **k):
        if not freed:
            raise _FakeOOM()  # nothing fits until the rest of the pipeline is off the card
        return real(x, *a, **k)

    monkeypatch.setattr(vae.decoder, "forward", decoder)
    assert O.install(vae)
    with torch.no_grad():
        vae.decode(z)
    assert freed == [1]
    assert fallbacks == []  # the wide tiles decoded after the card was freed; no fallback to the stock tiles


def test_ltx2_tiles_free_the_card_before_the_floor_tile(monkeypatch):
    from core.inference import video_ltx2_vae_tiles as lt

    tried = []
    freed = []

    def tiles(vae, z, temb, causal, th, tw, budgeted):
        tried.append((th, tw))
        if (th, tw) != (16, 16) and not freed:
            raise _FakeOOM()
        return torch.zeros(1)

    monkeypatch.setattr(lt, "_decode_tiles", tiles)
    monkeypatch.setattr(lt, "_plan", lambda vae, z: ((32, 32), True))
    monkeypatch.setattr(lt, "oom_free_first", lambda vae, z: (freed.append(1), True)[1])
    vae = types.SimpleNamespace()
    lt.tiled_decode(vae, torch.zeros(1, 4, 2, 64, 64), return_dict = False)
    assert tried == [(32, 32), (32, 32)] and freed == [1]


def test_ladder_is_the_outermost_decode_layer_in_both_pipelines():
    """Class ask (c): in the load order of diffusion.py and video.py the OOM retry is installed after every decode
    wrapper that can fall back on an OOM, so its scope is the outermost decode layer; inner fallbacks (wide tiles,
    LTX-2 tiles, untiled gate) call free_first before going cheaper."""
    import inspect
    import re

    from core.inference import diffusion as D
    from core.inference import video as V
    from core.inference import diffusion_vae_tiling as vt
    from core.inference import video_ltx2_vae_tiles as lt

    image = inspect.getsource(D)
    assert image.index("install_vae_oom_retry(") > image.index("install_wide_vae_tiles(getattr(pipe")
    video = inspect.getsource(V)
    oom = video.index("install_vae_oom_retry(")
    for wrapper in ("install_ltx2_vae_tiles(", "install_untiled_decode(", "install_rocm_vae_bf16_decode(pipe"):
        assert video.index(wrapper) < oom, wrapper
    assert re.search(r"oom_free_first\(self, z\)", inspect.getsource(vt))
    assert "oom_free_first(vae, z)" in inspect.getsource(lt.tiled_decode)


def test_scope_stays_outermost_over_the_untiled_gate(monkeypatch):
    """The untiled gate re-asserts its slot after each call; with the OOM scope installed over it, it must keep the
    scope (its __wrapped__ chain reaches the gate) instead of dropping it."""
    from core.inference import video_vae_untiled as U

    vae, pipe = _setup(cap = 1 << 20, untiled_need = 1, tiled_need = 1)
    O.uninstall(vae)
    vae.use_tiling = True
    vae.decode = lambda z, return_dict = True: vae._decode(z, return_dict)
    monkeypatch.setattr(U, "_enabled", lambda: True)
    family = next(iter(U._BYTES_PER_LATENT_PIXEL))
    monkeypatch.setattr(U, "_free_bytes", lambda d: 0)
    assert U.install_untiled_decode(types.SimpleNamespace(vae = vae), family)
    gate = vae.__dict__["decode"]
    assert O.install(vae, pipe = pipe)
    scope = vae.__dict__["decode"]
    assert getattr(scope, "_unsloth_vae_oom_scope", False) and scope.__wrapped__ is gate
    vae.decode(_z())
    assert vae.__dict__["decode"] is scope  # still outermost after a call


# -- first-decode cost: tiny-crop pairing and the on-disk verdict cache ------------------------------------------


def test_tiny_crop_pairs_like_the_full_crop():
    """The pairing traced on the smallest calibration crop equals the one traced on a full-size latent."""
    vae = AutoencoderKLSomethingNew()
    norms, silus = F._plain_group_norms(vae.decoder), F._plain_silus(vae.decoder)
    z = torch.randn(1, 8, 40, 40)
    with torch.no_grad():
        full = F._trace_group_norm_silu(norms, silus, lambda: vae._decode(z))
        for side in (F._PROBE_LATENTS, 8, 2):
            tiny = F._trace_group_norm_silu(norms, silus, lambda: vae._decode(z[..., :side, :side]))
            assert full[0] == tiny[0] and full[1] == tiny[1], side
            assert [s > 0 for s in full[3]["shared"]] == [s > 0 for s in tiny[3]["shared"]], side
    assert F._probe_latent(z).shape[-1] == F._PROBE_LATENTS


def _calibrated(monkeypatch, tmp_path, vae, **kw):
    monkeypatch.setenv(F.STRUCTURAL_CACHE_ENV, "1")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR", str(tmp_path))
    _fake_kernel(monkeypatch, **kw)
    traces = []
    real = F._trace_group_norm_silu
    monkeypatch.setattr(F, "_trace_group_norm_silu", lambda *a, **k: (traces.append(1), real(*a, **k))[1])
    F.install_structural(vae)
    with torch.no_grad():
        vae._decode(torch.randn(1, 8, 6, 6))
    return traces


def test_structural_verdict_is_cached_and_reused(monkeypatch, tmp_path):
    first = AutoencoderKLCleanNew()
    assert _calibrated(monkeypatch, tmp_path, first) == [1]
    assert first._unsloth_vae_structural.status == "ok" and not first._unsloth_vae_structural.cached
    assert len(list((tmp_path / "vae_structural").glob("*.json"))) == 1
    second = AutoencoderKLCleanNew()  # a later process: same class, config, layout, versions
    assert _calibrated(monkeypatch, tmp_path, second) == []  # no trace, no checks
    st = second._unsloth_vae_structural
    assert st.status == "ok" and st.cached and st.gn_silu == first._unsloth_vae_structural.gn_silu
    assert any(getattr(m.forward, "_unsloth_vae_fused", False) for m in F._plain_group_norms(second.decoder)
               if "forward" in m.__dict__) or any("forward" in m.__dict__ for m in second.decoder.modules())  # fmt: skip


def test_stale_cache_key_traces_again(monkeypatch, tmp_path):
    """Another torch / triton / config is another key: the old verdict is not used and the VAE is traced again."""
    assert _calibrated(monkeypatch, tmp_path, AutoencoderKLCleanNew()) == [1]
    monkeypatch.setattr(torch, "__version__", "0.0.0-other")
    assert _calibrated(monkeypatch, tmp_path, AutoencoderKLCleanNew()) == [1]
    assert len(list((tmp_path / "vae_structural").glob("*.json"))) == 2


def test_corrupt_or_mismatched_cache_entry_traces_again(monkeypatch, tmp_path):
    assert _calibrated(monkeypatch, tmp_path, AutoencoderKLCleanNew()) == [1]
    (entry,) = (tmp_path / "vae_structural").glob("*.json")
    entry.write_text('{"status": "ok", "paired": [true]}')  # wrong key / length
    assert _calibrated(monkeypatch, tmp_path, AutoencoderKLCleanNew()) == [1]


def test_cached_rejection_skips_fusion(monkeypatch, tmp_path):
    bad = AutoencoderKLCleanNew()
    norms = F._plain_group_norms(bad.decoder)
    assert _calibrated(monkeypatch, tmp_path, bad, wrong_site = norms[0]) == [1]
    assert bad._unsloth_vae_structural.status == "rejected"
    again = AutoencoderKLCleanNew()
    assert _calibrated(monkeypatch, tmp_path, again) == []
    assert again._unsloth_vae_structural.status == "rejected" and not any(
        "forward" in m.__dict__ for m in again.decoder.modules()
    )


def test_structural_cache_kill_switch(monkeypatch, tmp_path):
    assert _calibrated(monkeypatch, tmp_path, AutoencoderKLCleanNew()) == [1]
    monkeypatch.setenv(F.STRUCTURAL_CACHE_ENV, "0")
    traces = []
    real = F._trace_group_norm_silu
    monkeypatch.setattr(F, "_trace_group_norm_silu", lambda *a, **k: (traces.append(1), real(*a, **k))[1])
    vae = AutoencoderKLCleanNew()
    F.install_structural(vae)
    with torch.no_grad():
        vae._decode(torch.randn(1, 8, 6, 6))
    assert traces == [1]
