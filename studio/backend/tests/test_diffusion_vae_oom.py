# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""diffusion_vae_oom: the offload-first OOM ladder (CPU, fake OOMs and a fake card) and the structural fused
GroupNorm(+SiLU) match."""

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_vae_oom as O  # noqa: E402
from core.inference import diffusion_vae_fused as F  # noqa: E402


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
    real = vae._unsloth_vae_oom_inner

    def inner(z, *a, **k):
        if vae.__dict__.get(O._SESSION_ATTR) is not None and vae.__dict__[O._SESSION_ATTR].vae_offloaded:
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
    session.log.append(("moved:transformer", *session._component_undo("transformer", m)))
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
    with pytest.raises(_FakeOOM):
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
    session.log.append(("vae_offload", lambda: O._undo_vae_offload(vae, cpu),
                        lambda exc: O._defer(vae, cpu, "vae_offload", lambda: O._undo_vae_offload(vae, cpu), False, vae)))
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
    session.log.append(("moved:unet", *session._component_undo("unet", unet)))
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
        with pytest.raises(RuntimeError, match = "out of memory"):
            vae._decode(_z())
        assert [s[0] for s in vae.seen] == [False]  # the gate retries tiled itself


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
    calls, consumed, _ = F._trace_group_norm_silu(
        norms, silus, lambda: vae._decode(torch.randn(1, 8, 6, 6))
    )
    names = {id(m): n for n, m in vae.decoder.named_modules()}
    got = {names[id(m)]: (c, k) for m, c, k in zip(norms, calls, consumed)}
    assert got["block.norm1"] == (1, 1)
    assert got["norm_out"] == (1, 1)
    assert got["attn_norm"] == (1, 0)
    # norm2's output reaches the SiLU AND the residual: the trace sees the SiLU, the probe check must catch the rest
    assert got["block.norm2"] == (1, 1)


def _install_on_cpu(monkeypatch, vae):
    monkeypatch.setattr(F, "runtime_ok", lambda: True)
    monkeypatch.delenv(F.STRUCTURAL_ENV, raising = False)
    n = F.install_structural(vae)
    assert n == 4
    state = vae._unsloth_vae_structural
    # calibrate as on the first CUDA decode (CPU tensors take the reference GroupNorm, so it is exact here)
    z = torch.randn(1, 8, 6, 6)
    inner = vae.__dict__["_unsloth_vae_structural_inner"]
    with torch.no_grad():
        calls, consumed, ref = F._trace_group_norm_silu(
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
        out = vae.__dict__["_unsloth_vae_structural_inner"](z)[0]
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
    assert "_unsloth_vae_structural_inner" not in vae.__dict__
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


