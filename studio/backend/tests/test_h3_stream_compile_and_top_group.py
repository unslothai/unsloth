# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 streamed denoiser: the regional compile sits BELOW the group-offload hooks (no recompile when the hook
state changes), and the top-level group onloads from a pinned copy on the blocks' stream."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers.hooks")


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(self, x):
        return torch.relu(self.lin(x)) + 1


class _DiT(torch.nn.Module):
    _repeated_blocks = ["_Block"]

    def __init__(self):
        super().__init__()
        self.proj_in = torch.nn.Linear(8, 8)
        self.transformer_blocks = torch.nn.ModuleList([_Block() for _ in range(3)])

    def forward(self, x):
        x = self.proj_in(x)
        for b in self.transformer_blocks:
            x = b(x)
        return x


def _streamed_dit():
    from diffusers.hooks import apply_group_offloading

    torch.manual_seed(0)
    dit = _DiT().eval()
    apply_group_offloading(
        dit,
        onload_device = torch.device("cpu"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
    )
    return dit


def _graphs():
    from torch._dynamo.utils import counters
    return counters["stats"].get("unique_graphs", 0)


def _flip_hook_state(dit):
    """What the resident set does: replace a block group's onload / offload with no-ops."""
    from core.inference.video_minimax_h3_residency import h3_offload_groups, make_resident

    _top, blocks = h3_offload_groups(dit)
    make_resident(blocks[0])


def test_compiling_below_the_hooks_does_not_recompile_when_the_hook_state_changes(monkeypatch):
    from core.inference.video_minimax_h3_residency import compile_blocks_below_offload_hooks

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch._dynamo.reset()
    dit = _streamed_dit()
    x = torch.randn(2, 8)
    with torch.no_grad():
        eager = dit(x)
    kwargs = {"backend": "eager", "dynamic": False, "fullgraph": False}
    for block in dit.transformer_blocks:
        block.compile(**kwargs)
    dit._unsloth_regional_compile_kwargs = kwargs
    assert compile_blocks_below_offload_hooks(dit) == 3
    assert all(b._compiled_call_impl is None for b in dit.transformer_blocks)
    with torch.no_grad():
        out = dit(x)
    assert torch.equal(out, eager)
    before = _graphs()
    _flip_hook_state(dit)
    with torch.no_grad():
        assert torch.equal(dit(x), eager)
    assert _graphs() == before, "a hook-state change recompiled the block"


def test_the_old_placement_recompiles_on_the_same_change(monkeypatch, traced_offload_hooks):
    """Control: Module.compile traces the hooks, so the same flip compiles new graphs (the 9 s first steps)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch._dynamo.reset()
    dit = _streamed_dit()
    x = torch.randn(2, 8)
    for block in dit.transformer_blocks:
        block.compile(backend = "eager", dynamic = False, fullgraph = False)
    with torch.no_grad():
        dit(x)
    before = _graphs()
    _flip_hook_state(dit)
    with torch.no_grad():
        dit(x)
    assert _graphs() > before


def test_compile_below_hooks_kill_switch_and_missing_kwargs(monkeypatch):
    from core.inference.video_minimax_h3_residency import compile_blocks_below_offload_hooks

    dit = _streamed_dit()
    for block in dit.transformer_blocks:
        block.compile(backend = "eager")
    assert compile_blocks_below_offload_hooks(dit) == 0
    dit._unsloth_regional_compile_kwargs = {"backend": "eager"}
    monkeypatch.setenv("UNSLOTH_H3_COMPILE_BELOW_HOOKS", "0")
    assert compile_blocks_below_offload_hooks(dit) == 0
    assert all(b._compiled_call_impl is not None for b in dit.transformer_blocks)


def test_regional_compile_records_its_kwargs_for_the_streamed_path():
    import inspect

    from core.inference import diffusion_speed
    assert "_unsloth_regional_compile_kwargs" in inspect.getsource(
        diffusion_speed._compile_repeated_blocks
    )


class _FakeGroup:
    def __init__(self, stream):
        self.stream = stream
        self.low_cpu_mem_usage = True
        self.record_stream = False
        self.non_blocking = True
        self.cpu_param_dict = {}
        self.offload_to_disk_path = None
        self.modules = []
        self.parameters = []
        self.buffers = []

    def _init_cpu_param_dict(self):
        assert self.stream is not None and not self.low_cpu_mem_usage
        return {"w": "pinned"}

    def onload_(self):
        pass

    def offload_(self):
        pass


def test_the_top_level_group_gets_a_pinned_copy_and_the_block_stream(monkeypatch):
    import core.inference.video_minimax_h3_residency as res
    import core.inference.video_minimax_h3_te as te

    monkeypatch.setattr(te, "h3_te_pin_allowed", lambda *a: True)

    stream = object()
    top, block = _FakeGroup(None), _FakeGroup(stream)
    monkeypatch.setattr(res, "h3_offload_groups", lambda t: (top, [block]))
    assert res.pin_streamed_top_level_group(object())
    assert top.stream is stream and top.cpu_param_dict == {"w": "pinned"}
    assert top.record_stream and not top.non_blocking and not top.low_cpu_mem_usage
    assert not res.pin_streamed_top_level_group(object())


def test_the_top_level_pin_kill_switch_and_unstreamed_blocks(monkeypatch):
    import core.inference.video_minimax_h3_residency as res
    import core.inference.video_minimax_h3_te as te

    monkeypatch.setattr(te, "h3_te_pin_allowed", lambda *a: True)

    top = _FakeGroup(None)
    monkeypatch.setattr(res, "h3_offload_groups", lambda t: (top, [_FakeGroup(None)]))
    assert not res.pin_streamed_top_level_group(object())
    monkeypatch.setattr(res, "h3_offload_groups", lambda t: (top, [_FakeGroup(object())]))
    monkeypatch.setenv("UNSLOTH_H3_TOP_GROUP_PIN", "0")
    assert not res.pin_streamed_top_level_group(object())
    assert top.stream is None and top.cpu_param_dict == {}


def test_a_refused_pin_restores_the_group(monkeypatch):
    import core.inference.video_minimax_h3_residency as res
    import core.inference.video_minimax_h3_te as te

    monkeypatch.setattr(te, "h3_te_pin_allowed", lambda *a: True)

    top = _FakeGroup(None)

    def boom():
        raise RuntimeError("cudaHostAlloc failed")

    top._init_cpu_param_dict = boom
    monkeypatch.setattr(res, "h3_offload_groups", lambda t: (top, [_FakeGroup(object())]))
    assert not res.pin_streamed_top_level_group(
        object(), logger = types.SimpleNamespace(warning = lambda *a: None)
    )
    assert (
        top.stream is None
        and top.low_cpu_mem_usage
        and top.non_blocking
        and top.cpu_param_dict == {}
    )


def test_the_top_level_pin_honours_the_host_pin_policy(monkeypatch):
    """The pin-nothing override and the Windows / WSL pinned cap leave the top-level group pageable."""
    import core.inference.video_minimax_h3_residency as res
    import core.inference.video_minimax_h3_te as te

    top = _FakeGroup(None)
    monkeypatch.setattr(res, "h3_offload_groups", lambda t: (top, [_FakeGroup(object())]))
    monkeypatch.setattr(te, "h3_te_pin_allowed", lambda *a: False)
    assert not res.pin_streamed_top_level_group(object())
    assert top.stream is None and top.low_cpu_mem_usage and top.cpu_param_dict == {}


def _swap_ready(monkeypatch):
    import core.inference.video_minimax_h3_te as te

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(te, "h3_te_pin_allowed", lambda *a: True)
    real_pin = te.pin_module_in_place
    monkeypatch.setattr(
        te,
        "pin_module_in_place",
        lambda m, **k: real_pin(m, _arena_factory = lambda n: torch.zeros(n, dtype = torch.uint8), **k),
    )
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self, *a, **k: self.device.type == "cpu")


def test_a_rotating_vae_moves_by_repointing_at_its_pinned_copy(monkeypatch):
    from core.inference.video_minimax_h3_residency import install_pinned_swap

    _swap_ready(monkeypatch)
    vae = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.GroupNorm(2, 4))
    weight = vae[0].weight
    expect = {k: v.clone() for k, v in vae.state_dict().items()}
    assert install_pinned_swap(vae)
    assert vae[0].weight is weight
    host_ptr = weight.data_ptr()
    weight.data = weight.data.clone()
    vae.to("cpu")
    assert weight.data_ptr() == host_ptr, "the CPU move copied instead of re-pointing"
    for k, v in vae.state_dict().items():
        assert torch.equal(v, expect[k])


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
@pytest.mark.parametrize("pin_capped", [False, True], ids = ["uncapped", "windows_wsl_pin_cap"])
def test_a_pinned_swap_round_trip_on_cuda(monkeypatch, pin_capped):
    import core.inference.diffusion_memory as mem
    from core.inference.video_minimax_h3_residency import install_pinned_swap

    # Windows/WSL cap pinned memory, so h3_te_pin_allowed declines; opt in via override.
    monkeypatch.setattr(mem, "_pinned_memory_capped", lambda: pin_capped)
    monkeypatch.setenv(mem.GROUP_OFFLOAD_PIN_ENV, "1")
    vae = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.GroupNorm(2, 4))
    expect = {k: v.clone() for k, v in vae.state_dict().items()}
    assert install_pinned_swap(vae)
    host_ptr = vae[0].weight.data_ptr()
    vae.to("cuda")
    assert all(t.is_cuda for t in vae.parameters())
    x = torch.randn(1, 3, 8, 8, device = "cuda")
    y = vae(x)
    vae.to("cpu")
    assert vae[0].weight.data_ptr() == host_ptr
    for k, v in vae.state_dict().items():
        assert torch.equal(v, expect[k])
    assert torch.allclose(vae.cpu()(x.cpu()), y.cpu(), atol = 1e-5)


def test_the_pinned_swap_falls_back_on_dtype_moves_and_honours_its_kill_switch(monkeypatch):
    from core.inference.video_minimax_h3_residency import install_pinned_swap

    _swap_ready(monkeypatch)
    vae = torch.nn.Linear(4, 4)
    assert install_pinned_swap(vae)
    vae.to(torch.float16)
    assert vae.weight.dtype is torch.float16
    vae.to("meta")
    assert vae.weight.device.type == "meta" and vae.weight.dtype is torch.float16
    other = torch.nn.Linear(4, 4)
    monkeypatch.setenv("UNSLOTH_H3_VAE_PINNED_SWAP", "0")
    assert not install_pinned_swap(other)
    assert "to" not in other.__dict__


def test_the_pinned_swap_only_installs_from_the_parked_state(monkeypatch):
    from core.inference.video_minimax_h3_residency import install_pinned_swap

    _swap_ready(monkeypatch)
    vae = torch.nn.Linear(4, 4).to("meta")
    assert not install_pinned_swap(vae)
