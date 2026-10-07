# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``video_wan_fused.py``: the fused Wan block forward engages only for an fp16 Wan load (bf16 / fp32 keep
the stock forward), is bit-identical to the stock block on a real GPU, survives group-offload hooks, falls back to
stock on a self-check mismatch, and is restored on teardown."""

from __future__ import annotations

import ast
import inspect

import pytest

from core.inference import video_wan_fused as wf

torch = pytest.importorskip("torch")
transformer_wan = pytest.importorskip("diffusers.models.transformers.transformer_wan")
WanTransformerBlock = transformer_wan.WanTransformerBlock

_CUDA = torch.cuda.is_available() and not getattr(torch.version, "hip", None)
needs_cuda = pytest.mark.skipif(not _CUDA, reason = "needs an NVIDIA GPU")


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(wf.WAN_FUSED_ENV, raising = False)
    wf.uninstall()
    stock = WanTransformerBlock.forward
    yield
    wf.uninstall()
    assert WanTransformerBlock.forward is stock


def _fake_kernels(monkeypatch):
    # install logic without a GPU; on ROCm torch wanted() refuses HIP, so pretend a CUDA build too
    monkeypatch.setattr(wf, "_kernels", lambda: {"modnorm": None})
    monkeypatch.setattr(torch.version, "hip", None)


def test_stock_block_matches_the_fingerprint():
    from core.inference.diffusion_qwenimage21_rope import _digest
    assert _digest(WanTransformerBlock.forward) in wf._FINGERPRINTS


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_bf16_and_fp32_loads_keep_the_stock_forward(dtype):
    stock = WanTransformerBlock.forward
    assert wf.install(getattr(torch, dtype), "cuda") is False
    assert WanTransformerBlock.forward is stock and not wf.is_installed()


def test_cpu_load_keeps_the_stock_forward():
    stock = WanTransformerBlock.forward
    assert wf.install(torch.float16, "cpu") is False
    assert WanTransformerBlock.forward is stock


@pytest.mark.parametrize("value", ["0", "off", "false", "no"])
def test_kill_switch(monkeypatch, value):
    monkeypatch.setenv(wf.WAN_FUSED_ENV, value)
    stock = WanTransformerBlock.forward
    assert wf.install(torch.float16, "cuda") is False
    assert WanTransformerBlock.forward is stock


def test_changed_diffusers_block_keeps_the_stock_forward(monkeypatch):
    monkeypatch.setattr(wf, "_FINGERPRINTS", frozenset({"not-this-block"}))
    _fake_kernels(monkeypatch)
    stock = WanTransformerBlock.forward
    assert wf.install(torch.float16, "cuda") is False
    assert WanTransformerBlock.forward is stock


def test_install_is_idempotent_and_uninstall_restores(monkeypatch):
    _fake_kernels(monkeypatch)
    stock = WanTransformerBlock.forward
    assert wf.install(torch.float16, "cuda") is True
    patched = WanTransformerBlock.forward
    assert patched is not stock and getattr(patched, "_unsloth_wan_fused", False)
    assert wf.install(torch.float16, "cuda") is True
    assert WanTransformerBlock.forward is patched
    wf.uninstall()
    assert WanTransformerBlock.forward is stock
    wf.uninstall()


def test_install_for_pipe_scopes_to_wan(monkeypatch):
    import types

    _fake_kernels(monkeypatch)
    stock = WanTransformerBlock.forward
    other = types.SimpleNamespace(transformer = torch.nn.Linear(2, 2))
    assert wf.install_for_pipe(other, torch.float16, "cuda") is False
    assert WanTransformerBlock.forward is stock

    class _Wan(torch.nn.Module):
        pass

    _Wan.__module__ = wf._MODULE
    wan = types.SimpleNamespace(transformer = _Wan())
    assert wf.install_for_pipe(wan, torch.float16, "cuda") is True
    assert WanTransformerBlock.forward is not stock
    assert wf.install_for_pipe(other, torch.float16, "cuda") is False
    assert WanTransformerBlock.forward is stock


def test_cpu_tensors_take_the_stock_path(monkeypatch):
    _fake_kernels(monkeypatch)
    blk = _block(dim = 32, ffn = 64, heads = 2, device = "cpu", dtype = torch.float32)
    x, enc, temb, rot = _inputs(blk, 1, 8, 32, "cpu", torch.float32)
    with torch.no_grad():
        want = WanTransformerBlock.forward(blk, x, enc, temb, rot)
        assert wf.install(torch.float16, "cuda") is True
        got = blk(x, enc, temb, rot)
    assert torch.equal(want, got)
    assert wf.counts()["fused"] == 0 and wf.counts()["stock"] == 1


def _block(
    dim,
    ffn,
    heads,
    device,
    dtype,
    cross_attn_norm = True,
):
    torch.manual_seed(0)
    blk = WanTransformerBlock(
        dim, ffn, heads, "rms_norm_across_heads", cross_attn_norm = cross_attn_norm, eps = 1e-6
    )
    blk = blk.to(device = device, dtype = dtype).eval()
    # diffusers keeps these in float32 (WanTransformer3DModel._keep_in_fp32_modules)
    blk.scale_shift_table.data = blk.scale_shift_table.data.float()
    if cross_attn_norm:
        blk.norm2.float()
        with torch.no_grad():
            blk.norm2.weight.normal_(1.0, 0.1)
            blk.norm2.bias.normal_(0.0, 0.1)
    return blk


def _inputs(
    blk,
    B,
    L,
    D,
    device,
    dtype,
    per_token = True,
):
    g = torch.Generator(device = "cpu").manual_seed(1)
    x = (torch.randn(B, L, D, generator = g) * 8).to(device = device, dtype = dtype)
    enc = torch.randn(B, 16, D, generator = g).to(device = device, dtype = dtype)
    shape = (B, L, 6, D) if per_token else (B, 6, D)
    temb = (torch.randn(*shape, generator = g) * 0.5).to(device = device, dtype = dtype)
    head_dim = D // blk.attn1.heads
    freqs = torch.randn(1, L, 1, head_dim, generator = g).to(device)
    return x, enc, temb, (freqs.cos(), freqs.sin())


@needs_cuda
@pytest.mark.parametrize("per_token", [True, False], ids = ["temb_per_token", "temb_per_sample"])
@pytest.mark.parametrize("cross_attn_norm", [True, False], ids = ["norm2_affine", "norm2_identity"])
@pytest.mark.parametrize("batch", [1, 2])
def test_fused_block_is_bit_identical_to_stock_on_gpu(per_token, cross_attn_norm, batch):
    D = 256
    blk = _block(
        dim = D, ffn = 512, heads = 4, device = "cuda", dtype = torch.float16, cross_attn_norm = cross_attn_norm
    )
    x, enc, temb, rot = _inputs(blk, batch, 300, D, "cuda", torch.float16, per_token = per_token)
    x[0, :3, :5] = 300.0
    with torch.no_grad():
        want = WanTransformerBlock.forward(blk, x, enc, temb, rot)
        assert wf.install(torch.float16, "cuda") is True
        got = blk(x, enc, temb, rot)
        direct = wf._fused_forward(blk, x, enc, temb, rot)
    assert torch.equal(want, got)
    assert torch.equal(want, direct)
    assert wf.counts()["fused"] == 1


@needs_cuda
def test_fused_kernels_match_the_stock_op_chain_elementwise():
    """Each kernel against the exact stock expression, on Wan2.2-TI2V-5B's width (3072)."""
    import torch.nn.functional as F

    B, L, D = 1, 512, 3072
    g = torch.Generator(device = "cpu").manual_seed(3)
    x = (torch.randn(B, L, D, generator = g) * 8).half().cuda()
    a = (torch.randn(B, L, D, generator = g) * 3).half().cuda()
    t = (torch.randn(B, L, 6, D, generator = g) * 0.5).half().cuda()
    tbl = (torch.randn(1, 6, D, generator = g) / D**0.5).cuda()
    k = wf._kernels()
    sh, sc, gate, csh, csc, cgate = (tbl.unsqueeze(0) + t.float()).chunk(6, dim = 2)
    ln = F.layer_norm(x.float(), (D,), None, None, 1e-6)
    want = (ln * (1 + sc.squeeze(2)) + sh.squeeze(2)).type_as(x)
    mean, rstd = wf._stats(x, 1e-6)
    assert torch.equal(k["modnorm"](x, mean, rstd, t, tbl, 0, 1), want)
    want = (x.float() + a * gate.squeeze(2)).type_as(x)
    assert torch.equal(k["gate_residual"](x, a, t, tbl, 2), want)
    w = (1 + 0.1 * torch.randn(D, generator = g)).cuda()
    b = (0.1 * torch.randn(D, generator = g)).cuda()
    want = F.layer_norm(x.float(), (D,), w, b, 1e-6).type_as(x)
    assert torch.equal(k["affine_norm"](x, mean, rstd, w, b), want)


@needs_cuda
def test_modulation_kernels_index_past_two_gib_of_temb():
    """A per-token temb past 2**31 elements (1280x704, 529 frames: 116,510+ tokens at D=3072) must not wrap the row
    offset: the tail rows, beyond the int32 limit, still equal the stock expression."""
    import torch.nn.functional as F

    B, L, D = 1, 116_512, 3072
    if torch.cuda.mem_get_info()[0] < 8 << 30:
        pytest.skip("needs ~8 GiB free on the GPU")
    g = torch.Generator(device = "cuda").manual_seed(5)
    # fp16 straight from randn: an fp32 temb here alone would be 8 GiB.
    x = torch.randn(B, L, D, device = "cuda", dtype = torch.half, generator = g)
    a = torch.randn(B, L, D, device = "cuda", dtype = torch.half, generator = g)
    t = torch.randn(B, L, 6, D, device = "cuda", dtype = torch.half, generator = g).mul_(0.5)
    tbl = torch.randn(1, 6, D, device = "cuda", generator = g) / D**0.5
    k = wf._kernels()
    mean, rstd = wf._stats(x, 1e-6)
    got_n = k["modnorm"](x, mean, rstd, t, tbl, 0, 1)[:, -64:]
    got_g = k["gate_residual"](x, a, t, tbl, 2)[:, -64:]
    x, a, t = x[:, -64:], a[:, -64:], t[:, -64:]
    sh, sc, gate = (tbl.unsqueeze(0) + t.float()).chunk(6, dim = 2)[:3]
    ln = F.layer_norm(x.float(), (D,), None, None, 1e-6)
    assert torch.equal(got_n, (ln * (1 + sc.squeeze(2)) + sh.squeeze(2)).type_as(x))
    assert torch.equal(got_g, (x.float() + a * gate.squeeze(2)).type_as(x))


@needs_cuda
def test_fused_forward_survives_group_offload_hooks():
    """Installed before the offload hooks attach (as the loader does), the streamed blocks run the fused path."""
    pytest.importorskip("diffusers.hooks")
    from diffusers.hooks import apply_group_offloading

    D = 128

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(
                _block(dim = D, ffn = 256, heads = 2, device = "cpu", dtype = torch.float16) for _ in range(3)
            )

        def forward(self, x, enc, temb, rot):
            for blk in self.blocks:
                x = blk(x, enc, temb, rot)
            return x

    net = Net().eval()
    x, enc, temb, rot = _inputs(net.blocks[0], 1, 64, D, "cuda", torch.float16)
    with torch.no_grad():
        ref = Net().eval()
        ref.load_state_dict(net.state_dict())
        ref.cuda()
        want = ref(x, enc, temb, rot)
    assert wf.install(torch.float16, "cuda") is True
    apply_group_offloading(
        net,
        onload_device = torch.device("cuda"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
        use_stream = True,
    )
    with torch.no_grad():
        got = net(x, enc, temb, rot)
    assert torch.equal(want, got)
    assert wf.counts()["fused"] == 3


@needs_cuda
def test_self_check_mismatch_keeps_the_stock_forward(monkeypatch):
    blk = _block(dim = 128, ffn = 256, heads = 2, device = "cuda", dtype = torch.float16)
    x, enc, temb, rot = _inputs(blk, 1, 32, 128, "cuda", torch.float16)
    real = wf._fused_forward

    def _off_by_one(*args):
        return real(*args) + 1

    monkeypatch.setattr(wf, "_fused_forward", _off_by_one)
    with torch.no_grad():
        want = WanTransformerBlock.forward(blk, x, enc, temb, rot)
        assert wf.install(torch.float16, "cuda") is True
        got = [blk(x, enc, temb, rot) for _ in range(2)]
    assert all(torch.equal(want, g) for g in got)
    assert wf.counts()["fused"] == 0 and False in wf._VERIFIED.values()


@needs_cuda
def test_kernel_failure_after_the_self_check_keeps_the_stock_forward(monkeypatch):
    blk = _block(dim = 128, ffn = 256, heads = 2, device = "cuda", dtype = torch.float16)
    x, enc, temb, rot = _inputs(blk, 1, 32, 128, "cuda", torch.float16)
    real = wf._fused_forward
    calls = []

    def _jit_fails_after_verify(*args):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("PTX JIT compilation failed")
        return real(*args)

    monkeypatch.setattr(wf, "_fused_forward", _jit_fails_after_verify)
    with torch.no_grad():
        want = WanTransformerBlock.forward(blk, x, enc, temb, rot)
        assert wf.install(torch.float16, "cuda") is True
        got = [blk(x, enc, temb, rot) for _ in range(2)]
    assert all(torch.equal(want, g) for g in got)
    assert len(calls) == 2 and wf.counts() == {"fused": 0, "stock": 2}
    assert list(wf._VERIFIED.values()) == [False]


@needs_cuda
def test_self_check_out_of_memory_retries_instead_of_disabling(monkeypatch):
    blk = _block(dim = 128, ffn = 256, heads = 2, device = "cuda", dtype = torch.float16)
    x, enc, temb, rot = _inputs(blk, 1, 32, 128, "cuda", torch.float16)
    real = wf._fused_forward
    calls = []

    def _oom_once(*args):
        calls.append(1)
        if len(calls) == 1:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return real(*args)

    monkeypatch.setattr(wf, "_fused_forward", _oom_once)
    with torch.no_grad():
        want = WanTransformerBlock.forward(blk, x, enc, temb, rot)
        assert wf.install(torch.float16, "cuda") is True
        got = [blk(x, enc, temb, rot) for _ in range(2)]
    assert all(torch.equal(want, g) for g in got)
    assert wf.counts() == {"fused": 1, "stock": 1}
    assert list(wf._VERIFIED.values()) == [True]


@needs_cuda
def test_self_check_out_of_memory_twice_keeps_stock(monkeypatch):
    blk = _block(dim = 128, ffn = 256, heads = 2, device = "cuda", dtype = torch.float16)
    x, enc, temb, rot = _inputs(blk, 1, 32, 128, "cuda", torch.float16)
    calls = []

    def _always_oom(*args):
        calls.append(1)
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(wf, "_fused_forward", _always_oom)
    with torch.no_grad():
        want = WanTransformerBlock.forward(blk, x, enc, temb, rot)
        assert wf.install(torch.float16, "cuda") is True
        got = [blk(x, enc, temb, rot) for _ in range(4)]
    assert all(torch.equal(want, g) for g in got)
    assert len(calls) == 2 and wf.counts() == {"fused": 0, "stock": 4}
    assert list(wf._VERIFIED.values()) == [False]


@pytest.mark.skipif(
    not _CUDA or torch.cuda.device_count() < 2, reason = "needs two NVIDIA GPUs (non-current device)"
)
def test_kernels_launch_on_the_tensors_device(monkeypatch):
    blk = _block(dim = 128, ffn = 256, heads = 2, device = "cuda:1", dtype = torch.float16)
    x, enc, temb, rot = _inputs(blk, 1, 32, 128, "cuda:1", torch.float16)
    kernels = wf._kernels()
    seen = []
    for name in ("modnorm", "gate_residual", "rope"):
        real = kernels[name]

        def _record(
            *args,
            _real = real,
            **kwargs,
        ):
            seen.append(torch.cuda.current_device())
            return _real(*args, **kwargs)

        monkeypatch.setitem(kernels, name, _record)
    assert torch.cuda.current_device() == 0
    with torch.no_grad():
        want = WanTransformerBlock.forward(blk, x, enc, temb, rot)
        assert wf.install(torch.float16, "cuda:1") is True
        got = blk(x, enc, temb, rot)
    assert torch.equal(want, got) and wf.counts()["fused"] == 1
    assert seen and set(seen) == {1}


@needs_cuda
def test_grad_and_non_fp16_inputs_take_the_stock_path():
    blk = _block(dim = 128, ffn = 256, heads = 2, device = "cuda", dtype = torch.float16)
    x, enc, temb, rot = _inputs(blk, 1, 32, 128, "cuda", torch.float16)
    assert wf.install(torch.float16, "cuda") is True
    with torch.enable_grad():
        xg = x.clone().requires_grad_(True)
        blk(xg, enc, temb, rot)
    assert wf.counts() == {"fused": 0, "stock": 1}
    blk32 = _block(dim = 128, ffn = 256, heads = 2, device = "cuda", dtype = torch.float32)
    x32, enc32, temb32, rot32 = _inputs(blk32, 1, 32, 128, "cuda", torch.float32)
    with torch.no_grad():
        blk32(x32, enc32, temb32, rot32)
    assert wf.counts() == {"fused": 0, "stock": 2}


def _video_src() -> ast.AST:
    from core.inference import video
    return ast.parse(inspect.getsource(video))


def test_loader_installs_before_the_step_cache_and_offload_hooks():
    from core.inference import video

    src = (
        inspect.getsource(video.VideoBackend.load_pipeline)
        if hasattr(video, "VideoBackend")
        else None
    )
    if src is None:
        cls = next(
            c for c in vars(video).values() if inspect.isclass(c) and hasattr(c, "load_pipeline")
        )
        src = inspect.getsource(cls.load_pipeline)
    install = src.index("video_wan_fused.install_for_pipe(")
    assert install < src.index("apply_step_cache(")
    assert install < src.index("apply_memory_plan(")
    head = src[src.rindex("if effective_speed != SPEED_OFF:", 0, install) : install]
    assert "video_wan_fused" in src[:install] and head
    assert '"wan_fused_adaln"' in src


def test_teardown_and_rollback_restore_the_stock_forward():
    from core.inference import video
    cls = next(
        c
        for c in vars(video).values()
        if inspect.isclass(c) and hasattr(c, "_teardown_state_locked")
    )
    for name in ("_teardown_state_locked", "_rollback_precommit_globals"):
        body = ast.unparse(ast.parse(inspect.getsource(getattr(cls, name)).lstrip()))
        assert "video_wan_fused.uninstall()" in body, name


@needs_cuda
def test_fp16_cast_table_and_norm2_stay_bit_identical():
    """A pipeline moved with ``.to(float16)`` casts the float32-kept table and norm2 too; both upcast exactly."""
    D = 128
    blk = _block(dim = D, ffn = 256, heads = 2, device = "cuda", dtype = torch.float16)
    blk.scale_shift_table.data = blk.scale_shift_table.data.half()
    blk.norm2.half()
    x, enc, temb, rot = _inputs(blk, 1, 64, D, "cuda", torch.float16)
    with torch.no_grad():
        want = WanTransformerBlock.forward(blk, x, enc, temb, rot)
        assert wf.install(torch.float16, "cuda") is True
        got = blk(x, enc, temb, rot)
    assert torch.equal(want, got) and wf.counts()["fused"] == 1


@needs_cuda
def test_self_attention_rotary_is_bit_identical_and_scoped(monkeypatch):
    D = 256
    blk = _block(dim = D, ffn = 512, heads = 4, device = "cuda", dtype = torch.float16)
    x, enc, temb, rot = _inputs(blk, 2, 300, D, "cuda", torch.float16)
    assert wf.install(torch.float16, "cuda") is True
    assert wf._STATE.get("attn_call") is transformer_wan.WanAttnProcessor.__call__
    calls = []
    real = wf._kernels()["rope"]
    monkeypatch.setitem(wf._kernels(), "rope", lambda *a: calls.append(1) or real(*a))
    with torch.no_grad():
        want = blk.attn1(x, None, None, rot)
        got = wf._self_attention(blk.attn1, x, rot)
        assert got is not None and torch.equal(want, got) and len(calls) == 2
        # an fp16 table rounds each product to fp16 in the stock path: not this kernel's arithmetic
        assert wf._self_attention(blk.attn1, x, (rot[0].half(), rot[1].half())) is None
        blk.attn1.processor = type(
            "OtherProcessor",
            (transformer_wan.WanAttnProcessor,),
            {"__call__": lambda *a, **k: None},
        )()
        assert wf._self_attention(blk.attn1, x, rot) is None
    monkeypatch.setenv(wf.WAN_FUSED_ROPE_ENV, "0")
    assert wf._self_attention(blk.attn1, x, rot) is None


def test_changed_attention_processor_keeps_the_stock_rotary(monkeypatch):
    monkeypatch.setattr(wf, "_ATTN_FINGERPRINTS", frozenset({"not-this-processor"}))
    _fake_kernels(monkeypatch)
    assert wf.install(torch.float16, "cuda") is True
    assert "attn_call" not in wf._STATE


def test_attention_processor_matches_the_fingerprint():
    from core.inference.diffusion_qwenimage21_rope import _digest
    assert _digest(transformer_wan.WanAttnProcessor.__call__) in wf._ATTN_FINGERPRINTS


def test_regional_compile_traces_the_stock_block_without_recompiles(monkeypatch):
    """Under torch.compile the installed forward must trace the stock block and leave the module counters alone:
    dynamo guards on the globals a traced frame reads, so a bumped counter recompiles the block on every call (it hit
    the recompile limit and dropped the Wan regional compile to eager)."""
    import torch._dynamo as dynamo

    _fake_kernels(monkeypatch)
    blk = _block(dim = 32, ffn = 64, heads = 2, device = "cpu", dtype = torch.float32)
    x, enc, temb, rot = _inputs(blk, 1, 8, 32, "cpu", torch.float32)
    dynamo.reset()
    with torch.no_grad():
        want = WanTransformerBlock.forward(blk, x, enc, temb, rot)
        assert wf.install(torch.float16, "cuda") is True
        blk.compile(backend = "eager", fullgraph = True)
        with dynamo.config.patch(error_on_recompile = True):
            outs = [blk(x, enc, temb, rot) for _ in range(4)]
    dynamo.reset()
    assert all(torch.equal(want, o) for o in outs)
    assert wf.counts() == {"fused": 0, "stock": 0}
