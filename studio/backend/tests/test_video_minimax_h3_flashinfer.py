# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 FlashInfer SM120 block levers (CPU): off by default, refused off sm_120 / without flashinfer >= 0.7.3 /
for streamed or non-fp8/nvfp4 denoisers, exact weight repacks (row orders, NVFP4 scales, shared storage), the stock
forward on calls the operators cannot take, and the release of the replaced stock weights."""

import types

import pytest
import torch

from core.inference import video_minimax_h3_flashinfer as fi

ENVS = (fi.MASTER_ENV, fi.PRE_ENV, fi.OUT_ENV, fi.MLP_ENV, fi.KEEP_STOCK_ENV, fi.NV_Y_MARGIN_ENV)


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in ENVS:
        monkeypatch.delenv(name, raising = False)


def _fake_transformer(n = 2):
    blocks = []
    for _ in range(n):
        blk = torch.nn.Module()
        blk.attn = torch.nn.Module()
        blk.attn.to_q = torch.nn.Linear(8, 8, bias = False)
        blk.forward = lambda *a, **k: "stock"
        blocks.append(blk)
    return types.SimpleNamespace(transformer_blocks = blocks)


def test_off_by_default_touches_nothing():
    t = _fake_transformer()
    before = [b.forward for b in t.transformer_blocks]
    out = fi.install_h3_flashinfer(t)
    assert out["blocks"] == 0 and out["reason"] is None
    assert [b.forward for b in t.transformer_blocks] == before
    assert not any(hasattr(b, "_unsloth_h3fi") for b in t.transformer_blocks)


def test_refused_without_sm120_keeps_stock(monkeypatch):
    monkeypatch.setenv(fi.MASTER_ENV, "1")
    monkeypatch.setattr(
        fi,
        "flashinfer_unavailable",
        lambda device = None: "FlashInfer's MiniMax-H3 operators are sm_120 only",
    )
    monkeypatch.setattr(fi, "_scheme_of", lambda w: "fp8")
    t = _fake_transformer()
    before = [b.forward for b in t.transformer_blocks]
    out = fi.install_h3_flashinfer(t)
    assert out["blocks"] == 0 and "sm_120" in out["reason"]
    assert [b.forward for b in t.transformer_blocks] == before


def test_unavailable_reason_on_this_host():
    if torch.cuda.is_available() and tuple(torch.cuda.get_device_capability()) == (12, 0):
        pytest.skip("an sm_120 host: the answer depends on the installed flashinfer")
    reason = fi.flashinfer_unavailable()
    assert isinstance(reason, str) and reason


@pytest.mark.parametrize("resident,expect", [(False, "resident"), (True, "torchao fp8 / nvfp4")])
def test_streamed_or_bf16_denoiser_is_refused(monkeypatch, resident, expect):
    monkeypatch.setenv(fi.MASTER_ENV, "1")
    monkeypatch.setattr(fi, "flashinfer_unavailable", lambda device = None: None)
    out = fi.install_h3_flashinfer(_fake_transformer(), resident = resident)
    assert out["blocks"] == 0 and expect in out["reason"]


def test_int8_fused_qkv_blocks_are_refused_not_crashed(monkeypatch):
    monkeypatch.setenv(fi.MASTER_ENV, "1")
    monkeypatch.setattr(fi, "flashinfer_unavailable", lambda device = None: None)
    t = _fake_transformer()
    for b in t.transformer_blocks:
        del b.attn.to_q  # Studio's int8 path fuses to_q / to_k / to_v into to_qkv
        b.attn.to_qkv = torch.nn.Linear(8, 24, bias = False)
    out = fi.install_h3_flashinfer(t)
    assert out["blocks"] == 0 and "fp8 / nvfp4" in out["reason"]


def test_every_lever_off_is_a_reason(monkeypatch):
    monkeypatch.setenv(fi.MASTER_ENV, "1")
    for env in (fi.PRE_ENV, fi.OUT_ENV, fi.MLP_ENV):
        monkeypatch.setenv(env, "0")
    monkeypatch.setattr(fi, "flashinfer_unavailable", lambda device = None: None)
    monkeypatch.setattr(fi, "_scheme_of", lambda w: "fp8")
    out = fi.install_h3_flashinfer(_fake_transformer())
    assert out["blocks"] == 0 and "switched off" in out["reason"]


def _stub_install(
    monkeypatch,
    events,
    fail_pack_at = None,
    self_check = 0.01,
):
    monkeypatch.setenv(fi.MASTER_ENV, "1")
    monkeypatch.setattr(fi, "flashinfer_unavailable", lambda device = None: None)
    monkeypatch.setattr(fi, "_scheme_of", lambda w: "fp8")
    monkeypatch.setattr(fi, "register_ops", lambda: True)
    t = _fake_transformer(3)
    pos = {id(b): i for i, b in enumerate(t.transformer_blocks)}

    def pack(blk, scheme, levers):
        if pos[id(blk)] == fail_pack_at:
            raise RuntimeError("pack failed")
        events.append(("pack", pos[id(blk)]))
        return 1e-6, tuple(levers)

    def release(blk, levers):
        events.append(("release", pos[id(blk)]))

    monkeypatch.setattr(fi, "_pack", pack)
    monkeypatch.setattr(
        fi, "_self_check", lambda blk: events.append(("check", pos[id(blk)])) or self_check
    )
    monkeypatch.setattr(fi, "_release_stock", release)
    return t


def test_install_packs_and_releases_one_block_at_a_time(monkeypatch):
    events = []
    t = _stub_install(monkeypatch, events)
    out = fi.install_h3_flashinfer(t)
    assert out["blocks"] == 3 and out["reason"] is None
    # the load-time peak holds one block's copies: each block's stock weights go before the next block is packed
    assert events == [
        ("pack", 0),
        ("check", 0),
        ("release", 0),
        ("pack", 1),
        ("release", 1),
        ("pack", 2),
        ("release", 2),
    ]
    assert all(b.forward.__func__ is fi._fi_block_forward for b in t.transformer_blocks)


def test_install_failed_self_check_releases_nothing(monkeypatch):
    events = []
    t = _stub_install(monkeypatch, events, self_check = 1.0)
    before = [b.forward for b in t.transformer_blocks]
    out = fi.install_h3_flashinfer(t)
    assert out["blocks"] == 0 and "self-check" in out["reason"]
    assert [b.forward for b in t.transformer_blocks] == before
    assert not any(e[0] == "release" for e in events)


def test_install_pack_failure_midway_keeps_the_rest_stock(monkeypatch):
    events = []
    t = _stub_install(monkeypatch, events, fail_pack_at = 2)
    stock2 = t.transformer_blocks[2].forward
    out = fi.install_h3_flashinfer(t)
    assert out["blocks"] == 2 and "block 2" in out["reason"]
    assert t.transformer_blocks[2].forward is stock2
    assert ("release", 2) not in events


@pytest.mark.parametrize(
    "text,ok",
    [
        ("0.7.3", True),
        ("0.7.3rc1", True),
        ("0.7.3+cu130", True),
        ("0.8.0", True),
        ("0.7.2", False),
        ("0.6.6", False),
        ("1.0", True),
    ],
)
def test_version_floor(text, ok):
    assert (fi._version_tuple(text) >= fi.MIN_FLASHINFER) is ok


def _block_stub():
    calls = []

    def stock(*a):
        calls.append(a)
        return "stock-out"

    blk = types.SimpleNamespace(
        _unsloth_h3fi_stock = stock,
        adaln_proj = lambda temb: tuple(torch.zeros(3, 4) for _ in range(6)),
    )
    return blk, calls


def test_unsupported_calls_take_the_stock_forward(monkeypatch):
    monkeypatch.setitem(fi._T, "torch", torch)
    monkeypatch.setitem(fi._CFG, "released", False)
    blk, calls = _block_stub()
    before = fi.CALLS["stock"]
    x = torch.zeros(1, 4, 8, dtype = torch.bfloat16)
    assert (
        fi._fi_block_forward(blk, x, None, None, None, attention_mask = torch.ones(1)) == "stock-out"
    )
    assert fi._fi_block_forward(blk, x.float(), None, None, None) == "stock-out"
    assert len(calls) == 2 and fi.CALLS["stock"] == before + 2


def test_released_blocks_refuse_instead_of_running_without_weights(monkeypatch):
    monkeypatch.setitem(fi._T, "torch", torch)
    monkeypatch.setitem(fi._CFG, "released", True)
    blk, calls = _block_stub()
    with pytest.raises(RuntimeError, match = "UNSLOTH_H3_FLASHINFER=0"):
        fi._fi_block_forward(
            blk,
            torch.zeros(1, 4, 8, dtype = torch.bfloat16),
            None,
            None,
            None,
            attention_mask = torch.ones(1),
        )
    assert not calls


def test_qkv_row_order_is_head_kind_dim():
    rows = torch.arange(3 * fi.HEADS * fi.HEAD_DIM)
    out = fi._qkv_rows(rows.view(-1, 1)).view(-1)
    hd = fi.HEADS * fi.HEAD_DIM
    for h in (0, 7, fi.HEADS - 1):
        for kind in range(3):
            for d in (0, 5, fi.HEAD_DIM - 1):
                assert (
                    out[h * 3 * fi.HEAD_DIM + kind * fi.HEAD_DIM + d]
                    == kind * hd + h * fi.HEAD_DIM + d
                )


def test_fc1_rows_gate_then_value_interleaved_by_eight():
    rows = torch.arange(2 * fi.FFN)  # diffusers order: value rows, then gate rows
    out = fi._fc1_rows(rows.view(-1, 1)).view(-1)
    assert out[:8].tolist() == list(range(fi.FFN, fi.FFN + 8))  # gate of columns 0..7
    assert out[8:16].tolist() == list(range(0, 8))  # value of the same columns
    assert out[16:24].tolist() == list(range(fi.FFN + 8, fi.FFN + 16))
    assert sorted(out.tolist()) == rows.tolist()


def test_fc1_rows_match_flashinfer_interleave():
    mod = pytest.importorskip("flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_fc1_swiglu")
    rows = torch.arange(2 * fi.FFN).view(-1, 1)
    val, gate = rows.chunk(2, 0)
    assert torch.equal(
        fi._fc1_rows(rows), mod.interleave_minimax_h3_fc1_rows_sm120(torch.cat([gate, val], 0))
    )


def _nvfp4(w, pts):
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    return NVFP4Tensor.to_nvfp4(w, per_tensor_scale = pts, is_swizzled_scales = True)


def test_nvfp4_parts_dequantize_exactly_and_reswizzle_to_the_stored_scales():
    pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    g = torch.Generator().manual_seed(0)
    w = torch.randn(256, 512, generator = g) * 0.02
    t = _nvfp4(w, (w.abs().amax() / fi.E2M1_X_E4M3).float())
    codes, scale, pts = fi._nvfp4_parts(t)
    assert codes.shape == (256, 256) and scale.shape == (256, 32)
    assert torch.allclose(
        fi._nvfp4_dequant(codes, scale, pts), t.dequantize(torch.float32), rtol = 0, atol = 1e-8
    )
    assert torch.equal(fi._swizzle(scale), t.scale.contiguous().view(torch.uint8).reshape(-1))


class _Small:
    """H3-shaped block at small sizes (module constants patched): real torchao NVFP4 weights on CPU."""

    def __init__(self, monkeypatch, shared_qkv):
        pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
        monkeypatch.setattr(fi, "HIDDEN", 64)
        monkeypatch.setattr(fi, "HEADS", 2)
        monkeypatch.setattr(fi, "HEAD_DIM", 32)
        monkeypatch.setattr(fi, "FFN", 128)
        g = torch.Generator().manual_seed(3)
        blk = torch.nn.Module()
        blk.norm1, blk.norm2 = torch.nn.RMSNorm(64, eps = 1e-5), torch.nn.RMSNorm(64, eps = 1e-5)
        blk.attn = torch.nn.Module()
        blk.attn.norm_q, blk.attn.norm_k = (
            torch.nn.RMSNorm(32, eps = 1e-5),
            torch.nn.RMSNorm(32, eps = 1e-5),
        )
        blk.ff = torch.nn.Module()
        blk.ff.net = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Identity(), None])
        shapes = {
            "to_q": (64, 64),
            "to_k": (64, 64),
            "to_v": (64, 64),
            "o": (64, 64),
            "fc1": (256, 64),
            "fc2": (64, 128),
        }
        raw = {
            k: torch.randn(*s, generator = g) * (0.05 if k != "to_k" else 0.02)
            for k, s in shapes.items()
        }
        shared = max(raw[k].abs().amax() for k in ("to_q", "to_k", "to_v")) / fi.E2M1_X_E4M3
        self.raw = raw

        def lin(k):
            m = torch.nn.Linear(raw[k].shape[1], raw[k].shape[0], bias = False)
            pts = (
                shared
                if (shared_qkv and k in ("to_q", "to_k", "to_v"))
                else raw[k].abs().amax() / fi.E2M1_X_E4M3
            )
            m.weight = torch.nn.Parameter(_nvfp4(raw[k], pts.float()), requires_grad = False)
            return m

        blk.attn.to_q, blk.attn.to_k, blk.attn.to_v = lin("to_q"), lin("to_k"), lin("to_v")
        blk.attn.to_out = torch.nn.ModuleList([lin("o"), torch.nn.Dropout(0.0)])
        blk.ff.net[0].proj = lin("fc1")
        blk.ff.net[2] = lin("fc2")
        self.blk = blk


@pytest.mark.parametrize("shared_qkv", [True, False])
def test_nvfp4_pack_is_exact_and_pre_needs_one_qkv_scale(monkeypatch, shared_qkv):
    b = _Small(monkeypatch, shared_qkv)
    blk = b.blk
    o_codes, fc2_codes = blk.attn.to_out[0].weight.qdata, blk.ff.net[2].weight.qdata
    eps, took = fi._pack(blk, "nvfp4", ("pre", "out", "mlp"))
    assert eps == 1e-5
    assert took == (("pre", "out", "mlp") if shared_qkv else ("out", "mlp"))
    # out-projection / FC2: the very same storage as the stock weight, scales in FlashInfer's 128x4 layout
    assert fi._buf(blk, "o_q").data_ptr() == o_codes.data_ptr()
    assert fi._buf(blk, "fc2_q").data_ptr() == fc2_codes.data_ptr()
    for name, lin in (("o", blk.attn.to_out[0]), ("fc2", blk.ff.net[2])):
        codes, sc, pts = fi._nvfp4_parts(lin.weight)
        assert torch.equal(fi._buf(blk, f"{name}_s"), fi._swizzle(sc))
        assert fi._HOST[fi._buf(blk, f"{name}_q").data_ptr()] == 1.0 / pts
    # FC1: the same codes, rows permuted
    codes, sc, pts = fi._nvfp4_parts(blk.ff.net[0].proj.weight)
    assert torch.equal(fi._buf(blk, "fc1_q"), fi._fc1_rows(codes))
    assert fi._qkv_shared(blk) is shared_qkv
    if shared_qkv:
        q = torch.cat(
            [fi._nvfp4_parts(getattr(blk.attn, n).weight)[0] for n in ("to_q", "to_k", "to_v")], 0
        )
        assert torch.equal(fi._buf(blk, "qkv_q"), fi._qkv_rows(q))
    else:
        assert not hasattr(blk, "_h3fi_qkv_q")
    # buffers are non-persistent: the checkpoint's state dict is unchanged
    assert not any(k.startswith("_h3fi_") for k in blk.state_dict())
    fi._unpack(blk)
    assert not any(n.startswith("_h3fi_") for n, _ in blk.named_buffers())


def test_release_drops_only_the_replaced_stock_weights(monkeypatch):
    b = _Small(monkeypatch, shared_qkv = False)
    blk = b.blk
    _, took = fi._pack(blk, "nvfp4", ("pre", "out", "mlp"))
    fi._release_stock(blk, took)
    assert blk.attn.to_q.weight.numel() > 0  # pre stayed stock (unshared QKV scales): kept
    for m in (blk.attn.to_out[0], blk.ff.net[0].proj, blk.ff.net[2]):
        assert m.weight.numel() == 0 and m._unsloth_h3fi_released
    assert fi._buf(blk, "o_q").numel() > 0  # the shared storage lives on in the buffer


@pytest.mark.parametrize(
    "backend, engaged",
    [
        (None, True),
        ("native", True),
        ("_native_cudnn", True),
        ("_native_flash", False),
        ("sage", False),
    ],
)
def test_a_selected_attention_backend_the_block_cannot_reproduce_keeps_stock(
    monkeypatch, backend, engaged
):
    events = []
    t = _stub_install(monkeypatch, events)
    if backend is not None:
        for b in t.transformer_blocks:
            b.attn.processor = types.SimpleNamespace(_attention_backend = backend)
    before = [b.forward for b in t.transformer_blocks]
    out = fi.install_h3_flashinfer(t)
    if engaged:
        assert out["blocks"] == 3 and out["reason"] is None
    else:
        assert out["blocks"] == 0 and repr(backend) in out["reason"]
        assert [b.forward for b in t.transformer_blocks] == before and events == []


_NINJA_NO_NVCC = (
    "Ninja build failed. Ninja output:\n"
    "ninja: Entering directory `/root/.cache/flashinfer/0.7.3/120a/cached_ops/minimax_h3_sm120_quant_out_proj'\n"
    "[1/2]  /usr/local/cuda/bin/nvcc --generate-dependencies-with-compile " + "-I/x " * 200 + "\n"
    "FAILED: [code=127] /root/.cache/flashinfer/0.7.3/120a/cached_ops/out_proj.cuda.o\n"
    " /usr/local/cuda/bin/nvcc --generate-dependencies-with-compile " + "-I/x " * 200 + "\n"
    "/bin/sh: 1: /usr/local/cuda/bin/nvcc: not found\n"
    "ninja: build stopped: subcommand failed.\n"
)


@pytest.mark.parametrize(
    "exc,expect",
    [
        (RuntimeError(_NINJA_NO_NVCC), "nvcc: not found"),
        (FileNotFoundError(2, "No such file or directory", "ninja"), "ninja"),
        (
            RuntimeError(
                "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist"
            ),
            "Could not find nvcc",
        ),
    ],
)
def test_a_missing_jit_toolchain_keeps_stock_with_a_one_line_reason(monkeypatch, exc, expect):
    events = []
    t = _stub_install(monkeypatch, events)

    def self_check(blk):
        raise exc

    monkeypatch.setattr(fi, "_self_check", self_check)
    before = [b.forward for b in t.transformer_blocks]
    out = fi.install_h3_flashinfer(t)
    assert out["blocks"] == 0 and [b.forward for b in t.transformer_blocks] == before
    reason = out["reason"]
    assert expect in reason and "\n" not in reason and len(reason) <= 400
    assert "nvcc >= 12.8" in reason and "--generate-dependencies" not in reason
    assert not any(e[0] == "release" for e in events)


@pytest.mark.parametrize("resident,swap", [(False, False), (True, True)])
def test_streamed_or_int8_denoiser_never_probes_flashinfer(monkeypatch, resident, swap):
    monkeypatch.setenv(fi.MASTER_ENV, "1")

    def probe(device = None):
        raise AssertionError("flashinfer probed for a denoiser the levers refuse anyway")

    monkeypatch.setattr(fi, "flashinfer_unavailable", probe)
    if not swap:
        monkeypatch.setattr(fi, "_scheme_of", lambda w: "fp8")
    out = fi.install_h3_flashinfer(_fake_transformer(), resident = resident)
    assert out["blocks"] == 0 and out["reason"]


@pytest.mark.parametrize("margin", ["0", "-2", "nan", "inf", "x"])
def test_a_bad_y_margin_falls_back_to_the_default(monkeypatch, margin):
    events = []
    t = _stub_install(monkeypatch, events)
    monkeypatch.setenv(fi.NV_Y_MARGIN_ENV, margin)
    fi.install_h3_flashinfer(t)
    assert fi._CFG["y_margin"] == 4.0
