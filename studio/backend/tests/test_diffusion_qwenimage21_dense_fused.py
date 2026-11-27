# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 dense SwiGLU gate / value GEMM fusion (CPU stand-ins with the diffusers class names)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from core.inference import diffusion_qwenimage21_dense_fused as dfu  # noqa: E402


class QwenImage21SwiGLUFeedForward(nn.Module):
    """diffusers' module, same attributes and forward."""

    def __init__(self, hidden_size: int, mlp_hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, mlp_hidden_size, bias = False)
        self.out = nn.Linear(mlp_hidden_size, hidden_size, bias = False)
        self.gate_layer = nn.Linear(hidden_size, mlp_hidden_size, bias = False)
        self.activation_fn = nn.SiLU()

    def forward(self, hidden_states):
        return self.out(
            self.activation_fn(self.gate_layer(hidden_states)) * self.proj(hidden_states)
        )


class QwenImage21Transformer2DModel(nn.Module):
    def __init__(
        self,
        n: int = 2,
        dim: int = 32,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(QwenImage21SwiGLUFeedForward(dim, 3 * dim) for _ in range(n))

    def forward(self, x):
        for b in self.blocks:
            x = x + b(x)
        return x


@pytest.fixture(autouse = True)
def _cpu(monkeypatch):
    monkeypatch.setattr(dfu, "_DEVICE_TYPES", ("cpu",))
    monkeypatch.delenv(dfu.Q21_DENSE_FUSED_ENV, raising = False)
    torch.manual_seed(0)


def _model(dtype = torch.float32):
    return QwenImage21Transformer2DModel().to(dtype).eval()


def test_fused_matches_stock_and_shares_storage():
    m = _model()
    x = torch.randn(3, 7, 32)
    with torch.no_grad():
        ref = m(x)
    assert dfu.install(m)["gate_up"] == 2
    for b in m.blocks:
        fused = b.__dict__[dfu._FF_ATTR][0]
        assert (
            b.gate_layer.weight.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr()
        )
        assert b.proj.weight.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr()
        assert fused.shape == (2 * 96, 32)
    with torch.no_grad():
        out = m(x)
    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-5)
    # no second parameter set: the model's parameter bytes are unchanged
    assert sum(p.numel() for p in m.parameters()) == 2 * (3 * 96 * 32)


def test_bf16_matches_stock_within_one_rounding():
    m = _model(torch.bfloat16)
    x = torch.randn(2, 5, 32, dtype = torch.bfloat16)
    with torch.no_grad():
        ref = m(x).float()
    dfu.install(m)
    with torch.no_grad():
        out = m(x).float()
    torch.testing.assert_close(out, ref, rtol = 2e-2, atol = 2e-2)


def test_in_place_weight_write_reaches_the_fused_gemm():
    m = _model()
    x = torch.randn(2, 4, 32)
    dfu.install(m)
    with torch.no_grad():
        m.blocks[0].gate_layer.weight.add_(0.5)  # a LoRA fold / restore writes in place
        out = m(x)
        dfu.uninstall(m)
        ref = m(x)
    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-5)


def test_swapped_parts_fall_back_to_the_stock_forward():
    m = _model()
    x = torch.randn(2, 4, 32)
    dfu.install(m)
    b = m.blocks[0]
    b.gate_layer = nn.Linear(32, 96, bias = False)  # e.g. an adapter wrapper replacing the module
    assert dfu._intact(b) is None
    with torch.no_grad():
        got = b(x)
        want = b.out(b.activation_fn(b.gate_layer(x)) * b.proj(x))
    torch.testing.assert_close(got, want)
    b2 = m.blocks[1]
    b2.proj.weight = nn.Parameter(
        b2.proj.weight.detach().clone(), requires_grad = False
    )  # a new parameter
    assert dfu._intact(b2) is None


@pytest.mark.parametrize("why", ["kill", "offload", "bias", "hook", "subclass", "other_model"])
def test_declines(monkeypatch, why):
    m = _model()
    b = m.blocks[0]
    kw = {}
    if why == "kill":
        monkeypatch.setenv(dfu.Q21_DENSE_FUSED_ENV, "0")
    elif why == "offload":
        kw["offload_active"] = True
    elif why == "bias":
        b.gate_layer.bias = nn.Parameter(torch.zeros(96))
    elif why == "hook":
        b.proj.register_forward_pre_hook(lambda mod, args: None)
    elif why == "subclass":

        class QuantWeight(torch.Tensor):
            pass

        b.proj.weight = nn.Parameter(
            b.proj.weight.detach().as_subclass(QuantWeight), requires_grad = False
        )
    elif why == "other_model":
        m.__class__ = type(
            "FluxTransformer2DModel",
            (nn.Module,),
            {"forward": QwenImage21Transformer2DModel.forward},
        )
    n = dfu.install(m, **kw)
    if why in ("kill", "offload", "other_model"):
        assert not any(n.values())
        assert not any(dfu.is_installed(x) for x in m.modules())
    else:
        assert n["gate_up"] == 1  # the other block still fuses
        assert not dfu.is_installed(b)


def test_weights_off_the_gpu_defer_to_the_first_call(monkeypatch):
    m = _model()
    monkeypatch.setattr(dfu, "_DEVICE_TYPES", ("cuda",))
    assert dfu.install(m) is None
    assert not any(dfu.is_installed(x) for x in m.modules())
    monkeypatch.setattr(dfu, "_DEVICE_TYPES", ("cpu",))  # "placed" before the first forward
    with torch.no_grad():
        m(torch.randn(1, 2, 32))
    assert all(dfu.is_installed(b) for b in m.blocks)


def test_uninstall_restores_the_stock_forward():
    m = _model()
    dfu.install(m)
    dfu.uninstall(m)
    for b in m.blocks:
        assert not dfu.is_installed(b)
        assert "forward" not in b.__dict__


def test_compiles_without_graph_breaks():
    m = _model()
    dfu.install(m)
    torch._dynamo.reset()
    try:
        explain = torch._dynamo.explain(m.blocks[0])(torch.randn(2, 4, 32))
    finally:
        torch._dynamo.reset()
    assert explain.graph_break_count == 0
    assert explain.graph_count == 1


def test_qkv_records_go_through_the_shared_helper(monkeypatch):
    from core.inference import diffusion_qwenimage21_fused as q21f

    class QwenImage21Attention(nn.Module):
        def __init__(self, dim = 32):
            super().__init__()
            self.to_q = nn.Linear(dim, dim, bias = False)
            self.to_k = nn.Linear(dim, dim, bias = False)
            self.to_v = nn.Linear(dim, dim, bias = False)

    m = _model()
    m.attn = QwenImage21Attention()
    monkeypatch.setattr(q21f, "_patch_prepare", lambda logger = None: True)
    out = dfu.install(m)
    assert out == {"gate_up": 2, "qkv": 1}
    fused = q21f._qkv_intact(m.attn)
    assert fused is not None and fused.weight.shape == (96, 32)
    x = torch.randn(2, 3, 32)
    with torch.no_grad():
        q, k, v = fused(x).chunk(3, dim = -1)
        torch.testing.assert_close(q, m.attn.to_q(x))
        torch.testing.assert_close(v, m.attn.to_v(x))
    dfu.uninstall(m)
    assert q21f._qkv_intact(m.attn) is None
    # the int8 path's kill switch also keeps Q / K / V stock
    monkeypatch.setenv(q21f.Q21_CONVROT_FUSED_ENV, "0")
    m2 = _model()
    m2.attn = QwenImage21Attention()
    assert dfu.install(m2) == {"gate_up": 2, "qkv": 0}


def test_real_tiny_qwen21_stock_vs_fused_eager_and_compiled():
    qmod = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")
    from core.inference import diffusion_qwenimage21_fused as q21f

    torch.manual_seed(0)
    m = qmod.QwenImage21Transformer2DModel(
        num_layers = 2,
        attention_head_dim = 16,
        num_attention_heads = 2,
        context_in_dim = 24,
        in_channels = 8,
        out_channels = 8,
        axes_dims_rope = (4, 6, 6),
    ).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.2)
    g = torch.Generator().manual_seed(7)
    text, (th, tw) = 9, (4, 6)
    img_mask = torch.cat(
        [torch.zeros(text, dtype = torch.bool), torch.ones(th * tw // 4, dtype = torch.bool)]
    )[None]
    ehs = torch.randn(1, text, 24, generator = g)
    lat0 = torch.randn(1, th * tw, 8, generator = g)

    def _render(steps = 3):
        outs, lat = [], lat0
        kv = qmod.QwenImage21KVCache(len(m.transformer_blocks))
        with torch.inference_mode():
            for i in range(steps):
                out = m(
                    hidden_states = lat,
                    timestep = torch.full((1,), 1.0 - i / steps),
                    encoder_hidden_states = ehs,
                    encoder_hidden_states_mask = None,
                    img_shapes = [[(1, th, tw)]],
                    img_mask = img_mask,
                    kv_cache = kv,
                    kv_cache_mode = "extract" if i == 0 else "cached",
                    return_dict = False,
                )[0]
                outs.append(out.clone())
                lat = lat - 0.1 * out[:, -lat.shape[1] :]
        return outs

    ref = _render()
    try:
        out = dfu.install(m)
        assert out["gate_up"] == len(m.transformer_blocks)
        assert out["qkv"] == len(m.transformer_blocks)
        eager = _render()
        for a, b in zip(ref, eager):
            torch.testing.assert_close(a, b, rtol = 1e-5, atol = 1e-5)
        hits = []
        for mod in m.modules():
            fused = q21f._qkv_intact(mod) if type(mod).__name__ == "QwenImage21Attention" else None
            if fused is not None:
                fused.register_forward_pre_hook(lambda *_a: hits.append(1))
        torch._dynamo.reset()
        m.compile_repeated_blocks(backend = "eager", fullgraph = True, dynamic = False)
        compiled = _render()
        assert hits, "the compiled blocks never ran the fused Q / K / V projection"
        for a, b in zip(ref, compiled):
            torch.testing.assert_close(a, b, rtol = 1e-5, atol = 1e-5)
    finally:
        torch._dynamo.reset()
        dfu.uninstall(m)
        q21f.uninstall()
    assert qmod._qwenimage21_prepare_qkv.__name__ == "_qwenimage21_prepare_qkv"


def test_cfg_replica_drops_source_records_and_fuses_its_own(monkeypatch):
    from core.inference import diffusion_multigpu as mg
    from core.inference import diffusion_qwenimage21_fused as q21f

    class QwenImage21Attention(nn.Module):
        def __init__(self, dim = 32):
            super().__init__()
            self.to_q = nn.Linear(dim, dim, bias = False)
            self.to_k = nn.Linear(dim, dim, bias = False)
            self.to_v = nn.Linear(dim, dim, bias = False)

    monkeypatch.setattr(q21f, "_patch_prepare", lambda logger = None: True)
    m = _model()
    m.attn = QwenImage21Attention()
    assert dfu.install(m) == {"gate_up": 2, "qkv": 1}
    src_ptrs = {b.__dict__[dfu._FF_ATTR][0].untyped_storage().data_ptr() for b in m.blocks}
    replica = mg._copy_module(m, torch.device("cpu"))
    for b in replica.blocks:
        rec = b.__dict__[dfu._FF_ATTR]
        ptr = rec[0].untyped_storage().data_ptr()
        assert ptr not in src_ptrs  # not the source's fused copy
        assert b.gate_layer.weight.untyped_storage().data_ptr() == ptr  # the replica's own views
        assert dfu._intact(b) is not None
    fused = q21f._qkv_intact(replica.attn)
    assert fused is not None
    assert (
        replica.attn.to_q.weight.untyped_storage().data_ptr()
        == fused.weight.untyped_storage().data_ptr()
    )
    x = torch.randn(2, 3, 32)
    with torch.no_grad():
        torch.testing.assert_close(replica(x), m(x), rtol = 1e-5, atol = 1e-5)


def test_cfg_replica_rebuild_frees_the_copied_parts(monkeypatch):
    import gc
    import weakref

    from core.inference import diffusion_multigpu as mg

    m = _model()
    dfu.install(m)
    alive_after_rebuild = []
    orig = dfu.rebuild_replica

    def spy(source, replica):
        refs = [
            weakref.ref(b.gate_layer.weight) for b in replica.blocks
        ]  # the copies the deepcopy made
        out = orig(source, replica)
        gc.collect()
        # still inside _copy_module: nothing but its memo could keep the replaced copies alive here
        alive_after_rebuild.extend(r() is not None for r in refs)
        return out

    monkeypatch.setattr(dfu, "rebuild_replica", spy)
    replica = mg._copy_module(m, torch.device("cpu"))
    assert alive_after_rebuild == [False] * len(replica.blocks)
    assert all(dfu._intact(b) is not None for b in replica.blocks)


# Native int8 layers with int8 activations (the W8A8 linear on ROCm RDNA): same fusion over int8 codes.


def _native(
    linear,
    act_int8 = True,
    group = 16,
):
    from core.inference.diffusion_native_quant import NATIVE_INT8, native_linear_class
    return native_linear_class()(linear, NATIVE_INT8, act_int8 = act_int8, rot_group = group)


class QwenImage21Attention(nn.Module):
    def __init__(
        self,
        dim = 64,
        bias = True,
    ):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias = bias)
        self.to_k = nn.Linear(dim, dim, bias = bias)
        self.to_v = nn.Linear(dim, dim, bias = bias)


def _native_model(
    act_int8 = True,
    dim = 64,
    group = 16,
):
    m = QwenImage21Transformer2DModel(dim = dim).to(torch.bfloat16).eval()
    m.attn = QwenImage21Attention(dim).to(torch.bfloat16)
    for b in m.blocks:
        for name in ("gate_layer", "proj", "out"):
            setattr(b, name, _native(getattr(b, name), act_int8, group))
    for name in ("to_q", "to_k", "to_v"):
        setattr(m.attn, name, _native(getattr(m.attn, name), act_int8, group))
    return m


def _int8_rows(lin, x):
    """The W8A8 maths the GPU runs (rotation, row quant, int32 GEMM, row x column scale epilogue)."""
    x2 = x.reshape(-1, lin.in_features)
    if lin.rot_group:
        x2 = lin._rotate(x2)
    return lin._int_mm_rows(x2)


def test_native_int8_fusion_is_exact_and_shares_the_codes(monkeypatch):
    from core.inference import diffusion_qwenimage21_fused as q21f

    monkeypatch.setattr(q21f, "_patch_prepare", lambda logger = None: True)
    m = _native_model()
    x = torch.randn(40, 64, dtype = torch.bfloat16)
    with torch.no_grad():
        ref_ff = [
            torch.cat([_int8_rows(b.gate_layer, x), _int8_rows(b.proj, x)], -1) for b in m.blocks
        ]
        ref_qkv = torch.cat(
            [_int8_rows(getattr(m.attn, n), x) for n in ("to_q", "to_k", "to_v")], -1
        )
        ref_fwd = [b(x) for b in m.blocks]
    codes_before = sum(t.numel() for t in m.buffers() if t.dtype == torch.int8)
    assert dfu.install(m) == {"gate_up": 2, "qkv": 1}
    with torch.no_grad():
        for b, want, want_fwd in zip(m.blocks, ref_ff, ref_fwd):
            fused = dfu._intact(b)
            assert fused is not None and fused.out_features == 2 * 192
            assert (
                b.gate_layer.weight_q.untyped_storage().data_ptr()
                == fused.weight_q.untyped_storage().data_ptr()
            )
            assert (
                b.proj.weight_q.untyped_storage().data_ptr()
                == fused.weight_q.untyped_storage().data_ptr()
            )
            assert torch.equal(_int8_rows(fused, x), want)
            assert torch.equal(b(x), want_fwd)
        qkv = q21f._qkv_intact(m.attn)
        assert qkv is not None and qkv.out_features == 3 * 64
        assert torch.equal(_int8_rows(qkv, x), ref_qkv)
        assert torch.equal(
            qkv.bias.float(),
            torch.cat([m.attn.to_q.bias, m.attn.to_k.bias, m.attn.to_v.bias]).float(),
        )
    assert sum(t.numel() for t in m.buffers() if t.dtype == torch.int8) == codes_before
    dfu.uninstall(m)
    assert q21f._qkv_intact(m.attn) is None and dfu._intact(m.blocks[0]) is None


def test_native_int8_reloaded_part_falls_back(monkeypatch):
    from core.inference import diffusion_qwenimage21_fused as q21f

    monkeypatch.setattr(q21f, "_patch_prepare", lambda logger = None: True)
    m = _native_model()
    dfu.install(m)
    b = m.blocks[0]
    b.proj.weight_q = (
        b.proj.weight_q.clone()
    )  # load_state_dict(assign=True) / a move replaced the buffer
    assert dfu._intact(b) is None
    x = torch.randn(3, 64, dtype = torch.bfloat16)
    with torch.no_grad():
        torch.testing.assert_close(
            b(x), b.out(b.activation_fn(b.gate_layer(x)) * b.proj(x)), rtol = 0, atol = 0
        )
    m.attn.to_k.weight_q = m.attn.to_k.weight_q.clone()
    assert q21f._qkv_intact(m.attn) is None


@pytest.mark.parametrize("why", ["weight_only", "mixed", "hook", "group"])
def test_native_int8_declines(monkeypatch, why):
    from core.inference import diffusion_qwenimage21_fused as q21f

    monkeypatch.setattr(q21f, "_patch_prepare", lambda logger = None: True)
    m = _native_model(act_int8 = why != "weight_only")
    b = m.blocks[0]
    if why == "mixed":
        b.proj = nn.Linear(64, 192, bias = False).to(torch.bfloat16)
        m.attn.to_v = nn.Linear(64, 64).to(torch.bfloat16)
    elif why == "hook":
        b.proj.register_forward_hook(lambda *a: None)
        m.attn.to_v.register_forward_pre_hook(lambda *a: None)
    elif why == "group":
        b.proj = _native(nn.Linear(64, 192, bias = True).to(torch.bfloat16), True, 64)
        m.attn.to_v = _native(nn.Linear(64, 64).to(torch.bfloat16), True, 64)
    out = dfu.install(m)
    assert dfu._intact(b) is None
    assert q21f._qkv_intact(m.attn) is None
    if why == "weight_only":
        assert out == {"gate_up": 0, "qkv": 0}
    else:
        assert out == {"gate_up": 1, "qkv": 0}


def test_native_int8_compiles_without_graph_breaks():
    m = _native_model()
    dfu.install(m)
    torch._dynamo.reset()
    try:
        explain = torch._dynamo.explain(m.blocks[0])(torch.randn(2, 4, 64, dtype = torch.bfloat16))
    finally:
        torch._dynamo.reset()
    assert explain.graph_break_count == 0
    assert explain.graph_count == 1
