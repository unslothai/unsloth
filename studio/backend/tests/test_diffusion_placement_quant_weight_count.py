# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Placement prices a quantized denoiser's forward from its logical weights, not from ``parameters()``. CPU only."""

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from core.inference import diffusion_multigpu as mg  # noqa: E402
from core.inference import diffusion_native_quant as nq  # noqa: E402


class Flux2Transformer2DModel(nn.Module):
    """A klein-shaped stand-in: big linears in blocks, a norm, a float table buffer."""

    def __init__(
        self,
        d = 256,
        n = 3,
    ):
        super().__init__()
        self.x_embedder = nn.Linear(32, d)
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {"norm": nn.LayerNorm(d), "qkv": nn.Linear(d, 3 * d), "ff": nn.Linear(d, 4 * d)}
            )
            for _ in range(n)
        )
        self.register_buffer("rope", torch.randn(64, 16))


def _dense_count(module):
    return sum(p.numel() for p in module.parameters())


def _swap_linears(model, build):
    for name in [
        n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n != "x_embedder"
    ]:
        parent, _, leaf = name.rpartition(".")
        owner = model.get_submodule(parent)
        setattr(owner, leaf, build(getattr(owner, leaf)))
    return model


def _native(scheme, **kw):
    cls = nq.native_linear_class()
    return _swap_linears(
        Flux2Transformer2DModel().to(torch.bfloat16), lambda lin: cls(lin, scheme, **kw)
    )


def test_dense_denoiser_reads_exactly_as_before():
    model = Flux2Transformer2DModel()
    assert mg.logical_weight_elements(model) == _dense_count(model)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: _native(nq.NATIVE_INT8), id = "native-int8"),
        pytest.param(
            lambda: _native(nq.NATIVE_INT8, act_int8 = True, rot_group = 256),
            id = "native-int8-w8a8-convrot",
        ),
        pytest.param(lambda: _native(nq.NATIVE_FP8), id = "native-fp8"),
    ],
)
def test_buffered_quant_linears_count_their_logical_weights(build):
    dense = _dense_count(Flux2Transformer2DModel())
    model = build()
    # parameters() alone sees only the norms, biases and the dense embedder
    assert _dense_count(model) < 0.05 * dense
    assert mg.logical_weight_elements(model) == dense


def test_packed_4bit_codes_count_at_their_logical_size():
    """GGUF / bnb 4-bit pack two weights per stored element; a fused module's int8 codes count once each."""
    dense = _dense_count(Flux2Transformer2DModel())

    class Packed4(nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.in_features, self.out_features = lin.in_features, lin.out_features
            self.weight = nn.Parameter(
                torch.zeros(lin.out_features * lin.in_features // 2, dtype = torch.uint8),
                requires_grad = False,
            )
            self.bias = lin.bias

    model = _swap_linears(Flux2Transformer2DModel(), Packed4)
    assert _dense_count(model) < 0.6 * dense
    assert mg.logical_weight_elements(model) == dense

    class FusedCodes(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("codes", torch.zeros(64, 32, dtype = torch.int8))
            self.register_buffer("scale", torch.ones(64))

    assert mg.logical_weight_elements(FusedCodes()) == 64 * 32


class _Plan:
    offload_policy = "group"
    stream_transformer = True
    resident_transformer_mib = 0


def _price(monkeypatch, denoiser, streamed_mib, tokens):
    """replica_layout against a layer split on two idle T4s at the Kaggle-measured rates."""
    pipe = type("Pipe", (), {"transformer": denoiser, "transformer_2": None})()
    mg.note_default_request(pipe, guided = True, steps = 20, tokens = tokens, why = "default guidance 4")
    monkeypatch.setattr(mg, "_replica_card", lambda selection, need, **k: (1, "same-model card"))
    monkeypatch.setattr(mg, "replica_need_mib", lambda m: 4000)
    monkeypatch.setattr(mg, "_gemm_tflops", lambda ordinal, dtype: 46.5)
    import core.inference.diffusion_bg_compile as bg
    import core.inference.diffusion_memory as mem

    monkeypatch.setattr(mem, "_offload_cost_measurements", lambda: (10.2, 46.5))
    monkeypatch.setattr(bg, "load_time_enabled", lambda: False)
    monkeypatch.delenv(mg.CFG_REPLICA_PLACEMENT_ENV, raising = False)
    selection = mg.MultiGpuSelection(primary = 0, secondaries = (1,))
    sizes = {"transformer": (streamed_mib, "dit")}
    won, lost = mg.replica_layout(pipe, _Plan(), sizes, selection, alternative = "layer_split")
    return won, lost, mg._MODE_ESTIMATES.get(pipe)


def test_int8_denoiser_streaming_on_the_primary_is_priced_at_its_real_forward(monkeypatch):
    """klein-base-4B INT8 on 2x T4: parameters() priced a forward at 0.03 s (wrongly favoring a split); ~0.68 s is real."""
    model = _native(nq.NATIVE_INT8, act_int8 = True, rot_group = 256)
    # klein-base-4B's 3.88B weights x 4096 tokens, carried by this small stand-in as more tokens
    tokens = 3_877_455_360 * 4096 // _dense_count(Flux2Transformer2DModel())
    won, lost, est = _price(monkeypatch, model, 3876, tokens)
    assert won is not None and lost is None, lost
    assert est["chosen"] == "cfg_replica"
    assert 0.6 < est["compute_s"] < 0.8


def test_quantized_and_dense_denoisers_of_one_shape_price_alike(monkeypatch):
    """The decision depends on the weights a forward multiplies, not on how they are stored."""
    dense = Flux2Transformer2DModel().to(torch.bfloat16)
    quant = _native(nq.NATIVE_INT8)
    tokens = 3_877_455_360 * 4096 // _dense_count(dense)
    _w, _l, dense_est = _price(monkeypatch, dense, 3876, tokens)
    _w, _l, quant_est = _price(monkeypatch, quant, 3876, tokens)
    assert quant_est["compute_s"] == dense_est["compute_s"]
    assert quant_est["chosen"] == dense_est["chosen"]
