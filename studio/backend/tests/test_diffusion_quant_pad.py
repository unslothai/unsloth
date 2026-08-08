# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the small-M activation padding (``diffusion_quant_pad.py``).

Real torch, CPU only: the padding is a shape transform plus a slice, so a dense Linear proves
every structural property (shape, pad-row content, state-dict transparency, attribute
passthrough). The granularity gate is exercised against FAKE torchao weight layouts, so the
tests pin the ATTRIBUTES the probe reads rather than needing torchao installed. One CUDA-gated
test closes the loop on a genuinely int8-quantized Linear.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from core.inference.diffusion_quant_pad import (  # noqa: E402
    DEFAULT_PAD_TO,
    INT_MM_MIN_M,
    PadToMinM,
    activation_granularity_is_per_row,
    is_quantized_linear,
    matching_linear_fqns,
    padding_is_bitwise_exact,
    wrap_small_m_linears,
)


class _RecordingLinear(nn.Linear):
    """A Linear that remembers the activation it was handed, so the pad rows are inspectable."""

    def forward(self, x):
        self.seen = x.detach().clone()
        return super().forward(x)


def _fake_quant_tensor_type(**class_attrs):
    """A stand-in for a torchao weight subclass.

    The marker the probe looks for is ``__tensor_flatten__``, which every torchao tensor subclass
    defines. The granularity attributes are set on the TYPE rather than on the instance because
    ``nn.Parameter(subclass_tensor)`` returns ``tensor.detach()`` -- a fresh Python object of the
    same type -- so an instance attribute would not survive the assignment the real quantiser
    makes either."""

    def __tensor_flatten__(self):  # pragma: no cover - presence is the whole point
        return ["qdata"], None

    return type(
        "_FakeQuantTensor",
        (torch.Tensor,),
        {"__tensor_flatten__": __tensor_flatten__, **class_attrs},
    )


def _fake_quantized_linear(in_f = 8, out_f = 6, *, per_row = True, unknown = False):
    """A Linear whose weight advertises a torchao layout with a chosen activation granularity."""
    if unknown:
        attrs = {}
    elif per_row:
        attrs = {
            "input_quant_func": types.SimpleNamespace(
                __name__ = "_int8_symm_per_token_reduced_range_quant"
            )
        }
    else:
        attrs = {"input_quant_func": types.SimpleNamespace(__name__ = "_int8_symm_per_tensor_quant")}
    lin = nn.Linear(in_f, out_f)
    lin.weight = nn.Parameter(
        lin.weight.data.as_subclass(_fake_quant_tensor_type(**attrs)), requires_grad = False
    )
    return lin


# ── shape and value preservation ──────────────────────────────────────────────


@pytest.mark.parametrize("m", [1, 5, 10, 16, 17, 19, 64])
@pytest.mark.parametrize("lead", [(), (2,), (2, 3)])
def test_padding_returns_the_unpadded_result(m, lead):
    """The rows the caller asked for come back unchanged, at the caller's leading dims.

    The reference is the same module without the wrapper, so any difference is the padding's
    fault and nothing else's. Note the tolerance: this is a DENSE float Linear, where a taller
    GEMM can pick a different BLAS path and reassociate the accumulation (torch routes a single
    row through addmv and 32 rows through addmm). The exactness claim belongs to the int8 path,
    where the accumulation is integer and therefore order-independent -- see
    ``test_int8_padding_is_bitwise_exact_on_a_real_quantized_linear``."""
    torch.manual_seed(0)
    inner = nn.Linear(8, 6)
    wrapped = PadToMinM(inner, min_m = INT_MM_MIN_M, pad_to = DEFAULT_PAD_TO)
    x = torch.randn(*lead, m, 8)
    with torch.no_grad():
        got, want = wrapped(x), inner(x)
    assert got.shape == (*lead, m, 6) == want.shape
    assert torch.allclose(got, want, rtol = 0, atol = 1e-6)


def test_pad_rows_replicate_row_zero_rather_than_being_zeros():
    """An all-zero pad row has amax 0, so the activation quantizer divides by zero and the
    intermediate goes NaN. Replicating row 0 costs the same and keeps it finite -- and it is
    also what makes a per-tensor AMAX invariant, since it introduces no new element values."""
    inner = _RecordingLinear(8, 6)
    wrapped = PadToMinM(inner, min_m = 17, pad_to = 32)
    x = torch.randn(5, 8)
    wrapped(x)
    seen = inner.seen
    assert seen.shape == (32, 8), "activation must reach the GEMM at exactly pad_to rows"
    assert torch.equal(seen[:5], x), "the caller's rows must be untouched"
    for row in range(5, 32):
        assert torch.equal(seen[row], x[0]), f"pad row {row} is not a copy of row 0"
    assert not (seen[5:] == 0).all(), "pad rows must not be zeros"


def test_every_small_activation_normalises_to_one_row_count():
    """Below ``pad_to`` every activation reaches the GEMM at exactly ``pad_to`` rows, so one
    inductor graph covers every prompt length in the range rather than one graph per length.
    H3's seven eval prompts run at M = 10, 13, 13, 13, 14, 17, 19 -- which straddles the floor,
    so padding only up to ``min_m`` would leave three distinct shapes behind."""
    inner = _RecordingLinear(8, 6)
    wrapped = PadToMinM(inner, min_m = 17, pad_to = 32)
    for m in (1, 10, 13, 14, 16, 17, 19, 31):
        wrapped(torch.randn(m, 8))
        assert inner.seen.shape == (32, 8), f"M = {m} did not normalise to 32 rows"


def test_no_padding_at_or_above_pad_to():
    """A module that is small on one call and large on the next must pay nothing on the large
    one: at or above ``pad_to`` the activation reaches the GEMM at its own row count."""
    inner = _RecordingLinear(8, 6)
    wrapped = PadToMinM(inner, min_m = 17, pad_to = 32)
    wrapped(torch.randn(32, 8))
    assert inner.seen.shape == (32, 8)
    wrapped(torch.randn(4096, 8))
    assert inner.seen.shape == (4096, 8)


def test_zero_rows_return_the_projected_width():
    """torchao hands a zero-row activation back UNPROJECTED, so a downstream width-sensitive add
    crashes. The wrapper has no row 0 to replicate either, so it synthesises the empty result."""
    inner = _RecordingLinear(8, 6)
    wrapped = PadToMinM(inner)
    out = wrapped(torch.randn(0, 8))
    assert out.shape == (0, 6), "an empty activation must still come back 6 wide, not 8"
    assert not hasattr(inner, "seen"), "the inner GEMM must not be called with zero rows"


def test_pad_to_can_never_sit_below_min_m():
    """``pad_to`` exists to buy tiling and shape stability above the floor, never to undercut
    it: a 'padded' activation below ``min_m`` would still trip the assert it exists to clear."""
    inner = nn.Linear(8, 6)
    assert PadToMinM(inner, min_m = 17, pad_to = 8).pad_to == 17
    assert PadToMinM(inner, min_m = 17, pad_to = None).pad_to == 17
    assert PadToMinM(inner, min_m = 17, pad_to = 32).pad_to == 32


def test_forward_holds_no_mutable_integer_state():
    """dynamo guards on an nn.Module's integer attributes, so a counter incremented in forward
    would recompile on every call until the recompile limit silently drops the module to eager.
    The wrapper's ints must be exactly the two configured constants, before and after use."""
    def int_attrs(module):
        return {
            k: v
            for k, v in vars(module).items()
            if isinstance(v, int) and not isinstance(v, bool)
        }

    wrapped = PadToMinM(nn.Linear(8, 6), min_m = 17, pad_to = 32)
    before = int_attrs(wrapped)
    for m in (3, 3, 40, 3):
        wrapped(torch.randn(m, 8))
    assert before == int_attrs(wrapped) == {"min_m": 17, "pad_to": 32}


# ── drop-in transparency ──────────────────────────────────────────────────────


def test_attributes_pass_through_to_the_inner_linear():
    """diffusers' attention processors read ``to_q.weight.dtype`` and H3's blocks read
    ``context_embedder.weight``; without the passthrough the wrapper is a drop-in only until the
    first such access, which fails at render time rather than at wrap time."""
    inner = nn.Linear(8, 6)
    wrapped = PadToMinM(inner)
    assert wrapped.in_features == 8
    assert wrapped.out_features == 6
    assert wrapped.weight is inner.weight
    assert wrapped.bias is inner.bias
    with pytest.raises(AttributeError):
        wrapped.definitely_not_a_linear_attribute


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.context_embedder = nn.Linear(8, 6)
        self.other = nn.Linear(4, 4)


def test_state_dict_hides_the_wrapper_in_both_directions():
    """The prequant builder saves a state dict and the loader loads one into a fresh tree, so a
    wrapper that renamed ``context_embedder.weight`` to ``context_embedder.inner.weight`` would
    split the two halves of the checkpoint contract. It must be invisible either way round."""
    plain, wrapped_model = _Tiny(), _Tiny()
    wrapped_model.context_embedder = PadToMinM(wrapped_model.context_embedder)

    assert sorted(wrapped_model.state_dict()) == sorted(plain.state_dict())
    assert "context_embedder.inner.weight" not in wrapped_model.state_dict()

    # A checkpoint written from a wrapped tree loads into an unwrapped one, and the reverse.
    assert plain.load_state_dict(dict(wrapped_model.state_dict()), strict = True)
    assert wrapped_model.load_state_dict(dict(plain.state_dict()), strict = True)
    assert torch.equal(wrapped_model.context_embedder.weight, plain.context_embedder.weight)


# ── the granularity gate ──────────────────────────────────────────────────────


def test_activation_granularity_probe_reads_both_torchao_layouts():
    """v1 tensors expose the activation quantizer as ``input_quant_func`` and v2 ones as
    ``act_quant_kwargs.granularity``. An unrecognised layout answers None, not True: the caller
    treats unproven as unsafe."""
    assert activation_granularity_is_per_row(_fake_quantized_linear(per_row = True)) is True
    assert activation_granularity_is_per_row(_fake_quantized_linear(per_row = False)) is False
    assert activation_granularity_is_per_row(_fake_quantized_linear(unknown = True)) is None

    def _v2(granularity_cls_name):
        lin = nn.Linear(8, 6)
        kwargs = types.SimpleNamespace(granularity = type(granularity_cls_name, (), {})())
        lin.weight = nn.Parameter(
            lin.weight.data.as_subclass(_fake_quant_tensor_type(act_quant_kwargs = kwargs)),
            requires_grad = False,
        )
        return lin

    assert activation_granularity_is_per_row(_v2("PerRow")) is True
    assert activation_granularity_is_per_row(_v2("PerToken")) is True
    assert activation_granularity_is_per_row(_v2("PerTensor")) is False
    assert activation_granularity_is_per_row(_v2("PerGroup")) is False

    # A dense Linear is not quantized at all, so there is no activation granularity to report.
    assert activation_granularity_is_per_row(nn.Linear(8, 6)) is None


def test_wrap_raises_rather_than_skipping_an_unprovable_linear():
    """Silence is the worst outcome: a half-padded transformer compiles on the modules that were
    wrapped and crashes inside ``_int_mm`` on the ones that were not."""
    model = _Tiny()
    model.context_embedder = _fake_quantized_linear(per_row = False)
    with pytest.raises(RuntimeError, match = "provably per row"):
        wrap_small_m_linears(model, ["context_embedder"])
    assert isinstance(model.context_embedder, nn.Linear), "must not leave a partial wrap behind"

    model.context_embedder = _fake_quantized_linear(unknown = True)
    with pytest.raises(RuntimeError, match = "provably per row"):
        wrap_small_m_linears(model, ["context_embedder"])


def test_wrap_skips_dense_and_already_wrapped_linears():
    """One gate covers both. A dense ``F.linear`` has no row floor, so there is nothing to pad
    and nothing to prove; and a ``PadToMinM`` is not an ``nn.Linear`` either, so a second pass
    cannot nest the padding and double the row count."""
    model = _Tiny()
    assert wrap_small_m_linears(model, ["context_embedder"]) == ()
    assert isinstance(model.context_embedder, nn.Linear)

    model.context_embedder = _fake_quantized_linear()
    assert wrap_small_m_linears(model, ["context_embedder"]) == ("context_embedder",)
    assert isinstance(model.context_embedder, PadToMinM)
    assert wrap_small_m_linears(model, ["context_embedder"]) == ()
    assert not isinstance(model.context_embedder.inner, PadToMinM)
    # The wrapper forwards `weight` to its inner Linear, so the gate cannot lean on that alone.
    assert model.context_embedder.weight is model.context_embedder.inner.weight
    assert is_quantized_linear(model.context_embedder) is False


def test_wrap_ignores_names_absent_from_this_checkpoint_variant():
    """The pruned and dense H3 trees differ, so a family token list is a name list rather than a
    promise that every name exists."""
    assert wrap_small_m_linears(_Tiny(), ["token_refiner.refiner_blocks.0.attn.to_q"]) == ()


def test_matching_fqns_selects_quantized_linears_by_substring():
    """Same substring rule ``make_filter_fn`` applies to exclusions, so the pad list and the
    exclude list are read the same way -- and a dense Linear never enters the pad list."""
    model = _Tiny()
    model.context_embedder = _fake_quantized_linear()
    assert matching_linear_fqns(model, ("context_embedder",)) == ("context_embedder",)
    assert matching_linear_fqns(model, ("CONTEXT_EMBEDDER",)) == ("context_embedder",)
    assert matching_linear_fqns(model, ("other",)) == (), "dense Linears are not pad candidates"
    assert matching_linear_fqns(model, ()) == ()


def test_is_quantized_linear_only_accepts_a_torchao_weight():
    assert is_quantized_linear(_fake_quantized_linear()) is True
    assert is_quantized_linear(nn.Linear(8, 6)) is False
    assert is_quantized_linear(nn.LayerNorm(8)) is False


# ── the real thing ────────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "int8 dynamic quant needs CUDA")
def test_int8_padding_is_bitwise_exact_on_a_real_quantized_linear():
    """Closes the loop the fakes leave open: a genuinely torchao-quantized Linear, at the row
    counts H3's seven eval prompts produce (M = 10..19, straddling ``_int_mm``'s floor of 16)."""
    pytest.importorskip("torchao")
    from torchao.quantization import quantize_

    from core.inference.diffusion_transformer_quant import _make_quant_config, make_filter_fn

    lin = nn.Linear(1024, 768, bias = False).cuda().bfloat16().eval()
    quantize_(lin, _make_quant_config("int8"), filter_fn = make_filter_fn(0))
    assert is_quantized_linear(lin)
    assert activation_granularity_is_per_row(lin) is True
    for m in (1, 10, 13, 16, 17, 19, 64):
        assert padding_is_bitwise_exact(lin, m), f"padding changed the kept rows at M = {m}"
