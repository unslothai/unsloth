"""GPU-free unit tests for the EXL3 dequantization / transpose contract.

These guard the subtle placeholder-shape + transpose-detection logic that
Unsloth's LoRA kernels depend on (mirroring the bitsandbytes ``fast_dequantize``
contract). A mock stands in for ExLlamaV3's ``LinearEXL3`` so no GPU / exllamav3
install is needed.
"""

import unittest

import torch

from unsloth.exllama.quant_linear import (
    Exl3QuantState,
    exl3_fast_dequantize,
    get_exl3_quant_state,
)


class _MockInnerExl3:
    """Stands in for exllamav3.modules.quant.exl3.LinearEXL3.

    ``get_weight_tensor`` returns the weight in [in_features, out_features]
    layout, exactly like the real ExLlamaV3 layer.
    """

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.trellis = torch.zeros((1,), dtype = torch.int16)  # device anchor (cpu)
        # Deterministic, distinguishable data: w_in_out[i, j] = i*1000 + j
        i = torch.arange(in_features).unsqueeze(1)
        j = torch.arange(out_features).unsqueeze(0)
        self._w_in_out = (i * 1000 + j).to(torch.float16)

    def get_weight_tensor(self):
        return self._w_in_out  # [in, out]

    def forward(
        self,
        x,
        params = None,
        out_dtype = None,
    ):
        # Simulate a BUGGY fused kernel: return obviously-wrong output. If
        # ExllamaV3Linear.forward ever routes through this, the test below
        # (which compares against reconstruct-then-F.linear) will fail.
        rows = x.reshape(-1, x.shape[-1]).shape[0]
        return torch.full((rows, self.out_features), 999.0, dtype = torch.float16, device = x.device)


def _make_quant_state(in_features, out_features):
    inner = _MockInnerExl3(in_features, out_features)
    return Exl3QuantState(
        inner,
        in_features = in_features,
        out_features = out_features,
        compute_dtype = torch.float16,
    )


class TestDequantizeOrientation(unittest.TestCase):
    def test_dequantize_returns_out_in(self):
        qs = _make_quant_state(in_features = 4, out_features = 6)
        w = qs.dequantize()
        self.assertEqual(tuple(w.shape), (6, 4))  # [out, in]
        # w[out, in] must equal inner[in, out] transposed
        self.assertTrue(torch.equal(w, qs.exl3_linear.get_weight_tensor().t()))

    def test_fast_dequantize_transpose_flag(self):
        qs = _make_quant_state(in_features = 4, out_features = 6)
        w = exl3_fast_dequantize(qs, transpose = False)
        wt = exl3_fast_dequantize(qs, transpose = True)
        self.assertEqual(tuple(w.shape), (6, 4))  # [out, in]
        self.assertEqual(tuple(wt.shape), (4, 6))  # [in, out]
        self.assertTrue(torch.equal(w, wt.t()))

    def test_quant_state_shape_attr(self):
        qs = _make_quant_state(in_features = 4, out_features = 6)
        self.assertEqual(tuple(qs.shape), (6, 4))


class TestKernelFastDequantizeContract(unittest.TestCase):
    """The kernels/utils.py EXL3 branch must honour the bnb transpose contract.

    matmul_lora calls fast_dequantize(W, qs) and wants [out, in].
    fast_linear_forward / LoRA backward call fast_dequantize(W.t(), qs) and want
    [in, out]. The placeholder is shaped [out, 1] so W.t() has shape[0]==1.
    """

    def setUp(self):
        # Import the wrapped fast_dequantize from the kernels module. This pulls
        # the full unsloth import; conftest provides the GPU-free harness.
        from unsloth.kernels.utils import fast_dequantize
        self.fast_dequantize = fast_dequantize

    def test_placeholder_untransposed_gives_out_in(self):
        in_f, out_f = 5, 8
        qs = _make_quant_state(in_f, out_f)
        # Placeholder as built by ExllamaV3Linear: shape [out, 1].
        W = torch.zeros((out_f, 1), dtype = torch.float16)
        w = self.fast_dequantize(W, qs)
        self.assertEqual(tuple(w.shape), (out_f, in_f))

    def test_placeholder_transposed_gives_in_out(self):
        in_f, out_f = 5, 8
        qs = _make_quant_state(in_f, out_f)
        W = torch.zeros((out_f, 1), dtype = torch.float16)
        # matmul_lora path passes W; fast_linear_forward passes W.t().
        w_t = self.fast_dequantize(W.t(), qs)  # W.t() has shape [1, out]
        self.assertEqual(tuple(w_t.shape), (in_f, out_f))

    def test_square_layer_both_orientations(self):
        # Square layer (e.g. o_proj): orientation must still be data-correct.
        n = 6
        qs = _make_quant_state(n, n)
        W = torch.zeros((n, 1), dtype = torch.float16)
        w = self.fast_dequantize(W, qs)  # [out, in]
        w_t = self.fast_dequantize(W.t(), qs)  # [in, out]
        self.assertEqual(tuple(w.shape), (n, n))
        self.assertEqual(tuple(w_t.shape), (n, n))
        self.assertTrue(torch.equal(w, w_t.t()))

    def test_non_exl3_quant_state_delegates_to_bnb(self):
        # A None quant_state must pass through to the original bnb behaviour
        # (which returns W unchanged for quant_state=None).
        W = torch.randn((4, 3), dtype = torch.float16)
        out = self.fast_dequantize(W, None)
        self.assertTrue(torch.equal(out, W))


class TestGetQuantState(unittest.TestCase):
    def test_from_weight_tensor(self):
        qs = _make_quant_state(4, 6)
        W = torch.zeros((6, 1), dtype = torch.float16)
        W.quant_state = qs
        self.assertIs(get_exl3_quant_state(W), qs)

    def test_none_for_plain_tensor(self):
        self.assertIsNone(get_exl3_quant_state(torch.zeros((2, 2))))


class TestForwardUsesReconstruction(unittest.TestCase):
    """ExllamaV3Linear.forward must reconstruct+F.linear, not the fused kernel.

    Regression guard for a real bug: ExLlamaV3's fused trellis matmul
    (``inner.forward`` under ``transformers_fix``) produced wrong logits on some
    models even though the dequantized weights were correct. The mock inner's
    ``forward`` returns a sentinel (999.0); a correct ExllamaV3Linear.forward
    must instead equal ``F.linear(x, dequantized_weight)`` and never 999.
    """

    def test_forward_matches_dense_linear(self):
        from unsloth.exllama.quant_linear import ExllamaV3Linear

        in_f, out_f = 4, 6
        inner = _MockInnerExl3(in_f, out_f)
        layer = ExllamaV3Linear(in_f, out_f, inner, bias = None, compute_dtype = torch.float16)
        x = torch.randn(2, in_f, dtype = torch.float16)
        out = layer(x)
        # Expected: x @ dequant_weight.T, dequant = get_weight_tensor().t() = [out,in]
        W = layer.weight.quant_state.dequantize(dtype = torch.float16)  # [out, in]
        expected = torch.nn.functional.linear(x, W)
        self.assertEqual(tuple(out.shape), (2, out_f))
        self.assertTrue(torch.allclose(out.float(), expected.float(), atol = 1e-2))
        # And crucially NOT the buggy fused-kernel sentinel (999).
        self.assertFalse(torch.allclose(out.float(), torch.full_like(out.float(), 999.0)))


if __name__ == "__main__":
    unittest.main()
