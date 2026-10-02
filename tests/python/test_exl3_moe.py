"""GPU-free unit tests for memory-efficient quantized MoE experts.

Exercises Exl3QuantizedExperts.forward (reconstruct-on-forward with an LRU
cache) against a reference dense computation, using a mock LinearEXL3 so no GPU
or exllamav3 install is needed.
"""

import os
import unittest

import torch

from unsloth.exllama.moe import Exl3QuantizedExperts, _FusedGateUp


class _MockLinear:
    """Mock LinearEXL3: get_weight_tensor() returns [in, out] like the real one."""

    def __init__(self, in_features, out_features, seed):
        self.in_features = in_features
        self.out_features = out_features
        g = torch.Generator().manual_seed(seed)
        self._w = torch.randn(in_features, out_features, generator = g, dtype = torch.float16) * 0.05

    def get_weight_tensor(self):
        return self._w  # [in, out]


def _make_experts(
    num_experts = 4,
    hidden = 8,
    inter = 6,
    cache = 0,
):
    os.environ["UNSLOTH_EXL3_EXPERT_CACHE"] = str(cache)
    exp = Exl3QuantizedExperts(num_experts, hidden, inter, torch.nn.SiLU())
    for e in range(num_experts):
        gate = _MockLinear(hidden, inter, seed = e * 3 + 1)  # [hidden, inter]
        up = _MockLinear(hidden, inter, seed = e * 3 + 2)  # [hidden, inter]
        down = _MockLinear(inter, hidden, seed = e * 3 + 3)  # [inter, hidden]
        exp.set_expert(e, _FusedGateUp(gate, up), down)
    return exp


def _reference(exp, hidden_states, top_k_index, top_k_weights, act_fn):
    """Independent reference of the eager MoE experts forward."""
    final = torch.zeros_like(hidden_states)
    num_experts = exp.num_experts
    mask = torch.nn.functional.one_hot(top_k_index, num_classes = num_experts).permute(2, 1, 0)
    hit = torch.greater(mask.sum(dim = (-1, -2)), 0).nonzero()
    for ei in hit:
        e = int(ei[0])
        pos, tok = torch.where(mask[e])
        cur = hidden_states[tok]
        gu = (
            torch.cat(
                [exp._gate_up[e].gate.get_weight_tensor(), exp._gate_up[e].up.get_weight_tensor()],
                dim = 1,
            )
            .t()
            .to(cur.dtype)
        )
        gate, up = torch.nn.functional.linear(cur, gu).chunk(2, dim = -1)
        h = act_fn(gate) * up
        dn = exp._down[e].get_weight_tensor().t().to(cur.dtype)
        out = torch.nn.functional.linear(h, dn) * top_k_weights[tok, pos, None]
        final.index_add_(0, tok, out.to(final.dtype))
    return final


class TestExl3QuantizedExperts(unittest.TestCase):
    def _inputs(
        self,
        tokens = 5,
        hidden = 8,
        num_experts = 4,
        topk = 2,
    ):
        torch.manual_seed(0)
        hs = torch.randn(tokens, hidden, dtype = torch.float16)
        idx = torch.randint(0, num_experts, (tokens, topk))
        w = torch.softmax(torch.randn(tokens, topk), dim = -1).to(torch.float16)
        return hs, idx, w

    def test_forward_matches_reference_no_cache(self):
        exp = _make_experts(cache = 0)
        hs, idx, w = self._inputs()
        out = exp(hs, idx, w)
        ref = _reference(exp, hs, idx, w, exp.act_fn)
        self.assertEqual(tuple(out.shape), (5, 8))
        self.assertTrue(torch.allclose(out.float(), ref.float(), atol = 1e-2))

    def test_forward_matches_reference_with_cache(self):
        exp = _make_experts(cache = 2)  # small cap forces eviction
        hs, idx, w = self._inputs()
        out1 = exp(hs, idx, w)
        out2 = exp(hs, idx, w)  # second call hits cache
        ref = _reference(exp, hs, idx, w, exp.act_fn)
        self.assertTrue(torch.allclose(out1.float(), ref.float(), atol = 1e-2))
        # Cached result must be identical to the first (frozen weights).
        self.assertTrue(torch.equal(out1, out2))

    def test_cache_is_bounded(self):
        exp = _make_experts(num_experts = 4, cache = 2)
        hs, idx, w = self._inputs(num_experts = 4, topk = 4)  # likely hits all 4
        exp(hs, idx, w)
        # LRU cache must never exceed its cap.
        self.assertLessEqual(len(exp._gu_cache), 2)
        self.assertLessEqual(len(exp._lru), 2)

    def test_no_dense_params_in_state_dict(self):
        exp = _make_experts()
        # The quantized experts hold no dense nn.Parameters (they live outside the
        # module tree), so state_dict is empty - this is what keeps VRAM small.
        self.assertEqual(len(exp.state_dict()), 0)

    def test_grad_flows_to_upstream_activations(self):
        # The reconstructed base weights are frozen (no_grad), but gradients must
        # still flow to the LoRA-adapted UPSTREAM activations through F.linear.
        exp = _make_experts(cache = 8)
        upstream = torch.nn.Linear(8, 8, bias = False)  # simulate a trainable layer
        inp = torch.randn(6, 8)
        act = upstream(inp)
        idx = torch.randint(0, exp.num_experts, (6, 2))
        w = torch.softmax(torch.randn(6, 2), dim = -1)
        out = exp(act, idx, w)
        self.assertTrue(out.requires_grad)
        out.sum().backward()
        self.assertIsNotNone(upstream.weight.grad)
        self.assertGreater(upstream.weight.grad.norm().item(), 0.0)
        # Cached dense weights must never require grad (they are the frozen base).
        for t in list(exp._gu_cache.values()) + list(exp._dn_cache.values()):
            self.assertFalse(t.requires_grad)

    def test_reconstruction_targets_activation_device_and_dtype(self):
        # _dense must cast to the activation dtype AND (when given) device, so a
        # multi-GPU device_map does not put the weight and activation on
        # different devices. On CPU-only CI we can at least assert the dtype cast
        # and that passing device=cpu is honored without error.
        exp = _make_experts(cache = 0)
        cpu = torch.device("cpu")
        gu, dn = exp._get_dense(0, torch.float32, cpu)
        self.assertEqual(gu.dtype, torch.float32)
        self.assertEqual(dn.dtype, torch.float32)
        self.assertEqual(gu.device, cpu)
        # bf16 activations with fp16 stored weights must not raise a dtype error.
        hs = torch.randn(4, 8, dtype = torch.bfloat16)
        idx = torch.randint(0, exp.num_experts, (4, 2))
        w = torch.softmax(torch.randn(4, 2), dim = -1).to(torch.bfloat16)
        out = exp(hs, idx, w)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_module_to_does_not_touch_external_experts(self):
        # The quantized linears live outside the nn.Module tree via
        # object.__setattr__, so module.to()/._apply() (as accelerate calls)
        # must not move or cast them.
        exp = _make_experts()
        before = exp._gate_up[0].gate.get_weight_tensor().clone()
        exp.to(torch.float32)
        exp._apply(lambda t: t.to(torch.float64))
        after = exp._gate_up[0].gate.get_weight_tensor()
        self.assertEqual(after.dtype, torch.float16)  # unchanged
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(len(list(exp.parameters())), 0)
        self.assertEqual(len(list(exp.buffers())), 0)


class TestExpertBiasDetection(unittest.TestCase):
    """Bias-bearing experts (e.g. gpt_oss) must be detected so the quantized
    path skips them (it cannot represent per-expert bias)."""

    def test_detects_gate_up_and_down_bias(self):
        from unsloth.exllama.moe import _experts_have_bias

        class _M:
            pass

        m = _M()
        self.assertFalse(_experts_have_bias(m))
        m.gate_up_proj_bias = torch.zeros(4)
        self.assertTrue(_experts_have_bias(m))
        m2 = _M()
        m2.down_proj_bias = torch.zeros(4)
        self.assertTrue(_experts_have_bias(m2))


if __name__ == "__main__":
    unittest.main()
