"""ROCm builds whose flash varlen disagrees with its fake kernel cannot run it inside a compiled block.
Packed rows there must leave the block for the eager varlen wrapper, not attend over the dense mask."""

import os
import types

import pytest
import torch

import unsloth.utils.hf_packed_attention as hpa

# Each checkout tests its own tree, not an installed unsloth.
assert os.path.realpath(hpa.__file__).startswith(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + os.sep
), hpa.__file__


def _block_causal(lengths):
    total = sum(lengths)
    mask = torch.zeros(total, total, dtype = torch.bool)
    start = 0
    for n in lengths:
        mask[start : start + n, start : start + n] = torch.ones(n, n, dtype = torch.bool).tril()
        start += n
    return mask[None, None]


def _layer(n_heads = 4, head_dim = 16):
    config = types.SimpleNamespace(
        _attn_implementation = "sdpa", num_attention_heads = n_heads, num_key_value_heads = n_heads
    )
    attention = types.SimpleNamespace(config = config, head_dim = head_dim, num_key_value_groups = 1)
    return types.SimpleNamespace(self_attn = attention)


@pytest.fixture
def routes(monkeypatch):
    calls = {"dense": 0, "eager_wrapper": 0, "torch_varlen": 0}

    def dense(module, q, k, v, mask, **kwargs):
        calls["dense"] += 1
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = mask)
        return out.transpose(1, 2), None

    def torch_varlen(q, k, v, cu_q, cu_k, max_q, max_k, scale, window_size, **kwargs):
        calls["torch_varlen"] += 1
        return q * scale

    monkeypatch.setattr(hpa, "_ORIG_SDPA", [dense])
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(hpa, "_torch_varlen", lambda device: torch_varlen)
    monkeypatch.setattr(hpa, "_hip_window_ok", lambda device, window: True)
    monkeypatch.setattr(hpa, "_sdpa_policy_requested", lambda *args: False)
    monkeypatch.setattr(hpa, "_sdpa_requested", lambda *args: False)
    monkeypatch.setattr(hpa, "_static_window", lambda *args: None)
    monkeypatch.setattr(hpa, "_leaves_block", lambda *args: False)
    monkeypatch.setattr(hpa, "_BLOCK_PREP", {})
    import unsloth.utils.flash_impl as flash_impl

    monkeypatch.setattr(flash_impl, "flash_impl_allows", lambda *args: True)
    real_eager = hpa._sdpa_packed_varlen_eager

    def eager_wrapper(*args, **kwargs):
        calls["eager_wrapper"] += 1
        return real_eager(*args, **kwargs)

    monkeypatch.setattr(hpa, "_sdpa_packed_varlen_eager", eager_wrapper)
    return calls


def _prepare(lengths, dtype = torch.bfloat16):
    mask, psl = _block_causal(lengths), torch.tensor(lengths, dtype = torch.int32)
    hidden = torch.randn(1, sum(lengths), 8, dtype = dtype)
    before = hpa.HF_PACKED_ATTENTION_STATS["compiled"]
    varlen = hpa.block_varlen_kwargs(
        _layer(), hidden, {"packed_seq_lengths": psl, "attention_mask": mask}
    )
    return varlen, hpa.HF_PACKED_ATTENTION_STATS["compiled"] - before, mask, psl


def test_rocm_fake_kernel_mismatch_leaves_the_block_for_eager_varlen(monkeypatch, routes):
    monkeypatch.setattr(torch.version, "hip", "7.13")
    monkeypatch.setattr(hpa, "_varlen_in_graph", lambda *args: False)
    lengths = [3, 5]
    varlen, compiled, mask, psl = _prepare(lengths)
    assert varlen == hpa._LEAVE_BLOCK
    assert compiled == 0
    module = types.SimpleNamespace(is_causal = True, num_key_value_groups = 1, training = True)
    torch._dynamo.reset()
    monkeypatch.setattr(hpa, "IN_COMPILED_BLOCK", [True])
    fn = torch.compile(
        lambda q: hpa._sdpa_packed_varlen(
            module, q, q, q, mask, packed_seq_lengths = psl, **{hpa.BLOCK_VARLEN_KWARG: varlen}
        )[0],
        backend = "eager",
    )
    fn(torch.randn(1, 4, sum(lengths), 16, dtype = torch.bfloat16))
    torch._dynamo.reset()
    assert routes["eager_wrapper"] == 1
    assert routes["torch_varlen"] == 1
    assert routes["dense"] == 0


@pytest.mark.parametrize(
    "hip, in_graph",
    [(None, False), (None, True), ("7.13", True)],
)
def test_other_builds_keep_the_in_graph_route(monkeypatch, routes, hip, in_graph):
    """CUDA never leaves (whatever the probe says), nor does a ROCm build whose varlen traces."""
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(hpa, "_varlen_in_graph", lambda *args: in_graph)
    varlen, compiled, _, _ = _prepare([3, 5])
    assert isinstance(varlen, tuple)
    assert compiled == 1


def test_rocm_fp32_rows_keep_the_in_graph_route(monkeypatch, routes):
    """No half-precision varlen for fp32 rows: leaving the block would only add a graph break."""
    monkeypatch.setattr(torch.version, "hip", "7.13")
    monkeypatch.setattr(hpa, "_varlen_in_graph", lambda *args: False)
    varlen, compiled, _, _ = _prepare([3, 5], dtype = torch.float32)
    assert isinstance(varlen, tuple)
    assert compiled == 1
