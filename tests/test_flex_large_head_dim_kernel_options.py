# SPDX-License-Identifier: AGPL-3.0-or-later
"""FlexAttention above head_dim 256 needs explicit kernel_options, or the launch faults with
`CUDA error: misaligned address`."""

import pytest

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.models._utils as u


@pytest.mark.parametrize("head_dim", [32, 64, 128, 192, 256])
def test_no_kernel_options_at_or_below_256(head_dim):
    assert u._flex_kernel_options_for_head_dim(head_dim) is None


@pytest.mark.parametrize("head_dim", [264, 272, 288, 320, 384, 512])
def test_kernel_options_above_256(head_dim):
    # 264 is the first multiple of 8 above the boundary and already faults without these.
    assert u._flex_kernel_options_for_head_dim(head_dim) == {
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "BLOCK_M1": 16,
        "BLOCK_N1": 32,
        "BLOCK_M2": 32,
        "BLOCK_N2": 16,
    }


def test_block_m_is_32_not_64():
    # 64 fits the naive shared-memory estimate and still faults.
    assert u._flex_kernel_options_for_head_dim(512)["BLOCK_M"] == 32
    assert u._flex_kernel_options_for_head_dim(512)["BLOCK_N"] == 32


def test_a_non_integer_head_dim_is_left_alone():
    assert u._flex_kernel_options_for_head_dim(None) is None
    assert u._flex_kernel_options_for_head_dim("512") is None


class _Tensor:
    """Just enough of a tensor for the wrapper's head-dim probe."""

    def __init__(self, shape):
        self.shape = shape

    def dim(self):
        return len(self.shape)


def _record():
    seen = {}

    def flex_attention_forward(module, query, key, value, attention_mask, **kwargs):
        seen.update(kwargs)
        seen["called"] = True
        return ("out", "lse")

    return flex_attention_forward, seen


def _call(head_dim, **kwargs):
    original, seen = _record()
    wrapped = u._wrap_flex_attention_forward(original)
    query = _Tensor((1, 16, 2048, head_dim))
    assert wrapped(None, query, query, query, None, **kwargs) == ("out", "lse")
    assert seen["called"]
    return seen


def test_wrapper_injects_kernel_options_above_256():
    assert _call(512)["kernel_options"]["BLOCK_M"] == 32


def test_wrapper_leaves_256_alone():
    # Must be absent, not an empty dict, so torch keeps its own default.
    assert _call(256).get("kernel_options") is None


def test_a_caller_that_asked_for_something_keeps_it():
    got = _call(512, kernel_options = {"BLOCK_M": 16, "num_warps": 8})["kernel_options"]
    assert got["BLOCK_M"] == 16
    assert got["num_warps"] == 8
    assert got["BLOCK_N"] == 32


def test_other_kwargs_are_passed_through_untouched():
    seen = _call(512, scaling = 0.125, softcap = 30.0)
    assert seen["scaling"] == 0.125
    assert seen["softcap"] == 30.0


def test_a_non_4d_query_is_left_alone():
    original, seen = _record()
    wrapped = u._wrap_flex_attention_forward(original)
    wrapped(None, _Tensor((1, 2048, 512)), None, None, None)
    assert seen.get("kernel_options") is None


def test_wrapping_is_idempotent():
    original, _seen = _record()
    once = u._wrap_flex_attention_forward(original)
    assert u._wrap_flex_attention_forward(once) is once


def test_the_patch_is_installed_and_reapplying_it_changes_nothing():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    registered = ALL_ATTENTION_FUNCTIONS["flex_attention"]
    assert getattr(
        registered, "_unsloth_flex_kernel_options", False
    ), "importing unsloth must leave the flex attention function wrapped"
    assert u.patch_flex_attention_kernel_options()
    assert ALL_ATTENTION_FUNCTIONS["flex_attention"] is registered


def test_the_wrapper_keeps_the_original_signature():
    # Registered attention functions are introspected, so functools.wraps must keep the signature.
    import inspect

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    parameters = inspect.signature(ALL_ATTENTION_FUNCTIONS["flex_attention"]).parameters
    for name in ("module", "query", "key", "value", "attention_mask"):
        assert name in parameters


class _GradTensor(_Tensor):
    def __init__(
        self,
        shape,
        requires_grad,
        device_type = "cuda",
    ):
        super().__init__(shape)
        self.requires_grad = requires_grad
        self.device = type("device", (), {"type": device_type})()


def _call_with(query, **kwargs):
    original, seen = _record()
    u._wrap_flex_attention_forward(original)(None, query, query, query, None, **kwargs)
    return seen.get("kernel_options")


def test_a_call_that_needs_a_backward_forces_the_main_flex_kernel():
    assert _call_with(_GradTensor((2, 16, 121, 128), True)) == {"FORCE_USE_FLEX_ATTENTION": True}
    got = _call_with(_GradTensor((2, 16, 121, 512), True))
    assert got["FORCE_USE_FLEX_ATTENTION"] is True and got["BLOCK_M"] == 32


def test_inference_keeps_flex_decoding():
    import torch

    assert _call_with(_GradTensor((2, 16, 1, 128), False)) is None
    with torch.no_grad():
        assert _call_with(_GradTensor((2, 16, 121, 128), True)) is None
    assert _call_with(_GradTensor((2, 16, 121, 128), True, device_type = "cpu")) is None


def test_an_explicit_backend_is_not_combined_with_the_legacy_knob():
    # torch refuses BACKEND together with FORCE_USE_FLEX_ATTENTION.
    got = _call_with(
        _GradTensor((2, 16, 121, 128), True), kernel_options = {"BACKEND": "TRITON_DECODE"}
    )
    assert got == {"BACKEND": "TRITON_DECODE"}
    got = _call_with(
        _GradTensor((2, 16, 121, 128), True), kernel_options = {"FORCE_USE_FLEX_ATTENTION": False}
    )
    assert got == {"FORCE_USE_FLEX_ATTENTION": False}


def test_compiled_flex_backward_matches_eager_for_a_short_static_batch():
    # B > 1, static query 121 < 128, 16 * 121 floats per batch (not a multiple of 32): the shape
    # where flex_decoding's padded logsumexp gave grads off by 10x.
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device for the Triton flex kernels")
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError:
        pytest.skip("torch has no flex_attention")
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    torch._dynamo.reset()
    B, H, S, D = 2, 16, 121, 64
    gen = torch.Generator(device = "cuda").manual_seed(0)
    q, k, v, grad = (
        torch.randn(B, H, S, D, device = "cuda", dtype = torch.bfloat16, generator = gen)
        for _ in range(4)
    )
    mask = create_block_mask(
        lambda b, h, qi, ki: qi >= ki, B = B, H = None, Q_LEN = S, KV_LEN = S, device = "cuda"
    )
    module = torch.nn.Module().train()

    def grads(eager):
        qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
        if eager:
            from torch.nn.attention.flex_attention import flex_attention
            out = flex_attention(qq, kk, vv, block_mask = mask, scale = D**-0.5).transpose(1, 2)
        else:
            out, _ = ALL_ATTENTION_FUNCTIONS["flex_attention"](
                module, qq, kk, vv, mask, scaling = D**-0.5
            )
        out.backward(grad.transpose(1, 2))
        return [t.grad.float() for t in (qq, kk, vv)]

    for name, got, want in zip("qkv", grads(eager = False), grads(eager = True)):
        err = ((got - want).abs().max() / want.abs().max()).item()
        assert err < 0.05, f"d{name} relative error {err:.3g} vs eager flex attention"
