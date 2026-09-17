# SPDX-License-Identifier: AGPL-3.0-or-later
"""FlexAttention above head_dim 256 needs explicit kernel_options, or it faults.

Inductor's default flex template asks for BLOCK_M = BLOCK_N = 128, which at head_dim 512
wants 266 240 bytes of shared memory against a B200's 232 448 byte limit. Inductor emits the
kernel anyway and the launch dies with `CUDA error: misaligned address`. Reproduced through
the exact path this repo routes to, transformers' registered `flex_attention` function, by
scripts/flex_11102_gemma4_global_repro.py:

    unpatched  head_dim 512   FAIL  torch.AcceleratorError: CUDA error: misaligned address
    patched    head_dim 512   PASS  relRMS 0.00203, triton_tem_fused_flex_attention_0
    hd256      head_dim 256   PASS  unpatched, so the boundary really is > 256

Gemma 4's 5 full-attention layers use global_head_dim 512, so this is the difference between
running and crashing for the model that most needs the routing.

These pin the boundary, the injection, and that the head dims which run today are untouched.
"""

import pytest

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.models._utils as u


# ---------------------------------------------------------------------------------------------
# The boundary
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [32, 64, 128, 192, 256])
def test_no_kernel_options_at_or_below_256(head_dim):
    # Every head dim that runs on the default template today must keep running on it, or this
    # patch would be a silent performance change for models it has no business touching.
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
    # BLOCK_M/N=64 fits the naive shared-memory estimate and STILL faults; only 32 passes.
    # The measured rule, not a computed one, so guard against anyone "optimising" it back up.
    assert u._flex_kernel_options_for_head_dim(512)["BLOCK_M"] == 32
    assert u._flex_kernel_options_for_head_dim(512)["BLOCK_N"] == 32


def test_a_non_integer_head_dim_is_left_alone():
    assert u._flex_kernel_options_for_head_dim(None) is None
    assert u._flex_kernel_options_for_head_dim("512") is None


# ---------------------------------------------------------------------------------------------
# The injection
# ---------------------------------------------------------------------------------------------


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
    # Not "injects an empty dict": it must not appear at all, so torch keeps its own default.
    assert _call(256).get("kernel_options") is None


def test_a_caller_that_asked_for_something_keeps_it():
    # Someone tuning their own run must win over our floor, per key.
    got = _call(512, kernel_options = {"BLOCK_M": 16, "num_warps": 8})["kernel_options"]
    assert got["BLOCK_M"] == 16  # caller's
    assert got["num_warps"] == 8  # caller's
    assert got["BLOCK_N"] == 32  # ours, filling the gap


def test_other_kwargs_are_passed_through_untouched():
    seen = _call(512, scaling = 0.125, softcap = 30.0)
    assert seen["scaling"] == 0.125
    assert seen["softcap"] == 30.0


def test_a_non_4d_query_is_left_alone():
    # A few vision callers reuse the interface with a different rank; shape[-1] is not a head
    # dim there, so the probe must not fire.
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
    # Transformers and our own code introspect registered attention functions; functools.wraps
    # must keep inspect.signature following through to the real one.
    import inspect

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    parameters = inspect.signature(ALL_ATTENTION_FUNCTIONS["flex_attention"]).parameters
    for name in ("module", "query", "key", "value", "attention_mask"):
        assert name in parameters
