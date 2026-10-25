# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""sm80 flex backward: tuned bwd_* kernel_options at 128 < head_dim <= 256."""

import pytest

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.models._utils as u


@pytest.mark.parametrize("head_dim", [129, 192, 256])
def test_tuned_backward_options_in_range(head_dim):
    opts = u._flex_tuned_bwd_kernel_options(head_dim, True)
    assert opts == u._FLEX_TUNED_BWD_KERNEL_OPTIONS
    # backward only: the forward template keeps Inductor's default
    assert all(k.startswith("bwd_") for k in opts)
    # a fresh dict every call, so a caller merging its own keys never mutates the constant
    opts["bwd_num_warps"] = 99
    assert u._FLEX_TUNED_BWD_KERNEL_OPTIONS["bwd_num_warps"] == 4


@pytest.mark.parametrize("head_dim", [64, 128, 257, 512, None, "256"])
def test_tuned_backward_options_out_of_range(head_dim):
    assert u._flex_tuned_bwd_kernel_options(head_dim, True) is None


def test_tuned_backward_options_need_a_measured_device():
    assert u._flex_tuned_bwd_kernel_options(256, False) is None


def test_tuned_backward_tiles_divide_the_default_sparse_block():
    o = u._FLEX_TUNED_BWD_KERNEL_OPTIONS
    for k in ("bwd_BLOCK_M1", "bwd_BLOCK_N1", "bwd_BLOCK_M2", "bwd_BLOCK_N2"):
        assert 128 % o[k] == 0
    assert o["bwd_BLOCK_N1"] % o["bwd_BLOCK_M1"] == 0
    assert o["bwd_BLOCK_M2"] % o["bwd_BLOCK_N2"] == 0


@pytest.mark.parametrize("value,expected", [("0", False)])
def test_env_kill_switch(monkeypatch, value, expected):
    monkeypatch.setenv(u._FLEX_TUNED_BWD_ENV_VAR, value)
    assert u._flex_tuned_bwd_device_ok() is expected


@pytest.mark.parametrize(
    "hip,torch_version,caps,env,expected",
    [
        (None, "2.14.0+cu130", [(8, 0)], "", True),
        (None, "2.10.0+cu128", [(8, 0), (8, 0)], "", True),
        (None, "2.14.0", [(8, 0), (8, 6)], "", False),  # every visible device must be measured
        (None, "2.14.0", [(8, 6)], "", False),
        (None, "2.14.0", [(9, 0)], "", False),
        (None, "2.14.0", [(10, 0)], "", False),
        (None, "2.14.0", [(9, 0)], "1", True),  # forced on any CUDA device
        (None, "2.9.1", [(8, 0)], "", False),  # no bwd_ keys before 2.10
        (None, "2.9.1", [(8, 0)], "1", False),
        ("7.2.0", "2.14.0", [(8, 0)], "", False),  # ROCm never
        ("7.2.0", "2.14.0", [(8, 0)], "1", False),
    ],
)
def test_device_gate(monkeypatch, hip, torch_version, caps, env, expected):
    import torch

    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(torch, "__version__", torch_version)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(caps))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = 0: caps[index])
    monkeypatch.setenv(u._FLEX_TUNED_BWD_ENV_VAR, env)
    assert u._flex_tuned_bwd_device_ok() is expected


def test_wrapper_passes_tuned_backward_options(monkeypatch):
    import torch

    seen = {}

    def fake_flex(module, query, key, value, *args, **kwargs):
        seen.update(kwargs.get("kernel_options") or {})
        return query

    monkeypatch.setattr(u, "_flex_tuned_bwd_device_ok", lambda: True)
    wrapped = u._wrap_flex_attention_forward(fake_flex)

    class M(torch.nn.Module):
        is_causal = True

    q = torch.randn(1, 2, 8, 256, requires_grad = True)
    k = torch.randn(1, 2, 8, 256, requires_grad = True)
    v = torch.randn(1, 2, 8, 256, requires_grad = True)
    monkeypatch.setattr(u, "_flex_call_needs_backward", lambda *a: True)
    wrapped(M(), q, k, v, object())  # a non-None mask: stays on flex
    for key_, val in u._FLEX_TUNED_BWD_KERNEL_OPTIONS.items():
        assert seen[key_] == val
    assert seen["FORCE_USE_FLEX_ATTENTION"] is True


def _wrapped_call(monkeypatch, mask, **kwargs):
    import torch

    seen = {}

    def fake_flex(module, query, key, value, *args, **kw):
        seen.update(kw.get("kernel_options") or {})
        return query

    monkeypatch.setattr(u, "_flex_tuned_bwd_device_ok", lambda: True)
    monkeypatch.setattr(u, "_flex_call_needs_backward", lambda *a: True)
    wrapped = u._wrap_flex_attention_forward(fake_flex)

    class M(torch.nn.Module):
        is_causal = True

    q, k, v = (torch.randn(1, 2, 8, 256) for _ in range(3))
    wrapped(M(), q, k, v, mask, **kwargs)
    return seen


class _Mask:
    def __init__(self, block_size):
        self.BLOCK_SIZE = block_size


@pytest.mark.parametrize(
    "requested",
    [{"bwd_BLOCK_M1": 64}, {"BLOCK_M": 64, "BLOCK_N": 64}, {"num_warps": 8}, {"bwd_num_stages": 2}],
)
def test_caller_tiles_are_not_mixed_with_tuned_ones(monkeypatch, requested):
    # A partial caller set merged with ours could break N1 % M1 == 0 (Inductor rejects it).
    seen = _wrapped_call(monkeypatch, object(), kernel_options = dict(requested))
    for key_, val in requested.items():
        assert seen[key_] == val
    for key_ in u._FLEX_TUNED_BWD_KERNEL_OPTIONS:
        if key_ not in requested:
            assert key_ not in seen
    assert seen["FORCE_USE_FLEX_ATTENTION"] is True


def test_caller_non_tile_options_keep_tuned_backward(monkeypatch):
    seen = _wrapped_call(monkeypatch, object(), kernel_options = {"PRESCALE_QK": True})
    assert seen["PRESCALE_QK"] is True
    for key_, val in u._FLEX_TUNED_BWD_KERNEL_OPTIONS.items():
        assert seen[key_] == val


@pytest.mark.parametrize(
    "block_size, tuned",
    [
        ((128, 128), True),
        ((64, 128), True),
        ((32, 32), True),
        ((16, 16), False),
        ((128, 16), False),
    ],
)
def test_tuned_backward_only_when_tiles_divide_the_sparse_blocks(monkeypatch, block_size, tuned):
    # Inductor shrinks its default tiles to fit 16-wide sparse blocks; fixed 32-wide ones cannot.
    seen = _wrapped_call(monkeypatch, _Mask(block_size))
    assert all((key_ in seen) == tuned for key_ in u._FLEX_TUNED_BWD_KERNEL_OPTIONS)
    assert seen["FORCE_USE_FLEX_ATTENTION"] is True


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available()
    or __import__("torch").version.hip is not None
    or tuple(int(x) for x in __import__("torch").__version__.split("+")[0].split(".")[:2])
    < (2, 10),
    reason = "needs NVIDIA CUDA and torch >= 2.10 (bwd_ kernel_options)",
)
@pytest.mark.parametrize("case", ["block16", "caller_m1_64"])
def test_tuned_backward_compiles_for_small_blocks_and_partial_caller_tiles(monkeypatch, case):
    # Real Inductor compile of the backward with the options the wrapper emits (tuned path forced).
    import torch
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    torch._dynamo.reset()
    flex = torch.compile(flex_attention, fullgraph = True)

    def real_flex(module, query, key, value, mask, **kw):
        return flex(query, key, value, block_mask = mask, kernel_options = kw.get("kernel_options"))

    monkeypatch.setattr(u, "_flex_tuned_bwd_device_ok", lambda: True)
    wrapped = u._wrap_flex_attention_forward(real_flex)

    class M(torch.nn.Module):
        is_causal = True

    S = 128
    block = 16 if case == "block16" else 128
    mask = create_block_mask(
        lambda b, h, q, kv: q >= kv, None, None, S, S, device = "cuda", BLOCK_SIZE = block
    )
    q, k, v = (
        torch.randn(1, 2, S, 256, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
        for _ in range(3)
    )
    if case == "block16":
        # torch 2.10's Inductor cannot compile 16-wide sparse blocks at all (2.14 shrinks its tiles).
        try:
            flex(q, k, v, block_mask = mask).float().sum().backward()
        except Exception as error:
            pytest.skip(
                f"this torch cannot compile 16-wide sparse blocks without Unsloth: {type(error).__name__}"
            )
        q.grad = k.grad = v.grad = None
        torch._dynamo.reset()
    extra = {"kernel_options": {"bwd_BLOCK_M1": 64}} if case == "caller_m1_64" else {}
    out = wrapped(M(), q, k, v, mask, **extra)
    out.float().sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
