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
