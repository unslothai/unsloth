# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The registered Transformers flex wrapper keeps short no-grad queries off flex_decoding where unsloth_zoo found it
cannot compile (Triton 3.9 removed the block pointers torch 2.9 - 2.12's decode template builds)."""

import sys
import types

import pytest

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.models._utils as u


def _wrapped_call(
    monkeypatch,
    unavailable,
    q_len = 1,
    needs_backward = False,
    small_ok = False,
    head_dim = 64,
    **kwargs,
):
    import torch

    seen = {}

    def fake_flex(module, query, key, value, *args, **kw):
        seen["kernel_options"] = kw.get("kernel_options")
        return query

    monkeypatch.setattr(u, "_flex_decoding_unavailable", lambda: unavailable, raising = False)
    monkeypatch.setattr(u, "_flex_tuned_bwd_device_ok", lambda: False)
    monkeypatch.setattr(u, "_flex_small_smem_device_ok", lambda: small_ok)
    monkeypatch.setattr(u, "_flex_call_needs_backward", lambda *a: needs_backward)
    wrapped = u._wrap_flex_attention_forward(fake_flex)

    class M(torch.nn.Module):
        is_causal = True

    q = torch.randn(1, 2, q_len, head_dim)
    k = torch.randn(1, 2, 128, head_dim)
    wrapped(M(), q, k, k.clone(), object(), **kwargs)
    return seen["kernel_options"]


@pytest.mark.parametrize("q_len", [1, 5, 127])
def test_short_no_grad_query_takes_the_main_kernel_when_flex_decoding_cannot_compile(
    monkeypatch, q_len
):
    assert _wrapped_call(monkeypatch, True, q_len) == {"FORCE_USE_FLEX_ATTENTION": True}


@pytest.mark.parametrize("q_len", [1, 5])
def test_short_no_grad_query_keeps_auto_otherwise(monkeypatch, q_len):
    assert _wrapped_call(monkeypatch, False, q_len) is None


def test_backward_calls_are_unchanged(monkeypatch):
    for unavailable in (False, True):
        assert _wrapped_call(monkeypatch, unavailable, 5, needs_backward = True) == {
            "FORCE_USE_FLEX_ATTENTION": True
        }


def test_caller_options_stay_on_top(monkeypatch):
    opts = _wrapped_call(monkeypatch, True, 1, kernel_options = {"PRESCALE_QK": True})
    assert opts == {"FORCE_USE_FLEX_ATTENTION": True, "PRESCALE_QK": True}
    opts = _wrapped_call(monkeypatch, True, 1, kernel_options = {"FORCE_USE_FLEX_ATTENTION": False})
    assert opts == {"FORCE_USE_FLEX_ATTENTION": False}


def test_a_named_backend_is_left_alone(monkeypatch):
    assert _wrapped_call(monkeypatch, True, 1, kernel_options = {"BACKEND": "TRITON_DECODE"}) == {
        "BACKEND": "TRITON_DECODE"
    }


def test_flag_comes_from_unsloth_zoo(monkeypatch):
    stub = types.ModuleType("unsloth_zoo.flex_attention.utils")
    stub.FLEX_DECODING_UNAVAILABLE = True
    monkeypatch.setitem(sys.modules, "unsloth_zoo.flex_attention.utils", stub)
    assert u._flex_decoding_unavailable() is True
    stub.FLEX_DECODING_UNAVAILABLE = False
    assert u._flex_decoding_unavailable() is False


def test_an_unsloth_zoo_without_the_check_keeps_auto(monkeypatch):
    stub = types.ModuleType("unsloth_zoo.flex_attention.utils")
    monkeypatch.setitem(sys.modules, "unsloth_zoo.flex_attention.utils", stub)
    assert u._flex_decoding_unavailable() is False


def _select_torch_2_9(options):
    # torch 2.9 lowering: no BACKEND; flex_decoding unless FORCE_USE_FLEX_ATTENTION.
    return "main" if options.get("FORCE_USE_FLEX_ATTENTION", False) else "decode"


def _select_torch_2_12(options):
    # torch 2.10 - 2.12: BACKEND plus the legacy knob is rejected; AUTO picks flex_decoding unless forced.
    if "BACKEND" in options and options.get("FORCE_USE_FLEX_ATTENTION", False):
        raise ValueError("BACKEND cannot be combined with legacy FORCE_USE_FLEX_ATTENTION")
    backend = options.get("BACKEND", "AUTO")
    if backend == "TRITON_DECODE" or (
        backend == "AUTO" and not options.get("FORCE_USE_FLEX_ATTENTION", False)
    ):
        return "decode"
    return "main"


@pytest.mark.parametrize("select", [_select_torch_2_9, _select_torch_2_12])
@pytest.mark.parametrize("needs_backward", [False, True])
def test_an_explicit_auto_backend_takes_the_main_kernel_too(monkeypatch, needs_backward, select):
    # BACKEND="TRITON" would not stop flex_decoding on torch 2.9, so AUTO is dropped for the legacy knob.
    opts = _wrapped_call(monkeypatch, True, 1, needs_backward, kernel_options = {"BACKEND": "AUTO"})
    assert opts == {"FORCE_USE_FLEX_ATTENTION": True} and select(opts) == "main"
    opts = _wrapped_call(monkeypatch, False, 1, needs_backward, kernel_options = {"BACKEND": "AUTO"})
    if needs_backward:
        assert opts == {"FORCE_USE_FLEX_ATTENTION": True} and select(opts) == "main"
    else:
        assert opts == {"BACKEND": "AUTO"}
    opts = _wrapped_call(
        monkeypatch, True, 1, kernel_options = {"BACKEND": "AUTO", "FORCE_USE_FLEX_ATTENTION": False}
    )
    assert opts == {"BACKEND": "AUTO", "FORCE_USE_FLEX_ATTENTION": False}


@pytest.mark.parametrize("needs_backward", [False, True])
@pytest.mark.parametrize("backend", ["TRITON_DECODE", "FLASH", "TRITON"])
def test_a_named_backend_in_any_mapping_is_left_alone(monkeypatch, needs_backward, backend):
    # torch reads kernel_options through dict(); a read-only mapping must not hide its BACKEND.
    from types import MappingProxyType
    opts = _wrapped_call(
        monkeypatch, True, 1, needs_backward, kernel_options = MappingProxyType({"BACKEND": backend})
    )
    assert opts == {"BACKEND": backend}


def test_an_auto_backend_in_a_read_only_mapping_takes_the_main_kernel(monkeypatch):
    from types import MappingProxyType

    caller = MappingProxyType({"BACKEND": "AUTO", "PRESCALE_QK": True})
    opts = _wrapped_call(monkeypatch, True, 1, kernel_options = caller)
    assert opts == {"FORCE_USE_FLEX_ATTENTION": True, "PRESCALE_QK": True}
    assert dict(caller) == {"BACKEND": "AUTO", "PRESCALE_QK": True}


@pytest.mark.parametrize(
    "requested, main_kernel",
    [
        (None, True),
        ({"BACKEND": "AUTO"}, True),
        ({"FORCE_USE_FLEX_ATTENTION": False}, False),
        ({"BACKEND": "TRITON_DECODE"}, False),
    ],
)
def test_small_memory_tiles_follow_the_kernel_that_runs(monkeypatch, requested, main_kernel):
    # The fitted tiles are main-kernel tiles: a caller that keeps flex_decoding must not get them.
    kwargs = {} if requested is None else {"kernel_options": dict(requested)}
    opts = _wrapped_call(monkeypatch, True, 1, small_ok = True, head_dim = 256, **kwargs) or {}
    tiles = {k: opts.get(k) for k in u._FLEX_SMALL_SMEM_KERNEL_OPTIONS}
    if main_kernel:
        assert tiles == u._FLEX_SMALL_SMEM_KERNEL_OPTIONS
    else:
        assert all(v is None for v in tiles.values()) and opts == requested
