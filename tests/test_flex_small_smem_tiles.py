# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""sm86 / sm89 flex at 128 < head_dim <= 256: fwd_* / bwd_* tiles that fit 101376 bytes."""

from types import SimpleNamespace

import pytest

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.models._utils as u

T4 = ((7, 5), 65536)
A10 = ((8, 6), 101376)
RTX3090 = ((8, 6), 101376)
L4 = ((8, 9), 101376)
L40S = ((8, 9), 101376)
ORIN = ((8, 7), 166912)
A100 = ((8, 0), 166912)
H100 = ((9, 0), 232448)
B200 = ((10, 0), 232448)
RTX_PRO_6000 = ((12, 0), 101376)


@pytest.fixture
def devices(monkeypatch):
    monkeypatch.delenv(u._FLEX_SMALL_SMEM_ENV_VAR, raising = False)
    monkeypatch.setattr(u.torch.version, "hip", None, raising = False)
    monkeypatch.setattr(u.torch, "__version__", "2.10.0+cu130")

    def install(*cards):
        monkeypatch.setattr(u.torch.cuda, "is_available", lambda: bool(cards))
        monkeypatch.setattr(u.torch.cuda, "device_count", lambda: len(cards))
        monkeypatch.setattr(u.torch.cuda, "get_device_capability", lambda index = 0: cards[index][0])
        monkeypatch.setattr(
            u.torch.cuda,
            "get_device_properties",
            lambda index = 0: SimpleNamespace(shared_memory_per_block_optin = cards[index][1]),
        )

    return install


@pytest.mark.parametrize(
    "cards,expected",
    [
        ((A10,), True),
        ((RTX3090,), True),
        ((L4,), True),
        ((L40S, L4), True),
        ((T4,), False),  # pre-Ampere never reaches flex (_FLEX_ATTENTION_MIN_CAPABILITY)
        ((A100,), False),  # the default template fits: unchanged
        ((ORIN,), False),
        ((H100,), False),
        ((B200,), False),
        ((RTX_PRO_6000,), False),  # sm120 has its own Inductor table, not measured here
        ((L4, A100), False),  # every visible device must be in the class
        ((), False),
    ],
)
def test_device_gate(devices, cards, expected):
    devices(*cards)
    assert u._flex_small_smem_device_ok() is expected


@pytest.mark.parametrize(
    "version,expected", [("2.9.1", False), ("2.10.0", True), ("2.14.1+cu130", True)]
)
def test_needs_torch_with_prefixed_keys(devices, monkeypatch, version, expected):
    devices(L4)
    monkeypatch.setattr(u.torch, "__version__", version)
    assert u._flex_small_smem_device_ok() is expected


def test_kill_switch_and_rocm(devices, monkeypatch):
    devices(L4)
    monkeypatch.setenv(u._FLEX_SMALL_SMEM_ENV_VAR, "0")
    assert u._flex_small_smem_device_ok() is False
    monkeypatch.delenv(u._FLEX_SMALL_SMEM_ENV_VAR)
    monkeypatch.setattr(u.torch.version, "hip", "7.2.0", raising = False)
    assert u._flex_small_smem_device_ok() is False


def test_unreadable_device_is_off(devices, monkeypatch):
    devices(L4)

    def boom(index = 0):
        raise RuntimeError("probe")

    monkeypatch.setattr(u.torch.cuda, "get_device_properties", boom)
    assert u._flex_small_smem_device_ok() is False


@pytest.mark.parametrize("head_dim", [129, 192, 256])
def test_options_in_range(head_dim):
    opts = u._flex_small_smem_kernel_options(head_dim, True)
    assert opts == u._FLEX_SMALL_SMEM_KERNEL_OPTIONS
    assert all(k.startswith(("fwd_", "bwd_")) for k in opts)
    opts["fwd_num_stages"] = 99  # a fresh dict per call
    assert u._FLEX_SMALL_SMEM_KERNEL_OPTIONS["fwd_num_stages"] == 1


@pytest.mark.parametrize("head_dim", [64, 128, 257, 512, None, "256"])
def test_options_out_of_range(head_dim):
    assert u._flex_small_smem_kernel_options(head_dim, True) is None


def test_options_need_the_device():
    assert u._flex_small_smem_kernel_options(256, False) is None


def test_tiles_divide_the_default_sparse_block():
    o = u._FLEX_SMALL_SMEM_KERNEL_OPTIONS
    for k in (
        "fwd_BLOCK_M",
        "fwd_BLOCK_N",
        "bwd_BLOCK_M1",
        "bwd_BLOCK_N1",
        "bwd_BLOCK_M2",
        "bwd_BLOCK_N2",
    ):
        assert 128 % o[k] == 0
    assert o["bwd_BLOCK_N1"] % o["bwd_BLOCK_M1"] == 0
    assert o["bwd_BLOCK_M2"] % o["bwd_BLOCK_N2"] == 0


class _Causal:
    is_causal = True


def _wrapped(
    monkeypatch,
    small_ok = True,
    tuned_ok = False,
    short_tiles = False,
):
    seen = {}

    def fake_flex(module, query, key, value, *args, **kwargs):
        seen.clear()
        seen.update(kwargs.get("kernel_options") or {})
        return query

    monkeypatch.setattr(u, "_flex_small_smem_device_ok", lambda: small_ok)
    monkeypatch.setattr(u, "_flex_tuned_bwd_device_ok", lambda: tuned_ok)
    monkeypatch.setattr(u, "_flex_small_smem_short_query_tiles", lambda: short_tiles)
    return u._wrap_flex_attention_forward(fake_flex), seen


def _qkv(
    q_len,
    head_dim = 256,
    grad = True,
):
    import torch
    return [torch.randn(1, 2, q_len, head_dim, requires_grad = grad) for _ in range(3)]


def test_wrapper_passes_tiles_for_training_calls(monkeypatch):
    wrapped, seen = _wrapped(monkeypatch)
    monkeypatch.setattr(u, "_flex_call_needs_backward", lambda *a: True)
    wrapped(_Causal(), *_qkv(16), object())  # a non-None mask stays on flex
    for k, v in u._FLEX_SMALL_SMEM_KERNEL_OPTIONS.items():
        assert seen[k] == v
    assert seen["FORCE_USE_FLEX_ATTENTION"] is True


def test_wrapper_leaves_caller_tiles_alone(monkeypatch):
    # A partial caller set merged with ours could break the templates' tile divisibility checks.
    wrapped, seen = _wrapped(monkeypatch)
    monkeypatch.setattr(u, "_flex_call_needs_backward", lambda *a: True)
    wrapped(_Causal(), *_qkv(256), object(), kernel_options = {"fwd_num_stages": 3, "BLOCK_M": 16})
    assert seen["fwd_num_stages"] == 3 and seen["BLOCK_M"] == 16
    assert "fwd_BLOCK_N" not in seen and "bwd_num_stages" not in seen
    wrapped(_Causal(), *_qkv(256), object(), kernel_options = {"BACKEND": "TRITON"})
    assert seen["BACKEND"] == "TRITON" and seen["fwd_BLOCK_N"] == 64


class _Mask:
    def __init__(self, block):
        self.BLOCK_SIZE = (block, block)


def test_wrapper_skips_sparse_blocks_our_tiles_do_not_divide(monkeypatch):
    wrapped, seen = _wrapped(monkeypatch)
    monkeypatch.setattr(u, "_flex_call_needs_backward", lambda *a: True)
    wrapped(_Causal(), *_qkv(256), _Mask(16))
    assert "fwd_BLOCK_M" not in seen
    wrapped(_Causal(), *_qkv(256), _Mask(64))
    assert seen["fwd_BLOCK_M"] == 64


def test_wrapper_no_grad_prefill_gets_tiles_decode_does_not(monkeypatch):
    wrapped, seen = _wrapped(monkeypatch)
    monkeypatch.setattr(u, "_flex_call_needs_backward", lambda *a: False)
    wrapped(_Causal(), *_qkv(128, grad = False), object())
    assert seen.get("fwd_BLOCK_M") == 64
    assert "FORCE_USE_FLEX_ATTENTION" not in seen  # q_len >= 128 is the main kernel already
    wrapped(_Causal(), *_qkv(8, grad = False), object())
    assert seen == {}  # flex_decoding keeps its own config


def _gqa_qkv(
    q_len,
    q_heads,
    kv_heads,
    head_dim = 256,
):
    import torch
    q = torch.randn(1, q_heads, q_len, head_dim)
    return [q, torch.randn(1, kv_heads, q_len, head_dim), torch.randn(1, kv_heads, q_len, head_dim)]


@pytest.mark.parametrize(
    "q_len,q_heads,kv_heads,tiled",
    [
        (1, 4, 1, False),  # Gemma 3 1B decode step: 16 rows, flex_decoding fits
        (16, 4, 1, False),  # 64 rows still fit
        (17, 4, 1, True),  # 68 rows -> a 128-row tile: 114688 B
        (28, 4, 1, True),  # Gemma 3 1B / 270M chat prompt
        (28, 8, 4, False),  # Gemma 3 4B: 2 query heads per kv head, 56 rows
        (40, 8, 4, True),
        (127, 2, 2, True),
        (28, 6, 2, False),  # not a power of two: transformers repeats K / V, 28 rows
    ],
)
def test_wrapper_short_prompt_whose_decode_tile_does_not_fit_gets_tiles(
    monkeypatch, q_len, q_heads, kv_heads, tiled
):
    # A failed compile here pinned flex_attention to the eager fallback for the whole process.
    wrapped, seen = _wrapped(monkeypatch)
    monkeypatch.setattr(u, "_flex_call_needs_backward", lambda *a: False)
    wrapped(_Causal(), *_gqa_qkv(q_len, q_heads, kv_heads), object())
    if tiled:
        assert {
            k: seen.get(k) for k in u._FLEX_SMALL_SMEM_KERNEL_OPTIONS
        } == u._FLEX_SMALL_SMEM_KERNEL_OPTIONS
        # On the main kernel: flex_decoding with these tiles is wrong on torch 2.10, fails on 2.11.
        assert seen.get("FORCE_USE_FLEX_ATTENTION") is True
    else:
        assert seen == {}
    wrapped, seen = _wrapped(monkeypatch, small_ok = False)
    wrapped(_Causal(), *_gqa_qkv(q_len, q_heads, kv_heads), object())
    assert seen == {}  # other cards keep the defaults


@pytest.mark.parametrize(
    "version,expected",
    [
        ("2.10.0+cu128", True),
        ("2.11.0", True),
        ("2.13.1", True),
        ("2.14.0", False),
        ("2.14.1+cu130", False),
    ],
)
def test_short_query_tiles_torch_gate(monkeypatch, version, expected):
    monkeypatch.setattr(u.torch, "__version__", version)
    assert u._flex_small_smem_short_query_tiles() is expected


@pytest.mark.parametrize(
    "q_len,q_heads,kv_heads,forced",
    [
        (1, 4, 1, False),  # Gemma 3 1B decode step: tiles, flex_decoding stays eligible
        (4, 4, 1, False),  # 16 rows: flex_decoding's own BLOCK_M would be 16
        (8, 4, 1, False),  # 32 rows
        (16, 4, 1, False),  # 64 rows
        (16, 8, 4, False),  # Gemma 3 4B 16-token prompt, 32 rows
        (28, 8, 4, False),  # Gemma 3 4B prompt, 56 rows
        (28, 6, 2, False),  # repeated K / V, 28 rows
        (17, 4, 1, True),  # 68 rows: main kernel, as in the default gate
        (127, 2, 2, True),
    ],
)
def test_wrapper_short_no_grad_queries_get_tiles_before_torch_2_14(
    monkeypatch, q_len, q_heads, kv_heads, forced
):
    # A batch size change makes the batch dim dynamic, which moves even one-token decode from
    # flex_decoding to the main kernel, whose default tile does not fit (torch 2.10 / 2.11).
    import torch

    qkv = [t.to(torch.bfloat16) for t in _gqa_qkv(q_len, q_heads, kv_heads)]
    wrapped, seen = _wrapped(monkeypatch, short_tiles = True)
    monkeypatch.setattr(u, "_flex_call_needs_backward", lambda *a: False)
    wrapped(_Causal(), *qkv, object())
    assert {
        k: seen.get(k) for k in u._FLEX_SMALL_SMEM_KERNEL_OPTIONS
    } == u._FLEX_SMALL_SMEM_KERNEL_OPTIONS
    assert seen.get("FORCE_USE_FLEX_ATTENTION") is (True if forced else None)
    # flex_decoding derives SAFE_M_BOUNDARY from plain BLOCK_M, not fwd_BLOCK_M (torch < 2.14).
    assert seen.get("BLOCK_M") == (None if forced else 64)
    wrapped, seen = _wrapped(monkeypatch, small_ok = False, short_tiles = True)
    wrapped(_Causal(), *qkv, object())
    assert seen == {}  # other cards keep the defaults


@pytest.mark.parametrize(
    "head_dim,dtype,tiled",
    [
        (256, "bfloat16", True),
        (256, "float16", True),
        (256, "float32", False),
        (192, "bfloat16", False),
    ],
)
def test_short_query_tiles_only_where_measured(monkeypatch, head_dim, dtype, tiled):
    # fp32 needs 147456 B with these tiles on flex_decoding; head_dim 192 fails to build on 2.10.
    import torch

    wrapped, seen = _wrapped(monkeypatch, short_tiles = True)
    monkeypatch.setattr(u, "_flex_call_needs_backward", lambda *a: False)
    qkv = [t.to(getattr(torch, dtype)) for t in _gqa_qkv(1, 4, 1, head_dim = head_dim)]
    wrapped(_Causal(), *qkv, object())
    assert (seen.get("fwd_BLOCK_M") == 64) is tiled
    assert "FORCE_USE_FLEX_ATTENTION" not in seen


def test_short_query_tiles_need_the_small_card(monkeypatch):
    calls = []
    monkeypatch.setattr(u, "_flex_small_smem_device_ok", lambda: False)
    monkeypatch.setattr(u, "_flex_tuned_bwd_device_ok", lambda: False)
    monkeypatch.setattr(u, "_flex_small_smem_short_query_tiles", lambda: calls.append(1) or True)
    u._wrap_flex_attention_forward(lambda *a, **k: None)
    assert calls == []


def test_wrapper_untouched_off_class_and_outside_range(monkeypatch):
    wrapped, seen = _wrapped(monkeypatch, small_ok = False)
    monkeypatch.setattr(u, "_flex_call_needs_backward", lambda *a: True)
    wrapped(_Causal(), *_qkv(256), object())
    assert "fwd_BLOCK_M" not in seen
    wrapped, seen = _wrapped(monkeypatch, small_ok = True)
    wrapped(_Causal(), *_qkv(256, head_dim = 128), object())
    assert "fwd_BLOCK_M" not in seen


def test_a100_keeps_its_tuned_backward(monkeypatch):
    wrapped, seen = _wrapped(monkeypatch, small_ok = False, tuned_ok = True)
    monkeypatch.setattr(u, "_flex_call_needs_backward", lambda *a: True)
    wrapped(_Causal(), *_qkv(256), object())
    for k, v in u._FLEX_TUNED_BWD_KERNEL_OPTIONS.items():
        assert seen[k] == v
    assert "fwd_BLOCK_M" not in seen


def test_small_card_with_tuned_backward_forced_keeps_fitting_tiles(monkeypatch):
    # UNSLOTH_FLEX_TUNED_BACKWARD=1 on an L4: backward-only keys would leave the default forward
    # tile, which does not fit, so the full small-memory set wins.
    wrapped, seen = _wrapped(monkeypatch, small_ok = True, tuned_ok = True)
    monkeypatch.setattr(u, "_flex_call_needs_backward", lambda *a: True)
    wrapped(_Causal(), *_qkv(256), object())
    for k, v in u._FLEX_SMALL_SMEM_KERNEL_OPTIONS.items():
        assert seen[k] == v


@pytest.mark.parametrize("requested", [{"FORCE_USE_FLEX_ATTENTION": True}, {"BACKEND": "TRITON"}])
def test_wrapper_forced_main_kernel_no_grad_short_query_gets_tiles(monkeypatch, requested):
    # Gradient checkpointing runs each block's first forward under no_grad, and the block compile
    # hook forces the main kernel there, whose default head_dim 256 tiles do not fit (#15271).
    wrapped, seen = _wrapped(monkeypatch)
    monkeypatch.setattr(u, "_flex_call_needs_backward", lambda *a: False)
    wrapped(_Causal(), *_qkv(64, grad = False), object(), kernel_options = dict(requested))
    for k, v in u._FLEX_SMALL_SMEM_KERNEL_OPTIONS.items():
        assert seen[k] == v
    for k, v in requested.items():
        assert seen[k] == v
    wrapped(_Causal(), *_qkv(64, grad = False), object(), kernel_options = {"BACKEND": "AUTO"})
    assert "fwd_BLOCK_M" not in seen  # flex_decoding may still run: its own tiles


def test_wrapper_training_calls_add_no_sequence_guard(monkeypatch):
    # The grad check must short-circuit before query.shape[2] >= 128, or a dynamic-shape block
    # compiles once more when the sequence crosses 128.
    import torch

    monkeypatch.setattr(u, "_flex_call_needs_backward", lambda *a: True)

    def graphs_for(small_ok):
        wrapped, _ = _wrapped(monkeypatch, small_ok = small_ok)
        graphs = []

        def backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        torch._dynamo.reset()

        @torch.compile(backend = backend, dynamic = True)
        def block(q):
            return wrapped(_Causal(), q, q, q, None, is_causal = False) * 2

        for q_len in (256, 512, 64, 32):
            block(torch.randn(1, 2, q_len, 256))
        torch._dynamo.reset()
        return len(graphs)

    assert graphs_for(True) == graphs_for(False)
