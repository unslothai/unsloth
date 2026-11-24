# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The long variant of a kernel config (``long_keys`` / ``long``, a second compiled kernel): the card check compiles
and verifies it once, long calls run it only where it passed, and where it fails they run the short config, which
keeps serving every call. CPU: a faked sm89 card whose quantiser records the config and whose kernel is fp32
attention (or raises on the long variant)."""

from __future__ import annotations

import contextlib
import inspect

import pytest
import torch

from core.inference import diffusion_int8_attention as I

CAPABILITY = (8, 9)  # hd64 has long_keys there, hd128 has none
LONG_HD, PLAIN_HD = 64, 128


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(I.INT8_ATTENTION_ENV, raising = False)
    monkeypatch.delenv(I.IMPL_ENV, raising = False)
    monkeypatch.setattr(I, "_VERDICTS", {})
    monkeypatch.setattr(I, "_VERIFIED", set())
    monkeypatch.setattr(I, "_CUDA_FAILED", {})
    monkeypatch.setattr(I, "_BROKEN", set())
    monkeypatch.setattr(I, "_LOADED", {})
    monkeypatch.setattr(I, "_LONG_VERIFIED", set(), raising = False)
    monkeypatch.setattr(I, "_LONG_CHECKS", {}, raising = False)
    monkeypatch.setattr(I, "_LONG_BROKEN", set(), raising = False)
    stats = dict(I.STATS)
    yield
    I.STATS.clear()
    I.STATS.update(stats)


def _fake_card(monkeypatch, long_fails = False):
    """A CPU stand-in for an sm89 card. Returns the (keys, selected config) of every kernel call; with
    ``long_fails`` a call that selects the long variant raises at launch, as a failed compile or launch would."""
    from core.inference import rocm_bf16

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: CAPABILITY)
    monkeypatch.setattr(torch.cuda, "device", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)
    monkeypatch.setattr(I, "unsupported_reason", lambda q, k, v: None)

    def route(q, k, v):
        head_dim = int(q.shape[-1])
        impl, config = I._serving(0, head_dim)
        if config is None:
            return "no kernel config for this card", 0, None, None
        return (
            ("kernel failed on this card" if (0, head_dim) in I._BROKEN else None),
            0,
            impl,
            config,
        )

    monkeypatch.setattr(I, "_call_route", route)
    probe = I.probe_inputs
    monkeypatch.setattr(I, "probe_inputs", lambda device, *a, **k: probe("cpu", *a, **k))
    calls = []
    state = {"long_fails": long_fails}

    def quantise(
        q,
        k,
        v,
        config,
        scale = None,
        mode = I.MODE_MAIN,
    ):
        chosen = I.select_config(config, int(k.shape[1]))
        calls.append((int(k.shape[1]), chosen))
        long_keys = config.get("long_keys")
        return {
            "qkv": (q, k, v),
            "config": chosen,
            "long": bool(long_keys) and k.shape[1] >= long_keys,
        }

    def attend(p):
        if p["long"] and state["long_fails"]:
            raise RuntimeError("cuLaunchKernel(unsloth_i8attn wpc=8) failed with CUDA error 1")
        q, k, v = p["qkv"]
        return I.reference_attention(q, k, v).to(v.dtype)

    def cuda_attention(
        q,
        k,
        v,
        config,
        scale,
        index,
        mode = I.MODE_MAIN,
    ):
        # the eager hot path launches through _cuda_attention (one pass of the same two steps)
        return I.attend(I.quantise(q, k, v, config, scale, mode))

    # Stand-ins take the real signatures so a new argument fails by name, not as a TypeError.
    for name, fake in (("quantise", quantise), ("_cuda_attention", cuda_attention)):
        real = list(inspect.signature(getattr(I, name)).parameters)
        assert (
            list(inspect.signature(fake).parameters) == real
        ), f"update the fake {name}: it takes {real}"
    monkeypatch.setattr(I, "quantise", quantise)
    monkeypatch.setattr(I, "attend", attend)
    monkeypatch.setattr(I, "_cuda_attention", cuda_attention)
    return calls, state


def _call(
    keys,
    head_dim = LONG_HD,
    queries = 64,
):
    q, k, v = I.probe_inputs("cpu", torch.float32, keys, 2, head_dim, seed = keys)
    return q[:, :queries], k, v


def _short(head_dim = LONG_HD):
    return I.select_config(I.kernel_config(CAPABILITY, head_dim), 0)


def _long(head_dim = LONG_HD):
    config = I.kernel_config(CAPABILITY, head_dim)
    return I.select_config(config, int(config["long_keys"]))


def test_the_card_check_compiles_and_verifies_the_long_variant(monkeypatch):
    """(a) A config with long_keys: the check runs the long variant on a probe at or above long_keys (few heads,
    batch 1), records the pass, and real long calls then run it; the probe runs once per card / dtype."""
    calls, _ = _fake_card(monkeypatch)
    long_keys = int(I.kernel_config(CAPABILITY, LONG_HD)["long_keys"])
    found = I._check("cuda", torch.float32, 24, LONG_HD, None, timed = False)
    assert found["ok"] and found["long"]["ok"] and found["long"]["cos"] >= I.MIN_COSINE
    assert (0, LONG_HD, "torch.float32", "cuda") in I._VERIFIED
    assert (0, LONG_HD, "torch.float32", "cuda") in I._LONG_VERIFIED
    long_calls = [(keys, c) for keys, c in calls if keys >= long_keys]
    # the plain and the sharp probe, both on the long kernel
    assert len(long_calls) == 2 and all(c == _long() for _, c in long_calls)
    assert _long() != _short()
    # cached: a second check (another head count) does not probe the long variant again
    before = len(calls)
    I._check("cuda", torch.float32, 8, LONG_HD, None, timed = False)
    assert all(keys < long_keys for keys, _ in calls[before:])
    calls.clear()
    assert I.int8_attention_impl(*_call(long_keys)) is not None
    assert calls == [(long_keys, _long())] and I.STATS["int8"] >= 1 and not I._BROKEN


def test_a_long_variant_that_fails_its_check_leaves_long_calls_on_the_short_kernel(monkeypatch):
    """(b) The long variant raises in the check: the card keeps INT8 (the short kernel is verified), short calls run
    it, and long calls run the short config instead of native attention."""
    calls, _ = _fake_card(monkeypatch, long_fails = True)
    long_keys = int(I.kernel_config(CAPABILITY, LONG_HD)["long_keys"])
    found = I._check("cuda", torch.float32, 24, LONG_HD, None, timed = False)
    assert found["ok"] and (0, LONG_HD, "torch.float32", "cuda") in I._VERIFIED
    calls.clear()
    native = I.STATS["native"]
    I.int8_attention_impl(*_call(1000))
    I.int8_attention_impl(*_call(long_keys))
    I.int8_attention_impl(*_call(4 * long_keys))
    I.int8_attention_impl(*_call(1000))
    assert I.STATS["native"] == native and not I._BROKEN
    assert calls == [
        (1000, _short()),
        (long_keys, _short()),
        (4 * long_keys, _short()),
        (1000, _short()),
    ]
    assert not found["long"]["ok"] and found["long"]["failed"] and not I._LONG_VERIFIED


def test_a_long_variant_that_misses_the_accuracy_bar_is_not_used(monkeypatch):
    calls, _ = _fake_card(monkeypatch)
    real = I.attend

    def far_on_long(p):
        out = real(p)
        return out.flip(1) if p["long"] else out

    monkeypatch.setattr(I, "attend", far_on_long)
    long_keys = int(I.kernel_config(CAPABILITY, LONG_HD)["long_keys"])
    found = I._check("cuda", torch.float32, 4, LONG_HD, None, timed = False)
    assert found["ok"] and not found["long"]["ok"] and not found["long"].get("failed")
    calls.clear()
    I.int8_attention_impl(*_call(long_keys))
    assert calls == [(long_keys, _short())] and not I._BROKEN


def test_a_long_call_that_fails_at_runtime_disables_only_the_long_variant(monkeypatch):
    """(b) The long variant passed the check but a later long call raises: that call runs native attention, the
    long variant is off from then on, and short and long calls keep the verified short kernel."""
    calls, state = _fake_card(monkeypatch)
    long_keys = int(I.kernel_config(CAPABILITY, LONG_HD)["long_keys"])
    assert I._check("cuda", torch.float32, 4, LONG_HD, None, timed = False)["long"]["ok"]
    state["long_fails"] = True
    native = I.STATS["native"]
    I.int8_attention_impl(*_call(long_keys))
    assert I.STATS["native"] == native + 1
    assert I._LONG_BROKEN == {(0, LONG_HD, "cuda")} and not I._BROKEN
    calls.clear()
    I.int8_attention_impl(*_call(1000))
    I.int8_attention_impl(*_call(long_keys))
    assert calls == [(1000, _short()), (long_keys, _short())]
    assert I.STATS["native"] == native + 1


def test_a_short_kernel_failure_still_turns_the_card_off(monkeypatch):
    """Unchanged: a failing short-kernel call marks (index, head dim) broken, native from there on."""
    calls, _ = _fake_card(monkeypatch)
    assert I._check("cuda", torch.float32, 4, LONG_HD, None, timed = False)["ok"]

    def attend(p):
        raise RuntimeError("cuLaunchKernel failed")

    monkeypatch.setattr(I, "attend", attend)
    native = I.STATS["native"]
    I.int8_attention_impl(*_call(1000))
    assert I._BROKEN == {(0, LONG_HD)} and I.STATS["native"] == native + 1


def test_a_config_without_long_keys_is_checked_and_run_as_before(monkeypatch):
    """(c) No long_keys (sm89 hd128): the check runs only the plain and the sharp probe at 1000 tokens, records no
    long verdict, and every call length runs the one config."""
    calls, _ = _fake_card(monkeypatch)
    found = I._check("cuda", torch.float32, 24, PLAIN_HD, None, timed = False)
    assert found["ok"] and "long" not in found
    assert [keys for keys, _ in calls] == [1000, 1000]
    calls.clear()
    I.int8_attention_impl(*_call(1000, PLAIN_HD))
    I.int8_attention_impl(*_call(16384, PLAIN_HD, queries = 16))
    assert calls == [(1000, _short(PLAIN_HD)), (16384, _short(PLAIN_HD))]
    assert not I._LONG_VERIFIED and not I._LONG_BROKEN and not I._BROKEN


def test_an_explicit_config_is_run_as_given(monkeypatch):
    """A caller passing a config (benchmarks, the long probe itself) gets its long variant without a check."""
    calls, _ = _fake_card(monkeypatch)
    config = I.kernel_config(CAPABILITY, LONG_HD)
    long_keys = int(config["long_keys"])
    I.int8_attention(*_call(long_keys), config = config)
    assert calls == [(long_keys, _long())]
