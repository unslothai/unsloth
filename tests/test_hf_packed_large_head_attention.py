# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""CPU tests mock kernel eligibility; CUDA tests compare outputs and gradients."""

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch

from real_accelerator import has_real_cuda


@pytest.fixture
def hpa(monkeypatch):
    # Avoid Unsloth's model-loading and accelerator imports.
    root = Path(__file__).resolve().parents[1] / "unsloth"
    name = "_packed_hd512_test"
    for suffix, path in (("", root), (".utils", root / "utils"), (".kernels", root / "kernels")):
        package = types.ModuleType(name + suffix)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name + suffix, package)
    ad = types.ModuleType(name + ".utils.attention_dispatch")
    ad._VARLEN_INT32_GUARD_DISABLED = False
    ad.FLASH_VARLEN, ad.XFORMERS = "flash", "xformers"
    ad._varlen_backward_overflows_int32 = lambda *args: False
    ad.select_attention_backend = lambda **kwargs: None
    # Force efficient dispatch without model-backend discovery.
    ad._sdpa_backend_state = lambda *args: (
        True,
        int(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION),
    )
    monkeypatch.setitem(sys.modules, ad.__name__, ad)
    module_name = name + ".utils.hf_packed_attention"
    spec = importlib.util.spec_from_file_location(
        module_name, root / "utils/hf_packed_attention.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    yield module
    # Remove relative imports created outside monkeypatch.
    for key in list(sys.modules):
        if key.startswith(name + ".") and key not in (module_name, ad.__name__):
            sys.modules.pop(key, None)


def _cu(lengths, device = "cpu"):
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype = torch.int32, device = device
    )


def _mask(
    lengths,
    device = "cpu",
    window = None,
):
    doc = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths)).to(device)
    pos = torch.arange(sum(lengths), device = device)
    allowed = (doc[:, None] == doc[None, :]) & (pos[:, None] >= pos[None, :])
    if window is not None:
        allowed &= pos[None, :] > pos[:, None] - window
    return allowed[None, None]


@pytest.mark.parametrize("window", [None, 1, 5, 1024])
def test_intervals_equal_the_validated_document_mask(hpa, window):
    lengths = [1, 5, 2, 37, 3]
    lo, hi = hpa._packed_key_intervals(_cu(lengths), sum(lengths), window)
    keys = torch.arange(sum(lengths))
    reconstructed = (keys[None, :] >= lo[0, :, None]) & (keys[None, :] <= hi[0, :, None])
    assert torch.equal(reconstructed, _mask(lengths, window = window)[0, 0])
    assert lo.dtype == hi.dtype == torch.int32
    assert lo.is_contiguous() and hi.is_contiguous()


def test_intervals_reuse_and_invalidate_changed_boundaries(hpa):
    cu = _cu([3, 5])
    first = hpa._packed_key_intervals(cu, 8, None)
    second = hpa._packed_key_intervals(cu, 8, None)
    assert all(a is b for a, b in zip(first, second))
    cu[1] = 5
    changed = hpa._packed_key_intervals(cu, 8, None)
    assert changed[0].tolist() == [[0] * 5 + [5] * 3]
    assert changed[0] is not first[0]
    windowed = hpa._packed_key_intervals(cu, 8, 1)
    assert windowed[0].tolist() == [list(range(8))]
    assert hpa._packed_key_intervals(cu, 8, None)[0] is changed[0]


def test_inference_intervals_never_leak_into_training(hpa):
    cu = _cu([3, 5])
    with torch.inference_mode():
        lo, hi = hpa._packed_key_intervals(cu, 8, None)
        assert torch.is_inference(lo)
    train_lo, train_hi = hpa._packed_key_intervals(cu, 8, None)
    assert not torch.is_inference(train_lo) and not torch.is_inference(train_hi)
    with torch.inference_mode():
        inference_cu = _cu([3, 5])
        hpa._packed_key_intervals(inference_cu, 8, None)
        inference_cu[1] = 5
        changed, _ = hpa._packed_key_intervals(inference_cu, 8, None)
        assert changed.tolist() == [[0] * 5 + [5] * 3]


@pytest.fixture
def lh(hpa, monkeypatch):
    calls = []
    module = types.SimpleNamespace(
        _enabled = lambda: True,
        _shapes_qualify = lambda *args: True,
        _autocast_dtype_differs = lambda q: False,
        _math_sdpa_requested = lambda: False,
        _is_capturing = lambda device: False,
        _self_check = lambda *args: True,
        _SELF_CHECK = {},
        _warn_once = lambda message: None,
        LARGE_HEAD_ENGAGED = {"calls": 0, "fallback": 0},
    )

    def run(q, k, v, lo, hi, scale):
        calls.append((q, k, v, lo, hi, scale))
        return q.transpose(1, 2).contiguous()

    module.large_head_attention = run
    module.seen = calls
    package = sys.modules["_packed_hd512_test.kernels"]
    monkeypatch.setattr(package, "large_head_attention", module, raising = False)
    monkeypatch.setenv("UNSLOTH_HF_PACKED_HD512", "1")
    return module


@pytest.mark.parametrize(
    "reason",
    [
        "default",
        "head_dim",
        "disabled",
        "shape",
        "autocast",
        "math",
        "capture",
        "self_check",
        "compile",
        "rocm",
    ],
)
def test_large_head_refusals_do_not_launch(hpa, lh, monkeypatch, reason):
    q = torch.zeros(1, 4, 8, 256 if reason == "head_dim" else 512)
    if reason == "default":
        monkeypatch.delenv("UNSLOTH_HF_PACKED_HD512")
    elif reason == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif reason == "rocm":
        monkeypatch.setattr(torch.version, "hip", "test")
    elif reason == "autocast":
        monkeypatch.setattr(lh, "_autocast_dtype_differs", lambda *args: True)
    elif reason == "math":
        monkeypatch.setattr(lh, "_math_sdpa_requested", lambda: True)
    elif reason == "capture":
        monkeypatch.setattr(lh, "_is_capturing", lambda *args: True)
    else:
        attribute = {
            "disabled": "_enabled",
            "shape": "_shapes_qualify",
            "self_check": "_self_check",
        }.get(reason)
        if attribute:
            monkeypatch.setattr(lh, attribute, lambda *args: False)
    assert hpa._packed_large_head(q, q, q, _cu([3, 5]), 1.0, None) is None
    assert not lh.seen


@pytest.mark.parametrize("failure", ["refusal", "oom", "accelerator"])
def test_kernel_failures_preserve_cuda_errors(hpa, lh, monkeypatch, failure):
    class AcceleratorFault(RuntimeError):
        pass

    monkeypatch.setattr(hpa, "_ACCELERATOR_ERROR", AcceleratorFault)
    error = {
        "refusal": RuntimeError("unsupported kernel config"),
        "oom": torch.OutOfMemoryError("out of memory"),
        "accelerator": AcceleratorFault("CUDA fault"),
    }[failure]

    def fail(*args):
        raise error

    monkeypatch.setattr(lh, "large_head_attention", fail)
    q = torch.zeros(1, 4, 8, 512)
    if failure == "refusal":
        assert hpa._packed_large_head(q, q, q, _cu([3, 5]), 1.0, None) is None
        assert lh._SELF_CHECK[("cpu", q.dtype)] is False
        assert lh.LARGE_HEAD_ENGAGED["fallback"] == 1
    else:
        with pytest.raises(type(error), match = str(error)):
            hpa._packed_large_head(q, q, q, _cu([3, 5]), 1.0, None)


@pytest.mark.parametrize(
    "case", ["triton", "declined", "cross_document", "policy", "output_attentions", "disabled"]
)
def test_packed_dispatch_keeps_gqa_and_fallback_order(hpa, lh, monkeypatch, case):
    # Mock kernel eligibility but exercise mask validation and metadata.
    monkeypatch.setattr(hpa, "_static_window", lambda *args: None)
    monkeypatch.setattr(hpa, "_efficient_varlen_takes", lambda *args: case != "policy")
    monkeypatch.setattr(hpa, "_ORIG_SDPA", [lambda *args, **kwargs: ("original", None)])
    efficient_calls = []
    monkeypatch.setattr(
        hpa, "_efficient_varlen", lambda *args: efficient_calls.append(args) or "efficient"
    )
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    lengths = [3, 5]
    q, kv = torch.zeros(1, 32, 8, 512), torch.zeros(1, 4, 8, 512)
    mask = _mask(lengths)
    if case == "declined":
        monkeypatch.setattr(lh, "_self_check", lambda *args: False)
    elif case == "cross_document":
        mask[0, 0, 4, 1] = True
    elif case == "disabled":
        monkeypatch.setenv("UNSLOTH_HF_PACKED_HD512", "0")
    result, _ = hpa._sdpa_packed_varlen(
        types.SimpleNamespace(),
        q,
        kv,
        kv,
        mask,
        scaling = 1.0,
        packed_seq_lengths = torch.tensor(lengths),
        output_attentions = case == "output_attentions",
    )
    if case == "triton":
        assert result.shape == (1, 8, 32, 512)
        seen = lh.seen[0]
        assert seen[1] is kv and seen[2] is kv and seen[-1] == 1.0
        assert seen[3].tolist() == [[0] * 3 + [3] * 5]
        assert not efficient_calls
        assert hpa.HF_PACKED_ATTENTION_STATS["large_head"] == 1
        assert hpa.HF_PACKED_ATTENTION_STATS["fast"] == 1
    elif case in ("cross_document", "policy"):
        assert result == "original" and not efficient_calls and not lh.seen
    else:
        assert result == "efficient" and len(efficient_calls) == 1 and not lh.seen
        assert hpa.HF_PACKED_ATTENTION_STATS["efficient"] == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_hd512_block_preparation_ignores_other_layers_cached_route(hpa, monkeypatch, enabled):
    monkeypatch.setenv("UNSLOTH_HF_PACKED_HD512", "1" if enabled else "0")
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    monkeypatch.setattr(hpa, "_block_varlen_prepare", lambda *args: None)
    hpa._BLOCK_PREP.clear()
    hidden = torch.zeros(1, 8, 16)
    kwargs = {"packed_seq_lengths": torch.tensor([3, 5]), "attention_mask": _mask([3, 5])}
    small = types.SimpleNamespace(self_attn = types.SimpleNamespace(head_dim = 256))
    large = types.SimpleNamespace(self_attn = types.SimpleNamespace(head_dim = 512))
    assert hpa.block_varlen_kwargs(small, hidden, kwargs) is None
    assert hpa.block_varlen_kwargs(large, hidden, kwargs) == (hpa._LEAVE_BLOCK if enabled else None)


@pytest.mark.parametrize("in_block", [False, True])
def test_compiled_dispatch_only_leaves_unsloth_blocks(hpa, lh, monkeypatch, in_block):
    monkeypatch.setattr(hpa, "_static_window", lambda *args: None)
    monkeypatch.setattr(hpa, "_efficient_varlen_takes", lambda *args: True)
    monkeypatch.setattr(
        hpa, "_ORIG_SDPA", [lambda module, q, *args, **kwargs: (q.transpose(1, 2), None)]
    )
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    module = types.SimpleNamespace()
    q, kv = torch.zeros(1, 4, 8, 512), torch.zeros(1, 2, 8, 512)
    mask, lengths = _mask([3, 5]), torch.tensor([3, 5])
    kwargs = {"packed_seq_lengths": lengths}
    if in_block:
        kwargs[hpa.BLOCK_VARLEN_KWARG] = hpa._LEAVE_BLOCK

    def forward(q, kv):
        return hpa._sdpa_packed_varlen(module, q, kv, kv, mask, **kwargs)[0]

    compiled = torch.compile(forward, backend = "eager", fullgraph = not in_block)
    assert compiled(q, kv).shape == (1, 8, 4, 512)
    assert len(lh.seen) == int(in_block)
    assert hpa.HF_PACKED_ATTENTION_STATS["large_head"] == int(in_block)
    torch._dynamo.reset()


_CUDA = pytest.mark.skipif(
    not has_real_cuda() or (has_real_cuda() and torch.cuda.get_device_capability()[0] < 8),
    reason = "needs an sm80+ CUDA GPU",
)


@_CUDA
@pytest.mark.parametrize("n_kv", [4, 32])
@pytest.mark.parametrize("shared_kv", [False, True])
@pytest.mark.parametrize("checkpoint", ["none", "reentrant", "nonreentrant"])
@pytest.mark.parametrize("window", [None, 5])
def test_real_packed_kernel_matches_independent_documents(
    hpa, monkeypatch, n_kv, shared_kv, checkpoint, window
):
    pytest.importorskip("triton")
    from torch.utils.checkpoint import checkpoint as checkpoint_fn

    monkeypatch.setenv("UNSLOTH_HF_PACKED_HD512", "1")
    monkeypatch.delenv("UNSLOTH_HD512_ATTENTION", raising = False)
    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    lengths, H, D = [1, 17, 65, 3], 32, 512
    T = sum(lengths)
    q = torch.randn(1, H, T, D, device = "cuda", dtype = torch.bfloat16).requires_grad_()
    k = torch.randn(1, n_kv, T, D, device = "cuda", dtype = torch.bfloat16).requires_grad_()
    v = k if shared_kv else torch.randn_like(k).requires_grad_()
    mask = _mask(lengths, "cuda", window)
    psl = torch.tensor(lengths, dtype = torch.int32)
    module = types.SimpleNamespace(is_causal = True, sliding_window = window)
    scale = 1.0

    def packed(q, k, v):
        return hpa._sdpa_packed_varlen(
            module, q, k, v, mask, scaling = scale, packed_seq_lengths = psl
        )[0]

    out = (
        packed(q, k, v)
        if checkpoint == "none"
        else checkpoint_fn(packed, q, k, v, use_reentrant = checkpoint == "reentrant")
    )
    assert hpa.HF_PACKED_ATTENTION_STATS["large_head"] == 1
    inputs = (q, k) if shared_kv else (q, k, v)
    ref_inputs = tuple(x.detach().float().requires_grad_() for x in inputs)
    qr, kr = ref_inputs[:2]
    vr = kr if shared_kv else ref_inputs[2]
    refs, start = [], 0
    for length in lengths:
        rows = slice(start, start + length)
        keys = kr[:, :, rows].repeat_interleave(H // n_kv, dim = 1)
        vals = vr[:, :, rows].repeat_interleave(H // n_kv, dim = 1)
        refs.append(
            torch.nn.functional.scaled_dot_product_attention(
                qr[:, :, rows], keys, vals, attn_mask = _mask([length], "cuda", window), scale = scale
            ).transpose(1, 2)
        )
        start += length
    ref = torch.cat(refs, dim = 1)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.float())
    for actual, expected in zip(
        (out, *(x.grad for x in inputs)), (ref, *(x.grad for x in ref_inputs))
    ):
        assert torch.isfinite(actual).all()
        err = (actual.float() - expected).abs()
        assert err.max() <= 0.03 * expected.abs().max() + 0.02
        assert err.mean() <= 0.01 * expected.abs().mean() + 2e-3
