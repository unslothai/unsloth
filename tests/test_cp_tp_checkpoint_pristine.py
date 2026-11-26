# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The Ulysses chunked masked attention and the vocab-parallel cross entropy checkpoint their chunks with
torch's own non-reentrant checkpoint, also while Unsloth's smart gradient checkpointing has replaced
torch.utils.checkpoint.checkpoint with its forced-reentrant, CPU-offloading version: no chunk goes
through UnslothCheckpointFunction, torch.autograd.grad works (reentrant checkpoint refuses it), and the
grads, closed-over gpt-oss sinks included, match the unpatched run."""

import pytest
import torch
import torch.utils.checkpoint as torch_checkpoint

import unsloth  # noqa: F401
from unsloth.utils import ulysses_hybrid as uh

gc = pytest.importorskip("unsloth_zoo.gradient_checkpointing")


class _Attn(torch.nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.num_key_value_groups = groups
        self.is_causal = True


def _eager_with_sinks(calls):
    def attention(
        module,
        q,
        k,
        v,
        mask,
        s_aux = None,
        scaling = 1.0,
        **kwargs,
    ):
        calls.append(1)
        groups = module.num_key_value_groups
        k = k.repeat_interleave(groups, dim = 1)
        v = v.repeat_interleave(groups, dim = 1)
        scores = (q @ k.transpose(-1, -2)) * scaling + mask
        sink = s_aux.reshape(1, -1, 1, 1).expand(scores.shape[0], -1, scores.shape[2], 1)
        probs = torch.cat([scores, sink.to(scores.dtype)], dim = -1).softmax(dim = -1)[..., :-1]
        return (probs @ v).transpose(1, 2).contiguous(), None

    return attention


@pytest.fixture
def smart_gc(monkeypatch):
    """Unsloth's smart gradient checkpointing installed, as after FastModel / get_peft_model."""
    original = torch_checkpoint.checkpoint
    if torch.cuda.is_available():
        gc.patch_unsloth_smart_gradient_checkpointing(dtype = torch.float32)
    else:
        # The pinned offload buffers need an accelerator; install only the checkpoint replacement.
        gc._capture_pristine_checkpoint_once()
        monkeypatch.setattr(torch_checkpoint, "checkpoint", gc.unsloth_checkpoint)
    assert torch_checkpoint.checkpoint is not original
    yield
    if torch.cuda.is_available():
        gc.unpatch_unsloth_smart_gradient_checkpointing()
    torch_checkpoint.checkpoint = original


@pytest.fixture
def offload_calls(monkeypatch):
    """Calls into the reentrant, CPU-offloading UnslothCheckpointFunction."""
    calls = []
    apply = gc.UnslothCheckpointFunction.apply

    def counted(*args):
        calls.append(1)
        return apply(*args)

    monkeypatch.setattr(gc.UnslothCheckpointFunction, "apply", counted)
    return calls


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
needs_accelerator = pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "UnslothCheckpointFunction offloads through CUDA buffers"
)


def _inputs(
    total = 64,
    heads = 4,
    kv_heads = 2,
    dim = 8,
):
    g = torch.Generator().manual_seed(0)
    shapes = (
        (1, heads, total, dim),
        (1, kv_heads, total, dim),
        (1, kv_heads, total, dim),
        (2 * heads,),
    )
    q, k, v, sinks = (
        torch.randn(shape, generator = g).to(DEVICE).requires_grad_() for shape in shapes
    )
    grad = torch.randn(1, total, heads, dim, generator = g).to(DEVICE)
    return q, k, v, sinks, grad


def _chunked(
    q,
    k,
    v,
    sinks,
    calls,
    rows = 16,
):
    heads = slice(q.shape[1], 2 * q.shape[1])  # this rank's sinks: a non-leaf slice, as in Ulysses
    kwargs = {"s_aux": sinks[heads], "scaling": 0.3}
    return uh._chunked_masked_attention(
        _Attn(2), q, k, v, _eager_with_sinks(calls), kwargs, 24, None, None, rows
    )


def _reference():
    q, k, v, sinks, grad = _inputs()
    out = _chunked(q, k, v, sinks, [])
    return out.detach(), torch.autograd.grad(out, (q, k, v, sinks), grad)


def test_chunked_attention_skips_the_offloading_checkpoint(smart_gc, offload_calls):
    q, k, v, sinks, grad = _inputs()
    out = _chunked(q, k, v, sinks, [])
    assert offload_calls == []
    assert out.requires_grad


@needs_accelerator
def test_chunked_attention_grads_under_smart_gc(smart_gc):
    want_out, want = _reference_unpatched()
    q, k, v, sinks, grad = _inputs()
    calls = []
    out = _chunked(q, k, v, sinks, calls)
    assert len(calls) == 4  # one forward per chunk
    got = torch.autograd.grad(out, (q, k, v, sinks), grad)
    assert len(calls) == 8  # one recompute per chunk
    torch.testing.assert_close(out, want_out, rtol = 0, atol = 0)
    for name, a, b in zip(("dq", "dk", "dv", "dsinks"), got, want):
        torch.testing.assert_close(a, b, rtol = 1e-6, atol = 1e-6, msg = name)


@needs_accelerator
def test_chunked_attention_inside_layer_recompute(smart_gc):
    """The decoder layer is itself checkpointed (reentrant, by smart GC); the chunks run inside it."""
    want_out, want = _reference_unpatched()
    q, k, v, sinks, grad = _inputs()
    calls = []

    def layer(q, k, v, sinks):
        return _chunked(q, k, v, sinks, calls)

    out = torch_checkpoint.checkpoint(layer, q, k, v, sinks, use_reentrant = True)
    (out * grad).sum().backward()
    torch.testing.assert_close(out, want_out, rtol = 0, atol = 0)
    for name, a, b in zip(("dq", "dk", "dv", "dsinks"), (q.grad, k.grad, v.grad, sinks.grad), want):
        torch.testing.assert_close(a, b, rtol = 1e-6, atol = 1e-6, msg = name)
    # forward 4, layer recompute 4, one chunk recompute each 4 (main gives the same count and grads here)
    assert len(calls) == 12, len(calls)


_UNPATCHED = {}


def _reference_unpatched():
    if not _UNPATCHED:
        current = torch_checkpoint.checkpoint
        torch_checkpoint.checkpoint = getattr(
            torch_checkpoint, "_unsloth_pristine_checkpoint", None
        ) or getattr(torch_checkpoint, "_old_checkpoint", current)
        try:
            _UNPATCHED["value"] = _reference()
        finally:
            torch_checkpoint.checkpoint = current
    return _UNPATCHED["value"]


def test_vocab_parallel_ce_skips_the_offloading_checkpoint(smart_gc, offload_calls, monkeypatch):
    """vocab_parallel_cross_entropy's chunk loop, single rank: torch's own checkpoint, same grads."""
    from unsloth.models import tp_native as tpn

    g = torch.Generator().manual_seed(1)
    head = torch.nn.Linear(8, 32, bias = False).to(DEVICE)
    head.__dict__["_unsloth_tp_group"] = None
    head.__dict__["_unsloth_vocab_rows"] = (0, 32, 32)
    monkeypatch.setattr(tpn, "copy_to_tp", lambda x, group: x)

    def chunk(head, xs, ls, softcap, scale, group, lo, width):
        return torch.nn.functional.cross_entropy(head(xs), ls, ignore_index = -100, reduction = "sum")

    monkeypatch.setattr(tpn, "_vocab_ce_chunk", chunk)
    hidden = torch.randn(2, 12, 8, generator = g).to(DEVICE).requires_grad_()
    labels = torch.randint(0, 32, (2, 12), generator = g).to(DEVICE)
    want = torch.nn.functional.cross_entropy(
        head(hidden)[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1)
    )
    want_grads = torch.autograd.grad(want, (hidden, head.weight))
    loss = tpn.vocab_parallel_cross_entropy(head, hidden, labels, chunk_tokens = 5)
    assert offload_calls == []
    grads = torch.autograd.grad(loss, (hidden, head.weight))
    torch.testing.assert_close(loss, want)
    for a, b in zip(grads, want_grads):
        torch.testing.assert_close(a, b)
