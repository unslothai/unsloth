# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Unsloth's process-wide non-reentrant checkpoint inside a compiled region (CPU, no unsloth import).

Dynamo splits a traced checkpoint call's keyword arguments by inspect.signature(torch.utils.checkpoint.checkpoint);
DeepSeek-V4.1's chunked sparse attention (the torch backend: T4, CPU) checkpoints each query chunk with
use_reentrant = False inside its compiled attention, under the wrapper vision.py installs."""

import ast
import importlib.util
import inspect
import math
import os

import pytest
import torch
import torch.utils.checkpoint as torch_checkpoint

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION = os.path.join(HERE, "unsloth", "models", "vision.py")
PRISTINE = getattr(torch_checkpoint.checkpoint, "_unsloth_original", torch_checkpoint.checkpoint)


def _vision_wrapper(offload, hooks):
    """vision.py's `_nonre_checkpoint = non_reentrant_checkpoint(...)`, executed as written."""
    spec = importlib.util.spec_from_file_location(
        "_unsloth_non_reentrant_checkpoint",
        os.path.join(HERE, "unsloth", "models", "non_reentrant_checkpoint.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    src = open(VISION, encoding = "utf-8").read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_nonre_checkpoint" for t in node.targets
        ):
            ns = {
                "non_reentrant_checkpoint": module.non_reentrant_checkpoint,
                "_orig_checkpoint": PRISTINE,
                "_offload": offload,
                "_offload_activation_hooks": lambda owner = None: hooks(),
                "_gc_checkpoint_owner": lambda function: None,
            }
            exec(ast.get_source_segment(src, node), ns)
            return ns["_nonre_checkpoint"]
    pytest.fail("vision.py does not build _nonre_checkpoint with non_reentrant_checkpoint")


class _Hooks:
    def __init__(self):
        self.entered = 0

    def __call__(self):
        self.entered += 1
        return torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t)


@pytest.fixture
def installed(monkeypatch):
    hooks = _Hooks()

    def install(offload = True):
        wrapper = _vision_wrapper(offload, hooks)
        monkeypatch.setattr(torch_checkpoint, "checkpoint", wrapper)
        return wrapper

    torch._dynamo.reset()
    yield install, hooks
    torch._dynamo.reset()


def test_wrapper_keeps_torch_checkpoint_signature(installed):
    install, _ = installed
    wrapper = install()
    assert inspect.signature(wrapper) == inspect.signature(PRISTINE)
    assert "use_reentrant" in inspect.signature(wrapper).parameters
    assert wrapper._unsloth_original is PRISTINE and wrapper._unsloth_offloads is True


@pytest.mark.parametrize("offload", [True, False])
def test_checkpoint_with_use_reentrant_inside_a_compiled_region(installed, offload):
    install, hooks = installed
    install(offload)

    def block(
        x,
        w,
        scale = 1.0,
    ):
        return torch.tanh(x @ w) * scale

    def fn(x, w):
        a = torch_checkpoint.checkpoint(block, x, w, use_reentrant = False, scale = 0.5)
        return torch_checkpoint.checkpoint(block, a, w, use_reentrant = False).sum()

    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 8, generator = g, dtype = torch.float64)
    w = torch.randn(8, 8, generator = g, dtype = torch.float64)
    leaves = [t.clone().requires_grad_() for t in (x, w)]
    want = torch.autograd.grad(block(block(*leaves, scale = 0.5), leaves[1]).sum(), leaves)

    leaves = [t.clone().requires_grad_() for t in (x, w)]
    out = torch.compile(fn, backend = "aot_eager", fullgraph = True)(*leaves)
    got = torch.autograd.grad(out, leaves)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, rtol = 1e-12, atol = 1e-12)
    # The compiled graph recomputes on its own: offload hooks only around eager checkpoints.
    assert hooks.entered == 0
    leaves = [t.clone().requires_grad_() for t in (x, w)]
    torch.autograd.grad(fn(*leaves), leaves)
    assert hooks.entered == (2 if offload else 0)


def _naive(q, kv, sink, idxs, scale):
    """Row-by-row fp64 transcription of deepseek-ai's sparse_attn (K = V = kv, sink only in the denominator)."""
    b, s, h, d = q.shape
    out = torch.zeros(b, s, h, d, dtype = torch.float64)
    for bi in range(b):
        for t in range(s):
            rows = [int(i) for i in idxs[bi, t] if 0 <= int(i) < kv.shape[1]]
            for hi in range(h):
                if not rows:
                    continue
                logits = [float(q[bi, t, hi].double() @ kv[bi, r].double()) * scale for r in rows]
                m = max(logits)
                weights = [math.exp(x - m) for x in logits]
                denom = sum(weights) + math.exp(float(sink[hi]) - m)
                out[bi, t, hi] = sum(w * kv[bi, r].double() for w, r in zip(weights, rows)) / denom
    return out


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_chunked_sparse_attention_compiles_under_the_wrapper_and_matches_fp64(
    installed, monkeypatch, dtype
):
    """The T4 / CPU sparse attention path: chunked reference with a checkpoint per query chunk, compiled."""
    sa = pytest.importorskip("unsloth_zoo.sparse_attention")
    install, hooks = installed
    b, s, h, d, n, k = 1, 12, 2, 8, 10, 4
    monkeypatch.setattr(sa, "_TORCH_CHUNK_ELEMS", 4 * k * d * b)  # 3 chunks of 4 query rows
    g = torch.Generator().manual_seed(0)
    q = torch.randn(b, s, h, d, generator = g, dtype = torch.float64) * 0.5
    kv = torch.randn(b, n, d, generator = g, dtype = torch.float64) * 0.5
    sink = torch.randn(h, generator = g, dtype = torch.float64)
    idxs = torch.stack([torch.randperm(n, generator = g)[:k] for _ in range(s)]).unsqueeze(0)
    idxs[0, 2] = -1  # an empty row
    idxs[0, 5, 1] = -1
    idxs = idxs.to(torch.int32)
    go = torch.randn(b, s, h, d, generator = g, dtype = torch.float64)

    oracle = [t.clone().requires_grad_() for t in (q, kv, sink)]
    torch.testing.assert_close(
        _naive(*[t.detach() for t in oracle], idxs, 0.3),
        sa.sparse_attn_reference(*oracle, idxs, 0.3),
        rtol = 1e-12,
        atol = 1e-12,
    )
    want_grads = torch.autograd.grad(sa.sparse_attn_reference(*oracle, idxs, 0.3), oracle, go)

    install(offload = True)
    leaves = [t.to(dtype).requires_grad_() for t in (q, kv, sink)]
    compiled = torch.compile(
        lambda q, kv, sink: sa.sparse_attn_reference(q, kv, sink, idxs, 0.3),
        backend = "aot_eager",
        fullgraph = True,
    )
    out = compiled(*leaves)
    grads = torch.autograd.grad(out, leaves, go.to(dtype))
    tol = dict(rtol = 1e-12, atol = 1e-12) if dtype == torch.float64 else dict(rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(out.double(), _naive(q, kv, sink, idxs, 0.3), **tol)
    for a, b in zip(grads, want_grads):
        torch.testing.assert_close(a.double(), b, **tol)
    assert hooks.entered == 0
