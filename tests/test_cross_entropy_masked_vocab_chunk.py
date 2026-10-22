# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A vocab chunk that is entirely -inf (caller-masked vocabulary) must not turn the loss into NaN.

The chunked forward takes c = max(chunk); for an all -inf chunk that is -inf and logits - c is NaN.
16K chunks (T4 / A100 at vocab 128256) reach this where 64K chunks did not.
"""

import ast
import importlib.util
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn.functional as F

SOURCE = (
    pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "kernels" / "cross_entropy_loss.py"
)
VOCAB = 128256
CHUNK = 16384


def _inputs():
    generator = torch.Generator().manual_seed(14498)
    logits = (torch.randn(3, VOCAB, generator = generator) * 3).half()
    labels = torch.tensor([20000, 70000, -100])
    logits[0, :CHUNK] = -float("inf")  # first chunk masked, label in the second
    logits[1, 2 * CHUNK : 3 * CHUNK] = -float("inf")  # a middle chunk masked
    logits[2, :CHUNK] = -float("inf")  # ignored row with a masked chunk
    return logits, labels


def _reference(logits, labels):
    x = logits.double()
    active = labels != -100
    count = int(active.sum())
    loss = F.cross_entropy(x, labels, reduction = "sum") / count
    grad = torch.softmax(x, dim = -1)
    rows = torch.arange(labels.numel())[active]
    grad[rows, labels[active]] -= 1
    grad[~active] = 0
    return loss, grad / count


# Runs in a fresh interpreter: TRITON_INTERPRET must be set before triton is imported.
_INTERPRETER_SCRIPT = textwrap.dedent(
    """
    import ast, sys
    import torch
    import triton
    import triton.language as tl

    source, out = sys.argv[1], sys.argv[2]
    names = {"_chunked_cross_entropy_forward", "_cross_entropy_backward"}
    tree = ast.parse(open(source).read(), filename = source)
    body = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets))
    ]

    @triton.jit
    def triton_cast(x: tl.constexpr, dtype: tl.constexpr):
        # The interpreter cannot tl.cast a Python int stride.
        return tl.full((), x, dtype)

    @triton.jit
    def triton_tanh(x):
        return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0

    ns = dict(torch = torch, triton = triton, tl = tl, triton_cast = triton_cast, triton_tanh = triton_tanh)
    exec(compile(ast.Module(body = body, type_ignores = []), source, "exec"), ns)

    data = torch.load(sys.argv[3])
    logits, labels, block, warps = data["logits"], data["labels"], data["block"], data["warps"]
    rows, vocab = logits.shape
    n_chunks = -(-vocab // block)
    losses = torch.empty(rows, dtype = torch.float32)
    chunk_lse = torch.empty((rows, n_chunks), dtype = torch.float32)
    ns["_chunked_cross_entropy_forward"][(rows, n_chunks)](
        logits, logits.stride(0), losses, chunk_lse, labels,
        VOCAB_SIZE = vocab, N_CHUNKS = n_chunks, BLOCK_SIZE = block,
        DO_SOFTCAPPING = False, SOFTCAP = 0.0, DO_LOGIT_SCALING = False, LOGIT_SCALE = 0.0,
        EVICT = "", num_warps = warps,
    )
    # Same host-side reduction as Fast_CrossEntropyLoss.forward.
    logsumexp = torch.logsumexp(chunk_lse, dim = 1)
    losses += logsumexp
    losses.masked_fill_(labels == -100, 0)

    count = int((labels != -100).sum())
    dlosses = torch.full((rows,), 1.0 / count)
    dlogits = torch.empty_like(logits)
    ns["_cross_entropy_backward"][(rows, -(-vocab // 4096))](
        logits, logits.stride(0), dlogits, dlosses, dlosses.stride(0), logsumexp, labels,
        VOCAB_SIZE = vocab, BLOCK_SIZE = 4096,
        DO_SOFTCAPPING = False, SOFTCAP = 0.0, DO_LOGIT_SCALING = False, LOGIT_SCALE = 0.0,
        num_warps = 8,
    )
    torch.save({"loss": losses.sum() / count, "grad": dlogits}, out)
    """
)


def _check(loss, grad, logits, labels):
    expected_loss, expected_grad = _reference(logits, labels)
    assert torch.isfinite(loss), f"loss is {loss.item()}"
    assert torch.isfinite(grad).all(), f"{int((~torch.isfinite(grad)).sum())} non-finite gradients"
    torch.testing.assert_close(loss.double(), expected_loss, rtol = 1e-6, atol = 1e-5)
    torch.testing.assert_close(grad.double(), expected_grad, rtol = 2e-3, atol = 3e-7)


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason = "triton not installed")
@pytest.mark.parametrize("block,warps", [(CHUNK, 8), (65536, 32)])
def test_masked_vocab_chunk_matches_fp64_interpreter(tmp_path, block, warps):
    """Runs the real kernels from the source file under TRITON_INTERPRET=1 on CPU."""
    logits, labels = _inputs()
    data, out, script = tmp_path / "in.pt", tmp_path / "out.pt", tmp_path / "run.py"
    torch.save({"logits": logits, "labels": labels, "block": block, "warps": warps}, data)
    script.write_text(_INTERPRETER_SCRIPT)
    env = dict(os.environ, TRITON_INTERPRET = "1", CUDA_VISIBLE_DEVICES = "")
    result = subprocess.run(
        [sys.executable, str(script), str(SOURCE), str(out), str(data)],
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    got = torch.load(out)
    _check(got["loss"], got["grad"], logits, labels)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason = "CUDA Triton kernels required",
)
@pytest.mark.parametrize("config", [(CHUNK, 8), (65536, 32)])
def test_masked_vocab_chunk_matches_fp64_cuda(monkeypatch, config):
    from unsloth.kernels import cross_entropy_loss as ce_mod

    monkeypatch.setattr(ce_mod, "_chunked_forward_config", lambda device, vocab_size: config)
    logits, labels = _inputs()
    actual_logits = logits.cuda().requires_grad_()
    loss = ce_mod.fast_cross_entropy_loss(actual_logits[None], labels.cuda()[None])
    loss.backward()
    _check(loss.float().cpu(), actual_logits.grad.float().cpu(), logits, labels)


def test_chunked_forward_guards_an_all_masked_chunk():
    """Source pin: the chunk max is reset before the subtraction in the chunked forward."""
    tree = ast.parse(SOURCE.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_chunked_cross_entropy_forward"
    )
    text = ast.unparse(fn)
    guard = text.find("c = tl.where(c == -float('inf'), 0.0, c)")
    assert guard != -1
    assert guard < text.find("logsumexp = c + tl.log")
