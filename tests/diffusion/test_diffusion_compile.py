# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Chunked diffusion CE: parity with dense cross entropy, full-graph capture, no recompiles, sync-free objectives."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from unsloth.models import diffusion_loss
from unsloth.models.diffusion_loss import (
    chunked_cross_entropy,
    compile_backbone_layers,
    diffusion_compile_mode,
)


def _dense(
    hidden,
    targets,
    weights,
    weight = None,
    bias = None,
    softcap = None,
):
    z = hidden if weight is None else F.linear(hidden, weight, bias)
    z = z.float()
    if softcap is not None:
        z = torch.tanh(z / softcap) * softcap
    ce = F.cross_entropy(z.flatten(0, -2), targets.flatten(), reduction = "none", ignore_index = -100)
    return (ce * weights.flatten()).sum(), ce.sum()


@pytest.fixture
def small_chunks(monkeypatch):
    # Several chunks plus a padded tail even for tiny inputs.
    monkeypatch.setattr(diffusion_loss, "_CHUNK_BYTES", 4 * 11 * 8)
    monkeypatch.setattr(diffusion_loss, "_MIN_ROWS", 4)


@pytest.mark.parametrize("project", [True, False])
@pytest.mark.parametrize("softcap", [None, 3.0])
def test_chunked_ce_matches_dense_values_and_grads(small_chunks, project, softcap):
    torch.manual_seed(0)
    batch, length, width, vocab = 3, 13, 8, 11
    targets = torch.randint(0, vocab, (batch, length))
    targets[0, :4] = -100
    targets[2] = -100  # a fully ignored row
    weights = torch.rand(batch, length) * (targets != -100)
    if project:
        hidden = torch.randn(batch, length, width, requires_grad = True)
        weight = torch.randn(vocab, width, requires_grad = True)
        bias = torch.randn(vocab, requires_grad = True)
        params = [hidden, weight, bias]
    else:
        hidden = torch.randn(batch, length, vocab, requires_grad = True)
        weight = bias = None
        params = [hidden]
    ours, ours_ce, count = chunked_cross_entropy(hidden, targets, weights, weight, bias, softcap)
    ref, ref_ce = _dense(hidden, targets, weights, weight, bias, softcap)
    torch.testing.assert_close(ours, ref, rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(ours_ce, ref_ce, rtol = 1e-5, atol = 1e-6)
    assert int(count) == int((targets != -100).sum())
    for a, b in zip(torch.autograd.grad(ours, params), torch.autograd.grad(ref, params)):
        torch.testing.assert_close(a, b, rtol = 1e-5, atol = 1e-6)


def test_chunked_ce_all_ignored_is_zero_with_grad():
    hidden = torch.randn(2, 5, 7, requires_grad = True)
    weighted, total, count = chunked_cross_entropy(hidden, torch.full((2, 5), -100))
    assert float(weighted) == 0.0 and float(total) == 0.0 and int(count) == 0
    (grad,) = torch.autograd.grad(weighted, hidden)
    assert torch.count_nonzero(grad) == 0


def test_chunked_ce_frozen_head_gets_no_grad():
    hidden = torch.randn(2, 5, 4, requires_grad = True)
    head = torch.nn.Linear(4, 9, bias = False).requires_grad_(False)
    weighted, _, _ = chunked_cross_entropy(hidden, torch.randint(0, 9, (2, 5)), weight = head.weight)
    weighted.backward()
    assert hidden.grad is not None and head.weight.grad is None


def _kernel_args(
    rows = 16,
    width = 8,
    vocab = 11,
):
    return (
        torch.randn(rows, width),
        torch.randn(vocab, width),
        None,
        torch.randint(0, vocab, (rows,)),
        torch.rand(rows),
    )


def test_chunk_kernels_capture_without_graph_breaks():
    h, w, b, t, tw = _kernel_args()
    forward = torch._dynamo.explain(diffusion_loss._chunk_forward)(h, w, b, t, tw, 3.0)
    assert forward.graph_break_count == 0 and forward.graph_count == 1
    lse = torch.logsumexp(F.linear(h, w).float(), -1)
    backward = torch._dynamo.explain(diffusion_loss._chunk_backward)(
        h, w, b, t, tw, lse, torch.tensor(1.0), 3.0
    )
    assert backward.graph_break_count == 0 and backward.graph_count == 1


def _compiled_kernels_or_skip():
    diffusion_loss._COMPILED.clear()
    try:
        forward, _ = diffusion_loss._kernels(True)
        with torch.no_grad():  # as inside the autograd.Function, so the probe shares its guards
            forward(*_kernel_args(), None)
    except Exception as error:  # no C++ toolchain for inductor on this host
        pytest.skip(f"torch.compile unavailable here: {type(error).__name__}")
    diffusion_loss._COMPILED.clear()
    torch._dynamo.reset()


def test_compiled_loss_matches_eager_and_never_recompiles_on_new_masks():
    _compiled_kernels_or_skip()
    torch.manual_seed(0)
    hidden = torch.randn(4, 9, 8, requires_grad = True)
    weight = torch.randn(11, 8, requires_grad = True)
    with torch._dynamo.config.patch(error_on_recompile = True):
        for fraction in (0.1, 0.5, 0.9, 0.0):
            targets = torch.randint(0, 11, (4, 9)).masked_fill(torch.rand(4, 9) > fraction, -100)
            weights = torch.rand(4, 9) * (targets != -100)
            eager, _, _ = chunked_cross_entropy(hidden, targets, weights, weight)
            compiled, _, _ = chunked_cross_entropy(hidden, targets, weights, weight, compiled = True)
            torch.testing.assert_close(compiled, eager, rtol = 1e-5, atol = 1e-6)
            for a, b in zip(
                torch.autograd.grad(compiled, [hidden, weight]),
                torch.autograd.grad(eager, [hidden, weight]),
            ):
                torch.testing.assert_close(a, b, rtol = 1e-5, atol = 1e-6)


def test_compile_mode_parsing(monkeypatch):
    monkeypatch.delenv("UNSLOTH_COMPILE_DISABLE", raising = False)
    assert diffusion_compile_mode(None) == "auto"
    assert diffusion_compile_mode(SimpleNamespace(diffusion_compile = False)) is None
    assert diffusion_compile_mode(SimpleNamespace(diffusion_compile = "eager")) is None
    assert diffusion_compile_mode(SimpleNamespace(diffusion_compile = True)) == "full"
    assert diffusion_compile_mode(SimpleNamespace(diffusion_compile = "loss")) == "loss"
    with pytest.raises(ValueError, match = "diffusion_compile"):
        diffusion_compile_mode(SimpleNamespace(diffusion_compile = "fast"))
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "1")
    assert diffusion_compile_mode(SimpleNamespace(diffusion_compile = True)) is None


def test_compile_backbone_layers_is_idempotent():
    class _Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(3))
            self.head = torch.nn.Linear(4, 4)

    net = _Net()
    assert compile_backbone_layers(net) == 3
    assert compile_backbone_layers(net) == 0
    assert all(getattr(layer, "_unsloth_diffusion_compiled", False) for layer in net.layers)


def test_chunked_ce_backward_follows_forward_autocast():
    # fp32 weights under bf16 autocast: backward must rebuild the bf16 logits the forward normalised.
    torch.manual_seed(0)
    hidden = torch.randn(2, 6, 8, requires_grad = True)
    weight = torch.randn(13, 8, requires_grad = True)
    targets = torch.randint(0, 13, (2, 6))
    with torch.autocast("cpu", dtype = torch.bfloat16):
        ours, _, _ = chunked_cross_entropy(hidden, targets, weight = weight)
        ref = F.cross_entropy(
            F.linear(hidden, weight).float().flatten(0, 1), targets.flatten(), reduction = "sum"
        )
    torch.testing.assert_close(ours, ref, rtol = 1e-5, atol = 1e-5)
    for a, b in zip(
        torch.autograd.grad(ours, [hidden, weight]), torch.autograd.grad(ref, [hidden, weight])
    ):
        torch.testing.assert_close(a, b, rtol = 1e-3, atol = 1e-3)


def test_chunked_ce_empty_input_is_zero_with_grad():
    # DiffusionGemma's encoder AR slice of a one-token batch has no rows at all.
    hidden = torch.randn(0, 8, requires_grad = True)
    weight = torch.randn(11, 8, requires_grad = True)
    weighted, total, count = chunked_cross_entropy(
        hidden, torch.empty(0, dtype = torch.long), weight = weight
    )
    weighted.backward()
    assert float(weighted) == 0.0 and int(count) == 0
    assert weight.grad is not None and float(weight.grad.abs().sum()) == 0.0


def test_chunked_ce_leaves_the_autocast_cache_alone():
    # A later projection of the same weight in the same autocast region must still reach the weight (older
    # torch reused the gradient-less cast made inside the custom forward).
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    weight = torch.randn(11, 8, device = device, requires_grad = True)
    hidden = torch.randn(5, 8, device = device)
    targets = torch.randint(0, 11, (5,), device = device)

    def grads(with_ce):
        weight.grad = None
        with torch.autocast(device, dtype = torch.bfloat16):
            total = F.linear(hidden, weight).float().sum() * 0
            if with_ce:
                total = total + chunked_cross_entropy(hidden, targets, weight = weight)[0]
            total = total + F.linear(hidden, weight).float().sum()
        total.backward()
        return weight.grad.clone()

    ce_only = torch.func.grad(
        lambda w: _dense(hidden, targets, (targets >= 0).float(), weight = w)[0]
    )(weight.detach())
    torch.testing.assert_close(grads(True) - grads(False), ce_only, rtol = 5e-2, atol = 5e-2)


def test_chunked_ce_ignored_overflow_rows_do_not_poison_grads():
    hidden = torch.tensor([[1.0, 0.0], [float("inf"), 0.0]], requires_grad = True)
    targets = torch.tensor([0, -100])
    weighted, _, _ = chunked_cross_entropy(hidden, targets)
    weighted.backward()
    assert torch.isfinite(hidden.grad).all()
