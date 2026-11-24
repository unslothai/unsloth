# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The SDPA layout constraint answers mask alignment without a shape guard, so a caption length that is a multiple
of 8 does not recompile the compiled DiT blocks (Qwen-Image-2.1: a 27 s stall at step 16 on L4)."""

import types

import pytest
import torch

import core.training.diffusion_sdpa_align as sa


@pytest.fixture
def fresh(monkeypatch):
    import torch._inductor.lowering as lowering

    def original(fx_node, *args, **kwargs):
        from torch._inductor.virtualized import V
        return V.graph.sizevars.guard_or_false("aligned?"), args

    other = lambda *a, **k: None  # noqa: E731
    table = {"sdpa_fwd": original, "sdpa_bwd": original, "mm": other}
    monkeypatch.setattr(lowering, "sdpa_constraint", original)
    monkeypatch.setattr(lowering, "_maybe_layout_constraints", table)
    monkeypatch.setattr(sa, "_STATE", {})
    monkeypatch.delenv(sa.SDPA_ALIGN_GUARD_ENV, raising = False)
    return original, other, table


class _SizeVars:
    def __init__(self):
        self.guards = []

    def guard_or_false(self, expr):
        self.guards.append(expr)
        return True

    def statically_known_true(self, expr):
        return False


def test_wraps_only_the_sdpa_constraint_once(fresh):
    original, other, table = fresh
    assert sa.install_sdpa_align_without_guard() == 2
    assert table["mm"] is other
    assert table["sdpa_fwd"] is table["sdpa_bwd"] is not original
    assert table["sdpa_fwd"]._unsloth_original is original
    assert sa.install_sdpa_align_without_guard() == 2  # idempotent: not wrapped twice
    assert table["sdpa_fwd"]._unsloth_original is original


def test_the_wrapper_never_guards_and_restores_the_sizevars(fresh):
    from torch._inductor.virtualized import V

    _original, _other, table = fresh
    sa.install_sdpa_align_without_guard()
    sizevars = _SizeVars()
    with V.set_graph_handler(types.SimpleNamespace(sizevars = sizevars)):
        aligned, args = table["sdpa_fwd"](None, 1, 2)
        assert args == (1, 2)
        # Undecided without a guard: the misaligned path, and no guard recorded.
        assert aligned is False and sizevars.guards == []
        # Outside the constraint the real guarding method is back.
        assert "guard_or_false" not in vars(sizevars)
        assert sizevars.guard_or_false("x") is True and sizevars.guards == ["x"]


def test_env_keeps_torchs_guard(fresh, monkeypatch):
    original, _other, table = fresh
    monkeypatch.setenv(sa.SDPA_ALIGN_GUARD_ENV, "1")
    assert sa.install_sdpa_align_without_guard() == 0
    assert table["sdpa_fwd"] is original


def test_missing_inductor_internals_install_nothing(monkeypatch):
    import torch._inductor.lowering as lowering

    monkeypatch.setattr(sa, "_STATE", {})
    monkeypatch.delattr(lowering, "_maybe_layout_constraints")
    assert sa.install_sdpa_align_without_guard() == 0


def test_the_real_torch_registers_sdpa_under_the_wrapped_name():
    """The internals this relies on exist in the installed torch (a rename would silently keep the stall)."""
    import torch._inductor.lowering as lowering

    table = lowering._maybe_layout_constraints
    assert callable(lowering.sdpa_constraint)
    ops = [
        op
        for op, c in table.items()
        if c is lowering.sdpa_constraint
        or getattr(c, "_unsloth_original", None) is lowering.sdpa_constraint
    ]
    assert any("efficient_attention" in str(op) for op in ops)


def _graphs_over_text_lengths(lengths):
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()

    def attend(q, k, v, n: int):
        mask = torch.tril(torch.ones(n, n, dtype = torch.bool, device = q.device))[None, None]
        return torch.nn.functional.scaled_dot_product_attention(
            q[:, :, :n], k[:, :, :n], v[:, :, :n], attn_mask = mask
        )

    f = torch.compile(attend, dynamic = True, fullgraph = True)
    q = torch.randn(1, 2, 64, 64, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    for n in lengths:
        out = f(q, q, q, n)
        out.float().sum().backward()
        ref = attend(q.detach(), q.detach(), q.detach(), n)
        assert torch.allclose(out.detach().float(), ref.float(), atol = 2e-2, rtol = 2e-2)
    return counters["stats"]["unique_graphs"]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "the SDPA layout constraint only applies on CUDA"
)
def test_a_multiple_of_8_text_length_does_not_recompile(monkeypatch):
    import torch._functorch.config as functorch_config
    import torch._inductor.config as inductor_config
    import torch._inductor.lowering as lowering

    # Fresh compiles: a cached artifact carries the guards of the process that wrote it.
    monkeypatch.setattr(inductor_config, "fx_graph_cache", False)
    monkeypatch.setattr(functorch_config, "enable_autograd_cache", False)
    unwrapped = {
        op: getattr(c, "_unsloth_original", c)
        for op, c in lowering._maybe_layout_constraints.items()
    }
    monkeypatch.setattr(lowering, "_maybe_layout_constraints", unwrapped)
    monkeypatch.setattr(sa, "_STATE", {})
    monkeypatch.delenv(sa.SDPA_ALIGN_GUARD_ENV, raising = False)
    lengths = (27, 29, 24, 32, 35)
    # torch's own guard: the first multiple of 8 compiles a second graph (the test can fail).
    assert _graphs_over_text_lengths(lengths) == 2
    assert sa.install_sdpa_align_without_guard() > 0
    assert _graphs_over_text_lengths(lengths) == 1
