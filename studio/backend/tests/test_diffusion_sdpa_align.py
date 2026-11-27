# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The SDPA layout constraint answers mask alignment without a shape guard, so a caption length that is a multiple
of 8 does not recompile the compiled DiT blocks (Qwen-Image-2.1: a 27 s stall at step 16 on L4). Only alignment
questions change, the cache key is tagged while installed, and uninstall restores torch."""

import types

import pytest
import sympy
import torch

import core.training.diffusion_sdpa_align as sa


class _SizeVars:
    """Records what reaches torch's guarding method; statically knows only literal True / False."""

    def __init__(self):
        self.guards = []

    def guard_or_false(self, expr):
        self.guards.append(expr)
        return True

    def statically_known_true(self, expr):
        return expr == sympy.true


L = sympy.Symbol("s0", integer = True, positive = True)
ALIGN = sympy.And(sympy.Eq(sympy.Mod(L, 8), 0), sympy.Eq(sympy.Mod(L * L, 8), 0))
OTHER = sympy.Le(L, 64)


needs_internals = pytest.mark.skipif(
    not getattr(sa, "internals_supported", lambda: True)(),
    reason = "this torch lacks the SDPA alignment internals the wrapper patches",
)


@pytest.fixture
def fresh(monkeypatch):
    import torch._inductor.lowering as lowering

    def original(fx_node, *exprs):
        from torch._inductor.virtualized import V
        return [V.graph.sizevars.guard_or_false(e) for e in exprs]

    other = lambda *a, **k: None  # noqa: E731
    table = {"sdpa_fwd": original, "sdpa_bwd": original, "mm": other}
    monkeypatch.setattr(lowering, "sdpa_constraint", original)
    monkeypatch.setattr(lowering, "_maybe_layout_constraints", table)
    monkeypatch.setattr(sa, "_STATE", {})
    monkeypatch.delenv(sa.SDPA_ALIGN_GUARD_ENV, raising = False)
    cconfig = torch.compiler.config
    try:  # older torch has no cache_key_tag (and touching it leaves its config inconsistent)
        has_tag = isinstance(cconfig.cache_key_tag, str)
    except AttributeError:
        has_tag = False
    if has_tag:
        monkeypatch.setattr(cconfig, "cache_key_tag", "|other_tag")
    yield original, other, table
    sa.uninstall_sdpa_align_without_guard()


@needs_internals
def test_wraps_only_the_sdpa_constraint_once(fresh):
    original, other, table = fresh
    assert sa.install_sdpa_align_without_guard() == 2
    assert table["mm"] is other
    assert table["sdpa_fwd"] is table["sdpa_bwd"] is not original
    assert table["sdpa_fwd"]._unsloth_original is original
    assert sa.install_sdpa_align_without_guard() == 2  # idempotent: not wrapped twice
    assert table["sdpa_fwd"]._unsloth_original is original


@needs_internals
def test_only_alignment_questions_skip_the_guard(fresh):
    from torch._inductor.virtualized import V

    _original, _other, table = fresh
    sa.install_sdpa_align_without_guard()
    sizevars = _SizeVars()
    with V.set_graph_handler(types.SimpleNamespace(sizevars = sizevars)):
        aligned, other = table["sdpa_fwd"](None, ALIGN, OTHER)
        # The alignment question is undecided without a guard: the misaligned path, nothing recorded for it.
        assert aligned is False
        # Any other question still goes to torch's guarding method.
        assert other is True and sizevars.guards == [OTHER]
        assert "guard_or_false" not in vars(sizevars)


def test_alignment_question_detection():
    sv = _SizeVars()
    assert sa.is_alignment_question(ALIGN, sv)
    assert sa.is_alignment_question(sympy.Eq(sympy.Mod(L, 8), 0), sv)
    assert not sa.is_alignment_question(OTHER, sv)
    # An undecided non-alignment relation next to the alignment one keeps the guard.
    assert not sa.is_alignment_question(sympy.And(sympy.Eq(sympy.Mod(L, 8), 0), OTHER), sv)
    assert not sa.is_alignment_question("not sympy", sv)


@needs_internals
def test_cache_tag_while_installed_and_uninstall_restores(fresh):
    original, other, table = fresh
    cconfig = torch.compiler.config
    sa.install_sdpa_align_without_guard()
    assert cconfig.cache_key_tag == "|other_tag" + sa.CACHE_TAG
    assert sa.installed()
    assert sa.uninstall_sdpa_align_without_guard() == 2
    assert table["sdpa_fwd"] is table["sdpa_bwd"] is original and table["mm"] is other
    assert cconfig.cache_key_tag == "|other_tag"
    assert not sa.installed()
    assert sa.uninstall_sdpa_align_without_guard() == 0  # restoring twice is a no-op


@needs_internals
def test_env_keeps_torchs_guard(fresh, monkeypatch):
    original, _other, table = fresh
    monkeypatch.setenv(sa.SDPA_ALIGN_GUARD_ENV, "1")
    assert sa.install_sdpa_align_without_guard() == 0
    assert table["sdpa_fwd"] is original
    assert sa.CACHE_TAG not in (torch.compiler.config.cache_key_tag or "")


def test_missing_inductor_internals_install_nothing(monkeypatch):
    import torch._inductor.lowering as lowering

    monkeypatch.setattr(sa, "_STATE", {})
    monkeypatch.delattr(lowering, "_maybe_layout_constraints")
    assert sa.install_sdpa_align_without_guard() == 0


@needs_internals
def test_the_real_torch_registers_sdpa_under_the_wrapped_name():
    """The internals this relies on exist in the installed torch (a rename would silently keep the stall)."""
    import torch._inductor.lowering as lowering

    table = lowering._maybe_layout_constraints
    assert callable(lowering.sdpa_constraint)
    ops = [op for op, c in table.items() if c is lowering.sdpa_constraint]
    assert any("efficient_attention" in str(op) for op in ops)


def test_without_the_internals_the_wrapper_stays_out(fresh, monkeypatch):
    """torch 2.4 to 2.7: no SizeVarAllocator.statically_known_true (and is_aligned_realized_tensor does not guard
    through guard_or_false). Installing there used to raise a LoweringException on the first compiled SDPA."""
    from torch._inductor.sizevars import SizeVarAllocator

    original, _other, table = fresh
    monkeypatch.delattr(SizeVarAllocator, "statically_known_true", raising = False)
    assert not sa.internals_supported()
    assert sa.install_sdpa_align_without_guard() == 0
    assert table["sdpa_fwd"] is original and not sa.installed()


def test_without_is_aligned_realized_tensor_the_wrapper_stays_out(fresh, monkeypatch):
    import torch._inductor.ir as ir

    original, _other, table = fresh
    monkeypatch.delattr(ir, "is_aligned_realized_tensor", raising = False)
    assert sa.install_sdpa_align_without_guard() == 0
    assert table["sdpa_fwd"] is original


@needs_internals
def test_the_opt_out_takes_out_a_wrapper_this_process_installed(fresh, monkeypatch):
    original, _other, table = fresh
    assert sa.install_sdpa_align_without_guard() == 2
    monkeypatch.setenv(sa.SDPA_ALIGN_GUARD_ENV, "1")
    assert sa.install_sdpa_align_without_guard() == 0
    assert table["sdpa_fwd"] is table["sdpa_bwd"] is original
    assert not sa.installed() and sa.CACHE_TAG not in (torch.compiler.config.cache_key_tag or "")


_COMPILE_CHECK = """
import sys
sys.path.insert(0, sys.argv[1])
import torch
import core.training.diffusion_sdpa_align as sa

installer = sys.argv[2] == "installer"
if installer:
    print("wrapped", sa.install_sdpa_align_without_guard())


def attend(q, k, v):
    n = q.shape[-2]
    mask = torch.ones((n, n), dtype = torch.bool).tril()
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = mask)


f = torch.compile(attend, dynamic = True, fullgraph = True)
torch.manual_seed(0)
for n in (27, 24):
    q = torch.randn(1, 2, n, 16, requires_grad = True)
    out = f(q, q, q)
    ref = attend(q, q, q)
    torch.testing.assert_close(out, ref, rtol = 1e-4, atol = 1e-4)
    (g,) = torch.autograd.grad(out.sum(), q)
    (g_ref,) = torch.autograd.grad(ref.sum(), q)
    torch.testing.assert_close(g, g_ref, rtol = 1e-4, atol = 1e-4)
if installer:
    sa.uninstall_sdpa_align_without_guard()
print("COMPILED_SDPA_OK")
"""


def _run_compile_check(arm: str):
    import os
    import subprocess
    import sys

    backend = os.path.dirname(os.path.dirname(os.path.abspath(sa.__file__)))
    env = {k: v for k, v in os.environ.items() if k != sa.SDPA_ALIGN_GUARD_ENV}
    r = subprocess.run(
        [sys.executable, "-c", _COMPILE_CHECK, os.path.dirname(backend), arm],
        capture_output = True,
        text = True,
        env = env,
        timeout = 600,
    )
    return r, r.returncode == 0 and "COMPILED_SDPA_OK" in r.stdout


def test_compiled_sdpa_runs_with_the_installer_on_this_torch():
    """Real lowering on whatever torch runs the suite (torch 2.6 / 2.7 included), in a fresh interpreter: install,
    then a dynamic fullgraph masked SDPA forward and backward compiles and matches eager on CPU. On torch 2.6 the
    unconditional install raised LoweringException (SizeVarAllocator has no guard_or_false). The same graph is first
    compiled without the installer: a host whose CPU inductor cannot compile at all (no working C++ toolchain, as on
    a Windows runner without MSVC) skips with that reason instead of blaming the wrapper."""
    base, base_ok = _run_compile_check("baseline")
    if not base_ok:
        lines = [line.strip() for line in (base.stderr or base.stdout).splitlines() if line.strip()]
        errors = [line for line in lines if "Error" in line or "Exception" in line]
        reason = (errors or lines or [f"exit {base.returncode}"])[-1]
        pytest.skip(
            "CPU torch.compile of this SDPA graph fails without the installer on this host: "
            + reason[:300]
        )
    r, ok = _run_compile_check("installer")
    assert ok, (r.stdout[-2000:], r.stderr[-4000:])


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


@needs_internals
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "the SDPA layout constraint only applies on CUDA"
)
def test_a_multiple_of_8_text_length_does_not_recompile(monkeypatch):
    import torch._functorch.config as functorch_config
    import torch._inductor.config as inductor_config

    # Fresh compiles: a cached artifact carries the guards of the process that wrote it.
    monkeypatch.setattr(inductor_config, "fx_graph_cache", False)
    monkeypatch.setattr(functorch_config, "enable_autograd_cache", False)
    monkeypatch.setattr(sa, "_STATE", {})
    monkeypatch.delenv(sa.SDPA_ALIGN_GUARD_ENV, raising = False)
    lengths = (27, 29, 24, 32, 35)
    # torch's own guard: the first multiple of 8 compiles a second graph (the test can fail).
    assert _graphs_over_text_lengths(lengths) == 2
    try:
        assert sa.install_sdpa_align_without_guard() > 0
        assert _graphs_over_text_lengths(lengths) == 1
    finally:
        sa.uninstall_sdpa_align_without_guard()
    assert _graphs_over_text_lengths(lengths) == 2  # restored: torch's guard is back
