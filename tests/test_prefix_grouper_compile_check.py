# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""PrefixGrouper's one-time strict compile check.

unsloth_zoo sets dynamo's suppress_errors, so a flex kernel that fails to compile silently runs
eager flex_attention, which materialises the full T x T scores. The first compiled call per setup runs
with suppress_errors off, plus a forward + backward on copies of its inputs; a failure turns
PrefixGrouper off for the process.
"""

import importlib.util
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
flex = pytest.importorskip("torch.nn.attention.flex_attention")

UTILS = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "utils"
PKG = "_pg_check_pkg_under_test"


def _load():
    """prefix_grouper.py + its kernel under a throwaway package (no full unsloth import)."""
    pkg = types.ModuleType(PKG)
    pkg.__path__ = [str(UTILS)]
    names = (PKG, f"{PKG}.prefix_grouper_kernel", f"{PKG}.prefix_grouper")
    sys.modules[PKG] = pkg
    try:
        name = f"{PKG}.prefix_grouper"
        spec = importlib.util.spec_from_file_location(name, UTILS / "prefix_grouper.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        kernel = sys.modules[f"{PKG}.prefix_grouper_kernel"]
    finally:
        for n in names:
            sys.modules.pop(n, None)
    return module, kernel


def _cpu_flex(q, k, v, **kwargs):
    """Eager flex forward; CPU flex has no backward, so grads flow through a zero-weighted sum."""
    if q.requires_grad:
        # The real kernel saves the block mask and the mask_mod's label tensors for backward.
        bm = kwargs["block_mask"]
        cells = [c.cell_contents for c in (bm.mask_mod.__closure__ or ())]
        saved = [x for x in (*bm.as_tuple(), *cells) if isinstance(x, torch.Tensor)]
        if any(x.is_inference() for x in saved):
            raise RuntimeError("Inference tensors cannot be saved for backward.")
    out = flex.flex_attention(q.detach(), k.detach(), v.detach(), **kwargs)
    if q.requires_grad:
        out = out + 0 * (q.sum() + k.sum() + v.sum())
    return out


@pytest.fixture
def mods(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "1")
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "1")
    monkeypatch.delenv("UNSLOTH_GRPO_PREFIX_GROUPER_COMPILE_CHECK", raising = False)
    config = torch._dynamo.config
    for name in ("recompile_limit", "accumulated_recompile_limit", "suppress_errors"):
        if hasattr(config, name):
            monkeypatch.setattr(config, name, getattr(config, name))
    config.suppress_errors = True  # what unsloth_zoo sets
    pg, kernel = _load()
    calls = {"flex": 0, "strict": [], "mask": 0, "shapes": [], "grad": []}

    def eager_flex(
        *args,
        kernel_options = None,
        **kwargs,
    ):
        calls["flex"] += 1
        calls["strict"].append(torch._dynamo.config.suppress_errors)
        calls["shapes"].append(tuple(tuple(x.shape) for x in args[:3]))
        calls["grad"].append(args[0].requires_grad)
        return _cpu_flex(*args, **kwargs)

    def eager_mask(*args, **kwargs):
        calls["mask"] += 1
        return flex.create_block_mask(*args, **kwargs)

    monkeypatch.setattr(kernel, "_flex_attention_compiled", eager_flex)
    monkeypatch.setattr(kernel, "_create_block_mask_compiled", eager_mask)
    return pg, kernel, calls


def _qkv(
    T,
    H = 4,
    KVH = 2,
    D = 16,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(1, T, H, D, generator = g),
        torch.randn(1, T, KVH, D, generator = g),
        torch.randn(1, T, KVH, D, generator = g),
    )


def _seg(kernel, spec = ((20, [7, 13, 5]), (11, [9, 4]))):
    seg, _ = kernel.build_seg_info_multigroup([(p, list(r)) for p, r in spec], torch.device("cpu"))
    return seg


def test_success_runs_the_check_once_and_keeps_numerics(mods, capsys):
    pg, kernel, calls = mods
    seg = _seg(kernel)
    Q, K, V = _qkv(seg.T)
    rng = torch.get_rng_state()
    out = kernel.flex_shared_prefix_attention(Q, K, V, seg)
    # First call: the real call, then a grad forward on copies, both under strict mode and with the
    # real shapes (so the compiled graphs are the ones later calls reuse).
    assert calls["strict"] == [False, False]
    assert calls["grad"] == [False, True]
    assert len(set(calls["shapes"])) == 1
    assert torch.equal(torch.get_rng_state(), rng)  # the check draws no random numbers
    assert torch._dynamo.config.suppress_errors is True
    kernel.clear_block_mask_cache()
    again = kernel.flex_shared_prefix_attention(Q, K, V, seg)
    assert calls["flex"] == 3 and calls["strict"][-1] is True  # no re-check
    assert torch.equal(out, again)
    assert pg.prefix_grouper_enabled() is True
    assert kernel.compile_check_failed() is None
    assert capsys.readouterr().out == ""

    # Same output as with the check switched off.
    kernel._COMPILE_CHECKED.clear()
    kernel._COMPILE_CHECK_ON = False
    kernel.clear_block_mask_cache()
    assert torch.equal(kernel.flex_shared_prefix_attention(Q, K, V, seg), out)


def test_first_call_under_inference_mode(mods):
    """The trainer's first PrefixGrouper call is the no-grad logprob pass under inference_mode."""
    pg, kernel, calls = mods
    seg = _seg(kernel)
    Q, K, V = _qkv(seg.T)
    with torch.inference_mode():
        out = kernel.flex_shared_prefix_attention(Q, K, V, seg)
    assert kernel.compile_check_failed() is None and calls["flex"] == 2
    q, k, v = (x.clone().requires_grad_(True) for x in (Q, K, V))
    kernel.flex_shared_prefix_attention(q, k, v, seg).sum().backward()
    assert torch.equal(kernel.flex_shared_prefix_attention(Q, K, V, seg), out)


def test_new_head_layout_is_checked_once(mods):
    pg, kernel, calls = mods
    seg = _seg(kernel)
    kernel.flex_shared_prefix_attention(*_qkv(seg.T), seg)
    kernel.flex_shared_prefix_attention(*_qkv(seg.T, H = 8, KVH = 8, D = 32), seg)
    kernel.flex_shared_prefix_attention(*_qkv(seg.T, H = 8, KVH = 8, D = 32), seg)
    assert calls["flex"] == 2 + 2 + 1
    assert all(kernel._COMPILE_CHECKED.values()) and len(kernel._COMPILE_CHECKED) == 2


def test_compile_failure_turns_prefix_grouper_off_once(mods, monkeypatch, capsys):
    pg, kernel, calls = mods

    def broken(*args, **kwargs):
        calls["flex"] += 1
        raise RuntimeError(
            "Triton Error: out of resource: shared memory, Required: 104448, Hardware limit: 101376"
        )

    monkeypatch.setattr(kernel, "_flex_attention_compiled", broken)
    seg = _seg(kernel)
    Q, K, V = _qkv(seg.T)
    with pytest.raises(kernel.PrefixGrouperCompileError):
        kernel.flex_shared_prefix_attention(Q, K, V, seg)
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and "PrefixGrouper is off" in out and "104448" in out
    assert pg.prefix_grouper_enabled() is False
    assert torch._dynamo.config.suppress_errors is True  # patch restored after the failure
    # One retry with one-stage tiles, then later calls fail fast without a re-check or new log.
    with pytest.raises(kernel.PrefixGrouperCompileError):
        kernel.flex_shared_prefix_attention(Q, K, V, seg)
    assert calls["flex"] == 2
    assert capsys.readouterr().out == ""


def test_shared_memory_overflow_retries_with_one_stage(mods, monkeypatch, capsys):
    """torch 2.7 / 2.8 on 101376 B smem cards: 3-stage tiles overflow, num_stages=1 fits."""
    pg, kernel, calls = mods
    stages = []

    def three_stage_overflows(
        *args,
        kernel_options = None,
        **kwargs,
    ):
        stages.append(kernel_options.get("num_stages"))
        if kernel_options.get("num_stages") != 1:
            raise RuntimeError(
                "No valid triton configs. OutOfResources: out of resource: shared memory"
            )
        return _cpu_flex(*args, **kwargs)

    monkeypatch.setattr(kernel, "_flex_attention_compiled", three_stage_overflows)
    seg = _seg(kernel)
    Q, K, V = _qkv(seg.T)
    out = kernel.flex_shared_prefix_attention(Q, K, V, seg)
    assert stages == [None, 1, 1]
    assert pg.prefix_grouper_enabled() is True and capsys.readouterr().out == ""
    assert torch.equal(kernel.flex_shared_prefix_attention(Q, K, V, seg), out)
    assert stages[-1] == 1 and len(stages) == 4


def test_single_block_mask_is_built_eagerly(mods):
    """torch 2.7 Inductor fails when the first compiled mask is a single 128 block."""
    pg, kernel, calls = mods
    seg = _seg(kernel)
    assert seg.T_pad == 128
    kernel.get_block_mask(seg, torch.device("cpu"))
    assert calls["mask"] == 0
    kernel.get_block_mask(_seg(kernel, ((100, [30, 40]),)), torch.device("cpu"))
    assert calls["mask"] == 1


def test_backward_compile_failure_is_caught(mods, monkeypatch):
    pg, kernel, calls = mods

    class _Boom(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, g):
            raise RuntimeError("backward kernel failed to compile")

    def fwd_ok_bwd_broken(
        *args,
        kernel_options = None,
        **kwargs,
    ):
        return _Boom.apply(_cpu_flex(*args, **kwargs))

    monkeypatch.setattr(kernel, "_flex_attention_compiled", fwd_ok_bwd_broken)
    seg = _seg(kernel)
    with pytest.raises(kernel.PrefixGrouperCompileError):
        kernel.flex_shared_prefix_attention(*_qkv(seg.T), seg)
    assert pg.prefix_grouper_enabled() is False


def test_real_dynamo_failure_is_silent_without_the_check(mods, monkeypatch):
    """A backend failure is swallowed by suppress_errors on a plain call but caught by the check."""
    pg, kernel, calls = mods

    def failing_backend(gm, example_inputs):
        raise RuntimeError("simulated inductor codegen failure")

    compiled = torch.compile(flex.flex_attention, backend = failing_backend, dynamic = True)

    def via_dynamo(
        *args,
        kernel_options = None,
        **kwargs,
    ):
        return compiled(*args, **kwargs)

    torch._dynamo.reset()
    monkeypatch.setattr(kernel, "_flex_attention_compiled", via_dynamo)
    seg = _seg(kernel)
    Q, K, V = _qkv(seg.T)
    # Base behaviour: the failure is silent (eager fallback) when the check is off.
    kernel._COMPILE_CHECK_ON = False
    kernel.flex_shared_prefix_attention(Q, K, V, seg)
    assert kernel.compile_check_failed() is None
    kernel._COMPILE_CHECK_ON = True
    torch._dynamo.reset()
    with pytest.raises(kernel.PrefixGrouperCompileError):
        kernel.flex_shared_prefix_attention(Q, K, V, seg)
    assert "simulated inductor codegen failure" in kernel.compile_check_failed()
    assert pg.prefix_grouper_enabled() is False
    assert torch._dynamo.config.suppress_errors is True
    torch._dynamo.reset()


def test_oom_in_the_check_is_not_cached(mods, monkeypatch):
    pg, kernel, calls = mods
    state = {"n": 0}

    def oom_once(
        *args,
        kernel_options = None,
        **kwargs,
    ):
        state["n"] += 1
        if state["n"] == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return _cpu_flex(*args, **kwargs)

    monkeypatch.setattr(kernel, "_flex_attention_compiled", oom_once)
    seg = _seg(kernel)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        kernel.flex_shared_prefix_attention(*_qkv(seg.T), seg)
    assert pg.prefix_grouper_enabled() is True
    kernel.flex_shared_prefix_attention(*_qkv(seg.T), seg)
    assert kernel.compile_check_failed() is None


def test_eager_path_skips_the_check(mods):
    pg, kernel, calls = mods
    seg = _seg(kernel)
    kernel.flex_shared_prefix_attention(*_qkv(seg.T), seg, compiled = False)
    assert calls["flex"] == 0 and not kernel._COMPILE_CHECKED


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER_COMPILE_CHECK", "0")
    pg, kernel = _load()
    assert kernel._COMPILE_CHECK_ON is False


def test_oom_while_compiling_is_not_cached(mods, monkeypatch):
    """dynamo wraps an OOM raised while compiling in BackendCompilerFailed: still transient."""
    pg, kernel, calls = mods
    state = {"n": 0}

    def oom_backend(gm, example_inputs):
        state["n"] += 1
        if state["n"] == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return gm.forward

    compiled = torch.compile(flex.flex_attention, backend = oom_backend)

    def flex_oom_once(
        *args,
        kernel_options = None,
        **kwargs,
    ):
        return compiled(*args, **kwargs)

    monkeypatch.setattr(kernel, "_flex_attention_compiled", flex_oom_once)
    seg = _seg(kernel)
    with torch.no_grad(), pytest.raises(torch.cuda.OutOfMemoryError):
        kernel.flex_shared_prefix_attention(*_qkv(seg.T), seg)
    assert state["n"] == 1
    assert pg.prefix_grouper_enabled() is True
    assert kernel.compile_check_failed() is None and not kernel._COMPILE_CHECKED
    assert torch._dynamo.config.suppress_errors is True
