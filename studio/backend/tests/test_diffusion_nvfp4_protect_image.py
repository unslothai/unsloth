# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The NVFP4 per-step precision lever on the IMAGE backend."""

from __future__ import annotations

import ast
import os
import pathlib
import sys
import types

import pytest

from core.inference import diffusion_nvfp4_protect as pr

_INFERENCE_DIR = pathlib.Path(__file__).resolve().parents[1] / "core" / "inference"
_DIFFUSION = _INFERENCE_DIR / "diffusion.py"


def _generate_body() -> ast.FunctionDef:
    tree = ast.parse(_DIFFUSION.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "generate":
            return node
    raise AssertionError("diffusion.py has no generate()")


def _calls(node: ast.AST, name: str) -> list:
    found = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            target = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if target == name:
                found.append(sub)
    return found


def test_generate_arms_the_lever_with_the_effective_step_count():
    """Not ``steps``: an img2img at strength < 1 denoises a fraction of them."""
    generate = _generate_body()
    calls = _calls(generate, "protect_generation")
    assert len(calls) == 1, "generate() should arm the lever exactly once, per chunk"
    call = calls[0]
    assert len(call.args) >= 2
    assert isinstance(call.args[1], ast.Name) and call.args[1].id == "denoise_steps"


def test_the_effective_step_count_is_computed_outside_the_auto_cache_branch():
    """It used to be local to ``if state.cache_auto``."""
    generate = _generate_body()
    assignments = [
        node
        for node in ast.walk(generate)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "denoise_steps" for t in node.targets)
    ]
    assert len(assignments) == 1
    assert _calls(assignments[0], "effective_denoise_steps")
    for node in ast.walk(generate):
        if isinstance(node, ast.If):
            for sub in ast.walk(node):
                assert sub is not assignments[0], "denoise_steps is computed conditionally again"


def test_the_lever_wraps_the_chunk_render_itself():
    """The context is entered around ``pipe(**chunk_kwargs)``: one denoise loop per chunk."""
    generate = _generate_body()
    wrapped = []
    for node in ast.walk(generate):
        if not isinstance(node, ast.With):
            continue
        names = set()
        for item in node.items:
            expr = item.context_expr
            if isinstance(expr, ast.Name):
                names.add(expr.id)
            elif isinstance(expr, ast.Call):
                func = expr.func
                names.add(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ""))
        if "protect_ctx" in names:
            wrapped.append(names)
    assert wrapped, "the protect context is never entered"
    assert any("inference_mode" in names for names in wrapped)


def test_the_image_module_imports_the_lever_at_module_scope():
    source = _DIFFUSION.read_text(encoding = "utf-8")
    assert "from .diffusion_nvfp4_protect import protect_generation" in source


class _CachedScheduler:
    def __init__(self) -> None:
        self.calls = 0

    def step(self, *args, **kwargs):
        self.calls += 1
        return "stepped"


class _CapableLayer:
    """A stand-in for a converted flashinfer Linear: weak-referenceable, which is all the
    controller's registry asks of it."""


def _capable(ctl):
    """Register one, which is what makes a controller willing to arm. The caller must keep the
    returned object alive: the registry is weak."""
    layer = _CapableLayer()
    ctl.register_layer(layer)
    return layer


class _CachingPipe:
    """A denoise loop with an FBCache-shaped skip: the blocks are skipped, the LOOP is not."""

    def __init__(self, skip_steps) -> None:
        self.scheduler = _CachedScheduler()
        self.skip = set(skip_steps)

    def run(self, steps: int, forward) -> None:
        for i in range(steps):
            if i not in self.skip:
                forward()
            self.scheduler.step()


def test_a_cached_step_is_simply_not_protected():
    """The index still counts scheduler steps, so the lever protects the steps that are COMPUTED."""
    ctl = pr.NVFP4StepController("0,4,-1")
    layer = _capable(ctl)
    pipe = _CachingPipe(skip_steps = {4})  # step 4 is protected AND cached away
    seen: list = []
    with pr.protect_generation(pipe, 9, controller = ctl):
        pipe.run(9, lambda: seen.append((ctl.index, ctl.protected)))
    assert pipe.scheduler.calls == 9
    assert [index for index, _ in seen] == [0, 1, 2, 3, 5, 6, 7, 8]
    assert [index for index, protected in seen if protected] == [0, 8]
    assert ctl.protected_steps_seen == 3  # the controller still counts step 4 as protected


def test_the_cache_marker_and_the_lever_do_not_fight():
    """``GraphedForward`` bypasses itself while a step cache is engaged, and bypass wins."""
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_cuda_graph as cg

    module = types.SimpleNamespace(_unsloth_step_cache = "fbcache@0.12")
    handle = object.__new__(cg.GraphedForward)
    handle.module = module
    handle.enabled = True
    handle.bypassed = False
    handle.poisoned = False
    handle.orig = lambda *a, **kw: "eager"
    handle.stats = {"eager_calls": 0}
    assert handle(torch.zeros(1), return_dict = False) == "eager"
    assert handle.stats["eager_calls"] == 1


def test_each_chunk_restarts_the_schedule_at_step_zero():
    ctl = pr.NVFP4StepController("0,-1")
    layer = _capable(ctl)
    first: list = []
    second: list = []
    for sink in (first, second):
        pipe = _CachingPipe(skip_steps = ())
        with pr.protect_generation(pipe, 9, controller = ctl):
            pipe.run(9, lambda: sink.append((ctl.index, ctl.protected)))
    assert [i for i, p in first if p] == [0, 8]
    assert [i for i, p in second if p] == [0, 8]
    assert ctl.generations == 2
    assert ctl.protected is False


@pytest.mark.parametrize(
    "spec,steps,want",
    [
        ("auto", 9, (0, 8)),
        ("auto", 4, (0, 3)),
        ("0", 9, (0,)),
        ("0", 4, (0,)),
        ("all", 9, tuple(range(9))),
        ("all", 4, (0, 1, 2, 3)),
    ],
)
def test_the_image_schedules(spec, steps, want):
    assert pr.parse_protect_steps(spec, steps) == want


def test_suspend_protect_disarms_every_controller_it_reaches_and_restores_it():
    a, b = pr.NVFP4StepController("0"), pr.NVFP4StepController("auto")
    shared = pr.NVFP4StepController("all")
    modules = [
        types.SimpleNamespace(protect = a),
        types.SimpleNamespace(protect = b),
        types.SimpleNamespace(protect = shared),
        types.SimpleNamespace(protect = shared),  # deduped by identity
        types.SimpleNamespace(protect = None),
        types.SimpleNamespace(),
    ]
    with pr.suspend_protect(modules):
        assert not any(c.armed for c in (a, b, shared))
    assert all(c.armed for c in (a, b, shared))


def test_suspend_protect_restores_after_a_raise():
    ctl = pr.NVFP4StepController("0")
    with pytest.raises(RuntimeError):
        with pr.suspend_protect([types.SimpleNamespace(protect = ctl)]):
            raise RuntimeError("tuning blew up")
    assert ctl.armed is True


def test_prewarm_tunes_the_fp4_kernel_even_on_a_protected_step(monkeypatch):
    """A prewarm that fires at a protected step must still tune the FP4 kernel."""
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_nvfp4_linear as nl

    # The prewarm bails without flashinfer; the tuning loop itself is stubbed below.
    monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    layer = nl.nvfp4_linear_class()(
        64,
        32,
        wq = torch.zeros(32, 32, dtype = torch.uint8),
        w_sf = torch.ones(128, 4, dtype = torch.uint8),
        alpha = torch.tensor([0.25]),
        a_gsf = torch.tensor([1344.0]),
        bias = None,
    )
    ctl = pr.NVFP4StepController("0")
    layer.protect = ctl
    ctl.begin(2)
    assert ctl.protected is True

    branches: list = []
    original = nl._prewarm_shapes

    def _record(modules, shapes, *, logger, tuned_box):
        branches.append([m.protect.armed for m in modules])

    nl._prewarm_shapes = _record
    try:
        import torch.nn as nn
        tree = nn.Sequential(layer)
        nl.nvfp4_prewarm(tree, (1,))
    finally:
        nl._prewarm_shapes = original
    assert branches and branches[0] == [False]
    assert ctl.armed is True and ctl.protected is True


class _BlockHolding:
    """A denoiser-shaped module that HOLDS an NVFP4 layer without calling it."""

    def __new__(cls, inner):
        import torch.nn as nn
        class _Impl(nn.Module):
            def __init__(self, held):
                super().__init__()
                self.held = held

            def forward(
                self,
                hidden_states,
                return_dict = True,
            ):
                out = hidden_states * 2
                return (out,) if not return_dict else out

        return _Impl(inner)


def _cpu_nvfp4_tree(torch):
    import torch.nn as nn
    from core.inference import diffusion_nvfp4_linear as nl

    layer = nl.nvfp4_linear_class()(
        64,
        32,
        wq = torch.zeros(32, 32, dtype = torch.uint8),
        w_sf = torch.ones(128, 4, dtype = torch.uint8),
        alpha = torch.tensor([0.25]),
        a_gsf = torch.tensor([1344.0]),
        bias = None,
    )
    return nn.Sequential(layer), layer


def test_arming_the_lever_doubles_the_graph_cap_once(monkeypatch):
    """Every input shape becomes two calls, so the same shapes need twice the graphs."""
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_cuda_graph as cg

    monkeypatch.setenv(pr.PROTECT_STEPS_ENV, "0,-1")
    pr.reset_protect_controller()
    try:
        tree, _layer = _cpu_nvfp4_tree(torch)

        block = _BlockHolding(tree).eval()
        handle = cg.GraphedForward(block, max_graphs = 4).install().enable()
        before = handle.max_graphs
        block(hidden_states = torch.zeros(2, 64), return_dict = False)
        assert handle.protect_keyed is True
        assert handle.max_graphs == before * 2
        block(hidden_states = torch.zeros(2, 64), return_dict = False)
        assert handle.max_graphs == before * 2, "the cap must be raised once, not per call"
        cg.uninstall_all([handle])
    finally:
        monkeypatch.delenv(pr.PROTECT_STEPS_ENV, raising = False)
        pr.reset_protect_controller()


def test_an_unarmed_load_keeps_its_graph_cap(monkeypatch):
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_cuda_graph as cg

    monkeypatch.delenv(pr.PROTECT_STEPS_ENV, raising = False)
    pr.reset_protect_controller()
    tree, _layer = _cpu_nvfp4_tree(torch)
    block = _BlockHolding(tree).eval()
    handle = cg.GraphedForward(block, max_graphs = 4).install().enable()
    block(hidden_states = torch.zeros(2, 64), return_dict = False)
    assert handle.protect_keyed is False
    assert handle.max_graphs == 4
    cg.uninstall_all([handle])


def _cuda_or_skip():
    from core.inference import diffusion_nvfp4_ops as ops

    torch = pytest.importorskip("torch")
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        pytest.skip("CUDA devices are masked off for this process")
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if tuple(torch.cuda.get_device_capability(0)) not in ops.NVFP4_FLASHINFER_CAPS:
        pytest.skip("no flashinfer NVFP4 kernels on this card")
    pytest.importorskip("flashinfer")
    pytest.importorskip("torchao")
    return torch


def test_a_graphed_dit_captures_one_tuned_graph_per_branch_and_replays_both(monkeypatch):
    """Two captures, the FP4 tactic tuned before either, every replay bit-identical to the
    un-graphed forward of its own branch, and the second graph costing about what the first did."""
    torch = _cuda_or_skip()
    import torch.nn as nn
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
    from torchao.quantization import quantize_

    from core.inference import diffusion_cuda_graph as cg
    from core.inference import diffusion_nvfp4_linear as nl
    from core.inference import diffusion_nvfp4_ops as ops

    monkeypatch.setenv(pr.PROTECT_STEPS_ENV, "0,-1")
    ctl = pr.reset_protect_controller()
    nl.reset_tuned_shapes()
    prewarms: list = []
    real_prewarm = nl.nvfp4_prewarm

    def _counting_prewarm(module, shapes, **kw):
        tuned = real_prewarm(module, shapes, **kw)
        prewarms.append(tuned)
        return tuned

    monkeypatch.setattr(nl, "nvfp4_prewarm", _counting_prewarm)
    monkeypatch.setattr(cg, "_nvfp4_flashinfer_linears", cg._nvfp4_flashinfer_linears)

    try:
        with torch.cuda.device(0):
            torch.manual_seed(17)
            dit = (
                nn.Sequential(nn.Linear(3072, 3072), nn.SiLU(), nn.Linear(3072, 3072))
                .to("cuda", torch.bfloat16)
                .eval()
            )
            x = torch.randn(1024, 3072, device = "cuda", dtype = torch.bfloat16) * 0.05
            with torch.inference_mode():
                hidden = dit[1](dit[0](x))
            quantize_(dit, NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel = False))
            metadata = {
                "scheme": "nvfp4",
                "activation_scales_baked": True,
                "act_global_scales": {
                    "0": float(ops.global_scale(x)),
                    "2": float(ops.global_scale(hidden)),
                },
            }
            assert nl.convert_nvfp4_backend(dit, metadata, "flashinfer") == 2

            class DiT(nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner

                def forward(
                    self,
                    hidden_states,
                    return_dict = True,
                ):
                    out = self.inner(hidden_states)
                    return types.SimpleNamespace(sample = out) if return_dict else (out,)

            module = DiT(dit).eval()
            eager = {}
            ctl.begin(9)  # step 0: protected
            with torch.inference_mode():
                eager[True] = module(hidden_states = x, return_dict = False)[0].clone()
                for _ in range(8):
                    ctl.advance()
                    if not ctl.protected:
                        break
                assert ctl.protected is False
                eager[False] = module(hidden_states = x, return_dict = False)[0].clone()

            handle = cg.GraphedForward(module, max_graphs = 4).install().enable()

            def settled() -> tuple:
                """Reserved and allocated with the cache released, so the deltas are graph bytes."""
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                return torch.cuda.memory_reserved(), torch.cuda.memory_allocated()

            base_reserved, base_alloc = settled()
            torch.cuda.reset_peak_memory_stats()

            ctl.begin(9)  # protected again: first capture
            with torch.inference_mode():
                first = module(hidden_states = x, return_dict = False)[0].clone()
            after_one, _ = settled()
            one_graph_peak = torch.cuda.max_memory_allocated()

            for _ in range(8):
                ctl.advance()
                if not ctl.protected:
                    break
            with torch.inference_mode():
                second = module(hidden_states = x, return_dict = False)[0].clone()
            after_two, _ = settled()
            two_graph_peak = torch.cuda.max_memory_allocated()

            ctl.begin(9)
            with torch.inference_mode():
                first_again = module(hidden_states = x, return_dict = False)[0].clone()
            stats = dict(handle.stats)
            cap = handle.max_graphs
            cg.uninstall_all([handle])
    finally:
        monkeypatch.delenv(pr.PROTECT_STEPS_ENV, raising = False)
        pr.reset_protect_controller()

    assert stats["captures"] == 2, stats
    assert stats["replays"] == 3, stats
    assert stats["fallbacks"] == 0 and stats["cap_skips"] == 0, stats
    assert cap == 8, "arming the lever should have doubled the cap"
    assert prewarms and prewarms[0] > 0, prewarms
    assert torch.equal(first, eager[True])
    assert torch.equal(second, eager[False])
    assert torch.equal(first_again, first)
    assert not torch.equal(first, second)
    first_graph = after_one - base_reserved
    second_graph = after_two - after_one
    print(
        f"\n[graph memory] model={base_alloc / 2 ** 20:.1f} MiB "
        f"base_reserved={base_reserved / 2 ** 20:.1f} MiB "
        f"graph1=+{first_graph / 2 ** 20:.1f} MiB graph2=+{second_graph / 2 ** 20:.1f} MiB "
        f"peak_after_1={one_graph_peak / 2 ** 20:.1f} MiB "
        f"peak_after_2={two_graph_peak / 2 ** 20:.1f} MiB"
    )
    assert first_graph > 0, "the first capture reserved nothing; the measurement is not reading it"
    assert second_graph <= 1.5 * first_graph, (first_graph, second_graph)
    assert two_graph_peak < base_alloc + 2 * first_graph + (64 << 20)
