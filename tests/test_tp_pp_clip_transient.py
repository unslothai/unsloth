# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TP / PP gradient clipping: bounded square-sum transient, same norm and scaled gradients (CPU)."""

import importlib.machinery
import importlib.util
import os
import sys
import types
import weakref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK = 4096
NUMEL = 1 << 16  # 16 chunks


def _load(name):
    """unsloth/models/<name>.py without importing the unsloth package (which needs a GPU)."""
    full = f"unsloth.models.{name}"
    if full in sys.modules:
        return sys.modules[full]
    pkg = types.ModuleType("unsloth")
    pkg.__path__ = [os.path.join(REPO, "unsloth")]
    models = types.ModuleType("unsloth.models")
    models.__path__ = [os.path.join(REPO, "unsloth", "models")]
    for package in (pkg, models):
        package.__spec__ = importlib.machinery.ModuleSpec(package.__name__, None, is_package = True)
        package.__spec__.submodule_search_locations = package.__path__
    sys.modules.setdefault("unsloth", pkg)
    sys.modules.setdefault("unsloth.models", models)
    spec = importlib.util.spec_from_file_location(
        full, os.path.join(REPO, "unsloth", "models", f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


tpn = _load("tp_native")
ppn = _load("pp_native")


class _LiveBytes(TorchDispatchMode):
    """Peak bytes held by tensors the ops under this mode allocate (outputs that alias no input)."""

    def __init__(self):
        super().__init__()
        self.live, self.current, self.peak = {}, 0, 0

    def _free(self, ptr):
        self.current -= self.live.pop(ptr, 0)

    def __torch_dispatch__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        out = func(*args, **(kwargs or {}))
        inputs = {
            t.untyped_storage().data_ptr()
            for t in tree_leaves((args, kwargs))
            if isinstance(t, torch.Tensor)
        }
        for t in tree_leaves(out):
            if not isinstance(t, torch.Tensor):
                continue
            ptr = t.untyped_storage().data_ptr()
            if ptr == 0 or ptr in inputs or ptr in self.live:
                continue
            self.live[ptr] = t.untyped_storage().nbytes()
            self.current += self.live[ptr]
            self.peak = max(self.peak, self.current)
            weakref.finalize(t, self._free, ptr)
        return out


@pytest.fixture(autouse = True)
def _small_chunk(monkeypatch):
    monkeypatch.setattr(tpn, "_SQUARE_SUM_CHUNK", CHUNK, raising = False)


def _params(dtype, seed = 0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for numel in (NUMEL, NUMEL + 123, 37):  # chunked, chunked with a tail, single reduction
        p = torch.nn.Parameter(torch.zeros(numel, dtype = dtype))
        p.grad = (torch.randn(numel, generator = g) * 3).to(dtype)
        out.append(p)
    return out


def _pp_clip(params, max_norm):
    stage = types.SimpleNamespace(
        composed = False, num_stages = 1, device = torch.device("cpu"), _check_unscaled = lambda: None
    )
    stage._grad_copies = types.MethodType(ppn.PipelineParallel._grad_copies, stage)
    return ppn.PipelineParallel.clip_grad_norm_(stage, max_norm, params)


def _clip(kind, params, max_norm):
    return tpn.tp_clip_grad_norm_(params, max_norm) if kind == "tp" else _pp_clip(params, max_norm)


@pytest.mark.parametrize("kind", ["tp", "pp"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_clip_transient_is_bounded_by_the_chunk(kind, dtype):
    """Before: the whole gradient cast and squared (TP 4x, PP 8x a 16-bit gradient's bytes). After: slices."""
    params = _params(dtype)
    biggest = max(p.grad.numel() * p.grad.element_size() for p in params)
    with _LiveBytes() as mode:
        _clip(kind, params, 1e-3)
    acc = 4 if kind == "tp" else 8  # fp32 / fp64 accumulation
    # A slice cast plus its square, and a few scalars; far below one gradient.
    assert mode.peak <= 2 * CHUNK * acc + 4096, (kind, dtype, mode.peak, biggest)
    assert mode.peak < biggest


@pytest.mark.parametrize("kind", ["tp", "pp"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_clip_matches_the_whole_tensor_reduction(kind, dtype):
    params = _params(dtype)
    acc = torch.float32 if kind == "tp" else torch.float64
    expected = sum(p.grad.to(acc).square().sum() for p in params).sqrt()
    before = [p.grad.clone() for p in params]
    total = _clip(kind, params, 1e-3)
    # Summation order changes only inside the accumulation dtype's rounding (PP returns float32).
    torch.testing.assert_close(total.double(), expected.double(), rtol = 1e-6, atol = 0)
    coef = (1e-3 / (expected.double() + 1e-6)).item()
    for p, g in zip(params, before):
        torch.testing.assert_close(p.grad.double(), g.double() * coef, rtol = 1e-2, atol = 1e-6)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_tp_scaling_is_bitwise_the_whole_tensor_formula(dtype):
    """Scaling is elementwise: slicing the fp32 copy changes no element."""
    params = _params(dtype)
    before = [p.grad.clone() for p in params]
    total = torch.tensor(1234.5)
    tpn._scale_grads_(params, total, 1.0, 1e-6)
    coef = torch.clamp(1.0 / (total + 1e-6), max = 1.0)
    for p, g in zip(params, before):
        if dtype in (torch.float16, torch.bfloat16):
            ref = g.float().mul_(coef).to(dtype)
        else:
            ref = g * coef
        assert torch.equal(p.grad, ref), dtype


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_small_tensors_keep_the_single_reduction_bitwise(dtype):
    """At most one chunk: the exact same call as before."""
    g = torch.randn(CHUNK, dtype = torch.float32, generator = torch.Generator().manual_seed(1)).to(
        dtype
    )
    for acc in (torch.float32, torch.float64):
        assert torch.equal(tpn._square_sum(g, acc), g.to(acc).square().sum())


def test_square_sum_leaves_the_gradient_untouched():
    """An fp32 slice is the gradient itself: it is squared out of place."""
    g = torch.randn(64, 1100)
    keep = g.clone()
    tpn._square_sum(g, torch.float32)
    strided = g[:, :1000]  # non-contiguous, above one chunk
    torch.testing.assert_close(
        tpn._square_sum(strided, torch.float64), strided.double().square().sum(), rtol = 1e-12, atol = 0
    )
    assert torch.equal(g, keep)


class _MaxAlloc(TorchDispatchMode):
    """Largest fresh tensor (elements) any op under this mode allocates, however briefly it lives."""

    def __init__(self):
        super().__init__()
        self.numel = 0

    def __torch_dispatch__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        out = func(*args, **(kwargs or {}))
        inputs = {
            t.untyped_storage().data_ptr()
            for t in tree_leaves((args, kwargs))
            if isinstance(t, torch.Tensor) and t.layout == torch.strided
        }
        for t in tree_leaves(out):
            if isinstance(t, torch.Tensor) and t.layout == torch.strided:
                if t.untyped_storage().data_ptr() not in inputs:
                    self.numel = max(self.numel, t.untyped_storage().nbytes() // t.element_size())
        return out


def _strided_params(dtype, seed = 0):
    """Transposed gradients (the layout autograd keeps for a transposed parameter), above one chunk."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for rows, cols in ((256, 384), (300, 129), (3, 5)):
        p = torch.nn.Parameter(torch.zeros(cols, rows, dtype = dtype))
        p.grad = (torch.randn(rows, cols, generator = g) * 3).to(dtype).t()
        assert not p.grad.is_contiguous()
        out.append(p)
    return out


@pytest.mark.parametrize("kind", ["tp", "pp"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_strided_gradient_is_never_copied_whole(kind, dtype):
    """Before: reshape(-1) copied a strided gradient whole, and 16-bit scaling cast it whole to fp32."""
    params = _strided_params(dtype)
    before = [p.grad.clone() for p in params]
    acc = torch.float32 if kind == "tp" else torch.float64
    expected = sum(p.grad.to(acc).square().sum() for p in params).sqrt()
    with _MaxAlloc() as mode:
        total = _clip(kind, params, 1e-3)
    assert mode.numel <= CHUNK, (kind, dtype, mode.numel)
    torch.testing.assert_close(total.double(), expected.double(), rtol = 1e-6, atol = 0)
    coef = torch.clamp(1e-3 / (total.float() + 1e-6), max = 1.0)
    for p, g in zip(params, before):
        if kind == "tp" and dtype in (torch.float16, torch.bfloat16):
            assert torch.equal(
                p.grad, g.float().mul_(coef).to(dtype)
            ), dtype  # bitwise, slice by slice
        else:
            torch.testing.assert_close(
                p.grad.double(), g.double() * coef.item(), rtol = 1e-2, atol = 1e-6
            )


def test_strided_scaling_is_bitwise_and_bounded():
    for dtype in (torch.bfloat16, torch.float16):
        params = _strided_params(dtype)
        before = [p.grad.clone() for p in params]
        total = torch.tensor(1234.5)
        with _MaxAlloc() as mode:
            tpn._scale_grads_(params, total, 1.0, 1e-6)
        assert mode.numel <= CHUNK, (dtype, mode.numel)
        coef = torch.clamp(1.0 / (total + 1e-6), max = 1.0)
        for p, g in zip(params, before):
            assert torch.equal(p.grad, g.float().mul_(coef).to(dtype)), dtype


def test_only_non_overlapping_layouts_are_sliced_for_scaling():
    """Separate writes to slices of an aliased gradient would scale one element twice."""
    assert tpn._disjoint_strides(torch.empty(64, 96).t())
    assert tpn._disjoint_strides(torch.empty(8, 64, 96)[:, ::2].transpose(0, 2))
    assert not tpn._disjoint_strides(torch.empty(1, 96).expand(8, -1))
    assert not tpn._disjoint_strides(torch.empty(100).as_strided((10, 10), (5, 1)))


@pytest.mark.parametrize("kind", ["tp", "pp"])
def test_sparse_gradient_above_the_chunk(kind):
    """Before: reshape is not implemented for sparse tensors (a COO embedding gradient above the cutoff)."""
    rows, dim = NUMEL // 64, 64
    p = torch.nn.Parameter(torch.zeros(rows, dim))
    idx = torch.tensor([[3, 17, 900]])
    p.grad = torch.sparse_coo_tensor(idx, torch.randn(3, dim) * 3, (rows, dim)).coalesce()
    dense = p.grad.to_dense()
    acc = torch.float32 if kind == "tp" else torch.float64
    expected = dense.to(acc).square().sum().sqrt()
    total = _clip(kind, [p], 1e-3)
    torch.testing.assert_close(total.double(), expected.double(), rtol = 1e-6, atol = 0)
    coef = (1e-3 / (expected.double() + 1e-6)).item()
    torch.testing.assert_close(
        p.grad.to_dense().double(), dense.double() * coef, rtol = 1e-5, atol = 1e-7
    )


@pytest.mark.parametrize("numel", [37, NUMEL])
def test_norm_does_not_track_a_grad_that_requires_grad(numel):
    """Before: TP squared a requires_grad gradient under autograd, saving every slice for backward."""
    p = torch.nn.Parameter(torch.zeros(numel))
    p.grad = torch.randn(numel).requires_grad_()
    sq = tpn._square_sum(p.grad)
    assert sq.grad_fn is None and not sq.requires_grad
    assert tpn._square_sum(p.grad.detach()).item() == sq.item()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_single_chunk_holds_one_buffer(dtype):
    """Before: a fresh cast and its square were both live at the cutoff."""
    g = torch.randn(CHUNK).to(dtype)
    with _LiveBytes() as mode:
        out = tpn._square_sum(g, torch.float32)
    assert mode.peak <= CHUNK * 4 + 4096, mode.peak
    assert torch.equal(out, g.float().square().sum())


def _multi_lora():
    pkg = types.ModuleType("mlk_clip")
    pkg.__path__ = []
    sys.modules["mlk_clip"] = pkg
    utils = types.ModuleType("mlk_clip.utils")
    utils._base_quant_state = lambda layer, W: None
    utils.fast_dequantize = lambda W, q = None, dtype = None, **k: W
    utils.matmul_lora = lambda X, W, Wq, A, B, s, out = None: X @ W.t()
    utils.torch_amp_custom_fwd = utils.torch_amp_custom_bwd = lambda f: f
    sys.modules["mlk_clip.utils"] = utils
    spec = importlib.util.spec_from_file_location(
        "mlk_clip.multi_lora", os.path.join(REPO, "unsloth", "kernels", "multi_lora.py")
    )
    ml = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ml
    spec.loader.exec_module(ml)
    return ml


@pytest.mark.parametrize("contiguous", [True, False])
def test_multi_lora_expert_factor_is_squared_in_slices(monkeypatch, contiguous):
    """Before: per-adapter clipping cast and squared a whole expert factor gradient."""
    ml = _multi_lora()
    monkeypatch.setattr(ml, "_SQUARE_SUM_CHUNK", CHUNK)
    g = (torch.randn(16, 64, 96) * 3).to(torch.bfloat16)
    if not contiguous:
        g = g.transpose(1, 2)
    with _MaxAlloc() as mode:
        sq = ml._square_sum(g)
    assert mode.numel <= CHUNK, mode.numel
    torch.testing.assert_close(sq, g.float().pow(2).sum(), rtol = 1e-6, atol = 0)
    small = g[0, :8]
    assert torch.equal(ml._square_sum(small), small.float().pow(2).sum())
