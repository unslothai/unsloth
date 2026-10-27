# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""In-place dW accumulation must not bypass hooks registered on autograd Nodes.

A prehook on the weight's AccumulateGrad node, or a post hook on the GroupedGemmBackward node, may
rewrite dW before it reaches param.grad. Both must see the same gradient as the returned-gradient
path (UNSLOTH_MOE_DW_ACCUMULATE=0). Runs on CPU: the Triton GEMMs are swapped for dense matmuls and
the stream queries are mocked, so the autograd wrapper and the selector are the real code."""

import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch")
pytest.importorskip("triton")


@pytest.fixture
def gg(monkeypatch):
    stubs = {
        "unsloth": ROOT / "unsloth",
        "unsloth.kernels": ROOT / "unsloth" / "kernels",
        "unsloth.kernels.moe": ROOT / "unsloth" / "kernels" / "moe",
    }
    for name, path in stubs.items():
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, pkg)
    sys.modules["unsloth"].DEVICE_TYPE = "cuda"
    device_type = types.ModuleType("unsloth.device_type")
    device_type.cuda_visible_capabilities = lambda: ()
    monkeypatch.setitem(sys.modules, device_type.__name__, device_type)
    utils = types.ModuleType("unsloth.kernels.utils")
    utils.torch_gpu_device = lambda device: torch.cuda.device(device)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    for name in list(sys.modules):
        if name.startswith("unsloth.kernels.moe."):
            monkeypatch.delitem(sys.modules, name)
    try:
        module = importlib.import_module("unsloth.kernels.moe.grouped_gemm.interface")
    except Exception as e:  # pragma: no cover - depends on the installed Triton
        pytest.skip(f"grouped_gemm interface is unimportable here: {type(e).__name__}: {e}")

    # Single expert, dense CPU arithmetic in place of the Triton launches.
    def fwd(X, W, **kw):
        return X @ W[0].T

    def dX(dY, W, **kw):
        return dY @ W[0]

    def dense_dW(
        X,
        dY,
        dW_dtype = None,
        **kw,
    ):
        return (dY.T @ X).unsqueeze(0).to(dW_dtype or X.dtype)

    def dW_accumulate(dW, **kw):
        dW.add_(dense_dW(**kw).to(dW.dtype))

    monkeypatch.setattr(module, "grouped_gemm_forward", fwd)
    monkeypatch.setattr(module, "grouped_gemm_dX", dX)
    monkeypatch.setattr(module, "grouped_gemm_dW", dense_dW)
    monkeypatch.setattr(module, "grouped_gemm_dW_accumulate", dW_accumulate)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "default_stream", lambda *a, **k: 0)
    yield module
    for name in list(sys.modules):
        if name.startswith("unsloth.kernels.moe."):
            sys.modules.pop(name, None)


def _cfg():
    return types.SimpleNamespace(
        use_tma_load_x = False, use_tma_load_w = False, use_tma_load_dy = False,
        use_tma_store = False, BLOCK_SIZE_M = 32, BLOCK_SIZE_N = 32, BLOCK_SIZE_K = 32,
        num_warps = 4, num_stages = 2,
    )  # fmt: skip


def _scale(g, k):
    return None if g is None else g * k


def _run(
    gg,
    monkeypatch,
    enabled,
    hook = None,
):
    monkeypatch.setenv("UNSLOTH_MOE_DW_ACCUMULATE", "1" if enabled else "0")
    gen = torch.Generator().manual_seed(0)
    X = torch.randn(4, 3, generator = gen, requires_grad = True)
    W = torch.nn.Parameter(torch.randn(1, 2, 3, generator = gen))
    W.grad = torch.randn(1, 2, 3, generator = gen)
    G = torch.randn(4, 2, generator = gen)
    cfg = _cfg()
    Y = gg.GroupedGemm.apply(
        X,
        W,
        torch.tensor([4]),
        1,
        None,
        False,
        False,
        None,
        False,
        cfg,
        cfg,
        cfg,
        False,
        False,
        False,
    )
    handle = None
    if hook == "acc_prehook":
        handle = Y.grad_fn.next_functions[1][0].register_prehook(lambda gs: (_scale(gs[0], 2),))
    elif hook == "node_posthook":
        handle = Y.grad_fn.register_hook(lambda gi, go: (gi[0], _scale(gi[1], 3)) + tuple(gi[2:]))
    elif hook == "tensor_hook":
        handle = W.register_hook(lambda g: g * 5)
    elif hook == "removed":
        Y.grad_fn.next_functions[1][0].register_prehook(lambda gs: (gs[0] * 2,)).remove()
        Y.grad_fn.register_hook(lambda gi, go: None).remove()
    before = dict(gg.DW_ACCUMULATE_STATS)
    Y.backward(G)
    after = gg.DW_ACCUMULATE_STATS
    path = "inplace" if after["inplace"] > before["inplace"] else "fallback"
    if handle is not None:
        handle.remove()
    return W.grad.clone(), X.grad.clone(), path


@pytest.mark.parametrize("hook", ["acc_prehook", "node_posthook", "tensor_hook"])
def test_hooked_weight_matches_returned_gradient_path(gg, monkeypatch, hook):
    ref_w, ref_x, ref_path = _run(gg, monkeypatch, enabled = False, hook = hook)
    out_w, out_x, path = _run(gg, monkeypatch, enabled = True, hook = hook)
    assert ref_path == "fallback"
    torch.testing.assert_close(out_w, ref_w, rtol = 0, atol = 0)
    torch.testing.assert_close(out_x, ref_x, rtol = 0, atol = 0)
    assert path == "fallback"


@pytest.mark.parametrize("hook", [None, "removed"])
def test_unhooked_weight_still_accumulates_in_place(gg, monkeypatch, hook):
    ref_w, ref_x, _ = _run(gg, monkeypatch, enabled = False, hook = hook)
    out_w, out_x, path = _run(gg, monkeypatch, enabled = True, hook = hook)
    assert path == "inplace"
    torch.testing.assert_close(out_w, ref_w)
    torch.testing.assert_close(out_x, ref_x, rtol = 0, atol = 0)
