# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TiledMLP under 2-rank gloo DDP must produce the untiled all-reduced grads in every DDP mode."""

import importlib.util
import os
import pathlib
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

if not dist.is_available() or not dist.is_gloo_available():
    pytest.skip("needs gloo", allow_module_level = True)


def _tiled_mlp():
    # Spawned ranks skip conftest, so repeat its accelerator-less import of device_type.
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    if "unsloth_zoo.device_type" not in sys.modules and not torch.cuda.is_available():
        is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: True
        try:
            import unsloth_zoo.device_type  # noqa: F401
        finally:
            torch.cuda.is_available = is_available
    from unsloth_zoo.tiled_mlp import TiledMLP

    return TiledMLP


_PATCH = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_tiled_mlp_ddp.py"
_MODES = {
    "default": {},
    "find_unused_parameters": {"find_unused_parameters": True},
    "static_graph": {"static_graph": True},
    "gradient_as_bucket_view": {"gradient_as_bucket_view": True},
}


class _MLP(torch.nn.Module):
    def __init__(self, tiled):
        super().__init__()
        torch.manual_seed(0)
        self.up = torch.nn.Linear(8, 32)
        self.down = torch.nn.Linear(32, 8)
        self.frozen = torch.nn.Linear(8, 8).requires_grad_(False)
        self.tiled = tiled

    def mlp(self, x):
        return self.down(torch.nn.functional.gelu(self.up(x))) + self.frozen(x)

    def forward(self, x):
        if self.tiled:
            return _tiled_mlp().apply(self.mlp, self, x, False, 4, None)
        return self.mlp(x)


def _inputs(rank, step):
    g = torch.Generator().manual_seed(100 + rank)
    return (torch.randn(2, 16, 8, generator = g) * (step + 1)).requires_grad_(True)


def _grads(model, x):
    return [p.grad.tolist() for p in model.parameters() if p.requires_grad] + [x.grad.tolist()]


def _load_patch():
    spec = importlib.util.spec_from_file_location("_tiled_mlp_ddp_under_test", _PATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _worker(rank, init_file, queue):
    try:
        # Spawned workers lack conftest preloading and zoo GPU init refuses CPU-only runners.
        if not torch.cuda.is_available():
            os.environ["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
        _load_patch().patch_tiled_mlp_for_ddp()
        dist.init_process_group("gloo", init_method = f"file://{init_file}", rank = rank, world_size = 2)
        out = {}
        for mode, ddp_kwargs in _MODES.items():
            try:
                model = torch.nn.parallel.DistributedDataParallel(_MLP(tiled = True), **ddp_kwargs)
                out[mode] = []
                for step in range(3):
                    model.zero_grad(set_to_none = True)
                    x = _inputs(rank, step)
                    model(x).pow(2).sum().backward()
                    out[mode].append(_grads(model.module, x))
            except Exception as e:
                out[mode] = repr(e)
        queue.put((rank, out))
    except Exception as e:
        queue.put((rank, {mode: repr(e) for mode in _MODES}))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _reference(rank, step):
    model = _MLP(tiled = False)
    xs = [_inputs(r, step) for r in range(2)]
    for x in xs:
        model(x).pow(2).sum().backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad /= 2
    return _grads(model, xs[rank])


@pytest.fixture(scope = "module")
def ddp_grads(tmp_path_factory):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    init_file = tempfile.mktemp(dir = tmp_path_factory.mktemp("ddp"))
    procs = [ctx.Process(target = _worker, args = (r, init_file, queue)) for r in range(2)]
    for p in procs:
        p.start()
    results = dict(queue.get(timeout = 300) for _ in procs)
    for p in procs:
        p.join(30)
        if p.is_alive():
            p.kill()
    return results


@pytest.mark.parametrize("mode", list(_MODES))
def test_tiled_mlp_ddp_grads_match_untiled(mode, ddp_grads):
    for rank, by_mode in ddp_grads.items():
        grads = by_mode[mode]
        assert not isinstance(grads, str), grads
        for step, got in enumerate(grads):
            for a, b in zip(got, _reference(rank, step)):
                torch.testing.assert_close(torch.tensor(a), torch.tensor(b), rtol = 1e-4, atol = 1e-4)


def test_single_process_matches_stock_tiled_mlp():
    stock = _MLP(tiled = True)
    x = _inputs(0, 0)
    stock(x).pow(2).sum().backward()
    expected = _grads(stock, x)

    TiledMLP = _tiled_mlp()
    original_apply = TiledMLP.__dict__.get("apply")
    _load_patch().patch_tiled_mlp_for_ddp()
    try:
        patched = _MLP(tiled = True)
        x = _inputs(0, 0)
        patched(x).pow(2).sum().backward()
        got = _grads(patched, x)
    finally:
        if original_apply is not None:
            TiledMLP.apply = original_apply
        elif "apply" in TiledMLP.__dict__:
            del TiledMLP.apply
    for a, b in zip(got, expected):
        torch.testing.assert_close(torch.tensor(a), torch.tensor(b), rtol = 1e-5, atol = 1e-5)
