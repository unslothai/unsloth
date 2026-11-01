# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Tiled MLP under 2-rank gloo DDP matches untiled grads with no DDP patch; grad hooks fire once per backward."""

import os
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

if not dist.is_available() or not dist.is_gloo_available():
    pytest.skip("needs gloo", allow_module_level = True)

_MODES = {
    "default": {},
    "find_unused_parameters": {"find_unused_parameters": True},
    "static_graph": {"static_graph": True},
    "gradient_as_bucket_view": {"gradient_as_bucket_view": True},
}


def _registry():
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    if "unsloth_zoo.device_type" not in sys.modules and not torch.cuda.is_available():
        is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: True
        try:
            import unsloth_zoo.device_type  # noqa: F401
        finally:
            torch.cuda.is_available = is_available
    tiling_registry = pytest.importorskip(
        "unsloth_zoo.tiling_registry"
    )  # an older zoo has no registry
    return tiling_registry


class FooMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.up = torch.nn.Linear(8, 32)
        self.down = torch.nn.Linear(32, 8)
        self.frozen = torch.nn.Linear(8, 8).requires_grad_(False)

    def forward(self, x):
        return self.down(torch.nn.functional.gelu(self.up(x))) + self.frozen(x)


def _model(tiled):
    m = FooMLP()
    if tiled:
        _registry().install_module(m, "rows:8")
    return m


def _inputs(rank, step, frozen):
    g = torch.Generator().manual_seed(100 + rank)
    x = torch.randn(2, 16, 8, generator = g) * (step + 1)
    return x if frozen else x.requires_grad_(True)


def _grads(model, x):
    out = [p.grad.tolist() for p in model.parameters() if p.requires_grad]
    return out + ([x.grad.tolist()] if x.requires_grad else [])


def _worker(rank, init_file, queue):
    try:
        if not torch.cuda.is_available():
            os.environ["UNSLOTH_ZOO_DISABLE_GPU_INIT"] = "1"
        dist.init_process_group("gloo", init_method = f"file://{init_file}", rank = rank, world_size = 2)
        out = {}
        for mode, ddp_kwargs in _MODES.items():
            for frozen in (False, True):
                key = f"{mode}/{frozen}"
                try:
                    inner = _model(tiled = True)
                    fired = {}
                    for name, p in inner.named_parameters():
                        if p.requires_grad:
                            p.register_hook(
                                lambda g, name = name: (
                                    fired.__setitem__(name, fired.get(name, 0) + 1) or g
                                )
                            )
                    model = torch.nn.parallel.DistributedDataParallel(inner, **ddp_kwargs)
                    out[key] = []
                    for step in range(3):
                        model.zero_grad(set_to_none = True)
                        fired.clear()
                        x = _inputs(rank, step, frozen)
                        model(x).pow(2).sum().backward()
                        assert all(v == 1 for v in fired.values()) and len(fired) == 4, fired
                        out[key].append(_grads(model.module, x))
                except Exception as e:
                    out[key] = repr(e)
        queue.put((rank, out))
    except Exception as e:
        queue.put((rank, {"error": repr(e)}))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _reference(rank, step, frozen):
    model = _model(tiled = False)
    xs = [_inputs(r, step, frozen) for r in range(2)]
    for x in xs:
        model(x).pow(2).sum().backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad /= 2
    return _grads(model, xs[rank])


@pytest.fixture(scope = "module")
def ddp_grads(tmp_path_factory):
    _registry()  # skips here, before spawning, on a zoo without the registry
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


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("mode", list(_MODES))
def test_tiled_mlp_ddp_grads_match_untiled(mode, frozen, ddp_grads):
    for rank, by_mode in ddp_grads.items():
        assert "error" not in by_mode, by_mode
        grads = by_mode[f"{mode}/{frozen}"]
        assert not isinstance(grads, str), grads
        for step, got in enumerate(grads):
            want = _reference(rank, step, frozen)
            assert len(got) == len(want)
            for a, b in zip(got, want):
                torch.testing.assert_close(torch.tensor(a), torch.tensor(b), rtol = 1e-4, atol = 1e-4)
