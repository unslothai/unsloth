# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""One gloo rank for test_ddp_comm_defaults.py: a real accelerate Accelerator on CPU builds the DDP."""

from __future__ import annotations

import importlib.util
import os
import types

_DEFAULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unsloth",
    "models",
    "_ddp_defaults.py",
)


def load_defaults():
    # By path: importing the unsloth package needs a GPU.
    spec = importlib.util.spec_from_file_location("_unsloth_ddp_defaults_under_test", _DEFAULTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(kind):
    import torch

    torch.manual_seed(0)
    layers = [
        torch.nn.Linear(8, 16),
        torch.nn.GELU(),
        torch.nn.Linear(16, 16),
        torch.nn.GELU(),
        torch.nn.Linear(16, 5),
    ]
    model = torch.nn.Sequential(*layers)
    model.register_buffer("scale", torch.tensor(1.5), persistent = False)
    if kind == "peft":
        model.peft_config = {"default": object()}
    return model


def batch(rank, n = 6):
    import torch
    g = torch.Generator().manual_seed(1234 + rank)
    return torch.randn(n, 8, generator = g), torch.randint(0, 5, (n,), generator = g)


def loss_fn(model, x, y):
    import torch
    return torch.nn.functional.cross_entropy(model(x) * 1.0, y)


def run_rank(rank, port, queue, cases):
    import torch
    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        LOCAL_RANK = str(rank),
        WORLD_SIZE = "2",
        LOCAL_WORLD_SIZE = "2",
        ACCELERATE_USE_CPU = "true",
    )
    torch.set_num_threads(1)
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState, PartialState

    out = {}
    accelerator = Accelerator(cpu = True)
    try:
        defaults = load_defaults()
        for name, kind, env, args in cases:
            saved = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                accelerator.ddp_handler = None
                accelerator.prepare_model = types.MethodType(
                    type(accelerator).prepare_model, accelerator
                )
                trainer = types.SimpleNamespace(
                    accelerator = accelerator, args = types.SimpleNamespace(**args)
                )
                installed = defaults.install_unsloth_ddp_comm_defaults(trainer)
                model = build(kind)
                ddp = accelerator.prepare_model(model)
                opt = torch.optim.SGD(ddp.parameters(), lr = 0.1)
                x, y = batch(rank)
                loss = loss_fn(ddp, x, y)
                loss.backward()
                grads = [p.grad.detach().clone() for p in ddp.module.parameters()]
                storages = {p.grad.untyped_storage().data_ptr() for p in ddp.module.parameters()}
                opt.step()
                losses = [None, None]
                dist.all_gather_object(losses, float(loss))
                out[name] = dict(
                    installed = installed,
                    is_ddp = isinstance(ddp, torch.nn.parallel.DistributedDataParallel),
                    view = bool(ddp.gradient_as_bucket_view),
                    hooks = len(getattr(ddp, "_comm_hooks", []) or [])
                    or int(getattr(ddp, "_unsloth_test_hook", 0)),
                    n_grad_storages = len(storages),
                    losses = losses,
                    grads = [g.tolist() for g in grads],
                    params = [p.detach().tolist() for p in ddp.module.parameters()],
                )
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        queue.put((rank, out))
    except BaseException:
        import traceback
        queue.put((rank, {"error": traceback.format_exc()}))
        raise
    finally:
        dist.destroy_process_group()
        AcceleratorState._reset_state(True)
        PartialState._reset_state()
