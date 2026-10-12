# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Opt-in DDP `batched_grad_copy` (UNSLOTH_DDP_BATCHED_GRAD_COPY=1): the Trainer's
accelerate DDP kwargs handler gains `batched_grad_copy = True` only for plain DDP on a
torch whose DDP accepts it. `import unsloth` needs CUDA or XPU, so the helpers are
loaded from `_utils.py` source.
"""

import ast
import functools
import inspect
import os
import pathlib
import types

import pytest
import torch

accelerate = pytest.importorskip("accelerate")
from accelerate.utils import DistributedDataParallelKwargs, DistributedType

_UTILS = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"
_NAMES = (
    "_DDP_BGC_KWARG",
    "_DDP_BGC_HANDLER_CLASSES",
    "_DDP_BGC_PRINTED",
    "_ddp_batched_grad_copy_reason",
    "_ddp_batched_grad_copy_handler",
    "_enable_ddp_batched_grad_copy",
)
_HAS_KWARG = (
    "batched_grad_copy"
    in inspect.signature(torch.nn.parallel.DistributedDataParallel.__init__).parameters
)


def _helpers():
    tree = ast.parse(_UTILS.read_text(encoding = "utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _NAMES:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in _NAMES for t in node.targets
        ):
            nodes.append(node)
    found = {getattr(n, "name", None) or n.targets[0].id for n in nodes}
    assert found == set(_NAMES), f"missing helpers: {set(_NAMES) - found}"
    ns = {"os": os, "inspect": inspect, "torch": torch, "functools": functools}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(_UTILS), "exec"), ns)
    return ns


def _trainer(
    handler,
    distributed_type = DistributedType.MULTI_GPU,
    **state,
):
    seen = []

    def prepare_model(model, device_placement = None):
        h = accelerator.ddp_handler
        seen.append(h.to_kwargs() if h is not None else {})
        return model

    accelerator = types.SimpleNamespace(
        distributed_type = distributed_type,
        state = types.SimpleNamespace(**state),
        ddp_handler = handler,
        prepare_model = prepare_model,
    )
    return types.SimpleNamespace(accelerator = accelerator), seen


def _user_handler():
    return DistributedDataParallelKwargs(find_unused_parameters = True, bucket_cap_mb = 50)


@pytest.fixture
def opt_in(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DDP_BATCHED_GRAD_COPY", "1")


def test_off_by_default_leaves_handler_untouched(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DDP_BATCHED_GRAD_COPY", raising = False)
    ns = _helpers()
    handler = _user_handler()
    trainer, seen = _trainer(handler)
    assert ns["_enable_ddp_batched_grad_copy"](trainer) is False
    assert type(handler) is DistributedDataParallelKwargs
    trainer.accelerator.prepare_model(object())
    assert seen == [{"find_unused_parameters": True, "bucket_cap_mb": 50}]


@pytest.mark.skipif(not _HAS_KWARG, reason = "torch DDP has no batched_grad_copy")
def test_opt_in_adds_kwarg_and_keeps_user_fields(opt_in):
    ns = _helpers()
    handler = _user_handler()
    trainer, seen = _trainer(handler)
    assert ns["_enable_ddp_batched_grad_copy"](trainer) is True
    h = trainer.accelerator.ddp_handler
    assert h is handler and isinstance(h, DistributedDataParallelKwargs)
    assert "batched_grad_copy" not in h.__dict__
    assert h.to_dict() == _user_handler().to_dict()
    assert h.to_kwargs() == {
        "find_unused_parameters": True,
        "bucket_cap_mb": 50,
        "batched_grad_copy": True,
    }
    trainer.accelerator.prepare_model(object())
    assert seen[-1]["batched_grad_copy"] is True
    # A second trainer reuses the cached subclass, and re-applying is a no-op.
    other, _ = _trainer(DistributedDataParallelKwargs())
    ns["_enable_ddp_batched_grad_copy"](other)
    assert type(other.accelerator.ddp_handler) is type(h)
    ns["_enable_ddp_batched_grad_copy"](trainer)
    assert trainer.accelerator.ddp_handler.to_kwargs()["batched_grad_copy"] is True


@pytest.mark.skipif(not _HAS_KWARG, reason = "torch DDP has no batched_grad_copy")
def test_opt_in_without_handler_creates_default(opt_in):
    ns = _helpers()
    trainer, seen = _trainer(None)
    assert ns["_enable_ddp_batched_grad_copy"](trainer) is True
    trainer.accelerator.prepare_model(object())
    assert seen == [{"batched_grad_copy": True}]


@pytest.mark.skipif(not _HAS_KWARG, reason = "torch DDP has no batched_grad_copy")
def test_handler_replaced_before_prepare_still_gets_kwarg(opt_in):
    """transformers 4.x `_wrap_model` installs a fresh handler between __init__ and prepare."""
    ns = _helpers()
    trainer, seen = _trainer(DistributedDataParallelKwargs())
    ns["_enable_ddp_batched_grad_copy"](trainer)
    trainer.accelerator.ddp_handler = DistributedDataParallelKwargs(find_unused_parameters = True)
    trainer.accelerator.prepare_model(object())
    assert seen == [{"find_unused_parameters": True, "batched_grad_copy": True}]


def test_unsupported_torch_signature_not_added(opt_in, monkeypatch):
    ns = _helpers()

    class OldDDP(torch.nn.Module):
        def __init__(
            self,
            module,
            device_ids = None,
            bucket_cap_mb = None,
        ):
            super().__init__()

    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", OldDDP)
    handler = _user_handler()
    trainer, seen = _trainer(handler)
    assert ns["_enable_ddp_batched_grad_copy"](trainer) is False
    trainer.accelerator.prepare_model(object())
    assert seen == [{"find_unused_parameters": True, "bucket_cap_mb": 50}]


@pytest.mark.parametrize(
    "distributed_type, state",
    [
        (DistributedType.FSDP, {"fsdp_plugin": object()}),
        (DistributedType.DEEPSPEED, {"deepspeed_plugin": object()}),
        (DistributedType.MULTI_GPU, {"fsdp_plugin": object()}),
        (DistributedType.NO, {}),
    ],
)
def test_not_added_outside_plain_ddp(opt_in, distributed_type, state):
    ns = _helpers()
    handler = _user_handler()
    trainer, seen = _trainer(handler, distributed_type, **state)
    assert ns["_enable_ddp_batched_grad_copy"](trainer) is False
    assert type(trainer.accelerator.ddp_handler) is DistributedDataParallelKwargs
    trainer.accelerator.prepare_model(object())
    assert seen == [{"find_unused_parameters": True, "bucket_cap_mb": 50}]


def test_trainer_init_calls_it():
    """The Trainer.__init__ wrapper runs it after the original init."""
    tree = ast.parse(_UTILS.read_text(encoding = "utf-8"))
    init = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_unsloth_trainer_init"
    ]
    assert len(init) == 1
    calls = [ast.unparse(n.func) for n in ast.walk(init[0]) if isinstance(n, ast.Call)]
    assert "_enable_ddp_batched_grad_copy" in calls
    assert calls.index("_original_trainer_init") < calls.index("_enable_ddp_batched_grad_copy")
