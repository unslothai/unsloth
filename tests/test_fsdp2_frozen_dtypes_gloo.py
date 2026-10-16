# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""LoRA under FSDP2 puts float32 adapters next to frozen 16-bit weights in one unit. torch < 2.8
asserts one dtype over every parameter of a unit, frozen ones included; `unify_frozen_dtypes`
upcasts the frozen ones there. Two gloo ranks on CPU, real `fully_shard`, no accelerate needed.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import socket
import types

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
fsdp = pytest.importorskip("torch.distributed.fsdp")
if not hasattr(fsdp, "fully_shard") or not hasattr(fsdp, "MixedPrecisionPolicy"):
    pytest.skip("torch without the public FSDP2 API", allow_module_level = True)
if not dist.is_available() or not dist.is_gloo_available():
    pytest.skip("gloo backend unavailable", allow_module_level = True)

_FSDP2_PY = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "fsdp2.py"


def _fsdp2_module():
    # By path, so this runs on torch versions the rest of unsloth's imports do not cover.
    spec = importlib.util.spec_from_file_location("_unsloth_fsdp2_under_test", _FSDP2_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(16, 16, bias = False).to(torch.bfloat16).requires_grad_(False)
        self.lora_A = torch.nn.Linear(16, 4, bias = False)
        self.lora_B = torch.nn.Linear(4, 16, bias = False)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype))).to(x.dtype)


def _worker(rank, port, unify, out):
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    dist.init_process_group("gloo", rank = rank, world_size = 2)
    try:
        fsdp2 = _fsdp2_module()
        torch.manual_seed(0)
        model = torch.nn.Sequential(_Block(), _Block())
        policy = fsdp.MixedPrecisionPolicy(param_dtype = torch.bfloat16, reduce_dtype = torch.float32)
        plugin = types.SimpleNamespace(mixed_precision_policy = policy)
        acc = types.SimpleNamespace(
            state = types.SimpleNamespace(fsdp_plugin = plugin), mixed_precision = "bf16"
        )
        upcast = fsdp2.unify_frozen_dtypes(acc, model) if unify else False
        for block in model:
            fsdp.fully_shard(block, mp_policy = policy)
        fsdp.fully_shard(model, mp_policy = policy)
        try:
            model(torch.randn(2, 16, dtype = torch.bfloat16)).float().sum().backward()
            result = "ok"
        except AssertionError as e:
            result = str(e)
        if rank == 0:
            with open(out, "w") as f:
                f.write(f"{upcast}|{fsdp2._fsdp2_checks_frozen_dtypes()}|{result}")
    finally:
        dist.destroy_process_group()


def _run(unify, tmp_path):
    import torch.multiprocessing as mp

    out = str(tmp_path / f"unify_{unify}.txt")
    mp.spawn(_worker, args = (_free_port(), unify, out), nprocs = 2, join = True)
    upcast, frozen_checked, result = open(out).read().split("|", 2)
    return upcast == "True", frozen_checked == "True", result


def test_frozen_16bit_weights_next_to_float32_adapters_shard_and_train(tmp_path):
    upcast, frozen_checked, result = _run(True, tmp_path)
    assert result == "ok", result
    # Only torch < 2.8 needs (and gets) the upcast; newer torch keeps the frozen weights 16-bit.
    assert upcast is frozen_checked
    old_torch = tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 8)
    assert frozen_checked is old_torch


def test_without_the_upcast_old_torch_refuses_the_unit(tmp_path):
    _, frozen_checked, result = _run(False, tmp_path)
    if frozen_checked:
        assert "uniform original parameter dtype" in result
    else:
        assert result == "ok", result
