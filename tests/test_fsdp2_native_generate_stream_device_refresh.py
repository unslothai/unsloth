# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 generate streams decoder blocks through a `layers._unsloth_block_swap` stand-in.

A 16-bit model loaded on cuda:0 on every rank is moved to each rank's own GPU while it is sharded,
but the per-layer device published at load still says cuda:0. The fast decode loops repair that with
refresh_moved_layer_devices(layers, block_swap), which used to return early for any block swap, so
the streaming stand-in turned the repair off and rank 1 moved its activations to cuda:0 next to
weights gathered on cuda:1. Only a swapper that parks weights off the device may skip the repair.
CPU only: fake parameters report the devices."""

import ast
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import REPO, _load_native  # noqa: E402

_FUNCS = ("refresh_moved_layer_devices", "_device_of_parameters")


def _refresh_helper(calls):
    """refresh_moved_layer_devices from _utils.py without importing the unsloth package (needs a GPU)."""
    path = os.path.join(REPO, "unsloth", "models", "_utils.py")
    tree = ast.parse(open(path, encoding = "utf-8").read())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in _FUNCS]
    assert sorted(n.name for n in body) == sorted(_FUNCS)
    patching = types.ModuleType("unsloth_zoo.patching_utils")

    def verify_and_set_device(module):
        calls.append(module)
        module.__dict__["_per_layer_device"] = next(module.parameters()).device

    patching.verify_and_set_device = verify_and_set_device
    zoo = types.ModuleType("unsloth_zoo")
    zoo.patching_utils = patching
    saved = {k: sys.modules.get(k) for k in ("unsloth_zoo", "unsloth_zoo.patching_utils")}
    sys.modules["unsloth_zoo"] = zoo
    sys.modules["unsloth_zoo.patching_utils"] = patching
    namespace = {"torch": torch}
    exec(compile(ast.Module(body = body, type_ignores = []), path, "exec"), namespace)

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return namespace["refresh_moved_layer_devices"], restore


class _Layer:
    """Published on cuda:0 at load; its (sharded) weights now live on cuda:1."""

    def __init__(self):
        self.__dict__["_per_layer_device"] = torch.device("cuda", 0)

    def parameters(self):
        yield types.SimpleNamespace(device = torch.device("cuda", 1))


class _HostSwap:
    indices = (1,)  # offload_layers: block 1 lives in host RAM behind placeholders


def test_streaming_stand_in_keeps_the_device_repair():
    native = _load_native()
    calls = []
    refresh, restore = _refresh_helper(calls)
    try:
        layers = [_Layer() for _ in range(3)]
        refresh(layers, native._GenerateBlockStream({}))
        assert len(calls) == 3, "the FSDP2 streaming stand-in disabled the per-layer device repair"
        assert all(l.__dict__["_per_layer_device"] == torch.device("cuda", 1) for l in layers)

        calls.clear()
        layers = [_Layer() for _ in range(3)]
        refresh(layers, _HostSwap())  # real host block swap: placeholders are left alone
        assert calls == []
        refresh(layers, None)
        assert len(calls) == 3
    finally:
        restore()


class _OffloadedUnit:
    """A block unit whose shards sit on the host (CPUOffloadPolicy); unshard() gathers onto cuda:1."""

    def __init__(self):
        self.__dict__["_per_layer_device"] = torch.device("cuda", 0)  # published at load
        self.__dict__["_per_layer_device_index"] = 0
        self.device = torch.device("cpu")

    def unshard(self):
        self.device = torch.device("cuda", 1)

    def reshard(self):
        self.device = torch.device("cpu")

    def parameters(self):
        yield types.SimpleNamespace(device = self.device)


def test_streaming_enter_publishes_the_gathered_device():
    native = _load_native()
    unit = _OffloadedUnit()
    stream = native._GenerateBlockStream({0: unit})
    stream.enter(0)
    assert unit.__dict__["_per_layer_device"] == torch.device("cuda", 1)
    assert unit.__dict__["_per_layer_device_index"] == 1
    stream.leave(0)
    assert unit.__dict__["_per_layer_device"] == torch.device(
        "cuda", 1
    )  # where the next gather lands
