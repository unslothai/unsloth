# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""CPU checks for patch_trl_fused_lm_head_devices (loaded from rl.py source, TRL's utils replaced by a fake)."""

from __future__ import annotations

import ast
import functools
import os
import sys
import types
from collections import OrderedDict

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RL_SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl.py")
NAMES = (
    "_spans_several_devices",
    "_first_tensor_device",
    "_ApplyOnInputDevice",
    "patch_trl_fused_lm_head_devices",
)


class _Outputs(OrderedDict):
    """ModelOutput stand-in: a dict whose items TRL reads by attribute."""

    __getattr__ = OrderedDict.__getitem__


def _load():
    src = open(RL_SOURCE_PATH, encoding = "utf-8").read()
    ns = {"torch": torch, "functools": functools, "sys": sys}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in NAMES:
            exec(compile(ast.Module([node], []), RL_SOURCE_PATH, "exec"), ns)
    assert all(n in ns for n in NAMES)
    return ns


class _Model(torch.nn.Module):
    def __init__(self, device_map):
        super().__init__()
        if device_map is not None:
            self.hf_device_map = device_map

    def forward(
        self,
        input_ids = None,
        **kwargs,
    ):
        return "original"


def _fake_add_fused_lm_head(model, *args, **kwargs):
    # Like TRL: the head output lives on the head's device (CPU here), whatever device the inputs came on.
    def _fused_forward(
        self,
        *f_args,
        fused_lm_head = False,
        labels = None,
        **f_kwargs,
    ):
        if not fused_lm_head:
            return "unfused"
        rows, cols = labels.shape
        return _Outputs(
            loss = torch.zeros(()),
            log_probs = torch.zeros(rows, cols),
            entropy = None,
            label_mask = torch.ones(rows, cols, dtype = torch.bool),
        )

    model.forward = types.MethodType(_fused_forward, model)


class _Kernel:
    calls = []

    @staticmethod
    def apply(hidden, *args):
        _Kernel.calls.append(
            (hidden.device, torch.cuda.current_device() if torch.cuda.is_available() else None)
        )
        return hidden


@pytest.fixture
def fake_trl(monkeypatch):
    utils = types.ModuleType("trl.trainer.utils")
    utils.add_fused_lm_head = _fake_add_fused_lm_head
    utils._ChunkedLogProbFunction = _Kernel
    dpo = types.ModuleType("trl.trainer.dpo_trainer")
    dpo.add_fused_lm_head = _fake_add_fused_lm_head
    unrelated = types.ModuleType("trl.trainer.other")
    for name, module in (
        ("trl", types.ModuleType("trl")),
        ("trl.trainer", types.ModuleType("trl.trainer")),
        ("trl.trainer.utils", utils),
        ("trl.trainer.dpo_trainer", dpo),
        ("trl.trainer.other", unrelated),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["trl"].trainer = sys.modules["trl.trainer"]
    sys.modules["trl.trainer"].utils = utils
    return utils, dpo, unrelated


def test_rebinds_every_trl_module_and_is_idempotent(fake_trl):
    utils, dpo, unrelated = fake_trl
    ns = _load()
    ns["patch_trl_fused_lm_head_devices"]()
    patched = utils.add_fused_lm_head
    assert patched is not _fake_add_fused_lm_head
    assert patched._unsloth_device_patched
    assert dpo.add_fused_lm_head is patched
    assert not hasattr(unrelated, "add_fused_lm_head")
    ns["patch_trl_fused_lm_head_devices"]()
    assert utils.add_fused_lm_head is patched


@pytest.mark.parametrize("device_map", [None, {"": 0}, {"model": 0, "lm_head": 0}])
def test_single_device_model_keeps_trl_forward(fake_trl, device_map):
    utils, _, _ = fake_trl
    _load()["patch_trl_fused_lm_head_devices"]()
    model = _Model(device_map)
    utils.add_fused_lm_head(model)
    assert isinstance(model.forward, types.MethodType)
    assert model.forward.__func__.__name__ == "_fused_forward"
    assert utils._ChunkedLogProbFunction is _Kernel


def test_split_model_returns_outputs_on_input_device(fake_trl):
    utils, _, _ = fake_trl
    _load()["patch_trl_fused_lm_head_devices"]()
    model = _Model({"model.embed_tokens": 0, "model.layers.0": 1, "lm_head": 1})
    utils.add_fused_lm_head(model)
    # meta stands in for the input card: the fake head answers on the CPU, like a head on another card.
    input_ids = torch.empty(2, 5, dtype = torch.long, device = "meta")
    labels = torch.zeros(2, 5, dtype = torch.long)
    out = model(input_ids = input_ids, labels = labels, fused_lm_head = True)
    assert out.log_probs.device.type == "meta"
    assert out.label_mask.device.type == "meta"
    assert out.loss.device.type == "meta"
    assert out.entropy is None
    # Without the flag the forward is the original one, untouched.
    assert model(input_ids = input_ids) == "unfused"
    assert utils._ChunkedLogProbFunction.function is _Kernel
    shim = utils._ChunkedLogProbFunction
    utils.add_fused_lm_head(_Model({"model": 0, "lm_head": 1}))
    assert utils._ChunkedLogProbFunction is shim


def test_kernel_shim_passes_cpu_tensors_through():
    ns = _load()
    _Kernel.calls.clear()
    hidden = torch.ones(2, 3)
    assert ns["_ApplyOnInputDevice"](_Kernel).apply(hidden, 1, 2) is hidden
    assert _Kernel.calls[-1][0].type == "cpu"


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_kernel_shim_switches_to_the_hidden_device(monkeypatch, device_type):
    # No accelerator needed: the hidden states only have to report a device, and the backend is faked.
    ns = _load()
    backend = getattr(torch, device_type)
    entered = []

    class _Device:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            entered.append(self.device)

        def __exit__(self, *exc):
            entered.append(None)

    monkeypatch.setattr(backend, "current_device", lambda: 0)
    monkeypatch.setattr(backend, "device", _Device)
    hidden = types.SimpleNamespace(device = torch.device(device_type, 1))
    kernel = types.SimpleNamespace(apply = lambda h, *a: (entered[-1] if entered else "none", a))
    assert ns["_ApplyOnInputDevice"](kernel).apply(hidden, 7) == (
        torch.device(device_type, 1),
        (7,),
    )
    assert entered == [torch.device(device_type, 1), None]
    entered.clear()
    hidden.device = torch.device(device_type, 0)
    assert ns["_ApplyOnInputDevice"](kernel).apply(hidden) == ("none", ())
    assert entered == []


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason = "needs two CUDA devices"
)
def test_kernel_shim_runs_on_the_hidden_device():
    ns = _load()
    _Kernel.calls.clear()
    with torch.cuda.device(0):
        ns["_ApplyOnInputDevice"](_Kernel).apply(torch.ones(2, 3, device = "cuda:1"))
        assert torch.cuda.current_device() == 0
    assert _Kernel.calls[-1] == (torch.device("cuda", 1), 1)


def test_missing_trl_helper_is_a_no_op(monkeypatch):
    utils = types.ModuleType("trl.trainer.utils")
    monkeypatch.setitem(sys.modules, "trl", types.ModuleType("trl"))
    monkeypatch.setitem(sys.modules, "trl.trainer", types.ModuleType("trl.trainer"))
    monkeypatch.setitem(sys.modules, "trl.trainer.utils", utils)
    sys.modules["trl"].trainer = sys.modules["trl.trainer"]
    sys.modules["trl.trainer"].utils = utils
    _load()["patch_trl_fused_lm_head_devices"]()
    assert not hasattr(utils, "add_fused_lm_head")
