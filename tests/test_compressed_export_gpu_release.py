# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The compressed (FP8/NVFP4) export must free GPU weights before its
llm-compressor subprocess loads a second copy from disk -- including for
accelerate-dispatched multi-GPU shards (e.g. Studio's multi-GPU export load),
which the old single-device-only ``.to("cpu")`` skipped, leaving every GPU
holding a full copy alongside the subprocess's.

Loads only the two release/restore helpers from unsloth/save.py via AST (the
module itself needs torch/transformers), and exercises them with fakes.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

_SAVE_PY = Path(__file__).resolve().parent.parent / "unsloth" / "save.py"
_WANTED = {
    "_offload_model_for_quantize_subprocess",
    "_restore_model_after_quantize_subprocess",
}


class _FakeLogger:
    def __init__(self):
        self.warnings = []

    def warning_once(self, msg):
        self.warnings.append(msg)


def _load_helpers(fake_torch, fake_logger):
    tree = ast.parse(_SAVE_PY.read_text(encoding = "utf-8"))
    keep = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in _WANTED
    ]
    assert len(keep) == len(_WANTED), "release helpers missing from save.py"
    namespace = {"torch": fake_torch, "logger": fake_logger}
    exec(  # noqa: S102 - loading trusted repo source
        compile(ast.Module(body = keep, type_ignores = []), str(_SAVE_PY), "exec"),
        namespace,
    )
    return namespace


def _fake_torch(cuda_available = True):
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(is_available = lambda: cuda_available)
    return t


class _FakeModel:
    def __init__(
        self,
        device_map = None,
        devices = ("cuda:0",),
        quantized = False,
    ):
        if device_map is not None:
            self.hf_device_map = device_map
        self._devices = [types.SimpleNamespace(device = d) for d in devices]
        self.moved_to = []
        self.is_loaded_in_4bit = quantized

    def parameters(self):
        return iter(self._devices)

    def to(self, target):
        self.moved_to.append(str(target))
        return self


@pytest.fixture
def _fake_accelerate(monkeypatch):
    calls = {"removed": [], "dispatched": []}
    accel = types.ModuleType("accelerate")
    accel.dispatch_model = lambda model, device_map: calls["dispatched"].append(
        (model, dict(device_map))
    )
    hooks = types.ModuleType("accelerate.hooks")
    hooks.remove_hook_from_submodules = lambda model: calls["removed"].append(model)
    accel.hooks = hooks
    monkeypatch.setitem(sys.modules, "accelerate", accel)
    monkeypatch.setitem(sys.modules, "accelerate.hooks", hooks)
    return calls


def test_dispatched_multi_gpu_model_is_released_and_redispatched(_fake_accelerate):
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    device_map = {"model.embed": 0, "model.layers.0": 0, "model.layers.1": 1}
    model = _FakeModel(device_map = device_map, devices = ("cuda:0", "cuda:1"))

    token = ns["_offload_model_for_quantize_subprocess"](model)

    assert _fake_accelerate["removed"] == [model]  # hooks removed before the move
    assert model.moved_to == ["cpu"]
    assert token == ("dispatch", device_map)

    ns["_restore_model_after_quantize_subprocess"](model, token)
    assert _fake_accelerate["dispatched"] == [(model, device_map)]


def test_dispatched_move_failure_redispatches_and_returns_none(_fake_accelerate):
    # If .to("cpu") raises AFTER the accelerate hooks are removed, the model
    # must be re-dispatched (not left hookless/half-moved) and offload aborts.
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    device_map = {"model.embed": 0, "model.layers.1": 1}

    class _MoveFails(_FakeModel):
        def to(self, target):
            raise RuntimeError("host RAM cannot hold the sharded model")

    model = _MoveFails(device_map = device_map, devices = ("cuda:0", "cuda:1"))
    token = ns["_offload_model_for_quantize_subprocess"](model)
    assert token is None  # offload aborted
    assert _fake_accelerate["removed"] == [model]  # hooks were removed...
    assert _fake_accelerate["dispatched"] == [(model, device_map)]  # ...then restored


def test_single_device_move_failure_restores_and_returns_none():
    ns = _load_helpers(_fake_torch(), _FakeLogger())

    class _MoveFails(_FakeModel):
        def __init__(self):
            super().__init__(devices = ("cuda:0",))

        def to(self, target):
            self.moved_to.append(str(target))
            if target == "cpu":
                raise RuntimeError("move failed")
            return self

    model = _MoveFails()
    token = ns["_offload_model_for_quantize_subprocess"](model)
    assert token is None
    # attempted the cpu move, then restored back to the original device
    assert model.moved_to == ["cpu", "cuda:0"]


def test_cpu_spilled_map_still_releases_its_gpu_shards(_fake_accelerate):
    # accelerate spilled one module to CPU, but the rest is still resident on the
    # GPUs -- exactly the memory the subprocess/reload needs. Those weights are in
    # host RAM already, so moving is safe and the GPU shards get reclaimed.
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    device_map = {"model.embed": 0, "model.layers.0": 1, "model.layers.9": "cpu"}
    model = _FakeModel(device_map = device_map)

    token = ns["_offload_model_for_quantize_subprocess"](model)

    assert _fake_accelerate["removed"] == [model]
    assert model.moved_to == ["cpu"]
    assert token == ("dispatch", device_map)

    ns["_restore_model_after_quantize_subprocess"](model, token)
    assert _fake_accelerate["dispatched"] == [(model, device_map)]


def test_disk_offloaded_map_is_left_alone(_fake_accelerate):
    # disk/meta entries are NOT on the model: accelerate streams them from disk,
    # so removing hooks and moving would try to materialize the whole checkpoint
    # into RAM. Leave it untouched.
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    model = _FakeModel(device_map = {"model.embed": 0, "model.layers.9": "disk"})
    assert ns["_offload_model_for_quantize_subprocess"](model) is None
    assert model.moved_to == []
    assert _fake_accelerate["removed"] == []


def test_all_cpu_map_is_left_alone(_fake_accelerate):
    # Nothing on an accelerator: there is no GPU memory to reclaim, so do not
    # churn the hooks.
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    model = _FakeModel(device_map = {"model.embed": "cpu", "model.layers.0": "cpu"})
    assert ns["_offload_model_for_quantize_subprocess"](model) is None
    assert model.moved_to == []
    assert _fake_accelerate["removed"] == []


def test_single_device_model_keeps_plain_move():
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    model = _FakeModel(devices = ("cuda:0",))
    token = ns["_offload_model_for_quantize_subprocess"](model)
    assert model.moved_to == ["cpu"]
    assert token is not None and token[0] == "device"

    ns["_restore_model_after_quantize_subprocess"](model, token)
    assert model.moved_to[-1] == "cuda:0"


def test_quantized_model_is_released_when_the_stack_allows_it():
    # The Studio export path loads 4-bit by DEFAULT, so skipping quantized models
    # left a shard on every GPU while the subprocess/reload allocated another
    # copy. Release them too where the move is accepted.
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    model = _FakeModel(devices = ("cuda:0",), quantized = True)
    token = ns["_offload_model_for_quantize_subprocess"](model)
    assert token == ("device", "cuda:0")
    assert model.moved_to == ["cpu"]


def test_quantized_model_that_refuses_to_move_is_left_usable():
    # transformers rejects .to() for some bitsandbytes builds, and that refusal
    # raises before anything moves -- so the result must be exactly the old
    # behaviour: no token, model untouched, no exception escaping.
    ns = _load_helpers(_fake_torch(), _FakeLogger())

    class _Refuses(_FakeModel):
        def to(self, target):
            raise ValueError("`.to` is not supported for 4-bit bitsandbytes models")

    model = _Refuses(devices = ("cuda:0",), quantized = True)
    assert ns["_offload_model_for_quantize_subprocess"](model) is None


def test_no_cuda_is_noop_and_restore_none_is_noop():
    ns = _load_helpers(_fake_torch(cuda_available = False), _FakeLogger())
    model = _FakeModel()
    assert ns["_offload_model_for_quantize_subprocess"](model) is None
    ns["_restore_model_after_quantize_subprocess"](model, None)  # must not raise
    assert model.moved_to == []


def test_restore_failure_warns_instead_of_raising(_fake_accelerate):
    fake_logger = _FakeLogger()
    ns = _load_helpers(_fake_torch(), fake_logger)

    class _ExplodingModel(_FakeModel):
        def to(self, target):
            raise RuntimeError("device gone")

    model = _ExplodingModel(devices = ("cuda:0",))
    ns["_restore_model_after_quantize_subprocess"](model, ("device", "cuda:0"))
    assert fake_logger.warnings  # warned, did not raise


def test_lora_merge_budgets_per_device():
    # A merged tensor W lives on the GPU of its source layer, so a model sharded
    # across GPUs (device_map="balanced") must be budgeted against W's own
    # device, not GPU0 -- otherwise GPU1+ can OOM while only GPU0 is checked
    # (#7053). Pin the device-aware budget in the LoRA-merge save path.
    src = _SAVE_PY.read_text(encoding = "utf-8")
    tree = ast.parse(src)
    fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "unsloth_save_model"
        ),
        None,
    )
    assert fn is not None, "unsloth_save_model not found"
    body = ast.get_source_segment(src, fn)
    # Budget keyed on W's device, not a hardcoded device 0 / unqualified alloc.
    assert "torch.cuda.memory_allocated(W.device)" in body
    assert "_device_vram_budget(W.device)" in body
    assert "get_device_properties(0).total_memory * maximum_memory_usage" not in body


# ── the torchao ("portable" FP8/INT8) export shares the same release ──


def _fake_torch_xpu():
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(is_available = lambda: False)
    t.xpu = types.SimpleNamespace(is_available = lambda: True)
    return t


def test_dispatched_xpu_model_is_released(_fake_accelerate):
    # torchao runs on Intel GPUs too, so an XPU-dispatched shard must release
    # exactly like a CUDA one -- otherwise every XPU holds a full copy while the
    # torchao reload pulls another from disk.
    ns = _load_helpers(_fake_torch_xpu(), _FakeLogger())
    device_map = {"model.embed": "xpu:0", "model.layers.0": "xpu:1"}
    model = _FakeModel(device_map = device_map, devices = ("xpu:0", "xpu:1"))

    token = ns["_offload_model_for_quantize_subprocess"](model)

    assert _fake_accelerate["removed"] == [model]
    assert model.moved_to == ["cpu"]
    assert token == ("dispatch", device_map)

    ns["_restore_model_after_quantize_subprocess"](model, token)
    assert _fake_accelerate["dispatched"] == [(model, device_map)]


def test_single_device_xpu_model_is_released():
    ns = _load_helpers(_fake_torch_xpu(), _FakeLogger())
    model = _FakeModel(devices = ("xpu:0",))
    token = ns["_offload_model_for_quantize_subprocess"](model)
    assert token == ("device", "xpu:0")
    assert model.moved_to == ["cpu"]


def test_torchao_export_uses_the_shared_release():
    """The torchao path must not re-inline a single-device-only ``.to("cpu")``.

    A plain move is invalid on an accelerate-dispatched model, so handling only
    single-device models left a multi-GPU shard resident on every GPU while
    ``device_map="auto"`` loaded a second copy -- an OOM for any model large
    enough to have needed the sharded load in the first place.
    """
    src = _SAVE_PY.read_text(encoding = "utf-8")
    torchao = src.split("def _unsloth_save_torchao(", 1)[1].split("\ndef ", 1)[0]
    assert "_offload_model_for_quantize_subprocess(model)" in torchao
    assert "_restore_model_after_quantize_subprocess(model" in torchao
    # No hand-rolled single-device gate left behind.
    assert "len(_devs) == 1" not in torchao
