# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The compressed (FP8/NVFP4) export must free GPU weights before its llm-compressor
subprocess loads a second copy from disk, including for accelerate-dispatched multi-GPU
shards, which the old single-device-only ``.to("cpu")`` skipped and left resident.

Pulls the release/restore helpers out of unsloth/save.py via AST (importing the module
needs torch/transformers) and exercises them with fakes.
"""

from __future__ import annotations

import ast
import gc
import sys
import types
from pathlib import Path

import pytest

_SAVE_PY = Path(__file__).resolve().parent.parent / "unsloth" / "save.py"
_WANTED = {
    "_accelerate_dispatch_root",
    "_snapshot_dispatch_state",
    "_drop_accelerator_tied_param_cache",
    "_accelerate_move_guards",
    "_split_tensor_path",
    "_lookup_tensor",
    "_share_tensor",
    "_restore_dispatch_state",
    "_offload_model_for_quantize_subprocess",
    "_restore_model_after_quantize_subprocess",
}
_WANTED_ASSIGNS = {
    "_DISPATCH_SNAPSHOT_ATTR",
    "_ACCELERATE_MOVE_GUARDS",
}  # module constants the helpers close over


class _FakeLogger:
    def __init__(self):
        self.warnings = []

    def warning_once(self, msg):
        self.warnings.append(msg)


def _load_helpers(fake_torch, fake_logger):
    tree = ast.parse(_SAVE_PY.read_text(encoding = "utf-8"))
    keep = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in _WANTED)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in _WANTED_ASSIGNS for t in node.targets)
        )
    ]
    n_fns = sum(1 for node in keep if isinstance(node, ast.FunctionDef))
    assert n_fns == len(_WANTED), "release helpers missing from save.py"
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
    calls = {"removed": [], "dispatched": [], "dispatch_kwargs": [], "hooks_added": []}
    accel = types.ModuleType("accelerate")

    def _dispatch(model, device_map, **kwargs):
        calls["dispatched"].append((model, dict(device_map)))
        calls["dispatch_kwargs"].append(kwargs)

    accel.dispatch_model = _dispatch
    hooks = types.ModuleType("accelerate.hooks")
    hooks.remove_hook_from_submodules = lambda model: calls["removed"].append(model)
    hooks.add_hook_to_module = lambda module, hook: calls["hooks_added"].append((module, hook))
    accel.hooks = hooks
    monkeypatch.setitem(sys.modules, "accelerate", accel)
    monkeypatch.setitem(sys.modules, "accelerate.hooks", hooks)
    return calls


def test_dispatched_multi_gpu_model_is_released_and_redispatched(_fake_accelerate):
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    device_map = {"model.embed": 0, "model.layers.0": 0, "model.layers.1": 1}
    model = _FakeModel(device_map = device_map, devices = ("cuda:0", "cuda:1"))

    token = ns["_offload_model_for_quantize_subprocess"](model)

    assert _fake_accelerate["removed"] == [model]
    assert model.moved_to == ["cpu"]
    assert token == ("dispatch", device_map)

    ns["_restore_model_after_quantize_subprocess"](model, token)
    assert _fake_accelerate["dispatched"] == [(model, device_map)]


def test_dispatched_move_failure_redispatches_and_returns_none(_fake_accelerate):
    # If .to("cpu") raises after the hooks came off, the model must be re-dispatched, not left hookless and half-moved.
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    device_map = {"model.embed": 0, "model.layers.1": 1}

    class _MoveFails(_FakeModel):
        def to(self, target):
            raise RuntimeError("host RAM cannot hold the sharded model")

    model = _MoveFails(device_map = device_map, devices = ("cuda:0", "cuda:1"))
    token = ns["_offload_model_for_quantize_subprocess"](model)
    assert token is None  # offload aborted hooks were removed...
    assert _fake_accelerate["removed"] == [model]
    assert _fake_accelerate["dispatched"] == [(model, device_map)]


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
    # One module spilled to CPU, but the rest is the GPU memory the reload needs, and the spilled weights are already
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
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    model = _FakeModel(device_map = {"model.embed": 0, "model.layers.9": "disk"})
    assert ns["_offload_model_for_quantize_subprocess"](model) is None
    assert model.moved_to == []
    assert _fake_accelerate["removed"] == []


def test_all_cpu_map_is_left_alone(_fake_accelerate):
    # disk/meta entries are not on the model, so moving would materialize the whole checkpoint into RAM.
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    model = _FakeModel(device_map = {"model.embed": "cpu", "model.layers.0": "cpu"})
    assert ns["_offload_model_for_quantize_subprocess"](model) is None
    assert model.moved_to == []
    assert _fake_accelerate["removed"] == []


def test_single_device_model_keeps_plain_move():
    # Nothing on an accelerator:
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    model = _FakeModel(devices = ("cuda:0",))
    token = ns["_offload_model_for_quantize_subprocess"](model)
    assert model.moved_to == ["cpu"]
    assert token is not None and token[0] == "device"

    ns["_restore_model_after_quantize_subprocess"](model, token)
    assert model.moved_to[-1] == "cuda:0"


def test_quantized_model_is_released_when_the_stack_allows_it():
    # Unsloth exports load 4-bit by DEFAULT, so skipping quantized models left a shard on every GPU.
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    model = _FakeModel(devices = ("cuda:0",), quantized = True)
    token = ns["_offload_model_for_quantize_subprocess"](model)
    assert token == ("device", "cuda:0")
    assert model.moved_to == ["cpu"]


def test_quantized_model_that_refuses_to_move_is_left_usable():
    # transformers rejects .to() for some bitsandbytes builds and raises before anything moves, so the old behaviour
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
    assert fake_logger.warnings


def test_lora_merge_budgets_per_device():
    # A merged tensor W lives on the GPU of its source layer, so budget against W's own device, not GPU0, else a sharded
    # own device, not GPU0, else a sharded model OOMs GPU1+ (#7053).
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




def _fake_torch_xpu():
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(is_available = lambda: False)
    t.xpu = types.SimpleNamespace(is_available = lambda: True)
    return t


def test_dispatched_xpu_model_is_released(_fake_accelerate):
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
    # torchao runs on Intel GPUs too, so an XPU-dispatched shard must release exactly like a CUDA one.
    ns = _load_helpers(_fake_torch_xpu(), _FakeLogger())
    model = _FakeModel(devices = ("xpu:0",))
    token = ns["_offload_model_for_quantize_subprocess"](model)
    assert token == ("device", "xpu:0")
    assert model.moved_to == ["cpu"]


def test_torchao_export_uses_the_shared_release():
    """The torchao path must not re-inline a single-device-only ``.to("cpu")``.

    A plain move is invalid on a dispatched model, so single-device-only handling left
    a multi-GPU shard resident while ``device_map="auto"`` loaded a second copy.
    """
    src = _SAVE_PY.read_text(encoding = "utf-8")
    torchao = src.split("def _unsloth_save_torchao(", 1)[1].split("\ndef ", 1)[0]
    assert "_offload_model_for_quantize_subprocess(model)" in torchao
    assert "_restore_model_after_quantize_subprocess(model" in torchao
    # No hand-rolled single-device gate left behind.
    assert "len(_devs) == 1" not in torchao




# ── regressions for the multi-GPU dispatch branch ──
class _Child:
    """Minimal stand-in for an nn.Module leaf, enough for the dispatch walk."""

    def __init__(
        self,
        name = "inner",
        device_map = None,
    ):
        self._modules = {}
        self.__dict__["_name"] = name
        if device_map is not None:
            self.hf_device_map = device_map

    def named_modules(self):
        yield "", self
        for key, child in self._modules.items():
            for sub_name, sub in child.named_modules():
                yield (f"{key}.{sub_name}" if sub_name else key), sub

    def get_submodule(self, target):
        node = self
        for part in target.split("."):
            node = node._modules[part]
        return node

    def named_parameters(self, remove_duplicate = True):
        return iter(())

    def named_buffers(self, remove_duplicate = True):
        return iter(())


class _PeftLikeWrapper(_Child):
    """Proxies unknown attributes to the wrapped model, like ``PeftModelForCausalLM``:
    ``hasattr(wrapper, "_hf_hook")`` is True while ``delattr`` fails, which is what made
    the offload a silent no-op."""

    def __init__(self, inner):
        super().__init__(name = "wrapper")
        self._modules["base_model"] = inner
        self.moved_to = []

    def __getattr__(self, item):
        return getattr(self._modules["base_model"], item)

    def to(self, target):
        self.moved_to.append(str(target))
        return self

    def parameters(self):
        return iter(self._modules["base_model"]._devices)


def test_dispatch_root_is_the_inner_model_for_a_peft_style_wrapper(_fake_accelerate):
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    device_map = {"model.embed": 0, "model.layers.0": 1}
    inner = _Child(device_map = device_map)
    inner._devices = [types.SimpleNamespace(device = "cuda:0")]
    wrapper = _PeftLikeWrapper(inner)

    assert ns["_accelerate_dispatch_root"](wrapper) is inner

    token = ns["_offload_model_for_quantize_subprocess"](wrapper)
    # hooks must come off the INNER module, not the proxying wrapper
    assert _fake_accelerate["removed"] == [inner]
    assert wrapper.moved_to == ["cpu"]
    assert token == ("dispatch", device_map)


def test_dispatch_root_falls_back_to_the_model_it_was_given():
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    model = _FakeModel(device_map = {"model.embed": 0})
    assert ns["_accelerate_dispatch_root"](model) is model


def test_offload_failure_is_logged_not_swallowed():
    # A bare `return None` is indistinguishable from "nothing to move".
    fake_logger = _FakeLogger()
    ns = _load_helpers(_fake_torch(), fake_logger)

    class _Explodes(_FakeModel):
        @property
        def hf_device_map(self):
            raise RuntimeError("boom")

    assert ns["_offload_model_for_quantize_subprocess"](_Explodes()) is None
    assert any("boom" in w for w in fake_logger.warnings)


def test_restore_without_a_snapshot_forwards_skip_keys(_fake_accelerate):
    # dispatch_model() defaults skip_keys to None, which moves every forward kwarg to the executing device, wrong for
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    device_map = {"model.embed": 0, "model.layers.0": 1}
    model = _FakeModel(device_map = device_map, devices = ("cuda:0", "cuda:1"))
    model._skip_keys_device_placement = ["past_key_values"]

    ns["_restore_model_after_quantize_subprocess"](model, ("dispatch", device_map))

    assert _fake_accelerate["dispatched"] == [(model, device_map)]
    assert _fake_accelerate["dispatch_kwargs"] == [{"skip_keys": ["past_key_values"]}]


def test_snapshot_restores_a_forward_patched_after_the_dispatch(_fake_accelerate):
    """accelerate restores ``forward = _old_forward`` on removal, and ``_old_forward``
    is the forward from when the hook was FIRST attached. unsloth patches forwards after
    the dispatch, so a naive remove/re-add throws every fused kernel away for good."""
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    root = _Child(device_map = {"model.embed": 0, "mlp": 1})
    mlp = _Child(name = "mlp")
    root._modules["mlp"] = mlp

    stock_forward = lambda *a, **k: "stock"  # noqa: E731
    fused_forward = lambda *a, **k: "unsloth-fused"  # noqa: E731
    mlp._hf_hook = object()
    mlp._old_forward = stock_forward  # captured by accelerate at dispatch time installed by unsloth afterwards
    mlp.forward = fused_forward

    snapshot = ns["_snapshot_dispatch_state"](root)

    del mlp.__dict__["_hf_hook"]
    mlp.forward = mlp._old_forward
    del mlp.__dict__["_old_forward"]
    assert mlp.forward() == "stock"

    ns["_restore_dispatch_state"](root, snapshot)
    assert mlp.forward() == "unsloth-fused"
    assert mlp.__dict__["_old_forward"] is stock_forward


def test_snapshot_reties_shared_parameters(_fake_accelerate):
    """A CPU round trip repoints every tensor, so replaying the hooks alone leaves tied
    weights as independent copies: double VRAM, and updates to one never reach the other."""
    import torch

    root = _Child(device_map = {"embed": 0, "head": 0})
    shared = torch.nn.Parameter(torch.zeros(4, 4))
    for name in ("embed", "head"):
        child = _Child(name = name)
        child._parameters = {"weight": shared}
        child._buffers = {}
        root._modules[name] = child

    def named(remove_duplicate = True):
        seen, out = set(), []
        for mod_name, mod in root._modules.items():
            for attr, tensor in mod._parameters.items():
                if remove_duplicate and id(tensor) in seen:
                    continue
                seen.add(id(tensor))
                out.append((f"{mod_name}.{attr}", tensor))
        return iter(out)

    root.named_parameters = named
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    snapshot = ns_ties = ns["_snapshot_dispatch_state"](root)
    assert ns_ties[3] == [["embed.weight", "head.weight"]]

    root._modules["head"]._parameters["weight"] = torch.nn.Parameter(shared.detach().clone())
    assert (
        root._modules["embed"]._parameters["weight"].data_ptr()
        != root._modules["head"]._parameters["weight"].data_ptr()
    )

    ns["_restore_dispatch_state"](root, snapshot)
    assert (
        root._modules["embed"]._parameters["weight"].data_ptr()
        == root._modules["head"]._parameters["weight"].data_ptr()
    )


def test_meta_tensors_never_form_tie_groups(_fake_accelerate):
    """Offloaded parameters all sit on meta with storage pointer 0, so grouping by
    pointer alone would collapse them into one fake tie and overwrite them all."""
    import torch

    root = _Child(device_map = {"a": 0, "b": "cpu", "c": "cpu"})
    live = torch.nn.Parameter(torch.zeros(4, 4))
    offloaded = [
        torch.nn.Parameter(torch.empty(4, 4, device = "meta")),
        torch.nn.Parameter(torch.empty(8, 2, device = "meta")),
    ]

    def named(remove_duplicate = True):
        return iter([("a.weight", live), ("b.weight", offloaded[0]), ("c.weight", offloaded[1])])

    root.named_parameters = named
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    _hooks, places, _attrs, ties, _grads = ns["_snapshot_dispatch_state"](root)

    assert ties == []
    assert "b.weight" in places


def test_accelerate_move_guards_survive_the_replay(_fake_accelerate):
    """remove_hook_from_module also deletes the to/cuda/... guards dispatch_model
    installs to stop a caller moving an offloaded model."""
    ns = _load_helpers(_fake_torch(), _FakeLogger())
    root = _Child(device_map = {"": 0})
    guard = lambda *a, **k: "blocked"  # noqa: E731
    root._hf_hook = object()
    root.to = guard
    root.cuda = guard

    snapshot = ns["_snapshot_dispatch_state"](root)
    del root.__dict__["_hf_hook"], root.__dict__["to"], root.__dict__["cuda"]

    ns["_restore_dispatch_state"](root, snapshot)
    assert root.__dict__["to"] is guard
    assert root.__dict__["cuda"] is guard


def test_gradients_survive_the_offload_round_trip():
    """init_hook rebuilds the Parameter and drops .grad, so the snapshot has to carry it."""
    import torch

    root = _Child(device_map = {"": 0})
    weight = torch.nn.Parameter(torch.zeros(4, 4))
    weight.grad = torch.full((4, 4), 3.0)
    root._parameters = {"weight": weight}
    root.named_parameters = lambda remove_duplicate = True: iter([("weight", weight)])

    ns = _load_helpers(_fake_torch(), _FakeLogger())
    snapshot = ns["_snapshot_dispatch_state"](root)
    assert torch.equal(snapshot[4]["weight"], torch.full((4, 4), 3.0))

    # What init_hook does: same name, fresh Parameter, no grad.
    replacement = torch.nn.Parameter(torch.zeros(4, 4))
    assert replacement.grad is None
    root._parameters = {"weight": replacement}

    ns["_restore_dispatch_state"](root, snapshot)
    assert replacement.grad is not None, "the restore must put the gradient back"
    assert torch.equal(replacement.grad, torch.full((4, 4), 3.0))


def test_the_other_torchao_path_also_clears_the_failed_copy():
    """Both torchao paths must drop the copy and the traceback pinning it before restoring."""
    src = _SAVE_PY.read_text(encoding = "utf-8")
    body = src.split("\ndef _unsloth_save_torchao(", 1)[1].split("\ndef ", 1)[0]
    finally_block = body.split("    finally:", 1)[1]
    assert "del quantized_model" in finally_block
    assert "traceback.clear_frames" in finally_block
    restore_at = finally_block.index("_restore_model_after_quantize_subprocess")
    assert finally_block.index("del quantized_model") < restore_at
    assert finally_block.index("traceback.clear_frames") < restore_at


def test_cpu_spill_rejection_is_retryable():
    """bitsandbytes rejects a CPU-spilled map with a ValueError that says nothing about
    memory, so the single-device retry has to match it explicitly."""
    import importlib.util
    from pathlib import Path

    export_py = (
        Path(__file__).resolve().parent.parent
        / "studio"
        / "backend"
        / "core"
        / "export"
        / "export.py"
    )
    src = ast.parse(export_py.read_text(encoding = "utf-8"))
    keep = [
        n
        for n in src.body
        if isinstance(n, ast.FunctionDef) and n.name in {"_is_oom_error", "_is_cpu_spill_rejection"}
    ]
    assert len(keep) == 2
    namespace = {"torch": None}
    exec(  # noqa: S102 - loading trusted repo source
        compile(ast.Module(body = keep, type_ignores = []), str(export_py), "exec"), namespace
    )

    bnb = ValueError(
        "Some modules are dispatched on the CPU or the disk. Make sure you have enough "
        "GPU RAM to fit the quantized model."
    )
    assert not namespace["_is_oom_error"](bnb)
    assert namespace["_is_cpu_spill_rejection"](bnb)
    assert namespace["_is_oom_error"](RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"))
    assert not namespace["_is_cpu_spill_rejection"](RuntimeError("some other failure"))


def test_torchao_releases_the_quantized_copy_in_finally():
    """If save_pretrained raises, the quantized copy must still be dropped before the
    original is restored, or both are resident at once."""
    src = _SAVE_PY.read_text(encoding = "utf-8")
    body = src.split("def _unsloth_save_torchao_with_given_config(", 1)[1].split("\ndef ", 1)[0]
    finally_block = body.split("    finally:", 1)[1]
    assert "del quantized_model" in finally_block
    assert "_restore_model_after_quantize_subprocess(model, model_restore)" in finally_block
    # and the restore must come after the copy is dropped
    assert finally_block.index("del quantized_model") < finally_block.index(
        "_restore_model_after_quantize_subprocess"
    )
    # dropping the local is not enough: the live traceback still holds the frames
    assert "traceback.clear_frames" in finally_block
    assert finally_block.index("traceback.clear_frames") < finally_block.index(
        "_restore_model_after_quantize_subprocess"
    )


def test_a_live_traceback_pins_the_failed_copy_until_its_frames_are_cleared():
    """Why the clear_frames call above is load-bearing, on plain objects."""
    import sys
    import traceback
    import weakref

    class _Copy:
        pass

    def _build_and_fail(sink):
        copy = _Copy()  # noqa: F841 -- the point is that the frame retains it
        sink.append(weakref.ref(copy))
        raise RuntimeError("save_pretrained failed")

    def _run(clear_frames):
        # try/finally with the exception still in flight, exactly as in save.py
        sink = []
        alive = None
        try:
            try:
                _build_and_fail(sink)
            finally:
                if clear_frames:
                    exc = sys.exc_info()[1]
                    if exc is not None:
                        traceback.clear_frames(exc.__traceback__)
                gc.collect()
                alive = sink[0]() is not None
        except RuntimeError:
            pass
        return alive

    assert _run(clear_frames = False), "expected the traceback to pin the copy"
    assert not _run(clear_frames = True), "clear_frames must release it"
