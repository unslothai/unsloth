# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Full-finetune colocate weight sync: few load_weights calls, same tensors, same order."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import linecache
import sys
import textwrap
import types

import pytest


if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed; this test drives the real patch", allow_module_level = True)

import torch


_INIT_VLLM = """
def _init_vllm(self, model):
    if self.mode == "colocate":
        self.llm = LLM(
            model = model.name_or_path,
        )
"""

# TRL >= 1.10 VLLMGeneration.sync_weights, trimmed to the parts the patch reads.
_SYNC_WEIGHTS = """
def sync_weights(self):
    if self.mode == "colocate":
        for name, param in self._iter_named_params():
            self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights([(name, param)])
    self.llm.reset_prefix_cache()
"""

_GENERATE = """
def generate(self, prompts, **kwargs):
    self.sync_weights()
    return self.llm.generate(prompts)
"""


class FakeVllmModel:
    def __init__(self):
        self.calls = []

    def load_weights(self, weights):
        self.calls.append(list(weights))


class FakeEngine:
    shared_weights = False

    def __init__(self, *args, **kwargs):
        self.model = FakeVllmModel()
        runner = types.SimpleNamespace(model = self.model)
        worker = types.SimpleNamespace(model_runner = runner)
        self.llm_engine = types.SimpleNamespace(
            model_executor = types.SimpleNamespace(driver_worker = worker)
        )
        self.resets = 0

    def reset_prefix_cache(self):
        self.resets += 1

    def generate(self, *args, **kwargs):
        return ["generated"]


def _rl_replacements():
    from unsloth.models import rl_replacements
    return rl_replacements


def _build_fake_trl(monkeypatch, sync_src = _SYNC_WEIGHTS, version = "1.15.0"):
    class_src = (
        "class VLLMGeneration:\n"
        + "\n".join(
            textwrap.indent(textwrap.dedent(src), "    ")
            for src in (_INIT_VLLM, sync_src, _GENERATE)
        )
        + "\n"
    )
    filename = f"<fake_trl_{version}_vllm_generation_sync>"
    monkeypatch.setitem(
        linecache.cache,
        filename,
        (len(class_src), None, [line + "\n" for line in class_src.splitlines()], filename),
    )
    vllm_generation = types.ModuleType("trl.generation.vllm_generation")
    vllm_generation.__file__ = filename
    vllm_generation.__spec__ = importlib.machinery.ModuleSpec(
        name = "trl.generation.vllm_generation", loader = None, origin = filename
    )
    vllm_generation.LLM = FakeEngine
    exec(compile(class_src, filename, "exec"), vllm_generation.__dict__)

    generation = types.ModuleType("trl.generation")
    generation.__spec__ = importlib.machinery.ModuleSpec(
        name = "trl.generation", loader = None, origin = "<fake trl>"
    )
    generation.vllm_generation = vllm_generation
    trl = types.ModuleType("trl")
    trl.__spec__ = importlib.machinery.ModuleSpec(name = "trl", loader = None, origin = "<fake trl>")
    trl.__spec__.submodule_search_locations = []
    trl.__path__ = []
    trl.__version__ = version
    trl.generation = generation
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "trl.generation", generation)
    monkeypatch.setitem(sys.modules, "trl.generation.vllm_generation", vllm_generation)
    monkeypatch.setattr(
        _rl_replacements(), "importlib_version", lambda name: version if name == "trl" else "0"
    )
    return vllm_generation.VLLMGeneration


def _params():
    # 4 MiB + 4 MiB + 1 MiB + 4 MiB of float32.
    return [
        ("a.weight", torch.zeros(1024, 1024)),
        ("b.weight", torch.zeros(1024, 1024)),
        ("c.bias", torch.zeros(256, 1024)),
        ("d.weight", torch.zeros(1024, 1024)),
    ]


def _make_generation(cls, params, deepspeed = None, fsdp = None, peft = False):
    self = cls.__new__(cls)
    self.mode = "colocate"
    self.llm = FakeEngine()
    self.accelerator = types.SimpleNamespace(
        state = types.SimpleNamespace(deepspeed_plugin = deepspeed, fsdp_plugin = fsdp)
    )
    self.model = types.SimpleNamespace(peft_config = {"default": object()} if peft else None)
    self._iter_named_params = lambda: iter(params)
    return self


def _sent(self):
    return [pair for call in self.llm.model.calls for pair in call]


def _assert_same_tensors(sent, params):
    assert [n for n, _ in sent] == [n for n, _ in params]
    assert all(s is p for (_, s), (_, p) in zip(sent, params)), "tensors must be passed by reference"


def test_full_finetune_sync_is_one_load_weights_call(monkeypatch):
    monkeypatch.delenv("UNSLOTH_VLLM_WEIGHT_SYNC_BUCKET_MB", raising = False)
    cls = _build_fake_trl(monkeypatch)
    _rl_replacements().vllm_generation_init_patch()
    params = _params()
    self = _make_generation(cls, params)
    cls.sync_weights(self)
    assert len(self.llm.model.calls) == 1, f"expected one load_weights call, got {len(self.llm.model.calls)}"
    _assert_same_tensors(_sent(self), params)
    assert self.llm.resets == 1
    assert self._unsloth_bucketed_weight_syncs == 1


def test_bucket_size_caps_each_call(monkeypatch):
    monkeypatch.setenv("UNSLOTH_VLLM_WEIGHT_SYNC_BUCKET_MB", "8")
    cls = _build_fake_trl(monkeypatch)
    _rl_replacements().vllm_generation_init_patch()
    params = _params()
    self = _make_generation(cls, params)
    cls.sync_weights(self)
    assert [len(c) for c in self.llm.model.calls] == [2, 2]
    _assert_same_tensors(_sent(self), params)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(env = "0"),
        dict(deepspeed = object()),
        dict(fsdp = object()),
        dict(peft = True),
    ],
    ids = ["kill_switch", "deepspeed", "fsdp", "peft"],
)
def test_per_tensor_when_disabled_or_unsafe(monkeypatch, kwargs):
    env = kwargs.pop("env", None)
    if env is None:
        monkeypatch.delenv("UNSLOTH_VLLM_WEIGHT_SYNC_BUCKET_MB", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_VLLM_WEIGHT_SYNC_BUCKET_MB", env)
    cls = _build_fake_trl(monkeypatch)
    _rl_replacements().vllm_generation_init_patch()
    params = _params()
    self = _make_generation(cls, params, **kwargs)
    cls.sync_weights(self)
    assert [len(c) for c in self.llm.model.calls] == [1, 1, 1, 1]
    _assert_same_tensors(_sent(self), params)
    assert not getattr(self, "_unsloth_bucketed_weight_syncs", 0)


def test_other_sync_layouts_are_left_alone(monkeypatch):
    sync_src = """
def sync_weights(self):
    for name, param in self.model.named_parameters():
        self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights([(name, param.data)])
"""
    cls = _build_fake_trl(monkeypatch, sync_src = sync_src)
    _rl_replacements().vllm_generation_init_patch()
    params = _params()
    self = _make_generation(cls, params)
    self.model = types.SimpleNamespace(named_parameters = lambda: iter(params), peft_config = None)
    cls.sync_weights(self)
    assert [len(c) for c in self.llm.model.calls] == [1, 1, 1, 1]
