# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
import sys
import types
from pathlib import Path

import pytest

LOADER = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "loader.py"
GEMMA4_TYPES = ["gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text"]


class _Device:
    def __init__(
        self,
        type,
        index = None,
    ):
        self.type, self.index = type, index

    def __eq__(self, other):
        return (self.type, self.index) == (
            getattr(other, "type", None),
            getattr(other, "index", None),
        )


def _helper(
    monkeypatch,
    device_type,
    capability,
    target = "default",
):
    """The real helpers on a fake torch: `capability(device)` answers per device, `None` = current card.
    `target` is unsloth_zoo.vllm_utils.get_target_device's answer ("missing": an older zoo without it)."""
    fake_torch = types.ModuleType("torch")
    fake_torch.device = _Device
    fake_torch.cuda = types.SimpleNamespace(get_device_capability = capability)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    zoo_vllm = types.ModuleType("unsloth_zoo.vllm_utils")
    if target != "missing":
        engine = _Device("cuda", 0) if target == "default" else target
        zoo_vllm.get_target_device = lambda index = 0: engine
    monkeypatch.setitem(sys.modules, "unsloth_zoo", types.ModuleType("unsloth_zoo"))
    monkeypatch.setitem(sys.modules, "unsloth_zoo.vllm_utils", zoo_vllm)
    tree = ast.parse(LOADER.read_text(encoding = "utf-8"))
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_vllm_needs_newer_gpu",
            "_vllm_engine_device",
        ):
            wanted.append(node)
        elif isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "VLLM_NEEDS_SM80_MODEL_TYPES" for t in node.targets
        ):
            wanted.append(node)
    assert any(getattr(n, "name", None) == "_vllm_needs_newer_gpu" for n in wanted)
    namespace = {"DEVICE_TYPE": device_type}
    exec(compile(ast.Module(body = wanted, type_ignores = []), "loader.py", "exec"), namespace)
    return namespace["_vllm_needs_newer_gpu"]


@pytest.mark.parametrize("model_type", GEMMA4_TYPES)
def test_gemma4_falls_back_below_sm80(monkeypatch, model_type):
    f = _helper(monkeypatch, "cuda", lambda device = None: (7, 5))
    assert f([model_type]) == model_type


@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (9, 0), (12, 0)])
def test_gemma4_keeps_vllm_on_sm80_and_newer(monkeypatch, capability):
    f = _helper(monkeypatch, "cuda", lambda device = None: capability)
    assert f(GEMMA4_TYPES) is None


def test_other_models_keep_vllm_on_t4(monkeypatch):
    f = _helper(monkeypatch, "cuda", lambda device = None: (7, 5))
    for model_types in (["qwen3"], ["llama"], ["gemma3", "gemma3_text"], ["qwen3_5"]):
        assert f(model_types) is None


def test_non_cuda_devices_untouched(monkeypatch):
    def boom(device = None):
        raise AssertionError("capability must not be probed off CUDA")

    for device_type in ("hip", "xpu", "mps", "cpu"):
        f = _helper(monkeypatch, device_type, boom)
        assert f(["gemma4"]) is None


def test_failed_probe_keeps_vllm(monkeypatch):
    def broken(device = None):
        raise RuntimeError("no device")

    f = _helper(monkeypatch, "cuda", broken)
    assert f(["gemma4"]) is None


def _mixed_host(caps, current):
    # caps: per-card capability; a device-less query reads the current card, as torch documents.
    def capability(device = None):
        index = current if device is None else getattr(device, "index", device)
        return caps[current if index is None else index]

    return capability


@pytest.mark.parametrize("target", ["default", "missing"])
def test_gemma4_probes_the_engine_card_not_the_current_one(monkeypatch, target):
    # Current card cuda:1 is a T4, the engine's cuda:0 is an A100: vLLM can run Gemma 4 there.
    f = _helper(monkeypatch, "cuda", _mixed_host({0: (8, 0), 1: (7, 5)}, current = 1), target)
    assert f(["gemma4"]) is None


def test_gemma4_falls_back_when_the_engine_card_is_old(monkeypatch):
    # Current card cuda:1 is an H100, the engine's cuda:0 is a T4: Gemma 4 cannot run in vLLM there.
    f = _helper(monkeypatch, "cuda", _mixed_host({0: (7, 5), 1: (9, 0)}, current = 1))
    assert f(["gemma4"]) == "gemma4"


def test_gemma4_follows_zoos_target_device(monkeypatch):
    f = _helper(
        monkeypatch, "cuda", _mixed_host({0: (8, 0), 1: (7, 5)}, current = 0), _Device("cuda", 1)
    )
    assert f(["gemma4"]) == "gemma4"


@pytest.mark.parametrize("capability", [(7, 5), (8, 0), (9, 0)])
def test_single_gpu_answer_unchanged(monkeypatch, capability):
    # One card: the engine card is the current card, so the verdict matches the device-less query.
    expected = "gemma4" if capability[0] < 8 else None
    f = _helper(monkeypatch, "cuda", _mixed_host({0: capability}, current = 0))
    assert f(["gemma4"]) == expected


def test_fallback_drops_max_num_batched_tokens(capsys):
    # load_vllm takes it; transformers' from_pretrained rejects it once vLLM is switched off.
    tree = ast.parse(LOADER.read_text(encoding = "utf-8"))
    wanted = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "_drop_vllm_engine_kwargs")
        or (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "_VLLM_ENGINE_KWARGS" for t in node.targets)
        )
    ]
    namespace = {}
    exec(compile(ast.Module(body = wanted, type_ignores = []), "loader.py", "exec"), namespace)
    kwargs = {"max_num_batched_tokens": 4096, "token": "x"}
    namespace["_drop_vllm_engine_kwargs"](kwargs)
    assert kwargs == {"token": "x"}
