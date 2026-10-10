"""CPU checks for patch_trl_async_grpo (loaded from rl.py source, TRL's async module replaced by a fake)."""

from __future__ import annotations

import ast
import functools
import importlib
import os
import sys
import types
import warnings

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RL_SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl.py")
NAMES = (
    "_packed_seq_lengths_from_position_ids",
    "_is_unsloth_fast_backbone",
    "_unsloth_async_grpo_add_fused_lm_head",
    "patch_trl_async_grpo",
)


class _Logger:
    def info(self, *a, **k):
        pass


def _load(fast_backbone_types = ()):
    src = open(RL_SOURCE_PATH, encoding = "utf-8").read()
    ns = {
        "torch": torch,
        "functools": functools,
        "importlib": importlib,
        "warnings": warnings,
        "logger": _Logger(),
    }
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in NAMES:
            exec(compile(ast.Module([node], []), RL_SOURCE_PATH, "exec"), ns)
    assert all(n in ns for n in NAMES)
    ns["_is_unsloth_fast_backbone"] = lambda b: isinstance(b, fast_backbone_types)
    return ns


class FastBackbone(torch.nn.Module):
    pass


class CausalLM(torch.nn.Module):
    def __init__(
        self,
        backbone,
        attn = "sdpa",
    ):
        super().__init__()
        self.model = backbone
        self.config = types.SimpleNamespace(
            _attn_implementation = attn, _name_or_path = "org/Tiny-Model"
        )

    @property
    def base_model(self):
        return self.model

    def forward(self, *a, **k):
        return k


class Peft(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    @property
    def config(self):  # PeftModel forwards attribute lookups to the wrapped model
        return self.inner.config

    def get_base_model(self):
        return self.inner


@pytest.fixture
def fake_trl(monkeypatch):
    calls = {"fused": [], "create": [], "init": []}
    mod = types.ModuleType("trl.experimental.async_grpo.async_grpo_trainer")

    def create_model_from_path(model_id, **kw):
        calls["create"].append(model_id)
        return CausalLM(torch.nn.Module(), attn = "kernels-community/flash-attn3")

    def add_fused_lm_head(model, **kw):
        calls["fused"].append(model)
        orig = model.forward

        def fused(
            *a,
            fused_lm_head = False,
            **k,
        ):
            return {"fused": fused_lm_head, **orig(*a, **k)}

        model.forward = fused

    class AsyncGRPOConfig:
        def __init__(self, output_dir):
            self.output_dir = output_dir

    class AsyncGRPOTrainer:
        def __init__(
            self,
            model,
            reward_funcs = None,
            args = None,
            **kw,
        ):
            calls["init"].append((model, args))
            mod.add_fused_lm_head(mod.create_model_from_path(model))

    mod.create_model_from_path = create_model_from_path
    mod.add_fused_lm_head = add_fused_lm_head
    mod.AsyncGRPOConfig = AsyncGRPOConfig
    mod.AsyncGRPOTrainer = AsyncGRPOTrainer
    for name in ("trl", "trl.experimental", "trl.experimental.async_grpo"):
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        pkg.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package = True)
        monkeypatch.setitem(sys.modules, name, pkg)
    mod.__spec__ = importlib.machinery.ModuleSpec(mod.__name__, None)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    sys.modules["trl.experimental.async_grpo"].async_grpo_trainer = mod
    return mod, calls


def test_packed_seq_lengths_from_position_ids():
    f = _load()["_packed_seq_lengths_from_position_ids"]
    pos = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]])
    assert f(pos).tolist() == [3, 2, 4]
    assert f(pos).dtype == torch.int32
    assert f(torch.tensor([[0, 1, 2, 3]])) is None  # a single sequence needs no boundaries
    assert f(torch.tensor([[1, 2, 0, 1]])) is None  # does not start at a sequence start
    assert f(torch.tensor([[0, 1], [0, 1]])) is None  # padded batch, not one packed row
    assert f(None) is None


def test_unsloth_model_passes_through_and_gets_boundaries(fake_trl):
    mod, calls = fake_trl
    ns = _load(fast_backbone_types = (FastBackbone,))
    ns["patch_trl_async_grpo"]()
    lm = CausalLM(FastBackbone())
    peft = Peft(lm)
    trainer = mod.AsyncGRPOTrainer(peft, None)
    # Preloaded model is used as is, the fused head goes on the CausalLM under the PEFT wrapper.
    assert calls["create"] == []
    assert calls["fused"] == [lm]
    # Default args derived from the model name instead of crashing on model.split.
    assert calls["init"][0][1].output_dir == "Tiny-Model-AsyncGRPO"
    pos = torch.tensor([[0, 1, 0, 1, 2]])
    out = lm.forward(position_ids = pos, fused_lm_head = True)
    assert out["packed_seq_lengths"].tolist() == [2, 3]
    # The plain (generation) forward is untouched.
    assert "packed_seq_lengths" not in lm.forward(position_ids = pos)
    # String ids still go through TRL's loader.
    mod.AsyncGRPOTrainer("org/x", None, args = mod.AsyncGRPOConfig("o"))
    assert calls["create"] == ["org/x"]


def test_non_unsloth_sdpa_model_is_refused(fake_trl):
    mod, _ = fake_trl
    ns = _load()
    ns["patch_trl_async_grpo"]()
    with pytest.raises(ValueError, match = "attend to each other"):
        mod.AsyncGRPOTrainer(
            CausalLM(torch.nn.Module(), attn = "sdpa"), None, args = mod.AsyncGRPOConfig("o")
        )
    # flash attention reads position_ids resets itself: left to TRL, no boundaries injected.
    lm = CausalLM(torch.nn.Module(), attn = "flash_attention_2")
    mod.AsyncGRPOTrainer(lm, None, args = mod.AsyncGRPOConfig("o"))
    assert "packed_seq_lengths" not in lm.forward(
        position_ids = torch.tensor([[0, 1, 0, 1]]), fused_lm_head = True
    )


def test_patch_is_idempotent(fake_trl):
    mod, _ = fake_trl
    ns = _load()
    ns["patch_trl_async_grpo"]()
    first = (mod.create_model_from_path, mod.add_fused_lm_head, mod.AsyncGRPOTrainer.__init__)
    ns["patch_trl_async_grpo"]()
    assert (
        mod.create_model_from_path,
        mod.add_fused_lm_head,
        mod.AsyncGRPOTrainer.__init__,
    ) == first
