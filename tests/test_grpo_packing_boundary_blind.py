# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO keeps its packed forwards off for models whose layers mix packed rows (packing_boundary_blocker)."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
HELPER = "_unsloth_grpo_refuse_boundary_blind_packing"
FLAGS = (
    "_unsloth_seq_packing_nograd_ok",
    "_unsloth_seq_packing_grad_ok",
    "_unsloth_prefix_grouper_nograd_disabled",
)


@pytest.fixture(autouse = True)
def packing_module(monkeypatch):
    # The helper imports unsloth.utils.packing; load that file alone (importing unsloth needs a GPU).
    import importlib.util
    import sys
    import types

    path = SOURCE_PATH.parents[1] / "utils" / "packing.py"
    spec = importlib.util.spec_from_file_location("unsloth.utils.packing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("unsloth", "unsloth.utils"):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = []
            monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setitem(sys.modules, "unsloth.utils.packing", module)


def _helper():
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"))
    names = (HELPER, "_unsloth_grpo_packing_leaks", "_unsloth_grpo_enable_packed_varlen")
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"os": os, "torch": torch, "UNSLOTH_ENABLE_LOGGING": False}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return namespace[HELPER]


def _blind_module(name):
    return type(name, (torch.nn.Module,), {})()


def _qwen35_gdn():
    mixer = _blind_module("Qwen3_5GatedDeltaNet")
    mixer.conv1d = torch.nn.Conv1d(2, 2, 3)
    return mixer


class _Model(torch.nn.Module):
    def __init__(
        self,
        inner,
        model_type = "llama",
    ):
        super().__init__()
        self.layer = inner
        self.config = SimpleNamespace(model_type = model_type)


def _trainer():
    return SimpleNamespace(
        accelerator = SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m)
    )


@pytest.mark.parametrize(
    "model",
    [
        _Model(_qwen35_gdn()),
        _Model(_blind_module("RecurrentGemmaRglru")),
        _Model(_blind_module("ZayaCCAProjection")),
        _Model(_blind_module("DeepseekV4CSACompressor")),
        _Model(torch.nn.Linear(2, 2), model_type = "deepseek_v4"),
    ],
)
def test_every_packed_path_is_refused_once(model):
    trainer = _trainer()
    _helper()(trainer, model)
    assert model.__dict__["_unsloth_seq_packing_nograd_ok"] is False
    assert model.__dict__["_unsloth_seq_packing_grad_ok"] is False
    assert model.__dict__["_unsloth_prefix_grouper_nograd_disabled"] is True
    # Decided once per model: a later call does not re-walk it.
    model.__dict__["_unsloth_seq_packing_grad_ok"] = True
    _helper()(trainer, model)
    assert model.__dict__["_unsloth_seq_packing_grad_ok"] is True


def test_ref_model_scored_first_still_protects_the_policy():
    # TRL scores a separate ref_model before the policy when the old-logps pass is skipped.
    trainer = _trainer()
    ref = _Model(_blind_module("ZayaCCAProjection"))
    policy = _Model(_blind_module("ZayaCCAProjection"))
    _helper()(trainer, ref)
    _helper()(trainer, policy)
    for model in (ref, policy):
        assert model.__dict__["_unsloth_seq_packing_grad_ok"] is False
        assert model.__dict__["_unsloth_seq_packing_nograd_ok"] is False


@pytest.mark.parametrize(
    "leaks, scan_hit, can_pack", [(False, False, False), (None, True, False), (False, True, True)]
)
def test_wrapped_gdn_packing_requires_verified_boundaries(leaks, scan_hit, can_pack):
    model = _Model(_qwen35_gdn())
    model._unsloth_varlen_forward_wrapped = model.layer._unsloth_varlen_wrapped = True
    model.layer._unsloth_varlen_conv_hit = True
    model.layer._unsloth_varlen_scan_hit = scan_hit
    check = _helper()
    check.__globals__["_unsloth_grpo_packing_leaks"] = lambda *_: leaks
    trainer = _trainer()
    check(trainer, model)
    assert getattr(model, "_unsloth_seq_packing_grad_ok", True) is can_pack
    assert getattr(model, "_unsloth_seq_packing_nograd_ok", True) is can_pack

    # Gradient checkpointing keeps a verified mixer packed: the boundary stash survives the no-grad
    # per-row verify forward (tests/utils/test_hybrid_packing_default.py checks the replay itself).
    model = _Model(_qwen35_gdn())
    model._unsloth_varlen_forward_wrapped = model.layer._unsloth_varlen_wrapped = True
    model.layer._unsloth_varlen_conv_hit = True
    model.layer._unsloth_varlen_scan_hit = scan_hit
    model.is_gradient_checkpointing = True
    check(trainer, model)
    assert getattr(model, "_unsloth_seq_packing_grad_ok", True) is can_pack


def _conv(
    x,
    weight,
    bias = None,
    activation = None,
    seq_idx = None,
    **kwargs,
):
    # Depthwise causal conv that restarts at every seq_idx change; leak = True ignores seq_idx.
    cuts = [0, x.shape[-1]]
    if seq_idx is not None and not kwargs.get("leak"):
        ids = seq_idx.reshape(-1)
        cuts = [0] + (torch.nonzero(ids[1:] != ids[:-1]).flatten() + 1).tolist() + [x.shape[-1]]
    pad = weight.shape[-1] - 1
    outs = [
        torch.nn.functional.conv1d(
            x[:, :, s:e], weight.unsqueeze(1), bias, padding = pad, groups = x.shape[1]
        )[:, :, : e - s]
        for s, e in zip(cuts[:-1], cuts[1:])
    ]
    return torch.nn.functional.silu(torch.cat(outs, dim = -1))


def _leaky_conv(
    x,
    weight,
    bias = None,
    activation = None,
    seq_idx = None,
    **kwargs,
):
    return _conv(x, weight, bias, activation, seq_idx, leak = True)


def _scan(
    x,
    cu_seqlens = None,
    **kwargs,
):
    # Decayed running sum, reset at every cu_seqlens start.
    starts = set(cu_seqlens.tolist()[:-1]) if cu_seqlens is not None else {0}
    out, state = [], torch.zeros_like(x[:, 0])
    for t in range(x.shape[1]):
        state = (0 if t in starts else 0.7) * state + x[:, t]
        out.append(state)
    return torch.stack(out, dim = 1)


class Qwen3_5GatedDeltaNet(torch.nn.Module):
    """CPU stand-in with the instance kernels transformers' Qwen3.5 mixer dispatches through."""

    def __init__(self, conv = _conv):
        super().__init__()
        self.hidden_size = 8
        self.conv1d = torch.nn.Conv1d(8, 8, 4, groups = 8, bias = False)
        self.causal_conv1d_fn = conv
        self.chunk_gated_delta_rule = _scan

    def forward(
        self,
        hidden_states,
        cache_params = None,
        attention_mask = None,
    ):
        x = self.causal_conv1d_fn(
            x = hidden_states.transpose(1, 2), weight = self.conv1d.weight.squeeze(1), seq_idx = None
        )
        return self.chunk_gated_delta_rule(x.transpose(1, 2), cu_seqlens = None)


class _HybridLM(torch.nn.Module):
    def __init__(self, conv = _conv):
        super().__init__()
        torch.manual_seed(0)
        self.config = SimpleNamespace(
            model_type = "qwen3_5", layer_types = ["linear_attention", "full_attention"]
        )
        self.embed = torch.nn.Embedding(32, 8)
        self.linear_attn = Qwen3_5GatedDeltaNet(conv)
        self.head = torch.nn.Linear(8, 32)

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        input_ids = None,
        position_ids = None,
        packed_seq_lengths = None,
        use_cache = None,
    ):
        return SimpleNamespace(logits = self.head(self.linear_attn(self.embed(input_ids))))


@pytest.fixture
def hybrid_opt_in(monkeypatch):
    # The boundary shim packs gated-delta mixers by default on CUDA only; opt in for these CPU stand-ins.
    monkeypatch.setenv("UNSLOTH_EXPERIMENTAL_HYBRID_PACKING", "1")
    monkeypatch.delenv("UNSLOTH_GRPO_SEQ_PACKING", raising = False)


def test_unwrapped_gdn_packs_exactly_through_the_boundary_shim(hybrid_opt_in):
    """A freshly loaded GDN model: GRPO installs the shim, so whatever packing it allows matches per-row scoring."""
    model = _HybridLM()
    _helper()(_trainer(), model)
    assert model.__dict__.get("_unsloth_seq_packing_grad_ok") is not False
    assert model.__dict__.get("_unsloth_seq_packing_nograd_ok") is not False
    ids = torch.randint(0, 32, (1, 12), generator = torch.Generator().manual_seed(1))
    packed = model(
        input_ids = ids,
        position_ids = torch.arange(6).repeat(1, 2),
        packed_seq_lengths = torch.tensor([6, 6], dtype = torch.int32),
        use_cache = False,
    ).logits
    rows = torch.cat(
        [model(input_ids = ids[:, :6]).logits, model(input_ids = ids[:, 6:]).logits], dim = 1
    )
    torch.testing.assert_close(packed, rows)


@pytest.mark.parametrize("env", [{}, {"UNSLOTH_EXPERIMENTAL_HYBRID_PACKING": "0"}])
def test_gdn_without_exact_boundaries_scores_per_row(hybrid_opt_in, monkeypatch, env):
    """A kernel that drops seq_idx (the shim's probe rejects it), or the shim switched off: no packed GRPO forward,
    even though the leak is too small for the 0.7 nat packed self-verify."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    model = _HybridLM(conv = _leaky_conv if not env else _conv)
    _helper()(_trainer(), model)
    for flag in ("_unsloth_seq_packing_grad_ok", "_unsloth_seq_packing_nograd_ok"):
        assert model.__dict__[flag] is False


def test_grpo_packing_off_leaves_the_model_unpatched(hybrid_opt_in, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "0")
    model = _HybridLM()
    _helper()(_trainer(), model)
    assert "forward" not in model.__dict__
    assert not getattr(model.linear_attn, "_unsloth_varlen_wrapped", False)


def test_an_ordinary_model_keeps_packing():
    model = _Model(torch.nn.Linear(2, 2))
    _helper()(_trainer(), model)
    assert not any(flag in model.__dict__ for flag in FLAGS)


def test_the_generated_trainer_defines_and_calls_the_check():
    src = SOURCE_PATH.read_text(encoding = "utf-8")
    assert f'RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource({HELPER}))' in src
    tree = ast.parse(src)
    for outer in ("grpo_trainer__get_per_token_logps_and_entropies", "grpo_trainer_compute_loss"):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == outer)
        assert f"{HELPER}(self, model)" in ast.unparse(node), outer
