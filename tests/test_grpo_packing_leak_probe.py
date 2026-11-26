# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO turns packed forwards off when packed documents see each other (OPT's whole-row causal mask)."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")

SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
HELPER = "_unsloth_grpo_refuse_boundary_blind_packing"
PROBE = "_unsloth_grpo_packing_leaks"
VARLEN = "_unsloth_grpo_enable_packed_varlen"
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
    monkeypatch.delenv("UNSLOTH_GRPO_PACKING_LEAK_PROBE", raising = False)


def _namespace():
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"))
    nodes = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in (HELPER, PROBE, VARLEN)
    ]
    namespace = {"os": os, "torch": torch, "UNSLOTH_ENABLE_LOGGING": False}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return namespace


def _trainer():
    return SimpleNamespace(
        accelerator = SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m)
    )


def _tiny(kind):
    torch.manual_seed(0)
    if kind == "opt":
        config = transformers.OPTConfig(
            vocab_size = 256,
            hidden_size = 32,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            ffn_dim = 64,
            max_position_embeddings = 64,
            word_embed_proj_dim = 32,
        )
        cls = transformers.OPTForCausalLM
    elif kind == "gpt2":
        config = transformers.GPT2Config(
            vocab_size = 256, n_embd = 32, n_layer = 2, n_head = 4, n_positions = 64
        )
        cls = transformers.GPT2LMHeadModel
    else:
        config = transformers.LlamaConfig(
            vocab_size = 256,
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            max_position_embeddings = 64,
        )
        cls = transformers.LlamaForCausalLM
    config._attn_implementation = "sdpa"
    return cls(config).float()


def _packed_doc2(model, first, second):
    ids = torch.cat([first, second]).unsqueeze(0)
    pos = torch.cat([torch.arange(len(first)), torch.arange(len(second))]).unsqueeze(0)
    with torch.no_grad():
        out = model(
            input_ids = ids,
            position_ids = pos,
            packed_seq_lengths = torch.tensor([len(first), len(second)], dtype = torch.int32),
            use_cache = False,
        ).logits
    return out[0, len(first) :]


def test_opt_packed_documents_see_each_other():
    model = _tiny("opt").eval()
    g = torch.Generator().manual_seed(1)
    a, b, doc = (torch.randint(10, 256, (8,), generator = g) for _ in range(3))
    assert float((_packed_doc2(model, a, doc) - _packed_doc2(model, b, doc)).abs().max()) > 1e-3
    control = _tiny("llama").eval()
    assert float((_packed_doc2(control, a, doc) - _packed_doc2(control, b, doc)).abs().max()) == 0.0


def test_leaking_model_has_packing_refused():
    model = _tiny("opt").train()
    _namespace()[HELPER](_trainer(), model)
    assert model.__dict__.get("_unsloth_seq_packing_nograd_ok") is False
    assert model.__dict__.get("_unsloth_seq_packing_grad_ok") is False
    assert model.__dict__.get("_unsloth_prefix_grouper_nograd_disabled") is True
    assert model.training


@pytest.mark.parametrize("kind", ["llama", "gpt2"])
def test_isolating_models_keep_packing(kind):
    model = _tiny(kind).eval()
    namespace = _namespace()
    assert namespace[PROBE](_trainer(), model) is False
    namespace[HELPER](_trainer(), model)
    assert not any(flag in model.__dict__ for flag in FLAGS)
    assert not model.training


def test_kill_switch_keeps_the_old_behaviour(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_PACKING_LEAK_PROBE", "0")
    model = _tiny("opt")
    _namespace()[HELPER](_trainer(), model)
    assert not any(flag in model.__dict__ for flag in FLAGS)


def test_probe_runs_once_per_model(monkeypatch):
    model = _tiny("opt")
    namespace = _namespace()
    calls = []
    probe = namespace[PROBE]
    namespace[PROBE] = lambda self, m: calls.append(m) or probe(self, m)
    namespace[HELPER](_trainer(), model)
    model.__dict__["_unsloth_seq_packing_grad_ok"] = True
    namespace[HELPER](_trainer(), model)
    assert len(calls) == 1
    assert model.__dict__["_unsloth_seq_packing_grad_ok"] is True


def test_a_model_the_probe_cannot_run_keeps_packing():
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(model_type = "llama")
    assert _namespace()[PROBE](_trainer(), model) is None
    _namespace()[HELPER](_trainer(), model)
    assert not any(flag in model.__dict__ for flag in FLAGS)


def test_the_generated_trainer_carries_the_probe():
    src = SOURCE_PATH.read_text(encoding = "utf-8")
    assert f'RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource({PROBE}))' in src
    assert src.index(f"inspect.getsource({PROBE})") < src.index(f"inspect.getsource({HELPER})")


@pytest.mark.parametrize("kind", ["opt", "llama"])
def test_probe_restores_every_submodule_mode(kind):
    # A training model with one block deliberately kept in eval mode (its dropout must stay off): the probe's
    # eval() / restore must not flip that block back to training.
    model = _tiny(kind).train()
    frozen = model.model.decoder.layers[0] if kind == "opt" else model.model.layers[0]
    frozen.eval()
    before = {name: m.training for name, m in model.named_modules()}
    _namespace()[PROBE](_trainer(), model)
    assert {name: m.training for name, m in model.named_modules()} == before


def test_probe_restores_int4_quant_state_training():
    # Int4PackedLinear.train() also syncs quant_state.training: the restore must go through it, or the probe's
    # eval() leaves the packed layer's quantization state in inference mode for the rest of training.
    import importlib.util

    path = SOURCE_PATH.parent / "compressed_tensors_int4.py"
    spec = importlib.util.spec_from_file_location("_probe_compressed_tensors_int4", path)
    int4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(int4)
    model = _tiny("llama").train()
    layer = int4.Int4PackedLinear.__new__(int4.Int4PackedLinear)  # as the loader builds its shells
    torch.nn.Module.__init__(layer)
    qs = SimpleNamespace(training = True, _fast = None)
    layer.__dict__["_int4_quant_state"] = qs
    model.model.layers[
        1
    ].int4_layer = layer  # not on the forward path: only the mode bookkeeping is under test
    frozen = model.model.layers[0]
    frozen.eval()
    before = {name: m.training for name, m in model.named_modules()}
    _namespace()[PROBE](_trainer(), model)
    assert {name: m.training for name, m in model.named_modules()} == before
    assert layer.training and qs.training is True


def test_probe_restores_a_child_shared_by_two_parents():
    # One Dropout instance under two blocks: modules() yields it once, so restoring the later (training) block
    # recursively must not leave the shared child, saved in eval mode, in training mode.
    model = _tiny("llama").train()
    shared = torch.nn.Dropout(0.5)
    model.model.layers[0].shared_dropout = shared
    model.model.layers[1].shared_dropout = shared
    shared.eval()
    before = {name: m.training for name, m in model.named_modules(remove_duplicate = False)}
    _namespace()[PROBE](_trainer(), model)
    assert shared.training is False
    assert {name: m.training for name, m in model.named_modules(remove_duplicate = False)} == before
