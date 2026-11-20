# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO keeps its packed forwards off for models whose layers mix packed rows (packing_boundary_blocker)."""

import ast
import inspect
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
    names = (HELPER, "_unsloth_grpo_packing_leaks")
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"os": os, "torch": torch, "UNSLOTH_ENABLE_LOGGING": False}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return namespace[HELPER]


def _blind_module(name):
    return type(name, (torch.nn.Module,), {})()


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


class GatedDeltaNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(2, 2, 3)


@pytest.mark.parametrize("layer", [lambda: _blind_module("ZayaCCAProjection"), GatedDeltaNet])
def test_ref_model_scored_first_still_protects_the_policy(layer):
    # TRL scores a separate ref_model before the policy when the old-logps pass is skipped.
    trainer = _trainer()
    ref = _Model(layer())
    policy = _Model(layer())
    _helper()(trainer, ref)
    _helper()(trainer, policy)
    for model in (ref, policy):
        assert model.__dict__["_unsloth_seq_packing_grad_ok"] is False
        assert model.__dict__["_unsloth_seq_packing_nograd_ok"] is False


@pytest.mark.parametrize("outer, inner", [(False, False), (True, False), (False, True)])
def test_gdn_without_complete_boundary_support_skips_the_probe(outer, inner):
    check = _helper()
    check.__globals__["_unsloth_grpo_packing_leaks"] = lambda *_: pytest.fail("missing boundaries")
    model = _Model(GatedDeltaNet())
    model._unsloth_varlen_forward_wrapped = outer
    model.layer._unsloth_varlen_wrapped = inner
    check(_trainer(), model)
    assert model._unsloth_seq_packing_nograd_ok is False
    assert model._unsloth_seq_packing_grad_ok is False


@pytest.mark.parametrize(
    "probe, scan_hit", [(None, True), (True, True), (False, False), (False, True)]
)
def test_wrapped_gdn_needs_verified_boundaries(probe, scan_hit):
    model = _Model(GatedDeltaNet())
    model._unsloth_varlen_forward_wrapped = model.layer._unsloth_varlen_wrapped = True
    model.layer._unsloth_varlen_conv_hit = True
    model.layer._unsloth_varlen_scan_hit = scan_hit
    check = _helper()
    check.__globals__["_unsloth_grpo_packing_leaks"] = lambda *_: probe
    trainer = _trainer()
    check(trainer, model)
    if probe is False and scan_hit:
        assert not any(flag in model.__dict__ for flag in FLAGS)
        # A later checkpointed backward cannot reuse boundaries cleared by a per-row forward.
        model.is_gradient_checkpointing = True
        check(trainer, model)
        assert model._unsloth_seq_packing_grad_ok is False
        assert "_unsloth_seq_packing_nograd_ok" not in model.__dict__
    else:
        assert model._unsloth_seq_packing_grad_ok is False
        assert model._unsloth_seq_packing_nograd_ok is False


def test_qwen35_grpo_matches_independent_completion_gradients(monkeypatch):
    transformers = pytest.importorskip("transformers", minversion = "5.18.0")
    peft = pytest.importorskip("peft")
    zoo = pytest.importorskip("unsloth_zoo.rl_replacements")
    from transformers.models.qwen3_5 import modeling_qwen3_5

    # Run the reference PyTorch kernels even when optional CUDA extensions are installed.
    for name in ("causal_conv1d_fn", "torch_chunk_gated_delta_rule"):
        monkeypatch.setattr(modeling_qwen3_5, name, inspect.unwrap(getattr(modeling_qwen3_5, name)))
    monkeypatch.setattr(zoo, "device_synchronize", lambda: None)
    monkeypatch.setattr(zoo.grpo_accumulated_loss, "_pg_funcs", (), raising = False)
    monkeypatch.delenv("UNSLOTH_GRPO_SEQ_PACKING", raising = False)
    config = transformers.Qwen3_5TextConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        layer_types = ["linear_attention", "full_attention"],
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
        linear_num_key_heads = 2,
        linear_num_value_heads = 2,
        linear_key_head_dim = 16,
        linear_value_head_dim = 16,
        pad_token_id = 0,
        use_cache = False,
        attn_implementation = "sdpa",
    )
    with torch.random.fork_rng(devices = []):
        torch.manual_seed(3407)
        model = peft.get_peft_model(
            transformers.Qwen3_5ForCausalLM(config),
            peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["in_proj_qkv", "out_proj"]),
        )
    # Keep the unisolated logit differences below the numerical probe's absolute floor.
    with torch.no_grad():
        model.get_output_embeddings().weight.mul_(0.01)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    rows = torch.tensor([[2, 3, 4, 5, 6, 7], [2, 3, 4, 8, 9, 10]])
    advantages = torch.tensor([1.0, -1.0])
    targets = rows[:, -3:]
    per_row_logits = torch.cat([model(input_ids = row[None]).logits for row in rows])
    logps = per_row_logits[:, -4:-1].log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    reference = -(advantages[:, None] * (logps - logps.detach()).exp()).mean()
    params = [p for p in model.parameters() if p.requires_grad]
    expected_grads = torch.autograd.grad(reference, params)

    trainer = SimpleNamespace(
        model = model,
        beta = 0.0,
        num_generations = 2,
        use_vllm = False,
        _autocast_dtype = None,
        processing_class = SimpleNamespace(pad_token_id = 0),
        args = SimpleNamespace(
            unsloth_grpo_mini_batch = 1, unsloth_logit_chunk_multiplier = 1, steps_per_generation = 2
        ),
        accelerator = SimpleNamespace(
            unwrap_model = lambda m, **kwargs: m, num_processes = 1, scaler = None
        ),
    )
    _helper()(trainer, model)
    scores = []

    def capture_logits(module, args, output):
        scores.append(output.logits[:, -4:-1].detach())

    with model.register_forward_hook(capture_logits):
        loss = zoo.grpo_accumulated_loss(
            trainer,
            rows,
            torch.ones_like(rows),
            3,
            torch.ones_like(targets),
            advantages,
            None,
            None,
            loss_type = "grpo",
            num_items_in_batch = 6,
            num_processes = 1,
            current_gradient_accumulation_steps = 1,
            importance_sampling_level = "token",
        )[0]
    loss.backward()
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(
        torch.cat([p.grad.flatten() for p in params]),
        torch.cat([g.flatten() for g in expected_grads]),
        atol = 1e-9,
        rtol = 1e-5,
    )
    torch.testing.assert_close(torch.cat(scores), per_row_logits[:, -4:-1].detach())


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
