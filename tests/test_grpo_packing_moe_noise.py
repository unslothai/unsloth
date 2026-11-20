# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO packing / PrefixGrouper self-verify on MoE models: a sparse tail from top-k routing flips (bf16 noise that
batched vs per-row unpacked forwards show too) passes, while a cross-document leak or a broad shift still fails.
Dense models keep the max-abs rule. Runs the real no-grad block, the zoo grad rewrite and a tiny Qwen3-MoE."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_grpo_packed_raw_logits_nograd as harness  # noqa: E402
from _grpo_dispatch_source import load_dispatch_helpers  # noqa: E402


noise = load_dispatch_helpers()["_unsloth_seq_packing_moe_noise"]

MOE = SimpleNamespace(num_experts = 8, num_experts_per_tok = 2)
DENSE = SimpleNamespace(hidden_size = 8)


def _owner(config):
    return SimpleNamespace(config = config)


def _jitter(
    rows = 8,
    cols = 128,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    ref = -torch.rand(rows, cols, generator = g) * 4
    result = ref + torch.randn(rows, cols, generator = g) * 0.02
    return result, ref, torch.ones(rows, cols)


def _routing_tail(result):
    # 1% of the scored tokens off by about 1 nat, as on Qwen3-30B-A3B (max 0.9-1.2).
    flat = result.view(-1)
    flat[:: max(1, flat.numel() // 10)] -= 1.1
    return result


def test_routing_tail_passes_on_moe_and_counts():
    result, ref, cm = _jitter()
    result = _routing_tail(result)
    assert float((result - ref).abs().max()) >= 0.7  # the max-abs rule fails this batch
    model = _owner(MOE)
    assert noise(model, result, ref, cm) is True
    assert model._unsloth_seq_packing_moe_noise_passes == 1


def test_dense_model_keeps_max_abs_rule():
    result, ref, cm = _jitter()
    assert noise(_owner(DENSE), _routing_tail(result), ref, cm) is False
    assert noise(SimpleNamespace(), _routing_tail(result), ref, cm) is False


def test_text_config_moe_is_detected():
    class _Config:
        def get_text_config(self):
            return MOE

    result, ref, cm = _jitter()
    assert noise(_owner(_Config()), _routing_tail(result), ref, cm) is True


@pytest.mark.parametrize(
    "bug",
    ["one_document_leaks", "every_token_shifted", "too_many_outliers", "nan"],
)
def test_real_packing_bugs_still_fail_on_moe(bug):
    result, ref, cm = _jitter()
    if bug == "one_document_leaks":
        # Contamination of a single document: most of its tokens move, the batch mean stays small.
        result[3, 10:] -= 0.4
        assert float((result - ref).abs().mean()) < 0.1
    elif bug == "every_token_shifted":
        result -= 0.15  # e.g. wrong positions: every token a little off
    elif bug == "too_many_outliers":
        result.view(-1)[::20] -= 1.1  # 5% of tokens
    else:
        result[0, 0] = float("nan")
    result.view(-1)[0] -= 1.1
    assert noise(_owner(MOE), result, ref, cm) is False


def test_small_batches_and_masked_columns():
    result, ref, cm = _jitter(rows = 2, cols = 30)
    assert noise(_owner(MOE), _routing_tail(result), ref, cm) is False  # 60 scored tokens: too few
    result, ref, cm = _jitter()
    cm[:, :64] = 0
    result[:, :64] += 50.0  # garbage outside the loss mask is ignored
    assert noise(_owner(MOE), _routing_tail(result), ref, cm) is True


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GRPO_PACKING_MOE_NOISE", "0")
    result, ref, cm = _jitter()
    assert noise(_owner(MOE), _routing_tail(result), ref, cm) is False


# The real no-grad packed block.

BIG_KEEP = 34


def _big_batch(rows = 4, prompt = 3):
    g = torch.Generator().manual_seed(rows)
    ids = torch.randint(1, harness.VOCAB, (rows, prompt + BIG_KEEP), generator = g)
    ids[0, 0] = harness.PAD_ID
    return ids


class _MoeModel(harness._Model):
    """Exact packed forward unless armed: "routing" moves one completion logit in two rows by 1.2 nats (a routing
    flip), "broad" every position after the first segment by 9 nats (contamination)."""

    def __init__(self, config):
        super().__init__(hidden_states = False)
        self.config = config
        self.mode = None

    def forward(
        self,
        input_ids = None,
        packed_seq_lengths = None,
        **kwargs,
    ):
        out = super().forward(input_ids = input_ids, packed_seq_lengths = packed_seq_lengths, **kwargs)
        if packed_seq_lengths is None or self.mode is None:
            return out
        logits = out.logits.clone()
        T = input_ids.shape[1]
        if self.mode == "routing":
            ends = packed_seq_lengths.cumsum(0)
            positions = [int(ends[0]) - 3, int(ends[2]) - 5]
            shift = 1.2
        else:
            positions = list(range(int(packed_seq_lengths[0]), T - 1))
            shift = 9.0
        for p in positions:
            logits[0, p, int(input_ids[0, p + 1])] -= shift
        out.logits = logits
        return out


@pytest.fixture
def big_batches(monkeypatch):
    monkeypatch.setattr(harness, "_batch", _big_batch)
    monkeypatch.setattr(harness, "KEEP", BIG_KEEP)


def _run(config, mode):
    model = _MoeModel(config)
    model.mode = mode
    ns, *_ = harness._run_packed_block(model = model)
    return ns, model


def test_nograd_block_packs_through_routing_noise(big_batches):
    ns, model = _run(MOE, "routing")
    assert 0.7 <= ns["_pk_diff"] < 1.5
    assert ns["_pk_use"] is True
    assert model._unsloth_seq_packing_nograd_ok is True
    assert model._unsloth_seq_packing_nograd_verified_T == ns["_pk_T"]
    assert model._unsloth_seq_packing_moe_noise_passes == 1
    assert getattr(model, "_unsloth_seq_packing_nograd_unsafe_T", None) is None


def test_nograd_block_dense_routing_like_tail_still_falls_back(big_batches):
    ns, model = _run(DENSE, "routing")
    assert ns["_pk_use"] is False
    assert model._unsloth_seq_packing_nograd_unsafe_T == ns["_pk_T"]
    assert getattr(model, "_unsloth_seq_packing_moe_noise_passes", 0) == 0


def test_nograd_block_moe_contamination_still_disables(big_batches):
    ns, model = _run(MOE, "broad")
    assert ns["_pk_diff"] >= 1.5 and ns["_pk_use"] is False
    assert model._unsloth_seq_packing_nograd_ok is False
    assert getattr(model, "_unsloth_seq_packing_moe_noise_passes", 0) == 0


def test_prefix_grouper_nograd_verify_uses_the_moe_rule():
    rl_source = harness._SOURCE.read_text(encoding = "utf-8")
    assert (
        "if _pg_diff < _pg_tol_ok() or _unsloth_seq_packing_moe_noise(\n"
        "                            unwrapped_model, _pg_a, _pg_b, _pg_cm\n"
    ) in rl_source


def test_zoo_grad_verify_rewrite():
    rl = pytest.importorskip("unsloth.models.rl_replacements")
    zoo = pytest.importorskip("unsloth_zoo.rl_replacements")
    source = inspect.getsource(zoo.grpo_accumulated_loss)
    if "_pack_diff < 7e-1" not in source:
        pytest.skip("installed unsloth_zoo has no grad packing verify")
    patched = rl._patch_grpo_accumulated_loss_moe_noise(source)
    assert (
        "_unsloth_seq_packing_moe_noise(unwrapped_model, _pack_result, _pack_ref, _pack_cm)"
        in patched
    )
    if "_pg_diff < _pg_tol_ok()" in source:
        assert "_unsloth_seq_packing_moe_noise(unwrapped_model, _pg_a, _pg_b, _pg_cm)" in patched
    compile(patched, "<patched grpo_accumulated_loss>", "exec")
    assert rl._patch_grpo_accumulated_loss_moe_noise(patched) == patched
    items = [i for i in rl.RL_PRE_ITEMS["grpo_trainer"] if isinstance(i, str)]
    assert any("def _unsloth_seq_packing_moe_noise(" in i for i in items)
    assert any("_unsloth_seq_packing_moe_noise(unwrapped_model, _pack_result" in i for i in items)


# A real (tiny, random) Qwen3-MoE: a cross-document leak fails, last-layer routing flips pass.


def _tiny_qwen3_moe():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3MoeConfig(
        vocab_size = 96,
        hidden_size = 64,
        intermediate_size = 64,
        moe_intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        num_experts = 8,
        num_experts_per_tok = 2,
        max_position_embeddings = 256,
    )
    torch.manual_seed(0)
    model = transformers.Qwen3MoeForCausalLM(config).eval()
    with torch.no_grad():
        model.lm_head.weight.mul_(40.0)  # peaked next-token distributions, like a trained model
    return model


def _doc_logprobs(
    model,
    docs,
    packed_leak = False,
    flip = (),
):
    """Per-token logprobs of each document. packed_leak runs all documents as one causal row (positions reset per
    document, no block mask), i.e. what a backend ignoring packed_seq_lengths computes."""
    handle = None
    if flip:

        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            g = torch.Generator().manual_seed(1)
            for t in flip:
                hidden[0, t] += torch.randn(hidden.shape[-1], generator = g) * 4.0
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

        handle = model.model.layers[-1].mlp.register_forward_hook(hook)
    try:
        with torch.no_grad():
            if packed_leak:
                ids = torch.cat(docs).unsqueeze(0)
                pos = torch.cat([torch.arange(len(d)) for d in docs]).unsqueeze(0)
                lp = torch.log_softmax(model(input_ids = ids, position_ids = pos).logits.float(), -1)[0]
                out, start = [], 0
                for d in docs:
                    out.append(lp[start : start + len(d) - 1].gather(-1, d[1:, None])[:, 0])
                    start += len(d)
                return torch.stack(out)
            out = []
            for d in docs:
                lp = torch.log_softmax(model(input_ids = d[None]).logits.float(), -1)[0]
                out.append(lp[:-1].gather(-1, d[1:, None])[:, 0])
            return torch.stack(out)
    finally:
        if handle is not None:
            handle.remove()


def test_tiny_qwen3_moe_leak_fails_and_routing_flips_pass():
    model = _tiny_qwen3_moe()
    g = torch.Generator().manual_seed(0)
    docs = [torch.randint(0, 96, (48,), generator = g) for _ in range(4)]
    ref = _doc_logprobs(model, docs)
    cm = torch.ones_like(ref)

    leak = _doc_logprobs(model, docs, packed_leak = True)
    assert float((leak - ref).abs().max()) >= 0.7
    assert noise(model, leak, ref, cm) is False

    # One routing flip each in two documents' last-layer MoE output (per-row run, same rows).
    flipped = torch.cat(
        [_doc_logprobs(model, [docs[0]], flip = (20,)), _doc_logprobs(model, [docs[1]], flip = (30,))]
    )
    flipped = torch.cat([flipped, ref[2:]])
    assert float((flipped - ref).abs().max()) >= 0.7
    assert noise(model, flipped, ref, cm) is True


# Packed forwards on the transformers path get the varlen attention wrapper SFT packing installs.


def _boundary_namespace(
    monkeypatch,
    calls,
    blocker = None,
):
    import ast
    import os
    import types

    tree = ast.parse(harness._SOURCE.read_text(encoding = "utf-8"))
    names = ("_unsloth_grpo_refuse_boundary_blind_packing", "_unsloth_grpo_enable_packed_varlen")
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"os": os, "torch": torch, "UNSLOTH_ENABLE_LOGGING": False}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(harness._SOURCE), "exec"), ns)
    ns["_unsloth_grpo_packing_leaks"] = lambda self, unwrapped: False
    packing = types.ModuleType("unsloth.utils.packing")
    packing.packing_boundary_blocker = lambda model, model_types: blocker
    packing._enable_hf_packed_attention = lambda model = None, trainer = None: calls.append(
        (model, trainer)
    )
    for name in ("unsloth", "unsloth.utils"):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = []
            monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setitem(sys.modules, "unsloth.utils.packing", packing)
    return ns["_unsloth_grpo_refuse_boundary_blind_packing"]


def _grpo_trainer():
    return SimpleNamespace(
        accelerator = SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m)
    )


@pytest.mark.parametrize("env", [None, "0"], ids = ["packing-on", "packing-off"])
def test_packing_enables_hf_varlen_attention_once(monkeypatch, env):
    if env is None:
        monkeypatch.delenv("UNSLOTH_GRPO_SEQ_PACKING", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", env)
    calls = []
    refuse = _boundary_namespace(monkeypatch, calls)
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(model_type = "qwen3_moe")
    trainer = _grpo_trainer()
    refuse(trainer, model)
    refuse(trainer, model)
    assert calls == ([(model, trainer)] if env is None else [])


def test_blocked_packing_does_not_enable_varlen_attention(monkeypatch):
    calls = []
    refuse = _boundary_namespace(monkeypatch, calls, blocker = "RecurrentGemmaRglru")
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(model_type = "recurrent_gemma")
    refuse(_grpo_trainer(), model)
    assert calls == []
    assert model._unsloth_seq_packing_nograd_ok is False
