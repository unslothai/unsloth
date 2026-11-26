# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Hard routing load counter of unsloth/utils/moe_load.py (UNSLOTH_LOG_MOE_LOAD) on tiny OLMoE / Qwen3-MoE."""

import importlib.util
import pathlib

import pytest
import torch

pytest.importorskip("transformers")

ROOT = pathlib.Path(__file__).resolve().parents[1]
MOE_LOAD = ROOT / "unsloth" / "utils" / "moe_load.py"
REPOS = {
    "olmoe": "hf-internal-testing/tiny-random-OlmoeForCausalLM",
    "qwen3_moe": "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
}


def _load():
    spec = importlib.util.spec_from_file_location("_unsloth_moe_load", MOE_LOAD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


moe_load = _load()


def _model(family, router_logits = True):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(REPOS[family])
    config.output_router_logits = router_logits
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).float()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "gate" in name and "proj" not in name:
                param.normal_(0, 1.0)
    model.train()
    return model


def _batch(
    model,
    batch = 3,
    length = 7,
    seed = 1,
):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, model.config.vocab_size, (batch, length), generator = generator)


def _reference(
    model,
    ids,
    mask = None,
):
    """Per layer counts of top-k over the model's own router logits, and the model's own aux loss."""
    with torch.no_grad():
        out = model(input_ids = ids, attention_mask = mask, output_router_logits = True)
    top_k = model.config.num_experts_per_tok
    experts = model.config.num_experts
    keep = torch.ones(ids.numel(), dtype = torch.bool) if mask is None else mask.reshape(-1).bool()
    counts = torch.stack(
        [
            torch.bincount(logits[keep].topk(top_k, dim = -1).indices.reshape(-1), minlength = experts)
            for logits in out.router_logits
        ]
    )
    return counts, float(out.aux_loss)


def _ours(model):
    return sum(
        getattr(fn, "__self__", None).__class__.__name__ == "MoERoutingLoad"
        for fn in model._forward_hooks.values()
    )


@pytest.mark.parametrize("family", sorted(REPOS))
@pytest.mark.parametrize("padded", [False, True])
def test_counts_and_aux_match_the_model(family, padded):
    model = _model(family)
    ids = _batch(model)
    mask = None
    if padded:
        mask = torch.ones_like(ids)
        mask[0, :3] = 0
        mask[2, -2:] = 0
    expected, aux = _reference(model, ids, mask)
    counter = moe_load.MoERoutingLoad(model)
    assert len(counter) == model.config.num_hidden_layers
    assert counter.attach() and _ours(model) == 1
    counter.start()
    model(input_ids = ids, attention_mask = mask).logits.sum().backward()
    counter.stop()
    top_k = model.config.num_experts_per_tok
    real = ids.numel() if mask is None else int(mask.sum())
    assert torch.equal(counter.hard, expected)
    assert counter.tokens.tolist() == [real] * len(counter)
    assert counter.hard.sum(-1).tolist() == [real * top_k] * len(counter)
    assert torch.allclose(counter.soft.sum(-1), torch.full((len(counter),), float(real)), atol = 1e-3)
    stats = counter.stats()
    assert stats["moe_load_tokens"] == real
    assert stats["moe_load_assignments"] == real * top_k * len(counter)
    assert stats["moe_aux_loss"] == pytest.approx(aux, rel = 1e-5)
    assert stats["moe_load_max_mean"] >= 1.0 and stats["moe_soft_max_mean"] >= 1.0
    loads = [stats[f"moe_load_max_mean/layer_{i}"] for i in range(len(counter))]
    assert stats["moe_load_max_mean"] == max(loads)
    model(input_ids = ids)
    assert counter.hard.sum().item() == real * top_k * len(counter)
    counter.detach()
    assert _ours(model) == 0


@pytest.mark.parametrize("family", sorted(REPOS))
@pytest.mark.parametrize("use_reentrant", [False, True])
def test_gradient_checkpointing_counts_each_micro_batch_once(family, use_reentrant):
    model = _model(family)
    ids = _batch(model)
    expected, _ = _reference(model, ids)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs = {"use_reentrant": use_reentrant}
    )
    model.config.use_cache = False
    calls = []
    probe = model.model.layers[0].mlp.register_forward_pre_hook(lambda *a: calls.append(1))
    counter = moe_load.MoERoutingLoad(model)
    counter.attach()
    counter.start()
    embeds = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    for _ in range(2):  # two micro batches of one optimizer step
        model(inputs_embeds = embeds).logits.sum().backward()
    counter.stop()
    probe.remove()
    # The recompute really ran: the MoE block ran twice per micro batch, yet each micro batch counts once.
    assert len(calls) == 4
    assert counter.forwards == 2
    assert torch.equal(counter.hard, 2 * expected)
    assert counter.tokens.tolist() == [2 * ids.numel()] * len(counter)


def test_no_grad_and_inactive_forwards_are_not_counted():
    model = _model("olmoe")
    counter = moe_load.MoERoutingLoad(model)
    counter.attach()
    model(input_ids = _batch(model))  # attached, not started
    counter.start()
    with torch.no_grad():
        model(input_ids = _batch(model))
    counter.stop()
    assert counter.hard is None and counter.stats() == {}


def test_without_router_logits_nothing_is_counted():
    model = _model("qwen3_moe", router_logits = False)
    counter = moe_load.MoERoutingLoad(model)
    counter.attach()
    counter.start()
    model(input_ids = _batch(model))
    assert counter.stats() == {}


def test_counter_keeps_the_loss_and_grads():
    model = _model("qwen3_moe")
    ids = _batch(model)
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    base = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    model.zero_grad(set_to_none = True)
    counter = moe_load.MoERoutingLoad(model)
    counter.attach()
    counter.start()
    again = model(input_ids = ids, labels = ids).loss
    again.backward()
    assert counter.hard is not None
    assert torch.equal(loss.detach(), again.detach())
    head = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(head) == len(base) and all(torch.equal(a, b) for a, b in zip(base, head))


def test_interval_parsing():
    assert moe_load.moe_load_interval("0") == 0
    assert moe_load.moe_load_interval("") == 0
    assert moe_load.moe_load_interval("false") == 0
    assert moe_load.moe_load_interval("true") == 1
    assert moe_load.moe_load_interval("50") == 50


def test_trainer_logs_every_n_steps_beside_the_loss(tmp_path):
    from transformers import Trainer, TrainingArguments

    model = _model("olmoe")
    ids = _batch(model, batch = 16, length = 6)
    data = [{"input_ids": row, "labels": row} for row in ids]
    args = TrainingArguments(
        output_dir = str(tmp_path),
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 2,
        max_steps = 4,
        logging_steps = 1,
        report_to = [],
        use_cpu = True,
        save_strategy = "no",
        gradient_checkpointing = True,
    )
    trainer = Trainer(model = model, args = args, train_dataset = data)
    callback = moe_load.add_moe_load_callback(trainer, every = 2)
    assert trainer.callback_handler.callbacks[0] is callback
    assert moe_load.add_moe_load_callback(trainer) is callback
    trainer.train()
    logged = {h["step"]: h for h in trainer.state.log_history if "loss" in h}
    top_k = model.config.num_experts_per_tok
    layers = model.config.num_hidden_layers
    for step in (2, 4):
        # One optimizer step = 2 micro batches of 2 x 6 tokens, each counted once despite the recompute.
        assert logged[step]["moe_load_tokens"] == 2 * 2 * 6
        assert logged[step]["moe_load_assignments"] == 2 * 2 * 6 * top_k * layers
        assert logged[step]["moe_load_max_mean"] >= 1.0 and logged[step]["moe_aux_loss"] > 0
    for step in (1, 3):
        assert "moe_load_max_mean" not in logged[step]
    assert _ours(trainer.model) == 0 and callback.counter is None


def test_unsloth_trainer_init_wires_the_env_flag():
    source = (ROOT / "unsloth" / "trainer.py").read_text(encoding = "utf-8")
    assert "UNSLOTH_LOG_MOE_LOAD" in source and "add_moe_load_callback" in source
