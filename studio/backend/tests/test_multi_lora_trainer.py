# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Batched multi-LoRA worker on a tiny random Qwen3 (CPU, PyTorch fallback of the kernels)."""

from __future__ import annotations

import copy
import json
import math
import os
import queue
import types
from pathlib import Path

import sys

import pytest

# The repo root (pytest pythonpath) shadows PYTHONPATH.
if "unsloth" not in sys.modules:
    for _entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)[::-1]:
        if _entry and os.path.isfile(os.path.join(_entry, "unsloth", "kernels", "multi_lora.py")):
            sys.path.insert(0, _entry)

try:
    import torch
    from unsloth.kernels import multi_lora as ml
except Exception as exc:  # noqa: BLE001 - no multi-LoRA kernels or no importable unsloth here
    pytest.skip(f"unsloth.kernels.multi_lora unavailable: {exc}", allow_module_level = True)

from datasets import Dataset
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, ProcessorMixin, Qwen3Config, Qwen3ForCausalLM

from core.training import multi_lora_trainer as mlt

WORDS = [f"w{i}" for i in range(56)]


def _tokenizer():
    vocab = {"[PAD]": 0, "[EOS]": 1, "[U]": 2, "[B]": 3}
    vocab.update({w: i + 4 for i, w in enumerate(WORDS)})
    tok = Tokenizer(models.WordLevel(vocab = vocab, unk_token = "[PAD]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object = tok, pad_token = "[PAD]", eos_token = "[EOS]")


def _base():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        max_position_embeddings = 128,
        use_cache = False,
        tie_word_embeddings = False,
    )
    return Qwen3ForCausalLM(config).float()


def _qwen3_5(moe: bool):
    """Tiny Qwen3.5 text model: one Gated DeltaNet (linear attention) layer, one full attention layer; ``moe``
    swaps the MLPs for fused experts plus a shared expert, as in Qwen3.6-35B-A3B."""
    import transformers

    torch.manual_seed(0)
    common = dict(
        vocab_size = 64,
        hidden_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        max_position_embeddings = 128,
        use_cache = False,
        tie_word_embeddings = False,
        layer_types = ["linear_attention", "full_attention"],
        linear_num_key_heads = 2,
        linear_num_value_heads = 4,
        linear_key_head_dim = 8,
        linear_value_head_dim = 8,
        linear_conv_kernel_dim = 4,
    )
    if moe:
        config = transformers.Qwen3_5MoeTextConfig(
            **common,
            num_experts = 4,
            num_experts_per_tok = 2,
            moe_intermediate_size = 16,
            shared_expert_intermediate_size = 16,
        )
        return transformers.Qwen3_5MoeForCausalLM(config).float()
    config = transformers.Qwen3_5TextConfig(**common, intermediate_size = 64)
    return transformers.Qwen3_5ForCausalLM(config).float()


def _kind_targets(kind):
    # Lone runs train fused experts for the default targets; a bare "all-linear" string reaches no experts there.
    return {"target_modules": []} if kind == "moe" else {}


def _kind_base(kind):
    if kind == "dense":
        return _base()
    pytest.importorskip("transformers", minversion = "5.2")
    return _qwen3_5(moe = kind == "moe")


def _texts(n, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        k = int(torch.randint(4, 12, (1,), generator = g))
        out.append(
            " ".join(WORDS[int(i)] for i in torch.randint(0, 56, (k,), generator = g)) + " [EOS]"
        )
    return out


MEMBERS = [
    dict(run_id = "job_a", rows = 6, batch_size = 2, num_epochs = 1, lr = 1e-2, r = 4, alpha = 8, seed = 11),
    dict(run_id = "job_b", rows = 12, batch_size = 2, num_epochs = 2, lr = 5e-3, r = 8, alpha = 16, seed = 22),
    dict(
        run_id = "job_c",
        rows = 9,
        batch_size = 3,
        num_epochs = 1,
        lr = 2e-2,
        r = 2,
        alpha = 4,
        seed = 33,
        max_steps = 4,
    ),
]
EXPECTED_STEPS = {"job_a": 2, "job_b": 6, "job_c": 4}


def _member_config(m):
    return {
        "model_name": "tiny/qwen3",
        "local_datasets": [m["run_id"]],
        "format_type": "",
        "learning_rate": m["lr"],
        "num_epochs": m["num_epochs"],
        "batch_size": m["batch_size"],
        "lora_r": m["r"],
        "lora_alpha": m["alpha"],
        "random_seed": m["seed"],
        "max_steps": m.get("max_steps", 0),
        "optim": "adamw_torch",
        "warmup_steps": m.get("warmup_steps", 0),
        "weight_decay": m.get("weight_decay", 0.0),
        "gradient_accumulation_steps": 2,
        "packing": m.get("packing", False),
    }


class _Events:
    def __init__(self, on_event = None):
        self.items = []
        self.on_event = on_event

    def put(self, event):
        self.items.append(event)
        if self.on_event is not None:
            self.on_event(event)


def _run(
    monkeypatch,
    members,
    on_event = None,
    trainer_attrs = None,
    datasets = None,
    model_attrs = None,
    base = None,
    wrap_tokenizer = None,
    **extra,
):
    base = _base() if base is None else base
    for name, value in (model_attrs or {}).items():
        setattr(base, name, value)
    pristine = copy.deepcopy(base)
    tokenizer = _tokenizer()
    trainer = types.SimpleNamespace(
        model = None,
        tokenizer = None,
        should_stop = False,
        save_on_stop = True,
        load_in_4bit = False,
        is_cpt = False,
        **(trainer_attrs or {}),
    )
    data = {
        m["run_id"]: {
            "dataset": Dataset.from_dict({"text": _texts(m["rows"], m["seed"])}),
            "final_format": "",
        }
        for m in members
    }
    data.update(datasets or {})
    inboxes = []

    def load_base(trainer, config, hf_token, event_queue):
        trainer.model = base
        trainer.tokenizer = tokenizer if wrap_tokenizer is None else wrap_tokenizer(tokenizer)
        return False

    def stop_reader(stop_queue, inbox, trainer):
        inboxes.append(inbox)

    monkeypatch.setattr(mlt, "_new_trainer", lambda: trainer)
    monkeypatch.setattr(mlt, "_detect_model", lambda trainer, config, hf_token: None)
    monkeypatch.setattr(mlt, "_format_dataset", lambda t, cfg, tok: data[cfg["local_datasets"][0]])
    monkeypatch.setattr(mlt, "_load_base", load_base)
    monkeypatch.setattr(mlt, "_start_stop_reader", stop_reader)
    configs = [_member_config(m) for m in members]
    config = {
        **configs[0],
        "max_seq_length": 64,
        "gradient_checkpointing": "unsloth",
        "target_modules": ["all-linear"],
        "batch_members": [{"run_id": m["run_id"], "config": c} for m, c in zip(members, configs)],
        **extra,
    }
    events = _Events(None if on_event is None else lambda e: on_event(e, inboxes[0]))
    mlt.run_multi_lora_training(events, queue.Queue(), config)
    return events.items, base, pristine


def _member_events(events, run_id):
    return [e for e in events if e.get("member_run_id") == run_id]


def _adapter_logits(model, slot, input_ids):
    ml.set_multi_lora_batch(
        model, torch.full((input_ids.shape[0],), slot), seq_len = input_ids.shape[1]
    )
    with torch.no_grad():
        return model(input_ids = input_ids, use_cache = False).logits


@pytest.fixture(scope = "module")
def batched_run(tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path_factory.mktemp("studio_home")))
    try:
        yield _run(monkeypatch, MEMBERS)
    finally:
        monkeypatch.undo()


def test_each_member_reports_and_exports_on_its_own(batched_run):
    events, model, pristine = batched_run
    assert [e["type"] for e in events if "member_run_id" not in e][-1] == "complete"
    dirs = set()
    for m in MEMBERS:
        mine = _member_events(events, m["run_id"])
        kinds = [e["type"] for e in mine]
        assert kinds[0] == "output_dir" and kinds[-1] == "complete"
        assert set(kinds[1:-1]) == {"progress"}
        steps = [e["step"] for e in mine if e["type"] == "progress"]
        assert steps == list(range(EXPECTED_STEPS[m["run_id"]] + 1))
        tokens = [e["num_tokens"] for e in mine if e["type"] == "progress"]
        assert tokens == sorted(tokens) and tokens[-1] > 0
        assert all(
            e["total_steps"] == EXPECTED_STEPS[m["run_id"]] for e in mine if e["type"] == "progress"
        )
        out = mine[0]["output_dir"]
        assert mine[-1]["output_dir"] == out
        dirs.add(out)
        config = json.load(open(os.path.join(out, "adapter_config.json")))
        assert config["r"] == m["r"] and config["lora_alpha"] == m["alpha"]
        assert config["unsloth_training_method"] == "lora"
        assert os.path.exists(os.path.join(out, "tokenizer.json"))
    assert len(dirs) == len(MEMBERS)


def test_each_step_reports_the_rate_it_trained_with(batched_run):
    events, _model, _pristine = batched_run
    for m in MEMBERS:
        total = EXPECTED_STEPS[m["run_id"]]
        reported = [
            e["learning_rate"]
            for e in _member_events(events, m["run_id"])
            if e["type"] == "progress" and e["step"] > 0
        ]
        # Linear decay without warmup, read before each step: lr, then lr * (1 - 1/total), ...
        assert reported == pytest.approx([m["lr"] * (1 - k / total) for k in range(total)])


def test_exported_adapter_reloads_with_peft_and_matches(batched_run):
    from peft import PeftModel

    events, model, pristine = batched_run
    model.eval()
    input_ids = torch.randint(4, 60, (2, 10), generator = torch.Generator().manual_seed(5))
    for slot, m in enumerate(MEMBERS):
        out = _member_events(events, m["run_id"])[0]["output_dir"]
        state = load_file(os.path.join(out, "adapter_model.safetensors"))
        assert any(v.abs().sum() > 0 for k, v in state.items() if "lora_B" in k)
        peft_model = PeftModel.from_pretrained(copy.deepcopy(pristine), out).eval()
        with torch.no_grad():
            expected = peft_model(input_ids = input_ids).logits
        torch.testing.assert_close(
            _adapter_logits(model, slot, input_ids), expected, atol = 1e-5, rtol = 1e-5
        )


def test_finished_member_gets_no_more_updates(batched_run):
    events, model, pristine = batched_run
    out = _member_events(events, "job_a")[0]["output_dir"]
    exported = load_file(os.path.join(out, "adapter_model.safetensors"))
    current = ml.adapter_state_dict(model, 0)
    assert exported.keys() == current.keys()
    for key in exported:
        torch.testing.assert_close(current[key], exported[key], atol = 0, rtol = 0)


def test_stop_member_stops_only_that_member(monkeypatch):
    def on_event(event, inbox):
        if (
            event.get("member_run_id") == "job_b"
            and event["type"] == "progress"
            and event["step"] == 2
        ):
            inbox.put({"type": "stop_member", "run_id": "job_b", "save": True})

    events, model, pristine = _run(monkeypatch, MEMBERS, on_event)
    b = _member_events(events, "job_b")
    assert b[-1]["type"] == "stopped" and b[-1]["saved"] is True
    assert max(e["step"] for e in b if e["type"] == "progress") == 2
    assert os.path.exists(os.path.join(b[-1]["output_dir"], "adapter_model.safetensors"))
    for run_id in ("job_a", "job_c"):
        mine = _member_events(events, run_id)
        assert mine[-1]["type"] == "complete"
        assert max(e["step"] for e in mine if e["type"] == "progress") == EXPECTED_STEPS[run_id]
    assert events[-1]["type"] == "complete" and "member_run_id" not in events[-1]


def test_seed_init_is_independent_of_batch_order():
    first, second = _base(), _base()
    ml.inject_multi_lora(first, ["q_proj", "down_proj"], 2, r = [4, 8], lora_alpha = [8, 16])
    ml.inject_multi_lora(second, ["q_proj", "down_proj"], 2, r = [8, 4], lora_alpha = [16, 8])
    mlt._seed_adapter(first, 0, 11)
    mlt._seed_adapter(first, 1, 22)
    mlt._seed_adapter(second, 0, 22)
    mlt._seed_adapter(second, 1, 11)
    a, b = ml.adapter_state_dict(first, 0), ml.adapter_state_dict(second, 1)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol = 0, rtol = 0)
    assert all(v.abs().sum() == 0 for k, v in a.items() if "lora_B" in k)


def test_completion_masking_masks_the_prompt():
    tokenizer = _tokenizer()
    tokenizer._unsloth_input_part = "[U]"
    tokenizer._unsloth_output_part = "[B]"
    info = {
        "dataset": Dataset.from_dict({"text": ["[U] w1 w2 w3 [B] w4 w5 [EOS]"] * 3}),
        "final_format": "",
    }
    tok, rows = mlt._tokenize_member(
        info, tokenizer, {"train_on_completions": True}, "tiny/qwen3", 64
    )
    assert tok is tokenizer
    ids, labels = rows[0]
    response = ids.index(tokenizer.convert_tokens_to_ids("[B]")) + 1
    assert all(t == -100 for t in labels[:response])
    assert labels[response:] == ids[response:]


def test_batch_trains_through_sft_trainer_and_saves_checkpoints(monkeypatch, tmp_path):
    from trl import SFTTrainer

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    built = []
    original = mlt._build_sft_trainer

    def spy(*args, **kwargs):
        built.append(original(*args, **kwargs))
        return built[-1]

    monkeypatch.setattr(mlt, "_build_sft_trainer", spy)
    events, model, pristine = _run(monkeypatch, MEMBERS[:2], save_steps = 2)
    assert isinstance(built[0], SFTTrainer) and built[0].state.global_step == 6
    found = {}
    for m in MEMBERS[:2]:
        out = _member_events(events, m["run_id"])[0]["output_dir"]
        found[m["run_id"]] = sorted(p.name for p in Path(out).glob("checkpoint-*"))
        last = os.path.join(out, found[m["run_id"]][-1], "adapter_model.safetensors")
        final = load_file(os.path.join(out, "adapter_model.safetensors"))
        for key, value in load_file(last).items():
            torch.testing.assert_close(value, final[key], atol = 0, rtol = 0)
    assert found == {
        "job_a": ["checkpoint-2"],
        "job_b": ["checkpoint-2", "checkpoint-4", "checkpoint-6"],
    }


def test_gradients_are_clipped_per_member_every_step(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    seen = []
    real = ml.clip_grad_norm_per_adapter_

    def spy(model, max_norm):
        norms = real(model, max_norm)
        seen.append((max_norm, norms.clone()))
        return norms

    monkeypatch.setattr(ml, "clip_grad_norm_per_adapter_", spy)
    events, _model, _pristine = _run(monkeypatch, MEMBERS[:2], max_grad_norm = 0.01)
    assert len(seen) == EXPECTED_STEPS["job_b"]
    assert all(max_norm == 0.01 for max_norm, _ in seen)
    # Each member reports its own pre-clip norm, not one shared global norm.
    a = [
        e["grad_norm"]
        for e in _member_events(events, "job_a")
        if e["type"] == "progress" and e["step"]
    ]
    b = [
        e["grad_norm"]
        for e in _member_events(events, "job_b")
        if e["type"] == "progress" and e["step"]
    ]
    assert a == pytest.approx([float(n[0]) for _, n in seen[:2]])
    assert b == pytest.approx([float(n[1]) for _, n in seen])


def test_each_member_keeps_its_own_weight_decay_and_warmup(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    members = [
        {**MEMBERS[0], "weight_decay": 0.1, "warmup_steps": 1},
        {**MEMBERS[1], "weight_decay": 0.0, "warmup_steps": 0},
    ]
    built = []
    original = mlt._build_sft_trainer
    monkeypatch.setattr(
        mlt, "_build_sft_trainer", lambda *a, **k: built.append(original(*a, **k)) or built[-1]
    )
    events, _model, _pristine = _run(monkeypatch, members)
    assert [g["weight_decay"] for g in built[0].optimizer.param_groups] == [0.1, 0.0]
    for m in members:
        total = EXPECTED_STEPS[m["run_id"]]
        reported = [
            e["learning_rate"]
            for e in _member_events(events, m["run_id"])
            if e["type"] == "progress" and e["step"]
        ]
        warmup = m["warmup_steps"]
        # transformers' linear schedule: ramp over the member's own warmup, then decay over its own length.
        expected = [
            m["lr"] * (k / warmup if k < warmup else (total - k) / (total - warmup))
            for k in range(total)
        ]
        assert reported == pytest.approx(expected)


def test_targets_follow_the_single_job_rules():
    model = _base()
    every_linear = mlt._resolve_targets(model, ["all-linear"])
    assert (
        "q_proj" in every_linear and "down_proj" in every_linear and "lm_head" not in every_linear
    )
    # Explicit names beside "all-linear" win, as UnslothTrainer.prepare_model_for_training does.
    assert mlt._resolve_targets(model, ["all-linear", "q_proj"]) == ["q_proj"]
    assert mlt._resolve_targets(model, None) == mlt._DEFAULT_TARGETS


def test_a_text_diffusion_model_never_trains_batched(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    events, _model, _pristine = _run(
        monkeypatch, MEMBERS[:2], trainer_attrs = {"diffusion_lm_profile": "dream"}
    )
    assert [e["type"] for e in events if e.get("member_run_id")] == ["error", "error"]
    assert "text-diffusion" in events[-1]["error"]


def test_an_existing_lora_adapter_never_trains_batched(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    events, _model, _pristine = _run(
        monkeypatch, MEMBERS[:2], model_attrs = {"peft_config": {"default": object()}}
    )
    assert [e["type"] for e in events if e.get("member_run_id")] == ["error", "error"]
    assert "existing LoRA adapter" in events[-1]["error"]


def test_a_moe_model_never_trains_batched(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    config = copy.deepcopy(_base().config)
    config.num_experts = 8
    events, _model, _pristine = _run(monkeypatch, MEMBERS[:2], model_attrs = {"config": config})
    assert [e["type"] for e in events if e.get("member_run_id")] == ["error", "error"]
    assert "MoE" in events[-1]["error"]


def test_pretokenized_assistant_masks_become_labels():
    ids = [[5, 6, 7, 8], [9, 10, 11]]
    info = {
        "dataset": Dataset.from_dict(
            {"input_ids": ids, "assistant_masks": [[0, 0, 1, 1], [0, 1, 1]]}
        ),
        "final_format": "pretokenized",
    }
    _tok, rows = mlt._tokenize_member(info, _tokenizer(), {}, "tiny/qwen3", 64)
    assert [labels for _, labels in rows] == [[-100, -100, 7, 8], [-100, 10, 11]]


def test_a_pretokenized_mask_of_the_wrong_length_is_refused():
    info = {
        "dataset": Dataset.from_dict({"input_ids": [[5, 6, 7, 8]], "assistant_masks": [[0, 1]]}),
        "final_format": "pretokenized",
    }
    with pytest.raises(ValueError, match = "assistant_masks"):
        mlt._tokenize_member(info, _tokenizer(), {}, "tiny/qwen3", 64)


def test_a_pretokenized_token_outside_the_vocabulary_is_refused():
    info = {
        "dataset": Dataset.from_dict({"input_ids": [[5, 6, 900]]}),
        "final_format": "pretokenized",
    }
    with pytest.raises(ValueError, match = "outside this model's vocabulary"):
        mlt._tokenize_member(info, _tokenizer(), {}, "tiny/qwen3", 64, vocab_size = 64)


def test_one_members_bad_dataset_fails_only_that_member(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    bad = {
        "dataset": Dataset.from_dict({"input_ids": [[5, 6, 900]] * 4}),
        "final_format": "pretokenized",
    }
    events, _model, _pristine = _run(monkeypatch, MEMBERS[:2], datasets = {"job_a": bad})
    a, b = _member_events(events, "job_a"), _member_events(events, "job_b")
    assert [e["type"] for e in a] == ["error"] and "vocabulary" in a[0]["error"]
    assert b[-1]["type"] == "complete"
    assert events[-1]["type"] == "complete" and "member_run_id" not in events[-1]


@pytest.mark.parametrize("collator_masks_labels", [True, False])
def test_pretokenized_labels_and_assistant_masks(monkeypatch, collator_masks_labels):
    import utils.datasets.pretokenized as pretokenized

    monkeypatch.setattr(pretokenized, "_collator_masks_user_labels", lambda: collator_masks_labels)
    info = {
        "dataset": Dataset.from_dict(
            {
                "input_ids": [[5, 6, 7, 8]],
                "labels": [[5, 6, 7, 8]],
                "assistant_masks": [[0, 0, 1, 1]],
            }
        ),
        "final_format": "pretokenized",
    }
    _tok, rows = mlt._tokenize_member(info, _tokenizer(), {}, "tiny/qwen3", 64)
    # The single-job collator masks user labels only before TRL 1.7.
    assert rows[0][1] == ([-100, -100, 7, 8] if collator_masks_labels else [5, 6, 7, 8])


def test_batched_member_trains_as_it_would_alone(batched_run, monkeypatch):
    events, model, pristine = batched_run
    batched = load_file(
        os.path.join(_member_events(events, "job_b")[0]["output_dir"], "adapter_model.safetensors")
    )
    alone_events, _, _ = _run(monkeypatch, [MEMBERS[1]])
    alone = load_file(
        os.path.join(
            _member_events(alone_events, "job_b")[0]["output_dir"], "adapter_model.safetensors"
        )
    )
    batched_loss = [e["loss"] for e in _member_events(events, "job_b") if e["type"] == "progress"]
    alone_loss = [
        e["loss"] for e in _member_events(alone_events, "job_b") if e["type"] == "progress"
    ]
    assert batched_loss[1:] == pytest.approx(alone_loss[1:], rel = 1e-4)
    for key in alone:
        torch.testing.assert_close(batched[key], alone[key], atol = 1e-5, rtol = 1e-4)


@pytest.mark.parametrize("kind", ["dense", "hybrid", "moe"])
def test_a_neighbour_never_moves_a_members_training(monkeypatch, tmp_path, kind):
    """Job B batched beside A, then beside a different A (data, rate, rank, seed): B's losses and adapter match."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    other = dict(MEMBERS[0], rows = 10, lr = 3e-2, r = 2, alpha = 2, seed = 99)
    runs = []
    for neighbour in (MEMBERS[0], other):
        events, _, _ = _run(
            monkeypatch, [neighbour, MEMBERS[1]], base = _kind_base(kind), **_kind_targets(kind)
        )
        mine = _member_events(events, "job_b")
        losses = [e["loss"] for e in mine if e["type"] == "progress"]
        runs.append(
            (losses, load_file(os.path.join(mine[0]["output_dir"], "adapter_model.safetensors")))
        )
    (loss_1, adapter_1), (loss_2, adapter_2) = runs
    assert len(loss_1) == EXPECTED_STEPS["job_b"] + 1
    assert loss_2 == pytest.approx(loss_1, rel = 1e-5)
    assert adapter_1.keys() == adapter_2.keys()
    for key in adapter_1:
        torch.testing.assert_close(adapter_2[key], adapter_1[key], atol = 1e-6, rtol = 1e-5)


@pytest.mark.parametrize("kind", ["hybrid", "moe"])
def test_a_hybrid_or_moe_member_trains_as_it_would_alone(monkeypatch, tmp_path, kind):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    batched, _, _ = _run(monkeypatch, MEMBERS[:2], base = _kind_base(kind), **_kind_targets(kind))
    alone, _, _ = _run(monkeypatch, [MEMBERS[1]], base = _kind_base(kind), **_kind_targets(kind))
    out = {}
    for name, events in (("batched", batched), ("alone", alone)):
        mine = _member_events(events, "job_b")
        assert mine[-1]["type"] == "complete"
        out[name] = (
            [e["loss"] for e in mine if e["type"] == "progress"],
            load_file(os.path.join(mine[0]["output_dir"], "adapter_model.safetensors")),
            json.load(open(os.path.join(mine[0]["output_dir"], "adapter_config.json"))),
        )
    assert out["batched"][0][1:] == pytest.approx(out["alone"][0][1:], rel = 1e-4)
    for key, value in out["alone"][1].items():
        torch.testing.assert_close(out["batched"][1][key], value, atol = 1e-5, rtol = 1e-4)
    # Every adapter trained, dense and expert alike (injecting experts must not freeze the dense ones).
    trained = {
        k: v.abs().max() > 0 for k, v in out["batched"][1].items() if k.endswith("lora_B.weight")
    }
    assert trained and all(trained.values()), [k for k, ok in trained.items() if not ok][:3]
    if kind == "moe":
        config = out["batched"][2]
        assert sorted(config["target_parameters"]) == [
            "mlp.experts.down_proj",
            "mlp.experts.gate_up_proj",
        ]
        experts = [
            k for k in out["batched"][1] if ".mlp.experts." in k and k.endswith("lora_B.weight")
        ]
        assert len(experts) == 4 and all(out["batched"][1][k].abs().max() > 0 for k in experts)
    else:
        # Gated DeltaNet projections train exactly when all-linear asks for them, as alone.
        assert any(".linear_attn.in_proj_qkv." in k for k in out["batched"][1])


def test_an_allowlisted_moe_with_a_router_aux_loss_trains_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    pytest.importorskip("transformers", minversion = "5.2")
    base = _qwen3_5(moe = True)
    base.config.output_router_logits = True
    events, _, _ = _run(monkeypatch, MEMBERS[:2], base = base, target_modules = [])
    assert [e["type"] for e in events if e.get("member_run_id")] == ["error", "error"]
    assert "auxiliary loss" in events[-1]["error"]


def test_a_state_space_model_trains_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    base = _base()
    base.model.layers[0].mixer = type("MambaMixer", (torch.nn.Module,), {})()
    events, _, _ = _run(monkeypatch, MEMBERS[:2], base = base)
    assert [e["type"] for e in events if e.get("member_run_id")] == ["error", "error"]
    assert "state-space" in events[-1]["error"]


PACKED = [
    dict(
        run_id = "job_a",
        rows = 12,
        batch_size = 2,
        num_epochs = 2,
        lr = 1e-2,
        r = 4,
        alpha = 8,
        seed = 11,
        packing = True,
    ),
    dict(
        run_id = "job_b",
        rows = 16,
        batch_size = 1,
        num_epochs = 1,
        lr = 5e-3,
        r = 8,
        alpha = 16,
        seed = 22,
        packing = True,
    ),
    dict(run_id = "job_c", rows = 7, batch_size = 2, num_epochs = 1, lr = 2e-2, r = 2, alpha = 4, seed = 33),
]


def _final(events, run_id):
    mine = _member_events(events, run_id)
    losses = [e["loss"] for e in mine if e["type"] == "progress" and e["step"] > 0]
    state = load_file(os.path.join(mine[0]["output_dir"], "adapter_model.safetensors"))
    return losses, state


def _spy_packed(monkeypatch):
    seen = []
    real = mlt._collate_packed

    def spy(rows, slots, lengths):
        batch = real(rows, slots, lengths)
        seen.append(batch)
        return batch

    monkeypatch.setattr(mlt, "_collate_packed", spy)
    return seen


def test_packing_members_never_share_a_document_or_see_a_neighbour(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    seen = _spy_packed(monkeypatch)
    events, _, _ = _run(monkeypatch, PACKED, max_seq_length = 24)
    assert seen
    for batch in seen:
        ids, pos, lengths = (
            batch["adapter_ids"][0],
            batch["position_ids"][0],
            batch["packed_seq_lengths"],
        )
        starts = (pos == 0).nonzero().flatten()
        assert starts.tolist() == (torch.cumsum(lengths, 0) - lengths).tolist()
        # One adapter per document, and the label opening each document is never trained.
        for start, n in zip(starts.tolist(), lengths.tolist()):
            assert ids[start : start + n].unique().numel() == 1
        assert (batch["labels"][0, starts[1:]] == -100).all()
    rows_a = math.ceil(12 / 2)  # unpacked micro-batches per epoch
    steps_a = max(e["step"] for e in _member_events(events, "job_a") if e["type"] == "progress")
    assert steps_a < 2 * math.ceil(rows_a / 2)
    neighbour = [
        dict(PACKED[0]),
        dict(PACKED[1], rows = 9, seed = 99, lr = 3e-2),
        dict(PACKED[2], packing = True),
    ]
    other, _, _ = _run(monkeypatch, neighbour, max_seq_length = 24)
    alone, _, _ = _run(monkeypatch, [PACKED[0]], max_seq_length = 24)
    losses, state = _final(events, "job_a")
    for run in (other, alone):
        run_losses, run_state = _final(run, "job_a")
        assert run_losses == pytest.approx(losses, rel = 1e-4)
        # Float noise from the shared row's size only; a leaked neighbour moves the losses ~1e-3.
        for key in state:
            torch.testing.assert_close(run_state[key], state[key], atol = 1e-4, rtol = 1e-3)


def test_an_unpacked_member_beside_packed_ones_trains_as_it_would_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    mixed, _, _ = _run(monkeypatch, PACKED, max_seq_length = 24)
    alone, _, _ = _run(monkeypatch, [PACKED[2]], max_seq_length = 24)
    losses, state = _final(mixed, "job_c")
    alone_losses, alone_state = _final(alone, "job_c")
    assert losses == pytest.approx(alone_losses, rel = 1e-4)
    for key in state:
        torch.testing.assert_close(state[key], alone_state[key], atol = 1e-5, rtol = 1e-4)


def test_packing_is_ignored_where_the_single_job_ignores_it(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mlt, "_packs", lambda model: False)
    seen = _spy_packed(monkeypatch)
    events, _, _ = _run(monkeypatch, PACKED[:1], max_seq_length = 24)
    assert seen == []
    steps = max(e["step"] for e in _member_events(events, "job_a") if e["type"] == "progress")
    assert steps == 2 * math.ceil(math.ceil(12 / 2) / 2)


@pytest.mark.parametrize("varlen", [True, False])
def test_a_hybrid_packs_only_with_the_varlen_kernels(monkeypatch, varlen):
    transformers = pytest.importorskip("transformers", minversion = "5.2")
    from unsloth import trainer as unsloth_trainer

    config = transformers.Qwen3_5TextConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        layer_types = ["linear_attention", "full_attention"],
        linear_num_key_heads = 2,
        linear_num_value_heads = 4,
        linear_key_head_dim = 8,
        linear_value_head_dim = 8,
    )
    model = transformers.Qwen3_5ForCausalLM(config)
    calls = []

    def patch(m):
        calls.append(m)
        return varlen

    monkeypatch.setattr(unsloth_trainer, "patch_hybrid_linear_attention_varlen", patch)
    assert mlt._packs(model) is varlen and calls == [model]
    assert mlt._packs(_base()) is True


def test_pack_rows_keeps_each_documents_labels_with_its_tokens():
    rows = [([i] * n, [-100] + [i] * (n - 1)) for i, n in enumerate([5, 9, 3, 7, 4], start = 1)]
    packed, lengths = mlt._pack_rows(rows, 12)
    assert all(len(ids) <= 12 and sum(n) == len(ids) for (ids, _), n in zip(packed, lengths))
    docs = sorted(
        (ids[s : s + n], lab[s : s + n])
        for (ids, lab), ns in zip(packed, lengths)
        for s, n in zip([sum(ns[:k]) for k in range(len(ns))], ns)
    )
    assert docs == sorted(rows)


def _force_member_loss_scale(monkeypatch, poison = None):
    """fp16's per-member loss scale on this CPU run; ``poison`` = (slot, call) overflows that member there."""
    monkeypatch.setattr(mlt, "_fp16_loss_scale", lambda fp16, n: mlt._MemberLossScale(n))
    if poison is None:
        return
    real, calls = mlt._MemberLossScale.unscale_, []

    def unscale_(self, slot, params):
        if slot == poison[0]:
            calls.append(slot)
            if len(calls) == poison[1]:
                for p in params:
                    if p.grad is not None:
                        p.grad.fill_(float("inf"))
        return real(self, slot, params)

    monkeypatch.setattr(mlt._MemberLossScale, "unscale_", unscale_)


def test_member_loss_scale_trains_as_unscaled(batched_run, monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    events, _, _ = batched_run
    _force_member_loss_scale(monkeypatch)
    scaled, _, _ = _run(monkeypatch, MEMBERS)
    for m in MEMBERS:
        losses, state = _final(events, m["run_id"])
        scaled_losses, scaled_state = _final(scaled, m["run_id"])
        assert scaled_losses == pytest.approx(losses, rel = 1e-5)
        for key in state:
            torch.testing.assert_close(scaled_state[key], state[key], atol = 1e-6, rtol = 1e-5)


def test_an_fp16_overflow_skips_only_that_members_step(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    _force_member_loss_scale(monkeypatch)
    clean, _, _ = _run(monkeypatch, MEMBERS)
    _force_member_loss_scale(monkeypatch, poison = (0, 1))
    hit, _, _ = _run(monkeypatch, MEMBERS)
    for run_id in ("job_b", "job_c"):
        losses, state = _final(clean, run_id)
        hit_losses, hit_state = _final(hit, run_id)
        assert hit_losses == losses
        for key in state:
            torch.testing.assert_close(hit_state[key], state[key], atol = 0, rtol = 0)
    a = [e for e in _member_events(hit, "job_a") if e["type"] == "progress" and e["step"] > 0]
    # Its first step was skipped: no gradient norm, and the next step trains at the rate it would have used.
    assert a[0]["grad_norm"] is None and a[1]["grad_norm"] is not None
    assert a[1]["learning_rate"] == a[0]["learning_rate"]
    hit_a, clean_a = _final(hit, "job_a")[1], _final(clean, "job_a")[1]
    assert any(not torch.equal(hit_a[k], clean_a[k]) for k in clean_a)


def test_member_loss_scale_follows_grad_scaler_rules():
    scale = mlt._MemberLossScale(2, init_scale = 8.0, growth_interval = 2)
    p = torch.nn.Parameter(torch.zeros(3))
    for _ in range(2):
        p.grad = torch.full((3,), 16.0)
        assert scale.unscale_(1, [p]) and torch.equal(p.grad, torch.full((3,), 2.0))
    assert scale.scale == [8.0, 16.0]
    p.grad = torch.tensor([1.0, float("nan"), 0.0])
    assert not scale.unscale_(0, [p]) and p.grad is None
    assert scale.scale == [4.0, 16.0] and scale.good_steps == [0, 2 % 2]


def test_pack_rows_on_trl_without_packing_strategies(monkeypatch):
    import trl.data_utils

    real = trl.data_utils.pack_dataset

    def legacy(
        dataset,
        seq_length,
        map_kwargs = None,
    ):
        return real(dataset, seq_length, strategy = "wrapped")

    monkeypatch.setattr(trl.data_utils, "pack_dataset", legacy)
    rows = [([i] * n, [-100] + [i] * (n - 1)) for i, n in enumerate([5, 9, 3, 7, 4], start = 1)]
    packed, lengths = mlt._pack_rows(rows, 12)
    # Documents wrap across rows; each row is one sequence, as TRL's wrapped packing trains it.
    assert [ids for ids, _ in packed] == [
        [t for ids, _ in rows for t in ids][i : i + 12] for i in range(0, 28, 12)
    ]
    assert [labels for _, labels in packed] == [
        [t for _, lab in rows for t in lab][i : i + 12] for i in range(0, 28, 12)
    ]
    assert lengths == [[12], [12], [4]]


class _Processor(ProcessorMixin):
    """A vision model's processor: called with text, it reads it as images."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(
        self,
        images = None,
        text = None,
        **kwargs,
    ):
        raise ValueError("Incorrect image source.")

    def save_pretrained(self, path, **kwargs):
        self.tokenizer.save_pretrained(path)
        open(os.path.join(path, "preprocessor_config.json"), "w").write("{}")


def test_a_vision_models_processor_tokenizes_text_as_the_single_job_does(
    batched_run, monkeypatch, tmp_path
):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    events, _, _ = _run(monkeypatch, MEMBERS, wrap_tokenizer = _Processor)
    plain, _, _ = batched_run
    for m in MEMBERS:
        mine = _member_events(events, m["run_id"])
        assert mine[-1]["type"] == "complete", mine[-1]
        assert os.path.exists(os.path.join(mine[-1]["output_dir"], "preprocessor_config.json"))
        losses = [e["loss"] for e in mine if e["type"] == "progress"]
        assert losses == pytest.approx(
            [e["loss"] for e in _member_events(plain, m["run_id"]) if e["type"] == "progress"],
            rel = 1e-5,
        )


def test_pack_rows_packs_each_worker_shard_as_the_single_job_does():
    from datasets import Dataset
    from trl.data_utils import pack_dataset

    rows = [([i % 50 + 4] * 8, [i % 50 + 4] * 8) for i in range(16)]
    single = pack_dataset(
        Dataset.from_dict({"input_ids": [r[0] for r in rows], "labels": [r[1] for r in rows]}),
        2048,
        strategy = "bfd",
        map_kwargs = {"num_proc": 4},
    )
    packed, lengths = mlt._pack_rows(rows, 2048, num_proc = 4)
    assert len(packed) == len(single) == 4
    assert [ids for ids, _ in packed] == [list(x) for x in single["input_ids"]]
    assert lengths == [list(x) for x in single["seq_lengths"]]


def test_packing_is_refused_where_packed_documents_see_each_other(monkeypatch, tmp_path):
    import transformers.masking_utils as masking_utils

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    assert mlt._packed_forward_isolated(_base().eval())
    # Attention that ignores the document boundaries, as Qwen3.5's full attention did on transformers 5.3.
    monkeypatch.setattr(masking_utils, "find_packed_sequence_indices", lambda position_ids: None)
    assert not mlt._packed_forward_isolated(_base().eval())
    seen = _spy_packed(monkeypatch)
    events, _, _ = _run(monkeypatch, PACKED[:2], max_seq_length = 24)
    assert seen == []
    assert all(_member_events(events, m["run_id"])[-1]["type"] == "complete" for m in PACKED[:2])


def test_an_expert_only_moe_target_trains_batched(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    events, _, _ = _run(
        monkeypatch, MEMBERS[:2], base = _kind_base("moe"), target_modules = ["gate_up_proj"]
    )
    mine = _member_events(events, "job_b")
    assert mine[-1]["type"] == "complete", mine[-1]
    adapter = load_file(os.path.join(mine[0]["output_dir"], "adapter_model.safetensors"))
    assert adapter and all(".mlp.experts." in k for k in adapter), sorted(adapter)[:3]
    assert all(v.abs().max() > 0 for k, v in adapter.items() if k.endswith("lora_B.weight"))


def test_routing_check_runs_on_a_kernel_without_expert_support(monkeypatch):
    import sys
    import types

    import unsloth.kernels

    multi_lora_trainer = mlt
    from unsloth.kernels import multi_lora

    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    multi_lora.inject_multi_lora(model, ["0"], 2, 2, 2)
    # An Unsloth release that predates the expert kernel: only the dense layer class exists.
    old = types.SimpleNamespace(MultiLoraLinear = multi_lora.MultiLoraLinear)
    monkeypatch.setattr(unsloth.kernels, "multi_lora", old)
    monkeypatch.setitem(sys.modules, "unsloth.kernels.multi_lora", old)
    assert [name for name, _ in multi_lora_trainer._routed_modules(model)] == ["0"]
