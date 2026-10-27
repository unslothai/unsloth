# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Batched multi-LoRA worker on a tiny random Qwen3 (CPU, PyTorch fallback of the kernels)."""

from __future__ import annotations

import copy
import json
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
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

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
    **extra,
):
    base = _base()
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
        trainer.model, trainer.tokenizer = base, tokenizer
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
