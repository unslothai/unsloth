# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import json
import math
import sys

import pytest

torch = pytest.importorskip("torch")

from transformers import TrainerCallback, TrainingArguments

from unsloth import DecisionTrainer, FastDecisionModel
from unsloth.models import decision
from unsloth.models.decision_from_lm import default_head_config

TINY_QWEN3 = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TINY_QWEN3_5 = "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration"
HEAD = {
    "hidden_size": None,
    "width": 32,
    "routing_layers": 1,
    "layers": 1,
    "heads": 4,
    "feedforward": 64,
}


def _rows(n):
    rows = []
    for i in range(n):
        outage = i % 2 == 1
        rows.append(
            {
                "state": f"ticket {i}: "
                + ("the server is down again" if outage else "my invoice was paid"),
                "questions": {
                    "outage": {"type": "noul", "instructions": "Is a service down?"},
                    "team": {"type": "choice", "criteria": {"billing": "payments", "tech": "bugs"}},
                    "mood": {"type": "score", "criteria": ["calm", "annoyed", "angry"]},
                },
                "gold": {
                    "outage": outage,
                    "team": "tech" if outage else "billing",
                    "mood": 2 if outage else 0,
                },
            }
        )
    return rows


class _Losses(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(
        self,
        args,
        state,
        control,
        logs = None,
        **kwargs,
    ):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])


def _train(
    model,
    processor,
    items,
    tmp_path,
    steps = 30,
):
    losses = _Losses()
    DecisionTrainer(
        model = model,
        tokenizer = processor,
        train_dataset = items,
        head_learning_rate = 5e-3,
        callbacks = [losses],
        args = TrainingArguments(
            output_dir = str(tmp_path / "run"),
            per_device_train_batch_size = 8,
            max_steps = steps,
            learning_rate = 5e-3,
            logging_steps = 1,
            report_to = "none",
            save_strategy = "no",
        ),
    ).train()
    return losses.losses


_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model.safetensors-*-of-*.safetensors",
    "model-*-of-*.safetensors",
    "model.safetensors.index.json",
    "adapter_config.json",
    "adapter_model.safetensors",
    "joint_head.safetensors",
    "joint_head_config.json",
    "unsloth_decision_config.json",
    "joint_schema_model.py",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


def _assert_only_model_files(folder):
    # Nothing else: no download cache (.cache/huggingface), locks, staging or partial files.
    import fnmatch

    names = sorted(path.name for path in folder.iterdir())
    stray = [n for n in names if not any(fnmatch.fnmatch(n, p) for p in _MODEL_FILES)]
    assert not stray, (stray, names)


def test_default_head_config_scales_with_the_backbone():
    assert default_head_config(4096)["width"] == 1024
    assert default_head_config(2048)["width"] == 512
    small = default_head_config(1024, width = 256)
    assert small["heads"] == 4 and small["feedforward"] == 1024 and small["hidden_size"] == 1024


def _qwen3_5_runs_here():
    import importlib.util

    from real_accelerator import has_real_cuda

    # transformers routes Qwen3.5's conv to causal_conv1d whenever it imports, CUDA-only kernel.
    return has_real_cuda() or importlib.util.find_spec("causal_conv1d") is None


@pytest.mark.parametrize(
    "base",
    [
        TINY_QWEN3,
        pytest.param(
            TINY_QWEN3_5,
            marks = pytest.mark.skipif(
                not _qwen3_5_runs_here(),
                reason = "causal_conv1d is installed but there is no CUDA device",
            ),
        ),
    ],
)
def test_plain_lm_becomes_a_decision_model_that_trains_saves_and_reloads(base, tmp_path):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(base)
    hidden = (getattr(config, "text_config", None) or config).hidden_size
    model, processor = FastDecisionModel.from_pretrained(
        base, decision_head = "clef", head_config = {**HEAD, "hidden_size": hidden}, max_seq_length = 512
    )
    assert getattr(model, "is_clef", False)
    assert model.head.hidden_norm.weight.dtype == torch.float32
    again, _ = FastDecisionModel.from_pretrained(
        base, decision_head = "clef", head_config = {**HEAD, "hidden_size": hidden}, max_seq_length = 512
    )
    for a, b in zip(model.head.parameters(), again.head.parameters()):
        assert torch.equal(a.cpu(), b.cpu())
    del again

    items, report = FastDecisionModel.build_dataset(_rows(64), processor, model)
    assert report["skipped"] == 0 and len(items) == 64
    train, holdout = FastDecisionModel.split_holdout(items, fraction = 0.25)

    # Stage A: head only, backbone frozen.
    FastDecisionModel.freeze_backbone(model)
    assert not any(p.requires_grad for p in model.encoder.parameters())
    head_before = {k: v.clone() for k, v in model.head.state_dict().items()}
    _train(model, processor, train, tmp_path, steps = 5)
    assert any(not torch.equal(head_before[k], v) for k, v in model.head.state_dict().items())
    FastDecisionModel.unfreeze_backbone(model)

    # Stage B: LoRA + head.
    before = FastDecisionModel.evaluate(model, processor, holdout)
    model = FastDecisionModel.get_peft_model(model, r = 8, lora_alpha = 8)
    assert any("lora_" in n for n, p in model.encoder.named_parameters() if p.requires_grad)
    losses = _train(model, processor, train, tmp_path)
    after = FastDecisionModel.evaluate(model, processor, holdout)
    assert min(losses[-5:]) < losses[0] and after["loss"] < before["loss"]
    FastDecisionModel.calibrate(model, processor, holdout)

    state, questions = _rows(1)[0]["state"], _rows(1)[0]["questions"]
    answers = FastDecisionModel.predict(model, processor, state, questions)
    assert answers["outage"]["type"] == "noul" and 0 <= answers["outage"]["noul"] <= 1
    assert answers["team"]["choice"] in ("billing", "tech")
    assert sum(answers["team"]["probabilities"].values()) == pytest.approx(1, abs = 1e-3)
    assert answers["mood"]["legend"] == {"0": "calm", "1": "annoyed", "2": "angry"}

    # save_pretrained: only the adapters, the head and the configs; reloaded onto the base.
    adapters = tmp_path / "adapters"
    model.save_pretrained(str(adapters))
    assert (adapters / "adapter_config.json").is_file()
    assert (adapters / "joint_head.safetensors").is_file()
    assert not (adapters / "config.json").exists() and not list(adapters.glob("model*.safetensors"))
    assert (
        json.loads((adapters / "adapter_config.json").read_text())["base_model_name_or_path"]
        == base
    )
    from_adapters, adapter_processor = FastDecisionModel.from_pretrained(
        str(adapters), max_seq_length = 512
    )
    assert hasattr(from_adapters.encoder, "peft_config")
    assert from_adapters.decision_config["base_model"] == base
    again = FastDecisionModel.predict(from_adapters, adapter_processor, state, questions)
    # The head is stored in bf16; the adapters keep their own dtype.
    for name in questions:
        for key, value in answers[name].get("probabilities", {}).items():
            assert again[name]["probabilities"][key] == pytest.approx(value, abs = 0.03)
    assert again["outage"]["noul"] == pytest.approx(answers["outage"]["noul"], abs = 0.03)
    # A merged save from the adapter reload is a complete Clef folder again.
    remerged = tmp_path / "remerged"
    from_adapters.save_pretrained_merged(str(remerged))
    assert (remerged / "config.json").is_file() and not (remerged / "adapter_config.json").exists()
    del from_adapters

    out = tmp_path / "out"
    model.save_pretrained_merged(str(out))
    _assert_only_model_files(out)
    # Saving adapters over a merged folder (or the reverse) leaves no stale weights behind.
    model.save_pretrained(str(out / "swap"))
    model.save_pretrained_merged(str(out / "swap"))
    assert not (out / "swap" / "adapter_config.json").exists()
    model.save_pretrained(str(out / "swap"))
    assert not (out / "swap" / "config.json").exists()
    assert not list((out / "swap").glob("model*.safetensors"))
    assert json.loads((out / "joint_head_config.json").read_text())["hidden_size"] == hidden
    reloaded, reloaded_processor = FastDecisionModel.from_pretrained(str(out), max_seq_length = 512)
    assert reloaded.decision_config["temperature"] == model.decision_config["temperature"]
    merged_answers = FastDecisionModel.predict(reloaded, reloaded_processor, state, questions)
    assert merged_answers["outage"]["noul"] == pytest.approx(answers["outage"]["noul"], abs = 0.05)
    # calibrate() fits a head temperature that the save folds into the head weights.
    head_temperature = model.decision_config.get("head_temperature", 1.0)
    assert reloaded.decision_config.get("folded_temperature", 1.0) == pytest.approx(
        head_temperature
    )
    record = holdout[0]
    batch = {
        "input_ids": torch.tensor([record["input_ids"]]),
        "attention_mask": torch.ones(1, len(record["input_ids"]), dtype = torch.long),
        "records": [record["record"]],
    }
    device = next(model.parameters()).device
    # An earlier Clef load leaves UNSLOTH_HIGH_PRECISION_LAYERNORM set: float32 norms need autocast.
    amp_dtype = decision._clef_amp_dtype(model, device)
    autocast = torch.autocast(device.type, dtype = amp_dtype, enabled = amp_dtype is not None)
    with torch.no_grad(), autocast:
        model.eval()
        ours, _ = model(**{k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})
        theirs, _ = reloaded(
            **{
                k: v.to(next(reloaded.parameters()).device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
        )
    # The save folds 1 / T into the bf16 head; GPU training is not bit-reproducible, so the bound is relative
    # to the logit scale (30 B200 runs peaked at 5.5%; a fold left out or doubled is off by >= 13%).
    ours, theirs = ours.float().cpu(), theirs.float().cpu()
    mask = ours > -1e3
    expected = ours[mask] / head_temperature
    error = (theirs[mask] - expected).abs().max().item()
    scale = expected.abs().max().item()
    assert error <= 0.08 * scale + 0.05, (error, scale, head_temperature, expected, theirs[mask])
    if base == TINY_QWEN3_5:
        # The base repo has no joint_schema_model.py; Unsloth ships Cloudflare's, so their loader works.
        import importlib.util

        assert (out / "joint_schema_model.py").is_file()
        spec = importlib.util.spec_from_file_location(
            "converted_reference", out / "joint_schema_model.py"
        )
        reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reference
        spec.loader.exec_module(reference)
        released, _ = reference.load_release_model(str(out), device = device, dtype = torch.float32)
        with torch.no_grad():
            released_logits = released(
                {
                    **{k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)},
                    "records": batch["records"],
                    "media": {},
                }
            )[0]
        for row, z in enumerate(released_logits):
            assert int(z.argmax()) == int(theirs[row, : len(z)].argmax())


def test_float32_norms_train_without_a_precision_flag(tmp_path, monkeypatch):
    from real_accelerator import has_real_cuda

    if not has_real_cuda():
        pytest.skip("the CPU path loads without Unsloth's layernorm upcast")
    # Gemma 3 / 4, gpt-oss and Qwen3.5 loads set this and it outlives them.
    monkeypatch.setenv("UNSLOTH_HIGH_PRECISION_LAYERNORM", "1")
    model, processor = FastDecisionModel.from_pretrained(
        TINY_QWEN3, decision_head = "clef", head_config = {**HEAD, "hidden_size": 8}, max_seq_length = 512
    )
    norms = {p.dtype for n, p in model.encoder.named_parameters() if n.endswith("norm.weight")}
    assert torch.float32 in norms, norms
    items, _ = FastDecisionModel.build_dataset(_rows(16), processor, model)
    losses = _train(model, processor, items, tmp_path, steps = 2)
    assert len(losses) == 2 and all(map(math.isfinite, losses)), losses


def test_head_init_must_match_the_backbone(tmp_path):
    from safetensors.torch import save_file

    from unsloth.models.clef import JointSchemaHead

    head = JointSchemaHead(**{**HEAD, "hidden_size": 99})
    save_file(head.state_dict(), str(tmp_path / "joint_head.safetensors"))
    (tmp_path / "joint_head_config.json").write_text(json.dumps({**HEAD, "hidden_size": 99}))
    with pytest.raises(ValueError, match = "hidden size"):
        FastDecisionModel.from_pretrained(TINY_QWEN3, decision_head = "clef", head_init = str(tmp_path))


def test_head_init_warm_starts_from_a_matching_head(tmp_path):
    from safetensors.torch import save_file

    from unsloth.models.clef import JointSchemaHead

    from transformers import AutoConfig

    hidden = AutoConfig.from_pretrained(TINY_QWEN3).hidden_size
    config = {**HEAD, "hidden_size": hidden}
    head = JointSchemaHead(**config)
    with torch.no_grad():
        head.prior_logit_scale.fill_(1.25)
    save_file(head.state_dict(), str(tmp_path / "joint_head.safetensors"))
    (tmp_path / "joint_head_config.json").write_text(json.dumps(config))
    with pytest.warns(UserWarning, match = "warm start"):
        model, _ = FastDecisionModel.from_pretrained(
            TINY_QWEN3, decision_head = "clef", head_init = str(tmp_path)
        )
    assert float(model.head.prior_logit_scale) == 1.25


def test_unknown_decision_head_is_refused():
    with pytest.raises(ValueError, match = "decision_head"):
        FastDecisionModel.from_pretrained(TINY_QWEN3, decision_head = "pointer")


def _typed_decision_rows(n):
    # Shaped like LocalLLaMA/typed-decisions: JSON strings for state, questions and gold.
    import json

    rows = []
    for i in range(n):
        outage = i % 2 == 1
        state = {
            "ticket": f"T-{i}",
            "message": "the server is down again" if outage else "my invoice was paid twice",
        }
        questions = {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "invoices, refunds", "technical": "bugs, outages"},
            },
            "refund": {
                "type": "noul",
                "instructions": "Does the customer ask for a refund?",
                "criteria": {"false": "No refund.", "true": "A refund is asked for."},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "today"],
            },
        }
        p = 0.8
        gold = {
            "team": {
                "type": "choice",
                "label": "technical" if outage else "billing",
                "probabilities": {
                    "technical": p if outage else 1 - p,
                    "billing": 1 - p if outage else p,
                },
            },
            "refund": {
                "type": "noul",
                "label": str(not outage).lower(),
                "noul": 0.2 if outage else 0.9,
            },
            "urgency": {"type": "score", "label": "2" if outage else "0"},
        }
        rows.append(
            {
                "id": f"row-{i}",
                "workflow": "customer_service",
                "state": json.dumps(state),
                "questions": json.dumps(questions),
                "gold": json.dumps(gold),
                "n_questions": 3,
            }
        )
    return rows


@pytest.mark.skipif(
    not _qwen3_5_runs_here(), reason = "causal_conv1d is installed but there is no CUDA device"
)
def test_the_decision_notebook_runs_unchanged_on_a_plain_qwen3_5(tmp_path, monkeypatch):
    # unslothai/notebooks#380's call sequence, on a tiny Qwen3.5 with no decision_head argument.
    from datasets import Dataset
    from real_accelerator import has_real_cuda
    from transformers import TrainingArguments

    from unsloth import is_bfloat16_supported

    monkeypatch.chdir(tmp_path)
    four_bit = has_real_cuda()  # The notebook loads in 4-bit, which needs a GPU.
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name = TINY_QWEN3_5, max_seq_length = 2048, load_in_4bit = four_bit
    )
    assert getattr(model, "is_clef", False)
    model = FastDecisionModel.get_peft_model(
        model,
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    dataset = Dataset.from_list(_typed_decision_rows(40))
    items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model)
    assert report["skipped"] == 0 and report["total"] == 120 and report["truncated"] == 0
    train_items, eval_items = FastDecisionModel.split_holdout(items, seed = 3407)
    assert train_items and eval_items
    FastDecisionModel.evaluate(model, tokenizer, eval_items)

    # fp16 = True where bfloat16 is missing (a T4): DecisionTrainer settles Clef's precision.
    bf16 = is_bfloat16_supported() and has_real_cuda()
    trainer = DecisionTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = train_items,
        eval_dataset = eval_items,
        args = TrainingArguments(
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 4,
            warmup_steps = 1,
            max_steps = 2,
            learning_rate = 2e-4,
            lr_scheduler_type = "cosine",
            weight_decay = 0.01,
            bf16 = bf16,
            fp16 = has_real_cuda() and not bf16,
            eval_strategy = "epoch",
            logging_steps = 1,
            output_dir = "outputs",
            report_to = "none",
            seed = 3407,
        ),
    )
    trainer.train()
    # The trainer's own checkpoint: adapters and the exact float32 head, not the shared-weight backbone.
    from safetensors.torch import load_file

    checkpoint = tmp_path / "outputs" / "checkpoint-2"
    assert (checkpoint / "adapter_model.safetensors").is_file()
    saved = load_file(str(checkpoint / "joint_head.safetensors"))
    for name, value in model.head.state_dict().items():
        assert torch.equal(saved[name], value.detach().cpu().float()), name
    FastDecisionModel.calibrate(model, tokenizer, eval_items)
    test_items, _ = FastDecisionModel.build_dataset(
        Dataset.from_list(_typed_decision_rows(6)), tokenizer, model
    )
    assert 0 <= FastDecisionModel.evaluate(model, tokenizer, test_items)["accuracy"] <= 1

    FastDecisionModel.for_inference(model)
    state = "Hi, I was charged twice for invoice #4411. Please refund the duplicate today."
    questions = {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, errors",
                "sales": "pricing, new plans",
            },
        },
        "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["not urgent", "soon", "today"],
        },
    }
    answers = FastDecisionModel.predict(model, tokenizer, state, questions)
    assert answers["team"]["answer"] in ("billing", "technical", "sales")
    assert isinstance(answers["refund"]["answer"], bool)
    assert answers["urgency"]["answer"] in (0, 1, 2)
    for name, result in answers.items():
        probabilities = {k: round(v, 3) for k, v in result["probabilities"].items()}
        assert sum(probabilities.values()) == pytest.approx(1, abs = 0.01), name
    assert set(answers["refund"]["probabilities"]) == {"false", "true"}

    model.save_pretrained("decision_model")
    assert (tmp_path / "decision_model" / "adapter_config.json").is_file()
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name = "decision_model", max_seq_length = 2048, load_in_4bit = four_bit
    )
    again = FastDecisionModel.predict(model, tokenizer, state, questions)
    for name in questions:
        assert (
            again[name]["answer"] == answers[name]["answer"]
            or max(answers[name]["probabilities"].values()) < 0.6
        )
        for key, value in answers[name]["probabilities"].items():
            assert again[name]["probabilities"][key] == pytest.approx(value, abs = 0.05)
    model.save_pretrained_merged("decision_model_16bit", tokenizer, save_method = "merged_16bit")
    _assert_only_model_files(tmp_path / "decision_model_16bit")
    _assert_only_model_files(tmp_path / "decision_model")
    assert (tmp_path / "decision_model_16bit" / "config.json").is_file()
    assert (tmp_path / "decision_model_16bit" / "joint_head.safetensors").is_file()


def test_long_states_are_cut_in_training_but_read_in_full_by_predict(monkeypatch):
    # Training cuts the end of a long state; predict() reads up to CLEF_SERVE_MAX_LEN tokens, like serving.
    from unsloth.models import decision

    from transformers import AutoConfig

    hidden = AutoConfig.from_pretrained(TINY_QWEN3).hidden_size
    model, processor = FastDecisionModel.from_pretrained(
        TINY_QWEN3,
        decision_head = "clef",
        head_config = {**HEAD, "hidden_size": hidden},
        max_seq_length = 512,
    )
    long_row = _rows(1)[0]
    long_row["state"] = "the server is down again. " * 400
    items, report = FastDecisionModel.build_dataset([long_row] + _rows(3), processor, model)
    assert report["skipped"] == 0 and report["truncated"] == 1
    assert len(items[0]["input_ids"]) == 512
    seen = {}
    real = decision._clef_decide

    def spy(*args, **kwargs):
        seen["max_length"] = kwargs["max_length"]
        result = real(*args, **kwargs)
        seen["tokens"] = result["input_tokens"]
        return result

    monkeypatch.setattr(decision, "_clef_decide", spy)
    FastDecisionModel.predict(model, processor, long_row["state"], long_row["questions"])
    assert seen["max_length"] == decision.CLEF_SERVE_MAX_LEN and seen["tokens"] > 512


def test_backbone_stays_on_one_device_unless_the_caller_places_it(monkeypatch):
    from unsloth.models import decision, decision_from_lm, loader

    seen = []

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        seen.append(kwargs.get("device_map"))
        raise Captured

    monkeypatch.setattr(decision, "_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(loader.FastModel, "from_pretrained", capture)
    index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    for kwargs in ({}, {"device_map": "auto"}):
        with pytest.raises(Captured):
            decision_from_lm._load_backbone(TINY_QWEN3, 64, None, False, False, None, False, kwargs)
    # The default from-LM path: a plain LLM with no decision_head argument becomes a Clef model.
    with pytest.raises(Captured):
        FastDecisionModel.from_pretrained(TINY_QWEN3, max_seq_length = 64)
    assert seen == [{"": f"cuda:{index}"}, "auto", {"": f"cuda:{index}"}]


def _tiny_lm(
    monkeypatch,
    name = TINY_QWEN3,
    **kwargs,
):
    monkeypatch.setattr(decision, "_device", lambda: torch.device("cpu"))
    return FastDecisionModel.from_pretrained(
        name, decision_head = "clef", head_width = 32, max_seq_length = 256, **kwargs
    )


def test_a_gpt2_style_lm_runs_as_a_decision_model(monkeypatch):
    model, processor = _tiny_lm(monkeypatch, "trl-internal-testing/tiny-GPT2LMHeadModel")
    row = _rows(1)[0]
    answers = FastDecisionModel.predict(model, processor, row["state"], row["questions"])
    assert answers["team"]["choice"] in ("billing", "tech")


def test_adapters_refuse_a_full_finetune_and_keep_the_base_revision(tmp_path, monkeypatch):
    model, processor = _tiny_lm(monkeypatch, revision = "main")
    assert model.decision_config["base_revision"] == "main"
    model = FastDecisionModel.get_peft_model(model, r = 4, lora_alpha = 4)
    model.save_pretrained(str(tmp_path / "adapters"))
    adapter = json.loads((tmp_path / "adapters" / "adapter_config.json").read_text())
    assert adapter["revision"] == "main"
    with pytest.raises(ValueError, match = "save_pretrained_merged"):
        FastDecisionModel.from_pretrained(str(tmp_path / "adapters"), full_finetuning = True)
    reloaded, _ = FastDecisionModel.from_pretrained(str(tmp_path / "adapters"))
    assert reloaded.decision_config["base_revision"] == "main"


def test_a_full_finetune_restores_its_own_checkpoint(tmp_path, monkeypatch):
    model, processor = _tiny_lm(monkeypatch, full_finetuning = True)
    trainer = DecisionTrainer(
        model = model,
        tokenizer = processor,
        train_dataset = [],
        args = TrainingArguments(output_dir = str(tmp_path / "run"), report_to = "none"),
    )
    trainer._save(str(tmp_path / "ckpt"))
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    decision._load_clef_checkpoint(model, tmp_path / "ckpt")
    for k, v in model.state_dict().items():
        assert torch.equal(v, saved[k]), k
