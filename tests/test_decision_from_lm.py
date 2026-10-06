# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import json
import sys

import pytest

torch = pytest.importorskip("torch")

from transformers import TrainerCallback, TrainingArguments

from unsloth import DecisionTrainer, FastDecisionModel
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
    # The same seed gives the same fresh head.
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
    with torch.no_grad():
        model.eval()
        ours, _ = model(**{k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})
        theirs, _ = reloaded(
            **{
                k: v.to(next(reloaded.parameters()).device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
        )
    # Served logits: the save folds 1 / T into the head and stores the head and the merged backbone
    # in bf16, so the reloaded logits match ours / T up to bf16 rounding, which grows with their scale.
    ours, theirs = ours.float().cpu(), theirs.float().cpu()
    mask = ours > -1e3
    expected = ours[mask] / head_temperature
    error = (theirs[mask] - expected).abs().max().item()
    assert error <= 0.03 * expected.abs().max().item() + 0.05, (error, expected, theirs[mask])
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
