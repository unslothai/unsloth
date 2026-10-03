import json
import math
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from transformers import TrainerCallback, TrainingArguments

from unsloth import DecisionTrainer, FastDecisionModel
from unsloth.models import decision
from unsloth.models.decision import (
    DecisionDataCollator,
    DecisionDataError,
    _LengthGroupedBatches,
    _laya,
    _target,
)

laya = _laya()
# Every word the states, questions and options use, so the toy task below is learnable.
WORDS = (
    "the server is down again refund my card charge twice please help now noul choice score "
    "question : ? , does this need a reply which team should handle it how upset customer outage "
    "service billing charges and refunds level 0 1 2 calm annoyed angry false no statement not hold "
    "true yes holds"
).split()
QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does this need a reply now?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle it?",
        "criteria": {"outage": "service down", "billing": "charges and refunds"},
    },
    "mood": {
        "type": "score",
        "instructions": "How upset is the customer?",
        "criteria": ["calm", "annoyed", "angry"],
    },
}


def _row(i):
    outage = i % 2 == 0
    return {
        "state": "the server is down again help now" if outage else "refund my card charge twice",
        "questions": QUESTIONS,
        "gold": {
            "urgent": {"label": "true" if outage else "false"},
            "team": {"label": "outage" if outage else "billing"},
            "mood": 2 if outage else 0,
        },
    }


def _args(tmp_path, **overrides):
    return TrainingArguments(
        **{
            "output_dir": str(tmp_path / "run"),
            "per_device_train_batch_size": 8,
            "max_steps": 4,
            "learning_rate": 1e-3,
            "report_to": "none",
            "save_strategy": "no",
            **overrides,
        }
    )


def _tiny_encoder_config(**rope):
    from transformers import ModernBertConfig

    config = ModernBertConfig(
        vocab_size = 64,
        hidden_size = 64,
        intermediate_size = 96,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        pad_token_id = 0,
        cls_token_id = 2,
        sep_token_id = 1,
        global_attn_every_n_layers = 2,
        local_attention = 16,
    ).to_dict()
    for key in ("rope_parameters", "global_rope_theta", "local_rope_theta", "rope_theta"):
        config.pop(key, None)
    return {**config, **rope}


@pytest.fixture
def checkpoint(tmp_path):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["[PAD]", "[SEP]", "[CLS]", "[UNK]", "[MASK]"]
    vocab = {token: i for i, token in enumerate(specials + sorted(set(WORDS)))}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token = "[UNK]"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object = tokenizer,
        pad_token = "[PAD]",
        sep_token = "[SEP]",
        cls_token = "[CLS]",
        unk_token = "[UNK]",
        mask_token = "[MASK]",
    ).save_pretrained(str(tmp_path / "base" / "tokenizer"))
    (tmp_path / "base" / "encoder").mkdir()
    # The layout transformers 5 writes, as the published Laya checkpoints have it.
    (tmp_path / "base" / "encoder" / "config.json").write_text(
        json.dumps(
            _tiny_encoder_config(
                rope_parameters = {
                    "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
                    "sliding_attention": {"rope_theta": 160000.0, "rope_type": "default"},
                }
            ),
            indent = 2,
        )
    )
    config = {
        "encoder": "tiny",
        "head_layers": 1,
        "act_costs": {"escalate": 0.5},
        "max_len": 96,
        "head_max_len": 48,
        "temperature": [1.2, 1.1, 1.3],
        "temperature_by_options": {"noul:2": 0.7, "score:3-5": 0.9},
        "training": {"updates": 1000},
    }
    torch.manual_seed(0)
    model = laya.common.build_model(config, encoder_dir = str(tmp_path / "base" / "encoder"))
    save_file(
        {k: v.half().contiguous() for k, v in model.state_dict().items()},
        str(tmp_path / "base" / "model.safetensors"),
    )
    (tmp_path / "base" / "rl_agent_config.json").write_text(json.dumps(config))
    return tmp_path / "base"


@pytest.fixture
def fresh_laya():
    decision._laya.cache_clear()
    yield
    decision._laya.cache_clear()


def test_laya_is_the_copy_vendored_with_unsloth(monkeypatch, fresh_laya):
    # A pip installed laya (0.3.9+ has no agent or common attributes) is ignored and left in place.
    pip_laya = types.ModuleType("laya")
    monkeypatch.setitem(sys.modules, "laya", pip_laya)
    vendored = decision._laya()
    assert vendored is not pip_laya and sys.modules["laya"] is pip_laya
    assert Path(vendored.__file__).resolve() == decision._VENDORED_LAYA
    assert vendored.__version__ == "0.3.5" and hasattr(vendored.agent.Agent, "_to_internal")
    # Studio registers the same files as "laya" before training, and that module is the one used.
    studio_laya = types.ModuleType("laya")
    studio_laya.__file__ = str(decision._VENDORED_LAYA)
    monkeypatch.setitem(sys.modules, "laya", studio_laya)
    decision._laya.cache_clear()
    assert decision._laya() is studio_laya


def test_missing_vendored_laya_is_a_clear_error(monkeypatch, tmp_path, fresh_laya):
    monkeypatch.setattr(decision, "_VENDORED_LAYA", tmp_path / "laya" / "__init__.py")
    with pytest.raises(ImportError, match = "Unsloth: decision models need laya"):
        decision._laya()


def test_gold_labels_and_probabilities():
    noul, team, mood = (
        laya.agent.Agent._to_internal(QUESTIONS[n]) for n in ("urgent", "team", "mood")
    )
    assert _target(noul, {"probabilities": {"true": 0.3}}) == ([pytest.approx(0.7), 0.3], 0)
    assert _target(noul, {"noul": 0.8}) == ([pytest.approx(0.2), 0.8], 1)
    assert _target(team, {"probabilities": {"outage": 3, "billing": 1}}) == ([0.75, 0.25], 0)
    for label in (True, "TRUE", "True", " true"):
        assert _target(noul, label) == ([0.0, 1.0], 1)
    for label in (2, 2.0, "2", "2.0"):
        assert _target(mood, {"label": label}) == ([0.0, 0.0, 1.0], 2)
    unusable = [
        (mood, {"label": 2.5}),
        (mood, {"label": True}),
        (mood, math.inf),
        (mood, {"label": math.nan}),
        (noul, 1),
        (noul, {"noul": math.inf}),
        (noul, {"probabilities": {"true": math.inf}}),
        (team, {"probabilities": {"outage": math.nan, "billing": 1}}),
    ]
    for internal, gold in unusable:
        with pytest.raises(DecisionDataError):
            _target(internal, gold)


def test_temperature_fit_finds_the_temperature_of_the_soft_gold():
    # softmax([2, 0] / T) is [0.6, 0.4] at T = 2 / ln(1.5).
    logits = [torch.tensor([2.0, 0.0])] * 10
    items = [{"target": [0.6, 0.4]}] * 10
    assert decision._fit_temperature(logits, items) == pytest.approx(2 / math.log(1.5), rel = 1e-2)


def test_dataset_skips_bad_rows_and_names_the_most_common_reason(checkpoint):
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), use_gradient_checkpointing = False
    )
    assert model.decision_config["max_len"] == 1024
    assert model.decision_config["head_max_len"] == 256
    assert "training" not in model.decision_config

    nested = {**QUESTIONS, "team": {**QUESTIONS["team"], "criteria": [["outage"], ["billing"]]}}
    numbered = {**QUESTIONS, "team": {**QUESTIONS["team"], "criteria": [1, 2]}}
    rows = [
        {**_row(0), "gold": {"urgent": "true", "team": "outage"}},
        *[{**_row(i), "questions": nested} for i in range(1, 4)],
        {**_row(4), "questions": numbered},
        "not a row",
        {**_row(6), "gold": {**_row(6)["gold"], "urgent": {"noul": math.inf}}},
        {**_row(7), "gold": {**_row(7)["gold"], "mood": 1.5}},
        {"state": "x", "questions": "{}", "gold": {}},
        *[_row(i) for i in range(9, 30)],
    ]
    items, report = FastDecisionModel.build_dataset(rows, tokenizer, model)
    assert report["total"] == 3 * 28 + 1 and report["skipped"] == 8
    assert len(items) == report["total"] - report["skipped"]
    assert (
        report["reason"] == 'row 2: "team" needs criteria naming its options (and 3 more like it)'
    )

    batch = DecisionDataCollator(tokenizer.pad_token_id)(items[-3:])
    assert batch["marker_mask"].sum(-1).tolist() == [2, 2, 3]
    assert torch.allclose(batch["target"].sum(-1), torch.ones(3))


def test_holdout_takes_whole_rows_and_never_overshoots():
    items = [{"row": row} for row in range(50) for _ in range(3)]
    train, held = FastDecisionModel.split_holdout(items, 3407)
    assert len(held) == 15 and not {i["row"] for i in train} & {i["row"] for i in held}
    # One row with 100 decisions among single-decision rows: the target is 10.
    items = [{"row": row} for row, size in enumerate([1] * 9 + [100]) for _ in range(size)]
    for seed in range(20):
        train, held = FastDecisionModel.split_holdout(items, seed)
        assert len(held) <= 10 and len(train) >= 100
    single = [{"row": 0}, {"row": 0}]
    assert FastDecisionModel.split_holdout(single, 3407, fraction = 0.5) == (single, [])


def test_full_finetuning_trains_every_weight(checkpoint, capsys):
    model, _ = FastDecisionModel.from_pretrained(
        str(checkpoint), full_finetuning = True, use_gradient_checkpointing = False
    )
    assert all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())
    assert FastDecisionModel.get_peft_model(model) is model
    assert "Full finetuning is enabled" in capsys.readouterr().out
    assert not hasattr(model.encoder, "peft_config")
    with pytest.raises(NotImplementedError, match = "16-bit"):
        FastDecisionModel.from_pretrained(str(checkpoint), load_in_4bit = True)


def test_lora_freezes_a_16_bit_encoder_and_trains_the_head_in_fp32(checkpoint):
    model, _ = FastDecisionModel.from_pretrained(str(checkpoint), use_gradient_checkpointing = False)
    stored = torch.float32 if decision._device().type == "cpu" else torch.float16
    assert {p.dtype for p in model.encoder.parameters()} == {stored}
    assert not any(p.requires_grad for p in model.encoder.parameters())
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    assert head and all(p.requires_grad and p.dtype == torch.float32 for p in head)

    model, _ = FastDecisionModel.from_pretrained(
        str(checkpoint), dtype = torch.float16, use_gradient_checkpointing = False
    )
    model = FastDecisionModel.get_peft_model(model, r = 4, lora_alpha = 4)
    lora = [p for n, p in model.named_parameters() if "lora_" in n]
    assert lora and all(p.requires_grad and p.dtype == torch.float32 for p in lora)
    frozen = [p for n, p in model.encoder.named_parameters() if "lora_" not in n]
    assert all(not p.requires_grad and p.dtype == torch.float16 for p in frozen)
    assert all(b.dtype == torch.float32 for b in model.encoder.buffers() if b.is_floating_point())
    targets = {n.split(".")[-1] for n, m in model.encoder.named_modules() if hasattr(m, "lora_A")}
    assert targets == {"Wqkv", "Wo", "Wi"}
    with pytest.raises(RuntimeError, match = "already added LoRA"):
        FastDecisionModel.get_peft_model(model)


@pytest.mark.parametrize("lora", [True, False])
def test_head_learning_rate_and_weight_decay(checkpoint, tmp_path, lora):
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), full_finetuning = not lora, use_gradient_checkpointing = False
    )
    model = FastDecisionModel.get_peft_model(model, r = 4, lora_alpha = 4)
    items, _ = FastDecisionModel.build_dataset([_row(i) for i in range(4)], tokenizer, model)
    names = {id(p): n for n, p in model.named_parameters()}
    no_decay = {
        f"{prefix}.{name}"
        for prefix, module in model.named_modules()
        for name, _ in module.named_parameters(recurse = False)
        if isinstance(module, torch.nn.LayerNorm) or name.endswith("bias")
    }
    for head_learning_rate, head_lr in ((None, 1e-4), (0.0, 0.0)):
        trainer = DecisionTrainer(
            model = model,
            args = _args(tmp_path, learning_rate = 8e-4, weight_decay = 0.01),
            train_dataset = items,
            processing_class = tokenizer,
            head_learning_rate = head_learning_rate,
        )
        optimizer = trainer.create_optimizer()
        grouped = [
            (names[id(p)], group) for group in optimizer.param_groups for p in group["params"]
        ]
        assert {name for name, _ in grouped} == {
            n for n, p in model.named_parameters() if p.requires_grad
        }
        for name, group in grouped:
            assert group["lr"] == (8e-4 if name.startswith("encoder.") else head_lr)
            assert group["weight_decay"] == (0.0 if name in no_decay else 0.01)


def test_length_grouping_only_with_gradient_accumulation(checkpoint, tmp_path):
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), use_gradient_checkpointing = False
    )
    items, _ = FastDecisionModel.build_dataset([_row(i) for i in range(8)], tokenizer, model)
    for accumulation in (1, 4):
        trainer = DecisionTrainer(
            model = model,
            args = _args(tmp_path, gradient_accumulation_steps = accumulation),
            train_dataset = items,
            processing_class = tokenizer,
        )
        assert isinstance(trainer._get_train_sampler(), _LengthGroupedBatches) == (accumulation > 1)


def test_length_grouped_batches_mix_lengths_and_replay_an_epoch():
    lengths = [10] * 64 + [500] * 64
    sampler = _LengthGroupedBatches(lengths, 8, 3407)
    order = list(sampler)
    assert sorted(order) == list(range(128))
    batches = [order[i : i + 8] for i in range(0, 128, 8)]
    assert all(len({lengths[i] for i in batch}) == 1 for batch in batches)
    assert lengths[batches[0][0]] == 500
    assert {lengths[i] for batch in batches[:8] for i in batch} == {10, 500}
    # The order depends only on the epoch, so a resumed run replays the epoch it stopped in.
    assert list(sampler) == order
    sampler.set_epoch(1)
    assert list(sampler) != order
    sampler.set_epoch(0)
    assert list(sampler) == order


@pytest.mark.parametrize("lora", [True, False])
def test_train_calibrate_save_and_serve(checkpoint, tmp_path, lora):
    from safetensors.torch import load_file

    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), full_finetuning = not lora, use_gradient_checkpointing = False
    )
    model = FastDecisionModel.get_peft_model(model, r = 4, lora_alpha = 4)
    items, _ = FastDecisionModel.build_dataset([_row(i) for i in range(80)], tokenizer, model)
    train, held = FastDecisionModel.split_holdout(items, 3407, fraction = 0.5)
    midway = {}

    class SaveMidway(TrainerCallback):
        def on_step_end(
            self,
            args,
            state,
            control,
            model = None,
            **kwargs,
        ):
            if state.global_step == 2:
                model.save_pretrained_merged(tmp_path / "midway")
                midway.update(
                    {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                )

    def accuracy(prediction):
        hits = prediction.predictions.argmax(-1) == prediction.label_ids.argmax(-1)
        return {"accuracy": float(hits.mean())}

    trainer = DecisionTrainer(
        model = model,
        args = _args(tmp_path),
        train_dataset = train,
        eval_dataset = held,
        tokenizer = tokenizer,
        compute_metrics = accuracy,
        callbacks = [SaveMidway()],
    )
    trainer.train()
    # Saving midway kept the adapters on the model, so the steps after it still trained them.
    assert decision.is_decision_checkpoint(tmp_path / "midway")
    after = dict(model.named_parameters())
    assert set(midway) <= set(after)
    assert any(not torch.equal(after[n], p) for n, p in midway.items() if n.startswith("encoder."))

    metrics = trainer.evaluate()
    assert metrics["eval_loss"] > 0 and 0 <= metrics["eval_accuracy"] <= 1
    assert trainer.predict(held).predictions.shape == (len(held), 3)

    calibrated = FastDecisionModel.calibrate(model, tokenizer, held)
    assert calibrated["fitted_types"] == [0, 1, 2] and 0 <= calibrated["accuracy"] <= 1
    assert model.decision_config["temperature"] != [1.2, 1.1, 1.3]
    assert "temperature_by_options" not in model.decision_config

    model.decision_config["training"] = {"steps": 4}
    model.save_pretrained_merged(tmp_path / "out")
    saved = json.loads((tmp_path / "out" / "rl_agent_config.json").read_text())
    assert saved["fine_tuned"] is True and saved["training"] == {"steps": 4}
    assert saved["temperature"] == model.decision_config["temperature"]
    encoder_config = (checkpoint / "encoder" / "config.json").read_bytes()
    assert (tmp_path / "out" / "encoder" / "config.json").read_bytes() == encoder_config
    agent = laya.load(str(tmp_path / "out"), device = "cpu")
    assert agent.predict("the server is down again", QUESTIONS)["answers"]["team"]["choice"] in (
        "outage",
        "billing",
    )

    FastDecisionModel.for_inference(model)
    assert not model.training
    FastDecisionModel.for_training(model)
    assert model.training
    weights = load_file(str(tmp_path / "out" / "model.safetensors"))
    merged = model.encoder.merge_and_unload() if lora else model.encoder
    expected = {f"encoder.{k}": v for k, v in merged.state_dict().items()}
    expected.update((k, v) for k, v in model.state_dict().items() if not k.startswith("encoder."))
    assert weights.keys() == expected.keys()
    assert all(torch.equal(weights[k], v.detach().cpu().half()) for k, v in expected.items())


def test_toy_task_beats_the_base_model(checkpoint, tmp_path):
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), use_gradient_checkpointing = False
    )
    model = FastDecisionModel.get_peft_model(model, r = 8, lora_alpha = 16)
    items, _ = FastDecisionModel.build_dataset([_row(i) for i in range(120)], tokenizer, model)
    train, held = FastDecisionModel.split_holdout(items, 3407, fraction = 0.25)
    base = FastDecisionModel.evaluate(model, tokenizer, held)
    DecisionTrainer(
        model = model,
        args = _args(tmp_path, max_steps = 80, learning_rate = 5e-3, per_device_train_batch_size = 16),
        train_dataset = train,
        processing_class = tokenizer,
        head_learning_rate = 5e-3,
    ).train()
    tuned = FastDecisionModel.evaluate(model, tokenizer, held)
    # The random base can already score high on some transformers versions; 1.0 is the ceiling.
    assert tuned["accuracy"] >= min(base["accuracy"] + 0.25, 1.0)
    assert tuned["loss"] < base["loss"] / 2


def test_trainer_leaves_the_callers_arguments_alone(checkpoint, tmp_path, monkeypatch):
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), use_gradient_checkpointing = False
    )
    items, _ = FastDecisionModel.build_dataset([_row(i) for i in range(4)], tokenizer, model)
    args = _args(tmp_path, max_steps = 1, gradient_checkpointing = True)
    trainer = DecisionTrainer(model = model, args = args, train_dataset = items)
    assert args.remove_unused_columns is True and args.label_names is None
    assert args.gradient_checkpointing is True
    trainer.train()
    assert model.encoder.is_gradient_checkpointing
    # Without arguments, as Trainer allows.
    monkeypatch.chdir(tmp_path)
    trainer = DecisionTrainer(model = model, train_dataset = items, processing_class = tokenizer)
    batch = next(iter(trainer.get_train_dataloader()))
    assert {"marker_pos", "marker_mask", "target"} <= set(batch)


def test_for_training_turns_gradient_checkpointing_back_on(checkpoint):
    model, _ = FastDecisionModel.from_pretrained(str(checkpoint), use_gradient_checkpointing = False)
    assert not model.encoder.is_gradient_checkpointing
    FastDecisionModel.for_training(model)
    assert model.training and model.encoder.is_gradient_checkpointing
    FastDecisionModel.for_training(model, use_gradient_checkpointing = False)
    assert not model.encoder.is_gradient_checkpointing


def test_trainer_uses_one_gpu_and_the_set_batch_on_a_multi_gpu_machine(checkpoint, tmp_path):
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), use_gradient_checkpointing = False
    )
    args = _args(tmp_path, max_steps = 1)
    assert args.n_gpu <= 1
    args._n_gpu = 2
    trainer = DecisionTrainer(model = model, args = args, processing_class = tokenizer)
    assert trainer.args.n_gpu == 1 and args.n_gpu == 2
    assert trainer.args.train_batch_size == args.per_device_train_batch_size
    assert trainer._wrap_model(model) is model


def test_save_refuses_weights_float16_cannot_hold(checkpoint, tmp_path):
    model, _ = FastDecisionModel.from_pretrained(
        str(checkpoint), full_finetuning = True, use_gradient_checkpointing = False
    )
    model.save_pretrained_merged(tmp_path / "out")
    with torch.no_grad():
        model.scorer[1].weight[0, 0] = 7e4
    with pytest.raises(ValueError, match = "float16"):
        model.save_pretrained_merged(tmp_path / "out")
    assert decision.is_decision_checkpoint(tmp_path / "out")


def test_push_to_hub_merged_uploads_a_complete_checkpoint(checkpoint, monkeypatch):
    import huggingface_hub

    pushed = {}

    class Api:
        def __init__(self, token = None):
            pushed["token"] = token

        def create_repo(
            self,
            repo_id,
            private = None,
            exist_ok = False,
        ):
            pushed["private"] = private
            return types.SimpleNamespace(repo_id = f"me/{repo_id}")

        def upload_folder(self, folder_path, repo_id, **kwargs):
            pushed["repo_id"] = repo_id
            pushed["complete"] = decision.is_decision_checkpoint(folder_path)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    model, _ = FastDecisionModel.from_pretrained(str(checkpoint), use_gradient_checkpointing = False)
    model.push_to_hub_merged("laya-ft", token = "hf_token", private = True)
    assert pushed == {
        "token": "hf_token",
        "private": True,
        "repo_id": "me/laya-ft",
        "complete": True,
    }


def test_gradient_accumulation_matches_one_large_batch(checkpoint, tmp_path):
    grads = {}

    class Grab(TrainerCallback):
        def __init__(self, key):
            self.key = key

        def on_pre_optimizer_step(
            self,
            args,
            state,
            control,
            model = None,
            **kwargs,
        ):
            grads[self.key] = torch.cat(
                [p.grad.flatten() for p in model.parameters() if p.grad is not None]
            )

    for lora in (False, True):
        for accumulation in (1, 4):
            model, tokenizer = FastDecisionModel.from_pretrained(
                str(checkpoint),
                full_finetuning = not lora,
                dtype = torch.float32,
                use_gradient_checkpointing = False,
            )
            model = FastDecisionModel.get_peft_model(model, r = 4, lora_alpha = 4)
            model.eval()
            items, _ = FastDecisionModel.build_dataset(
                [_row(i) for i in range(16)], tokenizer, model
            )
            trainer = DecisionTrainer(
                model = model,
                args = _args(
                    tmp_path,
                    per_device_train_batch_size = 48 // accumulation,
                    gradient_accumulation_steps = accumulation,
                    max_steps = 1,
                    max_grad_norm = 0.0,
                    use_cpu = True,
                ),
                train_dataset = items,
                processing_class = tokenizer,
                callbacks = [Grab((lora, accumulation))],
            )
            trainer.model.train = lambda mode = True: trainer.model
            trainer.train()
        torch.testing.assert_close(grads[(lora, 1)], grads[(lora, 4)], rtol = 1e-3, atol = 1e-6)


def test_rope_parameters_build_the_trained_rope_on_every_transformers(tmp_path):
    from transformers import AutoModel

    def build(name, **rope):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(_tiny_encoder_config(**rope)))
        return AutoModel.from_config(decision._encoder_config(folder), attn_implementation = "sdpa")

    def rope_parameters(sliding):
        return {
            "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": sliding, "rope_type": "default"},
        }

    # How transformers 5 saves mmBERT, the same with both key styles, and with the 10000 default.
    saved = build("saved", rope_parameters = rope_parameters(160000.0))
    both = build(
        "both",
        rope_parameters = rope_parameters(160000.0),
        global_rope_theta = 160000.0,
        local_rope_theta = 160000.0,
    )
    default = build(
        "default",
        rope_parameters = rope_parameters(10000.0),
        global_rope_theta = 160000.0,
        local_rope_theta = 10000.0,
    )
    with torch.no_grad():
        # Random weights attend almost uniformly, which hides the rotary angles; sharpen the attention.
        for name, weight in saved.named_parameters():
            if name.endswith("Wqkv.weight"):
                weight.mul_(10)
    for model in (both, default):
        model.load_state_dict(saved.state_dict())
    ids = torch.randint(5, 64, (2, 24))
    outputs = [model.eval()(input_ids = ids).last_hidden_state for model in (saved, both, default)]
    torch.testing.assert_close(outputs[0], outputs[1])
    assert not torch.allclose(outputs[0], outputs[2], atol = 1e-2)
