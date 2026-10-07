# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import copy
import json
import math
import sys
import types
from pathlib import Path

import pytest
from real_accelerator import has_real_cuda

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


def test_long_runs_compile_the_encoder_layers_and_leave_them_eager(
    checkpoint, tmp_path, monkeypatch
):
    from unsloth.models import _decision_fast

    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), use_gradient_checkpointing = False
    )
    model = FastDecisionModel.get_peft_model(model, r = 4, lora_alpha = 8)
    items, _ = FastDecisionModel.build_dataset([_row(i) for i in range(8)], tokenizer, model)
    compiled, static = [], []
    monkeypatch.setattr(torch.nn.Module, "compile", lambda self, **kw: compiled.append((self, kw)))
    monkeypatch.setattr(_decision_fast, "_compile_static", lambda fn: static.append(fn) or fn)
    layers = _decision_fast._encoder_layers(model)
    assert len(layers) == 2
    trainer = DecisionTrainer(model = model, args = _args(tmp_path, max_steps = 1), train_dataset = items)
    # Short runs and CPUs stay eager; UNSLOTH_DECISION_COMPILE=1 forces it, =0 refuses it.
    monkeypatch.delenv("UNSLOTH_DECISION_COMPILE", raising = False)
    trainer.train()
    assert not compiled
    assert _decision_fast._wants_compile(model, 10**6) == next(model.parameters()).is_cuda
    monkeypatch.setattr(_decision_fast, "_on_gpu", lambda model: True)
    assert _decision_fast._wants_compile(model, _decision_fast.COMPILE_MIN_FORWARDS)
    assert not _decision_fast._wants_compile(model, _decision_fast.COMPILE_MIN_FORWARDS - 1)
    monkeypatch.setenv("UNSLOTH_DECISION_COMPILE", "1")
    trainer.train()
    # The test decisions are short, so each layer compiles once for static 64-token buckets.
    assert not compiled and len(static) == len(layers)
    assert all(layer._compiled_call_impl is None for layer in layers)
    monkeypatch.setattr(_decision_fast, "STATIC_MAX_LEN", 0)
    trainer.train()
    assert [m for m, _ in compiled] == layers and all(kw == {"dynamic": True} for _, kw in compiled)
    assert all(layer._compiled_call_impl is None for layer in layers)
    monkeypatch.setenv("UNSLOTH_DECISION_COMPILE", "0")
    assert not _decision_fast._wants_compile(model, 10**6)


def test_a_failing_compile_trains_eagerly(checkpoint, tmp_path, monkeypatch):
    from unsloth.models import _decision_fast as fast

    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), use_gradient_checkpointing = False
    )
    model = FastDecisionModel.get_peft_model(model, r = 4, lora_alpha = 8)
    items, _ = FastDecisionModel.build_dataset([_row(i) for i in range(8)], tokenizer, model)

    def broken(*args, **kwargs):
        raise RuntimeError("inductor cannot serve this platform")

    def compile(self, **kwargs):
        self._compiled_call_impl = broken

    monkeypatch.setattr(torch.nn.Module, "compile", compile)
    monkeypatch.setattr(fast, "_compile_static", lambda fn: broken)
    monkeypatch.setenv("UNSLOTH_DECISION_COMPILE", "1")
    trainer = DecisionTrainer(model = model, args = _args(tmp_path, max_steps = 2), train_dataset = items)
    trainer.train()
    assert trainer.state.global_step == 2
    assert model._unsloth_decision_compiled is False
    assert all(layer._compiled_call_impl is None for layer in fast._encoder_layers(model))
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_compiled_layers_checkpoint_with_torch_and_get_unsloths_back(
    checkpoint, tmp_path, monkeypatch
):
    import functools

    from unsloth.models import _decision_fast as fast

    torch_checkpoint = fast._torch_checkpoint()
    assert torch_checkpoint.__module__ == "torch.utils.checkpoint"

    model, _ = FastDecisionModel.from_pretrained(str(checkpoint), use_gradient_checkpointing = False)
    model = FastDecisionModel.get_peft_model(model, r = 4, lora_alpha = 8)
    layers = fast._encoder_layers(model)

    def unsloth_checkpoint(*args, **kwargs):
        raise AssertionError("compiled layers must not use Unsloth's checkpoint")

    offloaded = functools.partial(unsloth_checkpoint, use_reentrant = True)
    # unsloth_zoo patches torch.utils.checkpoint.checkpoint once a model loads with its checkpointing.
    monkeypatch.setattr(torch.utils.checkpoint, "_old_checkpoint", torch_checkpoint, raising = False)
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", unsloth_checkpoint)
    for layer in layers:
        layer._gradient_checkpointing_func = offloaded
    monkeypatch.setattr(torch.nn.Module, "compile", lambda self, **kw: None)
    monkeypatch.setattr(fast, "_wants_compile", lambda model, forwards: True)
    monkeypatch.setattr(fast, "_warm_up", lambda model, amp_dtype: None)
    with fast.compiled_encoder(model, 10**6) as compiled:
        assert compiled
        for layer in layers:
            func = layer._gradient_checkpointing_func
            assert func.func is torch_checkpoint and func.keywords == {"use_reentrant": False}
    assert all(layer._gradient_checkpointing_func is offloaded for layer in layers)

    def broken(model, amp_dtype):
        raise RuntimeError("inductor cannot serve this platform")

    monkeypatch.setattr(fast, "_warm_up", broken)
    with fast.compiled_encoder(model, 10**6) as compiled:
        assert not compiled
        assert all(layer._gradient_checkpointing_func is offloaded for layer in layers)

    # torch.compile itself can refuse (an unsupported Python): the second layer raises here.
    def refuse(self, **kwargs):
        if self is layers[1]:
            raise RuntimeError("Dynamo is not supported on this Python")
        self._compiled_call_impl = lambda *a, **k: None

    monkeypatch.setattr(torch.nn.Module, "compile", refuse)
    monkeypatch.setattr(fast, "_warm_up", lambda model, amp_dtype: None)
    with fast.compiled_encoder(model, 10**6) as compiled:
        assert not compiled
        assert all(layer._compiled_call_impl is None for layer in layers)
        assert all(layer._gradient_checkpointing_func is offloaded for layer in layers)


def test_a_resized_laya_vocabulary_saves_and_reloads(checkpoint, tmp_path):
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), use_gradient_checkpointing = False
    )
    grown = model.encoder.get_input_embeddings().num_embeddings + 8
    # The fixture's vocab equals its hidden size, so the covariance mean resizing samples from
    # is singular and can come out NaN on some GPUs.
    model.encoder.resize_token_embeddings(grown, mean_resizing = False)
    model.save_pretrained_merged(str(tmp_path / "out"), tokenizer)
    reloaded, _ = FastDecisionModel.from_pretrained(str(tmp_path / "out"))
    assert reloaded.encoder.get_input_embeddings().num_embeddings == grown


def test_full_clef_finetuning_never_quantizes_the_backbone(tmp_path, monkeypatch):
    from unsloth.models import loader

    seen = []

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        seen.append(kwargs)
        raise Captured

    monkeypatch.setattr(decision, "_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(loader.FastModel, "from_pretrained", capture)
    for full in (True, False):
        with pytest.raises(Captured):
            decision._load_clef(tmp_path, None, torch.bfloat16, True, full, None, False, {})
    assert seen[0].get("quantization_config") is None
    assert seen[1].get("quantization_config") is not None


def test_the_compile_warm_up_leaves_the_rng_where_it_was(checkpoint, monkeypatch):
    from unsloth.models import _decision_fast as fast

    model, _ = FastDecisionModel.from_pretrained(str(checkpoint), use_gradient_checkpointing = False)
    monkeypatch.setattr(torch.nn.Module, "compile", lambda self, **kw: None)
    monkeypatch.setattr(fast, "_wants_compile", lambda model, forwards: True)
    torch.manual_seed(0)
    before = torch.get_rng_state()
    with fast.compiled_encoder(model, 10**6) as compiled:
        assert compiled
        assert torch.equal(torch.get_rng_state(), before)


def test_the_compile_warm_up_forks_the_rng_of_the_models_own_accelerator(monkeypatch):
    from unsloth.models import _decision_fast as fast

    forked = []

    class Forked(Exception):
        pass

    def fork_rng(**kwargs):
        forked.append(kwargs)
        raise Forked

    model = types.SimpleNamespace(
        training = True,
        parameters = lambda: iter([torch.empty(1, device = "meta")]),
        train = lambda mode = True: None,
        encoder = types.SimpleNamespace(config = types.SimpleNamespace(vocab_size = 100)),
    )
    monkeypatch.setattr(torch.random, "fork_rng", fork_rng)
    with pytest.raises(Forked):
        fast._warm_up(model, None)
    # An XPU / MPS model forks its own RNG, not CUDA's (meta stands in for any accelerator).
    assert forked == [{"devices": [0], "device_type": "meta"}]


def test_clef_backbone_stays_on_one_device_unless_the_caller_places_it(tmp_path, monkeypatch):
    from unsloth.models import loader, loader_utils

    seen = []

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        seen.append(kwargs.get("device_map"))
        raise Captured

    monkeypatch.setattr(decision, "_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(loader.FastModel, "from_pretrained", capture)
    index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    for kwargs in ({}, {"device_map": None}, {"device_map": "auto"}):
        with pytest.raises(Captured):
            decision._load_clef(tmp_path, None, torch.bfloat16, False, False, None, False, kwargs)
    monkeypatch.setattr(loader_utils, "prepare_device_map", lambda: ({"": "cuda:1"}, True))
    with pytest.raises(Captured):
        decision._load_clef(tmp_path, None, torch.bfloat16, False, False, None, False, {})
    assert seen == [{"": f"cuda:{index}"}, {"": f"cuda:{index}"}, "auto", {"": "cuda:1"}]


def test_toy_task_beats_the_base_model(checkpoint, tmp_path, monkeypatch):
    # On CPU everywhere: the toy task plateaus near loss 0.45 and leaves it by step ~70 on CPU but
    # only after ~100 steps on a GPU (same curve otherwise, any precision), so 80 steps is a threshold
    # tuned to CPU arithmetic, not a GPU defect.
    monkeypatch.setattr(decision, "_device", lambda: torch.device("cpu"))
    # Eager too: a compiled encoder draws the same accuracy from a different 45-decision RNG stream.
    monkeypatch.setenv("UNSLOTH_DECISION_COMPILE", "0")
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), use_gradient_checkpointing = False
    )
    model = FastDecisionModel.get_peft_model(model, r = 8, lora_alpha = 16)
    items, _ = FastDecisionModel.build_dataset([_row(i) for i in range(120)], tokenizer, model)
    train, held = FastDecisionModel.split_holdout(items, 3407, fraction = 0.25)
    base = FastDecisionModel.evaluate(model, tokenizer, held)
    DecisionTrainer(
        model = model,
        args = _args(
            tmp_path,
            max_steps = 80,
            learning_rate = 5e-3,
            per_device_train_batch_size = 16,
            use_cpu = True,
        ),
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
    args._n_gpu = 2
    trainer = DecisionTrainer(model = model, args = args, processing_class = tokenizer)
    assert trainer.args.n_gpu == 1 and args.n_gpu == 2
    assert trainer.args.train_batch_size == args.per_device_train_batch_size
    assert trainer._wrap_model(model) is model


@pytest.mark.skipif(not has_real_cuda(), reason = "measures GPU memory")
def test_lora_save_does_not_copy_the_encoder_on_the_gpu(checkpoint, tmp_path):
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), use_gradient_checkpointing = False
    )
    model = FastDecisionModel.get_peft_model(model, r = 8, lora_alpha = 16)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    model.save_pretrained_merged(tmp_path / "out", tokenizer)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() <= before
    assert next(model.encoder.parameters()).is_cuda and hasattr(model.encoder, "peft_config")


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


TINY_QWEN3_5 = "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration"
CLEF_HEAD = {
    "hidden_size": 16,
    "width": 32,
    "routing_layers": 1,
    "layers": 1,
    "heads": 4,
    "feedforward": 64,
}


def _clef_reference():
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download("Cloudflare/clef-flash", "joint_schema_model.py")
    except Exception as exc:
        pytest.skip(f"Clef reference code unavailable: {exc}")
    import importlib.util

    spec = importlib.util.spec_from_file_location("clef_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


@pytest.fixture
def clef_checkpoint(tmp_path):
    import shutil

    from safetensors.torch import save_file
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    from unsloth.models.clef import JointSchemaHead

    if not has_real_cuda():
        try:
            import causal_conv1d  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("transformers calls causal_conv1d's CUDA-only kernel on CPU tensors")
    reference, path = _clef_reference()
    folder = tmp_path / "clef"
    torch.manual_seed(0)
    Qwen3_5ForConditionalGeneration.from_pretrained(
        TINY_QWEN3_5, dtype = torch.bfloat16
    ).save_pretrained(str(folder))
    AutoProcessor.from_pretrained(TINY_QWEN3_5).save_pretrained(str(folder))
    head = JointSchemaHead(**CLEF_HEAD)
    with torch.no_grad():
        for param in head.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    save_file(
        {k: v.to(torch.bfloat16).contiguous() for k, v in head.state_dict().items()},
        str(folder / "joint_head.safetensors"),
    )
    (folder / "joint_head_config.json").write_text(json.dumps(CLEF_HEAD))
    shutil.copyfile(path, folder / "joint_schema_model.py")
    return folder


def _clef_rows(n):
    rows = [_row(i) for i in range(n)]
    for row in rows:
        row["questions"] = {
            **QUESTIONS,
            "tags": {"type": "choice", "criteria": ["billing", "outage", "other"]},
        }
        row["gold"]["tags"] = row["gold"]["team"]["label"]
    return rows


def test_clef_encoding_is_token_identical_to_cloudflares():
    from transformers import AutoTokenizer

    from unsloth.models.clef import encode_record

    reference, _ = _clef_reference()
    tokenizer = AutoTokenizer.from_pretrained(TINY_QWEN3_5)
    for record in (
        {"state": {"invoice": {"total": 1250.0, "vendor": "Acmé"}}, "questions": QUESTIONS},
        {"state": "plain text", "questions": {"x": {"type": "noul", "criteria": {"true": "yes"}}}},
        {
            "state": "s" * 500,
            "questions": {"lvl": {"type": "score", "instructions": "", "criteria": [1, "b"]}},
        },
    ):
        for max_length in (100, 400, 16384):
            if max_length == 100:
                with pytest.raises(ValueError):
                    reference.encode_record(tokenizer, record, max_length = max_length)
                with pytest.raises(ValueError):
                    encode_record(tokenizer, record, max_length = max_length)
                continue
            ours = encode_record(tokenizer, record, max_length = max_length)
            theirs = reference.encode_record(tokenizer, record, max_length = max_length)
            assert ours.input_ids == theirs.input_ids
            assert [
                (q.question_id, q.question_type, q.question_span, q.option_spans, q.option_ids)
                for q in ours.questions
            ] == [
                (q.question_id, q.question_type, q.question_span, q.option_spans, q.option_ids)
                for q in theirs.questions
            ]


@pytest.mark.parametrize(
    "option",
    [
        {"use_dora": True},
        {"layers_to_transform": [0]},
        {"layers_pattern": "layers"},
        {"loftq_config": {"loftq_bits": 4}},
        {"init_lora_weights": "gaussian"},
    ],
)
def test_clef_lora_refuses_options_it_would_drop(monkeypatch, option):
    monkeypatch.setattr(decision, "_clef_peft_model", lambda model, **kwargs: "plain lora")
    model = types.SimpleNamespace(is_clef = True, encoder = types.SimpleNamespace())
    with pytest.raises(NotImplementedError, match = next(iter(option))):
        FastDecisionModel.get_peft_model(model, **option)
    assert FastDecisionModel.get_peft_model(model) == "plain lora"


def test_clef_metrics_score_in_the_runs_batch_size(monkeypatch):
    seen = []

    def clef_logits(
        model,
        items,
        pad_token_id,
        batch_size = 4,
    ):
        seen.append(batch_size)
        raise StopIteration

    monkeypatch.setattr(decision, "_clef_logits", clef_logits)
    model = types.SimpleNamespace(is_clef = True, decision_config = {})
    tokenizer = types.SimpleNamespace(pad_token_id = 0)
    for call in (FastDecisionModel.evaluate, FastDecisionModel.calibrate):
        with pytest.raises(StopIteration):
            call(model, tokenizer, [], batch_size = 1)
    with pytest.raises(StopIteration):
        FastDecisionModel.evaluate(model, tokenizer, [])
    assert seen == [1, 1, 4]


def test_clef_loads_scores_like_cloudflares_model_and_trains(clef_checkpoint, tmp_path):
    reference, _ = _clef_reference()
    model, processor = FastDecisionModel.from_pretrained(str(clef_checkpoint), max_seq_length = 512)
    assert getattr(model, "is_clef", False) and model.head.hidden_norm.weight.dtype == torch.float32
    device = next(model.parameters()).device
    backbone_dtype = next(model.encoder.parameters()).dtype
    released, _ = reference.load_release_model(
        str(clef_checkpoint), device = device, dtype = backbone_dtype
    )
    record = {"state": "the server is down again", "questions": QUESTIONS}
    encoded = reference.encode_record(processor.tokenizer, record)
    batch = reference.collate_records([encoded], processor.tokenizer.pad_token_id, device)
    with torch.no_grad():
        theirs = [z.float() for z in released(batch)[0]]
        ours, _ = model(batch["input_ids"], batch["attention_mask"], batch["records"])
        fp32_head = model.head
        model.head = copy.deepcopy(fp32_head).to(backbone_dtype)
        ours_same_dtype, _ = model(batch["input_ids"], batch["attention_mask"], batch["records"])
        model.head = fp32_head
    for row, z in enumerate(theirs):
        same = ours_same_dtype[row, : len(z)].float()
        assert torch.allclose(same, z, atol = 2e-2, rtol = 2e-2), (row, (same - z).abs().max(), same, z)
        fp32 = ours[row, : len(z)].float()
        assert torch.allclose(fp32, z, atol = 0.1), (row, (fp32 - z).abs().max(), fp32, z)

    items, report = FastDecisionModel.build_dataset(_clef_rows(64), processor, model)
    assert report["skipped"] == 0 and len(items) == 64 and len(items[0]["targets"]) == 4
    train, holdout = FastDecisionModel.split_holdout(items, fraction = 0.25)
    assert sum(len(i["labels"]) for i in holdout) <= 64
    before = FastDecisionModel.evaluate(model, processor, holdout)
    model = FastDecisionModel.get_peft_model(model, r = 8, lora_alpha = 8)
    losses = []

    class Losses(TrainerCallback):
        def on_log(
            self,
            args,
            state,
            control,
            logs = None,
            **kwargs,
        ):
            if logs and "loss" in logs:
                losses.append(logs["loss"])

    trainer = DecisionTrainer(
        model = model,
        tokenizer = processor,
        train_dataset = train,
        args = _args(tmp_path, max_steps = 30, learning_rate = 5e-3, logging_steps = 1),
        head_learning_rate = 5e-3,
        callbacks = [Losses()],
    )
    trainer.train()
    after = FastDecisionModel.evaluate(model, processor, holdout)
    assert min(losses[-5:]) < losses[0] and after["loss"] < before["loss"], (losses, before, after)
    calibration = FastDecisionModel.calibrate(model, processor, holdout)
    assert "accuracy" in calibration
    # Fitted to the gold labels, so the calibrated confidence tracks being right.
    head_temperature = model.decision_config["head_temperature"]
    assert (
        decision.HEAD_TEMPERATURE_RANGE[0] <= head_temperature <= decision.HEAD_TEMPERATURE_RANGE[1]
    )

    model.decision_config["training"] = {"steps": 30}
    model.save_pretrained_merged(str(tmp_path / "out"))
    assert (tmp_path / "out" / "joint_schema_model.py").is_file()
    reloaded, _ = FastDecisionModel.from_pretrained(str(tmp_path / "out"), max_seq_length = 512)
    # The parent run's record stays on disk but does not describe the next fine-tune.
    assert "training" not in reloaded.decision_config
    # The calibrated temperature over all questions is folded into the saved head; the per-type
    # temperatures are relative to it, so the reload serves the same probabilities.
    folded = reloaded.decision_config["folded_temperature"]
    assert folded == pytest.approx(model.decision_config["head_temperature"])
    assert "head_temperature" not in reloaded.decision_config
    assert reloaded.decision_config["temperature"] == model.decision_config["temperature"]
    released, _ = reference.load_release_model(
        str(tmp_path / "out"), device = device, dtype = torch.float32
    )
    with torch.no_grad():
        trained, _ = model(batch["input_ids"], batch["attention_mask"], batch["records"])
        again, _ = reloaded(batch["input_ids"], batch["attention_mask"], batch["records"])
        theirs = released(batch)[0]
    # bf16 reload rounding grows with the per-run logit scale, so the bound is relative (10 B200 runs
    # peaked at 1.6%); a fold left out or doubled is off by >= 20%.
    mask = trained > -1e3
    expected = (trained / folded).float()[mask]
    error = (again.float()[mask] - expected).abs().max().item()
    scale = expected.abs().max().item()
    assert error <= 0.03 * scale + 0.05, (error, scale, folded)
    for row, z in enumerate(theirs):
        assert int(z.argmax()) == int(again[row, : len(z)].argmax()), (row, z, again[row, : len(z)])


@pytest.mark.skipif(not has_real_cuda(), reason = "the fast kernels need a CUDA device")
def test_clef_backbone_runs_the_compiled_gated_delta_and_conv_kernels(clef_checkpoint):
    from transformers.utils.import_utils import is_causal_conv1d_available

    model, _ = FastDecisionModel.from_pretrained(str(clef_checkpoint), max_seq_length = 512)
    from transformers.utils.import_utils import is_flash_linear_attention_available

    if not is_flash_linear_attention_available():
        pytest.skip("flash-linear-attention is disabled on this GPU / Triton")
    layers = [m for m in model.encoder.modules() if type(m).__name__.endswith("GatedDeltaNet")]
    assert layers
    for layer in layers:
        forward = type(layer).forward
        assert forward.__code__.co_filename.endswith("unsloth_compiled_module_qwen3_5.py")
        delta = _chosen_kernel(
            layer, forward, "torch_chunk_gated_delta_rule", "chunk_gated_delta_rule"
        )
        conv = _chosen_kernel(layer, forward, "causal_conv1d_fn", "causal_conv1d_fn")
        assert delta is not None and ".fla." in "." + delta, delta
        if is_causal_conv1d_available():
            assert conv is not None and conv.startswith("causal_conv1d"), conv


def _chosen_kernel(layer, forward, global_name, attr):
    # transformers < 5.6 stores the picked kernel on the layer; later versions freeze it in a
    # closure under the hub-kernel wrappers of the module-level function.
    fn = vars(layer)[attr] if attr in vars(layer) else forward.__globals__.get(global_name)
    top = fn
    while fn is not None and hasattr(fn, "__code__"):
        cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))
        if "implementation" in cells:
            return cells["implementation"].cell_contents.__module__
        fn = getattr(fn, "__wrapped__", None)
    return getattr(top, "__module__", None)


def test_decision_loss_recipe_terms():
    torch.manual_seed(0)
    logits = torch.randn(5, 4)
    mask = torch.tensor([[1, 1, 1, 0]] * 3 + [[1, 1, 0, 0]] * 2, dtype = torch.bool)
    target = torch.zeros(5, 4)
    target[torch.arange(5), torch.tensor([0, 2, 1, 1, 0])] = 1.0
    ordinal = torch.tensor([True, True, False, False, False])
    plain = decision._soft_cross_entropy(logits, target, mask)
    assert torch.equal(decision._decision_loss(logits, target, mask, ordinal), plain)

    log_p = torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
    p = log_p.exp()
    k = mask.sum(-1, keepdim = True).float()
    smoothed = 0.9 * target + 0.1 * mask.float() / k
    expected_ce = -(smoothed * log_p).sum(-1).mean()
    expected_brier = ((p - target) ** 2 * mask).sum(-1).mean()
    levels = torch.arange(4.0)
    gold = target.argmax(-1).float()
    distance = (p * (levels - gold[:, None]).abs()).sum(-1) / (k[:, 0] - 1)
    expected_ordinal = distance[ordinal].mean()
    got = decision._decision_loss(
        logits,
        target,
        mask,
        ordinal,
        label_smoothing = 0.1,
        brier_weight = 0.5,
        ordinal_weight = 2.0,
    )
    torch.testing.assert_close(got, expected_ce + 0.5 * expected_brier + 2.0 * expected_ordinal)


def test_record_accuracy_needs_every_question_of_a_row_right():
    logits = [torch.tensor([2.0, 0.0]), torch.tensor([0.0, 2.0]), torch.tensor([2.0, 0.0])]
    items = [
        {"label": 0, "target": [1.0, 0.0], "row": 0, "qtype": 0},
        {"label": 0, "target": [1.0, 0.0], "row": 0, "qtype": 0},
        {"label": 0, "target": [1.0, 0.0], "row": 1, "qtype": 0},
    ]
    metrics = decision._metrics(logits, items, [1.0] * 3)
    assert metrics["accuracy"] == pytest.approx(2 / 3) and metrics["record_accuracy"] == 0.5


def _clef_head_inputs():
    from transformers import AutoTokenizer

    from unsloth.models.clef import JointSchemaHead, encode_record

    tokenizer = AutoTokenizer.from_pretrained(TINY_QWEN3_5)
    encoded = [
        encode_record(tokenizer, {"state": "state " * (i + 1), "questions": QUESTIONS})
        for i in range(2)
    ]
    batch = decision.ClefDataCollator(tokenizer.pad_token_id)(
        [
            {"input_ids": list(r.input_ids), "record": r, "targets": [[1.0]], "qtypes": [0]}
            for r in encoded
        ]
    )
    torch.manual_seed(0)
    head = JointSchemaHead(**CLEF_HEAD)
    with torch.no_grad():
        for param in head.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    hidden = torch.randn(*batch["input_ids"].shape, CLEF_HEAD["hidden_size"])
    lexical = torch.randn(int(batch["input_ids"].max()) + 1, CLEF_HEAD["hidden_size"])

    def run(module):
        with torch.no_grad():
            return [
                z
                for record in module(
                    hidden, batch["input_ids"], batch["attention_mask"], batch["records"], lexical
                )
                for z in record
            ]

    return head, run


@pytest.mark.parametrize("temperature", [1.7, 0.6])
def test_clef_temperature_folds_into_the_head_exactly(temperature):
    head, run = _clef_head_inputs()
    state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    assert decision._fold_temperature(state, temperature)
    folded = copy.deepcopy(head)
    folded.load_state_dict(state)
    for original, scaled in zip(run(head), run(folded)):
        torch.testing.assert_close(scaled, original / temperature, rtol = 1e-5, atol = 1e-6)


def test_clef_temperature_is_not_folded_past_the_heads_scale_clamp():
    head, _ = _clef_head_inputs()
    state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    state["joint_logit_scale"] = torch.tensor(math.log(90.0))
    before = {k: v.clone() for k, v in state.items()}
    # Sharpening by 2 would need a scale of 180, past the clamp at 100.
    assert not decision._fold_temperature(state, 0.5)
    assert all(torch.equal(before[k], v) for k, v in state.items())


def test_clef_collator_permutes_fields_but_keeps_targets_with_their_questions():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_QWEN3_5)
    model = types.SimpleNamespace(is_clef = True, decision_config = {"max_len": 2048})
    items, report = FastDecisionModel.build_dataset(_clef_rows(4), tokenizer, model)
    assert report["skipped"] == 0
    fixed = decision.ClefDataCollator(tokenizer.pad_token_id)
    shuffled = decision.ClefDataCollator(
        tokenizer.pad_token_id, tokenizer = tokenizer, max_len = 2048, permute_fields = True, seed = 1
    )
    orders = set()
    for _ in range(8):
        batch = shuffled(items[:1])
        record = batch["records"][0]
        orders.add(tuple(q.question_id for q in record.questions))
        by_name = dict(zip(items[0]["source"]["questions"], items[0]["targets"]))
        for row, question in enumerate(record.questions):
            assert (
                batch["target"][row, : len(question.option_ids)].tolist()
                == by_name[question.question_id]
            )
            assert bool(batch["ordinal"][row]) == (question.question_type == 2)
    assert len(orders) > 1
    assert fixed(items[:1])["records"][0] == items[0]["record"]


def test_a_failed_save_leaves_the_previous_laya_checkpoint_loadable(
    checkpoint, tmp_path, monkeypatch
):
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), use_gradient_checkpointing = False
    )
    out = tmp_path / "out"
    model.save_pretrained_merged(out)
    before = sorted(p.name for p in out.iterdir())
    original = (out / "model.safetensors").read_bytes()
    with torch.no_grad():
        for param in model.head.parameters():
            param.add_(1.0)

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(type(tokenizer), "save_pretrained", broken)
    with pytest.raises(OSError):
        model.save_pretrained_merged(out)
    assert sorted(p.name for p in out.iterdir()) == before
    assert (out / "model.safetensors").read_bytes() == original
    assert decision.is_decision_checkpoint(out)
    FastDecisionModel.from_pretrained(str(out), use_gradient_checkpointing = False)


def test_a_failed_save_leaves_the_previous_clef_checkpoint_loadable(
    clef_checkpoint, tmp_path, monkeypatch
):
    import safetensors.torch

    model, processor = FastDecisionModel.from_pretrained(str(clef_checkpoint), max_seq_length = 512)
    out = tmp_path / "out"
    model.save_pretrained_merged(str(out))
    head = (out / "joint_head.safetensors").read_bytes()
    before = sorted(p.name for p in out.iterdir())
    with torch.no_grad():
        for param in model.head.parameters():
            param.add_(1.0)
    # The backbone shards save, then the head write fails.
    original = safetensors.torch.save_file

    def failing_head(tensors, path, *args, **kwargs):
        if str(path).endswith("joint_head.safetensors"):
            raise OSError("disk full")
        return original(tensors, path, *args, **kwargs)

    monkeypatch.setattr(safetensors.torch, "save_file", failing_head)
    with pytest.raises(OSError):
        model.save_pretrained_merged(str(out))
    assert sorted(p.name for p in out.iterdir()) == before
    assert (out / "joint_head.safetensors").read_bytes() == head
    FastDecisionModel.from_pretrained(str(out), max_seq_length = 512)


def test_kl_regularization_needs_a_lora_clef(clef_checkpoint, tmp_path):
    model, processor = FastDecisionModel.from_pretrained(
        str(clef_checkpoint), max_seq_length = 512, full_finetuning = True
    )
    with pytest.raises(NotImplementedError, match = "LoRA Clef"):
        DecisionTrainer(model = model, args = _args(tmp_path), train_dataset = [], kl_weight = 0.1)


@pytest.mark.skipif(not has_real_cuda(), reason = "Unsloth's float16 path for Qwen3.5 needs a GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_clef_trains_in_float16_and_bfloat16(clef_checkpoint, tmp_path, dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation = False):
        pytest.skip("this GPU has no bfloat16")
    model, processor = FastDecisionModel.from_pretrained(
        str(clef_checkpoint), max_seq_length = 512, dtype = dtype
    )
    # float16 puts Qwen3.5 on Unsloth's float32 path (bfloat16 weights, no autocast), as on a T4.
    assert decision._clef_forced_float32(model) == (dtype == torch.float16)
    model = FastDecisionModel.get_peft_model(model, r = 8, lora_alpha = 8)
    items, _ = FastDecisionModel.build_dataset(_clef_rows(32), processor, model)
    grads = []

    class Grads(TrainerCallback):
        def on_log(
            self,
            args,
            state,
            control,
            logs = None,
            **kwargs,
        ):
            if logs and "loss" in logs:
                grads.append((logs["loss"], logs.get("grad_norm")))

    trainer = DecisionTrainer(
        model = model,
        tokenizer = processor,
        train_dataset = items,
        args = _args(
            tmp_path,
            max_steps = 6,
            logging_steps = 1,
            fp16 = dtype == torch.float16,
            bf16 = dtype == torch.bfloat16,
        ),
        callbacks = [Grads()],
    )
    assert trainer.args.fp16 is False and trainer.args.bf16 == (dtype == torch.bfloat16)
    trainer.train()
    assert grads and all(math.isfinite(loss) and math.isfinite(g) and g > 0 for loss, g in grads)
    assert math.isfinite(FastDecisionModel.evaluate(model, processor, items)["loss"])


def test_clef_calibration_fits_being_right_and_serves_through_the_head_temperature():
    # Underconfident logits: right 80% of the time with a soft gold that says 60%.
    generator = torch.Generator().manual_seed(0)
    logits, items = [], []
    for i in range(400):
        label = int(torch.randint(0, 3, (1,), generator = generator))
        right = bool(torch.rand(1, generator = generator) < 0.8)
        top = label if right else (label + 1) % 3
        z = torch.zeros(3)
        z[top] = 0.6
        target = [0.2, 0.2, 0.2]
        target[label] = 0.6
        logits.append(z)
        items.append({"label": label, "target": target, "qtype": i % 3, "row": i // 2})
    config = {"temperature": [1.0] * 3}
    plain = decision._metrics(logits, items, [1.0] * len(items))
    calibrated = decision._calibrate_clef(config, logits, items)
    assert calibrated["ece"] < plain["ece"] / 2 and calibrated["accuracy"] == plain["accuracy"]
    assert config["head_temperature"] < 1.0
    served = decision._served_temperatures(config, logits, items)
    assert served[0] == pytest.approx(
        config["head_temperature"]
        * decision._laya().common.clamp_temperature(config["temperature"][0])
    )


def test_clef_autocasts_only_in_bfloat16_and_never_on_the_float32_path(monkeypatch):
    model = torch.nn.Linear(1, 1)
    cuda = torch.device("cuda")
    monkeypatch.setattr(decision, "is_bfloat16_supported", lambda: True)
    assert decision._clef_amp_dtype(model, cuda) == torch.bfloat16
    model._unsloth_forced_float32 = True
    assert decision._clef_amp_dtype(model, cuda) is None
    model._unsloth_forced_float32 = False
    monkeypatch.setattr(decision, "is_bfloat16_supported", lambda: False)
    assert decision._clef_amp_dtype(model, cuda) is None
    assert decision._clef_amp_dtype(model, torch.device("cpu")) is None


@pytest.mark.parametrize("forced", [False, True])
def test_clef_trains_under_the_autocast_it_evaluates_in(tmp_path, monkeypatch, forced):
    # A previous trainer leaves fp16 in ACCELERATE_MIXED_PRECISION, which transformers 4.x reads.
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "fp16")
    monkeypatch.setattr(decision, "_amp_dtype", lambda device: torch.bfloat16)
    model = torch.nn.Linear(1, 1)
    model._unsloth_forced_float32 = forced
    args = _args(tmp_path)
    assert not args.bf16 and not args.fp16
    decision._clef_mixed_precision(model, args)
    expected = "no" if forced else "bf16"
    assert args.bf16 == (not forced) and not args.fp16, (args.bf16, args.fp16)
    assert decision.os.environ["ACCELERATE_MIXED_PRECISION"] == expected
    assert getattr(args, "mixed_precision", expected) == expected


def test_clef_calibration_with_every_holdout_decision_from_one_row():
    # A holdout row with many questions passes the item minimum with nothing to cross-validate on.
    logits = [torch.tensor([1.0, 0.0, -1.0]) for _ in range(12)]
    items = [{"label": i % 3, "target": [1 / 3] * 3, "qtype": i % 3, "row": 0} for i in range(12)]
    config = {"temperature": [1.0] * 3}
    calibrated = decision._calibrate_clef(config, logits, items)
    assert math.isfinite(calibrated["ece"]) and "head_temperature" in config


def test_clef_save_keeps_the_source_tokenizer_files_byte_identical(clef_checkpoint, tmp_path):
    model, processor = FastDecisionModel.from_pretrained(str(clef_checkpoint), max_seq_length = 512)
    out = tmp_path / "out"
    model.save_pretrained_merged(str(out))
    for name in ("tokenizer.json", "tokenizer_config.json", "processor_config.json"):
        if (clef_checkpoint / name).is_file():
            assert (out / name).read_bytes() == (clef_checkpoint / name).read_bytes(), name
    # Without it transformers warns of an "incorrect regex pattern" when loading the tokenizer.
    assert "transformers_version" in json.loads((out / "config.json").read_text())


def test_decision_forward_never_picks_cudnn_attention(checkpoint, tmp_path):
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), full_finetuning = True, use_gradient_checkpointing = False
    )
    items, _ = FastDecisionModel.build_dataset([_row(i) for i in range(4)], tokenizer, model)
    cudnn = []
    forward = model.forward

    def spy(*args, **kwargs):
        cudnn.append(torch.backends.cuda.cudnn_sdp_enabled())
        return forward(*args, **kwargs)

    model.forward = spy
    trainer = DecisionTrainer(
        model = model, args = _args(tmp_path), train_dataset = items, processing_class = tokenizer
    )
    device = next(model.parameters()).device
    batch = {
        k: v.to(device) for k, v in DecisionDataCollator(tokenizer.pad_token_id)(items).items()
    }
    trainer.compute_loss(model, batch)
    decision._logits(model, items, tokenizer.pad_token_id)
    assert cudnn and not any(cudnn)


def test_checkpointed_recompute_never_picks_cudnn_attention(checkpoint, tmp_path):
    # The recompute runs in backward: cuDNN there saved other tensors than the forward (CheckpointError).
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), full_finetuning = True, use_gradient_checkpointing = True
    )
    items, _ = FastDecisionModel.build_dataset([_row(i) for i in range(4)], tokenizer, model)
    layer = model.encoder.layers[0]
    cudnn = []
    forward = layer.forward

    def spy(*args, **kwargs):
        cudnn.append(torch.backends.cuda.cudnn_sdp_enabled())
        return forward(*args, **kwargs)

    layer.forward = spy
    trainer = DecisionTrainer(
        model = model, args = _args(tmp_path), train_dataset = items, processing_class = tokenizer
    )
    device = next(model.parameters()).device
    batch = {
        k: v.to(device) for k, v in DecisionDataCollator(tokenizer.pad_token_id)(items).items()
    }
    trainer.accelerator.backward(trainer.compute_loss(model, batch))
    assert len(cudnn) == 2 and not any(cudnn)


def test_logits_batch_similar_lengths_and_keep_the_callers_order(checkpoint):
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), full_finetuning = True, use_gradient_checkpointing = False
    )
    items, _ = FastDecisionModel.build_dataset([_row(i) for i in range(4)], tokenizer, model)
    assert len({len(item["input_ids"]) for item in items}) > 1
    widths = []
    forward = model.forward

    def spy(*args, **kwargs):
        widths.append(kwargs["input_ids"].shape[1])
        return forward(*args, **kwargs)

    model.forward = spy
    batched = decision._logits(model, items, tokenizer.pad_token_id, batch_size = 3)
    assert widths == sorted(widths)
    model.forward = forward
    for item, logits in zip(items, batched):
        alone = decision._logits(model, [item], tokenizer.pad_token_id)[0]
        assert logits.shape == (len(item["markers"]),)
        torch.testing.assert_close(logits, alone, rtol = 2e-2, atol = 2e-2)


@pytest.mark.parametrize("scaling", [1, 2])
def test_lean_lora_forward_matches_peft(checkpoint, scaling):
    model, _ = FastDecisionModel.from_pretrained(
        str(checkpoint), dtype = torch.float32, use_gradient_checkpointing = False
    )
    model = FastDecisionModel.get_peft_model(model, r = 4, lora_alpha = 4 * scaling)
    layers = [m for m in model.encoder.modules() if hasattr(m, "_unsloth_peft_forward")]
    from peft.tuners.lora.layer import Linear

    if Linear.forward.__name__ == "unsloth_forward":
        # An earlier FastModel load in this process compiled PEFT's forward; the lean one stays out.
        assert not layers
        return
    assert layers and all(m.forward.__func__ is decision._lean_lora_forward for m in layers)
    layer = layers[0]
    torch.manual_seed(0)
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_()
    device = layer.lora_A["default"].weight.device
    autocast = [False] + ([True] if device.type == "cuda" else [])
    for enabled in autocast:
        x = torch.randn(2, 5, layer.in_features, device = device, requires_grad = True)
        outputs = []
        for forward in (layer.forward, layer._unsloth_peft_forward):
            # The dtype the model trains in on this GPU: fp16 on a T4, which cannot run bf16.
            with torch.autocast(device.type, dtype = decision._amp_dtype(device), enabled = enabled):
                out = forward(x)
            grads = torch.autograd.grad(
                out.float().square().sum(), [x, layer.lora_A["default"].weight]
            )
            outputs.append((out, *grads))
        for lean, peft in zip(*outputs):
            assert lean.dtype == peft.dtype and torch.equal(lean, peft)
    # Disabled or merged adapters take PEFT's own path.
    with model.encoder.disable_adapter():
        x = torch.randn(2, 5, layer.in_features, device = device)
        torch.testing.assert_close(layer(x), layer.base_layer(x))


def test_compiled_layers_use_plain_sdpa_and_put_the_attention_back(checkpoint):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    from unsloth.models import _decision_fast

    model, _ = FastDecisionModel.from_pretrained(
        str(checkpoint), full_finetuning = True, use_gradient_checkpointing = False
    )
    config = model.encoder.config
    assert config._attn_implementation == "sdpa"
    with _decision_fast._compilable_attention(model, False):
        assert config._attn_implementation == "sdpa"
    with _decision_fast._compilable_attention(model, True):
        assert config._attn_implementation == "unsloth_decision_sdpa"
    assert config._attn_implementation == "sdpa"
    q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))
    mask = torch.ones(2, 1, 6, 6, dtype = torch.bool)
    mask[1, ..., 4:] = False
    module = model.encoder.layers[0].attn
    ours, _ = _decision_fast._encoder_sdpa(module, q, k, v, mask, scaling = 0.5)
    reference, _ = sdpa_attention_forward(module, q, k, v, mask, scaling = 0.5)
    torch.testing.assert_close(ours, reference)


def test_static_length_padding_leaves_the_loss_unchanged(checkpoint, tmp_path):
    from unsloth.models import _decision_fast

    model, tokenizer = FastDecisionModel.from_pretrained(
        str(checkpoint), full_finetuning = True, use_gradient_checkpointing = False
    )
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = 0.0
    items, _ = FastDecisionModel.build_dataset([_row(i) for i in range(4)], tokenizer, model)
    trainer = DecisionTrainer(
        model = model, args = _args(tmp_path), train_dataset = items, processing_class = tokenizer
    )
    device = next(model.parameters()).device
    batch = DecisionDataCollator(tokenizer.pad_token_id)(items)
    assert batch["input_ids"].shape[1] % _decision_fast.STATIC_MULTIPLE
    losses = []
    for multiple in (0, _decision_fast.STATIC_MULTIPLE):
        model._unsloth_pad_multiple = multiple
        inputs = {k: v.to(device) for k, v in batch.items()}
        padded = _decision_fast.pad_length(model, dict(inputs))
        assert padded["input_ids"].shape[1] % (multiple or 1) == 0
        with torch.no_grad():
            losses.append(trainer.compute_loss(model, inputs))
    model._unsloth_pad_multiple = 0
    torch.testing.assert_close(losses[0], losses[1])


def test_static_compiled_layers_run_eagerly_outside_training():
    from unsloth.models import _decision_fast

    calls = []
    layer = torch.nn.Linear(2, 2)
    call = _decision_fast._training_only(
        layer, lambda *a, **k: calls.append("compiled") or layer._call_impl(*a, **k)
    )
    x = torch.randn(1, 2)
    call(x)
    layer.eval()
    call(x)
    assert calls == ["compiled"]


@pytest.mark.parametrize("max_length, static", [(634, True), (4096, False), (None, False)])
def test_short_data_compiles_static_buckets_and_long_data_dynamic(
    checkpoint, monkeypatch, max_length, static
):
    from unsloth.models import _decision_fast

    monkeypatch.setenv("UNSLOTH_DECISION_COMPILE", "1")
    monkeypatch.setattr(_decision_fast, "_warm_up", lambda model, amp_dtype: None)
    model, _ = FastDecisionModel.from_pretrained(
        str(checkpoint), full_finetuning = True, use_gradient_checkpointing = False
    )
    layers = _decision_fast._encoder_layers(model)
    limit = torch._dynamo.config.recompile_limit
    with _decision_fast.compiled_encoder(model, 10, None, max_length) as compiled:
        assert compiled
        assert model._unsloth_pad_multiple == (_decision_fast.STATIC_MULTIPLE if static else 0)
        assert all(layer._compiled_call_impl is not None for layer in layers)
        assert all((layer._compiled_call_impl.__name__ == "call") == static for layer in layers)
        assert (torch._dynamo.config.recompile_limit > limit) == (static and limit < 28)
    assert all(layer._compiled_call_impl is None for layer in layers)
    assert model._unsloth_pad_multiple == 0
    assert torch._dynamo.config.recompile_limit == limit
