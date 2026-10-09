# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import ast
import json
import os
import platform
import sys
from pathlib import Path

import pytest

# Before any unsloth_zoo import: without torch off Apple Silicon it raises a plain ImportError.
if sys.platform != "darwin" or platform.machine() != "arm64":
    pytest.skip("the MLX backend", allow_module_level = True)
pytest.importorskip("unsloth_zoo.mlx.decision")
torch = pytest.importorskip("torch")

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from transformers import TrainingArguments

import unsloth
from unsloth import DecisionTrainer, FastDecisionModel

pytestmark = pytest.mark.skipif(not unsloth._IS_MLX, reason = "the MLX backend")

WORDS = (
    "the server is down again refund my card charge twice please help now noul choice score "
    "question : ? , does this need a reply which team should handle it how upset customer outage "
    "service billing charges and refunds level 0 1 2 calm annoyed angry false no statement not hold "
    "true yes holds"
).split()
TEAMS = {"outage": "service down", "billing": "charges and refunds"}
MOODS = ["calm", "annoyed", "angry"]
QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does this need a reply now?"},
    "team": {"type": "choice", "instructions": "Which team should handle it?", "criteria": TEAMS},
    "mood": {"type": "score", "instructions": "How upset is the customer?", "criteria": MOODS},
}
ENCODER = {"vocab_size": 64, "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 2}
ENCODER.update(num_attention_heads = 4, pad_token_id = 0, cls_token_id = 2, sep_token_id = 1)
ENCODER.update(global_attn_every_n_layers = 2, local_attention = 16)


def _row(i):
    state = "the server is down again help now" if i % 2 == 0 else "refund my card charge twice"
    gold = {"urgent": "true", "team": "outage", "mood": 2}
    gold = gold if i % 2 == 0 else {"urgent": "false", "team": "billing", "mood": 0}
    return {"state": state, "questions": QUESTIONS, "gold": gold}


ARGS = {"per_device_train_batch_size": 16, "max_steps": 80, "report_to": "none"}
ARGS.update(save_strategy = "no", eval_strategy = "steps", eval_steps = 40)


@pytest.fixture
def checkpoint(tmp_path):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import ModernBertConfig, PreTrainedTokenizerFast

    base = tmp_path / "base"
    specials = ["[PAD]", "[SEP]", "[CLS]", "[UNK]", "[MASK]"]
    vocab = {token: i for i, token in enumerate(specials + sorted(set(WORDS)))}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token = "[UNK]"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    names = dict(zip(("pad_token", "sep_token", "cls_token", "unk_token", "mask_token"), specials))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object = tokenizer, **names)
    tokenizer.save_pretrained(str(base / "tokenizer"))
    (base / "encoder").mkdir()
    encoder = json.dumps(ModernBertConfig(**ENCODER).to_dict())
    (base / "encoder" / "config.json").write_text(encoder)
    config = {"encoder": "tiny", "head_layers": 1, "act_costs": {"escalate": 0.5}, "max_len": 96}
    config.update(head_max_len = 48, temperature = [1.2, 1.1, 1.3], training = {"updates": 1000})
    torch.manual_seed(0)
    model = unsloth._decision_mlx._laya().common.build_model(
        config, encoder_dir = str(base / "encoder")
    )
    weights = {k: v.half().contiguous() for k, v in model.state_dict().items()}
    save_file(weights, str(base / "model.safetensors"))
    (base / "rl_agent_config.json").write_text(json.dumps(config))
    return base


@pytest.mark.parametrize("full", [False, True])
def test_train_calibrate_save_and_serve(checkpoint, tmp_path, full):
    model, tokenizer = FastDecisionModel.from_pretrained(str(checkpoint), full_finetuning = full)
    model = FastDecisionModel.get_peft_model(model, r = 8, lora_alpha = 16)
    rows = [_row(i) for i in range(120)]
    items, report = FastDecisionModel.build_dataset(rows, tokenizer, model)
    train, held = FastDecisionModel.split_holdout(items, 3407, fraction = 0.25)
    assert report["skipped"] == 0 and len(train) + len(held) == 360
    base = FastDecisionModel.evaluate(model, tokenizer, held)
    args = TrainingArguments(
        str(tmp_path / "run"), learning_rate = 1e-3 if full else 5e-3, warmup_steps = 8, **ARGS
    )
    trainer = DecisionTrainer(model, args, train, held, head_learning_rate = 5e-3)
    trainer.train()
    tuned = FastDecisionModel.evaluate(model, tokenizer, held)
    assert tuned["accuracy"] >= base["accuracy"] + 0.25 and tuned["loss"] < base["loss"] / 2
    assert [log["step"] for log in trainer.state.log_history if "eval_loss" in log] == [40, 80]
    assert trainer.args is not args and args.eval_steps == 40
    adapters = [v for k, v in tree_flatten(model.trainable_parameters()) if k.endswith("lora_b")]
    assert full != bool(adapters) and all(abs(v).max().item() > 0 for v in adapters)
    assert full == (
        "encoder.layers.0.attn.Wo.weight" in dict(tree_flatten(model.trainable_parameters()))
    )
    manual = DecisionTrainer(model, TrainingArguments(str(tmp_path / "run")), train[:16], held)
    assert manual.evaluate()["eval_loss"] > 0
    assert manual.evaluate(held[:8])["eval_loss"] != manual.evaluate()["eval_loss"]
    assert FastDecisionModel.calibrate(model, tokenizer, held)["fitted_types"] == [0, 1, 2]
    calibrated = FastDecisionModel.evaluate(model, tokenizer, held)
    assert model.decision_config["temperature"] != [1.2, 1.1, 1.3]
    model.save_pretrained_merged(tmp_path / "out")
    saved = json.loads((tmp_path / "out" / "rl_agent_config.json").read_text())
    assert saved["fine_tuned"] is True and "training" not in saved
    assert saved["temperature"] == model.decision_config["temperature"] and saved["max_len"] == 1024
    unsloth._decision_mlx._laya().load(str(tmp_path / "out"), device = "cpu")
    served, tokenizer = FastDecisionModel.from_pretrained(str(tmp_path / "out"))
    loss = FastDecisionModel.evaluate(served, tokenizer, held)["loss"]
    assert loss == pytest.approx(calibrated["loss"], abs = 2e-2)


def test_unsupported_options_are_refused(checkpoint, tmp_path):
    with pytest.raises(NotImplementedError, match = "load_in_4bit"):
        FastDecisionModel.from_pretrained(str(checkpoint), load_in_4bit = True)
    with pytest.raises(NotImplementedError, match = "bfloat16"):
        FastDecisionModel.from_pretrained(str(checkpoint), dtype = torch.bfloat16)
    model, _ = FastDecisionModel.from_pretrained(str(checkpoint), dtype = torch.float16)
    for option in ({"use_dora": True}, {"bias": "all"}, {"loftq_config": {"loftq_bits": 4}}):
        with pytest.raises(NotImplementedError, match = next(iter(option))):
            FastDecisionModel.get_peft_model(model, **option)
    with pytest.raises(NotImplementedError, match = "data_collator, optimizers"):
        DecisionTrainer(model, data_collator = len, optimizers = (None, None), brier_weight = 0.5)
    best = {"save_strategy": "best", "eval_strategy": "steps", "metric_for_best_model": "loss"}
    for option in ({"auto_find_batch_size": True}, best):
        with pytest.raises(NotImplementedError, match = next(iter(option))):
            DecisionTrainer(model, TrainingArguments(str(tmp_path), **option))
    ignored = {"data_seed": 1, "eval_on_start": True, "neftune_noise_alpha": 5.0}
    with pytest.warns(UserWarning, match = "ignores data_seed, eval_on_start, neftune_noise_alpha"):
        DecisionTrainer(model, TrainingArguments(str(tmp_path), **ignored))


def test_the_torch_trainer_options_reach_the_mlx_trainer(checkpoint, tmp_path):
    model, tokenizer = FastDecisionModel.from_pretrained(str(checkpoint))
    items, _ = FastDecisionModel.build_dataset([_row(i) for i in range(4)], tokenizer, model)
    schedule = {"eval_strategy": "steps", "eval_steps": 2, "save_steps": 2, "max_steps": 4}
    schedule.update(logging_first_step = True, label_smoothing_factor = 0.3, report_to = "none")
    args = TrainingArguments(str(tmp_path), **schedule)
    count = lambda prediction: {"rows": len(prediction.label_ids)}
    options = {"preprocess_logits_for_metrics": lambda logits, _: logits.argmax(dim = -1)}
    trainer = DecisionTrainer(model, args, items, items, compute_metrics = count, **options)
    assert trainer.objective == (0.3, 0.0, 0.0) and min(trainer.train_dataset[0]["target"]) == 0.0
    trainer.train()
    assert trainer.state.log_history[0]["step"] == 1 and (tmp_path / "checkpoint-2").is_dir()
    resumed = DecisionTrainer(model, args, items, items)
    resumed.train(resume_from_checkpoint = str(tmp_path / "checkpoint-2"))
    logs = resumed.state.log_history, trainer.state.log_history
    assert (
        logs[0][:2] == logs[1][:2] and len(logs[0]) == len(logs[1]) and logs[0][2:] != logs[1][2:]
    )
    assert trainer.predict(items[:3]).predictions.shape == (3,)
    assert trainer.evaluate()["eval_rows"] == 12


def test_transformers_callbacks_read_the_training_arguments(checkpoint, tmp_path):
    from transformers import EarlyStoppingCallback

    model, tokenizer = FastDecisionModel.from_pretrained(str(checkpoint))
    items, _ = FastDecisionModel.build_dataset([_row(0), _row(1)], tokenizer, model)
    args = {**ARGS, "eval_steps": 1, "metric_for_best_model": "eval_loss"}
    args = TrainingArguments(str(tmp_path), **args)
    # An improvement of 10 never happens, so the second evaluation stops the run.
    trainer = DecisionTrainer(model, args, items, items, callbacks = [EarlyStoppingCallback(1, 10.0)])
    trainer.train()
    assert trainer.state.global_step == 2 and trainer.args.eval_strategy == args.eval_strategy


def test_decision_helpers_are_one_torch_free_module_for_both_backends():
    import sys

    common = sys.modules["unsloth._decision_common"]
    assert unsloth._decision_mlx._metrics is common._metrics
    root = Path(unsloth.__file__).parent / "models"
    shared = ast.parse((root / "_decision_common.py").read_text(encoding = "utf-8")).body
    # Loaded by path on MLX, where unsloth.models and its torch imports are never imported.
    modules = [a.name for n in shared if isinstance(n, ast.Import) for a in n.names]
    modules += [n.module for n in shared if isinstance(n, ast.ImportFrom)]
    assert {name.split(".")[0] for name in modules}.isdisjoint({"torch", "transformers"})
    torch_module = ast.parse((root / "decision.py").read_text(encoding = "utf-8")).body
    for module in (
        torch_module,
        ast.parse((root / "decision_mlx.py").read_text(encoding = "utf-8")).body,
    ):
        assert {n.name for n in module if hasattr(n, "name")}.isdisjoint(common.__all__)
    taken = [n for n in torch_module if getattr(n, "module", None) == "_decision_common"]
    assert [a.name for n in taken for a in n.names] == ["*"]


@pytest.fixture
def clef_checkpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mlx_lm.models import qwen3_5
    from unsloth_zoo.mlx import decision as zoo

    text = {"model_type": "qwen3_5_text", "hidden_size": 64, "intermediate_size": 128}
    text.update(num_hidden_layers = 4, num_attention_heads = 2, num_key_value_heads = 1)
    # The default recurrent state is the full model's, megabytes per token whatever the hidden size.
    text.update(linear_num_value_heads = 4, linear_num_key_heads = 2)
    text.update(linear_key_head_dim = 32, linear_value_head_dim = 16)
    args = qwen3_5.ModelArgs(model_type = "qwen3_5", text_config = {**text, "vocab_size": 512})
    kernel = lambda value: value.swapaxes(1, 2) if value.ndim == 3 else value
    # The checkpoint names the decoder's tensors differently from the loaded model.
    names = (
        ("language_model.model.", "model.language_model."),
        ("language_model.lm_head.", "lm_head."),
    )

    def renamed(name, saving):
        for loaded, stored in names:
            old, new = (loaded, stored) if saving else (stored, loaded)
            if name.startswith(old):
                return new + name[len(old) :]
        return name

    def build(weights = None, quantized = False):
        model = qwen3_5.Model(args)
        model.set_dtype(mx.bfloat16)
        if weights:
            stored = mx.load(weights).items()
            model.update(tree_unflatten([(renamed(k, False), kernel(v)) for k, v in stored]))
        if quantized:
            # Stands in for the loader's 4-bit quantization, finely enough to compare with the saved model.
            nn.quantize(model, group_size = 64, bits = 8)
        return model

    def load(
        self,
        source,
        revision,
        dtype,
        token,
        adapter = None,
        load_in_4bit = False,
    ):
        encode = lambda text, add_special_tokens = False: [ord(c) % 512 for c in text]
        tokenizer = SimpleNamespace(encode = encode, pad_token_id = 0)
        model = build(f"{source}/model.safetensors", load_in_4bit)
        vars(self).update(model = model, tokenizer = tokenizer)

    folder, head = tmp_path / "clef", {"hidden_size": 64, "width": 32, "feedforward": 48}
    head.update(routing_layers = 1, layers = 1, heads = 4)
    folder.mkdir()
    weights = {renamed(k, True): kernel(v) for k, v in tree_flatten(build().parameters())}
    mx.save_safetensors(str(folder / "model.safetensors"), weights)
    weights = dict(tree_flatten(zoo._JointHead(**head).parameters()))
    mx.save_safetensors(str(folder / "joint_head.safetensors"), weights)
    (folder / "joint_head_config.json").write_text(json.dumps(head))
    (folder / "config.json").write_text("{}")
    monkeypatch.setattr(zoo.ClefModel, "_load", load)
    return folder


@pytest.mark.parametrize("full, four_bit", [(False, False), (False, True), (True, True)])
def test_clef_trains_calibrates_saves_and_reloads(
    clef_checkpoint, tmp_path, monkeypatch, full, four_bit
):
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(clef_checkpoint), full_finetuning = full, max_seq_length = 2048, load_in_4bit = four_bit
    )
    # Full fine-tuning drops the 4-bit request, as the torch loader does.
    quantized = any(name.endswith("scales") for name, _ in tree_flatten(model.parameters()))
    assert quantized == (four_bit and not full)
    model = FastDecisionModel.get_peft_model(model, r = 8, lora_alpha = 16)
    rows = [{**_row(0), "state": "long " * 1000}, *(_row(i) for i in range(1, 40))]
    rows.append({"state": "s", "questions": {"q": {"type": "noul"}}, "gold": {"q": True}})
    items, report = FastDecisionModel.build_dataset(rows, tokenizer, model)
    train, held = FastDecisionModel.split_holdout(items, 3407, fraction = 0.25)
    assert (report["total"], report["skipped"], len(items)) == (121, 1, 40)
    assert "instructions" in report["reason"] and sum(len(item["labels"]) for item in held) == 30
    assert max(len(item["input_ids"]) for item in items) == 2048 == len(items[0]["input_ids"])
    # Scored per question, typed as the shared metrics number them: choice, score, noul.
    assert [
        q["qtype"] for q in unsloth._decision_mlx._decision_logits(model, tokenizer, held[:1])[1]
    ] == [
        2,
        0,
        1,
    ]
    args = {**ARGS, "max_steps": 12, "eval_strategy": "no", "per_device_train_batch_size": 4}
    args.update(save_strategy = "steps", save_steps = 12)
    args = TrainingArguments(str(tmp_path / "run"), learning_rate = 1e-3, **args)
    options = {"kl_weight": 0.0 if full else 0.5, "permute_fields": True, "ordinal_weight": 0.3}
    options["brier_weight"] = 0.5
    trainer = DecisionTrainer(model, args, train, held, head_learning_rate = 1e-3, **options)
    assert trainer.train_dataset[0]["source"]["max_length"] == 2048 and trainer._permute
    assert trainer.objective == (0.0, 0.5, 0.3) and (trainer._reference is None) == full
    before = {name: mx.array(value) for name, value in tree_flatten(model.trainable_parameters())}
    trainer.train()
    after = dict(tree_flatten(model.trainable_parameters()))
    moved = {name.split(".")[0] for name in before if not mx.array_equal(before[name], after[name])}
    assert moved == {"encoder", "head"}
    assert (tmp_path / "run" / "checkpoint-12" / "adapter_model.safetensors").exists() != full
    trained = [name for name, _ in tree_flatten(model.trainable_parameters())]
    assert {name.split(".")[0] for name in trained} == {"encoder", "head"}
    assert full != any(name.endswith("lora_b") for name in trained)
    assert FastDecisionModel.calibrate(model, tokenizer, held)["fitted_types"] == [0, 1, 2]
    calibrated = FastDecisionModel.evaluate(model, tokenizer, held, batch_size = 4)
    # A backbone frozen after it trained is still saved as trained, and stays frozen.
    FastDecisionModel.freeze_backbone(model)
    with pytest.raises(ValueError, match = "saved over"):
        model.save_pretrained_merged(clef_checkpoint)
    assert not tree_flatten(model.encoder.trainable_parameters())
    model.save_pretrained_merged(tmp_path / "out")
    assert not tree_flatten(model.encoder.trainable_parameters())
    saved = json.loads((tmp_path / "out" / "unsloth_decision_config.json").read_text())
    assert saved["fine_tuned"] is True and saved["layout"] == "clef" and saved["max_len"] == 2048
    assert "head_temperature" in model.decision_config and "head_temperature" not in saved
    served, tokenizer = FastDecisionModel.from_pretrained(str(tmp_path / "out"))
    assert served.decision_config["temperature"] == saved["temperature"] != [1.0] * 3
    held = FastDecisionModel.build_dataset([rows[item["row"]] for item in held], tokenizer, served)
    loss = FastDecisionModel.evaluate(served, tokenizer, held[0])["loss"]
    assert loss == pytest.approx(calibrated["loss"], abs = 3e-2)
    # save_pretrained keeps LoRA adapters as adapters over their base; a full fine-tune has none and saves merged.
    model.save_pretrained(tmp_path / "lora")
    assert (tmp_path / "lora" / "adapter_model.safetensors").exists() != full
    assert (tmp_path / "lora" / "model.safetensors").exists() == full
    again, tokenizer = FastDecisionModel.from_pretrained(
        str(tmp_path / "lora"), load_in_4bit = four_bit
    )
    base = tmp_path / "lora" if full else clef_checkpoint
    reloaded = FastDecisionModel.evaluate(again, tokenizer, held[0])["loss"]
    assert again.decision_config["base_model"] == str(base) and reloaded == pytest.approx(
        loss, abs = 3e-2
    )
    if not full:
        # Loaded adapters are the ones that go on training, and they save as adapters again.
        with pytest.raises(RuntimeError, match = "already added"):
            FastDecisionModel.get_peft_model(again, r = 64)
        tuned = dict(tree_flatten(again.trainable_parameters()))
        assert sorted(tuned) == sorted(trained)
        assert {v.shape[1] for k, v in tuned.items() if k.endswith("lora_a")} == {8}
        again.save_pretrained(tmp_path / "again")
        # Without a decision config, the adapters still name their base.
        (tmp_path / "again" / "unsloth_decision_config.json").unlink()
        bare = FastDecisionModel.from_pretrained(str(tmp_path / "again"))[0]
        assert bare.decision_config["base_model"] == str(clef_checkpoint)
        # Merged, they go into the base's weights, which the adapter folder does not hold.
        again.save_pretrained_merged(tmp_path / "whole")
        whole, tokenizer = FastDecisionModel.from_pretrained(str(tmp_path / "whole"))
        merged = FastDecisionModel.evaluate(whole, tokenizer, held[0])["loss"]
        assert merged == pytest.approx(loss, abs = 3e-2)
    # The GGUF export converts a merged save, whatever the model trained through.
    seen, gguf = [], unsloth._decision_mlx._decision_gguf()
    monkeypatch.setattr(gguf, "_converter_dir", lambda *args: None)
    # The exporter reads the checkpoint through the helpers both backends share.
    assert gguf._layout(clef_checkpoint) == "clef" and gguf._layout(tmp_path) is None
    assert gguf._laya_max_head_tokens(tmp_path, {"head_max_len": 64}) == 256
    assert gguf.effective_temperatures({"temperature": [2.0, 1.0, 1.0]}, "laya")["choice"] == 2.0
    export = lambda folder, method, **kwargs: seen.append(
        (sorted(os.listdir(folder)), method, kwargs["output_dir"], Path(folder).name)
    )
    monkeypatch.setattr(gguf, "export_decision_gguf", export)
    # A merge an ended process left behind is swept, and this one is named for its own process.
    monkeypatch.setattr(gguf, "_pid_alive", lambda pid: False)
    (tmp_path / "gguf" / ".unsloth-merged-1-left").mkdir(parents = True)
    model.save_pretrained_gguf(tmp_path / "gguf", tokenizer)
    assert seen[0][3].startswith(f".unsloth-merged-{os.getpid()}-")
    assert "model.safetensors" in seen[0][0] and "adapter_config.json" not in seen[0][0]
    assert (
        seen[0][1:3] == ("q8_0", tmp_path / "gguf" / "gguf") and os.listdir(tmp_path / "gguf") == []
    )


def _predicted_shapes(answers):
    assert isinstance(answers["urgent"]["answer"], bool) and answers["team"]["answer"] in TEAMS
    assert answers["mood"]["answer"] in (0, 1, 2) and answers["team"]["type"] == "choice"
    assert sum(answers["mood"]["probabilities"].values()) == pytest.approx(1.0, abs = 1e-3)
    return max(answers["mood"]["probabilities"].values())


def test_predict_answers_at_the_calibrated_temperatures(checkpoint, monkeypatch):
    model, tokenizer = FastDecisionModel.from_pretrained(str(checkpoint))
    rows = [{**_row(0), "state": "down " * 2000}, _row(1)]
    assert FastDecisionModel.build_dataset(rows, tokenizer, model)[1]["truncated"] == 3
    # An untrained head answers uniformly, so each question's last option is made the likeliest.
    rising = lambda model, items, pad: [
        torch.arange(len(i["markers"])).float().numpy() for i in items
    ]
    monkeypatch.setattr(unsloth._decision_mlx._decision_zoo(), "decision_logits", rising)
    # One temperature per question type: choice, score, noul.
    model.decision_config["temperature"] = [0.5, 1.0, 5.0]
    answers = FastDecisionModel.predict(model, tokenizer, _row(0)["state"], QUESTIONS)
    confidence = [answers["team"]["confidence"], _predicted_shapes(answers)]
    confidence.append(answers["urgent"]["noul"])
    assert confidence == pytest.approx([0.8808, 0.6652, 0.5498], abs = 2e-4)
    assert list(answers) == list(QUESTIONS)
    picked = [answers[name]["answer"] for name in QUESTIONS]
    assert picked == [True, "billing", 2] and answers["team"]["choice"] == "billing"
    with pytest.raises(unsloth._decision_mlx.DecisionDataError, match = "non-empty"):
        FastDecisionModel.predict(model, tokenizer, "s", {})


def test_the_images_of_a_row_reach_the_clef_trainer_and_predict(
    clef_checkpoint, checkpoint, monkeypatch, tmp_path
):
    import base64
    import io

    from PIL import Image

    model, tokenizer = FastDecisionModel.from_pretrained(str(clef_checkpoint), max_seq_length = 2048)
    zoo = unsloth._decision_mlx._decision_zoo()
    buffer = io.BytesIO()
    Image.new("RGB", (5, 4)).save(buffer, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    rows = [
        {**_row(0), "images": [url, url]},
        {**_row(1), "images": json.dumps([url])},
        {**_row(2), "images": [None, {"bytes": None, "path": None}]},
    ]
    (tmp_path / "i.png").write_bytes(buffer.getvalue())
    rows += [
        {**_row(3), "images": {"bytes": buffer.getvalue()}},
        {**_row(4), "images": [{"bytes": b"cut"}]},
        {**_row(5), "images": [{"bytes": None, "path": str(tmp_path / "i.png")}]},
    ]
    # This decoder has no vision tower: a row with images is skipped, and says why.
    items, report = FastDecisionModel.build_dataset(rows, tokenizer, model)
    assert [item["row"] for item in items] == [2] and "does not support images" in report["reason"]

    seen, real, keep = [], zoo.clef_training_item, [True]

    def item(pipeline, state, questions, max_length, *images):
        # A row without images is asked as before, with no fifth argument.
        seen.append([getattr(image, "size", image) for image in images[0]] if images else None)
        return {
            **real(pipeline, state, questions, max_length),
            **({"images": images[0]} if keep and images else {}),
        }

    monkeypatch.setattr(zoo, "clef_training_item", item)
    items, report = FastDecisionModel.build_dataset(rows, tokenizer, model)
    assert [item["row"] for item in items] == [0, 1, 2, 3, 5] and seen == [
        [url, url],
        [url],
        None,
        [(5, 4)],
        [(5, 4)],
    ]
    assert items[0]["images"] == [url, url]
    keep.clear()
    assert "cannot identify image" in report["reason"]
    FastDecisionModel.predict(
        model, tokenizer, "s", {"q": {"type": "noul", "instructions": "i"}}, images = url
    )
    assert seen[-1] == [url]
    # An unsloth-zoo that takes no images says so, where skipping the rows would train without them.
    monkeypatch.setattr(
        zoo,
        "clef_training_item",
        lambda pipeline, state, questions, max_length = None: real(
            pipeline, state, questions, max_length
        ),
    )
    assert len(FastDecisionModel.build_dataset(rows[2:3], tokenizer, model)[0]) == 1
    question = {"q": {"type": "noul", "instructions": "i"}}
    for call in (
        lambda: FastDecisionModel.build_dataset(rows, tokenizer, model),
        lambda: FastDecisionModel.predict(model, tokenizer, "s", question, images = url),
    ):
        with pytest.raises(ImportError, match = "newer unsloth-zoo"):
            call()
    laya, laya_tokenizer = FastDecisionModel.from_pretrained(str(checkpoint))
    with pytest.raises(unsloth._decision_mlx.DecisionDataError, match = "Laya reads text only"):
        FastDecisionModel.predict(
            laya, laya_tokenizer, "s", {"q": {"type": "noul", "instructions": "i"}}, images = [url]
        )
    # An empty image cell is no image.
    assert FastDecisionModel.predict(
        laya, laya_tokenizer, "s", {"q": {"type": "noul", "instructions": "i"}}, images = "[]"
    )


@pytest.mark.parametrize("four_bit", [False, True])
def test_a_plain_language_model_trains_as_a_clef(clef_checkpoint, tmp_path, four_bit):
    for name in ("joint_head.safetensors", "joint_head_config.json"):
        (clef_checkpoint / name).unlink()
    with pytest.raises(NotImplementedError, match = "head_init"):
        FastDecisionModel.from_pretrained(str(clef_checkpoint), head_init = "org/clef")
    # Without head files the folder is a plain language model, which gets a new joint head.
    model, tokenizer = FastDecisionModel.from_pretrained(
        str(clef_checkpoint), head_width = 32, load_in_4bit = four_bit, random_state = 1
    )
    config = model.decision_config
    assert (config["base_model"], config["load_in_4bit"]) == (str(clef_checkpoint), four_bit)
    assert model._unsloth_pipeline.head_config["width"] == 32
    assert four_bit == any(name.endswith("scales") for name, _ in tree_flatten(model.parameters()))
    model = FastDecisionModel.get_peft_model(model, r = 8, lora_alpha = 16)
    names = lambda: {name for name, _ in tree_flatten(model.trainable_parameters())}
    adapters = names()
    FastDecisionModel.freeze_backbone(model)
    assert {name.split(".")[0] for name in names()} == {"head"}
    FastDecisionModel.unfreeze_backbone(model)
    # Exactly the weights that trained before: the adapters, not the base weights under them.
    assert names() == adapters and any(name.startswith("encoder") for name in adapters)
    confidence = _predicted_shapes(FastDecisionModel.predict(model, tokenizer, "s", QUESTIONS))
    model.save_pretrained_merged(tmp_path / "out")
    served, tokenizer = FastDecisionModel.from_pretrained(str(tmp_path / "out"))
    answers = FastDecisionModel.predict(served, tokenizer, "s", QUESTIONS)
    assert _predicted_shapes(answers) == pytest.approx(confidence, abs = 3e-2)
    assert served.decision_config["base_model"] == str(tmp_path / "out")
