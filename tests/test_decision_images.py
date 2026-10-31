# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import unsloth
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
from unsloth.models import decision
from unsloth.models.clef import JointSchemaHead, collate_records, encode_record


@pytest.fixture(scope = "module")
def processor():
    return AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")


@pytest.fixture
def row():
    return {
        "state": "Frames are ordered oldest to newest.",
        "images": [Image.new("RGB", (56, 56), color) for color in ("red", "green", "blue")],
        "questions": {
            "motion": {
                "type": "choice",
                "criteria": {"left": "moves left", "right": "moves right"},
            },
            "visible": {"type": "noul", "instructions": "Is an object visible?"},
        },
        "gold": {"motion": "right", "visible": True},
    }


def build(rows, processor):
    model = SimpleNamespace(is_clef = True, decision_config = {"max_len": 2048})
    return decision.FastDecisionModel.build_dataset(rows, processor, model)


def test_image_records_batch_and_permute_without_losing_order(processor, row):
    variants = [
        {**row, "images": []},
        {**row, "images": row["images"][:1]},
        row,
        {**row, "images": row["images"][::-1]},
    ]
    items, report = build(variants, processor)
    assert report["skipped"] == 0
    encoded = [
        encode_record(processor.tokenizer, r, max_length = 2048, processor = processor)
        for r in variants
    ]
    for item, record in zip(items, encoded):
        assert item["input_ids"] == list(record.input_ids)
        assert "pixel_values" not in (item["record"].media or {})
    expected = collate_records(encoded, processor.tokenizer.pad_token_id, torch.device("cpu"))
    collator = decision.ClefDataCollator(processor.tokenizer.pad_token_id, tokenizer = processor)
    batch = collator(items)
    for key in ("input_ids", "attention_mask"):
        torch.testing.assert_close(batch[key], expected[key])
    for key, value in expected["media"].items():
        torch.testing.assert_close(batch[key], value)
    assert not torch.equal(
        collator(items[2:3])["pixel_values"], collator(items[3:4])["pixel_values"]
    )
    shuffled = decision.ClefDataCollator(
        processor.tokenizer.pad_token_id,
        tokenizer = processor,
        max_len = 2048,
        permute_fields = True,
        seed = 1,
    )(items[2:3])
    assert [q.question_id for q in shuffled["records"][0].questions] == ["visible", "motion"]
    by_name = dict(zip(row["questions"], items[2]["targets"]))
    for target, question in zip(shuffled["target"], shuffled["records"][0].questions):
        assert target[: len(question.option_ids)].tolist() == by_name[question.question_id]
    torch.testing.assert_close(shuffled["pixel_values"], collator(items[2:3])["pixel_values"])


def test_image_columns_read_like_mlx(processor, row):
    import io

    image = row["images"][0]
    buffer = io.BytesIO()
    image.save(buffer, format = "PNG")
    forms = [[image], image, [{"bytes": buffer.getvalue(), "path": None}]]
    items, report = build([{**row, "images": form} for form in forms], processor)
    assert report["skipped"] == 0
    collator = decision.ClefDataCollator(processor.tokenizer.pad_token_id, tokenizer = processor)
    first = collator(items[:1])
    for item in items[1:]:
        assert item["input_ids"] == items[0]["input_ids"]
        torch.testing.assert_close(collator([item])["pixel_values"], first["pixel_values"])


def test_image_rows_require_a_processor(processor, row):
    items, report = build([row], processor.tokenizer)
    assert items == []
    assert report["skipped"] == len(row["questions"])
    assert "require a processor" in report["reason"]


@pytest.fixture
def model():
    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B")
    text = config.text_config
    text.hidden_size, text.intermediate_size, text.num_hidden_layers = 64, 128, 2
    text.num_attention_heads, text.num_key_value_heads, text.head_dim = 4, 2, 16
    text.layer_types = ["full_attention"] * 2
    text.rope_parameters = {
        **text.rope_parameters,
        "partial_rotary_factor": 1.0,
        "mrope_section": [2, 3, 3],
    }
    vision = config.vision_config
    vision.depth, vision.hidden_size, vision.intermediate_size = 1, 32, 64
    vision.num_heads, vision.out_hidden_size = 4, 64
    torch.manual_seed(13461)
    backbone = AutoModelForImageTextToText.from_config(config, attn_implementation = "eager")
    head = JointSchemaHead(
        hidden_size = 64,
        width = 32,
        routing_layers = 1,
        layers = 1,
        heads = 4,
        feedforward = 64,
    )
    model = decision.ClefDecisionModel(backbone, head).eval()
    model.decision_config = {"max_len": 2048, "temperature": [1.0] * 3}
    return model


def test_images_reach_forward_gradients_prediction_and_evaluation(processor, row, model):
    items, _ = build([row], processor)
    batch = decision.ClefDataCollator(processor.tokenizer.pad_token_id, tokenizer = processor)(items)
    calls = []
    hook = model.encoder.model.visual.register_forward_hook(lambda *args: calls.append(1))
    try:
        batch["pixel_values"] = batch["pixel_values"].detach().requires_grad_(True)
        logits, _ = model(**batch)
        loss = decision._decision_loss(logits, batch["target"], batch["marker_mask"])
        loss.backward()
        assert calls and torch.isfinite(loss)
        assert batch["pixel_values"].grad.abs().sum() > 0
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.encoder.model.visual.parameters()
        )
        changed, _ = model(**{**batch, "pixel_values": torch.zeros_like(batch["pixel_values"])})
        assert not torch.allclose(logits, changed, atol = 1e-5)
        calls.clear()
        answers = decision.FastDecisionModel.predict(
            model,
            processor,
            row["state"],
            row["questions"],
            images = row["images"],
        )
        assert calls and set(answers) == set(row["questions"])
        calls.clear()
        evaluated, _ = decision._clef_logits(
            model, items, processor.tokenizer.pad_token_id, processor = processor
        )
        assert calls and len(evaluated) == len(row["questions"])
        for expected, actual in zip(logits.detach(), evaluated):
            torch.testing.assert_close(expected, actual)
    finally:
        hook.remove()


def test_text_only_prediction_keeps_language_path(processor, row, model):
    calls = []
    hook = model.encoder.model.visual.register_forward_hook(lambda *args: calls.append(1))
    try:
        answers = decision.FastDecisionModel.predict(
            model, processor, row["state"], row["questions"]
        )
        tokenizer_only = decision.FastDecisionModel.predict(
            model,
            processor.tokenizer,
            row["state"],
            row["questions"],
        )
        assert answers == tokenizer_only
        assert not calls
    finally:
        hook.remove()
