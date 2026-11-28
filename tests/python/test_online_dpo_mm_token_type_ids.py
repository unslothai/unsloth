# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""TRL Online DPO ``_forward`` on a tiny Qwen2-VL / Qwen2.5-VL image batch after the full
RL_FUNCTIONS["online_dpo_trainer"] chain. TRL < 1.17 passes image_grid_thw without mm_token_type_ids, which
transformers 5.x M-RoPE models refuse; the patched forward must run and give the log-probs of a forward fed the
processor's mm_token_type_ids (completion tokens as text). On transformers 4.x the model takes no such argument and
the forward must be unchanged."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
import re
import textwrap

import pytest
import torch

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth.models.rl_replacements as rl

TINY = {
    "qwen2_vl": "trl-internal-testing/tiny-Qwen2VLForConditionalGeneration",
    "qwen2_5_vl": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
}


def _odpo_source():
    pytest.importorskip("trl")
    import trl.import_utils as iu

    iu._mergekit_available = iu._llm_blender_available = False
    for name in (
        "trl.experimental.online_dpo.online_dpo_trainer",
        "trl.trainer.online_dpo_trainer",
    ):
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        if spec is None or spec.origin is None:
            continue
        text = open(spec.origin, encoding = "utf-8").read()
        if "class OnlineDPOTrainer" not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ClassDef) and node.name == "OnlineDPOTrainer":
                for fn in node.body:
                    if isinstance(fn, ast.FunctionDef) and fn.name == "_forward":
                        return importlib.import_module(name), ast.get_source_segment(
                            text, fn, padded = True
                        )
    pytest.skip(reason = "TRL has no OnlineDPOTrainer._forward")


def _patched_forward():
    module, source = _odpo_source()
    if "vision_inputs" not in source:
        pytest.skip(reason = "this TRL's Online DPO has no vision inputs")
    ns = dict(vars(module))
    ns.update(os = os, re = re, torch = torch, inspect = inspect, textwrap = textwrap)
    for item in rl.RL_PRE_ITEMS["online_dpo_trainer"]:
        exec(compile(item, "<online dpo pre item>", "exec"), ns)
    new = source
    for edit in rl.RL_FUNCTIONS["online_dpo_trainer"]:
        new = edit("_forward", new)
    exec(compile(textwrap.dedent(new), "<online dpo _forward>", "exec"), ns)
    return ns["_forward"]


class _Accelerator:
    device = torch.device("cpu")

    def unwrap_model(self, model):
        return model


def _batch(processor):
    from PIL import Image

    g = torch.Generator().manual_seed(0)
    image = Image.fromarray((torch.rand(56, 56, 3, generator = g) * 255).to(torch.uint8).numpy())
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "Describe it."}],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt = True)
    out = processor(images = [image, image], text = [text, text], return_tensors = "pt")
    completion_ids = torch.randint(100, 1000, (2, 6), generator = g)
    completion_mask = torch.ones_like(completion_ids)
    completion_mask[1, 4:] = 0
    return out, completion_ids, completion_mask


def _reference(model, out, completion_ids, completion_mask, mm):
    ids = torch.cat([out["input_ids"], completion_ids], dim = 1)
    kwargs = dict(
        attention_mask = torch.cat([out["attention_mask"], completion_mask], dim = 1),
        pixel_values = out["pixel_values"],
        image_grid_thw = out["image_grid_thw"],
    )
    if mm is not None:
        kwargs["mm_token_type_ids"] = torch.cat([mm, mm.new_zeros(completion_ids.shape)], dim = 1)
    logits = model(ids, **kwargs).logits[:, out["input_ids"].size(1) - 1 : -1]
    return torch.take_along_dim(
        logits.log_softmax(-1), completion_ids.unsqueeze(-1), dim = 2
    ).squeeze(-1)


@pytest.mark.parametrize("arch", sorted(TINY))
def test_online_dpo_forward_passes_mm_token_type_ids(arch):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    forward = _patched_forward()
    processor = AutoProcessor.from_pretrained(TINY[arch])
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_pretrained(
        TINY[arch], torch_dtype = torch.float32
    ).eval()
    out, completion_ids, completion_mask = _batch(processor)
    takes_mm = "mm_token_type_ids" in inspect.signature(type(model).forward).parameters
    mm = out.get("mm_token_type_ids")
    if takes_mm and mm is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        mm = (out["input_ids"] == image_token_id).long()

    trainer = type("T", (), {"_forward": forward})()
    trainer.max_length = 4096
    trainer.accelerator = _Accelerator()
    trainer.processing_class = processor
    seen = {}
    original = model.forward

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    model.forward = spy
    # What TRL < 1.17 hands _forward: the processor's image tensors, never mm_token_type_ids.
    vision_inputs = {"pixel_values": out["pixel_values"], "image_grid_thw": out["image_grid_thw"]}
    with torch.no_grad():
        got = trainer._forward(
            model,
            out["input_ids"],
            out["attention_mask"],
            completion_ids,
            completion_mask,
            vision_inputs,
        )
        model.forward = original
        want = _reference(model, out, completion_ids, completion_mask, mm if takes_mm else None)
    if takes_mm:
        expected = torch.cat([mm, mm.new_zeros(completion_ids.shape)], dim = 1)
        assert torch.equal(seen["mm_token_type_ids"].to(expected.dtype), expected)
    else:
        assert "mm_token_type_ids" not in seen
    torch.testing.assert_close(got, want)


def test_online_dpo_mm_token_type_ids_rewrite_is_scoped():
    edit = rl.online_dpo_trainer__forward_mm_token_type_ids
    text_only = (
        "def _forward(self, model, prompt_ids, prompt_mask, completion_ids, completion_mask):\n"
        "    output = model(prompt_completion_ids, attention_mask=prompt_completion_mask)\n"
    )
    assert edit("_forward", text_only) == text_only
    vision = (
        "def _forward(self, model, prompt_ids, prompt_mask, completion_ids, completion_mask, vision_inputs=None):\n"
        "    output = model(prompt_completion_ids, **model_kwargs)\n"
    )
    once = edit("_forward", vision)
    assert once.count("_unsloth_online_dpo_mm_types(") == 1
    assert edit("_forward", once) == once
    assert edit("training_step", vision) == vision
