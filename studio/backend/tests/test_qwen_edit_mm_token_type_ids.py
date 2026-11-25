# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-Edit prompt embeddings match a Qwen2.5-VL forward given the processor's own inputs (diffusers #14189)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
pytest.importorskip("PIL")

from PIL import Image

from core.inference import diffusion_qwen_vl_mm_types as mm_types

REPO = "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"
TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


@pytest.fixture(scope = "module")
def tiny_vl():
    from huggingface_hub import snapshot_download
    try:
        path = snapshot_download(REPO, local_files_only = True)
        processor = transformers.AutoProcessor.from_pretrained(path)
        model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, dtype = torch.float32
        ).eval()
    except Exception as exc:  # noqa: BLE001 - LocalEntryNotFoundError and friends
        pytest.skip(f"{REPO} not in the Hub cache: {exc}")
    return processor, model


def _edit_pipe(processor, model):
    pipeline_cls = getattr(diffusers, "QwenImageEditPlusPipeline", None)
    if pipeline_cls is None:
        pytest.skip("this diffusers has no QwenImageEditPlusPipeline")

    class _Pipe:
        _get_qwen_prompt_embeds = pipeline_cls._get_qwen_prompt_embeds
        _extract_masked_hidden = pipeline_cls._extract_masked_hidden

    pipe = _Pipe()
    pipe.processor = processor
    pipe.text_encoder = model
    pipe.prompt_template_encode = TEMPLATE
    pipe.prompt_template_encode_start_idx = 0
    return pipe


def _images():
    gen = torch.Generator().manual_seed(0)
    return [
        Image.fromarray((torch.rand(112, 140, 3, generator = gen) * 255).byte().numpy()),
        Image.fromarray((torch.rand(84, 84, 3, generator = gen) * 255).byte().numpy()),
    ]


def _reference(processor, model, prompt, images):
    text = "".join(
        f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>" for i in range(len(images))
    )
    inputs = processor(
        text = [TEMPLATE.format(text + prompt)], images = images, padding = True, return_tensors = "pt"
    )
    with torch.no_grad():
        return model(**inputs, output_hidden_states = True).hidden_states[-1][0]


def test_edit_prompt_embeds_match_the_processor_reference(tiny_vl):
    processor, model = tiny_vl
    images = _images()
    prompt = "Make the sky purple"
    pipe = _edit_pipe(processor, model)
    reference = _reference(processor, model, prompt, images)
    accepts = "mm_token_type_ids" in inspect.signature(model.forward).parameters
    with torch.no_grad():
        stock, _ = pipe._get_qwen_prompt_embeds(prompt, images, device = torch.device("cpu"))
    # transformers 5.x: the stock pipeline drops the ids and its image tokens get 1D positions.
    assert torch.equal(stock[0], reference) != accepts

    engaged = mm_types.forward_mm_token_type_ids(pipe)
    assert engaged == accepts
    assert mm_types.forward_mm_token_type_ids(pipe) == engaged

    with torch.no_grad():
        embeds, mask = pipe._get_qwen_prompt_embeds(prompt, images, device = torch.device("cpu"))
    assert mask.shape[1] == reference.shape[0]
    assert torch.equal(embeds[0], reference)

    # The hook is scoped to the prompt encode: a direct encoder call afterwards is unchanged.
    assert not model._forward_pre_hooks


def test_ids_match_the_processor(tiny_vl):
    processor, model = tiny_vl
    if not hasattr(processor, "create_mm_token_type_ids"):
        pytest.skip("this transformers' processor has no create_mm_token_type_ids")
    inputs = processor(
        text = [TEMPLATE.format("Picture 1: <|vision_start|><|image_pad|><|vision_end|>edit")],
        images = _images()[:1],
        return_tensors = "pt",
    )
    expected = torch.tensor(processor.create_mm_token_type_ids(inputs["input_ids"].tolist()))
    assert torch.equal(mm_types.mm_token_type_ids(inputs["input_ids"], model.config), expected)


def test_text_only_and_kill_switch(tiny_vl, monkeypatch):
    processor, model = tiny_vl
    model_kwargs = {"input_ids": torch.tensor([[1, 2, 3]])}
    _, out = mm_types._add_mm_token_type_ids(model, (), dict(model_kwargs))
    assert "mm_token_type_ids" not in out  # no image: 1D positions are already the 3D ones
    monkeypatch.setenv("UNSLOTH_QWEN_VL_MM_TOKEN_TYPES", "0")
    pipe = _edit_pipe(processor, model)
    original = pipe._get_qwen_prompt_embeds
    assert mm_types.forward_mm_token_type_ids(pipe) is False
    assert pipe._get_qwen_prompt_embeds == original


def test_loader_installs_it():
    source = (
        Path(__file__).resolve().parents[1] / "core" / "inference" / "diffusion.py"
    ).read_text(encoding = "utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "forward_mm_token_type_ids"
    ]
    assert len(calls) == 1
