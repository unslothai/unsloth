"""LLaVA processors saved without num_additional_image_tokens (#2225, #3783)."""

from __future__ import annotations

import types

import pytest
import torch

transformers = pytest.importorskip("transformers")

from unsloth.tokenizer_utils import (
    _apply_post_load_tokenizer_fixes,
    _fix_llava_num_additional_image_tokens,
)

TINY = {
    "llava": "peft-internal-testing/tiny-LlavaForConditionalGeneration",
    "llava_next": "trl-internal-testing/tiny-LlavaNextForConditionalGeneration",
}


def _config(vision_model_type):
    return types.SimpleNamespace(vision_config = types.SimpleNamespace(model_type = vision_model_type))


def _load(repo):
    try:
        model = transformers.AutoModelForImageTextToText.from_pretrained(repo)
        processor = transformers.AutoProcessor.from_pretrained(repo)
    except OSError as e:
        pytest.skip(f"cannot fetch {repo}: {e}")
    return model.eval(), processor


def _forward(model, processor):
    from PIL import Image

    image = Image.new("RGB", (640, 480), (120, 30, 200))
    inputs = processor(images = image, text = "<image>\nDescribe", return_tensors = "pt")
    with torch.no_grad():
        model(**inputs)


@pytest.mark.parametrize("arch", sorted(TINY))
def test_processor_missing_key_is_repaired(arch):
    model, processor = _load(TINY[arch])
    # The unsloth/llava-* 16-bit repos ship processor_config.json without the key.
    processor.num_additional_image_tokens = 0
    with pytest.raises(ValueError, match = "Image features and image tokens do not match"):
        _forward(model, processor)

    fixed = _apply_post_load_tokenizer_fixes(processor, fix_tokenizer = False, config = model.config)
    assert fixed is processor and processor.num_additional_image_tokens == 1
    _forward(model, processor)


def test_saved_processor_keeps_repair(tmp_path):
    model, processor = _load(TINY["llava"])
    processor.num_additional_image_tokens = 0
    _fix_llava_num_additional_image_tokens(processor, config = model.config)
    processor.save_pretrained(tmp_path)
    reloaded = transformers.AutoProcessor.from_pretrained(tmp_path)
    assert reloaded.num_additional_image_tokens == 1


@pytest.mark.parametrize(
    "value, vision_model_type, config_style",
    [
        (1, "clip_vision_model", "object"),  # already right
        (0, "siglip_vision_model", "object"),  # no CLS token: 0 is right
        (0, None, "object"),
        (0, "clip_vision_model", "none"),
    ],
)
def test_other_processors_untouched(value, vision_model_type, config_style):
    processor = types.SimpleNamespace(num_additional_image_tokens = value)
    config = None if config_style == "none" else _config(vision_model_type)
    _fix_llava_num_additional_image_tokens(processor, config = config)
    assert processor.num_additional_image_tokens == value


def test_dict_vision_config_and_text_tokenizer():
    processor = types.SimpleNamespace(num_additional_image_tokens = 0)
    config = types.SimpleNamespace(vision_config = {"model_type": "clip_vision_model"})
    _fix_llava_num_additional_image_tokens(processor, config = config)
    assert processor.num_additional_image_tokens == 1

    tokenizer = types.SimpleNamespace()
    _fix_llava_num_additional_image_tokens(tokenizer, config = _config("clip_vision_model"))
    assert not hasattr(tokenizer, "num_additional_image_tokens")
