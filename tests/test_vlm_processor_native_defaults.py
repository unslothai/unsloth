# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A VLM repo without preprocessor_config.json (Step-3.7-Flash) still gets its native processor.

Without it the remote processor's patch_pixel_values are ignored by the native forward ("tokens: 493,
features: 169"). Functions are ast-extracted from vision.py; hub resolver and offline check stubbed.
"""

import ast
import json
import os

import pytest

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISION = os.path.join(HERE, "unsloth", "models", "vision.py")


def _load(*names):
    source = open(VISION, encoding = "utf-8").read()

    def _tokenizer_fast(
        name,
        padding_side = "left",
        **kwargs,
    ):
        return transformers.PreTrainedTokenizerFast.from_pretrained(name, padding_side = padding_side)

    ns = {
        "os": os,
        "_hub_repo_or_local_path": lambda name, **kwargs: name,
        "_load_pretrained_tokenizer_fast": _tokenizer_fast,
        "_is_offline_related_error": lambda e: False,
    }
    wanted = set(names)
    for node in ast.parse(source).body:
        targets = [t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)]
        name = node.name if isinstance(node, ast.FunctionDef) else (targets or [None])[0]
        if name in wanted:
            exec(ast.get_source_segment(source, node), ns)
            wanted.discard(name)
    return ns, wanted


NS, MISSING = _load(
    "_NATIVE_DEFAULT_IMAGE_PROCESSOR_TYPES",
    "_missing_torchvision_error",
    "_preprocessor_config_exists",
    "_native_default_image_processor",
    "_construct_vlm_processor_fallback",
)
Step3p7Processor = getattr(transformers, "Step3p7Processor", None)


def test_functions_exist():
    assert not MISSING, f"not found in vision.py: {sorted(MISSING)}"


def _tokenizer_repo(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    specials = [
        "<unk>",
        "<im_patch>",
        "<patch_start>",
        "<patch_end>",
        "<patch_newline>",
        "<im_start>",
        "<im_end>",
    ]
    vocab = {t: i for i, t in enumerate(specials + ["hi", "there"])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(specials)
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"unk_token": "<unk>"}))
    return str(tmp_path)


def _step_repo(tmp_path):
    _tokenizer_repo(tmp_path)
    config = transformers.Step3p7Config()
    config.save_pretrained(str(tmp_path))
    assert not (tmp_path / "preprocessor_config.json").exists()
    return str(tmp_path)


@pytest.mark.skipif(Step3p7Processor is None, reason = "no native step3p7 in this transformers")
def test_repo_without_preprocessor_config_gets_native_processor(tmp_path):
    repo = _step_repo(tmp_path)
    with pytest.raises(Exception):
        transformers.AutoImageProcessor.from_pretrained(repo)
    # The loader passes the text sub-config's model_type (step3p5); the top-level one wins.
    processor, err = NS["_construct_vlm_processor_fallback"](repo, "step3p5", None, False)
    assert err is None, err
    assert type(processor).__name__ == "Step3p7Processor"
    ip = processor.image_processor
    assert ip.size["height"] == 728 and ip.patch_size == 504
    from PIL import Image

    batch = processor(
        images = [Image.new("RGB", (800, 300))], text = ["<im_patch> hi"], return_tensors = "pt"
    )
    assert {"pixel_values", "pixel_values_local", "num_local_patches"} <= set(batch)
    n_local = int(batch["num_local_patches"][0])
    n_tokens = int((batch["input_ids"] == processor.image_token_id).sum())
    assert n_tokens == processor.num_image_feature_size + n_local * processor.num_patch_feature_size


def test_unknown_model_type_returns_none(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "not_a_real_model_type"}))
    assert NS["_native_default_image_processor"](str(tmp_path), "not_a_real_model_type") is None


@pytest.mark.skipif(Step3p7Processor is None, reason = "no native step3p7 in this transformers")
def test_broken_preprocessor_config_is_not_replaced_by_defaults(tmp_path):
    repo = _step_repo(tmp_path)
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"image_processor_type": "NoSuchImageProcessor"})
    )
    processor, err = NS["_construct_vlm_processor_fallback"](repo, "step3p5", None, False)
    assert processor is None and err is not None


def test_unlisted_vlm_without_preprocessor_config_still_fails(tmp_path):
    # LLaVA has a registered image processor, but its class defaults need not match the checkpoint.
    repo = _tokenizer_repo(tmp_path)
    transformers.LlavaConfig().save_pretrained(repo)
    processor, err = NS["_construct_vlm_processor_fallback"](repo, "llava", None, False)
    assert processor is None and err is not None
