# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A VLM repo without preprocessor_config.json still gets its native processor. No GPU needed.

stepfun-ai/Step-3.7-Flash(-FP8) ships only a remote `Step3VLProcessor` whose sizes are hardcoded,
so there is no preprocessor_config.json. Loaded untrusted, Unsloth builds the native
`Step3p7ForConditionalGeneration`, and `_construct_vlm_processor_fallback` died in
`AutoImageProcessor.from_pretrained`, leaving a bare tokenizer. The remote processor then produced
`patch_pixel_values` / `num_patches`, which the native forward ignores: "Image features and image
tokens do not match, tokens: 493, features: 169". The fallback now builds the image processor
transformers registers for the checkpoint's model_type at its class defaults.

The two functions are extracted from vision.py with ast so nothing heavy has to import; the
helpers they call are the real transformers / tokenizers objects, with the hub resolver and the
offline classifier stubbed for a local directory.
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
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(ast.get_source_segment(source, node), ns)
            wanted.discard(node.name)
    return ns, wanted


NS, MISSING = _load("_native_default_image_processor", "_construct_vlm_processor_fallback")
Step3p7Processor = getattr(transformers, "Step3p7Processor", None)


def test_functions_exist():
    assert not MISSING, f"not found in vision.py: {sorted(MISSING)}"


def _step_repo(tmp_path):
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
    config = transformers.Step3p7Config()
    config.save_pretrained(str(tmp_path))
    assert not (tmp_path / "preprocessor_config.json").exists()
    return str(tmp_path)


@pytest.mark.skipif(Step3p7Processor is None, reason = "no native step3p7 in this transformers")
def test_repo_without_preprocessor_config_gets_native_processor(tmp_path):
    repo = _step_repo(tmp_path)
    with pytest.raises(Exception):  # the defect's trigger: transformers alone cannot build it
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
