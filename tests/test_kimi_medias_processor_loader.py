# SPDX-License-Identifier: AGPL-3.0-only

"""FastVisionModel patches Kimi K2.5 / K2.7 medias= processors; the block is ast-extracted since importing unsloth needs a GPU."""

import ast
import os
import sys
import types

import pytest

VISION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unsloth",
    "models",
    "vision.py",
)


def _from_pretrained():
    tree = ast.parse(open(VISION_PATH, encoding = "utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FastBaseModel":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "from_pretrained":
                    return item
    raise AssertionError("FastBaseModel.from_pretrained not found")


def _patch_block():
    found = [
        node
        for node in ast.walk(_from_pretrained())
        if isinstance(node, ast.If)
        and "patch_medias_processor" in ast.unparse(node)
        and ast.unparse(node.test) == "hasattr(tokenizer, 'image_processor')"
    ]
    assert len(found) == 1, "from_pretrained does not patch Kimi medias= processors"
    return compile(ast.Module(body = found, type_ignores = []), VISION_PATH, "exec")


def _run_block(tokenizer, vision_utils_module, monkeypatch):
    zoo = types.ModuleType("unsloth_zoo")
    zoo.vision_utils = vision_utils_module
    monkeypatch.setitem(sys.modules, "unsloth_zoo", zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.vision_utils", vision_utils_module)
    exec(_patch_block(), {"tokenizer": tokenizer})


class _KimiLikeProcessor:
    def __init__(self):
        self.image_processor = object()
        self.tokenizer = object()

    def __call__(
        self,
        messages = None,
        medias = None,
        text = None,
        return_tensors = "pt",
        **kwargs,
    ):
        raise ValueError("Provide either 'messages' or both 'medias' and 'text'")


def test_loader_applies_patch_medias_processor_to_the_processor(monkeypatch):
    seen = []
    vision_utils = types.ModuleType("unsloth_zoo.vision_utils")
    vision_utils.patch_medias_processor = lambda processor: seen.append(processor) or True
    processor = _KimiLikeProcessor()
    _run_block(processor, vision_utils, monkeypatch)
    assert seen == [processor]


def test_loader_skips_on_unsloth_zoo_without_the_helper(monkeypatch):
    _run_block(_KimiLikeProcessor(), types.ModuleType("unsloth_zoo.vision_utils"), monkeypatch)


def test_loader_leaves_text_tokenizers_alone(monkeypatch):
    seen = []
    vision_utils = types.ModuleType("unsloth_zoo.vision_utils")
    vision_utils.patch_medias_processor = lambda processor: seen.append(processor) or True
    _run_block(types.SimpleNamespace(tokenizer = object()), vision_utils, monkeypatch)
    assert seen == []


def test_patched_kimi_processor_takes_text_and_images(monkeypatch):
    real = pytest.importorskip("unsloth_zoo.vision_utils")
    if not hasattr(real, "patch_medias_processor"):
        pytest.skip(
            "installed unsloth_zoo predates patch_medias_processor (unslothai/unsloth-zoo#1442)"
        )
    torch = pytest.importorskip("torch")
    from transformers.feature_extraction_utils import BatchFeature

    class _Tokenizer:
        # Encodes each word as its length, so a token's id is its length too. unsloth_zoo
        # (#1442) counts the image placeholders left after tokenizing through this.
        def convert_tokens_to_ids(self, token):
            return len(token)

        def __call__(
            self,
            text,
            return_tensors = "pt",
            padding = False,
            **kwargs,
        ):
            texts = [text] if isinstance(text, str) else text
            ids = [[len(word) for word in t.split()] for t in texts]
            width = max(map(len, ids))
            return {
                "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in ids]),
                "attention_mask": torch.tensor(
                    [[1] * len(row) + [0] * (width - len(row)) for row in ids]
                ),
            }

    class _MediaProcessor:
        def preprocess(
            self,
            medias,
            return_tensors = "pt",
        ):
            return BatchFeature(
                {
                    "pixel_values": torch.zeros(len(medias), 3, 4, 4),
                    "grid_thws": torch.ones(len(medias), 3, dtype = torch.long),
                }
            )

    class KimiK25ProcessorStandIn(_KimiLikeProcessor):
        image_token = "<|media_pad|>"

        def __init__(self):
            self.image_processor = _MediaProcessor()
            self.tokenizer = _Tokenizer()

    processor = KimiK25ProcessorStandIn()
    with pytest.raises(ValueError, match = "medias"):
        processor(text = ["a <|media_pad|> b"], images = [object()])
    monkeypatch.setitem(sys.modules, "unsloth_zoo.vision_utils", real)
    exec(_patch_block(), {"tokenizer": processor})
    out = processor(text = ["a <|media_pad|> b", "c <|media_pad|>"], images = [object(), object()])
    assert out["input_ids"].shape == (2, 3)
    assert out["pixel_values"].shape[0] == 2
    assert out["grid_thws"].shape == (2, 3)
