# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Processor selection for repo-code VLMs that register only AutoModel /
AutoModelForCausalLM.

nvidia/Nemotron-3-Nano-Omni maps only AutoModel / AutoModelForCausalLM, so
auto_model is not a VLM class and FastBaseModel picked AutoTokenizer, returning
a bare tokenizer although the repo ships an AutoProcessor. DeepSeek-OCR (no
loadable AutoProcessor) and Nemotron-Nano-VL (AutoProcessor resolves to the
tokenizer) must keep loading through AutoTokenizer.

The selection statements and `_acquire_processor` are exec'd straight from
vision.py with stub processor classes, so no network, weights or GPU are used.
"""

import ast
import importlib.util
import types
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")

VISION = Path(__file__).parents[1] / "unsloth" / "models" / "vision.py"
if not VISION.exists():  # copied out of the repo: read the tree on PYTHONPATH
    VISION = (
        Path(importlib.util.find_spec("unsloth").submodule_search_locations[0])
        / "models"
        / "vision.py"
    )
_SELECTION_NAMES = {
    "is_vlm",
    "is_whisper",
    "needs_processor",
    "is_vlm_config",
    "auto_processor",
    "try_repo_processor",
}


def _from_pretrained():
    tree = ast.parse(VISION.read_text(encoding = "utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FastBaseModel")
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained"
    )


def _selection_nodes():
    nodes = []
    for node in _from_pretrained().body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) in _SELECTION_NAMES for t in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "_acquire_processor":
            nodes.append(node)
    return nodes


class _Tokenizer:
    padding_side = "left"


class _Processor:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.image_processor = object()


def _stub(result, calls, name):
    def from_pretrained(path, **kwargs):
        calls.append((name, kwargs))
        if isinstance(result, Exception):
            raise result
        return result() if isinstance(result, type) else result

    return type(name, (), {"from_pretrained": staticmethod(from_pretrained)})


def _select(
    auto_processor_result,
    *,
    has_vision_config = True,
    text_only = False,
):
    calls = []
    fake_processor = _stub(auto_processor_result, calls, "AutoProcessor")
    fake_tokenizer = _stub(_Tokenizer, calls, "AutoTokenizer")
    config = types.SimpleNamespace()
    if has_vision_config:
        config.vision_config = types.SimpleNamespace()
    namespace = {
        "auto_model": transformers.AutoModelForCausalLM,
        "auto_config": config,
        "text_only": text_only,
        "whisper_language": None,
        "whisper_task": None,
        "AutoModelForVision2Seq": getattr(
            transformers, "AutoModelForVision2Seq", transformers.AutoModelForImageTextToText
        ),
        "AutoModelForImageTextToText": transformers.AutoModelForImageTextToText,
        "_multimodal_auto_classes": lambda: (),
        "AutoProcessor": fake_processor,
        "AutoTokenizer": fake_tokenizer,
        "tokenizer_name": "org/repo",
        "token": None,
        "trust_remote_code": True,
        "kwargs": {},
        "_tokenizer_revision": "abc123",
        "model_type_arch": "remote_vlm",
        "get_auto_processor": lambda *a, **k: None,
        "_construct_vlm_processor_fallback": lambda *a, **k: (None, None),
        "_is_offline_related_error": lambda e: False,
    }
    module = ast.Module(body = _selection_nodes(), type_ignores = [])
    ast.fix_missing_locations(module)
    exec(compile(module, str(VISION), "exec"), namespace)
    tok, err = namespace["_acquire_processor"](False)
    return tok, err, calls


def test_repo_autoprocessor_is_used_for_automodel_only_vlm():
    """Nemotron-3-Nano-Omni: the defect. Fails before the fix (bare tokenizer)."""
    tok, err, calls = _select(_Processor)
    assert isinstance(tok, _Processor), f"got {type(tok).__name__}, calls={calls}"
    assert hasattr(tok, "image_processor")
    name, kwargs = calls[0]
    assert name == "AutoProcessor"
    assert kwargs["padding_side"] == "left"
    assert kwargs["revision"] == "abc123"
    assert kwargs["trust_remote_code"] is True


def test_missing_autoprocessor_falls_back_to_tokenizer():
    """DeepSeek-OCR: AutoProcessor raises 'Unrecognized processing class'."""
    tok, err, calls = _select(ValueError("Unrecognized processing class"))
    assert isinstance(tok, _Tokenizer)
    assert err is None
    assert [c[0] for c in calls][-1] == "AutoTokenizer"


def test_autoprocessor_that_resolves_to_a_tokenizer_is_not_kept():
    """Nemotron-Nano-VL: AutoProcessor returns the tokenizer; keep the AutoTokenizer load."""
    tok, err, calls = _select(_Tokenizer)
    assert isinstance(tok, _Tokenizer)
    assert [c[0] for c in calls][-1] == "AutoTokenizer"


def test_text_model_never_tries_autoprocessor():
    tok, err, calls = _select(_Processor, has_vision_config = False)
    assert isinstance(tok, _Tokenizer)
    assert [c[0] for c in calls] == ["AutoTokenizer"]


def test_text_only_vlm_never_tries_autoprocessor():
    tok, err, calls = _select(_Processor, text_only = True)
    assert isinstance(tok, _Tokenizer)
    assert [c[0] for c in calls] == ["AutoTokenizer"]
