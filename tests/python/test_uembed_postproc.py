# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the opt-in EOS post-processor used by UEmbed (Qwen3.5) embedders.

UEmbed tokenizes `<content> <|endoftext|> x num_eos_tokens` and pools the position that
precedes that block (see `uembed_pooling.OffsetLastTokenPooling`), so the tokenizer must
actually emit the block. `build_eos_post_processor` attaches a
`tokenizers.processors.TemplateProcessing` that does exactly that.

Everything here is CPU-only and deterministic: the tokenizer is a synthetic WordLevel
fast tokenizer built in-process, so there is no download and no model.

Layers:
- Baseline characterization: a tokenizer with `num_eos_tokens = 0` is byte-for-byte the
  tokenizer it started as (these pass on the pre-change tree).
- Behavioural tests: the ACTUAL token id sequence is asserted, never the template string
  that was used to build it.
- Structural wiring test: proves the call sits behind a `num_eos_tokens > 0` guard without
  importing unsloth (the package import needs an accelerator + unsloth_zoo, which CPU boxes
  lack).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ST_SOURCE_PATH = _REPO_ROOT / "unsloth" / "models" / "sentence_transformer.py"
_POOLING_SOURCE_PATH = _REPO_ROOT / "unsloth" / "models" / "uembed_pooling.py"

_EOS_TOKEN = "<|endoftext|>"
_EOS_ID = 7
_VOCAB = {"<unk>": 0, "hello": 1, "world": 2, _EOS_TOKEN: _EOS_ID}
_CONTENT_IDS = [1, 2]  # "hello world" under _VOCAB


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _load_uembed_pooling():
    """Load `unsloth.models.uembed_pooling`, falling back to a direct file load.

    `import unsloth` runs the accelerator / unsloth_zoo gate, which legitimately refuses
    to import on a CPU-only machine. The module itself only needs torch (and, lazily,
    tokenizers), so the fallback executes the exact same source file rather than a stub.
    """
    try:
        from unsloth.models import uembed_pooling  # noqa: PLC0415
        return uembed_pooling
    except Exception:  # accelerator gate / missing unsloth_zoo / heavy optional deps
        pass

    name = "unsloth_uembed_pooling_direct"
    if name in sys.modules:
        return sys.modules[name]
    assert _POOLING_SOURCE_PATH.exists(), f"missing module file: {_POOLING_SOURCE_PATH}"
    spec = importlib.util.spec_from_file_location(name, _POOLING_SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "module")
def uembed():
    return _load_uembed_pooling()


def _fast_tokenizer(vocab: dict[str, int] | None = None):
    """A real (tiny, synthetic) `PreTrainedTokenizerFast` -- no download, no network."""
    pytest.importorskip("tokenizers", reason = "tokenizers not importable")
    transformers = pytest.importorskip("transformers", reason = "transformers not importable")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    backend = Tokenizer(WordLevel(dict(_VOCAB if vocab is None else vocab), unk_token = "<unk>"))
    backend.pre_tokenizer = Whitespace()
    return transformers.PreTrainedTokenizerFast(tokenizer_object = backend, unk_token = "<unk>")


def _ids(tokenizer, *text: str) -> list[int]:
    return tokenizer(*text)["input_ids"]


def _st_source_tree() -> ast.Module:
    return ast.parse(_ST_SOURCE_PATH.read_text(encoding = "utf-8"))


def _function_def(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {_ST_SOURCE_PATH}")


def _calls_to(node: ast.AST, func_name: str) -> list[ast.Call]:
    calls = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == func_name:
                calls.append(child)
    return calls


# --------------------------------------------------------------------------------------
# baseline characterization -- these pass BEFORE the feature exists
# --------------------------------------------------------------------------------------
def test_baseline_untouched_tokenizer_appends_nothing():
    tokenizer = _fast_tokenizer()

    assert _ids(tokenizer, "hello world") == _CONTENT_IDS
    assert tokenizer.backend_tokenizer.post_processor is None


def test_num_eos_tokens_zero_leaves_the_tokenizer_byte_for_byte_unchanged(uembed):
    """The default (non-UEmbed) path: no post-processor, no padding-side rewrite."""
    tokenizer = _fast_tokenizer()
    tokenizer.padding_side = "left"

    result = uembed.build_eos_post_processor(tokenizer, 0)

    assert result is None
    assert tokenizer.backend_tokenizer.post_processor is None
    assert _ids(tokenizer, "hello world") == _ids(_fast_tokenizer(), "hello world")
    assert tokenizer.padding_side == "left", "num_eos_tokens = 0 must not touch padding_side"


# --------------------------------------------------------------------------------------
# behaviour -- assert the ACTUAL token ids
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("num_eos_tokens", [1, 16])
def test_exactly_n_trailing_eos_ids_are_appended(uembed, num_eos_tokens):
    tokenizer = _fast_tokenizer()

    uembed.build_eos_post_processor(tokenizer, num_eos_tokens)
    ids = _ids(tokenizer, "hello world")

    assert ids[: len(_CONTENT_IDS)] == _CONTENT_IDS, "content ids must be unchanged"
    assert ids[len(_CONTENT_IDS) :] == [_EOS_ID] * num_eos_tokens
    assert len(ids) == len(_CONTENT_IDS) + num_eos_tokens
    assert ids.count(_EOS_ID) == num_eos_tokens, "the block is the only source of EOS ids"


def test_neighbouring_block_sizes_differ(uembed):
    """Guards the suite itself: an off-by-one template cannot pass silently."""
    lengths = []
    for num_eos_tokens in (1, 2, 3):
        tokenizer = _fast_tokenizer()
        uembed.build_eos_post_processor(tokenizer, num_eos_tokens)
        lengths.append(len(_ids(tokenizer, "hello world")))

    assert lengths == [len(_CONTENT_IDS) + n for n in (1, 2, 3)]


def test_pair_encoding_appends_the_block_after_both_segments(uembed):
    tokenizer = _fast_tokenizer()

    uembed.build_eos_post_processor(tokenizer, 3)
    ids = _ids(tokenizer, "hello", "world")

    assert ids == [1, _EOS_ID, _EOS_ID, _EOS_ID, 2, _EOS_ID, _EOS_ID, _EOS_ID]


def test_empty_input_still_yields_exactly_the_block(uembed):
    """Malformed/degenerate input: an empty string must not lose or duplicate the block."""
    tokenizer = _fast_tokenizer()

    uembed.build_eos_post_processor(tokenizer, 16)

    assert _ids(tokenizer, "") == [_EOS_ID] * 16


def test_padding_side_is_forced_right(uembed):
    """Offset pooling counts back from the last real token, so padding must be on the right."""
    tokenizer = _fast_tokenizer()
    tokenizer.padding_side = "left"

    uembed.build_eos_post_processor(tokenizer, 16)

    assert tokenizer.padding_side == "right"


def test_a_processor_is_unwrapped_to_its_inner_tokenizer(uembed):
    """The model load path may return a processor object rather than a bare tokenizer."""

    class _Processor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.image_processor = object()

    tokenizer = _fast_tokenizer()
    processor = _Processor(tokenizer)

    uembed.build_eos_post_processor(processor, 16)

    assert _ids(tokenizer, "hello world") == _CONTENT_IDS + [_EOS_ID] * 16


def test_eos_token_missing_from_the_vocabulary_raises(uembed):
    """Silently appending `unk` ids would corrupt every embedding the model produces."""
    tokenizer = _fast_tokenizer({"<unk>": 0, "hello": 1, "world": 2})

    with pytest.raises(ValueError, match = "endoftext"):
        uembed.build_eos_post_processor(tokenizer, 16)


def test_tokenizer_without_a_fast_backend_raises(uembed):
    class _SlowTokenizer:
        padding_side = "left"

        def convert_tokens_to_ids(self, token):
            return _EOS_ID

    with pytest.raises(ValueError, match = "fast"):
        uembed.build_eos_post_processor(_SlowTokenizer(), 16)


@pytest.mark.parametrize("value", [-1, 1.5, "16", None, True])
def test_malformed_num_eos_tokens_raises(uembed, value):
    with pytest.raises(ValueError, match = "num_eos_tokens"):
        uembed.build_eos_post_processor(_fast_tokenizer(), value)


# --------------------------------------------------------------------------------------
# opt-in wiring
# --------------------------------------------------------------------------------------
def test_post_processor_is_wired_behind_a_num_eos_tokens_guard():
    """`build_eos_post_processor` may only run when the checkpoint asks for a block."""
    from_pretrained = _function_def(_st_source_tree(), "from_pretrained")

    calls = _calls_to(from_pretrained, "build_eos_post_processor")
    assert len(calls) == 1, "expected exactly one post-processor attachment"

    guards = [
        node
        for node in ast.walk(from_pretrained)
        if isinstance(node, ast.If) and _calls_to(node, "build_eos_post_processor")
    ]
    positive_guards = [
        node
        for node in guards
        if any(
            isinstance(compare, ast.Compare)
            and isinstance(compare.left, ast.Name)
            and compare.left.id == "num_eos_tokens"
            and isinstance(compare.ops[0], ast.Gt)
            and isinstance(compare.comparators[0], ast.Constant)
            and compare.comparators[0].value == 0
            for compare in ast.walk(node.test)
        )
    ]
    assert len(positive_guards) == 1, "attachment must sit behind `num_eos_tokens > 0`"

    assert _calls_to(
        from_pretrained, "read_num_eos_tokens"
    ), "num_eos_tokens must come from the checkpoint's sparse_info.json"
    for call in calls:
        assert not any(
            isinstance(argument, ast.Constant) for argument in call.args
        ), "the block size must not be hardcoded"
