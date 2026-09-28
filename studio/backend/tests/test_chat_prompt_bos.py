# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A rendered chat prompt must carry exactly one BOS.

Most templates emit it, so letting the tokenizer add another doubles it; zephyr and
tinyllama-chat emit none, so suppressing specials unconditionally drops it. The deciding
fact is whether the rendered text already starts with BOS.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_STUBBED: list[str] = []


def _stub_if_missing(
    name,
    attrs = (),
    named_spec = False,
):
    """Register a stub for a dep this job does not install. A real install is left alone.

    Same helper and reason as test_vision_client_tools.py: core.inference.inference imports
    unsloth and trl at module scope, which studio-backend-ci.yml does not install, so
    unstubbed this file fails COLLECTION and takes the whole job down.

    ``named_spec`` gives the stub a real ModuleSpec, which only torchao needs: transformers
    probes it with find_spec, which raises ValueError on ``__spec__ = None``.
    """
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - unusable here either way, so stub it
        pass
    _STUBBED.append(name)
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None) if named_spec else None
    module.__version__ = "0.0.0"
    module.__getattr__ = lambda _attr: MagicMock()
    for attr in attrs:
        setattr(module, attr, MagicMock())
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, module)


# Fires only where torchao is installed but unusable against the local torch, in which
# case transformers.quantizers imports it and poisons transformers for every later module.
for _torchao in (
    "torchao",
    "torchao.prototype",
    "torchao.prototype.safetensors",
    "torchao.prototype.safetensors.safetensors_support",
    "torchao.prototype.safetensors.safetensors_utils",
    "torchao.quantization",
    "torchao.dtypes",
    "torchao.float8",
    "torchao.utils",
):
    _stub_if_missing(_torchao, named_spec = True)

_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("unsloth_zoo")
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

from core.inference.inference import _prompt_already_has_bos  # noqa: E402

# Drop the stubs now the name is bound; one left behind outlives this module.
for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


def _tokenizer(auto_adds_bos: bool):
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast

    vocab = {"<s>": 0, "</s>": 1, "hello": 2, "world": 3, "user": 4}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token = None))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object = backend, bos_token = "<s>", eos_token = "</s>")
    if auto_adds_bos:
        tokenizer.add_bos_token = True
    return tokenizer


def _encode(tokenizer, prompt):
    add_special_tokens = not _prompt_already_has_bos(tokenizer, prompt)
    return tokenizer(prompt, add_special_tokens = add_special_tokens)["input_ids"]


def _leading_bos(ids, bos_token_id):
    count = 0
    for token_id in ids:
        if token_id != bos_token_id:
            break
        count += 1
    return count


@pytest.mark.parametrize("auto_adds_bos", [True, False])
@pytest.mark.parametrize("template_emits_bos", [True, False])
def test_rendered_prompt_carries_exactly_one_bos(template_emits_bos, auto_adds_bos):
    tokenizer = _tokenizer(auto_adds_bos)
    if not (template_emits_bos or auto_adds_bos):
        pytest.skip("nothing supplies BOS, so there is none to count")
    prompt = ("<s> " if template_emits_bos else "") + "user hello world"

    assert _leading_bos(_encode(tokenizer, prompt), tokenizer.bos_token_id) == 1


def test_template_that_omits_bos_keeps_the_tokenizer_one():
    # Blanket add_special_tokens = False regressed exactly this shape.
    tokenizer = _tokenizer(auto_adds_bos = True)
    prompt = "user hello world"

    assert _prompt_already_has_bos(tokenizer, prompt) is False
    assert _encode(tokenizer, prompt)[0] == tokenizer.bos_token_id


def test_template_that_emits_bos_is_not_doubled():
    tokenizer = _tokenizer(auto_adds_bos = True)
    prompt = "<s> user hello world"

    assert _prompt_already_has_bos(tokenizer, prompt) is True
    assert _leading_bos(_encode(tokenizer, prompt), tokenizer.bos_token_id) == 1


def test_tokenizer_without_a_bos_token_is_left_alone():
    tokenizer = _tokenizer(auto_adds_bos = False)
    tokenizer.bos_token = None

    assert _prompt_already_has_bos(tokenizer, "user hello") is False


def test_a_tokenizer_that_raises_is_treated_as_having_no_bos():
    class _Raises:
        bos_token_id = 0

        def __call__(self, *args, **kwargs):
            raise RuntimeError("needs an image")

    assert _prompt_already_has_bos(_Raises(), "user hello") is False


def test_a_processor_is_probed_through_its_inner_tokenizer():
    import types

    tokenizer = _tokenizer(auto_adds_bos = True)
    processor = types.SimpleNamespace(tokenizer = tokenizer)

    assert _prompt_already_has_bos(processor, "<s> user hello") is True
    assert _prompt_already_has_bos(processor, "user hello") is False


def test_empty_prompt_does_not_index_out_of_range():
    assert _prompt_already_has_bos(_tokenizer(auto_adds_bos = True), "") is False
