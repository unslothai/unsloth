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

"""Gemma 4 base tokenizers must prepend <bos> at load time.

unsloth/gemma-4-* base mirrors ship without add_bos_token: true while
google/gemma-4-* includes it. Without the runtime fix, generation repeats
degenerate text. See unslothai/unsloth#7903.

Detection keys off the loaded tokenizer / config, not the Hub repo name, so
local folders and extra quant suffixes still get the fix.
"""

import types
from unittest.mock import patch

import pytest

import unsloth.tokenizer_utils as tu


class _Tok:
    def __init__(
        self,
        add_bos_token = False,
        bos_token_id = 2,
        processor_class = None,
        chat_template = None,
        eos_token = "<eos>",
        init_kwargs = None,
    ):
        self.add_bos_token = add_bos_token
        self.bos_token_id = bos_token_id
        self.processor_class = processor_class
        self.chat_template = chat_template
        self.eos_token = eos_token
        if init_kwargs is not None:
            self.init_kwargs = init_kwargs


class _Proc:
    def __init__(
        self,
        tokenizer,
        processor_class = "Gemma4Processor",
        chat_template = None,
    ):
        self.tokenizer = tokenizer
        self.processor_class = processor_class
        self.chat_template = chat_template


def _gemma4_base(**kwargs):
    kwargs.setdefault("processor_class", "Gemma4Processor")
    return _Tok(**kwargs)


def test_gemma4_from_processor_class():
    assert tu._is_gemma4_tokenizer(_gemma4_base()) is True


def test_gemma4_from_init_kwargs():
    tok = _Tok(init_kwargs = {"processor_class": "Gemma4Processor"})
    assert tu._is_gemma4_tokenizer(tok) is True


def test_gemma4_from_processor_wrapper():
    proc = _Proc(_Tok())
    assert tu._is_gemma4_tokenizer(proc) is True


def test_gemma3_processor_is_not_gemma4():
    tok = _Tok(processor_class = "Gemma3Processor")
    assert tu._is_gemma4_tokenizer(tok) is False
    assert tu._needs_gemma4_base_bos(tok) is False


def test_plain_tokenizer_is_not_gemma4():
    assert tu._is_gemma4_tokenizer(_Tok()) is False


def test_gemma4_config_model_type():
    config = types.SimpleNamespace(model_type = "gemma4", text_config = None)
    assert tu._is_gemma4_config(config) is True
    assert tu._needs_gemma4_base_bos(_Tok(), config = config) is True


def test_gemma4_config_nested_text_config():
    config = types.SimpleNamespace(
        model_type = "gemma4",
        text_config = types.SimpleNamespace(model_type = "gemma4_text"),
    )
    assert tu._is_gemma4_config(config) is True


def test_name_alone_does_not_trigger_fix():
    tok = _Tok()
    # Repo / folder names are ignored: a generic tokenizer must not flip BOS.
    fixed = tu._fix_gemma4_base_bos_token(tok)
    assert fixed.add_bos_token is False


def test_fix_sets_flag_for_quant_and_local_shapes():
    tok = _gemma4_base(add_bos_token = False)
    fixed = tu._fix_gemma4_base_bos_token(tok)
    assert fixed.add_bos_token is True


def test_fix_sets_flag_on_wrapped_processor():
    inner = _Tok(add_bos_token = False)
    proc = _Proc(inner)
    tu._fix_gemma4_base_bos_token(proc)
    assert inner.add_bos_token is True


def test_fix_skips_chat_template_that_emits_bos():
    tok = _gemma4_base(
        add_bos_token = False,
        chat_template = "{{- bos_token -}}{{ messages }}",
    )
    fixed = tu._fix_gemma4_base_bos_token(tok)
    assert fixed.add_bos_token is False


def test_fix_skips_turn_eos_instruct():
    tok = _gemma4_base(add_bos_token = False, eos_token = "<turn|>")
    fixed = tu._fix_gemma4_base_bos_token(tok)
    assert fixed.add_bos_token is False


def test_fix_honors_fix_tokenizer_false():
    tok = _gemma4_base(add_bos_token = False)
    fixed = tu._apply_post_load_tokenizer_fixes(tok, fix_tokenizer = False)
    assert fixed.add_bos_token is False


def test_load_correct_tokenizer_enables_bos_for_gemma4_base():
    def from_pretrained(model_name, **kwargs):
        return _gemma4_base(add_bos_token = False)

    with patch.object(tu, "AutoTokenizer", types.SimpleNamespace(from_pretrained = from_pretrained)):
        result = tu._load_correct_tokenizer("/models/gemma-4-31B-bnb-4bit", fix_tokenizer = True)

    assert result.add_bos_token is True


def test_load_correct_tokenizer_skips_instruct():
    def from_pretrained(model_name, **kwargs):
        return _gemma4_base(
            add_bos_token = False,
            chat_template = "{{- bos_token -}}",
            eos_token = "<turn|>",
        )

    with patch.object(tu, "AutoTokenizer", types.SimpleNamespace(from_pretrained = from_pretrained)):
        result = tu._load_correct_tokenizer("unsloth/gemma-4-E2B-it", fix_tokenizer = True)

    assert result.add_bos_token is False


def test_load_correct_tokenizer_uses_model_config_when_tokenizer_is_generic():
    # Stripped local tokenizers have no processor_class; FastLanguageModel still
    # has model.config.model_type == gemma4.
    def from_pretrained(model_name, **kwargs):
        return _Tok(add_bos_token = False)

    config = types.SimpleNamespace(model_type = "gemma4", text_config = None)
    with patch.object(tu, "AutoTokenizer", types.SimpleNamespace(from_pretrained = from_pretrained)):
        result = tu._load_correct_tokenizer(
            "/models/local-gemma4-bnb-4bit",
            fix_tokenizer = True,
            config = config,
        )

    assert result.add_bos_token is True


def test_fastmodel_processor_path_heals_from_config():
    # FastModel loads Gemma4Processor, then heals after the processor is final.
    inner = _Tok(add_bos_token = False)
    processor = types.SimpleNamespace(
        tokenizer = inner,
        image_processor = object(),
        chat_template = None,
    )
    config = types.SimpleNamespace(model_type = "gemma4", text_config = None)

    fixed = tu._apply_post_load_tokenizer_fixes(processor, fix_tokenizer = True, config = config)

    assert fixed is processor
    assert inner.add_bos_token is True


def test_fastmodel_processor_path_skips_instruct_template():
    inner = _Tok(add_bos_token = False, chat_template = "{{- bos_token -}}")
    processor = types.SimpleNamespace(
        tokenizer = inner,
        image_processor = object(),
        chat_template = "{{- bos_token -}}{{ messages }}",
    )
    config = types.SimpleNamespace(model_type = "gemma4", text_config = None)

    tu._apply_post_load_tokenizer_fixes(processor, fix_tokenizer = True, config = config)
    assert inner.add_bos_token is False


@pytest.mark.e2e
@pytest.mark.slow
def test_gemma4_e2b_hub_tokenizer_prepends_bos():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tok = tu.load_correct_tokenizer("unsloth/gemma-4-E2B", fix_tokenizer = True)
    assert tok.add_bos_token is True
    ids = tok("This book is largely concerned with Hobbits,")["input_ids"]
    assert ids[0] == tok.bos_token_id

    # Control: raw Hub tokenizer still omits BOS without the fix.
    raw = AutoTokenizer.from_pretrained("unsloth/gemma-4-E2B", trust_remote_code = True)
    raw_ids = raw("This book is largely concerned with Hobbits,")["input_ids"]
    assert raw_ids[0] != raw.bos_token_id


def test_chat_template_bos_is_preserved_when_tokenizer_auto_adds():
    tok = _gemma4_base(
        add_bos_token = True,
        chat_template = "{{ bos_token }}{% for m in messages %}{{ m }}{% endfor %}",
    )
    tu._fix_gemma4_base_bos_token(tok)
    assert tok.add_bos_token is True
    assert tok.chat_template.startswith("{{ bos_token }}")


@pytest.mark.parametrize("prefix", ["", " \n"])
def test_real_tokenizer_chat_bos_survives_save_reload(tmp_path, prefix):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        models.WordLevel({"[UNK]": 0, "[PAD]": 1, "<bos>": 2, "Hello": 3}, unk_token = "[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single = "<bos> $A", special_tokens = [("<bos>", 2)]
    )
    tok = PreTrainedTokenizerFast(tokenizer_object = backend, bos_token = "<bos>", unk_token = "[UNK]")
    tok.chat_template = prefix + "{{ bos_token }}Hello"
    config = types.SimpleNamespace(model_type = "gemma4")
    tu._fix_gemma4_base_bos_token(tok, config = config)
    tok.save_pretrained(tmp_path)
    tok = PreTrainedTokenizerFast.from_pretrained(tmp_path)
    tu._fix_gemma4_base_bos_token(tok, config = config)
    messages = [{"role": "user", "content": "Hello"}]
    encoded = tok.apply_chat_template(messages, tokenize = True)
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    assert ids == [2, 3]
    rendered = tok.apply_chat_template(messages, tokenize = False)
    assert tok(rendered, add_special_tokens = False)["input_ids"] == [2, 3]
    assert tok("Hello")["input_ids"] == [2, 3]


def test_export_helper_strips_dict_chat_template_without_crash():
    tok = _gemma4_base(
        add_bos_token = True,
        chat_template = {
            "default": "{{ bos_token }}{% for m in messages %}{{ m }}{% endfor %}",
            "tool_use": "{% for m in messages %}{{ m }}{% endfor %}",
        },
    )
    tu._dedupe_bos_chat_template(tok)
    assert "{{ bos_token }}" not in tok.chat_template["default"]
    assert tok.chat_template["tool_use"].startswith("{% for m in messages %}")


def test_instruct_template_is_not_stripped_when_tokenizer_does_not_add_bos():
    tok = _gemma4_base(
        add_bos_token = False,
        chat_template = "{{- bos_token -}}{{ messages }}",
        eos_token = "<turn|>",
    )
    tu._fix_gemma4_base_bos_token(tok)
    assert tok.add_bos_token is False
    assert "bos_token" in tok.chat_template


# Real-backend tests. The fakes above accept any attribute, so they cannot tell a working repair
# from an inert one; these build a real tokenizers backend in memory, no network needed.


def _build_fast_tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast

    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"<bos>": 0, "<eos>": 1, "hello": 2, "world": 3}, unk_token = None
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object = backend, bos_token = "<bos>", eos_token = "<eos>")


def _backend_honors_add_bos_token():
    # Pre-5.x fast tokenizers store add_bos_token without changing what they emit. Gemma 4 needs
    # transformers >= 5.5.0 anyway (loader.py SUPPORTS_GEMMA4), so record it rather than fail.
    try:
        tokenizer = _build_fast_tokenizer()
    except Exception:
        return False
    tokenizer.add_bos_token = True
    return tokenizer("hello world")["input_ids"][0] == tokenizer.bos_token_id


requires_working_add_bos_token = pytest.mark.skipif(
    not _backend_honors_add_bos_token(),
    reason = "this transformers treats add_bos_token as an inert attribute",
)


def _real_tokenizer(add_bos = False):
    tokenizer = _build_fast_tokenizer()
    # Gemma 4 is identified by its processor, not by this toy vocabulary.
    tokenizer.processor_class = "Gemma4Processor"
    if add_bos:
        tokenizer.add_bos_token = True
    return tokenizer


def _ids(
    tokenizer,
    text = "hello world",
    **kwargs,
):
    return tokenizer(text, **kwargs)["input_ids"]


@requires_working_add_bos_token
def test_real_backend_gains_exactly_one_bos():
    tok = _real_tokenizer()
    assert _ids(tok)[0] != tok.bos_token_id
    tu._fix_gemma4_base_bos_token(tok)
    ids = _ids(tok)
    assert ids[0] == tok.bos_token_id and ids[1] != tok.bos_token_id


@requires_working_add_bos_token
def test_real_backend_repair_is_idempotent():
    tok = _real_tokenizer()
    tu._fix_gemma4_base_bos_token(tok)
    once = _ids(tok)
    tu._fix_gemma4_base_bos_token(tok)
    assert _ids(tok) == once


@requires_working_add_bos_token
def test_real_backend_add_special_tokens_false_never_gains_bos():
    tok = _real_tokenizer()
    tu._fix_gemma4_base_bos_token(tok)
    assert tok.bos_token_id not in _ids(tok, add_special_tokens = False)


@requires_working_add_bos_token
def test_real_backend_already_correct_tokenizer_is_left_alone():
    # google base mirrors report add_bos_token = False and still prepend, so a repair keyed on
    # the attribute would rebuild a post_processor that already works.
    tok = _real_tokenizer(add_bos = True)
    before = str(tok._tokenizer.post_processor)
    ids_before = _ids(tok)
    tu._fix_gemma4_base_bos_token(tok)
    assert _ids(tok) == ids_before
    assert str(tok._tokenizer.post_processor) == before


@requires_working_add_bos_token
def test_real_backend_without_bos_token_does_not_claim_success():
    tok = _real_tokenizer()
    tok.bos_token = None
    tu._fix_gemma4_base_bos_token(tok)
    assert not getattr(tok, "add_bos_token", False) or _ids(tok)[0] == tok.bos_token_id


@pytest.mark.parametrize(
    "model_type, expected",
    [
        ("gemma4", True),
        ("gemma4_text", True),
        ("gemma-4", True),
        ("gemma3", False),
        ("gemma3n", False),
        ("gemma3n_text", False),
        ("gemma2", False),
        ("llama", False),
        # A future Gemma 4.5 is a different model with its own BOS policy.
        ("gemma_45", False),
        ("gemma-4.5", False),
        # Substring matching would catch unsloth's own diffusion_gemma4.
        ("diffusion_gemma4", False),
    ],
)
def test_config_model_type_detection_is_anchored(model_type, expected):
    config = types.SimpleNamespace(model_type = model_type, text_config = None)
    assert tu._is_gemma4_config(config) is expected


@pytest.mark.parametrize(
    "architecture, expected",
    [
        ("Gemma4ForConditionalGeneration", True),
        ("DiffusionGemma4ForConditionalGeneration", False),
        ("Gemma3ForConditionalGeneration", False),
    ],
)
def test_config_architectures_detection_is_anchored(architecture, expected):
    config = types.SimpleNamespace(
        model_type = "unknown", text_config = None, architectures = [architecture]
    )
    assert tu._is_gemma4_config(config) is expected


@requires_working_add_bos_token
def test_bos_token_inside_a_jinja_comment_does_not_suppress_the_fix():
    tok = _real_tokenizer()
    tok.chat_template = "{# bos_token is handled elsewhere #}{{ messages }}"
    tu._fix_gemma4_base_bos_token(tok)
    assert _ids(tok)[0] == tok.bos_token_id


@requires_working_add_bos_token
def test_bos_token_emitted_by_the_template_still_suppresses_the_fix():
    tok = _real_tokenizer()
    tok.chat_template = "{{- bos_token -}}{{ messages }}"
    tu._fix_gemma4_base_bos_token(tok)
    assert _ids(tok)[0] != tok.bos_token_id
