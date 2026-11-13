# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
from datasets import Dataset
from tokenizers import Tokenizer, models
from transformers import CLIPImageProcessor, LlavaProcessor, PreTrainedTokenizerFast

from unsloth.chat_templates import apply_chat_template, get_chat_template
from unsloth.ollama_template_mappers import OLLAMA_TEMPLATES
from unsloth.save import create_ollama_modelfile

ALPACA = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""

CONVERSATION = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
CONVERSATIONS = Dataset.from_dict({"conversations": [CONVERSATION]})


def _tokenizer(name):
    vocab = {"<|begin_of_text|>": 0, "<|end_of_text|>": 1, "[UNK]": 2}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token = "[UNK]")),
        bos_token = "<|begin_of_text|>",
        eos_token = "<|end_of_text|>",
        unk_token = "[UNK]",
    )
    tokenizer.name_or_path = name
    return tokenizer


def test_base_model_gets_the_trained_template():
    tokenizer = _tokenizer("unsloth/llama-3-8b-bnb-4bit")
    apply_chat_template(CONVERSATIONS, tokenizer = tokenizer, chat_template = ALPACA)

    modelfile = create_ollama_modelfile(tokenizer, "unsloth/llama-3-8b", "x.gguf")

    assert modelfile is not None
    assert modelfile.startswith("FROM x.gguf\n")
    assert "### Instruction:\n{{ .Prompt }}" in modelfile
    assert "### Response:\n{{ .Response }}<|end_of_text|>" in modelfile
    assert "__FILE_LOCATION__" not in modelfile


def test_instruct_model_trained_on_a_custom_template_keeps_it():
    tokenizer = _tokenizer("unsloth/llama-3-8b-Instruct")
    apply_chat_template(CONVERSATIONS, tokenizer = tokenizer, chat_template = ALPACA)

    modelfile = create_ollama_modelfile(tokenizer, "unsloth/llama-3-8b-Instruct", "x.gguf")

    assert "### Instruction:" in modelfile
    assert "<|start_header_id|>" not in modelfile


def test_stock_template_picked_after_training_leaves_the_mapped_template():
    tokenizer = _tokenizer("unsloth/llama-3-8b-Instruct")
    apply_chat_template(CONVERSATIONS, tokenizer = tokenizer, chat_template = ALPACA)
    tokenizer = get_chat_template(tokenizer, chat_template = "alpaca")

    modelfile = create_ollama_modelfile(tokenizer, "unsloth/llama-3-8b-Instruct", "x.gguf")

    assert "<|start_header_id|>" in modelfile
    assert "### Instruction:" not in modelfile


def test_instruct_model_without_a_custom_template_uses_its_own():
    tokenizer = _tokenizer("unsloth/llama-3-8b-Instruct")

    modelfile = create_ollama_modelfile(tokenizer, "unsloth/llama-3-8b-Instruct", "x.gguf")

    assert "FROM x.gguf\n" in modelfile
    assert "<|start_header_id|>" in modelfile
    assert getattr(tokenizer, "_ollama_modelfile", None) is None


def test_unmapped_model_without_a_template_has_no_modelfile():
    assert create_ollama_modelfile(_tokenizer("some/model"), "some/model", "x.gguf") is None


def test_processor_template_survives_the_gguf_unwrap():
    processor = LlavaProcessor(
        image_processor = CLIPImageProcessor(), tokenizer = _tokenizer("some/vlm"), patch_size = 14
    )
    processor = get_chat_template(processor, chat_template = "alpaca")

    modelfile = create_ollama_modelfile(processor.tokenizer, "some/vlm", "x.gguf")

    assert modelfile is not None
    assert "### Instruction:" in modelfile


def test_system_message_passed_to_get_chat_template_reaches_the_modelfile():
    tokenizer = get_chat_template(
        _tokenizer("some/model"), chat_template = "alpaca", system_message = "You are a pirate."
    )

    modelfile = create_ollama_modelfile(tokenizer, "some/model", "x.gguf")

    assert modelfile.endswith('SYSTEM """You are a pirate."""')
    assert modelfile.count("SYSTEM ") == 1
    assert "Below are some instructions" not in modelfile


def test_system_message_replaces_the_default_of_a_mapped_model():
    name = "unsloth/Qwen2.5-0.5B-Instruct"
    tokenizer = get_chat_template(
        _tokenizer(name), chat_template = "qwen-2.5", system_message = "Be brief."
    )

    modelfile = create_ollama_modelfile(tokenizer, name, "x.gguf")

    assert modelfile.endswith('SYSTEM """Be brief."""')
    assert "You are Qwen" not in modelfile
    assert "<|im_start|>system" in modelfile


def test_qwen_tool_prompt_uses_the_same_system_message_as_the_modelfile():
    name = "unsloth/Qwen2.5-0.5B-Instruct"
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    tokenizer = get_chat_template(
        _tokenizer(name), chat_template = "qwen-2.5", system_message = "Be brief."
    )
    stock = get_chat_template(_tokenizer(name), chat_template = "qwen-2.5")

    prompt = tokenizer.apply_chat_template(CONVERSATION, tools = tools, tokenize = False)

    assert prompt.startswith("<|im_start|>system\nBe brief.\n\n# Tools")
    assert create_ollama_modelfile(tokenizer, name, "x.gguf").endswith('SYSTEM """Be brief."""')
    assert stock.apply_chat_template(CONVERSATION, tools = tools, tokenize = False).startswith(
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
    )


def test_system_message_is_added_when_the_modelfile_has_none():
    tokenizer = get_chat_template(
        _tokenizer("some/model"), chat_template = "llama-3.1", system_message = "Be brief."
    )

    modelfile = create_ollama_modelfile(tokenizer, "some/model", "x.gguf")

    assert modelfile.endswith('PARAMETER min_p 0.1\nSYSTEM """Be brief."""')
    assert modelfile.count("SYSTEM ") == 1


def test_empty_system_message_drops_the_default_from_the_modelfile():
    tokenizer = get_chat_template(
        _tokenizer("some/model"), chat_template = "unsloth", system_message = ""
    )

    modelfile = create_ollama_modelfile(tokenizer, "some/model", "x.gguf")

    assert "SYSTEM " not in modelfile
    assert ">>> Assistant: {{ .Response }}" in modelfile


def test_default_system_message_leaves_the_stock_modelfile():
    tokenizer = get_chat_template(_tokenizer("some/model"), chat_template = "alpaca")

    modelfile = create_ollama_modelfile(tokenizer, "some/model", "x.gguf")

    assert tokenizer._ollama_modelfile == OLLAMA_TEMPLATES["alpaca"]
    assert tokenizer._ollama_system_message is None
    assert modelfile.endswith(
        'SYSTEM """Below are some instructions that describe some tasks. '
        'Write responses that appropriately complete each request."""'
    )


def test_system_message_without_a_template_default_leaves_the_modelfile():
    # chatml has no default, so get_chat_template only warns and the Jinja template ignores it.
    tokenizer = get_chat_template(
        _tokenizer("some/model"), chat_template = "chatml", system_message = "Be brief."
    )
    stock = get_chat_template(_tokenizer("some/model"), chat_template = "chatml")

    modelfile = create_ollama_modelfile(tokenizer, "some/model", "x.gguf")

    assert modelfile == create_ollama_modelfile(stock, "some/model", "x.gguf")


def test_system_message_keeps_the_mapped_template_choice():
    # qwen-2.5 and qwen-2.5-coder train the same; the mapper picks the coder Modelfile, which adds FIM.
    name = "unsloth/Qwen2.5-Coder-7B-Instruct"
    tokenizer = get_chat_template(
        _tokenizer(name), chat_template = "qwen-2.5", system_message = "Be brief."
    )

    modelfile = create_ollama_modelfile(tokenizer, name, "x.gguf")

    assert "<|fim_prefix|>" in modelfile
    assert modelfile.endswith('SYSTEM """Be brief."""')
    assert modelfile.count("SYSTEM ") == 1


def test_system_message_reaches_the_modelfile_of_a_processor():
    processor = LlavaProcessor(
        image_processor = CLIPImageProcessor(), tokenizer = _tokenizer("some/vlm"), patch_size = 14
    )
    processor = get_chat_template(processor, chat_template = "alpaca", system_message = "Be brief.")

    modelfile = create_ollama_modelfile(processor.tokenizer, "some/vlm", "x.gguf")

    assert modelfile.endswith('SYSTEM """Be brief."""')


def test_later_get_chat_template_without_system_message_restores_the_default():
    tokenizer = get_chat_template(
        _tokenizer("some/model"), chat_template = "alpaca", system_message = "Be brief."
    )
    tokenizer = get_chat_template(tokenizer, chat_template = "alpaca")

    modelfile = create_ollama_modelfile(tokenizer, "some/model", "x.gguf")

    assert "Be brief." not in modelfile
    assert 'SYSTEM """Below are some instructions' in modelfile


def test_system_message_is_written_verbatim():
    message = 'Say "hi" and use {braces}, {__EOS_TOKEN__}, \\1 and """ freely.'
    tokenizer = get_chat_template(
        _tokenizer("some/model"), chat_template = "alpaca", system_message = message
    )

    modelfile = create_ollama_modelfile(tokenizer, "some/model", "x.gguf")

    assert modelfile.endswith(f'SYSTEM """{message}"""')


def test_system_message_that_ollama_would_cut_short_is_rejected():
    tokenizer = get_chat_template(
        _tokenizer("some/model"), chat_template = "alpaca", system_message = 'Use """ \nfor code.'
    )

    with pytest.raises(ValueError, match = "Ollama Modelfile"):
        create_ollama_modelfile(tokenizer, "some/model", "x.gguf")
