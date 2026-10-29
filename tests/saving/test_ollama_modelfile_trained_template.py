# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from datasets import Dataset
from tokenizers import Tokenizer, models
from transformers import CLIPImageProcessor, LlavaProcessor, PreTrainedTokenizerFast

from unsloth.chat_templates import apply_chat_template, get_chat_template
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
