# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""gpt-oss developer block must match openai-harmony, not the HF chat_template.jinja.

The HF template (openai/gpt-oss-120b discussion #85) appends "\\n\\n" after the
instructions even without tools; harmony and vLLM do not.
"""

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from unsloth.chat_templates import get_chat_template


@pytest.fixture(autouse = True)
def _scratch_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather.",
            "parameters": SCHEMA,
        },
    }
]
TOOLS_BLOCK = (
    "# Tools\n\n## functions\n\nnamespace functions {\n\n// Get the weather.\n"
    "type get_weather = (_: {\ncity: string,\n}) => any;\n\n} // namespace functions"
)
# Rendered by openai-harmony 0.0.8 (HarmonyEncodingName.HARMONY_GPT_OSS).
EXPECTED = {
    ("Be helpful.", False): "<|start|>developer<|message|># Instructions\n\nBe helpful.<|end|>",
    ("Be helpful.", True): "<|start|>developer<|message|># Instructions\n\nBe helpful.\n\n"
    + TOOLS_BLOCK
    + "<|end|>",
    ("", True): "<|start|>developer<|message|>" + TOOLS_BLOCK + "<|end|>",
}


def _render(name, instructions, tools, role):
    vocab = {"<unk>": 0, "<|startoftext|>": 1, "<|return|>": 2, "<pad>": 3}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>")),
        bos_token = "<|startoftext|>",
        eos_token = "<|return|>",
        unk_token = "<unk>",
        pad_token = "<pad>",
    )
    tokenizer = get_chat_template(tokenizer, name)
    messages = [{"role": role, "content": instructions}] if instructions else []
    messages.append({"role": "user", "content": "Hi"})
    kwargs = {"tools": TOOLS} if tools else {}
    return tokenizer.apply_chat_template(
        messages, tokenize = False, add_generation_prompt = True, **kwargs
    )


def _developer_block(text):
    start = text.index("<|start|>developer<|message|>")
    return text[start : text.index("<|end|>", start) + len("<|end|>")]


CASES = [
    pytest.param(key, role, id = f"{case}-{role}")
    for key, case in zip(EXPECTED, ["instr", "instr_tools", "tools"])
    for role in (["system", "developer"] if key[0] else ["system"])
]


@pytest.mark.parametrize("name", ["gpt-oss", "gptoss"])
@pytest.mark.parametrize("key, role", CASES)
def test_developer_block_matches_harmony(name, key, role):
    instructions, tools = key
    out = _render(name, instructions, tools, role)
    assert _developer_block(out) == EXPECTED[key]


@pytest.mark.parametrize("key", list(EXPECTED), ids = ["instr", "instr_tools", "tools"])
def test_expected_strings_match_live_harmony(key):
    harmony = pytest.importorskip("openai_harmony")
    instructions, tools = key
    dev = harmony.DeveloperContent.new()
    if instructions:
        dev = dev.with_instructions(instructions)
    if tools:
        dev = dev.with_function_tools(
            [harmony.ToolDescription.new("get_weather", "Get the weather.", parameters = SCHEMA)]
        )
    enc = harmony.load_harmony_encoding(harmony.HarmonyEncodingName.HARMONY_GPT_OSS)
    convo = harmony.Conversation.from_messages(
        [
            harmony.Message.from_role_and_content(harmony.Role.DEVELOPER, dev),
            harmony.Message.from_role_and_content(harmony.Role.USER, "Hi"),
        ]
    )
    assert _developer_block(enc.decode(enc.render_conversation(convo))) == EXPECTED[key]
