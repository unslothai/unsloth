# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL's tool loop appends `_get_tool_suffix_ids` after the sampled tool call. Its dummy prefix is trimmed at the
last EOS, which for a tokenizer whose EOS is the end-of-turn token (unsloth/gemma-4-*: <turn|>) but whose tool-call
turn ends with another stop token (<|tool_response>) is the end of the USER turn, so the suffix carries a second,
empty-argument dummy tool call. These tests run the TRL function body through Unsloth's source edit."""

import ast
import os
import re
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_REPLACEMENTS = os.path.join(ROOT, "unsloth", "models", "rl_replacements.py")

# TRL main / 1.14 - 1.18 `GRPOTrainer._get_tool_suffix_ids` (text-only branch; the VLM branch is unchanged).
TRL_SOURCE = '''
def _get_tool_suffix_ids(self, tool_messages):
    """Get token IDs for tool result formatting by using a minimal dummy conversation."""
    dummy_tool_calls = [{"type": "function", "function": {"name": tool_messages[0]["name"], "arguments": {}}}]
    dummy_messages = [
        {"role": "user", "content": "dummy"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": dummy_tool_calls,
        },
    ]
    prefix_ids = self.processing_class.apply_chat_template(
        dummy_messages,
        add_generation_prompt=False,
        tokenize=True,
        chat_template=self.chat_template,
        return_dict=False,
        **self.chat_template_kwargs,
    )
    full_ids = self.processing_class.apply_chat_template(
        dummy_messages + tool_messages,
        add_generation_prompt=True,
        tokenize=True,
        chat_template=self.chat_template,
        return_dict=False,
        **self.chat_template_kwargs,
    )
    eos_positions = [i for i, tok_id in enumerate(prefix_ids) if tok_id == self._tokenizer.eos_token_id]
    if eos_positions:
        prefix_ids = prefix_ids[: eos_positions[-1] + 1]

    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("Unexpected tokenization: the EOS-trimmed prefix IDs are not a prefix of the full IDs.")
    return full_ids[len(prefix_ids) :]
'''


def _load_edit():
    """Load trainer__get_tool_suffix_ids and its regex from rl_replacements.py without importing unsloth."""
    src = open(RL_REPLACEMENTS, encoding = "utf-8").read()
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_TOOL_SUFFIX_EOS_TRIM" for t in node.targets
        ):
            keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "trainer__get_tool_suffix_ids":
            keep.append(node)
    assert (
        len(keep) == 2
    ), "trainer__get_tool_suffix_ids / _TOOL_SUFFIX_EOS_TRIM missing from rl_replacements.py"
    ns = {"re": re}
    exec(compile(ast.Module(body = keep, type_ignores = []), RL_REPLACEMENTS, "exec"), ns)
    return ns["trainer__get_tool_suffix_ids"]


class _ToyTokenizer:
    """One token per piece. Two families: "chatml" (turn ends <|im_end|>\\n) and "gemma4" (turn ends <turn|>\\n,
    a tool call ends <|tool_response> with no <turn|>, the tool result continues the same model turn)."""

    def __init__(self, family, eos):
        self.family = family
        self.vocab = {}
        self.eos_token_id = self._id(eos)

    def _id(self, piece):
        return self.vocab.setdefault(piece, len(self.vocab))

    def _pieces(self, messages, add_generation_prompt):
        out = []
        for i, m in enumerate(messages):
            calls = m.get("tool_calls") or []
            if self.family == "chatml":
                if m["role"] == "tool":
                    if i == 0 or messages[i - 1]["role"] != "tool":
                        out += ["<|im_start|>", "user", "\n"]
                    out += ["<tool_response>", m["content"], "</tool_response>"]
                    if i + 1 == len(messages) or messages[i + 1]["role"] != "tool":
                        out += ["<|im_end|>", "\n"]
                    continue
                out += ["<|im_start|>", m["role"], "\n", m["content"]]
                for c in calls:
                    out += [
                        "<tool_call>",
                        c["function"]["name"],
                        repr(c["function"]["arguments"]),
                        "</tool_call>",
                    ]
                out += ["<|im_end|>", "\n"]
            else:
                if m["role"] == "tool":
                    out += ["response:", m["name"], m["content"], "<tool_response|>"]
                    continue
                out += [
                    "<|turn>",
                    "model" if m["role"] == "assistant" else m["role"],
                    "\n",
                    m["content"],
                ]
                if calls:
                    for c in calls:
                        out += [
                            "<|tool_call>",
                            c["function"]["name"],
                            repr(c["function"]["arguments"]),
                            "<tool_call|>",
                        ]
                    out += ["<|tool_response>"]
                else:
                    out += ["<turn|>", "\n"]
        if add_generation_prompt:
            out += (
                ["<|im_start|>", "assistant", "\n"]
                if self.family == "chatml"
                else ["<|turn>", "model", "\n"]
            )
        return out

    def apply_chat_template(
        self,
        messages,
        add_generation_prompt = False,
        tokenize = True,
        **kwargs,
    ):
        return [self._id(p) for p in self._pieces(messages, add_generation_prompt)]

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv[i] for i in ids)


class _Trainer:
    def __init__(self, tok):
        self.processing_class = tok
        self._tokenizer = tok
        self.chat_template = None
        self.chat_template_kwargs = {}


def _bind(source):
    ns = {}
    exec(textwrap.dedent(source), ns)
    return ns["_get_tool_suffix_ids"]


def _reference_suffix(tok, stop_ids):
    """What the template itself renders after the sampled tool call (the first stop token of the assistant turn)."""
    prompt = [{"role": "user", "content": "weather in Paris?"}]
    call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
            }
        ],
    }
    tool = [{"role": "tool", "name": "get_weather", "content": "18 C"}]
    head = tok.apply_chat_template(prompt, add_generation_prompt = True)
    full = tok.apply_chat_template(prompt + [call] + tool, add_generation_prompt = True)
    stop = next(j for j in range(len(head), len(full)) if full[j] in stop_ids)
    return tool, full[stop + 1 :]


CASES = [
    # family, eos piece, generation stop pieces (generation_config.eos_token_id)
    ("chatml", "<|im_end|>", ("<|im_end|>",)),
    ("gemma4", "<eos>", ("<eos>", "<turn|>", "<|tool_response>")),  # google/gemma-4-*
    ("gemma4", "<turn|>", ("<eos>", "<turn|>", "<|tool_response>")),  # unsloth/gemma-4-*
]


@pytest.mark.parametrize("family,eos,stops", CASES)
def test_edited_suffix_matches_template(family, eos, stops):
    edit = _load_edit()
    edited = edit("_get_tool_suffix_ids", TRL_SOURCE)
    assert edited != TRL_SOURCE, "the EOS-trim edit no longer matches TRL's source"
    tok = _ToyTokenizer(family, eos)
    stop_ids = {tok._id(s) for s in stops}
    tool, want = _reference_suffix(tok, stop_ids)
    got = _bind(edited)(_Trainer(tok), tool)
    assert tok.decode(got) == tok.decode(want)


def test_unedited_trl_duplicates_the_dummy_call_for_turn_eos():
    """The flip: TRL's own trim on unsloth/gemma-4's EOS puts a dummy tool call into the suffix."""
    tok = _ToyTokenizer("gemma4", "<turn|>")
    stop_ids = {tok._id(s) for s in ("<eos>", "<turn|>", "<|tool_response>")}
    tool, want = _reference_suffix(tok, stop_ids)
    got = _bind(TRL_SOURCE)(_Trainer(tok), tool)
    assert got != want
    assert "<|tool_call>get_weather{}" in tok.decode(got)


def test_other_functions_untouched():
    edit = _load_edit()
    assert edit("_tool_call_loop", TRL_SOURCE) == TRL_SOURCE


def test_registered_for_grpo_and_distillation():
    src = open(RL_REPLACEMENTS, encoding = "utf-8").read()
    assert 'RL_FUNCTIONS["grpo_trainer"].append(trainer__get_tool_suffix_ids)' in src
    assert 'RL_FUNCTIONS["distillation_trainer"].append(trainer__get_tool_suffix_ids)' in src


@pytest.mark.parametrize("filename", ["grpo_trainer.py", "distillation_trainer.py"])
def test_edit_matches_installed_trl(filename):
    """Where the installed TRL has the tool loop, the edit must still find TRL's trim (read from source, no import)."""
    import importlib.util

    spec = importlib.util.find_spec("trl")
    if spec is None or spec.origin is None:
        pytest.skip("trl not installed")
    path = os.path.join(os.path.dirname(spec.origin), "trainer", filename)
    if not os.path.exists(path):
        pytest.skip(f"installed TRL has no trainer/{filename}")
    src = open(path, encoding = "utf-8").read()
    m = re.search(r"\n    def _get_tool_suffix_ids\(.*?(?=\n    def )", src, flags = re.DOTALL)
    if m is None or "eos_positions" not in m.group(0):
        pytest.skip(f"installed TRL {filename} has no EOS-trimmed _get_tool_suffix_ids")
    assert _load_edit()("_get_tool_suffix_ids", m.group(0)) != m.group(0)
