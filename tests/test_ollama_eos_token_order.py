# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The prefix collapse depends on order, so the result must not vary with PYTHONHASHSEED."""

import ast
import json
import os
import subprocess
import sys

_CHAT_TEMPLATES = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "chat_templates.py")

# `<unk>` shares its first three characters with `<unused*>`, which makes order matter.
_TOKENS = ["<pad>", "<eos>", "<unk>", "<unused0>", "<unused1>", "<unused2>"]
_BOS = "<pad>"

_RUNNER = """
import ast, json, sys
source, tokens, bos = json.loads(sys.argv[1]), json.loads(sys.argv[2]), json.loads(sys.argv[3])
namespace = {}
exec(compile(source, "get_ollama_eos_tokens", "exec"), namespace)


class _Tokenizer:
    def __init__(self, tokens, bos_token):
        self._tokens, self.bos_token = tokens, bos_token

    @property
    def added_tokens_decoder(self):
        return {index: token for index, token in enumerate(self._tokens)}


print(json.dumps(sorted(namespace["get_ollama_eos_tokens"](_Tokenizer(tokens, bos), []))))
"""


def _extract_source():
    with open(_CHAT_TEMPLATES, encoding = "utf-8") as f:
        source = f.read()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_ollama_eos_tokens":
            return ast.get_source_segment(source, node)
    raise AssertionError("get_ollama_eos_tokens not found in chat_templates.py")


def _run(
    seed,
    tokens = _TOKENS,
    bos = _BOS,
):
    environment = dict(os.environ, PYTHONHASHSEED = str(seed))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _RUNNER,
            json.dumps(_extract_source()),
            json.dumps(tokens),
            json.dumps(bos),
        ],
        capture_output = True,
        text = True,
        env = environment,
        check = True,
    )
    return json.loads(completed.stdout)


def test_ollama_eos_tokens_do_not_depend_on_hash_seed():
    results = [_run(seed) for seed in range(8)]
    assert len(set(map(tuple, results))) == 1, (
        "get_ollama_eos_tokens returned different stop tokens for the same tokenizer "
        f"under different PYTHONHASHSEEDs: {sorted(set(map(tuple, results)))}"
    )


def test_ollama_eos_tokens_keep_unk_beside_an_unused_family():
    for seed in range(8):
        assert _run(seed) == ["<eos>", "<unk>", "<unused"]


def test_ollama_eos_tokens_keep_short_tokens_whole():
    # Collapsing "<" (a 3-char token's half) rewrote every token and left no usable stop.
    for seed in range(8):
        assert _run(seed, ["<pad>", "<eos>", "<eot>", "<s>"], "<pad>") == [
            "<eos>",
            "<eot>",
            "<s>",
        ]
        assert _run(seed, ["<pad>", "</a>", "</b>", "</c>", "<eos>"], "<pad>") == [
            "</a>",
            "</b>",
            "</c>",
            "<eos>",
        ]


def test_ollama_eos_tokens_keep_unrelated_characters():
    for seed in range(8):
        assert _run(seed, ["<pad>", "eop", "ezz", "abc", "def"], "<pad>") == [
            "abc",
            "def",
            "eop",
            "ezz",
        ]


def test_ollama_eos_tokens_still_collapse_real_families():
    for seed in range(8):
        assert _run(seed, ["<pad>", "<eos>", "<eot>", "<eom>", "eop"], "<pad>") == ["<eo", "eop"]


def test_ollama_eos_tokens_drop_whitespace_only_stops():
    for seed in range(4):
        tokens = ["<pad>", "\t", "\t\t", "\t\t\t", "\n\n", "<end_of_turn>"]
        assert _run(seed, tokens, "<pad>") == ["<end_of_turn>"]


_RUNNER_ADDED = _RUNNER.replace(
    "return {index: token for index, token in enumerate(self._tokens)}",
    "return {index: AddedToken(*token) for index, token in enumerate(self._tokens)}",
).replace(
    "class _Tokenizer:",
    "class AddedToken(str):\n    def __new__(cls, content, special):\n        token = str.__new__(cls, content)\n        token.special = special\n        return token\n\n\nclass _Tokenizer:",
)


def test_ollama_eos_tokens_skip_non_special_html_tags():
    # Phi-3.5's non-special `</s>` is a real end token; Gemma's non-special HTML tags are answer text.
    tokens = [
        ["<pad>", True],
        ["</s>", False],
        ["<b>", False],
        ["</table>", False],
        ["<think>", False],
        ["<end_of_turn>", True],
    ]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _RUNNER_ADDED,
            json.dumps(_extract_source()),
            json.dumps(tokens),
            json.dumps("<pad>"),
        ],
        capture_output = True,
        text = True,
        check = True,
    )
    assert json.loads(completed.stdout) == ["</s>", "<end_of_turn>", "<think>"]
