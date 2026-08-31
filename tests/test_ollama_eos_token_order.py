"""Regression test for get_ollama_eos_tokens depending on set iteration order.

`get_ollama_eos_tokens` collapses a family of added tokens into the prefix they
share, so `<unused0>`, `<unused1>`, ... become one `<unused` stop token in the
Ollama Modelfile. It does that by rewriting `joined_text` as it walks the token
list, which makes the result order-dependent: a family member has to be seen
before any shorter token that merely shares its prefix. The list came from
`set(...) - set(...)`, and set iteration order over strings varies with
PYTHONHASHSEED, so a Gemma-shaped vocabulary exported `["<eos>", "<unk>",
"<unused"]` on some runs and `["<eos>", "<un"]` on others, losing `<unk>` and
handing Ollama a two-character stop token in its place.

chat_templates.py cannot be imported without a GPU, so the function is
ast-extracted the way tests/test_bad_mappings_redirect.py extracts its own.
"""

import ast
import json
import os
import subprocess
import sys

_CHAT_TEMPLATES = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "chat_templates.py")

# A Gemma-shaped vocabulary: `<unk>` sits next to a `<unused*>` family whose members
# share its first three characters, which is what makes the order matter.
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


def _run(seed):
    """Run the function in a fresh interpreter under a fixed PYTHONHASHSEED."""
    environment = dict(os.environ, PYTHONHASHSEED = str(seed))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _RUNNER,
            json.dumps(_extract_source()),
            json.dumps(_TOKENS),
            json.dumps(_BOS),
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
    """The shorter `<unk>` must survive; only the family collapses to its prefix."""
    for seed in range(8):
        assert _run(seed) == ["<eos>", "<unk>", "<unused"]
