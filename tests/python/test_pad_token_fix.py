"""_fix_pad_token dispatch in unsloth/tokenizer_utils.py.

It must delegate to unsloth_zoo's shared fix_pad_token when present (single
source of truth), and fall back to a no-op against an older unsloth_zoo. Static
+ CPU-only: _fix_pad_token is exec'd in isolation so the test never imports
torch / transformers / unsloth.
"""

import ast
import os
import sys
import types

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOK_PATH = os.path.join(REPO_ROOT, "unsloth", "tokenizer_utils.py")

WANTED = {
    "_fix_pad_token",
}


def _load_pad_helpers():
    """Exec only the pad-token helpers with a stub logger (no heavy imports)."""
    tree = ast.parse(open(TOK_PATH).read())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & WANTED:
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in WANTED:
            nodes.append(node)
    module = ast.Module(body = nodes, type_ignores = [])
    ast.fix_missing_locations(module)
    ns = {"logger": types.SimpleNamespace(warning = lambda *a, **k: None)}
    exec(compile(module, TOK_PATH, "exec"), ns)
    return ns


class FakeTok:
    def __init__(self, vocab, pad, eos):
        self._v = dict(vocab)
        self.pad_token = pad
        self.eos_token = eos

    @property
    def pad_token_id(self):
        return self._v.get(self.pad_token)

    @property
    def eos_token_id(self):
        return self._v.get(self.eos_token)

    def get_vocab(self):
        return dict(self._v)


def _block_shared_module(monkeypatch):
    # Stub parent so importing it is cheap, and mark the submodule absent.
    monkeypatch.setitem(sys.modules, "unsloth_zoo", types.ModuleType("unsloth_zoo"))
    monkeypatch.setitem(sys.modules, "unsloth_zoo.pad_token", None)


def test_fix_pad_token_none_is_noop():
    ns = _load_pad_helpers()
    assert ns["_fix_pad_token"](None) is None


def test_fallback_keeps_pad_named_token(monkeypatch):
    ns = _load_pad_helpers()
    _block_shared_module(monkeypatch)
    # A pad-named token (e.g. <|vision_pad|>) is a valid pad -> fallback keeps it.
    tok = FakeTok(
        {"<|endoftext|>": 1, "<|im_end|>": 2, "<|vision_pad|>": 3},
        pad = "<|vision_pad|>",
        eos = "<|im_end|>",
    )
    ns["_fix_pad_token"](tok)
    assert tok.pad_token == "<|vision_pad|>"
    assert tok.pad_token != tok.eos_token


def test_fallback_leaves_valid_pad_untouched(monkeypatch):
    ns = _load_pad_helpers()
    _block_shared_module(monkeypatch)
    tok = FakeTok({"<s>": 1, "</s>": 2, "<pad>": 0}, pad = "<pad>", eos = "</s>")
    ns["_fix_pad_token"](tok)
    assert tok.pad_token == "<pad>"


def test_fix_pad_token_delegates_to_shared_module(monkeypatch):
    ns = _load_pad_helpers()
    calls = {}

    fake = types.ModuleType("unsloth_zoo.pad_token")

    def fix_pad_token(tokenizer, allow_add = True):
        calls["allow_add"] = allow_add
        tokenizer.pad_token = "SHARED"

    fake.fix_pad_token = fix_pad_token
    monkeypatch.setitem(sys.modules, "unsloth_zoo", types.ModuleType("unsloth_zoo"))
    monkeypatch.setitem(sys.modules, "unsloth_zoo.pad_token", fake)

    tok = FakeTok({"a": 1}, pad = "x", eos = "y")
    ns["_fix_pad_token"](tok)
    # Delegated, and crucially with allow_add=False (no model here to resize).
    assert tok.pad_token == "SHARED"
    assert calls["allow_add"] is False
