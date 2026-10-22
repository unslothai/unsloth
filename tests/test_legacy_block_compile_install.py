# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""FastLanguageModel LoRA installs the per-block compile, with its own kill switch."""

import ast
import pathlib

import pytest

_LLAMA = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "llama.py"


def test_patch_peft_model_installs_block_compile():
    tree = ast.parse(_LLAMA.read_text(encoding = "utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FastLlamaModel")
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "patch_peft_model"
    )
    final = [n for n in fn.body if isinstance(n, ast.Return)][-1]
    before = fn.body[fn.body.index(final) - 1]
    assert isinstance(before, ast.Expr) and ast.unparse(before) == "_legacy_block_compile(model)"


@pytest.fixture
def llama(monkeypatch):
    mod = pytest.importorskip("unsloth.models.llama")
    from unsloth.models import _block_compile

    calls = []
    monkeypatch.setattr(
        _block_compile, "patch_block_compile", lambda model: calls.append(model) or ["X"]
    )
    monkeypatch.delenv("UNSLOTH_BLOCK_COMPILE_LEGACY", raising = False)
    return mod, calls


@pytest.mark.parametrize(
    "env, traceable, installed",
    [
        (None, True, True),
        (None, False, False),
        ("1", False, True),
        ("0", True, False),
    ],
)
def test_legacy_gate(llama, monkeypatch, env, traceable, installed):
    mod, calls = llama
    from unsloth.kernels import fast_lora

    monkeypatch.setattr(fast_lora, "TRACE_LORA_FUNCTIONS", traceable)
    if env is not None:
        monkeypatch.setenv("UNSLOTH_BLOCK_COMPILE_LEGACY", env)
    model = object()
    assert mod._legacy_block_compile(model) == (["X"] if installed else [])
    assert calls == ([model] if installed else [])


def test_packed_calls_keep_the_unsloth_layer_forward(monkeypatch):
    # FastLanguageModel's layer forwards read packed lengths on the host: packed calls stay eager.
    import importlib.util
    import sys

    import torch

    path = _LLAMA.parent / "_block_compile.py"
    spec = importlib.util.spec_from_file_location("_ublock_legacy", path)
    M = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = M
    spec.loader.exec_module(M)
    calls = []

    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            calls.append(1)
            return fn(*args, **kw)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    # Other test modules set UNSLOTH_COMPILE_DISABLE=1 at import, which xdist workers collect
    # before this runs; any of these keeps every call on the stock forward.
    for name in ("UNSLOTH_COMPILE_DISABLE", "UNSLOTH_BLOCK_COMPILE", "UNSLOTH_FORCE_FLOAT32"):
        monkeypatch.delenv(name, raising = False)

    def forward(
        self,
        hidden_states,
        packed_seq_lengths = None,
        **kwargs,
    ):
        return self.proj(hidden_states)

    forward.__module__ = "unsloth.models.llama"

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.proj = torch.nn.Linear(8, 8)

    Legacy = type(
        "LegacyToyDecoderLayer", (torch.nn.Module,), {"__init__": __init__, "forward": forward}
    )
    M._wrap(Legacy, {})
    layer = Legacy()
    x = torch.randn(1, 4, 8, requires_grad = True)
    layer(x, packed_seq_lengths = torch.tensor([2, 2])).sum().backward()
    assert calls == []
    layer(x).sum().backward()
    assert calls
