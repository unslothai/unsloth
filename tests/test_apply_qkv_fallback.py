# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""apply_qkv/apply_o fall back to module-level originals when the per-instance attribute is absent."""

from __future__ import annotations

import ast
import importlib
import os
import pathlib

import pytest
import torch
import unsloth  # noqa: F401

from real_accelerator import has_real_cuda
from unsloth.models import llama as llama_module

# gemma.py, qwen2.py and the vision/MoE loaders reuse LlamaAttention_fast_forward, so the
# llama row covers them.
ATTENTION_FILES = (
    "llama.py",
    "cohere.py",
    "falcon_h1.py",
    "gemma2.py",
    "granite.py",
    "mistral.py",
    "qwen3.py",
)

SIBLING_FILES = tuple(f for f in ATTENTION_FILES if f != "llama.py")

FALLBACKS = (("apply_qkv", "original_apply_qkv"), ("apply_o", "original_apply_o"))

MODELS_DIR = pathlib.Path(llama_module.__file__).resolve().parent


def _tree(filename: str) -> ast.Module:
    return ast.parse((MODELS_DIR / filename).read_text(encoding = "utf-8"))


def _bare_self_calls(tree: ast.Module, attr: str) -> list[int]:
    """Line numbers of every `self.<attr>(...)` call."""
    hits = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            hits.append(node.lineno)
    return hits


def _fallback_calls(tree: ast.Module, attr: str, fallback: str) -> list[int]:
    """Line numbers of every `getattr(self, "<attr>", <fallback>)(...)` call."""
    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Call)):
            continue
        inner = node.func
        if not (isinstance(inner.func, ast.Name) and inner.func.id == "getattr"):
            continue
        if len(inner.args) != 3:
            continue
        target, name, default = inner.args
        if not (isinstance(target, ast.Name) and target.id == "self"):
            continue
        if not (isinstance(name, ast.Constant) and name.value == attr):
            continue
        if isinstance(default, ast.Name) and default.id == fallback:
            hits.append(node.lineno)
    return hits


@pytest.mark.parametrize("filename", ATTENTION_FILES)
@pytest.mark.parametrize("attr,fallback", FALLBACKS)
def test_no_bare_instance_attribute_call(filename, attr, fallback):
    """A bare self.apply_qkv(...) is the defect; there must be none left."""
    bare = _bare_self_calls(_tree(filename), attr)
    assert bare == [], (
        f"unsloth/models/{filename} calls self.{attr}(...) with no fallback at "
        f"line(s) {bare}; use getattr(self, {attr!r}, {fallback}) so an instance that "
        "never reached the loader's attach step still resolves (unsloth#1713, unsloth#2587)"
    )


@pytest.mark.parametrize("filename", ATTENTION_FILES)
@pytest.mark.parametrize("attr,fallback", FALLBACKS)
def test_call_site_uses_the_module_level_fallback(filename, attr, fallback):
    """And the call site really is the getattr form, not something else that happens to parse."""
    calls = _fallback_calls(_tree(filename), attr, fallback)
    assert (
        calls
    ), f"unsloth/models/{filename} has no getattr(self, {attr!r}, {fallback})(...) call site"


@pytest.mark.parametrize("filename", SIBLING_FILES)
@pytest.mark.parametrize("attr,fallback", FALLBACKS)
def test_sibling_files_import_the_fallback_explicitly(filename, attr, fallback):
    """`from .llama import *` would also deliver these names, but only for as long as
    llama.py has no __all__ and neither name is renamed. The call sites are load bearing,
    so the import is explicit."""
    tree = _tree(filename)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "llama" and node.level == 1
        for alias in node.names
    }
    assert fallback in imported, (
        f"unsloth/models/{filename} must carry an explicit `from .llama import {fallback}`; "
        f"found {sorted(imported)}"
    )


@pytest.mark.parametrize("filename", ATTENTION_FILES)
@pytest.mark.parametrize("attr,fallback", FALLBACKS)
def test_fallback_resolves_to_the_same_function_in_every_module(filename, attr, fallback):
    """The import audit above is static; this one loads each module and checks the object."""
    module = importlib.import_module(f"unsloth.models.{filename[:-3]}")
    assert getattr(module, fallback) is getattr(llama_module, fallback)


def _plain_llama_attention():
    """A transformers LlamaAttention built outside unsloth's loader, as in unsloth#1713."""
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaAttention

    config = LlamaConfig(
        hidden_size = 32,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        intermediate_size = 64,
        num_hidden_layers = 1,
        vocab_size = 64,
    )
    return LlamaAttention(config, layer_idx = 0)


def test_plain_attention_has_neither_attribute():
    """The premise: nothing attaches apply_qkv / apply_o to an instance unsloth did not load."""
    attention = _plain_llama_attention()
    assert not hasattr(attention, "apply_qkv")
    assert not hasattr(attention, "apply_o")


def test_fallback_runs_on_an_instance_that_never_reached_the_loader():
    """The fallback is not just resolvable, it produces the projections the forward wants."""
    attention = _plain_llama_attention()
    hidden_states = torch.randn(1, 3, attention.config.hidden_size)

    apply_qkv = getattr(attention, "apply_qkv", llama_module.original_apply_qkv)
    apply_o = getattr(attention, "apply_o", llama_module.original_apply_o)
    assert apply_qkv is llama_module.original_apply_qkv
    assert apply_o is llama_module.original_apply_o

    Q, K, V = apply_qkv(attention, hidden_states)
    head_dim = attention.head_dim
    config = attention.config
    assert Q.shape == (1, 3, config.num_attention_heads * head_dim)
    assert K.shape == (1, 3, config.num_key_value_heads * head_dim)
    assert V.shape == K.shape

    attn_output = apply_o(attention, torch.randn(1, 3, config.num_attention_heads * head_dim))
    assert attn_output.shape == (1, 3, config.hidden_size)


def test_instance_attribute_still_wins_over_the_fallback():
    """This is the whole safety argument: the fused LoRA kernels are still selected, so
    the getattr default only ever fires where the old code raised."""
    from unsloth.kernels import apply_lora_o, apply_lora_qkv

    attention = _plain_llama_attention()
    attention.apply_qkv = apply_lora_qkv
    attention.apply_o = apply_lora_o

    assert getattr(attention, "apply_qkv", llama_module.original_apply_qkv) is apply_lora_qkv
    assert getattr(attention, "apply_o", llama_module.original_apply_o) is apply_lora_o

    attention.apply_qkv = llama_module.original_apply_qkv
    attention.apply_o = llama_module.original_apply_o
    assert getattr(attention, "apply_qkv", None) is llama_module.original_apply_qkv
    assert getattr(attention, "apply_o", None) is llama_module.original_apply_o


def _plain_model_dtype() -> "torch.dtype":
    """sm_75 cards like the T4 lack bfloat16; UNSLOTH_TEST_PLAIN_DTYPE forces either dtype branch."""
    forced = os.environ.get("UNSLOTH_TEST_PLAIN_DTYPE", "auto")
    if forced == "float16":
        return torch.float16
    if forced == "bfloat16":
        return torch.bfloat16
    try:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    except Exception:
        return torch.float16


# has_real_cuda(), not torch.cuda.is_available(): the zoo CUDA spoof patches the latter True
# process-wide. Enforced by tests/python/test_accelerator_skip_guards.py.
@pytest.mark.gpu
@pytest.mark.skipif(
    not has_real_cuda(),
    reason = "loads a real checkpoint through FastLanguageModel; needs an accelerator",
)
def test_apply_qkv_fallback_end_to_end():
    """unsloth#1713 as reported: load with unsloth, then build a plain transformers model of
    the same architecture in the same process and run it. Needs a real accelerator because
    the patched forward uses triton rope and the attention dispatch."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from unsloth import FastLanguageModel

    model_name = "unsloth/Llama-3.2-1B-Instruct"
    model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length = 512,
        load_in_4bit = True,
    )

    config = AutoConfig.from_pretrained(model_name)
    config.num_hidden_layers = 1

    # Pre-Ampere cards cannot run bf16. Set config dtype too: on transformers 4.57.6 the keyword
    # leaves config.dtype at the checkpoint value, mismatching activations and weights.
    dtype = _plain_model_dtype()
    for attribute in ("torch_dtype", "dtype"):
        try:
            setattr(config, attribute, str(dtype).replace("torch.", ""))
        except Exception:
            pass
    try:
        plain = AutoModelForCausalLM.from_config(config, dtype = dtype)
    except TypeError:
        # transformers 4.x spelling, before `dtype` became the keyword name.
        plain = AutoModelForCausalLM.from_config(config, torch_dtype = dtype)
    plain = plain.to("cuda").eval()
    assert next(plain.parameters()).dtype is dtype
    for layer in plain.model.layers:
        assert not hasattr(layer.self_attn, "apply_qkv")

    with torch.no_grad():
        out = plain(input_ids = torch.randint(0, 128, (1, 8), device = "cuda"))
    assert out.logits.shape == (1, 8, config.vocab_size)

    from unsloth.kernels import apply_lora_o, apply_lora_qkv

    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        random_state = 0,
    )
    first = model.model.model.layers[0].self_attn
    assert first.apply_qkv is apply_lora_qkv
    assert first.apply_o is apply_lora_o
