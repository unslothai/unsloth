# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An export must not declare an MTP head its weights do not contain.

Qwen3.5 ships its multi-token prediction head as top-level `mtp.*` tensors and
declares it with `mtp_num_hidden_layers` (inside `text_config` on the
multimodal configs). transformers has no MTP module for the architecture and
lists `^mtp.*` in `_keys_to_ignore_on_load_unexpected`, so the head is gone as
soon as the model loads and no merge or re-save can restore it. Measured on
Qwen/Qwen3.5-0.8B, a `save_pretrained_merged` of a full finetune writes 473
tensors, none of them `mtp.*`, beside a config that still declares
`mtp_num_hidden_layers = 1`. Consumers that trust the config then look for
weights that are not in the file: llama.cpp's converter asserts on the missing
layer, which is why `convert_to_gguf` already reconciles the same key, and vLLM
resolves its MTP draft config from it (unsloth#7681).
"""

import ast
import json
import types
from pathlib import Path

import pytest

SAVE_PY = Path(__file__).resolve().parents[1] / "unsloth" / "save.py"


@pytest.fixture(scope = "module")
def tree():
    return ast.parse(SAVE_PY.read_text(encoding = "utf-8"))


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in save.py")


def _calls(node):
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


# ---- every merged writer has to reconcile the declaration ----------------


def test_generic_save_reconciles_the_exported_folder(tree):
    """`unsloth_generic_save` is the writer a full finetune goes through, and
    the one the reported repro used."""
    assert "reconcile_mtp_config" in _calls(_func(tree, "unsloth_generic_save"))


def test_generic_save_guards_the_push_branch_too(tree):
    """A push has no local folder to repair afterwards, so that branch fixes
    the config for the duration of the write instead."""
    assert "_mtp_config_matching_tensors" in _calls(_func(tree, "unsloth_generic_save"))


def test_unsloth_save_model_strips_the_declaration(tree):
    """The text-model writer already swaps in a scrubbed config dict for the
    save, so the declaration is dropped there rather than in a second pass."""
    assert "_strip_absent_mtp_declaration" in _calls(_func(tree, "unsloth_save_model"))


def test_the_push_write_is_inside_the_guard(tree):
    """Structural, not just "the name appears": the `push_to_hub` call has to be
    lexically inside the `with` block, else the guard exits before the write."""
    func = _func(tree, "unsloth_generic_save")
    guarded = False
    for node in ast.walk(func):
        if not isinstance(node, ast.With):
            continue
        names = {
            item.context_expr.func.id
            for item in node.items
            if isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
        }
        if "_mtp_config_matching_tensors" in names and "push_to_hub" in _calls(node):
            guarded = True
    assert guarded, "model.push_to_hub is not inside _mtp_config_matching_tensors"


# ---- the dict-level stripper ---------------------------------------------


@pytest.fixture(scope = "module")
def save_module():
    # The behavioural half needs the reconciliation primitives, which live in
    # unsloth_zoo. The two packages are released and upgraded separately, so
    # skip rather than fail when an older zoo is installed; the structural AST
    # tests above still run, and they are the ones that pin this repo's wiring.
    pytest.importorskip(
        "unsloth_zoo.saving_utils",
        reason = "unsloth_zoo.saving_utils is unavailable",
    )
    from unsloth_zoo import saving_utils
    for name in ("MTP_CONFIG_KEY", "mtp_head_is_present"):
        if not hasattr(saving_utils, name):
            pytest.skip(f"installed unsloth_zoo has no {name}")
    # `unsloth` first, then the submodule, and skip rather than error when the
    # stack will not import on this runner. Two ways that happens, neither of
    # them anything to do with what is under test: a platform with no triton
    # wheel, and a cold import of `unsloth.save` that walks into
    # unsloth.models.vision and back into a half initialised unsloth.save.
    # `importorskip` only skips on ModuleNotFoundError, so the circular case
    # needs catching by hand, and one skip with a reason beats fourteen setup
    # errors that say nothing about this file.
    pytest.importorskip("unsloth", reason = "unsloth is not importable on this runner")
    try:
        import unsloth.save as module
    except ImportError as error:
        pytest.skip(f"unsloth.save is not importable on this runner: {error}")
    return module


BODY = ("model.language_model.embed_tokens.weight", "model.visual.blocks.0.attn.qkv.weight")
WITH_MTP = BODY + ("mtp.fc.weight",)


def test_stripper_removes_a_nested_declaration(save_module):
    config = {"model_type": "qwen3_5", "text_config": {"num_hidden_layers": 24, "mtp_num_hidden_layers": 1}}
    assert save_module._strip_absent_mtp_declaration(config, BODY) is True
    assert "mtp_num_hidden_layers" not in config["text_config"]
    assert config["text_config"]["num_hidden_layers"] == 24


def test_stripper_removes_a_top_level_declaration(save_module):
    config = {"model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1}
    assert save_module._strip_absent_mtp_declaration(config, BODY) is True
    assert "mtp_num_hidden_layers" not in config


def test_stripper_keeps_a_declaration_the_weights_back(save_module):
    config = {"text_config": {"mtp_num_hidden_layers": 1}}
    assert save_module._strip_absent_mtp_declaration(config, WITH_MTP) is False
    assert config["text_config"]["mtp_num_hidden_layers"] == 1


def test_stripper_keeps_a_head_stored_as_extra_layers(save_module):
    """The DeepSeek-V3 / GLM spelling keeps the head in extra `layers.N` blocks
    past `num_hidden_layers` rather than under `mtp.`. Both writers in this PR
    now decide "has a head" with the zoo's `mtp_head_is_present`, so neither can
    strip a declaration that spelling backs."""
    config = {"num_hidden_layers": 24, "mtp_num_hidden_layers": 1}
    names = BODY + ("model.layers.24.mlp.up_proj.weight",)
    assert save_module._strip_absent_mtp_declaration(config, names) is False
    assert config["mtp_num_hidden_layers"] == 1


def test_stripper_strips_when_every_layer_is_within_the_count(save_module):
    """The other side of the same rule: a body layer is not a head."""
    config = {"num_hidden_layers": 24, "mtp_num_hidden_layers": 1}
    names = BODY + ("model.layers.23.mlp.up_proj.weight",)
    assert save_module._strip_absent_mtp_declaration(config, names) is True
    assert "mtp_num_hidden_layers" not in config


def test_stripper_finds_the_layer_count_outside_the_declaring_container(save_module):
    """A multimodal config can declare the key at the top level while keeping
    `num_hidden_layers` in `text_config`. Reading the count only out of the
    declaring container returns None there, which silently disables the check."""
    config = {
        "mtp_num_hidden_layers": 1,
        "text_config": {"num_hidden_layers": 24},
    }
    names = BODY + ("model.language_model.layers.24.mlp.up_proj.weight",)
    assert save_module._strip_absent_mtp_declaration(config, names) is False
    assert config["mtp_num_hidden_layers"] == 1


def test_stripper_is_a_noop_without_a_declaration(save_module):
    config = {"model_type": "llama", "num_hidden_layers": 16}
    assert save_module._strip_absent_mtp_declaration(config, BODY) is False
    assert config == {"model_type": "llama", "num_hidden_layers": 16}


def test_stripper_never_raises(save_module):
    assert save_module._strip_absent_mtp_declaration(None, BODY) is False
    assert save_module._strip_absent_mtp_declaration({"mtp_num_hidden_layers": 1}, None) is False


# ---- the around-the-write guard restores the live config -----------------


def _fake_model(declared = 1, nested = True):
    text_config = types.SimpleNamespace(num_hidden_layers = 24)
    if nested and declared is not None:
        text_config.mtp_num_hidden_layers = declared
    config = types.SimpleNamespace(text_config = text_config)
    if not nested and declared is not None:
        config.mtp_num_hidden_layers = declared
    return types.SimpleNamespace(config = config)


@pytest.mark.parametrize("nested", [True, False])
def test_guard_hides_then_restores_the_declaration(save_module, nested):
    model = _fake_model(nested = nested)
    holder = model.config.text_config if nested else model.config
    with save_module._mtp_config_matching_tensors(model, BODY):
        assert not hasattr(holder, "mtp_num_hidden_layers")
    assert holder.mtp_num_hidden_layers == 1


def test_guard_leaves_a_backed_declaration_in_place(save_module):
    model = _fake_model()
    with save_module._mtp_config_matching_tensors(model, WITH_MTP):
        assert model.config.text_config.mtp_num_hidden_layers == 1
    assert model.config.text_config.mtp_num_hidden_layers == 1


def test_guard_restores_even_when_the_write_raises(save_module):
    """A failed save must not leave the caller's model config mutated."""
    model = _fake_model()
    with pytest.raises(RuntimeError):
        with save_module._mtp_config_matching_tensors(model, BODY):
            raise RuntimeError("write failed")
    assert model.config.text_config.mtp_num_hidden_layers == 1


def test_guard_does_nothing_when_the_tensor_names_are_unknown(save_module):
    """`None` means "we do not know what is being written", which must never
    license editing the config."""
    model = _fake_model()
    with save_module._mtp_config_matching_tensors(model, None):
        assert model.config.text_config.mtp_num_hidden_layers == 1
    assert model.config.text_config.mtp_num_hidden_layers == 1


def test_guard_never_raises_on_a_model_without_a_config(save_module):
    with save_module._mtp_config_matching_tensors(types.SimpleNamespace(), BODY):
        pass
