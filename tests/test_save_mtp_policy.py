# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An export must not declare an MTP head its weights do not contain: transformers drops
`^mtp.*` on load for Qwen3.5, and llama.cpp's converter then asserts on the missing layer
the config still promises. unsloth#7681."""

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


def test_generic_save_reconciles_the_exported_folder(tree):
    assert "reconcile_mtp_config" in _calls(_func(tree, "unsloth_generic_save"))


def test_generic_save_guards_the_push_branch_too(tree):
    """A push has no local folder to repair afterwards."""
    assert "_mtp_config_matching_tensors" in _calls(_func(tree, "unsloth_generic_save"))


def test_unsloth_save_model_strips_the_declaration(tree):
    assert "_strip_absent_mtp_declaration" in _calls(_func(tree, "unsloth_save_model"))


def test_the_push_write_is_inside_the_guard(tree):
    """Structural: outside the `with`, the guard exits before the write."""
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


def test_the_push_guard_reads_the_resident_tensors_when_no_state_dict_was_built(tree):
    """lora and merged_4bit push with `state_dict is None`, which no-ops the guard."""
    func = _func(tree, "unsloth_generic_save")
    derived_from_model = False
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "_mtp_tensor_names" for t in node.targets):
            continue
        for sub in ast.walk(node.value):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "state_dict"
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id == "model"
            ):
                derived_from_model = True
    assert derived_from_model, (
        "_mtp_tensor_names is never derived from model.state_dict(), so the push guard "
        "does nothing for save methods that build no state_dict of their own"
    )


@pytest.fixture(scope = "module")
def save_module():
    # Skip rather than fail on an older zoo; the AST tests above pin this repo's wiring.
    pytest.importorskip(
        "unsloth_zoo.saving_utils",
        reason = "unsloth_zoo.saving_utils is unavailable",
    )
    from unsloth_zoo import saving_utils

    for name in ("MTP_CONFIG_KEY", "mtp_head_is_present"):
        if not hasattr(saving_utils, name):
            pytest.skip(f"installed unsloth_zoo has no {name}")
    # importorskip only skips ModuleNotFoundError, so catch the circular import by hand.
    pytest.importorskip("unsloth", reason = "unsloth is not importable on this runner")
    try:
        import unsloth.save as module
    except ImportError as error:
        pytest.skip(f"unsloth.save is not importable on this runner: {error}")
    return module


BODY = ("model.language_model.embed_tokens.weight", "model.visual.blocks.0.attn.qkv.weight")
WITH_MTP = BODY + ("mtp.fc.weight",)


def test_stripper_removes_a_nested_declaration(save_module):
    config = {
        "model_type": "qwen3_5",
        "text_config": {"num_hidden_layers": 24, "mtp_num_hidden_layers": 1},
    }
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
    """DeepSeek-V3 / GLM keep the head in extra `layers.N` blocks, not under `mtp.`."""
    config = {"num_hidden_layers": 24, "mtp_num_hidden_layers": 1}
    names = BODY + ("model.layers.24.mlp.up_proj.weight",)
    assert save_module._strip_absent_mtp_declaration(config, names) is False
    assert config["mtp_num_hidden_layers"] == 1


def test_stripper_strips_when_every_layer_is_within_the_count(save_module):
    config = {"num_hidden_layers": 24, "mtp_num_hidden_layers": 1}
    names = BODY + ("model.layers.23.mlp.up_proj.weight",)
    assert save_module._strip_absent_mtp_declaration(config, names) is True
    assert "mtp_num_hidden_layers" not in config


def test_stripper_finds_the_layer_count_outside_the_declaring_container(save_module):
    """`num_hidden_layers` can sit in `text_config` while the key is at the top level."""
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
    model = _fake_model()
    with pytest.raises(RuntimeError):
        with save_module._mtp_config_matching_tensors(model, BODY):
            raise RuntimeError("write failed")
    assert model.config.text_config.mtp_num_hidden_layers == 1


def test_guard_does_nothing_when_the_tensor_names_are_unknown(save_module):
    """`None` means unknown, which must never license editing the config."""
    model = _fake_model()
    with save_module._mtp_config_matching_tensors(model, None):
        assert model.config.text_config.mtp_num_hidden_layers == 1
    assert model.config.text_config.mtp_num_hidden_layers == 1


def test_guard_never_raises_on_a_model_without_a_config(save_module):
    with save_module._mtp_config_matching_tensors(types.SimpleNamespace(), BODY):
        pass


@pytest.fixture
def zoo_without_the_helpers(save_module, monkeypatch):
    import sys

    real = sys.modules.get("unsloth_zoo.saving_utils")
    stand_in = types.ModuleType("unsloth_zoo.saving_utils")
    for name in dir(real or types.ModuleType("empty")):
        if name in ("MTP_CONFIG_KEY", "mtp_head_is_present", "reconcile_mtp_config"):
            continue
        if name.startswith("__"):
            continue
        setattr(stand_in, name, getattr(real, name))
    monkeypatch.setitem(sys.modules, "unsloth_zoo.saving_utils", stand_in)
    return stand_in


def _capture_warnings(save_module, monkeypatch):
    said = []
    logger = save_module.logger
    monkeypatch.setattr(logger, "warning_once", lambda message, *a, **kw: said.append(message))
    monkeypatch.setattr(logger, "warning", lambda message, *a, **kw: said.append(message))
    return said


def test_an_older_zoo_leaves_the_config_alone_and_says_nothing(
    save_module, zoo_without_the_helpers, monkeypatch
):
    """The import failure shared a warning with real problems, so every save printed it."""
    said = _capture_warnings(save_module, monkeypatch)
    config = {
        "model_type": "qwen3_5",
        "text_config": {"num_hidden_layers": 24, "mtp_num_hidden_layers": 1},
    }

    assert save_module._strip_absent_mtp_declaration(config, BODY) is False
    assert config["text_config"]["mtp_num_hidden_layers"] == 1
    assert said == [], said


def test_an_older_zoo_does_not_make_the_push_guard_complain(
    save_module, zoo_without_the_helpers, monkeypatch
):
    said = _capture_warnings(save_module, monkeypatch)

    class _Holder:
        pass

    text = _Holder()
    setattr(text, "mtp_num_hidden_layers", 1)
    text.num_hidden_layers = 24
    config = _Holder()
    config.text_config = text
    model = _Holder()
    model.config = config

    with save_module._mtp_config_matching_tensors(model, BODY):
        assert getattr(text, "mtp_num_hidden_layers") == 1
    assert getattr(text, "mtp_num_hidden_layers") == 1
    assert said == [], said


def test_a_real_failure_is_still_reported(save_module, monkeypatch):
    said = _capture_warnings(save_module, monkeypatch)

    class _Exploding(dict):
        def get(self, *args, **kwargs):
            raise RuntimeError("config is not readable")

    assert save_module._strip_absent_mtp_declaration(_Exploding(), BODY) is False
    assert said and "config is not readable" in said[0], said


def test_the_manual_merge_restores_the_config_when_the_write_fails(tree):
    """With the restore on the success path only, a failed upload mutated the live model."""
    func = _func(tree, "unsloth_save_model")
    restoring = False
    for node in ast.walk(func):
        if not isinstance(node, ast.Try) or not node.finalbody:
            continue
        wrote = "save_pretrained" in _calls(node) or "upload_folder" in _calls(node)
        restores = any(
            isinstance(sub, ast.Attribute) and sub.attr == "config"
            for stmt in node.finalbody
            for sub in ast.walk(stmt)
        ) and any(
            isinstance(sub, ast.Name) and sub.id == "old_config"
            for stmt in node.finalbody
            for sub in ast.walk(stmt)
        )
        if wrote and restores:
            restoring = True
    assert restoring, (
        "the scrubbed-config write is not in a try/finally that restores old_config, so a "
        "failed save leaves the caller's model stripped"
    )
