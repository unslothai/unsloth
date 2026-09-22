# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An export must not declare an MTP head its weights do not contain: transformers drops
`^mtp.*` on load for Qwen3.5, and llama.cpp's converter then asserts on the missing layer
the config still promises. unsloth#7681."""

import ast
import json
import os
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


def test_unsloth_save_model_strips_the_declaration(tree):
    assert "_strip_absent_mtp_declaration" in _calls(_func(tree, "unsloth_save_model"))


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


@pytest.fixture(scope = "module")
def save_module_any():
    """`unsloth.save` itself, without the MTP-aware zoo the stripper tests need.

    What this file's other behavioural fixture skips on is the installed zoo exporting the MTP
    helpers. The cost question below is about this repo's own control flow, so it is answerable
    on any zoo and should not be skipped with them.
    """
    pytest.importorskip("unsloth", reason = "unsloth is not importable on this runner")
    try:
        import unsloth.save as module
    except ImportError as error:
        pytest.skip(f"unsloth.save is not importable on this runner: {error}")
    return module


def test_a_local_save_does_not_collect_the_resident_state_dict(
    save_module_any, monkeypatch, tmp_path
):
    """A local save reconciles the folder it just wrote, so reading the resident tensors for it
    bought nothing and cost a second full collection on top of save_pretrained's own. On an
    offloaded or sharded model that materialises every weight, and on a distributed one it is
    a collective the other ranks are not making, so it can stall rather than merely be slow.
    """
    import torch

    collected = []

    class _Model:
        config = None

        def state_dict(self):
            collected.append("state_dict")
            # A real tensor, so the 16bit branch below can cast it as it always does.
            return {"model.embed_tokens.weight": torch.zeros(1)}

        def save_pretrained(self, directory, **kwargs):
            os.makedirs(directory, exist_ok = True)

    from unsloth_zoo import saving_utils

    monkeypatch.setattr(saving_utils, "reconcile_mtp_config", lambda *_a, **_k: None, raising = False)
    for method in ("lora", "merged_4bit_forced"):
        collected.clear()
        save_module_any.unsloth_generic_save(
            _Model(),
            None,
            save_directory = str(tmp_path / method),
            save_method = method,
            push_to_hub = False,
        )
        assert collected == [], (method, collected)

    # A 16bit save still builds one, because that state dict is what gets WRITTEN.
    collected.clear()
    save_module_any.unsloth_generic_save(
        _Model(),
        None,
        save_directory = str(tmp_path / "16bit"),
        save_method = "merged_16bit",
        push_to_hub = False,
    )
    assert collected == ["state_dict"], collected


def test_the_written_tensor_names_reach_the_reconciler(save_module_any, monkeypatch, tmp_path):
    """Reading the names back off disk is not always possible, so hand over the ones already held.

    `_checkpoint_tensor_names` declines to unpickle an unindexed `pytorch_model.bin` just to list
    names, so a `safe_serialization = False` export reconciles against "unknown" and keeps an
    `mtp_num_hidden_layers` the weights do not carry. Whenever a state dict was built it IS the
    thing being written, so passing its keys costs nothing and removes the blind spot.
    """
    import torch

    seen = []

    class _Model:
        config = None

        def state_dict(self):
            return {"model.embed_tokens.weight": torch.zeros(1)}

        def save_pretrained(self, directory, **kwargs):
            os.makedirs(directory, exist_ok = True)

    from unsloth_zoo import saving_utils

    monkeypatch.setattr(
        saving_utils,
        "reconcile_mtp_config",
        lambda directory, tensor_names = None: seen.append(tensor_names),
        raising = False,
    )
    save_module_any.unsloth_generic_save(
        _Model(),
        None,
        save_directory = str(tmp_path / "16bit"),
        save_method = "merged_16bit",
        push_to_hub = False,
    )
    assert seen == [["model.embed_tokens.weight"]], seen

    # No state dict of its own means the names really are unknown here, and unknown must stay
    # unknown rather than licence a guess.
    seen.clear()
    save_module_any.unsloth_generic_save(
        _Model(),
        None,
        save_directory = str(tmp_path / "lora"),
        save_method = "lora",
        push_to_hub = False,
    )
    assert seen == [None], seen
