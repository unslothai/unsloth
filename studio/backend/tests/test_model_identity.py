# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import ast
from pathlib import Path
from types import SimpleNamespace

from utils.models.model_identity import restore_hf_cache_repo_identity


_SNAPSHOT = (
    "/home/user/.cache/huggingface/hub/"
    "models--unsloth--Llama-3.2-1B-Instruct/snapshots/0123456789abcdef"
)
_TRAINER = Path(__file__).resolve().parent.parent / "core" / "training" / "trainer.py"


def test_training_loader_restores_selected_repo_identity_for_pinned_snapshot():
    tree = ast.parse(_TRAINER.read_text(encoding = "utf-8"))
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UnslothTrainer"
    )
    load_model = next(
        node
        for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_model"
    )
    restore_call = next(
        node
        for node in ast.walk(load_model)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "restore_hf_cache_repo_identity"
    )

    assert [ast.unparse(argument) for argument in restore_call.args] == [
        "self.model",
        "lookup_name",
    ]
    expected = next(
        keyword.value for keyword in restore_call.keywords if keyword.arg == "expected_repo_id"
    )
    assert isinstance(expected, ast.Name)
    assert expected.id == "model_name"


def test_pinned_training_load_restores_standard_model_identity():
    config = SimpleNamespace(_name_or_path = _SNAPSHOT, model_type = "llama")
    model = SimpleNamespace(config = config)

    restored = restore_hf_cache_repo_identity(
        model,
        _SNAPSHOT,
        expected_repo_id = "unsloth/Llama-3.2-1B-Instruct",
    )

    assert restored == "unsloth/Llama-3.2-1B-Instruct"
    assert vars(config) == {
        "_name_or_path": "unsloth/Llama-3.2-1B-Instruct",
        "model_type": "llama",
    }


def test_legacy_adapter_identity_is_repaired_only_in_memory():
    model_config = SimpleNamespace(_name_or_path = "/outputs/run/checkpoint-100")
    adapter_config = SimpleNamespace(
        base_model_name_or_path = _SNAPSHOT,
        r = 16,
    )
    model = SimpleNamespace(
        config = model_config,
        peft_config = {"default": adapter_config},
    )

    restored = restore_hf_cache_repo_identity(model, _SNAPSHOT)

    assert restored == "unsloth/Llama-3.2-1B-Instruct"
    assert vars(model_config) == {"_name_or_path": "/outputs/run/checkpoint-100"}
    assert vars(adapter_config) == {
        "base_model_name_or_path": "unsloth/Llama-3.2-1B-Instruct",
        "r": 16,
    }


def test_repo_mismatch_leaves_pinned_training_metadata_unchanged():
    config = SimpleNamespace(_name_or_path = _SNAPSHOT)
    model = SimpleNamespace(config = config)

    restored = restore_hf_cache_repo_identity(
        model,
        _SNAPSHOT,
        expected_repo_id = "another/model",
    )

    assert restored is None
    assert config._name_or_path == _SNAPSHOT


def test_ordinary_local_model_and_existing_hub_id_are_unchanged():
    local_config = SimpleNamespace(_name_or_path = "/models/private-model")
    local_model = SimpleNamespace(config = local_config)
    hub_config = SimpleNamespace(_name_or_path = "unsloth/Llama-3.2-1B-Instruct")
    hub_model = SimpleNamespace(config = hub_config)

    assert restore_hf_cache_repo_identity(local_model, "/models/private-model") is None
    assert restore_hf_cache_repo_identity(hub_model, "unsloth/Llama-3.2-1B-Instruct") is None
    assert local_config._name_or_path == "/models/private-model"
    assert hub_config._name_or_path == "unsloth/Llama-3.2-1B-Instruct"


def test_incomplete_cache_layout_is_not_treated_as_a_snapshot():
    incomplete = "/models--unsloth--Llama-3.2-1B-Instruct/snapshots"
    config = SimpleNamespace(_name_or_path = incomplete)

    assert restore_hf_cache_repo_identity(SimpleNamespace(config = config), incomplete) is None
    assert config._name_or_path == incomplete


def test_windows_cache_snapshot_is_supported_but_regular_local_path_is_unchanged():
    snapshot = (
        r"C:\Users\user\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct"
        r"\snapshots\0123456789abcdef"
    )
    snapshot_config = SimpleNamespace(_name_or_path = snapshot)
    local_config = SimpleNamespace(_name_or_path = r"C:\models\private-model")

    assert (
        restore_hf_cache_repo_identity(
            SimpleNamespace(config = snapshot_config),
            snapshot,
            expected_repo_id = "unsloth/Llama-3.2-1B-Instruct",
        )
        == "unsloth/Llama-3.2-1B-Instruct"
    )
    assert snapshot_config._name_or_path == "unsloth/Llama-3.2-1B-Instruct"
    assert (
        restore_hf_cache_repo_identity(
            SimpleNamespace(config = local_config),
            r"C:\models\private-model",
        )
        is None
    )
    assert local_config._name_or_path == r"C:\models\private-model"
