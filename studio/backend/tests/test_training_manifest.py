# SPDX-License-Identifier: AGPL-3.0-only

import json
from pathlib import Path

import pytest

from core.training.manifest import (
    MANIFEST_FILENAME,
    ManifestError,
    atomic_write_manifest,
    build_manifest,
    migrate_manifest,
    parse_manifest,
)


def _config(**overrides):
    value = {
        "model_name": "unsloth/example-model",
        "device_backend": "cuda",
        "training_type": "LoRA/QLoRA",
        "use_lora": True,
        "hf_dataset": "org/data",
        "train_split": "train",
        "max_seq_length": 2048,
        "random_seed": 3407,
        "learning_rate": "2e-4",
    }
    value.update(overrides)
    return value


def test_manifest_round_trip(tmp_path):
    original = build_manifest(_config(), expected_checkpoint_step = 12)
    path = atomic_write_manifest(tmp_path, original)
    assert path.name == MANIFEST_FILENAME
    assert parse_manifest(path) == original


def test_manifest_recursively_redacts_secrets(tmp_path):
    manifest = build_manifest(
        _config(
            hf_token = "hf-secret",
            wandb_token = "wandb-secret",
            subject = "private-user",
            custom_format_mapping = {"nested": {"password": "secret"}},
            s3_config = {"access_key": "key", "secret_key": "secret"},
        )
    )
    text = json.dumps(manifest)
    for secret in ("hf-secret", "wandb-secret", "private-user", "secret", "key"):
        assert secret not in text


def test_manifest_preserves_checkpoint_upload_preferences():
    manifest = build_manifest(
        _config(push_to_hub = True, hub_model_id = "org/training-checkpoints")
    )

    assert manifest["training_arguments"]["push_to_hub"] is True
    assert manifest["training_arguments"]["hub_model_id"] == "org/training-checkpoints"


def test_atomic_write_interruption_preserves_previous_file(monkeypatch, tmp_path):
    first = build_manifest(_config(), expected_checkpoint_step = 1)
    path = atomic_write_manifest(tmp_path, first)

    def interrupted(source, destination):
        raise OSError("simulated interruption")

    monkeypatch.setattr("core.training.manifest.os.replace", interrupted)
    with pytest.raises(OSError, match = "interruption"):
        atomic_write_manifest(tmp_path, build_manifest(_config(), expected_checkpoint_step = 2))
    assert json.loads(path.read_text()) == first
    assert not list(tmp_path.glob("*.tmp"))


def test_schema_v1_migration_warns():
    with pytest.warns(UserWarning, match = "migrated"):
        migrated = migrate_manifest({"schema_version": 1, "model": "org/model"})
    assert migrated["schema_version"] == 2
    assert migrated["base_model"] == "org/model"


def test_unknown_schema_is_rejected():
    with pytest.raises(ManifestError, match = "Unsupported"):
        migrate_manifest({"schema_version": 999})
