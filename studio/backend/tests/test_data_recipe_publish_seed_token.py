# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from core.data_recipe import huggingface as recipe_hf

_SOURCES = [
    {
        "seed_type": "hf",
        "path": "datasets/org/name/data/*.parquet",
        "token": "hf_FAKEFAKEFAKE",
        "endpoint": "https://huggingface.co",
    },
    {"seed_type": "github_repo", "repos": ["org/name"], "token": "ghp_FAKEFAKEFAKE"},
]


def _stub_data_designer(monkeypatch, seen):
    class _UploadError(Exception):
        pass

    class _Client:
        def __init__(self, token):
            pass

        def __getattr__(self, name):
            return lambda **kwargs: None

        def _upload_config_files(self, *, repo_id, metadata_path, builder_config_path):
            # The real client gates each of the two files on .exists() on its own.
            if metadata_path.exists():
                seen["uploaded_metadata"] = metadata_path.name
            if builder_config_path.exists():
                seen["uploaded"] = builder_config_path.read_text(encoding = "utf-8")
                seen["uploaded_name"] = builder_config_path.name

    class _Card:
        @classmethod
        def from_metadata(cls, *, builder_config, **kwargs):
            seen["card"] = json.dumps(builder_config)
            return SimpleNamespace(text = "", push_to_hub = lambda *args, **kwargs: None)

    modules = {
        "data_designer.engine.storage.artifact_storage": {
            "FINAL_DATASET_FOLDER_NAME": "parquet-files",
            "METADATA_FILENAME": "metadata.json",
            "PROCESSORS_OUTPUTS_FOLDER_NAME": "processors-files",
            "SDG_CONFIG_FILENAME": "builder_config.json",
        },
        "data_designer.integrations.huggingface.client": {
            "HuggingFaceHubClient": _Client,
            "HuggingFaceHubClientUploadError": _UploadError,
        },
        "data_designer.integrations.huggingface.dataset_card": {
            "DataDesignerDatasetCard": _Card,
        },
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)


def _publish(monkeypatch, tmp_path, builder_config):
    seen: dict = {}
    _stub_data_designer(monkeypatch, seen)
    monkeypatch.setattr(recipe_hf, "_resolve_recipe_artifact_path", lambda _: tmp_path)
    (tmp_path / "metadata.json").write_text("{}", encoding = "utf-8")
    if builder_config is not None:
        (tmp_path / "builder_config.json").write_text(json.dumps(builder_config), encoding = "utf-8")
    recipe_hf.publish_recipe_dataset(
        artifact_path = str(tmp_path),
        repo_id = "org/dataset",
        description = "d",
        hf_token = "hf_publish",
    )
    return seen


@pytest.mark.parametrize("source", _SOURCES)
def test_publish_keeps_the_seed_token_out_of_the_hub(monkeypatch, tmp_path, source):
    recipe = {
        "columns": [{"name": "c", "column_type": "expression", "token": "kept"}],
        "seed_config": {"source": source, "sampling_strategy": "ordered"},
    }
    builder_config = {"data_designer": recipe, "library_version": "0.5.4"}
    seen = _publish(monkeypatch, tmp_path, builder_config)

    assert seen["uploaded_name"] == "builder_config.json"
    assert "FAKEFAKEFAKE" not in seen["uploaded"]
    assert "FAKEFAKEFAKE" not in seen["card"]

    expected_source = {k: v for k, v in source.items() if k != "token"}
    assert json.loads(seen["uploaded"]) == {
        "data_designer": {
            **recipe,
            "seed_config": {"source": expected_source, "sampling_strategy": "ordered"},
        },
        "library_version": "0.5.4",
    }

    on_disk = json.loads((tmp_path / "builder_config.json").read_text(encoding = "utf-8"))
    assert on_disk == builder_config


def test_publish_without_a_seed_uploads_the_config_unchanged(monkeypatch, tmp_path):
    builder_config = {"data_designer": {"columns": [], "seed_config": None}}
    seen = _publish(monkeypatch, tmp_path, builder_config)
    assert json.loads(seen["uploaded"]) == builder_config


def test_publish_without_a_builder_config_uploads_none(monkeypatch, tmp_path):
    seen = _publish(monkeypatch, tmp_path, None)
    assert seen["uploaded_metadata"] == "metadata.json"
    assert "uploaded" not in seen
    assert seen["card"] == "null"
