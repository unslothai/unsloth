# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from types import ModuleType, SimpleNamespace

import pytest

from core.data_recipe import huggingface as recipe_hf


class _Hub:
    def __init__(
        self,
        *,
        private,
        deny_settings = False,
    ):
        self.private = private
        self.deny_settings = deny_settings
        self.calls = []
        self.uploads = []

    def create_repo(self, *, repo_id, repo_type, exist_ok, private):
        self.calls.append("create_repo")

    def update_repo_settings(self, *, repo_id, private, repo_type):
        self.calls.append("update_repo_settings")
        assert repo_type == "dataset"
        if self.deny_settings:
            raise PermissionError("403 Forbidden")
        self.private = private

    def repo_info(self, *, repo_id, repo_type):
        return SimpleNamespace(private = self.private)

    def upload(self, name):
        self.uploads.append((name, self.private))


def _stub_data_designer(monkeypatch, hub):
    class _UploadError(Exception):
        pass

    class _Client:
        def __init__(self, token):
            self._api = hub

        def _validate_repo_id(self, *, repo_id):
            pass

        def _validate_dataset_path(self, *, base_dataset_path):
            pass

        def _create_or_get_repo(self, *, repo_id, private):
            hub.create_repo(repo_id = repo_id, repo_type = "dataset", exist_ok = True, private = private)

        def _upload_main_dataset_files(self, *, repo_id, parquet_folder):
            hub.upload("data")

        def _upload_images_folder(self, *, repo_id, images_folder):
            hub.upload("images")

        def _upload_processor_files(self, *, repo_id, processors_folder):
            hub.upload("processors")

        def _upload_config_files(self, *, repo_id, metadata_path, builder_config_path):
            hub.upload("metadata.json")

    class _Card:
        @classmethod
        def from_metadata(cls, **kwargs):
            return SimpleNamespace(
                text = "", push_to_hub = lambda *args, **kwargs: hub.upload("README.md")
            )

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


def _publish(monkeypatch, tmp_path, hub, *, private):
    _stub_data_designer(monkeypatch, hub)
    monkeypatch.setattr(recipe_hf, "_resolve_recipe_artifact_path", lambda _: tmp_path)
    (tmp_path / "metadata.json").write_text("{}", encoding = "utf-8")
    return recipe_hf.publish_recipe_dataset(
        artifact_path = str(tmp_path),
        repo_id = "org/dataset",
        description = "d",
        hf_token = "hf_publish",
        private = private,
    )


def test_private_publish_to_an_existing_public_repo_makes_it_private_first(monkeypatch, tmp_path):
    hub = _Hub(private = False)
    _publish(monkeypatch, tmp_path, hub, private = True)

    assert hub.private is True
    assert hub.uploads
    assert all(private for _, private in hub.uploads)


def test_private_publish_uploads_nothing_when_the_repo_cannot_be_made_private(
    monkeypatch, tmp_path
):
    hub = _Hub(private = False, deny_settings = True)
    with pytest.raises(recipe_hf.RecipeDatasetPublishError, match = "private"):
        _publish(monkeypatch, tmp_path, hub, private = True)

    assert hub.uploads == []


def test_public_publish_leaves_repo_settings_alone(monkeypatch, tmp_path):
    hub = _Hub(private = True)
    _publish(monkeypatch, tmp_path, hub, private = False)

    assert "update_repo_settings" not in hub.calls
    assert hub.private is True
