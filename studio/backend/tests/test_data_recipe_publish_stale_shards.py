# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
import huggingface_hub.hf_api as hf_api_module

from core.data_recipe import huggingface as recipe_hf


class _Hub(HfApi):
    def __init__(self, files):
        super().__init__(token = "hf_publish")
        self.files = dict(files)

    def repo_exists(self, *args, **kwargs):
        return True

    def create_repo(self, *args, **kwargs):
        pass

    def list_repo_files(self, *args, **kwargs):
        return list(self.files)

    def create_commit(self, repo_id, operations, **kwargs):
        for op in operations:
            if isinstance(op, CommitOperationDelete):
                del self.files[op.path_in_repo]
            elif isinstance(op, CommitOperationAdd):
                self.files[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()


def _stub_data_designer(monkeypatch, hub):
    class _Client:
        def __init__(self, token):
            self._api = hub

        def __getattr__(self, name):
            return lambda **kwargs: None

        def _upload_main_dataset_files(self, *, repo_id, parquet_folder):
            hub.upload_folder(
                repo_id = repo_id,
                folder_path = str(parquet_folder),
                path_in_repo = "data",
                repo_type = "dataset",
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
            "HuggingFaceHubClientUploadError": type("_UploadError", (Exception,), {}),
        },
        "data_designer.integrations.huggingface.dataset_card": {
            "DataDesignerDatasetCard": SimpleNamespace(
                from_metadata = lambda **kwargs: SimpleNamespace(
                    text = "", push_to_hub = lambda *args, **kwargs: None
                )
            ),
        },
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)


def test_republishing_fewer_shards_removes_the_old_ones(monkeypatch, tmp_path):
    monkeypatch.setattr(hf_api_module, "is_xet_available", lambda: False)
    hub = _Hub(
        {
            "README.md": b"card",
            "metadata.json": b"{}",
            "data/batch_00000.parquet": b"old-0",
            "data/batch_00001.parquet": b"old-1",
            "data/batch_00002.parquet": b"old-2",
        }
    )
    _stub_data_designer(monkeypatch, hub)
    monkeypatch.setattr(recipe_hf, "_resolve_recipe_artifact_path", lambda _: tmp_path)
    (tmp_path / "metadata.json").write_text("{}", encoding = "utf-8")
    (tmp_path / "parquet-files").mkdir()
    (tmp_path / "parquet-files" / "batch_00000.parquet").write_bytes(b"new-0")

    recipe_hf.publish_recipe_dataset(
        artifact_path = str(tmp_path),
        repo_id = "org/dataset",
        description = "d",
        hf_token = "hf_publish",
    )

    assert hub.files == {
        "README.md": b"card",
        "metadata.json": b"{}",
        "data/batch_00000.parquet": b"new-0",
    }
