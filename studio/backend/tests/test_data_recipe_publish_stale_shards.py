# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from huggingface_hub import CommitInfo, CommitOperationAdd, CommitOperationDelete, HfApi
import huggingface_hub.hf_api as hf_api_module
import pytest

from core.data_recipe import huggingface as recipe_hf


class _Hub(HfApi):
    def __init__(
        self,
        files,
        *,
        fail_folder = None,
    ):
        super().__init__(token = "hf_publish")
        self.files = dict(files)
        self.fail_folder = fail_folder
        self.commits = []
        self.client_calls = []

    def list_repo_files(self, *args, **kwargs):
        return list(self.files)

    def create_commit(self, repo_id, operations, **kwargs):
        if any(op.path_in_repo.startswith(f"{self.fail_folder}/") for op in operations):
            raise RuntimeError("upload failed")
        deleted = {op.path_in_repo for op in operations if isinstance(op, CommitOperationDelete)}
        assert not deleted & {
            op.path_in_repo for op in operations if isinstance(op, CommitOperationAdd)
        }
        self.commits.append(operations)
        for op in operations:
            if isinstance(op, CommitOperationDelete):
                del self.files[op.path_in_repo]
            elif isinstance(op, CommitOperationAdd):
                self.files[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
        return CommitInfo(
            commit_url = f"https://huggingface.co/datasets/{repo_id}/commit/0",
            commit_message = "",
            commit_description = "",
            oid = "0",
        )


def _publish(monkeypatch, tmp_path, hub, local_files):
    # Single-commit path; huggingface_hub 1.x with Xet may split the upload into several commits.
    monkeypatch.setattr(hf_api_module, "is_xet_available", lambda: False, raising = False)

    class _Client:
        def __init__(self, token):
            self._api = hub

        def __getattr__(self, name):
            hub.client_calls.append(name)
            return lambda **kwargs: None

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
    monkeypatch.setattr(recipe_hf, "_resolve_recipe_artifact_path", lambda _: tmp_path)
    (tmp_path / "metadata.json").write_text("{}", encoding = "utf-8")
    for rel, data in local_files.items():
        (tmp_path / rel).parent.mkdir(parents = True, exist_ok = True)
        (tmp_path / rel).write_bytes(data)

    recipe_hf.publish_recipe_dataset(
        artifact_path = str(tmp_path),
        repo_id = "org/dataset",
        description = "d",
        hf_token = "hf_publish",
    )


_UNRELATED = {
    ".gitattributes": b"attrs",
    "README.md": b"card",
    "data/train-00000-of-00001.parquet": b"other-tool",
    "data/test/batch_00000.parquet": b"other-split",
    "data/batch_archive/batch_00000.parquet": b"archived",
    "data/batch_00001.parquet.bak": b"backup",
    "misc/batch_00003.parquet": b"other-config",
}


def test_republishing_fewer_batches_removes_only_old_batches(monkeypatch, tmp_path):
    hub = _Hub(
        {
            **_UNRELATED,
            "data/batch_00000.parquet": b"old-0",
            "data/batch_00001.parquet": b"old-1",
            "data/batch_00002.parquet": b"old-2",
            "summary/batch_00000.parquet": b"old-s0",
            "summary/batch_00001.parquet": b"old-s1",
        }
    )
    _publish(
        monkeypatch,
        tmp_path,
        hub,
        {
            "parquet-files/batch_00000.parquet": b"new-0",
            "processors-files/summary/batch_00000.parquet": b"new-s0",
        },
    )

    assert hub.files == {
        **_UNRELATED,
        "data/batch_00000.parquet": b"new-0",
        "summary/batch_00000.parquet": b"new-s0",
    }
    assert len(hub.commits) == 2


def test_first_publish_to_an_empty_repo_uploads_the_batches(monkeypatch, tmp_path):
    hub = _Hub({".gitattributes": b"attrs"})
    _publish(monkeypatch, tmp_path, hub, {"parquet-files/batch_00000.parquet": b"new-0"})

    assert hub.files == {".gitattributes": b"attrs", "data/batch_00000.parquet": b"new-0"}


@pytest.mark.parametrize(
    "fail_folder, message", [("data", "parquet files"), ("summary", "processor outputs")]
)
def test_failed_upload_is_reported_as_a_publish_error(monkeypatch, tmp_path, fail_folder, message):
    hub = _Hub({"data/batch_00000.parquet": b"old-0"}, fail_folder = fail_folder)
    with pytest.raises(recipe_hf.RecipeDatasetPublishError, match = message):
        _publish(
            monkeypatch,
            tmp_path,
            hub,
            {
                "parquet-files/batch_00000.parquet": b"new-0",
                "processors-files/summary/batch_00000.parquet": b"new-s0",
            },
        )


@pytest.mark.parametrize("processor", ["data", "Data", "images", "README.md"])
def test_an_old_run_with_a_processor_on_a_main_dataset_path_is_not_published(
    monkeypatch, tmp_path, processor
):
    hub = _Hub({"data/batch_00000.parquet": b"published-main"})
    with pytest.raises(recipe_hf.RecipeDatasetPublishError, match = "main dataset"):
        _publish(
            monkeypatch,
            tmp_path,
            hub,
            {
                "parquet-files/batch_00000.parquet": b"new-0",
                f"processors-files/{processor}/batch_00000.parquet": b"processor-0",
            },
        )

    assert hub.files == {"data/batch_00000.parquet": b"published-main"}
    assert hub.commits == []
    assert "_create_or_get_repo" not in hub.client_calls
