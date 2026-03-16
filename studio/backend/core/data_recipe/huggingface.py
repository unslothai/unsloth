# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
from pathlib import Path

from utils.paths import recipe_datasets_root, resolve_dataset_path

_DATA_DESIGNER_FOOTER = (
    '<sub style="white-space: nowrap;">Made with ❤️ using 🎨 '
    '<a href="https://github.com/NVIDIA-NeMo/DataDesigner">NeMo Data Designer</a></sub>'
)
_UNSLOTH_STUDIO_FOOTER = (
    '<sub style="white-space: nowrap;">Made with ❤️ using 🦥 ' "Unsloth Studio</sub>"
)


class RecipeDatasetPublishError(ValueError):
    """Raised when a recipe dataset cannot be published to Hugging Face."""


def _resolve_recipe_artifact_path(artifact_path: str) -> Path:
    root = recipe_datasets_root().expanduser().resolve()
    candidate = resolve_dataset_path(artifact_path).expanduser()
    resolved = candidate.resolve(strict = False)

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RecipeDatasetPublishError(
            "This execution artifact is outside the Recipe Studio dataset storage."
        ) from exc

    if not resolved.exists():
        raise RecipeDatasetPublishError("Execution artifacts are no longer available.")
    if not resolved.is_dir():
        raise RecipeDatasetPublishError(
            "Execution artifact path is not a dataset folder."
        )

    return resolved


def publish_recipe_dataset(
    *,
    artifact_path: str,
    repo_id: str,
    description: str,
    hf_token: str | None = None,
    private: bool = False,
) -> str:
    dataset_path = _resolve_recipe_artifact_path(artifact_path)

    try:
        from data_designer.engine.storage.artifact_storage import (
            FINAL_DATASET_FOLDER_NAME,
            METADATA_FILENAME,
            PROCESSORS_OUTPUTS_FOLDER_NAME,
            SDG_CONFIG_FILENAME,
        )
        from data_designer.integrations.huggingface.client import (
            HuggingFaceHubClient,
            HuggingFaceHubClientUploadError,
        )
        from data_designer.integrations.huggingface.dataset_card import (
            DataDesignerDatasetCard,
        )
    except ImportError as exc:
        raise RecipeDatasetPublishError(
            "NeMo Data Designer Hugging Face integration is not installed."
        ) from exc

    try:
        client = HuggingFaceHubClient(token = hf_token)
        client._validate_repo_id(repo_id = repo_id)
        client._validate_dataset_path(base_dataset_path = dataset_path)
        client._create_or_get_repo(repo_id = repo_id, private = private)

        metadata_path = dataset_path / METADATA_FILENAME
        builder_config_path = dataset_path / SDG_CONFIG_FILENAME

        with metadata_path.open(encoding = "utf-8") as fh:
            metadata = json.load(fh)

        builder_config = None
        if builder_config_path.exists():
            with builder_config_path.open(encoding = "utf-8") as fh:
                builder_config = json.load(fh)

        card = DataDesignerDatasetCard.from_metadata(
            metadata = metadata,
            builder_config = builder_config,
            repo_id = repo_id,
            description = description,
            tags = None,
        )
        card.text = card.text.replace(_DATA_DESIGNER_FOOTER, _UNSLOTH_STUDIO_FOOTER)
        # Data Designer currently drops the explicit token when pushing the
        # dataset card. Push it ourselves so auth stays request-local.
        card.push_to_hub(repo_id, token = hf_token, repo_type = "dataset")

        client._upload_main_dataset_files(
            repo_id = repo_id,
            parquet_folder = dataset_path / FINAL_DATASET_FOLDER_NAME,
        )
        client._upload_images_folder(
            repo_id = repo_id,
            images_folder = dataset_path / "images",
        )
        client._upload_processor_files(
            repo_id = repo_id,
            processors_folder = dataset_path / PROCESSORS_OUTPUTS_FOLDER_NAME,
        )
        client._upload_config_files(
            repo_id = repo_id,
            metadata_path = metadata_path,
            builder_config_path = builder_config_path,
        )

        return f"https://huggingface.co/datasets/{repo_id}"
    except HuggingFaceHubClientUploadError as exc:
        raise RecipeDatasetPublishError(str(exc)) from exc
