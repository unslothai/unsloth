# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from core.training.account_jobs import account_hf_token
import json
from pathlib import Path
import tempfile

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
    try:
        candidate = resolve_dataset_path(artifact_path).expanduser()
    except ValueError as exc:
        # Outside every dataset root, so it never reaches the check below: a 500, not a refusal.
        raise RecipeDatasetPublishError(
            "This execution artifact is outside the Recipe Studio dataset storage."
        ) from exc
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
        raise RecipeDatasetPublishError("Execution artifact path is not a dataset folder.")

    return resolved


def _drop_token_keys(value: object) -> None:
    if isinstance(value, dict):
        value.pop("token", None)
        for item in value.values():
            _drop_token_keys(item)
    elif isinstance(value, list):
        for item in value:
            _drop_token_keys(item)


def _scrub_seed_source_tokens(builder_config: object) -> None:
    if not isinstance(builder_config, dict):
        return
    for config in (builder_config, builder_config.get("data_designer")):
        if not isinstance(config, dict):
            continue
        seed_config = config.get("seed_config")
        if isinstance(seed_config, dict):
            _drop_token_keys(seed_config.get("source"))


def publish_recipe_dataset(
    *,
    artifact_path: str,
    repo_id: str,
    description: str,
    hf_token: str | None = None,
    private: bool = False,
    link_endpoint: str | None = None,
) -> str:
    hf_token = account_hf_token(hf_token)
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
            _scrub_seed_source_tokens(builder_config)

        card = DataDesignerDatasetCard.from_metadata(
            metadata = metadata,
            builder_config = builder_config,
            repo_id = repo_id,
            description = description,
            tags = None,
        )
        card.text = card.text.replace(_DATA_DESIGNER_FOOTER, _UNSLOTH_STUDIO_FOOTER)
        # Data Designer drops the explicit token, so push the card ourselves to keep auth request-local.
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
        with tempfile.TemporaryDirectory() as scrubbed_dir:
            scrubbed_config_path = Path(scrubbed_dir) / SDG_CONFIG_FILENAME
            if builder_config is not None:
                with scrubbed_config_path.open("w", encoding = "utf-8") as fh:
                    json.dump(builder_config, fh, indent = 2, ensure_ascii = False)
            client._upload_config_files(
                repo_id = repo_id,
                metadata_path = metadata_path,
                builder_config_path = scrubbed_config_path,
            )

        from utils.hf_endpoint import get_hf_endpoint

        # The upload went to get_hf_endpoint(); this URL is for the browser, where
        # a loopback mirror is not the same host. The caller passes what its client
        # can reach.
        return f"{link_endpoint or get_hf_endpoint()}/datasets/{repo_id}"
    except HuggingFaceHubClientUploadError as exc:
        raise RecipeDatasetPublishError(str(exc)) from exc
