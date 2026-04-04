# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

from .jsonable import to_jsonable
from .local_callable_validators import (
    register_oxc_local_callable_validators,
    split_oxc_local_callable_validators,
)

_IMAGE_CONTEXT_PATCHED = False


def _encode_bytes_to_base64(value: bytes | bytearray) -> str:
    return base64.b64encode(bytes(value)).decode("utf-8")


def _load_image_file_to_base64(
    path_value: str, *, base_path: str | None = None
) -> str | None:
    try:
        path = Path(path_value)
        candidates: list[Path] = []
        if path.is_absolute():
            candidates.append(path)
        else:
            if base_path:
                candidates.append(Path(base_path) / path)
            candidates.append(Path.cwd() / path)

        for candidate in candidates:
            if not candidate.exists() or not candidate.is_file():
                continue
            with candidate.open("rb") as f:
                return _encode_bytes_to_base64(f.read())
    except (OSError, TypeError, ValueError):
        return None
    return None


def _pil_image_to_base64(value: Any) -> str | None:
    try:
        from PIL.Image import Image as PILImage  # type: ignore
    except ImportError:
        return None
    if not isinstance(value, PILImage):
        return None
    buffer = io.BytesIO()
    image_format = str(getattr(value, "format", "") or "").upper()
    if image_format not in {"PNG", "JPEG", "JPG", "WEBP", "GIF"}:
        image_format = "PNG"
    value.save(buffer, format = image_format)
    return _encode_bytes_to_base64(buffer.getvalue())


def _normalize_image_context_value(value: Any, *, base_path: str | None = None) -> Any:
    if isinstance(value, str):
        return value

    if isinstance(value, (bytes, bytearray)):
        return _encode_bytes_to_base64(value)

    pil_base64 = _pil_image_to_base64(value)
    if pil_base64 is not None:
        return pil_base64

    if isinstance(value, dict):
        url = value.get("url")
        if isinstance(url, str):
            return url

        image_url = value.get("image_url")
        if isinstance(image_url, str):
            return image_url
        if isinstance(image_url, dict):
            nested_url = image_url.get("url")
            if isinstance(nested_url, str):
                return nested_url

        inline_data = value.get("data")
        if isinstance(inline_data, str):
            return inline_data

        raw_bytes = value.get("bytes")
        if isinstance(raw_bytes, (bytes, bytearray)):
            return _encode_bytes_to_base64(raw_bytes)
        if isinstance(raw_bytes, str) and raw_bytes.strip():
            return raw_bytes

        path_value = value.get("path")
        if isinstance(path_value, str) and path_value.strip():
            if as_base64 := _load_image_file_to_base64(path_value, base_path = base_path):
                return as_base64
            return path_value

    return value


def _apply_data_designer_image_context_patch() -> None:
    global _IMAGE_CONTEXT_PATCHED
    if _IMAGE_CONTEXT_PATCHED:
        return

    try:
        from data_designer.config.models import ImageContext
    except ImportError:
        return

    if getattr(ImageContext, "_unsloth_image_context_patch_applied", False):
        _IMAGE_CONTEXT_PATCHED = True
        return

    original_auto_resolve = ImageContext._auto_resolve_context_value

    def _patched_auto_resolve(
        self: Any, context_value: Any, base_path: str | None
    ) -> Any:
        normalized = _normalize_image_context_value(context_value, base_path = base_path)
        return original_auto_resolve(self, normalized, base_path)

    ImageContext._auto_resolve_context_value = _patched_auto_resolve
    setattr(ImageContext, "_unsloth_image_context_patch_applied", True)
    _IMAGE_CONTEXT_PATCHED = True


def build_model_providers(recipe: dict[str, Any]):
    from data_designer.config.models import ModelProvider

    providers: list[ModelProvider] = []
    for provider in recipe.get("model_providers", []):
        api_key = provider.get("api_key")
        api_key_env = provider.get("api_key_env")
        if not api_key and api_key_env:
            api_key = os.getenv(api_key_env)
        providers.append(
            ModelProvider(
                name = provider["name"],
                endpoint = provider["endpoint"],
                provider_type = provider.get("provider_type", "openai"),
                api_key = api_key,
                extra_headers = provider.get("extra_headers"),
                extra_body = provider.get("extra_body"),
            )
        )

    return providers


def _recipe_has_llm_columns(recipe: dict[str, Any]) -> bool:
    for column in recipe.get("columns", []):
        if not isinstance(column, dict):
            continue
        column_type = column.get("column_type")
        if isinstance(column_type, str) and column_type.startswith("llm-"):
            return True
    return False


def _validate_recipe_runtime_support(
    recipe: dict[str, Any],
    model_providers: list[Any],
) -> None:
    if _recipe_has_llm_columns(recipe) and not model_providers:
        raise ValueError("Add a Provider connection block before running this recipe.")


def build_mcp_providers(
    recipe: dict[str, Any],
) -> list:
    from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider

    providers: list[MCPProvider | LocalStdioMCPProvider] = []
    for provider in recipe.get("mcp_providers", []):
        if not isinstance(provider, dict):
            continue
        provider_type = provider.get("provider_type")
        if provider_type == "stdio":
            env = provider.get("env")
            if not isinstance(env, dict):
                env = {}
            args = provider.get("args")
            if not isinstance(args, list):
                args = []
            providers.append(
                LocalStdioMCPProvider(
                    name = str(provider.get("name", "")),
                    command = str(provider.get("command", "")),
                    args = [str(value) for value in args],
                    env = {str(key): str(value) for key, value in env.items()},
                )
            )
            continue

        if provider_type in {"sse", "streamable_http"}:
            api_key = provider.get("api_key")
            api_key_env = provider.get("api_key_env")
            if not api_key and api_key_env:
                api_key = os.getenv(str(api_key_env))
            providers.append(
                MCPProvider(
                    name = str(provider.get("name", "")),
                    endpoint = str(provider.get("endpoint", "")),
                    provider_type = str(provider_type),
                    api_key = str(api_key) if api_key else None,
                )
            )
    return providers


def build_config_builder(recipe: dict[str, Any]):
    _apply_data_designer_image_context_patch()
    from data_designer.config import DataDesignerConfigBuilder
    from data_designer.config.processors import ProcessorType

    recipe_core = {
        key: value
        for key, value in recipe.items()
        if key not in {"model_providers", "mcp_providers"}
    }
    recipe_core, oxc_local_callable_specs = split_oxc_local_callable_validators(
        recipe_core
    )
    builder = DataDesignerConfigBuilder.from_config({"data_designer": recipe_core})
    register_oxc_local_callable_validators(
        builder = builder,
        specs = oxc_local_callable_specs,
    )

    # DataDesignerConfigBuilder.from_config currently skips processors.
    # Re-attach explicitly so drop_columns/schema_transform survive API payload.
    for processor in recipe_core.get("processors") or []:
        if not isinstance(processor, dict):
            continue
        processor_type_raw = processor.get("processor_type")
        if not isinstance(processor_type_raw, str):
            continue
        kwargs = {k: v for k, v in processor.items() if k != "processor_type"}
        builder.add_processor(
            processor_type = ProcessorType(processor_type_raw),
            **kwargs,
        )

    return builder


def create_data_designer(
    recipe: dict[str, Any],
    *,
    artifact_path: str | None = None,
):
    _apply_data_designer_image_context_patch()
    from data_designer.interface.data_designer import DataDesigner

    model_providers = build_model_providers(recipe)
    _validate_recipe_runtime_support(recipe, model_providers)

    # DataDesigner requires at least one model provider in its registry even
    # when the pipeline contains no LLM columns.  Supply a lightweight stub
    # so sampler/expression-only recipes can run without a real provider.
    if not model_providers:
        from data_designer.config.models import ModelProvider

        model_providers = [
            ModelProvider(
                name = "_unused",
                endpoint = "http://localhost",
                provider_type = "openai",
                api_key = None,
            )
        ]

    return DataDesigner(
        artifact_path = artifact_path,
        model_providers = model_providers,
        mcp_providers = build_mcp_providers(recipe),
    )


def validate_recipe(recipe: dict[str, Any]) -> None:
    builder = build_config_builder(recipe)
    designer = create_data_designer(recipe)
    designer.validate(builder)


def preview_recipe(
    recipe: dict[str, Any],
    num_records: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    builder = build_config_builder(recipe)
    designer = create_data_designer(recipe)
    results = designer.preview(builder, num_records = num_records)

    dataset: list[dict[str, Any]] = []
    if results.dataset is not None:
        raw_rows = results.dataset.to_dict(orient = "records")
        dataset = [to_jsonable(row) for row in raw_rows]

    artifacts = (
        None
        if results.processor_artifacts is None
        else to_jsonable(results.processor_artifacts)
    )
    analysis = (
        None
        if results.analysis is None
        else to_jsonable(results.analysis.model_dump(mode = "json"))
    )

    return dataset, artifacts, analysis
