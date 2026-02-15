from __future__ import annotations

import os
from typing import Any

def _to_jsonable(value: Any) -> Any:
    # pydantic/fastapi can't serialize numpy arrays/scalars.
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        np = None  # type: ignore

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    # pandas Timestamp/date-like
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except Exception:
            pass

    return value


def build_model_providers(recipe: dict[str, Any]):
    from data_designer.config.default_model_settings import get_default_providers
    from data_designer.config.models import ModelProvider

    providers: list[ModelProvider] = []
    for provider in recipe.get("model_providers", []):
        api_key = provider.get("api_key")
        api_key_env = provider.get("api_key_env")
        if not api_key and api_key_env:
            api_key = os.getenv(api_key_env)
        providers.append(
            ModelProvider(
                name=provider["name"],
                endpoint=provider["endpoint"],
                provider_type=provider.get("provider_type", "openai"),
                api_key=api_key,
                extra_headers=provider.get("extra_headers"),
                extra_body=provider.get("extra_body"),
            )
        )

    # DataDesigner currently expects at least one provider even if they only use static samplers,
    # but it's fine it gives a warning only.
    return providers or get_default_providers()


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
                    name=str(provider.get("name", "")),
                    command=str(provider.get("command", "")),
                    args=[str(value) for value in args],
                    env={str(key): str(value) for key, value in env.items()},
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
                    name=str(provider.get("name", "")),
                    endpoint=str(provider.get("endpoint", "")),
                    api_key=str(api_key) if api_key else None,
                )
            )
    return providers


def build_config_builder(recipe: dict[str, Any]):
    from data_designer.config import DataDesignerConfigBuilder

    recipe_core = {
        key: value
        for key, value in recipe.items()
        if key not in {"model_providers", "mcp_providers"}
    }
    return DataDesignerConfigBuilder.from_config({"data_designer": recipe_core})


def create_data_designer(recipe: dict[str, Any]):
    from data_designer.interface.data_designer import DataDesigner

    return DataDesigner(
        model_providers=build_model_providers(recipe),
        mcp_providers=build_mcp_providers(recipe),
    )


def validate_recipe(recipe: dict[str, Any]) -> None:
    builder = build_config_builder(recipe)
    designer = create_data_designer(recipe)
    designer.validate(builder)


def preview_recipe(
    recipe: dict[str, Any],
    num_records: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    builder = build_config_builder(recipe)
    designer = create_data_designer(recipe)
    results = designer.preview(builder, num_records=num_records)

    dataset: list[dict[str, Any]] = []
    if results.dataset is not None:
        raw_rows = results.dataset.to_dict(orient="records")
        dataset = [_to_jsonable(row) for row in raw_rows]

    artifacts = (
        None
        if results.processor_artifacts is None
        else _to_jsonable(results.processor_artifacts)
    )

    return dataset, artifacts
