# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from core.training.account_jobs import account_path, managed_account, validate_recipe_access
import base64
import functools
import hashlib
import io
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from utils.paths import recipe_datasets_root

from .jsonable import _is_pandas_missing, to_jsonable
from .local_callable_validators import (
    register_oxc_local_callable_validators,
    split_oxc_local_callable_validators,
)
from .text_format_validators import (
    register_text_format_local_callable_validators,
    split_text_format_local_callable_validators,
)

_IMAGE_CONTEXT_PATCHED = False


def _encode_bytes_to_base64(value: bytes | bytearray) -> str:
    return base64.b64encode(bytes(value)).decode("utf-8")


def _load_image_file_to_base64(path_value: str, *, base_path: str | None = None) -> str | None:
    account_path(
        Path(base_path) / path_value
        if base_path and not Path(path_value).is_absolute()
        else path_value
    )
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
            # The cwd fallback is a different file from the one checked above.
            try:
                account_path(candidate)
            except HTTPException:
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
        from data_designer.config.models import ImageContext  # pyright: ignore[reportMissingImports]
        from data_designer.config.utils.image_helpers import is_image_path  # pyright: ignore[reportMissingImports]
    except ImportError:
        return

    if getattr(ImageContext, "_unsloth_image_context_patch_applied", False):
        _IMAGE_CONTEXT_PATCHED = True
        return

    original_auto_resolve = ImageContext._auto_resolve_context_value

    def _patched_auto_resolve(self: Any, context_value: Any, base_path: str | None) -> Any:
        normalized = _normalize_image_context_value(context_value, base_path = base_path)
        if not managed_account():
            return original_auto_resolve(self, normalized, base_path)
        # Managed image paths load under the workspace rule; with no base_path the engine never opens files.
        if (
            isinstance(normalized, str)
            and not normalized.lower().startswith(("http://", "https://", "data:"))
            and is_image_path(normalized)
        ):
            if as_base64 := _load_image_file_to_base64(normalized, base_path = base_path):
                return self._format_base64_context(as_base64)
        return original_auto_resolve(self, normalized, None)

    ImageContext._auto_resolve_context_value = _patched_auto_resolve
    setattr(ImageContext, "_unsloth_image_context_patch_applied", True)
    _IMAGE_CONTEXT_PATCHED = True


def _blank_missing_prompt_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return value


def _allow_empty_prompt(*_args: Any, **_kwargs: Any) -> None:
    return None


# Filters that stringify their input before finalize runs, so a missing cell would still render "None" / "nan".
_STRINGIFYING_FILTERS = (
    "capitalize",
    "escape",
    "forceescape",
    "lower",
    "replace",
    "string",
    "title",
    "trim",
    "truncate",
    "urlencode",
)


def _blank_missing_filter_input(filter_func: Any) -> Any:
    value_index = 1 if hasattr(filter_func, "jinja_pass_arg") else 0

    @functools.wraps(filter_func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if len(args) > value_index:
            args = (
                *args[:value_index],
                _blank_missing_prompt_value(args[value_index]),
                *args[value_index + 1 :],
            )
        return filter_func(*args, **kwargs)

    return wrapper


def _apply_data_designer_prompt_blank_patch() -> None:
    try:
        from data_designer.engine.column_generators.utils.prompt_renderer import (  # pyright: ignore[reportMissingImports]
            RecordBasedPromptRenderer,
        )
    except ImportError:
        return

    if getattr(RecordBasedPromptRenderer, "_unsloth_prompt_blank_patch_applied", False):
        return

    original_prepare = RecordBasedPromptRenderer.prepare_jinja2_multi_template_renderer

    def _patched_prepare(self: Any, template_name: str, *args: Any, **kwargs: Any) -> None:
        # render() prepares on every record; only patch the env this call creates, else filter wrappers nest per row.
        already_prepared = self._template_prepared_in_multi_template_renderer(template_name)
        original_prepare(self, template_name, *args, **kwargs)
        if already_prepared:
            return
        env = self._render_func_registry[template_name].func.__self__
        env.finalize = _blank_missing_prompt_value
        env._assert_rendered_text_not_empty = _allow_empty_prompt
        for name in _STRINGIFYING_FILTERS:
            if name in env.filters:
                env.filters[name] = _blank_missing_filter_input(env.filters[name])

    RecordBasedPromptRenderer.prepare_jinja2_multi_template_renderer = _patched_prepare
    setattr(RecordBasedPromptRenderer, "_unsloth_prompt_blank_patch_applied", True)


def _is_none(value: Any) -> bool:
    return value is None


def _apply_data_designer_expression_blank_patch() -> None:
    # "" is a real text Expression value (an unanswered optional field); data-designer 0.5 fails the run on it and
    # later releases drop the row. Number / bool dtypes keep that, since "" cannot be cast.
    try:
        from data_designer.engine.column_generators.generators.expression import (  # pyright: ignore[reportMissingImports]
            ExpressionColumnGenerator,
        )
    except ImportError:
        return

    if getattr(ExpressionColumnGenerator, "_unsloth_expression_blank_patch_applied", False):
        return

    original_prepare = ExpressionColumnGenerator.prepare_jinja2_template_renderer

    @functools.wraps(original_prepare)
    def _patched_prepare(self: Any, *args: Any, **kwargs: Any) -> None:
        original_prepare(self, *args, **kwargs)
        env = getattr(getattr(self, "_template_render_fn", None), "func", None)
        env = getattr(env, "__self__", None)
        if env is not None and self.config.dtype == "str":
            env._assert_rendered_text_not_empty = _allow_empty_prompt
            self._is_empty_rendered_expression = _is_none

    ExpressionColumnGenerator.prepare_jinja2_template_renderer = _patched_prepare
    setattr(ExpressionColumnGenerator, "_unsloth_expression_blank_patch_applied", True)


def _is_missing_cell(value: Any) -> bool:
    return (
        value is None
        or (isinstance(value, float) and math.isnan(value))
        or _is_pandas_missing(value)
    )


def _as_json_text(value: Any) -> Any:
    if _is_missing_cell(value):
        return None
    if isinstance(value, str):
        return value
    value = to_jsonable(value)
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii = False, default = str)


def _unify_cell_types(values: list[Any]) -> list[Any]:
    """Make one path of a column Arrow-writable: dicts and lists recurse, and a path whose rows disagree
    (text in one row, an object in another) is stored as JSON text instead of failing the batch."""
    import pyarrow

    present = [value for value in values if not _is_missing_cell(value)]
    if not present:
        return values
    if all(isinstance(value, dict) for value in present):
        keys = list(dict.fromkeys(key for value in present for key in value))
        if keys and all(isinstance(key, str) for key in keys):
            fields = {
                key: _unify_cell_types(
                    [value.get(key) if isinstance(value, dict) else None for value in values]
                )
                for key in keys
            }
            return [
                {key: fields[key][index] for key in value} if isinstance(value, dict) else value
                for index, value in enumerate(values)
            ]
    elif all(isinstance(value, (list, tuple)) for value in present):
        flat = _unify_cell_types([item for value in present for item in value])
        result, offset = [], 0
        for value in values:
            if isinstance(value, (list, tuple)):
                result.append(flat[offset : offset + len(value)])
                offset += len(value)
            else:
                result.append(value)
        return result
    elif not any(isinstance(value, (dict, list, tuple)) for value in present):
        kinds = {type(value) for value in present}
        # Arrow infers a mixed path from its first value (a date then a datetime drops the time); int/float is safe.
        if len(kinds) > 1 and not kinds <= {int, float}:
            return [_as_json_text(value) for value in values]
        try:
            pyarrow.array(present, from_pandas = True)
            return values
        except (pyarrow.ArrowInvalid, pyarrow.ArrowTypeError):
            pass
    return [_as_json_text(value) for value in values]


def _unify_dataframe_cell_types(dataframe: Any) -> tuple[Any, list[str]]:
    import pyarrow

    unified = dataframe.copy()
    changed: list[str] = []
    for column in unified.columns:
        if unified[column].dtype != object:
            continue
        values = unified[column].tolist()
        try:
            pyarrow.array(values, from_pandas = True)
            continue
        except (pyarrow.ArrowInvalid, pyarrow.ArrowTypeError):
            pass
        unified[column] = _unify_cell_types(values)
        changed.append(str(column))
    return unified, changed


_MIXED_TYPE_WARNED: set[tuple[str, ...]] = set()


def write_parquet_with_unified_types(dataframe: Any, path: Any, write: Any) -> Any:
    """Run ``write(dataframe)``; if Arrow refuses a column whose rows hold different types at the same path
    (unslothai/unsloth#5848: an LLM Structured field returned as text in one row and an object in another),
    store those paths as JSON text and write again."""
    import pyarrow
    try:
        return write(dataframe)
    except (pyarrow.ArrowInvalid, pyarrow.ArrowTypeError) as exc:
        unified, changed = _unify_dataframe_cell_types(dataframe)
        if not changed:
            raise
        try:
            result = write(unified)
        except (pyarrow.ArrowInvalid, pyarrow.ArrowTypeError):
            raise exc from None
        # data-designer rewrites a batch once per column stage; say it once.
        key = (str(path), *changed)
        if key not in _MIXED_TYPE_WARNED:
            _MIXED_TYPE_WARNED.add(key)
            logging.getLogger("data_designer.unsloth").warning(
                "Column(s) %s returned different value types across rows; mismatched fields were "
                "saved as JSON text.",
                ", ".join(repr(name) for name in changed),
            )
        return result


def _apply_data_designer_mixed_type_parquet_patch() -> None:
    try:
        from data_designer.engine.storage.artifact_storage import (  # pyright: ignore[reportMissingImports]
            ArtifactStorage,
        )
    except ImportError:
        return

    if getattr(ArtifactStorage, "_unsloth_mixed_type_parquet_patch_applied", False):
        return

    original_write = ArtifactStorage.write_parquet_file

    @functools.wraps(original_write)
    def _patched_write(
        self: Any,
        parquet_file_name: str,
        dataframe: Any,
        batch_stage: Any,
        subfolder: str | None = None,
    ) -> Any:
        return write_parquet_with_unified_types(
            dataframe,
            Path(str(getattr(self, "base_dataset_path", ""))) / parquet_file_name,
            lambda frame: original_write(
                self, parquet_file_name, frame, batch_stage, subfolder = subfolder
            ),
        )

    ArtifactStorage.write_parquet_file = _patched_write
    setattr(ArtifactStorage, "_unsloth_mixed_type_parquet_patch_applied", True)


def _apply_data_designer_seed_uri_quote_patch() -> None:
    # Seeds are read as FROM '<uri>': quote it for DuckDB. No ImportError guard: never run unquoted.
    from data_designer.engine.resources.seed_reader import SeedReader  # pyright: ignore[reportMissingImports]

    if getattr(SeedReader, "_unsloth_seed_uri_quote_patch_applied", False):
        return

    original_attach = SeedReader.attach

    @functools.wraps(original_attach)
    def _patched_attach(self: Any, *args: Any, **kwargs: Any) -> None:
        # Calls the class method, so attaching twice never quotes twice.
        self.get_dataset_uri = lambda: type(self).get_dataset_uri(self).replace("'", "''")
        original_attach(self, *args, **kwargs)

    SeedReader.attach = _patched_attach
    setattr(SeedReader, "_unsloth_seed_uri_quote_patch_applied", True)


def _ignore_sqlfluff_inline_config(*_args: Any, **_kwargs: Any) -> None:
    return None


def _apply_data_designer_sql_inline_config_patch() -> None:
    # sqlfluff applies inline "-- sqlfluff:" config from the linted text (recipe data); code_lang sets the dialect.
    try:
        from data_designer.engine.validators.sql import SQLValidator  # pyright: ignore[reportMissingImports]
    except ImportError:
        return

    if getattr(SQLValidator, "_unsloth_sql_inline_config_patch_applied", False):
        return

    original_validate = SQLValidator._validate_query

    @functools.wraps(original_validate)
    def _patched_validate(self: Any, content: str) -> Any:
        # sqlfluff is imported lazily, on the first SQL cell.
        from sqlfluff.core import FluffConfig  # pyright: ignore[reportMissingImports]
        if FluffConfig.process_raw_file_for_config is not _ignore_sqlfluff_inline_config:
            FluffConfig.process_raw_file_for_config = _ignore_sqlfluff_inline_config
        return original_validate(self, content)

    SQLValidator._validate_query = _patched_validate
    setattr(SQLValidator, "_unsloth_sql_inline_config_patch_applied", True)


def _require_public_provider_endpoint(endpoint: str) -> None:
    """Managed endpoints need HTTPS unless private switch is on; the metadata block holds even then."""
    if not managed_account():
        return
    from urllib.parse import urlsplit

    from core.inference.providers import (
        managed_private_url_hint,
        provider_address_excluding_metadata,
        public_provider_address,
    )
    from utils.managed_provider_url_settings import get_managed_private_provider_urls_allowed

    url = str(endpoint or "")
    if get_managed_private_provider_urls_allowed():
        try:
            # The switch lifts HTTPS-only and public-only, not http(s)-only.
            if urlsplit(url).scheme not in ("http", "https"):
                raise ValueError("Provider endpoints must use http or https.")
            provider_address_excluding_metadata(url)
        except ValueError as exc:
            raise HTTPException(
                status_code = 403, detail = f"Recipe provider endpoint refused: {exc}"
            ) from exc
        return

    try:
        if urlsplit(url).scheme != "https":
            raise ValueError(
                "Managed accounts may only use HTTPS provider endpoints."
                + managed_private_url_hint()
            )
        public_provider_address(url)
    except ValueError as exc:
        raise HTTPException(
            status_code = 403, detail = f"Recipe provider endpoint refused: {exc}"
        ) from exc


def install_public_egress_guard() -> None:
    """Job subprocess only: refuses rebinding hosts at connect; with the private switch on, metadata
    only."""
    if not managed_account():
        return
    from utils.managed_provider_url_settings import get_managed_private_provider_urls_allowed

    import ipaddress
    import socket

    from core.inference.providers import _metadata_address

    resolve = socket.getaddrinfo

    def guarded_getaddrinfo(host, port, *args, **kwargs):
        # Per lookup, not captured at install: a worker outlives the switch it started under.
        private_allowed = get_managed_private_provider_urls_allowed()
        infos = resolve(host, port, *args, **kwargs)
        for info in infos:
            address = str(info[4][0]).split("%", 1)[0]
            if _metadata_address(address):
                raise socket.gaierror(
                    f"Managed accounts may not reach cloud metadata services: {host!r}"
                )
            if not private_allowed and not ipaddress.ip_address(address).is_global:
                raise socket.gaierror(
                    f"Managed accounts may only reach public-network addresses: {host!r}"
                )
        return infos

    def guarded_gethostbyname(host):
        return str(guarded_getaddrinfo(host, None, socket.AF_INET)[0][4][0])

    socket.getaddrinfo = guarded_getaddrinfo
    socket.gethostbyname = guarded_gethostbyname


def build_model_providers(recipe: dict[str, Any]):
    from data_designer.config.models import ModelProvider  # pyright: ignore[reportMissingImports]

    providers: list[ModelProvider] = []
    for provider in recipe.get("model_providers", []):
        _require_public_provider_endpoint(provider.get("endpoint"))
        api_key = provider.get("api_key")
        api_key_env = provider.get("api_key_env")
        if not api_key and api_key_env and not managed_account():
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


def _validate_recipe_runtime_support(recipe: dict[str, Any], model_providers: list[Any]) -> None:
    if _recipe_has_llm_columns(recipe) and not model_providers:
        raise ValueError("Add a Provider connection block before running this recipe.")


def recipe_has_stdio_mcp(recipe: dict[str, Any]) -> bool:
    """True when the recipe asks for a local (stdio) MCP provider, i.e. a command
    this host would run. Routes gate on it to keep that behind a UI session."""
    providers = recipe.get("mcp_providers") or []
    if not isinstance(providers, list):
        return False
    return any(
        isinstance(provider, dict) and provider.get("provider_type") == "stdio"
        for provider in providers
    )


# Each env fallback is read only when its literal key is empty.
_ENV_CREDENTIAL_KEYS = {"api_key_env": "api_key", "token_env": "token", "hf_token_env": "hf_token"}
_LITERAL_CREDENTIAL_KEYS = set(_ENV_CREDENTIAL_KEYS.values())


def _names_env_credential(value: Any) -> bool:
    if isinstance(value, list):
        return any(_names_env_credential(item) for item in value)
    if not isinstance(value, dict):
        return False
    for key, item in value.items():
        if key in _ENV_CREDENTIAL_KEYS and item and not value.get(_ENV_CREDENTIAL_KEYS[key]):
            return True
        if key in _LITERAL_CREDENTIAL_KEYS and isinstance(item, str) and item in os.environ:
            return True
        if isinstance(item, (dict, list)) and _names_env_credential(item):
            return True
    return False


def recipe_names_host_env_credential(recipe: Any) -> bool:
    """A bare api_key/token name resolves from this host's env; top-level local providers are
    exempt, since the routes rewire exactly those to this server and drop external auth."""
    if isinstance(recipe, dict) and isinstance(recipe.get("model_providers"), list):
        rest = {key: value for key, value in recipe.items() if key != "model_providers"}
        external = [
            provider
            for provider in recipe["model_providers"]
            if not (isinstance(provider, dict) and provider.get("is_local") is True)
        ]
        return _names_env_credential(rest) or _names_env_credential(external)
    return _names_env_credential(recipe)


def _hf_seed_sends_host_token_elsewhere(recipe: Any) -> bool:
    """A tokenless HF seed sends the host's HF token to whatever endpoint it names."""
    if not isinstance(recipe, dict):
        return False
    seed_config = recipe.get("seed_config")
    source = seed_config.get("source") if isinstance(seed_config, dict) else None
    if not isinstance(source, dict) or source.get("seed_type") != "hf":
        return False
    if str(source.get("token") or "").strip():
        return False
    typed = str(source.get("endpoint") or "").strip()
    if not typed:
        return False
    from utils.hf_endpoint import get_hf_endpoint

    return typed.rstrip("/") != get_hf_endpoint().rstrip("/")


def require_ui_session_for_host_env_credentials(recipe: Any, via_api_key: bool) -> None:
    """API key / keyless callers bring their own credentials; UI sessions keep the env lookup."""
    if via_api_key and (
        recipe_names_host_env_credential(recipe) or _hf_seed_sends_host_token_elsewhere(recipe)
    ):
        raise HTTPException(
            status_code = 403,
            detail = "Recipes sent with an API key cannot read credentials from the server's "
            "environment. Put the provider key in api_key instead of naming an environment variable.",
        )


def _require_confinable_mcp_transport(provider_type: str) -> None:
    """Refuse network MCP for managed accounts: the engine opens its own connections and cannot use chat's confined transport."""
    if provider_type not in {"sse", "streamable_http"} or not managed_account():
        return
    raise HTTPException(
        status_code = 403,
        detail = (
            "Recipe MCP servers are unavailable for managed accounts until the recipe "
            "engine uses the account-confined MCP transport."
        ),
    )


def build_mcp_providers(recipe: dict[str, Any]) -> list:
    from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider  # pyright: ignore[reportMissingImports]

    # stdio providers spawn a local subprocess; same gate as chat MCP.
    from core.inference.mcp_client import stdio_mcp_enabled

    stdio_allowed = stdio_mcp_enabled()

    providers: list[MCPProvider | LocalStdioMCPProvider] = []
    for provider in recipe.get("mcp_providers", []):
        if not isinstance(provider, dict):
            continue
        provider_type = provider.get("provider_type")
        if provider_type == "stdio":
            if not stdio_allowed:
                continue
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
            _require_confinable_mcp_transport(provider_type)
            api_key = provider.get("api_key")
            api_key_env = provider.get("api_key_env")
            if not api_key and api_key_env and not managed_account():
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


def _strip_frontend_model_config_metadata(recipe: dict[str, Any]) -> dict[str, Any]:
    model_configs = recipe.get("model_configs")
    if not isinstance(model_configs, list):
        return recipe

    changed = False
    next_model_configs: list[Any] = []
    for model_config in model_configs:
        if isinstance(model_config, dict) and "gguf_variant" in model_config:
            next_model_config = dict(model_config)
            next_model_config.pop("gguf_variant", None)
            next_model_configs.append(next_model_config)
            changed = True
            continue
        next_model_configs.append(model_config)

    if not changed:
        return recipe

    return {
        **recipe,
        "model_configs": next_model_configs,
    }


def validate_recipe_names(recipe: dict[str, Any]) -> None:
    """Processor names must be one folder: Data Designer stores them as `processors-files/<name>`
    and neither previews nor lists a nested one. Column names are sanitized by the validator patch."""
    processors = recipe.get("processors")
    for item in processors if isinstance(processors, list) else []:
        name = item.get("name") if isinstance(item, dict) else None
        if isinstance(name, str) and (
            "/" in name or "\\" in name or name in (".", "..") or re.match(r"^[A-Za-z]:", name)
        ):
            raise ValueError(
                f"Processor names can't contain '/' or '\\', start with a drive letter, or be '.' / '..': {name!r}"
            )


def _safe_module_name(record_id: Any, column_name: str) -> str:
    # The code validator writes `<tmp>/<module>.py`: keep the column name readable but path-free.
    safe = re.sub(r"[^\w-]", "_", str(column_name))
    if safe != column_name:
        safe += "_" + hashlib.sha256(str(column_name).encode("utf-8")).hexdigest()[:8]
    return f"{record_id}_{safe}"


def _apply_data_designer_python_validator_path_patch() -> None:
    try:
        from data_designer.engine.validators.python import (  # pyright: ignore[reportMissingImports]
            RECORD_ID_COLUMN_NAME,
            PythonValidator,
        )
    except ImportError:
        return

    if getattr(PythonValidator, "_unsloth_python_validator_path_patch_applied", False):
        return

    def _patched_write(self: Any, row: Any, code_column: str, path: str) -> None:
        root = Path(path).resolve()
        target = (
            root / f"{self._get_module_name(row[RECORD_ID_COLUMN_NAME], code_column)}.py"
        ).resolve()
        if target.parent != root:
            raise ValueError(f"Code validator file left its temp folder: {code_column!r}")
        with open(target, "w", encoding = "utf-8") as file:
            file.write(row[code_column])

    PythonValidator._get_module_name = staticmethod(_safe_module_name)
    PythonValidator._write_code_to_file = _patched_write
    setattr(PythonValidator, "_unsloth_python_validator_path_patch_applied", True)


def build_config_builder(recipe: dict[str, Any]):
    validate_recipe_access(recipe)
    validate_recipe_names(recipe)
    _apply_data_designer_image_context_patch()
    from data_designer.config import DataDesignerConfigBuilder  # pyright: ignore[reportMissingImports]
    from data_designer.config.processors import ProcessorType  # pyright: ignore[reportMissingImports]

    recipe_core = {
        key: value
        for key, value in recipe.items()
        if key not in {"model_providers", "mcp_providers"}
    }
    recipe_core = _strip_frontend_model_config_metadata(recipe_core)
    recipe_core, oxc_local_callable_specs = split_oxc_local_callable_validators(recipe_core)
    recipe_core, text_format_specs = split_text_format_local_callable_validators(recipe_core)
    builder = DataDesignerConfigBuilder.from_config({"data_designer": recipe_core})
    register_oxc_local_callable_validators(
        builder = builder,
        specs = oxc_local_callable_specs,
    )
    register_text_format_local_callable_validators(
        builder = builder,
        specs = text_format_specs,
    )

    # from_config skips processors; re-attach them.
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


def create_data_designer(recipe: dict[str, Any], *, artifact_path: str | None = None):
    validate_recipe_access(recipe)
    account_path(artifact_path)
    _apply_data_designer_image_context_patch()
    _apply_data_designer_prompt_blank_patch()
    _apply_data_designer_expression_blank_patch()
    _apply_data_designer_mixed_type_parquet_patch()
    _apply_data_designer_seed_uri_quote_patch()
    _apply_data_designer_sql_inline_config_patch()
    _apply_data_designer_python_validator_path_patch()
    from data_designer.interface.data_designer import DataDesigner  # pyright: ignore[reportMissingImports]

    if artifact_path is None:
        # Packaged Unsloth can run with cwd=/, so pin the artifact root.
        artifact_path = str(recipe_datasets_root())

    recipe = _strip_frontend_model_config_metadata(recipe)
    model_providers = build_model_providers(recipe)
    _validate_recipe_runtime_support(recipe, model_providers)

    # DataDesigner requires >=1 model provider even with no LLM columns.
    if not model_providers:
        from data_designer.config.models import ModelProvider  # pyright: ignore[reportMissingImports]
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
    recipe: dict[str, Any], num_records: int
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    builder = build_config_builder(recipe)
    designer = create_data_designer(recipe)
    results = designer.preview(builder, num_records = num_records)

    dataset: list[dict[str, Any]] = []
    if results.dataset is not None:
        raw_rows = results.dataset.to_dict(orient = "records")
        dataset = [to_jsonable(row) for row in raw_rows]

    artifacts = (
        None if results.processor_artifacts is None else to_jsonable(results.processor_artifacts)
    )
    analysis = (
        None if results.analysis is None else to_jsonable(results.analysis.model_dump(mode = "json"))
    )

    return dataset, artifacts, analysis
