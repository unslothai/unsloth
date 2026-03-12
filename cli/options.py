# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Generate Typer CLI options from Pydantic models."""

import functools
import inspect
from pathlib import Path
from typing import Any, Callable, Optional, get_args, get_origin

import typer
from pydantic import BaseModel


def _python_name_to_cli_flag(name: str) -> str:
    """Convert python_name to --cli-flag."""
    return "--" + name.replace("_", "-")


def _unwrap_optional(annotation: Any) -> Any:
    """Unwrap Optional[X] to X."""
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        if type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                return non_none[0]
    return annotation


def _is_bool_field(annotation: Any) -> bool:
    """Check if field is a boolean (including Optional[bool])."""
    return _unwrap_optional(annotation) is bool


def _is_list_type(annotation: Any) -> bool:
    """Check if type is a List."""
    return get_origin(annotation) is list


def _get_python_type(annotation: Any) -> type:
    """Get the Python type for annotation."""
    unwrapped = _unwrap_optional(annotation)
    if unwrapped in (str, int, float, bool, Path):
        return unwrapped
    return str


def _collect_config_fields(config_class: type[BaseModel]) -> list[tuple[str, Any]]:
    """
    Collect all fields from a config class, flattening nested models. Returns list of
    (name, field_info) tuples. Raises ValueError on duplicate field names.
    """
    fields = []
    seen_names: set[str] = set()

    for name, field_info in config_class.model_fields.items():
        annotation = field_info.annotation
        # Skip nested models - recurse into them
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            for nested_name, nested_field in annotation.model_fields.items():
                if nested_name in seen_names:
                    raise ValueError(f"Duplicate field name '{nested_name}' in config")
                seen_names.add(nested_name)
                fields.append((nested_name, nested_field))
        else:
            if name in seen_names:
                raise ValueError(f"Duplicate field name '{name}' in config")
            seen_names.add(name)
            fields.append((name, field_info))
    return fields


def add_options_from_config(config_class: type[BaseModel]) -> Callable:
    """
    Decorator that adds CLI options for all fields in a Pydantic config model.

    The decorated function should declare a `config_overrides: dict = None` parameter
    which will receive a dict of all CLI-provided config values.
    """
    fields = _collect_config_fields(config_class)
    field_names = {
        name for name, field_info in fields if not _is_list_type(field_info.annotation)
    }

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        original_params = list(sig.parameters.values())
        original_param_names = {p.name for p in original_params}

        # Build new parameters: config fields first, then original params
        new_params = []

        for field_name, field_info in fields:
            # Skip fields already defined in function signature (e.g., with envvar)
            if field_name in original_param_names:
                continue
            annotation = field_info.annotation
            if _is_list_type(annotation):
                continue

            flag_name = _python_name_to_cli_flag(field_name)
            help_text = field_info.description or ""

            if _is_bool_field(annotation):
                default = typer.Option(
                    None,
                    f"{flag_name}/--no-{field_name.replace('_', '-')}",
                    help = help_text,
                )
                param = inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default = default,
                    annotation = Optional[bool],
                )
            else:
                py_type = _get_python_type(annotation)
                default = typer.Option(None, flag_name, help = help_text)
                param = inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default = default,
                    annotation = Optional[py_type],
                )
            new_params.append(param)

        # Add original params, excluding config_overrides (will be injected)
        for param in original_params:
            if param.name != "config_overrides":
                new_params.append(param)

        new_sig = sig.replace(parameters = new_params)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            config_overrides = {}
            for key in list(kwargs.keys()):
                if key in field_names:
                    if kwargs[key] is not None:
                        config_overrides[key] = kwargs[key]
                    # Only delete if not an explicitly declared parameter
                    if key not in original_param_names:
                        del kwargs[key]

            kwargs["config_overrides"] = config_overrides
            return func(*args, **kwargs)

        wrapper.__signature__ = new_sig
        return wrapper

    return decorator
