# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the service-level wiring of tool and custom validators.

build_config_builder must split both marker families out of the recipe and
re-register them as real LOCAL_CALLABLE columns with working callables. This
is the integration seam between the two validator modules and data_designer.
"""

import base64
import json

import pytest

pytest.importorskip("structlog")
pytest.importorskip("pandas")
data_designer = pytest.importorskip("data_designer")

from core.data_recipe import service  # noqa: E402


def _b64url(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _tool_marker() -> str:
    spec = {
        "ext": "go",
        "command": "test -s {file}",
        "scaffold": [
            {"path": "go.mod", "content": "module example.com/check\n\ngo 1.21\n"},
            {"path": "main.go", "content": "{source}"},
        ],
    }
    return "unsloth_tool_validator:" + _b64url(json.dumps(spec, separators = (",", ":")))


def _custom_marker() -> str:
    source = (
        "def validate(df):\n    df['is_valid'] = df.iloc[:, 0].str.len() > 0\n    return df\n"
    )
    return "unsloth_custom_validator:" + _b64url(source)


def _recipe() -> dict:
    return {
        "model_providers": [],
        "mcp_providers": [],
        "model_configs": [
            {
                "alias": "coding-model",
                "model": "",
                "provider": "openai-compatible",
                "inference_parameters": {},
            }
        ],
        "columns": [
            {
                "column_type": "llm-code",
                "name": "code_implementation",
                "drop": False,
                "model_alias": "coding-model",
                "prompt": "Write Go code.",
                "with_trace": "none",
                "extract_reasoning_content": False,
                "code_lang": "go",
            },
            {
                "column_type": "validation",
                "name": "tool_check",
                "drop": True,
                "target_columns": ["code_implementation"],
                "validator_type": "local_callable",
                "validator_params": {"validation_function": _tool_marker()},
                "batch_size": 10,
            },
            {
                "column_type": "validation",
                "name": "custom_check",
                "drop": True,
                "target_columns": ["code_implementation"],
                "validator_type": "local_callable",
                "validator_params": {"validation_function": _custom_marker()},
                "batch_size": 10,
            },
        ],
        "processors": [],
    }


def test_build_config_builder_registers_tool_and_custom_callables():
    config = service.build_config_builder(_recipe()).build()

    validation_columns = [column for column in config.columns if column.column_type == "validation"]
    assert [column.name for column in validation_columns] == ["tool_check", "custom_check"]

    from data_designer.config.validator_params import ValidatorType

    for column in validation_columns:
        assert column.validator_type == ValidatorType.LOCAL_CALLABLE
        assert callable(column.validator_params.validation_function)

    llm_columns = [column for column in config.columns if column.column_type == "llm-code"]
    assert [column.name for column in llm_columns] == ["code_implementation"]


def test_build_config_builder_tool_callable_runs_with_scaffold():
    config = service.build_config_builder(_recipe()).build()
    tool_column = next(column for column in config.columns if column.name == "tool_check")
    import pandas as pd

    out = tool_column.validator_params.validation_function(
        pd.DataFrame({"code_implementation": ["package main\n\nfunc main() {}\n"]})
    )
    assert list(out["is_valid"]) == [True]


def test_build_config_builder_custom_callable_runs():
    config = service.build_config_builder(_recipe()).build()
    custom_column = next(column for column in config.columns if column.name == "custom_check")
    import pandas as pd

    out = custom_column.validator_params.validation_function(
        pd.DataFrame({"code_implementation": ["hello", ""]})
    )
    assert list(out["is_valid"]) == [True, False]


def test_build_config_builder_leaves_non_marker_columns_untouched():
    recipe = _recipe()
    recipe["columns"].append(
        {
            "column_type": "sampler",
            "name": "domain",
            "drop": False,
            "sampler_type": "category",
            "params": {"values": ["a", "b"]},
        }
    )
    config = service.build_config_builder(recipe).build()
    assert any(column.name == "domain" for column in config.columns)
