# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the tool (command-template) validator that rides the OXC machinery.

Also guards the shared split/register path used by the OXC validator, since the
tool validator extends the same ``local_callable_validators.py`` module.
"""

import base64
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("structlog")
pytest.importorskip("pandas")


def _load_module():
    backend_root = Path(__file__).resolve().parent.parent
    module_path = backend_root / "core" / "data_recipe" / "local_callable_validators.py"
    spec = importlib.util.spec_from_file_location(
        "local_callable_validators_under_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module.__name__] = module
    spec.loader.exec_module(module)
    return module


tool = _load_module()


def _tool_marker(file_ext: str, command: str) -> str:
    payload = (
        base64.urlsafe_b64encode(json.dumps({"ext": file_ext, "command": command}).encode("utf-8"))
        .decode("ascii")
        .rstrip("=")
    )
    return f"{tool.TOOL_VALIDATION_FN_MARKER}:{payload}"


def _tool_column(
    *,
    name: str = "tool_check",
    file_ext: str,
    command: str,
    batch_size: int = 7,
) -> dict:
    return {
        "column_type": "validation",
        "name": name,
        "drop": False,
        "target_columns": ["code"],
        "validator_type": "local_callable",
        "validator_params": {"validation_function": _tool_marker(file_ext, command)},
        "batch_size": batch_size,
    }


def test_tool_marker_round_trip():
    spec = tool._parse_tool_spec(
        column = _tool_column(file_ext = "go", command = "go vet ./...", batch_size = 7)
    )
    assert spec is not None
    assert spec.name == "tool_check"
    assert spec.file_ext == "go"
    assert spec.command == "go vet ./..."
    assert spec.batch_size == 7
    assert spec.drop is False


@pytest.mark.parametrize(
    "bad_ext",
    ["../x", "a/b", "", "a" * 30, ".", "x y"],
)
def test_tool_spec_rejects_bad_extensions(bad_ext):
    spec = tool._parse_tool_spec(column = _tool_column(file_ext = bad_ext, command = "go vet ./..."))
    assert spec is None


def test_tool_spec_rejects_empty_command():
    spec = tool._parse_tool_spec(column = _tool_column(file_ext = "go", command = "  "))
    assert spec is None


def test_tool_spec_rejects_non_tool_markers():
    column = {
        "column_type": "validation",
        "name": "oxc",
        "target_columns": ["code"],
        "validator_type": "local_callable",
        "validator_params": {"validation_function": "unsloth_oxc_validator:javascript:syntax:auto"},
        "batch_size": 10,
    }
    assert tool._parse_tool_spec(column = column) is None


def test_split_tool_extracts_and_leaves_others():
    recipe = {
        "columns": [
            _tool_column(name = "go_one", file_ext = "go", command = "go vet ./..."),
            {
                "column_type": "validation",
                "name": "oxc_one",
                "target_columns": ["code"],
                "validator_type": "local_callable",
                "validator_params": {
                    "validation_function": "unsloth_oxc_validator:javascript:syntax:auto"
                },
                "batch_size": 10,
            },
            {
                "column_type": "validation",
                "name": "python_one",
                "target_columns": ["code"],
                "validator_type": "code",
                "validator_params": {"code_lang": "python"},
                "batch_size": 10,
            },
        ]
    }
    sanitized, specs = tool.split_tool_local_callable_validators(recipe)
    assert len(specs) == 1
    assert specs[0].name == "go_one"
    assert [column["name"] for column in sanitized["columns"]] == ["oxc_one", "python_one"]


def test_tool_callable_runs_successful_command():
    import pandas as pd

    fn = tool._build_tool_validation_function("txt", "test -s {file}")
    out = fn(pd.DataFrame({"code": ["hello world"]}))
    assert list(out["is_valid"]) == [True]
    assert out["tool_output"].iloc[0] == ""


def test_tool_callable_runs_failing_command():
    import pandas as pd

    fn = tool._build_tool_validation_function("txt", "false")
    out = fn(pd.DataFrame({"code": ["x"]}))
    assert list(out["is_valid"]) == [False]
    assert out["error_count"].iloc[0] == 1
    assert out["error_message"].iloc[0] != ""


def test_tool_callable_missing_binary_is_graceful():
    import pandas as pd

    fn = tool._build_tool_validation_function(
        "txt",
        "definitely_not_a_real_tool_xyz_123 {file}",
    )
    out = fn(pd.DataFrame({"code": ["x"]}))
    assert list(out["is_valid"]) == [False]
    assert "not found" in out["error_message"].iloc[0].lower()


def test_tool_callable_empty_df():
    import pandas as pd

    fn = tool._build_tool_validation_function("txt", "true")
    out = fn(pd.DataFrame({"code": []}))
    assert "is_valid" in out.columns
    assert len(out) == 0


def test_tool_callable_timeout_is_graceful(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(tool, "_TOOL_RUN_TIMEOUT_SECONDS", 1)
    fn = tool._build_tool_validation_function("txt", "sleep 5")
    out = fn(pd.DataFrame({"code": ["x"]}))
    assert list(out["is_valid"]) == [False]
    assert "timed out" in out["error_message"].iloc[0]


@pytest.mark.skipif(shutil.which("go") is None, reason = "go toolchain not installed")
def test_go_scaffold_and_vet(tmp_path):
    import pandas as pd

    go_source = (
        "package main\n\n" 'import "fmt"\n\n' "func main() {\n" '\tfmt.Println("hi")\n' "}\n"
    )
    fn = tool._build_tool_validation_function("go", "go vet ./...")
    out = fn(pd.DataFrame({"code": [go_source]}))
    assert list(out["is_valid"]) == [True]


def test_register_tool_creates_local_callable_column():
    data_designer = pytest.importorskip("data_designer")
    from data_designer.config.column_configs import ValidationColumnConfig
    from data_designer.config.validator_params import ValidatorType

    class _StubBuilder:
        def __init__(self) -> None:
            self.columns = []

        def add_column(self, column) -> None:
            self.columns.append(column)

    builder = _StubBuilder()
    tool.register_tool_local_callable_validators(
        builder = builder,
        specs = [
            tool.ToolLocalCallableValidatorSpec(
                name = "tc",
                drop = False,
                target_columns = ["code"],
                batch_size = 5,
                file_ext = "go",
                command = "go vet ./...",
            )
        ],
    )
    assert len(builder.columns) == 1
    column = builder.columns[0]
    assert isinstance(column, ValidationColumnConfig)
    assert column.validator_type == ValidatorType.LOCAL_CALLABLE
    assert callable(column.validator_params.validation_function)
