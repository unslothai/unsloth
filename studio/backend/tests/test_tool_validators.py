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


def _tool_marker(
    file_ext: str,
    command: str,
    scaffold: list[dict] | None = None,
) -> str:
    spec: dict = {"ext": file_ext, "command": command}
    if scaffold is not None:
        spec["scaffold"] = scaffold
    payload = base64.urlsafe_b64encode(json.dumps(spec).encode("utf-8")).decode("ascii").rstrip("=")
    return f"{tool.TOOL_VALIDATION_FN_MARKER}:{payload}"


def _tool_column(
    *,
    name: str = "tool_check",
    file_ext: str,
    command: str,
    scaffold: list[dict] | None = None,
    batch_size: int = 7,
) -> dict:
    return {
        "column_type": "validation",
        "name": name,
        "drop": False,
        "target_columns": ["code"],
        "validator_type": "local_callable",
        "validator_params": {"validation_function": _tool_marker(file_ext, command, scaffold)},
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
    assert spec.scaffold == ()


def test_tool_spec_parses_scaffold_rows():
    scaffold = [
        {"path": "go.mod", "content": "module example.com/check\n\ngo 1.21\n"},
        {"path": "main.go", "content": "{source}"},
    ]
    spec = tool._parse_tool_spec(
        column = _tool_column(
            file_ext = "go",
            command = "go vet {file}",
            scaffold = scaffold,
        )
    )
    assert spec is not None
    assert spec.scaffold == (
        ("go.mod", "module example.com/check\n\ngo 1.21\n"),
        ("main.go", "{source}"),
    )


def test_tool_spec_scaffold_is_optional():
    spec = tool._parse_tool_spec(column = _tool_column(file_ext = "go", command = "go vet ./..."))
    assert spec is not None
    assert spec.scaffold == ()


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


@pytest.mark.parametrize(
    "bad_path",
    [
        "../escape.txt",
        "a/../../b.txt",
        "/absolute.txt",
        "C:/drive.txt",
        "a\\backslash.txt",
        ".",
        "..",
        "a/./b",
    ],
)
def test_tool_spec_rejects_bad_scaffold_paths(bad_path):
    spec = tool._parse_tool_spec(
        column = _tool_column(
            file_ext = "txt",
            command = "cat {file}",
            scaffold = [{"path": bad_path, "content": "x"}],
        )
    )
    assert spec is None


@pytest.mark.parametrize(
    "scaffold",
    [
        ["not-a-dict"],
        [{"path": "a.txt", "content": 42}],
        [{"path": "a.txt"}],
        [{"path": "a.txt", "content": "x"}, {"path": "b.txt", "content": "x"}] * 6,
        [{"path": "big.txt", "content": "x" * (32 * 1024)}],
    ],
)
def test_tool_spec_rejects_bad_scaffold_entries(scaffold):
    spec = tool._parse_tool_spec(
        column = _tool_column(
            file_ext = "txt",
            command = "cat {file}",
            scaffold = scaffold,
        )
    )
    assert spec is None


def test_tool_spec_skips_empty_scaffold_paths():
    spec = tool._parse_tool_spec(
        column = _tool_column(
            file_ext = "txt",
            command = "cat {file}",
            scaffold = [
                {"path": "", "content": "ignored"},
                {"path": "main.txt", "content": "{source}"},
            ],
        )
    )
    assert spec is not None
    assert spec.scaffold == (("main.txt", "{source}"),)


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


def test_tool_max_workers_at_least_one():
    assert tool._tool_max_workers() >= 1


def test_tool_batch_empty_returns_empty():
    results = tool._run_tool_batch(
        file_ext = "txt",
        command = "true",
        scaffold = (),
        code_values = [],
        max_workers = 3,
    )
    assert results == []


def test_tool_batch_parallel_preserves_row_order():
    results = tool._run_tool_batch(
        file_ext = "txt",
        command = "cat {file}",
        scaffold = (),
        code_values = ["first", "second", "third"],
        max_workers = 3,
    )
    assert [result["is_valid"] for result in results] == [True, True, True]
    assert [result["tool_output"] for result in results] == ["first", "second", "third"]


def test_tool_batch_parallel_uses_unique_temp_dirs():
    results = tool._run_tool_batch(
        file_ext = "go",
        command = "sh -c 'test -s {file} && test -f {dir}/go.mod'",
        scaffold = (("go.mod", "module example.com/check\n"), ("main.go", "{source}")),
        code_values = ["a", "b", "c", "d"],
        max_workers = 4,
    )
    assert [result["is_valid"] for result in results] == [True, True, True, True]


def test_tool_batch_parallel_runs_faster_than_serial():
    import time

    start = time.monotonic()
    results = tool._run_tool_batch(
        file_ext = "txt",
        command = "sleep 0.5",
        scaffold = (),
        code_values = ["a", "b", "c"],
        max_workers = 3,
    )
    elapsed = time.monotonic() - start
    assert len(results) == 3
    assert all(result["is_valid"] for result in results)
    # Serial would take ~1.5s; parallel ~0.5s. Keep a wide margin for CI.
    assert elapsed < 1.4, f"parallel batch took {elapsed:.2f}s"


def test_parallel_batch_rows_get_distinct_file_paths():
    """Each concurrent row must write to its own temp dir: no shared {file}."""
    results = tool._run_tool_batch(
        file_ext = "go",
        command = "echo {file}",
        scaffold = (("go.mod", "module example.com/check\n"), ("main.go", "{source}")),
        code_values = ["a", "b", "c", "d"],
        max_workers = 4,
    )
    paths = [result["tool_output"] for result in results]
    assert all(path.endswith("main.go") for path in paths)
    assert len(set(paths)) == 4, f"rows shared temp paths: {paths}"


def test_parallel_batch_staggered_completion_keeps_row_order():
    """Rows finishing out of order must still map back to their input rows."""
    command = 'sh -c \'if [ "$(cat {file})" = "slow" ]; then sleep 0.6; fi; cat {file}\''
    results = tool._run_tool_batch(
        file_ext = "txt",
        command = command,
        scaffold = (),
        code_values = ["slow", "fast", "fast"],
        max_workers = 3,
    )
    assert [result["tool_output"] for result in results] == ["slow", "fast", "fast"]


def test_parallel_batch_mixed_results_keep_per_row_attribution():
    """Failing rows must carry their own error, not a neighbor's."""
    import pandas as pd

    fn = tool._build_tool_validation_function(
        "txt",
        'sh -c \'test "$(cat {file})" = "good"\'',
    )
    out = fn(pd.DataFrame({"code": ["good", "bad", "good", "bad"]}))
    assert list(out["is_valid"]) == [True, False, True, False]
    assert out["error_message"].iloc[0] == ""
    assert out["error_message"].iloc[1] != ""
    assert out["error_message"].iloc[2] == ""
    assert out["error_message"].iloc[3] != ""


def test_parallel_batch_concurrent_timeouts_are_graceful(monkeypatch):
    import time

    monkeypatch.setattr(tool, "_TOOL_RUN_TIMEOUT_SECONDS", 1)
    start = time.monotonic()
    results = tool._run_tool_batch(
        file_ext = "txt",
        command = "sleep 5",
        scaffold = (),
        code_values = ["a", "b", "c"],
        max_workers = 3,
    )
    elapsed = time.monotonic() - start
    assert len(results) == 3
    for result in results:
        assert result["is_valid"] is False
        assert "timed out" in result["error_message"]
    # All three time out at ~1s concurrently; a hang or deadlock would blow this.
    assert elapsed < 4, f"concurrent timeouts took {elapsed:.2f}s"


def test_parallel_batch_workers_capped_to_row_count():
    results = tool._run_tool_batch(
        file_ext = "txt",
        command = "true",
        scaffold = (),
        code_values = ["a", "b"],
        max_workers = 8,
    )
    assert len(results) == 2
    assert all(result["is_valid"] for result in results)


def test_parallel_batch_zero_or_negative_workers_fall_back_serial():
    for workers in (0, -1):
        results = tool._run_tool_batch(
            file_ext = "txt",
            command = "cat {file}",
            scaffold = (),
            code_values = ["a", "b"],
            max_workers = workers,
        )
        assert [result["tool_output"] for result in results] == ["a", "b"]


def test_serial_and_parallel_batches_are_equivalent():
    command = 'sh -c \'test "$(cat {file})" = "good"\''
    code_values = ["good", "bad", "good", "bad", "good"]
    serial = tool._run_tool_batch(
        file_ext = "txt",
        command = command,
        scaffold = (),
        code_values = code_values,
        max_workers = 1,
    )
    parallel = tool._run_tool_batch(
        file_ext = "txt",
        command = command,
        scaffold = (),
        code_values = code_values,
        max_workers = 4,
    )
    assert serial == parallel


def test_cached_tool_callable_is_thread_safe():
    """The lru_cached callable is shared; concurrent calls must not interfere."""
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd

    fn = tool._build_tool_validation_function(
        "txt",
        'sh -c \'test "$(cat {file})" = "good"\'',
    )
    df = pd.DataFrame({"code": ["good", "bad", "good"]})
    with ThreadPoolExecutor(max_workers = 4) as executor:
        outputs = list(executor.map(lambda _: fn(df), range(4)))
    for output in outputs:
        assert list(output["is_valid"]) == [True, False, True]
        assert output["error_message"].iloc[0] == ""
        assert output["error_message"].iloc[1] != ""


def test_tool_callable_file_placeholder_uses_scaffold_source():
    import pandas as pd

    fn = tool._build_tool_validation_function(
        "txt",
        'sh -c \'test "$(cat {file})" = "hello world"\'',
        (("src/check.txt", "{source}"),),
    )
    out = fn(pd.DataFrame({"code": ["hello world", "other"]}))
    assert list(out["is_valid"]) == [True, False]


def test_tool_callable_file_placeholder_path_is_scaffold_path():
    import pandas as pd

    fn = tool._build_tool_validation_function(
        "txt",
        "echo {file}",
        (("src/check.txt", "{source}"),),
    )
    out = fn(pd.DataFrame({"code": ["x"]}))
    assert out["tool_output"].iloc[0].endswith("src/check.txt")


def test_tool_callable_file_placeholder_falls_back_to_main_ext():
    import pandas as pd

    fn = tool._build_tool_validation_function(
        "txt",
        "echo {file}",
        (("notes.txt", "config"),),
    )
    out = fn(pd.DataFrame({"code": ["x"]}))
    assert out["tool_output"].iloc[0].endswith("main.txt")


def test_tool_callable_writes_nested_scaffold_parents():
    import pandas as pd

    fn = tool._build_tool_validation_function(
        "rs",
        "test -f src/main.rs",
        (("Cargo.toml", '[package]\nname = "check"\n'), ("src/main.rs", "{source}")),
    )
    out = fn(pd.DataFrame({"code": ["fn main() {}"]}))
    assert list(out["is_valid"]) == [True]


def test_tool_callable_scaffold_path_escape_is_graceful():
    import pandas as pd

    out = tool._run_tool_single(
        file_ext = "txt",
        command = "true",
        scaffold = (("../escape.txt", "x"),),
        code_value = "hello",
    )
    assert out["is_valid"] is False
    assert "escapes" in out["error_message"]


def test_tool_callable_legacy_go_marker_fails_gracefully_without_scaffold():
    import pandas as pd

    fn = tool._build_tool_validation_function("go", "go vet ./...")
    out = fn(pd.DataFrame({"code": ["package main\n\nfunc main() {}\n"]}))
    assert list(out["is_valid"]) == [False]
    assert out["error_message"].iloc[0] != ""


@pytest.mark.skipif(shutil.which("go") is None, reason = "go toolchain not installed")
def test_go_scaffold_and_vet():
    import pandas as pd

    go_source = (
        "package main\n\n" 'import "fmt"\n\n' "func main() {\n" '\tfmt.Println("hi")\n' "}\n"
    )
    fn = tool._build_tool_validation_function(
        "go",
        "go vet ./...",
        (("go.mod", "module example.com/check\n\ngo 1.21\n"), ("main.go", "{source}")),
    )
    out = fn(pd.DataFrame({"code": [go_source]}))
    assert list(out["is_valid"]) == [True]


@pytest.mark.skipif(shutil.which("go") is None, reason = "go toolchain not installed")
def test_go_vet_direct_file_reference():
    import pandas as pd

    good = "package main\n\nfunc main() {}\n"
    bad = "package main\n\nfunc main() { undefined }\n"
    fn = tool._build_tool_validation_function(
        "go",
        "go vet {file}",
        (("go.mod", "module example.com/check\n\ngo 1.21\n"), ("main.go", "{source}")),
    )
    out = fn(pd.DataFrame({"code": [good, bad]}))
    assert list(out["is_valid"]) == [True, False]
    assert out["error_message"].iloc[1] != ""


@pytest.mark.skipif(shutil.which("cargo") is None, reason = "cargo toolchain not installed")
def test_cargo_scaffold_and_check():
    import pandas as pd

    rs_source = 'fn main() {\n    println!("hi");\n}\n'
    fn = tool._build_tool_validation_function(
        "rs",
        "cargo check",
        (
            ("Cargo.toml", '[package]\nname = "check"\nversion = "0.1.0"\nedition = "2021"\n'),
            ("src/main.rs", "{source}"),
        ),
    )
    out = fn(pd.DataFrame({"code": [rs_source]}))
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
                scaffold = (("go.mod", "module example.com/check\n"), ("main.go", "{source}")),
            )
        ],
    )
    assert len(builder.columns) == 1
    column = builder.columns[0]
    assert isinstance(column, ValidationColumnConfig)
    assert column.validator_type == ValidatorType.LOCAL_CALLABLE
    assert callable(column.validator_params.validation_function)
