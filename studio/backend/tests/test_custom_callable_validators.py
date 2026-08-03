# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the advanced custom validator path (user-supplied Python source).

This module intentionally does NOT exercise the OXC/tool validator machinery.
It is the standalone regression net for ``custom_callable_validators.py``.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("structlog")
pytest.importorskip("pandas")


def _load_module():
    backend_root = Path(__file__).resolve().parent.parent
    module_path = backend_root / "core" / "data_recipe" / "custom_callable_validators.py"
    spec = importlib.util.spec_from_file_location(
        "custom_callable_validators_under_test",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module.__name__] = module
    spec.loader.exec_module(module)
    return module


custom = _load_module()

VALID_SOURCE = (
    "def validate(df):\n"
    "    df = df.copy()\n"
    "    df['is_valid'] = df.iloc[:, 0].str.len() > 3\n"
    "    df['error_message'] = df['is_valid'].map({True: '', False: 'too short'})\n"
    "    return df"
)


def _custom_column(*, name: str = "custom_check", source: str) -> dict:
    return {
        "column_type": "validation",
        "name": name,
        "drop": False,
        "target_columns": ["code"],
        "validator_type": "local_callable",
        "validator_params": {
            "validation_function": (
                f"{custom.CUSTOM_VALIDATION_FN_MARKER}:"
                f"{custom.encode_validation_source(source)}"
            )
        },
        "batch_size": 10,
    }


def test_source_round_trip_preserves_unicode_and_newlines():
    source = (
        "def validate(df):\n"
        "    import pandas as pd\n"
        "    df['is_valid'] = df['code'].str.contains('好')\n"
        "    return df\n"
    )
    encoded = custom.encode_validation_source(source)
    assert ":" not in encoded
    assert custom.decode_validation_source(encoded) == source


def test_decode_rejects_invalid_base64():
    assert custom.decode_validation_source("!!not base64!!") == ""


def test_decode_rejects_oversized_marker():
    big = "A" * (custom.CUSTOM_MARKER_MAX_CHARS + 1)
    assert custom.decode_validation_source(big) == ""
    assert custom.decode_validation_source("A" * 100) != ""


def test_split_skips_oversized_source():
    recipe = {
        "columns": [
            _custom_column(
                name = "too_big",
                source = "x" * (custom.CUSTOM_SOURCE_MAX_CHARS + 1),
            )
        ]
    }
    sanitized, specs = custom.split_custom_callable_validators(recipe)
    assert specs == []
    assert len(sanitized["columns"]) == 1


def test_parse_batch_size_is_clamped():
    assert custom._parse_batch_size(100000) == custom.BATCH_SIZE_MAX
    assert custom._parse_batch_size(512) == 512
    assert custom._parse_batch_size(0) == 10
    assert custom._parse_batch_size("nope") == 10


def test_split_extracts_custom_and_leaves_other_columns():
    recipe = {
        "columns": [
            _custom_column(name = "custom_one", source = VALID_SOURCE),
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
    sanitized, specs = custom.split_custom_callable_validators(recipe)
    assert len(specs) == 1
    assert specs[0].name == "custom_one"
    assert specs[0].source == VALID_SOURCE
    assert specs[0].batch_size == 10
    assert [column["name"] for column in sanitized["columns"]] == ["oxc_one", "python_one"]


def test_split_skips_malformed_marker():
    recipe = {
        "columns": [
            {
                "column_type": "validation",
                "name": "bad",
                "target_columns": ["code"],
                "validator_type": "local_callable",
                "validator_params": {"validation_function": "unsloth_custom_validator:!!bad!!"},
                "batch_size": 10,
            }
        ]
    }
    sanitized, specs = custom.split_custom_callable_validators(recipe)
    assert specs == []
    assert len(sanitized["columns"]) == 1


def test_callable_contract():
    import pandas as pd

    fn = custom._build_custom_validation_function(VALID_SOURCE)
    out = fn(pd.DataFrame({"code": ["abc", "abcdef"]}))
    assert list(out["is_valid"]) == [False, True]
    assert list(out["error_message"]) == ["too short", ""]


def test_callable_pre_injected_names_need_no_imports():
    import pandas as pd

    source = (
        "def validate(df):\n"
        "    subprocess.run(['true'])\n"
        "    with tempfile.TemporaryDirectory() as raw:\n"
        "        marker = Path(raw) / 'marker.txt'\n"
        "        marker.write_text('ok')\n"
        "        content = marker.read_text()\n"
        "    return pd.DataFrame({'is_valid': [content == 'ok'] * len(df)})\n"
    )
    fn = custom._build_custom_validation_function(source)
    out = fn(pd.DataFrame({"code": ["a", "b"]}))
    assert list(out["is_valid"]) == [True, True]


def test_callable_swallows_user_exception():
    """User exception details stay server-side: rows get a generic message."""
    import structlog

    import pandas as pd

    fn = custom._build_custom_validation_function(
        "def validate(df):\n    raise RuntimeError('/home/user/.ssh/id_rsa boom')\n"
    )
    with structlog.testing.capture_logs() as events:
        out = fn(pd.DataFrame({"code": ["a", "b"]}))
    assert list(out["is_valid"]) == [False, False]
    assert out["error_message"].iloc[0] == "Custom validator raised an error."
    assert "/home/user" not in out["error_message"].iloc[0]
    # The full exception must still reach the server log for debugging.
    assert any(event.get("event") == "Custom validator raised during execution" for event in events)


def test_callable_rejects_missing_is_valid_column():
    import pandas as pd

    fn = custom._build_custom_validation_function(
        "def validate(df):\n    import pandas as pd\n    return pd.DataFrame({'score': [1, 2]})\n"
    )
    out = fn(pd.DataFrame({"code": ["a", "b"]}))
    assert list(out["is_valid"]) == [False, False]
    assert "is_valid" in out["error_message"].iloc[0]


def test_callable_rejects_mismatched_row_count():
    """A validate that returns fewer/more rows than the input must degrade to
    per-row failures instead of misattributing results across rows."""
    import pandas as pd

    fewer = custom._build_custom_validation_function(
        "def validate(df):\n"
        "    import pandas as pd\n"
        "    return pd.DataFrame({'is_valid': [True]})\n"
    )
    out = fewer(pd.DataFrame({"code": ["a", "b", "c"]}))
    assert len(out) == 3
    assert list(out["is_valid"]) == [False, False, False]
    assert "1 rows for 3 input rows" in out["error_message"].iloc[0]

    more = custom._build_custom_validation_function(
        "def validate(df):\n"
        "    import pandas as pd\n"
        "    return pd.DataFrame({'is_valid': [True, True, True, True]})\n"
    )
    out = more(pd.DataFrame({"code": ["a", "b"]}))
    assert len(out) == 2
    assert list(out["is_valid"]) == [False, False]
    assert "4 rows for 2 input rows" in out["error_message"].iloc[0]

    exact = custom._build_custom_validation_function(
        "def validate(df):\n"
        "    import pandas as pd\n"
        "    return pd.DataFrame({'is_valid': [True, False]})\n"
    )
    out = exact(pd.DataFrame({"code": ["a", "b"]}))
    assert list(out["is_valid"]) == [True, False]


def test_callable_empty_df():
    import pandas as pd

    fn = custom._build_custom_validation_function(VALID_SOURCE)
    out = fn(pd.DataFrame({"code": []}))
    assert "is_valid" in out.columns
    assert len(out) == 0


def test_build_rejects_missing_validate():
    with pytest.raises(ValueError, match = "validate"):
        custom._build_custom_validation_function("x = 1")


def test_build_rejects_non_callable_validate():
    with pytest.raises(ValueError, match = "validate"):
        custom._build_custom_validation_function("validate = 42")


def test_build_rejects_syntax_error():
    with pytest.raises(ValueError, match = "compile"):
        custom._build_custom_validation_function("def validate(df:")


def test_full_loop_without_data_designer():
    import pandas as pd

    recipe = {"columns": [_custom_column(name = "cc", source = VALID_SOURCE)]}
    sanitized, specs = custom.split_custom_callable_validators(recipe)
    assert len(specs) == 1
    assert sanitized["columns"] == []
    fn = custom._build_custom_validation_function(specs[0].source)
    out = fn(pd.DataFrame({"code": ["a", "abcdef"]}))
    assert list(out["is_valid"]) == [False, True]


def test_register_creates_local_callable_column():
    data_designer = pytest.importorskip("data_designer")
    from data_designer.config.column_configs import ValidationColumnConfig
    from data_designer.config.validator_params import ValidatorType

    class _StubBuilder:
        def __init__(self) -> None:
            self.columns = []

        def add_column(self, column) -> None:
            self.columns.append(column)

    builder = _StubBuilder()
    custom.register_custom_callable_validators(
        builder = builder,
        specs = [
            custom.CustomCallableValidatorSpec(
                name = "cc",
                drop = False,
                target_columns = ["code"],
                batch_size = 5,
                source = VALID_SOURCE,
            )
        ],
    )
    assert len(builder.columns) == 1
    column = builder.columns[0]
    assert isinstance(column, ValidationColumnConfig)
    assert column.validator_type == ValidatorType.LOCAL_CALLABLE
    assert callable(column.validator_params.validation_function)
