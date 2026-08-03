# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the per-run local-execution consent attestation gate."""

import pytest

pytest.importorskip("structlog")
pytest.importorskip("fastapi")

from core.data_recipe.service import (  # noqa: E402  (backend root on sys.path via conftest)
    assert_local_execution_consent,
    recipe_requires_local_execution_consent,
)

TOOL_MARKER = "unsloth_tool_validator:eyJleHQiOiJ0eHQiLCJjb21tYW5kIjoiZWNobyB7ZmlsZX0ifQ"
CUSTOM_MARKER = "unsloth_custom_validator:ZGVmIHZhbGlkYXRlKGRmKTogcmV0dXJuIGRm"


def _validation_column(validation_function: str) -> dict:
    return {
        "column_type": "validation",
        "name": "check",
        "target_columns": ["code"],
        "validator_type": "local_callable",
        "validator_params": {"validation_function": validation_function},
        "batch_size": 10,
    }


def test_requires_consent_for_tool_and_custom_markers():
    assert (
        recipe_requires_local_execution_consent({"columns": [_validation_column(TOOL_MARKER)]})
        is True
    )
    assert (
        recipe_requires_local_execution_consent({"columns": [_validation_column(CUSTOM_MARKER)]})
        is True
    )


def test_no_consent_for_code_oxc_or_other_columns():
    code_column = {
        "column_type": "validation",
        "name": "code_check",
        "target_columns": ["code"],
        "validator_type": "code",
        "validator_params": {"code_lang": "python"},
        "batch_size": 10,
    }
    oxc_column = _validation_column("unsloth_oxc_validator:javascript:syntax:auto")
    llm_column = {
        "column_type": "llm",
        "name": "code_gen",
        "llm_type": "code",
        "output_format": "python",
    }
    recipe = {"columns": [code_column, oxc_column, llm_column]}
    assert recipe_requires_local_execution_consent(recipe) is False


def test_missing_or_false_consent_raises():
    recipe = {"columns": [_validation_column(TOOL_MARKER)]}
    with pytest.raises(ValueError, match = "local_execution_consent"):
        assert_local_execution_consent(recipe, None)
    with pytest.raises(ValueError, match = "local_execution_consent"):
        assert_local_execution_consent(recipe, {})
    with pytest.raises(ValueError, match = "local_execution_consent"):
        assert_local_execution_consent(recipe, {"local_execution_consent": False})


def test_true_consent_passes():
    recipe = {"columns": [_validation_column(CUSTOM_MARKER)]}
    assert_local_execution_consent(recipe, {"local_execution_consent": True})


def test_no_consent_needed_for_recipes_without_markers():
    recipe = {"columns": []}
    assert_local_execution_consent(recipe, {})
    assert_local_execution_consent(recipe, None)


def test_consent_is_never_read_from_the_recipe_itself():
    """A recipe dict carrying its own consent key must still be rejected:
    attestation has to come from the run section of the request."""
    recipe = {
        "columns": [_validation_column(TOOL_MARKER)],
        "local_execution_consent": True,
    }
    with pytest.raises(ValueError, match = "local_execution_consent"):
        assert_local_execution_consent(recipe, None)
