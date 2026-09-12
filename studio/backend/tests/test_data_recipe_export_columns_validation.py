# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.data_recipe.export_columns import (
    column_names_after_drop_processors,
    filter_studio_validation_violations,
    recipe_would_export_columns,
)
from core.data_recipe.service import validate_recipe


@pytest.fixture(name = "local_seed_parquet")
def fixture_local_seed_parquet() -> Path:
    tmpdir = Path(tempfile.mkdtemp())
    data_file = tmpdir / "data.parquet"
    pq.write_table(
        pa.table(
            {
                "instruction": ["a"],
                "input": ["b"],
                "output": ["c"],
            }
        ),
        data_file,
    )
    return data_file


def _base_recipe(data_file: Path) -> dict:
    return {
        "model_providers": [
            {
                "name": "p",
                "endpoint": "http://127.0.0.1:9",
                "provider_type": "openai",
            }
        ],
        "model_configs": [
            {
                "alias": "m",
                "model": "x",
                "provider": "p",
                "inference_parameters": {},
            }
        ],
        "seed_config": {
            "source": {"seed_type": "local", "path": str(data_file)},
            "sampling_strategy": "ordered",
            "selection_strategy": {"start": 0, "end": 1},
        },
    }


def test_column_names_after_drop_processors_respects_explicit_names() -> None:
    from data_designer.config.processors import ProcessorType  # pyright: ignore[reportMissingImports]

    processors = [
        type(
            "Drop",
            (),
            {
                "processor_type": ProcessorType.DROP_COLUMNS,
                "column_names": ["input", "missing"],
            },
        )()
    ]
    remaining = column_names_after_drop_processors(
        ["instruction", "input", "output"],
        processors,
    )
    assert remaining == {"instruction", "output"}


def test_recipe_would_export_columns_counts_seed_survivors(local_seed_parquet: Path) -> None:
    from data_designer.engine.compiler import (  # pyright: ignore[reportMissingImports]
        _add_internal_row_id_column_if_needed,
        _resolve_and_add_seed_columns,
    )
    from core.data_recipe.service import build_config_builder, create_data_designer

    recipe = {
        **_base_recipe(local_seed_parquet),
        "columns": [
            {
                "column_type": "llm-text",
                "name": "gen",
                "drop": True,
                "model_alias": "m",
                "prompt": "rewrite {{ output }}",
            }
        ],
        "processors": [
            {
                "processor_type": "drop_columns",
                "name": "drop_seed_columns",
                "column_names": ["output"],
            }
        ],
    }
    builder = build_config_builder(recipe)
    designer = create_data_designer(recipe)
    resource_provider = designer._create_resource_provider(
        "validate-configuration",
        builder,
    )
    config = builder.build()
    _resolve_and_add_seed_columns(config, resource_provider.seed_reader)
    _add_internal_row_id_column_if_needed(config)

    assert recipe_would_export_columns(config.columns, config.processors) is True


def test_validate_recipe_allows_seed_export_when_llm_marked_drop(local_seed_parquet: Path) -> None:
    recipe = {
        **_base_recipe(local_seed_parquet),
        "columns": [
            {
                "column_type": "llm-text",
                "name": "generated_instruction",
                "drop": True,
                "model_alias": "m",
                "prompt": "Write instruction for {{ output }}",
            }
        ],
        "processors": [
            {
                "processor_type": "drop_columns",
                "name": "drop_seed_columns",
                "column_names": ["output"],
            }
        ],
    }
    validate_recipe(recipe)


def test_filter_studio_validation_violations_still_fails_when_nothing_exports(
    local_seed_parquet: Path,
) -> None:
    from data_designer.engine.compiler import (  # pyright: ignore[reportMissingImports]
        _add_internal_row_id_column_if_needed,
        _get_allowed_references,
        _resolve_and_add_seed_columns,
    )
    from data_designer.engine.validation import (  # pyright: ignore[reportMissingImports]
        ViolationLevel,
        ViolationType,
        validate_data_designer_config,
    )
    from core.data_recipe.service import build_config_builder, create_data_designer

    recipe = {
        **_base_recipe(local_seed_parquet),
        "columns": [
            {
                "column_type": "llm-text",
                "name": "gen",
                "drop": True,
                "model_alias": "m",
                "prompt": "x",
            }
        ],
        "processors": [
            {
                "processor_type": "drop_columns",
                "name": "drop_all",
                "column_names": ["instruction", "input", "output"],
            }
        ],
    }
    builder = build_config_builder(recipe)
    designer = create_data_designer(recipe)
    resource_provider = designer._create_resource_provider(
        "validate-configuration",
        builder,
    )
    config = builder.build()
    _resolve_and_add_seed_columns(config, resource_provider.seed_reader)
    _add_internal_row_id_column_if_needed(config)
    violations = validate_data_designer_config(
        columns = config.columns,
        processor_configs = config.processors or [],
        allowed_references = _get_allowed_references(config),
    )
    filtered = filter_studio_validation_violations(
        violations,
        columns = config.columns,
        processor_configs = config.processors or [],
    )
    assert any(
        violation.type == ViolationType.ALL_COLUMNS_DROPPED
        and violation.level == ViolationLevel.ERROR
        for violation in filtered
    )
