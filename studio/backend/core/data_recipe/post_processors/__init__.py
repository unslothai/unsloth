# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio-owned post-processors.

`data_designer.config.processors.ProcessorType` is a fixed enum we cannot
extend, so we run our own processors out-of-band: in `service.py` we strip
studio-owned types from the config we hand to `data_designer`, then in the
worker we call `apply_studio_post_processors` on the output parquet after
`designer.create()` returns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .json_document_score import run_json_document_score

STUDIO_PROCESSOR_TYPES: frozenset[str] = frozenset({"json_document_score"})


def is_studio_processor_type(processor_type: Any) -> bool:
    return isinstance(processor_type, str) and processor_type in STUDIO_PROCESSOR_TYPES


def apply_studio_post_processors(
    *,
    base_dataset_path: Path,
    processors: list[dict[str, Any]],
) -> None:
    """Run every studio-owned processor against the recipe's parquet output.

    Non-studio processor entries are ignored (they were handled upstream by
    `data_designer`). Processors run in declaration order; later processors
    see earlier processors' output.
    """
    parquet_dir = base_dataset_path / "parquet-files"
    for processor in processors:
        if not isinstance(processor, dict):
            continue
        processor_type = processor.get("processor_type")
        if not is_studio_processor_type(processor_type):
            continue
        if processor_type == "json_document_score":
            run_json_document_score(
                parquet_dir,
                prediction_column=str(processor.get("prediction_column", "")),
                reference_column=str(processor.get("reference_column", "")),
                schema=processor.get("schema"),
                default_comparator=str(processor.get("default_comparator", "string")),
                score_column=str(processor.get("score_column", "doc_score")),
                breakdown_column=(
                    processor.get("breakdown_column")
                    if processor.get("breakdown_column")
                    else None
                ),
            )
