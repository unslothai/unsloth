# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Personal inventory details, kept separate from model identity and Hub capability tags."""

from typing import Annotated, Optional

from pydantic import BaseModel, Field, StringConstraints, field_validator


class ModelAnnotation(BaseModel):
    """User-visible details only; changing them never renames a repository or file."""

    display_name: Annotated[str, StringConstraints(strip_whitespace = True, max_length = 160)] = ""
    labels: list[
        Annotated[str, StringConstraints(strip_whitespace = True, min_length = 1, max_length = 64)]
    ] = Field(default_factory = list, max_length = 20)
    notes: str = Field(default = "", max_length = 8000)

    @field_validator("labels")
    @classmethod
    def unique_labels(cls, labels: list[str]) -> list[str]:
        """Preserve the user's order and spelling while removing repeated labels."""
        return list(dict.fromkeys(labels))


class ModelAnnotationUpdate(BaseModel):
    """Replace one inventory row's personal details in the acting account."""

    model_id: Annotated[
        str, StringConstraints(strip_whitespace = True, min_length = 1, max_length = 4096)
    ]
    annotation: ModelAnnotation


class ModelAnnotationsResponse(BaseModel):
    annotations: dict[str, ModelAnnotation]


class ModelAnnotationResponse(BaseModel):
    annotation: Optional[ModelAnnotation]
