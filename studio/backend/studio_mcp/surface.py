# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool text for one caller's surface. ``<<tool: shown with it | shown without it>>`` in a docstring,
hint or field description keeps a caller from reading about a tool it was not given."""

from __future__ import annotations

import re
from typing import Any, Collection

_SEGMENT = re.compile(r"<<(\w+): (.*?)(?: \| (.*?))?>>", re.DOTALL)


def render(text: Any, tools: Collection[str]) -> Any:
    if not isinstance(text, str) or "<<" not in text:
        return text
    rendered = _SEGMENT.sub(
        lambda match: match.group(2) if match.group(1) in tools else (match.group(3) or ""),
        text,
    )
    # A dropped segment may leave a doubled or trailing space before punctuation.
    return re.sub(r" {2,}", " ", re.sub(r" +([.,;:)])", r"\1", rendered)).strip()


def render_schema(schema: Any, tools: Collection[str]) -> Any:
    """``render`` every string in a JSON schema, so field descriptions follow the same rule."""
    if isinstance(schema, dict):
        return {key: render_schema(value, tools) for key, value in schema.items()}
    if isinstance(schema, list):
        return [render_schema(value, tools) for value in schema]
    return render(schema, tools)
