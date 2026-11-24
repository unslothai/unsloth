# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""ast.get_source_segment, memoised on the source text and the node's coordinates.

get_source_segment re-splits the whole source on every call, which for the 14k-line
unsloth/models/_utils.py is most of what several tests spend. The slice it returns is a pure
function of (source, lineno, col_offset, end_lineno, end_col_offset), so caching it changes
nothing a caller sees: callers still parse the current source and exec the returned text
afresh, so warnings, compile errors and code objects are exactly what they were.
"""

import ast
from functools import lru_cache
from types import SimpleNamespace


@lru_cache(maxsize = 4096)
def _segment_at(source, lineno, col_offset, end_lineno, end_col_offset):
    location = SimpleNamespace(
        lineno = lineno,
        col_offset = col_offset,
        end_lineno = end_lineno,
        end_col_offset = end_col_offset,
    )
    return ast.get_source_segment(source, location)


def cached_source_segment(source, node):
    """Same result as ast.get_source_segment(source, node) (padded = False)."""
    span = (
        getattr(node, "lineno", None),
        getattr(node, "col_offset", None),
        getattr(node, "end_lineno", None),
        getattr(node, "end_col_offset", None),
    )
    if None in span:
        return None
    return _segment_at(source, *span)
