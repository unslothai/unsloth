# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""ast.get_source_segment, memoised on the source text and the node's coordinates.

get_source_segment re-splits the whole source on every call, which for the 14k-line
unsloth/models/_utils.py is most of what several tests spend. Its result is a pure function of
(source, lineno, col_offset, end_lineno, end_col_offset), so caching it changes nothing a caller
sees: callers still parse the current source and exec the returned text afresh, so warnings,
compile errors and code objects are exactly what they were. If anything has replaced
ast.get_source_segment (a monkeypatch), the replacement is called with the real node, uncached.

Test files load this by path under one module name, so they collect without tests/conftest.py
and share one cache.
"""

import ast
from functools import lru_cache
from types import SimpleNamespace

_STANDARD = ast.get_source_segment


@lru_cache(maxsize = 1024)
def _segment_at(source, lineno, col_offset, end_lineno, end_col_offset):
    location = SimpleNamespace(
        lineno = lineno,
        col_offset = col_offset,
        end_lineno = end_lineno,
        end_col_offset = end_col_offset,
    )
    return _STANDARD(source, location)


def cached_source_segment(source, node):
    """Same result as ast.get_source_segment(source, node) (padded = False)."""
    if ast.get_source_segment is not _STANDARD:
        return ast.get_source_segment(source, node)
    span = (
        getattr(node, "lineno", None),
        getattr(node, "col_offset", None),
        getattr(node, "end_lineno", None),
        getattr(node, "end_col_offset", None),
    )
    if None in span:
        return _STANDARD(source, node)
    return _segment_at(source, *span)
