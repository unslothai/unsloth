# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fails grouped tests by name when ``--dist loadgroup`` stops grouping, instead of letting them flake.

pytest-xdist puts an ``xdist_group`` test on its group's worker by rewriting ``item._nodeid`` to ``<id>@<group>``.
pytest 9.2 builds ``Item.nodeid`` from a structured id instead, so that write is lost (pytest-xdist 3.8.0): every
grouped test is scheduled like an ungrouped one, and pairs that must share a worker, such as the
``*_reset_between_tests`` guards, fail whenever they land apart."""

from __future__ import annotations

from typing import Any


def ungrouped(config: Any, items: list) -> list[str]:
    """Node ids carrying an ``xdist_group`` mark that loadgroup did not tag; empty outside an xdist loadgroup worker."""
    if not getattr(config.option, "loadgroup", False) or not hasattr(config, "workerinput"):
        return []
    return [
        item.nodeid
        for item in items
        if item.get_closest_marker("xdist_group") and "@" not in item.nodeid
    ]


def check(config: Any, items: list) -> None:
    """Each grouped test that lost its tag fails with the reason; a collection error here is swallowed by xdist."""
    lost = set(ungrouped(config, items))
    if not lost:
        return
    import pytest

    message = (
        "--dist loadgroup did not tag this xdist_group test, so its group can run on different workers. This "
        "pytest / pytest-xdist pair drops the group suffix; keep pytest below the release that breaks it (CI pins "
        "pytest<9.2)."
    )

    def _fail(*args, **kwargs):
        pytest.fail(message, pytrace = False)

    for item in items:
        if item.nodeid in lost:
            item.runtest = _fail
