# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A report's URLs are held to the source catalog whatever case their scheme is written in:
browsers treat `HTTPS://` as https, so the check has to as well."""

import pytest

from core.research.citations import _validate_report


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (
            '```mermaid\nflowchart TD\n  A@{ img: "HTTPS://nope.example/x?leak=1" }\n```',
            '```mermaid\nflowchart TD\n  A@{ img: " }\n```',
        ),
        ("See HTTPS://nope.example/raw for more.", "See  for more."),
        ("See <Https://nope.example/auto> now.", "See  now."),
        ("See [the page](HTTP://nope.example/link) now.", "See the page now."),
    ],
)
def test_uncatalogued_urls_are_dropped_in_any_case(report, expected):
    assert _validate_report(report, [], []) == expected


def test_a_catalogued_source_is_still_cited():
    sources = [{"url": "https://unsloth.ai/docs", "title": "Docs"}]
    assert (
        _validate_report("Read https://unsloth.ai/docs today.", sources, [])
        == "Read [Docs](https://unsloth.ai/docs) today."
    )


@pytest.mark.parametrize(
    "report",
    ["Read HTTPS://unsloth.ai/docs today.", "Read [the docs](Https://unsloth.ai/docs) today."],
)
def test_a_catalogued_source_with_an_uppercase_scheme_is_cited(report):
    sources = [{"url": "https://unsloth.ai/docs", "title": "Docs"}]
    assert _validate_report(report, sources, []) == "Read [Docs](https://unsloth.ai/docs) today."
