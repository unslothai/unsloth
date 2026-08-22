# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every link in the PR template has to survive being pasted into a pull request body.

The template is not rendered from `.github/`. GitHub copies it verbatim into the PR
description, which is rendered at `/<owner>/<repo>/pull/<number>`, so a relative target is
resolved from THAT path. `../CONTRIBUTING.md` becomes `/<owner>/<repo>/CONTRIBUTING.md`,
which is not a repository route and 404s for every contributor who clicks it. Only absolute
URLs, or in-page anchors, are safe here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parents[2] / ".github" / "PULL_REQUEST_TEMPLATE.md"

# [text](target), ignoring images.
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)")
# [label]: target -- the reference-style definition. It carries a target exactly as an inline link
# does, and resolves against the same wrong base, so a guard that reads only the inline form would
# pass a template whose links all 404. The optional <> wrapper and the trailing title are both
# standard and must not end up in the target.
REF_DEF = re.compile(r"^[ ]{0,3}\[[^\]]+\]:[ \t]*<?([^>\s]+)>?", re.MULTILINE)


def _targets(text: str) -> list[str]:
    return LINK.findall(text) + REF_DEF.findall(text)


def test_template_exists() -> None:
    assert TEMPLATE.is_file(), f"{TEMPLATE} is missing"


def test_no_relative_link_targets_in_the_pr_template() -> None:
    relative = [
        target
        for target in _targets(TEMPLATE.read_text(encoding = "utf-8"))
        if not target.startswith(("http://", "https://", "mailto:", "#"))
    ]
    assert not relative, (
        f"{len(relative)} relative link target(s) in .github/PULL_REQUEST_TEMPLATE.md: "
        f"{relative}. The template is rendered inside a PR body at /owner/repo/pull/N, not "
        "from .github/, so a relative target resolves off the repository tree and 404s. Use "
        "the absolute https://github.com/... blob URL instead."
    )


@pytest.mark.parametrize(
    "target,relative",
    [
        ("../CONTRIBUTING.md", True),
        ("CONTRIBUTING.md", True),
        ("/unslothai/unsloth/blob/main/CONTRIBUTING.md", True),
        ("https://github.com/unslothai/unsloth/blob/main/CONTRIBUTING.md", False),
        ("#testing", False),
    ],
)
def test_the_guard_classifies_targets_the_way_github_resolves_them(
    target: str, relative: bool
) -> None:
    """The guard itself, against the shapes it has to separate.

    Without this the test above passes for a template with no links at all, which is exactly
    the state it is supposed to catch on the way back.
    """
    found = _targets(f"See [CONTRIBUTING.md]({target}) for the full rule.")
    assert found == [target]
    is_relative = not target.startswith(("http://", "https://", "mailto:", "#"))
    assert is_relative is relative


@pytest.mark.parametrize(
    "body, caught",
    [
        ("See [guidelines][contrib].\n\n[contrib]: ../CONTRIBUTING.md\n", True),
        ("See [guidelines][contrib].\n\n[contrib]: <../CONTRIBUTING.md>\n", True),
        ('See [g][c].\n\n[c]: ../CONTRIBUTING.md "Contributing"\n', True),
        ("[c]: https://github.com/unslothai/unsloth/blob/main/CONTRIBUTING.md\n", False),
        ("See [inline](../CONTRIBUTING.md).\n", True),
        ("![shot](../docs/a.png)\n", False),
    ],
)
def test_the_scan_reads_reference_definitions_as_well_as_inline_links(body, caught) -> None:
    """A reference-style link resolves against the same wrong base as an inline one.

    Without this the guard read only `[text](target)`, so a template that moved to
    `[guidelines][contrib]` would have passed while every rendered link still 404'd.
    """
    relative = [
        t for t in _targets(body) if not t.startswith(("http://", "https://", "mailto:", "#"))
    ]
    assert bool(relative) is caught, f"targets found: {_targets(body)}"
