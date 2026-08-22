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

# [text](target), ignoring images and reference-style definitions.
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)")


def _targets(text: str) -> list[str]:
    return LINK.findall(text)


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
