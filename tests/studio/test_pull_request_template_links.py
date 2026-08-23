# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every link in the PR template has to survive being pasted into a pull request body.

GitHub copies the template verbatim into the PR description, rendered at
`/<owner>/<repo>/pull/<number>`, so a DOCUMENT-relative target resolves from THAT path:
`../CONTRIBUTING.md` becomes `/<owner>/<repo>/CONTRIBUTING.md`, which 404s. A ROOT-relative
target is safe, since the leading slash discards the PR path and resolves against the host.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parents[2] / ".github" / "PULL_REQUEST_TEMPLATE.md"

# [text](target), ignoring images. The <> wrapper is not part of the URI (CommonMark 0.31.2
# section 6.3), so keeping the brackets would read an absolute `<https://...>` as relative.
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(<?([^)<>\s]+)>?")
# [label]: target -- reference definitions resolve against the same wrong base, so a guard reading
# only the inline form would pass a template whose links all 404. The optional <> wrapper and the
# trailing title must not end up in the target.
REF_DEF = re.compile(r"^[ ]{0,3}\[[^\]]+\]:[ \t]*<?([^>\s]+)>?", re.MULTILINE)


def _targets(text: str) -> list[str]:
    return LINK.findall(text) + REF_DEF.findall(text)


def _resolves_off_the_pr_path(target: str) -> bool:
    """True when GitHub would resolve `target` from `/<owner>/<repo>/pull/<number>`.

    A leading slash is NOT such a target: it resolves against the host, landing on the same
    page as the absolute URL.
    """
    if target.startswith(("http://", "https://", "mailto:", "#", "/")):
        return False
    return True


def test_template_exists() -> None:
    assert TEMPLATE.is_file(), f"{TEMPLATE} is missing"


def test_no_relative_link_targets_in_the_pr_template() -> None:
    relative = [
        target
        for target in _targets(TEMPLATE.read_text(encoding = "utf-8"))
        if _resolves_off_the_pr_path(target)
    ]
    assert not relative, (
        f"{len(relative)} relative link target(s) in .github/PULL_REQUEST_TEMPLATE.md: "
        f"{relative}. The template is rendered inside a PR body at /owner/repo/pull/N, not "
        "from .github/, so a document-relative target resolves off the repository tree and "
        "404s. Use the absolute https://github.com/... blob URL, or a root-relative "
        "/owner/repo/blob/... target, either of which ignores the PR path."
    )


@pytest.mark.parametrize(
    "target,relative",
    [
        ("../CONTRIBUTING.md", True),
        ("CONTRIBUTING.md", True),
        ("./docs/CONTRIBUTING.md", True),
        # Root-relative: the leading slash drops the PR path, so rejecting it would fail CI
        # on a valid link.
        ("/unslothai/unsloth/blob/main/CONTRIBUTING.md", False),
        ("https://github.com/unslothai/unsloth/blob/main/CONTRIBUTING.md", False),
        ("#testing", False),
    ],
)
def test_the_guard_classifies_targets_the_way_github_resolves_them(
    target: str, relative: bool
) -> None:
    """The guard itself, against the shapes it has to separate.

    Without this the test above also passes for a template with no links at all.
    """
    found = _targets(f"See [CONTRIBUTING.md]({target}) for the full rule.")
    assert found == [target]
    assert _resolves_off_the_pr_path(target) is relative


@pytest.mark.parametrize(
    "target",
    [
        "https://github.com/unslothai/unsloth/blob/main/CONTRIBUTING.md",
        "/unslothai/unsloth/blob/main/CONTRIBUTING.md",
        "../CONTRIBUTING.md",
    ],
)
def test_the_scan_unwraps_angle_bracketed_inline_destinations(target: str) -> None:
    """`[text](<target>)` is the same link as `[text](target)`: the brackets wrap the
    destination and are not part of the URI.
    """
    found = _targets(f"See [CONTRIBUTING.md](<{target}>) for the full rule.")
    assert found == [target]
    assert _resolves_off_the_pr_path(target) is _resolves_off_the_pr_path(found[0])


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
    """A reference-style link resolves against the same wrong base as an inline one, so a
    template that moved to `[guidelines][contrib]` would have passed while every rendered
    link 404'd.
    """
    relative = [
        t for t in _targets(body) if not t.startswith(("http://", "https://", "mailto:", "#"))
    ]
    assert bool(relative) is caught, f"targets found: {_targets(body)}"
