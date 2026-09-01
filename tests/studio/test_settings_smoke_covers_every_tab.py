# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The settings smoke must drive every tab the dialog actually has.

`playwright_settings_tabs.py` keeps its own list of tab ids. A settings page
added to `settings-dialog.tsx` without touching that list ships untested, and
nothing says so: the smoke keeps passing, because it still drives the twelve it
knows about.

That is not hypothetical. The keyboard-shortcuts page took the dialog to
thirteen tabs while the smoke's list stayed at twelve, so the new page had no
browser coverage at all. The same drift also broke the smoke's `nav != 12`
assertion, which then failed as "blocking the data panel took the dialog down"
while reporting `dialog: True` -- a stale constant reading like an
error-handling regression.

Both halves are pinned here: the lists agree, and the smoke no longer hardcodes
a nav size that any new page invalidates.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "tests" / "studio" / "playwright_settings_tabs.py"
DIALOG = ROOT / "studio" / "frontend" / "src" / "features" / "settings" / "settings-dialog.tsx"


def _smoke_tabs() -> list[str]:
    """The TABS list literal the smoke drives."""
    text = SMOKE.read_text(encoding = "utf-8")
    block = re.search(r"^TABS = \[(.*?)^\]", text, re.S | re.M)
    assert block, "could not find the TABS list in playwright_settings_tabs.py"
    return re.findall(r'"([a-z0-9-]+)"', block.group(1))


def _dialog_tabs() -> list[str]:
    """The tab ids the dialog renders, in declaration order."""
    text = DIALOG.read_text(encoding = "utf-8")
    # Scope to the tab array first:
    block = re.search(r"SETTINGS_TABS[^=]*=\s*\[(.*?)^\]", text, re.S | re.M)
    region = block.group(1) if block else text
    return re.findall(r'\bid:\s*"([a-z0-9-]+)"', region)


def test_the_smoke_drives_every_tab_the_dialog_defines() -> None:
    smoke, dialog = set(_smoke_tabs()), set(_dialog_tabs())
    missing = sorted(dialog - smoke)
    assert not missing, (
        f"settings-dialog.tsx defines tabs the settings smoke never opens: {missing}. "
        "Add them to TABS in tests/studio/playwright_settings_tabs.py, or the page "
        "ships with no browser coverage and the smoke still goes green."
    )


def test_the_smoke_does_not_drive_a_tab_that_no_longer_exists() -> None:
    smoke, dialog = set(_smoke_tabs()), set(_dialog_tabs())
    stale = sorted(smoke - dialog)
    assert not stale, (
        f"the settings smoke drives tabs the dialog no longer defines: {stale}. "
        "It will fail looking for a selector that cannot appear."
    )


def test_the_chunk_fail_tab_is_one_the_dialog_has() -> None:
    """The blocked-panel run names its tab in the workflow, not in the smoke.

    `CHUNK_FAIL` defaults to the empty string, so the tab comes from
    `PW_CHUNK_FAIL` in studio-frontend-ci.yml. Renaming that tab would leave the
    run blocking nothing at all, and the smoke would still report PASS on a
    panel it never broke.
    """
    workflow = (ROOT / ".github" / "workflows" / "studio-frontend-ci.yml").read_text(
        encoding = "utf-8"
    )
    targets = re.findall(r"PW_CHUNK_FAIL:\s*([a-z0-9-]+)", workflow)
    assert targets, "no PW_CHUNK_FAIL in studio-frontend-ci.yml; the blocked-panel run is not wired"
    dialog = set(_dialog_tabs())
    unknown = sorted(t for t in set(targets) if t not in dialog)
    assert not unknown, (
        f"studio-frontend-ci.yml blocks settings tabs the dialog does not define: "
        f"{unknown}. The run would block nothing and assert against an unbroken panel."
    )


def test_the_nav_assertion_is_not_hardcoded() -> None:
    """A literal count here is what made a new settings page look like a regression."""
    text = SMOKE.read_text(encoding = "utf-8")
    assert 'state["nav"] != nav_before' in text, (
        "the blocked-panel check must compare the nav against the size measured "
        "before blocking, not a literal. A hardcoded count fails the moment a "
        "settings page is added, and reports it as the dialog being taken down."
    )
    assert not re.search(
        r'state\["nav"\]\s*!=\s*\d+', text
    ), "found a hardcoded nav count in the blocked-panel check"


def test_the_guard_reads_both_files() -> None:
    """Neither list may be empty, or every assertion above passes vacuously."""
    assert len(_smoke_tabs()) >= 5, _smoke_tabs()
    assert len(_dialog_tabs()) >= 5, _dialog_tabs()
