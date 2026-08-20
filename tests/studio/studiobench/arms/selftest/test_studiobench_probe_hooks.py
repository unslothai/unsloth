# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two halves of the external-probe channel, pinned so neither can rot on its own.

A probe needs a way IN (`SBENCH_EXTRA_INIT_SCRIPT`, appended after the scene scripts) and a way
OUT (`SBENCH_PAGE_CONSOLE`, a prefix filter on the page's console). Neither is exercised by a
normal run, because both are off unless the environment asks for them, which is exactly the
property that makes them safe and exactly the property that lets them break unnoticed: delete the
console filter and the probe still installs, still samples, and reports nothing, which reads as
"the arm did not fire".

The off-by-default half is the one worth stating. A hook that changed the page even slightly when
unset would make every scored run a probe run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

STUDIOBENCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDIOBENCH.parent))

PROBE = STUDIOBENCH / "arms" / "content_visibility_probe.js"
MAIN = STUDIOBENCH / "__main__.py"


@pytest.fixture(scope = "module")
def main_src() -> str:
    return MAIN.read_text(encoding = "utf-8")


@pytest.fixture(scope = "module")
def probe_src() -> str:
    return PROBE.read_text(encoding = "utf-8")


def test_both_hooks_are_read_from_the_environment(main_src: str):
    for var in ("SBENCH_EXTRA_INIT_SCRIPT", "SBENCH_PAGE_CONSOLE"):
        assert f'os.environ.get("{var}")' in main_src, (
            f"{var} is no longer read by __main__; a probe that relies on it will install and "
            "report nothing, which is indistinguishable from an arm that did not fire"
        )


def test_the_hooks_are_off_unless_asked_for(main_src: str, monkeypatch):
    """Unset means NOTHING is appended and no listener is attached."""

    monkeypatch.delenv("SBENCH_EXTRA_INIT_SCRIPT", raising = False)
    monkeypatch.delenv("SBENCH_PAGE_CONSOLE", raising = False)
    import os

    assert os.environ.get("SBENCH_EXTRA_INIT_SCRIPT") is None
    assert os.environ.get("SBENCH_PAGE_CONSOLE") is None
    # Both call sites are guarded by a plain truthiness check on the variable, so an unset
    # variable cannot reach `add_init_script` or `page.on`. Pinned as source, because the failure
    # this guards is a refactor that hoists either call out of its `if`.
    for guarded in ("if extra_init:", "if console_prefix:"):
        assert guarded in main_src, f"{guarded!r} is gone; the hook may no longer be opt-in"


def test_the_probe_exists_and_is_not_a_stub(probe_src: str):
    assert PROBE.is_file()
    assert len(probe_src) > 4_000


def test_the_probe_prefix_is_the_one_the_console_filter_expects(probe_src: str):
    """The filter is an exact prefix match, so a probe with a different prefix is silent."""

    assert 'var PREFIX = "CVPOT ";' in probe_src
    assert "CVPOT " in (STUDIOBENCH / "CONTRIBUTING-perf.md").read_text(encoding = "utf-8"), (
        "the documented invocation and the probe's own prefix have to agree, or the recipe in "
        "CONTRIBUTING-perf.md produces an empty log and a false NOT RUN"
    )


def test_the_event_counter_is_the_one_potency_rests_on(probe_src: str):
    """`ev_skip` is the only potency signal here that no author CSS can fake."""

    assert "contentvisibilityautostatechange" in probe_src
    assert "ev_skip" in probe_src
    # The geometry route is a documented false negative and must stay documented rather than
    # quietly removed: someone will reimplement it otherwise.
    assert "KNOWN FALSE NEGATIVE" in probe_src
