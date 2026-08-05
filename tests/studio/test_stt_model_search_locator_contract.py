# SPDX-License-Identifier: AGPL-3.0-only
"""The dictation model search box must be reachable by test id, not by copy.

`playwright_extra_ui.py` located this input with
`get_by_placeholder("Search model")`. That placeholder is rendered through
`t("settings.voice.dictation.sttModelSearchPlaceholder")`, so it is both
translated and free to be reworded, and #7835 reworded the English string to
"Search any model on HF". `get_by_placeholder` matches on substring, and
"Search model" is not a substring of that, so the step failed with

    Locator.fill: Timeout 60000ms exceeded.
      waiting for get_by_placeholder("Search model")

That takes the whole Chat UI Tests job down, on every PR, for a copy edit that
touched no behaviour. The results container beside it already had
`data-testid="stt-model-results"`; the input now has one too.
"""

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
VOICE_TAB = REPO / "studio/frontend/src/features/settings/tabs/voice-tab.tsx"
EXTRA_UI = REPO / "tests/studio/playwright_extra_ui.py"
EN_LOCALE = REPO / "studio/frontend/src/i18n/locales/en.ts"

TEST_ID = "stt-model-search"


def test_the_input_carries_the_test_id():
    source = VOICE_TAB.read_text(encoding = "utf-8")
    assert f'data-testid="{TEST_ID}"' in source, VOICE_TAB


def test_the_test_id_sits_on_the_search_input_not_somewhere_else():
    """A test id on the wrong element would still satisfy the check above."""
    source = VOICE_TAB.read_text(encoding = "utf-8")
    index = source.index(f'data-testid="{TEST_ID}"')
    block = source[max(0, index - 400) : index + 400]
    assert "sttModelSearchPlaceholder" in block, block


def test_the_driver_uses_it():
    source = EXTRA_UI.read_text(encoding = "utf-8")
    assert f'get_by_test_id("{TEST_ID}")' in source, EXTRA_UI


def test_no_playwright_step_locates_this_input_by_its_copy():
    """The regression itself. Any placeholder-based locator here is one copy
    edit away from taking the job down again."""
    source = EXTRA_UI.read_text(encoding = "utf-8")
    offenders = [
        line.strip()
        for line in source.splitlines()
        if "get_by_placeholder" in line and re.search(r"[Ss]earch\s+model", line)
    ]
    assert offenders == [], offenders


def test_the_english_copy_is_still_free_to_change():
    """States the point: this file must keep passing whatever the wording is,
    so it deliberately does not assert the string."""
    source = EN_LOCALE.read_text(encoding = "utf-8")
    assert "sttModelSearchPlaceholder:" in source, EN_LOCALE
