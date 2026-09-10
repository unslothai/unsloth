# SPDX-License-Identifier: AGPL-3.0-only
"""The dictation model search box must be reachable by test id, not by copy.

`playwright_extra_ui.py` used `get_by_placeholder("Search model")`; #7835
reworded that placeholder, so `Locator.fill` timed out and took the Chat UI
Tests job down on every PR. The input now carries a test id.
"""

import ast
import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
VOICE_TAB = REPO / "studio/frontend/src/features/settings/tabs/voice-tab.tsx"
SETTINGS_DIALOG = REPO / "studio/frontend/src/features/settings/settings-dialog.tsx"
EXTRA_UI = REPO / "tests/studio/playwright_extra_ui.py"
EN_LOCALE = REPO / "studio/frontend/src/i18n/locales/en.ts"

# Every element the dictation step drives, and the i18n key it replaced.
TEST_IDS = {
    "dictation-engine-trigger": "settings.voice.dictation.engineLabel",
    "dictation-engine-model": "settings.voice.dictation.engineModel",
    "stt-model-trigger": "settings.voice.dictation.sttModelLabel",
    "stt-model-search": "settings.voice.dictation.sttModelSearchPlaceholder",
}
TEST_ID = "stt-model-search"
# Tab buttons come from one map, so the test id is templated.
TAB_TEST_ID = "data-testid={`settings-tab-${tab.id}`}"


@pytest.mark.parametrize("test_id", sorted(TEST_IDS))
def test_the_element_carries_the_test_id(test_id):
    source = VOICE_TAB.read_text(encoding = "utf-8")
    assert f'data-testid="{test_id}"' in source, (test_id, VOICE_TAB)


@pytest.mark.parametrize("test_id,key", sorted(TEST_IDS.items()))
def test_the_test_id_sits_on_the_right_element(test_id, key):
    """A test id on the wrong element still passes above, so require it beside its key."""
    source = VOICE_TAB.read_text(encoding = "utf-8")
    index = source.index(f'data-testid="{test_id}"')
    block = source[max(0, index - 400) : index + 400]
    assert key in block, (test_id, block)


@pytest.mark.parametrize("test_id", sorted(TEST_IDS))
def test_the_driver_uses_it(test_id):
    source = EXTRA_UI.read_text(encoding = "utf-8")
    assert f'get_by_test_id("{test_id}")' in source, (test_id, EXTRA_UI)


def test_the_voice_settings_tab_is_reachable_by_test_id():
    """The step's first click is the Voice tab, whose label is translated too."""
    source = SETTINGS_DIALOG.read_text(encoding = "utf-8")
    assert TAB_TEST_ID in source, SETTINGS_DIALOG
    assert 'id: "voice"' in source, SETTINGS_DIALOG
    assert 'get_by_test_id("settings-tab-voice")' in EXTRA_UI.read_text(encoding = "utf-8")


# Locators that resolve through user-visible copy (get_by_role only with a name).
COPY_LOCATORS = (
    "get_by_placeholder",
    "get_by_label",
    "get_by_text",
    "get_by_alt_text",
    "get_by_title",
)
# The per-line pattern this guard replaced.
PER_LINE = r"get_by_(placeholder|label)\(|get_by_role\([^)]*name\s*="


def copy_locator_calls(source, first_line, last_line):
    """Copy-bound locator calls starting between two 1-based lines.

    Walks the AST, so a call split over lines is one node. Parses the whole
    file because the sliced step alone is not parseable.
    """
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if not first_line <= node.lineno <= last_line:
            continue
        attr = node.func.attr
        named_role = attr == "get_by_role" and any(kw.arg == "name" for kw in node.keywords)
        if attr in COPY_LOCATORS or named_role:
            segment = ast.get_source_segment(source, node) or attr
            offenders.append(f"line {node.lineno}: {' '.join(segment.split())}")
    return offenders


def line_range(source, start, end):
    """1-based line numbers of two character offsets."""
    return source.count("\n", 0, start) + 1, source.count("\n", 0, end) + 1


def test_the_dictation_step_binds_to_no_translated_copy_at_all():
    """The whole step, not just the input: a reword anywhere in it repeats the outage."""
    source = EXTRA_UI.read_text(encoding = "utf-8")
    start = source.index("Voice model picker: real mouse-wheel scrolling")
    end = source.index("results.hover()", start)
    offenders = copy_locator_calls(source, *line_range(source, start, end))
    assert offenders == [], offenders


MULTI_LINE_SAMPLE = """page.get_by_role(
    "button",
    name = re.compile(r"^Voice$"),
).first.click()
page.get_by_test_id("stt-model-search").fill("whisper")
"""


def test_the_guard_catches_a_multi_line_copy_locator():
    """A split call hides `name =` from any per-line match."""
    assert [l for l in MULTI_LINE_SAMPLE.splitlines() if re.search(PER_LINE, l)] == []
    offenders = copy_locator_calls(MULTI_LINE_SAMPLE, 1, 5)
    assert len(offenders) == 1, offenders
    assert offenders[0].startswith("line 1: page.get_by_role("), offenders


def test_the_guard_passes_test_id_only_code():
    """It must not fire on the locators the step is supposed to use."""
    clean = 'page.get_by_test_id("stt-model-search").fill("whisper")\npage.get_by_role("dialog")\n'
    assert copy_locator_calls(clean, 1, 2) == []


def test_the_guard_ignores_calls_outside_the_range():
    assert copy_locator_calls(MULTI_LINE_SAMPLE, 5, 5) == []


def test_no_playwright_step_locates_this_input_by_its_copy():
    """The regression itself: one copy edit away from taking the job down again."""
    source = EXTRA_UI.read_text(encoding = "utf-8")
    offenders = [
        line.strip()
        for line in source.splitlines()
        if "get_by_placeholder" in line and re.search(r"[Ss]earch\s+model", line)
    ]
    assert offenders == [], offenders


def test_ci_actually_runs_this_file():
    """Repo-root pytest discovery skips this file, so a workflow must name it."""
    workflow = (REPO / ".github/workflows/studio-ui-smoke.yml").read_text(encoding = "utf-8")
    assert f"pytest tests/studio/{Path(__file__).name}" in workflow, workflow
    assert "tests/studio/**" in workflow, "the workflow must trigger on this path"


def test_the_english_copy_is_still_free_to_change():
    """Assert the key, not the string, so the wording stays free to change."""
    source = EN_LOCALE.read_text(encoding = "utf-8")
    assert "sttModelSearchPlaceholder:" in source, EN_LOCALE
