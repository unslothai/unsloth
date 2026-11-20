"""The settings-tabs browser smoke expects each control moved out of Appearance under a named
Chat tab section. Read that expectation from the driver and check it against chat-tab.tsx, so a
regrouping of the Chat tab fails here, in the unit tests, rather than only in the Playwright run."""

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "tests" / "studio" / "playwright_settings_tabs.py"
CHAT_TAB = REPO / "studio" / "frontend" / "src" / "features" / "settings" / "tabs" / "chat-tab.tsx"

_SECTION = re.compile(r"<SettingsSection\b[^>]*?title=\{t\(\s*\"([^\"]+)\"")
_LABEL = re.compile(r"t\(\s*\"([^\"]+)\"")


def _expected() -> tuple:
    tree = ast.parse(DRIVER.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "COMPOSER_SEARCH_SECTIONS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("COMPOSER_SEARCH_SECTIONS not found in playwright_settings_tabs.py")


def _sections_by_label(source: str) -> dict:
    """Map every t("...") key in chat-tab.tsx to the title key of the SettingsSection it sits in."""
    starts = [(m.start(), m.group(1)) for m in _SECTION.finditer(source)]
    owner: dict = {}
    for m in _LABEL.finditer(source):
        enclosing = [title for pos, title in starts if pos < m.start()]
        if enclosing:
            owner.setdefault(m.group(1), set()).add(enclosing[-1])
    return owner


def test_driver_names_the_section_each_moved_control_lives_in():
    owner = _sections_by_label(CHAT_TAB.read_text(encoding = "utf-8"))
    expected = _expected()
    assert expected, "the driver lists no moved controls"
    for label_key, section_key in expected:
        assert label_key in owner, f"{label_key} is no longer rendered in chat-tab.tsx"
        assert owner[label_key] == {section_key}, (
            f"{label_key} sits under {sorted(owner[label_key])} in chat-tab.tsx, "
            f"but playwright_settings_tabs.py expects {section_key}"
        )
