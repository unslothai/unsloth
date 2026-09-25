# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`_en_catalog.en_string` resolves the key it is asked for, and every key the drivers ask for exists.

The drivers look controls up by `en_string(...)` in browser jobs that take minutes to reach the
lookup. A key renamed in en.ts would surface there as a locator timeout; checked here, it fails in
a second with the key's name.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from _en_catalog import EN_LOCALE_TS, aria_label_selector, en_string

HERE = Path(__file__).resolve().parent

SAMPLE = """\
// A comment with a key: "that is not one"
export const en = {
  composer: {
    title: "Composer",
    wrapped:
      "A value on the line after its key",
    braces: "Literal { and } inside a string",
  },
  settings: {
    title: "Settings",
    chat: {
      title: "Chat",
      "quoted-key": "Quoted",
      single: 'Delete "{name}"?',
      escaped: 'It\\'s here',
      tick: `Run \\`unsloth\\` now`,
      templated: `Hello ${name}`,
      literalDollar: `Type \\${name} as written`,
      hex: "Context\\x20window",
      codePoint: "Smile \\u{1F600}",
      continued: "one \\\ntwo",
    },
  },
};
"""


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "en.ts"
    path.write_text(SAMPLE, encoding = "utf-8")
    return path


@pytest.mark.parametrize(
    "key, expected",
    [
        ("composer.title", "Composer"),
        ("settings.title", "Settings"),
        ("settings.chat.title", "Chat"),
        ("composer.wrapped", "A value on the line after its key"),
        ("composer.braces", "Literal { and } inside a string"),
        ("settings.chat.quoted-key", "Quoted"),
        ("settings.chat.single", 'Delete "{name}"?'),
        ("settings.chat.escaped", "It's here"),
        ("settings.chat.tick", "Run `unsloth` now"),
        ("settings.chat.literalDollar", "Type ${name} as written"),
        ("settings.chat.hex", "Context window"),
        ("settings.chat.codePoint", "Smile \U0001f600"),
        ("settings.chat.continued", "one two"),
    ],
)
def test_a_key_resolves_by_its_full_path(sample, key, expected):
    assert en_string(key, sample) == expected


def test_a_missing_key_fails_naming_it(sample):
    with pytest.raises(KeyError, match = "settings.chat.gone"):
        en_string("settings.chat.gone", sample)


def test_a_template_with_placeholders_is_refused(sample):
    with pytest.raises(ValueError, match = "templated"):
        en_string("settings.chat.templated", sample)


@pytest.mark.parametrize(
    "label, selector",
    [
        ("Plain text", '[aria-label="Plain text"]'),
        ('Delete "{name}"?', '[aria-label="Delete \\"{name}\\"?"]'),
        ("C:\\models", '[aria-label="C:\\\\models"]'),
        ("two\nlines", '[aria-label="two\\a lines"]'),
        ("carriage\rreturn", '[aria-label="carriage\\d return"]'),
        ("form\ffeed", '[aria-label="form\\c feed"]'),
        ("tab\there", '[aria-label="tab\\9 here"]'),
    ],
)
def test_a_label_is_quoted_as_a_css_string(label, selector):
    assert aria_label_selector(label) == selector


def test_a_comment_is_not_read_as_a_key(sample):
    with pytest.raises(KeyError):
        en_string("key", sample)


def test_the_shipped_catalog_resolves():
    assert en_string("composerSettings.showContext")
    assert en_string("settings.chat.showResponseModel")
    # Single-quoted in en.ts because it contains double quotes.
    assert en_string("shell.dialog.deleteChat.description").endswith('"{name}"?')


def test_every_key_the_studio_tests_ask_for_exists():
    asked = {}
    for path in sorted(HERE.glob("*.py")):
        if path.name in {"_en_catalog.py", Path(__file__).name}:
            continue
        for key in re.findall(
            r"""en_string\(\s*["']([^"']+)["']""", path.read_text(encoding = "utf-8")
        ):
            asked.setdefault(key, path.name)
    assert asked, "no test looks a label up through en_string any more"
    missing = {}
    for key, where in asked.items():
        try:
            en_string(key)
        except KeyError:
            missing[key] = where
    assert not missing, f"keys asked for but not in {EN_LOCALE_TS.name}: {missing}"


def _github_glob(pattern: str) -> re.Pattern[str]:
    """A workflow `paths` pattern as a regex: `**` crosses `/` (`**/` also matches nothing), `*` and `?` do not."""
    out = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            # `**/` may match zero directories: `a/**/b` selects `a/b` too.
            out.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("**", index):
            out.append(".*")
            index += 2
        elif pattern[index] == "*":
            out.append("[^/]*")
            index += 1
        elif pattern[index] == "?":
            out.append("[^/]")
            index += 1
        else:
            out.append(re.escape(pattern[index]))
            index += 1
    return re.compile("".join(out))


def _paths_include(paths: list[str], path: str) -> bool:
    """Whether a `paths` filter selects `path`: patterns apply in order, the last match wins."""
    included = False
    for pattern in paths:
        negated = pattern.startswith("!")
        if _github_glob(pattern[1:] if negated else pattern).fullmatch(path):
            included = not negated
    return included


@pytest.mark.parametrize(
    "paths, included",
    [
        (["tests/studio/_en_catalog.py"], True),
        (["tests/studio/*"], True),
        (["tests/**"], True),
        (["tests/*"], False),
        (["tests/studio/*", "!tests/studio/_en_catalog.py"], False),
        (["!tests/studio/_en_catalog.py", "tests/studio/*"], True),
        (["tests/**", "!tests/studio/**", "tests/studio/_en_*.py"], True),
        (["tests/studio/**/_en_catalog.py"], True),
        (["tests/**/_en_catalog.py"], True),
        (["tests/**/studio/_en_catalog.py"], True),
    ],
)
def test_the_paths_filter_is_read_in_order(paths, included):
    assert _paths_include(paths, "tests/studio/_en_catalog.py") is included


def _pull_request_runs_on(workflow: dict, path: str) -> bool:
    """Whether a pull request that changes only `path` triggers `workflow`."""
    # PyYAML reads a bare `on:` key as True and a quoted `"on":` as the string.
    triggers = workflow.get(True, workflow.get("on"))
    if isinstance(triggers, str):
        triggers = [triggers]
    if isinstance(triggers, list):
        return "pull_request" in triggers
    if not isinstance(triggers, dict) or "pull_request" not in triggers:
        return False
    event = triggers["pull_request"] or {}
    if "paths" in event:
        return _paths_include(event["paths"] or [], path)
    if "paths-ignore" in event:
        # Same ordered reading: a match excludes, a later `!` match puts it back.
        return not _paths_include(event["paths-ignore"] or [], path)
    return True


@pytest.mark.parametrize(
    "workflow, runs",
    [
        ({True: {"pull_request": None}}, True),
        ({True: "pull_request"}, True),
        ({True: ["push", "pull_request"]}, True),
        ({True: {"push": None}}, False),
        ({"on": {"pull_request": {"paths": ["tests/other/**"]}}}, False),
        ({"on": {"pull_request": {"paths": ["tests/studio/**"]}}}, True),
        ({True: {"pull_request": {"paths": ["tests/studio/**/_en_catalog.py"]}}}, True),
        ({True: {"pull_request": {"paths-ignore": ["tests/studio/**"]}}}, False),
        ({True: {"pull_request": {"paths-ignore": ["docs/**"]}}}, True),
        (
            {
                True: {
                    "pull_request": {"paths-ignore": ["tests/**", "!tests/studio/_en_catalog.py"]}
                }
            },
            True,
        ),
    ],
)
def test_the_pull_request_trigger_is_read_in_full(workflow, runs):
    assert _pull_request_runs_on(workflow, "tests/studio/_en_catalog.py") is runs


def test_every_workflow_running_a_catalog_driver_also_triggers_on_the_catalog_reader():
    """A PR that changes only `_en_catalog.py` must still run the browser drivers built on it."""
    import yaml

    repo = HERE.parents[1]
    reader = "tests/studio/_en_catalog.py"
    drivers = [
        path.name
        for path in sorted(HERE.glob("*.py"))
        if not path.name.startswith("test_")
        and path.name != "_en_catalog.py"
        and "_en_catalog" in path.read_text(encoding = "utf-8")
    ]
    assert drivers, "no browser driver reads the catalog any more"
    unguarded = {}
    for workflow in sorted((repo / ".github" / "workflows").glob("*.y*ml")):
        text = workflow.read_text(encoding = "utf-8")
        runs = [name for name in drivers if f"tests/studio/{name}" in text]
        if not runs:
            continue
        if not _pull_request_runs_on(yaml.safe_load(text), reader):
            unguarded[workflow.name] = runs
    assert not unguarded, f"these workflows run catalog drivers but skip {reader}: {unguarded}"
