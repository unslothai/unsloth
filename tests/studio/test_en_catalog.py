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

from _en_catalog import EN_LOCALE_TS, _decode, aria_label_selector, en_string

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
      joined: "Context " + "window usage",
      noted: "Context " /* note */ + "window usage",
      upper: "context".toUpperCase(),
      picked: flag ? "short" : "long",
      separated: "a\\\u2028b",
      annotated /* translator note */: "Annotated",
      lineNoted // translator note
        : "Line noted",
      last: "Last" // trailing comment
    },
    overridden: {
      before: "Before",
      nested: { inner: "Inner" },
      ...shared,
      after: "After",
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
        ("settings.chat.last", "Last"),
        ("settings.chat.annotated", "Annotated"),
        ("settings.chat.lineNoted", "Line noted"),
        ("settings.chat.separated", "ab"),
        ("settings.overridden.after", "After"),
    ],
)
def test_a_key_resolves_by_its_full_path(sample, key, expected):
    assert en_string(key, sample) == expected


@pytest.mark.parametrize("terminator", ["\n", "\r\n", "\r", "\u2028", "\u2029"])
def test_a_continuation_over_any_line_terminator_contributes_nothing(terminator):
    # Decoded directly: reading a file with read_text folds CR LF to LF before the tokenizer.
    assert _decode(f'"a\\{terminator}b"') == "ab"


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


def test_a_concatenated_value_is_refused_not_truncated(sample):
    for key in (
        "settings.chat.joined",
        "settings.chat.noted",
        "settings.chat.upper",
        "settings.chat.picked",
        "settings.overridden.before",
        "settings.overridden.nested.inner",
    ):
        with pytest.raises(ValueError, match = "expression"):
            en_string(key, sample)


def test_a_label_with_nul_is_refused_by_the_selector():
    with pytest.raises(ValueError, match = "NUL"):
        aria_label_selector("a\0b")


def test_a_surrogate_pair_joins_and_a_lone_half_is_refused_by_the_selector():
    assert _decode('"\\uD83D\\uDE00"') == "\U0001f600"
    with pytest.raises(ValueError, match = "surrogate"):
        aria_label_selector(_decode('"\\uD800"'))


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


def _imports_catalog(path: Path) -> bool:
    """Whether a module imports `_en_catalog`, in either `import` or `from ... import` form."""
    import ast

    for node in ast.walk(ast.parse(path.read_text(encoding = "utf-8"))):
        if isinstance(node, ast.Import) and any(a.name == "_en_catalog" for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "_en_catalog":
            return True
    return False


@pytest.mark.parametrize(
    "source, imports",
    [
        ("from _en_catalog import en_string\n", True),
        ("import _en_catalog\n", True),
        ("import os, _en_catalog as catalog\n", True),
        ("# from _en_catalog import en_string\n", False),
        ("x = 'from _en_catalog import en_string'\n", False),
    ],
)
def test_both_import_forms_mark_a_catalog_driver(tmp_path, source, imports):
    path = tmp_path / "driver.py"
    path.write_text(source, encoding = "utf-8")
    assert _imports_catalog(path) is imports


COMPOSER_WORKFLOW = HERE.parents[1] / ".github" / "workflows" / "studio-composer-compatibility.yml"


# Where each catalog driver runs, pinned literally: the step's `if:` and the matrix leg it
# needs. A new driver, or a restructured workflow, updates this table with it.
DRIVER_STEPS = {
    "playwright_composer_settings.py": (
        "matrix.suite == 'browsers'",
        {"suite": "browsers"},
        # Continues `run_driver "<log>" \`, whose failure the loop turns into the exit status.
        "python tests/studio/playwright_composer_settings.py || status=1",
    ),
    "selenium_composer_safari.py": (
        "${{ !cancelled() && matrix.suite == 'safari' }}",
        {"os": "macos-latest", "suite": "safari"},
        # Last command of a bash -e step, so its exit status is the step's.
        "python tests/studio/selenium_composer_safari.py",
    ),
}


def test_the_composer_workflow_runs_on_a_catalog_only_change():
    """Both browser drivers find their controls through `_en_catalog.py`, and this workflow is
    the one that runs them. A PR that changes only the reader has to run it too.

    Deliberately literal rather than an evaluator of Actions expressions and shell: the reader
    is listed by name in `pull_request.paths` with no other filter; every catalog driver is
    in `DRIVER_STEPS`; and each runs from a step whose `if:` is exactly the pinned one, on a
    matrix leg that exists, as exactly the pinned command line (so `|| true`, an `echo` of it,
    or any other respelling fails). A workflow restructured some other way updates this test
    with it.
    """
    import yaml

    workflow = yaml.safe_load(COMPOSER_WORKFLOW.read_text(encoding = "utf-8"))
    triggers = workflow.get(True, workflow.get("on"))  # PyYAML reads a bare `on:` as True.
    pull_request = triggers["pull_request"]
    assert "tests/studio/_en_catalog.py" in pull_request["paths"]
    assert not {"paths-ignore", "branches", "branches-ignore", "types"} & set(pull_request)
    drivers = sorted(
        path.name
        for path in HERE.glob("*.py")
        if not path.name.startswith(("test_", "_")) and _imports_catalog(path)
    )
    assert drivers, "no browser driver reads the catalog any more"
    assert set(drivers) <= set(DRIVER_STEPS), f"pin where these drivers run: {drivers}"
    for name in drivers:
        condition, leg, command = DRIVER_STEPS[name]
        found = [
            (job, step)
            for job in (workflow.get("jobs") or {}).values()
            for step in job.get("steps") or []
            if command in (line.strip() for line in str(step.get("run") or "").splitlines())
        ]
        assert found, f"no step runs exactly {command!r}"
        assert any(
            "if" not in job
            and str(step.get("if")) == condition
            and any(
                all(include.get(key) == value for key, value in leg.items())
                for include in ((job.get("strategy") or {}).get("matrix") or {}).get("include", [])
            )
            for job, step in found
        ), f"{name} no longer runs from a step gated on {condition!r} with the leg {leg}"
