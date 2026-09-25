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
