# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Regression guard for the Colab-intro strip in the Unsloth Docker image.

cells[0] alone misses two shapes: the badge in cells[0] with the sentence in cells[1],
and the sentence wrapped in a single-line HTML comment.

The widening has to stay narrow: the scan stops at the first non-markdown cell so it
never reaches prose between code cells, it stays idempotent, and it must leave
unsloth_nb_content_sig's middle digest alone or the boot refresh re-strips forever.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STRIP_PATH = REPO_ROOT / "docker" / "unsloth_nb_strip_colab.py"

INTRO = 'To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!\n'
BADGE = '<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/X.ipynb">badge</a>\n'


@pytest.fixture(scope = "module")
def strip():
    assert STRIP_PATH.is_file(), f"missing {STRIP_PATH}"
    spec = importlib.util.spec_from_file_location("unsloth_nb_strip_under_test", STRIP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(src):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [src],
    }


def nb(*cells):
    return {"cells": list(cells), "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def text(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def has_intro(notebook):
    return any("to run this, press" in text(c).lower() for c in notebook["cells"])


def test_intro_in_cell_zero_is_still_stripped(strip):
    doc = nb(md(INTRO, "\n", BADGE), code("print(1)"))
    assert strip._strip_intro(doc) is True
    assert not has_intro(doc)
    assert BADGE in text(doc["cells"][0]), "the badge row must survive the strip"


def test_intro_in_cell_one_behind_the_badge_is_stripped(strip):
    doc = nb(md(BADGE), md(INTRO, "\n", "You will learn how to do data prep.\n"), code("print(1)"))
    assert strip._strip_intro(doc) is True
    assert not has_intro(doc)
    assert "You will learn how to do data prep.\n" in text(doc["cells"][1])


def test_intro_inside_a_single_line_html_comment_is_stripped(strip):
    commented = "<!-- " + INTRO.rstrip("\n") + " -->\n"
    doc = nb(md(commented, '<div class="align-center">\n'), code("print(1)"))
    assert strip._strip_intro(doc) is True
    assert not has_intro(doc)
    assert '<div class="align-center">\n' in text(doc["cells"][0])


def test_multi_line_html_comment_is_left_alone(strip):
    # half-removing a multi-line comment leaves a `<!--` that swallows the cell
    doc = nb(md("<!-- " + INTRO, "still inside the comment\n", "-->\n"), code("print(1)"))
    assert strip._strip_intro(doc) is False
    assert has_intro(doc)


def test_strip_stops_at_the_first_code_cell(strip):
    # a markdown cell AFTER code is prose, not the header block
    later = md("Explanation.\n", INTRO)
    doc = nb(md(BADGE), code("print(1)"), later)
    assert strip._strip_intro(doc) is False
    assert text(doc["cells"][2]) == "Explanation.\n" + INTRO


def test_strip_is_idempotent(strip):
    doc = nb(md(BADGE), md(INTRO, "\n", "rest\n"), code("print(1)"))
    assert strip._strip_intro(doc) is True
    once = copy.deepcopy(doc)
    assert strip._strip_intro(doc) is False, "a second boot must be a no-op"
    assert doc == once


def test_a_notebook_without_the_intro_is_untouched(strip):
    doc = nb(md(BADGE, "# Title\n"), code("print(1)"))
    before = copy.deepcopy(doc)
    assert strip._strip_intro(doc) is False
    assert doc == before


def test_source_given_as_a_string_is_handled(strip):
    doc = nb(
        {"cell_type": "markdown", "metadata": {}, "source": BADGE},
        {"cell_type": "markdown", "metadata": {}, "source": INTRO + "\nrest\n"},
        code("print(1)"),
    )
    assert strip._strip_intro(doc) is True
    assert not has_intro(doc)
    assert isinstance(doc["cells"][1]["source"], str)


def test_end_to_end_write_back_is_valid_json(strip, tmp_path):
    p = tmp_path / "N.ipynb"
    p.write_text(json.dumps(nb(md(BADGE), md(INTRO, "\n", "rest\n"), code("print(1)"))))
    assert strip.strip_notebook(str(p)) is True
    reloaded = json.loads(p.read_text())
    assert not has_intro(reloaded)
    assert strip.strip_notebook(str(p)) is False
