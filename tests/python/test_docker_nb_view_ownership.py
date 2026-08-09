# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""The categorized notebook VIEW may only delete the links it created.

`unsloth_nb_view.py` rebuilds "/workspace/Unsloth Notebooks" on every boot, and
that directory is also JupyterLab's landing dir, so `_clear_view()` promises to
remove only the tool's own symlinks. Every link the tool creates points at
DEST/nb/<file>, but the ownership predicate accepted ANY target under DEST, so a
user's own symlink into the notebooks checkout -- e.g. a shortcut to their own
notebook saved beside it, which the sync script explicitly supports ("kept
existing user file" / "In DEST but never recorded") -- was classified as
tool-owned and deleted on the next boot.

Behavioural: builds a real DEST/VIEW pair on disk and runs build_view twice.
No docker, no network.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VIEW_PATH = REPO_ROOT / "docker" / "unsloth_nb_view.py"

README = (
    "### Main Notebooks\n"
    "[Llama](nb/Llama3_2_%281B_and_3B%29_Conversational.ipynb)\n"
    "### Gemma\n"
    "[Gemma](nb/Gemma3_%284B%29.ipynb)\n"
)


@pytest.fixture(scope = "module")
def view_mod():
    assert VIEW_PATH.is_file(), f"missing {VIEW_PATH}"
    spec = importlib.util.spec_from_file_location("unsloth_nb_view_under_test", VIEW_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tree(tmp_path: Path):
    dest = tmp_path / "unsloth-notebooks"
    view = tmp_path / "Unsloth Notebooks"
    (dest / "nb").mkdir(parents = True)
    view.mkdir()
    for name in ("Llama3_2_(1B_and_3B)_Conversational.ipynb", "Gemma3_(4B).ipynb"):
        (dest / "nb" / name).write_text("{}", encoding = "utf-8")
    (dest / "README.md").write_text(README, encoding = "utf-8")
    # The user's own notebook, saved inside the checkout (supported by the sync
    # script), plus their own folder of shortcuts in the landing dir.
    (dest / "my_work").mkdir()
    (dest / "my_work" / "experiment.ipynb").write_text("{}", encoding = "utf-8")
    return dest, view


def link(target: Path, at: Path) -> None:
    at.parent.mkdir(parents = True, exist_ok = True)
    os.symlink(os.path.relpath(target, at.parent), at)


def test_a_user_link_to_their_own_file_in_the_checkout_survives(view_mod, tree):
    dest, view = tree
    own = view / "00 My favourites" / "experiment.ipynb"
    link(dest / "my_work" / "experiment.ipynb", own)

    view_mod.build_view(str(dest), str(view))

    assert os.path.islink(own), (
        "a symlink the user created in the landing dir, pointing at their own "
        "file inside the notebooks checkout, was deleted by _clear_view"
    )
    assert os.path.realpath(own) == os.path.realpath(dest / "my_work" / "experiment.ipynb")


def test_a_user_link_outside_the_checkout_survives(view_mod, tree, tmp_path: Path):
    dest, view = tree
    outside = tmp_path / "datasets"
    outside.mkdir()
    own = view / "datasets"
    link(outside, own)

    view_mod.build_view(str(dest), str(view))

    assert os.path.islink(own)


def test_the_tools_own_stale_links_are_still_cleaned_up(view_mod, tree):
    dest, view = tree
    view_mod.build_view(str(dest), str(view))
    generated = view / "02 Gemma" / "Gemma3_(4B).ipynb"
    assert os.path.islink(generated)

    # Upstream drops the notebook: its generated link (now stale, and pointing
    # into DEST/nb) has to go, and the emptied folder with it.
    (dest / "nb" / "Gemma3_(4B).ipynb").unlink()
    (dest / "README.md").write_text(
        "### Main Notebooks\n[Llama](nb/Llama3_2_%281B_and_3B%29_Conversational.ipynb)\n",
        encoding = "utf-8",
    )
    view_mod.build_view(str(dest), str(view))

    assert not os.path.islink(generated) and not os.path.exists(generated)
    assert not (view / "02 Gemma").exists()


def test_a_rebuild_is_stable_for_the_links_it_owns(view_mod, tree):
    dest, view = tree
    view_mod.build_view(str(dest), str(view))
    first = sorted(str(p.relative_to(view)) for p in view.rglob("*"))
    view_mod.build_view(str(dest), str(view))
    assert sorted(str(p.relative_to(view)) for p in view.rglob("*")) == first
