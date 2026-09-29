# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""The Colab-intro cleanup must not overwrite a save it did not see.

`strip_notebook` read the file, serialised the cleaned copy and then `os.replace`d it
unconditionally, while JupyterLab was already serving $DEST. A save landing in that
window was destroyed, and `migrate` then recorded the cleaned hash, so the state
machine treats the notebook as pristine forever after.
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


@pytest.fixture(scope = "module")
def strip():
    assert STRIP_PATH.is_file(), f"missing {STRIP_PATH}"
    spec = importlib.util.spec_from_file_location("unsloth_nb_strip_race", STRIP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def notebook(*sources):
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": list(src)} for src in sources
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(path: Path, nb) -> None:
    path.write_text(json.dumps(nb, indent = 1, ensure_ascii = False) + "\n", encoding = "utf-8")


@pytest.fixture
def racing(strip, tmp_path: Path):
    real_dump = strip.json.dump
    state = {"save": None, "path": None, "fired": 0}

    def dump(obj, fp, *args, **kwargs):
        out = real_dump(obj, fp, *args, **kwargs)
        if state["save"] is not None and state["fired"] == 0:
            state["fired"] = 1
            Path(state["path"]).write_text(state["save"], encoding = "utf-8")  # Ctrl+S
        return out

    strip.json.dump = dump
    try:
        yield state
    finally:
        strip.json.dump = real_dump


def test_a_save_during_the_cleanup_is_not_overwritten(strip, racing, tmp_path: Path):
    path = tmp_path / "Llama.ipynb"
    write(path, notebook([INTRO, "\n", "# Llama\n"]))

    edited = notebook([INTRO, "\n", "# Llama\n", "\n", "my own notes, saved from JupyterLab\n"])
    racing["save"] = json.dumps(edited, indent = 1, ensure_ascii = False) + "\n"
    racing["path"] = str(path)

    strip.strip_notebook(str(path))

    on_disk = json.loads(path.read_text(encoding = "utf-8"))
    assert on_disk == edited, (
        "the user's save landed after strip_notebook read the file and was "
        "overwritten by the cleaned copy of the OLD content; the sync contract "
        "is that user edits always win"
    )


def test_the_recorded_hash_still_matches_the_file_after_a_racing_save(
    strip, racing, tmp_path: Path
):
    # migrate() rewrites STATE with the post-strip hash, so a clobbered save is also
    # recorded as pristine and every later refresh overwrites it again
    dest = tmp_path / "unsloth-notebooks"
    dest.mkdir()
    path = dest / "Llama.ipynb"
    write(path, notebook([INTRO, "\n", "# Llama\n"]))
    before = strip._sha256(str(path))
    state = tmp_path / ".unsloth_sync_state"
    state.write_text(f"{before}  Llama.ipynb\n", encoding = "utf-8")

    edited = notebook([INTRO, "\n", "# Llama\n", "\n", "my own notes\n"])
    racing["save"] = json.dumps(edited, indent = 1, ensure_ascii = False) + "\n"
    racing["path"] = str(path)

    strip.migrate(str(state), str(dest))

    recorded = state.read_text(encoding = "utf-8").split("  ", 1)[0]
    on_disk = strip._sha256(str(path))
    assert json.loads(path.read_text(encoding = "utf-8")) == edited
    assert recorded != on_disk, (
        "a file the user saved during the cleanup must NOT end up recorded as "
        "managed-and-pristine, or the next refresh overwrites it too"
    )


def test_the_normal_no_race_cleanup_still_strips_and_rewrites(strip, tmp_path: Path):
    path = tmp_path / "Llama.ipynb"
    original = notebook([INTRO, "\n", "# Llama\n"])
    write(path, copy.deepcopy(original))

    assert strip.strip_notebook(str(path)) is True
    cleaned = json.loads(path.read_text(encoding = "utf-8"))
    assert cleaned["cells"][0]["source"] == ["# Llama\n"]
    assert strip.strip_notebook(str(path)) is False  # idempotent


@pytest.fixture
def racing_after_replace(strip, tmp_path: Path):
    """The OTHER window: after os.replace published, before migrate() hashes."""
    real_replace = strip.os.replace
    state = {"save": None, "path": None, "fired": 0}

    def replace(src, dst, *args, **kwargs):
        out = real_replace(src, dst, *args, **kwargs)
        if state["save"] is not None and state["fired"] == 0 and str(dst) == state["path"]:
            state["fired"] = 1
            Path(state["path"]).write_text(state["save"], encoding = "utf-8")  # Ctrl+S
        return out

    strip.os.replace = replace
    try:
        yield state
    finally:
        strip.os.replace = real_replace


def test_a_save_landing_after_the_replace_is_not_recorded_as_pristine(
    strip, racing_after_replace, tmp_path: Path
):
    # rename(2) is atomic, but migrate()'s re-read of the published file is not
    dest = tmp_path / "unsloth-notebooks"
    dest.mkdir()
    path = dest / "Llama.ipynb"
    write(path, notebook([INTRO, "\n", "# Llama\n"]))
    before = strip._sha256(str(path))
    state = tmp_path / ".unsloth_sync_state"
    state.write_text(f"{before}  Llama.ipynb\n", encoding = "utf-8")

    edited = notebook([INTRO, "\n", "# Llama\n", "\n", "saved right after the replace\n"])
    racing_after_replace["save"] = json.dumps(edited, indent = 1, ensure_ascii = False) + "\n"
    racing_after_replace["path"] = str(path)

    strip.migrate(str(state), str(dest))

    assert (
        racing_after_replace["fired"] == 1
    ), "the window was never exercised; this test would pass vacuously"
    recorded = state.read_text(encoding = "utf-8").split("  ", 1)[0]
    assert json.loads(path.read_text(encoding = "utf-8")) == edited
    assert recorded != strip._sha256(
        str(path)
    ), "the user's own save was recorded as the cleaned, sync-owned version"


def test_the_recorded_hash_is_the_cleaned_copy_when_nobody_races(strip, tmp_path: Path):
    # over-reach guard: STATE must adopt the cleaned hash, or every boot re-strips it
    dest = tmp_path / "unsloth-notebooks"
    dest.mkdir()
    path = dest / "Llama.ipynb"
    write(path, notebook([INTRO, "\n", "# Llama\n"]))
    state = tmp_path / ".unsloth_sync_state"
    state.write_text(f"{strip._sha256(str(path))}  Llama.ipynb\n", encoding = "utf-8")

    strip.migrate(str(state), str(dest))

    recorded = state.read_text(encoding = "utf-8").split("  ", 1)[0]
    assert recorded == strip._sha256(str(path))


def test_cleanup_keeps_the_notebooks_owner_and_mode(strip, tmp_path: Path):
    """A bind-mounted notebook matching the baked template is managed WITHOUT being
    copied, so the host user keeps owning it -- and os.replace swaps the directory
    entry, so a root-written staging file would hand it back root-owned."""
    import os
    import stat as _stat

    nb_path = tmp_path / "Managed.ipynb"
    nb_path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "markdown",
                        "source": ["To run this, press *Runtime* > Run all\n"],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding = "utf-8",
    )
    os.chmod(nb_path, 0o640)
    before = _stat.S_IMODE(os.stat(nb_path).st_mode)
    assert before == 0o640

    assert strip.strip_notebook(str(nb_path)) is True
    after = _stat.S_IMODE(os.stat(nb_path).st_mode)
    assert after == before, f"cleanup changed the mode {oct(before)} -> {oct(after)}"
    assert "To run this" not in nb_path.read_text(encoding = "utf-8")
