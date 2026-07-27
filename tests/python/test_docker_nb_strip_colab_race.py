# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""The Colab-intro cleanup must not overwrite a save it did not see.

`unsloth_sync_notebooks.sh` forks the GitHub refresh into a DETACHED child before
the entrypoint execs the container command, so JupyterLab is already serving
$DEST while that child runs. When the refresh copied anything the child re-arms
`finalize()`, which runs `unsloth_nb_strip_colab.py --state ... --dest ...`, i.e.
`migrate()` -> `strip_notebook()` over every owned+unedited notebook.

`strip_notebook` read the file, parsed it, serialised the cleaned copy and then
`os.replace`d it unconditionally. A user save that landed in that window was
destroyed, and `migrate` then recorded the cleaned file's hash, so the state
machine treats the notebook as pristine forever after -- the same
check-then-write hole that was closed in the refresh loop itself (the publish
there now re-reads the hash immediately before the rename).

Behavioural: the save is injected inside the window, while the helper serialises
the cleaned copy (the widest part of it: json parse + dump of a notebook that is
often megabytes). No docker, no network.
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
    """Fire a user save inside the window: after strip_notebook read the file,
    while it is serialising the cleaned copy."""
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
    # migrate() rewrites STATE with the post-strip hash. If the write above is
    # allowed to clobber a save, the state ALSO says "pristine", so every later
    # refresh happily overwrites the notebook again.
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
    # Guard the fix from over-reaching: with nobody else writing, the cleanup
    # must still strip the Colab sentence and publish the result.
    path = tmp_path / "Llama.ipynb"
    original = notebook([INTRO, "\n", "# Llama\n"])
    write(path, copy.deepcopy(original))

    assert strip.strip_notebook(str(path)) is True
    cleaned = json.loads(path.read_text(encoding = "utf-8"))
    assert cleaned["cells"][0]["source"] == ["# Llama\n"]
    assert strip.strip_notebook(str(path)) is False  # idempotent
