# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""A notebook deleted or renamed upstream has to leave the tree.

The refresh loop walks the CLONE, so a file that vanished upstream is never visited,
and the state file is then REPLACED by what that loop recorded. The copy we published
stays on disk with no record, the next refresh reads it as user-owned, and
unsloth_nb_view.py files it under "Other Notebooks" for good. Upstream deleted 10 and
renamed 7 nb/ notebooks in the last year, so they accumulate.

Only a file that still hashes to what the sync itself wrote is removed, which is what
keeps a user's edit safe, so that is asserted here rather than assumed.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNC = REPO_ROOT / "docker" / "unsloth_sync_notebooks.sh"

behavioural = pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("bash", "git", "sha256sum")),
    reason = "needs bash, git and sha256sum",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd = cwd,
        check = True,
        capture_output = True,
        env = dict(
            os.environ,
            GIT_AUTHOR_NAME = "t",
            GIT_AUTHOR_EMAIL = "t@e",
            GIT_COMMITTER_NAME = "t",
            GIT_COMMITTER_EMAIL = "t@e",
        ),
    )


FILES = {
    "nb/keep.ipynb": "keep-v1",
    "nb/doomed.ipynb": "doomed-v1",
    "nb/edited.ipynb": "edited-v1",
}


def _setup(tmp_path: Path):
    remote = tmp_path / "remote"
    (remote / "nb").mkdir(parents = True)
    for rel, body in FILES.items():
        (remote / rel).write_text(body, encoding = "utf-8")
    _git(remote, "init", "-q", "-b", "main")
    _git(remote, "add", "-A")
    _git(remote, "commit", "-qm", "one")

    template = tmp_path / "template"
    (template / "nb").mkdir(parents = True)
    dest = tmp_path / "dest"
    (dest / "nb").mkdir(parents = True)
    lines = []
    for rel, body in FILES.items():
        (template / rel).write_text(body, encoding = "utf-8")
        (dest / rel).write_text(body, encoding = "utf-8")
        lines.append(f"{_sha256(dest / rel)}  {rel}")
    (dest / ".unsloth_sync_state").write_text("\n".join(lines) + "\n", encoding = "utf-8")
    return remote, template, dest


def _advance(remote: Path):
    """Upstream drops two notebooks and changes the third."""
    (remote / "nb" / "doomed.ipynb").unlink()
    (remote / "nb" / "edited.ipynb").unlink()
    (remote / "nb" / "keep.ipynb").write_text("keep-v2", encoding = "utf-8")
    _git(remote, "add", "-A")
    _git(remote, "commit", "-qm", "two")


def _refresh(tmp_path: Path, remote: Path, template: Path, dest: Path, **extra):
    return subprocess.run(
        ["bash", str(SYNC)],
        capture_output = True,
        text = True,
        timeout = 300,
        env = dict(
            os.environ,
            UNSLOTH_NB_REFRESH_CHILD = "1",
            UNSLOTH_NOTEBOOKS_TEMPLATE = str(template),
            UNSLOTH_NOTEBOOKS_DIR = str(dest),
            UNSLOTH_NOTEBOOKS_REPO = str(remote),
            UNSLOTH_SKIP_NOTEBOOK_VIEW = "1",
            UNSLOTH_KEEP_COLAB_INTRO = "1",
            **extra,
        ),
    )


def _state(dest: Path) -> dict:
    path = dest / ".unsloth_sync_state"
    out = {}
    if path.is_file():
        for line in path.read_text(encoding = "utf-8").splitlines():
            parts = line.split("  ", 1)
            if len(parts) == 2:
                out[parts[1]] = parts[0]
    return out


@behavioural
def test_a_pristine_notebook_deleted_upstream_is_removed(tmp_path: Path):
    remote, template, dest = _setup(tmp_path)
    _advance(remote)
    # the user edited one of the two that upstream dropped
    (dest / "nb" / "edited.ipynb").write_text("MY OWN WORK", encoding = "utf-8")

    run = _refresh(tmp_path, remote, template, dest)
    assert run.returncode == 0, run.stdout + run.stderr

    assert not (
        dest / "nb" / "doomed.ipynb"
    ).exists(), "the notebook upstream deleted is still on disk, so the view keeps listing it"
    assert (dest / "nb" / "edited.ipynb").read_text(
        encoding = "utf-8"
    ) == "MY OWN WORK", "a file that no longer hashes to what we wrote is the user's, edited or not"
    assert (dest / "nb" / "keep.ipynb").read_text(encoding = "utf-8") == "keep-v2"

    state = _state(dest)
    assert "nb/doomed.ipynb" not in state
    assert state["nb/keep.ipynb"] == _sha256(dest / "nb" / "keep.ipynb")
    assert "1 removed upstream" in run.stdout, run.stdout


@behavioural
def test_the_removal_can_be_turned_off(tmp_path: Path):
    """Non-vacuity as well as the opt-out: with the switch on, the same run leaves it."""
    remote, template, dest = _setup(tmp_path)
    _advance(remote)

    run = _refresh(tmp_path, remote, template, dest, UNSLOTH_KEEP_REMOVED_NOTEBOOKS = "1")
    assert run.returncode == 0, run.stdout + run.stderr

    assert (dest / "nb" / "doomed.ipynb").exists()
    assert "0 removed upstream" in run.stdout, run.stdout


@behavioural
def test_a_case_only_rename_does_not_delete_the_file_just_published(tmp_path: Path):
    """The clone is case-sensitive while the destination may be a macOS or Windows bind
    mount, so the old path would resolve to the file the refresh has just written."""
    remote, template, dest = _setup(tmp_path)
    _git(remote, "mv", "nb/doomed.ipynb", "nb/Doomed.ipynb")
    _git(remote, "commit", "-qm", "rename case only")

    run = _refresh(tmp_path, remote, template, dest)
    assert run.returncode == 0, run.stdout + run.stderr

    assert (dest / "nb" / "Doomed.ipynb").exists()
    assert "0 removed upstream" in run.stdout, run.stdout


@behavioural
def test_nothing_is_removed_when_upstream_removed_nothing(tmp_path: Path):
    remote, template, dest = _setup(tmp_path)
    (remote / "nb" / "keep.ipynb").write_text("keep-v2", encoding = "utf-8")
    _git(remote, "add", "-A")
    _git(remote, "commit", "-qm", "two")

    run = _refresh(tmp_path, remote, template, dest)
    assert run.returncode == 0, run.stdout + run.stderr

    for rel in FILES:
        assert (dest / rel).exists(), rel
    assert "0 removed upstream" in run.stdout, run.stdout
