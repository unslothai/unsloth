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


def _refuse_rm(tmp_path: Path, target: Path) -> Path:
    """A `rm` earlier on PATH that fails for one path, the way a writable single-FILE
    bind mount fails with EBUSY. The publish above already works around that case for
    rename, so it is not hypothetical here."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok = True)
    stub = bindir / "rm"
    stub.write_text(
        "#!/bin/sh\n"
        'for a in "$@"; do\n'
        f'  [ "$a" = "{target}" ] && exit 1\n'
        "done\n"
        'exec /bin/rm "$@"\n',
        encoding = "utf-8",
    )
    stub.chmod(0o755)
    return bindir


@behavioural
def test_a_removal_that_cannot_be_unlinked_keeps_its_record_and_retries(tmp_path: Path):
    """Ignoring the failure dropped the record while leaving the file, which is the
    exact state the removal exists to prevent: the next refresh reads the stale copy as
    user-owned and never tries again. It also left `failed` at zero, so the sync marker
    was stamped and the next start exited before it looked."""
    remote, template, dest = _setup(tmp_path)
    _advance(remote)
    # upstream dropped this one too, but the user owns it now, so doomed is the only
    # removal candidate and the counters below are about it alone
    (dest / "nb" / "edited.ipynb").write_text("MY OWN WORK", encoding = "utf-8")
    doomed = dest / "nb" / "doomed.ipynb"
    bindir = _refuse_rm(tmp_path, doomed)

    run = _refresh(
        tmp_path, remote, template, dest, PATH = f"{bindir}{os.pathsep}{os.environ['PATH']}"
    )
    assert run.returncode == 0, run.stdout + run.stderr

    assert doomed.exists(), "precondition: the stub refused the unlink"
    assert _state(dest).get("nb/doomed.ipynb") == _sha256(
        doomed
    ), "the file stayed but its record was dropped, so it is now unmanaged"
    assert "0 removed upstream" in run.stdout, run.stdout
    assert not (
        dest / ".unsloth_sync_commit"
    ).is_file(), "the commit was stamped over a failure, so the next start exits before retrying"

    run2 = _refresh(tmp_path, remote, template, dest)
    assert run2.returncode == 0, run2.stdout + run2.stderr
    assert not doomed.exists()
    assert "nb/doomed.ipynb" not in _state(dest)
    assert "1 removed upstream" in run2.stdout, run2.stdout


@behavioural
def test_the_successful_removal_still_stamps_the_commit(tmp_path: Path):
    """Non-vacuity for the assertion above: without the stub the same run records it,
    so the marker check is testing the failure and not the environment."""
    remote, template, dest = _setup(tmp_path)
    _advance(remote)
    (dest / "nb" / "edited.ipynb").write_text("MY OWN WORK", encoding = "utf-8")

    run = _refresh(tmp_path, remote, template, dest)
    assert run.returncode == 0, run.stdout + run.stderr

    assert (dest / ".unsloth_sync_commit").is_file()
    assert "1 removed upstream" in run.stdout, run.stdout


def _refuse_mv_to(tmp_path: Path, target: Path) -> Path:
    """An `mv` earlier on PATH that fails for one destination, the way a full or
    read-only /workspace fails when the state file is published. Every other move,
    including the same-dir rename each notebook is published with, passes through."""
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok = True)
    stub = bindir / "mv"
    stub.write_text(
        "#!/bin/sh\n"
        'for a in "$@"; do\n'
        f'  [ "$a" = "{target}" ] && exit 1\n'
        "done\n"
        'exec /bin/mv "$@"\n',
        encoding = "utf-8",
    )
    stub.chmod(0o755)
    return bindir


@behavioural
def test_a_state_that_cannot_be_published_holds_the_sync_marker(tmp_path: Path):
    """Stamping the commit over a failed state publish wedges the tree permanently.
    The next boot exits on remote == last, and once upstream moves again the stale
    hashes no longer match the notebooks this run wrote, so all of them read as
    user-edited and are never updated again. `failed` already holds the marker back
    for a notebook that could not be written; the state file has to count too."""
    remote, template, dest = _setup(tmp_path)
    _advance(remote)
    before = _state(dest)
    bindir = _refuse_mv_to(tmp_path, dest / ".unsloth_sync_state")

    run = _refresh(
        tmp_path, remote, template, dest, PATH = f"{bindir}{os.pathsep}{os.environ['PATH']}"
    )
    assert run.returncode == 0, run.stdout + run.stderr

    assert _state(dest) == before, "precondition: the stub refused the state publish"
    assert not (
        dest / ".unsloth_sync_commit"
    ).is_file(), "the commit was stamped over an unpublished state, so the tree is wedged"


@behavioural
def test_the_state_temp_file_is_published_by_a_same_directory_rename(tmp_path: Path):
    """Holding the marker is not on its own enough, which is why the temp file sits
    beside the state file rather than in /tmp. The notebooks are published BEFORE the
    state is, so a publish that fails on its own leaves disk ahead of state, and no
    marker check undoes that: the next run reads every notebook it just wrote as
    user-edited, because they no longer hash to the stale record. A sibling makes the
    publish an atomic rename and makes it fail WITH the notebook writes, which already
    hold the marker. Asserted on the script text because the failure it removes cannot
    be provoked once the rename is atomic."""
    source = SYNC.read_text(encoding = "utf-8")
    assert 'TMPSTATE="$STATE.tmp"' in source, (
        "the refresh state temp file left $DEST, so publishing it is a cross-device "
        "copy that can fail after the notebooks have already landed"
    )
    stray = [
        line.strip()
        for line in source.splitlines()
        if "mktemp" in line and "TMPSTATE=" in line and "||" not in line
    ]
    assert not stray, f"the refresh state must not be staged outside $DEST: {stray}"


@behavioural
def test_a_published_state_still_stamps_the_commit(tmp_path: Path):
    """Non-vacuity for the assertion above: the identical run without the stub records
    the marker, so that check is testing the failure and not the environment."""
    remote, template, dest = _setup(tmp_path)
    _advance(remote)

    run = _refresh(tmp_path, remote, template, dest)
    assert run.returncode == 0, run.stdout + run.stderr
    assert (dest / ".unsloth_sync_commit").is_file()
