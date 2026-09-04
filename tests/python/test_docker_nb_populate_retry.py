# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""First-boot population must not claim a commit it did not finish copying.

A `cp -a` that fails is skipped silently and gets no state entry, so phase 1b never
restores it -- and stamping the commit anyway tells the phase 2 refresh it is already
synced, so the notebook is gone for good on an offline container.

Driven end to end against the real script with the refresh disabled. The blocked path
is a plain FILE where a directory has to be, which is ENOTDIR for root too, so this
cannot silently pass under a root CI container the way a chmod would.
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

TEMPLATE_COMMIT = "a" * 40

behavioural = pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("bash", "sha256sum")),
    reason = "needs bash and sha256sum",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _template(tmp_path: Path) -> Path:
    template = tmp_path / "template"
    (template / "sub").mkdir(parents = True)
    (template / "a.ipynb").write_text("A", encoding = "utf-8")
    (template / "sub" / "b.ipynb").write_text("B", encoding = "utf-8")
    (template / ".unsloth_template_commit").write_text(TEMPLATE_COMMIT + "\n", encoding = "utf-8")
    return template


def _run(tmp_path: Path, template: Path, dest: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SYNC)],
        capture_output = True,
        text = True,
        timeout = 300,
        env = dict(
            os.environ,
            UNSLOTH_NOTEBOOKS_TEMPLATE = str(template),
            UNSLOTH_NOTEBOOKS_DIR = str(dest),
            # phase 2 needs a clone; the offline half must retry on its own
            UNSLOTH_SKIP_NOTEBOOK_REFRESH = "1",
            UNSLOTH_SKIP_NOTEBOOK_VIEW = "1",
            UNSLOTH_KEEP_COLAB_INTRO = "1",
        ),
    )


def _state(dest: Path) -> dict:
    path = dest / ".unsloth_sync_state"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding = "utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


@behavioural
def test_a_failed_copy_leaves_the_commit_unstamped_and_retries_next_start(tmp_path: Path):
    template = _template(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    # a regular file, so the copy into it is ENOTDIR for root too
    (dest / "sub").write_text("blocked", encoding = "utf-8")

    first = _run(tmp_path, template, dest)
    assert first.returncode == 0, first.stdout + first.stderr

    assert (dest / "a.ipynb").read_text(encoding = "utf-8") == "A"
    assert "a.ipynb" in _state(dest)
    assert "sub/b.ipynb" not in _state(dest), "a copy that failed must not be recorded"
    assert not (dest / ".unsloth_sync_commit").exists(), (
        "stamping the template commit after an incomplete populate makes the "
        "phase 2 refresh exit early, so the missing notebook never comes back"
    )
    assert (dest / ".unsloth_sync_partial").exists()

    (dest / "sub").unlink()

    second = _run(tmp_path, template, dest)
    assert second.returncode == 0, second.stdout + second.stderr

    assert (dest / "sub" / "b.ipynb").read_text(encoding = "utf-8") == "B"
    assert _state(dest)["sub/b.ipynb"] == _sha256(dest / "sub" / "b.ipynb")
    assert _state(dest)["a.ipynb"] == _sha256(dest / "a.ipynb")
    assert (dest / ".unsloth_sync_commit").read_text(encoding = "utf-8").strip() == (TEMPLATE_COMMIT)
    assert not (dest / ".unsloth_sync_partial").exists()


@behavioural
def test_a_clean_populate_stamps_the_commit_and_does_not_re_run(tmp_path: Path):
    template = _template(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()

    assert _run(tmp_path, template, dest).returncode == 0
    assert (dest / ".unsloth_sync_commit").read_text(encoding = "utf-8").strip() == (TEMPLATE_COMMIT)
    assert not (dest / ".unsloth_sync_partial").exists()
    before = _state(dest)
    assert set(before) == {"a.ipynb", "sub/b.ipynb"}

    # a re-run of phase 1 would drop this file's managed record, so an identical
    # state is what proves the block was skipped
    (dest / "a.ipynb").write_text("edited", encoding = "utf-8")

    assert _run(tmp_path, template, dest).returncode == 0
    assert _state(dest) == before
    assert (dest / "a.ipynb").read_text(encoding = "utf-8") == "edited"


@behavioural
def test_the_partial_marker_is_not_recorded_as_a_notebook(tmp_path: Path):
    # record_state() walks every file under DEST, so a dotfile missing from its skip
    # list reaches users as a notebook to sync
    template = _template(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "sub").write_text("blocked", encoding = "utf-8")

    assert _run(tmp_path, template, dest).returncode == 0
    assert (dest / ".unsloth_sync_partial").exists()
    assert ".unsloth_sync_partial" not in _state(dest)

    source = SYNC.read_text(encoding = "utf-8")
    skip = source[source.index("record_state() {") :]
    skip = skip[: skip.index("printf")]
    assert ".unsloth_sync_partial" in skip


@behavioural
def test_the_retry_keeps_records_the_refresh_added(tmp_path: Path):
    """A retry rebuilds the state from the TEMPLATE alone, so anything the refresh did
    in between is thrown away: a template file whose bytes upstream changed now differs
    from the template and hits the "kept existing user file" branch, and a notebook that
    exists only upstream is never visited at all. Both become user-owned for good, while
    the commit marker is stamped anyway so it looks converged."""
    template = _template(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "sub").write_text("blocked", encoding = "utf-8")

    assert _run(tmp_path, template, dest).returncode == 0
    assert (dest / ".unsloth_sync_partial").exists()

    # what the refresh does between the two boots
    (dest / "a.ipynb").write_text("A-v2-from-upstream", encoding = "utf-8")
    (dest / "remote_only.ipynb").write_text("R", encoding = "utf-8")
    (dest / ".unsloth_sync_state").write_text(
        f"{_sha256(dest / 'a.ipynb')}  a.ipynb\n"
        f"{_sha256(dest / 'remote_only.ipynb')}  remote_only.ipynb\n",
        encoding = "utf-8",
    )

    (dest / "sub").unlink()
    second = _run(tmp_path, template, dest)
    assert second.returncode == 0, second.stdout + second.stderr

    state = _state(dest)
    assert state.get("a.ipynb") == _sha256(
        dest / "a.ipynb"
    ), "the refreshed copy is ours, not the user's; dropping its record freezes it"
    assert state.get("remote_only.ipynb") == _sha256(
        dest / "remote_only.ipynb"
    ), "a notebook that exists only upstream is never walked by the populate loop"
    assert state.get("sub/b.ipynb") == _sha256(dest / "sub" / "b.ipynb")
    assert (dest / "a.ipynb").read_text(
        encoding = "utf-8"
    ) == "A-v2-from-upstream", "the retry must not overwrite the newer copy with the baked template"
    assert len(state) == 3, state


@behavioural
def test_a_first_boot_records_only_what_it_populated(tmp_path: Path):
    """Non-vacuity for the merge above: with no prior state there is nothing to carry
    forward, so a stray file in DEST must not acquire a record."""
    template = _template(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "not_ours.ipynb").write_text("N", encoding = "utf-8")

    assert _run(tmp_path, template, dest).returncode == 0
    assert set(_state(dest)) == {"a.ipynb", "sub/b.ipynb"}


# Phase 1b restores a notebook the user deleted from the BAKED template, and records
# the template's hash for it. When the refresh had already moved that notebook past
# the image, the restore silently walks it backwards -- and phase 2 exits on
# `remote == last`, so the sync marker has to come off or it stays there until
# upstream happens to commit again.

UPSTREAM_COMMIT = "b" * 40


def _restored_run(tmp_path: Path, template: Path, dest: Path, recorded: str, body: str | None):
    """State says `recorded` for a.ipynb; `body` is what is on disk, None to delete it."""
    (dest / ".unsloth_sync_state").write_text(
        f"{recorded}  a.ipynb\n{_sha256(template / 'sub' / 'b.ipynb')}  sub/b.ipynb\n",
        encoding = "utf-8",
    )
    (dest / ".unsloth_sync_commit").write_text(UPSTREAM_COMMIT + "\n", encoding = "utf-8")
    if body is None:
        (dest / "a.ipynb").unlink()
    else:
        (dest / "a.ipynb").write_text(body, encoding = "utf-8")
    return _run(tmp_path, template, dest)


@behavioural
def test_a_notebook_restored_backwards_drops_the_sync_marker(tmp_path: Path):
    template, dest = _template(tmp_path), tmp_path / "dest"
    assert _run(tmp_path, template, dest).returncode == 0

    # the refresh had taken a.ipynb past the baked "A"; the user then deletes it
    run = _restored_run(tmp_path, template, dest, "c" * 64, None)
    assert run.returncode == 0, run.stdout + run.stderr

    assert (dest / "a.ipynb").read_text(encoding = "utf-8") == "A", "it was restored"
    assert not (dest / ".unsloth_sync_commit").is_file(), (
        "the marker survived a downgrade, so the refresh exits on remote == last and "
        "the notebook stays on the image's older copy"
    )
    assert _state(dest)["a.ipynb"] == _sha256(template / "a.ipynb")
    assert "1 needing a refresh" in run.stdout, run.stdout


@behavioural
def test_a_restore_that_changes_nothing_keeps_the_marker(tmp_path: Path):
    """Non-vacuity and the cost control: an ordinary delete of a notebook that never
    moved past the image must not force a full clone on the next start."""
    template, dest = _template(tmp_path), tmp_path / "dest"
    assert _run(tmp_path, template, dest).returncode == 0

    run = _restored_run(tmp_path, template, dest, _sha256(template / "a.ipynb"), None)
    assert run.returncode == 0, run.stdout + run.stderr

    assert (dest / "a.ipynb").read_text(encoding = "utf-8") == "A"
    assert (dest / ".unsloth_sync_commit").read_text(encoding = "utf-8").strip() == UPSTREAM_COMMIT
    assert "0 needing a refresh" in run.stdout, run.stdout


@behavioural
def test_a_notebook_still_on_disk_is_not_touched(tmp_path: Path):
    """The restore is only for files that are GONE; a user's edit stays and keeps the
    marker, because nothing was walked backwards."""
    template, dest = _template(tmp_path), tmp_path / "dest"
    assert _run(tmp_path, template, dest).returncode == 0

    run = _restored_run(tmp_path, template, dest, "c" * 64, "MY OWN WORK")
    assert run.returncode == 0, run.stdout + run.stderr

    assert (dest / "a.ipynb").read_text(encoding = "utf-8") == "MY OWN WORK"
    assert (dest / ".unsloth_sync_commit").read_text(encoding = "utf-8").strip() == UPSTREAM_COMMIT


# --- a stale state staging file the populate cannot truncate --------------------------
# A run killed between the truncate and the mv leaves $STATE.tmp behind. When that run
# was a different uid (root once, then `--user`) the populate cannot empty it, every
# append fails, and the mv -- which needs write on DEST, not on the file -- publishes
# the FOREIGN file as our state. 0444 on a file this uid owns reproduces that, since
# open-for-write consults the owner bits.
@behavioural
def test_a_stale_unwritable_state_temp_is_not_published_as_our_state(tmp_path: Path):
    template = _template(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    stale = dest / ".unsloth_sync_state.tmp"
    stale.write_text("deadbeef  stale.ipynb\n", encoding = "utf-8")
    stale.chmod(0o444)

    run = _run(tmp_path, template, dest)
    assert run.returncode == 0, run.stdout + run.stderr

    state = _state(dest)
    assert (
        "stale.ipynb" not in state
    ), "a leftover from an interrupted run was adopted as the state of this one"
    assert state == {
        "a.ipynb": _sha256(dest / "a.ipynb"),
        "sub/b.ipynb": _sha256(dest / "sub" / "b.ipynb"),
    }, state
    assert (dest / ".unsloth_sync_commit").read_text(encoding = "utf-8").strip() == TEMPLATE_COMMIT
    assert not (dest / ".unsloth_sync_partial").exists()


@behavioural
def test_an_unstageable_state_leaves_the_marker_off_instead_of_copying(tmp_path: Path):
    """Fail CLOSED when the staging path cannot be cleared either, exactly as the
    refresh child does: notebooks we cannot record read as user edits on the next run,
    which is `kept`, not `failed`, so the marker would be stamped over them for good.
    A DIRECTORY at the staging path is unlinkable and untruncatable for root too."""
    template = _template(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / ".unsloth_sync_state.tmp").mkdir()

    run = _run(tmp_path, template, dest)
    assert run.returncode == 0, run.stdout + run.stderr

    assert not (dest / ".unsloth_sync_state").exists(), run.stdout + run.stderr
    assert not (
        dest / ".unsloth_sync_commit"
    ).exists(), "stamping the commit here strands every notebook this run copied"
    assert (dest / ".unsloth_sync_partial").exists()
    assert not (dest / "a.ipynb").exists(), "nothing may be published without a record"
    assert "could not be staged" in run.stdout, run.stdout
