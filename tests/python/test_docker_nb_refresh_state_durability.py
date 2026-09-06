# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""A refreshed notebook must never end up on disk without a state record.

That single combination is unrecoverable. Withholding the commit marker is not
enough, because the truncated state is still published: the NEXT refresh reads the
unrecorded notebook as a user edit, keeps it, finds nothing failed, and stamps the
marker over it, so it is unmanaged for good.

Driven end to end against the real script with a local repository standing in for
upstream. RLIMIT_FSIZE fills the disk part-way through, sized so the clone still
fits: git's index costs ceil((62 + len(path)) / 8) * 8 per entry while the state
costs 67 + len(path), so a 17-character name makes the state outgrow the index and
opens a window between them. Without that the clone always dies first and the state
write is never reached.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNC = REPO_ROOT / "docker" / "unsloth_sync_notebooks.sh"

NOTEBOOKS = 500
CAP_KIB = 40  # between the clone's 40067-byte index and the 42000-byte state

needs_git = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("sha256sum") is None,
    reason = "the refresh path needs git and sha256sum",
)


def _upstream(tmp_path: Path) -> Path:
    up = tmp_path / "up"
    up.mkdir()
    for i in range(1, NOTEBOOKS + 1):
        (up / f"nb{i:09d}.ipynb").write_text("N", encoding = "utf-8")
    env = dict(os.environ, GIT_CONFIG_GLOBAL = "/dev/null", GIT_CONFIG_SYSTEM = "/dev/null")
    subprocess.run(["git", "init", "-q", "."], cwd = up, check = True, env = env)
    subprocess.run(["git", "add", "-A"], cwd = up, check = True, env = env)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"],
        cwd = up,
        check = True,
        env = env,
    )
    return up


def _template(tmp_path: Path) -> Path:
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / "seed.ipynb").write_text("T", encoding = "utf-8")
    (tpl / ".unsloth_template_commit").write_text("a" * 40 + "\n", encoding = "utf-8")
    return tpl


def _run(
    tpl: Path,
    dest: Path,
    up: Path,
    *,
    cap_kib: int | None = None,
    refresh: bool,
    keep_removed: bool = False,
):
    def _cap():
        import resource
        import signal

        # SIGXFSZ would kill the script; the shell has to SEE the write error
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        n = cap_kib * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (n, n))

    env = dict(
        os.environ,
        UNSLOTH_NOTEBOOKS_TEMPLATE = str(tpl),
        UNSLOTH_NOTEBOOKS_DIR = str(dest),
        UNSLOTH_NOTEBOOKS_REPO = str(up),
        UNSLOTH_SKIP_NOTEBOOK_VIEW = "1",
        UNSLOTH_KEEP_COLAB_INTRO = "1",
    )
    if keep_removed:
        env["UNSLOTH_KEEP_REMOVED_NOTEBOOKS"] = "1"
    if refresh:
        # run the refresh inline; the real one detaches and discards its output
        env["UNSLOTH_NB_REFRESH_CHILD"] = "1"
    else:
        env["UNSLOTH_SKIP_NOTEBOOK_REFRESH"] = "1"
    return subprocess.run(
        ["bash", str(SYNC)],
        capture_output = True,
        text = True,
        timeout = 600,
        env = env,
        preexec_fn = _cap if cap_kib else None,
    )


def _recorded(dest: Path) -> set[str]:
    state = dest / ".unsloth_sync_state"
    if not state.exists():
        return set()
    out = set()
    for line in state.read_text(encoding = "utf-8").splitlines():
        _, _, rel = line.partition("  ")
        if rel:
            out.add(rel)
    return out


def _published(dest: Path) -> set[str]:
    return {p.name for p in dest.glob("*.ipynb")}


@needs_git
def test_no_notebook_is_published_without_a_record(tmp_path: Path):
    """The invariant. A file we wrote but could not record is the unrecoverable
    state, so it must be rolled back rather than left behind."""
    tpl, dest, up = _template(tmp_path), tmp_path / "dest", _upstream(tmp_path)
    dest.mkdir()
    _run(tpl, dest, up, refresh = False)

    run = _run(tpl, dest, up, cap_kib = CAP_KIB, refresh = True)
    assert "could not be written" in run.stdout, (
        "the cap did not bite; this test proves nothing unless some append failed\n"
        + run.stdout
        + run.stderr
    )

    orphans = _published(dest) - _recorded(dest) - {"seed.ipynb"}
    assert not orphans, (
        f"{len(orphans)} notebook(s) on disk with no state record; the next refresh "
        f"reads them as user edits and stops updating them: {sorted(orphans)[:5]}"
    )


@needs_git
def test_the_next_refresh_recovers_everything_that_was_rolled_back(tmp_path: Path):
    """Rollback is only correct if the retry actually restores them."""
    tpl, dest, up = _template(tmp_path), tmp_path / "dest", _upstream(tmp_path)
    dest.mkdir()
    _run(tpl, dest, up, refresh = False)

    first = _run(tpl, dest, up, cap_kib = CAP_KIB, refresh = True)
    assert "could not be written" in first.stdout, first.stdout + first.stderr
    # strictly fewer than the full set: `< NOTEBOOKS + 1` was vacuous, since the
    # run yields NOTEBOOKS either way and seed.ipynb is dropped as deleted upstream
    assert (
        len(_published(dest)) < NOTEBOOKS
    ), "nothing was rolled back, so the retry below proves nothing"

    second = _run(tpl, dest, up, refresh = True)
    assert "kept (your edits)" in second.stdout, second.stdout
    kept = int(second.stdout.split("updated, ")[1].split(" kept")[0])
    assert kept == 0, (
        f"{kept} notebook(s) became user-owned after a disk-full refresh; they would "
        f"never be updated again\n{second.stdout}"
    )
    # seed.ipynb is template-only, so the refresh drops it as deleted upstream
    assert len(_recorded(dest)) == NOTEBOOKS, sorted(_recorded(dest))[:5]


def _commit_upstream_change(up: Path):
    """A second commit so the next refresh does not exit early on remote == last."""
    (up / f"nb{NOTEBOOKS + 1:09d}.ipynb").write_text("N", encoding = "utf-8")
    env = dict(os.environ, GIT_CONFIG_GLOBAL = "/dev/null", GIT_CONFIG_SYSTEM = "/dev/null")
    subprocess.run(["git", "add", "-A"], cwd = up, check = True, env = env)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "more"],
        cwd = up,
        check = True,
        env = env,
    )


@needs_git
def test_a_user_edited_notebook_is_never_rolled_back(tmp_path: Path):
    """The rollback removes OUR copy so a lost record is recoverable. It must never
    touch a notebook the user changed.

    Every SECOND notebook is edited, not every twenty-fifth. The cap bites near the
    end of the walk, so a sparse edited set can miss the failing range entirely and
    the rollback is then never asked about a user edit -- which is how the first
    version of this test passed without executing the code it named. The assertion
    that some pristine notebook WAS removed is what keeps it honest.
    """
    tpl, dest, up = _template(tmp_path), tmp_path / "dest", _upstream(tmp_path)
    dest.mkdir()
    _run(tpl, dest, up, refresh = False)
    _run(tpl, dest, up, refresh = True)  # uncapped: everything published and recorded
    assert len(_published(dest)) == NOTEBOOKS

    edited = {f"nb{i:09d}.ipynb" for i in range(2, NOTEBOOKS + 1, 2)}
    for name in edited:
        (dest / name).write_text("USER EDIT", encoding = "utf-8")
    pristine = {f"nb{i:09d}.ipynb" for i in range(1, NOTEBOOKS + 1, 2)}
    _commit_upstream_change(up)

    run = _run(tpl, dest, up, cap_kib = CAP_KIB, refresh = True)
    assert "could not be written" in run.stdout, run.stdout + run.stderr

    survivors = _published(dest)
    # non-vacuity: the rollback has to have actually fired somewhere in this run
    assert pristine - survivors, (
        "no notebook was rolled back, so this test never asked the rollback about a "
        "user edit and proves nothing"
    )
    lost = edited - survivors
    assert not lost, f"user edits destroyed by the rollback: {sorted(lost)[:5]}"
    for name in sorted(edited):
        assert (dest / name).read_text(
            encoding = "utf-8"
        ) == "USER EDIT", f"{name} was overwritten while the disk was full"


# ---------------------------------------------------------------------------
# The refresh child reads $STATE too, and IT is the copy that runs by default.
# The guard added for the section 1b reader did not cover it, and the test that
# was supposed to prove it passed only because its helper sets
# UNSLOTH_SKIP_NOTEBOOK_REFRESH=1. These deliberately do not.
# ---------------------------------------------------------------------------
def _json_upstream(tmp_path: Path, count: int) -> Path:
    """Valid notebook JSON, so the body-aware comparison can report SAME and the
    `unchanged` branch is reachable. With one-byte files it never is, which is why
    the rollback below had no coverage."""
    import json

    up = tmp_path / "up"
    up.mkdir()
    body = json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["print(1)\n"],
                    "metadata": {},
                    "outputs": [],
                    "execution_count": None,
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )
    for i in range(1, count + 1):
        (up / f"nb{i:09d}.ipynb").write_text(body, encoding = "utf-8")
    env = dict(os.environ, GIT_CONFIG_GLOBAL = "/dev/null", GIT_CONFIG_SYSTEM = "/dev/null")
    subprocess.run(["git", "init", "-q", "."], cwd = up, check = True, env = env)
    subprocess.run(["git", "add", "-A"], cwd = up, check = True, env = env)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"],
        cwd = up,
        check = True,
        env = env,
    )
    return up


@needs_git
def test_the_refresh_child_never_republishes_an_unreadable_state_as_empty(tmp_path: Path):
    """No UNSLOTH_SKIP_NOTEBOOK_REFRESH here: that flag is what hid this."""
    tpl, dest, up = _template(tmp_path), tmp_path / "dest", _upstream(tmp_path)
    dest.mkdir()
    _run(tpl, dest, up, refresh = False)
    _run(tpl, dest, up, refresh = True)
    before = _recorded(dest)
    assert len(before) == NOTEBOOKS

    state = dest / ".unsloth_sync_state"
    state.chmod(0o000)
    try:
        run = _run(tpl, dest, up, refresh = True)
    finally:
        state.chmod(0o644)

    assert run.returncode == 0, run.stdout + run.stderr
    assert state.stat().st_size > 0, (
        "the refresh child published an EMPTY state over a valid one; every notebook "
        "it described is now read as a user edit and frozen\n" + run.stdout
    )
    assert _recorded(dest) == before, run.stdout
    assert "0 updated" not in run.stdout, (
        "the refresh ran with an empty LAST, which is the defect itself\n" + run.stdout
    )


@needs_git
def test_a_failed_record_on_an_unchanged_notebook_rolls_it_back(tmp_path: Path):
    """drop_unrecordable's removal, which nothing exercised before.

    Its other call sites either hand it a user-edited file (hash gate declines) or a
    missing one (early return), so the suite could call it seven times and roll back
    zero. The `unchanged` branch is the reachable one that must actually remove.
    """
    tpl, dest = _template(tmp_path), tmp_path / "dest"
    up = _json_upstream(tmp_path, NOTEBOOKS)
    dest.mkdir()
    _run(tpl, dest, up, refresh = False)
    _run(tpl, dest, up, refresh = True)
    assert len(_published(dest)) == NOTEBOOKS

    _commit_upstream_change(up)
    run = _run(tpl, dest, up, cap_kib = CAP_KIB, refresh = True)
    assert "could not be written" in run.stdout, run.stdout + run.stderr
    assert "kept (only header/footer changed upstream)" in run.stdout, run.stdout

    survivors = _published(dest)
    rolled_back = NOTEBOOKS - len([n for n in survivors if n.startswith("nb")])
    assert rolled_back > 0, (
        "no unchanged notebook was rolled back, so drop_unrecordable's removal is "
        "still unexercised\n" + run.stdout
    )
    # and the retry has to restore them
    second = _run(tpl, dest, up, refresh = True)
    assert len(_published(dest)) == NOTEBOOKS + 1, second.stdout


@needs_git
def test_keeping_a_removed_notebook_keeps_its_record_too(tmp_path: Path):
    """UNSLOTH_KEEP_REMOVED_NOTEBOOKS kept the FILE and dropped its RECORD, so the
    next refresh read it as a user edit -- and turning the option back off never
    recovered it, because by then it is no longer in the state."""
    tpl, dest, up = _template(tmp_path), tmp_path / "dest", _upstream(tmp_path)
    dest.mkdir()
    _run(tpl, dest, up, refresh = False)
    _run(tpl, dest, up, refresh = True)
    victim = f"nb{1:09d}.ipynb"
    assert victim in _recorded(dest)

    (up / victim).unlink()
    env = dict(os.environ, GIT_CONFIG_GLOBAL = "/dev/null", GIT_CONFIG_SYSTEM = "/dev/null")
    subprocess.run(["git", "add", "-A"], cwd = up, check = True, env = env)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "del"],
        cwd = up,
        check = True,
        env = env,
    )

    run = _run(tpl, dest, up, refresh = True, keep_removed = True)
    assert (dest / victim).exists(), "the opt-out did not keep the file: " + run.stdout
    assert victim in _recorded(dest), (
        "the file was kept but its record was dropped, so the next refresh reads a "
        "notebook the user never touched as a user edit\n" + run.stdout
    )


@needs_git
def test_an_unrecorded_notebook_identical_to_upstream_is_adopted(tmp_path: Path):
    """When the publish rollback cannot unlink -- a single-FILE bind mount gives
    EBUSY there, the same case the rename at line 562 already works around -- the
    published bytes stay on disk with no record, and no further append is possible in
    that run. Rolling the bytes back instead would not help, because the file is
    still unrecorded and so still read as a user edit, and discarding the staged
    state would strand every notebook the run DID record. The only repair is for a
    later refresh to notice that a file identical to the clone was never a user edit
    and take it back under management."""
    tpl, dest, up = _template(tmp_path), tmp_path / "dest", _upstream(tmp_path)
    dest.mkdir()
    _run(tpl, dest, up, refresh = False)
    _run(tpl, dest, up, refresh = True)

    orphan = f"nb{7:09d}.ipynb"
    state = dest / ".unsloth_sync_state"
    surviving = [
        ln
        for ln in state.read_text(encoding = "utf-8").splitlines()
        if not ln.endswith("  " + orphan)
    ]
    state.write_text("\n".join(surviving) + "\n", encoding = "utf-8")
    # the run that lost the record withheld the marker, so the next start refreshes
    (dest / ".unsloth_sync_commit").unlink(missing_ok = True)
    assert orphan not in _recorded(dest)
    assert (dest / orphan).exists()

    run = _run(tpl, dest, up, refresh = True)
    assert orphan in _recorded(dest), (
        "a notebook byte-identical to the clone but missing from the state was read "
        "as a user edit, so it stays unmanaged and stops tracking upstream for "
        "good\n" + run.stdout + run.stderr
    )


@needs_git
def test_a_genuinely_edited_unrecorded_notebook_is_still_left_alone(tmp_path: Path):
    """The adoption above must key on the content matching the clone EXACTLY. An
    unrecorded file whose bytes differ is the real user-edit case and must keep its
    hands-off treatment."""
    tpl, dest, up = _template(tmp_path), tmp_path / "dest", _upstream(tmp_path)
    dest.mkdir()
    _run(tpl, dest, up, refresh = False)
    _run(tpl, dest, up, refresh = True)

    orphan = f"nb{7:09d}.ipynb"
    state = dest / ".unsloth_sync_state"
    surviving = [
        ln
        for ln in state.read_text(encoding = "utf-8").splitlines()
        if not ln.endswith("  " + orphan)
    ]
    state.write_text("\n".join(surviving) + "\n", encoding = "utf-8")
    (dest / ".unsloth_sync_commit").unlink(missing_ok = True)
    (dest / orphan).write_text("MINE, do not touch", encoding = "utf-8")

    run = _run(tpl, dest, up, refresh = True)
    assert (dest / orphan).read_text(encoding = "utf-8") == "MINE, do not touch", (
        "an unrecorded notebook the user had edited was overwritten\n" + run.stdout + run.stderr
    )
    assert orphan not in _recorded(dest), (
        "an unrecorded notebook the user had edited was adopted into the state\n" + run.stdout
    )


def _head(up: Path) -> str:
    env = dict(os.environ, GIT_CONFIG_GLOBAL = "/dev/null", GIT_CONFIG_SYSTEM = "/dev/null")
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd = up,
        check = True,
        env = env,
        capture_output = True,
        text = True,
    )
    return out.stdout.strip()


def _delete_upstream(up: Path, victim: str) -> None:
    (up / victim).unlink()
    env = dict(os.environ, GIT_CONFIG_GLOBAL = "/dev/null", GIT_CONFIG_SYSTEM = "/dev/null")
    subprocess.run(["git", "add", "-A"], cwd = up, check = True, env = env)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "del"],
        cwd = up,
        check = True,
        env = env,
    )


@needs_git
def test_a_kept_removed_notebook_survives_a_failed_state_append(tmp_path: Path):
    """Every other caller may delete a notebook it could not record, because the clone
    still holds a copy to re-publish next start. This one may not: upstream DELETED
    the file, so the copy in $DEST is the last one in existence. Dropping it destroys
    exactly what UNSLOTH_KEEP_REMOVED_NOTEBOOKS was set to preserve, and no retry
    recovers it -- not with the option on, not with it off, because by then it is in
    neither the clone nor the state."""
    tpl, dest, up = _template(tmp_path), tmp_path / "dest", _upstream(tmp_path)
    dest.mkdir()
    _run(tpl, dest, up, refresh = False)
    _run(tpl, dest, up, refresh = True)
    victim = f"nb{1:09d}.ipynb"
    assert victim in _recorded(dest)
    _delete_upstream(up, victim)

    # The cap has to let the main loop's 499 records through and fail on the removal
    # loop's single extra append, or the run aborts before ever reaching the branch.
    # A record costs 67 + len(rel) = 84 B, so the window is 499*84 = 41916 <= cap <
    # 500*84 = 42000, and 41 KiB = 41984 is the only multiple of 1024 inside it.
    run = _run(tpl, dest, up, cap_kib = 41, refresh = True, keep_removed = True)

    assert (dest / victim).exists(), (
        "the last surviving copy of a notebook the user asked to KEEP was deleted "
        "because its record would not fit\n" + run.stdout + run.stderr
    )
    assert not (dest / ".unsloth_sync_commit").exists() or (
        dest / ".unsloth_sync_commit"
    ).read_text(encoding = "utf-8").strip() != _head(up), (
        "the record was lost but the run still counted as a success and stamped the "
        "marker, so the next start exits early instead of retrying\n" + run.stdout
    )


# ---------------------------------------------------------------------------
# Section 1b and the populate retry, both driven with the refresh switched off so
# only the parent's own state writers are in play.
# ---------------------------------------------------------------------------
def _big_template(tmp_path: Path, count: int) -> Path:
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    for i in range(1, count + 1):
        (tpl / f"nb{i:09d}.ipynb").write_text("N", encoding = "utf-8")
    (tpl / ".unsloth_template_commit").write_text("a" * 40 + "\n", encoding = "utf-8")
    return tpl


@needs_git
def test_an_abandoned_restore_puts_the_tree_back(tmp_path: Path):
    """Section 1b restores notebooks that are missing, then rewrites the state. When
    that rewrite is abandoned the old state is kept, and it describes the tree as it
    was BEFORE the restores -- so the restored files have to go back too. Leaving them
    means the next refresh sees baked content where the state holds post-refresh
    hashes and reads every one as a user edit."""
    tpl, dest, up = _big_template(tmp_path, NOTEBOOKS), tmp_path / "dest", _upstream(tmp_path)
    dest.mkdir()
    _run(tpl, dest, up, refresh = False)
    before = _recorded(dest)
    assert len(before) == NOTEBOOKS

    # the LAST 50, so they straddle the point where the appends start failing;
    # victims processed before it are all restored no matter what the guard does
    victims = [f"nb{i:09d}.ipynb" for i in range(NOTEBOOKS - 49, NOTEBOOKS + 1)]
    for name in victims:
        (dest / name).unlink()

    run = _run(tpl, dest, up, cap_kib = CAP_KIB, refresh = False)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "could not be rewritten" in run.stdout, (
        "the cap did not bite in section 1b; this test proves nothing\n" + run.stdout
    )

    assert _recorded(dest) == before, "the old state must be kept intact\n" + run.stdout
    still_there = [n for n in victims if (dest / n).exists()]
    assert not still_there, (
        f"{len(still_there)} notebook(s) were restored while the state that describes "
        f"them was abandoned, so the next refresh reads them as user edits: "
        f"{still_there[:5]}"
    )

    # and it must STOP restoring once the rewrite is known to be doomed, rather than
    # keep touching the tree and rely on the undo to clean up after it
    restored = (
        int(run.stdout.split("restored ")[1].split(" ")[0]) if "restored " in run.stdout else 0
    )
    assert restored < len(victims), (
        f"section 1b restored all {restored} notebooks after deciding the state could "
        f"not be written; the guard is evaluated once instead of per iteration"
    )


@needs_git
def test_a_lost_merge_record_does_not_publish_the_short_state(tmp_path: Path):
    """The merge loop is the only source of records for notebooks that exist upstream
    but not in the baked template. Nothing can re-derive them, so a lost one must
    abandon the staged state rather than publish it and merely withhold the marker.
    A failed COPY is different and still publishes, because the next boot re-walks the
    template."""
    tpl, dest, up = _big_template(tmp_path, NOTEBOOKS), tmp_path / "dest", _upstream(tmp_path)
    dest.mkdir()
    _run(tpl, dest, up, refresh = False)

    # notebooks the refresh had added: present in $DEST and in the state, absent from
    # the baked template, so only the merge loop can carry their records forward
    state = dest / ".unsloth_sync_state"
    extra = [f"up_only_{i:04d}.ipynb" for i in range(100)]
    import hashlib

    with state.open("a", encoding = "utf-8") as f:
        for name in extra:
            (dest / name).write_text("U", encoding = "utf-8")
            f.write(hashlib.sha256(b"U").hexdigest() + "  " + name + "\n")
    (dest / ".unsloth_sync_partial").write_text("", encoding = "utf-8")

    run = _run(tpl, dest, up, cap_kib = CAP_KIB + 2, refresh = False)
    assert run.returncode == 0, run.stdout + run.stderr

    survived = _recorded(dest)
    lost = [n for n in extra if n not in survived]
    assert not lost, (
        f"{len(lost)} upstream-only record(s) were dropped and the short state was "
        f"published anyway; nothing can re-derive them: {lost[:5]}\n" + run.stdout
    )
