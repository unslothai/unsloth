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
    assert len(_published(dest)) < NOTEBOOKS + 1, "nothing was rolled back"

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
    """The rollback removes OUR copy to make a lost record recoverable. It must never
    touch a notebook the user changed, whose record is also the one most likely to be
    dropped, since an edited file takes the `kept` branch on every refresh."""
    tpl, dest, up = _template(tmp_path), tmp_path / "dest", _upstream(tmp_path)
    dest.mkdir()
    _run(tpl, dest, up, refresh = False)
    _run(tpl, dest, up, refresh = True)  # uncapped: everything published and recorded

    edited = {f"nb{i:09d}.ipynb" for i in range(1, NOTEBOOKS + 1, 25)}
    for name in edited:
        (dest / name).write_text("USER EDIT", encoding = "utf-8")
    _commit_upstream_change(up)

    run = _run(tpl, dest, up, cap_kib = CAP_KIB, refresh = True)
    assert "could not be written" in run.stdout, run.stdout + run.stderr

    lost = {n for n in edited if not (dest / n).exists()}
    assert not lost, f"user edits destroyed by the rollback: {sorted(lost)[:5]}"
    for name in sorted(edited):
        assert (dest / name).read_text(encoding = "utf-8") == "USER EDIT", name
