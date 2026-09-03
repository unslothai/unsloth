# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Regression guard for the notebook-sync race in the Unsloth Docker image.

unsloth_sync_notebooks.sh populates /workspace/unsloth-notebooks on boot and then
refreshes from GitHub in a DETACHED child, so container start is never blocked on
a network fetch. The parent forked that child and exited immediately, which fired
its `trap finalize EXIT` -- the Colab-intro strip plus the categorized-view
rebuild -- while the child was concurrently `cp -a`-ing refreshed notebooks into
the same tree and rewriting the same state file. Both processes also ran
build_categorized_view, which tears down and rebuilds the symlink farm.

Six identical fresh-container boots reported "cleaned" 279 / 289 / 293 / 297 /
300 / 306 / 307 / 315 / 330 notebooks; two consecutive `docker run`s of the same
image printed 378 and 372. Worse than the noise, the lost writes were permanent:
a notebook the child copied while the parent was hashing it ended up with a
recorded hash that no longer matched the file, so the strip treated it as
user-edited and skipped it on every later boot. That is where 10 of the 23
notebooks still carrying the Colab intro came from. Setting
UNSLOTH_SKIP_NOTEBOOK_REFRESH=1 -- i.e. never forking the child -- made the
result stable and correctly idempotent, which is what pinned the cause.

The fix keeps the refresh detached and fixes the ORDERING instead: one exclusive
lock covers a whole invocation so the child cannot start work until the parent
has exited, the parent runs the finalize explicitly BEFORE it forks (so the order
holds even on a host without flock), the finalize is run-once, and the child
re-arms it only when the refresh actually copied something.

Static: parses the shell script. No docker, no GPU, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNC = REPO_ROOT / "docker" / "unsloth_sync_notebooks.sh"


@pytest.fixture(scope = "module")
def sync() -> str:
    assert SYNC.is_file(), f"missing {SYNC}"
    return SYNC.read_text(encoding = "utf-8")


def test_the_refresh_is_still_detached(sync: str):
    # The whole point of the child is that a 60s ls-remote + clone must not delay
    # container startup. A fix that simply made the refresh synchronous would
    # pass every other test here and regress boot time.
    assert re.search(
        r'UNSLOTH_NB_REFRESH_CHILD=1 "\$0" >/dev/null 2>&1 &', sync
    ), "the GitHub refresh must stay a detached child"


def test_an_exclusive_lock_serialises_the_two_processes(sync: str):
    assert "lock_acquire()" in sync and "lock_release()" in sync
    assert re.search(
        r"flock -w \"\$LOCK_WAIT\" 9", sync
    ), "the lock must be a real exclusive flock, and must not block forever"


def test_the_lock_is_taken_before_anything_mutates_the_tree(sync: str):
    lock = sync.index("\nlock_acquire\n")
    populate = sync.index("# 1) First-boot populate")
    assert lock < populate, (
        "populate / restore / refresh all rewrite the state file; the lock has to "
        "cover them, not just the strip"
    )


def test_a_missing_flock_degrades_instead_of_hanging(sync: str):
    block = sync[sync.index("lock_acquire()") : sync.index("lock_release()")]
    assert "command -v flock" in block and "return 0" in block, (
        "a host without flock, or a $DEST that cannot hold the lock file, must "
        "fall back to running unlocked rather than failing the boot"
    )


def test_the_parent_finalizes_before_it_forks(sync: str):
    fork = sync.index('UNSLOTH_NB_REFRESH_CHILD=1 "$0"')
    block = sync[sync.index('if [ "${UNSLOTH_NB_REFRESH_CHILD:-0}" != "1" ]; then') : fork]
    assert re.search(r"^\s*finalize\s*$", block, re.M), (
        "the strip and view rebuild must be done BEFORE the child exists; running "
        "them from the EXIT trap after the fork is the race itself"
    )


def test_finalize_runs_at_most_once(sync: str):
    block = sync[sync.index("finalize() {") : sync.index("trap 'finalize; lock_release' EXIT")]
    assert (
        '[ "$_FINALIZED" = "1" ] && return 0' in block
    ), "the explicit pre-fork call and the EXIT trap must not strip twice"
    assert "_FINALIZED=1" in block


def test_the_exit_trap_still_covers_the_early_exits(sync: str):
    # Offline / no-git / UNSLOTH_SKIP_NOTEBOOK_REFRESH all exit before the fork
    # site, and still need the view built.
    assert "trap 'finalize; lock_release' EXIT" in sync


def test_the_child_does_not_repeat_the_parents_finalize(sync: str):
    tail = sync[sync.index("# --- refresh child ---") :]
    assert re.search(r"^_FINALIZED=1\s*$", tail, re.M), (
        "the parent already stripped and built the view for the tree as it "
        "stands; an unconditional second pass makes an up-to-date boot noisy"
    )


def test_the_child_re_arms_the_finalize_only_after_it_copies(sync: str):
    tail = sync[sync.index("refreshed from GitHub") :]
    assert re.search(
        r'if \[ "\$updated" -gt 0 \]; then\s*\n\s*_FINALIZED=0\s*\n\s*finalize', tail
    ), (
        "freshly copied notebooks arrive with the upstream Colab intro and have "
        "to be stripped, but only when something was actually copied"
    )


def test_the_lock_file_is_not_recorded_as_a_notebook(sync: str):
    block = sync[sync.index("record_state() {") :]
    block = block[: block.index("\n}")]
    assert ".unsloth_sync.lock) continue" in block, (
        "the lock file lives in $DEST next to the state file and must be excluded "
        "from the managed-file state like the other metadata"
    )


def test_the_lock_lives_beside_the_state_it_protects(sync: str):
    assert re.search(r'^LOCK="\$DEST/\.unsloth_sync\.lock"', sync, re.M), (
        "keeping the lock in $DEST also serialises two containers sharing the "
        "notebooks volume, which /tmp would not"
    )


# --- concurrent-publish safety ------------------------------------------------
# The detach above is deliberate, but entrypoint.sh runs `sync_notebooks` and then
# `exec "$@"`, so the child is still copying while JupyterLab serves the same tree.
# `cp -a` writes THROUGH the destination inode, so it both exposes half-written
# JSON to a reader and destroys a save made after the recorded-hash check. The
# publish therefore has to go via a same-dir temp plus an atomic rename.


def test_the_refresh_publishes_each_notebook_atomically(sync: str):
    block = sync[sync.index("while IFS= read -r -d '' f; do") :]
    block = block[: block.index("done < <(find")]
    assert re.search(
        r'cp -a "\$f" "\$new"', block
    ), "the refresh must copy into a staging file, not onto the live notebook"
    assert re.search(
        r'mv -f "\$new" "\$dst"', block
    ), "the staged copy must be published with an atomic rename"


def test_the_staging_file_is_hidden_and_beside_the_destination(sync: str):
    assert re.search(r'new="\$\(dirname "\$dst"\)/\.unsloth_nb_new\.\$\$"', sync), (
        "the staging file must be dot-prefixed (invisible in the file browser), "
        "per-PID (two containers on one volume) and in the destination directory "
        "(a rename cannot cross filesystems)"
    )


def test_the_recorded_hash_is_rechecked_immediately_before_publishing(sync: str):
    block = sync[sync.index("while IFS= read -r -d '' f; do") :]
    block = block[: block.index("done < <(find")]
    recheck = block.index('cp -a "$f" "$new"')
    assert re.search(
        r'if \[ -e "\$dst" \] && \[ "\$\(hash_of "\$dst"\)" != "\$\{LAST\[\$rel\]:-\}" \]',
        block[recheck:],
    ), (
        "the earlier check sits before middle_unchanged (a python subprocess), so "
        "the hash has to be re-read once the staging copy is complete or a save "
        "made in between is silently overwritten"
    )


def test_a_pristine_pre_existing_file_is_not_rewritten_on_first_boot(sync: str):
    block = sync[sync.index('if [ ! -f "$STATE" ]; then') :]
    block = block[: block.index('mv "$STATE.tmp" "$STATE"')]
    assert "kept existing user file" in block
    # A bind-mounted file whose bytes already match the template used to fall
    # through to `cp -a`, i.e. --preserve=all stamping root:root, the baked mode
    # and the build mtime onto the host user's own file. Record, don't copy.
    same = block.index("kept existing user file")
    tail = block[same:]
    assert tail.index("$STATE.tmp") < tail.index('cp -a "$TEMPLATE/$rel"'), (
        "an existing file with the template's exact bytes must be recorded as "
        "managed without being copied over"
    )


def test_the_recorded_hash_is_the_staged_copy_not_the_published_file(sync: str):
    # rename(2) is atomic, but the hash taken AFTER it is a second, unprotected
    # read: JupyterLab is already serving $DEST while the refresh child runs, so
    # a save landing between the rename and that read is recorded as the
    # sync-owned pristine version, and the NEXT refresh is then allowed to
    # overwrite the user's work. The staging file is dot-prefixed and per-PID, so
    # hashing it before publishing cannot race with anything.
    block = sync[sync.index("while IFS= read -r -d '' f; do") :]
    block = block[: block.index("done < <(find")]
    assert re.search(
        r'staged="\$\(hash_of "\$new"\)"', block
    ), "the published hash must be taken from the staging copy"
    assert block.index('staged="$(hash_of "$new")"') < block.index(
        'mv -f "$new" "$dst"'
    ), "the staged hash must be taken BEFORE the rename that publishes it"
    publish = block.index('mv -f "$new" "$dst"')
    tail = block[publish:]
    assert re.search(
        r"printf '%s  %s\\n' \"\$staged\" \"\$rel\"", tail
    ), "the state line must record the staged hash, not a re-read of $dst"
    assert not re.search(r"printf '%s  %s\\n' \"\$\(hash_of \"\$dst\"\)\"", tail), (
        "re-reading $dst after the rename adopts whatever save landed in that "
        "window as the pristine version"
    )


# --- behavioural: the same race, driven end to end ---------------------------
# The refresh child is re-entered directly (UNSLOTH_NB_REFRESH_CHILD=1), against
# a LOCAL git remote, with a `mv` shim that performs the real rename and then
# writes the user's bytes -- i.e. the Ctrl+S that lands inside the window.

import hashlib  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

_NEEDS = ("bash", "git", "sha256sum", "mv")

behavioural = pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in _NEEDS),
    reason = "needs bash, git, sha256sum and mv",
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


def _remote_with(tmp_path: Path, body: str) -> Path:
    remote = tmp_path / "remote"
    remote.mkdir()
    _git(remote, "init", "-q", "-b", "main")
    (remote / "x.ipynb").write_text(body, encoding = "utf-8")
    _git(remote, "add", "x.ipynb")
    _git(remote, "commit", "-qm", "one")
    return remote


def _advance(remote: Path, body: str) -> None:
    (remote / "x.ipynb").write_text(body, encoding = "utf-8")
    _git(remote, "add", "x.ipynb")
    _git(remote, "commit", "-qm", "next")


def _env(tmp_path: Path, remote: Path, dest: Path, *, save_bytes: str | None) -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok = True)
    real_mv = shutil.which("mv")
    shim = bin_dir / "mv"
    if save_bytes is None:
        shim.write_text(f'#!/usr/bin/env bash\nexec "{real_mv}" "$@"\n', encoding = "utf-8")
    else:
        # Rename for real, then land the user's save in the window between the
        # rename and the hash the script records. Fires once, on the notebook only.
        shim.write_text(
            "#!/usr/bin/env bash\n"
            f'"{real_mv}" "$@" || exit $?\n'
            'dst="${@: -1}"\n'
            f'if [ "$dst" = "{dest / "x.ipynb"}" ] && [ ! -e "{tmp_path / ".fired"}" ]; then\n'
            f'  : > "{tmp_path / ".fired"}"\n'
            f'  printf %s {save_bytes!r} > "$dst"\n'
            "fi\n",
            encoding = "utf-8",
        )
    shim.chmod(0o755)
    return dict(
        os.environ,
        PATH = f"{bin_dir}{os.pathsep}" + os.environ["PATH"],
        UNSLOTH_NB_REFRESH_CHILD = "1",
        UNSLOTH_NOTEBOOKS_TEMPLATE = str(tmp_path / "template"),
        UNSLOTH_NOTEBOOKS_DIR = str(dest),
        UNSLOTH_NOTEBOOKS_REPO = str(remote),
        UNSLOTH_SKIP_NOTEBOOK_VIEW = "1",
        UNSLOTH_KEEP_COLAB_INTRO = "1",
        UNSLOTH_NOTEBOOK_BODY_AWARE = "0",
    )


def _recorded(dest: Path) -> str:
    for line in (dest / ".unsloth_sync_state").read_text().splitlines():
        parts = line.split("  ", 1)
        if len(parts) == 2 and parts[1] == "x.ipynb":
            return parts[0]
    return ""


def _seed(tmp_path: Path, body: str) -> Path:
    # The baked template has to exist: phase 1 exits early without it, and the
    # refresh child never runs.
    template = tmp_path / "template"
    template.mkdir(exist_ok = True)
    (template / "x.ipynb").write_text(body, encoding = "utf-8")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "x.ipynb").write_text(body, encoding = "utf-8")
    (dest / ".unsloth_sync_state").write_text(
        f"{_sha256(dest / 'x.ipynb')}  x.ipynb\n", encoding = "utf-8"
    )
    (dest / ".unsloth_sync_commit").write_text("0" * 40 + "\n", encoding = "utf-8")
    return dest


@behavioural
def test_a_save_landing_after_the_rename_is_not_recorded_as_pristine(tmp_path: Path):
    remote = _remote_with(tmp_path, "v1")
    _advance(remote, "v2")
    dest = _seed(tmp_path, "v1")
    subprocess.run(
        ["bash", str(SYNC)],
        env = _env(tmp_path, remote, dest, save_bytes = "USER EDIT"),
        capture_output = True,
        text = True,
        timeout = 180,
    )
    live = (dest / "x.ipynb").read_text()
    assert live == "USER EDIT", f"the shim did not land the save: {live!r}"
    assert _recorded(dest) != _sha256(
        dest / "x.ipynb"
    ), "the user's own save was recorded as the sync-owned pristine version"
    assert (
        _recorded(dest) == hashlib.sha256(b"v2").hexdigest()
    ), "the recorded hash must be the bytes this refresh published"


@behavioural
def test_a_save_in_that_window_survives_the_next_refresh(tmp_path: Path):
    # The consequence of the above: with the user's bytes recorded as pristine,
    # the next refresh sees hash(dst) == recorded and overwrites their work.
    remote = _remote_with(tmp_path, "v1")
    _advance(remote, "v2")
    dest = _seed(tmp_path, "v1")
    subprocess.run(
        ["bash", str(SYNC)],
        env = _env(tmp_path, remote, dest, save_bytes = "USER EDIT"),
        capture_output = True,
        text = True,
        timeout = 180,
    )
    _advance(remote, "v3")
    subprocess.run(
        ["bash", str(SYNC)],
        env = _env(tmp_path, remote, dest, save_bytes = None),
        capture_output = True,
        text = True,
        timeout = 180,
    )
    assert (
        dest / "x.ipynb"
    ).read_text() == "USER EDIT", "the user's notebook edit was overwritten by the upstream refresh"


@behavioural
def test_an_unraced_refresh_still_publishes_and_records_upstream(tmp_path: Path):
    # Over-reach guard: without a save in the window the refresh must still
    # update the notebook and record the bytes it wrote.
    remote = _remote_with(tmp_path, "v1")
    _advance(remote, "v2")
    dest = _seed(tmp_path, "v1")
    subprocess.run(
        ["bash", str(SYNC)],
        env = _env(tmp_path, remote, dest, save_bytes = None),
        capture_output = True,
        text = True,
        timeout = 180,
    )
    assert (dest / "x.ipynb").read_text() == "v2"
    assert _recorded(dest) == hashlib.sha256(b"v2").hexdigest()
