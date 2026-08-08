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
    return SYNC.read_text()


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
