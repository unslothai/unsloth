# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A ref is a branch, a tag OR a commit, and the installer has to take all three.

CONTRIBUTING-perf.md tells a reader to measure an already-merged change as `merge commit` against
`merge commit^1`. Both of those are commit shas, and `git clone --branch <sha>` resolves against
the remote's advertised branches and tags: it exits with `Remote branch <sha> not found in upstream
origin` before anything is installed. The reused-clone path was broken the same way by
`reset --hard origin/<sha>`, which is not a name that exists either.

Run against a local repository over the filesystem, so the test needs no network.

    python -m pytest tests/studio/studiobench/runtime/selftest/test_studiobench_lifecycle_refs.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.runtime.lifecycle import checkout_ref  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason = "git is not installed")


def _git(
    *args,
    cwd,
    check = True,
):
    return subprocess.run(
        ["git", *args],
        cwd = str(cwd),
        text = True,
        capture_output = True,
        check = check,
        env = {
            "GIT_AUTHOR_NAME": "studiobench",
            "GIT_AUTHOR_EMAIL": "studiobench@example.invalid",
            "GIT_COMMITTER_NAME": "studiobench",
            "GIT_COMMITTER_EMAIL": "studiobench@example.invalid",
            # No user or system git config: an `init.defaultBranch`, a `commit.gpgsign` or a
            # `clone.defaultRemoteName` on the machine running the tests would otherwise decide what this
            # repository looks like.
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "HOME": str(cwd),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        },
    )


def _origin(tmp_path: Path) -> tuple[Path, str, str]:
    """A repository with two commits on `main`. Returns (path, first sha, second sha)."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git("init", "--initial-branch=main", cwd = origin)
    (origin / "install.sh").write_text("first\n")
    _git("add", "-A", cwd = origin)
    _git("commit", "-m", "first", cwd = origin)
    first = _git("rev-parse", "HEAD", cwd = origin).stdout.strip()
    (origin / "install.sh").write_text("second\n")
    _git("commit", "-am", "second", cwd = origin)
    second = _git("rev-parse", "HEAD", cwd = origin).stdout.strip()
    # A bare mirror, because a fetch from a non-bare checkout of the same branch is a special case and
    # the real remote is a server.
    bare = tmp_path / "origin.git"
    _git("clone", "--bare", str(origin), str(bare), cwd = tmp_path)
    return bare, first, second


def test_clone_branch_cannot_take_a_commit(tmp_path):
    """The reason this module does not use `git clone --branch <ref>`, asserted rather than said."""

    bare, first, _second = _origin(tmp_path)
    got = _git(
        "clone", "--branch", first, str(bare), str(tmp_path / "byclone"), cwd = tmp_path, check = False
    )
    assert got.returncode != 0
    assert "not found in upstream origin" in (got.stderr + got.stdout)


def test_checkout_ref_takes_a_branch_a_tag_and_a_commit(tmp_path):
    bare, first, second = _origin(tmp_path)
    _git("tag", "v1", first, cwd = bare)

    repo = tmp_path / "repo"
    _git("clone", str(bare), str(repo), cwd = tmp_path)

    assert checkout_ref(repo, "main") == second
    assert (repo / "install.sh").read_text() == "second\n"

    # THE CASE THAT WAS BROKEN: a bare commit sha, which is what `merge commit^1` resolves to.
    assert checkout_ref(repo, first) == first
    assert (repo / "install.sh").read_text() == "first\n"

    assert checkout_ref(repo, "v1") == first
    # And a local expression the remote will not serve by name.
    assert checkout_ref(repo, f"{second}^1") == first


def test_an_unknown_ref_says_so(tmp_path):
    bare, _first, _second = _origin(tmp_path)
    repo = tmp_path / "repo"
    _git("clone", str(bare), str(repo), cwd = tmp_path)
    with pytest.raises(RuntimeError, match = "could not be resolved"):
        checkout_ref(repo, "no-such-ref")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
