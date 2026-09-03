# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""First-boot population must not claim a commit it did not finish copying.

`unsloth_sync_notebooks.sh` phase 1 lays the baked notebooks down, records what
it wrote in `.unsloth_sync_state`, and stamps the template commit into
`.unsloth_sync_commit`. A `cp -a` that fails (a destination subpath that is not
a writable directory, a full disk) is skipped silently and gets no state entry,
so phase 1b never restores it. Stamping the commit anyway told the phase 2
refresh that this commit was already synced, so it exited early too, and the
notebook was gone for good on an offline container.

Driven end to end against the real script with the refresh disabled, so these
are offline assertions with no network and no git. The blocked path is a plain
FILE where a directory has to be, which fails for root as well, so the test
does not silently pass under a root CI container the way a chmod would.
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
    (template / ".unsloth_template_commit").write_text(
        TEMPLATE_COMMIT + "\n", encoding = "utf-8"
    )
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
            # Offline: phase 2 needs a clone, and the point here is that the
            # offline half of the retry works on its own.
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
    # `sub` is a regular file, so `mkdir -p dest/sub` and the copy into it both
    # fail with ENOTDIR, for root too.
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

    # The obstruction clears (the operator fixes the mount, the disk drains).
    (dest / "sub").unlink()

    second = _run(tmp_path, template, dest)
    assert second.returncode == 0, second.stdout + second.stderr

    assert (dest / "sub" / "b.ipynb").read_text(encoding = "utf-8") == "B"
    assert _state(dest)["sub/b.ipynb"] == _sha256(dest / "sub" / "b.ipynb")
    assert _state(dest)["a.ipynb"] == _sha256(dest / "a.ipynb")
    assert (dest / ".unsloth_sync_commit").read_text(encoding = "utf-8").strip() == (
        TEMPLATE_COMMIT
    )
    assert not (dest / ".unsloth_sync_partial").exists()


@behavioural
def test_a_clean_populate_stamps_the_commit_and_does_not_re_run(tmp_path: Path):
    # Control: the retry gate must not turn every start back into a populate.
    template = _template(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()

    assert _run(tmp_path, template, dest).returncode == 0
    assert (dest / ".unsloth_sync_commit").read_text(encoding = "utf-8").strip() == (
        TEMPLATE_COMMIT
    )
    assert not (dest / ".unsloth_sync_partial").exists()
    before = _state(dest)
    assert set(before) == {"a.ipynb", "sub/b.ipynb"}

    # An edit the user makes between starts. Phase 1 re-running would drop its
    # managed record (the hash no longer matches the template), so the state
    # staying identical is what proves the block was skipped.
    (dest / "a.ipynb").write_text("edited", encoding = "utf-8")

    assert _run(tmp_path, template, dest).returncode == 0
    assert _state(dest) == before
    assert (dest / "a.ipynb").read_text(encoding = "utf-8") == "edited"


@behavioural
def test_the_partial_marker_is_not_recorded_as_a_notebook(tmp_path: Path):
    # record_state() walks every file under DEST; a new dotfile that is not in
    # its skip list would be published to users as a notebook to sync.
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
