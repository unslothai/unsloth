# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""The GitHub refresh must not take a bind-mounted notebook away from its owner.

`unsloth_sync_notebooks.sh` publishes a refreshed notebook by copying the freshly
cloned upstream file to a same-dir staging name and renaming it over the
destination. rename(2) swaps the DIRECTORY ENTRY: the staged inode survives with
its own owner and mode and the destination's inode is discarded, so the staged
copy's root:root 0644 -- inherited from the clone via `cp -a` -- became the
published file's identity.

That matters because a host-owned file really can be under sync management. The
first-boot populate adopts a pre-existing file whose bytes already match the
baked template (the documented `-v $PWD/notebooks:/workspace/unsloth-notebooks`
bind mount of the same checkout) and records it as managed WITHOUT copying,
precisely so `cp -a` does not stamp the baked root:root over the host user's
file. The refresh then undid that: measured in a container against a real bind
mount, a managed notebook went from `65534:65534 0664` to `0:0 0644` across the
publish and the host user could no longer write it.

The fix is the bash twin of the one already applied to `unsloth_run.py`
(`_stage_metadata` before its `os.replace`): give the staged copy the
destination's mode and owner before the rename, best effort. The EBUSY
single-file-bind-mount fallback gets the same treatment -- `cp -a` onto an
existing inode chowns it too, a plain `cp` does not.

Behavioural: runs the real script against a local git "remote". No docker, no
network. The uid half needs root, so it is asserted statically here and was
verified in a container; the mode half is exercised end to end.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNC = REPO_ROOT / "docker" / "unsloth_sync_notebooks.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("git") is None
    or shutil.which("sha256sum") is None,
    reason = "needs bash, git and sha256sum",
)

REL = "nb/Llama.ipynb"
# The managed notebook's mode on the host. Deliberately neither 0644 nor 0664,
# so the assertion holds whatever umask the clone is checked out under.
HOST_MODE = 0o640


def _nb(code: str) -> str:
    return json.dumps(
        {
            "cells": [
                {"cell_type": "code", "source": ["!pip install unsloth\n"], "metadata": {}},
                {"cell_type": "code", "source": [code], "metadata": {}},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )


V1 = _nb("model = FastLanguageModel.from_pretrained('llama-3')\n")
V2 = _nb("model = FastLanguageModel.from_pretrained('llama-4')\n")


def _git(*args, cwd: Path):
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd = str(cwd), check = True, capture_output = True,
    )


def _world(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Baked template, a host bind mount of the same bytes, and an upstream repo."""
    template = tmp_path / "template"
    (template / "nb").mkdir(parents = True)
    (template / REL).write_text(V1, encoding = "utf-8")
    (template / ".unsloth_template_commit").write_text("old\n", encoding = "utf-8")

    dest = tmp_path / "dest"
    (dest / "nb").mkdir(parents = True)
    # Same bytes as the template: populate adopts this as managed and leaves the
    # host user's inode alone.
    (dest / REL).write_text(V1, encoding = "utf-8")
    os.chmod(dest / REL, HOST_MODE)

    remote = tmp_path / "remote"
    (remote / "nb").mkdir(parents = True)
    (remote / REL).write_text(V2, encoding = "utf-8")
    _git("init", "-q", "-b", "main", cwd = remote)
    _git("add", "-A", cwd = remote)
    _git("commit", "-qm", "bump", cwd = remote)
    return template, dest, remote


def _run(tmp_path: Path, template: Path, dest: Path, remote: Path, *, path_prefix = None):
    env = dict(os.environ)
    env.update(
        UNSLOTH_NOTEBOOKS_TEMPLATE = str(template),
        UNSLOTH_NOTEBOOKS_DIR = str(dest),
        UNSLOTH_NOTEBOOKS_REPO = str(remote),
        # Run the refresh inline instead of forking it, so the assertions below
        # are not racing a detached child.
        UNSLOTH_NB_REFRESH_CHILD = "1",
        UNSLOTH_SKIP_NOTEBOOK_VIEW = "1",
        UNSLOTH_KEEP_COLAB_INTRO = "1",
    )
    if path_prefix is not None:
        env["PATH"] = str(path_prefix) + os.pathsep + env["PATH"]
    return subprocess.run(
        ["bash", str(SYNC)], capture_output = True, text = True, env = env, timeout = 120,
    )


def _assert_refreshed_and_still_the_owners(dest: Path, res):
    live = dest / REL
    assert live.read_text(encoding = "utf-8") == V2, (
        f"the upstream change must reach the container; stdout={res.stdout!r} "
        f"stderr={res.stderr!r}"
    )
    mode = stat.S_IMODE(live.stat().st_mode)
    assert mode == HOST_MODE, (
        f"the published notebook kept mode 0o{mode:o}, not the host user's "
        f"0o{HOST_MODE:o}: the refresh replaced their file's identity with the "
        f"freshly cloned copy's"
    )


def test_the_refresh_keeps_the_destinations_metadata(tmp_path: Path):
    template, dest, remote = _world(tmp_path)
    res = _run(tmp_path, template, dest, remote)
    _assert_refreshed_and_still_the_owners(dest, res)


def test_the_ebusy_fallback_keeps_the_destinations_metadata(tmp_path: Path):
    # A single-FILE bind mount cannot be renamed over. Model it with an `mv` that
    # refuses only the staging rename, so the in-place copy fallback runs; it used
    # to be `cp -a`, which chowns the destination inode it writes through.
    binp = tmp_path / "bin"
    binp.mkdir()
    stub = binp / "mv"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in *.unsloth_nb_new.*) exit 1 ;; esac\n'
        'exec /bin/mv "$@"\n',
        encoding = "utf-8",
    )
    stub.chmod(0o755)

    template, dest, remote = _world(tmp_path)
    res = _run(tmp_path, template, dest, remote, path_prefix = binp)
    _assert_refreshed_and_still_the_owners(dest, res)


def test_the_owner_half_is_applied_too(tmp_path: Path):
    # Reproducing the reported 65534 -> 0 needs root, which the suite does not
    # have; the container run that did have it showed exactly that. Pin the chown
    # so a later edit cannot silently drop it and leave only the mode fixed.
    src = SYNC.read_text(encoding = "utf-8")
    block = src[src.index("stage_metadata() {") : src.index("cp_keep_meta() {")]
    assert 'chmod --reference="$2" "$1"' in block
    assert 'chown --reference="$2" "$1"' in block
    assert block.count("|| true") == 2, (
        "both must be best effort: a filesystem that refuses them must not cost "
        "the user their refresh"
    )
    assert '[ -e "$2" ] || return 0' in block, (
        "a brand-new notebook has no destination metadata to inherit"
    )
