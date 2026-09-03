# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""The GitHub refresh must not take a bind-mounted notebook away from its owner.

rename(2) swaps the DIRECTORY ENTRY, so the staged inode's root:root 0644 becomes the
published file's identity -- and a host-owned file really can be under sync
management, since first-boot populate adopts one matching the baked template WITHOUT
copying. `cp -a` onto an existing inode chowns it too; plain `cp` does not.
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
    shutil.which("bash") is None
    or shutil.which("git") is None
    or shutil.which("sha256sum") is None,
    reason = "needs bash, git and sha256sum",
)

REL = "nb/Llama.ipynb"
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
        cwd = str(cwd),
        check = True,
        capture_output = True,
    )


def _world(tmp_path: Path) -> tuple[Path, Path, Path]:
    template = tmp_path / "template"
    (template / "nb").mkdir(parents = True)
    (template / REL).write_text(V1, encoding = "utf-8")
    (template / ".unsloth_template_commit").write_text("old\n", encoding = "utf-8")

    dest = tmp_path / "dest"
    (dest / "nb").mkdir(parents = True)
    (dest / REL).write_text(V1, encoding = "utf-8")
    os.chmod(dest / REL, HOST_MODE)

    remote = tmp_path / "remote"
    (remote / "nb").mkdir(parents = True)
    (remote / REL).write_text(V2, encoding = "utf-8")
    _git("init", "-q", "-b", "main", cwd = remote)
    _git("add", "-A", cwd = remote)
    _git("commit", "-qm", "bump", cwd = remote)
    return template, dest, remote


def _run(
    tmp_path: Path,
    template: Path,
    dest: Path,
    remote: Path,
    *,
    path_prefix = None,
):
    env = dict(os.environ)
    env.update(
        UNSLOTH_NOTEBOOKS_TEMPLATE = str(template),
        UNSLOTH_NOTEBOOKS_DIR = str(dest),
        UNSLOTH_NOTEBOOKS_REPO = str(remote),
        UNSLOTH_NB_REFRESH_CHILD = "1",
        UNSLOTH_SKIP_NOTEBOOK_VIEW = "1",
        UNSLOTH_KEEP_COLAB_INTRO = "1",
    )
    if path_prefix is not None:
        env["PATH"] = str(path_prefix) + os.pathsep + env["PATH"]
    return subprocess.run(
        ["bash", str(SYNC)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 120,
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
    # an `mv` that refuses only the staging rename models a single-FILE bind mount
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
    src = SYNC.read_text(encoding = "utf-8")
    block = src[src.index("stage_metadata() {") : src.index("cp_keep_meta() {")]
    assert 'chmod --reference="$2" "$1"' in block
    assert 'chown --reference="$2" "$1"' in block
    assert block.count("|| true") == 2, (
        "both must be best effort: a filesystem that refuses them must not cost "
        "the user their refresh"
    )
    assert (
        '[ -e "$2" ] || return 0' in block
    ), "a brand-new notebook has no destination metadata to inherit"


@pytest.mark.skipif(
    os.geteuid() == 0,
    reason = "root holds CAP_DAC_OVERRIDE, so chmod 0500 does not stop the write",
)
def test_a_failed_publish_does_not_claim_the_commit_is_synced(tmp_path: Path):
    """A publish that cannot be written must stay retryable: stamping $SYNCED anyway
    short-circuits the next boot on `remote == last`. Skipped under root, which
    bypasses the chmod (CAP_DAC_OVERRIDE)."""
    template, dest, remote = _world(tmp_path)
    nb_dir = dest / "nb"
    os.chmod(nb_dir, 0o500)  # publish into nb/ now fails, DEST root stays writable
    try:
        res = _run(tmp_path, template, dest, remote)
    finally:
        os.chmod(nb_dir, 0o700)

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd = remote,
        capture_output = True,
        text = True,
        check = True,
    ).stdout.strip()
    synced = dest / ".unsloth_sync_commit"
    marker = synced.read_text(encoding = "utf-8").strip() if synced.exists() else ""
    assert marker != head, (
        "the sync marker was advanced to the upstream commit even though the "
        "publish failed, so the next start short-circuits on remote == last and "
        f"never retries; marker={marker!r} stdout={res.stdout!r} "
        f"stderr={res.stderr!r}"
    )
    assert (dest / REL).read_text(encoding = "utf-8") == V1


# mkdir(2) gives a new DIRECTORY the caller's uid and only setgid carries down, so a
# category folder upstream adds lands root:root and the user cannot write into it

SYNC_SH = REPO_ROOT / "docker" / "unsloth_sync_notebooks.sh"


def _function_block(source: str, name: str) -> str:
    start = source.index(f"{name}() {{")
    end = source.index("\n}\n", start) + len("\n}\n")
    return source[start:end]


def _drive_mkdir_keep_owner(tmp_path: Path, target: Path) -> list:
    source = SYNC_SH.read_text(encoding = "utf-8")
    block = _function_block(source, "mkdir_keep_owner")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok = True)
    log = tmp_path / "chown.log"
    shim = bin_dir / "chown"
    shim.write_text(
        "#!/usr/bin/env bash\n" f'printf "%s\\n" "$*" >> "{log}"\n' "exit 0\n",
        encoding = "utf-8",
    )
    shim.chmod(0o755)

    driver = tmp_path / "driver.sh"
    driver.write_text(
        "#!/usr/bin/env bash\nset -u\n" + block + f'\nmkdir_keep_owner "{target}"\n',
        encoding = "utf-8",
    )
    result = subprocess.run(
        ["bash", str(driver)],
        capture_output = True,
        text = True,
        timeout = 120,
        env = dict(os.environ, PATH = f"{bin_dir}{os.pathsep}" + os.environ["PATH"]),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if not log.exists():
        return []
    return [line for line in log.read_text(encoding = "utf-8").splitlines() if line]


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_every_created_notebook_directory_is_chowned_to_its_nearest_ancestor(tmp_path: Path):
    anchor = tmp_path / "notebooks"
    anchor.mkdir()
    target = anchor / "AMD" / "vision"

    calls = _drive_mkdir_keep_owner(tmp_path, target)

    assert target.is_dir(), "the directory still has to be created"
    assert calls == [
        f"--reference={anchor} {anchor / 'AMD'}",
        f"--reference={anchor} {target}",
    ], f"both created levels must be fixed, outermost first: {calls}"


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_an_existing_notebook_directory_is_left_alone(tmp_path: Path):
    existing = tmp_path / "notebooks"
    existing.mkdir()
    assert _drive_mkdir_keep_owner(tmp_path, existing) == []


def test_every_directory_creating_site_routes_through_the_helper():
    """Three sites create directories inside $DEST; none may call bare mkdir."""
    source = SYNC_SH.read_text(encoding = "utf-8")
    body = source[source.index("mkdir_keep_owner() {") :]
    body = body[body.index("\n}\n") :]  # everything after the helper itself
    stray = [line.strip() for line in body.splitlines() if "mkdir -p" in line and "$DEST/" in line]
    assert not stray, f"these still create a directory as root inside $DEST: {stray}"
    assert (
        body.count("mkdir_keep_owner ") == 3
    ), "expected populate, restore and publish to route through the helper"
