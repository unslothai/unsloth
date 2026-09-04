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
    assert (
        live.read_text(encoding = "utf-8") == V2
    ), f"the upstream change must reach the container; stdout={res.stdout!r} stderr={res.stderr!r}"
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
        '#!/usr/bin/env bash\ncase "$*" in *.unsloth_nb_new.*) exit 1 ;; esac\nexec /bin/mv "$@"\n',
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
    # This used to assert `[ -e "$2" ] || return 0`, i.e. that a brand-new
    # notebook was left alone. That was the defect: with nothing to inherit
    # from, the early return published the clone's root:root 0644 and the host
    # user could not edit a notebook upstream had just added. It now falls
    # through to own_like_dir instead.
    assert 'if [ ! -e "$2" ]; then' in block
    assert (
        "own_like_dir" in block
    ), "a brand-new notebook must take the owner of the directory it lands in"


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
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{log}"\nexit 0\n',
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
    """Four sites create $DEST or a directory inside it; none may call bare mkdir."""
    source = SYNC_SH.read_text(encoding = "utf-8")
    # The WHOLE file, not just what follows the helper: `mkdir -p "$DEST"` sat above
    # it, and the root it created as root:root is the anchor every other site
    # inherits from, so scoping this scan to the tail is what let that one through.
    stray = [
        line.strip() for line in source.splitlines() if "mkdir -p" in line and '"$DEST' in line
    ]
    assert not stray, f"these still create a directory as root inside $DEST: {stray}"
    body = source[source.index("mkdir_keep_owner() {") :]
    body = body[body.index("\n}\n") :]  # everything after the helper itself
    # Real invocations only: counting the substring also counted the word where a
    # comment merely names the helper, so prose could satisfy or break this.
    calls = [
        line.strip() for line in body.splitlines() if line.strip().startswith("mkdir_keep_owner ")
    ]
    assert len(calls) == 4, (
        "expected the notebook root, populate, restore and publish to route "
        f"through the helper: {calls}"
    )


# --- a notebook that has no destination to inherit from -------------------------
# stage_metadata returned early when $2 did not exist, so a notebook upstream had
# just added kept the clone's root:root 0644 through the rename, and the two
# `cp -a` copies in populate/restore kept the TEMPLATE's. unsloth_run.py's
# _stage_metadata has had a new-file branch for this since the earlier ownership
# fix; the shell twin did not.


def _drive_sh(tmp_path: Path, snippet: str, *funcs: str) -> list:
    """Run shipped shell functions with `chown` replaced by a recorder on PATH."""
    source = SYNC_SH.read_text(encoding = "utf-8")
    blocks = "\n".join(_function_block(source, name) for name in funcs)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok = True)
    log = tmp_path / "chown.log"
    shim = bin_dir / "chown"
    shim.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{log}"\nexit 0\n',
        encoding = "utf-8",
    )
    shim.chmod(0o755)

    driver = tmp_path / "driver.sh"
    driver.write_text(
        "#!/usr/bin/env bash\nset -u\numask 022\n" + blocks + "\n" + snippet, encoding = "utf-8"
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
def test_a_brand_new_notebook_gets_the_destination_directorys_owner(tmp_path: Path):
    dest_dir = tmp_path / "nb"
    dest_dir.mkdir()
    staged = dest_dir / ".unsloth_nb_new.1"
    staged.write_text("{}", encoding = "utf-8")
    os.chmod(staged, 0o600)  # what the clone / mkstemp hands over

    calls = _drive_sh(
        tmp_path,
        f'stage_metadata "{staged}" "{dest_dir / "new.ipynb"}"\n',
        "own_like_dir",
        "stage_metadata",
    )

    assert calls == [
        f"--reference={dest_dir} {staged}"
    ], f"a new notebook must take the owner of the directory it lands in: {calls}"
    # 0666 & ~022, the mode a plain write would have produced.
    assert stat.S_IMODE(os.stat(staged).st_mode) == 0o644, oct(
        stat.S_IMODE(os.stat(staged).st_mode)
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_an_existing_notebook_still_inherits_from_the_file_not_the_directory(tmp_path: Path):
    dest_dir = tmp_path / "nb"
    dest_dir.mkdir()
    live = dest_dir / "x.ipynb"
    live.write_text("old", encoding = "utf-8")
    staged = dest_dir / ".unsloth_nb_new.1"
    staged.write_text("{}", encoding = "utf-8")

    calls = _drive_sh(
        tmp_path,
        f'stage_metadata "{staged}" "{live}"\n',
        "own_like_dir",
        "stage_metadata",
    )

    assert calls == [f"--reference={live} {staged}"], (
        "the existing-destination branch must keep copying from the FILE, which "
        "is what preserves a mode the user chose themselves"
    )


def test_both_template_copies_hand_the_file_to_the_host_user():
    """The sibling guard: populate and restore both `cp -a` from the template,
    which preserves its root:root 0644, so each needs the ownership fix. Fixing
    the publish path alone is how this class of bug keeps coming back."""
    source = SYNC_SH.read_text(encoding = "utf-8")
    copies = [
        i
        for i, line in enumerate(source.splitlines())
        if 'cp -a "$TEMPLATE/$rel" "$DEST/$rel"' in line
    ]
    assert len(copies) == 2, f"expected the populate and restore copies, got {copies}"
    lines = source.splitlines()
    for i in copies:
        window = "\n".join(lines[i : i + 4])
        assert (
            "own_like_dir" in window
        ), f"the copy at line {i + 1} publishes the template's root:root mode"


# --- the notebook root itself ---------------------------------------------------
# `mkdir -p "$DEST"` ran as root before any helper was involved, so on first boot
# under a host-owned bind mount (UNSLOTH_NOTEBOOKS_DIR=/workspace/host/notebooks,
# with -v $PWD:/workspace/host) the notebook root landed root:root. Every later
# mkdir_keep_owner anchors on the NEAREST EXISTING ancestor and own_like_dir copies
# the owner of the directory a file lands in, so that one root:root directory is
# then inherited by every category folder and every notebook underneath it, and the
# host user cannot edit or delete their own notebooks. unsloth_run.py's
# _makedirs_as_host has always chowned the leaf it creates; the shell twin did not.


@pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")
def test_the_notebook_root_is_created_with_its_ancestors_owner(tmp_path: Path):
    template = tmp_path / "template"
    (template / "nb").mkdir(parents = True)
    (template / REL).write_text(V1, encoding = "utf-8")
    (template / ".unsloth_template_commit").write_text("old\n", encoding = "utf-8")

    host = tmp_path / "host"  # the bind mount, owned by the host user
    host.mkdir()
    dest = host / "notebooks"  # first boot: does not exist yet

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "chown.log"
    shim = bin_dir / "chown"
    shim.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{log}"\nexit 0\n',
        encoding = "utf-8",
    )
    shim.chmod(0o755)

    env = dict(os.environ)
    env.update(
        UNSLOTH_NOTEBOOKS_TEMPLATE = str(template),
        UNSLOTH_NOTEBOOKS_DIR = str(dest),
        UNSLOTH_SKIP_NOTEBOOK_REFRESH = "1",
        UNSLOTH_SKIP_NOTEBOOK_VIEW = "1",
        UNSLOTH_KEEP_COLAB_INTRO = "1",
        PATH = f"{bin_dir}{os.pathsep}" + os.environ["PATH"],
    )
    res = subprocess.run(
        ["bash", str(SYNC)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 120,
    )

    assert (
        dest / REL
    ).is_file(), f"populate must still run; stdout={res.stdout!r} stderr={res.stderr!r}"
    calls = [
        line
        for line in (log.read_text(encoding = "utf-8").splitlines() if log.exists() else [])
        if line
    ]
    assert f"--reference={host} {dest}" in calls, (
        "the notebook root was created as root:root, so every directory and "
        f"notebook under it inherits root ownership from it: {calls}"
    )
