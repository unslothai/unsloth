# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The baked IPython profile must stay writable under `docker run --user <uid>`.

`ENV IPYTHONDIR=/opt/unsloth-nb/ipython` is what makes the notebook startup hook
load for EVERY kernel regardless of uid. But IPython does not fail loudly on a
profile it cannot write: `IPython.paths.get_ipython_dir()` checks the directory
with `os.access(..., W_OK)` and, when it is not writable, warns once on stderr and
substitutes `tempfile.mkdtemp()`. A fresh temp directory has no `profile_default/
startup/`, so the hook never runs -- no transformers sidecar activation, no
`%pip` / `%uv` magic, no colab-compat -- and the notebook still executes, wrongly,
with nothing in the output to say so.

Making only the TOP directory writable is worse: `ProfileDir.check_dirs()` then
creates `security/`, `log/`, `pid/` inside `profile_default` and raises
PermissionError, which kills the kernel before it replies to kernel_info.

So this replays the Dockerfile's own profile-setup commands into a temp tree,
re-maps each path's OTHER bits onto its owner bits (which is exactly what a
foreign uid sees), and then asks the installed IPython what it would really do.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "docker"
DOCKERFILE = DOCKER_DIR / "Dockerfile"

_IPY_DIR = "/opt/unsloth-nb/ipython"


def _profile_setup_commands() -> list[str]:
    """The `mkdir` / `cp` / `chmod` steps of the RUN layer that builds the profile."""
    text = DOCKERFILE.read_text(encoding = "utf-8")
    # the layer is a single `RUN set -eux \` ... continuation block
    start = text.index("COPY unsloth_nb_compat.py")
    block = text[start : text.index("\nENV PATH=", start)]
    joined = block.replace("\\\n", "\n")
    cmds = []
    for raw in joined.split("\n"):
        line = raw.strip()
        line = re.sub(r"^&&\s*", "", line)
        if not line or _IPY_DIR not in line:
            continue
        if line.split(" ", 1)[0] not in ("mkdir", "cp", "chmod"):
            continue
        cmds.append(line)
    return cmds


def _as_foreign_uid(root: Path) -> None:
    """Give every path the permissions a DIFFERENT uid would get.

    The image builds this tree as root with a 022 umask, so a container started
    with `--user <uid>` sees only the `other` bits. The test process owns the tree,
    and `os.access` answers on the OWNER bits, so copy other->owner to make the
    question the same one.
    """
    # deepest first: dropping a directory's owner bits would hide its children
    paths = sorted([root, *root.rglob("*")], key = lambda p: len(p.parts), reverse = True)
    for path in paths:
        mode = stat.S_IMODE(path.stat().st_mode)
        other = mode & 0o007
        path.chmod((mode & ~0o700) | (other << 6))


_PROBE = textwrap.dedent(
    """
    import json, os, sys
    from IPython.paths import get_ipython_dir
    from IPython.core.profiledir import ProfileDir

    out = {"ipdir": get_ipython_dir(), "error": None, "startup": []}
    try:
        pd = ProfileDir.find_profile_dir_by_name(out["ipdir"], "default")
        pd.check_dirs()
        out["startup"] = sorted(os.listdir(pd.startup_dir))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    print("PROBE" + json.dumps(out))
    """
)


@pytest.fixture(scope = "module")
def staged_profile(tmp_path_factory):
    pytest.importorskip("IPython")
    root = tmp_path_factory.mktemp("unsloth-nb")
    shutil.copy(DOCKER_DIR / "unsloth_ipython_startup.py", root / "unsloth_ipython_startup.py")
    cmds = _profile_setup_commands()
    assert cmds, "no profile setup commands found in the Dockerfile"
    assert any(c.startswith("cp ") for c in cmds), "the startup hook is no longer copied in"
    script = "set -eu\n" + "\n".join(c.replace("/opt/unsloth-nb", str(root)) for c in cmds)
    subprocess.run(["bash", "-c", script], check = True)
    _as_foreign_uid(root / "ipython")
    yield root / "ipython"
    # hand the owner bits back, or pytest cannot remove its own tmp tree
    for path in sorted(root.rglob("*"), key = lambda p: len(p.parts), reverse = True):
        try:
            path.chmod(stat.S_IMODE(path.stat().st_mode) | 0o700)
        except OSError:
            pass


def _probe(ipython_dir: Path) -> dict:
    env = dict(os.environ)
    env["IPYTHONDIR"] = str(ipython_dir)
    env.pop("PYTHONWARNINGS", None)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env = env,
        capture_output = True,
        text = True,
        timeout = 300,
    )
    line = [l for l in proc.stdout.splitlines() if l.startswith("PROBE")]
    assert line, f"probe produced no result:\n{proc.stdout}\n{proc.stderr}"
    import json

    return json.loads(line[0][len("PROBE") :])


def test_ipython_keeps_the_baked_profile_for_a_non_root_uid(staged_profile: Path):
    got = _probe(staged_profile)
    assert got["ipdir"] == str(staged_profile), (
        "IPython discarded IPYTHONDIR and fell back to "
        f"{got['ipdir']}: the profile is not writable by a non-root uid, so the "
        "startup hook silently does not load under `docker run --user <uid>`"
    )
    assert got["error"] is None, (
        f"IPython could not use the baked profile: {got['error']}. The kernel dies "
        "here; ProfileDir creates security/, log/ and pid/ inside profile_default"
    )
    assert (
        "00-unsloth-nb.py" in got["startup"]
    ), f"the startup hook is not in the profile IPython uses: {got['startup']}"


def test_the_shared_startup_hook_stays_read_only(staged_profile: Path):
    hook = staged_profile / "profile_default" / "startup" / "00-unsloth-nb.py"
    assert hook.is_file()
    for path in (hook, hook.parent):
        mode = stat.S_IMODE(path.stat().st_mode)
        assert not mode & 0o022, (
            f"{path} is writable by other uids; the shared hook runs in every kernel "
            "and must not be replaceable from a notebook"
        )
    # profile_default has to be world-writable for the test above to pass; without
    # the sticky bit that also lets any uid rename startup/ out of the way
    parent = stat.S_IMODE((staged_profile / "profile_default").stat().st_mode)
    if parent & 0o002:
        assert parent & stat.S_ISVTX, (
            "profile_default is world-writable without the sticky bit, so any uid "
            "can unlink or rename the read-only startup hook"
        )
