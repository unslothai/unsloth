# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""openssh-server's postinst writes host keys at build time; the published image must not carry them."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER = REPO_ROOT / "docker"
STUDIO = DOCKER / "Dockerfile.studio"
LAUNCH = DOCKER / "studio_launch.sh"


def _runs(path: Path) -> list[str]:
    text = re.sub(r"\\\r?\n", " ", path.read_text(encoding = "utf-8"))
    return [ln.strip()[4:] for ln in text.splitlines() if ln.strip().startswith("RUN ")]


def _sshd_run() -> str:
    (run,) = [r for r in _runs(STUDIO) if "openssh-server" in r]
    return run


def _cleanup_steps() -> list[str]:
    return [s.strip() for s in _sshd_run().split("&&") if "/etc/ssh/ssh_host_" in s]


def test_host_keys_are_dropped_in_the_layer_that_creates_them():
    # A later RUN would only whiteout the keys; the lower layer still ships them.
    run = _sshd_run()
    steps = _cleanup_steps()
    assert steps == ["rm -f /etc/ssh/ssh_host_*"], steps
    assert run.index("openssh-server") < run.index(steps[0])


def test_nothing_after_the_install_bakes_keys_back():
    later = " ".join(_runs(STUDIO)[_runs(STUDIO).index(_sshd_run()) + 1 :])
    assert "ssh-keygen" not in later
    assert "openssh-server" not in later
    assert "dpkg-reconfigure" not in later


def test_the_launcher_mints_keys_before_enabling_sshd():
    launch = LAUNCH.read_text(encoding = "utf-8")
    gate = launch.index('if [[ -n "${PUBLIC_SSH_KEY}" ]]')
    keygen = launch.index("ssh-keygen -A", gate)
    enable = launch.index("export UNSLOTH_ENABLE_SSHD=true", gate)
    assert keygen < enable


def _fingerprint(root: Path) -> str:
    out = subprocess.run(
        ["ssh-keygen", "-lf", str(root / "etc/ssh/ssh_host_ed25519_key.pub")],
        capture_output = True,
        text = True,
        check = True,
    )
    return out.stdout.split()[1]


def _boot(image: Path, container: Path) -> str:
    shutil.copytree(image, container)
    # studio_launch.sh runs `ssh-keygen -A`, which only fills in missing keys.
    subprocess.run(["ssh-keygen", "-A", "-f", str(container)], check = True, capture_output = True)
    return _fingerprint(container)


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("ssh-keygen") is None,
    reason = "runs the Dockerfile step under a POSIX sh with ssh-keygen",
)
@pytest.mark.parametrize("cleanup", [True, False], ids = ["dockerfile", "no-cleanup"])
def test_each_container_gets_its_own_host_key(tmp_path: Path, cleanup: bool):
    image = tmp_path / "image"
    (image / "etc/ssh").mkdir(parents = True)
    subprocess.run(["ssh-keygen", "-A", "-f", str(image)], check = True, capture_output = True)
    built = _fingerprint(image)
    if cleanup:
        for step in _cleanup_steps():
            subprocess.run(["sh", "-c", step.replace("/etc/ssh", f"{image}/etc/ssh")], check = True)
        assert not list((image / "etc/ssh").glob("ssh_host_*"))

    a = _boot(image, tmp_path / "a")
    b = _boot(image, tmp_path / "b")
    if cleanup:
        assert len({built, a, b}) == 3
    else:
        assert a == b == built
