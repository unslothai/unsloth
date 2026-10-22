# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An upload into a sandbox never hard-links a host file in, even if a tool swaps the in-flight temp name for a
symlink while the bytes are being written."""

import os
from pathlib import Path

import pytest

from core.inference import tools

pytestmark = pytest.mark.skipif(
    not hasattr(os, "symlink") or os.name == "nt", reason = "needs POSIX symlinks"
)


def _swapping_chunks(workdir: Path, secret: Path):
    yield b"first "
    # What a tool running in the sandbox can do mid-upload: replace the temp name with a link to a host file.
    (tmp,) = [p for p in workdir.iterdir() if p.name.startswith(".tmp-")]
    tmp.unlink()
    tmp.symlink_to(secret)
    yield b"second"


def test_a_swapped_temp_symlink_is_not_followed_into_the_sandbox(tmp_path):
    workdir = tmp_path / "sandbox"
    workdir.mkdir()
    secret = tmp_path / "auth.db"
    secret.write_bytes(b"JWT-SECRET")

    try:
        tools._install_upload(str(workdir), "up.txt", _swapping_chunks(workdir, secret), False)
    except Exception:
        pass  # refusing the upload is fine too; what must not happen is the link below

    dest = workdir / "up.txt"
    if os.path.lexists(dest):
        assert os.lstat(dest).st_ino != os.stat(secret).st_ino
        assert not (dest.is_file() and not dest.is_symlink() and dest.read_bytes() == b"JWT-SECRET")
    assert secret.read_bytes() == b"JWT-SECRET"
    assert os.stat(secret).st_nlink == 1


def test_a_normal_upload_is_a_regular_file(tmp_path):
    workdir = tmp_path / "sandbox"
    workdir.mkdir()
    assert (
        tools._install_upload(str(workdir), "up.txt", iter([b"hello ", b"world"]), False) is False
    )
    dest = workdir / "up.txt"
    assert dest.is_file() and not dest.is_symlink()
    assert dest.read_bytes() == b"hello world"
    assert [p.name for p in workdir.iterdir()] == ["up.txt"]
