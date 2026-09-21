# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the installer modules are allowed to import just to be imported.

Every `unsloth studio` command loads `install_llama_prebuilt`, and through it
`prebuilt_core`, before it knows whether this run will download anything. The
desktop's launch preflight loads it to stat a directory. So the module-scope
import list is on the launch path, and filelock alone costs about 25 ms because
it pulls in asyncio.

These tests pin the deferral rather than the timing: a wall-clock budget on a
shared runner is a flake, while "filelock is not in sys.modules after the import"
is the same answer on every machine. Each runs in a fresh interpreter, since
anything already imported by the test session would hide a regression.
"""

import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
STUDIO_DIR = PACKAGE_ROOT / "studio"

# Imported by the archive and locking paths, which an import must not reach.
# asyncio is here because it is filelock's cost rather than its own.
DEFERRED = ("filelock", "asyncio", "zipfile", "tarfile")


def _modules_after_importing(statement: str) -> set[str]:
    """The interesting part of sys.modules in a fresh interpreter."""
    probe = (
        f"{statement}\n"
        "import json, sys\n"
        f"print(json.dumps(sorted(m for m in {DEFERRED!r} if m in sys.modules)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output = True,
        text = True,
        cwd = str(PACKAGE_ROOT),
    )
    assert result.returncode == 0, result.stderr
    import json

    return set(json.loads(result.stdout.strip().splitlines()[-1]))


@pytest.mark.parametrize(
    "statement",
    [
        "import sys; sys.path.insert(0, 'studio'); import prebuilt_core",
        "import sys; sys.path.insert(0, 'studio'); import install_llama_prebuilt",
    ],
    ids = ["prebuilt_core", "install_llama_prebuilt"],
)
def test_importing_an_installer_does_not_load_the_download_stack(statement):
    """A marker read, a health check or a `--help` must not pay for filelock."""
    assert _modules_after_importing(statement) == set()


def test_the_lock_still_uses_filelock_when_it_is_installed():
    """Deferred, not dropped: the real lock is what makes concurrent installs safe."""
    sys.path.insert(0, str(STUDIO_DIR))
    import prebuilt_core

    file_lock, file_lock_timeout = prebuilt_core.filelock_classes()
    pytest.importorskip("filelock")
    assert file_lock is not None and file_lock_timeout is not None
    assert file_lock.__module__.startswith("filelock")


def test_a_missing_filelock_still_answers_with_the_pid_fallback(tmp_path, monkeypatch):
    """The ImportError arm, which is what a stripped environment hits.

    Exercised through install_lock rather than by asserting on the tuple, since
    the point of the fallback is that the lock still works.
    """
    sys.path.insert(0, str(STUDIO_DIR))
    import prebuilt_core

    monkeypatch.setattr(prebuilt_core, "_FILELOCK_CLASSES", (None, None))
    lock_path = tmp_path / "install.lock"
    with prebuilt_core.install_lock(lock_path, timeout = 5):
        # The fallback writes a real file so a crashed holder can be spotted.
        assert lock_path.is_file()
    assert not lock_path.exists()


def test_the_verdict_is_cached_so_a_missing_filelock_is_not_reimported(monkeypatch):
    """An ImportError is not cheap to repeat, and install_lock is called per install."""
    sys.path.insert(0, str(STUDIO_DIR))
    import prebuilt_core

    monkeypatch.setattr(prebuilt_core, "_FILELOCK_CLASSES", None)
    monkeypatch.delitem(sys.modules, "filelock", raising = False)

    class CountingBlocker:
        """Refuses filelock and counts how many times it was asked."""

        def __init__(self):
            self.calls = 0

        def find_spec(
            self,
            name,
            path = None,
            target = None,
        ):
            if name == "filelock" or name.startswith("filelock."):
                self.calls += 1
                raise ImportError("filelock is not installed in this environment")
            return None

    blocker = CountingBlocker()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])

    assert prebuilt_core.filelock_classes() == (None, None)
    assert prebuilt_core.filelock_classes() == (None, None)
    # Twice asked, once attempted: the second call read the cache.
    assert blocker.calls == 1, blocker.calls
