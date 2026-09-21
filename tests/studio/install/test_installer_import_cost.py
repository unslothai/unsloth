# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the installer modules may import just to be imported.

Every `unsloth studio` command loads `install_llama_prebuilt`, and the desktop
preflight loads it to stat a directory, so the module-scope import list is on the
launch path; filelock alone costs ~25 ms because it pulls in asyncio.

Pinned as "not in sys.modules" rather than as a wall-clock budget, which would flake
on a shared runner. Fresh interpreter each time: the test session has its own imports.
"""

import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
STUDIO_DIR = PACKAGE_ROOT / "studio"

# asyncio is here because it is filelock's cost, not its own.
DEFERRED = ("filelock", "asyncio", "zipfile", "tarfile")


def _modules_after_importing(statement: str) -> set[str]:
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
    """The ImportError arm, through install_lock: the point is that it still locks."""
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
