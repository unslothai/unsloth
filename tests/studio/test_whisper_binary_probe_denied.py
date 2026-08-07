# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An unreadable whisper.cpp install must read as engine-unavailable, not raise.

setup.ps1 now leaves a denied `<STUDIO_HOME>/whisper.cpp` in place instead of
aborting the run, so the backend is the first thing to probe it. `Path.is_file()`
propagates EACCES, and `stt_status` does not catch it, so an unguarded probe turns
into a 500 on the one endpoint that reports *both* dictation engines.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SIDECAR = REPO_ROOT / "studio" / "backend" / "core" / "inference" / "stt_ggml_sidecar.py"


def _is_runnable():
    """Exec the real function alone: importing the module pulls in the whole
    backend (structlog, fastapi), which this check does not need."""
    tree = ast.parse(SIDECAR.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_is_runnable":
            namespace: dict = {"os": os, "sys": sys, "Path": Path}
            exec(compile(ast.Module([node], []), str(SIDECAR), "exec"), namespace)
            return namespace["_is_runnable"]
    raise AssertionError("_is_runnable not found in stt_ggml_sidecar.py")


def _can_deny() -> bool:
    """Probe rather than infer. Guessing from euid silently drops the only
    behavioural test in any root container, which is most CI images."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        locked = Path(tmp) / "locked"
        locked.mkdir()
        (locked / "probe").write_text("", encoding = "utf-8")
        locked.chmod(0o000)
        try:
            (locked / "probe").is_file()
            return False
        except OSError:
            return True
        finally:
            locked.chmod(0o755)


denial_capable = pytest.mark.skipif(
    sys.platform == "win32" or not _can_deny(),
    reason = "this host cannot produce a read denial, so the check would pass vacuously",
)


def test_a_readable_executable_is_still_runnable(tmp_path):
    binary = tmp_path / "whisper-server"
    binary.write_text("", encoding = "utf-8")
    binary.chmod(0o755)
    assert _is_runnable()(binary) is True


def test_a_missing_binary_is_not_runnable(tmp_path):
    assert _is_runnable()(tmp_path / "whisper-server") is False


@denial_capable
def test_an_unreadable_install_dir_reads_as_unavailable_not_an_exception(tmp_path):
    install = tmp_path / "whisper.cpp"
    install.mkdir()
    binary = install / "whisper-server"
    binary.write_text("", encoding = "utf-8")
    binary.chmod(0o755)
    install.chmod(0o000)
    try:
        # Negative control: the denial is real on this filesystem.
        with pytest.raises(OSError):
            binary.is_file()
        assert _is_runnable()(binary) is False
    finally:
        install.chmod(0o755)
