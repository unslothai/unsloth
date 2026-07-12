# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic tests for the sandbox sitecustomize path-remap shim.

The shim (``core/inference/sandbox_site/sitecustomize.py``) runs at interpreter
startup inside every sandboxed tool subprocess and remaps ChatGPT
code-interpreter habit paths (``/mnt/data`` etc.) onto the per-conversation
working directory. Importing it calls ``_install()``, which monkeypatches
``builtins.open`` / ``io.open`` / ``os.makedirs`` / ``os.mkdir`` /
``pathlib.Path.mkdir`` process-wide, so these tests
load it into a throwaway module and restore those globals immediately, then
exercise the pure ``_remap()`` function directly -- no subprocess, and no real
``/mnt`` or ``/tmp`` writes. The mkdir test keeps the patch installed under a
``chdir`` into ``tmp_path`` so the only real writes land in that temp dir.
"""

from __future__ import annotations

import builtins
import importlib.util
import io
import os
import pathlib
from pathlib import Path

_SHIM = (
    Path(__file__).resolve().parent.parent
    / "core"
    / "inference"
    / "sandbox_site"
    / "sitecustomize.py"
)


def _load_shim():
    """Import the shim without leaving its open()/mkdir patches installed."""
    saved = (builtins.open, io.open, os.makedirs, os.mkdir, pathlib.Path.mkdir)
    spec = importlib.util.spec_from_file_location("_sandbox_sitecustomize_under_test", _SHIM)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # runs _install(), patching the globals
    finally:
        # Undo the process-wide patch so the test process stays clean.
        builtins.open, io.open, os.makedirs, os.mkdir, pathlib.Path.mkdir = saved
    mod._notified = True  # silence the one-shot stderr notice in tests
    return mod


def test_always_remap_prefixes_map_into_cwd(monkeypatch, tmp_path):
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    assert mod._remap("/mnt/data/out.txt") == os.path.join(cwd, "out.txt")
    assert mod._remap("/mnt/data") == cwd
    # Unrelated absolute and relative paths pass straight through.
    assert mod._remap("/etc/passwd") == "/etc/passwd"
    assert mod._remap("relative.txt") == "relative.txt"


def test_write_fallback_remaps_hallucinated_absolute_path(monkeypatch, tmp_path):
    # Models invent absolute paths from seeing their CWD (e.g.
    # /home/ubuntu/Sandbox/x.html). Prefix lists cannot enumerate these, so a
    # write/create-mode open on an absolute path outside the CWD whose parent is
    # missing is redirected to the basename in the CWD.
    mod = _load_shim()
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    cwd = os.getcwd()
    hallucinated = "/home/ubuntu/Sandbox/flappy_bird.html"
    for mode in ("w", "a", "x", "w+"):
        assert mod._remap_open(hallucinated, mode) == os.path.join(cwd, "flappy_bird.html")
    # A nested missing tree collapses to just the basename in the CWD.
    assert mod._remap_open("/no/such/tree/report.txt", "w") == os.path.join(cwd, "report.txt")


def test_write_fallback_never_touches_read_modes(monkeypatch, tmp_path):
    # Reading a real system file (or a genuinely missing one) must fail or
    # succeed truthfully -- the fallback is write-only.
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    for mode in ("r", "rb", "r+"):
        assert mod._remap_open("/etc/definitely_missing_xyz.conf", mode) == (
            "/etc/definitely_missing_xyz.conf"
        )


def test_write_fallback_passes_through_existing_external_dir(monkeypatch, tmp_path):
    # A write to an absolute path whose parent directory exists is a deliberate,
    # working target (e.g. a real writable dir) and must NOT be redirected.
    mod = _load_shim()
    external = tmp_path / "external"
    external.mkdir()
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    target = str(external / "out.txt")
    assert mod._remap_open(target, "w") is target


def test_write_fallback_leaves_relative_and_bytes_paths(monkeypatch, tmp_path):
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    # Relative paths are already inside the CWD.
    assert mod._remap_open("out.txt", "w") == "out.txt"
    # Bytes paths are left untouched (os.getcwd() is str; prefix remap skips
    # non-str too).
    assert mod._remap_open(b"/no/such/tree/x.bin", "w") == b"/no/such/tree/x.bin"


def test_remap_open_still_applies_prefix_remaps(monkeypatch, tmp_path):
    # The prefix remaps run first in open(), for reads and writes alike, and
    # preserve subpaths -- the write-mode fallback is only the last resort.
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    assert mod._remap_open("/mnt/data/sub/out.txt", "w") == os.path.join(cwd, "sub", "out.txt")
    assert mod._remap_open("/mnt/data/sub/out.txt", "r") == os.path.join(cwd, "sub", "out.txt")


def test_tmp_outputs_is_a_conditional_prefix():
    mod = _load_shim()
    assert "/tmp/outputs" in mod._CONDITIONAL_PREFIXES
    # It must NOT be in the always-remap set: /tmp exists on the host, so an
    # unconditional remap could shadow a real /tmp/outputs the user code made.
    assert "/tmp/outputs" not in mod._PREFIXES


def test_tmp_outputs_remapped_only_while_absent(monkeypatch, tmp_path):
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    # Point the conditional prefix at a real temp location so we can toggle its
    # existence on disk instead of mocking os.path.exists.
    cond = str(tmp_path / "cond_outputs")
    monkeypatch.setattr(mod, "_CONDITIONAL_PREFIXES", (cond,))

    # Absent: heal the habit path into the working directory (preserved/served).
    assert not os.path.exists(cond)
    assert mod._remap(cond + "/plot.png") == os.path.join(cwd, "plot.png")
    assert mod._remap(cond) == cwd

    # Present (the user's own code created it): pass through, never shadowed.
    os.makedirs(cond)
    assert mod._remap(cond + "/plot.png") == cond + "/plot.png"
    assert mod._remap(cond) == cond


def test_pathlib_mkdir_parents_remaps_convention_path(monkeypatch, tmp_path):
    # `Path('/mnt/data').mkdir(parents=True, exist_ok=True)` is a stock
    # code-interpreter setup line. pathlib drives it through os.mkdir (not
    # os.makedirs) per component and, on FileExistsError, Path.is_dir()/os.stat
    # -- so the shim must patch os.mkdir AND Path.mkdir for the whole
    # parents/exist_ok dance to land in the working directory instead of raising
    # before any open() runs. This keeps the shim's mkdir patches installed
    # (unlike _load_shim) under a chdir into tmp_path, so the only real writes
    # land in that temp dir, and restores every patched global in finally.
    saved = (builtins.open, io.open, os.makedirs, os.mkdir, pathlib.Path.mkdir)
    spec = importlib.util.spec_from_file_location("_sandbox_sitecustomize_mkdir", _SHIM)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(mod)  # installs the os.mkdir / Path.mkdir patches
        mod._notified = True
        # Bare convention path maps onto the CWD itself, which already exists:
        # exist_ok=True must be honoured against the mapped location, not raise.
        pathlib.Path("/mnt/data").mkdir(parents = True, exist_ok = True)
        # A nested convention path is created inside the CWD, parents and all.
        pathlib.Path("/mnt/data/plots/run1").mkdir(parents = True, exist_ok = True)
        assert os.path.isdir(os.path.join(cwd, "plots", "run1"))
        # Re-running the same setup is idempotent: exist_ok is evaluated on the
        # mapped path (which now exists), not the never-present /mnt/data.
        pathlib.Path("/mnt/data/plots/run1").mkdir(parents = True, exist_ok = True)

        # Passthrough: real paths are created verbatim through both patches,
        # never remapped into the CWD.
        real_dir = tmp_path / "real_via_path"
        pathlib.Path(str(real_dir)).mkdir()
        assert real_dir.is_dir()
        real_os = tmp_path / "real_via_os"
        os.mkdir(str(real_os))
        assert real_os.is_dir()
    finally:
        builtins.open, io.open, os.makedirs, os.mkdir, pathlib.Path.mkdir = saved
