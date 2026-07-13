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

import pytest

_SHIM = (
    Path(__file__).resolve().parent.parent
    / "core"
    / "inference"
    / "sandbox_site"
    / "sitecustomize.py"
)


def _load_shim():
    """Import the shim without leaving its open()/mkdir patches installed."""
    saved = (builtins.open, io.open, os.open, os.makedirs, os.mkdir, pathlib.Path.mkdir)
    spec = importlib.util.spec_from_file_location("_sandbox_sitecustomize_under_test", _SHIM)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # runs _install(), patching the globals
    finally:
        # Undo the process-wide patch so the test process stays clean.
        (builtins.open, io.open, os.open, os.makedirs, os.mkdir, pathlib.Path.mkdir) = saved
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


def test_write_fallback_never_clobbers_same_basename(monkeypatch, tmp_path):
    # A same-named file already in the working directory is an unrelated
    # persistent conversation file. Redirecting an invented absolute path (with
    # a missing parent) onto it would truncate/append to data the model never
    # asked to touch. The fallback must REFUSE the redirect and return the
    # original path so the real open() raises FileNotFoundError and the existing
    # file is preserved. Semantics: refuse-on-collision for every create mode.
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)

    existing = tmp_path / "report.txt"
    existing.write_text("KEEP-ME")

    requested = "/definitely_missing_parent_7083/report.txt"
    for mode in ("w", "a", "x", "w+", "a+"):
        # Refused: returns the original absolute path unchanged (no redirect).
        assert mod._remap_open(requested, mode) == requested

    # And opening the refused path really does raise, leaving the file intact.
    with pytest.raises(FileNotFoundError):
        open(mod._remap_open(requested, "w"), "w")
    assert existing.read_text() == "KEEP-ME"

    # No collision -> still healed into the working directory as before.
    fresh = "/definitely_missing_parent_7083/brand_new.txt"
    assert mod._remap_open(fresh, "w") == os.path.join(os.getcwd(), "brand_new.txt")


@pytest.mark.parametrize("mode", ["r+", "rb+"])
def test_read_update_modes_never_redirected_even_with_missing_parent(monkeypatch, tmp_path, mode):
    # r+ / rb+ REQUIRE the target to already exist; they never create. A "+" in
    # the mode must not qualify as creation, or a missing absolute path would be
    # redirected onto a same-basename workspace file and opened for read/update,
    # corrupting unrelated data. The parent here is missing, so only the mode
    # predicate protects the victim.
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)

    victim = tmp_path / "victim.txt"
    victim.write_text("original")

    requested = "/definitely_missing_parent_xyz/victim.txt"
    assert mod._remap_open(requested, mode) == requested
    with pytest.raises(FileNotFoundError):
        open(mod._remap_open(requested, mode), mode)
    assert victim.read_text() == "original"


def test_existing_convention_prefix_is_not_shadowed(monkeypatch, tmp_path):
    # A convention prefix (/mnt/data etc.) is remapped ONLY while it is absent.
    # If a real host directory exists at that prefix it must pass through so its
    # own filesystem semantics apply -- a real read succeeds, a missing file
    # under an EXISTING real prefix is created there by a write, never shadowed
    # by a CWD file.
    mod = _load_shim()
    external = tmp_path / "real_prefix"
    external.mkdir()
    (external / "data.txt").write_text("real external content")

    workdir = tmp_path / "conversation"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(mod, "_PREFIXES", (str(external),))
    monkeypatch.setattr(mod, "_CONDITIONAL_PREFIXES", ())

    target = str(external / "data.txt")
    # Prefix exists -> pass through for read and write.
    assert mod._remap(target) == target
    assert mod._remap_open(target, "r") == target
    assert mod._remap_open(target, "w") == target
    # A missing file under the EXISTING real prefix is left alone (parent
    # exists), so the real directory creates it -- not a CWD shadow.
    missing = str(external / "new.txt")
    assert mod._remap_open(missing, "w") == missing

    # Remove the prefix directory -> healing resumes (absent prefix).
    (external / "data.txt").unlink()
    external.rmdir()
    assert mod._remap(target) == os.path.join(os.getcwd(), "data.txt")


def test_os_open_and_path_touch_remap_convention_path(monkeypatch, tmp_path):
    # Path.touch() and other low-level creators go through os.open, not
    # builtins/io.open. Keep the shim's patches installed (like the mkdir test)
    # under a chdir into tmp_path so os.open is patched, and confirm a
    # convention path is healed into the working directory instead of raising.
    saved = (builtins.open, io.open, os.open, os.makedirs, os.mkdir, pathlib.Path.mkdir)
    spec = importlib.util.spec_from_file_location("_sandbox_sitecustomize_osopen", _SHIM)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(mod)  # installs the os.open patch
        mod._notified = True
        pathlib.Path("/mnt/data/touched.txt").touch()
        assert os.path.isfile(os.path.join(cwd, "touched.txt"))
        # Direct os.open with create flags is healed too.
        fd = os.open("/mnt/data/via_os_open.txt", os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
        assert os.path.isfile(os.path.join(cwd, "via_os_open.txt"))
    finally:
        (builtins.open, io.open, os.open, os.makedirs, os.mkdir, pathlib.Path.mkdir) = saved


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
    saved = (builtins.open, io.open, os.open, os.makedirs, os.mkdir, pathlib.Path.mkdir)
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
        (builtins.open, io.open, os.open, os.makedirs, os.mkdir, pathlib.Path.mkdir) = saved
