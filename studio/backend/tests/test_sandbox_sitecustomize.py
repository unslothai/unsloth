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


def _save_patch_targets():
    """Snapshot every global the shim patches, so tests can restore them.

    On Python < 3.11 the shim also repoints ``pathlib._NormalAccessor.open``
    (pathlib captured the original io.open at import there); the accessor is
    absent on 3.11+, so the snapshot skips it.
    """
    accessor = getattr(pathlib, "_NormalAccessor", None)
    return (
        (builtins.open, io.open, os.open, os.makedirs, os.mkdir, pathlib.Path.mkdir),
        accessor,
        accessor.open if accessor is not None else None,
    )


def _restore_patch_targets(saved):
    """Undo _save_patch_targets so the test process stays clean."""
    globals_tuple, accessor, accessor_open = saved
    (builtins.open, io.open, os.open, os.makedirs, os.mkdir, pathlib.Path.mkdir) = globals_tuple
    if accessor is not None:
        accessor.open = accessor_open


def _load_shim():
    """Import the shim without leaving its open()/mkdir patches installed."""
    saved = _save_patch_targets()
    spec = importlib.util.spec_from_file_location("_sandbox_sitecustomize_under_test", _SHIM)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # runs _install(), patching the globals
    finally:
        # Undo the process-wide patch so the test process stays clean.
        _restore_patch_targets(saved)
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


def test_prefix_remap_contains_parent_traversal_inside_cwd(monkeypatch, tmp_path):
    # A hallucinated habit path can carry '..' in its suffix. The remapped target
    # must stay under the per-conversation CWD, never climbing into a sibling
    # session's directory: '..' components are dropped, the rest of the subpath kept.
    mod = _load_shim()
    workdir = tmp_path / "session_current" / "work"
    workdir.mkdir(parents = True)
    monkeypatch.chdir(workdir)
    cwd = os.getcwd()

    for escaping in (
        "/mnt/data/../other_session/file",
        "/mnt/data/../../secrets.txt",
        "/mnt/data/a/../../b/c.txt",
        "/mnt/data/./sub/./x.txt",
    ):
        mapped = mod._remap(escaping)
        # Never escapes the CWD subtree.
        assert mapped == cwd or mapped.startswith(cwd + os.sep), (escaping, mapped)
        assert os.path.realpath(mapped).startswith(os.path.realpath(cwd))
    # '../other_session/file' collapses to CWD/other_session/file.
    assert mod._remap("/mnt/data/../other_session/file") == os.path.join(
        cwd, "other_session", "file"
    )
    # A bare '/mnt/data/..' with nothing left maps onto the CWD itself.
    assert mod._remap("/mnt/data/..") == cwd


def test_write_fallback_refuses_dotdot_basename(monkeypatch, tmp_path):
    # basename('/no/such/tree/..') == '..'; joining that onto the CWD would target
    # its parent (outside the sandbox). The fallback must refuse such non-filename
    # basenames and return the path unchanged so the real open raises.
    mod = _load_shim()
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    for escaping in ("/no/such/tree/..", "/no/such/tree/.", "/no/such/tree/"):
        assert mod._remap_open(escaping, "w") == escaping


def test_write_fallback_remaps_hallucinated_absolute_path(monkeypatch, tmp_path):
    # Models invent absolute paths from their CWD (e.g. /home/ubuntu/Sandbox/x.html),
    # which prefix lists cannot enumerate. A write/create-mode open on an absolute
    # path outside the CWD whose parent is missing is redirected to the basename in the CWD.
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
    # Reading a real (or genuinely missing) file must succeed/fail truthfully --
    # the fallback is write-only.
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    for mode in ("r", "rb", "r+"):
        assert mod._remap_open("/etc/definitely_missing_xyz.conf", mode) == (
            "/etc/definitely_missing_xyz.conf"
        )


def test_write_fallback_passes_through_existing_external_dir(monkeypatch, tmp_path):
    # A write to an absolute path whose parent dir exists is a deliberate, working
    # target and must NOT be redirected.
    mod = _load_shim()
    external = tmp_path / "external"
    external.mkdir()
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    target = str(external / "out.txt")
    assert mod._remap_open(target, "w") is target


def test_write_fallback_never_clobbers_same_basename(monkeypatch, tmp_path):
    # A same-named CWD file is an unrelated persistent conversation file.
    # Redirecting an invented absolute path (missing parent) onto it would clobber
    # data the model never asked to touch, so the fallback refuses on collision for
    # every create mode: it returns the original path and the real open() raises.
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


def test_write_fallback_reserves_same_target_on_repeated_writes(monkeypatch, tmp_path):
    # Iterative overwrite of the SAME invented path must keep landing on the CWD
    # target the fallback first healed it to. Once ./app.html exists, a naive
    # anti-clobber guard would return the original (parent-missing) path and every
    # regenerate would raise; the fallback must recognise its own prior remap and re-serve it.
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    invented = "/home/ubuntu/Sandbox/app.html"
    target = os.path.join(cwd, "app.html")

    # First write: healed into the CWD, and create the file so the collision guard
    # would trigger on the next call without the fix.
    assert mod._remap_open(invented, "w") == target
    with open(mod._remap_open(invented, "w"), "w") as fh:
        fh.write("v1")

    # Repeated overwrites of the same invented path stay on the same target.
    for _ in range(3):
        assert mod._remap_open(invented, "w") == target
    with open(mod._remap_open(invented, "w"), "w") as fh:
        fh.write("v2")
    assert Path(target).read_text() == "v2"

    # A DIFFERENT invented source colliding on basename is still refused, so it can
    # never clobber the artifact the first path owns.
    other = "/opt/other/app.html"
    assert mod._remap_open(other, "w") == other


def test_write_fallback_reserves_healed_target_across_separate_runs(monkeypatch, tmp_path):
    # Each tool call is a FRESH subprocess, so the in-process remap map is empty on
    # the next run while the healed file persists in the working directory. A second
    # run overwriting the SAME invented path (whose healed basename now exists) must
    # still re-serve that target via the on-disk sidecar, else the model could never
    # overwrite last turn's artifact. Each _load_shim() simulates a brand-new interpreter.
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    invented = "/home/ubuntu/Sandbox/app.html"
    target = os.path.join(cwd, "app.html")

    # Run 1: heal the invented path, create the file, persist source->target to the sidecar.
    run1 = _load_shim()
    assert run1._remap_open(invented, "w") == target
    with open(run1._remap_open(invented, "w"), "w") as fh:
        fh.write("v1")

    # Run 2: brand-new interpreter, nothing in memory -- still recognises its prior
    # heal from the sidecar and re-serves it, even though ./app.html now exists
    # (which without the sidecar would trip the anti-clobber guard and raise).
    run2 = _load_shim()
    assert run2._remapped_writes == {}
    assert run2._remap_open(invented, "w") == target
    with open(run2._remap_open(invented, "w"), "w") as fh:
        fh.write("v2")
    assert Path(target).read_text() == "v2"

    # A DIFFERENT invented source colliding only on basename is still refused across
    # runs: the sidecar records solely the source it healed, so an unrelated path
    # can never adopt/clobber the artifact.
    other = "/opt/other/app.html"
    assert run2._remap_open(other, "w") == other

    # A foreign CWD file (created directly, never healed) stays protected in a later
    # run from an invented path sharing its basename.
    (tmp_path / "notes.txt").write_text("KEEP-ME")
    run3 = _load_shim()
    assert run3._remap_open("/some/missing/notes.txt", "w") == "/some/missing/notes.txt"
    with pytest.raises(FileNotFoundError):
        open(run3._remap_open("/some/missing/notes.txt", "w"), "w")
    assert (tmp_path / "notes.txt").read_text() == "KEEP-ME"


@pytest.mark.parametrize("mode", ["r+", "rb+"])
def test_read_update_modes_never_redirected_even_with_missing_parent(monkeypatch, tmp_path, mode):
    # r+ / rb+ REQUIRE the target to exist and never create; a "+" must not qualify
    # as creation, or a missing absolute path would be redirected onto a same-basename
    # workspace file and corrupt it. The parent is missing, so only the mode predicate
    # protects the victim.
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
    # A convention prefix (/mnt/data etc.) is remapped ONLY while absent. If a real
    # host directory exists there it must pass through so its own filesystem semantics
    # apply: a real read succeeds, and a missing file under it is created there by a
    # write, never shadowed by a CWD file.
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
    # A missing file under the EXISTING real prefix is left alone (parent exists),
    # so the real directory creates it -- not a CWD shadow.
    missing = str(external / "new.txt")
    assert mod._remap_open(missing, "w") == missing

    # Remove the prefix directory -> healing resumes (absent prefix).
    (external / "data.txt").unlink()
    external.rmdir()
    assert mod._remap(target) == os.path.join(os.getcwd(), "data.txt")


def test_os_open_and_path_touch_remap_convention_path(monkeypatch, tmp_path):
    # Path.touch() and other low-level creators go through os.open, not builtins/io.open.
    # Keep the shim's patches installed under a chdir into tmp_path so os.open is
    # patched, and confirm a convention path is healed into the CWD instead of raising.
    saved = _save_patch_targets()
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
        _restore_patch_targets(saved)


def test_path_write_read_text_remap_convention_path(monkeypatch, tmp_path):
    # Path.open / write_text / read_text route through io.open (3.11+) or the captured
    # accessor open (< 3.11). Keep the patches installed under a chdir into tmp_path
    # and confirm a convention path is healed into the CWD on every version. This is
    # the hermetic guard for the 3.10 accessor path a plain io.open patch misses.
    saved = _save_patch_targets()
    spec = importlib.util.spec_from_file_location("_sandbox_sitecustomize_writetext", _SHIM)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(mod)  # installs the io.open / accessor patch
        mod._notified = True
        pathlib.Path("/mnt/data/note.txt").write_text("pathlib remap")
        assert os.path.isfile(os.path.join(cwd, "note.txt"))
        # read_text goes through the same mapped path and sees what was written.
        assert pathlib.Path("/mnt/data/note.txt").read_text() == "pathlib remap"
        # A real absolute path passes through both patches untouched.
        real = tmp_path / "real.txt"
        pathlib.Path(str(real)).write_text("verbatim")
        assert real.read_text() == "verbatim"
    finally:
        _restore_patch_targets(saved)


def test_write_fallback_leaves_relative_and_bytes_paths(monkeypatch, tmp_path):
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    # Relative paths are already inside the CWD.
    assert mod._remap_open("out.txt", "w") == "out.txt"
    # Bytes paths are left untouched (prefix remap skips non-str).
    assert mod._remap_open(b"/no/such/tree/x.bin", "w") == b"/no/such/tree/x.bin"


def test_remap_open_still_applies_prefix_remaps(monkeypatch, tmp_path):
    # The prefix remap runs first and preserves subpaths. A write heals onto the CWD
    # unconditionally; the write-mode fallback is only the last resort.
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    assert mod._remap_open("/mnt/data/sub/out.txt", "w") == os.path.join(cwd, "sub", "out.txt")
    # A read whose mapped target does NOT exist keeps the original path: a missing
    # input stays truthful, not silently redirected into the CWD.
    assert mod._remap_open("/mnt/data/sub/out.txt", "r") == "/mnt/data/sub/out.txt"


def test_prefix_read_heals_only_when_mapped_target_exists(monkeypatch, tmp_path):
    # A convention-prefix READ must not redirect onto the CWD when the mapped target
    # is absent -- that masks a genuine missing-input error and could serve an
    # unrelated same-basename workdir file. It heals only when the mapped CWD target
    # exists, so re-reading an artifact an earlier write produced still works.
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()

    # Mapped target absent: read keeps the original path (truthful miss).
    assert mod._remap_open("/mnt/data/input.csv", "r") == "/mnt/data/input.csv"
    with pytest.raises(FileNotFoundError):
        open(mod._remap_open("/mnt/data/input.csv", "r"))

    # r+ (never creates) behaves the same: no redirect while absent.
    assert mod._remap_open("/mnt/data/input.csv", "r+") == "/mnt/data/input.csv"

    # A write heals onto the CWD and creates the artifact...
    mapped = mod._remap_open("/mnt/data/input.csv", "w")
    assert mapped == os.path.join(cwd, "input.csv")
    with open(mapped, "w") as fh:
        fh.write("col\n1\n")

    # ...and now a READ of the same convention path heals onto that existing artifact.
    read_target = mod._remap_open("/mnt/data/input.csv", "r")
    assert read_target == os.path.join(cwd, "input.csv")
    with open(read_target) as fh:
        assert fh.read() == "col\n1\n"


def test_prefix_boundary_not_matched_by_similar_paths(monkeypatch, tmp_path):
    # The prefix match is anchored on a segment boundary (prefix or prefix + '/'), so
    # a sibling merely sharing the textual prefix must NOT be remapped: /workspace2
    # is not /workspace, /mnt/database is not /mnt/data.
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    for unrelated in ("/workspace2/file.txt", "/mnt/database/x", "/home/sandboxed/y"):
        assert mod._remap(unrelated) == unrelated
        # And through open() for a read too (no silent redirect).
        assert mod._remap_open(unrelated, "r") == unrelated


def test_tmp_outputs_is_a_conditional_prefix():
    mod = _load_shim()
    assert "/tmp/outputs" in mod._CONDITIONAL_PREFIXES
    # NOT in the always-remap set: /tmp exists on the host, so an unconditional remap
    # could shadow a real /tmp/outputs the user code made.
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
    # `Path('/mnt/data').mkdir(parents=True, exist_ok=True)` is a stock setup line.
    # pathlib drives it through os.mkdir per component and Path.is_dir()/os.stat on
    # FileExistsError, so the shim must patch os.mkdir AND Path.mkdir for the whole
    # parents/exist_ok dance to land in the CWD instead of raising. Keeps the mkdir
    # patches installed under a chdir into tmp_path and restores them in finally.
    saved = _save_patch_targets()
    spec = importlib.util.spec_from_file_location("_sandbox_sitecustomize_mkdir", _SHIM)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(mod)  # installs the os.mkdir / Path.mkdir patches
        mod._notified = True
        # Bare convention path maps onto the CWD, which already exists: exist_ok=True
        # must be honoured against the mapped location, not raise.
        pathlib.Path("/mnt/data").mkdir(parents = True, exist_ok = True)
        # A nested convention path is created inside the CWD, parents and all.
        pathlib.Path("/mnt/data/plots/run1").mkdir(parents = True, exist_ok = True)
        assert os.path.isdir(os.path.join(cwd, "plots", "run1"))
        # Idempotent: exist_ok is evaluated on the mapped path (which now exists),
        # not the never-present /mnt/data.
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
        _restore_patch_targets(saved)


def test_read_of_missing_prefix_path_emits_no_notice(monkeypatch, tmp_path, capsys):
    # A read of a missing convention path keeps the original path and must not spend
    # the one-shot notice; a genuine remap afterward still notifies.
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    mod._notified = False  # re-arm the one-shot notice for this test
    # Read of a missing prefixed path: original kept, no notice, flag unspent.
    assert mod._remap_open("/mnt/data/missing.csv", "r") == "/mnt/data/missing.csv"
    assert mod._notified is False
    assert "does not exist" not in capsys.readouterr().err
    # A committed write then heals and fires the notice exactly once.
    assert mod._remap_open("/mnt/data/out.txt", "w") == os.path.join(os.getcwd(), "out.txt")
    assert mod._notified is True
    assert "/mnt/data does not exist in this sandbox" in capsys.readouterr().err


def test_os_open_trunc_without_creat_missing_stays_truthful(monkeypatch, tmp_path):
    # O_TRUNC / O_APPEND without O_CREAT cannot create a missing file, so the shim
    # treats them as a read: a missing convention path stays truthful (the error
    # names the caller's path) and nothing is created in the CWD.
    saved = _save_patch_targets()
    spec = importlib.util.spec_from_file_location("_sandbox_sitecustomize_trunc", _SHIM)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.chdir(tmp_path)
    try:
        spec.loader.exec_module(mod)
        mod._notified = True
        with pytest.raises(FileNotFoundError) as exc:
            os.open("/mnt/data/missing_xyz.bin", os.O_WRONLY | os.O_TRUNC)
        assert exc.value.filename == "/mnt/data/missing_xyz.bin"
        assert not os.path.exists(os.path.join(os.getcwd(), "missing_xyz.bin"))
    finally:
        _restore_patch_targets(saved)
