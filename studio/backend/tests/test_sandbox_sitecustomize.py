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
        spec.loader.exec_module(mod)
    finally:
        # Undo the process-wide patch so the test process stays clean.
        _restore_patch_targets(saved)
    mod._notified = True
    return mod


def test_always_remap_prefixes_map_into_cwd(monkeypatch, tmp_path):
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    assert mod._remap("/mnt/data/out.txt") == os.path.join(cwd, "out.txt")
    assert mod._remap("/mnt/data") == cwd
    assert mod._remap("/etc/passwd") == "/etc/passwd"
    assert mod._remap("relative.txt") == "relative.txt"


def test_prefix_remap_contains_parent_traversal_inside_cwd(monkeypatch, tmp_path):
    # A hallucinated habit path can carry '..' in its suffix: dropped, so it cannot climb into a sibling session.
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
        assert mapped == cwd or mapped.startswith(cwd + os.sep), (escaping, mapped)
        assert os.path.realpath(mapped).startswith(os.path.realpath(cwd))
    assert mod._remap("/mnt/data/../other_session/file") == os.path.join(
        cwd, "other_session", "file"
    )
    assert mod._remap("/mnt/data/..") == cwd


def test_write_fallback_refuses_dotdot_basename(monkeypatch, tmp_path):
    # basename('/no/such/tree/..') == '..', so joining it targets the parent: refuse non-filename basenames.
    mod = _load_shim()
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    for escaping in ("/no/such/tree/..", "/no/such/tree/.", "/no/such/tree/"):
        assert mod._remap_open(escaping, "w") == escaping


def test_write_fallback_remaps_hallucinated_absolute_path(monkeypatch, tmp_path):
    # Models invent absolute paths from their CWD, which no prefix list enumerates, so a write open takes the basename.
    mod = _load_shim()
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    cwd = os.getcwd()
    hallucinated = "/home/ubuntu/Sandbox/flappy_bird.html"
    for mode in ("w", "a", "x", "w+"):
        assert mod._remap_open(hallucinated, mode) == os.path.join(cwd, "flappy_bird.html")
    assert mod._remap_open("/no/such/tree/report.txt", "w") == os.path.join(cwd, "report.txt")


def test_write_fallback_never_touches_read_modes(monkeypatch, tmp_path):
    # The fallback is write-only: reading a real or genuinely missing file must succeed or fail truthfully.
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    for mode in ("r", "rb", "r+"):
        assert mod._remap_open("/etc/definitely_missing_xyz.conf", mode) == (
            "/etc/definitely_missing_xyz.conf"
        )


def test_write_fallback_passes_through_existing_external_dir(monkeypatch, tmp_path):
    # A write to an absolute path whose parent exists is a deliberate target and must NOT be redirected.
    mod = _load_shim()
    external = tmp_path / "external"
    external.mkdir()
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    target = str(external / "out.txt")
    assert mod._remap_open(target, "w") is target


def test_write_fallback_never_clobbers_same_basename(monkeypatch, tmp_path):
    # Redirecting an invented path onto a same-named CWD file would clobber data, so it refuses on collision.
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)

    existing = tmp_path / "report.txt"
    existing.write_text("KEEP-ME")

    requested = "/definitely_missing_parent_7083/report.txt"
    for mode in ("w", "a", "x", "w+", "a+"):
        assert mod._remap_open(requested, mode) == requested

    with pytest.raises(FileNotFoundError):
        open(mod._remap_open(requested, "w"), "w")
    assert existing.read_text() == "KEEP-ME"

    fresh = "/definitely_missing_parent_7083/brand_new.txt"
    assert mod._remap_open(fresh, "w") == os.path.join(os.getcwd(), "brand_new.txt")


def test_write_fallback_reserves_same_target_on_repeated_writes(monkeypatch, tmp_path):
    # Once ./app.html exists a naive anti-clobber guard raises on every regenerate, so it re-serves its own remap.
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    invented = "/home/ubuntu/Sandbox/app.html"
    target = os.path.join(cwd, "app.html")

    assert mod._remap_open(invented, "w") == target
    with open(mod._remap_open(invented, "w"), "w") as fh:
        fh.write("v1")

    for _ in range(3):
        assert mod._remap_open(invented, "w") == target
    with open(mod._remap_open(invented, "w"), "w") as fh:
        fh.write("v2")
    assert Path(target).read_text() == "v2"

    # A DIFFERENT invented source colliding on basename is still refused, so it cannot clobber the artifact.
    other = "/opt/other/app.html"
    assert mod._remap_open(other, "w") == other


def test_write_fallback_reserves_healed_target_across_separate_runs(monkeypatch, tmp_path):
    # Each tool call is a FRESH subprocess, so the in-process map is empty; the on-disk sidecar allows the overwrite.
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    invented = "/home/ubuntu/Sandbox/app.html"
    target = os.path.join(cwd, "app.html")

    run1 = _load_shim()
    assert run1._remap_open(invented, "w") == target
    with open(run1._remap_open(invented, "w"), "w") as fh:
        fh.write("v1")

    # A brand-new interpreter still recognises its prior heal from the sidecar, which would else trip the guard.
    run2 = _load_shim()
    assert run2._remapped_writes == {}
    assert run2._remap_open(invented, "w") == target
    with open(run2._remap_open(invented, "w"), "w") as fh:
        fh.write("v2")
    assert Path(target).read_text() == "v2"

    # The sidecar records solely the source it healed, so an unrelated path can never adopt the artifact across runs.
    other = "/opt/other/app.html"
    assert run2._remap_open(other, "w") == other

    # A foreign CWD file, created directly and never healed, stays protected in a later run.
    (tmp_path / "notes.txt").write_text("KEEP-ME")
    run3 = _load_shim()
    assert run3._remap_open("/some/missing/notes.txt", "w") == "/some/missing/notes.txt"
    with pytest.raises(FileNotFoundError):
        open(run3._remap_open("/some/missing/notes.txt", "w"), "w")
    assert (tmp_path / "notes.txt").read_text() == "KEEP-ME"


@pytest.mark.parametrize("mode", ["r+", "rb+"])
def test_read_update_modes_never_redirected_even_with_missing_parent(monkeypatch, tmp_path, mode):
    # r+ / rb+ require the target and never create, so a "+" must not qualify as creation.
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
    # A convention prefix is remapped ONLY while absent: a real directory passes through with its own semantics.
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
    assert mod._remap(target) == target
    assert mod._remap_open(target, "r") == target
    assert mod._remap_open(target, "w") == target
    # A missing file under an EXISTING real prefix is created there, not shadowed by a CWD file.
    missing = str(external / "new.txt")
    assert mod._remap_open(missing, "w") == missing

    (external / "data.txt").unlink()
    external.rmdir()
    assert mod._remap(target) == os.path.join(os.getcwd(), "data.txt")


def test_os_open_and_path_touch_remap_convention_path(monkeypatch, tmp_path):
    # Path.touch() goes through os.open, not io.open, so the patches stay installed under a chdir into tmp_path.
    saved = _save_patch_targets()
    spec = importlib.util.spec_from_file_location("_sandbox_sitecustomize_osopen", _SHIM)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(mod)
        mod._notified = True
        pathlib.Path("/mnt/data/touched.txt").touch()
        assert os.path.isfile(os.path.join(cwd, "touched.txt"))
        fd = os.open("/mnt/data/via_os_open.txt", os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
        assert os.path.isfile(os.path.join(cwd, "via_os_open.txt"))
    finally:
        _restore_patch_targets(saved)


def test_path_write_read_text_remap_convention_path(monkeypatch, tmp_path):
    # Path.open / write_text route through io.open (3.11+) or the captured accessor open: this guards the 3.10 path.
    saved = _save_patch_targets()
    spec = importlib.util.spec_from_file_location("_sandbox_sitecustomize_writetext", _SHIM)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(mod)
        mod._notified = True
        pathlib.Path("/mnt/data/note.txt").write_text("pathlib remap")
        assert os.path.isfile(os.path.join(cwd, "note.txt"))
        assert pathlib.Path("/mnt/data/note.txt").read_text() == "pathlib remap"
        real = tmp_path / "real.txt"
        pathlib.Path(str(real)).write_text("verbatim")
        assert real.read_text() == "verbatim"
    finally:
        _restore_patch_targets(saved)


def test_write_fallback_leaves_relative_and_bytes_paths(monkeypatch, tmp_path):
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    assert mod._remap_open("out.txt", "w") == "out.txt"
    # Bytes paths are left untouched (the prefix remap skips non-str).
    assert mod._remap_open(b"/no/such/tree/x.bin", "w") == b"/no/such/tree/x.bin"


def test_remap_open_still_applies_prefix_remaps(monkeypatch, tmp_path):
    # The prefix remap runs first and preserves subpaths; the write-mode fallback is only the last resort.
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    assert mod._remap_open("/mnt/data/sub/out.txt", "w") == os.path.join(cwd, "sub", "out.txt")
    # A read whose mapped target does not exist keeps the original path, so a missing input stays truthful.
    assert mod._remap_open("/mnt/data/sub/out.txt", "r") == "/mnt/data/sub/out.txt"


def test_prefix_read_heals_only_when_mapped_target_exists(monkeypatch, tmp_path):
    # Redirecting an absent convention-prefix READ would mask a missing-input error, so it heals only if it exists.
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()

    assert mod._remap_open("/mnt/data/input.csv", "r") == "/mnt/data/input.csv"
    with pytest.raises(FileNotFoundError):
        open(mod._remap_open("/mnt/data/input.csv", "r"))

    assert mod._remap_open("/mnt/data/input.csv", "r+") == "/mnt/data/input.csv"

    mapped = mod._remap_open("/mnt/data/input.csv", "w")
    assert mapped == os.path.join(cwd, "input.csv")
    with open(mapped, "w") as fh:
        fh.write("col\n1\n")

    read_target = mod._remap_open("/mnt/data/input.csv", "r")
    assert read_target == os.path.join(cwd, "input.csv")
    with open(read_target) as fh:
        assert fh.read() == "col\n1\n"


def test_prefix_boundary_not_matched_by_similar_paths(monkeypatch, tmp_path):
    # The prefix match is anchored on a segment boundary: /workspace2 is not /workspace.
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    for unrelated in ("/workspace2/file.txt", "/mnt/database/x", "/home/sandboxed/y"):
        assert mod._remap(unrelated) == unrelated
        assert mod._remap_open(unrelated, "r") == unrelated


def test_tmp_outputs_is_a_conditional_prefix():
    mod = _load_shim()
    assert "/tmp/outputs" in mod._CONDITIONAL_PREFIXES
    # /tmp exists on the host, so an unconditional remap could shadow a real /tmp/outputs the user code made.
    assert "/tmp/outputs" not in mod._PREFIXES


def test_tmp_outputs_remapped_only_while_absent(monkeypatch, tmp_path):
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    # Point the conditional prefix at a real temp location so existence toggles on disk, without mocking.
    cond = str(tmp_path / "cond_outputs")
    monkeypatch.setattr(mod, "_CONDITIONAL_PREFIXES", (cond,))

    assert not os.path.exists(cond)
    assert mod._remap(cond + "/plot.png") == os.path.join(cwd, "plot.png")
    assert mod._remap(cond) == cwd

    os.makedirs(cond)
    assert mod._remap(cond + "/plot.png") == cond + "/plot.png"
    assert mod._remap(cond) == cond


def test_pathlib_mkdir_parents_remaps_convention_path(monkeypatch, tmp_path):
    # pathlib drives mkdir(parents=True, exist_ok=True) through os.mkdir per component plus
    # Path.is_dir()/os.stat, so the shim must patch os.mkdir AND Path.mkdir to land in the CWD.
    saved = _save_patch_targets()
    spec = importlib.util.spec_from_file_location("_sandbox_sitecustomize_mkdir", _SHIM)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(mod)
        mod._notified = True
        # exist_ok=True must be honoured against the mapped location, which already exists, not raise.
        pathlib.Path("/mnt/data").mkdir(parents = True, exist_ok = True)
        pathlib.Path("/mnt/data/plots/run1").mkdir(parents = True, exist_ok = True)
        assert os.path.isdir(os.path.join(cwd, "plots", "run1"))
        # exist_ok is evaluated on the mapped path, not the never-present /mnt/data.
        pathlib.Path("/mnt/data/plots/run1").mkdir(parents = True, exist_ok = True)

        real_dir = tmp_path / "real_via_path"
        pathlib.Path(str(real_dir)).mkdir()
        assert real_dir.is_dir()
        real_os = tmp_path / "real_via_os"
        os.mkdir(str(real_os))
        assert real_os.is_dir()
    finally:
        _restore_patch_targets(saved)


def test_read_of_missing_prefix_path_emits_no_notice(monkeypatch, tmp_path, capsys):
    # A read of a missing convention path must not spend the one-shot notice; a genuine remap afterward still notifies.
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    mod._notified = False
    assert mod._remap_open("/mnt/data/missing.csv", "r") == "/mnt/data/missing.csv"
    assert mod._notified is False
    assert "does not exist" not in capsys.readouterr().err
    assert mod._remap_open("/mnt/data/out.txt", "w") == os.path.join(os.getcwd(), "out.txt")
    assert mod._notified is True
    assert "/mnt/data does not exist in this sandbox" in capsys.readouterr().err


def test_os_open_trunc_without_creat_missing_stays_truthful(monkeypatch, tmp_path):
    # O_TRUNC / O_APPEND without O_CREAT cannot create, so the shim reads them and the path stays truthful.
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
