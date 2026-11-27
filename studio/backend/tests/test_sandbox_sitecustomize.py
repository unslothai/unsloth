# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Loads the sitecustomize path-remap shim into a throwaway module and restores patched globals after."""

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
    """Snapshot the globals the shim patches; the pathlib accessor is included only before Python 3.11."""
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


def test_windows_backslash_spelling_of_a_prefix_maps_into_cwd(monkeypatch):
    """On Windows str(Path("/mnt/data/x")) is "\\mnt\\data\\x"; the forward-slash prefix match alone missed it, so a
    write fell back to the basename heal and the matching read failed (staging Windows leg, #16210)."""
    import ntpath
    import types

    mod = _load_shim()
    cwd = "C:\\sandbox\\work"
    fake_os = types.SimpleNamespace(
        **{name: getattr(os, name) for name in dir(os) if not name.startswith("__")}
    )
    fake_os.sep = "\\"
    fake_os.path = types.SimpleNamespace(
        **{name: getattr(ntpath, name) for name in dir(ntpath) if not name.startswith("__")}
    )
    fake_os.path.exists = lambda p: False
    fake_os.getcwd = lambda: cwd
    monkeypatch.setattr(mod, "os", fake_os)
    assert mod._remap("\\mnt\\data\\out.txt") == ntpath.join(cwd, "out.txt")
    assert mod._remap("/mnt/data/sub\\out.txt") == ntpath.join(cwd, "sub", "out.txt")
    assert mod._remap("\\mnt\\data") == cwd
    assert mod._remap("\\mnt\\data\\..\\..\\escape.txt") == ntpath.join(cwd, "escape.txt")
    # A drive-qualified path is a real host location, never a code-interpreter convention.
    assert mod._remap("D:\\mnt\\data\\out.txt") == "D:\\mnt\\data\\out.txt"
    assert mod._remap("\\\\server\\share\\mnt\\data\\x") == "\\\\server\\share\\mnt\\data\\x"
    assert mod._remap("data\\out.txt") == "data\\out.txt"


@pytest.mark.skipif(
    os.path.exists("/mnt/data"),
    reason = "the host has a real /mnt/data, which the shim never shadows",
)
def test_a_pathlib_write_then_read_of_mnt_data_round_trips_in_a_child(tmp_path):
    """End to end in a real interpreter with the shim on PYTHONPATH, on whatever OS runs the suite. Windows spells the
    path with backslashes, which the prefix remap must still catch on the read as well as the write."""
    import subprocess
    import sys

    code = (
        "from pathlib import Path\n"
        "p = Path('/mnt/data/round_trip.txt')\n"
        "p.write_text('round trip')\n"
        "print(p.read_text())\n"
    )
    env = {**os.environ, "PYTHONPATH": str(_SHIM.parent)}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd = tmp_path,
        env = env,
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "round trip"
    assert (tmp_path / "round_trip.txt").read_text(encoding = "utf-8") == "round trip"


def test_prefix_remap_contains_parent_traversal_inside_cwd(monkeypatch, tmp_path):
    # '..' components are dropped so the remap never escapes the per-conversation CWD.
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
    # basename '..' would target the CWD's parent, so the fallback must refuse it.
    mod = _load_shim()
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    for escaping in ("/no/such/tree/..", "/no/such/tree/.", "/no/such/tree/"):
        assert mod._remap_open(escaping, "w") == escaping


def test_write_fallback_remaps_hallucinated_absolute_path(monkeypatch, tmp_path):
    # Write-mode opens on an absolute path with a missing parent go to the CWD basename.
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
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    for mode in ("r", "rb", "r+"):
        assert mod._remap_open("/etc/definitely_missing_xyz.conf", mode) == (
            "/etc/definitely_missing_xyz.conf"
        )


def test_write_fallback_passes_through_existing_external_dir(monkeypatch, tmp_path):
    mod = _load_shim()
    external = tmp_path / "external"
    external.mkdir()
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    target = str(external / "out.txt")
    assert mod._remap_open(target, "w") is target


def test_write_fallback_never_clobbers_same_basename(monkeypatch, tmp_path):
    # A same-named CWD file is unrelated data: refuse on collision so the real open() raises.
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
    # The fallback must recognise its own prior remap, or every regenerate after the first raises.
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

    other = "/opt/other/app.html"
    assert mod._remap_open(other, "w") == other


def test_write_fallback_reserves_healed_target_across_separate_runs(monkeypatch, tmp_path):
    # Each tool call is a fresh subprocess; the on-disk sidecar carries the remap across runs.
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    invented = "/home/ubuntu/Sandbox/app.html"
    target = os.path.join(cwd, "app.html")

    run1 = _load_shim()
    assert run1._remap_open(invented, "w") == target
    with open(run1._remap_open(invented, "w"), "w") as fh:
        fh.write("v1")

    run2 = _load_shim()
    assert run2._remapped_writes == {}
    assert run2._remap_open(invented, "w") == target
    with open(run2._remap_open(invented, "w"), "w") as fh:
        fh.write("v2")
    assert Path(target).read_text() == "v2"

    other = "/opt/other/app.html"
    assert run2._remap_open(other, "w") == other

    (tmp_path / "notes.txt").write_text("KEEP-ME")
    run3 = _load_shim()
    assert run3._remap_open("/some/missing/notes.txt", "w") == "/some/missing/notes.txt"
    with pytest.raises(FileNotFoundError):
        open(run3._remap_open("/some/missing/notes.txt", "w"), "w")
    assert (tmp_path / "notes.txt").read_text() == "KEEP-ME"


@pytest.mark.parametrize("mode", ["r+", "rb+"])
def test_read_update_modes_never_redirected_even_with_missing_parent(monkeypatch, tmp_path, mode):
    # r+ / rb+ never create, so "+" must not count as a create mode.
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
    # A convention prefix is remapped only while absent; a real host dir passes through.
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
    missing = str(external / "new.txt")
    assert mod._remap_open(missing, "w") == missing

    (external / "data.txt").unlink()
    external.rmdir()
    assert mod._remap(target) == os.path.join(os.getcwd(), "data.txt")


def test_os_open_and_path_touch_remap_convention_path(monkeypatch, tmp_path):
    # Path.touch() goes through os.open, not builtins/io.open.
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
    # Path.open uses io.open on 3.11+ and the captured accessor open below 3.11.
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
    assert mod._remap_open(b"/no/such/tree/x.bin", "w") == b"/no/such/tree/x.bin"


def test_remap_open_still_applies_prefix_remaps(monkeypatch, tmp_path):
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    assert mod._remap_open("/mnt/data/sub/out.txt", "w") == os.path.join(cwd, "sub", "out.txt")
    assert mod._remap_open("/mnt/data/sub/out.txt", "r") == "/mnt/data/sub/out.txt"


def test_prefix_read_heals_only_when_mapped_target_exists(monkeypatch, tmp_path):
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
    # The prefix match is segment-anchored: /mnt/database is not /mnt/data.
    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    for unrelated in ("/workspace2/file.txt", "/mnt/database/x", "/home/sandboxed/y"):
        assert mod._remap(unrelated) == unrelated
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
    cond = str(tmp_path / "cond_outputs")
    monkeypatch.setattr(mod, "_CONDITIONAL_PREFIXES", (cond,))

    assert not os.path.exists(cond)
    assert mod._remap(cond + "/plot.png") == os.path.join(cwd, "plot.png")
    assert mod._remap(cond) == cwd

    os.makedirs(cond)
    assert mod._remap(cond + "/plot.png") == cond + "/plot.png"
    assert mod._remap(cond) == cond


def test_pathlib_mkdir_parents_remaps_convention_path(monkeypatch, tmp_path):
    # Path.mkdir(parents, exist_ok) goes through os.mkdir and stat, so both are patched.
    saved = _save_patch_targets()
    spec = importlib.util.spec_from_file_location("_sandbox_sitecustomize_mkdir", _SHIM)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(mod)
        mod._notified = True
        pathlib.Path("/mnt/data").mkdir(parents = True, exist_ok = True)
        pathlib.Path("/mnt/data/plots/run1").mkdir(parents = True, exist_ok = True)
        assert os.path.isdir(os.path.join(cwd, "plots", "run1"))
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
    # O_TRUNC / O_APPEND without O_CREAT cannot create, so they are treated as reads.
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


@pytest.mark.parametrize(
    "init, namespace",
    [
        ("__path__ = __import__('pkgutil').extend_path(__path__, __name__)\n", True),
        ("from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n", True),
        ("__import__('pkg_resources').declare_namespace(__name__)\n", True),
        (
            "try:\n    __import__('pkg_resources').declare_namespace(__name__)\n"
            "except ImportError:\n    import pkgutil\n"
            "    __path__ = pkgutil.extend_path(__path__, __name__)\n",
            True,
        ),
        ("# extend_path and declare_namespace are not used here\n", False),
        ("from pkgutil import extend_path\n", False),
        ("DOC = '__path__ = extend_path(__path__, __name__)'\n", False),
        ("if False:\n    __import__('pkg_resources').declare_namespace(__name__)\n", False),
        ("def setup():\n    __path__ = extend_path(__path__, __name__)\n", False),
        ("paths = extend_path(['x'], 'other')\n", False),
        ("this is not python(\n", False),
        (
            "try:\n    import _no_such_module\n"
            "except ImportError:\n    __path__ = __import__('pkgutil').extend_path(__path__, __name__)\n",
            False,
        ),
        (
            "try:\n    import os\n"
            "except ImportError:\n    pass\n"
            "else:\n    __path__ = __import__('pkgutil').extend_path(__path__, __name__)\n",
            True,
        ),
    ],
)
def test_host_packages_finder_only_treats_declared_namespaces_as_open(tmp_path, init, namespace):
    """Only a real extend_path / declare_namespace declaration opens a regular-looking package to
    the chat folder's portion; text that merely mentions them does not."""
    finder = _load_shim()._HostPackagesFinder
    origin = tmp_path / "__init__.py"
    origin.write_text(init, encoding = "utf-8")
    assert finder._extends_path(str(origin)) is namespace


def _run_with_host_packages(tmp_path, folder, code):
    """Run ``code`` in a fresh interpreter with the shim installed for ``folder``."""
    import subprocess
    import sys

    script = f"import runpy; runpy.run_path({str(_SHIM)!r})\n" + code
    env = {**os.environ, "_UNSLOTH_HOST_SESSION_PACKAGES": str(folder)}
    env.pop("PYTHONPATH", None)
    done = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd = tmp_path,
        env = env,
        capture_output = True,
        text = True,
        timeout = 60,
    )
    return done.stdout + done.stderr


def test_host_packages_metadata_honours_an_explicit_search_path(tmp_path):
    """An exact-name lookup scoped to explicit paths never answers from the chat folder."""
    folder = tmp_path / "packages"
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    info = folder / "chatmeta-1.0.dist-info"
    info.mkdir(parents = True)
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: chatmeta\nVersion: 1.0\n", encoding = "utf-8"
    )
    out = _run_with_host_packages(
        tmp_path,
        folder,
        "import importlib.metadata as m\n"
        "print('DEFAULT', m.version('chatmeta'))\n"
        f"print('TRUSTED', len(list(m.distributions(name = 'chatmeta', path = [{str(trusted)!r}]))))\n"
        f"print('LISTED', len(list(m.distributions(name = 'chatmeta', path = [{str(folder)!r}]))))\n",
    )
    assert "DEFAULT 1.0" in out, out
    assert "TRUSTED 0" in out, out
    assert "LISTED 1" in out, out


def test_host_packages_pep420_portion_joins_the_namespace_path(tmp_path):
    """A submodule served from the chat folder's PEP 420 portion sees that portion through the
    namespace's __path__ (importlib.resources, pkgutil.iter_modules); the host portion stays first."""
    folder = tmp_path / "packages"
    host = tmp_path / "host"
    (host / "chatns").mkdir(parents = True)
    (host / "chatns" / "host_mod.py").write_text("V = 'host'\n", encoding = "utf-8")
    (folder / "chatns").mkdir(parents = True)
    (folder / "chatns" / "data.txt").write_text("chat-data", encoding = "utf-8")
    (folder / "chatns" / "chat_mod.py").write_text(
        "import importlib.resources\n"
        "V = importlib.resources.files('chatns').joinpath('data.txt').read_text()\n",
        encoding = "utf-8",
    )
    out = _run_with_host_packages(
        tmp_path,
        folder,
        f"import sys; sys.path.insert(0, {str(host)!r})\n"
        "import os, pkgutil, chatns.host_mod, chatns.chat_mod\n"
        "print('NS', chatns.chat_mod.V,"
        " [os.path.basename(os.path.dirname(p)) for p in chatns.__path__],"
        " sorted(i.name for i in pkgutil.iter_modules(chatns.__path__)))\n",
    )
    assert "NS chat-data ['host', 'packages'] ['chat_mod', 'host_mod']" in out, out


def test_a_raw_connect_to_a_name_resolving_to_a_metadata_host_is_refused(monkeypatch):
    import socket

    mod = _load_shim()
    answer = {"alias.test": "169.254.169.254", "plain.test": "93.184.215.14"}
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, *a, **k: [(socket.AF_INET, 1, 6, "", (answer[host], 0))],
    )
    with pytest.raises(PermissionError, match = "cloud-metadata host"):
        mod._metadata_connect_hook("socket.connect", (None, ("alias.test", 80)))
    assert mod._metadata_connect_hook("socket.connect", (None, ("plain.test", 80))) is None
    assert mod._metadata_connect_hook("socket.connect", (None, ("127.0.0.1", 80))) is None


def _child(
    code,
    cwd,
    *,
    shim = True,
    flags = (),
):
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if shim:
        env["PYTHONPATH"] = str(_SHIM.parent)
    proc = subprocess.run(
        [sys.executable, *flags, "-c", code],
        cwd = cwd,
        env = env,
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_the_shim_imports_neither_json_nor_pathlib_at_startup(tmp_path):
    """They pull re, enum, functools and more: most of every sandboxed Python launch. Loaded on first use instead.
    Run with -S and the shim imported directly, so no .pth file (an editable install's finder) preloads them."""
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(_SHIM.parent)!r})\n"
        "before = set(sys.modules)\n"
        "import sitecustomize\n"
        "print(' '.join(sorted(set(sys.modules) - before)))\n"
    )
    added = set(_child(code, tmp_path, shim = False, flags = ("-S",)).split())
    assert "sitecustomize" in added
    assert not added & {"json", "pathlib", "re", "enum"}, sorted(added)


def test_the_shim_resolves_the_stdlib_paths_once_at_startup(tmp_path):
    """realpath of every sys.path entry is slow on Windows (more inside the container): one pass per launch."""
    code = (
        "import os, sys\n"
        f"sys.path.insert(0, {str(_SHIM.parent)!r})\n"
        "real = os.path.realpath\n"
        "calls = []\n"
        "os.path.realpath = lambda *a, **k: calls.append(a) or real(*a, **k)\n"
        "entries = len(sys.path)\n"
        "import sitecustomize\n"
        "os.path.realpath = real\n"
        "print(len(calls), entries)\n"
    )
    calls, entries = map(int, _child(code, tmp_path, shim = False, flags = ("-S",)).split())
    assert calls <= entries + 1, (calls, entries)


@pytest.mark.skipif(
    os.path.exists("/mnt/data"),
    reason = "the host has a real /mnt/data, which the shim never shadows",
)
@pytest.mark.parametrize(
    "spelling", ["import pathlib\nP = pathlib.Path", "from pathlib import Path as P"]
)
def test_path_mkdir_is_remapped_when_pathlib_is_imported_after_startup(tmp_path, spelling):
    code = (
        f"{spelling}\n"
        "import os, sys\n"
        "P('/mnt/data/made/inner').mkdir(parents = True, exist_ok = True)\n"
        "print(os.path.isdir('made/inner'), 'pathlib._local' in sys.modules or True)\n"
    )
    assert _child(code, tmp_path).split()[0] == "True"
    assert (tmp_path / "made" / "inner").is_dir()


@pytest.mark.parametrize("name", ["pathlib", "json"])
def test_a_find_spec_probe_does_not_use_up_the_lazy_stdlib_import(tmp_path, name):
    """importlib.util.find_spec only probes: the import after it must still be answered (and pathlib patched)."""
    code = (
        "import importlib.util, sys, sysconfig, os\n"
        f"assert importlib.util.find_spec({name!r}) is not None\n"
        f"import {name}\n"
        "stdlib = os.path.realpath(sysconfig.get_paths()['stdlib'])\n"
        f"patched = {name!r} != 'pathlib' or '_path_mkdir' in {name}.Path.mkdir.__qualname__\n"
        f"print(os.path.realpath({name}.__file__).startswith(stdlib), patched)\n"
    )
    (tmp_path / f"{name}.py").write_text("raise SystemExit('planted copy ran')\n")
    assert _child(code, tmp_path).split() == ["True", "True"]


def test_modules_that_import_pathlib_get_the_patched_one(tmp_path):
    code = "import zipfile, pathlib\nprint(pathlib.Path.mkdir.__qualname__)\n"
    assert "_path_mkdir" in _child(code, tmp_path)


@pytest.mark.parametrize("name", ["json", "pathlib"])
def test_a_planted_copy_in_the_cwd_does_not_replace_the_stdlib_module(tmp_path, name):
    (tmp_path / f"{name}.py").write_text("raise SystemExit('planted copy ran')\n")
    code = (
        f"import {name}, sys, sysconfig, os\n"
        f"stdlib = os.path.realpath(sysconfig.get_paths()['stdlib'])\n"
        f"print(os.path.realpath({name}.__file__).startswith(stdlib))\n"
    )
    assert _child(code, tmp_path) == "True"


def test_the_lazy_finder_resolves_from_the_stdlib_and_hands_back_the_real_loader(tmp_path):
    code = (
        "import sys\n"
        "finder = sys.meta_path[0]\n"
        "print(type(finder).__name__)\n"
        "import pathlib, json\n"
        "print(type(pathlib.__loader__).__name__, type(json.__spec__.loader).__name__)\n"
        "print(any(type(f).__name__ == '_StdlibLazyFinder' for f in sys.meta_path))\n"
        "import helper_mod\n"
        "print(helper_mod.VALUE)\n"
    )
    (tmp_path / "helper_mod.py").write_text("VALUE = 'cwd-module'\n")
    first, loaders, still_there, helper = _child(code, tmp_path).splitlines()
    assert first == "_StdlibLazyFinder"
    assert "_StdlibLoader" not in loaders
    # Never removed (importlib walks the live meta_path), and inert after: cwd modules import as usual.
    assert still_there == "True"
    assert helper == "cwd-module"


def test_imports_on_other_threads_never_miss_a_module_while_the_lazy_import_runs(tmp_path):
    """meta_path is not edited after startup, so a thread importing during the first json/pathlib import still
    finds every finder (an insert/remove there could make importlib skip one)."""
    for i in range(20):
        (tmp_path / f"cwd_mod_{i}.py").write_text(f"V = {i}\n")
    code = (
        "import sys, threading, importlib\n"
        "errors = []\n"
        "go = threading.Event()\n"
        "def worker(i):\n"
        "    go.wait()\n"
        "    try:\n"
        "        assert importlib.import_module(f'cwd_mod_{i}').V == i\n"
        "    except Exception as exc:\n"
        "        errors.append(repr(exc))\n"
        "threads = [threading.Thread(target = worker, args = (i,)) for i in range(20)]\n"
        "[t.start() for t in threads]\n"
        "go.set()\n"
        "import pathlib, json\n"
        "[t.join() for t in threads]\n"
        "print(len(errors), '_path_mkdir' in pathlib.Path.mkdir.__qualname__)\n"
    )
    assert _child(code, tmp_path).split() == ["0", "True"]


def test_the_sidecar_round_trips_in_a_child_without_json_preloaded(tmp_path):
    code = (
        "with open('/no/such/dir/healed.txt', 'w') as fh: fh.write('x')\n"
        "import json\n"
        "print(json.load(open('.unsloth_sandbox_remap.json')))\n"
    )
    out = _child(code, tmp_path)
    assert "/no/such/dir/healed.txt" in out and "healed.txt" in out
    assert (tmp_path / "healed.txt").read_text() == "x"


def test_the_sidecar_is_skipped_when_a_planted_json_is_already_loaded(monkeypatch, tmp_path):
    import sys
    import types

    mod = _load_shim()
    monkeypatch.chdir(tmp_path)
    planted = types.ModuleType("json")
    planted.__file__ = str(tmp_path / "json.py")
    planted.load = planted.dump = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("planted json used")
    )
    monkeypatch.setitem(sys.modules, "json", planted)
    assert mod._stdlib_json() is None
    assert mod._load_sidecar(str(tmp_path)) == {}
    mod._record_sidecar(str(tmp_path), "/a/b.txt", str(tmp_path / "b.txt"))
    assert not (tmp_path / ".unsloth_sandbox_remap.json").exists()
