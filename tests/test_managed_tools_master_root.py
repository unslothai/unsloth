# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_HOME names the tree, not the Studio directory inside it, and the native runtimes are
siblings of studio/, the spelling studio/setup.sh and scripts/build_whisper_cpp.sh already use. A
resolver deriving them from studio_root() would look in <root>/studio/<tool> for what the
installer put at <root>/<tool>, so managed Node and whisper.cpp go missing and run.py pins the
wrong llama.cpp path into every worker.

Run in a subprocess per case: these modules read the environment at import time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "studio" / "backend"

PROBE = """
import json, os, sys
sys.path.insert(0, os.environ["_BACKEND"])
from utils.paths import storage_roots as sr
from utils.node_runtime import managed_node_dir
from core.inference.stt_ggml_sidecar import _managed_whisper_cpp_dir

# studio_root() is called constantly, so a warning it emits for a supported
# layout is not one line, it is a flooded log.
_warnings = []
sr.logger.warning = lambda msg, *a, **k: _warnings.append(msg % a if a else msg)

print(json.dumps({
    "studio": str(sr.studio_root()),
    "master": None if sr.unsloth_home() is None else str(sr.unsloth_home()),
    "node": str(managed_node_dir()),
    "whisper": str(_managed_whisper_cpp_dir()),
    "warnings": _warnings,
}))
"""


def _resolve(env_overrides: dict[str, str], home: Path) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "_BACKEND": str(BACKEND),
    }
    # A hand-built environment has to carry what the interpreter needs: Windows python exits 1
    # with no usable message when SYSTEMROOT is absent, which read as "the resolver answered
    # wrongly" on every Windows runner.
    for name in ("SYSTEMROOT", "SystemRoot", "COMSPEC", "PATHEXT", "TEMP", "TMP", "WINDIR"):
        value = os.environ.get(name)
        if value:
            env.setdefault(name, value)
    env.update(env_overrides)
    out = subprocess.run([sys.executable, "-c", PROBE], env = env, capture_output = True, text = True)
    # Not check = True: the child's stderr is the only thing that says why, and swallowing it is
    # how a dead interpreter passes for a wrong answer.
    assert out.returncode == 0, f"probe failed ({out.returncode}): {out.stderr.strip()}"
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_portable_root_puts_the_tools_beside_studio(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "portable"
    r = _resolve({"UNSLOTH_HOME": str(root)}, home)
    assert r["studio"] == str(root / "studio")
    assert r["master"] == str(root)
    assert r["node"] == str(root / "node")
    assert r["whisper"] == str(root / "whisper.cpp")


def test_a_default_install_is_untouched(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    r = _resolve({}, home)
    assert r["master"] is None
    assert r["studio"] == str(home / ".unsloth" / "studio")
    assert r["node"] == str(home / ".unsloth" / "node")
    assert r["whisper"] == str(home / ".unsloth" / "whisper.cpp")


def test_a_plain_custom_studio_home_is_untouched(tmp_path):
    # No UNSLOTH_HOME: the tools stay children of the Studio root, as before.
    home = tmp_path / "home"
    home.mkdir()
    custom = tmp_path / "custom"
    r = _resolve({"UNSLOTH_STUDIO_HOME": str(custom)}, home)
    assert r["master"] is None
    assert r["studio"] == str(custom)
    assert r["node"] == str(custom / "node")
    assert r["whisper"] == str(custom / "whisper.cpp")


def test_a_flat_root_keeps_the_tools_at_that_root(tmp_path):
    # UNSLOTH_HOME == UNSLOTH_STUDIO_HOME, so "beside studio/" and "inside it" are one directory.
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "flat"
    r = _resolve({"UNSLOTH_HOME": str(root), "UNSLOTH_STUDIO_HOME": str(root)}, home)
    assert r["studio"] == str(root)
    assert r["node"] == str(root / "node")
    assert r["whisper"] == str(root / "whisper.cpp")
    # Path.parents excludes the path itself, so the equality check is what keeps this warning off.
    assert r["warnings"] == []


def test_a_studio_home_outside_the_master_root_still_warns(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    r = _resolve(
        {
            "UNSLOTH_HOME": str(tmp_path / "portable"),
            "UNSLOTH_STUDIO_HOME": str(tmp_path / "elsewhere"),
        },
        home,
    )
    assert any("not self-contained" in w for w in r["warnings"])


def test_the_builder_and_the_resolver_agree_on_the_same_directory(tmp_path):
    # build_whisper_cpp.sh installs under UNSLOTH_HOME; a resolver one level off reports
    # dictation unavailable with whisper-server sitting right there.
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "portable"
    built = root / "whisper.cpp"
    built.mkdir(parents = True)
    r = _resolve({"UNSLOTH_HOME": str(root)}, home)
    assert r["whisper"] == str(built)


def _record_note(studio: Path, master: Path) -> None:
    """What setup.sh writes at the end of a master-root install."""
    (studio / "share").mkdir(parents = True, exist_ok = True)
    (studio / "share" / ".unsloth-master-root").write_text(f"{master}\n", encoding = "utf-8")


def test_a_recorded_master_root_outlives_the_command_that_set_it(tmp_path):
    # UNSLOTH_HOME is settable for one command, which installs node and whisper.cpp BESIDE
    # studio/, and nothing persists it: the next launch resolved both one level down, at
    # <studio>/node and <studio>/whisper.cpp, with the real trees untouched next door.
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "portable"
    studio = root / "studio"
    studio.mkdir(parents = True)
    _record_note(studio, root)
    # UNSLOTH_STUDIO_HOME alone: what the installer's launcher actually persists.
    r = _resolve({"UNSLOTH_STUDIO_HOME": str(studio)}, home)
    assert r["master"] == str(root)
    assert r["studio"] == str(studio)
    assert r["node"] == str(root / "node")
    assert r["whisper"] == str(root / "whisper.cpp")
    assert r["warnings"] == []


def test_a_note_whose_root_has_since_moved_is_ignored(tmp_path):
    # A note licenses this process to adopt a root for caches and runtimes both. One naming a
    # tree that is no longer there must not win over the layout in front of it.
    home = tmp_path / "home"
    home.mkdir()
    studio = tmp_path / "custom"
    studio.mkdir()
    _record_note(studio, tmp_path / "gone")
    r = _resolve({"UNSLOTH_STUDIO_HOME": str(studio)}, home)
    assert r["master"] is None
    assert r["node"] == str(studio / "node")


def test_a_note_copied_into_an_unrelated_install_is_ignored(tmp_path):
    # The recorded root exists and has a studio/ child, but it is not THIS install's studio
    # directory, so the note travelled rather than described. Checking only is_dir() would
    # redirect this install's runtimes into someone else's tree.
    home = tmp_path / "home"
    home.mkdir()
    other = tmp_path / "other"
    (other / "studio").mkdir(parents = True)
    studio = tmp_path / "custom"
    studio.mkdir()
    _record_note(studio, other)
    r = _resolve({"UNSLOTH_STUDIO_HOME": str(studio)}, home)
    assert r["master"] is None
    assert r["node"] == str(studio / "node")


def test_an_empty_note_is_not_a_root(tmp_path):
    # A truncated or zero-length note must read as "no record", not as the current directory.
    home = tmp_path / "home"
    home.mkdir()
    studio = tmp_path / "custom"
    (studio / "share").mkdir(parents = True)
    (studio / "share" / ".unsloth-master-root").write_text("\n", encoding = "utf-8")
    r = _resolve({"UNSLOTH_STUDIO_HOME": str(studio)}, home)
    assert r["master"] is None
    assert r["node"] == str(studio / "node")


def test_an_explicit_studio_home_without_a_note_does_not_borrow_anothers(tmp_path):
    # Two installs on one box, only the legacy tree carrying a note: studio_root() stays on the
    # named tree, so reading past it would send the runtimes and the portable caches to the
    # OTHER install while Studio ran from here.
    home = tmp_path / "home"
    legacy_master = home / ".unsloth"
    (legacy_master / "studio").mkdir(parents = True)
    _record_note(legacy_master / "studio", legacy_master)
    named = tmp_path / "named"
    named.mkdir()
    r = _resolve({"UNSLOTH_STUDIO_HOME": str(named)}, home)
    assert r["master"] is None
    assert r["studio"] == str(named)
    assert r["node"] == str(named / "node")
    assert r["whisper"] == str(named / "whisper.cpp")


def test_a_flat_recorded_root_is_still_honoured(tmp_path):
    # The flat layout is supported, and the note then sits in the root it names rather than in a
    # studio/ child: an exact <root>/studio match would refuse it, containment satisfies it.
    home = tmp_path / "home"
    home.mkdir()
    flat = tmp_path / "flat"
    (flat / "share").mkdir(parents = True)
    (flat / "share" / ".unsloth-master-root").write_text(f"{flat}\n", encoding = "utf-8")
    r = _resolve({"UNSLOTH_STUDIO_HOME": str(flat)}, home)
    assert r["master"] == str(flat)
    assert r["node"] == str(flat / "node")
    assert r["warnings"] == []


def test_a_default_install_reads_no_note(tmp_path):
    # Nothing writes the note for a default install, so the legacy tree must not acquire a
    # master root by accident: this is the path every existing user is on.
    home = tmp_path / "home"
    (home / ".unsloth" / "studio" / "share").mkdir(parents = True)
    r = _resolve({}, home)
    assert r["master"] is None
    assert r["node"] == str(home / ".unsloth" / "node")


_DISCOVERY_PROBE = """
import json, os, sys
sys.path.insert(0, os.environ["_BACKEND"])
from pathlib import Path
from utils.paths.storage_roots import studio_root, unsloth_home
from utils.llama_cpp_path_settings import mark_managed_llama_cpp_path

# Replay run.py's module-level block, which is what a real server start does
# before anything asks where llama-server is.
resolved = studio_root().resolve()
if resolved != (Path.home() / ".unsloth" / "studio"):
    os.environ.setdefault("UNSLOTH_STUDIO_HOME", str(resolved))
    managed = (unsloth_home() or resolved) / "llama.cpp"
    os.environ.setdefault("UNSLOTH_LLAMA_CPP_PATH", str(managed))
    mark_managed_llama_cpp_path(managed)

from core.inference.llama_cpp import LlamaCppBackend

print(json.dumps({
    "exported": os.environ.get("UNSLOTH_LLAMA_CPP_PATH"),
    "found": LlamaCppBackend._find_llama_server_binary(),
}))
"""


def _install_llama_server(directory: Path) -> Path:
    # The name _find_llama_server_binary looks for on THIS platform: it appends .exe on Windows,
    # so a fixture that only ever writes the POSIX name made discovery correctly answer None and
    # all three discovery tests fail on a Windows runner for a reason in the fixture.
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    binary = directory / "build" / "bin" / name
    binary.parent.mkdir(parents = True, exist_ok = True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
    binary.chmod(0o755)
    return binary


def _discover(env_overrides: dict[str, str], home: Path) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "_BACKEND": str(BACKEND),
    }
    # Same reason as _resolve: the interpreter's own requirements travel with it.
    for name in ("SYSTEMROOT", "SystemRoot", "COMSPEC", "PATHEXT", "TEMP", "TMP", "WINDIR"):
        value = os.environ.get(name)
        if value:
            env.setdefault(name, value)
    env.update(env_overrides)
    out = subprocess.run(
        [sys.executable, "-c", _DISCOVERY_PROBE],
        env = env,
        capture_output = True,
        text = True,
    )
    assert out.returncode == 0, f"discovery probe failed ({out.returncode}): {out.stderr.strip()}"
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_discovery_finds_the_llama_server_the_master_root_holds(tmp_path):
    # The managed marker makes discovery SKIP the env var for its own derivation, so the two
    # must name one directory or every GGUF model reports no runtime.
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "portable"
    (root / "studio").mkdir(parents = True)
    binary = _install_llama_server(root / "llama.cpp")

    result = _discover({"UNSLOTH_HOME": str(root)}, home)

    assert result["exported"] == str(root / "llama.cpp")
    assert result["found"] == str(binary)


def test_discovery_still_prefers_a_plain_custom_studio_root(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    custom = tmp_path / "custom"
    custom.mkdir()
    binary = _install_llama_server(custom / "llama.cpp")

    result = _discover({"UNSLOTH_STUDIO_HOME": str(custom)}, home)

    assert result["found"] == str(binary)


def test_discovery_still_finds_a_legacy_install(tmp_path):
    home = tmp_path / "home"
    (home / ".unsloth" / "studio").mkdir(parents = True)
    binary = _install_llama_server(home / ".unsloth" / "llama.cpp")

    result = _discover({}, home)

    assert result["found"] == str(binary)
