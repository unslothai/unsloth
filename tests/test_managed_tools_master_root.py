# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_HOME names the tree, not the Studio directory inside it, and the
native runtimes are siblings of studio/, which is the spelling studio/setup.sh
and scripts/build_whisper_cpp.sh already use. A resolver that derived them from
studio_root() would look in <root>/studio/<tool> for something the installer put
at <root>/<tool>, so managed Node and whisper.cpp would silently go missing and
run.py would pin the wrong llama.cpp path into every worker's environment.

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
    env.update(env_overrides)
    out = subprocess.run(
        [sys.executable, "-c", PROBE], env = env, capture_output = True, text = True, check = True
    )
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
    # No UNSLOTH_HOME, so this stays on the pre-existing derivation where the
    # tools are children of the Studio root.
    home = tmp_path / "home"
    home.mkdir()
    custom = tmp_path / "custom"
    r = _resolve({"UNSLOTH_STUDIO_HOME": str(custom)}, home)
    assert r["master"] is None
    assert r["studio"] == str(custom)
    assert r["node"] == str(custom / "node")
    assert r["whisper"] == str(custom / "whisper.cpp")


def test_a_flat_root_keeps_the_tools_at_that_root(tmp_path):
    # UNSLOTH_HOME == UNSLOTH_STUDIO_HOME: the master root IS the Studio root,
    # so "beside studio/" and "inside it" are the same directory.
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "flat"
    r = _resolve({"UNSLOTH_HOME": str(root), "UNSLOTH_STUDIO_HOME": str(root)}, home)
    assert r["studio"] == str(root)
    assert r["node"] == str(root / "node")
    assert r["whisper"] == str(root / "whisper.cpp")
    # Path.parents excludes the path itself, so an equality check is what keeps
    # this supported layout from warning that it is not self-contained.
    assert r["warnings"] == []


def test_a_studio_home_outside_the_master_root_still_warns(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    r = _resolve(
        {"UNSLOTH_HOME": str(tmp_path / "portable"), "UNSLOTH_STUDIO_HOME": str(tmp_path / "elsewhere")},
        home,
    )
    assert any("not self-contained" in w for w in r["warnings"])


def test_the_builder_and_the_resolver_agree_on_the_same_directory(tmp_path):
    # scripts/build_whisper_cpp.sh reads UNSLOTH_HOME as the root it installs
    # under. The resolver has to land on the same path or dictation reports the
    # engine unavailable while whisper-server sits on disk one level up.
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "portable"
    built = root / "whisper.cpp"
    built.mkdir(parents = True)
    r = _resolve({"UNSLOTH_HOME": str(root)}, home)
    assert r["whisper"] == str(built)
