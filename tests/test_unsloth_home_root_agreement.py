# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""UNSLOTH_HOME has to mean the same install to the CLI and to the backend: if
the two disagree the backend writes studio.db, auth and the pid file under
<UNSLOTH_HOME>/studio while the CLI reads ~/.unsloth/studio.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_PROBE = """
import json, sys
sys.path.insert(0, {backend!r})
import os
from unsloth_cli.commands import studio as cli
from utils.paths.storage_roots import studio_root, unsloth_home
cli._ensure_studio_env_exported()
master = unsloth_home()
print(json.dumps({{
    "cli": str(cli.STUDIO_HOME),
    "cli_is_custom": bool(cli._STUDIO_HOME_IS_CUSTOM),
    "backend": str(studio_root()),
    "cli_llama": os.environ.get("UNSLOTH_LLAMA_CPP_PATH"),
    "backend_llama": str((master or studio_root()) / "llama.cpp"),
}}))
"""


def _probe(env_overrides: dict) -> dict:
    env = dict(os.environ)
    for key in ("UNSLOTH_HOME", "UNSLOTH_STUDIO_HOME", "STUDIO_HOME"):
        env.pop(key, None)
    env.update(env_overrides)
    env["PYTHONPATH"] = str(REPO_ROOT)
    source = _PROBE.format(backend = str(REPO_ROOT / "studio" / "backend"))
    out = subprocess.run(
        [sys.executable, "-c", source],
        capture_output = True,
        text = True,
        cwd = str(REPO_ROOT),
        env = env,
        check = True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_unsloth_home_resolves_to_one_studio_root(tmp_path):
    master = tmp_path / "portable"
    result = _probe({"UNSLOTH_HOME": str(master)})

    assert result["backend"] == str(master / "studio")
    assert result["cli"] == result["backend"]
    # Custom, so `unsloth studio ...` re-exports UNSLOTH_STUDIO_HOME.
    assert result["cli_is_custom"] is True


def test_studio_home_still_outranks_unsloth_home(tmp_path):
    explicit = tmp_path / "explicit"
    result = _probe(
        {"UNSLOTH_HOME": str(tmp_path / "portable"), "UNSLOTH_STUDIO_HOME": str(explicit)}
    )

    assert result["backend"] == str(explicit)
    assert result["cli"] == str(explicit)


def test_no_unsloth_home_keeps_the_legacy_default(tmp_path):
    result = _probe({"HOME": str(tmp_path / "home"), "USERPROFILE": str(tmp_path / "home")})

    legacy = str(tmp_path / "home" / ".unsloth" / "studio")
    assert result["cli"] == legacy
    assert result["backend"] == legacy
    assert result["cli_is_custom"] is False


def test_the_cli_exports_the_llama_cpp_path_the_backend_will_use(tmp_path):
    # run.py keeps a non-blank UNSLOTH_LLAMA_CPP_PATH, so a CLI that exports
    # <root>/studio/llama.cpp pins that wrong path for the server and every
    # worker, and llama_cpp.py then cannot find the managed llama-server.
    master = tmp_path / "portable"
    result = _probe({"UNSLOTH_HOME": str(master)})

    assert result["cli_llama"] == str(master / "llama.cpp")
    assert result["cli_llama"] == result["backend_llama"]


def test_a_plain_custom_root_still_keeps_llama_cpp_inside_it(tmp_path):
    explicit = tmp_path / "explicit"
    result = _probe({"UNSLOTH_STUDIO_HOME": str(explicit)})

    assert result["cli_llama"] == str(explicit / "llama.cpp")
    assert result["cli_llama"] == result["backend_llama"]


def test_a_legacy_install_still_keeps_llama_cpp_at_the_legacy_path(tmp_path):
    home = tmp_path / "home"
    (home / ".unsloth" / "studio").mkdir(parents = True)
    result = _probe({"HOME": str(home), "USERPROFILE": str(home)})

    # Not custom, so nothing is exported at all and the backend default stands.
    assert result["cli_llama"] is None
    assert result["backend_llama"] == str(home / ".unsloth" / "studio" / "llama.cpp")


_MAIN_PROBE = """
import json, sys, types
sys.path.insert(0, {backend!r})
# main.py pulls in the whole app on import, so exec only its module-level
# llama.cpp block, which is what decides the managed path and marks it.
import os, re
from pathlib import Path
src = Path({backend!r}, "main.py").read_text(encoding = "utf-8")
# Anchored on the master-root lookup rather than the `if`, because the guard
# reads _MASTER_ROOT and a slice starting at the `if` would not define it.
start = src.index('from utils.paths.storage_roots import unsloth_home as _unsloth_home')
end = src.index('# The studio bundles unsloth_zoo', start)
if 'if _STUDIO_ROOT_RESOLVED != _LEGACY_STUDIO_ROOT' not in src[start:end]:
    raise AssertionError("the extracted slice no longer contains the export guard")
from utils.paths.storage_roots import studio_root as _studio_root
_LEGACY_STUDIO_ROOT = (Path.home() / ".unsloth" / "studio").resolve()
_STUDIO_ROOT_RESOLVED = _studio_root().resolve()
marked = []
mod = types.ModuleType("utils.llama_cpp_path_settings")
mod.mark_managed_llama_cpp_path = lambda p: marked.append(str(p))
sys.modules["utils.llama_cpp_path_settings"] = mod
exec(compile(src[start:end], "main.py", "exec"), globals())
print(json.dumps({{
    "exported": os.environ.get("UNSLOTH_LLAMA_CPP_PATH"),
    "marked": marked,
}}))
"""


def _main_probe(env_overrides: dict) -> dict:
    env = dict(os.environ)
    for key in ("UNSLOTH_HOME", "UNSLOTH_STUDIO_HOME", "STUDIO_HOME", "UNSLOTH_LLAMA_CPP_PATH"):
        env.pop(key, None)
    env.update(env_overrides)
    env["PYTHONPATH"] = str(REPO_ROOT)
    source = _MAIN_PROBE.format(backend = str(REPO_ROOT / "studio" / "backend"))
    out = subprocess.run(
        [sys.executable, "-c", source],
        capture_output = True,
        text = True,
        cwd = str(REPO_ROOT),
        env = env,
        check = True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_main_marks_the_same_llama_cpp_path_run_py_exports(tmp_path):
    # A direct `uvicorn main:app` exports this outright; under run.py main.py
    # keeps the correct value but would mark the wrong one managed, which makes
    # the bundled path look like an immutable user override.
    master = tmp_path / "portable"
    result = _main_probe({"UNSLOTH_HOME": str(master)})

    assert result["exported"] == str(master / "llama.cpp")
    assert result["marked"] == [str(master / "llama.cpp")]


def test_main_leaves_a_plain_custom_root_alone(tmp_path):
    explicit = tmp_path / "explicit"
    result = _main_probe({"UNSLOTH_STUDIO_HOME": str(explicit)})

    assert result["exported"] == str(explicit / "llama.cpp")
    assert result["marked"] == [str(explicit / "llama.cpp")]


def test_main_exports_for_a_master_root_that_is_the_legacy_path(tmp_path):
    # A portable install pointed at the legacy Studio path owns llama.cpp beside
    # it, at <root>/llama.cpp. Guarding on the legacy equality alone skipped the
    # export, so a bare `uvicorn main:app` in a fresh shell left unsloth_zoo on
    # ~/.unsloth/llama.cpp and never saw the installed runtime.
    home = tmp_path / "home"
    legacy = home / ".unsloth" / "studio"
    legacy.mkdir(parents = True)
    result = _main_probe(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "UNSLOTH_HOME": str(legacy),
            "UNSLOTH_STUDIO_HOME": str(legacy),
        }
    )

    assert result["exported"] == str(legacy / "llama.cpp")
    assert result["marked"] == [str(legacy / "llama.cpp")]


def test_main_still_exports_nothing_without_a_master_root(tmp_path):
    # The other half: a plain legacy install has no master root, so the guard
    # must stay closed rather than pinning the default it already resolves to.
    home = tmp_path / "home"
    (home / ".unsloth" / "studio").mkdir(parents = True)
    result = _main_probe({"HOME": str(home), "USERPROFILE": str(home)})

    assert result["exported"] is None
    assert result["marked"] == []
