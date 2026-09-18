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

"""UNSLOTH_HOME has to mean the same install to the CLI and to the backend: if the two disagree
the backend writes studio.db, auth and the pid file under <UNSLOTH_HOME>/studio while the CLI
reads ~/.unsloth/studio.
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
    "exported_studio_home": os.environ.get("UNSLOTH_STUDIO_HOME"),
    "cli_llama": os.environ.get("UNSLOTH_LLAMA_CPP_PATH"),
    "exported_master": os.environ.get("UNSLOTH_HOME"),
    "backend_llama": str((master or studio_root()) / "llama.cpp"),
}}))
"""


def _probe(env_overrides: dict) -> dict:
    env = dict(os.environ)
    # UNSLOTH_LLAMA_CPP_PATH too, as _main_probe does: an inherited value measures the runner.
    for key in ("UNSLOTH_HOME", "UNSLOTH_STUDIO_HOME", "STUDIO_HOME", "UNSLOTH_LLAMA_CPP_PATH"):
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
    # run.py keeps a non-blank UNSLOTH_LLAMA_CPP_PATH, so a CLI exporting
    # <root>/studio/llama.cpp pins that wrong path for every worker.
    master = tmp_path / "portable"
    result = _probe({"UNSLOTH_HOME": str(master)})

    assert result["cli_llama"] == str(master / "llama.cpp")
    assert result["cli_llama"] == result["backend_llama"]


def test_the_cli_recovers_a_recorded_master_root_for_the_setup_subprocess(tmp_path):
    # `UNSLOTH_HOME=/mnt/portable unsloth studio update` leaves nothing in a later environment.
    # The backend recovers the root from the note, so a CLI that did not would hand setup a plain
    # Studio root: setup refreshes the runtimes at <master>/studio/ while the backend keeps
    # launching the stale ones at <master>/. Exported, not merely read, because setup is a
    # subprocess.
    master = tmp_path / "portable"
    studio = master / "studio"
    (studio / "share").mkdir(parents = True)
    (studio / "share" / ".unsloth-master-root").write_text(f"{master}\n", encoding = "utf-8")
    result = _probe({"UNSLOTH_STUDIO_HOME": str(studio)})

    assert result["exported_master"] == str(master)
    assert result["cli_llama"] == str(master / "llama.cpp")
    assert result["cli_llama"] == result["backend_llama"]


def test_the_cli_refuses_a_note_carried_in_from_another_master_root(tmp_path):
    # A Studio tree copied from master root A to B keeps a note naming A. Exporting it would
    # point an update at the original install; the Studio directory has to lie inside the root
    # its note names, which is the rule storage_roots and both uninstallers apply.
    original = tmp_path / "original"
    (original / "studio").mkdir(parents = True)
    copied = tmp_path / "copied" / "studio"
    (copied / "share").mkdir(parents = True)
    (copied / "share" / ".unsloth-master-root").write_text(f"{original}\n", encoding = "utf-8")
    result = _probe({"UNSLOTH_STUDIO_HOME": str(copied)})

    assert result["exported_master"] is None
    assert result["cli_llama"] == str(copied / "llama.cpp")
    assert result["cli_llama"] == result["backend_llama"]


def test_a_plain_custom_root_still_keeps_llama_cpp_inside_it(tmp_path):
    explicit = tmp_path / "explicit"
    result = _probe({"UNSLOTH_STUDIO_HOME": str(explicit)})

    assert result["cli_llama"] == str(explicit / "llama.cpp")
    assert result["cli_llama"] == result["backend_llama"]


def test_a_master_root_the_cli_declined_is_still_told_to_the_backend(tmp_path):
    """The one root the CLI picks that the backend would not pick for itself.

    install.sh and install.ps1 do not read UNSLOTH_HOME yet, so _resolve_studio_home keeps an
    exported-but-uninstalled master root off the legacy install that actually exists. That
    fallback is not custom, and returning early on non-custom exported nothing at all, so
    studio_root() in the backend went on honouring UNSLOTH_HOME with no install check: the CLI
    ran the legacy venv while the backend under it wrote studio.db, auth and the pid file to
    <master>/studio. The other tests here point the master root at an empty directory AND leave
    the legacy root uninstalled, so the fallback is never entered by them.
    """
    home = tmp_path / "home"
    # The sentinel _looks_like_installer_managed_studio_home reads.
    conf = home / ".unsloth" / "studio" / "share" / "studio.conf"
    conf.parent.mkdir(parents = True)
    conf.write_text("installed\n", encoding = "utf-8")

    result = _probe(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "UNSLOTH_HOME": str(tmp_path / "portable"),
        }
    )

    legacy = str(home / ".unsloth" / "studio")
    assert result["cli"] == legacy
    # Not custom: the root is the Unsloth-owned legacy one, and false is what lets the
    # installers replace the tree without an owner marker.
    assert result["cli_is_custom"] is False
    assert result["exported_studio_home"] == legacy
    assert result["backend"] == legacy


def test_a_whitespace_studio_home_does_not_defeat_the_export(tmp_path):
    """The same fallback as above, with an inherited UNSLOTH_STUDIO_HOME of "   ".

    Every resolver strips before deciding, so "   " means unset to _resolve_studio_home and to
    studio_root alike. The export guard read the raw value, where "   " is truthy, so it left the
    whitespace in place: the CLI ran the legacy install and the backend under it resolved
    <master>/studio, which is the split this file exists to prevent. Blank was already covered by
    the guard; whitespace-only was the shape that got through.
    """
    home = tmp_path / "home"
    conf = home / ".unsloth" / "studio" / "share" / "studio.conf"
    conf.parent.mkdir(parents = True)
    conf.write_text("installed\n", encoding = "utf-8")

    legacy = str(home / ".unsloth" / "studio")
    for inherited in ("   ", "", "\t\n"):
        result = _probe(
            {
                "HOME": str(home),
                "USERPROFILE": str(home),
                "UNSLOTH_HOME": str(tmp_path / "portable"),
                "UNSLOTH_STUDIO_HOME": inherited,
                "UNSLOTH_LLAMA_CPP_PATH": inherited,
            }
        )
        assert result["cli"] == legacy, inherited
        assert result["exported_studio_home"] == legacy, inherited
        assert result["backend"] == legacy, inherited
        # The runtimes live beside studio/ at the master root, and a whitespace value left in
        # place would be scanned as a directory rather than replaced.
        assert result["cli_llama"] == str(tmp_path / "portable" / "llama.cpp"), inherited


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
    # A direct `uvicorn main:app` exports this outright; under run.py main.py would instead
    # mark the wrong path managed, making the bundled one look like an override.
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
    # A portable install pointed at the legacy Studio path still owns <root>/llama.cpp, so the
    # legacy equality alone skipped the export and left unsloth_zoo on ~/.unsloth/llama.cpp.
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
    # The other half: no master root, so the guard stays closed rather than pinning the default.
    home = tmp_path / "home"
    (home / ".unsloth" / "studio").mkdir(parents = True)
    result = _main_probe({"HOME": str(home), "USERPROFILE": str(home)})

    assert result["exported"] is None
    assert result["marked"] == []
