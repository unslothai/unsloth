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

"""CLI and backend must agree on UNSLOTH_HOME, or studio.db is written where the CLI never looks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse = True)
def isolated_home(tmp_path, monkeypatch):
    for key in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(key, str(tmp_path / "home"))


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


def _probe(env_overrides: dict, prefix: Path | None = None) -> dict:
    env = dict(os.environ)
    # Clear UNSLOTH_LLAMA_CPP_PATH too, as _main_probe does: an inherited value measures the runner.
    for key in ("UNSLOTH_HOME", "UNSLOTH_STUDIO_HOME", "STUDIO_HOME", "UNSLOTH_LLAMA_CPP_PATH"):
        env.pop(key, None)
    env.update(env_overrides)
    env["PYTHONPATH"] = str(REPO_ROOT)
    source = _PROBE.format(backend = str(REPO_ROOT / "studio" / "backend"))
    if prefix is not None:
        source = f"import sys\nsys.prefix = {str(prefix)!r}\n" + source
    out = subprocess.run(
        [sys.executable, "-c", source],
        capture_output = True,
        text = True,
        cwd = str(REPO_ROOT),
        env = env,
        check = True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("installed", [False, True])
def test_cli_infers_only_a_managed_interpreter_root(tmp_path, installed):
    root = tmp_path / "custom"
    if installed:
        (root / "share").mkdir(parents = True)
        (root / "share" / "studio.conf").write_text("installed\n", encoding = "utf-8")
    result = _probe({}, prefix = root / "unsloth_studio")
    expected = root if installed else tmp_path / "home" / ".unsloth" / "studio"
    assert result["cli"] == str(expected)
    assert result["backend"] == str(expected)
    assert result["cli_is_custom"] is installed


def test_cli_keeps_custom_root_when_resolution_fails(tmp_path, monkeypatch):
    from unsloth_cli.commands import studio

    custom = tmp_path / "custom"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(custom))

    def unavailable(self):
        raise OSError("filesystem unavailable")

    monkeypatch.setattr(Path, "resolve", unavailable)
    assert studio._resolve_studio_home() == (custom, True)


def test_unsloth_home_resolves_to_one_studio_root(tmp_path):
    master = tmp_path / "portable"
    result = _probe({"UNSLOTH_HOME": str(master)})

    assert result["backend"] == str(master / "studio")
    assert result["cli"] == result["backend"]
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
    # run.py keeps a non-blank UNSLOTH_LLAMA_CPP_PATH, so a wrong export pins every worker.
    master = tmp_path / "portable"
    result = _probe({"UNSLOTH_HOME": str(master)})

    assert result["cli_llama"] == str(master / "llama.cpp")
    assert result["cli_llama"] == result["backend_llama"]


def test_the_cli_recovers_a_recorded_master_root_for_the_setup_subprocess(tmp_path):
    # The backend recovers the root from the note, so the CLI must export it to setup too.
    master = tmp_path / "portable"
    studio = master / "studio"
    (studio / "share").mkdir(parents = True)
    (studio / "share" / ".unsloth-master-root").write_text(f"{master}\n", encoding = "utf-8")
    result = _probe({"UNSLOTH_STUDIO_HOME": str(studio)})

    assert result["exported_master"] == str(master)
    assert result["cli_llama"] == str(master / "llama.cpp")
    assert result["cli_llama"] == result["backend_llama"]


def test_the_cli_and_the_backend_accept_the_same_nested_studio_root(tmp_path):
    # Studio dir inside the root but not its studio/ child; a stricter CLI rule would diverge.
    master = tmp_path / "root"
    studio = master / "custom" / "studio"
    (studio / "share").mkdir(parents = True)
    (studio / "share" / ".unsloth-master-root").write_text(f"{master}\n", encoding = "utf-8")
    result = _probe({"UNSLOTH_STUDIO_HOME": str(studio)})

    assert result["exported_master"] == str(master)
    assert result["cli"] == str(studio)
    assert result["cli_llama"] == str(master / "llama.cpp")
    assert result["cli_llama"] == result["backend_llama"]


def test_the_cli_refuses_a_note_carried_in_from_another_master_root(tmp_path):
    # A copied Studio tree keeps a note naming the old root; it must lie inside that root.
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
    home = tmp_path / "home"
    install_id = home / ".unsloth" / "studio" / "share" / "studio_install_id"
    install_id.parent.mkdir(parents = True)
    install_id.write_text("ab" * 32 + "\n", encoding = "utf-8")

    result = _probe(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "UNSLOTH_HOME": str(tmp_path / "portable"),
        }
    )

    legacy = str(home / ".unsloth" / "studio")
    assert result["cli"] == legacy
    # Not custom: false lets installers replace the legacy tree without an owner marker.
    assert result["cli_is_custom"] is False
    assert result["exported_studio_home"] == legacy
    assert result["backend"] == legacy


def test_a_new_master_install_wins_over_an_existing_legacy_install(tmp_path):
    home = tmp_path / "home"
    master = tmp_path / "portable"
    for root in (home / ".unsloth" / "studio", master / "studio"):
        (root / "share").mkdir(parents = True)
        (root / "share" / "studio.conf").write_text("installed\n", encoding = "utf-8")
    result = _probe({"HOME": str(home), "USERPROFILE": str(home), "UNSLOTH_HOME": str(master)})
    assert result["cli"] == str(master / "studio")
    assert result["backend"] == result["cli"]
    assert result["cli_is_custom"] is True
    assert result["cli_llama"] == str(master / "llama.cpp")


def test_update_keeps_an_existing_legacy_install(tmp_path):
    home = tmp_path / "home"
    legacy = home / ".unsloth" / "studio"
    (legacy / "share").mkdir(parents = True)
    (legacy / "share" / "studio_install_id").write_text("ab" * 32 + "\n", encoding = "utf-8")
    source = """
import contextlib, json, os
from unsloth_cli.commands import studio as cli
calls = []
def capture(**kwargs):
    calls.append(os.environ.get('UNSLOTH_STUDIO_HOME'))
cli._studio_runtime_launch_guard = lambda **kwargs: contextlib.nullcontext()
cli._studio_runtime_gate.consume_runtime_gate_handoff = lambda: False
cli._studio_runtime_gate.ensure_managed_environment_is_idle = lambda root: None
class Transaction(contextlib.nullcontext):
    def __enter__(self): return self
    def validate_launcher(self): pass
cli._WindowsLauncherUpdateTransaction = Transaction
cli._run_setup_script = capture
cli._refresh_desktop_shortcuts = capture
cli.update(local=False, package='unsloth', verbose=False, verify=False, stage=False)
print(json.dumps(calls))
"""
    env = dict(os.environ, HOME = str(home), USERPROFILE = str(home), PYTHONPATH = str(REPO_ROOT))
    for key in ("UNSLOTH_STUDIO_HOME", "STUDIO_HOME", "UNSLOTH_STUDIO_STAGE_ROOT"):
        env.pop(key, None)
    env["UNSLOTH_HOME"] = str(tmp_path / "portable")
    run = subprocess.run(
        [sys.executable, "-c", source],
        env = env,
        cwd = REPO_ROOT,
        capture_output = True,
        text = True,
        check = True,
        timeout = 60,
    )
    assert json.loads(run.stdout.strip().splitlines()[-1]) == [str(legacy), str(legacy)]
    assert not (tmp_path / "portable").exists()


def test_a_whitespace_studio_home_does_not_defeat_the_export(tmp_path):
    """Export guard must strip whitespace too: every resolver treats a whitespace-only value as unset."""
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
        # Runtimes live beside studio/; a whitespace value would be scanned as a directory.
        assert result["cli_llama"] == str(tmp_path / "portable" / "llama.cpp"), inherited


def test_a_legacy_install_still_keeps_llama_cpp_at_the_legacy_path(tmp_path):
    home = tmp_path / "home"
    (home / ".unsloth" / "studio").mkdir(parents = True)
    result = _probe({"HOME": str(home), "USERPROFILE": str(home)})

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
    # Under run.py main.py would mark the wrong path managed, unlike a direct uvicorn launch.
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
    # A portable install at the legacy Studio path still owns <root>/llama.cpp.
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
    home = tmp_path / "home"
    (home / ".unsloth" / "studio").mkdir(parents = True)
    result = _main_probe({"HOME": str(home), "USERPROFILE": str(home)})

    assert result["exported"] is None
    assert result["marked"] == []
