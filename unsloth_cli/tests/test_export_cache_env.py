# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth export` / `list-checkpoints` must pin the cache env before importing the export
backend.

Neither calls ensure_studio_backend_path(), so before the seeding moved into
studio_backend_imports() they reached studio.backend.core.export with UNSLOTH_COMPILE_LOCATION
unset, and the export subprocess (mp spawn, which inherits os.environ) then let unsloth_zoo
resolve its relative default against the shell's working directory (#8865).
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Runs in its own interpreter: the seeding is process-global (os.environ plus a one-shot flag),
# so an in-process assertion would depend on test ordering.
PROBE = textwrap.dedent(
    """
    import json, os, sys, types

    import typer
    from typer.testing import CliRunner

    command_name = sys.argv[1]
    recorded = {}

    # Stand in for the heavy backend (unsloth, torch) so the probe stays
    # import-light. The CLI's own import statement, and therefore everything that
    # has to happen before it, is untouched.
    class ExportBackend:
        def __init__(self):
            recorded["compile_location"] = os.environ.get("UNSLOTH_COMPILE_LOCATION")

        def load_checkpoint(self, **kwargs):
            return True, "loaded"

        def export_merged_model(self, **kwargs):
            return True, "exported", kwargs["save_directory"]

        def scan_checkpoints(self, **kwargs):
            return []

    # Each stub keeps the REAL directory on __path__ instead of []. The point is to skip
    # the heavy studio.backend.core.export import, not to make the whole studio tree
    # unimportable: unsloth_cli/__init__ eagerly imports commands.start, which pulls
    # studio.backend.utils.coding_agents, and an empty __path__ turned that into
    # ModuleNotFoundError. Submodules still load from disk; the package __init__ files
    # are the thing that never runs, which is what kept this probe light to begin with.
    repo_root = sys.argv[2]
    for name, relative in (
        ("studio", "studio"),
        ("studio.backend", "studio/backend"),
        ("studio.backend.core", "studio/backend/core"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [os.path.join(repo_root, *relative.split("/"))]
        sys.modules[name] = module
    export_module = types.ModuleType("studio.backend.core.export")
    export_module.ExportBackend = ExportBackend
    sys.modules["studio.backend.core.export"] = export_module

    from unsloth_cli.commands import export as export_command

    app = typer.Typer()
    if command_name == "export":
        app.command()(export_command.export)
        args = ["checkpoint", "exported"]
    else:
        app.command()(export_command.list_checkpoints)
        args = []
    result = CliRunner().invoke(app, args)
    recorded["exit_code"] = result.exit_code
    print("PROBE " + json.dumps(recorded))
    """
)


@pytest.mark.parametrize("command_name", ["export", "list-checkpoints"])
def test_export_commands_seed_compile_location(tmp_path, command_name):
    unsloth_home = tmp_path / "unsloth"
    workdir = tmp_path / "cwd"
    workdir.mkdir()

    env = dict(os.environ)
    env.pop("UNSLOTH_COMPILE_LOCATION", None)
    env.update(
        HOME = str(tmp_path / "home"),
        UNSLOTH_HOME = str(unsloth_home),
        PYTHONPATH = str(REPO_ROOT),
    )
    completed = subprocess.run(
        [sys.executable, "-c", PROBE, command_name, str(REPO_ROOT)],
        cwd = str(workdir),
        env = env,
        capture_output = True,
        text = True,
        timeout = 300,
    )
    assert completed.returncode == 0, completed.stderr
    line = next(l for l in completed.stdout.splitlines() if l.startswith("PROBE "))
    recorded = json.loads(line[len("PROBE ") :])

    assert recorded["exit_code"] == 0, recorded
    location = recorded["compile_location"]
    assert location, "UNSLOTH_COMPILE_LOCATION was unset when the export backend was imported"
    # Absolute and inside the install root, not resolved against the CWD.
    assert Path(location).is_absolute()
    assert Path(location) == unsloth_home / "studio" / "compiled_cache"
    assert not (workdir / "unsloth_compiled_cache").exists()
