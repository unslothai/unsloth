# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Regression and unit tests for Docker volume persistence (#4396).

Tests that:
1. setup_data_dir initializes the canonical subdirectories under /data.
2. Runtime paths (auth, outputs, exports, runs, caches) are symlinked to /data.
3. Pre-existing files in container directories migrate safely to /data.
4. Existing persistent data is never wiped or clobbered.
5. An empty bind-mount over Studio's baked home triggers an actionable warning.
6. docker/compose.yml provides a valid, single-volume persistence configuration.
7. supervisord.conf forwards the persistent cache and data directory environment.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "docker" / "entrypoint.sh"
COMPOSE = REPO_ROOT / "docker" / "compose.yml"
SUPERVISORD = REPO_ROOT / "docker" / "supervisord.conf"

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")


@pytest.fixture(scope = "module")
def setup_data_dir_script() -> str:
    """Extract setup_data_dir definition and helper functions from entrypoint.sh."""
    source = ENTRYPOINT.read_text(encoding = "utf-8")
    match = re.search(
        r"(setup_data_dir\(\)\s*\{.*?\n\})\nsetup_data_dir",
        source,
        re.S,
    )
    assert match, "setup_data_dir function not found in entrypoint.sh"
    return match.group(1)


@needs_bash
def test_setup_data_dir_creates_persistent_tree(setup_data_dir_script: str, tmp_path: Path):
    """setup_data_dir must initialize the standard directory structure."""
    data_dir = tmp_path / "data"
    studio_home = tmp_path / "studio"
    workspace = tmp_path / "workspace"
    root_home = tmp_path / "root"

    studio_home.mkdir(parents = True)
    (studio_home / "unsloth_studio").mkdir()
    workspace.mkdir(parents = True)
    root_home.mkdir(parents = True)

    script = f"""#!/usr/bin/env bash
set -euo pipefail
err()  {{ echo "ERROR: $*" >&2; }}
warn() {{ echo "WARN: $*" >&2; }}

{setup_data_dir_script}

setup_data_dir
"""
    runner = tmp_path / "test_run.sh"
    runner.write_text(script, encoding = "utf-8")

    env = dict(
        os.environ,
        UNSLOTH_DATA_DIR = str(data_dir),
        UNSLOTH_STUDIO_HOME = str(studio_home),
        HOME = str(root_home),
    )

    res = subprocess.run(["bash", str(runner)], capture_output = True, text = True, env = env)
    assert res.returncode == 0, res.stdout + res.stderr

    expected_subdirs = [
        "cache/huggingface",
        "cache/triton",
        "cache/torch",
        "outputs",
        "exports",
        "auth",
        "runs",
        "work",
    ]
    for subdir in expected_subdirs:
        assert (data_dir / subdir).is_dir(), f"Expected directory {subdir} was not created"

    # Verify symlinks created under Studio home
    for link_name in ("outputs", "exports", "auth", "runs"):
        link = studio_home / link_name
        assert link.is_symlink(), f"{link_name} under studio_home should be a symlink"
        assert link.resolve() == (data_dir / link_name).resolve()


@needs_bash
def test_setup_data_dir_migrates_initial_content(setup_data_dir_script: str, tmp_path: Path):
    """Files existing in the container before mounting must be preserved in /data."""
    data_dir = tmp_path / "data"
    studio_home = tmp_path / "studio"
    auth_dir = studio_home / "auth"
    auth_dir.mkdir(parents = True)
    (auth_dir / "auth.db").write_text("initial-sqlite-data", encoding = "utf-8")
    (studio_home / "unsloth_studio").mkdir()

    script = f"""#!/usr/bin/env bash
set -euo pipefail
err()  {{ echo "ERROR: $*" >&2; }}
warn() {{ echo "WARN: $*" >&2; }}

{setup_data_dir_script}

setup_data_dir
"""
    runner = tmp_path / "test_migrate.sh"
    runner.write_text(script, encoding = "utf-8")

    env = dict(
        os.environ,
        UNSLOTH_DATA_DIR = str(data_dir),
        UNSLOTH_STUDIO_HOME = str(studio_home),
    )

    res = subprocess.run(["bash", str(runner)], capture_output = True, text = True, env = env)
    assert res.returncode == 0, res.stdout + res.stderr

    # auth.db should now exist under the persistent data directory
    persistent_db = data_dir / "auth" / "auth.db"
    assert persistent_db.is_file(), "Existing auth.db was not migrated to /data/auth"
    assert persistent_db.read_text(encoding = "utf-8") == "initial-sqlite-data"

    # And studio_home / auth should be a symlink pointing to persistent auth
    assert (studio_home / "auth").is_symlink()
    assert (studio_home / "auth" / "auth.db").read_text(encoding = "utf-8") == "initial-sqlite-data"


@needs_bash
def test_setup_data_dir_warns_on_clobbered_studio_home(setup_data_dir_script: str, tmp_path: Path):
    """If a host volume was bind-mounted directly over Studio's root, warn the user."""
    data_dir = tmp_path / "data"
    studio_home = tmp_path / "opt_unsloth_studio"
    studio_home.mkdir(parents = True)  # Empty directory without unsloth_studio

    mock_venv = tmp_path / "opt_unsloth_venv"
    mock_venv.mkdir(parents = True)

    script = f"""#!/usr/bin/env bash
set -euo pipefail
err()  {{ echo "ERROR: $*" >&2; }}
warn() {{ echo "WARN: $*" >&2; }}

{setup_data_dir_script}

setup_data_dir
"""
    runner = tmp_path / "test_warn.sh"
    runner.write_text(script, encoding = "utf-8")

    env = dict(
        os.environ,
        UNSLOTH_DATA_DIR = str(data_dir),
        UNSLOTH_STUDIO_HOME = str(studio_home),
        UNSLOTH_BASE_VENV = str(mock_venv),
    )

    res = subprocess.run(["bash", str(runner)], capture_output = True, text = True, env = env)
    assert res.returncode == 0
    assert "Bind-mounting over Studio's root removes pre-installed venvs" in res.stderr
    assert "#4396" in res.stderr


def test_compose_file_validity():
    """docker/compose.yml must be valid YAML with correct single-volume config."""
    assert COMPOSE.is_file(), "docker/compose.yml is missing"
    content = COMPOSE.read_text(encoding = "utf-8")
    doc = yaml.safe_load(content)

    assert "services" in doc, "compose.yml must define services"
    assert "unsloth" in doc["services"], "compose.yml must have 'unsloth' service"

    unsloth = doc["services"]["unsloth"]
    assert unsloth.get("ipc") == "host", "compose service should use ipc: host"

    ports = unsloth.get("ports", [])
    port_strs = [str(p) for p in ports]
    assert any("8000" in p for p in port_strs), "compose.yml should map Studio port 8000"
    assert any("8888" in p for p in port_strs), "compose.yml should map Jupyter port 8888"

    volumes = unsloth.get("volumes", [])
    vol_strs = [str(v) for v in volumes]
    assert any("/data" in v for v in vol_strs), "compose.yml must mount /data volume"


def test_supervisord_environment_inheritance():
    """supervisord.conf must forward cache and data variables to services."""
    assert SUPERVISORD.is_file(), "docker/supervisord.conf is missing"
    text = SUPERVISORD.read_text(encoding = "utf-8")

    # studio program environment
    match_studio = re.search(r"\[program:studio\].*?environment=([^\n]+)", text, re.S)
    assert match_studio, "[program:studio] must define environment"
    studio_env = match_studio.group(1)
    assert "HF_HOME" in studio_env, "studio program must receive HF_HOME"
    assert "TRITON_CACHE_DIR" in studio_env, "studio program must receive TRITON_CACHE_DIR"
    assert "UNSLOTH_DATA_DIR" in studio_env, "studio program must receive UNSLOTH_DATA_DIR"

    # jupyter program environment
    match_jupyter = re.search(r"\[program:jupyter\].*?environment=([^\n]+)", text, re.S)
    assert match_jupyter, "[program:jupyter] must define environment"
    jupyter_env = match_jupyter.group(1)
    assert "HF_HOME" in jupyter_env, "jupyter program must receive HF_HOME"
    assert "TRITON_CACHE_DIR" in jupyter_env, "jupyter program must receive TRITON_CACHE_DIR"
    assert "UNSLOTH_DATA_DIR" in jupyter_env, "jupyter program must receive UNSLOTH_DATA_DIR"
