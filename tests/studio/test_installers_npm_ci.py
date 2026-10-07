# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Studio installs of a committed package-lock.json go through `npm ci`, never `npm install`.

`npm install` rewrites the lockfile when package.json drifts from it and installs whatever
the drift asks for, so the audited lockfile stops being what gets installed.
"""

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
LOCKED_DIRS = ("studio", "studio/frontend", "studio/backend/core/data_recipe/oxc-validator")
# `npm install` survives only as the no-lockfile fallback and the global bun install.
_BARE_INSTALL = re.compile(r"\bnpm\s+install\b(?!\s+-g\b)")


def _read(rel):
    return (REPO / rel).read_text(encoding = "utf-8")


def test_lockfiles_are_committed():
    for d in LOCKED_DIRS:
        assert (REPO / d / "package-lock.json").is_file(), d


def test_installers_pick_npm_ci_when_a_lockfile_exists():
    sh = _read("studio/setup.sh")
    assert sh.count("[ -f package-lock.json ] && _NPM_INSTALL=ci") == 2
    assert 'npm "${_NPM_INSTALL:-install}"' in sh
    build = _read("build.sh")
    assert "[ -f package-lock.json ] && _npm_install=ci" in build
    ps1 = _read("studio/setup.ps1")
    assert ps1.count('if (Test-Path "package-lock.json") { "ci" } else { "install" }') == 3
    for rel, text in (("studio/setup.sh", sh), ("build.sh", build), ("studio/setup.ps1", ps1)):
        code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
        bare = [
            l.strip()
            for l in code.splitlines()
            if _BARE_INSTALL.search(l) and "Write-StudioLine" not in l
        ]
        assert bare == [], f"{rel}: {bare}"


def test_bun_only_runs_against_a_committed_bun_lock():
    gate = "if [ ! -f package-lock.json ] && [ -f bun.lock ] && command -v bun &>/dev/null; then"
    assert gate in _read("studio/setup.sh")
    assert gate in _read("build.sh")
    assert (
        '$UseBun = -not (Test-Path "package-lock.json") -and (Test-Path "bun.lock") -and'
        in _read("studio/setup.ps1")
    )
    for rel in ("studio/setup.sh", "build.sh"):
        assert 'bun install "${_NPM_REGISTRY_ARGS' not in _read(rel), rel
        assert 'bun install --frozen-lockfile "${_NPM_REGISTRY_ARGS' in _read(rel), rel
    assert "{ bun install @NpmRegistryArgs }" not in _read("studio/setup.ps1")


def test_bun_is_not_provisioned_when_npm_ci_will_run():
    sh = _read("studio/setup.sh")
    skip = sh.index('elif [ -f "$SCRIPT_DIR/frontend/package-lock.json" ]; then')
    assert (
        skip
        < sh.index('elif [ "$NODE_SOURCE" = bundled ]; then', skip)
        < sh.index("npm install -g bun", skip)
    )
    ps1 = _read("studio/setup.ps1")
    gate = 'if (-not (Test-Path (Join-Path $FrontendDir "package-lock.json")) -and -not (Get-Command bun'
    assert ps1.index(gate) < ps1.index("npm install -g bun")


def test_workflows_install_lockfiled_dirs_with_npm_ci():
    for name in (
        "release-desktop.yml",
        "studio-tauri-smoke.yml",
        "desktop-app-clean-machine-ci.yml",
    ):
        doc = yaml.safe_load(_read(f".github/workflows/{name}"))
        for job_id, job in doc["jobs"].items():
            for step in job.get("steps") or []:
                run = step.get("run") if isinstance(step, dict) else None
                if isinstance(run, str):
                    bare = [
                        l.strip()
                        for l in run.splitlines()
                        if _BARE_INSTALL.search(l) and not l.lstrip().startswith("#")
                    ]
                    assert bare == [], f"{name}:{job_id}: {bare}"
