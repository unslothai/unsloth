# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

STAGE_DIR_NAME = ".update-stage"
STAGE_ROOT_ENV = "UNSLOTH_STUDIO_STAGE_ROOT"
SHELL_VERSION_ENV = "UNSLOTH_TAURI_SHELL_VERSION"
READY_MARKER = "READY.json"
VENV_NAME = "unsloth_studio"
MIN_FREE_BYTES = 1 << 30
PROBE_TIMEOUT_SECONDS = 300

RELOCATABLE_SHEBANG = (
    "#!/bin/sh\n"
    '\'\'\'exec\' "$(dirname -- "$(realpath -- "$0")")"/\'python\' "$0" "$@"\n'
    "' '''\n"
)


class StageError(RuntimeError):
    pass


def is_staging() -> bool:
    return bool(os.environ.get(STAGE_ROOT_ENV))


def runtime_root(studio_home: Path) -> Path:
    override = (os.environ.get(STAGE_ROOT_ENV) or "").strip()
    return Path(override) if override else studio_home


def stage_root(studio_home: Path) -> Path:
    return studio_home / STAGE_DIR_NAME


def venv_python(venv: Path) -> Path:
    if platform.system() == "Windows":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _copy_command(source: Path, destination: Path) -> Optional[list[str]]:
    system = platform.system()
    if system == "Darwin":
        return ["cp", "-c", "-R", str(source), str(destination)]
    if system == "Linux":
        return ["cp", "-a", "--reflink=auto", str(source), str(destination)]
    return None


def clone_tree(source: Path, destination: Path) -> None:
    command = _copy_command(source, destination)
    if command is not None:
        result = subprocess.run(command, capture_output = True, text = True)
        if result.returncode == 0:
            return
        shutil.rmtree(destination, ignore_errors = True)
    shutil.copytree(source, destination, symlinks = True)


def _is_venv_python_shebang(first_line: bytes) -> bool:
    if not first_line.startswith(b"#!"):
        return False
    target = first_line[2:].strip()
    return b"/bin/python" in target and not target.startswith(b"/usr/bin/env")


def make_relocatable(venv: Path) -> int:
    cfg = venv / "pyvenv.cfg"
    lines = cfg.read_text(encoding = "utf-8").splitlines()
    if not any(line.split("=", 1)[0].strip() == "relocatable" for line in lines):
        lines.append("relocatable = true")
        cfg.write_text("\n".join(lines) + "\n", encoding = "utf-8")
    if platform.system() == "Windows":
        return 0
    rewritten = 0
    for script in (venv / "bin").iterdir():
        if script.is_symlink() or not script.is_file():
            continue
        data = script.read_bytes()
        first_line, newline, rest = data.partition(b"\n")
        if not newline or not _is_venv_python_shebang(first_line):
            continue
        script.write_bytes(RELOCATABLE_SHEBANG.encode("utf-8") + rest)
        rewritten += 1
    return rewritten


def discard(root: Path) -> None:
    shutil.rmtree(root, ignore_errors = True)


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd = str(cwd),
        env = env,
        capture_output = True,
        text = True,
        timeout = PROBE_TIMEOUT_SECONDS,
    )


def installed_version(venv: Path, env: dict[str, str]) -> str:
    result = _run(
        [
            str(venv_python(venv)),
            "-c",
            "import importlib.metadata as m; print(m.version('unsloth'))",
        ],
        cwd = venv.parent,
        env = env,
    )
    version = result.stdout.strip()
    if result.returncode != 0 or not version:
        raise StageError(f"staged environment has no importable unsloth: {result.stderr.strip()}")
    return version


def probe_cli(venv: Path, env: dict[str, str]) -> None:
    result = _run(
        [str(venv_python(venv)), "-X", "utf8", "-m", "unsloth_cli", "-h"],
        cwd = venv.parent,
        env = env,
    )
    if result.returncode != 0:
        raise StageError(f"staged CLI failed to start: {result.stderr.strip()[-2000:]}")


def child_environment(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env[STAGE_ROOT_ENV] = str(root)
    env.pop("PYTHONHOME", None)
    env.pop("VIRTUAL_ENV", None)
    return env


def run_staged_update(root: Path, args: list[str]) -> int:
    python = venv_python(root / VENV_NAME)
    command = [str(python), "-X", "utf8", "-m", "unsloth_cli", "studio", "update", *args]
    return subprocess.call(command, cwd = str(root), env = child_environment(root))


def write_ready_marker(root: Path, backend_version: str, shell_version: Optional[str]) -> Path:
    marker = root / READY_MARKER
    payload = {
        "backend_version": backend_version,
        "shell_version": shell_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    marker.write_text(json.dumps(payload, indent = 2) + "\n", encoding = "utf-8")
    return marker


def stage(
    studio_home: Path,
    *,
    update_args: list[str],
    echo: Callable[[str], None],
    run_update: Callable[[Path, list[str]], int] = run_staged_update,
) -> dict:
    live = studio_home / VENV_NAME
    if not (live / "pyvenv.cfg").is_file():
        raise StageError(f"no managed environment at {live}")
    root = stage_root(studio_home)
    discard(root)
    if shutil.disk_usage(studio_home).free < MIN_FREE_BYTES:
        raise StageError("not enough free disk space to prepare an update")
    root.mkdir(parents = True)
    try:
        echo("[TAURI:STEP] clone")
        clone_tree(live, root / VENV_NAME)
        make_relocatable(root / VENV_NAME)
        echo("[TAURI:STEP] update")
        if run_update(root, update_args) != 0:
            raise StageError("staged update failed")
        echo("[TAURI:STEP] verify")
        env = child_environment(root)
        version = installed_version(root / VENV_NAME, env)
        probe_cli(root / VENV_NAME, env)
        shell_version = (os.environ.get(SHELL_VERSION_ENV) or "").strip() or None
        write_ready_marker(root, version, shell_version)
    except BaseException:
        discard(root)
        raise
    return {"backend_version": version, "root": str(root)}
