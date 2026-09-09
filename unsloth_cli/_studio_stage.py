# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The passive half of the 805-807 background update contract.

Those shells run the INSTALLED (old) CLI with `--stage`, which clones the venv, runs this
package's setup inside the clone and on POSIX imports `finalize_for_activation` from the staged
tree. This wheel stages nothing, but those names must stay: stdlib only, importable under -I.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

STAGE_DIR_NAME = ".update-stage"
STAGE_ROOT_ENV = "UNSLOTH_STUDIO_STAGE_ROOT"
# Where a staged update parks the uv cache it used; the live marker is written only once the
# stage is accepted, so an update that never activates cannot redirect the environment.
UV_CACHE_MARKER = "uv-cache-dir"
SHELL_VERSION_ENV = "UNSLOTH_TAURI_SHELL_VERSION"
VENV_NAME = "unsloth_studio"
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


def venv_python(venv: Path) -> Path:
    if platform.system() == "Windows":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _is_venv_python_shebang(first_line: bytes) -> bool:
    if not first_line.startswith(b"#!"):
        return False
    target = first_line[2:].strip()
    return b"/bin/python" in target and not target.startswith(b"/usr/bin/env")


def _is_venv_python_shell_wrapper(lines: list[bytes]) -> bool:
    return (
        len(lines) >= 3
        and lines[0] == b"#!/bin/sh"
        and lines[1].startswith(b"'''exec' ")
        and b"/bin/python" in lines[1]
        and b'"$0" "$@"' in lines[1]
        and lines[2] == b"' '''"
    )


def _relocatable_script(body: bytes, original: int) -> bytes:
    """The rewritten script, never shorter than the one the installer recorded: RELOCATABLE_SHEBANG is
    82 bytes, so past ~68 characters of venv path the rewrite would shrink it and install_manifest.py
    calls that damage. The padding sits between shebang and body, unread by /bin/sh."""
    shebang = RELOCATABLE_SHEBANG.encode("utf-8")
    deficit = original - len(shebang) - len(body)
    if deficit <= 0:
        return shebang + body
    # `# ` plus the newline is 3 bytes. Longer than the original is fine; only shorter reads as damage.
    return shebang + b"# " + b"#" * max(deficit - 3, 0) + b"\n" + body


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
        lines = data.splitlines(keepends = True)
        if not lines:
            continue
        first_line = lines[0].rstrip(b"\r\n")
        if _is_venv_python_shebang(first_line):
            body = b"".join(lines[1:])
        elif _is_venv_python_shell_wrapper([line.rstrip(b"\r\n") for line in lines[:3]]):
            body = b"".join(lines[3:])
        else:
            continue
        script.write_bytes(_relocatable_script(body, len(data)))
        rewritten += 1
    return rewritten


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd = str(cwd),
        env = env,
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        timeout = PROBE_TIMEOUT_SECONDS,
    )


def probe_cli(venv: Path, env: dict[str, str]) -> None:
    result = _run(
        [str(venv_python(venv)), "-I", "-X", "utf8", "-m", "unsloth_cli", "-h"],
        cwd = venv.parent,
        env = env,
    )
    if result.returncode != 0:
        raise StageError(f"staged CLI failed to start: {result.stderr.strip()[-2000:]}")


def console_script(venv: Path) -> Path:
    return venv / "bin" / "unsloth"


def probe_console_script(venv: Path, env: dict[str, str]) -> None:
    """POSIX runs this file, not `python -m`, so it is the probe that counts. Windows is exempt: quarantine
    can take the unsigned launcher stub off a working install."""
    if platform.system() == "Windows":
        return
    script = console_script(venv)
    if not script.is_file():
        raise StageError(f"staged environment has no launcher at {script}")
    try:
        result = _run([str(script), "-h"], cwd = venv.parent, env = env)
    except OSError as exc:
        # A shebang naming a missing interpreter fails here, not with a return code.
        raise StageError(f"staged launcher is not executable: {exc}") from exc
    if result.returncode != 0:
        raise StageError(f"staged launcher failed to start: {result.stderr.strip()[-2000:]}")


def finalize_for_activation(root: Path) -> None:
    venv = root / VENV_NAME
    make_relocatable(venv)
    env = child_environment(root)
    probe_cli(venv, env)
    probe_console_script(venv, env)


def child_environment(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env[STAGE_ROOT_ENV] = str(root)
    scripts = root / VENV_NAME / ("Scripts" if platform.system() == "Windows" else "bin")
    env["PATH"] = str(scripts) + os.pathsep + env.get("PATH", "")
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    return env
