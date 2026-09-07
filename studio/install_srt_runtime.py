#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Install the locked native tool helper during setup, never during a tool call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def install(*, offline: bool = False, windows_install: bool = False) -> int:
    if sys.platform not in ("linux", "darwin", "win32"):
        raise RuntimeError("SRT does not support this platform.")
    if windows_install and sys.platform != "win32":
        raise RuntimeError("--windows-install requires Windows.")
    root = Path(__file__).resolve().parent / "backend/core/inference/srt_runtime"
    node, npm = shutil.which("node"), shutil.which("npm")
    if not node or not npm:
        raise RuntimeError(
            "SRT requires Node >=20.11 and npm during setup; Required remains unavailable."
        )
    version = subprocess.run(
        [node, "--version"], check = True, capture_output = True, text = True
    ).stdout.strip()
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", version)
    if not match or tuple(map(int, match.groups())) < (20, 11, 0):
        raise RuntimeError("SRT requires Node >=20.11; Required remains unavailable.")
    manifest = json.loads((root / "package.json").read_text(encoding = "utf-8"))
    lock = json.loads((root / "package-lock.json").read_text(encoding = "utf-8"))
    dependency = "@anthropic-ai/sandbox-runtime"
    if manifest.get("dependencies", {}).get(dependency) != "0.0.75":
        raise RuntimeError("Unexpected SRT dependency pin.")
    if lock.get("packages", {}).get("node_modules/" + dependency, {}).get("version") != "0.0.75":
        raise RuntimeError("SRT lock does not match the reviewed version.")
    patch = root / "apply_patch.mjs"
    if not patch.is_file():
        raise RuntimeError("SRT reviewed empty-root and network-forwarder patch is missing.")
    # Invoke npm's JS entrypoint directly on Windows: npm.cmd otherwise needs
    # command-shell parsing, including installation paths containing spaces.
    npm_command = [npm]
    if sys.platform == "win32":
        npm_cli = Path(npm).parent / "node_modules/npm/bin/npm-cli.js"
        if not npm_cli.is_file():
            raise RuntimeError("Cannot locate the selected npm CLI; rerun Node setup.")
        npm_command = [node, str(npm_cli)]
    command = [*npm_command, "ci", "--ignore-scripts", "--no-audit", "--no-fund"]
    if offline:
        command.append("--offline")
    subprocess.run(command, cwd = root, check = True, timeout = 300)
    subprocess.run([node, str(patch)], cwd = root, check = True, timeout = 30)
    subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            "const m = await import(process.argv[1]); m.verifyInstallation();",
            (root / "bridge.mjs").as_uri(),
        ],
        cwd = root,
        check = True,
        timeout = 30,
    )
    if sys.platform == "win32":
        if windows_install:
            print(
                "Installing upstream SRT's local sandbox account, group, registry state and WFP filters; Windows may request elevation.",
                flush = True,
            )
            cli = root / "node_modules/@anthropic-ai/sandbox-runtime/dist/cli.js"
            subprocess.run([node, str(cli), "windows-install"], cwd = root, check = True, timeout = 150)
        else:
            print(
                "SRT helper verified. Complete one-time privileged setup with: python studio/install_srt_runtime.py --windows-install"
            )
        print(
            "Windows SRT uses the upstream alpha account/ACL/WFP model; availability requires a live probe."
        )
    elif sys.platform == "darwin":
        print(
            "SRT helper verified. macOS uses native Seatbelt; install ripgrep if missing. Availability requires a live probe."
        )
    elif not shutil.which("bwrap"):
        print(
            "SRT dependencies verified; install bubblewrap through your OS package manager. Required remains unavailable until its live probe passes."
        )
    else:
        print(
            "SRT dependencies verified; Linux Required availability is determined by the live probe (Preview, not fully qualified)."
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--offline", action = "store_true", help = "Use only the existing npm cache; no registry access"
    )
    parser.add_argument(
        "--windows-install",
        action = "store_true",
        help = "Explicitly provision upstream Windows sandbox account and WFP filters (may request elevation)",
    )
    args = parser.parse_args()
    try:
        return install(offline = args.offline, windows_install = args.windows_install)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(
            f"SRT setup unavailable: {exc}. Required mode will not execute on the host.",
            file = sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
