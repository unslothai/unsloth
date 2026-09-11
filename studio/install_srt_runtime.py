#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Install the locked native tool helper during setup, never during a tool call."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile


def _port_range(value):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(port) is not int for port in value)
    ):
        raise RuntimeError("Windows proxy range requires two integer ports.")
    low, high = value
    if not 1024 <= low < high <= 65535 or high - low >= 100:
        raise RuntimeError(
            "Windows proxy range must contain 2 to 100 ports between 1024 and 65535."
        )
    return [low, high]


def _range_available(ports):
    # Keep all candidate sockets open together so the check requires the whole
    # range. The native probe still handles races after these are released.
    with ExitStack() as stack:
        for port in range(ports[0], ports[1] + 1):
            listener = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            try:
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                listener.bind(("127.0.0.1", port))
            except OSError:
                return False
    return True


def _select_windows_range(preferred, *, explicit = False):
    preferred = preferred or [60080, 60089]
    if _range_available(preferred):
        return preferred
    if explicit:
        raise RuntimeError("The requested Windows proxy port range is reserved or in use.")
    for low in range(20000, 20640, 10):
        candidate = [low, low + 9]
        if _range_available(candidate):
            return candidate
    raise RuntimeError("No usable Windows proxy port range found; tool isolation remains blocked.")


def _existing_windows_ready(root):
    command = [
        sys.executable,
        "-I",
        "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from core.inference.srt_probe import probe; "
        "from core.inference.srt_windows_read_lease import shutdown; "
        "ready = probe(force=True)[0]; shutdown(); sys.exit(0 if ready else 1)",
        str(root.parents[2]),
    ]
    try:
        # Include the bounded read-grant shutdown and recovery budgets. Returning
        # success before cleanup would leave setup racing an active grant owner.
        result = subprocess.run(command, capture_output = True, timeout = 240, check = False)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def install(
    *,
    offline: bool = False,
    windows_install: bool = False,
    windows_proxy_port_range = None,
    windows_force: bool = False,
) -> int:
    if sys.platform != "win32":
        raise RuntimeError("SRT does not support this platform.")
    if windows_install and sys.platform != "win32":
        raise RuntimeError("--windows-install requires Windows.")
    if (windows_proxy_port_range is not None or windows_force) and not windows_install:
        raise RuntimeError("Windows configuration options require --windows-install.")
    root = Path(__file__).resolve().parent / "backend/core/inference/srt_runtime"
    settings_path = root / "installed-runtime-settings.json"
    selected_range = None
    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding = "utf-8"))
        if not isinstance(settings, dict) or set(settings) != {"windowsProxyPortRange"}:
            raise RuntimeError("Invalid installed SRT runtime settings.")
        selected_range = _port_range(settings["windowsProxyPortRange"])
    if windows_proxy_port_range is not None:
        selected_range = _port_range(windows_proxy_port_range)
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
            if (
                not windows_force
                and windows_proxy_port_range is None
                and _existing_windows_ready(root)
            ):
                print(
                    "Existing Windows SRT isolation passed its live check; keeping its account and firewall configuration."
                )
                return 0
            selected_range = _select_windows_range(
                selected_range, explicit = windows_proxy_port_range is not None
            )
            print(
                "Installing upstream SRT's local sandbox account, group, registry state and WFP filters; Windows may request elevation.",
                flush = True,
            )
            cli = root / "node_modules/@anthropic-ai/sandbox-runtime/dist/cli.js"
            install_command = [node, str(cli), "windows-install"]
            if selected_range is not None:
                install_command.extend(
                    ["--proxy-port-range", f"{selected_range[0]}-{selected_range[1]}"]
                )
            if windows_force:
                install_command.append("--force")
            subprocess.run(install_command, cwd = root, check = True, timeout = 150)
            if selected_range is not None:
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode = "w", encoding = "utf-8", dir = root, delete = False
                    ) as stream:
                        temporary = stream.name
                        json.dump({"windowsProxyPortRange": selected_range}, stream)
                    os.replace(temporary, settings_path)
                    temporary = None
                finally:
                    if temporary is not None:
                        Path(temporary).unlink(missing_ok = True)
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
    parser.add_argument(
        "--windows-proxy-port-range",
        nargs = 2,
        type = int,
        metavar = ("START", "END"),
        help = "Select 2 to 100 proxy ports for explicit Windows setup",
    )
    parser.add_argument(
        "--windows-force",
        action = "store_true",
        help = "Explicitly reconcile conflicting upstream Windows setup",
    )
    args = parser.parse_args()
    try:
        return install(
            offline = args.offline,
            windows_install = args.windows_install,
            windows_proxy_port_range = args.windows_proxy_port_range,
            windows_force = args.windows_force,
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(
            f"SRT setup unavailable: {exc}. Required mode will not execute on the host.",
            file = sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
