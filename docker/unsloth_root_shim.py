#!/opt/unsloth-venv/bin/python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""restore root only for system-install commands in host-owned notebook kernels."""

import os, sys


_TOOLS = {
    "apt": "/usr/bin/apt",
    "apt-get": "/usr/bin/apt-get",
    "dpkg": "/usr/bin/dpkg",
}
_SUDO_FLAGS = {
    "-b",
    "--background",
    "-E",
    "--preserve-env",
    "-H",
    "--set-home",
    "-K",
    "--remove-timestamp",
    "-k",
    "--reset-timestamp",
    "-n",
    "--non-interactive",
    "-S",
    "--stdin",
}


def _become_root():
    try:
        os.setgid(0)
        os.setuid(0)
    except OSError as exc:
        raise SystemExit(f"unsloth notebook install could not restore root: {exc}") from exc


def _sudo_command(argv):
    command = []
    for arg in argv:
        if command:
            command.append(arg)
        elif arg == "--":
            continue
        elif arg in _SUDO_FLAGS or arg.startswith("--preserve-env="):
            continue
        elif arg.startswith("-"):
            raise SystemExit(f"unsloth notebook sudo does not support option: {arg}")
        else:
            command.append(arg)
    if not command:
        raise SystemExit("unsloth notebook sudo requires a command")
    return command


def main():
    tool = os.path.basename(sys.argv[0])
    _become_root()
    if tool == "sudo":
        command = _sudo_command(sys.argv[1:])
        os.execvpe(command[0], command, os.environ)
        return
    real = _TOOLS.get(tool)
    if real is None:
        raise SystemExit(f"unknown unsloth notebook root tool: {tool}")
    os.execv(real, [real, *sys.argv[1:]])


if __name__ == "__main__":
    main()
