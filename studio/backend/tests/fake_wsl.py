#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A stand-in for wsl.exe on Linux CI: guest commands run on this host, every call is logged.

FAKE_WSL_LOG names a JSON-lines log; FAKE_WSL_STATUS is the ``--status`` exit code.
"""

import json
import os
import subprocess
import sys


def main(argv):
    with open(os.environ["FAKE_WSL_LOG"], "a", encoding = "utf-8") as log:
        log.write(
            json.dumps(
                {
                    "argv": argv,
                    "wslenv": os.environ.get("WSLENV", ""),
                    "shared": {
                        key.split("/")[0]: os.environ.get(key.split("/")[0])
                        for key in os.environ.get("WSLENV", "").split(":")
                        if key
                    },
                }
            )
            + "\n"
        )
    if argv[:1] == ["--status"]:
        return int(os.environ.get("FAKE_WSL_STATUS", "0"))
    if argv[:1] in (["--import"], ["--unregister"], ["--terminate"]):
        return 0
    if "--" not in argv:
        return 2
    command = argv[argv.index("--") + 1 :]
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
