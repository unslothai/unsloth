# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Run every requested browser and retain individual verdicts under temp/."""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from _playwright_robust import stop_process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--browsers", nargs = "+", default = ["chromium", "firefox", "webkit", "chrome", "msedge"]
    )
    parser.add_argument("--output", default = "temp/queue-validation/compatibility")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = (root / args.output).resolve()
    output.relative_to(root)
    output.mkdir(parents = True, exist_ok = True)
    temp = root / "temp/queue-validation/tmp"
    temp.mkdir(parents = True, exist_ok = True)
    env = {**os.environ, "TMPDIR": str(temp), "TMP": str(temp), "TEMP": str(temp)}
    verdicts = []
    for browser in args.browsers:
        for script in ("playwright_prompt_queue_actions.py", "playwright_composer_settings.py"):
            current = {
                **env,
                "PW_ENGINE": browser if browser in ("firefox", "webkit") else "chromium",
            }
            current.pop("PW_CHANNEL", None)
            current.pop("PW_EXECUTABLE", None)
            if browser in ("chrome", "msedge"):
                current["PW_CHANNEL"] = browser
            log = output / f"{browser}-{script.removesuffix('.py')}.log"
            with log.open("w", encoding = "utf-8") as stream:
                group = (
                    {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                    if os.name == "nt"
                    else {"start_new_session": True}
                )
                process = subprocess.Popen(
                    [sys.executable, str(root / "tests/studio" / script)],
                    cwd = root,
                    env = current,
                    stdout = stream,
                    stderr = subprocess.STDOUT,
                    **group,
                )
                try:
                    code = process.wait(timeout = 180)
                except subprocess.TimeoutExpired:
                    stop_process(process)
                    stream.write("FAIL: simulation exceeded 180 seconds\n")
                    code = 124
            verdict = {
                "os": platform.platform(),
                "browser": browser,
                "script": script,
                "exit_code": code,
                "log": str(log.relative_to(root)),
            }
            verdicts.append(verdict)
            print(json.dumps(verdict), flush = True)
    (output / "results.json").write_text(json.dumps(verdicts, indent = 2) + "\n", encoding = "utf-8")
    return int(any(v["exit_code"] for v in verdicts))


if __name__ == "__main__":
    raise SystemExit(main())
