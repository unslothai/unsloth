# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The permission wrapper waits for a just-stopped server to let go of auth.db.

The Windows UI lane stops its own server and then calls run-studio-permission-browser.sh on the
same UNSLOTH_STUDIO_HOME. A killed server on Windows can hold auth.db open for a few seconds, so
the wrapper's `rm -rf "$studio_home/auth"` failed with "Device or resource busy" and, under
`set -e`, failed the lane after every suite had passed (run 39669502772).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / ".github" / "scripts" / "run-studio-permission-browser.sh"
BASH = shutil.which("bash")


def _source() -> str:
    return SCRIPT.read_text(encoding = "utf-8")


def _code(text: str) -> str:
    return "\n".join(re.sub(r"(^|\s)#.*$", "", line) for line in text.split("\n"))


def _wipe_auth() -> str:
    match = re.search(r"^wipe_auth\(\) \{\n.*?^\}\n", _source(), re.M | re.S)
    assert match, "run-studio-permission-browser.sh no longer defines wipe_auth()"
    return match.group(0)


def _run(
    tmp_path: Path,
    busy_calls: int,
    tries: int = 30,
) -> tuple[subprocess.CompletedProcess, int]:
    """Runs the script's wipe_auth with an rm that reports a busy file for its first calls."""
    assert BASH, "bash is required to run the wrapper's own code"
    shims = tmp_path / "bin"
    shims.mkdir()
    calls = tmp_path / "rm_calls"
    real_rm = shutil.which("rm")
    (shims / "rm").write_text(
        "#!/usr/bin/env bash\n"
        f'n=$(( $(cat "{calls}" 2>/dev/null || echo 0) + 1 )); echo "$n" > "{calls}"\n'
        f'if [ "$n" -le {busy_calls} ]; then\n'
        "  echo \"rm: cannot remove 'auth/auth.db': Device or resource busy\" >&2; exit 1\n"
        "fi\n"
        f'exec "{real_rm}" "$@"\n',
        encoding = "utf-8",
    )
    (shims / "sleep").write_text("#!/usr/bin/env bash\nexit 0\n", encoding = "utf-8")
    for shim in shims.iterdir():
        shim.chmod(0o755)
    home = tmp_path / "home"
    (home / "auth").mkdir(parents = True)
    (home / "auth" / "auth.db").write_text("x", encoding = "utf-8")
    body = f'set -euo pipefail\nstudio_home="{home}"\n{_wipe_auth()}wipe_auth\necho wiped\n'
    env = dict(os.environ, PATH = f"{shims}{os.pathsep}{os.environ['PATH']}")
    env["STUDIO_AUTH_WIPE_TRIES"] = str(tries)
    result = subprocess.run([BASH, "-c", body], env = env, capture_output = True, text = True, timeout = 60)
    return result, int(calls.read_text())


def test_a_briefly_held_auth_dir_is_wiped_once_released(tmp_path):
    result, calls = _run(tmp_path, busy_calls = 4)
    assert result.returncode == 0, result.stderr
    assert "wiped" in result.stdout
    assert calls == 5
    assert not (tmp_path / "home" / "auth").exists()


def test_an_auth_dir_that_stays_held_fails_loudly(tmp_path):
    result, calls = _run(tmp_path, busy_calls = 99, tries = 5)
    assert result.returncode != 0
    assert "wiped" not in result.stdout
    assert "still held open" in result.stderr
    assert calls == 6, "five quiet tries and one final attempt that prints rm's own error"
    assert "Device or resource busy" in result.stderr


def test_a_release_on_the_final_attempt_is_a_success(tmp_path):
    result, calls = _run(tmp_path, busy_calls = 5, tries = 5)
    assert result.returncode == 0, result.stderr
    assert "wiped" in result.stdout
    assert calls == 6
    assert not (tmp_path / "home" / "auth").exists()


def test_the_wrapper_wipes_only_through_the_retry_before_booting():
    code = _code(_source())
    helper = _code(_wipe_auth())
    outside = code.replace(helper, "")
    assert not re.search(
        r"rm -rf \"\$studio_home/auth\"", outside
    ), "a bare rm of $studio_home/auth fails the lane while the previous server still holds auth.db"
    calls = [m.start() for m in re.finditer(r"^wipe_auth\s*$", code, re.M)]
    boot = code.index("unsloth studio -H")
    assert calls and calls[0] < boot, "the auth dir must be wiped before the server boots"
