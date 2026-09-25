# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""
agent-guides-install.sh must not go red because a vendor dropped an option we
only passed to save time.

The curl|bash installers track the vendor's main branch and exit on any option
they do not know. On 2026-09-24 hermes-agent rewrote its argument parser without
--no-skills, and every PR's `connection (hermes, stable)` job failed three times
over with `unknown option: --no-skills` before the agent was ever launched.

These run the script's own curl_bash against a local installer shaped like the
vendor's, so they need bash and curl but no network.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "agent-guides-install.sh"
HERMES_URL = "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not shutil.which("bash") or not shutil.which("curl"),
    reason = "runs the installer helpers under bash with a file:// installer",
)


def _source() -> str:
    return SCRIPT.read_text(encoding = "utf-8")


def _function(name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", _source(), re.M | re.S)
    assert match, f"{name} is gone from {SCRIPT.name}"
    return match.group(0)


def _hermes_args() -> list[str]:
    """The options the hermes recipe passes, exactly as the script writes them."""
    match = re.search(
        rf'curl_bash "{re.escape(HERMES_URL)}" \\\n(.*?)\\\n',
        _source(),
        re.S,
    )
    assert match, "the hermes recipe no longer calls curl_bash on the vendor installer"
    return shlex.split(match.group(1))


def _installer(tmp_path: Path, known: list[str]) -> Path:
    """A vendor installer: records its arguments, and exits on any it does not parse."""
    record = tmp_path / "argv"
    labels = "|".join(known)
    script = tmp_path / "install.sh"
    # Recorded before parsing, so a rejected run still shows what it was given.
    script.write_text(
        f"printf '%s\\n' \"$@\" > {shlex.quote(str(record))}\n"
        "while [ $# -gt 0 ]; do\n"
        '    case "$1" in\n'
        f"        {labels}) shift ;;\n"
        '        *) echo "unknown option: $1" >&2; exit 1 ;;\n'
        "    esac\n"
        "done\n",
        encoding = "utf-8",
    )
    return script


def _curl_bash(tmp_path: Path, installer: Path, args: list[str]) -> subprocess.CompletedProcess:
    helpers = tmp_path / "helpers.sh"
    # sleep is the retry backoff: three rejected attempts would otherwise cost a minute.
    helpers.write_text(
        "sleep() { :; }\n" + _function("installer_args") + _function("curl_bash"),
        encoding = "utf-8",
    )
    command = (
        "set -uo pipefail\n"
        f"LOG={shlex.quote(str(tmp_path / 'install.log'))}\n"
        f". {shlex.quote(str(helpers))}\n"
        f"curl_bash {shlex.quote(installer.as_uri())} {' '.join(shlex.quote(a) for a in args)}\n"
    )
    return subprocess.run(["bash", "-c", command], capture_output = True, text = True, timeout = 60)


def _received(tmp_path: Path) -> list[str]:
    return (tmp_path / "argv").read_text(encoding = "utf-8").split()


def test_the_hermes_recipe_installs_once_the_vendor_drops_no_skills(tmp_path: Path) -> None:
    # The parser hermes-agent shipped on 2026-09-24.
    installer = _installer(
        tmp_path,
        ["--non-interactive", "--skip-setup", "--skip-browser|--no-playwright|-SkipBrowser"],
    )
    result = _curl_bash(tmp_path, installer, _hermes_args())
    assert result.returncode == 0, (tmp_path / "install.log").read_text(encoding = "utf-8")
    assert _received(tmp_path) == ["--non-interactive", "--skip-setup", "--skip-browser"]
    assert "no longer takes --no-skills" in (tmp_path / "install.log").read_text(encoding = "utf-8")


def test_an_optional_flag_the_installer_still_parses_is_passed(tmp_path: Path) -> None:
    installer = _installer(tmp_path, ["--non-interactive", "--no-skills"])
    result = _curl_bash(tmp_path, installer, ["--non-interactive", "?--no-skills"])
    assert result.returncode == 0
    assert _received(tmp_path) == ["--non-interactive", "--no-skills"]


def test_a_flag_is_not_kept_for_a_longer_one_that_starts_with_it(tmp_path: Path) -> None:
    installer = _installer(tmp_path, ["--non-interactive", "--skip-setup"])
    result = _curl_bash(tmp_path, installer, ["--non-interactive", "?--skip"])
    assert result.returncode == 0
    assert _received(tmp_path) == ["--non-interactive"]


def test_a_required_flag_the_installer_dropped_still_fails(tmp_path: Path) -> None:
    """Only the ?-marked niceties are forgiven: losing one we depend on is real drift."""
    installer = _installer(tmp_path, ["--skip-setup"])
    result = _curl_bash(tmp_path, installer, ["--non-interactive", "?--skip-setup"])
    assert result.returncode != 0
    assert _received(tmp_path) == ["--non-interactive", "--skip-setup"]


def test_hermes_still_passes_non_interactive_as_required() -> None:
    assert "--non-interactive" in _hermes_args()
