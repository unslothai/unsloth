# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two Windows UI lanes must not share the state that used to serialise them.

Five Playwright suites ran end to end for ~788s of a 20.6 minute job. They are
disjoint and their ports were already distinct; what forced the sequence was shared
auth state, because boot-studio-api-only.sh hardcoded ~/.unsloth/studio/auth and every
"pass the bootstrap password" step read that same file back.

Every failure this file guards against LOOKS LIKE SUCCESS, which is why they are worth
pinning:

  * A lane losing its own UNSLOTH_STUDIO_HOME. Both lanes then wipe and re-seed one
    auth directory while the other is logging in with the password it just read. That
    is a race, so it fails intermittently and on someone else's PR.
  * Backgrounding without wait-and-collect. `&` defeats `set -e`; the step exits 0 and
    the job goes green having run neither suite to completion.
  * Waiting on only the first lane. The second lane's failure is then invisible, which
    is the specific regression #9158 called out for the Linux indicator engines.
  * A lane home with no venv link. UNSLOTH_STUDIO_HOME is the CLI's INSTALL root, so a
    bare directory makes `unsloth studio` exit "Unsloth Studio not set up" before it
    binds a port.
  * A lane home with no llama.cpp path. Setting the variable makes the root custom, and
    unsloth_cli/commands/studio.py then resolves UNSLOTH_LLAMA_CPP_PATH under it rather
    than the legacy ~/.unsloth/llama.cpp. These lanes load a real GGUF, so the model
    load fails rather than falling back.

The assertions read the script and the workflow rather than a list written here, so a
list cannot agree with itself while the scripts move.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
LANE = REPO / ".github" / "scripts" / "run-studio-ui-lane.sh"
BOOT = REPO / ".github" / "scripts" / "boot-studio-api-only.sh"
WORKFLOW = REPO / ".github" / "workflows" / "studio-windows-ui-smoke.yml"

LANES = ("chat", "extra")


def _strip_comments(text: str) -> str:
    """Assertions must not be satisfied by the prose that explains them.

    Every one of these scripts documents the thing being asserted in a comment
    directly above it, so a substring check against the raw file passes even after the
    code it describes is deleted. This has already bitten this repo once.
    """
    out = []
    for line in text.split("\n"):
        stripped = re.sub(r"(^|\s)#.*$", "", line)
        out.append(stripped)
    return "\n".join(out)


def _lane_body() -> str:
    return _strip_comments(LANE.read_text(encoding = "utf-8"))


def _step() -> dict:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8")) or {}
    for step in doc["jobs"]["ui-smoke"]["steps"]:
        if "lane" in str(step.get("name", "")).lower():
            return step
    raise AssertionError(
        "no step in ui-smoke runs the UI lanes. If they were deliberately put back in "
        "sequence, delete this file; if the step was renamed, retarget it."
    )


def _step_body() -> str:
    return _strip_comments(str(_step().get("run", "")))


def test_the_boot_script_honours_a_per_lane_studio_home() -> None:
    """The enabling change. Without it the lanes share one auth directory."""
    body = _strip_comments(BOOT.read_text(encoding = "utf-8"))
    assert "UNSLOTH_STUDIO_HOME" in body, (
        "boot-studio-api-only.sh no longer reads UNSLOTH_STUDIO_HOME, so it is back to "
        "wiping the one legacy auth directory. Two concurrent lanes then race: one wipes "
        "the .bootstrap_password the other just minted and is about to log in with, and "
        "it fails intermittently rather than every time."
    )
    assert not re.search(r"rm -rf\s+~?/?\.unsloth/studio/auth", body), (
        "boot-studio-api-only.sh wipes a hardcoded auth path again; it must go through "
        "the resolved per-lane home"
    )


def test_each_lane_gets_its_own_port_and_studio_home() -> None:
    body = _lane_body()
    ports = set(re.findall(r"PORT=(\d{4,5})", body))
    assert len(ports) >= len(LANES), (
        f"the lanes do not have distinct boot ports: {sorted(ports)}. Two servers on one "
        f"port means the second never binds."
    )
    assert re.search(r"home=.*\$\{?LANE", body) or re.search(r"\.studio-lane-\$LANE", body), (
        "the lane home does not vary by lane, so both lanes share one UNSLOTH_STUDIO_HOME "
        "and the auth wipe races"
    )
    assert re.search(r"export\s+UNSLOTH_STUDIO_HOME=", body), (
        "the lane never exports UNSLOTH_STUDIO_HOME, so the boot script and the two "
        "browser scripts all fall back to the shared legacy home"
    )


def test_each_lane_links_the_installed_venv_and_pins_llama_cpp() -> None:
    """A bare per-lane home is not a usable Unsloth root; see the module docstring."""
    body = _lane_body()
    assert "unsloth_studio" in body and re.search(r"mklink|ln -sfn", body), (
        "the lane home does not link the installed venv. UNSLOTH_STUDIO_HOME is the "
        "CLI's install root, so `unsloth studio` exits 'Unsloth Studio not set up' "
        "before binding a port."
    )
    assert re.search(r"export\s+UNSLOTH_LLAMA_CPP_PATH=", body), (
        "the lane does not pin UNSLOTH_LLAMA_CPP_PATH. A custom UNSLOTH_STUDIO_HOME "
        "makes the CLI resolve llama.cpp UNDER that home instead of ~/.unsloth/llama.cpp "
        "(unsloth_cli/commands/studio.py), and these lanes load a real GGUF."
    )


def test_the_lane_boot_does_not_write_the_shared_github_env() -> None:
    """Concurrent lanes appending one file is the shared state this change removes."""
    body = _lane_body()
    assert "env -u GITHUB_ENV" in body, (
        "the lane boots without unsetting GITHUB_ENV. boot-studio-api-only.sh appends "
        "the pid there whenever it is set, and inside a step it always is, so both lanes "
        "would append to the one file the runner reads back -- a lost pid and exactly the "
        "kind of shared mutable state the lanes exist to avoid."
    )


def test_the_step_runs_the_lanes_concurrently() -> None:
    body = _step_body()
    assert re.search(r"run-studio-ui-lane\.sh.*&\s*$", body, re.M) or re.search(
        r"run-studio-ui-lane\.sh[^\n]*\n[^\n]*&\s*$", body, re.M
    ), (
        "the lanes are not backgrounded, so they run one after another and the change "
        "buys nothing"
    )


def test_the_step_waits_on_every_lane_and_propagates_failure() -> None:
    """`&` defeats set -e: without this the step exits 0 having run nothing."""
    body = _step_body()
    assert "wait " in body, "the step never waits on the lanes, so it cannot see them fail"
    assert re.search(r"rc=1", body) and re.search(r'exit\s+"?\$\{?rc', body), (
        "the step does not collect a failing lane into its own exit status. Backgrounded "
        "work does not trip `set -e`, so the job would go green with a failed lane."
    )
    waits = len(re.findall(r"\bwait\b", body))
    loops = len(re.findall(r"\bfor\s+\w+\s+in\s+\$pids", body))
    assert loops >= 1 or waits >= len(LANES), (
        "the step waits on fewer lanes than it starts, so one lane's breakage hides "
        "another's -- the regression #9158 called out for the Linux indicator engines"
    )


def test_every_suite_that_used_to_be_a_step_still_runs() -> None:
    """The silent failure: a lane that quietly drops a suite still goes green."""
    body = _lane_body()
    for suite in (
        "tests/studio/playwright_chat_ui.py",
        "tests/studio/playwright_extra_ui.py",
        "tests/studio/playwright_update_banner_layout.py",
        "run-studio-indicator-browser.sh",
        "run-studio-permission-browser.sh",
    ):
        assert suite in body, (
            f"{suite} ran as a step before the lanes and is not run by any lane now. "
            f"Nothing else in CI covers it, so dropping it turns nothing red."
        )


def test_the_guard_is_reading_real_files() -> None:
    """Every assertion above passes vacuously if these stop being found."""
    for path in (LANE, BOOT, WORKFLOW):
        assert path.is_file(), path
    assert len(_lane_body()) > 500, "lane script body looks empty after comment stripping"
    assert len(_step_body()) > 100, "workflow step body looks empty after comment stripping"
