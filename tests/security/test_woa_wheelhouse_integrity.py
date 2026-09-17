# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Every input to a wheel we Authenticode sign has to be immutable.

woa-wheelhouse.yml builds pyarrow and sqlite-vec for win_arm64, signs the native images
inside those wheels with the Azure Trusted Signing credentials, and `--clobber`s them onto
the rolling release that install.ps1 fetches by default on Windows ARM64. The build pulls
its ARM64 support from a third-party fork of Arrow and EXECUTES that fork's batch files, so
a branch ref there would mean whoever can push to it decides what we sign. The smoke tests
prove the wheel works and the signature check proves we signed it; neither says anything
about provenance. So the ref is pinned to a commit, and this is the gate that keeps it that
way.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "woa-wheelhouse.yml"

COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def _steps(workflow, job):
    return workflow["jobs"][job]["steps"]


def _dispatch_inputs(workflow):
    # YAML 1.1 reads a bare `on` as the boolean true, so that is the key PyYAML hands back.
    triggers = workflow["on"] if "on" in workflow else workflow[True]
    return triggers["workflow_dispatch"]["inputs"]


def test_arm64_overlay_is_pinned_to_a_commit():
    ref = _workflow()["env"]["ARROW_WOA_REF"]
    assert COMMIT_SHA.match(str(ref)), (
        f"ARROW_WOA_REF is {ref!r}. A branch or tag on {_workflow()['env']['ARROW_WOA_REPO']} "
        "is mutable, and its batch files are executed to build a wheel we sign and publish."
    )


def test_overlay_checkout_uses_that_pin():
    checkouts = [
        step
        for step in _steps(_workflow(), "build-pyarrow")
        if str(step.get("uses", "")).startswith("actions/checkout")
    ]
    overlay = [
        step
        for step in checkouts
        if step.get("with", {}).get("repository") == "${{ env.ARROW_WOA_REPO }}"
    ]
    assert len(overlay) == 1, f"expected one overlay checkout, got {len(overlay)}"
    assert (
        overlay[0]["with"]["ref"] == "${{ env.ARROW_WOA_REF }}"
    ), "the overlay checkout must take the pinned ref, not a literal branch"


def test_the_run_re_checks_the_pin_before_copying_the_batch_files():
    """A pin nobody verifies is a comment. The build asserts it at run time too."""
    steps = _steps(_workflow(), "build-pyarrow")
    names = [step.get("name") for step in steps]
    assert "Confirm the overlay is the pinned commit" in names, names
    guard = names.index("Confirm the overlay is the pinned commit")
    overlay = names.index("Overlay the ARM64 build support onto the release tag")
    assert guard < overlay, "the pin has to be confirmed before the fork's files are copied"
    body = steps[guard]["run"]
    assert "rev-parse HEAD" in body
    assert "[0-9a-f]{40}" in body


def test_upstream_sources_are_tags_not_branches():
    """apache/arrow and asg017/sqlite-vec come from the dispatch inputs, which default to
    release tags. A default that drifted to a branch would reopen the same hole."""
    inputs = _dispatch_inputs(_workflow())
    assert inputs["arrow_tag"]["default"].startswith("apache-arrow-")
    assert re.match(r"^v\d+\.\d+\.\d+$", inputs["sqlite_vec_tag"]["default"])


def test_publishing_is_off_by_default():
    """`publish: true` is what reaches the signing environment and the release."""
    publish = _dispatch_inputs(_workflow())["publish"]
    assert publish["type"] == "boolean"
    assert publish["default"] is False
