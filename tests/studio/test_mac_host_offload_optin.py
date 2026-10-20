# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Only macOS GGUF CI may set UNSLOTH_ALLOW_HOST_OFFLOAD: its runner's Metal GPU is paravirtual."""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
ENV_VAR = "UNSLOTH_ALLOW_HOST_OFFLOAD"

# The mac GGUF phases were absorbed into the Mac UI job; the opt-out moved to its env.
MAC_GGUF = "studio-mac-ui-smoke.yml"
OTHER_GGUF = ["studio-inference-smoke.yml", "studio-windows-inference-smoke.yml"]

TRUTHY = ("1", 1, "true", "True", "yes")


def _doc(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding = "utf-8"))


def test_the_mac_gguf_job_opts_out_at_job_level():
    """Set at job level so later phases inherit it; a phase without it gets an HTTP 400 from the load."""
    jobs = _doc(MAC_GGUF)["jobs"]
    assert len(jobs) == 1, f"expected one bundled job in {MAC_GGUF}, got {list(jobs)}"
    env = next(iter(jobs.values())).get("env") or {}
    assert env.get(ENV_VAR) in TRUTHY, (
        f"{MAC_GGUF} no longer sets {ENV_VAR} at job level. Every phase there runs CPU-only "
        f"because the runner's Metal device is paravirtual, so the whole model sits in host "
        f"RAM and the #8883 guard declines the load with HTTP 400."
    )


def test_the_opt_out_explains_itself_in_place():
    """A bare env var here reads like a workaround someone can tidy away."""
    src = (WORKFLOWS / MAC_GGUF).read_text(encoding = "utf-8")
    head = src[: src.index(ENV_VAR)]
    comment = head[head.rindex("\n      HF_HOME") :] if "\n      HF_HOME" in head else head
    for phrase in ("paravirtual", "8883"):
        assert phrase.lower() in comment.lower(), (
            f"the {ENV_VAR} block no longer explains {phrase!r}; without the reason the next "
            f"person removes it and Mac GGUF CI goes red again"
        )


@pytest.mark.parametrize("name", OTHER_GGUF)
def test_no_other_gguf_workflow_disables_the_guard(name):
    doc = _doc(name)
    offenders = []
    # A top-level `env:` reaches every job, so it is the broadest way to set this.
    if (doc.get("env") or {}).get(ENV_VAR) is not None:
        offenders.append("workflow env (applies to every job)")
    for jid, job in (doc.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        if (job.get("env") or {}).get(ENV_VAR) is not None:
            offenders.append(f"{jid} (job env)")
        for step in job.get("steps") or []:
            if (step.get("env") or {}).get(ENV_VAR) is not None:
                offenders.append(f"{jid}: {step.get('name')}")
    assert not offenders, (
        f"{name} disables the host-offload guard in {offenders}. Those runners have real "
        f"memory and a real device; silencing the guard there means a regression that "
        f"host-offloads a model it should decline would pass CI green."
    )


def test_the_guard_still_has_its_own_tests():
    """The mac opt-out must not be mistaken for the guard being untested."""
    owned = [
        REPO / "studio" / "backend" / "tests" / "test_host_offload_ram_guard.py",
        REPO / "studio" / "backend" / "tests" / "test_llama_cpp_placement.py",
    ]
    for path in owned:
        assert path.exists(), f"{path.name} is gone; the guard's coverage went with it"
        assert ENV_VAR in path.read_text(
            encoding = "utf-8"
        ), f"{path.name} no longer exercises {ENV_VAR}"
