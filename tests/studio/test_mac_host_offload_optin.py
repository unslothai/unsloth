# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Only the macOS GGUF job opts out of the host-offload guard, and only deliberately.

Mac Unsloth GGUF CI went red on every main run from ee68d9e2a onwards, the merge of #8883
("refuse a gguf that cannot fit in free vram plus available ram"). The chain that makes macOS
special, from the failing run's own server log:

  1. GitHub's macOS runners expose a PARAVIRTUAL Metal device.
  2. Unsloth refuses to offload to one, because paravirtual Apple GPUs return corrupt output:
     "Forcing gpu_layers=0 ... this Mac's Metal device is virtualised".
  3. So the launch is `--gpu-layers 0 --device none` and the WHOLE model is a host mapping,
     not the partial spill the new guard was written to price.
  4. The guard then measures honestly and declines: about 3 GB wanted, about 2 GB usable.

The guard is correct -- the runner really cannot hold gemma-4-E2B UD-Q4_K_XL plus mmproj-F16
in 2 GB, and it had only been getting away with it because the prompts are tiny and the
mapping is paged. That is precisely the gamble the guard exists to stop taking on a user's
machine. CI takes it knowingly via UNSLOTH_ALLOW_HOST_OFFLOAD.

What this file protects is the blast radius of that decision. The escape hatch disables a real
safety net, so it belongs on the one platform whose GPU is fake and nowhere else: if it spread
to the Linux or Windows GGUF jobs, a genuine regression that made Unsloth try to host-offload
a model it should have declined would sail through CI green.
"""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
ENV_VAR = "UNSLOTH_ALLOW_HOST_OFFLOAD"

# The mac GGUF phases were absorbed into the Mac UI job, which is why this is not the file the docstring above was
# written against. The opt-out moved with them, to that job's env, and is still the only place in the repo it appears.
MAC_GGUF = "studio-mac-ui-smoke.yml"
OTHER_GGUF = ["studio-inference-smoke.yml", "studio-windows-inference-smoke.yml"]

TRUTHY = ("1", 1, "true", "True", "yes")


def _doc(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding = "utf-8"))


def test_the_mac_gguf_job_opts_out_at_job_level():
    """Job level, so a phase added later inherits it rather than failing mysteriously.

    A phase that misses it does not fail loudly: the load returns HTTP 400 and the test
    reports an unexpected status, several layers away from the actual cause.
    """
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
    # Workflow level first. GitHub propagates a top-level `env:` to every job, so it is the BROADEST way to set this
    # and was the one place a job/step scan could not see: one line at the top of the Linux or Windows workflow would
    # disable the guard across every job in it while this test, whose whole subject is blast radius, stayed green.
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
