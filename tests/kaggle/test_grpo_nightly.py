# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO runs nightly, and the reason is a measurement rather than a preference.

Over nine T4 sessions the leg hit an intermittent illegal memory access in
vLLM's standby sleep on Turing in FOUR of them. Everything else about it is
sound -- 0.95 utilisation confirmed on both Colab and Kaggle, sleep/wake
surviving three cycles, a non-zero ``reward_std`` on every step once the reward
function stopped saturating -- but a 44% red in front of every PR, for a race no
reader can act on, is exactly how a check gets switched off before the day it is
right. At nightly cadence a clean run still arrives most days and a red one
costs nobody a merge.

So the rules here are about the SHAPE that makes that possible:

* the schedule exists and fires the GRPO leg specifically;
* a leg list REPLACES ``--all-kernels`` rather than filtering after it, because
  the kernel plan and the expected payload count come out of the same call and
  a filter applied afterwards leaves the launcher waiting on payloads nobody
  built;
* the schedule bypasses the sampling gate. A nightly sampled at 15% is a
  weekly, and the difference is invisible until someone goes looking for a
  result that never existed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "kaggle-t4-notebook-ci.yml"
TEXT = WORKFLOW.read_text(encoding = "utf-8")
DOC = yaml.safe_load(TEXT)
TRIGGERS = DOC.get(True) or DOC.get("on")

sys.path.insert(0, str(ROOT / ".github" / "scripts" / "kaggle_t4_ci"))
import legs  # noqa: E402


def test_there_is_a_nightly_schedule():
    assert "schedule" in TRIGGERS, "nothing runs the legs that cannot be per-PR"
    crons = [entry["cron"] for entry in TRIGGERS["schedule"]]
    assert len(crons) == 1, crons
    minute, hour = crons[0].split()[:2]
    assert minute.isdigit() and hour.isdigit(), "the nightly must be a fixed time"


def test_the_nightly_does_not_pile_onto_the_hour_mark():
    """Every scheduled workflow on GitHub asks for :00, and this one has no
    reason to join that queue."""
    minute = TRIGGERS["schedule"][0]["cron"].split()[0]
    assert minute not in ("0", "00", "30"), f"minute {minute} is the mark everyone else picks"


def _nightly_legs():
    """The leg names the schedule actually selects, read off the workflow."""
    # Anchored on LEG_LIST, not on the schedule fallback in general. The gate
    # bypass a few lines above reads `github.event_name == 'schedule' && 'true'`
    # and a loose pattern picks THAT up: the first version of this helper
    # reported the nightly leg set as ["true"].
    match = re.search(r"LEG_LIST:.*github\.event_name == 'schedule' && '([a-z_,]+)'", TEXT)
    assert match, "the schedule selects no leg list at all"
    return [name for name in match.group(1).split(",") if name]


def test_the_schedule_runs_the_grpo_leg():
    assert "grpo" in _nightly_legs(), (
        "the schedule must select the leg; a nightly that runs the wired set "
        "is just another copy of the per-PR run"
    )


def test_the_schedule_also_runs_the_multi_gpu_leg():
    """multi_gpu is here for a different reason from grpo and the distinction
    is worth keeping: grpo is nightly because it CRASHES 44% of the time,
    multi_gpu because it costs makespan. It passes on hardware -- the ab3 A/B
    has it green in both arms -- and the brief it was built to was multi-GPU
    coverage at no wall-clock cost, which it measurably is not (+172.4s and
    +39.7s over two same-commit pairs, slower in both).

    Nightly is where that coverage is free. Without this the DEVICE_COUNT > 1
    bindings in unsloth's kernels go back to being tested nowhere at all, which
    is the state this leg was written to end."""
    assert "multi_gpu" in _nightly_legs(), (
        "the nightly no longer runs multi_gpu, so unsloth's DEVICE_COUNT > 1 "
        "code path is covered by nothing: every other leg is pinned to one card"
    )


def test_the_schedule_also_runs_the_latest_compile_leg():
    """Third reason again, and the distinction is the point: grpo is nightly
    because it crashes 44% of sessions, multi_gpu because it costs makespan at
    the margin, latest_compile because it does not FIT.

    Its DONE record is 1323.0s (unsloth-probe-lcleg-tmpdir-ac53ca) and at
    12.73GB peak it admits no co-tenant, so it wants a whole card for 22
    minutes. The per-PR kernel's only slack is gpu1's 776.3s idle block while
    Studio holds gpu0. 1323 does not go into 776.

    Without this the leg is wired nowhere and gemma-4-E2B-it on the newest
    transformers and trl -- the only thing that caught zoo #1103 -- is tested
    by nothing. That is the state it was built to end, and the reason the leg
    is nightly rather than deleted."""
    assert "latest_compile" in _nightly_legs(), (
        "the nightly no longer runs latest_compile, so nothing anywhere loads "
        "gemma-4-E2B-it on the newest transformers and trl, which is the "
        "pairing that found unsloth-zoo #1103"
    )


def test_every_leg_the_nightly_names_exists():
    """A typo here produces a build that selects nothing and a run that proves
    nothing, with no error anywhere."""
    named = _nightly_legs()
    assert named, "no scheduled leg name found at all"
    for name in named:
        assert name in legs.LEGS, f"the nightly names {name!r}, which is not a leg"


def test_the_nightly_set_fits_in_one_kernel():
    """`--legs` builds ONE kernel, and MAX_LEGS_PER_KERNEL is what the driver's
    scheduling was measured against. A list longer than that silently packs a
    kernel nobody has run."""
    assert len(_nightly_legs()) <= legs.MAX_LEGS_PER_KERNEL


def test_no_nightly_leg_is_ALSO_in_the_per_pr_set():
    """The whole point, and it applies to each of them. If grpo were wired into
    KERNELS the 44% crash rate would be back in front of every PR; if multi_gpu
    were, the makespan it was moved here to avoid would be back too. Either way
    the nightly becomes a second copy of the per-PR run."""
    wired = {name for kernel in legs.KERNELS for name in kernel}
    both = sorted(set(_nightly_legs()) & wired)
    assert not both, (
        f"{both} run nightly AND per-PR, so the nightly is pointless and every "
        f"PR carries whatever these were moved off the critical path to avoid"
    )


def test_a_leg_list_replaces_all_kernels_rather_than_filtering_after_it():
    """``--all-kernels`` derives BOTH the kernel plan and the payload count the
    launcher waits on. A filter applied afterwards leaves it expecting payloads
    that were never built, which times out rather than failing."""
    assert 'KERNEL_SELECT="--legs $LEG_LIST"' in TEXT
    assert 'KERNEL_SELECT="--all-kernels"' in TEXT
    assert "$KERNEL_SELECT \\" in TEXT
    assert (
        "--all-kernels \\" not in TEXT
    ), "--all-kernels is still hardcoded, so the override cannot take effect"


def test_the_schedule_bypasses_the_sampling_gate():
    """A nightly sampled at 15% is a weekly, and the difference is invisible
    until someone goes looking for a result that never existed."""
    assert "github.event_name == 'schedule' && 'true'" in TEXT, (
        "the schedule does not force the gate, so most nights it will draw a "
        "stand-down and report nothing"
    )


def test_the_leg_list_default_survives_a_schedule_event():
    """``inputs`` is null on a schedule, so an input default cannot supply the
    value; it has to come from the fallback."""
    assert "inputs.legs || (github.event_name == 'schedule'" in TEXT


# ------------------------------------------------- the command line it composes

# Every rule above reads the workflow as TEXT, each was true, and the nightly
# still never ran: the build step also emitted --with-studio unconditionally,
# which build_kernel.py refuses next to --legs, so runs 33587255856 and
# 33716011285 (every scheduled run there has ever been) died on
# `--with-studio requires --all-kernels`. Two guards, neither able to see the
# other. So the rules below run the step's own shell body and hand the argv it
# composes to the real parser.


def _build_step():
    """The `Build the kernel notebooks` step, off the parsed YAML."""
    for job in DOC["jobs"].values():
        for step in job.get("steps") or []:
            if step.get("name") == "Build the kernel notebooks":
                return step
    raise AssertionError("the build step is gone, so nothing here can be true")


def _compose_argv(
    event,
    tmp_path,
    legs_input = "",
    studio_concurrent = "",
    github_output = "",
):
    """Run the build step's shell body and return the argv it would invoke.

    Executed by bash, not pattern-matched, so the branches decide what is
    emitted exactly as on a runner. The only substitution is `python`, a stub
    on PATH that records its arguments.
    """
    import json
    import os
    import shlex
    import subprocess

    step = _build_step()
    # `inputs` is null on push and on a schedule, so every input expression is
    # empty on both triggers modelled here. Only LEG_LIST differs, and it comes
    # from the workflow's own fallback rather than a copy kept here.
    nightly = ",".join(_nightly_legs())
    env_values = {
        "LEG_LIST": legs_input or (nightly if event == "schedule" else ""),
        "SKIP_BAND": "",
        "MAX_STEPS": "10",
        "REF_STEPS": "10",
    }
    for key in step.get("env") or {}:
        assert key in env_values, f"the build step gained {key}, which this rule does not model"

    body = step["run"]
    # `steps.*.outputs.*` are refs the builder only echoes, so any hex will do.
    body = body.replace("${{ steps.ref.outputs.ref }}", "0" * 40)
    body = body.replace("${{ steps.pins.outputs.zoo_ref }}", "1" * 40)
    body = body.replace("${{ inputs.shared_wheels }}", "")
    body = body.replace("${{ inputs.studio_concurrent }}", studio_concurrent)
    assert "${{" not in body, f"an unmodelled expression survives: {body}"

    bindir = tmp_path / "bin"
    bindir.mkdir(parents = True, exist_ok = True)
    argvfile = tmp_path / "argv.json"
    (bindir / "python").write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"open({str(argvfile)!r}, 'w').write(json.dumps(sys.argv[1:]))\n",
        encoding = "utf-8",
    )
    (bindir / "python").chmod(0o755)

    env = dict(os.environ, PATH = f"{bindir}{os.pathsep}{os.environ['PATH']}", **env_values)
    # A real file, so the rules read what the step published, not what it printed.
    env["GITHUB_OUTPUT"] = github_output or str(tmp_path / "github_output")
    # `bash -e`, which is what GitHub runs a `run:` block under on Linux.
    proc = subprocess.run(
        ["bash", "-e", "-c", body],
        cwd = ROOT,
        env = env,
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 0, f"the step body itself failed:\n{proc.stderr}"
    assert argvfile.exists(), f"the body never invoked python:\n{proc.stdout}\n{proc.stderr}"
    argv = json.loads(argvfile.read_text(encoding = "utf-8"))
    assert argv and argv[0].endswith("build_kernel.py"), argv
    return argv, proc.stdout + proc.stderr, shlex.join(argv)


def _run_builder(argv, tmp_path):
    import subprocess
    import sys

    out = [str(tmp_path / "kernel") if a == "kernel" else a for a in argv]
    assert out != argv, "the step no longer writes to `kernel`, so this rule writes into the repo"
    return subprocess.run(
        [sys.executable, *out],
        cwd = ROOT,
        capture_output = True,
        text = True,
    )


def test_the_nightly_command_line_is_one_the_builder_accepts(tmp_path):
    """THE RULE THAT WOULD HAVE CAUGHT IT.

    Not "the flags look right": the step's own shell composes the argv and the
    real builder is handed it.
    """
    argv, log, printed = _compose_argv("schedule", tmp_path)
    proc = _run_builder(argv, tmp_path)
    assert proc.returncode == 0, (
        f"the nightly builds nothing:\n  {printed}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "--legs" in argv, argv
    assert "--all-kernels" not in argv, argv


def test_the_per_pr_command_line_is_one_the_builder_accepts(tmp_path):
    """The pair to the rule above: narrowing one branch until it parses while
    breaking the other would satisfy that one on its own."""
    argv, log, printed = _compose_argv("push", tmp_path)
    proc = _run_builder(argv, tmp_path)
    assert proc.returncode == 0, (
        f"the per-PR run builds nothing:\n  {printed}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "--all-kernels" in argv, argv
    assert "--legs" not in argv, argv


def test_studio_rides_the_wired_set_and_only_the_wired_set(tmp_path):
    """The pair that was mutually exclusive, asserted in BOTH directions:
    dropping Studio from the nightly is only a fix if the per-PR kernel still
    carries it, else the payload is covered by nothing."""
    nightly, _, _ = _compose_argv("schedule", tmp_path)
    assert "--with-studio" not in nightly, (
        "the nightly still asks for Studio alongside a leg list, which "
        "build_kernel.py refuses outright"
    )
    per_pr, _, _ = _compose_argv("push", tmp_path)
    assert (
        "--with-studio" in per_pr
    ), "no trigger packs Studio in any more, so the whole Studio payload runs nowhere"
    assert "--studio-args" in per_pr, per_pr


def test_studio_concurrent_still_reaches_the_builder_on_the_per_pr_run(tmp_path):
    """The flag moved inside the Studio branch. One that stops being passed is
    the failure run 32674263571 shipped: the variant of an A/B ran the
    control's schedule and nothing was red."""
    argv, _, _ = _compose_argv("push", tmp_path)
    assert "--studio-concurrent" in argv, (
        "the default per-PR build no longer shares a card, so Studio waits for "
        "both to drain and the makespan claim in this workflow's header is gone"
    )


def test_a_leg_list_dispatch_is_told_studio_is_not_aboard(tmp_path):
    """A run that drops Studio and a run whose Studio silently stopped being
    packed look identical in the summary, so the first one says so."""
    _, log, _ = _compose_argv("schedule", tmp_path)
    assert "Studio is not in this kernel" in log, log


def test_an_explicit_leg_dispatch_takes_the_same_path_as_the_nightly(tmp_path):
    """`legs` is a dispatch input as well as a schedule fallback and reaches
    the same branch, so a hand dispatch hit the identical build failure."""
    argv, _, printed = _compose_argv("workflow_dispatch", tmp_path, legs_input = "grpo")
    assert "--with-studio" not in argv, printed
    proc = _run_builder(argv, tmp_path)
    assert proc.returncode == 0, f"{printed}\n{proc.stdout}\n{proc.stderr}"


def test_studio_concurrent_false_actually_removes_the_flag(tmp_path):
    """The OTHER half of the input, which the text rules cannot see.

    A branch that hardcoded the flag instead of reading the variable would
    still contain every string they look for. Only running it with the input
    `false` and finding the flag gone says the switch works, and that dispatch
    is the only way Studio's own two-card device selection is under test.
    """
    off, _, printed = _compose_argv("workflow_dispatch", tmp_path, studio_concurrent = "false")
    assert "--studio-concurrent" not in off, (
        f"studio_concurrent=false still shares a card, so the two-card Studio "
        f"dispatch is unreachable: {printed}"
    )
    assert "--with-studio" in off, "turning sharing off must not drop Studio itself"
    proc = _run_builder(off, tmp_path)
    assert proc.returncode == 0, f"{printed}\n{proc.stdout}\n{proc.stderr}"

    on, _, _ = _compose_argv("workflow_dispatch", tmp_path)
    assert (
        "--studio-concurrent" in on
    ), "the default stopped sharing, which is a makespan regression"


def test_the_studio_reporter_is_told_when_studio_is_not_aboard(tmp_path):
    """The build step publishes whether it packed Studio; the reporter gates
    on it.

    `own_verdict` answers an EMPTY `studio-gpu` report set with `partial`,
    carrying the notebook kernel's reason, so an ungated reporter renders
    "Unsloth GPU smoke: PARTIAL" on every leg-list run about a payload that was
    never aboard. Not red, which is worse: it reads like a result.
    """
    for event, expected in (("schedule", "false"), ("push", "true")):
        outfile = tmp_path / f"out_{event}"
        outfile.write_text("", encoding = "utf-8")
        argv, log, _ = _compose_argv(
            event,
            tmp_path / event,
            github_output = str(outfile),
        )
        written = dict(
            line.split("=", 1)
            for line in outfile.read_text(encoding = "utf-8").splitlines()
            if "=" in line
        )
        assert written.get("studio") == expected, (
            f"{event} publishes studio={written.get('studio')!r}, so the "
            f"reporter gate reads the wrong answer"
        )
        assert ("--with-studio" in argv) is (expected == "true"), argv


def test_the_studio_report_step_reads_that_output():
    """The output above is only worth publishing if something gates on it."""
    for job in DOC["jobs"].values():
        for step in job.get("steps") or []:
            if step.get("name") == "Report Studio":
                assert "steps.build.outputs.studio == 'true'" in step["if"], step["if"]
                return
    raise AssertionError("the Studio reporter step is gone")
