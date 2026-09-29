# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Guard the parallel zoo run, the serial reruns for tests that cannot share workers,
and the split that moved all of them into their own job.

The zoo suite used to run inside the `consolidated` cell, serially, for 418s of that
cell's 17.2 minutes. It now runs in `consolidated-zoo`, a second matrix job over the
same three (transformers, TRL) combos. Two things that were previously true by
construction have to be asserted now that there are two jobs:

  - the suite still runs under all three pins, and still runs at all. A matrix that
    loses a combo, or a job whose steps drift away from the zoo ones, reduces coverage
    without failing anything.
  - the two jobs still install the same environment. The install lives in
    .github/actions/core-cpu-setup so there is one copy of it, but the four steps above
    that action (checkout, setup-python, the pip cache restore) and the job-level `env`
    and `runs-on` are per-job and can drift silently. A zoo job on a different
    transformers than the cell it was split out of would still be green, and would be
    testing something nobody asked for.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "consolidated-tests-ci.yml"
)
ACTION = (
    Path(__file__).resolve().parents[2] / ".github" / "actions" / "core-cpu-setup" / "action.yml"
)

# The two halves of Core. Both run the same matrix; only their test steps differ.
CORE_JOBS = ("consolidated", "consolidated-zoo")
SETUP_ACTION = "./.github/actions/core-cpu-setup"

# (ignored path, why it cannot share a worker with the rest of the suite)
ISOLATED = [
    ("tests/test_mlx_generate.py", "8 failures under xdist that serial does not produce"),
    (
        "tests/test_moe_bnb4bit_per_expert_conversions.py",
        "6 failures under xdist that serial does not produce",
    ),
    ("tests/test_hf_xet_fallback.py", "sub-second wall-clock margins under CPU contention"),
    # MLX shims alter sys.modules, so these files each need a fresh process.
    ("tests/test_mlx_neftune_quant_map.py", "the mlx shim un-skips it and the stub then raises"),
    ("tests/test_mlx_gated_delta_vjp.py", "the mlx shim changes which backend it exercises"),
    (
        "tests/test_gemma3_forced_float32_boundary_dtype.py",
        "the mlx shim leaves patch_Gemma3MLP unable to install its forward",
    ),
]

ZOO_MARKER = "--dist loadfile tests/"

# Deselected because it needs a GPU.
MLX_DESELECT = (
    "tests/test_mlx_finetune_last_n_layers.py::"
    "test_get_peft_model_passes_finetune_last_n_layers_through"
)


def _commands() -> list[str]:
    """Every `python -m pytest ...` invocation, line continuations resolved."""
    text = WORKFLOW.read_text(encoding = "utf-8")
    joined = re.sub(r"\\\s*\n\s*", " ", text)
    return [
        line.strip()
        for line in joined.splitlines()
        if "python -m pytest" in line and not line.lstrip().startswith("#")
    ]


def _zoo_parallel() -> str:
    hits = [c for c in _commands() if ZOO_MARKER in c and "-n 4" in c]
    assert len(hits) == 1, (
        f"expected exactly one parallel zoo pytest run, found {len(hits)}. "
        f"This guard cannot check a command it cannot identify."
    )
    return hits[0]


def _zoo_mlx_group() -> str:
    """Return the serial MLX group command."""
    hits = [c for c in _commands() if "$mlx_group" in c]
    assert len(hits) == 1, f"expected exactly one serial mlx group run, found {len(hits)}"
    return hits[0]


def _zoo_serial(path: str) -> str:
    """Return the dedicated rerun command for one isolated file."""
    hits = [c for c in _commands() if "-n 4" not in c and path in c]
    assert len(hits) == 1, f"expected exactly one serial rerun naming {path}, found {len(hits)}"
    return hits[0]


def test_the_zoo_suite_actually_runs_in_parallel() -> None:
    """If the -n is dropped the ignores below become pure test deletion."""
    cmd = _zoo_parallel()
    assert "--dist loadfile" in cmd, (
        "the parallel zoo run does not use --dist loadfile. 34 of the 236 zoo test files "
        "touch sys.modules or importlib.reload, so tests within a file have to stay on "
        "one worker and in order; the default `load` splits them per test."
    )


@pytest.mark.parametrize("path,reason", ISOLATED, ids = lambda v: v.split("/")[-1])
def test_an_isolated_file_is_ignored_by_the_parallel_run(path: str, reason: str) -> None:
    cmd = _zoo_parallel()
    covered = f"--ignore={path}" in cmd or (
        path.rsplit("/", 1)[-1].startswith("test_mlx_")
        and "--ignore-glob='tests/test_mlx_*.py'" in cmd
    )
    assert covered, (
        f"{path} ({reason}) is not ignored by the parallel zoo run, so it goes back to "
        f"failing intermittently depending on which worker picks it up"
    )


@pytest.mark.parametrize("path,reason", ISOLATED, ids = lambda v: v.split("/")[-1])
def test_an_isolated_file_still_runs_serially(path: str, reason: str) -> None:
    """The silent half. An ignore with no rerun deletes the tests and stays green."""
    assert path in _zoo_serial(path), (
        f"{path} is ignored from the parallel run but never run again. Its tests are "
        f"simply not executed, and nothing else in CI would say so."
    )


@pytest.mark.parametrize("path,reason", ISOLATED, ids = lambda v: v.split("/")[-1])
def test_the_serial_rerun_is_not_itself_parallel(path: str, reason: str) -> None:
    """Rerunning these under xdist would reproduce exactly what it exists to avoid."""
    assert "-n " not in _zoo_serial(path), (
        f"the serial rerun of {path} passes -n, which puts it back in the parallel "
        f"session whose ordering is what breaks it"
    )


def test_the_serial_reruns_tolerate_an_empty_collection() -> None:
    """Treat pytest exit 5 from a module-level skip as non-fatal."""
    for path, _ in ISOLATED:
        assert "_keep" in _zoo_serial(path), (
            f"the rerun of {path} feeds its status straight into rc, so a module-level "
            f"skip (pytest exit 5, nothing collected) fails the whole job"
        )
    assert 'if [ "$1" = 5 ]' in WORKFLOW.read_text(
        encoding = "utf-8"
    ), "the isolated rerun no longer tolerates pytest's no-tests-collected exit"


@pytest.mark.parametrize("path,reason", ISOLATED, ids = lambda v: v.split("/")[-1])
def test_an_isolated_file_does_not_share_its_rerun(path: str, reason: str) -> None:
    """Give every contaminating file its own process."""
    cmd = _zoo_serial(path)
    others = [other for other, _ in ISOLATED if other != path and other in cmd]
    assert not others, (
        f"{path} shares its rerun process with {others}. These files are isolated "
        f"because they cannot share a process, and that includes each other."
    )


def test_the_deselects_survive_on_the_parallel_run() -> None:
    """Keep each deselect with the command that owns its file."""
    cmd = _zoo_parallel()
    assert (
        cmd.count("--deselect") == 2
    ), f"the parallel zoo run carries {cmd.count('--deselect')} deselects, expected 2"
    group = _zoo_mlx_group()
    assert MLX_DESELECT in group, (
        f"{MLX_DESELECT} is deselected nowhere now that test_mlx_finetune_last_n_layers.py "
        f"runs in the serial mlx group, so it fails on a GPU-less runner instead"
    )


def test_the_mlx_family_leaves_the_parallel_run_as_a_glob() -> None:
    """Exclude all MLX tests because their partial shims contaminate workers."""
    assert "--ignore-glob='tests/test_mlx_*.py'" in _zoo_parallel(), (
        "the parallel zoo run no longer excludes the mlx family as a glob, so the next "
        "test_mlx_*.py added upstream goes back to poisoning whichever file follows it"
    )


def test_the_mlx_group_runs_serially_and_skips_the_per_file_three() -> None:
    """The group is serial on purpose, and must not double-run the per-file isolated mlx."""
    assert (
        "-n " not in _zoo_mlx_group()
    ), "the mlx group runs under xdist, which is the arrangement it exists to avoid"
    # Per-file exclusions are applied while building the group, not on pytest itself.
    text = WORKFLOW.read_text(encoding = "utf-8")
    for path, _ in ISOLATED:
        name = path.rsplit("/", 1)[-1]
        if not name.startswith("test_mlx_"):
            continue
        stem = name[len("test_mlx_") : -len(".py")]
        assert f"{stem}|" in text or f"{stem})" in text, (
            f"{path} has its own process but is not excluded from the mlx group's file "
            f"list, so it runs twice and brings its shim back into that session"
        )


def test_an_empty_mlx_group_stops_the_step_instead_of_collecting_everything() -> None:
    """The group is passed unquoted, so an empty list is not an empty run: with nothing
    in ``mlx_group`` the command collects the whole rootdir instead, green and far
    slower. The glob only has to stop matching once, upstream renaming the family say."""
    text = WORKFLOW.read_text(encoding = "utf-8")
    assert 'if [ -z "$mlx_group" ]' in text, (
        "nothing checks that the mlx group glob matched anything, so an empty glob "
        "silently turns this step into a serial run of the entire suite"
    )


def _doc() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def _job(jid: str) -> dict:
    jobs = _doc()["jobs"]
    assert jid in jobs, f"{jid} is gone from the workflow; the split it guards is undone"
    return jobs[jid]


def _job_commands(jid: str) -> list[str]:
    """Every `python -m pytest ...` invocation inside one job, continuations resolved."""
    found = []
    for step in _job(jid)["steps"]:
        joined = re.sub(r"\\\s*\n\s*", " ", str(step.get("run", "")))
        found += [
            " ".join(line.split())
            for line in joined.splitlines()
            if "python -m pytest" in line and not line.lstrip().startswith("#")
        ]
    return found


def _preamble(jid: str) -> list[dict]:
    """The steps up to and including the shared setup action, which every job repeats."""
    out: list[dict] = []
    for step in _job(jid)["steps"]:
        out.append(step)
        if str(step.get("uses", "")) == SETUP_ACTION:
            return out
    raise AssertionError(
        f"{jid} never calls {SETUP_ACTION}, so it no longer installs the environment "
        f"the other half of Core installs"
    )


def test_the_zoo_suite_runs_beside_core_and_not_inside_it() -> None:
    """The point of the split. Back inside `consolidated` it is 7.6 min of serial wait."""
    inside = [c for c in _job_commands("consolidated") if ZOO_MARKER in c]
    assert not inside, (
        f"the zoo suite is running inside the `consolidated` cell again ({inside}), so "
        f"every pull request waits through it before the rest of Core can finish"
    )
    beside = [c for c in _job_commands("consolidated-zoo") if ZOO_MARKER in c]
    assert (
        len(beside) == 1
    ), f"expected the parallel zoo run in the `consolidated-zoo` job, found {len(beside)}"


def test_the_zoo_job_runs_every_zoo_step_that_left_the_cell() -> None:
    """Two steps moved. A move that drops one is a silent deletion of its tests."""
    zoo = " \n".join(_job_commands("consolidated-zoo"))
    assert "_zoo_apply_fused_lm_head_shim.py" in zoo, (
        "unsloth_zoo.compiler.test_apply_fused_lm_head moved out of the consolidated "
        "cell but is not run by the zoo job either, so it runs nowhere"
    )
    for path, _ in ISOLATED:
        assert path in zoo, (
            f"{path}'s serial rerun is not in the zoo job. It is ignored by the parallel "
            f"run, so wherever its rerun went, it has to have gone with it"
        )


@pytest.mark.parametrize("jid", CORE_JOBS)
def test_both_halves_of_core_run_the_same_three_combos(jid: str) -> None:
    """Coverage is '3 pins x the same suite'. A matrix that drifts quietly ends that."""
    expected = _doc()["jobs"]["consolidated"]["strategy"]["matrix"]["combo"]
    assert [c["id"] for c in expected] == [
        "t4576-trl0latest",
        "tlatest5-trl1latest",
        "pyproject",
    ], "the Core combo ids changed; update this guard deliberately, not by accident"
    assert _job(jid)["strategy"]["matrix"]["combo"] == expected, (
        f"{jid}'s matrix no longer matches `consolidated`'s. GitHub Actions has no way to "
        f"share a matrix between jobs, so these are two copies, and a copy that drifts "
        f"means the two halves of Core are testing different (transformers, TRL) pins "
        f"while still reporting as one gate"
    )


def test_both_halves_of_core_share_one_install_preamble() -> None:
    """The install has one definition; the four steps around it are still per-job.

    Checkout, setup-python and the pip cache restore are duplicated by necessity, and a
    difference in any of them (a different interpreter, a cache scoped to other files)
    makes the zoo job test a stack the cell it was split from never runs.
    """
    a, b = (_preamble(jid) for jid in CORE_JOBS)
    assert len(a) == len(b), (
        f"the two Core jobs run {len(a)} and {len(b)} preamble steps. They install the "
        f"same environment, so their preambles have to be the same steps in the same order"
    )
    for left, right in zip(a, b):
        # The pip cache `name` is the one field that MUST differ: a shared name is a
        # shared key, and only the first job to finish on main would ever save.
        # tests/studio/test_pip_cache_naming.py owns that rule.
        left, right = dict(left), dict(right)
        if "pip-cache-restore" in str(left.get("uses", "")):
            left["with"] = {k: v for k, v in left["with"].items() if k != "name"}
            right["with"] = {k: v for k, v in right["with"].items() if k != "name"}
        assert left == right, (
            f"the Core preambles have drifted at step "
            f"{left.get('name') or left.get('uses')!r}:\n  consolidated:      {left}\n"
            f"  consolidated-zoo:  {right}"
        )


def test_both_halves_of_core_share_one_environment_and_one_runner() -> None:
    """`env` and `runs-on` are job-level and cannot be factored into the action.

    `runs-on` is included on purpose. The label is not cosmetic here: measured on this
    repo, `ubuntu-latest` queues behind the org's backlog for a median 44.9 min while any
    other Ubuntu label walks past in minutes, so two halves of one gate on two different
    labels would make the split buy nothing.
    """
    a, b = (_job(jid) for jid in CORE_JOBS)
    assert a["env"] == b["env"], (
        f"the two Core jobs no longer share a job-level env:\n  only in consolidated: "
        f"{ {k: v for k, v in a['env'].items() if b['env'].get(k) != v} }\n"
        f"  only in consolidated-zoo: "
        f"{ {k: v for k, v in b['env'].items() if a['env'].get(k) != v} }"
    )
    assert a["runs-on"] == b["runs-on"], (
        f"the two halves of Core run on different labels ({a['runs-on']} vs "
        f"{b['runs-on']}), so one of them queues behind a backlog the other skips"
    )


def test_the_shared_preamble_is_not_also_inlined() -> None:
    """A caller that re-adds an install step is how one definition becomes two."""
    body = "\n".join(str(step.get("run", "")) for jid in CORE_JOBS for step in _job(jid)["steps"])
    for marker in ("pip install -e .", "download.pytorch.org/whl/cpu", "git clone"):
        assert marker not in body, (
            f"{marker!r} is inlined in a Core job again. The install lives in "
            f"{SETUP_ACTION} so both halves cannot drift apart; a second copy is the "
            f"drift, and it is silent until the two jobs disagree about a version"
        )
    action = yaml.safe_load(ACTION.read_text(encoding = "utf-8"))
    run = "\n".join(str(step.get("run", "")) for step in action["runs"]["steps"])
    for marker in ("pip install -e .", "download.pytorch.org/whl/cpu", "git clone"):
        assert marker in run, f"{SETUP_ACTION} no longer does {marker!r}"


def test_a_skipped_isolated_file_is_named_in_the_log() -> None:
    """Exit 5 is tolerated, so the file that produced it has to be identifiable: an
    expected module-level skip and a file that stopped collecting for a new reason both
    exit 5 and both stay green."""
    text = WORKFLOW.read_text(encoding = "utf-8")
    for path, _ in ISOLATED:
        assert f'_keep "$?" {path}' in text, (
            f"the rerun of {path} does not pass its own name to _keep, so a silent "
            f"empty collection is reported without saying which file it was"
        )
