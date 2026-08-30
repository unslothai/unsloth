# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""
The unsloth_zoo suite runs in parallel, minus the files that must not share a worker.

Measured on a staging runner, whole suite, all three matrix cells:

    cell                      serial   -n 4 --dist loadfile
    HF=latest + TRL=latest      580s                  253s
    HF=4.57.6 + TRL<1           517s                  229s
    HF=default + TRL=default    568s                  246s

Across two runs of all three cells, four agreed exactly -- failure sets and skip
sets both -- and two produced 14 failures serial does not: 8 in
test_mlx_generate.py, 6 in test_moe_bnb4bit_per_expert_conversions.py.

The same 14 every time, and not the same cell: the second run moved them from
HF=default to HF=latest. That rules out a dependency combination and leaves worker
scheduling, with roughly one cell per run drawing the losing order. It also means a
single green run proves nothing here, which is why the pin stays even though four
of six observations were clean.

Both files pass alone, and pass under xdist alone, so they are self-contained; what
breaks them is another file running first in the same worker, an ordering serial
never produces because serial is alphabetical. Both causes are now known, and they
are different:

  test_moe_bnb4bit_per_expert_conversions.py -- test_vllm_to_hf_conversion.py put a
  bitsandbytes.functional carrying only dequantize_4bit into sys.modules and never
  removed it, so the later `from bitsandbytes.functional import QuantState` that
  temporary_patches/moe_utils_bnb4bit.py does at import time failed with "(unknown
  location)". m sorts before v, so serial never saw it. Fixed upstream in
  unsloth_zoo#1076, with a teardown hook that fails the next one.

  test_mlx_generate.py -- CAUSE NOT ESTABLISHED. It installs the MLX-on-torch shim
  at its own import time and the shim's docstring requires that to happen "BEFORE
  any unsloth_zoo MLX module is imported", which looked like the answer: the eight
  failures are all isinstance checks reporting

      TypeError: requests[0] must be GenerationRequest.

  which is what two copies of a class look like. But conftest imports unsloth_zoo
  and pulls unsloth_zoo.mlx in before any test module loads, so that precondition
  is violated on every run INCLUDING the ones that pass, and the file passes alone.
  So the ordering contract is not the trigger, or not the whole one. I checked this
  by asserting the precondition and watching it fail a green run.

  The pin stands on the observation rather than the explanation: eight tests here
  fail under xdist and pass serially, reproducibly, on the same commit. Whoever
  picks this up next should not start from the shim docstring -- I did, and it is a
  dead end.

So each is ignored from the parallel pass and run again in a process of its own --
its own, not one shared by all of them, since they contaminate each other too. That
pairing is the thing this file guards, because half of it is silent: drop the serial
pass and the ignores simply delete the tests from CI while the job stays green. That
is strictly worse than the failures, which at least announce themselves.

Same shape as test_backend_ci_parallel_isolation for studio-backend-ci, and for the
same reason: an ignore and its serial rerun are two edits held together by nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "consolidated-tests-ci.yml"
)

# (ignored path, why it cannot share a worker with the rest of the suite)
ISOLATED = [
    ("tests/test_mlx_generate.py", "8 failures under xdist that serial does not produce"),
    (
        "tests/test_moe_bnb4bit_per_expert_conversions.py",
        "6 failures under xdist that serial does not produce",
    ),
    # Not ordering: wall clock. 27 sub-second sleeps against a stall detector, one of
    # them commented "within the unmeasurable window". Under four workers it reported
    # "no progress for 0s" -- nothing stalled, the test was descheduled.
    ("tests/test_hf_xet_fallback.py", "sub-second wall-clock margins under CPU contention"),
    # The MLX trio. test_mlx_generate.py's shim is the shared cause: it lands in
    # sys.modules at import time, which un-skips the neftune file (45 skips become
    # NotImplementedError out of the stub) and breaks the transformers patching the
    # gemma3 file asserts on. Under xdist that shim reaches them through a worker; in a
    # shared serial rerun it reaches them directly, which is why each gets its own
    # process below rather than a seat in one.
    ("tests/test_mlx_neftune_quant_map.py", "the mlx shim un-skips it and the stub then raises"),
    ("tests/test_mlx_gated_delta_vjp.py", "the mlx shim changes which backend it exercises"),
    (
        "tests/test_gemma3_forced_float32_boundary_dtype.py",
        "the mlx shim leaves patch_Gemma3MLP unable to install its forward",
    ),
]

# The zoo suite is the only pytest run in this workflow driven out of the cloned zoo
# checkout, and it is told apart by the deselect it carries rather than by step order,
# so re-ordering or renaming steps does not quietly point this guard at another command.
ZOO_MARKER = "--dist loadfile tests/"

# Deselected because it needs a GPU. It rides whichever command owns its file.
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
    """The serial run of the mlx family, found by the glob it builds its file list from."""
    hits = [c for c in _commands() if "$mlx_group" in c]
    assert len(hits) == 1, f"expected exactly one serial mlx group run, found {len(hits)}"
    return hits[0]


def _zoo_serial(path: str) -> str:
    """The rerun command for one isolated file. One file per command, on purpose."""
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
    # The mlx three are covered by the family glob rather than by name, which is the
    # point of the glob: a new test_mlx_*.py is excluded without anyone listing it.
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
    """One process per file turns pytest's exit 5 into a job failure unless handled.

    test_moe_bnb4bit_per_expert_conversions.py skips itself at module level on
    transformers 4.57.6: pytest prints "1 skipped" and returns 5, because nothing was
    collected. The shared invocation this replaced never saw it -- one skip among the
    session's other tests, exit 0 -- so the split has to keep that verdict.
    """
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
    """One process each, because they contaminate each other as well as the suite.

    Sharing one rerun process cost 14 failures and 6 errors on the cloned zoo at
    52aff0d, all of them gone when each file got its own process.
    """
    cmd = _zoo_serial(path)
    others = [other for other, _ in ISOLATED if other != path and other in cmd]
    assert not others, (
        f"{path} shares its rerun process with {others}. These files are isolated "
        f"because they cannot share a process, and that includes each other."
    )


def test_the_deselects_survive_on_the_parallel_run() -> None:
    """
    The deselects are the CUDA-only and known-broken cases. They lived on the single
    command that this change split; a split that dropped one would turn a deliberate
    'deselected' into a runtime failure on a GPU-less runner. Two ride the parallel run,
    and the mlx one moved with its file when the family left for the serial group.
    """
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
    """Naming the victims is whack-a-mole: they move every run, so exclude the source.

    Every tests/mlx_simulation stub installs itself into sys.modules, each importing file
    installs only the subset it needs, and a partial mlx stops a later file in the same
    worker from skipping. A glob keeps a newly added test_mlx_*.py out of the parallel
    pass without anyone remembering to list it.
    """
    assert "--ignore-glob='tests/test_mlx_*.py'" in _zoo_parallel(), (
        "the parallel zoo run no longer excludes the mlx family as a glob, so the next "
        "test_mlx_*.py added upstream goes back to poisoning whichever file follows it"
    )


def test_the_mlx_group_runs_serially_and_skips_the_per_file_three() -> None:
    """The group is serial on purpose, and must not double-run the per-file isolated mlx."""
    assert (
        "-n " not in _zoo_mlx_group()
    ), "the mlx group runs under xdist, which is the arrangement it exists to avoid"
    # The exclusion lives on the `ls | grep -v` that builds the list, not on the pytest
    # line, so it is read off the step text rather than the command.
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
