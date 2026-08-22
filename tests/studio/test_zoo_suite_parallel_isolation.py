# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""
The unsloth_zoo suite runs in parallel, minus two files that must not share a worker.

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

So the pair is ignored from the parallel pass and run again in a process of their
own. That pairing is the thing this file guards, because half of it is silent:
drop the serial pass and the ignores simply delete 51 tests from CI while the job
stays green. That is strictly worse than the 14 failures, which at least announce
themselves.

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
]

# The zoo suite is the only pytest run in this workflow driven out of the cloned zoo
# checkout, and it is told apart by the deselect it carries rather than by step order,
# so re-ordering or renaming steps does not quietly point this guard at another command.
ZOO_MARKER = "tests/test_mlx_finetune_last_n_layers.py::test_get_peft_model_passes_finetune_last_n_layers_through"


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


def _zoo_serial() -> str:
    hits = [c for c in _commands() if "-n 4" not in c and all(path in c for path, _ in ISOLATED)]
    assert (
        len(hits) == 1
    ), f"expected exactly one serial rerun naming both isolated files, found {len(hits)}"
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
    assert f"--ignore={path}" in _zoo_parallel(), (
        f"{path} ({reason}) is not ignored by the parallel zoo run, so it goes back to "
        f"failing intermittently depending on which worker picks it up"
    )


@pytest.mark.parametrize("path,reason", ISOLATED, ids = lambda v: v.split("/")[-1])
def test_an_isolated_file_still_runs_serially(path: str, reason: str) -> None:
    """The silent half. An ignore with no rerun deletes the tests and stays green."""
    assert path in _zoo_serial(), (
        f"{path} is ignored from the parallel run but never run again. Its tests are "
        f"simply not executed, and nothing else in CI would say so."
    )


def test_the_serial_rerun_is_not_itself_parallel() -> None:
    """Rerunning the pair under xdist would reproduce exactly what it exists to avoid."""
    assert "-n " not in _zoo_serial(), (
        "the serial rerun of the isolated files passes -n, which puts them back in the "
        "parallel session whose ordering is what breaks them"
    )


def test_the_deselects_survive_on_the_parallel_run() -> None:
    """
    The three deselects are the CUDA-only and known-broken cases. They lived on the
    single command that this change split in two; a split that dropped them would turn
    a deliberate 'deselected' into a runtime failure on a GPU-less runner.
    """
    cmd = _zoo_parallel()
    assert (
        cmd.count("--deselect") == 3
    ), f"the parallel zoo run carries {cmd.count('--deselect')} deselects, expected 3"
