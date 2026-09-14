# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Guard the parallel zoo run and the serial reruns for tests that cannot share workers."""

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
