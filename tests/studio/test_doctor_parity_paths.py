# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Doctor must not call two different usernames a capability divergence.

Provisioning explicitly supports a peer with a different home, so on such a pair
`/home/alice/.unsloth/.../torchrun` and `/home/bob/.unsloth/.../torchrun` are the same
capability. Compared as raw strings they read as a divergence, doctor returns failure, and it
warns of a deadlock while both tools are present and equivalent.

Only the home prefix is folded. A tool at `/usr/bin` on one node and inside the managed
environment on the other is a real difference and must still be reported.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _doctor():
    spec = importlib.util.spec_from_file_location(
        "unsloth_doctor_for_parity", REPO / "unsloth_cli" / "commands" / "doctor.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _node(home: str, **overrides) -> dict:
    probe = {
        "home": home,
        "which_torchrun": f"{home}/.unsloth/studio/unsloth_studio/bin/torchrun",
        "which_python3": f"{home}/.unsloth/studio/unsloth_studio/bin/python3",
        "executable": f"{home}/.unsloth/studio/unsloth_studio/bin/python",
        "python_version": "3.13.1",
    }
    probe.update(overrides)
    return probe


def test_different_homes_are_not_a_divergence() -> None:
    doctor = _doctor()
    assert doctor.compare_parity(_node("/home/alice"), _node("/home/bob")) == []


def test_a_tool_outside_the_managed_environment_still_diverges() -> None:
    """The case the comparison exists for: same name, genuinely different thing."""
    doctor = _doctor()
    diverged = doctor.compare_parity(
        _node("/home/alice"),
        _node("/home/bob", which_torchrun = "/usr/bin/torchrun"),
    )
    assert [k for k, _, _ in diverged] == ["which_torchrun"], diverged


def test_a_missing_tool_still_diverges() -> None:
    doctor = _doctor()
    diverged = doctor.compare_parity(
        _node("/home/alice"),
        _node("/home/bob", which_torchrun = None),
    )
    assert [k for k, _, _ in diverged] == ["which_torchrun"], diverged


def test_non_path_probes_are_untouched() -> None:
    """Only `which_*` and `executable` are folded; a version difference is still a difference."""
    doctor = _doctor()
    diverged = doctor.compare_parity(
        _node("/home/alice"),
        _node("/home/bob", python_version = "3.12.4"),
    )
    assert [k for k, _, _ in diverged] == ["python_version"], diverged


def test_the_probe_reports_the_home_it_normalises_against() -> None:
    """Without it there is nothing to fold against, and the comparison silently does nothing."""
    doctor = _doctor()
    assert "r['home'] = os.path.expanduser('~')" in doctor.parity_probe_source()
    assert "home" in doctor.PARITY_SKIP, "the home itself would read as a divergence"


# ── --data validation, which belongs before the model load ──────────────────────
# `reps = (need + len(ids) - 1) // len(ids)` divides by the row count, so an empty file raised
# ZeroDivisionError, and only after both ranks had loaded and materialised the model.


def _pipeline():
    spec = importlib.util.spec_from_file_location(
        "spark_pipeline_for_data", REPO / "studio" / "spark_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_empty_dataset_is_rejected(tmp_path) -> None:
    pipeline = _pipeline()
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding = "utf-8")
    assert "no rows" in (pipeline.dataset_problem(str(empty)) or "")

    whitespace = tmp_path / "blank.jsonl"
    whitespace.write_text("\n\n   \n", encoding = "utf-8")
    assert "no rows" in (pipeline.dataset_problem(str(whitespace)) or "")


def test_an_unreadable_dataset_says_so(tmp_path) -> None:
    pipeline = _pipeline()
    assert "could not be read" in (pipeline.dataset_problem(str(tmp_path / "missing.jsonl")) or "")


def test_a_dataset_with_rows_is_accepted(tmp_path) -> None:
    pipeline = _pipeline()
    good = tmp_path / "good.jsonl"
    good.write_text('{"q": "a", "a": "b"}\n', encoding = "utf-8")
    assert pipeline.dataset_problem(str(good)) is None


def test_the_check_runs_before_the_model_is_built() -> None:
    """A late check costs the most expensive part of the run to reach an input error."""
    source = (REPO / "studio" / "spark_pipeline.py").read_text(encoding = "utf-8")
    assert source.index("dataset_problem(args.data)") < source.index(
        "model, cfg, _ = build_stage_model("
    )
