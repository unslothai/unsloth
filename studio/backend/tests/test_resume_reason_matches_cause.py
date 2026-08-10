# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The refusal reason has to match the reason the run was actually refused.

``300fe6321`` and ``a98e721f4`` set out to stop a provenance refusal being reported as a
checkpoint problem. They overshot: both sites asked
``resource_provenance_resume_blocker`` whenever ``can_resume_run`` said no, but that
function refuses for several reasons and the blocker is computed independently of which
one fired. Since ``initialize_resource_provenance`` writes ``{"version": 1, "status":
"pending"}`` at the start of every run, the blocker answers "The model revision used by
this run was not attested." for any Hub-model run -- including one whose checkpoint is
simply missing, which is the *more* common way to be unresumable. So the fix traded one
misdiagnosis for another and the new one covered the bigger population.

The discriminator is ``has_resume_state``, the same one ``can_resume_run`` short-circuits
on: with no saved trainer state the checkpoint is the cause and the client's own wording
is right; with the state intact a refusal really is provenance's doing.
"""

import json

import pytest

from core.training.provenance import RESOURCE_PROVENANCE_KEY
from routes import training_history


_ATTESTATION = "The model revision used by this run was not attested."


def _row(tmp_path, **overrides):
    row = {
        "id": "run-1",
        "status": "error",
        "model_name": "unsloth/Llama-3.2-1B-Instruct",
        "dataset_name": "yahma/alpaca-cleaned",
        "started_at": "2026-08-06T00:00:00Z",
        "output_dir": str(tmp_path / "run-1"),
        # A real run's config: the trained model plus what initialize_resource_provenance writes.
        "config_json": json.dumps(
            {
                "model_name": "unsloth/Llama-3.2-1B-Instruct",
                RESOURCE_PROVENANCE_KEY: {"version": 1, "status": "pending"},
            }
        ),
    }
    row.update(overrides)
    return row


@pytest.fixture
def unresumable(monkeypatch):
    monkeypatch.setattr(training_history, "can_resume_run", lambda *a, **k: False)
    monkeypatch.setattr(training_history, "artifacts_present", lambda *a, **k: True)
    monkeypatch.setattr(
        training_history, "_preview_fields", lambda *a, **k: {"has_preview_model": False}
    )


def test_a_missing_checkpoint_is_not_reported_as_an_attestation_problem(tmp_path, unresumable):
    """The regression: no trainer state on disk, so the checkpoint is the real cause."""
    (tmp_path / "run-1").mkdir()

    summary = training_history._summary_from_row(_row(tmp_path), False)

    assert summary.can_resume is False
    assert summary.resume_blocked_reason != _ATTESTATION
    assert summary.resume_blocked_reason is None, (
        "with no saved trainer state the client's own checkpoint wording is the correct "
        "diagnosis; handing it a provenance sentence blames the wrong thing"
    )


def test_an_intact_checkpoint_still_gets_the_provenance_reason(tmp_path, unresumable, monkeypatch):
    """The case 300fe6321 was actually for must keep working."""
    monkeypatch.setattr(training_history, "has_resume_state", lambda *a, **k: True, raising = False)
    from core.training import resume as resume_mod

    monkeypatch.setattr(resume_mod, "has_resume_state", lambda output_dir: True)

    summary = training_history._summary_from_row(_row(tmp_path), False)

    assert summary.resume_blocked_reason == _ATTESTATION


def test_a_resumable_run_carries_no_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(training_history, "can_resume_run", lambda *a, **k: True)
    monkeypatch.setattr(training_history, "artifacts_present", lambda *a, **k: True)
    monkeypatch.setattr(
        training_history, "_preview_fields", lambda *a, **k: {"has_preview_model": False}
    )

    summary = training_history._summary_from_row(_row(tmp_path), False)

    assert summary.can_resume is True
    assert summary.resume_blocked_reason is None


def test_a_row_with_no_output_dir_is_not_diagnosed_as_provenance(tmp_path, unresumable):
    summary = training_history._summary_from_row(_row(tmp_path, output_dir = None), False)

    assert summary.resume_blocked_reason is None


def test_the_start_route_gates_the_substitution_on_the_checkpoint(tmp_path):
    """Wiring contract: the route must not ask the blocker unconditionally.

    Driving the real async start path here would mean standing up the whole request
    stack; what regressed is a missing guard, so pin the guard -- but over the AST, not
    the source text. A substring search is satisfied by the explanatory comment that
    sits right next to the code, which makes it pass with the guard deleted.
    """
    import ast
    import inspect

    from routes import training as training_routes

    tree = ast.parse(inspect.getsource(training_routes))

    def guarded_blocker_calls(node):
        """`if ... has_resume_state ... :` blocks that contain the blocker lookup."""
        for sub in ast.walk(node):
            if not isinstance(sub, ast.If):
                continue
            test_names = {n.id for n in ast.walk(sub.test) if isinstance(n, ast.Name)} | {
                n.attr for n in ast.walk(sub.test) if isinstance(n, ast.Attribute)
            }
            if "has_resume_state" not in test_names:
                continue
            body_names = {
                n.id
                for n in ast.walk(ast.Module(body = sub.body, type_ignores = []))
                if isinstance(n, ast.Name)
            }
            if "resource_provenance_resume_blocker" in body_names:
                yield sub

    all_blocker_names = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id == "resource_provenance_resume_blocker"
    ]
    assert all_blocker_names, "the blocker lookup is gone from the route entirely"

    guarded = list(guarded_blocker_calls(tree))
    assert guarded, (
        "the provenance reason is substituted without an enclosing has_resume_state "
        "check, so a run with a missing checkpoint is told its model revision was not "
        "attested"
    )
