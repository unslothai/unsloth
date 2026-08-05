# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A provenance-blocked resume must not be reported as a checkpoint problem.

``can_resume_run`` gained a provenance clause in this rework, so it now returns False for
runs whose checkpoint is entirely intact -- most realistically after the pinned model
snapshot is evicted from the HF cache. The start route answered every False with

    "Resume checkpoint must belong to a stopped or errored run with complete saved
     trainer state."

which points the user at trainer state that is fine. ``exact_resume_resource_requirements``
already raises with the precise reason; it was being discarded.
"""

import pytest

from core.training.provenance import (
    RESOURCE_PROVENANCE_KEY,
    resource_provenance_allows_resume,
    resource_provenance_resume_blocker,
)


def _config(**overrides):
    config = {
        RESOURCE_PROVENANCE_KEY: {"version": 1, "status": "complete"},
        "model_name": "unsloth/Llama-3.2-1B-Instruct",
    }
    config.update(overrides)
    return config


@pytest.fixture
def resources_available(monkeypatch):
    """Satisfy the exact-resource check so a test can isolate the status logic.

    Without this the requirements check raises "model revision was not attested" for any
    synthetic config, which is correct behaviour but masks what these cases are about.
    """
    from core.training import provenance as provenance_mod
    monkeypatch.setattr(provenance_mod, "exact_resume_resource_requirements", lambda config: None)


def test_a_run_without_provenance_is_not_blocked(resources_available):
    """Runs written before this rework carry no marker and must stay resumable."""
    config = _config()
    config.pop(RESOURCE_PROVENANCE_KEY)

    assert resource_provenance_resume_blocker(config) is None
    assert resource_provenance_allows_resume(config) is True


@pytest.mark.parametrize("status", ["pending", "incomplete", "complete"])
def test_resumable_statuses_report_no_blocker(status, resources_available):
    config = _config(**{RESOURCE_PROVENANCE_KEY: {"version": 1, "status": status}})

    assert resource_provenance_resume_blocker(config) is None
    assert resource_provenance_allows_resume(config) is True


def test_an_unresumable_status_explains_itself(resources_available):
    config = _config(**{RESOURCE_PROVENANCE_KEY: {"version": 1, "status": "failed"}})

    blocker = resource_provenance_resume_blocker(config)

    assert blocker, "a refused resume must carry a reason"
    assert "checkpoint" not in blocker.lower(), (
        "the checkpoint is intact here; naming it sends the user after the wrong thing"
    )
    assert "failed" in blocker
    assert resource_provenance_allows_resume(config) is False


def test_missing_exact_resources_surface_their_own_message(monkeypatch):
    """The precise reason from the requirements check must reach the caller."""
    from core.training import provenance as provenance_mod

    message = "The exact model snapshot for this run is no longer available."

    def unavailable(config):
        raise provenance_mod.ExactResumeResourcesUnavailable(message)

    monkeypatch.setattr(provenance_mod, "exact_resume_resource_requirements", unavailable)

    assert provenance_mod.resource_provenance_resume_blocker(_config()) == message
    assert provenance_mod.resource_provenance_allows_resume(_config()) is False


def test_the_two_helpers_cannot_disagree(resources_available):
    """allows_resume is defined in terms of the blocker, so they stay in step."""
    for status in ("pending", "incomplete", "complete", "failed", "bogus", None):
        config = _config(**{RESOURCE_PROVENANCE_KEY: {"version": 1, "status": status}})
        blocked = resource_provenance_resume_blocker(config) is not None
        assert blocked is not resource_provenance_allows_resume(config)


def test_the_start_route_prefers_the_provenance_reason():
    """Wiring contract: the generic checkpoint text must not be the only answer.

    Kept narrow -- it asserts the blocker is consulted in the resume rejection branch and
    that its result overrides the default, which is exactly what regressed.
    """
    import inspect

    from routes import training as training_routes

    source = inspect.getsource(training_routes)
    branch = source.split("if not resume_run or not await asyncio.to_thread(", 1)[1]
    branch = branch.split("resume_checkpoint = await", 1)[0]

    assert "resource_provenance_resume_blocker" in branch
    assert branch.index("resource_provenance_resume_blocker") < branch.index(
        "raise HTTPException"
    ), "the reason must be resolved before the error is raised"
