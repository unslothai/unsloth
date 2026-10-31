# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A provenance-blocked resume must not get the checkpoint message; the exact blocker reason is used."""

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
    """Stubs the exact-resource check, which rejects any synthetic config as an unattested revision."""
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


@pytest.mark.parametrize(
    "marker",
    [
        {"status": "pending"},
        {"version": 2, "status": "pending"},
    ],
    ids = ["missing-version", "wrong-version"],
)
def test_malformed_pending_marker_reports_validation_blocker(marker):
    config = _config(**{RESOURCE_PROVENANCE_KEY: marker})

    assert resource_provenance_resume_blocker(config) == "The resource provenance is invalid."
    assert resource_provenance_allows_resume(config) is False


def test_an_unresumable_status_explains_itself(resources_available):
    config = _config(**{RESOURCE_PROVENANCE_KEY: {"version": 1, "status": "failed"}})

    blocker = resource_provenance_resume_blocker(config)

    assert blocker, "a refused resume must carry a reason"
    assert (
        "checkpoint" not in blocker.lower()
    ), "the checkpoint is intact here; naming it sends the user after the wrong thing"
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
    """The start route must let the provenance reason override the generic checkpoint text."""
    import inspect

    from routes import training as training_routes

    source = inspect.getsource(training_routes)
    branch = source.split("if not resume_run or not await asyncio.to_thread(", 1)[1]
    branch = branch.split("resume_checkpoint = await", 1)[0]

    assert "resource_provenance_resume_blocker" in branch
    assert branch.index("resource_provenance_resume_blocker") < branch.index(
        "raise HTTPException"
    ), "the reason must be resolved before the error is raised"
