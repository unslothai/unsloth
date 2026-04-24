# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from core.data_recipe.jobs.parse import apply_update, parse_log_message
from core.data_recipe.jobs.types import Job
from routes.data_recipe.validate import _GITHUB_VALIDATE_NOTE, validate
from models.data_recipe import RecipePayload


def test_github_page_log_updates_source_progress_without_cursor():
    job = Job(job_id = "job-1")
    job.source_progress_estimated_total = 200

    update = parse_log_message(
        "[unslothai/unsloth] issues page 2 (+15) cursor=abc123 remaining=2960"
    )

    assert update is not None
    apply_update(job, update)

    progress = job.source_progress
    assert progress is not None
    assert progress.source == "github"
    assert progress.status == "fetching"
    assert progress.repo == "unslothai/unsloth"
    assert progress.resource == "issues"
    assert progress.page == 2
    assert progress.page_items == 15
    assert progress.fetched_items == 15
    assert progress.estimated_total == 200
    assert progress.rate_remaining == 2960
    assert progress.message is not None
    assert "cursor" not in progress.message
    assert "abc123" not in progress.message


def test_github_rate_limit_log_updates_source_progress():
    job = Job(job_id = "job-1")

    update = parse_log_message("Rate limit hit. Sleeping 123s until reset.")

    assert update is not None
    apply_update(job, update)

    progress = job.source_progress
    assert progress is not None
    assert progress.status == "rate_limited"
    assert progress.retry_after_sec == 123
    assert "resume automatically" in (progress.message or "")


def test_github_real_sample_prs_and_trial_limit_are_parsed():
    job = Job(job_id = "job-1")

    for message in (
        "[unslothai/unsloth] PRs page 4 (+25) cursor=abc123 remaining=4983",
        "Trial limit reached for PRs (100)",
    ):
        update = parse_log_message(message)
        assert update is not None
        apply_update(job, update)

    progress = job.source_progress
    assert progress is not None
    assert progress.repo == "unslothai/unsloth"
    assert progress.resource == "pulls"
    assert progress.page == 4
    assert progress.fetched_items == 25
    assert progress.rate_remaining == 4983
    assert progress.message == "GitHub pulls trial limit reached (100)."


def test_github_validate_skips_live_access_with_honest_note():
    response = validate(
        RecipePayload(
            recipe = {
                "seed_config": {
                    "source": {
                        "seed_type": "github_repo",
                        "repos": ["unslothai/unsloth"],
                        "item_types": ["issues"],
                        "limit": 1,
                    }
                },
                "columns": [{"column_type": "expression", "name": "x", "expr": "1"}],
            }
        )
    )

    assert response.valid is True
    assert response.raw_detail == _GITHUB_VALIDATE_NOTE
