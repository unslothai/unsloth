# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from core.data_recipe.jobs.constants import STAGE_SAMPLING
from core.data_recipe.jobs.parse import apply_update, parse_log_message
from core.data_recipe.jobs.types import Job


def test_current_sampling_log_sets_stage_and_rows():
    # data-designer's current phrasing (seen in issue #5848 logs). Without this
    # the sampling stage never fires and the row total used for overall progress
    # is lost. The legacy "Preparing samplers ..." line is covered below.
    update = parse_log_message("🌱 Sampling 25 records from seed dataset")

    assert update is not None
    assert update.stage == STAGE_SAMPLING
    assert update.rows == 25

    job = Job(job_id = "job-1")
    apply_update(job, update)
    assert job.stage == STAGE_SAMPLING
    assert job.rows == 25


def test_legacy_sampling_log_still_parsed():
    update = parse_log_message("Preparing samplers to generate 50 records across 3 columns")

    assert update is not None
    assert update.stage == STAGE_SAMPLING
    assert update.rows == 50
    assert update.cols == 3
