# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

# stages parsed from data-designer logs
STAGE_CREATE = "create"
STAGE_PREVIEW = "preview"
STAGE_DAG = "dag"
STAGE_HEALTHCHECK = "healthcheck"
STAGE_SAMPLING = "sampling"
STAGE_SOURCE = "source"
STAGE_COLUMN_CONFIG = "column_config"
STAGE_GENERATING = "generating"
STAGE_BATCH = "batch"
STAGE_PROFILING = "profiling"

USAGE_RESET_STAGES = {
    STAGE_CREATE,
    STAGE_PREVIEW,
    STAGE_DAG,
    STAGE_HEALTHCHECK,
    STAGE_SAMPLING,
    STAGE_GENERATING,
    STAGE_PROFILING,
}

# job event types emitted by worker/manager
EVENT_JOB_ENQUEUED = "job.enqueued"
EVENT_JOB_STARTED = "job.started"
EVENT_JOB_CANCELLING = "job.cancelling"
EVENT_JOB_CANCELLED = "job.cancelled"
EVENT_JOB_COMPLETED = "job.completed"
EVENT_JOB_ERROR = "job.error"
