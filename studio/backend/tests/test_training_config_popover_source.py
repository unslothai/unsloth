# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Source-level regression guards for the Training Config popover data source
(#6853).

The live Training Progress popover used to read the editable form store
(useTrainingConfigStore) while a run was active, so it showed stale/static
values whenever the user touched the form after starting the run; only the
History view read the run's saved config snapshot. These guards pin the fixed
wiring: both views feed ProgressSection a config override mapped from
GET /api/train/runs/{id}, and ProgressSection prefers that override whenever
one is present -- not only for historical views.
"""

from __future__ import annotations

from pathlib import Path

_STUDIO_FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "src" / "features" / "studio"


def _read(rel: str) -> str:
    return (_STUDIO_FRONTEND / rel).read_text(encoding = "utf-8")


def test_progress_section_prefers_override_over_form_store():
    src = _read("sections/progress-section.tsx")
    # Fields key on the override's presence, not isHistorical, so History shows blanks rather than live form values.
    assert "const cfg = configOverride ?? (isHistorical ? undefined : config)" in src
    assert "const cfgEpochs = cfg?.epochs" in src
    assert "isHistorical ? configOverride?.epochs" not in src


def test_live_view_fetches_the_active_run_config():
    src = _read("live-training-view.tsx")
    assert "getTrainingRun(" in src
    assert "mapRunConfigToOverride(" in src
    assert "configOverride={runConfigOverride}" in src


def test_live_view_fetches_as_soon_as_the_job_id_exists():
    # start_training() inserts the run row BEFORE the pump consumes any event, so the job id is the
    # whole readiness condition: gating on a first step or a terminal phase would show the wrong
    # config for the entire pre-step window of a long load.
    src = _read("live-training-view.tsx")
    assert "if (!runtime.jobId) {" in src
    assert "[runtime.jobId, fetchedRunConfig, fetchAttempt]" in src
    assert "runRowReady" not in src


def test_live_view_retries_the_transient_row_miss():
    # A lookup racing that commit can 404 and no dep changes on failure, so the retry must be explicit and bounded.
    src = _read("live-training-view.tsx")
    assert "RUN_CONFIG_FETCH_RETRIES" in src
    assert "RUN_CONFIG_FETCH_RETRY_MS" in src
    assert "setFetchAttempt(" in src
    assert "attempts >= RUN_CONFIG_FETCH_RETRIES" in src
    assert "fetchAttempt?.jobId === jobId ? fetchAttempt.count : 0" in src
    assert "clearTimeout(retryTimer)" in src


def test_live_view_prefers_saved_training_method():
    # The method label and LoRA rows come from the run snapshot, not the form, which may have changed.
    src = _read("live-training-view.tsx")
    assert "runConfigOverride?.trainingMethod ?? config.trainingMethod" in src


def test_history_view_uses_the_shared_mapper():
    src = _read("historical-training-view.tsx")
    assert "mapRunConfigToOverride(detail.config)" in src
    assert "num_epochs" not in src


def test_shared_mapper_matches_backend_config_keys():
    src = _read("sections/run-config-override.ts")
    # The mapper reads the run config the backend snapshots at job start, so pin the key set.
    for key in (
        "training_type",
        "load_in_4bit",
        "num_epochs",
        "batch_size",
        "learning_rate",
        "max_steps",
        "max_seq_length",
        "warmup_steps",
        "optim",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "use_rslora",
        "use_loftq",
        "use_dora",
    ):
        assert key in src, f"run-config mapper lost backend key {key}"
