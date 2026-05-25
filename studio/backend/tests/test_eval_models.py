# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from models import (
    EvalStartRequest, EvalRunSummary, EvalRunDetail, EvalResultRow,
    EvalProgress, MetricInfo,
)


def test_eval_start_request_defaults():
    req = EvalStartRequest(
        model_identifier="hf/m",
        dataset={"is_local": True, "path": "d.jsonl", "split": "train"},
        input_column="q", reference_column="a",
        metric_name="exact_match",
    )
    assert req.limit == 100
    assert req.metric_config == {}
    assert req.temperature == 0.0
    assert req.max_new_tokens == 256
    assert req.system_prompt == ""


def test_eval_progress_roundtrip():
    p = EvalProgress(run_id="r", status="running", done=1, total=2, avg_score=0.5)
    assert p.model_dump()["done"] == 1
