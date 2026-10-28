# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core.benchmark import resolve_model_details
from core.benchmark.orchestrator import BenchmarkOrchestrator
from core.benchmark.recipe_heldout import (
    eval_metrics,
    parse_judge_verdict,
    run_benchmark,
    score_with_validator,
)
from models.benchmark import BenchmarkRunRequest
from routes import benchmark_recipe as route

ROWS = [
    {"question": "2+2?", "answer": "4"},
    {"question": "Capital of France?", "answer": "Paris"},
    {"question": "3*3?", "answer": "9"},
]


def test_reference_scorer_matches_whole_words_only():
    assert score_with_validator("reference", "The answer is 4.", "4")[0]
    assert not score_with_validator("reference", "It is 14", "4")[0]
    assert not score_with_validator("reference", "anything", "")[0]


@pytest.mark.parametrize(
    "scorer, output, ok",
    [
        ("python", "```python\ndef f():\n    return 1\n```", True),
        ("python", "def f(:\n", False),
        ("json", '```json\n{"a": 1}\n```', True),
        ("json", "{a: 1}", False),
        ("markdown", "```py\nunclosed", False),
    ],
)
def test_validator_scorers(scorer, output, ok):
    assert score_with_validator(scorer, output, None)[0] is ok


def test_judge_verdict_ignores_thinking():
    assert parse_judge_verdict("<think>maybe FAIL</think>PASS\nfine") is True
    assert parse_judge_verdict("FAIL: wrong") is False
    assert parse_judge_verdict("no idea") is None


def test_candidate_against_baseline_and_eval_samples():
    answers = {
        "candidate": {"2+2?": "4", "Capital of France?": "Paris", "3*3?": "6"},
        "baseline": {"2+2?": "5", "Capital of France?": "Paris", "3*3?": "6"},
    }
    result = run_benchmark(
        {
            "rows": ROWS,
            "prompt_column": "question",
            "reference_column": "answer",
            "scorer": "reference",
            "with_baseline": True,
        },
        chat = lambda target, messages: answers[target][messages[-1]["content"]],
    )
    assert result["candidate"]["score"] == pytest.approx(2 / 3)
    assert result["baseline"]["score"] == pytest.approx(1 / 3)
    assert result["delta"] == pytest.approx(1 / 3)
    failed = result["failed_examples"][0]
    assert failed["prompt"] == "3*3?" and failed["output"] == "6" and failed["reference"] == "9"
    sample = result["samples"]["candidate"][2]
    assert sample == {
        "doc_id": 2,
        "question": "3*3?",
        "target": "9",
        "response": "6",
        "raw_response": "Answer does not contain the reference.",
        "correct": False,
    }
    assert eval_metrics(result["candidate"])[0]["name"] == "acc,none"


def test_judge_and_row_errors():
    def chat(target, messages):
        if target == "judge":
            return "PASS" if "Paris" in messages[-1]["content"].split("Answer to grade:")[1] else "FAIL"
        if messages[-1]["content"] == "3*3?":
            raise RuntimeError("HTTP 500: boom")
        return "Paris"

    result = run_benchmark(
        {"rows": ROWS, "prompt_column": "question", "scorer": "judge", "rubric": "Says Paris."},
        chat = chat,
    )
    assert result["candidate"] == {"total": 3, "scored": 2, "passed": 2, "errors": 1, "score": 1.0}
    assert "baseline" not in result
    assert result["failed_examples"][0]["reason"] == "HTTP 500: boom"
    # stored samples share the score's denominator
    assert [s["doc_id"] for s in result["samples"]["candidate"]] == [0, 1]
    assert result["errored"] == [{"index": 2, "error": "HTTP 500: boom"}]


def test_stops_on_cancel():
    calls = []
    result = run_benchmark(
        {"rows": ROWS, "prompt_column": "question", "reference_column": "answer", "scorer": "reference"},
        chat = lambda target, messages: calls.append(1) or "4",
        is_cancelled = lambda: len(calls) >= 1,
    )
    assert result["cancelled"] is True
    assert result["candidate"]["total"] == 1


def test_lm_eval_limit_reaches_simple_evaluate():
    _, kwargs = resolve_model_details("model.gguf", task = "gsm8k", limit = 10)
    assert kwargs["limit"] == 10
    _, kwargs = resolve_model_details("model.gguf", task = "gsm8k")
    assert "limit" not in kwargs


def test_recipe_task_is_in_the_registry():
    from routes import benchmark as evals

    task = next(t for t in evals._CURATED_TASKS if t.id == "recipe_heldout")
    assert task.runner == "recipe"
    assert all(t.runner == "lm_eval" for t in evals._CURATED_TASKS if t.id != "recipe_heldout")


def _request(**recipe):
    body = {
        "rows": ROWS,
        "prompt_column": "question",
        "reference_column": "answer",
        "scorer": "judge",
        "judge_alias": "local_model",
        "model_alias": "local_model",
        "baseline": {"mode": "adapter_off"},
        "model_providers": [{"name": "local", "is_local": True}],
        "model_configs": [{"alias": "local_model", "model": "", "provider": "local"}],
    }
    body.update(recipe)
    return BenchmarkRunRequest.model_validate(
        {"checkpoint_path": "", "model_source": "local", "task": "recipe_heldout", "recipe": body}
    )


def _fake_inject(recipe, request, expect_gen = None):
    for provider in recipe["model_providers"]:
        if provider.pop("is_local", None):
            provider.update(endpoint = "http://127.0.0.1:8888/v1", api_key = "sk-unsloth-test")
    return 7


@pytest.fixture
def harness(monkeypatch):
    backend = BenchmarkOrchestrator()
    stored, revoked, bodies = {}, [], []
    monkeypatch.setattr(route, "get_benchmark_backend", lambda: backend)
    monkeypatch.setattr(route, "_inject_local_providers", _fake_inject)
    monkeypatch.setattr(route, "_revoke_internal_api_key_safe", revoked.append)
    monkeypatch.setattr(route, "managed_account", lambda: False)
    monkeypatch.setattr(route, "current_account", lambda: None)
    monkeypatch.setattr(route, "_loaded_local_model_identity", lambda: (True, "unsloth/Qwen3-0.6B", ""))
    monkeypatch.setattr(
        route, "_store", lambda run_id, model, summary, samples, duration: stored.update({run_id: (model, summary, samples)})
    )

    def fake_post(client, target, messages):
        bodies.append(dict(target["body"]))
        if "Answer to grade" in messages[-1]["content"]:
            return "PASS" if "right" in messages[-1]["content"] else "FAIL"
        return "right" if target["body"].get("use_adapter") else "wrong"

    monkeypatch.setattr(route, "post_chat", fake_post)
    return SimpleNamespace(backend = backend, stored = stored, revoked = revoked, bodies = bodies)


def _run(request, via_api_key = False):
    return asyncio.run(route.run_recipe_heldout(request, SimpleNamespace(), ("admin", None), via_api_key))


def test_adapter_off_runs_through_the_evals_slot_and_storage(harness):
    response = _run(_request())
    recipe = response.details["recipe"]
    assert response.success
    assert recipe["candidate"]["score"] == 1.0 and recipe["baseline"]["score"] == 0.0
    assert recipe["delta"] == 1.0
    candidate_id, baseline_id = recipe["run_ids"]["candidate"], recipe["run_ids"]["baseline"]
    assert harness.stored[candidate_id][0] == "unsloth/Qwen3-0.6B"
    assert harness.stored[baseline_id][0] == "unsloth/Qwen3-0.6B (adapter off)"
    assert len(harness.stored[candidate_id][2]) == 3
    # the judge never runs with the adapter it grades
    judge_bodies = [b for b in harness.bodies if b.get("use_adapter") is False]
    assert len(judge_bodies) == 9  # 3 baseline answers + 6 judge calls
    assert harness.revoked == [7]
    assert not harness.backend.is_active()
    assert harness.backend.get_last_op_status() == "success"


def test_a_local_judge_never_grades_with_the_adapter_on_even_without_a_baseline(harness):
    response = _run(_request(baseline = {"mode": "none"}, local_adapter = True))
    assert response.success
    judge = [b for b in harness.bodies if b.get("use_adapter") is False]
    assert len(judge) == 3 and "baseline" not in response.details["recipe"]


def test_a_busy_slot_refuses_and_frees_nothing(harness):
    seq = harness.backend.try_begin(None)
    with pytest.raises(HTTPException) as exc:
        _run(_request())
    assert exc.value.status_code == 409
    harness.backend.release(seq)


def test_judge_needs_a_judge_model(harness):
    with pytest.raises(HTTPException) as exc:
        _run(_request(judge_alias = None))
    assert exc.value.status_code == 400
    assert not harness.backend.is_active()


def test_without_recipe_inputs_it_points_at_flows(harness):
    request = BenchmarkRunRequest.model_validate(
        {"checkpoint_path": "x", "model_source": "local", "task": "recipe_heldout"}
    )
    with pytest.raises(HTTPException) as exc:
        _run(request)
    assert "Flows" in exc.value.detail


def test_managed_account_cannot_reach_external_providers(harness, monkeypatch):
    monkeypatch.setattr(route, "managed_account", lambda: True)
    request = _request(
        scorer = "reference",
        judge_alias = None,
        baseline = {"mode": "none"},
        model_alias = "remote",
        model_providers = [{"name": "openai", "endpoint": "https://example.com/v1"}],
        model_configs = [{"alias": "remote", "model": "gpt", "provider": "openai"}],
    )
    with pytest.raises(HTTPException) as exc:
        _run(request)
    assert exc.value.status_code == 403
