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
    resolve_chat_target,
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
    assert parse_judge_verdict("It does not pass the check.\nFAIL") is None
    assert parse_judge_verdict("**FAIL** - it does not pass") is False


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


def test_no_delta_when_the_two_sides_scored_different_rows():
    def chat(target, messages):
        if target == "baseline" and messages[-1]["content"] == "2+2?":
            raise RuntimeError("boom")
        return {"2+2?": "4", "Capital of France?": "Paris", "3*3?": "9"}[messages[-1]["content"]]

    result = run_benchmark(
        {
            "rows": ROWS,
            "prompt_column": "question",
            "reference_column": "answer",
            "scorer": "reference",
            "with_baseline": True,
        },
        chat = chat,
    )
    assert result["candidate"]["score"] == 1.0 and result["baseline"]["score"] == 1.0
    assert result["delta"] is None


def test_judge_and_row_errors():
    def chat(target, messages):
        if target == "judge":
            return (
                "PASS"
                if "Paris" in messages[-1]["content"].split("Answer to grade:")[1]
                else "FAIL"
            )
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
        {
            "rows": ROWS,
            "prompt_column": "question",
            "reference_column": "answer",
            "scorer": "reference",
        },
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


def _fake_inject(
    recipe,
    request,
    expect_gen = None,
):
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
    monkeypatch.setattr("core.benchmark.hold.loaded_identity", lambda: "unsloth/Qwen3-0.6B")
    monkeypatch.setattr(
        route,
        "_store",
        lambda run_id, model, summary, samples, duration: stored.update(
            {run_id: (model, summary, samples)}
        ),
    )

    def fake_post(client, target, messages):
        bodies.append(dict(target["body"]))
        if "Answer to grade" in messages[-1]["content"]:
            return "PASS" if "right" in messages[-1]["content"] else "FAIL"
        return "right" if target["body"].get("use_adapter") else "wrong"

    monkeypatch.setattr(route, "post_chat", fake_post)
    return SimpleNamespace(backend = backend, stored = stored, revoked = revoked, bodies = bodies)


def _run(request, via_api_key = False):
    return asyncio.run(
        route.run_recipe_heldout(request, SimpleNamespace(), ("admin", None), via_api_key)
    )


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
    # every candidate answer turns the adapter back on, and so does the closing request
    assert [b.get("use_adapter") for b in harness.bodies] == [True, False] * 3 + [True]
    assert response.details["recipe"]["candidate"]["score"] == 1.0


def test_a_run_that_turned_the_adapter_off_ends_with_it_back_on(harness):
    _run(_request(baseline = {"mode": "none"}, local_adapter = True))
    # The judge's adapter-off request sticks on the server; the last request puts the LoRA back.
    assert harness.bodies[-2].get("use_adapter") is False
    assert harness.bodies[-1] == {**harness.bodies[0], "max_tokens": 1}
    assert harness.bodies[-1]["use_adapter"] is True


def test_a_run_without_an_adapter_sends_no_restore(harness):
    _run(_request(scorer = "reference", judge_alias = None, baseline = {"mode": "none"}))
    assert len(harness.bodies) == 3 and all("use_adapter" not in b for b in harness.bodies)


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


def test_two_providers_sharing_a_name_are_refused(harness):
    # Providers are found by name, so a second "local" would be the one the run calls.
    request = _request(
        scorer = "reference",
        judge_alias = None,
        baseline = {"mode": "none"},
        model_providers = [
            {"name": "local", "is_local": True},
            {
                "name": "local",
                "endpoint": "https://example.com/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
        ],
    )
    with pytest.raises(HTTPException) as exc:
        _run(request)
    assert exc.value.status_code == 400 and harness.bodies == []
    assert not harness.backend.is_active()


def test_a_managed_account_never_reads_a_server_env_key(monkeypatch):
    monkeypatch.setenv("BENCH_TEST_KEY", "sk-from-the-server")
    configs = {"remote": {"alias": "remote", "model": "gpt", "provider": "openai"}}
    providers = {
        "openai": {
            "name": "openai",
            "endpoint": "https://example.com/v1",
            "api_key_env": "BENCH_TEST_KEY",
        }
    }
    monkeypatch.setattr("core.training.account_jobs.managed_account", lambda: True)
    assert "sk-from-the-server" not in str(resolve_chat_target("remote", configs, providers))
    monkeypatch.setattr("core.training.account_jobs.managed_account", lambda: False)
    assert "sk-from-the-server" in str(resolve_chat_target("remote", configs, providers))


def test_stop_leaves_the_key_until_the_worker_turns_the_adapter_back_on(harness, monkeypatch):
    import threading

    release, events = threading.Event(), []

    def slow_post(client, target, messages):
        events.append(("post", target["body"].get("use_adapter"), target["body"].get("max_tokens")))
        if len(events) == 1:
            harness.backend.cancel()
            release.wait(10)
        return "right"

    monkeypatch.setattr(route, "post_chat", slow_post)
    monkeypatch.setattr(
        route, "_revoke_internal_api_key_safe", lambda key: events.append(("revoke", key))
    )
    response = _run(_request(scorer = "reference", judge_alias = None))
    assert response.details == {"cancelled": True}
    # The worker is still mid-run, so the route left the key alone.
    assert not any(e[0] == "revoke" for e in events)
    release.set()
    harness.backend._worker.result(timeout = 10)
    assert events[-2] == ("post", True, 1) and events[-1] == ("revoke", 7)


def test_reference_match_keeps_signs_and_decimals():
    assert score_with_validator("reference", "1", "-1")[0] is False
    assert score_with_validator("reference", "4.5", "4")[0] is False
    assert score_with_validator("reference", "The answer is 4.", "4")[0] is True
    assert score_with_validator("reference", "-1", "-1")[0] is True
    assert score_with_validator("reference", "1,000 apples", "1000")[0] is True


def test_every_local_scoring_request_withdraws_studios_tools(harness):
    # After the local provider is injected (is_local is popped from it), candidate, baseline and
    # judge requests still go out with Studio's tools off.
    response = _run(_request())
    assert response.success and harness.bodies
    assert all(
        b.get("enable_tools") is False and b.get("tool_choice") == "none" for b in harness.bodies
    )


@pytest.mark.parametrize(
    "output, reference, passed",
    [
        ("C#", "C++", False),
        ("x < 1", "x > 1", False),
        (".5", "5", False),
        ("0.5", ".5", True),
        ("Paris.", "paris", True),
        ("The answer is 42!", "42", True),
        ("x - 1", "x 1", False),
    ],
)
def test_reference_matching_keeps_symbols_that_change_the_answer(output, reference, passed):
    from core.benchmark.recipe_heldout import score_with_validator
    assert score_with_validator("reference", output, reference)[0] is passed


def test_each_row_is_asked_with_its_own_system_message():
    rows = [
        {"system": "Reply with RED", "q": "What label?", "a": "RED"},
        {"system": "Reply with BLUE", "q": "What label?", "a": "BLUE"},
        {"system": "", "q": "What label?", "a": "GREEN"},
    ]

    def chat(target, messages):
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        return system.rsplit(" ", 1)[-1] if system else "none"

    spec = {"rows": rows, "prompt_column": "q", "reference_column": "a", "scorer": "reference"}
    result = run_benchmark(
        {**spec, "system_column": "system", "system_prompt": "Reply with GREEN"}, chat = chat
    )
    # A blank cell falls back to the run's prompt.
    assert result["candidate"]["passed"] == 3
    assert run_benchmark(spec, chat = chat)["candidate"]["passed"] == 0


def test_the_route_asks_each_row_with_its_own_system_message(harness, monkeypatch):
    asked = []

    def post(client, target, messages):
        asked.append(messages[0]["content"] if messages[0]["role"] == "system" else None)
        return "4"

    monkeypatch.setattr(route, "post_chat", post)
    rows = [{**r, "rule": f"Rule {i}"} for i, r in enumerate(ROWS)]
    _run(_request(rows = rows, scorer = "reference", baseline = {"mode": "none"}, system_column = "rule"))
    assert asked == ["Rule 0", "Rule 1", "Rule 2"]


def test_a_structured_reference_is_scored_on_the_answer_training_used():
    # SQuAD-style answers train as their text, so "Paris" matches, not the dict's repr.
    rows = [{"question": "Capital of France?", "answer": {"text": ["Paris"], "answer_start": [0]}}]
    result = run_benchmark(
        {
            "rows": rows,
            "prompt_column": "question",
            "reference_column": "answer",
            "scorer": "reference",
        },
        chat = lambda target, messages: "Paris",
    )
    assert result["candidate"]["score"] == pytest.approx(1.0)


def test_a_scoring_holds_the_gpu_lease_so_training_cannot_evict_its_model(monkeypatch):
    from core.inference import gpu_arbiter

    refusals = []

    def scoring(*a, **k):
        # Another caller (a training start, a video load) asks while the rows are scored.
        import contextvars
        refusals.append(contextvars.Context().run(gpu_arbiter.lease_refusal))
        return {"ok": True}

    monkeypatch.setattr(route, "run_benchmark", scoring)
    monkeypatch.setattr(route, "_restore_adapter", lambda *a: None)
    route._recipe_task({"targets": {}, "spec": {}, "is_cancelled": lambda: False})
    assert isinstance(refusals[0], gpu_arbiter.GpuLeaseHeldError)
    assert not gpu_arbiter.lease_held()


def test_an_inequality_is_not_scored_as_an_equality():
    assert not score_with_validator("reference", "x = 1", "x != 1")[0]
    assert score_with_validator("reference", "So x != 1.", "x != 1")[0]
    assert score_with_validator("reference", "Hello!", "hello")[0]


def test_an_lm_eval_run_holds_the_model_for_its_whole_life(monkeypatch):
    from core.benchmark import orchestrator
    from core.inference import gpu_arbiter
    from core.benchmark import hold as bench_routes

    seen = []

    def evaluate(params):
        import contextvars
        seen.append(
            (
                bench_routes.scoring_active(),
                contextvars.Context().run(gpu_arbiter.lease_refusal) is not None,
            )
        )
        return {"results": {}}

    monkeypatch.setattr(orchestrator, "_run_benchmark_task", evaluate)
    assert orchestrator._held_benchmark_task({}) == {"results": {}}
    assert seen == [(True, True)]
    assert not bench_routes.scoring_active() and not gpu_arbiter.lease_held()


def test_the_judge_sees_the_system_instructions_the_answer_was_given():
    judged = []

    def chat(target, messages):
        if target == "judge":
            judged.append(messages[-1]["content"])
            return "FAIL\nIt broke the rule." if "Always reply in French" in judged[-1] else "PASS"
        return "Hello"

    result = run_benchmark(
        {
            "rows": [{"question": "Greet me"}],
            "prompt_column": "question",
            "scorer": "judge",
            "system_prompt": "Always reply in French.",
        },
        chat = chat,
    )
    assert "System instructions the answer was given:\nAlways reply in French." in judged[0]
    assert result["candidate"]["score"] == pytest.approx(0.0)


def test_a_local_model_is_stored_under_its_quant(monkeypatch):
    from core.benchmark import hold as bench_routes

    monkeypatch.setattr(bench_routes, "loaded_identity", lambda: "unsloth/Qwen3-0.6B-GGUF:Q8_0")
    configs = {"candidate": {"provider": "local", "model": ""}}
    # Two quants of one repo are two models in history and on the scoreboard.
    assert route._model_label("candidate", configs, {"local"}) == "unsloth/Qwen3-0.6B-GGUF:Q8_0"


def test_a_conversational_prompt_is_asked_as_turns_and_scored_on_the_answer():
    asked = []

    def chat(target, messages):
        asked.append(messages)
        return "Paris"

    rows = [
        {
            "prompt": [
                {"role": "system", "content": "Answer with one word."},
                {"role": "user", "content": "Capital of France?"},
            ],
            "completion": [{"role": "assistant", "content": "Paris"}],
        },
        {
            "prompt": [{"role": "user", "content": "Capital of Italy?"}],
            "completion": [{"role": "assistant", "content": "Rome"}],
        },
    ]
    result = run_benchmark(
        {
            "rows": rows,
            "prompt_column": "prompt",
            "reference_column": "completion",
            "scorer": "reference",
            "system_prompt": "Be helpful.",
        },
        chat = chat,
    )
    # The row's own system turn is kept, not doubled; a prompt without one gets the run's.
    assert asked[0] == rows[0]["prompt"]
    assert asked[1] == [{"role": "system", "content": "Be helpful."}, *rows[1]["prompt"]]
    # Scored against the assistant's text, not the message list as JSON.
    assert result["candidate"]["passed"] == 1
    failed = result["failed_examples"][0]
    assert failed["prompt"] == "Capital of Italy?" and failed["reference"] == "Rome"


def test_a_set_system_prompt_replaces_a_conversation_s_own():
    asked, judged = [], []

    def chat(target, messages):
        if target == "judge":
            judged.append(messages[-1]["content"])
            return "PASS"
        asked.append(messages)
        return "Hello"

    prompt = [
        {"role": "system", "content": "Answer in French."},
        {"role": "user", "content": "Greet me"},
    ]
    spec = {"rows": [{"prompt": prompt}], "prompt_column": "prompt", "scorer": "judge"}
    run_benchmark(
        {**spec, "system_prompt": "Answer in English.", "system_override": True}, chat = chat
    )
    assert asked[0] == [
        {"role": "system", "content": "Answer in English."},
        {"role": "user", "content": "Greet me"},
    ]
    assert "Answer in English." in judged[0] and "French" not in judged[0]
    # Training's own prompt (not an override) leaves the row's system turn in place.
    run_benchmark({**spec, "system_prompt": "Answer in English."}, chat = chat)
    assert asked[1] == prompt and "Answer in French." in judged[1]


def test_a_set_system_prompt_replaces_a_system_column_too():
    def chat(target, messages):
        return messages[0]["content"].rsplit(" ", 1)[-1] if messages[0]["role"] == "system" else ""

    spec = {
        "rows": [{"system": "Reply with RED", "q": "Which colour?", "a": "GREEN"}],
        "prompt_column": "q",
        "reference_column": "a",
        "system_column": "system",
        "system_prompt": "Reply with GREEN",
        "scorer": "reference",
    }
    assert run_benchmark({**spec, "system_override": True}, chat = chat)["candidate"]["passed"] == 1
    # Not an override: the row's own system message wins, as training sent it.
    assert run_benchmark(spec, chat = chat)["candidate"]["passed"] == 0


def test_a_whole_conversation_is_asked_without_its_answer_and_scored_on_it():
    asked, judged = [], []

    def chat(target, messages):
        if target == "judge":
            judged.append(messages[-1]["content"])
            return "PASS"
        asked.append(messages)
        return "Paris"

    turns = [
        {"role": "user", "content": "Capital of France?"},
        {"role": "assistant", "content": "Paris"},
    ]
    rows = [{"messages": turns}]
    spec = {"rows": rows, "prompt_column": "messages", "scorer": "reference"}
    result = run_benchmark(spec, chat = chat)
    # The held-out answer is never sent, and it is what the answer is scored against.
    assert asked[0] == turns[:1]
    assert result["candidate"]["passed"] == 1
    assert result["samples"]["candidate"][0]["question"] == "Capital of France?"
    assert result["samples"]["candidate"][0]["target"] == "Paris"
    run_benchmark({**spec, "scorer": "judge"}, chat = chat)
    assert "Instruction:\nCapital of France?\n\nReference answer:\nParis" in judged[0]


def test_a_reference_with_thinking_is_matched_as_the_answer_is():
    answer = "<think>Two plus two equals four.</think>4"
    result = run_benchmark(
        {
            "rows": [{"q": "2+2?", "a": answer}],
            "prompt_column": "q",
            "reference_column": "a",
            "scorer": "reference",
        },
        chat = lambda target, messages: answer,
    )
    assert result["candidate"]["passed"] == 1


def test_a_conversation_reference_is_its_closing_reply():
    from core.benchmark.recipe_heldout import reference_text

    chosen = [
        {"role": "user", "content": "Capital of France?"},
        {"role": "assistant", "content": "Paris"},
        {"role": "user", "content": "And Germany?"},
        {"role": "assistant", "content": "Berlin"},
    ]
    assert reference_text(chosen) == "Berlin"
    assert reference_text([{"role": "assistant", "content": "Paris"}]) == "Paris"


def test_a_prompt_with_its_own_reference_keeps_its_assistant_prefill():
    asked = []

    def chat(target, messages):
        asked.append(messages)
        return "8"

    prompt = [
        {"role": "user", "content": "Complete this sequence."},
        {"role": "assistant", "content": "2, 4, 6,"},
    ]
    result = run_benchmark(
        {
            "rows": [{"prompt": prompt, "answer": "8"}],
            "prompt_column": "prompt",
            "reference_column": "answer",
            "scorer": "reference",
        },
        chat = chat,
    )
    assert asked[0] == prompt and result["candidate"]["passed"] == 1
    assert "2, 4, 6," in result["samples"]["candidate"][0]["question"]


def test_a_single_message_object_reads_as_its_text():
    from core.benchmark.recipe_heldout import reference_text

    assert reference_text({"role": "assistant", "content": "Paris"}) == "Paris"
    assert reference_text({"from": "gpt", "value": "Paris"}) == "Paris"
    # A SQuAD answer is not a message.
    assert reference_text({"text": ["Paris"], "answer_start": [0]}) == "Paris"
