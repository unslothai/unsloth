# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.training import rl_samples
from core.training.rl_samples import SampleRecorder, record_samples
from core.training.training import TrainingBackend

SPECS = [{"name": "format", "weight": 1.0}, {"name": "correct", "weight": 2.0}]
PROMPTS = [[{"role": "system", "content": "s"}, {"role": "user", "content": "What is 48 + 24?"}]] * 4
COMPLETIONS = [[{"role": "assistant", "content": f"answer {i}"}] for i in range(4)]


def _funcs():
    def format(prompts = None, completions = None, **kw):
        return [0.5, 0.0, 0.5, 0.0]

    def correct(prompts = None, completions = None, **kw):
        return [2.0, 2.0, None, float("nan")]

    return [format, correct]


def _run(funcs, **kwargs):
    return [f(prompts = PROMPTS, completions = COMPLETIONS, **kwargs) for f in funcs]


def test_one_group_per_batch_with_every_reward_and_the_weighted_total():
    sent = []
    funcs = record_samples(_funcs(), SPECS, sent.append, group_size = 2)
    out = _run(funcs, answer = ["72"] * 4, trainer_state = SimpleNamespace(global_step = 7))
    assert out[0] == [0.5, 0.0, 0.5, 0.0]  # scores pass through untouched
    [group] = sent
    assert group["step"] == 7 and group["prompt"] == "What is 48 + 24?" and group["answer"] == "72"
    assert [i["completion"] for i in group["items"]] == ["answer 0", "answer 1"]
    assert group["items"][0]["rewards"] == {"format": 0.5, "correct": 2.0}
    assert group["items"][0]["total"] == 4.5
    assert group["items"][1]["total"] == 4.0


def test_missing_and_nan_scores_become_none_and_count_as_zero():
    sent = []
    _run(record_samples(_funcs(), SPECS, sent.append, group_size = 4))
    items = sent[0]["items"]
    assert items[2]["rewards"]["correct"] is None and items[3]["rewards"]["correct"] is None
    assert items[2]["total"] == 0.5


def test_wrappers_keep_the_names_trl_logs_under():
    funcs = record_samples(_funcs(), SPECS, lambda _: None, group_size = 2)
    assert [f.__name__ for f in funcs] == ["format", "correct"]


def test_no_sink_leaves_the_functions_alone():
    funcs = _funcs()
    assert record_samples(funcs, SPECS, None, group_size = 2) is funcs


def test_groups_are_throttled():
    now = [0.0]
    sent = []
    recorder = SampleRecorder(["format", "correct"], [1.0, 2.0], sent.append, 2, min_interval = 10, clock = lambda: now[0])
    funcs = [recorder.wrap(i, f) for i, f in enumerate(_funcs())]
    _run(funcs)
    now[0] = 5.0
    _run(funcs)
    now[0] = 11.0
    _run(funcs)
    assert len(sent) == 2


def test_a_broken_sink_never_stops_training():
    def sink(_):
        raise RuntimeError("queue closed")

    funcs = record_samples(_funcs(), SPECS, sink, group_size = 2)
    assert _run(funcs)[0] == [0.5, 0.0, 0.5, 0.0]


def test_long_text_is_trimmed():
    sent = []
    funcs = record_samples(_funcs(), SPECS, sent.append, group_size = 1)
    long = [[{"role": "assistant", "content": "x" * 10_000}]] * 4
    for f in funcs:
        f(prompts = ["p" * 5_000] * 4, completions = long)
    assert len(sent[0]["items"][0]["completion"]) == rl_samples.MAX_COMPLETION_CHARS + 1
    assert sent[0]["prompt"].startswith("…") and len(sent[0]["prompt"]) == rl_samples.MAX_PROMPT_CHARS + 1


def _sample_event(step):
    return {"type": "samples", "step": step, "prompt": "q", "answer": "72", "items": [{"completion": "c", "rewards": {"r": 1.0}, "total": 1.0}]}


def test_the_backend_keeps_the_newest_groups_in_order():
    backend = TrainingBackend()
    for step in range(TrainingBackend.RL_SAMPLE_LIMIT + 3):
        backend._handle_event(_sample_event(step))
    assert len(backend.rl_samples) == TrainingBackend.RL_SAMPLE_LIMIT
    assert backend.rl_samples[-1]["seq"] == TrainingBackend.RL_SAMPLE_LIMIT + 3
    assert backend.rl_samples[0]["step"] == 3


@pytest.fixture
def client(monkeypatch):
    from auth.authentication import get_current_subject
    from routes import training as training_routes

    backend = TrainingBackend()
    backend.current_job_id = "job-1"
    for step in (1, 2, 3):
        backend._handle_event(_sample_event(step))
    monkeypatch.setattr(training_routes, "get_training_backend", lambda: backend)
    monkeypatch.setattr(training_routes, "job_is_foreign", lambda _b: False)
    app = FastAPI()
    app.include_router(training_routes.router, prefix = "/api/train")
    app.dependency_overrides[get_current_subject] = lambda: "test"
    return TestClient(app)


def test_the_route_returns_only_newer_groups(client):
    body = client.get("/api/train/rl-samples", params = {"after": 1}).json()
    assert body["job_id"] == "job-1"
    assert [s["seq"] for s in body["samples"]] == [2, 3]


def test_the_route_refuses_a_stale_job(client):
    response = client.get("/api/train/rl-samples", params = {"expected_job_id": "old"})
    assert response.status_code == 409
