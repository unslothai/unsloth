# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from eval.runner import run_eval, EvalSummary
from eval.metrics.registry import make_scorer


def _echo_generate(reference_by_input):
    def generate(messages, system_prompt, **gen):
        user = messages[-1]["content"]
        return reference_by_input.get(user, "")
    return generate


def test_runs_all_examples_and_averages():
    examples = [("inA", "x"), ("inB", "y")]
    generate = _echo_generate({"inA": "x", "inB": "WRONG"})
    collected = []
    summary = run_eval(
        examples=examples,
        generate=generate,
        scorer=make_scorer("exact_match", {}),
        system_prompt="", template=None, gen_params={},
        should_cancel=lambda: False,
        on_result=lambda idx, res, pred, inp, ref: collected.append((idx, res.score)),
    )
    assert isinstance(summary, EvalSummary)
    assert summary.num_scored == 2
    assert abs(summary.avg_score - 0.5) < 1e-9
    assert summary.status == "completed"
    assert collected == [(0, 1.0), (1, 0.0)]


def test_template_renders_input():
    seen = {}
    def generate(messages, system_prompt, **gen):
        seen["content"] = messages[-1]["content"]
        return "x"
    run_eval(
        examples=[("world", "x")], generate=generate,
        scorer=make_scorer("exact_match", {}),
        system_prompt="sys", template="Hello {input}!", gen_params={},
        should_cancel=lambda: False, on_result=lambda *a: None,
    )
    assert seen["content"] == "Hello world!"


def test_cancellation_stops_early():
    calls = {"n": 0}
    def generate(messages, system_prompt, **gen):
        calls["n"] += 1
        return "x"
    summary = run_eval(
        examples=[("a", "x"), ("b", "x"), ("c", "x")],
        generate=generate, scorer=make_scorer("exact_match", {}),
        system_prompt="", template=None, gen_params={},
        should_cancel=lambda: calls["n"] >= 1,  # cancel after first generate
        on_result=lambda *a: None,
    )
    assert summary.status == "cancelled"
    assert summary.num_scored == 1


def test_generation_error_does_not_abort():
    def generate(messages, system_prompt, **gen):
        if messages[-1]["content"] == "boom":
            raise RuntimeError("kaboom")
        return "x"
    scores = []
    summary = run_eval(
        examples=[("ok", "x"), ("boom", "x"), ("ok", "x")],
        generate=generate, scorer=make_scorer("exact_match", {}),
        system_prompt="", template=None, gen_params={},
        should_cancel=lambda: False,
        on_result=lambda idx, res, *a: scores.append((idx, res.score, res.error)),
    )
    assert summary.num_scored == 3
    assert scores[1][1] == 0.0 and scores[1][2] is not None  # errored example -> 0 + error
    assert summary.status == "completed"
