# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import math

import pytest

from core.flows import memory as mem
from core.flows.nodes import memory as nodes
from core.flows.registry import FlowNodeError
from core.flows.specs import SPECS_BY_TYPE

VOCAB = ("knee", "run", "hills", "oatmeal", "breakfast", "sister", "visit", "party", "cafe")


def bag(text: str) -> list[float]:
    words = mem.tokens(text)
    return [1.0 if any(w.startswith(v) for w in words) else 0.0 for v in VOCAB] + [0.01]


class Ctx:
    def __init__(
        self,
        block,
        inputs,
        embeddings = True,
        **params,
    ):
        self.params = {p["id"]: p["default"] for p in SPECS_BY_TYPE[block]["params"]}
        self.params.update(params)
        self.inputs, self.seed = inputs, 7
        self.embeddings, self.calls, self.logs, self.prompts = embeddings, [], [], []
        self.answers = []

    async def call(
        self,
        method,
        path,
        json = None,
        **_,
    ):
        self.calls.append(path)
        if not self.embeddings:
            raise FlowNodeError("No embedding model is loaded", status = 400)
        return {"data": [{"index": i, "embedding": bag(t)} for i, t in enumerate(json["input"])]}

    async def llm_json(self, *, prompt, schema, **_):
        self.prompts.append(prompt)
        return self.answers.pop(0)

    async def llm_text(self, *, prompt, **_):
        self.prompts.append(prompt)
        return "A runner."

    def log(
        self,
        message,
        level = "info",
    ):
        self.logs.append((level, message))


def run(coro):
    return asyncio.run(coro)


def stream_of(*rows):
    s = mem.empty_stream()
    for text, importance, t in rows:
        s["records"].append(mem.new_record(s, text, "observation", importance, t))
    return s


def test_normalise_is_min_max_and_flat_terms_count_for_nothing():
    assert mem.normalise([2.0, 4.0, 3.0]) == [0.0, 1.0, 0.5]
    assert mem.normalise([5.0, 5.0]) == [0.0, 0.0]


def test_recency_decays_per_hour_since_last_access_or_by_rank():
    records = [{"t_access": 10.0}, {"t_access": 0.0}]
    hours = mem.recency(records, now = 10.0, decay = 0.995, mode = "hours")
    assert hours[0] == 1.0 and math.isclose(hours[1], 0.995**10)
    assert mem.recency(records, now = 10.0, decay = 0.99, mode = "rank") == [1.0, 0.99]


def test_score_sums_weighted_normalised_terms():
    s = stream_of(("ran hills, knee hurts", 8, 0.0), ("ate oatmeal for breakfast", 2, 9.0))
    scores = mem.score(s["records"], now = 10.0, query = "knee pain after hills", weights = {})
    # First record: oldest (recency 0), most important (1) and the only relevant one (1): 0 + 2 + 3.
    assert scores[0] == {"recency": 0.0, "importance": 1.0, "relevance": 1.0, "score": 5.0}
    assert scores[1]["score"] == 0.5
    assert mem.top(s["records"], scores, 1) == [0]


def test_paper_weights_are_a_param_away():
    s = stream_of(("knee", 1, 0.0), ("oatmeal", 10, 10.0))
    equal = mem.score(
        s["records"],
        now = 10.0,
        query = "knee",
        weights = {"recency": 1, "importance": 1, "relevance": 1},
    )
    assert [round(x["score"], 3) for x in equal] == [1.0, 2.0]


def test_keyword_relevance_ignores_stop_words():
    assert mem.keyword_relevance("what about the knee", "my knee hurts") == 1.0
    assert mem.keyword_relevance("the a of", "anything") == 0.0


def test_cap_keeps_cited_records_and_drops_the_least_important():
    s = stream_of(("a", 1, 0.0), ("b", 9, 0.0), ("c", 2, 1.0))
    s["records"].append(mem.new_record(s, "insight", "reflection", 5, 2.0, cites = ["m0"]))
    mem.cap(s, 3)
    assert [r["id"] for r in s["records"]] == ["m0", "m1", "m3"]


def test_as_stream_accepts_a_record_list_and_continues_ids():
    s = mem.as_stream([{"id": "m4", "text": "x"}])
    assert s["next_id"] == 5 and s["kind"] == mem.STREAM_KIND
    with pytest.raises(ValueError):
        mem.as_stream("not a stream")


def test_remember_rates_importance_in_one_call_and_embeds_in_one_call():
    ctx = Ctx("memory.add", {"items": ["ran hills", "ate oatmeal"], "now": 5.0})
    ctx.answers = [{"ratings": [7]}]  # one rating short: the missing one defaults to 5
    out = run(nodes.add(ctx))
    added = out["added"]
    assert [r["importance"] for r in added] == [7, 5]
    assert len(ctx.prompts) == 1 and ctx.calls == ["/v1/embeddings"]
    assert all(isinstance(r["embedding"], list) for r in added)
    assert out["memory"]["since_reflection"] == 12.0 and out["memory"]["next_id"] == 2


def test_remember_with_fixed_importance_makes_no_model_call():
    ctx = Ctx(
        "memory.add",
        {"items": "met Ana at the cafe", "now": 1.0},
        importance = "fixed",
        fixed_importance = 3,
    )
    out = run(nodes.add(ctx))
    assert out["added"][0]["importance"] == 3 and ctx.prompts == []


def test_recall_uses_embeddings_and_marks_records_used():
    s = stream_of(("ran hills, knee hurts", 5, 0.0), ("ate oatmeal for breakfast", 5, 0.0))
    ctx = Ctx("memory.retrieve", {"memory": s, "query": "is my knee ok", "now": 3.0}, top_k = 1)
    out = run(nodes.retrieve(ctx))
    assert [r["text"] for r in out["records"]] == ["ran hills, knee hurts"]
    assert out["context"] == "1. ran hills, knee hurts"
    assert ctx.calls == ["/v1/embeddings"]  # the query and both missing record vectors in one call
    by_id = {r["id"]: r for r in out["memory"]["records"]}
    assert by_id["m0"]["t_access"] == 3.0 and by_id["m1"]["t_access"] == 0.0
    assert "embedding" not in out["records"][0]


def test_recall_falls_back_to_keywords_without_an_embedding_model():
    s = stream_of(("ran hills, knee hurts", 5, 0.0), ("ate oatmeal for breakfast", 5, 0.0))
    ctx = Ctx(
        "memory.retrieve", {"memory": s, "query": "oatmeal", "now": 3.0}, embeddings = False, top_k = 1
    )
    out = run(nodes.retrieve(ctx))
    assert out["records"][0]["text"] == "ate oatmeal for breakfast"
    assert ctx.logs and ctx.logs[0][0] == "warning"


def test_reflect_waits_for_the_threshold():
    s = stream_of(("ran hills", 5, 0.0))
    s["since_reflection"] = 20.0
    ctx = Ctx("memory.reflect", {"memory": s, "now": 1.0})
    out = run(nodes.reflect(ctx))
    assert (
        out["reflected"] is False and ctx.prompts == [] and out["memory"]["records"] == s["records"]
    )


def test_reflect_uses_four_model_calls_and_cites_its_evidence():
    s = stream_of(
        ("ran hills, knee hurts", 8, 0.0),
        ("knee still sore after the run", 7, 1.0),
        ("ate oatmeal", 2, 2.0),
    )
    s["since_reflection"] = 200.0
    ctx = Ctx("memory.reflect", {"memory": s, "now": 3.0, "name": "Maya"})
    ctx.answers = [
        {"questions": ["How is Maya's knee?", "What does Maya eat?", "Is Maya training?"]},
        {
            "insights": [
                {
                    "text": "Maya's knee is injured from running",
                    "cites": [1, 2, 99],
                    "importance": 8,
                }
            ]
        },
        {"insights": []},
        {"insights": [{"text": "Maya keeps training despite pain", "cites": [1], "importance": 6}]},
    ]
    out = run(nodes.reflect(ctx))
    assert len(ctx.prompts) == 4 and out["reflected"] is True
    insights = [r for r in out["memory"]["records"] if r["kind"] == "reflection"]
    assert [r["text"] for r in insights] == out["insights"]
    assert len(insights[0]["cites"]) == 2  # citation 99 is out of range and dropped
    assert set(insights[0]["cites"]) <= {"m0", "m1", "m2"}
    assert out["memory"]["since_reflection"] == 0.0
    assert all(isinstance(r.get("embedding"), list) for r in insights)


def test_summary_reads_three_recalls_with_one_model_call():
    s = stream_of(("Maya runs every morning", 6, 0.0))
    ctx = Ctx("agent.summary", {"memory": s, "name": "Maya", "now": 1.0})
    out = run(nodes.summary(ctx))
    assert out["summary"] == "A runner." and len(ctx.prompts) == 1
    assert "Maya runs every morning" in ctx.prompts[0]
    assert ctx.calls == ["/v1/embeddings"]


def test_plan_keeps_known_places_in_time_order():
    ctx = Ctx("agent.plan", {"summary": "Maya, a runner.", "places": ["park", "cafe"]}, name = "Maya")
    ctx.answers = [
        {
            "plan": [
                {"start": "12:00", "minutes": 60, "activity": "lunch", "place": "cafe"},
                {"start": "07:30", "minutes": 45, "activity": "easy run", "place": "park"},
                {"start": "15:00", "minutes": 60, "activity": "swim", "place": "pool"},
                {"start": "25:00", "minutes": 60, "activity": "sleep", "place": "park"},
            ]
        }
    ]
    out = run(nodes.plan(ctx))
    assert [e["activity"] for e in out["plan"]] == ["easy run", "lunch"]
    assert out["text"].splitlines()[0] == "07:30 (45 min) easy run at park"


def test_plan_schema_limits_places_to_the_list():
    schema = nodes._plan_schema(["park"], 4)
    item = schema["properties"]["plan"]["items"]
    assert item["properties"]["place"]["enum"] == ["park"]
    assert schema["properties"]["plan"]["maxItems"] == 4
