# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import random

import pytest

from core.inference import voice_log as vl


@pytest.fixture(autouse = True)
def _log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_VOICE_LOG_DIR", str(tmp_path))
    return tmp_path


def _judge(text, p, outcome = None, thread_id = "t1"):
    row_id = vl.append("judge", {"text": text}, vl.judge_verdict(p), p, 12.0, thread_id = thread_id)
    if outcome:
        vl.set_outcome(row_id, outcome)
    return row_id


def test_append_writes_only_training_fields(tmp_path):
    row_id = vl.append("judge", {"text": "what's the"}, "hold", 0.2, 9.5, thread_id = "abc", source = "api")
    line = json.loads((tmp_path / "laya-log.jsonl").read_text().strip())
    assert line["id"] == row_id
    assert set(line) == {"id", "t", "model", "thread_id", "source", "input", "verdict", "confidence", "latency_ms", "outcome"}
    assert line["source"] == "api" and line["outcome"] is None and line["thread_id"] == "abc"


def test_outcome_merges_on_read_and_latest_wins():
    row_id = _judge("tell me a joke", 0.9)
    vl.set_outcome(row_id, "sent")
    vl.set_outcome(row_id, "continued_quickly", {"gap_ms": 800})
    (row,) = vl.read_rows()
    assert row["outcome"] == "continued_quickly"
    assert row["outcome_detail"] == {"gap_ms": 800}
    assert vl.error_kind(row) == "finished_but_continued"


def test_agreement_rules():
    assert vl.error_kind({"model": "judge", "verdict": "hold", "outcome": "flushed"}) == "held_but_flushed"
    assert vl.error_kind({"model": "judge", "verdict": "hold", "outcome": "continued"}) is None
    assert vl.error_kind({"model": "interject", "verdict": "ignore", "outcome": "repeated"}) == "ignored_but_repeated"
    assert vl.error_kind({"model": "interject", "verdict": "queue", "outcome": "reasked"}) == "queued_but_reasked"
    assert vl.error_kind({"model": "interject", "verdict": "restart", "outcome": "ok"}) is None
    assert vl.error_kind({"model": "filler", "verdict": "say", "outcome": "echoed"}) == "echoed"
    assert vl.error_kind({"model": "filler", "verdict": "say", "outcome": "too_late"}) == "too_late"
    assert vl.error_kind({"model": "filler", "verdict": "quiet", "outcome": "echoed"}) is None
    assert vl.error_kind({"model": "addressed", "verdict": "other", "outcome": "repeated"}) == "missed_address"
    assert vl.error_kind({"model": "addressed", "verdict": "reading", "outcome": "repeated"}) == "missed_address"
    assert vl.error_kind({"model": "addressed", "verdict": "assistant", "outcome": "cut"}) is None
    assert vl.error_kind({"model": "addressed", "verdict": "other", "outcome": "ok"}) is None
    assert vl.error_kind({"model": "judge", "verdict": "hold", "outcome": None}) is None


def test_stats_counts_accuracy_errors_and_latency():
    _judge("one", 0.9, "sent")
    _judge("two", 0.9, "continued_quickly")
    _judge("three", 0.1, "flushed")
    _judge("four", 0.1, "continued")
    _judge("five", 0.5)  # no outcome yet: counted as a verdict, not in accuracy
    fid = vl.append("filler", {"text": "fix my code", "class": "fix"}, "say", 0.8, 3.0)
    vl.set_outcome(fid, "echoed")
    s = vl.stats()
    judge = s["models"]["judge"]
    assert judge["verdicts"] == 5 and judge["with_outcome"] == 4
    assert judge["accuracy"] == 0.5
    assert judge["errors"] == {"finished_but_continued": 1, "held_but_flushed": 1}
    assert judge["latency_p50_ms"] == 12.0 and judge["latency_p95_ms"] == 12.0
    assert s["models"]["filler"]["errors"] == {"echoed": 1, "too_late": 0}
    assert s["models"]["interject"]["accuracy"] is None
    assert s["models"]["addressed"] == {"verdicts": 0, "with_outcome": 0, "accuracy": None, "errors": {"missed_address": 0},
                                        "latency_p50_ms": None, "latency_p95_ms": None}
    assert s["rows"] == 6 and s["log_bytes"] > 0


def test_export_defaults_to_disagreements_in_training_shape():
    _judge("so what I wanted", 0.9, "continued_quickly")
    _judge("tell me a joke", 0.9, "sent")
    _judge("can you", 0.2, "flushed", thread_id = "t2")
    rows = vl.export_rows()
    assert [(r["text"], r["label"]) for r in rows] == [("so what I wanted", "incomplete"), ("can you", "complete")]
    assert rows[0]["model"] == "judge" and rows[0]["verdict"] == "finished" and rows[0]["outcome"] == "continued_quickly"
    assert rows[0]["thread_id"] == "t1" and "t" in rows[0] and rows[0]["confidence"] == 0.9
    assert [r["text"] for r in vl.export_rows(thread_id = "t2")] == ["can you"]
    assert len(vl.export_rows(only = "all")) == 3
    assert vl.export_rows(model = "filler") == []


def test_export_mix_adds_one_agreeing_row_per_disagreement():
    _judge("a", 0.9, "continued_quickly")
    for text in ("b", "c", "d"):
        _judge(text, 0.9, "sent")
    rows = vl.export_rows(mix = True, rng = random.Random(1))
    assert len(rows) == 2
    assert sorted(r["label"] for r in rows) == ["complete", "incomplete"]


def test_export_since_and_interject_and_filler_labels():
    iid = vl.append("interject", {"prompt": "write a poem", "utterance": "make it short"}, "ignore", 0.7, 5.0)
    vl.set_outcome(iid, "repeated")
    fid = vl.append("filler", {"text": "fix my code", "class": "fix"}, "say", 0.8, 3.0)
    vl.set_outcome(fid, "spoken")
    rows = vl.export_rows(only = "all")
    assert rows[0]["text"] == "make it short" and rows[0]["label"] == "restart" and rows[0]["prompt"] == "write a poem"
    assert rows[1]["label"] == "fix"
    aid = vl.append("addressed", {"text": "grandma say hi to it", "prev": "assistant"}, "other", 0.9, 2.0)
    vl.set_outcome(aid, "repeated")
    bid = vl.append("addressed", {"text": "what time is it", "prev": "none"}, "assistant", 0.8, 2.0)
    vl.set_outcome(bid, "cut")
    rows = vl.export_rows(only = "all")
    assert rows[2]["label"] == "assistant" and rows[2]["verdict"] == "other" and rows[2]["prev"] == "assistant"
    assert rows[3]["label"] == "assistant" and rows[3]["outcome"] == "cut"
    assert [r["label"] for r in vl.export_rows()] == ["restart", "assistant"]
    assert vl.export_rows(only = "all", since = "2999-01-01") == []
    assert "\n" in vl.to_jsonl(rows) and json.loads(vl.to_jsonl(rows).splitlines()[0])["model"] == "interject"


def test_clear_and_thread_delete(tmp_path):
    a = _judge("a", 0.9, "sent", thread_id = "keep")
    _judge("b", 0.9, "sent", thread_id = "gone")
    assert vl.delete_thread_rows(["gone", None]) == 1
    rows = vl.read_rows()
    assert [r["id"] for r in rows] == [a] and rows[0]["outcome"] == "sent"
    assert len((tmp_path / "laya-outcomes.jsonl").read_text().splitlines()) == 1
    assert vl.delete_thread_rows(["nobody"]) == 0
    assert vl.clear() == 1
    assert vl.read_rows() == [] and not (tmp_path / "laya-log.jsonl").exists()
    assert vl.stats()["rows"] == 0
