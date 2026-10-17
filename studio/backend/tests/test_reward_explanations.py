# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Check preview explanations against the reward callable used during training."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.training import rewards
from routes.rewards import router


def spec(body):
    return rewards.parse_reward_markdown(f"---\nname: example\nkind: rule\n---\n{body}")


@pytest.mark.parametrize(
    "body,text,reference,score,reason",
    [
        ("type: exact_match\nscore: {match: 2, miss: -1}", "72", "72", 2, "matched"),
        ("type: exact_match\nscore: {match: 2, miss: -1}", "73", "72", -1, "not_matched"),
        ("type: exact_match\nmissing: -2", "72", None, -2, "missing_reference"),
        (
            "type: exact_match\nextract: {between: ['<a>', '</a>']}\nmissing: -2",
            "72",
            "72",
            -2,
            "missing_response",
        ),
        (
            "type: regex\npattern: foo\nmode: search\nscore: {match: 1}",
            "a foo b",
            None,
            1,
            "matched",
        ),
        ("type: regex\npattern: foo\nscore: {miss: -1}", "a foo b", None, -1, "not_matched"),
        ("type: json_schema\nscore: {match: 1}", "```json\n{}\n```", None, 1, "matched"),
        ("type: json_schema\nscore: {miss: -1}", "{invalid}", None, -1, "invalid_json"),
        ("type: json_schema\nscore: {miss: -1}", "[]", None, -1, "wrong_json_type"),
        (
            "type: json_schema\nschema: {required: [a, b]}\nscore: {miss: -1}",
            '{"a": 1}',
            None,
            -1,
            "missing_json_keys",
        ),
        (
            "type: json_schema\nschema: {type: number}\nscore: {miss: -1}",
            "true",
            None,
            -1,
            "wrong_json_type",
        ),
        (
            "type: length\nmax_chars: 3\nscore: {under: 1, over: -1}",
            "א🙂a",
            None,
            1,
            "within_limit",
        ),
        ("type: length\nmax_chars: 3\nscore: {under: 1, over: -1}", "four", None, -1, "over_limit"),
        (
            "type: numeric\nbands: [{within: 0.1, score: 2}]\nelse: -1",
            "110",
            "100",
            2,
            "within_tolerance",
        ),
        (
            "type: numeric\nbands: [{within: 0.1, score: 2}]\nelse: -1",
            "111",
            "100",
            -1,
            "outside_tolerance",
        ),
        (
            "type: numeric\nbands: [{within: 0.1, score: 2}]\nmissing: -2",
            "nan",
            "100",
            -2,
            "invalid_response_number",
        ),
        (
            "type: numeric\nbands: [{within: 0.1, score: 2}]\nmissing: -2",
            "1",
            "inf",
            -2,
            "invalid_reference_number",
        ),
    ],
)
def test_explanation_preserves_training_scores(body, text, reference, score, reason):
    reward = spec(body)
    [preview] = rewards.preview_scores([reward], text, reference, explain = True)
    assert preview["score"] == score
    assert preview["explanation"]["reason"] == reason
    assert rewards.make_reward_func(reward)(completions = [text], answer = [reference]) == [score]
    assert rewards.preview_scores([reward], text, reference) == [
        {"name": "example", "score": score}
    ]


def test_exact_match_records_extraction_normalization_and_fallback():
    reward = spec("""type: exact_match
extract: {between: ['<answer>', '</answer>']}
reference_extract: {regex: '#### (.*)'}
normalize: [strip, remove_commas, lower]
score: {match: 2}
""")
    [result] = rewards.preview_scores(
        [reward], "<answer> 1,000 </answer>", "work #### 1000", explain = True
    )
    details = result["explanation"]
    assert details["response"] == " 1,000 "
    assert details["reference"] == "1000"
    assert details["normalized_response"] == details["normalized_reference"] == "1000"
    assert details["normalizers"] == ["strip", "remove_commas", "lower"]
    assert details["reference_fallback"] is False
    [fallback] = rewards.preview_scores([reward], "<answer>1000</answer>", "1000", explain = True)
    assert fallback["explanation"]["reference_fallback"] is True
    assert fallback["score"] == 2


def test_numeric_zero_reference_uses_absolute_error_and_first_matching_band():
    reward = spec("type: numeric\nbands: [{within: 1, score: 1}, {within: 0.1, score: 2}]")
    [result] = rewards.preview_scores([reward], "0.05", "0", explain = True)
    assert result["score"] == 2
    assert result["explanation"]["error_kind"] == "absolute"
    assert result["explanation"]["error"] == "0.05"
    assert result["explanation"]["tolerance"] == 0.1


def test_overflowing_numeric_error_is_json_serializable():
    reward = spec("type: numeric\nbands: [{within: 0.1, score: 1}]\nelse: -1")
    [result] = rewards.preview_scores([reward], "1e308", "1e-308", explain = True)
    assert result["score"] == -1
    assert result["explanation"]["reason"] == "outside_tolerance"
    json.dumps(result, allow_nan = False)


def test_json_explanation_identifies_missing_keys():
    reward = spec("type: json_schema\nschema: {required: [a, b]}")
    [result] = rewards.preview_scores([reward], '{"a": 1}', explain = True)
    assert result["explanation"]["missing_keys"] == ["b"]


def test_regex_timeout_is_not_reported_as_an_ordinary_mismatch(monkeypatch):
    reward = spec("type: regex\npattern: a+\nscore: {miss: -1}")

    class TimedOut:
        def fullmatch(self, *args, **kwargs):
            raise TimeoutError

    monkeypatch.setattr(rewards._rx, "compile", lambda *args, **kwargs: TimedOut())
    [result] = rewards.preview_scores([reward], "a", explain = True)
    assert result["score"] == -1
    assert result["explanation"]["reason"] == "regex_timeout"


def test_preview_api_explanations_are_optional_and_weights_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(rewards, "_user_root", lambda: tmp_path / "rewards")
    app = FastAPI()
    app.include_router(router, prefix = "/api/rewards")
    app.dependency_overrides[get_current_subject] = lambda: "test"
    client = TestClient(app)
    payload = {
        "rewards": [{"name": "exact-answer", "weight": -0.5}],
        "completion": "<answer>72</answer>",
        "reference": "72",
    }
    plain = client.post("/api/rewards/preview", json = payload).json()
    assert plain == {"scores": [{"name": "exact-answer", "score": 2, "weighted": -1}], "total": -1}
    response = client.post("/api/rewards/preview", json = {**payload, "explain": True})
    assert response.status_code == 200, response.text
    explained = response.json()
    assert explained["total"] == -1
    assert explained["scores"][0]["weighted"] == -1
    assert explained["scores"][0]["explanation"]["normalized_response"] == "72"
