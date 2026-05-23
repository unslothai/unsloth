# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the /api/scoring/score route (document scorer integration)."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from routes.scoring import router


def _client() -> TestClient:
    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    app.include_router(router, prefix="/api/scoring")
    return TestClient(app)


def test_score_perfect_document():
    body = {
        "ground_truth": {"total": 100, "currency": "USD"},
        "prediction": {"total": 100, "currency": "USD"},
        "schema": {"total": "money", "currency": "categorical"},
    }
    r = _client().post("/api/scoring/score", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["score"] == 1.0
    assert data["breakdown"]["children"]["total"]["score"] == 1.0


def test_score_partial_with_breakdown():
    body = {
        "ground_truth": {"total": 100, "currency": "USD"},
        "prediction": {"total": 90, "currency": "EUR"},
        "schema": {"total": "money", "currency": "categorical"},
    }
    r = _client().post("/api/scoring/score", json=body)
    assert r.status_code == 200
    data = r.json()
    # total -> 0.9, currency -> 0.0, mean over 2 leaves -> 0.45
    assert abs(data["score"] - 0.45) < 1e-9
    assert data["breakdown"]["children"]["currency"]["score"] == 0.0
    assert abs(data["breakdown"]["children"]["total"]["score"] - 0.9) < 1e-9


def test_score_schema_none_defaults_to_string():
    body = {"ground_truth": {"a": "x"}, "prediction": {"a": "x"}}
    r = _client().post("/api/scoring/score", json=body)
    assert r.status_code == 200
    assert r.json()["score"] == 1.0


def test_score_invalid_schema_returns_400():
    body = {
        "ground_truth": {"a": 1},
        "prediction": {"a": 1},
        "schema": "not_a_comparator",
    }
    r = _client().post("/api/scoring/score", json=body)
    assert r.status_code == 400


def test_score_without_breakdown():
    body = {
        "ground_truth": {"a": "x"},
        "prediction": {"a": "x"},
        "return_key_scores": False,
    }
    r = _client().post("/api/scoring/score", json=body)
    assert r.status_code == 200
    assert r.json()["breakdown"] is None
