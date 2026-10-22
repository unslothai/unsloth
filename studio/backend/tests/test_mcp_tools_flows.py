# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""list_flows, run_flow, get_job(kind="flow") and cancel(kind="flow") over a fake Studio."""

from .mcp_harness import bodies, run_tool


def _calls(studio) -> list[tuple[str, str]]:
    return [(c[0], c[1]) for c in studio.state.calls]


SNAPSHOT = {
    "run": {
        "id": "run-1",
        "flow_id": "f1",
        "status": "completed",
        "error": None,
        "outputs": {"story": "Once upon a time"},
    },
    "nodes": {"a": {"status": "done"}, "b": {"status": "cached"}, "c": {"status": "skipped"}},
    "last_seq": 9,
}
PAYLOADS = {
    ("GET", "/api/flows/"): {
        "flows": [{"id": "f1", "name": "Comic", "kind": "pipeline", "nodeCount": 5}]
    },
    ("GET", "/api/flows/examples"): {
        "examples": [{"id": "comic", "name": "Comic Strip Machine", "kind": "pipeline"}]
    },
    ("POST", "/api/flows/"): {"id": "f2"},
    ("POST", "/api/flows/examples/{example_id}/copy"): {"id": "f3"},
    ("POST", "/api/flows/{flow_id}/runs"): {"run_id": "run-1"},
    ("GET", "/api/flows/runs/{run_id}"): SNAPSHOT,
    ("POST", "/api/flows/runs/{run_id}/cancel"): {"status": "cancelling"},
}


def test_list_flows_lists_saved_and_examples(monkeypatch):
    result, _ = run_tool(monkeypatch, PAYLOADS, "list_flows", {})
    out = result["structuredContent"]
    assert out["flows"][0]["id"] == "f1" and out["flows"][0]["node_count"] == 5
    assert out["examples"][0]["example"] is True


def test_run_flow_by_id_returns_outputs(monkeypatch):
    result, studio = run_tool(
        monkeypatch, PAYLOADS, "run_flow", {"flow_id": "f1", "params": {"premise": "a moon"}}
    )
    out = result["structuredContent"]
    assert out["kind"] == "flow" and out["status"] == "completed"
    assert out["flow"]["outputs"] == {"story": "Once upon a time"}
    assert out["progress_percent"] == 100.0
    assert bodies(studio, "/api/flows/f1/runs") == [{"params": {"premise": "a moon"}}]


def test_run_flow_from_document_or_example(monkeypatch):
    result, studio = run_tool(
        monkeypatch, PAYLOADS, "run_flow", {"document": {"format": "unsloth-flow"}}
    )
    assert [p for m, p in _calls(studio) if m == "POST"][:2] == [
        "/api/flows/",
        "/api/flows/f2/runs",
    ]
    result, studio = run_tool(monkeypatch, PAYLOADS, "run_flow", {"example_id": "comic"})
    assert ("POST", "/api/flows/f3/runs") in _calls(studio)


def test_run_flow_needs_exactly_one_source(monkeypatch):
    result, _ = run_tool(monkeypatch, PAYLOADS, "run_flow", {})
    assert result.get("isError")
    result, _ = run_tool(monkeypatch, PAYLOADS, "run_flow", {"flow_id": "f1", "example_id": "x"})
    assert result.get("isError")


def test_get_job_and_cancel_flow(monkeypatch):
    result, _ = run_tool(monkeypatch, PAYLOADS, "get_job", {"kind": "flow", "id": "run-1"})
    out = result["structuredContent"]
    assert out["flow"]["nodes_done"] == 3 and out["flow"]["nodes_total"] == 3
    result, _ = run_tool(monkeypatch, PAYLOADS, "get_job", {"kind": "flow"})
    assert result.get("isError")
    result, studio = run_tool(monkeypatch, PAYLOADS, "cancel", {"kind": "flow", "id": "run-1"})
    assert result["structuredContent"]["cancelled"] is True
    assert ("POST", "/api/flows/runs/run-1/cancel") in _calls(studio)
