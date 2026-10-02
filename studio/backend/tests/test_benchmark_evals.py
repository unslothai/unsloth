# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from core.benchmark import orchestrator as orch
from core.benchmark.parse import extract_samples, parse_run_summary, pick_default_metric
from core.benchmark.resolve import resolve_model_details
from routes import benchmark as benchmark_routes
from utils.account_context import AccountContext, run_as


# Shapes as lm_eval 0.4.x writes them: "<metric>,<filter>" plus "<metric>_stderr,<filter>".
ARC_RESULTS = {
    "results": {
        "arc_easy": {
            "alias": "arc_easy",
            "acc,none": 0.62,
            "acc_stderr,none": 0.0099,
            "acc_norm,none": 0.58,
            "acc_norm_stderr,none": 0.0101,
        }
    },
    "n-shot": {"arc_easy": 0},
    "n-samples": {"arc_easy": {"original": 2376, "effective": 2}},
    "samples": {
        "arc_easy": [
            {
                "doc_id": 0,
                "doc": {"question": "Which is a liquid?", "choices": {"text": ["ice", "water"]}},
                "target": 1,
                "resps": [[[-2.5, False]], [[-0.3, True]]],
                "filtered_resps": [[-2.5, False], [-0.3, True]],
                "acc": 1.0,
                "acc_norm": 1.0,
            }
        ]
    },
}

GSM8K_RESULTS = {
    "results": {
        "gsm8k": {
            "alias": "gsm8k",
            "exact_match,strict-match": 0.25,
            "exact_match_stderr,strict-match": 0.04,
            "exact_match,flexible-extract": 0.3,
            "exact_match_stderr,flexible-extract": 0.05,
        }
    },
    "samples": {
        "gsm8k": [
            {
                "doc_id": 3,
                "doc": {"question": "2+2?"},
                "target": "#### 4",
                "resps": [["The answer is 4"]],
                "filtered_resps": ["4"],
                "exact_match,strict-match": 1.0,
            }
        ]
    },
}


def test_loglikelihood_metrics_keep_filter_suffix_and_stderr():
    summary = parse_run_summary("benchmark_arc_easy_20261002_120000", ARC_RESULTS)
    by_name = {m["name"]: m for m in summary["metrics"]}
    assert set(by_name) == {"acc,none", "acc_norm,none"}
    assert by_name["acc,none"]["stderr"] == "0.0099"
    assert pick_default_metric(summary["metrics"]) == "acc_norm,none"


def test_generation_stderr_rows_are_not_metrics():
    summary = parse_run_summary("benchmark_gsm8k_20261002_120000", GSM8K_RESULTS)
    names = [m["name"] for m in summary["metrics"]]
    assert names == ["exact_match,strict-match", "exact_match,flexible-extract"]
    assert summary["metrics"][0]["stderr"] == "0.04"
    assert pick_default_metric(summary["metrics"]) == "exact_match,strict-match"


@pytest.mark.parametrize("results,task", [(ARC_RESULTS, "arc_easy"), (GSM8K_RESULTS, "gsm8k")])
def test_samples_fit_the_text_columns(results, task):
    samples = extract_samples(results, task)
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE eval_samples (run_id TEXT, doc_id INTEGER, question TEXT, target TEXT,"
        " response TEXT, raw_response TEXT, correct INTEGER)"
    )
    conn.executemany(
        "INSERT INTO eval_samples VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "r",
                s["doc_id"],
                s["question"],
                s["target"],
                s["response"],
                s["raw_response"],
                int(s["correct"]),
            )
            for s in samples
        ],
    )
    assert conn.execute("SELECT COUNT(*) FROM eval_samples").fetchone()[0] == 1
    if task == "arc_easy":
        assert json.loads(samples[0]["response"]) == [[-2.5, False], [-0.3, True]]
    else:
        assert samples[0]["response"] == "4"


def test_run_targets_the_bound_studio_address():
    _, kwargs = resolve_model_details(
        "model.gguf", task = "arc_easy", server_url = "http://127.0.0.1:18921"
    )
    assert kwargs["model"] == "unsloth-studio-gguf"
    assert kwargs["model_args"]["base_url"] == "http://127.0.0.1:18921"

    req = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace(server_port = 18921)))
    assert benchmark_routes._studio_base_url(req) == "http://127.0.0.1:18921"
    req.app.state.server_request_host = "::1"
    assert benchmark_routes._studio_base_url(req) == "http://[::1]:18921"
    req.app.state.server_port = None
    assert benchmark_routes._studio_base_url(req) is None


def _orchestrator(monkeypatch, task):
    monkeypatch.setattr(orch, "_run_benchmark_task", task)
    return orch.BenchmarkOrchestrator()


def test_successful_run_stays_active_until_finished(monkeypatch):
    backend = _orchestrator(monkeypatch, lambda params: {"results": {"t": {}}})
    seq = backend.get_op_seq() + 1
    assert backend.run({}) == {"results": {"t": {}}}
    assert backend.is_active() and backend.get_last_op_status() is None
    backend.finish(op_seq = seq + 1)  # stale: another run's sequence
    assert backend.is_active()
    backend.finish(op_seq = seq)
    assert not backend.is_active()
    assert (backend.get_last_op_status(), backend.get_op_seq()) == ("success", seq)


def test_failed_and_cancelled_runs_report_their_outcome(monkeypatch):
    def boom(params):
        raise ValueError("server said 401")

    backend = _orchestrator(monkeypatch, boom)
    with pytest.raises(RuntimeError, match = "401"):
        backend.run({})
    assert not backend.is_active()
    assert backend.get_last_op_status() == "error"

    release = threading.Event()
    backend = _orchestrator(monkeypatch, lambda params: release.wait(5) or {"results": {}})
    result = {}
    worker = threading.Thread(target = lambda: result.setdefault("r", backend.run({})))
    worker.start()
    while not backend.is_active():
        pass
    assert backend.cancel()
    worker.join(5)
    release.set()
    assert result["r"] == {}
    assert backend.was_cancelled() and backend.get_last_op_status() == "cancelled"
    backend.finish()
    assert backend.get_last_op_status() == "cancelled"


def test_another_accounts_run_is_hidden(monkeypatch):
    backend = SimpleNamespace(_result_account = AccountContext("alice-id", "alice"))
    bob = AccountContext("bob-id", "bob")
    monkeypatch.setattr(benchmark_routes, "_has_managed_accounts", lambda: True)
    assert benchmark_routes._run_hidden_from(backend, bob)
    assert not benchmark_routes._run_hidden_from(backend, backend._result_account)
    monkeypatch.setattr(benchmark_routes, "_has_managed_accounts", lambda: False)
    assert not benchmark_routes._run_hidden_from(backend, bob)


def test_run_records_the_starting_account(monkeypatch):
    backend = _orchestrator(monkeypatch, lambda params: {"results": {}})
    alice = AccountContext("alice-id", "alice")
    run_as(alice, backend.run, {})
    assert backend._result_account == alice


def test_studio_gguf_backend_authenticates_and_uses_v1_tokenize():
    pytest.importorskip("lm_eval.models.gguf")
    from core.benchmark.gguf_client import StudioGGUFLM

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, body):
            seen.append((self.command, self.path.split("?")[0], self.headers.get("Authorization")))
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._reply({"total_slots": 4})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self._reply({"tokens": [1, 2, 3]})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    try:
        lm = StudioGGUFLM(
            base_url = f"http://127.0.0.1:{server.server_port}", model = "m.gguf", api_key = "sk-test"
        )
        assert lm._tokenize("hello", False) == [1, 2, 3]
        assert lm._detect_total_slots() == 4
    finally:
        server.shutdown()
    assert ("POST", "/v1/tokenize", "Bearer sk-test") in seen
    assert ("GET", "/props", "Bearer sk-test") in seen


def test_run_from_an_executor_records_the_requesting_account(monkeypatch):
    # The route runs backend.run in an executor thread, which does not inherit
    # the request's account context; it passes the account explicitly.
    import asyncio

    from utils.account_context import arun_as

    backend = _orchestrator(monkeypatch, lambda params: {"results": {}})
    alice = AccountContext("alice-id", "alice")

    async def route_like():
        from utils.account_context import current_account

        account = current_account()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: backend.run({}, account))

    asyncio.run(arun_as(alice, route_like()))
    assert backend._result_account == alice
