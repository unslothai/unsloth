# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The recipe worker defaults Data Designer usage telemetry off and keeps an explicit opt-in."""

import json
import multiprocessing as mp
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("data_designer")


def _serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    return server


@pytest.fixture(name = "endpoints")
def fixture_endpoints():
    posts: list[bytes] = []

    class Llm(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            reply = json.dumps(
                {
                    "id": "c",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "x",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    class Telemetry(BaseHTTPRequestHandler):
        def do_POST(self):
            posts.append(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    llm, telemetry = _serve(Llm), _serve(Telemetry)
    yield (
        f"http://127.0.0.1:{llm.server_address[1]}/v1",
        f"http://127.0.0.1:{telemetry.server_address[1]}/v1.1/events/json",
        posts,
    )
    llm.shutdown()
    telemetry.shutdown()


def _run_preview(tmp_path: Path, llm_endpoint: str) -> dict:
    from core.data_recipe.jobs.worker import run_job_process

    seed = tmp_path / "seed.csv"
    seed.write_text("question\nWho?\nWhen?\n", encoding = "utf-8")
    recipe = {
        "model_providers": [{"name": "p", "endpoint": llm_endpoint, "provider_type": "openai"}],
        "model_configs": [
            {"alias": "m", "model": "x", "provider": "p", "inference_parameters": {}}
        ],
        "seed_config": {
            "source": {"seed_type": "local", "path": str(seed)},
            "sampling_strategy": "ordered",
        },
        "columns": [
            {
                "column_type": "llm-text",
                "name": "answer",
                "model_alias": "m",
                "prompt": "{{ question }}",
            }
        ],
    }
    run = {"rows": 2, "_job_id": "telemetry", "execution_type": "preview"}
    # Spawn like jobs/manager.py: data_designer reads the switch once, at import, in the child.
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target = run_job_process,
        kwargs = {"event_queue": queue, "recipe": recipe, "run": run},
    )
    proc.start()
    try:
        while True:
            event = queue.get(timeout = 120)
            if event.get("type") in ("job.completed", "job.error"):
                return event
    finally:
        proc.join(60)
        if proc.is_alive():
            proc.kill()


def test_worker_defaults_data_designer_telemetry_off(tmp_path, monkeypatch, endpoints) -> None:
    llm_endpoint, telemetry_endpoint, posts = endpoints
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NEMO_TELEMETRY_ENDPOINT", telemetry_endpoint)
    monkeypatch.delenv("NEMO_TELEMETRY_ENABLED", raising = False)

    event = _run_preview(tmp_path, llm_endpoint)

    assert event["type"] == "job.completed", event.get("error")
    assert [row["answer"] for row in event["dataset"]] == ["ok", "ok"]
    assert posts == []


@pytest.mark.parametrize("value", ["true", "1"])
def test_worker_keeps_explicit_telemetry_opt_in(tmp_path, monkeypatch, endpoints, value) -> None:
    llm_endpoint, telemetry_endpoint, posts = endpoints
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NEMO_TELEMETRY_ENDPOINT", telemetry_endpoint)
    monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", value)

    event = _run_preview(tmp_path, llm_endpoint)

    assert event["type"] == "job.completed", event.get("error")
    assert len(posts) == 1


def test_worker_sets_the_default_before_any_data_designer_import() -> None:
    import ast
    import inspect

    from core.data_recipe.jobs import worker

    body = ast.parse(inspect.getsource(worker.run_job_process)).body[0].body
    calls = [
        index
        for index, node in enumerate(body)
        if isinstance(node, ast.Expr)
        and "NEMO_TELEMETRY_ENABLED" in ast.unparse(node)
        and "setdefault" in ast.unparse(node)
    ]
    assert calls, "run_job_process must setdefault NEMO_TELEMETRY_ENABLED"
    before = body[: calls[0]]
    assert not any("data_designer" in ast.unparse(node) for node in before)
    assert not any(
        isinstance(node, (ast.Expr, ast.Assign))
        and isinstance(getattr(node, "value", None), ast.Call)
        for node in before
    )
