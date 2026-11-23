# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise the smoke driver against a cold HTTP contract, without claiming MLX coverage."""

import importlib.util
import json
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest


@pytest.mark.parametrize("scenario", ["real-mlx", "no-mlx-settles"])
def test_mac_smoke_requests_detection_before_waiting_for_a_verdict(monkeypatch, tmp_path, scenario):
    spec = importlib.util.spec_from_file_location(
        "_mac_verdict_demand_smoke", Path(__file__).with_name("mac_capability_verdict_smoke.py")
    )
    smoke = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, smoke)
    spec.loader.exec_module(smoke)
    monkeypatch.setitem(
        sys.modules, "utils.mlx_repair", SimpleNamespace(mlx_stack_available = lambda: True)
    )
    monkeypatch.setattr(smoke, "POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr(smoke, "SETTLE_BUDGET_S", 0.5)
    monkeypatch.setattr(smoke, "STABLE_HOLD_S", 0.05)
    monkeypatch.setattr(smoke, "wait_for_health", lambda *args: True)
    monkeypatch.setattr(smoke, "tail_log", lambda *args: None)

    requested = threading.Event()
    requests = []
    chat_only = scenario == "no-mlx-settles"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path)
            auth = self.headers.get("Authorization") == "Bearer smoke-test"
            demand = parse_qs(path.query).get("detect_hardware") == ["true"]
            requests.append((path.path, demand, auth))
            status = 200
            if path.path == "/api/system":
                body = {"device_backend": "mlx"}
            elif path.path == "/api/health":
                if demand and auth:
                    requested.set()
                elif demand:
                    status = 401
                body = {"status": "healthy", "chat_only": chat_only if requested.is_set() else True}
                if auth:
                    body.update(version = "test", device_type = "mac" if requested.is_set() else None)
                    body["chat_only_reason"] = (
                        "mlx_unavailable" if requested.is_set() and chat_only else None
                    )
                if not requested.is_set():
                    body.update(hardware_detecting = True, hardware_detection_deferred = True)
            else:
                status, body = 404, {}
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target = server.serve_forever, daemon = True)
    server_thread.start()
    port = server.server_port
    poller = smoke.HealthPoller(f"http://127.0.0.1:{port}")
    poller.token = "smoke-test"

    @contextmanager
    def booted(*args):
        poller.start()
        try:
            yield poller, SimpleNamespace(error = None)
        finally:
            poller.stop()

    monkeypatch.setattr(smoke, "booted", booted)
    original_path = list(sys.path)
    try:
        if scenario == "real-mlx":
            smoke.scenario_real_mlx(port, tmp_path / "unused.log")
        else:
            smoke.scenario_no_mlx_settles(port, tmp_path / "unused.log", tmp_path)
        assert not smoke._failed, smoke._failed
        assert requested.is_set(), "ordinary health must be followed by an explicit feature demand"
        assert requests[0] == ("/api/health", False, True)
        assert [r for r in requests if r[1]] == [("/api/health", True, True)]
        assert any(not sample.published for sample in poller.snapshot())
        assert any(sample.published and sample.authed for sample in poller.snapshot())
    finally:
        poller.stop()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout = 2)
        sys.path[:] = original_path
