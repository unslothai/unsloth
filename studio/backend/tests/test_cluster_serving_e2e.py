# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A real N-worker split on loopback: ggml-rpc-servers, the planner's own --list-devices probe,
and a llama-server that loads across all of them and answers. Needs a llama.cpp build with RPC
and a small GGUF, so it only runs when UNSLOTH_TEST_RPC_BIN and UNSLOTH_TEST_GGUF are set."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import cluster_serving as cs
from core.inference import spark_serving as ss

BIN = os.environ.get("UNSLOTH_TEST_RPC_BIN")
GGUF = os.environ.get("UNSLOTH_TEST_GGUF")
pytestmark = pytest.mark.skipif(
    not (BIN and GGUF) or os.name == "nt", reason = "needs a llama.cpp RPC build and a GGUF"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(url: str, timeout: float) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout = 2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def test_a_three_worker_split_loads_on_every_worker_and_answers(monkeypatch, tmp_path):
    bin_dir = Path(BIN)
    procs, endpoints = [], []
    try:
        for i in range(3):
            port = _free_port()
            endpoints.append(f"127.0.0.1:{port}")
            procs.append(
                subprocess.Popen(
                    [
                        str(bin_dir / "ggml-rpc-server"),
                        "-H",
                        "127.0.0.1",
                        "-p",
                        str(port),
                        "-t",
                        "4",
                    ],
                    stdout = open(tmp_path / f"rpc{i}.log", "w", encoding = "utf-8"),
                    stderr = subprocess.STDOUT,
                    start_new_session = True,
                )
            )
        monkeypatch.delenv(ss.ENV_TOGGLE, raising = False)
        monkeypatch.setenv(cs.ENV_NODES, ",".join(endpoints))
        monkeypatch.setenv(cs.ENV_TOPOLOGY, "layer_split")
        monkeypatch.setattr(ss, "llama_server_binary", lambda: str(bin_dir / "llama-server"))
        cs.reset_for_tests()
        assert all(
            asyncio.run(ss.wait_for_port("127.0.0.1", int(e.split(":")[1]), 30)) for e in endpoints
        )
        request = SimpleNamespace(
            model_path = GGUF,
            gguf_variant = None,
            llama_extra_args = None,
            max_seq_length = 512,
            cache_type_kv = None,
            disable_vision = False,
        )
        out = asyncio.run(ss.before_load(request, 4))
        status = ss.status()
        assert status["topology"] == "layer_split", status
        port = _free_port()
        log = tmp_path / "llama-server.log"
        procs.append(
            subprocess.Popen(
                [str(bin_dir / "llama-server"), "-m", GGUF, "--port", str(port), "-c", "512"]
                + ["-ngl", "99", "-t", "4", "-lv", "4"]
                + out.llama_extra_args,
                stdout = open(log, "w", encoding = "utf-8"),
                stderr = subprocess.STDOUT,
                start_new_session = True,
            )
        )
        assert _wait(f"http://127.0.0.1:{port}/health", 300), log.read_text(encoding = "utf-8")[
            -2000:
        ]
        body = json.dumps({"prompt": "The capital of France is", "n_predict": 4}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/completion", body, {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout = 300) as r:
            assert json.loads(r.read())["content"].strip()
        text = log.read_text(encoding = "utf-8")
        for endpoint in endpoints:
            assert f"[{endpoint}] model buffer size" in text, endpoint
        # rpc-server takes one client at a time: held workers cannot answer a probe; the plan must hold.
        backend = SimpleNamespace(
            _process = SimpleNamespace(args = [str(bin_dir / "llama-server")] + out.llama_extra_args),
            is_loaded = True,
        )
        asyncio.run(ss.after_load(backend, 4))
        t0 = time.time()
        again = asyncio.run(
            ss.before_load(SimpleNamespace(**{**vars(request), "llama_extra_args": None}), 4)
        )
        assert ss.status()["topology"] == "layer_split", ss.status()
        assert again.llama_extra_args == out.llama_extra_args
        assert time.time() - t0 < 4 * cs.BUSY_PROBE_TIMEOUT_S
    finally:
        for p in reversed(procs):
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        cs.reset_for_tests()
