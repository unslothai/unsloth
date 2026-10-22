# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from routes import benchmarks as benchmarks_routes
from routes import inference as inference_routes
from routes import llama_bench

# Prints what `llama-bench -o jsonl --progress` does: progress on stderr, one JSON row per test.
FAKE = r"""
import json, sys, time
rows = [(512, 0, 1450.5), (0, 128, 61.25)]
for i, (p, n, ts) in enumerate(rows, 1):
    print(f"llama-bench: benchmark {i}/{len(rows)}: starting", file=sys.stderr, flush=True)
    if "--slow" in sys.argv:
        time.sleep(30)
    print(json.dumps({
        "build_commit": "abc123", "build_number": 9000, "gpu_info": "Radeon AI PRO R9700",
        "backends": "ROCm", "model_type": "qwen3 4B Q4_K_M", "model_size": 2500000000,
        "model_n_params": 4000000000, "n_prompt": p, "n_gen": n, "n_depth": 0,
        "n_gpu_layers": 99, "flash_attn": 1, "avg_ts": ts, "stddev_ts": 1.5,
        "samples_ts": [ts - 1, ts + 1], "avg_ns": 1,
    }), flush=True)
"""


class _Backend:
    is_loaded = True
    gguf_path = "/models/qwen.gguf"
    model_identifier = "unsloth/Qwen3-4B-GGUF"
    hf_variant = "Q4_K_M"


@pytest.fixture
def bench(tmp_path, monkeypatch):
    script = tmp_path / "fake_llama_bench.py"
    script.write_text(FAKE, encoding = "utf-8")
    slow = {"on": False}
    unloads = []

    async def unload(request, subject):
        unloads.append(request.model_path)

    monkeypatch.setattr(llama_bench, "find_llama_bench", lambda: tmp_path / "llama-bench")
    monkeypatch.setattr(
        llama_bench,
        "_command",
        lambda binary, gguf, request: [sys.executable, str(script)]
        + (["--slow"] if slow["on"] else []),
    )
    monkeypatch.setattr(inference_routes, "get_llama_cpp_backend", lambda: _Backend())
    monkeypatch.setattr(inference_routes, "_unload_model_impl", unload)
    monkeypatch.setattr(
        "core.inference.llama_cpp.LlamaCppBackend._llama_server_env_for_binary",
        staticmethod(lambda binary, **_: None),
    )
    monkeypatch.setattr(llama_bench, "_job", None)

    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    app.include_router(benchmarks_routes.router, prefix = "/api/benchmarks")
    return TestClient(app), slow, unloads


def _wait(client, until = ("done", "error", "cancelled")):
    for _ in range(200):
        job = client.get("/api/benchmarks/llama-bench/run").json()["job"]
        if job["status"] in until:
            return job
        time.sleep(0.05)
    raise AssertionError(f"llama-bench job never finished: {job}")


def test_a_run_unloads_the_model_parses_rows_and_saves_as_its_own_kind(bench):
    client, _, unloads = bench
    res = client.post("/api/benchmarks/llama-bench/run", json = {"repetitions": 2})
    assert res.status_code == 200, res.text
    assert unloads == ["unsloth/Qwen3-4B-GGUF"]
    job = _wait(client)
    assert job["status"] == "done"
    assert [r["test"] for r in job["rows"]] == ["pp512", "tg128"]
    assert job["rows"][1]["avg_ts"] == 61.25
    assert job["meta"]["gpu_info"] == "Radeon AI PRO R9700"

    # Saved beside the sweeps, but the sweeps' list doesn't pick it up.
    assert client.get("/api/benchmarks/runs").json()["runs"] == []
    runs = client.get("/api/benchmarks/runs", params = {"kind": "llama-bench"}).json()["runs"]
    assert [r["id"] for r in runs] == [job["id"]]
    assert runs[0]["outcomes"][0]["test"] == "pp512"
    assert runs[0]["ggufVariant"] == "Q4_K_M"


def test_a_second_run_is_refused_while_one_is_going_and_cancel_stops_it(bench):
    client, slow, _ = bench
    slow["on"] = True
    assert client.post("/api/benchmarks/llama-bench/run", json = {}).status_code == 200
    again = client.post("/api/benchmarks/llama-bench/run", json = {})
    assert again.status_code == 409
    client.delete("/api/benchmarks/llama-bench/run")
    assert _wait(client)["status"] == "cancelled"


def test_missing_binary_and_no_model_are_409s_that_unload_nothing(bench, monkeypatch):
    client, _, unloads = bench
    monkeypatch.setattr(llama_bench, "find_llama_bench", lambda: None)
    res = client.post("/api/benchmarks/llama-bench/run", json = {})
    assert res.status_code == 409
    assert res.json()["detail"]["error"] == "llama_bench_missing"
    assert client.get("/api/benchmarks/llama-bench/status").json()["available"] is False

    monkeypatch.setattr(llama_bench, "find_llama_bench", lambda: Path("llama-bench"))
    monkeypatch.setattr(_Backend, "is_loaded", False)
    assert client.post("/api/benchmarks/llama-bench/run", json = {}).status_code == 409
    assert unloads == []


def test_nothing_to_measure_is_a_400(bench):
    client, _, unloads = bench
    res = client.post(
        "/api/benchmarks/llama-bench/run", json = {"prompt_tokens": [0], "gen_tokens": []}
    )
    assert res.status_code == 400
    assert unloads == []


def test_the_command_leaves_newer_and_default_flags_off():
    request = llama_bench.LlamaBenchRequest(prompt_tokens = [512, 512, 128], gen_tokens = [128])
    cmd = llama_bench._command(Path("llama-bench"), "m.gguf", request)
    assert cmd[cmd.index("-p") + 1] == "128,512"
    assert "-d" not in cmd and "-fa" not in cmd and "-ngl" not in cmd

    request = llama_bench.LlamaBenchRequest(depths = [0, 4096], flash_attn = "on", n_gpu_layers = 20)
    cmd = llama_bench._command(Path("llama-bench"), "m.gguf", request)
    assert cmd[cmd.index("-d") + 1] == "0,4096"
    assert cmd[cmd.index("-fa") + 1] == "1"
    assert cmd[cmd.index("-ngl") + 1] == "20"


def test_managed_accounts_do_not_see_the_owners_model_or_run(bench):
    from utils.account_context import AccountContext, bind_account

    client, _slow, _unloads = bench
    assert client.post("/api/benchmarks/llama-bench/run", json = {}).status_code == 200
    _wait(client)

    async def managed():
        bind_account(AccountContext("acct-bob", "bob", "user"))
        return "bob"

    client.app.dependency_overrides[get_current_subject] = managed
    status = client.get("/api/benchmarks/llama-bench/status").json()
    assert status["model"] is None and status["ggufVariant"] is None and status["job"] is None
    assert client.get("/api/benchmarks/llama-bench/run").json() == {"job": None}


def test_api_key_callers_get_paths_redacted_from_the_error_and_log(bench, tmp_path, monkeypatch):
    client, _slow, _unloads = bench
    secret = str(tmp_path / "private" / "owner-model.gguf")
    failing = tmp_path / "failing.py"
    failing.write_text(
        f"import sys; print(\"main: error: failed to load model '{secret}'\", file=sys.stderr); sys.exit(1)\n",
        encoding = "utf-8",
    )
    monkeypatch.setattr(
        llama_bench, "_command", lambda binary, gguf, request: [sys.executable, str(failing)]
    )
    client.app.dependency_overrides[authenticated_via_api_key] = lambda: True
    assert client.post("/api/benchmarks/llama-bench/run", json = {}).status_code == 200
    job = _wait(client)
    assert job["status"] == "error"
    assert secret not in job["error"] and all(secret not in line for line in job["log"])


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX mode bits")
def test_a_non_executable_llama_bench_is_not_offered(tmp_path, monkeypatch):
    server = tmp_path / "llama-server"
    server.write_text("", encoding = "utf-8")
    bench_bin = tmp_path / "llama-bench"
    bench_bin.write_text("", encoding = "utf-8")
    bench_bin.chmod(0o644)
    monkeypatch.setattr(
        "core.inference.llama_cpp.LlamaCppBackend._find_llama_server_binary",
        staticmethod(lambda: str(server)),
    )
    assert llama_bench.find_llama_bench() is None
    bench_bin.chmod(0o755)
    assert llama_bench.find_llama_bench() == bench_bin
