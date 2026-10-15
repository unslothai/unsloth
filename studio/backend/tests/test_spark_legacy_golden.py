# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two-Spark path frozen as it was before N-node clusters existed: every launch argv, remote
command, router backend and status payload these scenarios produce must stay byte for byte the
same. SPARK_GOLDEN_RECORD=1 rewrites the fixture; only do that for a deliberate legacy change."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

import pytest

from core.inference import spark_serving as ss

from .spark_fake_llama import FakeLlama
from .test_spark_serving import (
    _FAKE_HELP_WITH_FLAG,
    _FAKE_HELP_WITHOUT_FLAG,
    _FakeBackend,
    _FakeRequest,
    _patch_remote,
    cluster,  # noqa: F401  (pytest fixture)
    write_fake_llama_server,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason = "the fake llama-server is a shell script")

GOLDEN = Path(__file__).parent / "fixtures" / "spark_legacy_golden.json"


def _norm(
    value: Any,
    tmp: Path,
    port: int = 0,
) -> Any:
    text = json.dumps(value, sort_keys = True, default = str)
    text = text.replace(str(tmp), "<TMP>")
    # The Studio home comes from a session fixture under pytest's numbered basetemp.
    text = re.sub(r'[^"]*/studio_homes\d+/home-\d+', "<HOME>", text)
    if port:
        text = text.replace(str(port), "<PORT>")
    return json.loads(text)


def _status(skip = ("relaunch_log",)) -> Dict[str, Any]:
    status = ss.status()
    return {k: v for k, v in status.items() if k not in skip}


def _check(name: str, actual: Any) -> None:
    data = json.loads(GOLDEN.read_text(encoding = "utf-8")) if GOLDEN.exists() else {}
    if os.environ.get("SPARK_GOLDEN_RECORD") == "1":
        data[name] = actual
        GOLDEN.parent.mkdir(parents = True, exist_ok = True)
        GOLDEN.write_text(json.dumps(data, indent = 1, sort_keys = True) + "\n", encoding = "utf-8")
        return
    assert name in data, f"no golden for {name}; record it deliberately"
    if actual != data[name] and os.environ.get("SPARK_GOLDEN_DUMP"):
        Path(os.environ["SPARK_GOLDEN_DUMP"]).write_text(
            json.dumps({"expected": data[name], "actual": actual}, indent = 1, sort_keys = True),
            encoding = "utf-8",
        )
    assert actual == data[name]


def _split(cluster_stub, monkeypatch, tmp_path, help_text, **request_kw):
    cluster_stub.topology = "layer_split"
    write_fake_llama_server(cluster_stub.bundle / "build" / "bin", help_text)
    model = tmp_path / "big.gguf"
    model.write_bytes(b"x" * 1024)
    calls, started = _patch_remote(monkeypatch)
    out = asyncio.run(ss.before_load(_FakeRequest(str(model), **request_kw), 32))
    return {
        "extra_args": out.llama_extra_args,
        "n_parallel": getattr(out, "n_parallel", None),
        "speculative_type": getattr(out, "speculative_type", None),
        "remote_calls": calls,
        "started": [(p.name, p.peer, p.argv) for p in started],
        "status": _status(),
        "planner": cluster_stub.planner_calls[-1],
    }


def test_layer_split_with_pipeline_groups(cluster, monkeypatch, tmp_path):
    out = _split(
        cluster, monkeypatch, tmp_path, _FAKE_HELP_WITH_FLAG, llama_extra_args = ["--seed", "1"]
    )
    _check("layer_split_groups", _norm(out, tmp_path))
    asyncio.run(ss.shutdown())


def test_layer_split_without_pipeline_groups(cluster, monkeypatch, tmp_path):
    out = _split(cluster, monkeypatch, tmp_path, _FAKE_HELP_WITHOUT_FLAG)
    _check("layer_split_plain", _norm(out, tmp_path))
    asyncio.run(ss.shutdown())


def test_layer_split_inherited_extras(cluster, monkeypatch, tmp_path):
    cluster.topology = "layer_split"
    write_fake_llama_server(cluster.bundle / "build" / "bin", _FAKE_HELP_WITH_FLAG)
    model = tmp_path / "big.gguf"
    model.write_bytes(b"x" * 1024)
    _patch_remote(monkeypatch)
    out = asyncio.run(
        ss.before_load(_FakeRequest(str(model)), 8, inherited_extra_args = ["--seed", "1"])
    )
    _check("layer_split_inherited", _norm({"extra_args": out.llama_extra_args}, tmp_path))
    asyncio.run(ss.shutdown())


def test_single_and_replicas_before_load(cluster, monkeypatch, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    calls, started = _patch_remote(monkeypatch)
    results = {}
    for topology in ("single", "replicas"):
        cluster.topology = topology
        request = _FakeRequest(str(model))
        out = asyncio.run(ss.before_load(request, 16))
        results[topology] = {
            "same_request": out is request,
            "extra_args": out.llama_extra_args,
            "status": _status(),
        }
    results["remote_calls"] = calls
    results["started"] = [(p.name, p.argv) for p in started]
    _check("single_and_replicas_before_load", _norm(results, tmp_path))


def test_replicas_after_load(cluster, monkeypatch, tmp_path):
    cluster.topology = "replicas"
    monkeypatch.setenv(ss.ENV_PEER, "127.0.0.1")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 100)
    calls, started = _patch_remote(
        monkeypatch, binary = "$HOME/.unsloth/llama.cpp/build/bin/llama-server"
    )

    async def scenario():
        fake = await FakeLlama("main").start()
        backend = _FakeBackend(fake.port, str(model))
        try:
            await ss.after_load(backend, 16)
            status = _status()
            router = status.pop("router") or {}
            out = {
                "status": status,
                "router_backends": [
                    (b["name"], b.get("primary"), b.get("slots"))
                    for b in router.get("backends", [])
                ],
                "started": [(p.name, p.peer, p.argv) for p in started],
                "remote_calls": calls,
                "routes_through_router": ss.route_base_url(backend) is not None,
            }
            return _norm(out, tmp_path, fake.port)
        finally:
            await ss.shutdown()
            await fake.stop()

    _check("replicas_after_load", asyncio.run(scenario()))


def test_disabled_status_and_arg_helpers(cluster, monkeypatch):
    cluster.spark = False
    ss.reset_peer_discovery_cache()
    out = {
        "disabled_status": ss.status(),
        "split_args": ss.layer_split_extra_args("192.168.200.13", 50052),
        "split_args_groups": ss.layer_split_extra_args("192.168.200.13", 50052, pipeline_groups = 2),
        "strip": ss.strip_layer_split_extra_args(
            ["--seed", "1"] + ss.layer_split_extra_args("10.0.0.2", 50052, pipeline_groups = 2)
        ),
        "strip_user_rpc": ss.strip_layer_split_extra_args(["--rpc", "10.0.0.9:1", "--seed", "1"]),
        "rpc_server_argv": ss.rpc_server_argv(
            "/b/ggml-rpc-server", bind = "10.0.0.2", port = 50052, cache = True
        ),
        "topologies": list(ss.TOPOLOGIES),
        "env_names": [
            ss.ENV_TOGGLE,
            ss.ENV_TOPOLOGY,
            ss.ENV_PEER,
            ss.ENV_PIPELINE_GROUPS,
            ss.ENV_MTP,
            ss.ENV_RPC_BIND,
        ],
    }
    _check("disabled_status_and_helpers", out)
