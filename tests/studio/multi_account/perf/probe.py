# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One isolated benchmark process. Product imports happen after selecting the revision."""

import argparse
import json
import math
import os
import sqlite3
import statistics
import sys
import time
from collections import Counter
from contextlib import closing
from pathlib import Path
from unittest.mock import patch


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "samples": len(samples),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
    }


def measure_cost(operation) -> dict:
    counters = Counter(connections = 0, queries = 0, statements = 0, mkdir_calls = 0, directories_created = 0)
    connect, mkdir = sqlite3.connect, os.mkdir

    def trace(statement):
        counters["statements"] += 1
        if statement.lstrip().upper().startswith("SELECT"):
            counters["queries"] += 1

    def open_connection(*args, **kwargs):
        counters["connections"] += 1
        conn = connect(*args, **kwargs)
        conn.set_trace_callback(trace)
        return conn

    def create_directory(*args, **kwargs):
        counters["mkdir_calls"] += 1
        result = mkdir(*args, **kwargs)
        counters["directories_created"] += 1
        return result

    with (
        patch.object(sqlite3, "connect", open_connection),
        patch.object(os, "mkdir", create_directory),
    ):
        operation()
    return dict(counters)


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--backend", type = Path, required = True)
    parser.add_argument("--helpers", type = Path, required = True)
    parser.add_argument("--mode", choices = ("cost", "timing"), required = True)
    parser.add_argument("--samples", type = int, default = 2000)
    parser.add_argument("--history-samples", type = int, default = 200)
    args = parser.parse_args()
    sys.path[:0] = [str(args.backend), str(args.helpers)]

    from seed import seed_legacy_install

    home = Path(os.environ["UNSLOTH_STUDIO_HOME"])
    seed_legacy_install(home)
    # A nonempty sidebar: 100 historical threads, identical across revisions.
    with closing(sqlite3.connect(home / "studio.db")) as conn:
        conn.executemany(
            "INSERT INTO chat_threads (id,title,model_type,created_at,updated_at) VALUES (?,?,'base',1000,1000)",
            [(f"history-{index}", f"Historical conversation {index}") for index in range(99)],
        )
        conn.commit()

    from fastapi.testclient import TestClient
    from support import bearer, make_app
    from utils.paths import storage_roots

    assert Path(storage_roots.__file__).resolve().is_relative_to(args.backend.resolve())
    # Before the contract, owner workspace_root was simply studio_root.
    workspace_root = getattr(storage_roots, "workspace_root", storage_roots.studio_root)
    headers = bearer("unsloth")
    with TestClient(make_app()) as client:

        def get(path):
            response = client.get(path, headers = headers)
            assert response.status_code == 200, (path, response.status_code, response.text)
            return response

        for _ in range(100):
            get("/api/auth/status")
            get("/account-probe")
            listing = get("/api/chat/threads")
        assert len(listing.json()["threads"]) == 100
        if args.mode == "cost":
            measurements = {
                "status": measure_cost(lambda: get("/api/auth/status")),
                "authenticated_get": measure_cost(lambda: get("/account-probe")),
                "workspace_1000": measure_cost(lambda: [workspace_root() for _ in range(1000)]),
            }
        else:
            measurements = {}
            for label, path, count in (
                ("status", "/api/auth/status", args.samples),
                ("history", "/api/chat/threads", args.history_samples),
            ):
                durations = []
                for _ in range(count):
                    start = time.perf_counter_ns()
                    get(path)
                    durations.append((time.perf_counter_ns() - start) / 1_000_000)
                measurements[label] = summarize(durations)
    print(json.dumps(measurements, sort_keys = True))


if __name__ == "__main__":
    main()
