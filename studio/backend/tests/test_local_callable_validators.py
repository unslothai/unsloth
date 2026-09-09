# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import re
import subprocess
from pathlib import Path

from core.data_recipe import local_callable_validators as validators


def test_oxc_batch_falls_back_when_node_times_out(monkeypatch, tmp_path):
    runner = tmp_path / "validate.mjs"
    runner.touch()
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise subprocess.TimeoutExpired(cmd = args[0], timeout = kwargs["timeout"])

    monkeypatch.setattr(validators, "_OXC_RUNNER_PATH", runner)
    monkeypatch.setattr(validators, "resolve_node_executable", lambda: "node")
    monkeypatch.setattr(validators, "ensure_dir", lambda path: tmp_path)
    monkeypatch.setattr(validators, "oxc_validator_tmp_root", lambda: tmp_path)
    monkeypatch.setattr(validators, "child_env_without_native_path_secret", lambda: {})
    monkeypatch.setattr(validators, "_windows_hidden_subprocess_kwargs", lambda: {})
    monkeypatch.setattr(validators.subprocess, "run", fake_run)

    results = validators._run_oxc_batch(
        node_lang = "js",
        validation_mode = "syntax",
        code_shape = "auto",
        code_values = ["const value = 1;", "const value = 2;"],
    )

    assert len(calls) == 1
    assert calls[0][1]["timeout"] == validators._OXC_TIMEOUT_S
    # The wrapper needs the same budget: oxlint is a grandchild this kill cannot reach.
    assert json.loads(calls[0][1]["input"])["timeout_ms"] == validators._OXC_TIMEOUT_S * 1000
    assert len(results) == 2
    assert all(result["is_valid"] is False for result in results)
    assert all(result["error_count"] == 1 for result in results)
    assert [result["error_message"] for result in results] == [
        "OXC validation timed out",
        "OXC validation timed out",
    ]


def test_the_wrapper_kills_oxlint_against_the_remaining_caller_budget():
    # Python's timeout SIGKILLs only the wrapper, so oxlint has to die inside validate.mjs,
    # against what is left of the caller's budget. On the source: no JS test runner ships.
    source = validators._OXC_RUNNER_PATH.read_text(encoding = "utf-8")

    assert re.search(
        r"mapBudgetMs\(payload\?\.timeout_ms\)", source
    ), "the oxlint budget must come from the timeout_ms the caller sends"

    # performance.now() is monotonic ms since process start, the same basis as the caller's
    # timeout; Date.now() would drift from it on a wall-clock step.
    remaining = re.search(r"const timeoutMs = ([^;]+);", source)
    assert remaining, "runLintBatch must compute what is left of the caller's budget"
    assert "performance.now()" in remaining.group(1)
    assert "Date.now()" not in remaining.group(1)

    fallback_budget = re.search(r"const OXLINT_DEFAULT_BUDGET_MS = ([\d_]+);", source)
    assert fallback_budget, "validate.mjs must declare OXLINT_DEFAULT_BUDGET_MS"
    assert (
        int(fallback_budget.group(1).replace("_", "")) == validators._OXC_TIMEOUT_S * 1000
    ), "the fallback budget must match the wait this process actually gives the wrapper"

    options = re.search(r"spawnSync\(oxlintBin, oxlintArgs, \{(.*?)\}\)", source, re.S)
    assert options, "oxlint must still be launched through spawnSync(oxlintBin, oxlintArgs, ...)"
    assert "timeout: timeoutMs" in options.group(
        1
    ), "oxlint's bound must be the computed remainder, not a constant"
    # SIGTERM is ignorable, and spawnSync then waits out the child anyway.
    assert 'killSignal: "SIGKILL"' in options.group(1)
