# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import re
import subprocess
from pathlib import Path

import pytest

from core.data_recipe import local_callable_validators as validators

_HAS_REAL_OXLINT = (
    validators.resolve_node_executable() is not None
    and (validators._OXC_TOOL_DIR / "node_modules" / "oxlint" / "package.json").is_file()
)
real_oxlint = pytest.mark.skipif(
    not _HAS_REAL_OXLINT, reason = "needs Node and oxc-validator/node_modules/oxlint"
)


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
    assert json.loads(calls[0][1]["input"])["timeout_ms"] == validators._OXC_TIMEOUT_S * 1000
    assert len(results) == 2
    assert all(result["is_valid"] is False for result in results)
    assert all(result["error_count"] == 1 for result in results)
    assert [result["error_message"] for result in results] == [
        "OXC validation timed out",
        "OXC validation timed out",
    ]


def test_the_wrapper_kills_oxlint_against_the_remaining_caller_budget():
    # Python's timeout kills only the wrapper; oxlint must die inside validate.mjs.
    source = validators._OXC_RUNNER_PATH.read_text(encoding = "utf-8")

    assert re.search(
        r"mapBudgetMs\(payload\?\.timeout_ms\)", source
    ), "the oxlint budget must come from the timeout_ms the caller sends"

    # performance.now() is monotonic like the caller's timeout; Date.now() drifts.
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
    # SIGTERM is ignorable, and spawnSync then waits out the child.
    assert 'killSignal: "SIGKILL"' in options.group(1)


@real_oxlint
@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("config_name", ["oxlint.config.ts", ".oxlintrc.json"])
def test_oxlint_never_loads_a_config_planted_around_the_snippets(
    monkeypatch, tmp_path, config_name, depth
):
    # The snippets land under the account's temp root, which its confined tools can write, while
    # oxlint runs unconfined: a config found there is the account's code run as the Studio user.
    marker = tmp_path / "config-was-imported"
    payload = f'import fs from "node:fs";\nfs.writeFileSync({json.dumps(str(marker))}, "");\n'
    oxc_tmp = tmp_path / "accounts" / "alice" / "oxc-validator"
    oxc_tmp.mkdir(parents = True)
    planted = oxc_tmp.parents[depth - 1] if depth else oxc_tmp
    if config_name.endswith(".ts"):
        (planted / config_name).write_text(payload + "export default {};\n", encoding = "utf-8")
    else:
        (planted / config_name).write_text('{"jsPlugins": ["./plugin.js"]}', encoding = "utf-8")
        (planted / "plugin.js").write_text(
            payload + 'export default { meta: { name: "planted" }, rules: {} };\n',
            encoding = "utf-8",
        )
    monkeypatch.setattr(validators, "oxc_validator_tmp_root", lambda: oxc_tmp)

    results = validators._run_oxc_batch(
        node_lang = "js",
        validation_mode = "lint",
        code_shape = "module",
        code_values = ["debugger;\n"],
    )

    assert not marker.exists()
    # Pinned to the shipped, empty config: the default rules still report.
    assert [result["code"] for result in results] == ["eslint(no-debugger)"]
