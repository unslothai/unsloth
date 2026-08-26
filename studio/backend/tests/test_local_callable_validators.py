# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
    assert len(results) == 2
    assert all(result["is_valid"] is False for result in results)
    assert all(result["error_count"] == 1 for result in results)
    assert [result["error_message"] for result in results] == [
        "OXC validation timed out",
        "OXC validation timed out",
    ]


def test_oxlint_spawn_is_bounded_below_the_python_timeout():
    # validate.mjs owns oxlint's lifetime: the Python timeout SIGKILLs only the Node
    # wrapper, so an oxlint bounded from Python alone survives as an orphan. The
    # backend ships no JS test runner, so the invariant is guarded on the source.
    source = validators._OXC_RUNNER_PATH.read_text(encoding = "utf-8")

    declared = re.search(r"const OXLINT_TIMEOUT_MS = ([\d_]+);", source)
    assert declared, "validate.mjs must declare OXLINT_TIMEOUT_MS"
    assert (
        int(declared.group(1).replace("_", "")) / 1000 < validators._OXC_TIMEOUT_S
    ), "oxlint must give up before the Python wait does, or Python kills the wrapper first"

    options = re.search(r"spawnSync\(oxlintBin, oxlintArgs, \{(.*?)\}\)", source, re.S)
    assert options, "oxlint must still be launched through spawnSync(oxlintBin, oxlintArgs, ...)"
    assert "timeout: OXLINT_TIMEOUT_MS" in options.group(1)
    # SIGTERM is ignorable, and spawnSync then waits out the child anyway.
    assert 'killSignal: "SIGKILL"' in options.group(1)
