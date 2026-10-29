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
def test_windows_launches_oxlint_through_its_bin_entry():
    source = validators._OXC_RUNNER_PATH.read_text(encoding = "utf-8")
    entry = re.search(r"const OXLINT_ENTRY = join\(TOOL_DIR, ([^)]*)\);", source)
    assert entry, "validate.mjs must name oxlint's bin entry for Windows"
    parts = [part.strip().strip('"') for part in entry.group(1).split(",")]
    package_dir = validators._OXC_TOOL_DIR / "node_modules" / "oxlint"
    package = json.loads((package_dir / "package.json").read_text(encoding = "utf-8"))
    bin_field = package["bin"]
    bin_path = bin_field["oxlint"] if isinstance(bin_field, dict) else bin_field
    assert validators._OXC_TOOL_DIR.joinpath(*parts) == (package_dir / bin_path).resolve(
        strict = False
    )
    assert "process.execPath" in source


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


@real_oxlint
def test_oxlint_lints_snippets_under_a_gitignored_temp_root(monkeypatch, tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check = True)
    (tmp_path / ".gitignore").write_text("*\n", encoding = "utf-8")
    oxc_tmp = tmp_path / "studio" / "oxc-validator"
    oxc_tmp.mkdir(parents = True)
    monkeypatch.setattr(validators, "oxc_validator_tmp_root", lambda: oxc_tmp)

    results = validators._run_oxc_batch(
        node_lang = "js",
        validation_mode = "lint",
        code_shape = "module",
        code_values = ["debugger;\n", "export const x = 1;\n"],
    )

    assert [result["code"] for result in results] == ["eslint(no-debugger)", None]


@real_oxlint
def test_oxlint_lints_a_batch_longer_than_one_command_line(monkeypatch, tmp_path):
    monkeypatch.setattr(validators, "oxc_validator_tmp_root", lambda: tmp_path)
    codes = ["debugger;\n", "export const x = 1;\n"] * 1500

    results = validators._run_oxc_batch(
        node_lang = "js",
        validation_mode = "lint",
        code_shape = "module",
        code_values = codes,
    )

    assert [result["code"] for result in results] == ["eslint(no-debugger)", None] * 1500


_HAS_REAL_PARSER = (
    validators.resolve_node_executable() is not None
    and (validators._OXC_TOOL_DIR / "node_modules" / "oxc-parser" / "package.json").is_file()
)
real_parser = pytest.mark.skipif(
    not _HAS_REAL_PARSER, reason = "needs Node and oxc-validator/node_modules/oxc-parser"
)

_PARITY_CODES = [
    "const value = 1;\n",
    "debugger;\n",
    "let broken = ;\n",
    "return 5",
    "const s = 'café \U0001f600';\nfoo(",
    "<div>{value}</div>",
    "",
    "var a = 1; a == null; if (a = 2) {}\n",
    "function f(x: number): string { return `${x}`; }\n",
    "export default function App() { return <p>{' '}</p>; }\n",
    "import x from 'y';\nx(\n",
    "\ud800 lone surrogate",
    "/* unterminated",
    "class A { #p = 1; m() { return this.#p; } }\n",
]


def _parity_batches(rng):
    batches = []
    for lang in ("js", "ts", "jsx", "tsx"):
        for mode in ("syntax", "lint", "syntax+lint"):
            for shape in ("auto", "module", "snippet"):
                size = rng.randint(1, 6)
                batches.append((lang, mode, shape, tuple(rng.sample(_PARITY_CODES, size))))
    return batches


def _run_batch(batch):
    lang, mode, shape, codes = batch
    return validators._run_oxc_batch(
        node_lang = lang, validation_mode = mode, code_shape = shape, code_values = list(codes)
    )


def _count_launches(monkeypatch):
    launches = {"popen": 0, "run": 0}
    real_popen, real_run = validators.subprocess.Popen, validators.subprocess.run

    def counting_popen(*args, **kwargs):
        # subprocess.run goes through Popen too.
        if "--serve" in args[0]:
            launches["popen"] += 1
        return real_popen(*args, **kwargs)

    def counting_run(*args, **kwargs):
        launches["run"] += 1
        return real_run(*args, **kwargs)

    monkeypatch.setattr(validators.subprocess, "Popen", counting_popen)
    monkeypatch.setattr(validators.subprocess, "run", counting_run)
    return launches


@real_oxlint
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_one_worker_answers_every_batch_exactly_like_a_process_per_batch(monkeypatch, seed):
    import random

    rng = random.Random(seed)
    batches = _parity_batches(rng)
    expected = {batch: json.dumps(_run_batch(batch)) for batch in batches}
    order = batches + rng.sample(batches, len(batches) // 2)
    rng.shuffle(order)

    launches = _count_launches(monkeypatch)
    with validators.oxc_validator_session():
        for batch in order:
            assert json.dumps(_run_batch(batch)) == expected[batch], batch
    assert launches == {"popen": 1, "run": 0}


@real_parser
def test_a_dead_worker_costs_one_batch_a_one_shot_run_and_is_replaced(monkeypatch):
    batch = ("ts", "syntax", "auto", ("let x: number = ;", "const ok = 1;"))
    expected = _run_batch(batch)
    launches = _count_launches(monkeypatch)
    with validators.oxc_validator_session():
        assert _run_batch(batch) == expected
        first = validators._OXC_SESSION._worker.proc
        first.kill()
        first.wait()
        assert _run_batch(batch) == expected
        assert launches == {"popen": 1, "run": 1}
        assert _run_batch(batch) == expected
        assert launches == {"popen": 2, "run": 1}
        assert validators._OXC_SESSION._worker.proc is not first


@real_parser
def test_the_worker_ends_with_its_session():
    with validators.oxc_validator_session():
        _run_batch(("js", "syntax", "auto", ("const a = 1;",)))
        proc = validators._OXC_SESSION._worker.proc
        assert proc.poll() is None
    assert proc.poll() is not None
    assert validators._OXC_SESSION is None


@pytest.mark.skipif(
    validators.resolve_node_executable() is None, reason = "needs Node to start the stub worker"
)
def test_a_worker_past_its_deadline_is_killed_and_the_batch_times_out(monkeypatch, tmp_path):
    silent = tmp_path / "silent.mjs"
    silent.write_text("process.stdin.resume();\n", encoding = "utf-8")
    monkeypatch.setattr(validators, "_OXC_RUNNER_PATH", silent)
    monkeypatch.setattr(validators, "_OXC_TIMEOUT_S", 1)
    monkeypatch.setattr(validators, "oxc_validator_tmp_root", lambda: tmp_path)
    with validators.oxc_validator_session():
        results = _run_batch(("js", "syntax", "auto", ("a", "b")))
        assert validators._OXC_SESSION._worker is None
    assert [result["error_message"] for result in results] == ["OXC validation timed out"] * 2


def test_the_kill_switch_keeps_one_process_per_batch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_OXC_PERSISTENT_WORKER", "0")
    with validators.oxc_validator_session():
        assert validators._OXC_SESSION is None


@real_parser
def test_the_worker_exits_when_its_caller_is_killed(tmp_path):
    psutil = pytest.importorskip("psutil")
    import sys
    import time

    backend = Path(validators.__file__).resolve().parents[2]
    script = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(backend)!r})\n"
        "from core.data_recipe import local_callable_validators as v\n"
        "with v.oxc_validator_session():\n"
        "    v._run_oxc_batch(node_lang='js', validation_mode='syntax', code_shape='auto',"
        " code_values=['const a = 1;'])\n"
        "    print(v._OXC_SESSION._worker.proc.pid, flush=True)\n"
        "    time.sleep(120)\n"
    )
    caller = subprocess.Popen([sys.executable, "-c", script], stdout = subprocess.PIPE, text = True)
    try:
        worker_pid = int(caller.stdout.readline())
    finally:
        caller.kill()
        caller.wait()

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            if psutil.Process(worker_pid).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.1)
    else:
        psutil.Process(worker_pid).kill()
        pytest.fail("the OXC worker outlived the process that started it")


@real_parser
def test_a_large_batch_costs_the_worker_about_what_it_costs_one_shot(monkeypatch):
    import time

    # A JSON line carries no raw newline until its end, so rescanning the buffer per stdin chunk
    # was quadratic: 18x one-shot for 32 MB, past the deadline on slower hosts.
    batch = (
        "js",
        "syntax",
        "auto",
        tuple("/*" + "a" * 3_200_000 + "*/\nconst x = 1;\n" for _ in range(10)),
    )
    started = time.monotonic()
    expected = _run_batch(batch)
    monkeypatch.setattr(validators, "_OXC_TIMEOUT_S", max(3.0, 5 * (time.monotonic() - started)))
    launches = _count_launches(monkeypatch)
    with validators.oxc_validator_session():
        assert _run_batch(batch) == expected
    assert launches == {"popen": 1, "run": 0}
    assert all(result["is_valid"] for result in expected)
