# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""React previews compile on the server: Node runs oxc-transform over the chat's
component and nothing else. The route must stay authenticated and bounded (size,
time, output, concurrency), and every failure has to come back as an answer the
preview can show, never a 500. The real-Node cases run wherever the OXC
validator's node_modules are installed."""

import asyncio
import inspect
import json
import shutil
import subprocess
import threading

import pytest
from fastapi import HTTPException

import core.inference.react_preview as rp
import routes.inference as inf_mod
from auth.authentication import get_current_subject

_HAS_REAL_COMPILER = rp.resolve_node_executable() is not None and rp._TRANSFORM_PACKAGE.is_file()
real_compiler = pytest.mark.skipif(
    not _HAS_REAL_COMPILER, reason = "needs Node and oxc-validator/node_modules/oxc-transform"
)

DASHBOARD = """import { useState } from "react";
import { Heart } from "lucide-react";

type Props = { start?: number };

export default function Dashboard({ start = 0 }: Props) {
  const [n, setN] = useState<number>(start);
  return <button className="p-2" onClick={() => setN(n + 1)}><Heart /> {n}</button>;
}
"""


@pytest.fixture(autouse = True)
def _empty_cache():
    rp._cache.clear()
    yield
    rp._cache.clear()


# --- real Node -------------------------------------------------------------------------


@real_compiler
def test_tsx_compiles_to_the_module_runner_shape():
    result = rp.compile_react_preview(DASHBOARD, "tsx")
    assert result["status"] == "ok", result
    assert {"react", "react/jsx-runtime", "lucide-react"} <= set(result["deps"])
    assert "__vite_ssr_import__" in result["code"]
    assert "Props" not in result["code"]  # types are gone


@real_compiler
def test_a_syntax_error_points_into_the_original_source():
    result = rp.compile_react_preview("const a = 1;\nconst b = <div>;\n", "tsx")
    assert result == {
        "status": "error",
        "diagnostics": [{"message": "Unexpected token", "line": 2, "column": 16}],
    }


@real_compiler
def test_columns_count_utf16_units_after_non_ascii_text():
    # oxc reports UTF-8 byte offsets. "日" is 3 bytes and the emoji 4, but 1 and 2 UTF-16
    # units, so a byte-based column would land 4 too far right (and line 1 shifts the offset).
    source = 'const a = "é😀";\r\nconst b = "日😀"; const c = <div>;\n'
    result = rp.compile_react_preview(source, "tsx")
    assert result["status"] == "error", result
    first = result["diagnostics"][0]
    assert (first["line"], first["column"]) == (2, 33)
    units = source.split("\r\n")[1].encode("utf-16-le")
    at = 2 * (first["column"] - 1)
    assert units[at : at + 2].decode("utf-16-le") == ";"


@real_compiler
def test_dynamic_import_is_rejected_at_compile_time():
    result = rp.compile_react_preview(
        'export default function App() { return null; }\nconst m = import("x");\n', "jsx"
    )
    assert result["status"] == "error"
    assert result["diagnostics"][0]["message"].startswith("Dynamic import() isn't supported")


@real_compiler
def test_a_missing_oxc_transform_reads_as_unavailable(monkeypatch, tmp_path):
    # The script resolves oxc-transform next to itself, so a copy outside the tool
    # directory exercises the guarded import, not the Python pre-check.
    script = tmp_path / "compile-react.mjs"
    shutil.copy(rp._SCRIPT, script)
    monkeypatch.setattr(rp, "_SCRIPT", script)
    assert rp.compile_react_preview(DASHBOARD, "tsx") == {
        "status": "unavailable",
        "reason": "transform_missing",
    }


# --- mocked Node -----------------------------------------------------------------------


class _FakeRun:
    def __init__(
        self,
        stdout = b"",
        returncode = 0,
        stderr = b"",
        exc = None,
    ):
        self.calls = []
        self.stdout, self.returncode, self.stderr, self.exc = stdout, returncode, stderr, exc

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if self.exc is not None:
            raise self.exc
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, self.stderr)


def _ok(code = "export {}", deps = ()):
    return json.dumps({"status": "ok", "code": code, "deps": list(deps)}).encode()


@pytest.fixture
def fake_node(monkeypatch, tmp_path):
    script = tmp_path / "compile-react.mjs"
    package = tmp_path / "package.json"
    script.write_text("")
    package.write_text("{}")
    monkeypatch.setattr(rp, "_SCRIPT", script)
    monkeypatch.setattr(rp, "_TRANSFORM_PACKAGE", package)
    monkeypatch.setattr(rp, "resolve_node_executable", lambda: "/opt/node/bin/node")
    monkeypatch.setattr(rp, "oxc_validator_tmp_root", lambda: tmp_path)
    monkeypatch.setattr(rp, "ensure_dir", lambda path: path)
    monkeypatch.setattr(
        rp, "child_env_without_native_path_secret", lambda: {"PATH": "/usr/bin", "NODE_PATH": "/x"}
    )
    monkeypatch.setattr(rp, "windows_hidden_subprocess_kwargs", lambda: {})

    def install(fake):
        monkeypatch.setattr(rp.subprocess, "run", fake)
        return fake

    return install


def test_node_runs_only_the_compile_script(fake_node, tmp_path):
    run = fake_node(_FakeRun(_ok()))
    rp.compile_react_preview("x", "jsx")
    ((args, kwargs),) = run.calls
    assert args == ["/opt/node/bin/node", str(rp._SCRIPT)]
    assert json.loads(kwargs["input"]) == {"source": "x", "lang": "jsx"}
    assert kwargs["timeout"] == rp.COMPILE_TIMEOUT_S == 10
    assert kwargs["cwd"] == str(rp._TOOL_DIR)
    env = kwargs["env"]
    assert "NODE_PATH" not in env
    assert env["PATH"].startswith("/opt/node/bin")
    assert env["TMPDIR"] == env["TMP"] == env["TEMP"] == str(tmp_path)


def test_the_result_is_passed_through(fake_node):
    fake_node(_FakeRun(_ok("code();", ["react", "react/jsx-runtime"])))
    assert rp.compile_react_preview("x", "tsx") == {
        "status": "ok",
        "code": "code();",
        "deps": ["react", "react/jsx-runtime"],
    }


def test_a_hung_node_is_a_timeout(fake_node):
    fake_node(_FakeRun(exc = subprocess.TimeoutExpired(["node"], rp.COMPILE_TIMEOUT_S)))
    assert rp.compile_react_preview("x", "tsx") == {"status": "unavailable", "reason": "timeout"}


def test_no_node_is_node_missing(fake_node, monkeypatch):
    run = fake_node(_FakeRun(_ok()))
    monkeypatch.setattr(rp, "resolve_node_executable", lambda: None)
    assert rp.compile_react_preview("x", "tsx") == {
        "status": "unavailable",
        "reason": "node_missing",
    }
    assert run.calls == []


def test_no_oxc_transform_is_transform_missing(fake_node, monkeypatch, tmp_path):
    run = fake_node(_FakeRun(_ok()))
    monkeypatch.setattr(rp, "_TRANSFORM_PACKAGE", tmp_path / "absent" / "package.json")
    assert rp.compile_react_preview("x", "tsx") == {
        "status": "unavailable",
        "reason": "transform_missing",
    }
    assert run.calls == []


@pytest.mark.parametrize(
    "fake",
    [
        _FakeRun(b"", returncode = 1, stderr = b"boom"),
        _FakeRun(b"x" * (rp.MAX_OUTPUT_BYTES + 1)),
        _FakeRun(b"not json"),
        _FakeRun(b"[1, 2]"),
        _FakeRun(json.dumps({"status": "ok", "code": 1, "deps": []}).encode()),
        _FakeRun(json.dumps({"status": "ok", "code": "", "deps": [1]}).encode()),
        _FakeRun(json.dumps({"status": "error", "diagnostics": []}).encode()),
        _FakeRun(json.dumps({"status": "error", "diagnostics": [{"line": 1}]}).encode()),
        _FakeRun(json.dumps({"status": "unavailable", "reason": "anything"}).encode()),
        _FakeRun(exc = OSError("exec format error")),
    ],
    ids = [
        "nonzero-exit",
        "oversized-stdout",
        "not-json",
        "not-an-object",
        "code-not-a-string",
        "dep-not-a-string",
        "no-diagnostics",
        "diagnostic-without-message",
        "unknown-reason",
        "launch-failed",
    ],
)
def test_anything_unexpected_is_failed(fake_node, fake):
    fake_node(fake)
    assert rp.compile_react_preview("x", "tsx") == {"status": "unavailable", "reason": "failed"}


def test_output_is_clipped(fake_node):
    diagnostics = [{"message": "m" * 5000, "line": -3, "column": True}] * 50
    fake_node(_FakeRun(json.dumps({"status": "error", "diagnostics": diagnostics}).encode()))
    result = rp.compile_react_preview("x", "tsx")
    assert len(result["diagnostics"]) == 20
    assert result["diagnostics"][0] == {"message": "m" * 2000, "line": 0, "column": 0}

    rp._cache.clear()
    fake_node(_FakeRun(_ok(deps = ["d" * 500] + [f"dep{i}" for i in range(300)])))
    deps = rp.compile_react_preview("x", "tsx")["deps"]
    assert len(deps) == 100
    assert max(map(len, deps)) == 200


def test_settled_results_are_cached_and_unavailable_is_not(fake_node, monkeypatch):
    run = fake_node(_FakeRun(_ok()))
    first = rp.compile_react_preview("same", "tsx")
    assert rp.compile_react_preview("same", "tsx") == first
    assert len(run.calls) == 1
    # The language is part of the key.
    rp.compile_react_preview("same", "jsx")
    assert len(run.calls) == 2

    monkeypatch.setattr(rp, "resolve_node_executable", lambda: None)
    assert rp.compile_react_preview("later", "tsx")["reason"] == "node_missing"
    # Node installed afterwards is picked up on the next try.
    monkeypatch.setattr(rp, "resolve_node_executable", lambda: "/opt/node/bin/node")
    assert rp.compile_react_preview("later", "tsx")["status"] == "ok"
    assert len(run.calls) == 3


def test_the_cache_is_bounded(fake_node):
    run = fake_node(_FakeRun(_ok()))
    for i in range(40):
        rp.compile_react_preview(f"s{i}", "tsx")
    assert len(rp._cache) == 32
    rp.compile_react_preview("s0", "tsx")  # evicted, so compiled again
    assert len(run.calls) == 41


def test_huge_outputs_are_not_cached(fake_node):
    run = fake_node(_FakeRun(_ok("x" * (1024 * 1024 + 1))))
    rp.compile_react_preview("big", "tsx")
    rp.compile_react_preview("big", "tsx")
    assert len(run.calls) == 2


def test_at_most_four_compiles_run_at_once(monkeypatch):
    lock = threading.Lock()
    release = threading.Event()
    started = threading.Semaphore(0)
    state = {"active": 0, "peak": 0}

    def blocking(source, lang):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        started.release()
        release.wait(10)
        with lock:
            state["active"] -= 1
        return {"status": "ok", "code": source, "deps": []}

    monkeypatch.setattr(rp, "compile_react_preview", blocking)

    async def main():
        monkeypatch.setattr(rp, "_SEMAPHORE", asyncio.Semaphore(rp.MAX_CONCURRENT))
        tasks = [
            asyncio.create_task(rp.compile_react_preview_async(f"s{i}", "tsx")) for i in range(6)
        ]
        for _ in range(4):
            assert await asyncio.to_thread(started.acquire, True, 10)
        await asyncio.sleep(0.2)
        assert state["active"] == 4  # the fifth waits for a slot
        release.set()
        return await asyncio.gather(*tasks)

    results = asyncio.run(main())
    assert [r["code"] for r in results] == [f"s{i}" for i in range(6)]
    assert state["peak"] == rp.MAX_CONCURRENT == 4


def test_a_cancelled_request_keeps_its_slot_until_node_exits(monkeypatch):
    release = threading.Event()
    started = []

    def blocking(source, lang):
        started.append(source)
        release.wait(10)
        return {"status": "ok", "code": source, "deps": []}

    monkeypatch.setattr(rp, "compile_react_preview", blocking)

    async def main():
        monkeypatch.setattr(rp, "_SEMAPHORE", asyncio.Semaphore(1))
        first = asyncio.create_task(rp.compile_react_preview_async("a", "tsx"))
        while not started:
            await asyncio.sleep(0.01)
        first.cancel()
        second = asyncio.create_task(rp.compile_react_preview_async("b", "tsx"))
        await asyncio.sleep(0.2)
        assert started == ["a"]
        release.set()
        assert (await second)["code"] == "b"
        with pytest.raises(asyncio.CancelledError):
            await first

    asyncio.run(main())


# --- route -----------------------------------------------------------------------------


def _compile(source: str, lang: str = "tsx"):
    request = inf_mod.ArtifactReactCompileRequest(source = source, lang = lang)
    return asyncio.run(inf_mod.compile_artifact_react_preview(request, current_subject = "u"))


def test_the_route_requires_a_signed_in_subject():
    parameter = inspect.signature(inf_mod.compile_artifact_react_preview).parameters[
        "current_subject"
    ]
    assert parameter.default.dependency is get_current_subject


def test_the_route_is_mounted_where_the_frontend_calls_it():
    routes = [
        route
        for route in inf_mod.studio_router.routes
        if getattr(route, "path", None) == "/artifact-react-compile"
    ]
    assert len(routes) == 1
    assert routes[0].methods == {"POST"}
    assert routes[0].include_in_schema is False


def test_sources_over_256_kib_are_refused(monkeypatch):
    calls = []

    async def fake(source, lang):
        calls.append((len(source), lang))
        return {"status": "ok", "code": "", "deps": []}

    monkeypatch.setattr(rp, "compile_react_preview_async", fake)
    assert _compile("a" * rp.MAX_SOURCE_BYTES)["status"] == "ok"
    with pytest.raises(HTTPException) as excinfo:
        _compile("a" * (rp.MAX_SOURCE_BYTES + 1))
    assert excinfo.value.status_code == 413
    # Bytes, not characters: 128 Ki two-byte characters plus one byte is over.
    with pytest.raises(HTTPException):
        _compile("é" * (rp.MAX_SOURCE_BYTES // 2) + "a")
    assert calls == [(rp.MAX_SOURCE_BYTES, "tsx")]


def test_lang_is_jsx_or_tsx_and_defaults_to_tsx():
    assert inf_mod.ArtifactReactCompileRequest(source = "").lang == "tsx"
    with pytest.raises(ValueError):
        inf_mod.ArtifactReactCompileRequest(source = "", lang = "python")
