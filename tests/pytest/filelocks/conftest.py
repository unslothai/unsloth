from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pytest
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("filelock-runner")
    group.addoption("--workers", action="store", type=int, default=8,
                    help="Default number of subprocess workers to launch per test")
    group.addoption("--subprocess-timeout", action="store", type=float, default=30.0,
                    help="Per-subprocess timeout in seconds")
    group.addoption("--no-output-truncation", action="store_true",
                    help="If set, do not truncate captured stdout/stderr on failure")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "workers(n): override subprocess worker count for this test")


@dataclass
class ProcResult:
    index: int
    cmd: List[str]
    payload: Dict[str, Any]
    returncode: Optional[int]  # None if timeout/terminated
    duration_s: float
    stdout: str
    stderr: str
    timeout: bool


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path.cwd()


# @pytest.fixture(autouse=True)
# def _session_env(monkeypatch: pytest.MonkeyPatch) -> None:
#     monkeypatch.setenv("PYTHONUNBUFFERED", "1")

@pytest.fixture(scope="session", autouse=True)
def _session_env():
    prev = os.environ.get("PYTHONUNBUFFERED")
    os.environ["PYTHONUNBUFFERED"] = "1"
    yield
    if prev is None:
        os.environ.pop("PYTHONUNBUFFERED", None)
    else:
        os.environ["PYTHONUNBUFFERED"] = prev

@pytest.fixture
def per_test_dir(tmp_path: Path) -> Path:
    """
    A clean directory you can use per test. Useful to isolate lock files,
    temp outputs, etc.
    """
    return tmp_path


@pytest.fixture
def workers(request: pytest.FixtureRequest) -> int:
    marker = request.node.get_closest_marker("workers")
    if marker:
        return int(marker.args[0])
    return int(request.config.getoption("--workers"))


@pytest.fixture
def subprocess_timeout(request: pytest.FixtureRequest) -> float:
    return float(request.config.getoption("--subprocess-timeout"))


@pytest.fixture
def truncate_outputs(request: pytest.FixtureRequest) -> bool:
    return not bool(request.config.getoption("--no-output-truncation"))


@pytest.fixture
def run_many(project_root: Path, subprocess_timeout: float):
    """
    Fan-out runner. Launches multiple Python subprocesses that execute
    a callable via tests/pytest/filelocks/workers.py, capturing stdout+stderr for each.
    """
    def _run_many(
        func: str,
        payloads: Iterable[Dict[str, Any]],
        *,
        cwd: Optional[Path] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
        python: str = sys.executable,
        max_parallel: Optional[int] = None,
    ) -> List[ProcResult]:
        payloads = list(payloads)
        if not payloads:
            raise ValueError("payloads must be a non-empty iterable of {'args': [...], 'kwargs': {...}} dicts")

        tmo = subprocess_timeout if timeout is None else timeout
        working_dir = str((cwd or project_root).resolve())

        # Ensure tests/ is importable so `-m tests.pytest.filelocks.workers` works
        env_combined = os.environ.copy()
        if env:
            env_combined.update(env)

        # Guarantee PYTHONPATH has the project root
        roots = [str(project_root.resolve()), working_dir]
        existing = [p for p in env_combined.get("PYTHONPATH", "").split(os.pathsep) if p]
        dedup = []
        for p in roots + existing:
            if p and p not in dedup:
                dedup.append(p)
        env_combined["PYTHONPATH"] = os.pathsep.join(dedup)

        base_cmd = [python, "-u", "-m", "tests.pytest.filelocks._worker", "--func", func]
        print(base_cmd)

        results: List[ProcResult] = []

        def launch_one(index: int, payload: Dict[str, Any]) -> ProcResult:
            start = time.perf_counter()
            try:
                proc = subprocess.run(
                    base_cmd,
                    input=json.dumps(payload),
                    cwd=working_dir,
                    env=env_combined,
                    capture_output=True,
                    text=True,
                    timeout=tmo,
                )
                duration = time.perf_counter() - start
                return ProcResult(
                    index=index,
                    cmd=base_cmd,
                    payload=payload,
                    returncode=proc.returncode,
                    duration_s=duration,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    timeout=False,
                )
            except subprocess.TimeoutExpired as e:
                duration = time.perf_counter() - start
                stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode() if e.stdout else "")
                stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode() if e.stderr else "")
                return ProcResult(
                    index=index,
                    cmd=base_cmd,
                    payload=payload,
                    returncode=None,
                    duration_s=duration,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=True,
                )

        max_workers = max_parallel or min(32, len(payloads))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(launch_one, i, payload) for i, payload in enumerate(payloads)]
            for fut in as_completed(futures):
                results.append(fut.result())

        # Keep results ordered by index for readability
        results.sort(key=lambda r: r.index)
        return results

    return _run_many


@pytest.fixture
def assert_all_ok(truncate_outputs: bool):
    """
    Assert helper that fails the test if ANY subprocess failed.
    It prints stdout, stderr, and the stdin payload for each failing worker.
    """
    def _assert_all_ok(results: List[ProcResult], *, show_bytes: int = 4000) -> None:
        failed = [
            r for r in results
            if r.timeout or (r.returncode is None) or (r.returncode != 0)
        ]
        if not failed:
            return

        def maybe_trunc(s: str) -> str:
            if not truncate_outputs or len(s) <= show_bytes:
                return s
            head = s[:show_bytes]
            tail = s[-show_bytes:]
            return f"{head}\n[...output truncated...]\n{tail}"

        sections = []
        sections.append(f"FAILURES DETECTED: {len(failed)}/{len(results)} subprocesses failed")
        for r in failed:
            rc = "TIMEOUT" if r.timeout else r.returncode
            payload_pretty = json.dumps(r.payload, indent=2, sort_keys=True)
            sections.append(
                textwrap.dedent(
                    f"""
                    ── worker #{r.index} ─────────────────────────────────────────
                    cmd: {' '.join(r.cmd)}
                    exit: {rc}, ran for {r.duration_s:.2f}s
                    stdin payload:
                    {payload_pretty}

                    ── stdout ───────────────────────────────────────────────────
                    {maybe_trunc(r.stdout).rstrip()}

                    ── stderr ───────────────────────────────────────────────────
                    {maybe_trunc(r.stderr).rstrip()}
                    """
                ).rstrip()
            )
        pytest.fail("\n\n".join(sections), pytrace=False)

    return _assert_all_ok

@pytest.fixture
def hf_cache_env(per_test_dir: Path) -> dict:
    hf_home = per_test_dir / "hf_home"
    (hf_home / "hub").mkdir(parents=True, exist_ok=True)
    # Keep tokenizers from forking extra threads & spamming logs in CI
    env = {
        "HF_HOME": str(hf_home),
        "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
        "TOKENIZERS_PARALLELISM": "false",
    }
    return env