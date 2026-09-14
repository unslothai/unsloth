# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Run isolated upload regressions on a native CI runner."""

import json
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request

repo = Path(__file__).resolve().parents[2]
root = Path(os.environ.get("RAG_SIM_ART_DIR", str(repo / "temp/rag-upload"))).resolve()
root.mkdir(parents = True, exist_ok = True)
for folder in ("tmp", "cache", "browsers"):
    (root / folder).mkdir(exist_ok = True)
env = os.environ.copy()
env.update(
    {
        "RAG_SIM_ART_DIR": str(root),
        "TMPDIR": str(root / "tmp"),
        "TMP": str(root / "tmp"),
        "TEMP": str(root / "tmp"),
        "UV_CACHE_DIR": str(root / "cache/uv"),
        "PIP_CACHE_DIR": str(root / "cache/pip"),
        "XDG_CACHE_HOME": str(root / "cache"),
        "HF_HOME": str(root / "cache/huggingface"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PLAYWRIGHT_BROWSERS_PATH": str(root / "browsers"),
        "npm_config_cache": str(root / "cache/npm"),
    }
)
failures = []


def run(
    name,
    args,
    cwd = repo,
    required = True,
):
    with (root / f"{name}.log").open("w", encoding = "utf-8") as log:
        result = subprocess.run(
            list(map(str, args)), cwd = cwd, env = env, stdout = log, stderr = subprocess.STDOUT
        )
    print(f"{name}: exit {result.returncode}", flush = True)
    if result.returncode:
        if required:
            failures.append(name)
        print(
            (root / f"{name}.log").read_text(encoding = "utf-8", errors = "replace")[-6000:], flush = True
        )
    return result.returncode == 0


def python_in(directory):
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


assert run("bootstrap", [sys.executable, "-m", "venv", root / "bootstrap"])
bootstrap = python_in(root / "bootstrap")
assert run("install-uv", [bootstrap, "-m", "pip", "install", "uv==0.11.0"])
# actions/setup-python ships macOS builds without --enable-loadable-sqlite-extensions, so
# sqlite-vec cannot load and every RAG test errors. Download an interpreter that has one.
# Read the path off the managed list rather than asking uv to resolve "3.12": both
# `uv venv --managed-python` and `uv python find --managed-python` answered with the
# runner's own framework build, while every path this list reports is one uv installed.
if hasattr(sqlite3.connect(":memory:"), "enable_load_extension"):
    interpreter = sys.executable
else:
    assert run("python-install", [bootstrap, "-m", "uv", "python", "install", "3.12"])
    listed = subprocess.run(
        [
            str(bootstrap),
            "-m",
            "uv",
            "python",
            "list",
            "--only-installed",
            "--managed-python",
            "--output-format",
            "json",
        ],
        cwd = repo,
        env = env,
        capture_output = True,
        text = True,
        check = True,
    ).stdout
    interpreter = next(
        (entry["path"] for entry in json.loads(listed) if entry["version"].startswith("3.12.")),
        "",
    )
    assert interpreter, f"uv installed no 3.12 to build the venv from: {listed}"
assert run(
    "venv", [bootstrap, "-m", "uv", "venv", "--clear", "--python", interpreter, root / "venv"]
)
python = python_in(root / "venv")
# Fail here rather than through a hundred RagExtensionUnavailable errors.
assert run(
    "sqlite-extensions",
    [python, "-c", "import sqlite3; sqlite3.connect(':memory:').enable_load_extension(True)"],
)
assert run(
    "dependencies",
    [
        bootstrap,
        "-m",
        "uv",
        "pip",
        "install",
        "--python",
        python,
        "-r",
        repo / "tests/studio/rag_upload_requirements.txt",
    ],
)
tests = sorted((repo / "studio/backend/tests").glob("test_rag_*.py"))
tests.append(repo / "studio/backend/tests/test_conversation_archive.py")
run(
    "backend",
    [
        python,
        repo / "tests/studio/rag_upload_pytest.py",
        *tests,
        "-q",
        "-rs",
        "--basetemp",
        root / "pytest",
        "-o",
        f"cache_dir={root / 'pytest-cache'}",
        f"--junitxml={root / 'backend.xml'}",
    ],
)
frontend = repo / "studio/frontend"
npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
if run("frontend-install", [npm, "ci"], cwd = frontend):
    node_tests = sorted(
        set(
            p
            for pattern in (
                "rag-*.test.ts",
                "project-attachment*.test.ts",
                "project-source*.test.ts",
            )
            for p in (frontend / "tests").glob(pattern)
        )
    )
    run(
        "frontend-unit", ["node", "--experimental-strip-types", "--test", *node_tests], cwd = frontend
    )
    run("typecheck", [npm, "run", "typecheck"], cwd = frontend)
    if run(
        "browser-install", [python, "-m", "playwright", "install", "chromium", "firefox", "webkit"]
    ):
        engines = ["chromium", "firefox", "webkit"]
        driver = repo / "tests/studio/rag_upload_browser_tests.py"
        if not run("browser-probe", [python, driver, "--probe", *engines], required = False):
            if sys.platform == "linux":
                env.update(
                    {
                        "RETRY_ATTEMPTS": "2",
                        "RETRY_ATTEMPT_TIMEOUT": "180",
                        "APT_ACQUIRE_RETRIES": "0",
                    }
                )
                run(
                    "browser-libraries",
                    [
                        "bash",
                        repo / ".github/scripts/retry-with-apt-lock.sh",
                        python,
                        "-m",
                        "playwright",
                        "install-deps",
                        *engines,
                    ],
                )
        if os.name == "nt":
            engines += ["chrome", "edge"]
        if sys.platform == "darwin" and env.get("GITHUB_ACTIONS") == "true":
            if run("safari-enable", ["sudo", "/usr/bin/safaridriver", "--enable"]):
                engines.append("safari")
        with (root / "browser-server.log").open("w", encoding = "utf-8") as log:
            server = subprocess.Popen(
                [
                    "node",
                    str(repo / "tests/studio/rag_upload_browser_server.mjs"),
                    str(frontend),
                    str(root),
                ],
                cwd = repo,
                env = env,
                stdout = log,
                stderr = subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 30
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("Browser fixture exited before becoming ready")
                    try:
                        urllib.request.urlopen("http://127.0.0.1:18948/__state", timeout = 1).close()
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise RuntimeError("Browser fixture did not start")
                        time.sleep(0.1)
                run(
                    "browsers",
                    [python, driver, *engines],
                )
            finally:
                server.terminate()
                try:
                    server.wait(timeout = 10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
(root / "summary.json").write_text(
    json.dumps(
        {"os": platform.platform(), "python": sys.version, "failed_steps": failures}, indent = 2
    ),
    encoding = "utf-8",
)
raise SystemExit(bool(failures))
