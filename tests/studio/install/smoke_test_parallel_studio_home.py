#!/usr/bin/env python3
"""Smoke test: N parallel install.sh runs with distinct UNSLOTH_STUDIO_HOME
values must produce N fully isolated installs whose backends can run
side by side without clashing.

Covers the env-override path added in #5190:

    install-time
        * N concurrent ``install.sh --local --no-torch`` runs against
          this checkout, each pinned to its own UNSLOTH_STUDIO_HOME and
          a redirected HOME, all exit 0.
        * Each STUDIO_HOME contains its own bin/, share/, llama.cpp/
          and unsloth_studio/ venv, with no cross-install absolute
          paths.
        * share/studio_install_id is unique across the N installs.
        * share/studio.conf exports UNSLOTH_EXE, UNSLOTH_STUDIO_HOME
          and UNSLOTH_LLAMA_CPP_PATH, all pointing inside this install.
        * share/launch-studio.sh has @@DATA_DIR@@ substituted to its
          own share/ at install time.
        * bin/unsloth is a symlink that resolves into its own venv.
        * The redirected HOME is left clean: no shell-rc append, no
          .desktop file, no Studio.app stub, no shared marker.

    runtime
        * N concurrent ``bin/unsloth studio`` launches each bind their
          own dynamically allocated free port and stay healthy.
        * /api/health is 200, status is healthy, chat_only is true
          under --no-torch.
        * The studio_root_id reported by /api/health on each backend
          equals that install's share/studio_install_id, so the
          runtime resolver agrees with the install-time write.
        * studio_root_id values are pairwise distinct.
        * GET / and GET /api/chat are 200 on every backend.
        * The Python interpreter behind each PID is the install's own
          venv python (the bin/unsloth shim does not cross-resolve).

This is an integration smoke runner, not a pytest unit test. It does
real installs (~1 minute end to end on a warm uv cache) and is meant
to be invoked explicitly:

    python tests/studio/install/smoke_test_parallel_studio_home.py
    python tests/studio/install/smoke_test_parallel_studio_home.py --n 6 --keep

Exits 0 on PASS, 1 on FAIL, 2 on infrastructure error. Artifacts land
under a temporary directory and are removed on PASS unless --keep is
set; on FAIL or ERROR they are kept regardless so logs can be
inspected.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
INSTALL_TIMEOUT_S = 600
HEALTH_TIMEOUT_S = 120
HEALTH_POLL_INTERVAL_S = 1.0


class TestFailure(AssertionError):
    pass


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[smoke {ts}] {msg}", flush = True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_one_install(
    label: str,
    repo: Path,
    studio_home: Path,
    fake_home: Path,
    uv_cache: Path,
    log_path: Path,
) -> tuple[str, int]:
    studio_home.mkdir(parents = True, exist_ok = True)
    fake_home.mkdir(parents = True, exist_ok = True)
    uv_cache.mkdir(parents = True, exist_ok = True)
    log_path.parent.mkdir(parents = True, exist_ok = True)
    env = os.environ.copy()
    env["HOME"] = str(fake_home)
    env["UNSLOTH_STUDIO_HOME"] = str(studio_home)
    env["UV_CACHE_DIR"] = str(uv_cache)
    env["NO_COLOR"] = "1"
    with log_path.open("w") as fh:
        proc = subprocess.run(
            ["bash", "install.sh", "--local", "--no-torch"],
            cwd = str(repo),
            env = env,
            stdout = fh,
            stderr = subprocess.STDOUT,
            timeout = INSTALL_TIMEOUT_S,
        )
    return label, proc.returncode


def _launch_backend(
    studio_home: Path, fake_home: Path, port: int, log_path: Path
) -> subprocess.Popen:
    log_path.parent.mkdir(parents = True, exist_ok = True)
    env = os.environ.copy()
    env["HOME"] = str(fake_home)
    # Pin UNSLOTH_STUDIO_HOME (and clear the alias) so the child cannot
    # inherit a Studio root from the caller's shell. Without this, a shell
    # that already exports either var would override the per-label sys.prefix
    # inference and every backend would resolve to the caller's install.
    env["UNSLOTH_STUDIO_HOME"] = str(studio_home)
    env.pop("STUDIO_HOME", None)
    # The child process inherits a dup of stdout via Popen, so closing the
    # parent's handle when this function returns is safe and avoids relying
    # on GC timing to release the fd.
    with log_path.open("w") as fh:
        return subprocess.Popen(
            [
                str(studio_home / "bin" / "unsloth"),
                "studio",
                "-H",
                "127.0.0.1",
                "-p",
                str(port),
                "--silent",
            ],
            env = env,
            stdout = fh,
            stderr = subprocess.STDOUT,
            start_new_session = True,
        )


def _wait_for_health(port: int, timeout: float) -> dict:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    url = f"http://127.0.0.1:{port}/api/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout = 2) as r:
                if r.status == 200:
                    return json.loads(r.read().decode())
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            last_err = e
        time.sleep(HEALTH_POLL_INTERVAL_S)
    raise TestFailure(
        f"port {port}: /api/health never returned 200 (last_err={last_err})"
    )


def _http_status(port: int, path: str, timeout: float = 5.0) -> int:
    url = f"http://127.0.0.1:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout = timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def _check_install_layout(label: str, studio_home: Path) -> dict:
    for d in ("bin", "share", "llama.cpp", "unsloth_studio"):
        if not (studio_home / d).is_dir():
            raise TestFailure(f"[{label}] missing {studio_home / d}")

    shim = studio_home / "bin" / "unsloth"
    if not shim.is_symlink():
        raise TestFailure(f"[{label}] {shim} is not a symlink")
    expected_target = (studio_home / "unsloth_studio" / "bin" / "unsloth").resolve()
    if shim.resolve() != expected_target:
        raise TestFailure(
            f"[{label}] shim resolves to {shim.resolve()}, expected {expected_target}"
        )

    install_id_path = studio_home / "share" / "studio_install_id"
    if not install_id_path.is_file():
        raise TestFailure(f"[{label}] missing {install_id_path}")
    install_id = install_id_path.read_text().strip()
    if len(install_id) < 32:
        raise TestFailure(f"[{label}] studio_install_id too short: {install_id!r}")

    conf = (studio_home / "share" / "studio.conf").read_text()
    must_contain = [
        f"UNSLOTH_EXE='{studio_home}/unsloth_studio/bin/unsloth'",
        f"export UNSLOTH_STUDIO_HOME='{studio_home}'",
        f"export UNSLOTH_LLAMA_CPP_PATH='{studio_home}/llama.cpp'",
    ]
    for needle in must_contain:
        if needle not in conf:
            raise TestFailure(
                f"[{label}] studio.conf missing line:\n  {needle}\n" f"actual:\n{conf}"
            )

    launcher = (studio_home / "share" / "launch-studio.sh").read_text()
    if "@@DATA_DIR@@" in launcher:
        raise TestFailure(f"[{label}] launch-studio.sh kept @@DATA_DIR@@ placeholder")
    expected_data_dir_line = f"DATA_DIR='{studio_home}/share'"
    if expected_data_dir_line not in launcher:
        raise TestFailure(
            f"[{label}] launch-studio.sh missing {expected_data_dir_line!r}"
        )

    return {"label": label, "studio_home": str(studio_home), "install_id": install_id}


def _check_fake_home_clean(fake_home: Path) -> None:
    forbidden = [
        ".bashrc",
        ".zshrc",
        ".profile",
        ".unsloth",
        Path(".local") / "share" / "applications" / "unsloth-studio.desktop",
        Path("Desktop") / "unsloth-studio.desktop",
        Path("Applications") / "Unsloth Studio.app",
    ]
    leaked = [str(p) for p in forbidden if (fake_home / p).exists()]
    if leaked:
        raise TestFailure(
            f"redirected HOME picked up persistent install pollution: {leaked}"
        )


def _backend_pid_python(pid: int) -> Path | None:
    """Resolve the binary backing a running PID. Linux exposes this at
    /proc/PID/exe; on platforms without /proc (macOS, BSD, Windows) we
    skip this check and rely on the install-time symlink + studio.conf
    invariants to catch cross-resolution. Returns None when /proc is
    unavailable so the caller can skip cleanly."""
    if sys.platform != "linux":
        return None
    proc_exe = Path(f"/proc/{pid}/exe")
    if not proc_exe.exists():
        return None
    return proc_exe.resolve()


def run(n_installs: int, keep: bool) -> int:
    if n_installs < 2:
        raise TestFailure("--n must be >= 2 to test for clashes")
    labels = [chr(ord("a") + i) for i in range(n_installs)]

    repo = PACKAGE_ROOT
    if not (repo / "install.sh").is_file():
        raise TestFailure(
            f"install.sh not found at {repo}; " "run from a clone of unslothai/unsloth"
        )

    test_root = Path(tempfile.mkdtemp(prefix = "unsloth_studio_clash_"))
    _log(f"test root: {test_root}")
    _log(f"repo: {repo}")

    backends: list[tuple[str, Path, Path, int, subprocess.Popen]] = []
    failed = False
    try:
        # ---- parallel installs --------------------------------------------
        _log(f"launching {n_installs} parallel installs (--local --no-torch)")
        with ThreadPoolExecutor(max_workers = n_installs) as pool:
            futures = []
            for label in labels:
                futures.append(
                    pool.submit(
                        _run_one_install,
                        label,
                        repo,
                        test_root / "installs" / label,
                        test_root / "fake_homes" / label,
                        test_root / "uv_caches" / label,
                        test_root / "logs" / f"install_{label}.log",
                    )
                )
            for fut in as_completed(futures):
                label, rc = fut.result()
                _log(f"  install {label}: exit {rc}")
                if rc != 0:
                    raise TestFailure(
                        f"install {label} failed (rc={rc}); see "
                        f"{test_root / 'logs' / f'install_{label}.log'}"
                    )

        # ---- install-layout invariants ------------------------------------
        _log("verifying install-time invariants")
        observed = []
        for label in labels:
            studio_home = test_root / "installs" / label
            obs = _check_install_layout(label, studio_home)
            observed.append(obs)
            _check_fake_home_clean(test_root / "fake_homes" / label)
        ids = [o["install_id"] for o in observed]
        if len(set(ids)) != len(ids):
            raise TestFailure(f"studio_install_id collision: {ids}")
        _log(f"  {len(ids)} unique studio_install_ids, all redirected HOMEs clean")

        # ---- parallel backend launches ------------------------------------
        _log(f"launching {n_installs} backends in parallel")
        for label in labels:
            port = _free_port()
            studio_home = test_root / "installs" / label
            fake_home = test_root / "fake_homes" / label
            log_path = test_root / "logs" / f"run_{label}.log"
            proc = _launch_backend(studio_home, fake_home, port, log_path)
            backends.append((label, studio_home, fake_home, port, proc))
            _log(f"  {label} -> port {port} (pid {proc.pid})")

        # ---- wait for health ----------------------------------------------
        _log("waiting for /api/health on each backend")
        health_payloads: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers = n_installs) as pool:
            fut_to_label = {
                pool.submit(_wait_for_health, port, HEALTH_TIMEOUT_S): label
                for (label, _sh, _fh, port, _p) in backends
            }
            for fut in as_completed(fut_to_label):
                label = fut_to_label[fut]
                health_payloads[label] = fut.result()
                _log(f"  {label}: healthy")

        # ---- runtime invariants -------------------------------------------
        _log("checking runtime invariants")
        seen_root_ids: set[str] = set()
        for (label, studio_home, _fh, port, proc), obs in zip(backends, observed):
            health = health_payloads[label]
            if health.get("status") != "healthy":
                raise TestFailure(f"[{label}] health status != healthy: {health}")
            if health.get("studio_root_id") != obs["install_id"]:
                raise TestFailure(
                    f"[{label}] runtime studio_root_id "
                    f"{health.get('studio_root_id')!r} != install_id "
                    f"{obs['install_id']!r}"
                )
            if not health.get("chat_only"):
                raise TestFailure(f"[{label}] chat_only is not true under --no-torch")
            if health["studio_root_id"] in seen_root_ids:
                raise TestFailure(
                    f"[{label}] studio_root_id collision at runtime: "
                    f"{health['studio_root_id']}"
                )
            seen_root_ids.add(health["studio_root_id"])

            for path in ("/", "/api/chat"):
                code = _http_status(port, path)
                if code != 200:
                    raise TestFailure(f"[{label}] GET {path} -> {code}")

            exe = _backend_pid_python(proc.pid)
            if exe is not None:
                expected_python = (
                    studio_home / "unsloth_studio" / "bin" / "python"
                ).resolve()
                if exe != expected_python:
                    raise TestFailure(
                        f"[{label}] PID {proc.pid} exe={exe}, expected {expected_python}"
                    )

        versions = {h.get("version") for h in health_payloads.values()}
        if len(versions) != 1:
            raise TestFailure(f"version mismatch across installs: {versions}")

        _log(
            f"PASS: all install + runtime invariants hold "
            f"(version={next(iter(versions))})"
        )
        return 0

    except TestFailure as e:
        _log(f"FAIL: {e}")
        failed = True
        return 1
    except Exception as e:
        _log(f"ERROR: {type(e).__name__}: {e}")
        failed = True
        return 2
    finally:
        for _lbl, _sh, _fh, _port, proc in backends:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout = 10)
                except Exception:
                    proc.kill()

        if keep or failed:
            _log(f"artifacts kept at {test_root}")
        else:
            shutil.rmtree(test_root, ignore_errors = True)
            _log(f"cleaned up {test_root}")


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__)
    ap.add_argument(
        "--n",
        type = int,
        default = 4,
        help = "number of parallel installs (default 4, must be >= 2)",
    )
    ap.add_argument(
        "--keep",
        action = "store_true",
        help = "leave the temp test root on disk even on PASS",
    )
    args = ap.parse_args()
    return run(args.n, args.keep)


if __name__ == "__main__":
    raise SystemExit(main())
