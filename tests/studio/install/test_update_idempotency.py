# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Second update must make no network calls and change nothing; needs UNSLOTH_IDEMPOTENCY_E2E=1."""

from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

import pytest


E2E = os.environ.get("UNSLOTH_IDEMPOTENCY_E2E") == "1"

PROXY = pathlib.Path(__file__).resolve().parent / "idempotency_proxy.py"
IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
# Each run's update log and proxy journal: pytest's tmp dir, or UNSLOTH_IDEMPOTENCY_ARTIFACTS to
# keep them (a failure is unreadable without them).
ARTIFACTS = os.environ.get("UNSLOTH_IDEMPOTENCY_ARTIFACTS", "")

# Hosts a no-op update must not touch; pypi.org and github.com are bounded separately.
PAYLOAD_HOSTS = (
    "files.pythonhosted.org",
    "objects.githubusercontent.com",
    "nodejs.org",
)
# Not a payload host: prebuilt metadata and payload share this host, and only the largest single
# transfer separates them (metadata <= 19 KB, payloads >= 567 KB, no Range).
PREBUILT_METADATA_HOST = "release-assets.githubusercontent.com"
PREBUILT_METADATA_CEILING = 256 * 1024

# HEAD /releases/latest, its 302, then one GET per metadata asset.
GITHUB_PER_PREBUILT = 4
# macOS llama.cpp skips that fast path: DEFAULT_MAX_MACOS_RELEASE_FALLBACKS (16) plans, 2 assets each.
MACOS_LLAMA_GITHUB = 2 * 16
MAX_GITHUB_DESKTOP = (
    (MACOS_LLAMA_GITHUB if IS_MACOS else GITHUB_PER_PREBUILT)
    + GITHUB_PER_PREBUILT
    + GITHUB_PER_PREBUILT  # an AMD iGPU host resolves llama twice (:8597)
    + 2  # the whisper/llama ggml_tree pairing (install_whisper_prebuilt.py:381)
    + 2  # slack for a retried fetch
)
# --local adds the triton ref probe (Linux) and uv's zoo git fetch.
MAX_GITHUB_LOCAL = MAX_GITHUB_DESKTOP + (0 if (IS_MACOS or IS_WINDOWS) else 1) + 2

# uv resolves each git+https requirement through the API on --local.
MACOS_LLAMA_API_GITHUB = 5
MAX_API_GITHUB_DESKTOP = MACOS_LLAMA_API_GITHUB if IS_MACOS else 0
MAX_API_GITHUB_LOCAL = MAX_API_GITHUB_DESKTOP + 1


def assert_read_metadata_but_no_payload(run) -> None:
    """The release metadata a no-op update legitimately reads, and nothing bigger."""
    largest = run.largest_transfer_from(PREBUILT_METADATA_HOST)
    assert largest <= PREBUILT_METADATA_CEILING, (
        f"{PREBUILT_METADATA_HOST} served a single {largest} byte transfer to an update with "
        f"nothing to do: that size is a prebuilt archive, not a release manifest. {run.report()}"
    )


# raw.githubusercontent.com serves the shortcut icon and the zoo pyproject; bounded by size.
ICON_FETCH_CEILING = 64 * 1024

# What the installers print when they decline to do work (setup.sh consumes the prebuilt installers'
# "already matches" and prints its own line).
NO_WORK_MARKERS = ("dependencies up to date", "prebuilt up to date", "sidecar current")

# Between the llama.cpp and whisper.cpp publishes there is no paired whisper; that is expected.
WHISPER_UNPAIRED = re.compile(r"whisper\.cpp\s+no compatible prebuilt \(")
# Printed whenever audio.cpp's installer ran, so it owes an answer from the install record.
AUDIO_CPP_STEP = re.compile(r"^\s*audio\.cpp\s+\S", re.MULTILINE)
# A warm-cache rebuild talks to no forbidden host, so the log line is the only witness.
FRONTEND_CURRENT_MARKER = re.compile(r"frontend\s+up to date")
# What setup.sh / setup.ps1 print when the version check found a newer release, and when
# it could not ask PyPI at all (the pass runs on purpose in both cases).
UPGRADE_MARKER = "available, updating..."
UPTODATE_MARKER = re.compile(r"python\s+\S+ \S+ is up to date")
PYPI_UNREACHABLE_MARKER = "could not reach PyPI, updating to be safe..."
# What --local installs from the checkout on every pass, so its RECORD moving is expected.
LOCAL_CORE = frozenset({"unsloth", "unsloth-zoo", "unsloth_zoo"})
# Full passes reinstall these at the same version: seed plugins, and click (sqlfluff<4 vs
# huggingface-hub 1.23+, recorded known_unmet).
FULL_PASS_CHURN = LOCAL_CORE | frozenset(
    {"data-designer-github-repo-seed", "data-designer-unstructured-seed", "click"}
)

DIST_LIST = (
    "import importlib.metadata as m, json; "
    "print(json.dumps(sorted(((d.metadata['Name'] or '').lower(), d.version) "
    "for d in m.distributions())))"
)
# Names and versions cannot see a same-version reinstall; the RECORD mtime can.
DIST_RECORDS = (
    "import importlib.metadata as m, json, os; "
    "out = []\n"
    "for d in m.distributions():\n"
    "    p = getattr(d, '_path', None) and (d._path / 'RECORD')\n"
    "    try:\n"
    "        st = os.stat(p)\n"
    "        out.append([(d.metadata['Name'] or '').lower(), d.version, st.st_mtime_ns, st.st_size])\n"
    "    except (OSError, TypeError):\n"
    "        out.append([(d.metadata['Name'] or '').lower(), d.version, None, None])\n"
    "print(json.dumps(sorted(out)))"
)


# Computed like the installer's known_unmet_index digest.
INSTALLER_DIR = pathlib.Path(__file__).resolve().parents[3] / "studio"
INDEX_DIGEST = (
    "import hashlib, sys\n"
    f"sys.path.insert(0, {str(INSTALLER_DIR)!r})\n"
    "import install_manifest\n"
    "index = install_manifest.installed_dependency_index()\n"
    "if index is None:\n"
    "    print('')\n"
    "else:\n"
    "    digest = hashlib.sha256()\n"
    "    for name in sorted(index):\n"
    "        digest.update(f'{name}=={index[name][0]}\\n'.encode('utf-8'))\n"
    "    print(digest.hexdigest())\n"
)


def _home() -> pathlib.Path:
    return pathlib.Path(os.environ.get("UNSLOTH_IDEMPOTENCY_HOME") or pathlib.Path.home())


def _studio_home() -> pathlib.Path:
    override = os.environ.get("UNSLOTH_IDEMPOTENCY_STUDIO_HOME")
    return pathlib.Path(override) if override else _home() / ".unsloth" / "studio"


def _unsloth_home() -> pathlib.Path:
    """Where the prebuilts (llama.cpp, whisper.cpp, node) live: setup.sh's UNSLOTH_HOME
    rule, which nests them under a CUSTOM Studio home and keeps ~/.unsloth otherwise."""
    if os.environ.get("UNSLOTH_IDEMPOTENCY_STUDIO_HOME"):
        return _studio_home()
    return _home() / ".unsloth"


def _venv_python() -> pathlib.Path:
    venv = _studio_home() / "unsloth_studio"
    return venv / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def _installed_version(venv_python: pathlib.Path) -> str:
    probe = subprocess.run(
        [
            str(venv_python),
            "-I",
            "-c",
            "import importlib.metadata as m; print(m.version('unsloth'))",
        ],
        capture_output = True,
        text = True,
        timeout = 120,
    )
    return probe.stdout.strip()


def _pypi_latest() -> str:
    import urllib.request
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/unsloth/json", timeout = 60) as fh:
            return json.load(fh)["info"]["version"]
    except Exception:  # noqa: BLE001 - no answer means do not run the non-local cases
        return ""


@pytest.fixture(scope = "session")
def install() -> pathlib.Path:
    # Gated here, not on the module, so the proxy self test still runs in ordinary CI.
    if not E2E:
        pytest.skip("opt-in end to end; set UNSLOTH_IDEMPOTENCY_E2E=1")
    venv_python = _venv_python()
    if not venv_python.is_file():
        # Opted in and nothing to measure: a skip would keep the job green while measuring nothing.
        pytest.fail(
            f"UNSLOTH_IDEMPOTENCY_E2E is set but there is no Studio install at {venv_python}"
        )
    return venv_python


class ProxyRun:
    """One update, measured."""

    def __init__(self, directory: pathlib.Path, log: str, summary: dict, seconds: float, rc: int):
        self.directory = directory
        self.log = log
        self.summary = summary
        self.seconds = seconds
        self.rc = rc

    def bytes_from(self, host: str) -> int:
        return sum(
            v["bytes_down"]
            for h, v in self.summary["by_host"].items()
            if h == host or h.endswith("." + host)
        )

    def connections_to(self, host: str) -> int:
        return sum(
            v["connections"]
            for h, v in self.summary["by_host"].items()
            if h == host or h.endswith("." + host)
        )

    def largest_transfer_from(self, host: str) -> int:
        """The biggest single connection from `host`. 0 when it was never contacted."""
        return max(
            (
                v["largest_bytes_down"]
                for h, v in self.summary["by_host"].items()
                if h == host or h.endswith("." + host)
            ),
            default = 0,
        )

    @property
    def connections(self) -> int:
        return self.summary.get("connections", 0)

    @property
    def refused(self) -> int:
        return self.summary.get("refused", 0)

    def report(self) -> str:
        hosts = ", ".join(
            f"{h}={v['bytes_down'] / 1e6:.1f}MB/{v['connections']}c"
            for h, v in list(self.summary["by_host"].items())[:8]
        )
        return (
            f"{self.seconds:.1f}s exit={self.rc} "
            f"down={self.summary['total_bytes_down'] / 1e6:.1f}MB "
            f"conns={self.connections} refused={self.refused} [{hosts}]"
        )


def _start_proxy(directory: pathlib.Path, *extra: str):
    log = directory / "proxy.jsonl"
    port_file = directory / "proxy.port"
    port_file.unlink(missing_ok = True)
    # The proxy appends: a retained journal would attribute the previous run's traffic to this one.
    log.unlink(missing_ok = True)
    process = subprocess.Popen(
        [
            sys.executable,
            str(PROXY),
            "serve",
            "--port",
            "0",
            "--log",
            str(log),
            "--port-file",
            str(port_file),
            *extra,
        ],
        stdout = (directory / "proxy.out").open("w"),
        stderr = subprocess.STDOUT,
    )
    for _ in range(100):
        if port_file.is_file() and port_file.read_text().strip():
            break
        if process.poll() is not None:
            raise RuntimeError((directory / "proxy.out").read_text())
        time.sleep(0.1)
    else:
        process.kill()
        raise RuntimeError("the proxy never reported a port")
    return process, f"http://127.0.0.1:{port_file.read_text().strip()}", log


def _wait_until(
    predicate,
    *,
    timeout: float = 5.0,
    interval: float = 0.05,
) -> bool:
    """Poll until `predicate` holds or the deadline passes. A monotonic deadline rather
    than a fixed sleep: the wait is for something the proxy is already doing, and a sleep
    long enough to be safe on a loaded CI box is wasted on every ordinary run."""
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


def _settle_journal(
    log_path: pathlib.Path,
    *,
    timeout: float = 5.0,
    quiet: float = 0.25,
    active_grace: float = 75.0,
) -> bool:
    """Wait for the proxy journal to go quiet before terminating it, or in-flight records are dropped."""
    active_path = log_path.with_name(log_path.name + ".active")

    def _workers_active() -> bool:
        # A worker blocked in an upstream connect has written nothing yet.
        try:
            return int(active_path.read_text().strip() or "0") > 0
        except (OSError, ValueError):
            return False

    size = -1
    stable_since = time.monotonic()
    deadline = stable_since + timeout
    grace_deadline = stable_since + active_grace
    while True:
        now = time.monotonic()
        try:
            current = log_path.stat().st_size
        except OSError:
            current = 0
        if current != size:
            size, stable_since = current, now
        elif now - stable_since >= quiet and not _workers_active():
            return True
        if now >= deadline and not _workers_active():
            return True
        if now >= grace_deadline:
            return not _workers_active()
        time.sleep(0.05)


def run_update(
    tmp_path: pathlib.Path,
    label: str,
    *,
    local: bool = True,
    offline: bool = False,
    refuse: bool = False,
    deny_hosts: str = "",
) -> ProxyRun:
    """One `unsloth studio update`, through the proxy, timed and logged."""
    directory = (pathlib.Path(ARTIFACTS) if ARTIFACTS else tmp_path) / label
    directory.mkdir(parents = True, exist_ok = True)
    extra = ["--refuse"] if refuse else (["--deny-hosts", deny_hosts] if deny_hosts else [])
    process, url, log_path = _start_proxy(directory, *extra)
    # Inherited, not replaced: a hand-built environment fails in ways that look like the product.
    env = dict(os.environ)
    env.pop("UNSLOTH_IDEMPOTENCY_E2E", None)
    # The pytest venv is not the venv under test.
    for leaked in (
        "VIRTUAL_ENV",
        "PYTHONPATH",
        "PYTHONHOME",
        "UV_CACHE_DIR",
        "UV_PYTHON",
        "UV_CONSTRAINT",
        "UV_CONFIG_FILE",
        "PIP_CONSTRAINT",
        "PIP_CONFIG_FILE",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "XDG_CONFIG_HOME",
        "UNSLOTH_STUDIO_HOME",
        "STUDIO_HOME",
        "NPM_CONFIG_PROXY",
        "NPM_CONFIG_HTTPS_PROXY",
        "NPM_CONFIG_NOPROXY",
    ):
        env.pop(leaked, None)
    env.update(
        HOME = str(_home()),
        USERPROFILE = str(_home()),
        UNSLOTH_SKIP_AUTOSTART = "1",
        UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK = "1",
        PYTHONUNBUFFERED = "1",
        HTTPS_PROXY = url,
        HTTP_PROXY = url,
        ALL_PROXY = url,
        https_proxy = url,
        http_proxy = url,
        NO_PROXY = "127.0.0.1,localhost",
        no_proxy = "127.0.0.1,localhost",
        # npm_config_https_proxy outranks HTTPS_PROXY (npm 11).
        npm_config_proxy = url,
        npm_config_https_proxy = url,
        npm_config_noproxy = "127.0.0.1,localhost",
        # Presence, not truthiness: otherwise the CLI reads a proxy default from PowerShell profiles.
        _UNSLOTH_PS_PROXY_DEFAULTS = "{}",
    )
    if os.environ.get("UNSLOTH_IDEMPOTENCY_STUDIO_HOME"):
        env["UNSLOTH_STUDIO_HOME"] = os.environ["UNSLOTH_IDEMPOTENCY_STUDIO_HOME"]
    if offline:
        env["UV_OFFLINE"] = "1"
    else:
        env.pop("UV_OFFLINE", None)
    command = [str(_venv_python()), "-I", "-X", "utf8", "-m", "unsloth_cli", "studio", "update"]
    if local:
        command.append("--local")
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd = str(_home()),
            env = env,
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 3600,
        )
        text = completed.stdout + completed.stderr
        rc = completed.returncode
    finally:
        seconds = time.time() - started
        # Read first: a dead proxy's refused CONNECT looks like no connection.
        proxy_died = process.poll() is not None
        settled = _settle_journal(log_path)
        process.terminate()
        try:
            process.wait(timeout = 10)
        except subprocess.TimeoutExpired:  # pragma: no cover - a wedged tunnel
            process.kill()
    (directory / "update.log").write_text(text, encoding = "utf-8")
    assert not proxy_died, (
        f"{label}: the measurement proxy exited (code {process.returncode}) before the update "
        "finished; its journal is incomplete and proves nothing about connections"
    )
    assert settled, (
        f"{label}: the update finished while a proxy worker was still between accept and "
        "its journal record; the journal is incomplete and proves nothing about connections"
    )
    summary = _load_proxy_module().summary(str(log_path))
    run = ProxyRun(directory, text, summary, seconds, rc)
    print(f"[idempotency] {label}: {run.report()}", flush = True)
    return run


def _load_proxy_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("idempotency_proxy", PROXY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(path: pathlib.Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _tree_state(root: pathlib.Path) -> dict | None:
    """File count and newest mtime. Cheap enough for a sidecar of 4000 files, and it
    moves the moment anything in the tree is rewritten."""
    if not root.is_dir():
        return None
    count = 0
    newest = 0.0
    total = 0
    for path in root.rglob("*"):
        try:
            stat = path.stat()
        except OSError:
            continue
        if path.is_file():
            count += 1
            total += stat.st_size
            newest = max(newest, stat.st_mtime)
    return {"files": count, "bytes": total, "newest_mtime": round(newest, 3)}


def snapshot(venv_python: pathlib.Path) -> dict:
    studio_home = _studio_home()
    venv = venv_python.parent.parent
    unsloth_home = _unsloth_home()
    state: dict = {}
    distributions = subprocess.run(
        [str(venv_python), "-I", "-c", DIST_LIST],
        capture_output = True,
        text = True,
        timeout = 300,
    )
    state["distributions"] = json.loads(distributions.stdout or "[]")
    records = subprocess.run(
        [str(venv_python), "-I", "-c", DIST_RECORDS],
        capture_output = True,
        text = True,
        timeout = 300,
    )
    state["dist_records"] = [
        [name, version, None, None] if name in FULL_PASS_CHURN else [name, version, mtime, size]
        for name, version, mtime, size in json.loads(records.stdout or "[]")
    ]

    index_digest = subprocess.run(
        [str(venv_python), "-I", "-c", INDEX_DIGEST],
        capture_output = True,
        text = True,
        timeout = 300,
    )
    state["installed_index_digest"] = (index_digest.stdout or "").strip() or None

    manifest_path = venv / "unsloth_install_manifest.json"
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
        # The one key ALLOWED to move: a pass that correctly did nothing still finished.
        manifest.pop("completed_at_ms", None)
    state["manifest"] = manifest

    # Byte for byte: the point of the prebuilt pre-checks is that they do not rewrite the marker.
    for marker in (
        "UNSLOTH_PREBUILT_INFO.json",
        "UNSLOTH_WHISPER_PREBUILT_INFO.json",
        "UNSLOTH_AUDIO_CPP_PREBUILT_INFO.json",
        "UNSLOTH_NODE_PREBUILT_INFO.json",
    ):
        found = sorted(unsloth_home.glob(f"*/{marker}")) if unsloth_home.is_dir() else []
        # The digest alone cannot see a rewrite with identical bytes; the mtime can.
        state[marker] = {
            str(p.relative_to(unsloth_home)): (_digest(p), p.stat().st_mtime_ns) for p in found
        }

    state["uv_cache_marker"] = _digest(studio_home / "cache" / "uv-cache-dir")
    state["no_torch_marker"] = (venv / ".unsloth-no-torch").is_file()
    for name in (".venv_t5_530", ".venv_t5_550", ".venv_t5_510"):
        state[name] = _tree_state(studio_home / name)

    binaries: dict[str, float] = {}
    if unsloth_home.is_dir():
        for pattern in (
            "llama.cpp/**/llama-server*",
            "llama.cpp/**/llama-quantize*",
            "whisper.cpp/**/whisper-server*",
            "audio.cpp/**/audiocpp_server*",
            "node/**/node*",
        ):
            for path in unsloth_home.glob(pattern):
                if path.is_file():
                    binaries[str(path.relative_to(unsloth_home))] = round(path.stat().st_mtime, 3)
    state["binaries"] = binaries
    return state


def diff(before: dict, after: dict) -> list[str]:
    return [key for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]


def expected_prebuilt_answers(log: str) -> int:
    """Prebuilt answers expected: llama.cpp, whisper.cpp unless unpaired, audio.cpp if its installer ran."""
    return (1 if WHISPER_UNPAIRED.search(log) else 2) + (1 if AUDIO_CPP_STEP.search(log) else 0)


@pytest.fixture(scope = "session")
def settled(install, tmp_path_factory):
    """One update, so whatever the install left half-decided is decided. Everything
    below measures the update AFTER this one."""
    directory = tmp_path_factory.mktemp("idempotency")
    run = run_update(directory, "run1-settle")
    assert run.rc == 0, f"the first update failed:\n{run.log[-8000:]}"
    return directory, snapshot(install)


def test_a_second_update_changes_nothing_on_disk(install, settled):
    directory, before = settled
    run = run_update(directory, "run2-noop")
    assert run.rc == 0, run.log[-8000:]
    after = snapshot(install)
    assert diff(before, after) == [], "a no-op update rewrote part of the install:\n" + "\n".join(
        f"  {key}:\n    before={before.get(key)!r}\n    after={after.get(key)!r}"
        for key in diff(before, after)
    )
    # A download parked in an npm/uv cache leaves the snapshot alone.
    for host in PAYLOAD_HOSTS:
        assert run.connections_to(host) == 0, f"{host}: {run.report()}"
        assert run.bytes_from(host) == 0, f"{host}: {run.report()}"
    assert_read_metadata_but_no_payload(run)


def test_a_second_update_downloads_no_payload(install, settled):
    """The measurement the timings cannot make: a fast run and a run that refetched
    three release manifests look the same on a warm cache."""
    directory, _ = settled
    run = run_update(directory, "run3-network")
    assert run.rc == 0, run.log[-8000:]
    for host in PAYLOAD_HOSTS:
        # Connections, not only bytes: a failed attempt is still a fetch the update decided to make.
        assert run.connections_to(host) == 0, (
            f"{host} was contacted {run.connections_to(host)} time(s) by an update with "
            f"nothing to do: {run.report()}"
        )
        assert run.bytes_from(host) == 0, (
            f"{host} served {run.bytes_from(host)} bytes to an update with nothing to "
            f"do: {run.report()}"
        )
    assert_read_metadata_but_no_payload(run)


def test_a_second_local_update_reuses_everything_it_can(install, settled):
    """--local always reruns the dependency pass, so it shows the bootstrap, sidecar and prebuilt skips."""
    directory, _ = settled
    run = run_update(directory, "run4-local", local = True)
    assert run.rc == 0, run.log[-8000:]
    # The full message: several other steps end in the same suffix.
    assert "pip bootstrap (satisfied, skipped)" in run.log, (
        "the pip bootstrap reinstalled pip on a venv that already had one:\n" + run.log[-8000:]
    )
    assert FRONTEND_CURRENT_MARKER.search(run.log), (
        "the frontend was rebuilt on a checkout whose dist was already current:\n" + run.log[-8000:]
    )
    assert run.log.count("sidecar current") == 3, (
        "a settled transformers sidecar was rebuilt:\n" + run.log[-8000:]
    )
    # One line each from llama.cpp and whisper.cpp: a single match would let one re-validate.
    assert run.log.count("prebuilt up to date") >= expected_prebuilt_answers(run.log), (
        "a prebuilt was re-validated instead of answered from its marker:\n" + run.log[-8000:]
    )
    assert "falling back to source build" not in run.log
    for host in PAYLOAD_HOSTS:
        assert run.connections_to(host) == 0, f"{host}: {run.report()}"
        assert run.bytes_from(host) == 0, f"{host}: {run.report()}"
    assert_read_metadata_but_no_payload(run)
    # uv resolves the zoo overlay's git ref through the API on --local.
    assert run.connections_to("api.github.com") <= MAX_API_GITHUB_LOCAL, run.report()
    assert run.connections_to("github.com") <= MAX_GITHUB_LOCAL, run.report()


def test_the_desktop_update_path_keeps_a_verified_install_offline(install, settled):
    """Offline update with UV_OFFLINE and a 403ing proxy must succeed with zero successful connections."""
    directory, before = settled
    run = run_update(directory, "run6-offline", local = False, offline = True, refuse = True)
    assert run.rc == 0, (
        "an update with nothing to do failed offline. That is the state a user on a "
        f"plane, or behind a corporate proxy, is in:\n{run.log[-8000:]}"
    )
    # Invoke-RestMethod ignores HTTPS_PROXY, so Windows reaches PyPI: offline keep, up to date, or
    # a newer release are all valid. The teeth: nothing passed the proxy, nothing on disk moved.
    assert (
        "keeping the verified install" in run.log
        or UPTODATE_MARKER.search(run.log)
        or UPGRADE_MARKER in run.log
    ), run.log[-8000:]
    assert run.connections == run.refused, run.report()
    assert diff(before, snapshot(install)) == []


def _replace_bytes(path: pathlib.Path, data: bytes) -> None:
    """Write *data* to a NEW inode at *path*, so a hardlinked original is left alone."""
    path.unlink()
    path.write_bytes(data)


def _sidecar_dirs() -> list[pathlib.Path]:
    return [
        path
        for path in (
            _studio_home() / name for name in (".venv_t5_530", ".venv_t5_550", ".venv_t5_510")
        )
        if path.is_dir()
    ]


def test_a_truncated_sidecar_file_rebuilds_only_that_sidecar(install, settled):
    directory, before = settled
    sidecars = _sidecar_dirs()
    if not sidecars:
        pytest.skip("no transformers sidecars in this install")
    target = sidecars[0]
    victim = next(
        (p for p in sorted(target.rglob("*.py")) if p.stat().st_size > 0),
        None,
    )
    assert victim is not None, f"no file to damage in {target}"
    saved = victim.read_bytes()
    # Replaced, not truncated: uv hardlinks --target trees from its cache.
    _replace_bytes(victim, b"")
    try:
        run = run_update(directory, "fault-sidecar", local = True)
        assert run.rc == 0, run.log[-8000:]
        # Judged before the cleanup, which would restore the bytes and advance the mtime.
        restored = victim.read_bytes() if victim.is_file() else b""
        assert restored != b"", "the update exited 0 and left the truncated sidecar file as it was"
        assert (
            restored == saved
        ), "the rebuilt sidecar file differs from the bytes the settled install had"
    finally:
        if victim.is_file() and victim.read_bytes() == b"":
            _replace_bytes(victim, saved)
    after = snapshot(install)
    changed = diff(before, after)
    assert target.name in changed, f"the damaged sidecar was not rebuilt (changed: {changed})"
    unexpected = sorted(set(changed) - {target.name, "manifest"})
    assert unexpected == [], f"the sidecar repair changed more than the damaged sidecar: {changed}"


def test_a_deleted_manifest_re_runs_the_pass_and_changes_nothing(install, settled):
    """The manifest is the only record of what the last pass did. Without it every step
    must run -- and every step must then find its work already done, so the venv comes
    out identical."""
    directory, _ = settled
    manifest = install.parent.parent / "unsloth_install_manifest.json"
    saved = manifest.read_bytes()
    # Baseline taken here: the fault tests before this one legitimately move records.
    before = snapshot(install)
    manifest.unlink()
    try:
        run = run_update(directory, "fault-manifest", local = True)
        assert run.rc == 0, run.log[-8000:]
        assert manifest.is_file(), "the pass exited 0 without rewriting the manifest"
    finally:
        if not manifest.is_file():
            manifest.write_bytes(saved)
    after = snapshot(install)
    assert (
        after["distributions"] == before["distributions"]
    ), "a pass with no evidence resolved to a different set of packages"
    assert after["manifest"] is not None, "the pass did not rewrite the manifest"
    untouched = [key for key in diff(before, after) if key not in ("manifest", "dist_records")]
    assert untouched == [], (
        "a pass with no evidence redid work on already-valid components: " + ", ".join(untouched)
    )
    was = {tuple(record) for record in before["dist_records"]}
    moved = sorted(record[0] for record in after["dist_records"] if tuple(record) not in was)
    assert moved == [], "a pass with no evidence reinstalled: " + ", ".join(moved)


def test_a_missing_llama_binary_makes_the_marker_check_decline(install, settled):
    """A missing llama-server must fail the marker check: the pre-check tests existence, not content."""
    directory, before = settled
    candidates = sorted(_unsloth_home().glob("llama.cpp/**/llama-server*"))
    victim = next((p for p in candidates if p.is_file() and p.stat().st_size > 1024), None)
    if victim is None:
        pytest.skip("no llama.cpp prebuilt in this install")
    saved = victim.read_bytes()
    mode = victim.stat().st_mode
    # The pre-check accepts EITHER path, so removing one leaves it answering from the other.
    sibling = _unsloth_home() / "llama.cpp" / "llama-server"
    sibling_target = (
        os.readlink(sibling)
        if sibling.is_symlink()
        else (sibling.read_bytes() if sibling.is_file() else None)
    )
    sibling_was_symlink = sibling.is_symlink()
    victim.unlink()
    if sibling_target is not None and sibling != victim:
        sibling.unlink()
    try:
        run = run_update(directory, "fault-llama", local = True)
        assert run.rc == 0, run.log[-8000:]
        assert run.largest_transfer_from(PREBUILT_METADATA_HOST) > PREBUILT_METADATA_CEILING, (
            "a missing runtime binary was still answered from the marker -- nothing larger "
            "than the release metadata every update reads came back: " + run.report()
        )
    finally:
        if not victim.exists():
            victim.write_bytes(saved)
        if sibling_target is not None and not sibling.exists():
            if sibling_was_symlink:
                os.symlink(sibling_target, sibling)
            else:
                sibling.write_bytes(sibling_target)
        victim.chmod(mode)
    assert snapshot(install)["distributions"] == before["distributions"]


def test_the_manifest_records_the_evidence_the_next_run_needs(install, settled):
    """Every skip depends on the manifest holding evidence; a pass recording none makes updates redo all."""
    _directory, _before = settled
    manifest_path = install.parent.parent / "unsloth_install_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    assert manifest is not None, "the install finished without a manifest"
    inputs = manifest.get("pass_inputs")
    assert isinstance(inputs, dict) and inputs, "no pass_inputs recorded"
    assert all(isinstance(v, str) and len(v) == 64 for v in inputs.values()), inputs
    steps = manifest.get("step_results")
    assert isinstance(steps, dict) and steps, "no step_results recorded"
    for key in ("studio.txt", "base.txt"):
        assert key in inputs, f"{key} is not in {sorted(inputs)}"
    assert manifest.get("installer_python_tag"), manifest.keys()


def _known_unmet_names(state: dict) -> set[str]:
    """Unmet names recorded for this exact installed set; a record for another set is ignored."""
    manifest = state.get("manifest") or {}
    # A record written against a different installed set says nothing about this one.
    digest = state.get("installed_index_digest")
    if not digest or manifest.get("known_unmet_index") != digest:
        return set()
    record = manifest.get("known_unmet") or {}
    names: set[str] = set()
    for entries in record.values() if isinstance(record, dict) else ():
        for entry in entries or ():
            text = str(entry).strip()
            if text and not text.startswith("<"):
                names.add(_dist_name(text.split()[0]))
    return names


def _dist_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _assert_same_distributions(before: dict, after: dict, message: str) -> None:
    """A known_unmet dist may legitimately differ, since the last step resolving it decides its version."""
    torn = _known_unmet_names(before) & _known_unmet_names(after)
    # Counter, not set: a duplicate dist-info for an exempted name must not be hidden.
    before_names = collections.Counter(_dist_name(n) for n, _ in before["distributions"])
    after_names = collections.Counter(_dist_name(n) for n, _ in after["distributions"])
    assert (
        before_names == after_names
    ), f"{message}: a distribution record was added or removed: {(before_names - after_names) + (after_names - before_names)}"
    assert [d for d in after["distributions"] if _dist_name(d[0]) not in torn] == [
        d for d in before["distributions"] if _dist_name(d[0]) not in torn
    ], f"{message} (known_unmet on both sides, not compared: {sorted(torn)})"


def test_full_deps_forces_every_step_and_still_changes_nothing(install, settled):
    """The escape hatch. It must do the work -- no "(satisfied, skipped)" anywhere --
    and arrive at the same venv, which is what makes the skips safe."""
    directory, before = settled
    env_backup = os.environ.get("UNSLOTH_STUDIO_FULL_DEPS")
    os.environ["UNSLOTH_STUDIO_FULL_DEPS"] = "1"
    try:
        run = run_update(directory, "fault-full-deps", local = True)
    finally:
        if env_backup is None:
            os.environ.pop("UNSLOTH_STUDIO_FULL_DEPS", None)
        else:
            os.environ["UNSLOTH_STUDIO_FULL_DEPS"] = env_backup
    assert run.rc == 0, run.log[-8000:]
    assert "(satisfied, skipped)" not in run.log, (
        "UNSLOTH_STUDIO_FULL_DEPS still skipped a step:\n" + run.log[-8000:]
    )
    after = snapshot(install)
    _assert_same_distributions(
        before,
        after,
        "doing every step produced a different venv from skipping the settled ones, so "
        "at least one skip was not equivalent to the work it replaced",
    )


def test_the_install_is_left_working(install, settled):
    """Runs last. The fault cases legitimately move mtimes -- a rebuilt sidecar is a
    rebuilt sidecar -- so this asserts what has to be true anyway: the same packages are
    installed, and the install reports itself complete."""
    _directory, before = settled
    _assert_install_working(install, before)


def _assert_install_working(
    install: pathlib.Path,
    before: dict,
    *,
    same_distributions: bool = True,
) -> None:
    """After an upgrade only the checkout's own packages must match; transitive moves may stay."""
    after = snapshot(install)
    if same_distributions:
        _assert_same_distributions(before, after, "the install's packages changed")
    else:
        core = lambda dists: sorted(d for d in dists if d[0] in LOCAL_CORE)  # noqa: E731
        assert core(after["distributions"]) == core(before["distributions"])
        assert core(after["distributions"]), "the checkout's own packages are gone"
    assert after["manifest"] is not None
    verify_env = {**os.environ, "HOME": str(_home()), "USERPROFILE": str(_home())}
    # The same root the measured runs used, or the CLI verifies a default one.
    if os.environ.get("UNSLOTH_IDEMPOTENCY_STUDIO_HOME"):
        verify_env["UNSLOTH_STUDIO_HOME"] = os.environ["UNSLOTH_IDEMPOTENCY_STUDIO_HOME"]
    verify = subprocess.run(
        [str(install), "-I", "-X", "utf8", "-m", "unsloth_cli", "studio", "verify-install"],
        env = verify_env,
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_the_harness_measures_a_real_proxy(tmp_path):
    """Exercises a refused CONNECT and an allowed one, since a silent proxy would pass every assertion."""
    import http.server
    import socket
    import threading
    import urllib.error
    import urllib.request
    from urllib.parse import urlsplit

    body = b"x" * 4096

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    upstream = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target = upstream.serve_forever, daemon = True).start()
    module = _load_proxy_module()
    process, url, log_path = _start_proxy(tmp_path, "--deny-hosts", "pypi.org")
    try:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": url, "https": url})
        )
        # A refused CONNECT is a URLError; urllib gets no tunnel to raise an HTTP status.
        with pytest.raises(urllib.error.URLError) as excinfo:
            opener.open("https://pypi.org/simple/", timeout = 30)
        assert "403" in str(excinfo.value)
        # The proxy journals before answering, so only an in-flight write is absorbed.
        if not _wait_until(lambda: module.summary(str(log_path))["connections"] >= 1):
            pytest.fail(f"the proxy answered 403 but never journalled the connection: {log_path}")
        proxy = urlsplit(url)
        port = upstream.server_address[1]
        with socket.create_connection((proxy.hostname, proxy.port), timeout = 30) as tunnel:
            tunnel.sendall(
                f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode()
            )
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = tunnel.recv(4096)
                assert chunk, "the proxy closed the tunnel before answering CONNECT"
                head += chunk
            assert head.split(b"\r\n", 1)[0].split()[1] == b"200", head
            tunnel.sendall(b"GET / HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
            received = b""
            while True:
                chunk = tunnel.recv(65536)
                if not chunk:
                    break
                received += chunk
        assert received.endswith(body), received[:200]
        counted = (
            lambda: module.summary(str(log_path))["by_host"]
            .get("127.0.0.1", {})
            .get("bytes_down", 0)
        )
        if not _wait_until(lambda: counted() >= len(body)):
            pytest.fail(
                f"the proxy relayed {len(received)} bytes but journalled {counted()} for the host: {log_path}"
            )
    finally:
        process.terminate()
        process.wait(timeout = 10)
        upstream.shutdown()
    summary = module.summary(str(log_path))
    assert summary["connections"] == 2 and summary["refused"] == 1
    assert summary["by_host"]["127.0.0.1"]["bytes_down"] >= len(body)
    assert summary["by_host"]["pypi.org"]["bytes_down"] == 0


# Last: this may be a real upgrade, so anything after would measure the released package.


def test_the_desktop_update_path_does_no_network_work(install, settled):
    """Desktop update is judged only when its version check short-circuited; a real upgrade may fetch."""
    directory, _ = settled
    before = snapshot(install)
    # Read BEFORE the run: after it the installed version may already be PyPI's.
    installed, latest = _installed_version(install), _pypi_latest()
    print(f"[idempotency] installed={installed!r} pypi={latest!r}", flush = True)
    run = run_update(directory, "run5-desktop", local = False)
    assert run.rc == 0, run.log[-8000:]
    took_fast_path = NO_WORK_MARKERS[0] in run.log
    if not took_fast_path:
        # Restore the checkout first: later workflow steps expect the code under test.
        restore = run_update(directory, "run5-restore", local = True)
        assert restore.rc == 0, restore.log[-8000:]
        _assert_install_working(install, before, same_distributions = UPGRADE_MARKER not in run.log)
        # Use the measured run's own log: a separate lookup can see another release.
        if UPGRADE_MARKER in run.log:
            pytest.skip(
                "installed version is not PyPI's latest; the desktop no-op path was not taken"
            )
        if PYPI_UNREACHABLE_MARKER in run.log:
            # Only the measured run's own lookup counts.
            pytest.skip("PyPI was unreachable; the desktop no-op path could not be judged")
        pytest.fail(
            f"the desktop update ran the dependency pass with unsloth {installed!r} installed "
            f"and {latest!r} on PyPI; the fast path was not taken:\n{run.log[-8000:]}"
        )
    for host in PAYLOAD_HOSTS:
        assert run.connections_to(host) == 0, f"{host}: {run.report()}"
        assert run.bytes_from(host) == 0, f"{host}: {run.report()}"
    assert run.bytes_from("raw.githubusercontent.com") <= ICON_FETCH_CEILING, run.report()
    for marker in NO_WORK_MARKERS:
        assert marker in run.log, f"{marker!r} missing from a no-op update:\n{run.log[-8000:]}"
    assert run.log.count("prebuilt up to date") >= expected_prebuilt_answers(run.log), run.log[
        -8000:
    ]
    assert run.log.count("sidecar current") == 3, run.log[-8000:]
    assert run.connections_to("pypi.org") <= 2, run.report()
    assert run.connections_to("github.com") <= MAX_GITHUB_DESKTOP, run.report()
    # Zero everywhere except macOS llama.cpp, which lacks the download-host fast path.
    assert run.connections_to("api.github.com") <= MAX_API_GITHUB_DESKTOP, run.report()
    assert_read_metadata_but_no_payload(run)
    assert diff(before, snapshot(install)) == []


# Verbatim from the Update CI log on 2026-09-24, run between the two publishes.
UNPAIRED_SH = (
    "  whisper.cpp    no compatible prebuilt (installed llama.cpp b11139-mix-a6922cc; whisper "
    "requires b11115-mix-a6922cc); curated whisper.cpp dictation is unavailable"
)


def test_an_unpaired_whisper_release_lowers_the_count_to_llama_alone():
    assert expected_prebuilt_answers(UNPAIRED_SH) == 1


def test_without_the_unpaired_line_both_prebuilts_must_answer():
    # The count of two catches a whisper.cpp that re-validated or went missing.
    assert expected_prebuilt_answers("  llama.cpp      prebuilt up to date") == 2


def test_an_installed_audio_cpp_owes_one_more_answer():
    log = "  llama.cpp      prebuilt up to date\n  whisper.cpp    prebuilt up to date\n"
    assert expected_prebuilt_answers(log + "  audio.cpp      prebuilt up to date\n") == 3
    assert expected_prebuilt_answers(log + "  audio.cpp      prebuilt installed\n") == 3
    # "audio.cpp" inside another line (a path, a skip substep) is not its step line.
    assert expected_prebuilt_answers(log + "  [verbose] audio.cpp: install skipped\n") == 2


def test_the_audio_cpp_step_pattern_matches_both_installers():
    root = pathlib.Path(__file__).resolve().parents[3]
    for script in ("studio/setup.sh", "studio/setup.ps1"):
        text = (root / script).read_text(encoding = "utf-8")
        assert 'step "audio.cpp" "prebuilt up to date"' in text, script
        assert 'step "audio.cpp" "update unavailable, existing prebuilt kept"' in text, script


def test_the_unpaired_pattern_matches_both_installers():
    root = pathlib.Path(__file__).resolve().parents[3]
    for script in ("studio/setup.sh", "studio/setup.ps1"):
        text = (root / script).read_text(encoding = "utf-8")
        assert re.search(r'step "whisper\.cpp" "no compatible prebuilt \(', text), script
