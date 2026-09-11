# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""End to end: a second `unsloth studio update` must do no network work and change
nothing on disk.

The unit half of this lives in test_dependency_pass_skips.py and in the prebuilt
installers' own suites. It cannot answer the question this file asks, because every skip
is conditional on evidence that only a real install produces: a manifest written by a
real dependency pass, sidecar trees with real RECORD files, prebuilt markers whose
hashes describe binaries that are really on disk.

Two things make the claim measurable rather than plausible:

  * The child's ONLY route out is a logging proxy (idempotency_proxy.py). Every
    connection attempt is recorded whether it succeeds, so "did no network work" is a
    byte count per host, not an impression of speed. A warm uv cache also finishes fast.
  * The install is snapshotted before and after. "Nothing changed" means a sorted
    distribution list, the manifest minus its timestamp, the prebuilt markers byte for
    byte, the uv cache marker, the no-torch marker, each sidecar's file count and mtime,
    and the mtime of every runtime binary.

Opt in, because it needs a real install:

    UNSLOTH_IDEMPOTENCY_E2E=1 pytest tests/studio/install/test_update_idempotency.py

By default it exercises the install under $HOME. UNSLOTH_IDEMPOTENCY_HOME points it at
an isolated one instead, which is how it is run locally against a scratch install:

    UNSLOTH_IDEMPOTENCY_E2E=1 UNSLOTH_IDEMPOTENCY_HOME=/path/to/fake/home pytest ...

Two update modes are measured, because they take different paths and only one of them
is the flow a desktop user gets:

  * `studio update` (no --local) is the desktop flow. When the installed version equals
    PyPI's latest it takes setup.sh's fast path, skips the dependency pass entirely, and
    the prebuilt pre-checks answer from their markers. This is where the byte counts are
    asserted. It only runs when the two versions do agree; otherwise a "no-op" update
    would legitimately be an upgrade, so those cases skip themselves rather than measure
    an upgrade and call it idempotency.
  * `studio update --local` reinstalls from the checkout, so the dependency pass always
    runs, and it runs WITHOUT last run's evidence: a checkout is a development install
    shape, and the pass refuses to skip steps on a manifest that may not describe the
    tree (install_python_stack._plan_pass). What it demonstrates is the rest: the pip
    bootstrap skip, the sidecars answering "current", the prebuilts answering from their
    markers, and that a full pass over a settled venv still downloads nothing. It is
    asserted from the LOG as well as from bytes: with a warm uv cache a full pass also
    downloads nothing, so bytes cannot tell a skip from a re-resolve there. A full pass
    has some churn of its own, named in FULL_PASS_CHURN; everything else must hold still.

The fault-injection cases DAMAGE that install and expect the update to repair exactly
the damaged part. They restore what they broke, but a failure mid-case can leave the
install in the damaged state, so never point this at an install you care about.

Reading a CI failure: run 1 is the SETTLE run and is deliberately not measured. The
first update performed by new installer code legitimately rewrites the manifest --
`step_results` and `pass_inputs` are recorded by the pass that introduces them, so an
install laid down by an older build has neither -- and it may repair torchao, which an
earlier release installed from the wrong index. Only runs 2 and later are asserted on.
A staging run confirmed the third (offline) run changed nothing at all, which is the
claim this branch makes; a diff reported against `run1-settle` is not that claim.
"""

from __future__ import annotations

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
# Each run's update log and proxy journal: pytest's tmp dir, or UNSLOTH_IDEMPOTENCY_ARTIFACTS to
# keep them (a failure is unreadable without them).
ARTIFACTS = os.environ.get("UNSLOTH_IDEMPOTENCY_ARTIFACTS", "")

# Hosts a no-op update must not touch. pypi.org and github.com are bounded separately (a version
# check and a latest-release HEAD are legitimate); these five are where the megabytes are.
PAYLOAD_HOSTS = (
    "files.pythonhosted.org",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "nodejs.org",
)
# NOT raw.githubusercontent.com: install.sh's shortcut refresh fetches rounded-512.png from it on an
# editable install with no built frontend (~8 KB); a bound keeps it from growing.
ICON_FETCH_CEILING = 64 * 1024

# What the installers print when they decline to do work (setup.sh consumes the prebuilt installers'
# "already matches" and prints its own line).
NO_WORK_MARKERS = ("dependencies up to date", "prebuilt up to date", "sidecar current")
# setup.sh's column-padded frontend line: a rebuild on a warm npm cache talks to no forbidden host,
# so the log line is the only witness.
FRONTEND_CURRENT_MARKER = re.compile(r"frontend\s+up to date")
# What setup.sh / setup.ps1 print when the version check found a newer release, and when
# it could not ask PyPI at all (the pass runs on purpose in both cases).
UPGRADE_MARKER = "available, updating..."
# What both shells print (step "python", column-padded) when PyPI's latest equals the
# installed version and the dependency pass is skipped for it.
UPTODATE_MARKER = re.compile(r"python\s+\S+ \S+ is up to date")
PYPI_UNREACHABLE_MARKER = "could not reach PyPI, updating to be safe..."
# What --local installs from the checkout on every pass, so its RECORD moving is expected.
LOCAL_CORE = frozenset({"unsloth", "unsloth-zoo", "unsloth_zoo"})
# What a FULL pass (no evidence: every --local pass, the pass after a deleted manifest) reinstalls
# at the same version: the two path-installed seed plugins, and click, caught between sqlfluff<4
# (click<=8.3.0) and huggingface-hub 1.23+ (click>=8.4.2) and recorded as known_unmet. A version
# CHANGE is still seen, since the distribution list is compared separately.
FULL_PASS_CHURN = LOCAL_CORE | frozenset(
    {"data-designer-github-repo-seed", "data-designer-unstructured-seed", "click"}
)

DIST_LIST = (
    "import importlib.metadata as m, json; "
    "print(json.dumps(sorted(((d.metadata['Name'] or '').lower(), d.version) "
    "for d in m.distributions())))"
)
# Names and versions cannot see a same-version reinstall; the RECORD mtime can. Per distribution, so
# a failure names the package.
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


# ── the install under test ──


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


# ── the proxy ──


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
    """Wait for the proxy's journal to stop growing, before the proxy is terminated.

    Returns False when a worker was still between accept and its record after
    `active_grace` seconds, longer than the proxy's 60 s upstream connect timeout: the
    journal is then incomplete and the caller must not read it as a zero-connection proof.

    A worker appends its record once the connection it describes has closed, so the child
    can exit -- or a client can see its response -- with records still in flight, and
    terminating the proxy at that moment drops them. A dropped record is a connection this
    harness would then report as never having happened, which is precisely the claim it
    exists to make. Waits for quiescence rather than for a count, because how many
    connections a run makes is the thing being measured.
    """
    active_path = log_path.with_name(log_path.name + ".active")

    def _workers_active() -> bool:
        # A worker blocked in an upstream connect has written nothing: a quiet journal alone is not
        # proof.
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
            # A journal still growing past the deadline with no worker in flight is bounded here.
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
    # The venv pytest runs in is not the venv under test: anything pointing the child at this
    # interpreter's environment has to go.
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
        # npm_config_https_proxy outranks HTTPS_PROXY (npm 11): an inherited value would route the
        # registry traffic around the proxy that counts it.
        npm_config_proxy = url,
        npm_config_https_proxy = url,
        npm_config_noproxy = "127.0.0.1,localhost",
        # Presence, not truthiness: without it the CLI reads an Invoke-WebRequest proxy default from
        # the PowerShell profiles, which would outrank the variables above.
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
        # Read first: a proxy that died mid-run journalled nothing after, and a CONNECT refused by a
        # dead listener looks like no connection at all.
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


# ── the snapshot ──


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
    # FULL_PASS_CHURN moves its RECORD on every full pass: mtime and size are not compared, name and
    # version still are.
    state["dist_records"] = [
        [name, version, None, None] if name in FULL_PASS_CHURN else [name, version, mtime, size]
        for name, version, mtime, size in json.loads(records.stdout or "[]")
    ]

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
            "node/**/node*",
        ):
            for path in unsloth_home.glob(pattern):
                if path.is_file():
                    binaries[str(path.relative_to(unsloth_home))] = round(path.stat().st_mtime, 3)
    state["binaries"] = binaries
    return state


def diff(before: dict, after: dict) -> list[str]:
    return [key for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]


# ── run 1 and run 2 ──


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
    # The same run's traffic too: a download parked in an npm or uv cache leaves the snapshot alone
    # and is quiet by the network case that follows.
    for host in PAYLOAD_HOSTS:
        assert run.connections_to(host) == 0, f"{host}: {run.report()}"
        assert run.bytes_from(host) == 0, f"{host}: {run.report()}"


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


def test_a_second_local_update_reuses_everything_it_can(install, settled):
    """--local is the CI and developer path. It always runs the dependency pass, without
    last run's evidence (a checkout is a development install shape, and a checkout can
    change in ways no requirements digest sees), so what it demonstrates is the rest:
    the pip bootstrap skip, all three sidecars answering "current" without a rebuild,
    and both prebuilts answering from their markers."""
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
    assert run.log.count("prebuilt up to date") >= 2, (
        "a prebuilt was re-validated instead of answered from its marker:\n" + run.log[-8000:]
    )
    assert "falling back to source build" not in run.log
    for host in PAYLOAD_HOSTS:
        assert run.connections_to(host) == 0, f"{host}: {run.report()}"
        assert run.bytes_from(host) == 0, f"{host}: {run.report()}"
    # The release is never listed: the marker checks cost one HEAD per prebuilt.
    assert run.connections_to("api.github.com") == 0, run.report()


# ── offline ──


def test_the_desktop_update_path_keeps_a_verified_install_offline(install, settled):
    """UV_OFFLINE plus a proxy that 403s everything. Every attempt is still recorded, so
    this asserts zero SUCCESSFUL connections, not merely zero bytes.

    Non-local only, and not because --local is uninteresting: --local re-overlays
    `unsloth-zoo @ git+main` on every run by design, so it cannot complete without a
    network on any branch of this code, and a run that fails mid-pass leaves no manifest
    behind. The flow a user is in on a plane is this one.

    Without the rule this PR adds, an unreachable PyPI means "update to be safe", which
    offline can only fail: the pass starts, the first git requirement 403s, and the
    update exits non-zero on an install that was already complete.
    """
    directory, before = settled
    run = run_update(directory, "run6-offline", local = False, offline = True, refuse = True)
    assert run.rc == 0, (
        "an update with nothing to do failed offline. That is the state a user on a "
        f"plane, or behind a corporate proxy, is in:\n{run.log[-8000:]}"
    )
    # Either the offline rule or the ordinary fast path: Invoke-RestMethod ignores HTTPS_PROXY, so
    # on Windows the version check can still answer "up to date". Either way nothing got through the
    # proxy and nothing on disk moved.
    assert "keeping the verified install" in run.log or UPTODATE_MARKER.search(run.log), run.log[
        -8000:
    ]
    assert run.connections == run.refused, run.report()
    assert diff(before, snapshot(install)) == []


# Fault injection: each case damages one thing and asserts the update repairs THAT thing, and where
# cheap to check, nothing else.


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
    # Replaced, not truncated in place: uv hardlinks --target trees from its cache (NTFS too), so an
    # in-place truncation would poison the cache and every tier with it.
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
    # Exactly the damaged component; the manifest may move, as it records the renewed evidence.
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
        # Judged before the cleanup puts the saved copy back.
        assert manifest.is_file(), "the pass exited 0 without rewriting the manifest"
    finally:
        if not manifest.is_file():
            manifest.write_bytes(saved)
    after = snapshot(install)
    assert (
        after["distributions"] == before["distributions"]
    ), "a pass with no evidence resolved to a different set of packages"
    assert after["manifest"] is not None, "the pass did not rewrite the manifest"
    # A pass with no evidence re-checks every component and must leave the valid ones alone; the
    # manifest itself is the one key allowed to differ.
    untouched = [key for key in diff(before, after) if key not in ("manifest", "dist_records")]
    assert untouched == [], (
        "a pass with no evidence redid work on already-valid components: " + ", ".join(untouched)
    )
    # Names and versions cannot see a same-version reinstall; the RECORD mtime can (FULL_PASS_CHURN
    # excepted).
    was = {tuple(record) for record in before["dist_records"]}
    moved = sorted(record[0] for record in after["dist_records"] if tuple(record) not in was)
    assert moved == [], "a pass with no evidence reinstalled: " + ", ".join(moved)


def test_a_damaged_llama_binary_makes_the_marker_check_decline(install, settled):
    """The pre-check answers "current" from hashes of the runtime binaries, so a damaged
    one has to send the update back to the release it was skipping.

    Measured on the network rather than in the log, because the log line the fast path
    produces is consumed by setup.sh and reprinted identically either way: reaching
    release-assets.githubusercontent.com at all means the release was listed, which is
    exactly the work the marker check exists to avoid.

    It asserts a DECLINE, not a repair. A truncated llama-server is not replaced by the
    full path either -- its own check is the marker fingerprint plus a tree walk, and it
    has never hashed the binaries. That is unchanged by this PR: the pre-check refuses to
    answer, and what happens next is what happened before.
    """
    directory, before = settled
    candidates = sorted(_unsloth_home().glob("llama.cpp/**/llama-server*"))
    victim = next((p for p in candidates if p.is_file() and p.stat().st_size > 1024), None)
    if victim is None:
        pytest.skip("no llama.cpp prebuilt in this install")
    saved = victim.read_bytes()
    mode = victim.stat().st_mode
    if IS_WINDOWS:
        # Damage the Windows validator detects too: it has no image-reading preflight.
        victim.unlink()
    else:
        victim.write_bytes(saved[: len(saved) // 4])
    try:
        run = run_update(directory, "fault-llama", local = True)
        assert run.rc == 0, run.log[-8000:]
        assert run.connections_to("release-assets.githubusercontent.com") > 0, (
            "a damaged runtime binary was still answered from the marker: " + run.report()
        )
    finally:
        victim.write_bytes(saved)
        victim.chmod(mode)
    assert snapshot(install)["distributions"] == before["distributions"]


def test_the_manifest_records_the_evidence_the_next_run_needs(install, settled):
    """Every skip is conditional on this. A pass that finished but recorded nothing is
    not a bug anyone would notice until the NEXT update quietly redoes everything -- or,
    worse, skips on a key it happens to find and cannot check.

    Not a test that an edited requirements file re-runs its step: --local, which is what
    this harness and CI install with, never skips anything (a checkout can change in ways
    no requirements digest sees), so there would be nothing to observe. That gate is
    covered by tests/studio/install/test_dependency_pass_skips.py.
    """
    _directory, _before = settled
    # The manifest on disk now: the previous case regenerated it, and that copy is what the next
    # update's skips read.
    manifest_path = install.parent.parent / "unsloth_install_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    assert manifest is not None, "the install finished without a manifest"
    inputs = manifest.get("pass_inputs")
    assert isinstance(inputs, dict) and inputs, "no pass_inputs recorded"
    assert all(isinstance(v, str) and len(v) == 64 for v in inputs.values()), inputs
    steps = manifest.get("step_results")
    assert isinstance(steps, dict) and steps, "no step_results recorded"
    # The keys a later run gates on, spelled the same way it will look them up.
    for key in ("studio.txt", "base.txt"):
        assert key in inputs, f"{key} is not in {sorted(inputs)}"
    assert manifest.get("installer_python_tag"), manifest.keys()


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
    assert after["distributions"] == before["distributions"], (
        "doing every step produced a different venv from skipping the settled ones, so "
        "at least one skip was not equivalent to the work it replaced"
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
    """The same packages as *before* are installed, and the CLI reports the install
    complete. Shared by the last ordered case and the desktop case's restore, which is
    the last product operation of the run and is otherwise judged by exit code alone.

    After a real upgrade to PyPI's release the restore reinstalls the checkout only where
    needed, and a transitive dependency the release moved that the checkout's ranges
    still admit legitimately stays: then only the checkout's own packages are held to
    *before*."""
    after = snapshot(install)
    if same_distributions:
        assert after["distributions"] == before["distributions"]
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
    """A proxy that silently failed to start would make every assertion above pass, and
    so would one that relays a tunnel without counting its bytes: the byte ceilings are
    the only bound on hosts that are allowed through. So both halves are exercised: a
    refused CONNECT, and an allowed one carrying a response of known size."""
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
        # A refused CONNECT is a URLError: urllib never gets a tunnel to raise an HTTP status
        # through.
        with pytest.raises(urllib.error.URLError) as excinfo:
            opener.open("https://pypi.org/simple/", timeout = 30)
        assert "403" in str(excinfo.value)
        # The proxy journals before it answers (Proxy.handle's refused branch), so only an in-flight
        # write is absorbed here; a proxy that stopped journalling must fail loudly.
        if not _wait_until(lambda: module.summary(str(log_path))["connections"] >= 1):
            pytest.fail(f"the proxy answered 403 but never journalled the connection: {log_path}")
        # An allowed tunnel by hand: CONNECT, then a GET through it; the bytes must match the
        # journal.
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


# The desktop path, last: when the installed version is not PyPI's latest this is a real upgrade,
# and anything measured after it would be measuring the released package.


def test_the_desktop_update_path_does_no_network_work(install, settled):
    """No --local: the flow the desktop app and the Repair button run. Its whole cost on
    a settled install should be one version check and the prebuilt HEADs.

    Judged only when the version check short-circuited, read off the measured run's own
    log rather than a separate request to PyPI made earlier (which can fail or see another
    release than the one the update saw). With a version
    mismatch this run IS an upgrade to PyPI's release, whose Node, sidecar, llama.cpp and
    whisper.cpp pins can differ from the checkout's, and an upgrade that rebuilds or
    downloads those is behaving correctly; a version bump would otherwise fail this
    smoke test for doing its job. The run still has to succeed on that branch, and the
    checkout is put back for the workflow steps after this harness.
    """
    directory, _ = settled
    # Its own baseline: the fault-injection cases before it legitimately moved things.
    before = snapshot(install)
    # Read BEFORE the run: after it the installed version may already be PyPI's.
    installed, latest = _installed_version(install), _pypi_latest()
    print(f"[idempotency] installed={installed!r} pypi={latest!r}", flush = True)
    run = run_update(directory, "run5-desktop", local = False)
    assert run.rc == 0, run.log[-8000:]
    took_fast_path = NO_WORK_MARKERS[0] in run.log
    if not took_fast_path:
        # The checkout goes back first either way: later workflow steps expect the code under test.
        # Held to the same bar as the ordered cases, not its exit code alone.
        restore = run_update(directory, "run5-restore", local = True)
        assert restore.rc == 0, restore.log[-8000:]
        _assert_install_working(install, before, same_distributions = UPGRADE_MARKER not in run.log)
        # Why the pass ran, off the measured run's own log: the separate lookup can see another
        # release.
        if UPGRADE_MARKER in run.log:
            # A real upgrade to PyPI's release, whose pins may differ: nothing below can be
            # asserted.
            pytest.skip(
                "installed version is not PyPI's latest; the desktop no-op path was not taken"
            )
        if PYPI_UNREACHABLE_MARKER in run.log:
            # The pass runs on purpose when PyPI cannot be asked: nothing to judge. Only the
            # measured run's own lookup counts, or this would look away from the equal-version pass.
            pytest.skip("PyPI was unreachable; the desktop no-op path could not be judged")
        # Equal versions, a reachable index, and the pass still ran: the regression this catches.
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
    # Each component, not the generic line once.
    assert run.log.count("prebuilt up to date") >= 2, run.log[-8000:]
    assert run.log.count("sidecar current") == 3, run.log[-8000:]
    # One version check and one latest-release HEAD per prebuilt; the bound stops a full listing
    # returning.
    assert run.connections_to("pypi.org") <= 2, run.report()
    assert run.connections_to("github.com") <= 6, run.report()
    # The other half of the prebuilt claim: the release itself is never listed.
    assert run.connections_to("api.github.com") == 0, run.report()
    assert run.connections_to("release-assets.githubusercontent.com") == 0, run.report()
    assert diff(before, snapshot(install)) == []
