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
    runs. That is what exercises the per-step skips, and it is asserted from the LOG:
    with a warm uv cache a full pass also downloads nothing, so bytes cannot tell a skip
    from a re-resolve there. Both are checked; neither alone is enough.

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
import subprocess
import sys
import time

import pytest


E2E = os.environ.get("UNSLOTH_IDEMPOTENCY_E2E") == "1"

PROXY = pathlib.Path(__file__).resolve().parent / "idempotency_proxy.py"
IS_WINDOWS = sys.platform == "win32"
# Where each run's update log and proxy journal are kept. Under pytest's tmp dir by
# default, which a passing run discards; point UNSLOTH_IDEMPOTENCY_ARTIFACTS at a real
# directory to keep them (CI uploads them, and a failure is unreadable without them).
ARTIFACTS = os.environ.get("UNSLOTH_IDEMPOTENCY_ARTIFACTS", "")

# Hosts a no-op update must not touch at all. pypi.org and github.com are listed
# separately because a version check and a "what is the latest release" HEAD are both
# legitimate, and small; these five are where the megabytes are.
PAYLOAD_HOSTS = (
    "files.pythonhosted.org",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "nodejs.org",
)
# NOT raw.githubusercontent.com. install.sh's desktop-shortcut refresh falls back to
# downloading rounded-512.png from it when it cannot find the icon in the installed
# frontend, which is the case for an editable --local install with no built frontend.
# ~8 KB, and it does not happen for the wheel installs users have -- but a bound keeps
# it from quietly becoming something bigger.
ICON_FETCH_CEILING = 64 * 1024

# What the installers print when they decline to do work. "already matches" is what the
# prebuilt installers log; setup.sh CONSUMES that and prints its own line, so the log a
# user (and this harness) sees carries these instead.
NO_WORK_MARKERS = ("dependencies up to date", "prebuilt up to date", "sidecar current")

DIST_LIST = (
    "import importlib.metadata as m, json; "
    "print(json.dumps(sorted(((d.metadata['Name'] or '').lower(), d.version) "
    "for d in m.distributions())))"
)
# Names and versions cannot see a reinstall at the same version, and a warm uv cache
# makes one download nothing; the dist-info RECORD is rewritten by every install, so
# its mtime can. Per distribution, so a failure names the package that moved.
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
    # The gate lives here rather than on the module so the proxy's own self test, which
    # needs nothing installed, still runs in ordinary CI. A measuring instrument that
    # quietly stopped measuring would make every assertion below pass.
    if not E2E:
        pytest.skip("opt-in end to end; set UNSLOTH_IDEMPOTENCY_E2E=1")
    venv_python = _venv_python()
    if not venv_python.is_file():
        # Opted in, and nothing to measure: a skip here would let every product-facing
        # case report itself skipped while the proxy self test keeps the job green.
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
    # The proxy appends and the summary reads the whole journal, so a retained artifacts
    # directory would attribute the previous invocation's traffic to this run.
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
) -> None:
    """Wait for the proxy's journal to stop growing, before the proxy is terminated.

    A worker appends its record once the connection it describes has closed, so the child
    can exit -- or a client can see its response -- with records still in flight, and
    terminating the proxy at that moment drops them. A dropped record is a connection this
    harness would then report as never having happened, which is precisely the claim it
    exists to make. Waits for quiescence rather than for a count, because how many
    connections a run makes is the thing being measured.
    """
    active_path = log_path.with_name(log_path.name + ".active")

    def _workers_active() -> bool:
        # The proxy publishes how many workers sit between accept and their record. A
        # worker blocked in an upstream connect has written nothing, so a quiet journal
        # alone is not proof that nothing is left to journal.
        try:
            return int(active_path.read_text().strip() or "0") > 0
        except (OSError, ValueError):
            return False

    size = -1
    stable_since = time.monotonic()
    deadline = stable_since + timeout
    while time.monotonic() < deadline:
        try:
            current = log_path.stat().st_size
        except OSError:
            current = 0
        if current != size:
            size, stable_since = current, time.monotonic()
        elif time.monotonic() - stable_since >= quiet and not _workers_active():
            return
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
    # Inherited, not replaced: the runner's PATH, SystemRoot and certificate bundle all
    # matter, and a hand-built environment here fails in ways that look like the product.
    env = dict(os.environ)
    env.pop("UNSLOTH_IDEMPOTENCY_E2E", None)
    # The venv pytest runs in is not the venv under test. Anything that would point the
    # child at this interpreter's environment has to go, or the update resolves against
    # the wrong cache, the wrong constraints, or the wrong site-packages -- and then
    # measures whatever that produced.
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
        _settle_journal(log_path)
        process.terminate()
        try:
            process.wait(timeout = 10)
        except subprocess.TimeoutExpired:  # pragma: no cover - a wedged tunnel
            process.kill()
    (directory / "update.log").write_text(text, encoding = "utf-8")
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
    state["dist_records"] = json.loads(records.stdout or "[]")

    manifest_path = venv / "unsloth_install_manifest.json"
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
        # The one key that is ALLOWED to move: it records when the pass finished, and a
        # pass that correctly did nothing still finished.
        manifest.pop("completed_at_ms", None)
    state["manifest"] = manifest

    # Byte for byte: a marker rewritten with identical content is still a rewrite, and
    # the whole point of the prebuilt pre-checks is that they do not rewrite it.
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
def desktop_path(install) -> bool:
    """Whether a non-local update can short-circuit: True only when the installed
    version is the one PyPI would hand back.

    Which EXPECTATIONS apply, not whether the case runs. The measurement a version
    mismatch invalidates is the fast path itself; everything a `studio update` must
    never do -- refetch a llama.cpp release, list a GitHub release, refetch a Node
    tarball -- holds whether or not unsloth itself is being upgraded, and dropping
    those assertions is how a regression in them reaches a release unnoticed.
    """
    installed, latest = _installed_version(install), _pypi_latest()
    print(f"[idempotency] installed={installed!r} pypi={latest!r}", flush = True)
    return bool(installed) and installed == latest


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


def test_a_second_update_downloads_no_payload(install, settled):
    """The measurement the timings cannot make: a fast run and a run that refetched
    three release manifests look the same on a warm cache."""
    directory, _ = settled
    run = run_update(directory, "run3-network")
    assert run.rc == 0, run.log[-8000:]
    for host in PAYLOAD_HOSTS:
        # Connections, not only bytes: an attempt that failed or timed out upstream is
        # still a payload fetch the update decided to make, and the installer can keep
        # the verified prebuilt and exit 0 after one.
        assert run.connections_to(host) == 0, (
            f"{host} was contacted {run.connections_to(host)} time(s) by an update with "
            f"nothing to do: {run.report()}"
        )
        assert run.bytes_from(host) == 0, (
            f"{host} served {run.bytes_from(host)} bytes to an update with nothing to "
            f"do: {run.report()}"
        )


def test_a_second_local_update_reuses_everything_it_can(install, settled):
    """--local is the CI and developer path. It always runs the dependency pass on
    purpose (a checkout can change in ways no requirements digest sees), so what it
    demonstrates is the rest: the pip bootstrap skip, all three sidecars answering
    "current" without a rebuild, and both prebuilts answering from their markers."""
    directory, _ = settled
    run = run_update(directory, "run4-local", local = True)
    assert run.rc == 0, run.log[-8000:]
    assert "(satisfied, skipped)" in run.log, (
        "the pip bootstrap reinstalled pip on a venv that already had one:\n" + run.log[-8000:]
    )
    assert run.log.count("sidecar current") == 3, (
        "a settled transformers sidecar was rebuilt:\n" + run.log[-8000:]
    )
    # One line each from llama.cpp and whisper.cpp: a single match would let one of the
    # two re-validate an installed prebuilt while the other answered from its marker.
    assert run.log.count("prebuilt up to date") >= 2, (
        "a prebuilt was re-validated instead of answered from its marker:\n" + run.log[-8000:]
    )
    assert "falling back to source build" not in run.log
    for host in PAYLOAD_HOSTS:
        assert run.connections_to(host) == 0, f"{host}: {run.report()}"
        assert run.bytes_from(host) == 0, f"{host}: {run.report()}"
    # The release is never listed on this path either: the marker checks cost one HEAD
    # on github.com per prebuilt, and a --local pass has no other business with the API.
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
    assert "keeping the verified install" in run.log, run.log[-8000:]
    assert run.connections == run.refused, run.report()
    assert diff(before, snapshot(install)) == []


# ── fault injection ──
#
# The direction that matters. A needless install costs seconds; a wrong skip ships a
# venv that dies on `import structlog`. Each case damages one thing and asserts that the
# update repairs THAT thing -- and, where it is cheap to check, that it does not decide
# to rebuild everything else while it is there.


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
    victim.write_bytes(b"")
    try:
        run = run_update(directory, "fault-sidecar", local = True)
        assert run.rc == 0, run.log[-8000:]
        # Judged before the cleanup below, which would otherwise restore the bytes
        # itself and advance the mtime the rebuild assertion reads.
        repaired = victim.is_file() and victim.read_bytes() != b""
        assert repaired, "the update exited 0 and left the truncated sidecar file as it was"
    finally:
        if victim.is_file() and victim.read_bytes() == b"":
            victim.write_bytes(saved)
    after = snapshot(install)
    changed = diff(before, after)
    assert target.name in changed, f"the damaged sidecar was not rebuilt (changed: {changed})"
    untouched = [
        name for name in (".venv_t5_530", ".venv_t5_550", ".venv_t5_510") if name != target.name
    ]
    assert [
        name for name in changed if name in untouched
    ] == [], f"an unrelated sidecar was rebuilt too: {changed}"


def test_a_deleted_manifest_re_runs_the_pass_and_changes_nothing(install, settled):
    """The manifest is the only record of what the last pass did. Without it every step
    must run -- and every step must then find its work already done, so the venv comes
    out identical."""
    directory, before = settled
    manifest = install.parent.parent / "unsloth_install_manifest.json"
    saved = manifest.read_bytes()
    manifest.unlink()
    try:
        run = run_update(directory, "fault-manifest", local = True)
        assert run.rc == 0, run.log[-8000:]
        # Judged before the cleanup below, which would otherwise put the saved copy back
        # and let the assertion after the snapshot pass on the test's own file.
        assert manifest.is_file(), "the pass exited 0 without rewriting the manifest"
    finally:
        if not manifest.is_file():
            manifest.write_bytes(saved)
    after = snapshot(install)
    assert (
        after["distributions"] == before["distributions"]
    ), "a pass with no evidence resolved to a different set of packages"
    assert after["manifest"] is not None, "the pass did not rewrite the manifest"


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
        # The Linux and macOS validators read the executable image, so a truncated
        # binary is caught; Windows has no such preflight and the marker match checks
        # that the executables exist. Damage the Windows validator detects.
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
    _directory, before = settled
    manifest = before["manifest"]
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
    after = snapshot(install)
    assert after["distributions"] == before["distributions"]
    assert after["manifest"] is not None
    verify = subprocess.run(
        [str(install), "-I", "-X", "utf8", "-m", "unsloth_cli", "studio", "verify-install"],
        env = {**os.environ, "HOME": str(_home()), "USERPROFILE": str(_home())},
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert verify.returncode == 0, verify.stdout + verify.stderr


def test_the_harness_measures_a_real_proxy(tmp_path):
    """A proxy that silently failed to start would make every assertion above pass."""
    module = _load_proxy_module()
    process, url, log_path = _start_proxy(tmp_path, "--refuse")
    try:
        import urllib.error
        import urllib.request

        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": url, "https": url})
        )
        # A refused CONNECT surfaces as URLError, not HTTPError: urllib never gets a
        # tunnel to raise an HTTP status through.
        with pytest.raises(urllib.error.URLError) as excinfo:
            opener.open("https://pypi.org/simple/", timeout = 30)
        assert "403" in str(excinfo.value)
        # The client sees the 403 off the socket; the journal is written by the proxy
        # worker. The proxy writes that record before it answers (see the refused branch
        # of Proxy.handle), so all that is left to absorb here is the write itself being
        # in flight -- but a proxy that had stopped journalling altogether must fail
        # loudly rather than be papered over, hence the deadline and the failure below.
        if not _wait_until(lambda: module.summary(str(log_path))["connections"] >= 1):
            pytest.fail(f"the proxy answered 403 but never journalled the connection: {log_path}")
    finally:
        process.terminate()
        process.wait(timeout = 10)
    summary = module.summary(str(log_path))
    assert summary["connections"] == 1 and summary["refused"] == 1


# ── the desktop path, last ──
#
# Deliberately the final case. When the installed version is not PyPI's latest this run
# is a real upgrade and mutates the session install; anything measured after it would be
# measuring the released package, and comparing it against the settled snapshot would
# fail on every version-bump commit with the offline behaviour perfectly correct.


def test_the_desktop_update_path_does_no_network_work(install, settled, desktop_path):
    """No --local: the flow the desktop app and the Repair button run. Its whole cost on
    a settled install should be one version check and the prebuilt HEADs.

    Runs on BOTH branches of desktop_path. Skipping the case when the installed version
    is not PyPI's latest also dropped the assertions that hold either way -- the release
    payload hosts, api.github.com, release-assets -- and those are the ones that catch a
    prebuilt regression regardless of whether the version check short-circuited. Only
    what genuinely depends on that short-circuit is relaxed below, each with its reason.
    """
    directory, _ = settled
    # Its own baseline, taken now: this is the last case, and the fault-injection cases
    # before it legitimately rebuilt a sidecar and advanced a binary's mtime.
    before = snapshot(install)
    run = run_update(directory, "run5-desktop", local = False)
    assert run.rc == 0, run.log[-8000:]
    # files.pythonhosted.org is the ONE payload host the version check governs: with a
    # version mismatch this run IS an upgrade, and an upgrade downloads the wheel it
    # upgrades to. The other three carry release payloads -- llama.cpp, whisper.cpp,
    # node -- which no upgrade of unsloth has any reason to refetch, so they are held at
    # zero on both branches.
    payload_hosts = (
        PAYLOAD_HOSTS
        if desktop_path
        else tuple(host for host in PAYLOAD_HOSTS if host != "files.pythonhosted.org")
    )
    for host in payload_hosts:
        assert run.connections_to(host) == 0, f"{host}: {run.report()}"
        assert run.bytes_from(host) == 0, f"{host}: {run.report()}"
    assert run.bytes_from("raw.githubusercontent.com") <= ICON_FETCH_CEILING, run.report()
    # "dependencies up to date" is setup.sh's fast-path line, and the fast path is
    # exactly what the version check decides; an upgrade legitimately runs the pass
    # instead. The prebuilt and sidecar markers are not the version check's business:
    # both are answered from disk whether or not the dependency pass runs.
    markers = (
        NO_WORK_MARKERS
        if desktop_path
        else tuple(marker for marker in NO_WORK_MARKERS if marker != "dependencies up to date")
    )
    for marker in markers:
        assert marker in run.log, f"{marker!r} missing from a no-op update:\n{run.log[-8000:]}"
    # Each component, not the generic line once: one prebuilt re-validating while the
    # other answers from its marker would otherwise pass here.
    assert run.log.count("prebuilt up to date") >= 2, run.log[-8000:]
    assert run.log.count("sidecar current") == 3, run.log[-8000:]
    # One version check, and one "what is the latest release" HEAD per prebuilt. The
    # bound is what stops this quietly becoming a full release listing again. Only the
    # pypi.org one is relaxed: a real dependency pass resolves against the index, and
    # how many connections that takes is the resolver's business. The github.com bound
    # belongs to the prebuilts, which do the same work either way.
    if desktop_path:
        assert run.connections_to("pypi.org") <= 2, run.report()
    assert run.connections_to("github.com") <= 6, run.report()
    # The other half of the prebuilt claim: the release itself is never listed. Both of
    # these carried traffic on every update before the marker checks, and both are where
    # the 13-63 s macOS re-validation went.
    assert run.connections_to("api.github.com") == 0, run.report()
    assert run.connections_to("release-assets.githubusercontent.com") == 0, run.report()
    if desktop_path:
        # A dependency pass rewrites the manifest and can move the distribution list, so
        # "nothing changed on disk" is only a claim about the short-circuited path. The
        # no-op case is asserted in full by test_a_second_update_changes_nothing_on_disk.
        assert diff(before, snapshot(install)) == []
    else:
        # This run WAS an upgrade to PyPI's release, so the checkout is no longer what is
        # installed. Put it back: the workflow steps after this harness run the CLI and
        # expect the code under test, not the released one.
        restore = run_update(directory, "run5-restore", local = True)
        assert restore.rc == 0, restore.log[-8000:]
