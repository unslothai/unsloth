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

The fault-injection cases DAMAGE that install and expect the update to repair exactly
the damaged part. They restore what they broke, but a failure mid-case can leave the
install in the damaged state, so never point this at an install you care about.
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

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PROXY = pathlib.Path(__file__).resolve().parent / "idempotency_proxy.py"
IS_WINDOWS = sys.platform == "win32"

# Hosts a no-op update must not touch at all. pypi.org and github.com are listed
# separately because a version check and a "what is the latest release" HEAD are both
# legitimate, and small; these five are where the megabytes are.
PAYLOAD_HOSTS = (
    "files.pythonhosted.org",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "nodejs.org",
    "raw.githubusercontent.com",
)

# What the installers print when they decline to do work. All three prebuilts share the
# "already matches" substring, which setup.sh and setup.ps1 also grep for.
NO_WORK_MARKERS = ("dependencies up to date", "already matches")

DIST_LIST = (
    "import importlib.metadata as m, json; "
    "print(json.dumps(sorted(((d.metadata['Name'] or '').lower(), d.version) "
    "for d in m.distributions())))"
)


# ── the install under test ──


def _home() -> pathlib.Path:
    return pathlib.Path(os.environ.get("UNSLOTH_IDEMPOTENCY_HOME") or pathlib.Path.home())


def _studio_home() -> pathlib.Path:
    override = os.environ.get("UNSLOTH_IDEMPOTENCY_STUDIO_HOME")
    return pathlib.Path(override) if override else _home() / ".unsloth" / "studio"


def _venv_python() -> pathlib.Path:
    venv = _studio_home() / "unsloth_studio"
    return venv / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


@pytest.fixture(scope = "session")
def install() -> pathlib.Path:
    # The gate lives here rather than on the module so the proxy's own self test, which
    # needs nothing installed, still runs in ordinary CI. A measuring instrument that
    # quietly stopped measuring would make every assertion below pass.
    if not E2E:
        pytest.skip("opt-in end to end; set UNSLOTH_IDEMPOTENCY_E2E=1")
    venv_python = _venv_python()
    if not venv_python.is_file():
        pytest.skip(f"no Studio install at {venv_python}; run install.sh first")
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
    directory = tmp_path / label
    directory.mkdir(parents = True, exist_ok = True)
    extra = ["--refuse"] if refuse else (["--deny-hosts", deny_hosts] if deny_hosts else [])
    process, url, log_path = _start_proxy(directory, *extra)
    # Inherited, not replaced: the runner's PATH, SystemRoot and certificate bundle all
    # matter, and a hand-built environment here fails in ways that look like the product.
    env = dict(os.environ)
    env.pop("UNSLOTH_IDEMPOTENCY_E2E", None)
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
    unsloth_home = _home() / ".unsloth"
    state: dict = {}
    distributions = subprocess.run(
        [str(venv_python), "-I", "-c", DIST_LIST],
        capture_output = True,
        text = True,
        timeout = 300,
    )
    state["distributions"] = json.loads(distributions.stdout or "[]")

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
        state[marker] = {str(p.relative_to(unsloth_home)): _digest(p) for p in found}

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


def test_a_second_update_downloads_nothing(install, settled):
    """The measurement the timings cannot make: a fast run and a run that refetched
    three release manifests look the same on a warm cache."""
    directory, _ = settled
    run = run_update(directory, "run3-network")
    assert run.rc == 0, run.log[-8000:]
    for host in PAYLOAD_HOSTS:
        assert (
            run.bytes_from(host) == 0
        ), f"{host} served {run.bytes_from(host)} bytes: {run.report()}"
    # A version check and the "what is the latest release" HEADs are allowed, and are
    # what the prebuilt fast paths spend instead of a full release listing.
    assert run.connections_to("pypi.org") <= 1, run.report()
    assert run.connections_to("github.com") <= 4, run.report()


def test_a_second_update_says_it_did_nothing(install, settled):
    """The disk and the network agree; the log has to as well, because a silent skip
    and a silent failure to notice work are indistinguishable to a user."""
    directory, _ = settled
    run = run_update(directory, "run4-log")
    assert run.rc == 0, run.log[-8000:]
    for marker in NO_WORK_MARKERS:
        assert marker in run.log, f"{marker!r} missing from a no-op update:\n{run.log[-8000:]}"
    assert "falling back to source build" not in run.log


# ── offline ──


def test_a_no_op_update_needs_no_network_at_all(install, settled):
    """UV_OFFLINE plus a proxy that 403s everything. Every attempt is still recorded,
    so this asserts zero CONNECTIONS, not merely zero bytes."""
    directory, before = settled
    run = run_update(directory, "run5-offline", offline = True, refuse = True)
    assert run.rc == 0, (
        "an update with nothing to do failed offline. That is the state a user on a "
        f"plane, or behind a corporate proxy, is in:\n{run.log[-8000:]}"
    )
    assert run.connections == run.refused, run.report()
    assert diff(before, snapshot(install)) == []


def test_the_non_local_path_keeps_a_verified_install_when_pypi_is_unreachable(install, settled):
    """The rule added for exactly this: without --local, an unreachable PyPI means
    "update to be safe", which offline can only fail. UV_OFFLINE says the caller knows
    there is no network, so a verified install is kept instead."""
    directory, before = settled
    run = run_update(directory, "run6-offline-nonlocal", local = False, offline = True, refuse = True)
    assert run.rc == 0, run.log[-8000:]
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
    finally:
        if not manifest.is_file():
            manifest.write_bytes(saved)
    after = snapshot(install)
    assert (
        after["distributions"] == before["distributions"]
    ), "a pass with no evidence resolved to a different set of packages"
    assert after["manifest"] is not None, "the pass did not rewrite the manifest"


def test_a_damaged_llama_binary_reinstalls_llama(install, settled):
    directory, before = settled
    candidates = sorted((_home() / ".unsloth").glob("llama.cpp/**/llama-quantize*"))
    victim = next((p for p in candidates if p.is_file() and p.stat().st_size > 1024), None)
    if victim is None:
        pytest.skip("no llama.cpp prebuilt in this install")
    saved = victim.read_bytes()
    mode = victim.stat().st_mode
    victim.write_bytes(saved[: len(saved) // 4])
    try:
        run = run_update(directory, "fault-llama", local = True)
        assert run.rc == 0, run.log[-8000:]
        after = snapshot(install)
        assert after["binaries"] != before["binaries"], "the truncated binary was not replaced"
        assert victim.stat().st_size == len(saved), "llama.cpp was not reinstalled whole"
    finally:
        if victim.is_file() and victim.stat().st_size != len(saved):
            victim.write_bytes(saved)
            victim.chmod(mode)


def test_an_edited_requirements_file_runs_that_step(install, settled):
    """The installed copy of studio.txt is what the digest is taken from, so editing it
    is the same signal a released requirements change is."""
    directory, before = settled
    installed_req = next(
        (p for p in (install.parent.parent).rglob("requirements/studio.txt") if p.is_file()),
        None,
    )
    if installed_req is None:
        pytest.skip("no installed requirements tree")
    saved = installed_req.read_bytes()
    installed_req.write_bytes(saved + b"\n# idempotency harness\n")
    try:
        run = run_update(directory, "fault-requirements", local = True)
        assert run.rc == 0, run.log[-8000:]
        assert "studio deps" in run.log
    finally:
        installed_req.write_bytes(saved)
    after = snapshot(install)
    assert (
        after["distributions"] == before["distributions"]
    ), "re-running one step changed the resolved packages"


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


def test_the_scratch_install_is_left_where_it_was_found(install, settled):
    """Runs last. Every case above restores what it broke; this is the assertion that
    they all did."""
    _directory, before = settled
    assert diff(before, snapshot(install)) == []


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
    finally:
        process.terminate()
        process.wait(timeout = 10)
    summary = module.summary(str(log_path))
    assert summary["connections"] == 1 and summary["refused"] == 1
