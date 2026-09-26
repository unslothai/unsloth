# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in serving environments. No engine packages are imported into Studio.

When Studio's own PyTorch and CUDA libraries are the ones an engine is locked to,
its environment runs on Studio's interpreter and holds only the packages Studio
lacks or pins differently; Studio's site-packages follow its own on sys.path.
Otherwise it is a complete isolated environment.

Installation is serialized across Studio processes. Environments are built at
their final paths (venv scripts contain absolute paths); only the active marker
is replaced. Runtime leases prevent removing an environment another Studio uses.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager, nullcontext
from pathlib import Path

PROFILES = {
    "vllm": {
        "version": "0.20.0",
        "module": "vllm",
        "cuda": "cu130",
        "driver": 580,
        # The vllm wheel is manylinux_2_35.
        "glibc": (2, 35),
        # FlashInfer fetches the trtllm kernels it uses on demand rather than the whole cubin wheel.
        "omit": ("flashinfer-cubin",),
    },
    "sglang": {
        "version": "0.5.12",
        "module": "sglang",
        "cuda": "cu130",
        "driver": 580,
        # Excluded: outlines-core 0.1.26 has no py3.13 wheel; only --grammar-backend outlines needs it.
        "omit": ("outlines", "outlines-core", "flashinfer-cubin"),
    },
}
PYTHON = (3, 13)
_BASE_PTH = "zz_unsloth_studio_base.pth"
_BASE_MODULE = "_unsloth_studio_base"
# Studio installs its own FlashInfer for NVFP4 diffusion; a jit-cache of another version fails the
# engine's flashinfer import ("flashinfer-jit-cache version ... does not match"), so the engine never sees them.
_STUDIO_BASE_SOURCE = """import pkgutil, site, sys

_HIDDEN = ("flashinfer", "flashinfer_jit_cache", "flashinfer_cubin")


class _Hidden:
    def __init__(self, finder):
        self._finder = finder

    def find_spec(self, name, target = None):
        if name.partition(".")[0] in _HIDDEN:
            return None
        return self._finder.find_spec(name, target)

    def invalidate_caches(self):
        self._finder.invalidate_caches()

    def iter_modules(self, prefix = ""):
        for info in pkgutil.iter_importer_modules(self._finder, prefix):
            if info[0][len(prefix):] not in _HIDDEN:
                yield info


for path in {paths!r}:
    site.addsitedir(path)
    for hook in sys.path_hooks:
        try:
            finder = hook(path)
        except ImportError:
            continue
        sys.path_importer_cache[path] = _Hidden(finder)
        break
"""
_REQUIREMENTS = Path(__file__).resolve().parents[2] / "requirements" / "engines"
_jobs: dict[str, dict] = {}
_cancels: dict[str, threading.Event] = {}
_lock = threading.RLock()


def engine_root() -> Path:
    from utils.paths.storage_roots import studio_root
    return studio_root() / "engines"


def profile(engine: str) -> dict:
    if engine not in PROFILES:
        raise ValueError("Unknown inference engine")
    return PROFILES[engine]


def requirements(engine: str) -> Path:
    profile(engine)
    return _REQUIREMENTS / f"{engine}-linux-{profile(engine)['cuda']}.txt"


def profile_digest(engine: str) -> str:
    return hashlib.sha256(requirements(engine).read_bytes()).hexdigest()


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins(engine: str) -> dict[str, tuple[str, str]]:
    """{name: (version, requirement line with its hashes)} from a uv-compiled lock."""
    entries: dict[str, tuple[str, str]] = {}
    name = None
    for line in requirements(engine).read_text(encoding = "utf-8").splitlines(keepends = True):
        match = re.match(r"([A-Za-z0-9][A-Za-z0-9_.\-]*)==([^\s;\\]+)", line)
        if match:
            name = _normalize(match.group(1))
            entries[name] = (match.group(2), line)
        elif name and line.startswith("    --hash="):
            entries[name] = (entries[name][0], entries[name][1] + line)
        else:
            name = None
    return entries


def _studio_site() -> list[str]:
    paths = sysconfig.get_paths()
    return sorted({paths["purelib"], paths["platlib"]})


def _studio_packages() -> dict[str, str]:
    """Distributions in Studio's own site-packages, excluding sidecars on sys.path."""
    import importlib.metadata as metadata
    return {
        _normalize(dist.metadata["Name"]): dist.version
        for dist in metadata.distributions(path = _studio_site())
    }


def _torch_runtime() -> set[str]:
    """Studio's torch and the native CUDA packages it loads, which one process can hold once."""
    import importlib.metadata as metadata
    from packaging.requirements import Requirement

    names, pending = {"torch"}, [("torch", ())]
    while pending:
        name, extras = pending.pop()
        dists = list(metadata.distributions(name = name, path = _studio_site()))
        for raw in dists[0].requires or [] if dists else []:
            requirement = Requirement(raw)
            child = _normalize(requirement.name)
            if (
                child not in names
                and child.startswith(("nvidia-", "cuda-toolkit", "triton"))
                and (
                    requirement.marker is None
                    or any(requirement.marker.evaluate({"extra": e}) for e in ("", *extras))
                )
            ):
                names.add(child)
                pending.append((child, tuple(requirement.extras)))
    return names


def _same_build(installed: str | None, locked: str | None, cuda: str) -> bool:
    """PyTorch's index labels its wheels (2.11.0+cu130); the PyPI lock names the same build unlabelled."""
    return installed is not None and installed in (locked, f"{locked}+{cuda}")


def _compat(engine: str) -> dict[str, list[str]]:
    """What the locked packages require of each other (engine_compat.py), or {} for a stale file."""
    path = requirements(engine).with_suffix(".compat.json")
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return {}
    return data["requires"] if data.get("lock_sha256") == profile_digest(engine) else {}


def _reusable(
    engine: str, lock: dict, studio: dict[str, str], provided: dict[str, str]
) -> dict[str, str]:
    """Studio's own versions every engine dependency accepts, kept only while their own
    requirements still hold against what the engine will see."""
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    import importlib.metadata as metadata

    compat = _compat(engine)
    if not compat:
        return {}
    hidden = {"flashinfer-python", "flashinfer-jit-cache", "flashinfer-cubin"}
    reuse = {
        name: studio[name]
        for name in lock
        if name in studio
        and name not in provided
        and name not in hidden
        and all(
            SpecifierSet(spec).contains(studio[name], prereleases = True)
            for spec in compat.get(name, ())
        )
    }
    requires = {}
    for name in reuse:
        dists = list(metadata.distributions(name = name, path = _studio_site()))
        requires[name] = [Requirement(r) for r in (dists[0].requires or [])] if dists else None
    while True:
        seen = {name: version for name, (version, _) in lock.items()} | provided | reuse
        drop = {
            name
            for name in reuse
            if requires[name] is None
            or any(
                _normalize(r.name) in seen
                and not r.specifier.contains(seen[_normalize(r.name)], prereleases = True)
                for r in requires[name]
                if r.marker is None or r.marker.evaluate({"extra": ""})
            )
        }
        if not drop:
            return reuse
        for name in drop:
            del reuse[name]


def install_plan(engine: str) -> dict:
    """Split the lock into packages Studio already provides and the ones to install."""
    lock = _pins(engine)
    studio = _studio_packages()
    cuda = profile(engine)["cuda"]
    shared = (
        sys.implementation.name == "cpython"
        and sys.version_info[:2] == PYTHON
        and "torch" in studio
        and all(
            _same_build(studio.get(name), lock.get(name, (None,))[0], cuda)
            for name in _torch_runtime()
        )
    )
    provided = {
        name: studio[name]
        for name, (version, _) in lock.items()
        if shared and _same_build(studio.get(name), version, cuda)
    }
    if shared:
        provided |= _reusable(engine, lock, studio, provided)
    return {
        "shared": shared,
        "provided": provided,
        "requirements": "".join(line for name, (_, line) in lock.items() if name not in provided),
    }


def _cuda_trees(env: Path, shared: bool) -> list[Path]:
    sites = [env / "lib" / "python{}.{}".format(*PYTHON) / "site-packages"]
    sites += [Path(path) for path in _studio_site()] if shared else []
    return [site / "nvidia" / "cu13" for site in sites if (site / "nvidia" / "cu13").is_dir()]


def link_cuda_home(env: Path, shared: bool) -> None:
    """FlashInfer JIT-compiles some kernels; this CUDA_HOME is the locked pip nvcc, since a host may
    have only the driver, or a toolkit of another CUDA major."""
    home = env / "cuda"
    (home / "lib64").mkdir(parents = True, exist_ok = True)
    trees = _cuda_trees(env, shared)
    for name, link in (
        ("bin", home / "bin"),
        ("nvvm", home / "nvvm"),
        ("include", home / "include"),
    ):
        source = next(
            (
                tree / name
                for tree in trees
                if (tree / name / ("nvcc" if name == "bin" else "")).exists()
            ),
            None,
        )
        if source is not None and not link.is_symlink():
            link.symlink_to(source)
    # nvcc links -lcudart, and the runtime wheel ships only the versioned soname.
    cudart = next(
        (
            tree / "lib" / "libcudart.so.13"
            for tree in trees
            if (tree / "lib" / "libcudart.so.13").exists()
        ),
        None,
    )
    if cudart is not None and not (home / "lib64" / "libcudart.so").is_symlink():
        (home / "lib64" / "libcudart.so").symlink_to(cudart)


def cuda_environment(info: dict) -> dict[str, str]:
    """CUDA_HOME for the engine's JIT compiles; CPATH covers headers split across a shared environment."""
    env = Path(info["path"])
    if not (env / "cuda" / "bin" / "nvcc").exists():
        return {}
    return {
        "CUDA_HOME": str(env / "cuda"),
        "CPATH": os.pathsep.join(
            str(tree / "include") for tree in _cuda_trees(env, bool(info.get("shared")))
        ),
    }


def stale(info: dict) -> bool:
    """True when Studio no longer provides what a shared engine environment was checked with."""
    if not info.get("shared"):
        return False
    if info.get("python") != platform.python_version():
        return True
    studio = _studio_packages()
    return any(studio.get(name) != version for name, version in info.get("provided", {}).items())


# Runs in the engine's interpreter, so it checks both layers as the engine imports them.
_CHECK = r"""
import importlib.metadata as metadata, json, re, sys
from packaging.requirements import Requirement

def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()

omitted = set(sys.argv[2].split(",")) - {""}
cuda = sys.argv[3]
provided = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
problems = []
for line in open(sys.argv[1], encoding = "utf-8"):
    match = re.match(r"([A-Za-z0-9][A-Za-z0-9_.\-]*)==([^\s;\\]+)", line)
    if not match:
        continue
    name, version = match.groups()
    try:
        found = metadata.version(name)
    except metadata.PackageNotFoundError:
        found = None
    if found not in (version, f"{version}+{cuda}", provided.get(normalize(name))):
        problems.append(f"{name}=={version} is required, found {found}")
        continue
    for raw in metadata.requires(name) or []:
        requirement = Requirement(raw)
        if normalize(requirement.name) in omitted or (
            requirement.marker and not requirement.marker.evaluate({"extra": ""})
        ):
            continue
        try:
            have = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            have = None
        if have is None or not requirement.specifier.contains(have, prereleases = True):
            problems.append(f"{name} requires {requirement}, found {have}")
if problems:
    sys.exit("\n".join(problems))
"""


_gpu_rows: dict = {}
_gpu_rows_lock = threading.Lock()
_gpu_probe_lock = threading.Lock()
_PENDING = object()


def _cached_rows(gpu_id):
    with _gpu_rows_lock:
        cached = _gpu_rows.get(gpu_id)
    if cached is not None and (cached[1] is not None or time.monotonic() < cached[0]):
        return cached[1]
    return _PENDING


def _probe_rows(gpu_id):
    rows = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,compute_cap",
                "--format=csv,noheader",
                *(["--id", str(gpu_id)] if gpu_id is not None else []),
            ],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 60,
        )
        if result.returncode == 0:
            rows = [line.split(",") for line in result.stdout.strip().splitlines()]
    except (OSError, subprocess.TimeoutExpired):
        pass
    with _gpu_rows_lock:
        _gpu_rows[gpu_id] = (time.monotonic() + 300, rows)
    return rows


def _driver_rows(gpu_id: int | None, *, wait: bool = True):
    """nvidia-smi rows, cached per process; a failure is retried after 5 minutes. A loaded
    multi-GPU host can take over 20s to answer, so ``wait=False`` (status polls) never runs the
    probe in the caller: it starts one in the background and returns ``_PENDING``."""
    rows = _cached_rows(gpu_id)
    if rows is not _PENDING:
        return rows
    if not wait:
        if _gpu_probe_lock.acquire(blocking = False):

            def probe():
                try:
                    _probe_rows(gpu_id)
                finally:
                    _gpu_probe_lock.release()

            threading.Thread(target = probe, daemon = True).start()
        return _PENDING
    with _gpu_probe_lock:
        rows = _cached_rows(gpu_id)
        return _probe_rows(gpu_id) if rows is _PENDING else rows


def support_reason(
    engine: str = "vllm",
    gpu_id: int | None = None,
    *,
    wait: bool = True,
) -> str | None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        return "Managed engines currently require Linux x86_64."
    glibc = profile(engine).get("glibc", (2, 34))
    if tuple(int(x) for x in (platform.libc_ver()[1] or "0.0").split(".")[:2]) < glibc:
        return f"{engine} requires glibc {glibc[0]}.{glibc[1]} or newer."
    rows = _driver_rows(gpu_id, wait = wait)
    if rows is _PENDING:
        return "Checking for a supported NVIDIA GPU."
    try:
        if any(
            int(row[0].strip().split(".")[0]) >= profile(engine)["driver"] and float(row[1]) >= 8.0
            for row in rows or ()
            if len(row) == 2
        ):
            return None
    except ValueError:
        pass
    return f"Requires an NVIDIA GPU with compute capability 8.0 or newer and driver {profile(engine)['driver']} or newer."


def _atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(data), encoding = "utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok = True)


@contextmanager
def engine_lease(
    engine: str,
    *,
    exclusive: bool = False,
    wait: float = 1.0,
):
    """``wait=0`` for status probes: real takers retry so a probe's instant hold never fails them."""
    profile(engine)
    import fcntl

    root = engine_root()
    root.mkdir(parents = True, exist_ok = True)
    with (root / f"{engine}.lock").open("a", encoding = "utf-8") as handle:
        deadline = time.monotonic() + wait
        while True:
            try:
                fcntl.flock(handle, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "This engine is running or being changed in another Studio instance."
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def installed(engine: str) -> dict | None:
    profile(engine)
    root = engine_root() / engine
    if root.is_symlink():
        return None
    try:
        info = json.loads((root / "active.json").read_text(encoding = "utf-8"))
        directory = info["directory"]
        if (
            not isinstance(directory, str)
            or not directory.startswith("env-")
            or Path(directory).name != directory
        ):
            return None
        path = root / directory
        if path.is_symlink() or not (path / "bin" / "python").is_file():
            return None
        return {**info, "path": str(path)}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def status(engine: str) -> dict:
    info = installed(engine)
    try:
        job = json.loads((engine_root() / f"{engine}.job.json").read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        job = {"state": "idle", "phase": None, "message": ""}
    if job.get("state") == "running":
        try:
            with engine_lease(engine, wait = 0):
                job = {
                    "state": "error",
                    "phase": None,
                    "message": "Installation was interrupted. Retry to repair it.",
                }
        except (RuntimeError, ImportError):
            pass
    outdated = bool(info and stale(info))
    in_use = False
    if info and job.get("state") != "running":
        try:
            with engine_lease(engine, exclusive = True, wait = 0):
                pass
        except RuntimeError:
            in_use = True
        except ImportError:
            pass
    return {
        "engine": engine,
        "version": profile(engine)["version"],
        "installed_version": info.get("version") if info else None,
        "installed": info is not None,
        "in_use": in_use,
        "current": bool(
            info and info.get("profile_digest") == profile_digest(engine) and not outdated
        ),
        "restored": bool(info and info.get("restored") and not outdated),
        "shared": bool(info and info.get("shared")),
        "can_rollback": bool(
            info and isinstance(info.get("previous"), dict) and not stale(info["previous"])
        ),
        "unsupported_reason": support_reason(engine, wait = False),
        "download_bytes": None,
        "additional_disk_bytes": None,
        "job": job,
    }


def _update(engine: str, **values) -> None:
    with _lock:
        job = _jobs.setdefault(engine, {})
        if "phase" in values and values["phase"] != job.get("phase"):
            values.setdefault("activity", "")
        job.update(values)
        root = engine_root()
        root.mkdir(parents = True, exist_ok = True)
        _atomic_json(root / f"{engine}.job.json", _jobs[engine])


def install_environment() -> dict[str, str]:
    # Do not inherit Studio's resolver overrides, Python overlays or credentials.
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "HOME",
            "PATH",
            "LANG",
            "LC_ALL",
            "TMPDIR",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "NO_PROXY",
            "https_proxy",
            "http_proxy",
            "no_proxy",
        }
    }
    from utils.paths.storage_roots import cache_root

    recorded_cache = None
    try:
        raw = (cache_root() / "uv-cache-dir").read_bytes().decode("utf-8-sig")
        raw = raw.removesuffix("\n").removesuffix("\r")
        if Path(raw).is_absolute() and Path(raw).is_dir() and os.access(raw, os.W_OK):
            recorded_cache = raw
    except (OSError, UnicodeError, ValueError):
        pass
    env["UV_CACHE_DIR"] = (
        os.environ.get("UV_CACHE_DIR") or recorded_cache or str(cache_root() / "uv")
    )
    env["PYTHONNOUSERSITE"] = "1"
    env["UV_NO_CONFIG"] = "1"
    env["UV_CONCURRENT_DOWNLOADS"] = "4"
    env["UV_HTTP_RETRIES"] = "5"
    # uv's clone fallback can hardlink: opt into clones only after probing both filesystems.
    env["UV_LINK_MODE"] = "copy"
    return env


def package_link_mode(destination: Path) -> str:
    """Probe Linux reflinks across the actual install paths, without keeping files."""
    import fcntl

    cache = Path(install_environment()["UV_CACHE_DIR"])
    try:
        cache.mkdir(parents = True, exist_ok = True)
        with (
            tempfile.TemporaryFile(dir = cache) as source,
            tempfile.TemporaryFile(dir = destination) as target,
        ):
            source.write(b"Studio package clone probe")
            source.flush()
            fcntl.ioctl(target.fileno(), 0x40049409, source.fileno())
        return "clone"
    except OSError:
        return "copy"


def _run(engine: str, argv: list[str], cancel: threading.Event) -> None:
    from utils.process_lifetime import (
        adopt_pid,
        child_popen_kwargs,
        terminate_pid,
        forget_pid,
        spawn_on_lifetime_thread,
        is_process_shutting_down,
    )

    tail: deque[str] = deque(maxlen = 20)
    if cancel.is_set() or is_process_shutting_down():
        raise RuntimeError("Installation cancelled or Studio is shutting down.")
    proc = spawn_on_lifetime_thread(
        lambda: subprocess.Popen(
            argv,
            env = install_environment(),
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            start_new_session = True,
            **child_popen_kwargs(),
        )
    )
    adopt_pid(proc.pid)

    def drain():
        from utils.log_redaction import redact_log_text

        last_update = 0.0
        activity = ""
        for line in proc.stdout:
            line = redact_log_text(line.rstrip())[-1000:]
            if not line:
                continue
            tail.append(line)
            if line.startswith(
                (
                    "Downloading ",
                    "Downloaded ",
                    "Resolved ",
                    "Prepared ",
                    "Installed ",
                    "Uninstalled ",
                )
            ):
                activity = line
            now = time.monotonic()
            if now - last_update >= 1:
                _update(engine, activity = activity, log = list(tail))
                last_update = now
        if tail:
            _update(engine, activity = activity, log = list(tail))

    reader = threading.Thread(target = drain, daemon = True)
    deadline = time.monotonic() + 3600
    try:
        # A shutdown sweep may have finished between spawn and adoption; recheck even on clean exit.
        if is_process_shutting_down() or cancel.is_set():
            raise RuntimeError("Installation cancelled or Studio is shutting down.")
        reader.start()
        while proc.poll() is None:
            if (engine_root() / f"{engine}.cancel").exists():
                cancel.set()
            if is_process_shutting_down():
                cancel.set()
            if cancel.wait(0.2):
                raise RuntimeError("Installation cancelled.")
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Installation timed out. Retry when the connection is available."
                )
        reader.join(timeout = 2)
        if proc.returncode:
            raise RuntimeError("Engine installation failed. " + "\n".join(tail))
    finally:
        if proc.poll() is None:
            terminate_pid(proc.pid, timeout = 5, owner_verified = True)
        proc.wait(timeout = 10)
        forget_pid(proc.pid)
        if reader.ident is not None:
            reader.join(timeout = 2)
        proc.stdout.close()


def _install(
    engine: str,
    cancel: threading.Event,
    lease = None,
) -> None:
    destination = None
    try:
        with nullcontext() if lease is not None else engine_lease(engine, exclusive = True):
            _update(engine, state = "running", phase = "creating", message = "Preparing installation")
            reason = support_reason(engine)
            if reason:
                raise RuntimeError(reason)
            uv = shutil.which("uv")
            if uv is None:
                raise RuntimeError("uv is unavailable. Repair the Studio installation and retry.")
            root = engine_root() / engine
            if root.is_symlink():
                raise RuntimeError("Engine directory must not be a symbolic link.")
            root.mkdir(parents = True, exist_ok = True)
            digest = profile_digest(engine)
            destination = root / ("env-" + uuid.uuid4().hex)
            plan = install_plan(engine)
            _update(
                engine,
                phase = "creating",
                message = "Preparing an engine environment on Studio's PyTorch"
                if plan["shared"]
                else "Preparing an isolated Python environment",
            )
            interpreter = (
                sys.executable if sys.version_info[:2] == PYTHON else "{}.{}".format(*PYTHON)
            )
            _run(engine, [uv, "venv", "--python", interpreter, str(destination)], cancel)
            python = str(destination / "bin" / "python")
            packages = destination / "engine-requirements.txt"
            packages.write_text(plan["requirements"], encoding = "utf-8")
            _update(
                engine, phase = "installing", message = "Downloading and installing engine packages"
            )
            _run(
                engine,
                [
                    uv,
                    "pip",
                    "sync",
                    "--python",
                    python,
                    "--link-mode",
                    package_link_mode(destination),
                    "--require-hashes",
                    "--only-binary",
                    ":all:",
                    "--index-url",
                    "https://pypi.org/simple",
                    str(packages),
                ],
                cancel,
            )
            if plan["shared"]:
                # Appended after the engine's own site-packages, so its pins win.
                site = destination / "lib" / "python{}.{}".format(*PYTHON) / "site-packages"
                (site / f"{_BASE_MODULE}.py").write_text(
                    _STUDIO_BASE_SOURCE.format(paths = _studio_site()), encoding = "utf-8"
                )
                (site / _BASE_PTH).write_text(f"import {_BASE_MODULE}\n", encoding = "utf-8")
            link_cuda_home(destination, plan["shared"])
            _update(engine, phase = "checking", message = "Checking the installed engine")
            _run(
                engine,
                [
                    python,
                    "-I",
                    "-c",
                    _CHECK,
                    str(requirements(engine)),
                    ",".join(profile(engine).get("omit", ())),
                    profile(engine)["cuda"],
                    json.dumps(plan["provided"]),
                ],
                cancel,
            )
            torch_version = _pins(engine)["torch"][0]
            _run(
                engine,
                [
                    python,
                    "-I",
                    "-c",
                    f"import {profile(engine)['module']}; import torch; import bitsandbytes; import torchao; assert torch.__version__.split('+')[0] == {torch_version!r}; assert torch.version.cuda == '13.0'",
                ],
                cancel,
            )
            from .engine_adapters import ADAPTERS

            module = ADAPTERS[engine].module
            _run(engine, [python, "-I", "-m", module, "--help"], cancel)
            if (engine_root() / f"{engine}.cancel").exists():
                cancel.set()
            if cancel.is_set():
                raise RuntimeError("Installation cancelled.")
            prior = installed(engine)
            if profile_digest(engine) != digest:
                raise RuntimeError(
                    "Studio's engine profile changed during installation. Retry to use the updated profile."
                )
            _atomic_json(
                root / "active.json",
                {
                    "directory": destination.name,
                    "version": profile(engine)["version"],
                    "profile_digest": digest,
                    "shared": plan["shared"],
                    "python": platform.python_version(),
                    "provided": plan["provided"],
                    "studio_prefix": sys.prefix,
                    "previous_directory": prior["directory"] if prior else None,
                    "previous": {
                        k: v
                        for k, v in prior.items()
                        if k not in ("path", "previous", "previous_directory")
                    }
                    if prior
                    else None,
                },
            )
            # Keep the active env and one previous version; every runtime lease (any Studio) is excluded.
            keep = {destination.name, prior["directory"] if prior else None}
            for old in root.glob("env-*"):
                if old.name not in keep and old.is_dir() and not old.is_symlink():
                    shutil.rmtree(old, ignore_errors = True)
            destination = None
            _record_manifest(engine)
            _update(engine, state = "success", phase = "ready", message = "Engine installed")
    except Exception as exc:
        _update(
            engine, state = "cancelled" if cancel.is_set() else "error", phase = None, message = str(exc)
        )
    finally:
        if destination is not None:
            shutil.rmtree(destination, ignore_errors = True)
        if lease is not None:
            lease.__exit__(None, None, None)


def start_install(engine: str) -> dict:
    profile(engine)
    with _lock:
        if _jobs.get(engine, {}).get("state") == "running":
            return status(engine)
        lease = engine_lease(engine, exclusive = True)
        lease.__enter__()
        cancel = threading.Event()
        _cancels[engine] = cancel
        try:
            (engine_root() / f"{engine}.cancel").unlink(missing_ok = True)
            _update(
                engine,
                state = "running",
                phase = "queued",
                message = "Preparing installation",
                activity = "",
                log = [],
            )
            threading.Thread(target = _install, args = (engine, cancel, lease), daemon = True).start()
        except Exception:
            lease.__exit__(None, None, None)
            raise
    return status(engine)


def cancel_install(engine: str) -> dict:
    profile(engine)
    with _lock:
        event = _cancels.get(engine)
        if event:
            event.set()
        if status(engine)["job"].get("state") == "running":
            (engine_root() / f"{engine}.cancel").touch()
    return status(engine)


def remove(engine: str) -> dict:
    with engine_lease(engine, exclusive = True):
        root = engine_root() / engine
        if root.is_symlink():
            raise RuntimeError("Engine directory must not be a symbolic link.")
        shutil.rmtree(root, ignore_errors = False) if root.exists() else None
        with _lock:
            _jobs.pop(engine, None)
        (engine_root() / f"{engine}.job.json").unlink(missing_ok = True)
        (engine_root() / f"{engine}.cancel").unlink(missing_ok = True)
    _record_manifest(engine)
    return status(engine)


def rollback(engine: str) -> dict:
    with engine_lease(engine, exclusive = True):
        info = installed(engine)
        previous = info.get("previous") if info else None
        directory = previous.get("directory") if isinstance(previous, dict) else None
        if (
            not isinstance(directory, str)
            or not directory.startswith("env-")
            or Path(directory).name != directory
        ):
            raise RuntimeError("No previous engine installation is available.")
        root = engine_root() / engine
        path = root / directory
        if path.is_symlink() or not (path / "bin" / "python").is_file():
            raise RuntimeError(
                "The previous engine installation is unavailable. Repair the engine instead."
            )
        if stale(previous):
            raise RuntimeError(
                "The previous engine installation was built on packages Studio no longer has. Repair the engine instead."
            )
        _atomic_json(
            root / "active.json",
            {
                **previous,
                "restored": True,
                "previous_directory": info["directory"],
                "previous": {
                    k: v
                    for k, v in info.items()
                    if k not in ("path", "previous", "previous_directory")
                },
            },
        )
        _update(
            engine, state = "success", phase = "ready", message = "Previous engine installation restored"
        )
    _record_manifest(engine)
    return status(engine)


def _record_manifest(engine: str) -> None:
    try:
        from studio.install_manifest import update_manifest

        info = installed(engine)
        from utils.paths.storage_roots import studio_root

        update_manifest(
            root = studio_root(),
            **{
                f"optional_engine_{engine}": {
                    "installed": info is not None,
                    "version": info.get("version") if info else None,
                    "profile_digest": info.get("profile_digest") if info else None,
                }
            },
        )
    except ImportError:
        pass
