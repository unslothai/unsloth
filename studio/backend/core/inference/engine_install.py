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

from utils.subprocess_compat import windows_hidden_subprocess_kwargs
import sys
import sysconfig
import tempfile
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager, nullcontext
from pathlib import Path

PRECISIONS = ("auto", "bf16", "fp16", "int4", "int8", "fp8")
ENGINE_NAMES = {"vllm": "vLLM", "sglang": "SGLang"}
PROFILES = {
    "vllm": {
        "module": "vllm",
        "cuda": "cu130",
        "driver": 580,
        # FlashInfer fetches the trtllm kernels it uses on demand rather than the whole cubin wheel.
        # CUTLASS DSL loads its newest flavour the driver runs, and driver 580 runs CUDA 13.
        "omit": ("flashinfer-cubin", "nvidia-cutlass-dsl-libs-cu12"),
        # Newest first; each release pins one torch build (vLLM 0.27+ needs torch 2.13, 0.20-0.26 torch 2.11).
        "releases": (
            {"version": "0.30.0", "torch": "2.13.0", "lock": "vllm-linux-cu130-torch213"},
            {"version": "0.26.0", "torch": "2.11.0", "lock": "vllm-linux-cu130"},
        ),
        # AMD GPUs: vLLM's own ROCm build, Python 3.12 only. Its torch loads the host's ROCm
        # from /opt/rocm instead of bundling it, so the environment is always isolated.
        "rocm": {
            "cuda": None,
            "driver": None,
            "index": "https://wheels.vllm.ai/rocm/0.30.0/rocm723",
            "python": (3, 12),
            "glibc": (2, 39),
            "rocm": (7, 2),
            "gfx": (
                "gfx90a",
                "gfx942",
                "gfx950",
                "gfx1100",
                "gfx1101",
                "gfx1150",
                "gfx1151",
                "gfx1200",
                "gfx1201",
            ),
            "omit": (),
            "platform": "rocm",
            # The ROCm build has no bitsandbytes quantization, and TorchAO's packed INT4 is CUDA-only.
            "bitsandbytes": False,
            "precisions": tuple(p for p in PRECISIONS if p != "int4"),
            "releases": ({"version": "0.30.0", "torch": "2.12.0", "lock": "vllm-linux-rocm723"},),
        },
    },
    "sglang": {
        "module": "sglang",
        "cuda": "cu130",
        "driver": 580,
        # Excluded: outlines-core 0.1.26 has no py3.13 wheel; only --grammar-backend outlines needs it.
        "omit": ("outlines", "outlines-core", "flashinfer-cubin", "nvidia-cutlass-dsl-libs-cu12"),
        "releases": (
            {
                "version": "0.5.20",
                "torch": "2.13.0",
                "lock": "sglang-linux-cu130-torch213",
                # cuda-tile 1.6.0rc5 is sdist-only and SGLang never imports it.
                "omit": (
                    "outlines",
                    "outlines-core",
                    "flashinfer-cubin",
                    "nvidia-cutlass-dsl-libs-cu12",
                    "cuda-tile",
                ),
            },
            {"version": "0.5.17", "torch": "2.11.0", "lock": "sglang-linux-cu130"},
        ),
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
        spec = self._finder.find_spec(name, target)
        # A regular nvidia package here (Colab ships nvidia/__init__.py) would hide the engine's
        # own nvidia/ tree; as a namespace portion both are found, the engine's first.
        if name == "nvidia" and spec is not None and spec.submodule_search_locations:
            spec.loader = None
        return spec

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


_kfd_has_amd_gpu: bool | None = None


def gpu_platform() -> str:
    """The GPU family the engine runs on: "rocm" for AMD, else "cuda". A host with both, and
    every Windows host, follows Studio's own PyTorch."""
    global _kfd_has_amd_gpu
    from utils.hardware import hardware

    # IS_ROCM reads False until startup's background detection settles; an early status poll
    # or install on an AMD host must not pick the CUDA profile meanwhile.
    if not hardware.DETECTION_COMPLETE.is_set():
        hardware.ensure_hardware_detected()
    if hardware.IS_ROCM:
        return "rocm"
    if platform.system() != "Linux" or os.path.exists("/proc/driver/nvidia/version"):
        return "cuda"
    # Asked many times per status poll; the GPUs do not change while Studio runs.
    if _kfd_has_amd_gpu is None:
        from utils.hardware.amd import amd_kfd_gpu_node_count

        count = amd_kfd_gpu_node_count()
        if count is None:
            return "cuda"
        _kfd_has_amd_gpu = count > 0
    return "rocm" if _kfd_has_amd_gpu else "cuda"


def _flavor(engine: str) -> dict:
    """The engine's profile for this host's GPU platform, before a release is picked."""
    base = {
        "platform": "cuda",
        "python": PYTHON,
        "precisions": PRECISIONS,
        "bitsandbytes": True,
        **{key: value for key, value in PROFILES[engine].items() if key != "rocm"},
    }
    if gpu_platform() == "rocm" and "rocm" in PROFILES[engine]:
        return {**base, **PROFILES[engine]["rocm"]}
    return base


def _release(engine: str) -> dict:
    """The release built on Studio's own torch, so the engine can share it; otherwise the newest,
    which then gets an isolated environment of its own."""
    from . import wsl_host

    flavor = _flavor(engine)
    releases = flavor["releases"]
    if wsl_host.active() or flavor["platform"] == "rocm":
        # The WSL guest has no Studio torch to share, so it always gets the newest.
        return releases[0]
    torch = _studio_packages().get("torch")
    for release in releases:
        if _same_build(torch, release["torch"], flavor["cuda"]):
            return release
    return releases[0]


def profile(engine: str) -> dict:
    if engine not in PROFILES:
        raise ValueError("Unknown inference engine")
    return {**_flavor(engine), **_release(engine)}


def _python(engine: str) -> tuple[int, int]:
    return profile(engine)["python"]


def _venv_python_args(engine: str, guest: bool = False) -> list[str]:
    """uv venv's interpreter. Triton compiles its HIP driver module against the interpreter's
    Python.h at run time, and Ubuntu's python3.12 has none without python3.12-dev, so the ROCm
    environment always gets a uv-managed CPython, which ships its headers. Studio's own
    interpreter serves only a local environment, never the WSL guest's."""
    version = "{}.{}".format(*_python(engine))
    if profile(engine)["platform"] == "rocm":
        return ["--python", version, "--python-preference", "only-managed"]
    if not guest and sys.version_info[:2] == _python(engine):
        return ["--python", sys.executable]
    return ["--python", version]


def requirements(engine: str) -> Path:
    return _REQUIREMENTS / f"{profile(engine)['lock']}.txt"


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


_studio_seen: tuple[tuple, dict[str, str]] | None = None


def _studio_packages() -> dict[str, str]:
    """Distributions in Studio's own site-packages, excluding sidecars on sys.path. Status polls
    read this several times; the scan is redone only when an install changes a site directory."""
    global _studio_seen
    import importlib.metadata as metadata

    sites = _studio_site()
    key = tuple((site, os.stat(site).st_mtime_ns) for site in sites if os.path.isdir(site))
    if _studio_seen is None or _studio_seen[0] != key:
        _studio_seen = (
            key,
            {
                _normalize(dist.metadata["Name"]): dist.version
                for dist in metadata.distributions(path = sites)
            },
        )
    return dict(_studio_seen[1])


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


# nvcc's cudafe stubs need the crt headers of its own release, so mixing releases breaks
# FlashInfer's JIT (Colab: crt 13.4.59 with nvcc 13.0.88, "macro __cudaLaunch passed 2 arguments").
# Shared only when Studio holds every one at the locked version, else the engine brings all of them.
_TOOLCHAIN = frozenset({"nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-nvvm", "nvidia-cuda-cccl"})


def _same_build(installed: str | None, locked: str | None, cuda: str) -> bool:
    """PyTorch's index labels its wheels (2.11.0+cu130); the PyPI lock names the same build unlabelled."""
    return installed is not None and installed in (locked, f"{locked}+{cuda}")


def _compat_file(engine: str) -> dict:
    """engine_compat.py's output for the lock, or {} for a missing or stale file."""
    path = requirements(engine).with_suffix(".compat.json")
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return {}
    return data if data.get("lock_sha256") == profile_digest(engine) else {}


def _compat(engine: str) -> dict[str, list[str]]:
    """What the locked packages require of each other."""
    return _compat_file(engine).get("requires", {})


def download_bytes(engine: str) -> int | None:
    """Wheel bytes an install of the selected release downloads: the lock minus what Studio provides."""
    from . import wsl_host

    sizes = _compat_file(engine).get("sizes")
    if not sizes:
        return None
    wsl = wsl_host.active()
    # A WSL engine is always a complete environment, plus the distro and uv on first setup.
    provided = {} if wsl else install_plan(engine)["provided"]
    omit = set(profile(engine).get("omit", ()))
    total = sum(
        size or 0 for name, size in sizes.items() if name not in provided and name not in omit
    )
    summary = wsl_host.summary() if wsl else {}
    if wsl and not summary.get("distro"):
        total += wsl_host.ROOTFS["size"] + wsl_host.UV["size"]
    if (
        wsl
        and profile(engine)["platform"] == "rocm"
        and summary.get("rocm") != wsl_host.ROCM_RELEASE
    ):
        total += wsl_host.ROCM_APT_BYTES + sum(
            spec["size"] for spec in (wsl_host.ROCDXG, wsl_host.ROCDXG_SMI)
        )
    return total


# Not engine dependencies, so absent from the compat file: the configs Studio's adapters build
# (Int8WeightOnlyConfig and friends) use TorchAO 0.17's API, and 0.18 removed their version 1.
_ADAPTER_REQUIRES = {"torchao": ["==0.17.0"]}


def _adapter_accepts(name: str, version: str) -> bool:
    from packaging.specifiers import SpecifierSet
    return all(
        SpecifierSet(spec).contains(version, prereleases = True)
        for spec in _ADAPTER_REQUIRES.get(name, ())
    )


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
    hidden = {"flashinfer-python", "flashinfer-jit-cache", "flashinfer-cubin", *_TOOLCHAIN}
    reuse = {
        name: studio[name]
        for name in lock
        if name in studio
        and name not in provided
        and name not in hidden
        and _adapter_accepts(name, studio[name])
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
    from packaging.specifiers import SpecifierSet

    lock = _pins(engine)
    studio = _studio_packages()
    cuda = profile(engine)["cuda"]
    compat = _compat(engine)
    runtime = _torch_runtime()

    def fits(name: str) -> bool:
        if _same_build(studio.get(name), lock.get(name, (None,))[0], cuda):
            return True
        # Torch is the exact build; a CUDA library torch takes as a range (nvjitlink) only has to
        # satisfy every locked package's requirement, since the engine then loads Studio's copy.
        specs = compat.get(name)
        return bool(
            name != "torch"
            and name not in _TOOLCHAIN
            and name in lock
            and name in studio
            and specs
            and all(SpecifierSet(spec).contains(studio[name], prereleases = True) for spec in specs)
        )

    # The ROCm build's torch loads /opt/rocm rather than bundled libraries: never Studio's.
    shared = (
        profile(engine)["platform"] == "cuda"
        and sys.implementation.name == "cpython"
        and sys.version_info[:2] == _python(engine)
        and "torch" in studio
        and all(fits(name) for name in runtime - _TOOLCHAIN)
    )
    provided = {
        name: studio[name]
        for name, (version, _) in lock.items()
        if shared
        and (_same_build(studio.get(name), version, cuda) or (name in runtime and fits(name)))
    }
    if not all(name in provided for name in _TOOLCHAIN & lock.keys()):
        provided = {name: version for name, version in provided.items() if name not in _TOOLCHAIN}
    if shared:
        provided |= _reusable(engine, lock, studio, provided)
    return {
        "shared": shared,
        "provided": provided,
        "requirements": "".join(line for name, (_, line) in lock.items() if name not in provided),
    }


def _cuda_trees(env: Path, shared: bool) -> list[Path]:
    sites = list(env.glob("lib/python3.*/site-packages"))
    sites += [Path(path) for path in _studio_site()] if shared else []
    return [site / "nvidia" / "cu13" for site in sites if (site / "nvidia" / "cu13").is_dir()]


def link_cuda_home(env: Path, shared: bool) -> None:
    """FlashInfer JIT-compiles some kernels; this CUDA_HOME is the locked pip nvcc, since a host may
    have only the driver, or a toolkit of another CUDA major."""
    home = env / "cuda"
    (home / "lib64").mkdir(parents = True, exist_ok = True)
    trees = _cuda_trees(env, shared)
    # One include tree, the engine's entries first: a Studio header such as curand_kernel.h
    # quote-includes crt/ from beside itself, which then resolves to the engine's crt.
    include = home / "include"
    if include.is_symlink():
        include.unlink()
    include.mkdir(exist_ok = True)
    for tree in trees:
        if (tree / "include").is_dir():
            for entry in (tree / "include").iterdir():
                if not (include / entry.name).is_symlink():
                    (include / entry.name).symlink_to(entry)
    for name, link in (
        ("bin", home / "bin"),
        ("nvvm", home / "nvvm"),
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
    """CUDA_HOME for the engine's JIT compiles; CPATH is its merged include tree for other compilers."""
    if info.get("host") == "wsl":
        return dict(info.get("cuda_environment") or {})
    env = Path(info["path"])
    if not (env / "cuda" / "bin" / "nvcc").exists():
        return {}
    return {"CUDA_HOME": str(env / "cuda"), "CPATH": str(env / "cuda" / "include")}


def stale(info: dict) -> bool:
    """True when Studio no longer provides what a shared engine environment was checked with."""
    if not info.get("shared"):
        return False
    if info.get("python") != platform.python_version():
        return True
    studio = _studio_packages()
    return any(
        studio.get(name) != version or not _adapter_accepts(name, version)
        for name, version in info.get("provided", {}).items()
    )


def built_for_this_gpu(engine: str, info: dict) -> bool:
    """Whether an environment runs on this host's GPU platform; one from before AMD support is CUDA.
    A restored environment skips the profile check, so this keeps it off the other vendor's GPU."""
    return info.get("platform", "cuda") == profile(engine)["platform"]


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
        from utils.hardware.nvidia import _nvidia_smi_executable
        result = subprocess.run(
            [
                _nvidia_smi_executable(),
                "--query-gpu=driver_version,compute_cap",
                "--format=csv,noheader",
                *(["--id", str(gpu_id)] if gpu_id is not None else []),
            ],
            capture_output = True,
            **windows_hidden_subprocess_kwargs(),
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
    from . import wsl_host

    if wsl_host.active():
        # WSL itself is not a prerequisite: installing the engine sets it up.
        reason = wsl_host.support_reason()
        if reason:
            return reason
    elif platform.system() != "Linux" or platform.machine() != "x86_64":
        return "Managed engines currently require Linux x86_64 or Windows x64."
    else:
        glibc = profile(engine).get("glibc", (2, 34))
        if tuple(int(x) for x in (platform.libc_ver()[1] or "0.0").split(".")[:2]) < glibc:
            return f"{ENGINE_NAMES[engine]} requires glibc {glibc[0]}.{glibc[1]} or newer."
    if gpu_platform() == "rocm":
        return _rocm_reason(engine, gpu_id, wait)
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


ROCM_HOME = Path("/opt/rocm")
# What vLLM's ROCm torch and kernels load from the host (their RUNPATH is /opt/rocm/lib).
ROCM_LIBRARIES = (
    "libamdhip64.so.7",
    "libhiprtc.so.7",
    "libMIOpen.so.1",
    "librocblas.so.5",
    "libhipblas.so.3",
    "libhipblaslt.so.1",
    "libhipfft.so.0",
    "libhiprand.so.1",
    "libhipsparse.so.4",
    "libhipsparselt.so.0",
    "libhipsolver.so.1",
    "librocsolver.so.0",
    "librccl.so.1",
    "librocprofiler-sdk.so.1",
    # rocprofiler-sdk loads it but does not depend on its package (hsa-amd-aqlprofile).
    "libhsa-amd-aqlprofile64.so.1",
    "libroctx64.so.4",
)
# Linked by that torch from outside ROCm: {soname: Ubuntu 24.04 package}.
ROCM_SYSTEM_LIBRARIES = {
    "libmpi.so.40": "libopenmpi3t64",
    "libmpi_cxx.so.40": "libopenmpi3t64",
    "libnuma.so.1": "libnuma1",
}


def rocm_version() -> tuple[int, int] | None:
    """The ROCm release installed in /opt/rocm, where the ROCm engine's torch looks for it."""
    try:
        text = (ROCM_HOME / ".info" / "version").read_text(encoding = "utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.match(r"\s*(\d+)\.(\d+)", text)
    return (int(match.group(1)), int(match.group(2))) if match else None


_rocm_arches: dict[int, str] | None = None


def _rocm_gpu_arches() -> dict[int, str]:
    """{physical GPU id: the gfx target it presents} from Studio's own ROCm torch, read once, or {}
    when the ordinals cannot be mapped to physical ids. The engine inherits Studio's environment,
    so an HSA_OVERRIDE_GFX_VERSION spoof applies to both."""
    global _rocm_arches
    if _rocm_arches is None:
        _rocm_arches = {}
        try:
            import torch
            from utils.hardware.hardware import (
                _props_gfx_arch,
                _rocm_device_ordinal_active,
                _rocm_visibility_masks_are_stacked,
                _torch_ordinal_physical_ids,
            )

            # Stacked or ordinal masks renumber the devices; the KFD topology then answers instead.
            # AMD's SDK / Radeon wheels leave torch.version.hip unset; their label carries ROCm.
            if (
                (getattr(torch.version, "hip", None) or "rocm" in torch.__version__.lower())
                and torch.cuda.is_available()
                and not _rocm_device_ordinal_active()
                and not _rocm_visibility_masks_are_stacked()
            ):
                count = torch.cuda.device_count()
                physical = _torch_ordinal_physical_ids(count) or list(range(count))
                for ordinal, gpu_id in enumerate(physical[:count]):
                    arch = _props_gfx_arch(torch.cuda.get_device_properties(ordinal))
                    if arch:
                        _rocm_arches[gpu_id] = arch
        except Exception:
            _rocm_arches = {}
    return _rocm_arches


def _unsupported_amd_gpu(found: list[str]) -> str:
    return (
        "Requires an AMD Instinct MI200, MI300 or MI350, Radeon RX 7700 to 7900, Radeon RX 9000, or Ryzen AI Max or AI 300 GPU"
        + (f" (found {', '.join(found)})." if found else ".")
    )


def _rocm_reason(
    engine: str,
    gpu_id: int | None,
    wait: bool = True,
) -> str | None:
    from . import wsl_host

    if "rocm" not in PROFILES[engine]:
        return f"{ENGINE_NAMES[engine]} requires an NVIDIA GPU. Use vLLM on AMD GPUs."
    wanted = profile(engine)
    if wsl_host.active():
        # Studio installs ROCm inside its WSL distro and checks the GPU there; a card Studio's torch
        # or the Windows driver already names refuses before that download. The engine launches
        # on the selected ordinal, so that GPU's own target decides when torch names it.
        arches = _rocm_gpu_arches() if gpu_id is not None else {}
        if gpu_id in arches:
            known = [arches[gpu_id]]
        else:
            from utils.hardware.hardware import get_physical_gpu_inventory
            known = [
                device["gfx"]
                for device in get_physical_gpu_inventory(block = wait).get("devices") or []
                if device.get("vendor") == "amd" and device.get("gfx")
            ]
        if known and not any(target in wanted["gfx"] for target in known):
            return _unsupported_amd_gpu(known)
        return None
    found = rocm_version()
    # The ROCm 7 libraries the engine links keep one soname for every 7.x release.
    if found is None or found[0] != wanted["rocm"][0] or found < wanted["rocm"]:
        return "Requires ROCm {}.{} or a newer {}.x release installed in /opt/rocm".format(
            *wanted["rocm"], wanted["rocm"][0]
        ) + (" (found {}.{}).".format(*found) if found else ".")
    from utils.hardware.amd import _a_bare_soname_resolves, amd_kfd_gpu_gfx_targets

    missing = [soname for soname in ROCM_LIBRARIES if not (ROCM_HOME / "lib" / soname).exists()]
    if missing:
        return (
            f"The ROCm installation in /opt/rocm is missing {', '.join(missing)}. Install the full "
            "ROCm package (on Ubuntu: sudo apt install rocm), then retry."
        )
    missing = {
        soname: package
        for soname, package in ROCM_SYSTEM_LIBRARIES.items()
        # The install and the engine both run without the caller's LD_LIBRARY_PATH.
        if not (ROCM_HOME / "lib" / soname).exists()
        and not _a_bare_soname_resolves(soname, with_ld_library_path = False)
    }
    if missing:
        return (
            f"Requires {', '.join(missing)}, which vLLM's AMD build links. "
            f"On Ubuntu: sudo apt install {' '.join(dict.fromkeys(missing.values()))}"
        )
    if not (os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")):
        # Triton builds its HIP driver module the first time the engine compiles a kernel.
        return "Requires a C compiler, which vLLM's AMD build uses at run time (on Ubuntu: sudo apt install gcc)."
    from utils.hardware.amd import amd_closed_nodes_block_the_runtime, amd_node_permission_hint

    # HIP opens /dev/kfd and an AMD render node; a container mapping only one initialises no GPU.
    if not os.access("/dev/kfd", os.R_OK | os.W_OK) or amd_closed_nodes_block_the_runtime():
        return amd_node_permission_hint() or (
            "Studio cannot open /dev/kfd and an AMD render node (/dev/dri/renderD*). Add your user "
            "to the render and video groups, then sign in again."
        )

    # The target each GPU presents to the engine (an HSA_OVERRIDE_GFX_VERSION spoof included),
    # else the one the kernel reports.
    arches = _rocm_gpu_arches()
    if gpu_id is None:
        selected = list(arches.values()) or amd_kfd_gpu_gfx_targets() or []
    elif gpu_id in arches:
        selected = [arches[gpu_id]]
    else:
        selected = (amd_kfd_gpu_gfx_targets() or [])[gpu_id : gpu_id + 1]
    selected = [target for target in selected if target]
    if gpu_id is not None and not selected:
        # Nothing names this GPU's target; vLLM refuses one it has no kernels for itself.
        return None
    if any(target in wanted["gfx"] for target in selected):
        return None
    return _unsupported_amd_gpu(selected)


def _atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(data), encoding = "utf-8")
        # Windows readers (including status polling in another Studio process) can
        # briefly deny deletion of the destination. Keep the previous JSON intact
        # and retry the atomic rename; permanent permission errors still surface.
        deadline = time.monotonic() + 1.0
        while True:
            try:
                os.replace(tmp, path)
                break
            except OSError as exc:
                if (
                    sys.platform != "win32"
                    or getattr(exc, "winerror", None) not in (5, 32, 33)
                    or time.monotonic() >= deadline
                ):
                    raise
                time.sleep(0.01)
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
    lock, unlock = _file_lock()

    root = engine_root()
    root.mkdir(parents = True, exist_ok = True)
    with (root / f"{engine}.lock").open("a", encoding = "utf-8") as handle:
        deadline = time.monotonic() + wait
        while True:
            try:
                lock(handle, exclusive)
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
            unlock(handle)


def _file_lock():
    """(lock, unlock) raising BlockingIOError when held: flock, or LockFileEx on Windows, which
    also has shared and exclusive modes (msvcrt.locking is exclusive only)."""
    if os.name != "nt":
        import fcntl
        return (
            lambda handle, exclusive: fcntl.flock(
                handle, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB
            ),
            lambda handle: fcntl.flock(handle, fcntl.LOCK_UN),
        )
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class Overlapped(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_void_p),
            ("InternalHigh", ctypes.c_void_p),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)

    def lock(handle, exclusive):
        flags = 0x1 | (
            0x2 if exclusive else 0
        )  # LOCKFILE_FAIL_IMMEDIATELY | LOCKFILE_EXCLUSIVE_LOCK
        if not kernel32.LockFileEx(
            wintypes.HANDLE(msvcrt.get_osfhandle(handle.fileno())),
            flags,
            0,
            1,
            0,
            ctypes.byref(Overlapped()),
        ):
            raise BlockingIOError(ctypes.get_last_error(), "lock held")

    def unlock(handle):
        kernel32.UnlockFileEx(
            wintypes.HANDLE(msvcrt.get_osfhandle(handle.fileno())),
            0,
            1,
            0,
            ctypes.byref(Overlapped()),
        )

    return lock, unlock


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
        if info.get("host") == "wsl":
            # Lives in the distro; checking it would boot WSL on every status poll, so the load checks.
            from .wsl_host import GUEST_ROOT
            return {**info, "path": f"{GUEST_ROOT}/engines/{engine}/{directory}"}
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
    wanted = profile(engine)
    return {
        "engine": engine,
        "version": wanted["version"],
        "installed_version": info.get("version") if info else None,
        "installed": info is not None,
        "in_use": in_use,
        "current": bool(
            info and info.get("profile_digest") == profile_digest(engine) and not outdated
        ),
        "restored": bool(
            info and info.get("restored") and not outdated and built_for_this_gpu(engine, info)
        ),
        "shared": bool(info and info.get("shared")),
        "can_rollback": bool(
            info
            and isinstance(info.get("previous"), dict)
            and not stale(info["previous"])
            and built_for_this_gpu(engine, info["previous"])
        ),
        "unsupported_reason": support_reason(engine, wait = False),
        "precisions": list(wanted["precisions"]),
        "platform": gpu_platform(),
        # Only priced while an install or update is on offer: the plan reads Studio's packages.
        "download_bytes": None
        if info and info.get("profile_digest") == profile_digest(engine) and not outdated
        else _safe_download_bytes(engine),
        "additional_disk_bytes": None,
        "job": job,
        **_host_status(),
    }


def _host_status() -> dict:
    from . import wsl_host
    return {"host": "wsl", "wsl": wsl_host.summary()} if wsl_host.active() else {"host": "local"}


def _safe_download_bytes(engine: str) -> int | None:
    try:
        return download_bytes(engine)
    except Exception:
        return None


def _update(engine: str, **values) -> None:
    with _lock:
        job = _jobs.setdefault(engine, {})
        if "phase" in values and values["phase"] != job.get("phase"):
            values.setdefault("activity", "")
        job.update(values)
        root = engine_root()
        root.mkdir(parents = True, exist_ok = True)
        _atomic_json(root / f"{engine}.job.json", _jobs[engine])


def driver_library_path(env: dict[str, str]) -> str | None:
    """The LD_LIBRARY_PATH entries that hold the NVIDIA driver. Hosts such as Colab keep libcuda
    outside the loader cache and reach it only this way; every other entry is dropped, so a CUDA
    runtime Studio's environment points at never shadows the engine's own."""
    entries = [
        entry
        for entry in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if entry and os.path.isabs(entry) and (Path(entry) / "libcuda.so.1").exists()
    ]
    return os.pathsep.join(entries) or None


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
    # The install checks import the engine, whose device detection needs the driver.
    if driver := driver_library_path(dict(os.environ)):
        env["LD_LIBRARY_PATH"] = driver
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


def _run(
    engine: str,
    argv: list[str],
    cancel: threading.Event,
    env: dict | None = None,
    *,
    stdin_pipe: bool = False,
) -> str:
    from utils.process_lifetime import (
        adopt_pid,
        child_popen_kwargs,
        terminate_pid,
        forget_pid,
        spawn_on_lifetime_thread,
        is_process_shutting_down,
    )

    tail: deque[str] = deque(maxlen = 20)
    if (engine_root() / f"{engine}.cancel").exists():
        cancel.set()
    if cancel.is_set() or is_process_shutting_down():
        raise RuntimeError("Installation cancelled or Studio is shutting down.")
    proc = spawn_on_lifetime_thread(
        lambda: subprocess.Popen(
            argv,
            env = install_environment() if env is None else env,
            stdin = subprocess.PIPE if stdin_pipe else None,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            start_new_session = True,
            # A windowed Studio has no console; wsl.exe would otherwise open one.
            creationflags = 0x08000000 if sys.platform == "win32" else 0,
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
        if cancel.is_set() or is_process_shutting_down():
            raise RuntimeError("Installation cancelled or Studio is shutting down.")
        if proc.returncode:
            raise RuntimeError("Engine installation failed. " + "\n".join(tail))
        return "\n".join(tail)
    finally:
        if proc.stdin is not None:
            # Closing the WSL runner's pipe stops its guest process group, including
            # apt/dpkg children; terminating wsl.exe alone can leave them running.
            try:
                proc.stdin.close()
                proc.wait(timeout = 15)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if proc.poll() is None:
            terminate_pid(proc.pid, timeout = 5, owner_verified = True)
        proc.wait(timeout = 10)
        forget_pid(proc.pid)
        if reader.ident is not None:
            reader.join(timeout = 2)
        proc.stdout.close()


def _index_arguments(engine: str) -> list[str]:
    """PyPI, plus the engine's own index first for the builds only it publishes (vLLM's ROCm torch)."""
    extra = profile(engine).get("index")
    return [
        "--index-url",
        "https://pypi.org/simple",
        *(["--extra-index-url", extra] if extra else []),
    ]


def _smoke_source(engine: str) -> str:
    """Imports the engine and the precision libraries its lock carries, on the locked torch build."""
    wanted = profile(engine)
    pins = _pins(engine)
    modules = [
        wanted["module"],
        "torch",
        *(name for name in ("bitsandbytes", "torchao") if name in pins),
    ]
    runtime = (
        "assert torch.version.hip"
        if wanted["platform"] == "rocm"
        else "assert torch.version.cuda == '13.0'"
    )
    return (
        "".join(f"import {module}\n" for module in modules)
        + f"assert torch.__version__.split('+')[0] == {pins['torch'][0].split('+')[0]!r}\n"
        + runtime
        + "\n"
    )


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
            from . import wsl_host

            if wsl_host.active():
                _install_wsl(engine, cancel)
                return
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
            _run(engine, [uv, "venv", *_venv_python_args(engine), str(destination)], cancel)
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
                    *_index_arguments(engine),
                    str(packages),
                ],
                cancel,
            )
            if plan["shared"]:
                # Appended after the engine's own site-packages, so its pins win.
                site = (
                    destination / "lib" / "python{}.{}".format(*_python(engine)) / "site-packages"
                )
                (site / f"{_BASE_MODULE}.py").write_text(
                    _STUDIO_BASE_SOURCE.format(paths = _studio_site()), encoding = "utf-8"
                )
                (site / _BASE_PTH).write_text(f"import {_BASE_MODULE}\n", encoding = "utf-8")
            if profile(engine)["platform"] == "cuda":
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
                    profile(engine)["cuda"] or "",
                    json.dumps(plan["provided"]),
                ],
                cancel,
            )
            _run(engine, [python, "-I", "-c", _smoke_source(engine)], cancel)
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
                    "platform": profile(engine)["platform"],
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
        from .wsl_host import Waiting
        _update(
            engine,
            state = "waiting"
            if isinstance(exc, Waiting)
            else "cancelled"
            if cancel.is_set()
            else "error",
            phase = None,
            message = str(exc),
        )
    finally:
        if destination is not None:
            shutil.rmtree(destination, ignore_errors = True)
        if lease is not None:
            lease.__exit__(None, None, None)


# Runs in the engine's interpreter inside WSL: the non-shared half of link_cuda_home and
# managed_engine._deep_gemm_unloadable, whose paths Windows cannot inspect.
_GUEST_FINALIZE = r"""
import json, sys
from pathlib import Path

env = Path(sys.argv[1])
site = next(env.glob("lib/python3.*/site-packages"))
tree = site / "nvidia" / "cu13"
home = env / "cuda"
(home / "lib64").mkdir(parents = True, exist_ok = True)
for name in ("bin", "nvvm", "include"):
    if (tree / name).exists() and not (home / name).is_symlink():
        (home / name).symlink_to(tree / name)
cudart = tree / "lib" / "libcudart.so.13"
if cudart.exists() and not (home / "lib64" / "libcudart.so").is_symlink():
    (home / "lib64" / "libcudart.so").symlink_to(cudart)
vendored = site / "vllm" / "third_party" / "deep_gemm"
tag = "cpython-3" + site.parent.name.removeprefix("python3.")
print(json.dumps({
    "cuda_environment": {"CUDA_HOME": str(home), "CPATH": str(tree / "include")}
    if (home / "bin" / "nvcc").exists() else {},
    "deep_gemm_unloadable": not (site / "deep_gemm").is_dir() and vendored.is_dir()
    and not any(vendored.glob(f"_C.{tag}-*.so")) and not any(vendored.glob("_C.abi3*.so")),
}))
"""
_PROXIES = ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy")


def _install_wsl(engine: str, cancel: threading.Event) -> None:
    """The same hash-pinned lock and checks as on Linux, run inside Studio's WSL distro. Always
    isolated: Studio's Windows packages cannot serve a Linux interpreter."""
    from . import wsl_host

    guest_root = wsl_host.GUEST_ROOT
    rocm = profile(engine)["platform"] == "rocm"
    progress = lambda text: _update(engine, activity = text)
    _update(engine, phase = "preparing_wsl", message = "Setting up the Unsloth WSL environment")
    wsl_host.prepare(progress, cancel, profile(engine)["platform"])
    if cancel.is_set():
        raise RuntimeError("Installation cancelled.")
    root = engine_root() / engine
    root.mkdir(parents = True, exist_ok = True)
    digest = profile_digest(engine)
    directory = "env-" + uuid.uuid4().hex
    base = f"{guest_root}/engines/{engine}"
    destination = f"{base}/{directory}"
    python = f"{destination}/bin/python"
    uv = f"{guest_root}/bin/uv"
    env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONNOUSERSITE": "1",
        "UV_CACHE_DIR": f"{guest_root}/uv-cache",
        "UV_PYTHON_INSTALL_DIR": f"{guest_root}/python",
        "UV_NO_CONFIG": "1",
        "UV_CONCURRENT_DOWNLOADS": "4",
        "UV_HTTP_RETRIES": "5",
        # Cache and environments share the distro's ext4 disk.
        "UV_LINK_MODE": "hardlink",
        "DEBIAN_FRONTEND": "noninteractive",
        **(wsl_host.rocm_environment() if rocm else {}),
    }
    secrets = {key: os.environ[key] for key in _PROXIES if os.environ.get(key)}

    def guest_run(argv):
        command, windows_env = wsl_host.guest_command(
            [f"{guest_root}/bin/run-engine", *argv], env = env, secrets = secrets
        )
        return _run(engine, command, cancel, env = windows_env, stdin_pipe = True)

    try:
        wsl_host.ensure_build_tools(guest_run, progress)
        if rocm:
            _install_wsl_rocm(engine, guest_run, progress, cancel)
        _update(engine, phase = "creating", message = "Preparing an isolated Python environment")
        guest_run([uv, "venv", *_venv_python_args(engine, guest = True), destination])
        _update(engine, phase = "installing", message = "Downloading and installing engine packages")
        lock = wsl_host.to_guest_path(requirements(engine))
        guest_run(
            [
                uv,
                "pip",
                "sync",
                "--python",
                python,
                "--require-hashes",
                "--only-binary",
                ":all:",
                *_index_arguments(engine),
                lock,
            ]
        )
        _update(engine, phase = "checking", message = "Checking the installed engine")
        from .engine_adapters import ADAPTERS

        scripts = {
            "check.py": _CHECK,
            "finalize.py": _GUEST_FINALIZE,
            "smoke.py": _smoke_source(engine),
        }
        for name, source in scripts.items():
            wsl_host.put(f"{destination}/{name}", source)
        guest_run(
            [
                python,
                "-I",
                f"{destination}/check.py",
                lock,
                ",".join(profile(engine).get("omit", ())),
                profile(engine)["cuda"] or "",
                "{}",
            ]
        )
        guest_run([python, "-I", f"{destination}/smoke.py"])
        guest_run([python, "-I", "-m", ADAPTERS[engine].module, "--help"])
        facts = json.loads(
            wsl_host.guest([python, "-I", f"{destination}/finalize.py", destination])
            .strip()
            .splitlines()[-1]
        )
        if (engine_root() / f"{engine}.cancel").exists():
            cancel.set()
        if cancel.is_set():
            raise RuntimeError("Installation cancelled.")
        if profile_digest(engine) != digest:
            raise RuntimeError(
                "Studio's engine profile changed during installation. Retry to use the updated profile."
            )
        prior = installed(engine)
        _atomic_json(
            root / "active.json",
            {
                "directory": directory,
                "version": profile(engine)["version"],
                "profile_digest": digest,
                "shared": False,
                "platform": profile(engine)["platform"],
                "python": "{}.{}".format(*_python(engine)),
                "provided": {},
                "host": "wsl",
                "distro": wsl_host.distro_name(),
                **facts,
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
        # Committed: from here a failed cleanup must never delete the environment just activated.
        destination = None
        keep = [directory, *([prior["directory"]] if prior else [])]
        try:
            wsl_host.guest(
                ["find", base, "-mindepth", "1", "-maxdepth", "1", "-name", "env-*"]
                + [arg for name in keep for arg in ("!", "-name", name)]
                + ["-exec", "rm", "-rf", "{}", "+"]
            )
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            pass
    finally:
        if destination is not None:
            try:
                wsl_host.guest(["rm", "-rf", destination])
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                pass
    _record_manifest(engine)
    _update(engine, state = "success", phase = "ready", message = "Engine installed")


def _install_wsl_rocm(engine: str, guest_run, progress, cancel: threading.Event) -> None:
    """ROCm's userspace and the DXG bridge in the distro, then the GPU check the host cannot make."""
    from . import wsl_host

    _update(
        engine, phase = "preparing_rocm", message = "Installing AMD ROCm in the Unsloth WSL environment"
    )
    key = wsl_host.ROCM_APT_KEY
    if wsl_host._sha256(key) != wsl_host.ROCM_APT_KEY_SHA256:
        raise RuntimeError(
            "Studio's copy of AMD's ROCm repository key is damaged. Reinstall Studio."
        )
    packages = [
        wsl_host.download(spec, spec["url"].rsplit("/", 1)[1], progress, cancel)
        for spec in (wsl_host.ROCDXG, wsl_host.ROCDXG_SMI)
    ]
    setup = f"{wsl_host.GUEST_ROOT}/bin/setup-rocm"
    wsl_host.put(setup, wsl_host.ROCM_SETUP, "755")
    guest_run([setup, *(wsl_host.to_guest_path(path) for path in (key, *packages))])
    found = sorted(
        set(
            re.findall(
                r"\bgfx[0-9a-f]+\b",
                wsl_host.guest(
                    ["/opt/rocm/bin/rocminfo"], env = wsl_host.rocm_environment(), timeout = 300
                ),
            )
        )
    )
    if not any(target in profile(engine)["gfx"] for target in found):
        raise RuntimeError(_unsupported_amd_gpu(found))
    wsl_host.write_state(rocm = wsl_host.ROCM_RELEASE)


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
        from . import wsl_host

        if (
            wsl_host.active()
            and not wsl_host.distro_ready()
            and wsl_host.wsl_exe() is not None
            and wsl_host.distro_name() in wsl_host.registered_distros()
        ):
            # Forgetting the engine now would strand its files inside the distro.
            raise RuntimeError("WSL is not responding, so the engine cannot be removed. Try again.")
        if wsl_host.active() and wsl_host.distro_ready():
            # The compile caches live beside the environments, not under them.
            wsl_host.guest(
                [
                    "rm",
                    "-rf",
                    f"{wsl_host.GUEST_ROOT}/engines/{engine}",
                    f"{wsl_host.GUEST_ROOT}/cache/{engine}",
                ]
            )
        shutil.rmtree(root, ignore_errors = False) if root.exists() else None
        with _lock:
            _jobs.pop(engine, None)
        (engine_root() / f"{engine}.job.json").unlink(missing_ok = True)
        (engine_root() / f"{engine}.cancel").unlink(missing_ok = True)
    _record_manifest(engine)
    return status(engine)


def remove_wsl_environment() -> list[dict]:
    """Unregister the private distro. Every engine lives in it, so all must be idle."""
    from contextlib import ExitStack
    from . import wsl_host

    if not wsl_host.active():
        raise RuntimeError("Studio only creates a WSL environment on Windows.")
    with ExitStack() as stack:
        for engine in PROFILES:
            stack.enter_context(engine_lease(engine, exclusive = True))
        wsl_host.unregister()
        for engine in PROFILES:
            root = engine_root() / engine
            if root.exists() and not root.is_symlink():
                shutil.rmtree(root)
            (engine_root() / f"{engine}.job.json").unlink(missing_ok = True)
            with _lock:
                _jobs.pop(engine, None)
    for engine in PROFILES:
        _record_manifest(engine)
    return [status(engine) for engine in PROFILES]


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
        if previous.get("host") == "wsl":
            from . import wsl_host
            try:
                wsl_host.guest(
                    ["test", "-f", f"{wsl_host.GUEST_ROOT}/engines/{engine}/{directory}/bin/python"]
                )
                present = True
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                present = False
        else:
            present = not path.is_symlink() and (path / "bin" / "python").is_file()
        if not present:
            raise RuntimeError(
                "The previous engine installation is unavailable. Repair the engine instead."
            )
        if stale(previous):
            raise RuntimeError(
                "The previous engine installation was built on packages Studio no longer has. Repair the engine instead."
            )
        if not built_for_this_gpu(engine, previous):
            raise RuntimeError(
                "The previous engine installation was built for another GPU platform. Update the engine instead."
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
