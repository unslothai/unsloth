# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install FlashInfer on demand for an NVFP4 load, without ever moving the resident torch stack.

The NVFP4 flashinfer backend (``diffusion_nvfp4_ops``) falls back to torchao whenever ``import
flashinfer`` fails, so a Blackwell host without the package silently runs the slower kernels. This
installs it the first time an NVFP4 load asks, under the same rules as the other lazy installs
(``diffusion_attention``, ``utils/ssm_runtime``): env opt-out, offline refusal, once per process.

What is installed is fixed, not resolved: ``flashinfer-python`` at the one version whose private
layout ``diffusion_nvfp4_dispatch`` and the fake impls in ``diffusion_nvfp4_ops`` were checked
against, plus the ``flashinfer-jit-cache`` built for the running CUDA, because without it every
kernel is JIT-compiled through nvcc, which a Studio host usually does not have. Every distribution
already installed is pinned to its exact version through a constraints file, so the resolver can
only ADD packages; the result is then verified in a fresh interpreter and rolled back on any drift.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import shutil
import subprocess
import weakref
import sys
import tempfile
import threading
from typing import Any, Callable, Optional

# auto (default) / 1 installs flashinfer when an NVFP4 load needs it; 0 never installs.
FLASHINFER_INSTALL_ENV = "UNSLOTH_NVFP4_FLASHINFER_INSTALL"

# The version the dispatch allowlist and the op fakes were read against. Exact, never a floor.
FLASHINFER_VERSION = "0.6.6"
FLASHINFER_PACKAGE = "flashinfer-python"
FLASHINFER_JIT_CACHE_PACKAGE = "flashinfer-jit-cache"
FLASHINFER_CUBIN_PACKAGE = "flashinfer-cubin"
FLASHINFER_JIT_CACHE_INDEX = "https://flashinfer.ai/whl/{tag}"
# CUDA builds flashinfer-jit-cache 0.6.6 is published for (linux x86_64 and aarch64 only).
FLASHINFER_JIT_CACHE_CUDA = ((12, 8), (12, 9), (13, 0))

# Reachability probe for the default index. A HEAD with a short timeout, so a host with no route
# refuses in seconds instead of leaving pip to retry for minutes inside a model load.
_PYPI_PROBE_URL = "https://pypi.org/simple/flashinfer-python/"
_CUSTOM_INDEX_ENVS = ("UV_INDEX_URL", "UV_DEFAULT_INDEX", "PIP_INDEX_URL")

# The jit-cache wheel is 1.2-1.8 GB; the timeout covers a slow link, not a hung resolver.
_INSTALL_TIMEOUT_S = 1800
_VERIFY_TIMEOUT_S = 300

_INSTALL_LOCK = threading.Lock()
# (ok, reason) of the one install this process attempted. Policy refusals (opt-out, offline, no
# network) are NOT recorded here, so a later load on a changed environment can still install.
_OUTCOME: Optional[tuple[bool, str]] = None
# The last reason any call gave for flashinfer being unavailable.
_LAST_REASON: Optional[str] = None
# The same, per backend object (image vs video), so a status route reports the reason for ITS loaded model and not
# whatever another backend's later load saw.
_REASONS: "weakref.WeakKeyDictionary[Any, Optional[str]]" = weakref.WeakKeyDictionary()

StatusCb = Optional[Callable[[str], None]]


def install_env() -> str:
    raw = os.environ.get(FLASHINFER_INSTALL_ENV, "auto").strip().lower()
    return "0" if raw in ("0", "false", "no", "off") else "auto"


def last_install_reason() -> Optional[str]:
    """Why flashinfer is not serving NVFP4 here, as the last ensure call saw it; None when it is."""
    return _LAST_REASON


def _cached_preflight_failure(index: Optional[int] = None) -> Optional[str]:
    """A memoised failed preflight, read without running one: this feeds a polled status route. With
    ``index`` only that device's record counts; without it, the first failed record on any device."""
    try:
        from . import diffusion_nvfp4_ops as ops
        with ops._PREFLIGHT_LOCK:
            records = (
                [ops._PREFLIGHT.get(index)] if index is not None else list(ops._PREFLIGHT.values())
            )
    except Exception:  # noqa: BLE001
        return None
    for record in records:
        if isinstance(record, dict) and not record.get("ok"):
            return f"flashinfer preflight failed: {record.get('reason', 'unknown')}"
    # A transient (allocation) failure is not memoised, yet it may be what put the resident model on torchao.
    transient = (
        [ops._PREFLIGHT_TRANSIENT.get(index)]
        if index is not None
        else list(ops._PREFLIGHT_TRANSIENT.values())
    )
    for record in transient:
        if isinstance(record, dict):
            return f"flashinfer preflight failed: {record.get('reason', 'unknown')}"
    return None


def _cuda_index(device: Any) -> Optional[int]:
    """The CUDA index a load targets, or None when it cannot be told without guessing."""
    if isinstance(device, int):
        return device
    try:
        import torch
        dev = torch.device(device) if device is not None else None
    except Exception:  # noqa: BLE001
        return None
    if dev is None or dev.type != "cuda":
        return None
    if dev.index is not None:
        return int(dev.index)
    try:
        return int(torch.cuda.current_device())
    except Exception:  # noqa: BLE001
        return None


def record_install_reason(
    owner: Any,
    ok: bool,
    reason: Optional[str],
    device: Any = None,
) -> None:
    """Bind an ensure outcome to ``owner`` (the loading backend) and the device it loaded on. Call it
    once the load has replaced the resident model, so a cancelled attempt cannot relabel it."""
    if owner is None:
        return
    try:
        _REASONS[owner] = (None if ok else reason, _cuda_index(device))
    except TypeError:  # not weak-referenceable: the process-wide reason stands in
        pass


def nvfp4_backend_fields(backend: Optional[str], owner: Any = None) -> dict:
    """The two status keys for a loaded NVFP4 denoiser: the backend it runs and, when that is torchao,
    why flashinfer is not serving it (the install refusal or failure, else a failed preflight). ``owner``
    is the backend object whose load asked; without it the process-wide last reason is used."""
    reason = None
    if backend == "torchao":
        if owner is not None and owner in _REASONS:
            own, index = _REASONS[owner]
            reason = own or _cached_preflight_failure(index)
        else:
            reason = (_LAST_REASON if owner is None else None) or _cached_preflight_failure()
    return {"transformer_quant_backend": backend, "transformer_quant_backend_reason": reason}


def reset_install_state() -> None:
    """Forget the per-process outcome. For tests."""
    global _OUTCOME, _LAST_REASON
    with _INSTALL_LOCK:
        _OUTCOME = None
        _LAST_REASON = None
        _REASONS.clear()


def _emit(
    logger: Any,
    status_cb: StatusCb,
    message: str,
    *,
    warn: bool = False,
) -> None:
    if logger is not None:
        (logger.warning if warn else logger.info)("nvfp4.flashinfer: %s", message)
    if status_cb is not None:
        try:
            status_cb(message)
        except Exception:  # noqa: BLE001 - status is best-effort; never fail a load over a UI message
            pass


def _finish(
    ok: bool,
    reason: str,
    logger: Any,
    status_cb: StatusCb,
    *,
    warn: Optional[bool] = None,
) -> tuple[bool, str]:
    global _LAST_REASON
    _LAST_REASON = None if ok else reason
    _emit(logger, status_cb, reason, warn = (not ok) if warn is None else warn)
    return ok, reason


ENV_LOCK_TIMEOUT_S = 900.0


def _env_lock_path() -> str:
    """One lock file per environment: next to its site-packages when writable, else in the temp dir."""
    import hashlib

    name = ".unsloth-flashinfer-install.lock"
    if os.access(sys.prefix, os.W_OK):
        return os.path.join(sys.prefix, name)
    digest = hashlib.sha256(os.path.realpath(sys.prefix).encode()).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), f"{name}.{digest}")


@contextlib.contextmanager
def _env_install_lock(timeout: float = ENV_LOCK_TIMEOUT_S):
    """Hold the cross-process install lock. Yields False when another process kept it past
    ``timeout``; yields True without locking when filelock is unavailable (the old behaviour)."""
    try:
        from filelock import FileLock, Timeout
    except ImportError:
        yield True
        return
    lock = FileLock(_env_lock_path())
    try:
        lock.acquire(timeout = timeout)
    except Timeout:
        yield False
        return
    try:
        yield True
    finally:
        lock.release()


def _await_inflight_install() -> None:
    """Wait for another Studio process installing flashinfer into this environment, without touching it.

    flashinfer resolves its jit-cache directory once, at import, so a process that imports it between the
    flashinfer-python and the jit-cache steps of another process's install JIT-compiles for its whole life.
    Only the pinned flashinfer-python without its jit-cache can be that window, so any other state (already
    imported, absent, another version, cache present) returns without touching the lock."""
    if "flashinfer" in sys.modules:
        return
    installed = _dist_version(FLASHINFER_PACKAGE)
    if installed is None or installed.split("+", 1)[0] != FLASHINFER_VERSION:
        return
    if _dist_version(FLASHINFER_JIT_CACHE_PACKAGE) is not None:
        return
    # A timeout falls through to the import, as before this wait existed.
    with _env_install_lock():
        pass


def _import_flashinfer() -> tuple[bool, str]:
    from .diffusion_nvfp4_ops import _flashinfer_available
    importlib.invalidate_caches()
    return _flashinfer_available()


def _dist_version(name: str) -> Optional[str]:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - unreadable metadata reads as absent
        return None


def _canonical(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(".", "-")


def installed_distributions() -> dict[str, str]:
    """``{canonical name: version}`` for every distribution this interpreter can see."""
    from importlib.metadata import distributions

    importlib.invalidate_caches()
    found: dict[str, str] = {}
    for dist in distributions():
        try:
            name = dist.metadata["Name"]
            version = dist.version
        except Exception:  # noqa: BLE001 - a corrupt dist-info is skipped, not fatal
            continue
        if name and version:
            # The first entry on sys.path wins at import time, so it is the one to pin.
            found.setdefault(_canonical(name), str(version))
    return found


def _host_refusal(device: Any) -> Optional[str]:
    """Why this host cannot run the flashinfer NVFP4 kernels at all, or None."""
    if not sys.platform.startswith("linux"):
        return f"flashinfer publishes Linux wheels only (this host is {sys.platform})"
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return f"torch is not importable ({type(exc).__name__})"
    if getattr(torch.version, "hip", None):
        return "flashinfer NVFP4 kernels are CUDA only (this torch is a ROCm build)"
    if not getattr(torch.version, "cuda", None):
        return "flashinfer NVFP4 kernels need a CUDA build of torch"
    from .diffusion_nvfp4_ops import NVFP4_FLASHINFER_CAPS, _device_capability

    capability = _device_capability(device)
    if capability is None:
        return "no CUDA device capability to check"
    if tuple(capability) not in NVFP4_FLASHINFER_CAPS:
        return "sm_%d%d is not in the flashinfer NVFP4 set %s" % (
            capability[0],
            capability[1],
            ", ".join(sorted("sm_%d%d" % c for c in NVFP4_FLASHINFER_CAPS)),
        )
    return None


def _parse_cuda(version: Optional[str]) -> Optional[tuple[int, int]]:
    try:
        major, minor = str(version).split(".")[:2]
        return int(major), int(minor)
    except Exception:  # noqa: BLE001
        return None


def jit_cache_tag(cuda_version: Optional[str]) -> Optional[str]:
    """The ``cuXYZ`` jit-cache index for the running CUDA, or None when none is published.

    The newest build with the same major and a minor no newer than the running one: CUDA minor
    version compatibility lets a 12.8 build load on a 12.9 runtime, never the other way round, and
    never across a major (the libraries' sonames change)."""
    running = _parse_cuda(cuda_version)
    if running is None:
        return None
    usable = [c for c in FLASHINFER_JIT_CACHE_CUDA if c[0] == running[0] and c[1] <= running[1]]
    if not usable:
        return None
    major, minor = max(usable)
    return f"cu{major}{minor}"


def _nvcc_available() -> bool:
    """What flashinfer's JIT would find: CUDA_HOME first, then PATH, then the default toolkit."""
    for root in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")):
        if root and os.path.isfile(os.path.join(root, "bin", "nvcc")):
            return True
    return bool(shutil.which("nvcc")) or os.path.isfile("/usr/local/cuda/bin/nvcc")


def _reachable(url: str) -> bool:
    try:
        from utils.wheel_utils import url_exists
        return bool(url_exists(url))
    except Exception:  # noqa: BLE001 - an unanswerable probe means no network
        return False


def _torch_probe(timeout: int = 60) -> Optional[dict]:
    """The resident torch as a FRESH interpreter sees it; the running one would never show drift."""
    try:
        from utils.wheel_utils import probe_torch_wheel_env
        return probe_torch_wheel_env(timeout = timeout)
    except Exception:  # noqa: BLE001
        return None


def _uv_executable() -> Optional[str]:
    try:
        from utils.mlx_repair import _uv_executable as find_uv
        return find_uv()
    except Exception:  # noqa: BLE001
        return shutil.which("uv")


def _child_env() -> dict[str, str]:
    try:
        from utils.child_stdio import utf8_child_env
        from utils.native_path_leases import child_env_without_native_path_secret
        return utf8_child_env(child_env_without_native_path_secret())
    except Exception:  # noqa: BLE001
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        return env


def _installer_prefix(uv: Optional[str], index_url: Optional[str] = None) -> list[str]:
    """The install command up to its packages. ``index_url`` is the one index this step must use; it
    replaces the mirror rather than joining it, since uv refuses a repeated ``--index-url``."""
    if uv:
        cmd = [uv, "pip", "install", "--python", sys.executable]
        # uv reads only its own index settings; a pip-only mirror would otherwise be skipped for pypi.org, which the
        # preflight did not probe because a mirror is configured.
        pip_index = (os.environ.get("PIP_INDEX_URL") or "").strip()
        if (
            index_url is None
            and pip_index
            and not (os.environ.get("UV_INDEX_URL") or os.environ.get("UV_DEFAULT_INDEX"))
        ):
            cmd += ["--index-url", pip_index]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
        # The converse: pip ignores uv's settings, and the preflight skipped the pypi.org probe for them.
        uv_index = (
            os.environ.get("UV_DEFAULT_INDEX") or os.environ.get("UV_INDEX_URL") or ""
        ).strip()
        if index_url is None and uv_index and not (os.environ.get("PIP_INDEX_URL") or "").strip():
            cmd += ["--index-url", uv_index]
    if index_url is not None:
        cmd += ["--index-url", index_url]
    return cmd


def _uninstall_cmd(uv: Optional[str], names: list[str]) -> list[str]:
    if uv:
        return [uv, "pip", "uninstall", "--python", sys.executable, *names]
    return [sys.executable, "-m", "pip", "uninstall", "-y", *names]


def _write_constraints(pins: dict[str, str]) -> str:
    """Every installed distribution at its exact version. Versions that are not PEP 440 are left
    out rather than handed to a resolver that would reject the whole file."""
    try:
        from packaging.version import InvalidVersion, Version
    except Exception:  # noqa: BLE001
        Version = None
        InvalidVersion = Exception
    lines = []
    for name, version in sorted(pins.items()):
        if Version is not None:
            try:
                Version(version)
            except InvalidVersion:
                continue
        lines.append(f"{name}=={version}\n")
    fd, path = tempfile.mkstemp(prefix = "unsloth_flashinfer_", suffix = ".txt")
    with os.fdopen(fd, "w", encoding = "utf-8") as fh:
        fh.writelines(lines)
    try:
        from utils.uv_path_safety import uv_safe_path
        return uv_safe_path(path)
    except Exception:  # noqa: BLE001
        return path


# Index sources the installers read from the environment. The jit-cache step drops them: an extra index outranks
# uv's --index-url, and under uv's first-index strategy one carrying any flashinfer-jit-cache hides the pinned index.
_INDEX_SOURCE_ENVS = (
    "UV_INDEX",
    "UV_EXTRA_INDEX_URL",
    "UV_FIND_LINKS",
    "UV_INDEX_URL",
    "UV_DEFAULT_INDEX",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "PIP_FIND_LINKS",
)


def _pinned_index_env() -> dict[str, str]:
    env = _child_env()
    for name in _INDEX_SOURCE_ENVS:
        env.pop(name, None)
    return env


def _run(
    run: Callable[..., Any],
    cmd: list[str],
    timeout: int,
    env: Optional[dict[str, str]] = None,
) -> tuple[bool, str]:
    try:
        result = run(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = timeout,
            env = _child_env() if env is None else env,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    output = str(getattr(result, "stdout", "") or "")
    return getattr(result, "returncode", 1) == 0, output.strip()[-2000:]


def _verify_child(run: Callable[..., Any]) -> tuple[bool, str]:
    """``import flashinfer`` in a fresh interpreter, which is where the next process will do it."""
    code = (
        "import flashinfer, sys; "
        "sys.stdout.write(str(getattr(flashinfer, '__version__', 'unknown')))"
    )
    ok, output = _run(run, [sys.executable, "-c", code], _VERIFY_TIMEOUT_S)
    if not ok:
        return False, f"import flashinfer failed after the install: {output[-300:]}"
    version = output.strip().splitlines()[-1] if output.strip() else ""
    if version != FLASHINFER_VERSION:
        return (
            False,
            f"flashinfer imported as {version or 'unknown'}, expected {FLASHINFER_VERSION}",
        )
    return True, version


def _drift(before: dict[str, str], after: dict[str, str]) -> dict[str, tuple[str, Optional[str]]]:
    return {name: (old, after.get(name)) for name, old in before.items() if after.get(name) != old}


def _rollback(
    run: Callable[..., Any], uv: Optional[str], before: dict[str, str], logger: Any
) -> str:
    """Remove what the install added and put back anything it moved. Returns a one-line summary."""
    after = installed_distributions()
    added = sorted(set(after) - set(before))
    moved = _drift(before, after)
    notes = []
    if added:
        ok, output = _run(run, _uninstall_cmd(uv, added), _VERIFY_TIMEOUT_S)
        # The count goes to the status line, the names to the log.
        notes.append(f"{'removed' if ok else 'could not remove'} {len(added)} added package(s)")
        if logger is not None:
            logger.warning(
                "nvfp4.flashinfer: rollback %s %s%s",
                "removed" if ok else "could not remove",
                ", ".join(added),
                "" if ok else f": {output[-500:]}",
            )
    if moved:
        # --no-deps: restore exactly these, and nothing a resolver would pull back in.
        specs = [f"{name}=={old}" for name, (old, _new) in sorted(moved.items())]
        ok, output = _run(run, _installer_prefix(uv) + ["--no-deps", *specs], _INSTALL_TIMEOUT_S)
        notes.append(("restored " if ok else "could not restore ") + ", ".join(specs))
        if not ok and logger is not None:
            logger.warning("nvfp4.flashinfer: rollback restore failed: %s", output[-500:])
    return "; ".join(notes) or "nothing to roll back"


def _install(
    device: Any, logger: Any, status_cb: StatusCb, run: Callable[..., Any]
) -> tuple[bool, str, bool]:
    """The install itself, called under ``_INSTALL_LOCK``. ``(ok, reason, memoise)``."""
    torch_before = _torch_probe()
    if not torch_before or not torch_before.get("cuda_version"):
        # Transient on a loaded box (the probe has a timeout), so not memoised.
        return False, "flashinfer not installed: the resident torch could not be probed", False
    tag = jit_cache_tag(torch_before.get("cuda_version"))
    if tag is None and not _nvcc_available():
        return (
            False,
            f"flashinfer not installed: no prebuilt {FLASHINFER_JIT_CACHE_PACKAGE} "
            f"{FLASHINFER_VERSION} for CUDA {torch_before.get('cuda_version')} and no nvcc to "
            "JIT-compile its kernels",
            True,
        )
    # A stray jit-cache or cubin of another version makes `import flashinfer` raise, and it is not
    # ours to replace.
    for extra in (FLASHINFER_JIT_CACHE_PACKAGE, FLASHINFER_CUBIN_PACKAGE):
        present = _dist_version(extra)
        if present is not None and present.split("+", 1)[0] != FLASHINFER_VERSION:
            return (
                False,
                f"flashinfer not installed: {extra} {present} is already installed and "
                f"flashinfer-python {FLASHINFER_VERSION} would refuse to import beside it",
                True,
            )
    # A configured mirror is the installer's to reach (and may carry credentials a HEAD cannot send), so pypi.org is
    # only probed when it is the index the installer will use. The jit-cache index is fixed, so it always is.
    probes = [] if any(os.environ.get(v) for v in _CUSTOM_INDEX_ENVS) else [_PYPI_PROBE_URL]
    if tag is not None:
        probes.append(
            FLASHINFER_JIT_CACHE_INDEX.format(tag = tag) + f"/{FLASHINFER_JIT_CACHE_PACKAGE}/"
        )
    for url in probes:
        if not _reachable(url):
            return False, f"flashinfer not installed: {url} is not reachable", False

    before = installed_distributions()
    uv = _uv_executable()
    constraints = _write_constraints(before)
    size_hint = " (about 1.5 GB of prebuilt kernels)" if tag else ""
    _emit(
        logger,
        status_cb,
        f"installing {FLASHINFER_PACKAGE} {FLASHINFER_VERSION}"
        f"{f' with {FLASHINFER_JIT_CACHE_PACKAGE} for {tag}' if tag else ''} for NVFP4{size_hint}",
    )
    try:
        steps = [
            # Constrained, not --no-deps: flashinfer cannot import without apache-tvm-ffi and friends,
            # and the constraints make every package already here immovable, torch included.
            (
                _installer_prefix(uv)
                + [
                    "--only-binary",
                    ":all:",
                    "-c",
                    constraints,
                    f"{FLASHINFER_PACKAGE}=={FLASHINFER_VERSION}",
                ],
                None,
            ),
        ]
        if tag is not None:
            steps.append(
                (
                    _installer_prefix(uv, FLASHINFER_JIT_CACHE_INDEX.format(tag = tag))
                    + [
                        "--only-binary",
                        ":all:",
                        "--no-deps",
                        f"{FLASHINFER_JIT_CACHE_PACKAGE}=={FLASHINFER_VERSION}",
                    ],
                    _pinned_index_env(),
                )
            )
        failure = None
        for cmd, step_env in steps:
            ok, output = _run(run, cmd, _INSTALL_TIMEOUT_S, step_env)
            if not ok:
                failure = "the installer failed: " + " ".join(output.split())[-400:]
                break
    finally:
        try:
            os.unlink(constraints)
        except OSError:
            pass

    if failure is None:
        moved = _drift(before, installed_distributions())
        torch_after = _torch_probe()
        if moved:
            failure = "the install changed " + ", ".join(
                f"{name} {old} -> {new or 'removed'}" for name, (old, new) in sorted(moved.items())
            )
        elif not torch_after or (
            torch_after.get("torch_version"),
            torch_after.get("cuda_version"),
        ) != (torch_before.get("torch_version"), torch_before.get("cuda_version")):
            failure = "torch reads differently after the install (%s CUDA %s -> %s)" % (
                torch_before.get("torch_version"),
                torch_before.get("cuda_version"),
                "unreadable"
                if not torch_after
                else f"{torch_after.get('torch_version')} CUDA {torch_after.get('cuda_version')}",
            )
        else:
            ok, detail = _verify_child(run)
            if not ok:
                failure = detail

    if failure is not None:
        rolled = _rollback(run, uv, before, logger)
        return False, f"flashinfer install rolled back ({rolled}): {failure}", True

    # The dispatch fast path memoises "flashinfer missing"; forget it now that it is not.
    try:
        from . import diffusion_nvfp4_dispatch, diffusion_nvfp4_ops
        diffusion_nvfp4_dispatch.forget_availability()
        diffusion_nvfp4_ops.reset_preflight_cache()
    except Exception:  # noqa: BLE001
        pass
    ok, detail = _import_flashinfer()
    if not ok:
        return False, f"flashinfer installed but does not import in this process ({detail})", True
    return True, f"installed flashinfer {detail} for NVFP4", True


def ensure_flashinfer_for_nvfp4(
    device: Any = None,
    *,
    logger: Any = None,
    status_cb: StatusCb = None,
    run: Callable[..., Any] = subprocess.run,
    local_files_only: bool = False,
    owner: Any = None,
) -> tuple[bool, str]:
    """See ``_ensure``. ``owner`` (the loading backend object) keys the reason its status route reports,
    recorded at once; a loader that can still be cancelled with a model resident passes no owner and
    calls ``record_install_reason`` after the swap. ``local_files_only`` loads never install."""
    ok, reason = _ensure(
        device, logger = logger, status_cb = status_cb, run = run, local_files_only = local_files_only
    )
    record_install_reason(owner, ok, reason, device)
    return ok, reason


def _ensure(
    device: Any = None,
    *,
    logger: Any = None,
    status_cb: StatusCb = None,
    run: Callable[..., Any] = subprocess.run,
    local_files_only: bool = False,
) -> tuple[bool, str]:
    """Make ``import flashinfer`` work for an NVFP4 load on ``device``, installing it if allowed.

    ``(ok, reason)``; never raises. Only an eligible host installs (Linux, CUDA torch, a device in
    the flashinfer NVFP4 set), only once per process, and never while offline. The caller does not
    need the answer to be True: ``select_nvfp4_backend`` still decides, and a False here leaves it
    on torchao exactly as before, with the reason logged and kept for the status route."""
    global _OUTCOME
    try:
        from .diffusion_nvfp4_ops import BACKEND_TORCHAO, nvfp4_backend_env

        if nvfp4_backend_env() == BACKEND_TORCHAO:
            return _finish(False, "UNSLOTH_NVFP4_BACKEND=torchao", None, None)
        _await_inflight_install()
        present, detail = _import_flashinfer()
        if present:
            return _finish(True, f"flashinfer {detail} already installed", None, None)
        installed = _dist_version(FLASHINFER_PACKAGE)
        if installed is not None:
            # Present but broken is not repaired: the user may have built or pinned it.
            return _finish(
                False,
                f"flashinfer-python {installed} is installed but does not import ({detail})",
                logger,
                status_cb,
            )
        if local_files_only:
            return _finish(
                False, "local-only load: flashinfer is not downloaded", logger, status_cb
            )
        if install_env() == "0":
            return _finish(
                False,
                f"flashinfer is not installed and {FLASHINFER_INSTALL_ENV}=0",
                logger,
                status_cb,
                warn = False,
            )
        refusal = _host_refusal(device)
        if refusal is not None:
            # Not a failure worth a warning: this host would never select flashinfer anyway.
            return _finish(False, f"flashinfer not installed: {refusal}", logger, None, warn = False)
        cached = _OUTCOME
        if cached is not None:
            return _finish(cached[0], cached[1], None, None)
        from utils.utils import hf_env_offline

        # UV_OFFLINE too: the install probes PyPI and the flashinfer index before uv ever runs.
        if hf_env_offline() or os.environ.get("UV_OFFLINE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return _finish(
                False,
                "flashinfer is not installed and Unsloth is in offline mode",
                logger,
                status_cb,
            )
        with _INSTALL_LOCK:
            # A concurrent load may have installed it while this one waited.
            if _OUTCOME is not None:
                return _finish(_OUTCOME[0], _OUTCOME[1], None, None)
            # The rollback diffs installed distributions against a snapshot, so the whole transaction must
            # also exclude another Studio process installing into the same environment.
            with _env_install_lock() as held:
                if not held:
                    return _finish(
                        False,
                        "flashinfer not installed: another process is installing into this "
                        "environment",
                        logger,
                        status_cb,
                    )
                present, detail = _import_flashinfer()
                if present:
                    return _finish(True, f"flashinfer {detail} already installed", None, None)
                ok, reason, memoise = _install(device, logger, status_cb, run)
            if memoise:
                _OUTCOME = (ok, reason)
        return _finish(ok, reason, logger, status_cb)
    except Exception as exc:  # noqa: BLE001 - an install helper must never break a model load
        return _finish(
            False, f"flashinfer install skipped ({type(exc).__name__}: {exc})", logger, None
        )
