# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install FlashInfer on demand for an NVFP4 load, without ever moving the resident torch stack.

Pinned flashinfer-python plus the CUDA-matched jit-cache; every installed distribution is constrained to its
exact version, verified in a fresh interpreter, and rolled back on any drift.
"""

from __future__ import annotations

import ast
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
FLASHINFER_JIT_CACHE_CUDA = ((12, 8), (12, 9), (13, 0))

# Short-timeout HEAD: a host with no route refuses in seconds, not minutes of pip retries.
_PYPI_PROBE_URL = "https://pypi.org/simple/flashinfer-python/"
_CUSTOM_INDEX_ENVS = ("UV_INDEX_URL", "UV_DEFAULT_INDEX", "PIP_INDEX_URL")
# uv ranks these above its default index and stops at the first that has a package; pip never reads them.
_UV_INDEX_ENVS = ("UV_INDEX", "UV_EXTRA_INDEX_URL", "UV_FIND_LINKS")
_PIP_EXTRA_INDEX_ENVS = ("PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS")

_INSTALL_TIMEOUT_S = 1800
_VERIFY_TIMEOUT_S = 300

_INSTALL_LOCK = threading.Lock()
# Outcome of this process's one install; policy refusals are not recorded, so a changed env can still install.
_OUTCOME: Optional[tuple[bool, str]] = None
_LAST_REASON: Optional[str] = None
# Per backend object, so each status route reports its own model's reason.
_REASONS: "weakref.WeakKeyDictionary[Any, Optional[str]]" = weakref.WeakKeyDictionary()

StatusCb = Optional[Callable[[str], None]]


def install_env() -> str:
    raw = os.environ.get(FLASHINFER_INSTALL_ENV, "auto").strip().lower()
    return "0" if raw in ("0", "false", "no", "off") else "auto"


def last_install_reason() -> Optional[str]:
    """Why flashinfer is not serving NVFP4 here, as the last ensure call saw it; None when it is."""
    return _LAST_REASON


def _cached_preflight_failure(index: Optional[int] = None) -> Optional[str]:
    """A memoised failed preflight, without running one; ``index`` limits it to that device."""
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
    """Bind an ensure outcome to ``owner``; call after the swap so a cancelled load cannot relabel it."""
    if owner is None:
        return
    try:
        _REASONS[owner] = (None if ok else reason, _cuda_index(device))
    except TypeError:  # not weak-referenceable: the process-wide reason stands in
        pass


def nvfp4_backend_fields(backend: Optional[str], owner: Any = None) -> dict:
    """Status keys for a loaded NVFP4 denoiser: its backend and, on torchao, why flashinfer is not serving it."""
    reason = None
    from .diffusion_nvfp4_flag import nvfp4_diffusion_enabled

    if backend == "torchao" and nvfp4_diffusion_enabled():
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
    """Hold the cross-process install lock; False on timeout, True unlocked when filelock is missing."""
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


def _await_inflight_install(wait: bool = True) -> bool:
    """Wait out another process's flashinfer install here: importing mid-install misses the jit-cache for life.
    False when it outlived the wait; ``wait=False`` tries once."""
    if "flashinfer" in sys.modules:
        return True
    installed = _dist_version(FLASHINFER_PACKAGE)
    if installed is None or installed.split("+", 1)[0] != FLASHINFER_VERSION:
        return True
    with _env_install_lock() if wait else _env_install_lock(0) as held:
        return held


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
    """The ``cuXYZ`` jit-cache index: same major, newest minor <= running (minor compat only), or None."""
    running = _parse_cuda(cuda_version)
    if running is None:
        return None
    usable = [c for c in FLASHINFER_JIT_CACHE_CUDA if c[0] == running[0] and c[1] <= running[1]]
    if not usable:
        return None
    major, minor = max(usable)
    return f"cu{major}{minor}"


def _jit_cache_version(tag: Optional[str]) -> Optional[str]:
    """The exact ``flashinfer-jit-cache`` version for a ``cuXYZ`` tag, local segment included."""
    return None if tag is None else f"{FLASHINFER_VERSION}+{tag}"


def _same_version(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    try:
        from packaging.version import Version
        return Version(a) == Version(b)
    except Exception:  # noqa: BLE001
        return a.strip().lower() == b.strip().lower()


def _nvcc_available() -> bool:
    """What flashinfer's JIT would find: CUDA_HOME first, then PATH, then the default toolkit."""
    for root in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")):
        if root and os.path.isfile(os.path.join(root, "bin", "nvcc")):
            return True
    return bool(shutil.which("nvcc")) or os.path.isfile("/usr/local/cuda/bin/nvcc")


def _reachable(url: str) -> bool:
    """A HEAD answered; a failed TLS handshake still proves a route (the installer's trust settings decide TLS)."""
    import ssl
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(urllib.request.Request(url, method = "HEAD"), timeout = 10):
            return True
    except urllib.error.HTTPError:
        return False
    except urllib.error.URLError as exc:
        return isinstance(exc.reason, ssl.SSLError)
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


_TRUE = ("1", "y", "yes", "t", "true", "on")
_FALSE = ("0", "n", "no", "f", "false", "off")


def _flag(value: Any) -> Optional[bool]:
    """A pip / uv boolean setting; None when unset or unreadable."""
    if isinstance(value, bool):
        return value
    text = str(value if value is not None else "").strip().lower()
    return True if text in _TRUE else False if text in _FALSE else None


def _pip_settings(run: Callable[..., Any]) -> dict[str, str]:
    """pip's effective ``install`` options (env over ``[install]`` over ``[global]``); empty when pip cannot answer."""
    try:
        result = run(
            [sys.executable, "-m", "pip", "config", "list"],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 60,
            env = _child_env(),
        )
    except Exception:  # noqa: BLE001
        return {}
    if getattr(result, "returncode", 1) != 0:
        return {}
    layers: dict[str, dict[str, str]] = {"global": {}, "install": {}, ":env:": {}}
    for line in str(getattr(result, "stdout", "") or "").splitlines():
        key, sep, raw = line.partition("=")
        section, dot, name = key.strip().partition(".")
        if not sep or not dot or section not in layers:
            continue
        try:
            value = ast.literal_eval(raw.strip())
        except Exception:  # noqa: BLE001
            value = raw.strip()
        layers[section][name.strip().lower().replace("_", "-")] = str(value)
    merged: dict[str, str] = {}
    for section in ("global", "install", ":env:"):
        merged.update(layers[section])
    return merged


class _Unreadable(Exception):
    pass


def _uv_config_tables() -> Optional[list[tuple[dict, dict]]]:
    """``(top level, [pip])`` per uv config file ``uv pip install`` reads, highest priority first; None if unparseable."""
    explicit = (os.environ.get("UV_CONFIG_FILE") or "").strip()
    if not explicit and _flag(os.environ.get("UV_NO_CONFIG")):
        return []
    try:
        # novermin -- 3.11; the tomli / None fallback below is the guard, which vermin cannot see.
        import tomllib  # novermin
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            tomllib = None

    def _load(path: str, pyproject: bool = False) -> Optional[dict]:
        if not os.path.isfile(path):
            return None
        if tomllib is None:
            raise _Unreadable(path)
        try:
            with open(path, "rb") as fh:
                data = tomllib.load(fh)
        except Exception as exc:  # noqa: BLE001
            raise _Unreadable(path) from exc
        if pyproject:
            data = (data.get("tool") or {}).get("uv")
        return data if isinstance(data, dict) else None

    found: list[Optional[dict]] = []
    try:
        if explicit:
            found.append(_load(explicit))
        else:
            try:
                directory = os.getcwd()
            except OSError:
                directory = ""
            while directory:
                if os.path.isfile(os.path.join(directory, "uv.toml")):
                    found.append(_load(os.path.join(directory, "uv.toml")))
                    break
                # A pyproject.toml without [tool.uv] does not stop the search, as in uv.
                project = _load(os.path.join(directory, "pyproject.toml"), pyproject = True)
                if project is not None:
                    found.append(project)
                    break
                parent = os.path.dirname(directory)
                directory = "" if parent == directory else parent
            home = os.environ.get("XDG_CONFIG_HOME") or ""
            if not os.path.isabs(home):
                home = os.path.join(os.path.expanduser("~"), ".config")
            found.append(_load(os.path.join(home, "uv", "uv.toml")))
            system = "/etc/uv/uv.toml"
            for root in (os.environ.get("XDG_CONFIG_DIRS") or "/etc/xdg").split(":"):
                if not root:
                    break
                if os.path.isfile(os.path.join(root, "uv", "uv.toml")):
                    system = os.path.join(root, "uv", "uv.toml")
                    break
            found.append(_load(system))
    except _Unreadable:
        return None
    tables = []
    for data in found:
        if data is not None:
            pip = data.get("pip")
            tables.append((data, pip if isinstance(pip, dict) else {}))
    return tables


_CONFIG_INDEX_KEYS = ("index", "index-url", "extra-index-url", "find-links")


def _installer_config(uv: Optional[str], run: Callable[..., Any]) -> dict[str, Any]:
    """Installer config beyond our flags (env + files): ``refusal`` an unhonorable setting, ``mirror`` pypi.org may
    not be the index, ``own_index`` a default index of its own, ``unprobed`` a proxy the probe would not use."""
    config: dict[str, Any] = {
        "refusal": None,
        "mirror": False,
        "own_index": False,
        "unprobed": False,
    }
    if os.environ.get("ALL_PROXY", os.environ.get("all_proxy")) and not (
        os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    ):
        # uv and pip route through ALL_PROXY; urllib ignores it.
        config["unprobed"] = True
    if uv:
        config["mirror"] = any(os.environ.get(v) for v in _CUSTOM_INDEX_ENVS + _UV_INDEX_ENVS)
        tables = _uv_config_tables()
        if tables is None:
            # A config file this interpreter cannot read may name a mirror; the installer decides, not pypi.org.
            config["mirror"] = True
            return config

        def _first(key: str) -> Any:
            for section in (1, 0):
                for table in tables:
                    if key in table[section]:
                        return table[section][key]
            return None

        def _default(entry: Any) -> bool:
            return isinstance(entry, dict) and bool(entry.get("default"))

        for top, pip in tables:
            for table in (top, pip):
                if any(table.get(k) for k in _CONFIG_INDEX_KEYS):
                    config["mirror"] = True
                entries = table.get("index")
                if table.get("index-url") or (
                    isinstance(entries, list) and any(_default(i) for i in entries)
                ):
                    config["own_index"] = True
        offline = _flag(os.environ.get("UV_OFFLINE"))
        checks = (
            (_flag(_first("offline")) if offline is None else offline, "uv is configured offline"),
            (_flag(_first("no-index")), "uv is configured with no-index"),
            (
                _flag(_first("no-deps")),
                "uv is configured with no-deps, and flashinfer needs its dependencies",
            ),
            (
                _first("target"),
                "uv is configured to install into a target directory, not this environment",
            ),
            (_first("prefix"), "uv is configured to install into a prefix, not this environment"),
            # A same-version replacement is invisible to the version-only drift check, so it cannot be rolled back.
            (
                _flag(_first("reinstall")) or _first("reinstall-package"),
                "uv is configured to reinstall packages, which could replace installed ones unchecked",
            ),
        )
        config["refusal"] = next((why for hit, why in checks if hit), None)
        return config

    settings = _pip_settings(run)
    config["pip_settings"] = settings
    # pip still installs from an extra index or find-links when pypi.org is unreachable.
    config["mirror"] = any(
        os.environ.get(v) for v in _CUSTOM_INDEX_ENVS + _PIP_EXTRA_INDEX_ENVS
    ) or any(settings.get(k) for k in ("index-url", "extra-index-url", "find-links"))
    config["own_index"] = bool(settings.get("index-url"))
    if settings.get("proxy"):
        config["unprobed"] = True
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    checks = (
        (_flag(settings.get("no-index")), "pip is configured with no-index"),
        (
            _flag(settings.get("no-deps")),
            "pip is configured with no-deps, and flashinfer needs its dependencies",
        ),
        (
            settings.get("target"),
            "pip is configured to install into a target directory, not this environment",
        ),
        (
            settings.get("prefix"),
            "pip is configured to install into a prefix, not this environment",
        ),
        (
            settings.get("root"),
            "pip is configured to install under another root, not this environment",
        ),
        (
            _flag(settings.get("force-reinstall")) or _flag(settings.get("ignore-installed")),
            "pip is configured to force-reinstall or ignore installed packages, which could replace them unchecked",
        ),
        (
            _flag(settings.get("require-virtualenv")) and not in_venv,
            "pip requires a virtualenv and this interpreter is not in one",
        ),
    )
    config["refusal"] = next((why for hit, why in checks if hit), None)
    return config


def _installer_prefix(
    uv: Optional[str],
    index_url: Optional[str] = None,
    *,
    own_index: bool = False,
) -> list[str]:
    """The install command up to its packages; ``index_url`` replaces any mirror (uv refuses a repeated --index-url)."""
    if uv:
        cmd = [uv, "pip", "install", "--python", sys.executable]
        # uv ignores pip-only mirrors (and pip uv's, below); the preflight skipped pypi.org because one is configured.
        pip_index = (os.environ.get("PIP_INDEX_URL") or "").strip()
        if (
            index_url is None
            and pip_index
            and not own_index
            and not (os.environ.get("UV_INDEX_URL") or os.environ.get("UV_DEFAULT_INDEX"))
        ):
            cmd += ["--index-url", pip_index]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
        uv_index = (
            os.environ.get("UV_DEFAULT_INDEX") or os.environ.get("UV_INDEX_URL") or ""
        ).strip()
        if (
            index_url is None
            and uv_index
            and not own_index
            and not (os.environ.get("PIP_INDEX_URL") or "").strip()
        ):
            cmd += ["--index-url", uv_index]
    if index_url is not None:
        cmd += ["--index-url", index_url]
        if uv:
            # --index outranks configured indexes; under uv's first-index strategy one listing jit-cache hides ours.
            cmd += ["--index", index_url]
    return cmd


def _uninstall_cmd(uv: Optional[str], names: list[str]) -> list[str]:
    if uv:
        return [uv, "pip", "uninstall", "--python", sys.executable, *names]
    return [sys.executable, "-m", "pip", "uninstall", "-y", *names]


def _pinnable(dists: dict[str, str]) -> dict[str, str]:
    """Distributions a constraints file can hold; non-PEP 440 versions are dropped so the file stays valid."""
    try:
        from packaging.version import InvalidVersion, Version
    except Exception:  # noqa: BLE001
        return dict(dists)
    pins = {}
    for name, version in dists.items():
        try:
            Version(version)
        except InvalidVersion:
            continue
        pins[name] = version
    return pins


def _write_constraints(pins: dict[str, str]) -> str:
    """Every installed distribution at its exact version (see ``_pinnable``)."""
    lines = [f"{name}=={version}\n" for name, version in sorted(_pinnable(pins).items())]
    fd, path = tempfile.mkstemp(prefix = "unsloth_flashinfer_", suffix = ".txt")
    with os.fdopen(fd, "w", encoding = "utf-8") as fh:
        fh.writelines(lines)
    try:
        from utils.uv_path_safety import uv_safe_path
        return uv_safe_path(path)
    except Exception:  # noqa: BLE001
        return path


# Env index sources the jit-cache step drops: under uv's first-index strategy they can hide the pinned index.
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


# pip transport settings (proxy, CA) the jit-cache step keeps once pip.conf is skipped.
_PIP_TRANSPORT_SETTINGS = ("proxy", "cert", "client-cert", "trusted-host", "timeout", "retries")


def _pinned_index_env(pip_settings: Optional[dict[str, str]] = None) -> dict[str, str]:
    """The jit-cache step's env: pip.conf is skipped (its extra indexes could beat the pinned one) and the transport
    settings in ``pip_settings`` are re-exported."""
    env = _child_env()
    for name in _INDEX_SOURCE_ENVS:
        env.pop(name, None)
    if pip_settings is not None:
        env["PIP_CONFIG_FILE"] = os.devnull
        for key in _PIP_TRANSPORT_SETTINGS:
            name = "PIP_" + key.upper().replace("-", "_")
            value = pip_settings.get(key)
            if value and name not in env:
                env[name] = value
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
        return False, f"import flashinfer failed after the install: {_quoted(output, 300)}"
    version = output.strip().splitlines()[-1] if output.strip() else ""
    if version != FLASHINFER_VERSION:
        return (
            False,
            f"flashinfer imported as {version or 'unknown'}, expected {FLASHINFER_VERSION}",
        )
    return True, version


def _drift(before: dict[str, str], after: dict[str, str]) -> dict[str, tuple[str, Optional[str]]]:
    return {name: (old, after.get(name)) for name, old in before.items() if after.get(name) != old}


def _reported_installs(output: str) -> set[str]:
    """Canonical names the installer says it installed: uv's `` + name==version`` lines, pip's
    ``Successfully installed name-version ...``."""
    names: set[str] = set()
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("+ ") and "==" in line:
            names.add(_canonical(line[2:].split("==", 1)[0]))
        elif line.startswith("Successfully installed "):
            for item in line[len("Successfully installed ") :].split():
                if "-" in item:
                    names.add(_canonical(item.rsplit("-", 1)[0]))
    return names


def _requirement_closure(roots: set[str], candidates: set[str]) -> set[str]:
    """``roots`` plus installed dependencies within ``candidates``, for an install that reported nothing."""
    from importlib.metadata import distribution

    try:
        from packaging.requirements import Requirement
    except Exception:  # noqa: BLE001
        Requirement = None
    found: set[str] = set()
    pending = [name for name in roots if name in candidates]
    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        try:
            requires = distribution(name).requires or []
        except Exception:  # noqa: BLE001 - absent or unreadable metadata adds nothing
            continue
        for spec in requires:
            try:
                if Requirement is None:
                    continue
                req = Requirement(spec)
                if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                    continue
            except Exception:  # noqa: BLE001
                continue
            dep = _canonical(req.name)
            if dep in candidates and dep not in found:
                pending.append(dep)
    return found


def _quoted(output: str, limit: int) -> str:
    """Installer output for a reason or log line: index URLs can carry credentials the user must not see."""
    from utils.log_redaction import redact_log_text
    return redact_log_text(" ".join(str(output).split())[-limit:])


def _rollback(
    run: Callable[..., Any],
    uv: Optional[str],
    before: dict[str, str],
    logger: Any,
    own_index: bool = False,
    reported: Optional[set[str]] = None,
    unreported_step: bool = False,
) -> str:
    """Remove what this install reported adding and restore what it moved; returns a one-line summary."""
    after = installed_distributions()
    new = set(after) - set(before)
    ours = set(reported or ()) & new
    if not ours or unreported_step:
        # A step that died without its summary may have half-installed what no report names.
        ours |= _requirement_closure({FLASHINFER_PACKAGE, FLASHINFER_JIT_CACHE_PACKAGE}, new)
    added = sorted(ours)
    if new - ours and logger is not None:
        logger.info(
            "nvfp4.flashinfer: rollback leaves %s (not installed by this transaction)",
            ", ".join(sorted(new - ours)),
        )
    moved = _drift(before, after)
    if reported and not unreported_step:
        # Revert only what our installer reported: concurrent changes by another installer are not ours.
        moved = {name: change for name, change in moved.items() if name in reported}
    elif unreported_step:
        # No report: constraints pinned everything, so only unpinnable packages can be ours.
        pinned = _pinnable(before)
        moved = {name: change for name, change in moved.items() if name not in pinned}
    notes = []
    if added:
        ok, output = _run(run, _uninstall_cmd(uv, added), _VERIFY_TIMEOUT_S)
        notes.append(f"{'removed' if ok else 'could not remove'} {len(added)} added package(s)")
        if logger is not None:
            logger.warning(
                "nvfp4.flashinfer: rollback %s %s%s",
                "removed" if ok else "could not remove",
                ", ".join(added),
                "" if ok else f": {_quoted(output, 500)}",
            )
    if moved:
        # --no-deps: restore exactly these, and nothing a resolver would pull back in.
        specs = [f"{name}=={old}" for name, (old, _new) in sorted(moved.items())]
        ok, output = _run(
            run,
            _installer_prefix(uv, own_index = own_index) + ["--no-deps", *specs],
            _INSTALL_TIMEOUT_S,
        )
        notes.append(("restored " if ok else "could not restore ") + ", ".join(specs))
        if not ok and logger is not None:
            logger.warning("nvfp4.flashinfer: rollback restore failed: %s", _quoted(output, 500))
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
    # A stray jit-cache or cubin of another version breaks `import flashinfer`; not ours to replace.
    for extra in (FLASHINFER_JIT_CACHE_PACKAGE, FLASHINFER_CUBIN_PACKAGE):
        present = _dist_version(extra)
        if present is not None and present.split("+", 1)[0] != FLASHINFER_VERSION:
            return (
                False,
                f"flashinfer not installed: {extra} {present} is already installed and "
                f"flashinfer-python {FLASHINFER_VERSION} would refuse to import beside it",
                True,
            )
    # flashinfer checks only the public version, so a jit-cache for another CUDA would be kept; not ours to replace.
    cache = _dist_version(FLASHINFER_JIT_CACHE_PACKAGE)
    if cache is not None and not _same_version(cache, _jit_cache_version(tag)):
        return (
            False,
            f"flashinfer not installed: {FLASHINFER_JIT_CACHE_PACKAGE} {cache} is already installed and "
            f"does not match CUDA {torch_before.get('cuda_version')}"
            + (f" (expected {_jit_cache_version(tag)})" if tag else ""),
            True,
        )
    uv = _uv_executable()
    config = _installer_config(uv, run)
    if config["refusal"]:
        # Configuration, not a failure: not memoised, so a later load after it changes can still install.
        return False, f"flashinfer not installed: {config['refusal']}", False
    # A configured mirror is the installer's to reach; probe pypi.org only when it is the index (and not proxied).
    probes = [] if config["mirror"] else [_PYPI_PROBE_URL]
    if tag is not None:
        probes.append(
            FLASHINFER_JIT_CACHE_INDEX.format(tag = tag) + f"/{FLASHINFER_JIT_CACHE_PACKAGE}/"
        )
    if config["unprobed"]:
        probes = []
    for url in probes:
        if not _reachable(url):
            return False, f"flashinfer not installed: {url} is not reachable", False

    before = installed_distributions()
    constraints = _write_constraints(before)
    size_hint = " (about 1.5 GB of prebuilt kernels)" if tag else ""
    _emit(
        logger,
        status_cb,
        f"installing {FLASHINFER_PACKAGE} {FLASHINFER_VERSION}"
        f"{f' with {FLASHINFER_JIT_CACHE_PACKAGE} for {tag}' if tag else ''} for NVFP4{size_hint}",
    )
    reported: set[str] = set()
    try:
        steps = []
        # jit-cache FIRST: an interrupted install must not leave an importable flashinfer that reads as installed.
        if tag is not None:
            steps.append(
                (
                    _installer_prefix(uv, FLASHINFER_JIT_CACHE_INDEX.format(tag = tag))
                    + [
                        "--only-binary",
                        ":all:",
                        "--no-deps",
                        # The full local version: `==0.6.6` would accept a cache built for another CUDA.
                        f"{FLASHINFER_JIT_CACHE_PACKAGE}=={_jit_cache_version(tag)}",
                    ],
                    _pinned_index_env(None if uv else config.get("pip_settings", {})),
                )
            )
        steps.append(
            # Constrained, not --no-deps: flashinfer needs apache-tvm-ffi; constraints keep everything present immovable.
            (
                _installer_prefix(uv, own_index = config["own_index"])
                + [
                    "--only-binary",
                    ":all:",
                    "-c",
                    constraints,
                    f"{FLASHINFER_PACKAGE}=={FLASHINFER_VERSION}",
                ],
                None,
            )
        )
        failure = None
        unreported_step = False
        for cmd, step_env in steps:
            ok, output = _run(run, cmd, _INSTALL_TIMEOUT_S, step_env)
            step_reported = _reported_installs(output)
            reported |= step_reported
            if not ok:
                failure = "the installer failed: " + _quoted(output, 400)
                unreported_step = not step_reported
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
        rolled = _rollback(
            run, uv, before, logger, config["own_index"], reported, unreported_step = unreported_step
        )
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
    """See ``_ensure``. ``owner`` keys the status reason (a cancellable loader passes none and calls
    ``record_install_reason`` after the swap). Switch off: returns at once, no import, lock, subprocess or network."""
    from .diffusion_nvfp4_flag import nvfp4_diffusion_enabled

    if not nvfp4_diffusion_enabled():
        return False, "NVFP4 is disabled in this build"
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
    """Make ``import flashinfer`` work for an NVFP4 load on ``device``. ``(ok, reason)``; never raises; False leaves
    ``select_nvfp4_backend`` on torchao."""
    global _OUTCOME
    try:
        from .diffusion_nvfp4_ops import BACKEND_TORCHAO, nvfp4_backend_env

        if nvfp4_backend_env() == BACKEND_TORCHAO:
            return _finish(False, "UNSLOTH_NVFP4_BACKEND=torchao", None, None)
        # A load that may not install checks the lock once instead of waiting behind another process's download.
        declined = (
            "local-only load"
            if local_files_only
            else f"{FLASHINFER_INSTALL_ENV}=0"
            if install_env() == "0"
            else None
        )
        if not _await_inflight_install(wait = declined is None):
            # Not memoised: a later load imports once that install has finished.
            return _finish(
                False,
                "flashinfer not ready: another process is still installing it into this environment"
                + (f" ({declined}: not waiting for it)" if declined else ""),
                logger,
                status_cb,
            )
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
            # The rollback diffs a snapshot, so also exclude other Studio processes installing here.
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
