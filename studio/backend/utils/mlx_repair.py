# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Best-effort MLX self-heal for Apple Silicon. On macOS, Unsloth enables Train/Export only when the MLX training/export stack is usable (see utils.hardware.hardware.detect_hardware -> CHAT_ONLY). MLX is pulled only transitively via unsloth-zoo, and a resolver backtrack (mlx-vlm -> transformers>=5 vs the single-env transformers pin) can silently drop it, leaving Train/Export greyed out after a reinstall/update, so this reinstalls mlx by name on a background thread and re-detects, reopening the gate without a manual `unsloth studio update`. The install mirrors the main Apple Silicon installer (install_python_stack.py): it points UV_OVERRIDE at overrides-darwin-arm64.txt so the resolver keeps the Unsloth transformers pin AND installs a current mlx-vlm, and it requires the same minimum versions unsloth-zoo declares so a backtracked old mlx-vlm (which still imports but breaks VLM Train/Export) is never accepted as healthy. Mirrors the runtime backend self-heal already used for causal-conv1d (core.training.worker._ensure_causal_conv1d_fast_path): default-on, best-effort, opt out with UNSLOTH_DISABLE_MLX_AUTOREPAIR=1."""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import structlog

from utils.uv_path_safety import uv_safe_path

logger = structlog.get_logger(__name__)

DISABLE_ENV_VAR = "UNSLOTH_DISABLE_MLX_AUTOREPAIR"
# Match uv's stable leading clause only; the tail has been reworded across releases.
_UNRESOLVED_PYTHON_MARKER = "No virtual environment or system Python installation found"
# unsloth-zoo's darwin floors; older mlx-vlm/mlx-lm import but break Train or batching.
_MLX_MIN_VERSIONS = {"mlx": "0.22.0", "mlx-lm": "0.31.2", "mlx-vlm": "0.4.4"}
_MLX_PACKAGE_NAMES = tuple(_MLX_MIN_VERSIONS)
_MLX_RUNTIME_IMPORTS = ("mlx.core", "mlx_lm", "mlx_lm.sample_utils", "mlx_vlm")
# Pinned install specs (mlx breaks in patch releases); keep in sync with unsloth-zoo.
# mlx 0.32.3 avoids the 0.32.2 GQA-8 decode bug.
_MLX_INSTALL_SPECS = {
    "mlx": "==0.32.3",
    "mlx-lm": "==0.31.3",
    "mlx-vlm": ">=0.4.4,<=0.7.4",
}
MLX_PACKAGES = tuple(f"{name}{spec}" for name, spec in _MLX_INSTALL_SPECS.items())


def _zoo_declared_specifier(package: str) -> str:
    """The version range the INSTALLED unsloth-zoo declares for `package`, or "".

    The specs above track the zoo in the repository, but the self-heal runs against whatever zoo is on
    the machine, and it never upgrades it. mlx-vlm 0.7.1 passes `cache` to `gated_delta_update`, which a
    zoo predating that keyword does not accept, so admitting 0.7.1 next to an older zoo raises TypeError
    at the first Qwen3.5 VLM training step, after mlx_stack_available() has already cleared the
    chat-only gate. Reading the installed zoo's own requirement is what keeps the two in step without
    naming a zoo version here: it widens on its own the moment a zoo that declares 0.7.1 is installed.
    """
    try:
        from importlib.metadata import requires
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib on every supported Python
        return ""
    try:
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
    except ImportError:
        return ""
    try:
        declared = requires("unsloth_zoo") or ()
    except Exception:
        return ""
    wanted = canonicalize_name(package)
    for raw in declared:
        try:
            requirement = Requirement(raw)
        except Exception:
            continue
        if canonicalize_name(requirement.name) == wanted and str(requirement.specifier):
            return str(requirement.specifier)
    return ""


def _install_packages() -> tuple[str, ...]:
    """MLX_PACKAGES, with mlx-vlm narrowed to what the installed unsloth-zoo declares.

    Only mlx-vlm: it is the one whose call shape the zoo has to match, and mlx/mlx-lm are pinned
    exactly at both ends, so intersecting those would just empty the range on any zoo a patch release
    behind. Both specifiers are passed and uv intersects them.
    """
    packages = []
    for name, spec in _MLX_INSTALL_SPECS.items():
        declared = _zoo_declared_specifier(name) if name == "mlx-vlm" else ""
        packages.append(f"{name}{spec},{declared}" if declared else f"{name}{spec}")
    return tuple(packages)


_MLX_REINSTALL_ARGS = tuple(
    arg for name in _MLX_PACKAGE_NAMES for arg in ("--reinstall-package", name)
)
# Wheels only: sdist build hooks run arbitrary code in this unattended install.
_ONLY_BINARY_ARG = "--only-binary=:all:"
# Env allowlist for the unattended install: excludes secrets, index and cache redirects.
_MLX_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "all_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        # uv's rustls reads these, not the CA bundle vars above.
        "UV_SYSTEM_CERTS",
        "UV_NATIVE_TLS",
    }
)
_REPAIR_TIMEOUT_S = 900

# At most once per process.
_attempted = False
_attempted_lock = threading.Lock()
# Written under _attempted_lock together with _attempted.
_repair_thread: Optional[threading.Thread] = None
# Bounds post-install verification imports, which can hang on a broken stack.
_repair_started_at: Optional[float] = None
_WORKER_BUDGET_S = _REPAIR_TIMEOUT_S + 300
_repair_clock = time.monotonic
_environment_mutated = False


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        return True
    except Exception:
        return False


# Per-part cap for _bounded(): folds and truncates free-form import errors and versions.
_BLOCKER_PART_CAP = 80
_BLOCKER_LINE_CAP = 200


def _bounded(text: str, cap: int = _BLOCKER_PART_CAP) -> str:
    """One line, capped, with an ellipsis marking anything dropped."""
    folded = " ".join(str(text).split())
    if len(folded) > cap:
        folded = folded[: cap - 3].rstrip() + "..."
    return folded


def _one_line(exc: BaseException) -> str:
    """An exception's message as one bounded line."""
    return _bounded(str(exc))


def _mlx_runtime_import_blocker() -> Optional[str]:
    """The first runtime import that will not load, and why. None when all do."""
    for module in _MLX_RUNTIME_IMPORTS:
        try:
            importlib.import_module(module)
        except Exception as exc:
            return f"{module} does not import ({type(exc).__name__}: {_one_line(exc)})"
    return None


def _mlx_runtime_imports_available() -> bool:
    return _mlx_runtime_import_blocker() is None


def _mlx_version_blockers() -> list[str]:
    """Every MLX package that is missing or below the minimum, named."""
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _dist_version

        from packaging.version import Version
    except Exception as exc:
        return [f"the version check could not run ({type(exc).__name__}: {_one_line(exc)})"]
    blockers: list[str] = []
    for name, minimum in _MLX_MIN_VERSIONS.items():
        try:
            installed = _dist_version(name)
        except PackageNotFoundError:
            blockers.append(f"{name} is not installed (needs >={minimum})")
            continue
        except Exception as exc:
            blockers.append(f"{name} could not be read ({type(exc).__name__}: {_one_line(exc)})")
            continue
        try:
            if Version(installed) < Version(minimum):
                blockers.append(f"{name} {_bounded(installed)} is older than {minimum}")
        except Exception as exc:
            blockers.append(
                f"{name} {_bounded(installed)} is unreadable "
                f"({type(exc).__name__}: {_one_line(exc)})"
            )
    return blockers


def _mlx_versions_satisfy_minimums() -> bool:
    return not _mlx_version_blockers()


def mlx_stack_blockers() -> list[str]:
    """Why this host cannot train with MLX, in the order the gate checks it. The gate itself is all-or-nothing, and "run `unsloth studio update`" is no help to someone who has just run it: a resolver backtrack leaves a stack that is present but unusable, and nothing said which package or which import was the problem. Same order as ``mlx_stack_available`` so the two cannot disagree. Empty means the stack is usable."""
    versions = _mlx_version_blockers()
    if versions:
        return [_bounded(line, _BLOCKER_LINE_CAP) for line in versions]
    blocker = _mlx_runtime_import_blocker()
    return [_bounded(blocker, _BLOCKER_LINE_CAP)] if blocker else []


def mlx_stack_available() -> bool:
    """`import mlx.core` works AND mlx/mlx-lm/mlx-vlm meet unsloth-zoo's minimums. Check distribution versions before imports so a too-old but importable MLX module is not loaded into this process before repair can replace it."""
    if not _mlx_versions_satisfy_minimums():
        return False
    return _mlx_runtime_imports_available()


def mlx_repair_in_flight() -> bool:
    """True while the one-time self-heal can still overturn a chat-only verdict. Ask only about a host whose MLX stack has just been measured as unusable, which is what detect_hardware's "mlx_unavailable" verdict means: this answers "has the repair finished", not "does this host need one", since the not-yet-started branch would otherwise have to re-probe the stack, and on the host that matters that means re-running the failing mlx imports on the event loop for every health poll. Detection runs on the warm thread and the repair is scheduled after it, so such a host settles chat-only first and only flips once the reinstall lands; both halves of that window count, since both publish an answer the repair is about to replace: the stretch before the worker starts, and the worker itself. False the moment it has finished, whichever way it went, so a host that genuinely cannot train still gets a final verdict, as it does when the self-heal is opted out of, or cannot apply at all, or when a worker outlives _WORKER_BUDGET_S without finishing. The not-yet-started half is unbounded here on purpose: this module cannot tell a repair that is moments away from starting from one whose scheduler never arrives, so callers holding a verdict back on the strength of it pair this with mlx_repair_started() and bound that half themselves."""
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return False
    if not is_apple_silicon():
        return False
    # A --no-torch install declines the self-heal, so the verdict settles now.
    if _installed_without_torch():
        return False
    with _attempted_lock:
        attempted, thread, started_at = _attempted, _repair_thread, _repair_started_at
    if not attempted:
        return True
    if thread is None or not thread.is_alive():
        return False
    if started_at is not None and _repair_clock() - started_at >= _WORKER_BUDGET_S:
        return False
    return True


def mlx_repair_started() -> bool:
    """True once start_mlx_autorepair_if_needed() has claimed the one-time latch. Splits mlx_repair_in_flight()'s True into its two halves for callers that treat them differently: a live worker is a reinstall that legitimately runs for many minutes, while "not started yet" is a promise nothing has kept yet. Reads the latch rather than the thread handle, so a worker whose start() blew up still counts as started and falls through to in_flight's aliveness check."""
    with _attempted_lock:
        return _attempted


def _uv_executable() -> str | None:
    """Find uv even when macOS GUI launchers start with a minimal PATH."""
    found = shutil.which("uv")
    if found:
        return found
    for candidate in (
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
        Path("/opt/homebrew/bin/uv"),
        Path("/usr/local/bin/uv"),
    ):
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        except OSError:
            continue
    return None


def _venv_root() -> str | None:
    """The venv directory this interpreter runs from, or None outside a venv. `sys.prefix` differs from `sys.base_prefix` exactly when a venv is active; confirm the marker file so a half-deleted tree is never named as the target."""
    if sys.prefix == sys.base_prefix:
        return None
    try:
        if (Path(sys.prefix) / "pyvenv.cfg").is_file():
            return sys.prefix
    except OSError:
        pass
    return None


def _uv_install_cmd(*args: str) -> list[str] | None:
    uv = _uv_executable()
    if not uv:
        return None
    return [uv, "pip", "install", "--python", sys.executable, *args]


def _mlx_install_env() -> dict[str, str]:
    """Minimal, allowlisted environment for the unattended mlx install. The self-heal runs without confirmation on the default startup path, so it forwards only the variables uv genuinely needs (see _MLX_ENV_ALLOWLIST) instead of the full Unsloth environment: secrets and package-source redirects in os.environ are dropped so a malicious resolver-selected artifact cannot read Unsloth secrets or be steered to a hostile index. Mirror the main installer (install_python_stack.py) by pointing UV_OVERRIDE at overrides-darwin-arm64.txt, which keeps mlx-vlm/mlx-lm on the Unsloth Transformers floor: without it, uv keeps the Unsloth transformers pin only by silently backtracking mlx-vlm to an old, unsupported version (uv honours UV_OVERRIDE; plain pip ignores it, so the transformers constraint below is the pip-path safety net). We set UV_OVERRIDE ourselves, so a poisoned one in the process env is ignored. VIRTUAL_ENV is set from sys.prefix rather than forwarded from os.environ, for the same reason: it names the environment uv must install into, and taking it from the process env would let a caller redirect the install elsewhere. It does NOT rescue a venv whose bin/python has stopped resolving: an explicit --python outranks VIRTUAL_ENV, so uv reports the same unresolved-interpreter error either way, and nothing passable to `uv pip install` recovers that state, since --target and --prefix do exit 0 but resolve against whatever ambient interpreter uv finds and write a wrong-ABI or off-sys.path install, which is worse than staying chat-only because it defeats the mlx_stack_available() gate. That case is detected and reported instead: see the _UNRESOLVED_PYTHON_MARKER branch in attempt_mlx_repair."""
    env = {key: os.environ[key] for key in _MLX_ENV_ALLOWLIST if key in os.environ}
    if (venv_root := _venv_root()) is not None:
        env["VIRTUAL_ENV"] = venv_root
    override = (
        Path(__file__).resolve().parents[1]
        / "requirements"
        / "single-env"
        / "overrides-darwin-arm64.txt"
    )
    if override.is_file():
        # uv truncates UV_OVERRIDE at the first space.
        env.setdefault("UV_OVERRIDE", uv_safe_path(override))
    return env


def _transformers_constraint_args() -> tuple[list[str], str | None]:
    """Pin transformers to the running version for the mlx install. The install must never upgrade transformers underneath a running Unsloth (the single-env install pins a compatible default). With UV_OVERRIDE set this is belt-and-suspenders; on the plain-pip path (no UV_OVERRIDE support) it is the actual guard, since the resolver either finds an mlx build compatible with the pin or fails, leaving us chat-only rather than breaking Unsloth. Returns (pip args, temp file path to clean up). Read the version from installed metadata rather than `import transformers`: transformers can have valid metadata yet fail to import (e.g. an incompatible huggingface_hub), and in that case we still want to pin it so the mlx install cannot quietly upgrade it out from under Unsloth."""
    from importlib.metadata import PackageNotFoundError, version as _dist_version

    try:
        transformers_version = _dist_version("transformers")
    except PackageNotFoundError:
        return [], None
    except Exception:
        return [], None
    fd, path = tempfile.mkstemp(prefix = "mlx_repair_", suffix = ".txt")
    with os.fdopen(fd, "w", encoding = "utf-8") as fh:
        fh.write(f"transformers=={transformers_version}\n")
    return ["--constraint", path], path


def attempt_mlx_repair(*, timeout: int = _REPAIR_TIMEOUT_S) -> bool:
    """Install a usable mlx/mlx-lm/mlx-vlm stack by name into the running venv. Best-effort; returns True iff the resulting stack meets unsloth-zoo's minimums (so a backtracked old mlx-vlm is rejected, not accepted). transformers is held at its pinned version so the install can never upgrade it underneath Unsloth."""
    global _environment_mutated
    # Inside the try: a failure here must leave Unsloth chat-only, not crash the thread.
    constraint_path = None
    try:
        constraint_args, constraint_path = _transformers_constraint_args()
        packages = _install_packages()
        cmd = _uv_install_cmd(
            "--upgrade",
            _ONLY_BINARY_ARG,
            *_MLX_REINSTALL_ARGS,
            *constraint_args,
            *packages,
        )
        if cmd is None:
            logger.warning(
                "MLX self-heal requires uv so Unsloth can apply dependency overrides; "
                "staying chat-only. Run `unsloth studio update` to restore uv."
            )
            return False
        logger.info("MLX self-heal: installing %s", ", ".join(packages))
        # Set before the wait: a partial uv reinstall already mutated the env.
        _environment_mutated = True
        result = subprocess.run(
            cmd,
            env = _mlx_install_env(),
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("MLX self-heal timed out after %ss; staying chat-only", timeout)
        return False
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("MLX self-heal could not start: %s", exc)
        return False
    finally:
        if constraint_path and os.path.exists(constraint_path):
            try:
                os.remove(constraint_path)
            except OSError:
                pass
    if result.returncode != 0:
        tail = (result.stdout or "")[-2000:]
        if _UNRESOLVED_PYTHON_MARKER in (result.stdout or ""):
            _environment_mutated = False
            logger.warning(
                "MLX self-heal could not use the Unsloth environment at %s: uv did not "
                "recognise it as a virtual environment. This usually means the venv's "
                "bin/python points at an interpreter that has since been upgraded or "
                "removed. Train/Export stay disabled until the environment is rebuilt: "
                "run `unsloth studio update`. uv said:\n%s",
                _venv_root() or sys.prefix,
                tail,
            )
            return False
        logger.warning("MLX self-heal failed (staying chat-only):\n%s", tail)
        return False
    importlib.invalidate_caches()
    if not mlx_stack_available():
        logger.warning(
            "MLX self-heal produced an incomplete or too-old MLX stack "
            "(need %s); staying chat-only.",
            # Quote the floors, not the install pins.
            ", ".join(f"{name}>={ver}" for name, ver in _MLX_MIN_VERSIONS.items()),
        )
        return False
    return True


def _run_repair_and_redetect(epoch: Optional[int] = None) -> None:
    repaired = attempt_mlx_repair()
    if not repaired and not _environment_mutated:
        return
    try:
        from utils.hardware import hardware as hw

        # Scope to the start epoch so a shutdown mid-install discards the re-detect.
        with hw.owning_detection_epoch(epoch):
            hw.detect_hardware()
        if epoch is not None and hw.current_detection_epoch() != epoch:
            # The scoped pass declined; re-detect under the live epoch or the Mac stays chat-only.
            hw.detect_hardware()
        if repaired:
            logger.info(
                "MLX self-heal succeeded; Train/Export enabled (reload the page). chat_only=%s",
                hw.CHAT_ONLY,
            )
        else:
            logger.info(
                "MLX self-heal installed but the stack is still unusable; re-measured so "
                "the reason matches what is now on disk: %s",
                hw.CHAT_ONLY_DETAIL,
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("MLX installed but hardware re-detection failed: %s", exc)


def _installed_without_torch() -> bool:
    """True when this venv was installed --no-torch (GGUF-only). Unknown reads as False: an install predating the manifest keeps today's repair behaviour rather than silently losing it."""
    try:
        from studio.install_manifest import recorded_no_torch
        return recorded_no_torch() is True
    except Exception:
        return False


def start_mlx_autorepair_if_needed() -> bool:
    """If this is an Apple Silicon host whose MLX stack is missing or too old, reinstall it on a daemon thread (off the startup critical path) and re-detect on success. True iff a repair thread was started; False off Apple Silicon, when already attempted this process, when the venv was installed --no-torch, or when disabled via UNSLOTH_DISABLE_MLX_AUTOREPAIR=1. An adequate stack starts no repair but still overturns a verdict that contradicts it."""
    global _attempted, _repair_thread, _repair_started_at
    if not is_apple_silicon():
        return False
    from utils.hardware import hardware as _hw

    # Opt-out declines a reinstall, but the overturn still runs when someone waits on it.
    no_torch = _installed_without_torch()
    opted_out = os.environ.get(DISABLE_ENV_VAR) == "1" or no_torch
    if opted_out and not _hw.verdict_blames_the_mlx_stack():
        return False
    # Read before measuring so a shutdown discards results published from it.
    epoch = _hw.current_detection_epoch()
    if mlx_stack_available():
        # Runs early and can race a transformers import, yielding a partial module.
        if _hw.overturn_the_mlx_verdict(epoch):
            logger.info(
                "MLX stack measures usable after the warm, against a chat-only verdict "
                "from before it; re-detected. Train/Export are back (reload the page)."
            )
        return False
    if opted_out:
        if no_torch and _hw.settle_the_no_torch_verdict(epoch):
            logger.info(
                "MLX stack measures unusable on a --no-torch install; Train/Export stay "
                "off by request. Reinstall without --no-torch to enable them."
            )
        return False

    with _attempted_lock:
        if _attempted:
            return False
        _attempted = True
        _repair_thread = threading.Thread(
            target = _run_repair_and_redetect,
            args = (epoch,),
            daemon = True,
            name = "mlx-autorepair",
        )
        # Stamped before start() so the budget covers the worker's whole life.
        _repair_started_at = _repair_clock()
        _repair_thread.start()
    # Log outside the lock so blocked stdout cannot stall mlx_repair_in_flight().
    logger.warning(
        "Apple Silicon without a usable MLX stack; attempting a one-time background "
        "reinstall of mlx/mlx-lm/mlx-vlm to re-enable Train/Export. "
        "Set %s=1 to disable.",
        DISABLE_ENV_VAR,
    )
    return True
