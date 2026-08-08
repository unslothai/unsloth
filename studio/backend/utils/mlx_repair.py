# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Best-effort MLX self-heal for Apple Silicon.

On macOS, Unsloth enables Train/Export only when the MLX training/export stack is
usable (see utils.hardware.hardware.detect_hardware -> CHAT_ONLY). MLX is pulled
only transitively via unsloth-zoo, and a resolver backtrack (mlx-vlm ->
transformers>=5 vs the single-env transformers pin) can silently drop it, leaving
Train/Export greyed out after a reinstall/update. This reinstalls mlx by name on
a background thread, then re-detects so the gate re-opens without a manual
`unsloth studio update`.

The install mirrors the main Apple Silicon installer (install_python_stack.py):
it points UV_OVERRIDE at overrides-darwin-arm64.txt so the resolver keeps the
Unsloth transformers pin AND installs a current mlx-vlm, and it requires the same
minimum versions unsloth-zoo declares so a backtracked old mlx-vlm (which still
imports but breaks VLM Train/Export) is never accepted as healthy.

Mirrors the runtime backend self-heal already used for tilelang
(core.training.worker._ensure_tilelang_backend_unconditional): default-on,
best-effort, opt out with UNSLOTH_DISABLE_MLX_AUTOREPAIR=1.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

import structlog

from utils.uv_path_safety import uv_safe_path

logger = structlog.get_logger(__name__)

DISABLE_ENV_VAR = "UNSLOTH_DISABLE_MLX_AUTOREPAIR"
# uv's wording when --python names a path it will not install into. Matched on the
# stable leading clause only: uv appends the offending path and a "run `uv venv`"
# hint, and has reworded the tail across releases.
_UNRESOLVED_PYTHON_MARKER = "No virtual environment or system Python installation found"
# Minimum versions unsloth-zoo requires on Apple Silicon (its pyproject darwin
# deps). mlx-vlm especially must be >=0.4.4: an older one still imports but
# breaks VLM Train/Export, so installing it would wrongly clear chat-only.
_MLX_MIN_VERSIONS = {"mlx": "0.22.0", "mlx-lm": "0.22.0", "mlx-vlm": "0.4.4"}
_MLX_PACKAGE_NAMES = tuple(_MLX_MIN_VERSIONS)
_MLX_RUNTIME_IMPORTS = ("mlx.core", "mlx_lm", "mlx_lm.sample_utils", "mlx_vlm")
MLX_PACKAGES = tuple(f"{name}>={version}" for name, version in _MLX_MIN_VERSIONS.items())
_MLX_REINSTALL_ARGS = tuple(
    arg for name in _MLX_PACKAGE_NAMES for arg in ("--reinstall-package", name)
)
# Require pre-built wheels for the unattended self-heal. A source distribution's
# PEP 517 build backend runs arbitrary code at install time, and this install is
# default-on, resolver-driven, and runs before the post-install stack check can
# reject anything. mlx/mlx-metal ship wheels only (no sdist on PyPI) and
# mlx-lm/mlx-vlm publish py3-none-any wheels, so requiring wheels does not break a
# healthy self-heal; if a wheel is genuinely unavailable the install fails and
# Unsloth stays chat-only (the existing safe fallback) until `unsloth studio update`.
_ONLY_BINARY_ARG = "--only-binary=:all:"
# Allowlist of environment variables forwarded to the install subprocess. The
# self-heal runs without confirmation on the default startup path, so it must not
# hand resolver/build code the full Unsloth environment. Everything outside this
# set is dropped, which excludes three dangerous classes by construction:
#   * secrets (HF_TOKEN, AWS_*, WANDB_API_KEY, ...) that a malicious wheel/sdist
#     build hook would otherwise read straight out of os.environ;
#   * package-source redirects (UV_INDEX*, UV_DEFAULT_INDEX, UV_FIND_LINKS,
#     PIP_INDEX_URL, ...) so a poisoned process env cannot silently repoint the
#     install at an attacker-controlled index/find-links;
#   * cache-dir redirects (UV_CACHE_DIR, XDG_CACHE_HOME) so a poisoned env cannot
#     point uv at an attacker-staged cache (cache poisoning / symlink writes). uv
#     falls back to its safe user-owned default cache, reused across runs anyway.
# uv still honours on-disk config (uv.toml / pip.conf), so a corporate mirror
# configured there keeps working; only process-env redirects are dropped. We set
# UV_OVERRIDE ourselves in _mlx_install_env, so a poisoned one here is ignored.
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
        # proxies + custom CA bundles so installs behind a corporate gateway work
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
    }
)
_REPAIR_TIMEOUT_S = 900

# Attempt at most once per process; success is sticky (mlx then imports and the
# guard short-circuits on the next boot).
_attempted = False
_attempted_lock = threading.Lock()
# The worker started by start_mlx_autorepair_if_needed, so callers can tell "still
# installing" from "done, whichever way it went". Written under _attempted_lock together
# with the latch above, since mlx_repair_in_flight() reads the pair as one state.
_repair_thread: Optional[threading.Thread] = None


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        return True
    except Exception:
        return False


def _mlx_runtime_imports_available() -> bool:
    for module in _MLX_RUNTIME_IMPORTS:
        try:
            importlib.import_module(module)
        except Exception:
            return False
    return True


def _mlx_versions_satisfy_minimums() -> bool:
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _dist_version

        from packaging.version import Version
    except Exception:
        return False
    for name, minimum in _MLX_MIN_VERSIONS.items():
        try:
            if Version(_dist_version(name)) < Version(minimum):
                return False
        except PackageNotFoundError:
            return False
        except Exception:
            return False
    return True


def mlx_stack_available() -> bool:
    """`import mlx.core` works AND mlx/mlx-lm/mlx-vlm meet unsloth-zoo's minimums.

    Check distribution versions before imports so a too-old but importable MLX
    module is not loaded into this process before repair can replace it."""
    if not _mlx_versions_satisfy_minimums():
        return False
    return _mlx_runtime_imports_available()


def mlx_repair_in_flight() -> bool:
    """True while the one-time self-heal can still overturn a chat-only verdict.

    Ask only about a host whose MLX stack has just been measured as unusable, which is
    what detect_hardware's "mlx_unavailable" verdict means. This answers "has the repair
    finished", not "does this host need one": the not-yet-started branch would otherwise
    have to re-probe the stack, and on the host that matters that means re-running the
    failing mlx imports on the event loop for every health poll.

    Detection runs on the warm thread and the repair is scheduled after it, so such a host
    settles chat-only first and only flips once the reinstall lands. Both halves of that
    window count, since both publish an answer the repair is about to replace: the stretch
    before the worker starts, and the worker itself. False the moment it has finished,
    whichever way it went, so a host that genuinely cannot train still gets a final
    verdict -- as it does when the self-heal is opted out of, or cannot apply at all.

    The not-yet-started half is unbounded here on purpose: this module cannot tell a
    repair that is moments away from starting from one whose scheduler never arrives.
    Callers holding a verdict back on the strength of it pair this with
    mlx_repair_started() and bound that half themselves."""
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return False
    if not is_apple_silicon():
        return False
    with _attempted_lock:
        attempted, thread = _attempted, _repair_thread
    if not attempted:
        return True
    return thread is not None and thread.is_alive()


def mlx_repair_started() -> bool:
    """True once start_mlx_autorepair_if_needed() has claimed the one-time latch.

    Splits mlx_repair_in_flight()'s True into its two halves for callers that treat them
    differently: a live worker is a reinstall that legitimately runs for many minutes,
    while "not started yet" is a promise nothing has kept yet. Reads the latch rather
    than the thread handle, so a worker whose start() blew up still counts as started and
    falls through to in_flight's aliveness check."""
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
    """The venv directory this interpreter runs from, or None outside a venv.

    `sys.prefix` differs from `sys.base_prefix` exactly when a venv is active.
    Confirm the marker file so a half-deleted tree is never named as the target."""
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
    """Minimal, allowlisted environment for the unattended mlx install.

    The self-heal runs without confirmation on the default startup path, so it
    forwards only the variables uv genuinely needs (see _MLX_ENV_ALLOWLIST) instead
    of the full Unsloth environment: secrets and package-source redirects in
    os.environ are dropped so a malicious resolver-selected artifact cannot read
    Unsloth secrets or be steered to a hostile index.

    Mirror the main installer (install_python_stack.py) by pointing UV_OVERRIDE at
    overrides-darwin-arm64.txt, which relaxes mlx-vlm/mlx-lm's transformers>=5
    requirement to >=4.57.6. Without it, uv keeps the Unsloth transformers pin only
    by silently backtracking mlx-vlm to an old, unsupported version (uv honours
    UV_OVERRIDE; plain pip ignores it, so the transformers constraint below is the
    pip-path safety net). We set UV_OVERRIDE ourselves, so a poisoned one in the
    process env is ignored.

    VIRTUAL_ENV is set from sys.prefix rather than forwarded from os.environ, for
    the same reason: it names the environment uv must install into, and taking it
    from the process env would let a caller redirect the install elsewhere.

    It does NOT rescue a venv whose bin/python has stopped resolving. An explicit
    --python outranks VIRTUAL_ENV, so uv reports the same unresolved-interpreter
    error either way; this once claimed otherwise, and named an interpreter-probe
    helper that was deleted with it. Nothing passable to `uv pip install` recovers
    that state -- --target and --prefix do exit 0, but resolve against whatever
    ambient interpreter uv finds and write a wrong-ABI or off-sys.path install,
    which is worse than staying chat-only because it defeats the
    mlx_stack_available() gate. That case is detected and reported instead: see
    the _UNRESOLVED_PYTHON_MARKER branch in attempt_mlx_repair."""
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
        # uv truncates UV_OVERRIDE at the first space (issue #6503).
        env.setdefault("UV_OVERRIDE", uv_safe_path(override))
    return env


def _transformers_constraint_args() -> tuple[list[str], str | None]:
    """Pin transformers to the running version for the mlx install.

    The install must never upgrade transformers underneath a running Unsloth
    (the single-env install pins transformers==4.57.6). With UV_OVERRIDE set this
    is belt-and-suspenders; on the plain-pip path (no UV_OVERRIDE support) it is
    the actual guard -- the resolver either finds an mlx build compatible with the
    pin or fails, leaving us chat-only rather than breaking Unsloth. Returns
    (pip args, temp file path to clean up).

    Read the version from installed metadata rather than `import transformers`:
    transformers can have valid metadata yet fail to import (e.g. an incompatible
    huggingface_hub), and in that case we still want to pin it so the mlx install
    cannot quietly upgrade it out from under Unsloth."""
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
    """Install a usable mlx/mlx-lm/mlx-vlm stack by name into the running venv.
    Best-effort; returns True iff the resulting stack meets unsloth-zoo's minimums
    (so a backtracked old mlx-vlm is rejected, not accepted). transformers is held
    at its pinned version so the install can never upgrade it underneath Unsloth."""
    # Prepare the constraint inside the try: this runs on a daemon thread, so an
    # exception here (e.g. tempfile.mkstemp failing on a full disk or bad TMPDIR)
    # must leave Unsloth chat-only, not crash the background self-heal thread.
    constraint_path = None
    try:
        constraint_args, constraint_path = _transformers_constraint_args()
        cmd = _uv_install_cmd(
            "--upgrade",
            _ONLY_BINARY_ARG,
            *_MLX_REINSTALL_ARGS,
            *constraint_args,
            *MLX_PACKAGES,
        )
        if cmd is None:
            logger.warning(
                "MLX self-heal requires uv so Unsloth can apply dependency overrides; "
                "staying chat-only. Run `unsloth studio update` to restore uv."
            )
            return False
        logger.info("MLX self-heal: installing %s", ", ".join(MLX_PACKAGES))
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
            # uv will not install into this environment at all, so the venv is broken
            # rather than merely missing MLX, and no install flag recovers it from in
            # here: `--python <venv>/bin/python`, `--python <venv>` and a bare
            # VIRTUAL_ENV all report the same error against an interpreter uv cannot
            # resolve. Only rebuilding the environment fixes it, so name the command
            # that does. uv's own text says to run `uv venv`, which would build an
            # environment Unsloth does not manage.
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
            ", ".join(f"{name}>={ver}" for name, ver in _MLX_MIN_VERSIONS.items()),
        )
        return False
    return True


def _run_repair_and_redetect(epoch: Optional[int] = None) -> None:
    if not attempt_mlx_repair():
        return
    try:
        from utils.hardware import hardware as hw

        # A pip install, so shutdown can land anywhere inside it. Scoping to the epoch read
        # before start() discards the re-detect rather than republish for a dead lifespan.
        with hw.owning_detection_epoch(epoch):
            hw.detect_hardware()  # flips CHAT_ONLY / DEVICE now that mlx imports
        if epoch is not None and hw.current_detection_epoch() != epoch:
            # The scoped pass declined: this repair outlived its lifespan. The install
            # still succeeded, so the live lifespan holds a verdict measured before mlx
            # existed and the _attempted latch blocks any later repair. Re-detect under
            # the live epoch, or a now-capable Mac stays chat-only until a restart.
            hw.detect_hardware()
        logger.info(
            "MLX self-heal succeeded; Train/Export enabled (reload the page). chat_only=%s",
            hw.CHAT_ONLY,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("MLX installed but hardware re-detection failed: %s", exc)


def start_mlx_autorepair_if_needed() -> bool:
    """If this is an Apple Silicon host whose MLX stack is missing or too old,
    reinstall it on a daemon thread (off the startup critical path) and re-detect
    on success. Returns True iff a repair thread was started. No-op (returns False)
    off Apple Silicon, when the stack is already adequate, when already attempted
    this process, or when disabled via UNSLOTH_DISABLE_MLX_AUTOREPAIR=1."""
    global _attempted, _repair_thread
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return False
    if not is_apple_silicon():
        return False
    if mlx_stack_available():
        return False
    from utils.hardware import hardware as _hw

    with _attempted_lock:
        if _attempted:
            return False
        _attempted = True
        # Built and started inside the lock that claims the latch, so the pair is never
        # half-published: a reader landing between the two sees "attempted, nothing alive"
        # and settles the very verdict this repair is about to overturn. The worker never
        # takes this lock, so the start() handshake cannot deadlock against it.
        # The epoch is read here rather than in the thread: it may not run for a while, and
        # reading it there would bind the pass to a later shutdown.
        _repair_thread = threading.Thread(
            target = _run_repair_and_redetect,
            args = (_hw.current_detection_epoch(),),
            daemon = True,
            name = "mlx-autorepair",
        )
        _repair_thread.start()
    # Logged outside the lock: a blocked stdout must not hold up mlx_repair_in_flight(),
    # which /api/health calls on the event loop.
    logger.warning(
        "Apple Silicon without a usable MLX stack; attempting a one-time background "
        "reinstall of mlx/mlx-lm/mlx-vlm to re-enable Train/Export. "
        "Set %s=1 to disable.",
        DISABLE_ENV_VAR,
    )
    return True
