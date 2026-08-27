# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import contextlib
import functools
import importlib.util
import hashlib
import hmac
import http.client
import json
import os
import platform
import re
import secrets
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import types
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple
import typer

from unsloth_cli import _studio_deps, _studio_runtime_gate
from unsloth_cli._inference import SpeculativeType
from unsloth_cli.commands import _password_prompt

studio_app = typer.Typer(help = "Unsloth Studio commands.")


def _enable_verbose_access_logs() -> None:
    """Restore every per-request access log by disabling the burst dedup and the
    quiet-poll heartbeat. Inherited by the spawned/re-exec'd server via the env."""
    os.environ["UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS"] = "0"
    os.environ["UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS"] = "0"


# Resolve install root: UNSLOTH_STUDIO_HOME, then STUDIO_HOME alias, then
# sys.prefix inference (so a direct call to <root>/bin/unsloth resolves after
# the installer's env var has expired), then legacy ~/.unsloth/studio.
# UNSLOTH_STUDIO_HOME wins when both env vars are set.
# Both halves, and the 8 KB ceiling, are Test-UnslothCmdShimFile's in install.ps1 and
# _IsUnslothCmdShim's in scripts/uninstall.ps1. Bytes, not text: the shim is written
# without a BOM but an edited copy may carry one, and a decode error here would be
# indistinguishable from "not ours".
_CMD_SHIM_MARKERS = (b"unsloth-studio-managed-launcher", b"from unsloth_cli import app")
_CMD_SHIM_MAX_BYTES = 8192


def _looks_like_installer_managed_studio_home(candidate: Path) -> bool:
    """Sentinel check (studio.conf or bin shim) so a dev venv named
    unsloth_studio is not misidentified as a custom Unsloth root.

    On Windows bin\\unsloth.cmd counts too. Only install.sh writes
    share/studio.conf, so on a custom-root Windows install the generated
    unsloth.exe is the only sentinel there is, and antivirus quarantine deletes
    it -- after which this returns False, the root falls back to
    ~/.unsloth/studio, and every `unsloth studio ...` reads and writes the wrong
    installation. The .cmd is written by the same installer for the same
    directory, so it answers the same question.

    It has to be OUR .cmd: this decides which tree the CLI manages, and the
    directory is on PATH, so any file of that name would otherwise be enough to
    point a custom root at itself. Same marker Test-UnslothCmdShimFile in
    install.ps1 and the uninstaller's recursive-delete guard require.
    """
    if (candidate / "share" / "studio.conf").is_file():
        return True
    if platform.system() != "Windows":
        return (candidate / "bin" / "unsloth").is_file()
    if (candidate / "bin" / "unsloth.exe").is_file():
        return True
    return _is_managed_cmd_shim(candidate / "bin" / "unsloth.cmd")


def _is_managed_cmd_shim(path: Path) -> bool:
    """Whether *path* is the .cmd shim this installer generates.

    Read as bytes and matched on the marker line install.ps1 writes. A hand
    rolled wrapper that happens to invoke the CLI is not this, and must not be
    taken as proof of an installer-managed root.
    """
    try:
        if path.stat().st_size > _CMD_SHIM_MAX_BYTES:
            return False
        body = path.read_bytes()
    except OSError:
        return False
    return all(marker in body for marker in _CMD_SHIM_MARKERS)


def _resolve_studio_home() -> tuple[Path, bool]:
    override = (os.environ.get("UNSLOTH_STUDIO_HOME") or "").strip()
    if not override:
        override = (os.environ.get("STUDIO_HOME") or "").strip()
    if override:
        try:
            return Path(override).expanduser().resolve(), True
        except (OSError, ValueError):
            return Path(override).expanduser(), True
    try:
        prefix = Path(sys.prefix).resolve()
        if prefix.name == "unsloth_studio":
            inferred = prefix.parent
            legacy = (Path.home() / ".unsloth" / "studio").resolve()
            if inferred != legacy and _looks_like_installer_managed_studio_home(inferred):
                return inferred, True
    except (OSError, ValueError):
        pass
    return Path.home() / ".unsloth" / "studio", False


STUDIO_HOME, _STUDIO_HOME_IS_CUSTOM = _resolve_studio_home()


def _ensure_studio_env_exported() -> None:
    """Re-export UNSLOTH_STUDIO_HOME / UNSLOTH_LLAMA_CPP_PATH only for real
    custom roots so subprocesses inherit the right install. Called from each
    studio subcommand entry rather than at import time, to avoid leaking env
    state into unrelated importers (tests, --help, CLI introspection).
    """
    if not _STUDIO_HOME_IS_CUSTOM:
        return
    # Truthy-check (not setdefault) so a blank UNSLOTH_STUDIO_HOME= does not
    # suppress the inferred custom root.
    if not os.environ.get("UNSLOTH_STUDIO_HOME"):
        os.environ["UNSLOTH_STUDIO_HOME"] = str(STUDIO_HOME)
    # When override == legacy default, llama.cpp stays at ~/.unsloth/llama.cpp.
    try:
        _legacy_studio = (Path.home() / ".unsloth" / "studio").resolve()
        _is_legacy = STUDIO_HOME.resolve() == _legacy_studio
    except (OSError, ValueError):
        _is_legacy = STUDIO_HOME == (Path.home() / ".unsloth" / "studio")
    if _is_legacy:
        _llama_dir = Path.home() / ".unsloth" / "llama.cpp"
    else:
        _llama_dir = STUDIO_HOME / "llama.cpp"
    if not os.environ.get("UNSLOTH_LLAMA_CPP_PATH"):
        os.environ["UNSLOTH_LLAMA_CPP_PATH"] = str(_llama_dir)


BOOTSTRAP_PASSWORD_FILE = ".bootstrap_password"
DESKTOP_SECRET_FILE = ".desktop_secret"
# Mirrors studio/backend/auth/storage.py's sentinel; see the helpers below.
UNDELIVERED_CREDENTIAL_FILE = ".credential_undelivered"
DEFAULT_ADMIN_USERNAME = "unsloth"
DESKTOP_SECRET_PREFIX = "desktop-"
API_KEY_PBKDF2_SALT_KEY = "api_key_pbkdf2_salt"
DESKTOP_SECRET_HASH_KEY = "desktop_secret_hash"
DESKTOP_SECRET_CREATED_AT_KEY = "desktop_secret_created_at"
PBKDF2_ITERATIONS = 100_000
_START_API_KEY_MARKER_ENV = "_UNSLOTH_START_API_KEY_MARKER"
_CLOUDFLARE_INTENT_ENV = "_UNSLOTH_CLOUDFLARE_INTENT"


def _consume_start_api_key_marker_env() -> bool:
    """Consume the one-shot readiness marker passed across an Unsloth re-exec."""
    return os.environ.pop(_START_API_KEY_MARKER_ENV, None) == "1"


def _preserve_cloudflare_intent(cloudflare: Optional[bool], secure: bool) -> None:
    """Carry the user's tri-state choice across compatibility re-execs."""
    if _CLOUDFLARE_INTENT_ENV in os.environ:
        return
    if secure or cloudflare is True:
        intent = "enabled"
    elif cloudflare is False:
        intent = "disabled"
    else:
        intent = "unset"
    os.environ[_CLOUDFLARE_INTENT_ENV] = intent


# __file__ is unsloth_cli/commands/studio.py -- two parents up is the package root
# (either site-packages or the repo root for editable installs).
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _should_hide_windows_subprocesses() -> bool:
    """Hide child console windows only for non-interactive Windows launches."""
    if platform.system() != "Windows":
        return False
    try:
        return not sys.stdout.isatty()
    except (AttributeError, OSError, ValueError):
        return True


def _windows_hidden_subprocess_kwargs() -> dict[str, object]:
    """Return Windows-only Popen kwargs that suppress transient console windows."""
    if not _should_hide_windows_subprocesses():
        return {}

    kwargs: dict[str, object] = {}
    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if create_no_window:
        kwargs["creationflags"] = create_no_window

    startupinfo_factory = getattr(subprocess, "STARTUPINFO", None)
    startf_use_showwindow = getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
    sw_hide = getattr(subprocess, "SW_HIDE", 0)
    if startupinfo_factory is not None and startf_use_showwindow:
        startupinfo = startupinfo_factory()
        startupinfo.dwFlags |= startf_use_showwindow
        startupinfo.wShowWindow = sw_hide
        kwargs["startupinfo"] = startupinfo

    return kwargs


# Windows materialises the `unsloth` entry point as a generated, unsigned .exe that
# Application Control denies while the signed interpreter beside it still runs, so
# every managed invocation goes through the interpreter (issue #8490).
#
# Byte-identical to WINDOWS_CLI_ENTRYPOINT in studio/src-tauri/src/process.rs and to
# $script:UnslothCliTrampoline in install.ps1, which carries the full rationale for
# both halves and for the deliberate absence of -I.
_WINDOWS_CLI_ENTRYPOINT = (
    "import sys, os; sys.path[:1] = [x for x in sys.path[:1] if getattr(sys.flags, 'safe_path', False) or x not in ('', os.getcwd())]; "
    "sys.argv[0] = 'unsloth'; from unsloth_cli import app; sys.exit(app())"
)

# Asks the managed interpreter whether the trampoline's import would succeed, by
# performing that exact import. The sys.path[0] scrub is
# _WINDOWS_CLI_ENTRYPOINT's, kept a separate literal because a parity test
# literal_evals that constant; the two must agree or the probe answers for a
# different sys.path than the launch uses.
#
# `from unsloth_cli import app` rather than a find_spec, deliberately. find_spec
# locates without executing, which sounds like the lighter question and is the
# wrong one: it answers True for an empty unsloth_cli/ directory (a namespace
# package), for a package whose __init__ raises, and for one whose dependencies
# an interrupted install never fetched. Every one of those is a venv the
# trampoline cannot start, and this probe gates the headless-public strip of
# .bootstrap_password, so a false pass there is a public Unsloth with no login
# page and no plaintext recovery credential. Paying the import is what makes the
# probe's answer the launch's answer.
_MANAGED_CLI_IMPORT_PROBE = (
    "import sys, os; sys.path[:1] = [x for x in sys.path[:1] if getattr(sys.flags, 'safe_path', False) or x not in ('', os.getcwd())]; "
    "from unsloth_cli import app; sys.exit(0)"
)

# Seconds, and generous: this is a bare interpreter start plus the CLI package
# import, and it can run from cold on a machine whose antivirus is scanning the
# venv it just quarantined a file out of. The caller treats a timeout as "no
# verdict" rather than as a missing package, so overrunning it is not fatal.
_MANAGED_CLI_IMPORT_PROBE_TIMEOUT = 60

# CreateProcess refuses a program blocked by an Application Control policy with
# ERROR_ACCESS_DISABLED_BY_POLICY, which Python surfaces as OSError.winerror.
_ERROR_ACCESS_DISABLED_BY_POLICY = 1260


def _managed_cli_argv(
    python: Path,
    *args: str,
    isolated: bool = False,
) -> List[str]:
    """argv that runs the managed `unsloth` CLI through *python*.

    -X utf8 rather than PYTHONUTF8 so the encoding holds even for a caller that
    scrubs the environment. -X utf8 precedes -I because -I implies -E, which
    would discard PYTHONUTF8 but cannot touch a command-line -X.

    *isolated* mirrors the Isolation enum in studio/src-tauri/src/process.rs and
    carries the same rule: the default is inherit, because -I implies -E and -s
    and so drops PYTHONPATH, PYTHONWARNINGS, PYTHONHASHSEED and user
    site-packages that the console script honoured, an observable difference on
    machines with no policy at all. Only a caller that must reproduce an
    already-isolated launch asks for it, and today that is the desktop updater's
    health probe, matching build_update_command's Isolation::Isolated.
    """
    flags = ["-X", "utf8", "-I"] if isolated else ["-X", "utf8"]
    return [str(python), *flags, "-c", _WINDOWS_CLI_ENTRYPOINT, *args]


def _is_application_control_block(error: OSError) -> bool:
    """True when Windows refused to start a program because a policy blocks it.

    Distinct from a missing or corrupt executable: nothing ran, so the failure
    says nothing about the program itself.
    """
    return getattr(error, "winerror", None) == _ERROR_ACCESS_DISABLED_BY_POLICY


@contextlib.contextmanager
def _studio_runtime_launch_guard(*, inherited: bool = False):
    guard = _studio_runtime_gate.studio_runtime_launch_guard(
        STUDIO_HOME,
        inherited = inherited,
    )
    try:
        acquired = guard.__enter__()
    except _studio_runtime_gate.StudioRuntimeGateBusy:
        typer.echo(
            "Error: Unsloth installation is modifying the managed environment. "
            "Wait for it to finish, then try again.",
            err = True,
        )
        raise typer.Exit(1)
    except OSError as exc:
        typer.echo(f"Error: could not coordinate the Unsloth launch: {exc}", err = True)
        raise typer.Exit(1)

    try:
        yield acquired
    finally:
        guard.__exit__(None, None, None)


def _stream_for_subprocess(stream):
    """Return *stream* if it has a real OS file descriptor, else None.

    subprocess.run on Windows refuses to inherit std handles unless
    they're passed explicitly (otherwise close_fds=True forces
    bInheritHandles=False, and a CREATE_NO_WINDOW child ends up with
    no stdio at all). When sys.stdout / sys.stderr is a real fd-backed
    stream we want to hand it through; when it's been captured by a
    test harness (pytest's capsys, an in-memory wrapper, etc) we fall
    back to None so subprocess uses its default.
    """
    if stream is None:
        return None
    try:
        stream.fileno()
    except (AttributeError, OSError, ValueError):
        return None
    return stream


def _display_host_for_bind(run_mod, host: str) -> str:
    return run_mod._resolve_external_ip() if host in ("0.0.0.0", "::") else host


def _loopback_bind_host_for(host: str) -> str:
    return "::1" if host == "::" else "127.0.0.1"


def _url_host(host: str) -> str:
    return (
        f"[{host}]" if ":" in host and not (host.startswith("[") and host.endswith("]")) else host
    )


def _emit_run_cloudflare_notice(
    run_mod, host: str, display_host: str, actual_port: int, secure: bool
) -> None:
    from unsloth_cli._tool_policy import is_external_host

    if not is_external_host(host):
        return
    run_mod._verify_global_reachability(display_host, actual_port)
    run_mod._print_cloudflare_line(
        secure = secure,
        loopback_host = _loopback_bind_host_for(host),
    )


def _studio_venv_python() -> Optional[Path]:
    """Return the studio venv Python binary, or None if not set up."""
    if platform.system() == "Windows":
        p = STUDIO_HOME / "unsloth_studio" / "Scripts" / "python.exe"
    else:
        p = STUDIO_HOME / "unsloth_studio" / "bin" / "python"
    return p if p.is_file() else None


def _managed_cli_site_packages_layout(python: Path) -> bool:
    """On-disk hint that the venv holding *python* still carries the CLI.

    Weaker than the import probe below and only used when the probe could not be
    run at all: an empty ``unsloth_cli/`` or an orphaned dist-info left by an
    interrupted install answers yes here without being importable.

    The dist-info is accepted alongside the package directory because an
    editable install of the checkout leaves a .pth and no unsloth_cli/ here.
    """
    site_packages = python.parent.parent / "Lib" / "site-packages"
    if (site_packages / "unsloth_cli").is_dir():
        return True
    return any(site_packages.glob("unsloth-*.dist-info"))


def _managed_cli_package_present(python: Path) -> bool:
    """Whether the venv holding *python* can still import the package the CLI runs.

    Windows only, and only asked when the console script is gone. The generated
    unsloth.exe is what proves a CLI on POSIX, but on Windows nothing launches it
    any more (issue #8490) and antivirus quarantine deletes the unsigned stub
    while leaving the environment perfectly able to run.

    Asked of the interpreter rather than of site-packages, because the two are
    not the same claim. What Windows launches is the trampoline's
    ``from unsloth_cli import app``, so an orphaned ``unsloth-*.dist-info`` (a
    moved editable checkout, an interrupted install) or an empty ``unsloth_cli/``
    directory is metadata, not a runnable CLI. It matters here specifically:
    this gate stands in front of the headless-public strip of
    .bootstrap_password, so passing a venv that then fails to import is the one
    outcome the gate's placement exists to prevent -- a public Unsloth with no
    login page and no plaintext recovery credential.

    The probe runs the trampoline's own ``from unsloth_cli import app`` rather
    than a cheaper spec lookup (see _MANAGED_CLI_IMPORT_PROBE for why the cheaper
    one answers a different question), with the same sys.path[0] scrub applied so
    an ``unsloth_cli`` directory in the caller's cwd cannot answer for the venv.
    """
    if platform.system() != "Windows":
        return False
    try:
        probe = subprocess.run(
            [str(python), "-X", "utf8", "-c", _MANAGED_CLI_IMPORT_PROBE],
            capture_output = True,
            timeout = _MANAGED_CLI_IMPORT_PROBE_TIMEOUT,
            # Same as every other managed-interpreter probe here: a non-interactive
            # Windows launch must not flash a console window (issue #8490's sibling).
            **_windows_hidden_subprocess_kwargs(),
        )
    except subprocess.TimeoutExpired:
        # Slow is not broken. A cold venv under an antivirus scan can take longer
        # than the probe waits, and the re-exec this gate stands in front of has
        # no timeout at all, so it would still come up. Fall back to the on-disk
        # layout rather than aborting an install that works -- the failure this
        # whole fallback exists to remove.
        return _managed_cli_site_packages_layout(python)
    except (OSError, subprocess.SubprocessError):
        # No verdict for a different reason: the interpreter would not start at
        # all (missing, or denied by an Application Control policy). The re-exec
        # runs THAT interpreter, so it is going to fail the same way, and the
        # on-disk layout cannot say otherwise. Fail closed, because the caller
        # strips .bootstrap_password on a headless public launch before it
        # re-execs: passing here would leave a public Unsloth with no login page
        # and no plaintext recovery credential, which is worse than telling the
        # user to re-run setup.
        return False
    return probe.returncode == 0


def _hsa_override_gfx_arch(value: Optional[str]) -> Optional[str]:
    """gfx arch named by an HSA_OVERRIDE_GFX_VERSION value, or None if unreadable.

    libhsakmt (topology.c) reads it as a major.minor.stepping triple
    (``sscanf(envvar, "%u.%u.%u%c") != 3`` rejects anything else) and the target
    name concatenates the stepping in hex, which is why 9.0.10 is gfx90a:
    11.0.0 -> gfx1100, 11.5.1 -> gfx1151.

    Kept in sync with _hsa_override_gfx_arch in studio/install_python_stack.py and
    in install.sh.
    """
    if not value:
        return None
    # [0-9] rather than str.isdigit()/\d, both of which accept non-ASCII digits.
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value.strip()):
        return None
    major, minor, step = (int(p) for p in value.strip().split("."))
    # Steppings are a single hex nibble; anything wider is not a real target.
    if not (0 <= step <= 15) or major <= 0 or minor > 9:
        return None
    return f"gfx{major}{minor}{step:x}"


def _torch_requires_rocm_metapackage(venv_dir: Path) -> bool:
    """Whether the installed torch actually resolves through the ``rocm`` meta-package.

    AMD's per-gfx wheels depend on it; the generic pytorch.org ROCm wheels vendor their
    own runtime and depend on nothing, so after a switch between them the meta-package is
    left behind describing a family with no bearing on what torch loads. Unknown shapes
    answer False: refusing to arbitrate leaves the environment untouched.
    """
    for sp_pattern in ("lib/python*/site-packages", "Lib/site-packages"):
        for sp in venv_dir.glob(sp_pattern):
            for info in sp.glob("torch-*.dist-info"):
                if not re.fullmatch(r"torch-[^-]+\.dist-info", info.name):
                    continue
                metadata = info / "METADATA"
                if not metadata.is_file():
                    continue
                try:
                    text = metadata.read_text(encoding = "utf-8", errors = "replace")
                except OSError:
                    return False
                for line in text.splitlines():
                    if not line.lower().startswith("requires-dist:"):
                        continue
                    # `Requires-Dist: rocm[libraries,devel]==7.13.0` and plain `rocm` both
                    # count; `rocm-sdk-core` does not, it is a component not the arbiter.
                    if re.search(r"requires-dist:\s*rocm(?![-_a-z0-9])", line, re.IGNORECASE):
                        return True
                return False
    return False


def _installed_rocm_single_arch(venv_dir: Path) -> Optional[str]:
    """gfx arch the ROCm runtime in *venv_dir* ACTIVELY carries kernels for, or None.

    AMD's per-gfx index ships one runtime distribution per architecture,
    ``rocm-sdk-libraries-<family>``, and the torch beside it holds code objects for
    that family alone. Which one is live has to come from the ``rocm`` meta-package,
    whose ``Requires-Dist`` names the family AMD's torch resolved (verified on
    repo.amd.com/rocm/whl/gfx1151: ``rocm-sdk-libraries-gfx1151==7.13.0; extra ==
    "libraries"``). Globbing for a ``rocm_sdk_libraries_gfx*`` directory instead
    would read an ORPHAN: ``rocm`` upgrades in place across a family switch while the
    superseded runtime keeps its own distribution name and is never uninstalled, so a
    venv that has changed families holds both. Same reasoning and hazard as
    _installed_rocm_wheel_family in studio/install_python_stack.py.

    None means "do not act": no ``rocm``, unreadable metadata, more than one family
    named, or a MULTI-arch family such as gfx120x-all, whose runtime carries kernels
    for several ISAs and so contradicts no override.
    """
    # The bare `rocm` metadata only describes a LIVE install while torch still resolves
    # through it. Switching from AMD's per-gfx index to a generic pytorch.org one does not
    # uninstall it: the generic wheels vendor their own ROCm libraries and depend on no
    # meta-package, so `rocm` is orphaned and pip never removes it. Reading it then would
    # name the OLD family and clear an override the generic wheels may be the only reason
    # the GPU works at all.
    if not _torch_requires_rocm_metapackage(venv_dir):
        return None
    _metadata: Optional[Path] = None
    for sp_pattern in ("lib/python*/site-packages", "Lib/site-packages"):
        for sp in venv_dir.glob(sp_pattern):
            for info in sp.glob("rocm-*.dist-info"):
                # rocm-sdk-core and rocm-sdk-libraries-* also start "rocm-"; only the
                # bare `rocm` meta-package arbitrates.
                if (
                    re.fullmatch(r"rocm-[^-]+\.dist-info", info.name)
                    and (info / "METADATA").is_file()
                ):
                    _metadata = info / "METADATA"
                    break
    if _metadata is None:
        return None
    try:
        _text = _metadata.read_text(encoding = "utf-8", errors = "replace")
    except OSError:
        return None
    _families = set()
    for _line in _text.splitlines():
        if not _line.lower().startswith("requires-dist:"):
            continue
        _m = re.search(r"rocm[-_]sdk[-_]libraries[-_]([0-9a-zA-Z]+)", _line)
        if _m:
            _families.add(_m.group(1).lower())
    if len(_families) != 1:
        return None  # nothing to arbitrate with, or two runtimes and no tie-break
    _family = _families.pop()
    # Single ISA only: gfx120x-all style families cover several architectures, so an
    # override naming one of them is contradicted by nothing.
    return _family if re.fullmatch(r"gfx[0-9a-f]+", _family) else None


def _clear_hsa_override_contradicting_install(venv_dir: Path) -> Optional[str]:
    """Drop an HSA_OVERRIDE_GFX_VERSION no installed kernel can satisfy (#7331).

    libhsakmt (topology.c) writes the variable's major.minor.stepping straight into
    the KFD node's EngineId and ROCr names the agent from that, so the override
    decides the ISA every later process sees. Against per-gfx wheels, which hold code
    objects for one architecture, an override naming a different one leaves the
    runtime asking for kernels the install does not contain and every launch fails on
    the first allocation, exactly as before the routing fix.

    install.sh clears it for the one launch it performs itself, but that unset dies
    with the installer: `unsloth studio update` runs install_python_stack.py as a
    child (studio/setup.sh:1444) and every later launch inherits the user's shell
    instead. This is the chokepoint the exec, the Windows Popen and the in-process
    paths all pass through.

    Keyed on the INSTALL, never on a hardware probe: on a generic multi-arch index
    there is nothing to contradict and the override is often the only thing making
    the GPU usable. Returns the installed arch when the variable was dropped.
    """
    raw = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    # Windows ROCm ignores the variable entirely, so there is nothing to correct.
    if not raw or platform.system() == "Windows":
        return None
    arch = _installed_rocm_single_arch(venv_dir)
    if arch is None:
        return None
    named = _hsa_override_gfx_arch(raw)
    # Unreadable: libhsakmt rejects it too, so it is not this spoof to undo.
    if named is None or named == arch:
        return None
    os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)
    return arch


def _clear_hsa_override_before_launch(silent: bool = False) -> Optional[str]:
    """Run the #7331 spoof clear for whichever entry point is about to launch.

    Every launch needs it, not just plain ``unsloth studio``: the group callback
    returns early once a subcommand is named, and ``unsloth run`` is bound straight
    to ``studio_run``, so both would otherwise reach llama-server and the backend
    with the contradicting override still set. Idempotent, so chained entry points
    are free to call it twice.
    """
    _venv = STUDIO_HOME / "unsloth_studio"
    _arch = _clear_hsa_override_contradicting_install(
        Path(sys.prefix) if sys.prefix.startswith(str(_venv)) else _venv
    )
    if _arch is not None and not silent:
        typer.echo(
            f"Cleared HSA_OVERRIDE_GFX_VERSION: this install carries {_arch} kernels "
            f"only, so the runtime has to report the real arch. Remove the export "
            f"from your shell profile as well, or the next terminal restores it.",
            err = True,
        )
    return _arch


def _find_run_py() -> Optional[Path]:
    """Find studio/backend/run.py.

    No CWD dependency — works from any directory.
    Since studio/ is now a proper package (has __init__.py), it lives in
    site-packages after pip install, right next to unsloth_cli/.
    """
    # 1. Relative to __file__ (site-packages or editable repo root)
    run_py = _PACKAGE_ROOT / "studio" / "backend" / "run.py"
    if run_py.is_file():
        return run_py
    # 2. Unsloth venv's site-packages (Linux + Windows layouts)
    for pattern in (
        "lib/python*/site-packages/studio/backend/run.py",
        "Lib/site-packages/studio/backend/run.py",
    ):
        for match in (STUDIO_HOME / "unsloth_studio").glob(pattern):
            return match
    return None


def _install_state() -> dict:
    """verify_install() result for this install root.

    STUDIO_HOME is an extra search root so a CLI installed outside the managed
    venv still inspects the venv the desktop app launches.
    """
    return _studio_deps.install_state(extra_roots = (STUDIO_HOME / "unsloth_studio",))


_RUN_MODULE = None


def _load_run_module():
    """Import studio.backend.run without relying on package resolution.

    `studio update` can leave a partial ``site-packages/studio/backend/``
    tree (plugin build artefacts only). That shadowed tree wins over an
    editable install and breaks ``from studio.backend.run import ...``.
    Loading by file path sidesteps the conflict.
    """
    global _RUN_MODULE
    if _RUN_MODULE is not None:
        return _RUN_MODULE

    run_py = _find_run_py()
    if run_py is None:
        raise ImportError("Could not find studio/backend/run.py. Re-run: unsloth studio setup")

    loaded = sys.modules.get("studio.backend.run")
    if loaded is not None:
        # __file__ can be None for namespace packages from partial trees.
        loaded_path = Path(getattr(loaded, "__file__", None) or "").resolve()
        if loaded_path == run_py.resolve():
            _RUN_MODULE = loaded
            return _RUN_MODULE

    spec = importlib.util.spec_from_file_location("studio.backend.run", run_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load studio backend from {run_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["studio.backend.run"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("studio.backend.run", None)
        raise
    _RUN_MODULE = module
    return _RUN_MODULE


def _find_setup_script(repo_root: Optional[Path] = None) -> Optional[Path]:
    """Find studio/setup.sh or studio/setup.ps1.

    No CWD dependency — works from any directory.

    `repo_root` is the explicit --local checkout, when there is one. Its setup
    script has to win: the scripts build the frontend under their own
    $SCRIPT_DIR, and the editable install of `repo_root` removes the installed
    tree that the installed copy's script would have built into. studio/frontend
    /dist is gitignored, so a fresh checkout would then have no frontend at all.
    """
    name = "setup.ps1" if platform.system() == "Windows" else "setup.sh"
    # 0. The checkout the caller actually asked to install from. No fallback:
    #    dropping back to the installed copy's script is the exact behaviour
    #    this branch exists to stop, so a checkout without one is unusable
    #    rather than a reason to use somebody else's.
    if repo_root is not None:
        s = repo_root / "studio" / name
        return s if s.is_file() else None
    # 1. Relative to __file__ (site-packages or editable repo root)
    s = _PACKAGE_ROOT / "studio" / name
    if s.is_file():
        return s
    # 2. Unsloth venv's site-packages
    for pattern in (
        f"lib/python*/site-packages/studio/{name}",
        f"Lib/site-packages/studio/{name}",
    ):
        for match in (STUDIO_HOME / "unsloth_studio").glob(pattern):
            return match
    return None


# Mirror in studio/backend/run.py argparse + backend denylist test;
# bumping the cap in one place only desyncs.
_PARALLEL_MIN = 1
_PARALLEL_MAX = 64
_PARALLEL_DEFAULT_RUN = 4  # pre-PR hardcoded for `unsloth studio run`
# New Chat leaves the previous conversation generating and the admission queue caps decodes at
# the slot count, so at 1 every extra chat queues. _slots_that_fit_on_gpu() may cut it back.
_PARALLEL_DEFAULT_PLAIN = 4


def _resolve_secure(secure: bool, not_secure: bool) -> bool:
    """Reconcile the deprecated --not-secure alias with --secure/--no-secure.

    Typer parses --secure and --not-secure as independent options, so the alias
    cannot lean on Click's last-wins ordering the way --secure/--no-secure do.
    Restore that ordering from argv: --not-secure only forces secure off when it
    is the last of the secure flags on the command line, matching the backend's
    BooleanOptionalAction.
    """
    if not not_secure:
        return secure
    last_secure = max(
        (i for i, a in enumerate(sys.argv) if a in ("--secure", "--no-secure")),
        default = -1,
    )
    last_not_secure = max(
        (i for i, a in enumerate(sys.argv) if a == "--not-secure"),
        default = -1,
    )
    return secure if last_secure > last_not_secure else False


def _iter_editable_studio_source_roots(venv_dir: Path):
    """Yield repo roots from setuptools `__editable___*_finder.py` files in
    *venv_dir*'s site-packages whose MAPPING includes a `studio` entry.

    Returns the parent dir of the mapped `studio` package (i.e. the repo
    root), so callers can append `/studio/...` to reach any subdir.
    """
    import ast
    import re

    for sp_pattern in ("lib/python*/site-packages", "Lib/site-packages"):
        for sp in venv_dir.glob(sp_pattern):
            for finder in sp.glob("__editable___*_finder.py"):
                try:
                    src = finder.read_text(encoding = "utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                # Tolerate single- or multi-line dict literals; [^}]* still
                # rejects nested dicts, which the setuptools template never
                # emits for editable installs.
                m = re.search(r"^MAPPING\s*(?::[^=]*)?=\s*(\{[^}]*\})", src, re.M | re.S)
                if not m:
                    continue
                try:
                    mapping = ast.literal_eval(m.group(1))
                except (SyntaxError, ValueError):
                    continue
                # Defensive: literal_eval can return a set / list / None if the
                # matched literal is not a dict (regex captures `{...}`).
                if not isinstance(mapping, dict):
                    continue
                studio_pkg = mapping.get("studio")
                if studio_pkg:
                    yield Path(studio_pkg).parent


def _find_frontend_dist() -> Optional[Path]:
    """Locate a built `studio/frontend/dist` (containing index.html).

    Probes (in order): package-local default, installer venv site-packages,
    editable source roots referenced from the installer venv. Returns None
    if nothing servable is found, so callers can decide to error or proceed
    in `--api-only` mode.

    Fixes the silent 404 when another `unsloth` on PATH shadows the
    installer's binary and points `_PACKAGE_ROOT` at a site-packages copy
    that never received a vite build.
    """
    candidates: List[Path] = [_PACKAGE_ROOT / "studio" / "frontend" / "dist"]
    venv_dir = STUDIO_HOME / "unsloth_studio"
    for pattern in (
        "lib/python*/site-packages/studio/frontend/dist",
        "Lib/site-packages/studio/frontend/dist",
    ):
        candidates.extend(venv_dir.glob(pattern))
    for repo_root in _iter_editable_studio_source_roots(venv_dir):
        candidates.append(repo_root / "studio" / "frontend" / "dist")
    seen: set[Path] = set()
    for c in candidates:
        try:
            resolved = c.resolve()
        except OSError:
            resolved = c
        if resolved in seen:
            continue
        seen.add(resolved)
        if (c / "index.html").is_file():
            return c
    return None


# ── helpers for `unsloth studio run` ────────────────────────────────


def _wait_for_server(port: int, timeout: int = 30) -> bool:
    """Poll ``GET /api/health`` until the server responds 200 or *timeout* expires."""
    import urllib.request
    import urllib.error

    url = f"http://127.0.0.1:{port}/api/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout = 2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError, ConnectionError):
            pass
        time.sleep(0.5)
    return False


def _create_api_key_inprocess(name: str) -> str:
    """Create an API key via direct storage call (no HTTP needed).

    Bypasses the ``must_change_password`` gate that blocks HTTP
    ``POST /api/auth/api-keys`` on fresh installs.  Safe because the
    CLI already has filesystem access to ``~/.unsloth/studio``.
    """
    storage = _load_backend_auth_storage()

    raw_key, _row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME,
        name = name,
    )
    return raw_key


def _load_backend_auth_storage():
    run_py = _find_run_py()
    backend_dir = run_py.parent if run_py is not None else _PACKAGE_ROOT / "studio" / "backend"
    if backend_dir.is_dir() and str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    auth_dir = backend_dir / "auth"
    storage_py = auth_dir / "storage.py"
    loaded = sys.modules.get("auth.storage")
    if loaded is not None:
        # __file__ can be None for namespace packages from partial trees.
        loaded_path = Path(getattr(loaded, "__file__", None) or "").resolve()
        if loaded_path == storage_py.resolve():
            return loaded

    package = sys.modules.get("auth")
    package_paths = [Path(path).resolve() for path in getattr(package, "__path__", [])]
    if package is None or auth_dir.resolve() not in package_paths:
        package = types.ModuleType("auth")
        package.__path__ = [str(auth_dir)]
        package.__package__ = "auth"
        package.__file__ = str(auth_dir / "__init__.py")
        sys.modules["auth"] = package

    spec = importlib.util.spec_from_file_location("auth.storage", storage_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load backend auth storage from {storage_py}")
    storage = importlib.util.module_from_spec(spec)
    sys.modules["auth.storage"] = storage
    spec.loader.exec_module(storage)

    return storage


def _write_auth_secret(path: Path, secret: str) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    fd, tmp_name = tempfile.mkstemp(prefix = f".{path.name}.", dir = path.parent)
    tmp_path = Path(tmp_name)
    try:
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass
        # newline pins LF: text mode writes CRLF on Windows, and `$(cat ...)`
        # strips the LF but leaves the CR glued to the credential.
        with os.fdopen(fd, "w", encoding = "utf-8", newline = "\n") as f:
            fd = -1
            # Newline so `cat` doesn't run it into the shell prompt; readers strip.
            f.write(secret + "\n")
        os.replace(tmp_path, path)
    except Exception:
        if fd >= 0:
            os.close(fd)
        tmp_path.unlink(missing_ok = True)
        raise
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _connect_auth_db() -> sqlite3.Connection:
    auth_dir = STUDIO_HOME / "auth"
    auth_dir.mkdir(parents = True, exist_ok = True)
    conn = sqlite3.connect(auth_dir / "auth.db")
    # A live server writes this DB while the CLI runs; the default lock wait is zero.
    conn.execute("PRAGMA busy_timeout=5000")
    # Mirror backend storage.get_connection: this path can create auth/ and
    # auth.db (the pre-exposure gate writes here first), and sqlite3.connect
    # makes the DB 0644 under a 022 umask. Keep both private.
    for _path, _mode in ((auth_dir, 0o700), (auth_dir / "auth.db", 0o600)):
        try:
            os.chmod(_path, _mode)
        except OSError:
            pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS auth_user (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            jwt_secret TEXT NOT NULL,
            must_change_password INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            id INTEGER PRIMARY KEY,
            token_hash TEXT NOT NULL,
            username TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            is_desktop INTEGER NOT NULL DEFAULT 0,
            secret_gen TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            key_prefix TEXT NOT NULL,
            key_hash TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            last_used_at TEXT,
            expires_at TEXT,
            is_active INTEGER NOT NULL DEFAULT 1
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS app_secrets (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    # Mirror backend storage.get_connection: one-time link tokens live in the SAME
    # auth.db. The CLI never mints them, but _cli_update_password revokes any
    # outstanding rows in its password transaction (as the backend does), so the
    # table must exist here even on a DB the CLI created before the backend ran.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS link_tokens (
            jti        TEXT PRIMARY KEY,
            username   TEXT NOT NULL,
            expires_at TEXT NOT NULL
        );
        """
    )
    auth_columns = {row[1] for row in conn.execute("PRAGMA table_info(auth_user)")}
    if "must_change_password" not in auth_columns:
        conn.execute(
            "ALTER TABLE auth_user ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0"
        )
    refresh_columns = {row[1] for row in conn.execute("PRAGMA table_info(refresh_tokens)")}
    if "is_desktop" not in refresh_columns:
        conn.execute("ALTER TABLE refresh_tokens ADD COLUMN is_desktop INTEGER NOT NULL DEFAULT 0")
    if "secret_gen" not in refresh_columns:
        conn.execute("ALTER TABLE refresh_tokens ADD COLUMN secret_gen TEXT")
    conn.commit()
    return conn


def _pbkdf2_hex(value: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        value.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    ).hex()


def _hash_password(password: str) -> tuple[str, str]:
    salt = secrets.token_hex(16)
    pwd_hash = _pbkdf2_hex(password, salt.encode("utf-8"))
    return salt, pwd_hash


def _get_or_create_api_key_pbkdf2_salt(conn: sqlite3.Connection) -> bytes:
    row = conn.execute(
        "SELECT value FROM app_secrets WHERE key = ?",
        (API_KEY_PBKDF2_SALT_KEY,),
    ).fetchone()
    if row is None:
        salt_hex = secrets.token_hex(32)
        conn.execute(
            "INSERT OR IGNORE INTO app_secrets (key, value) VALUES (?, ?)",
            (API_KEY_PBKDF2_SALT_KEY, salt_hex),
        )
        row = conn.execute(
            "SELECT value FROM app_secrets WHERE key = ?",
            (API_KEY_PBKDF2_SALT_KEY,),
        ).fetchone()
    return bytes.fromhex(row[0])


def _ensure_cli_default_admin(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT 1 FROM auth_user WHERE username = ?",
        (DEFAULT_ADMIN_USERNAME,),
    ).fetchone()
    if row is not None:
        return

    bootstrap_password = secrets.token_urlsafe(32)
    password_salt, password_hash = _hash_password(bootstrap_password)
    conn.execute(
        """
        INSERT INTO auth_user (
            username,
            password_salt,
            password_hash,
            jwt_secret,
            must_change_password
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            DEFAULT_ADMIN_USERNAME,
            password_salt,
            password_hash,
            secrets.token_urlsafe(64),
            1,
        ),
    )
    _write_auth_secret(
        STUDIO_HOME / "auth" / BOOTSTRAP_PASSWORD_FILE,
        bootstrap_password,
    )


def _create_desktop_secret_in_cli() -> str:
    raw_secret = DESKTOP_SECRET_PREFIX + secrets.token_urlsafe(48)
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect_auth_db()
    try:
        _ensure_cli_default_admin(conn)
        secret_hash = _pbkdf2_hex(raw_secret, _get_or_create_api_key_pbkdf2_salt(conn))
        conn.execute(
            "INSERT OR REPLACE INTO app_secrets (key, value) VALUES (?, ?)",
            (DESKTOP_SECRET_HASH_KEY, secret_hash),
        )
        conn.execute(
            "INSERT OR REPLACE INTO app_secrets (key, value) VALUES (?, ?)",
            (DESKTOP_SECRET_CREATED_AT_KEY, now),
        )
        conn.commit()
        return raw_secret
    finally:
        conn.close()


def _should_prompt_password_change(
    *, cloudflare: Optional[bool], host: str, secure: bool, api_only: bool
) -> bool:
    """Whether this launch will expose Unsloth through the Cloudflare tunnel.

    CLI mirror of run.py's _cloudflare_tunnel_should_start, minus the Colab
    case (Colab launches never come through this CLI path). --secure implies
    the tunnel; --cloudflare only tunnels non-api-only wildcard binds.
    """
    if secure:
        return True
    if cloudflare is not True:
        return False
    return host in ("0.0.0.0", "::") and not api_only


def _prompt_streams_interactive() -> bool:
    """The prompt needs a real terminal for input and for the masked echo."""
    try:
        return sys.stdin.isatty() and sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def _undelivered_credential_path():
    """Sentinel marking an admin password committed but never shown to anyone.

    CLI mirror of storage.mark_credential_undelivered / credential_undelivered;
    the two must agree on the filename and on storing the committed
    password_hash, because either side may write it and the other may read it
    (the CLI rotates before re-exec'ing the backend).
    """
    return STUDIO_HOME / "auth" / UNDELIVERED_CREDENTIAL_FILE


def _mark_credential_undelivered(password_hash: str) -> None:
    """Best-effort: a failure only costs the retry its guard, never the launch."""
    path = _undelivered_credential_path()
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(password_hash, encoding = "utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except OSError:
        pass


def _clear_credential_undelivered() -> None:
    try:
        _undelivered_credential_path().unlink(missing_ok = True)
    except OSError:
        pass


def _credential_undelivered(password_hash: str) -> bool:
    """True when *password_hash* is the hash of a password that was never shown.

    Matching on the hash (not mere existence) is what makes this self-healing:
    `unsloth studio reset-password` rewrites the row, so the sentinel stops
    matching even on a machine where the unlink failed. Stale or torn sentinels
    are removed and reported False rather than refusing a launch unprovably.
    """
    path = _undelivered_credential_path()
    try:
        if not path.is_file():
            return False
        recorded = path.read_text(encoding = "utf-8").strip()
    except OSError:
        return False
    if recorded and hmac.compare_digest(recorded, password_hash):
        return True
    _clear_credential_undelivered()
    return False


def _one_time_secret_console_stream(*, skip = None):
    """Return an interactive-terminal stream to surface a one-time secret, or None.

    Mirrors run.py's ``_one_time_secret_stream`` fail-closed contract for the CLI
    parent: the auto-generated admin password must reach the operator's terminal
    BEFORE we rotate away the seeded bootstrap credential. Prefers stderr, then
    stdout, and requires a real TTY. A writable non-tty stream -- a ``> file``
    redirect, nohup.out, a systemd-journald pipe -- is NOT an ephemeral console:
    surfacing the credential there PERSISTS the plaintext where log consumers can
    read it (CWE-532), so it is skipped. Returns None when neither stream is a
    usable TTY -- a Windows pythonw/service wrapper (both absent), a closed stream,
    or a headless (nohup/systemd) launch with redirected output -- in which case
    the caller fails closed rather than rotate away and lose (or leak) the only
    credential. The CLI parent has no session-log tee, so no _TeeStream unwrapping
    is needed here.

    *skip* excludes an already-resolved console so a delivery that RAISED on it can
    retry the other one (see _deliver_auto_generated_credentials). The remaining
    candidate still has to pass every check above, so the retry cannot downgrade to
    a redirected, non-tty surface.
    """
    for candidate in (sys.stderr, sys.stdout):
        try:
            if candidate is None or getattr(candidate, "closed", False):
                continue
            if skip is not None and candidate is skip:
                continue
            if not callable(getattr(candidate, "write", None)):
                continue
            # A writable non-tty stream is a redirected file/journal/pipe that would
            # persist the one-time credential (CWE-532); only a real terminal is an
            # ephemeral surface. isatty() may be absent/raise on a wrapper stream ->
            # treated as non-interactive by the except below.
            if not candidate.isatty():
                continue
        except (AttributeError, ValueError):
            continue
        return candidate
    return None


def _generate_reset_password() -> str:
    """Readable 4-word passphrase; the user has to type this one back in."""
    try:
        import diceware
        return diceware.get_passphrase(
            options = diceware.handle_options(args = ["-n", "4", "-d", "", "-c"])
        )
    except Exception:
        return secrets.token_urlsafe(24)


def _cli_update_password(
    conn: sqlite3.Connection,
    username: str,
    new_password: str,
    *,
    revoke_api_keys: bool = False,
    require_must_change: bool = False,
) -> bool:
    """CLI mirror of backend update_password + change-password route effects.

    One transaction: rehash, rotate the JWT secret, clear must_change_password,
    revoke refresh tokens (PR #6651 finding), revoke outstanding one-time link
    tokens, drop the desktop secret, and (for a reset) the API keys the old
    credential could have minted. File cleanup happens after commit; a failed
    unlink must not roll the change back. Returns whether the row was written.

    ``require_must_change`` makes it a compare-and-set on must_change_password,
    mirroring backend storage.update_password: an auto-generated launch credential
    must not overwrite a password a user chose in a Studio tab between the
    must_change read and this write. Returns False when that guard rejects the
    update, and then NOTHING else is revoked -- the password that won the race
    belongs to another writer, and its sessions, link tokens, desktop secret and
    API keys are not ours to destroy. That is why the early return sits above
    ``if revoke_api_keys:`` in particular: that DELETE has no WHERE clause and
    would wipe every key for every user while leaving the password as the winner
    set it. The guard stays spelled ``require_must_change and rowcount == 0``
    rather than a bare rowcount check, because the unguarded callers
    (``reset_password``, ``_apply_supplied_password_before_launch``) rely on the
    revocations running even when the UPDATE matches nothing, e.g. a renamed
    admin row.

    Revoking link_tokens here mirrors backend storage.update_password: a link
    token is signed with a key derived from the JWT secret rotated in this same
    statement, so a leftover row would let a concurrent exchange that read the old
    key before the rotation still consume its jti and mint a session. Deleting the
    rows in the SAME transaction as the rotation closes that race.
    """
    password_salt, password_hash = _hash_password(new_password)
    guard = " AND must_change_password = 1" if require_must_change else ""
    with conn:
        cursor = conn.execute(
            f"""
            UPDATE auth_user
            SET password_salt = ?, password_hash = ?, jwt_secret = ?, must_change_password = 0
            WHERE username = ?{guard}
            """,
            (password_salt, password_hash, secrets.token_urlsafe(64), username),
        )
        if require_must_change and cursor.rowcount == 0:
            return False
        conn.execute("DELETE FROM refresh_tokens WHERE username = ?", (username,))
        conn.execute("DELETE FROM link_tokens WHERE username = ?", (username,))
        conn.execute(
            "DELETE FROM app_secrets WHERE key IN (?, ?)",
            (DESKTOP_SECRET_HASH_KEY, DESKTOP_SECRET_CREATED_AT_KEY),
        )
        if revoke_api_keys:
            conn.execute("DELETE FROM api_keys")
    for stale in (BOOTSTRAP_PASSWORD_FILE, DESKTOP_SECRET_FILE):
        stale_path = STUDIO_HOME / "auth" / stale
        try:
            stale_path.unlink(missing_ok = True)
        except OSError as exc:
            # The hash is already committed, so a failed unlink must NOT roll the
            # change back. But a locked-yet-writable file (Windows AV, read-only
            # auth dir) must be truncated: otherwise its stale plaintext survives
            # and generate_bootstrap_password() would re-validate this revoked
            # credential if auth.db is ever recreated. Mirrors backend
            # clear_bootstrap_password().
            try:
                stale_path.write_text("", encoding = "utf-8")
                cleared = True
            except OSError:
                cleared = False
            if cleared:
                typer.echo(
                    f"Warning: could not remove stale {stale} file ({exc}); cleared its "
                    "contents so the old credential cannot be reused.",
                    err = True,
                )
            else:
                typer.echo(
                    f"Warning: could not remove or clear stale {stale} file ({exc}); the "
                    "old credential is still on disk. Remove it manually to prevent reuse "
                    "after a reset.",
                    err = True,
                )
    return True


def _echo_auto_generated_credentials(
    username: str,
    password: str,
    *,
    out = None,
) -> None:
    """Surface an auto-generated admin credential once, on the parent's console.

    Writes to the pre-resolved *out* stream (the one the caller verified was usable
    before it rotated the seeded recovery password) so the credential lands on the
    exact console the fail-closed preflight checked; falls back to ``typer.echo`` on
    stderr for callers that do not resolve one. Mirrors run.py's
    ``_print_auto_generated_credentials``. Never logged elsewhere and never
    persisted; the re-exec'd child then sees must_change=0 and no-ops.
    """
    line = "=" * 70
    banner = (
        f"\n{line}\n"
        "  Unsloth Studio admin login (auto-generated for this public launch)\n"
        f"    Username: {username}\n"
        f"    Password: {password}\n"
        "  Save this now: it is shown once, not written to disk, and not in the\n"
        "  process list. Rotate later with `unsloth studio reset-password`.\n"
        f"{line}"
    )
    if out is None:
        typer.echo(banner, err = True)
    else:
        print(banner, file = out, flush = True)


def _deliver_auto_generated_credentials(username: str, password: str, *, out) -> bool:
    """Echo the one-time credential to *out*, retrying once on the other console.

    Mirrors run.py's ``_deliver_one_time_credential``. The console preflight runs
    before the rotation, but this write happens after ``_cli_update_password`` has
    committed the generated password and removed the seeded bootstrap credential.
    A terminal that disappears in between (a dropped SSH session; writes to the
    orphaned pty raise OSError EIO) would make the echo raise, aborting the launch
    with a live password nobody has ever seen. Retry once on the other console
    (re-resolved through the same tty/closed/writable preflight, so the retry can
    never land the credential in a redirected file or journal), and report whether
    it reached a console at all so the caller can fail closed instead of crashing.
    """
    fallback = _one_time_secret_console_stream(skip = out)
    # Never retry the stream that just failed (a stubbed resolver could return it).
    for stream in (out, fallback if fallback is not out else None):
        if stream is None:
            continue
        try:
            _echo_auto_generated_credentials(username, password, out = stream)
            return True
        except Exception:
            continue
    return False


def _log_secret_free_delivery_failure() -> None:
    """Explain an undeliverable one-time credential, WITHOUT echoing the secret.

    Reached only when every console refused the banner, so this message may itself
    fail to land; it is best-effort and deliberately carries no password (writing
    the value anywhere else would persist it, CWE-532). The non-zero exit is the
    part the caller can always rely on.
    """
    try:
        typer.echo(
            "Error: the auto-generated Unsloth admin password could not be shown: "
            "the console went away after the pre-rotation check. It is now the live "
            "password but was never displayed, so nothing can recover it. Reset the "
            "credential with `unsloth studio reset-password`, then relaunch.",
            err = True,
        )
    except Exception:
        pass


def _apply_supplied_password_before_launch(supplied_password: "str | None") -> None:
    """Non-interactively set the INITIAL admin password (from --password /
    UNSLOTH_STUDIO_PASSWORD / stdin) before the server binds, while the account
    still has its auto-generated bootstrap password.

    Only ever sets the FIRST password: an already-set one is a hard error (an
    override would be an auth bypass on a public launch), and an invalid value
    fails closed. Runs in the parent before any re-exec so the secret never
    crosses to the child argv.
    """
    if not supplied_password:
        return
    try:
        conn = _connect_auth_db()
    except (OSError, sqlite3.Error) as exc:
        typer.echo(
            f"Error: --password could not open the Unsloth auth database ({exc}); not starting.",
            err = True,
        )
        raise typer.Exit(1)
    try:
        _ensure_cli_default_admin(conn)
        conn.commit()
        row = conn.execute(
            "SELECT password_salt, password_hash, must_change_password "
            "FROM auth_user WHERE username = ?",
            (DEFAULT_ADMIN_USERNAME,),
        ).fetchone()
        if not row:
            typer.echo(
                "Error: --password could not initialize the admin account; not starting.",
                err = True,
            )
            raise typer.Exit(1)
        if not row[2]:
            typer.echo(
                "Error: an Unsloth admin password is already set; --password only sets "
                "the initial password. Change it in the UI, or run `unsloth studio "
                "reset-password` for a new one.",
                err = True,
            )
            raise typer.Exit(1)
        password_salt, password_hash = row[0], row[1]

        def _is_current_password(candidate: str) -> bool:
            return hmac.compare_digest(
                _pbkdf2_hex(candidate, password_salt.encode("utf-8")), password_hash
            )

        problem = _password_prompt.validate_new_password(supplied_password, _is_current_password)
        if problem is not None:
            typer.echo(f"Error: {problem} Not starting.", err = True)
            raise typer.Exit(1)
        _cli_update_password(conn, DEFAULT_ADMIN_USERNAME, supplied_password)
        typer.echo(f"Password updated for '{DEFAULT_ADMIN_USERNAME}'.", err = True)
    except (OSError, sqlite3.Error) as exc:
        # Any DB failure fails closed (typer.Exit is not caught here, so the
        # deliberate Exit(1) branches above propagate unchanged).
        typer.echo(
            f"Error: --password could not update the Unsloth auth database ({exc}); not starting.",
            err = True,
        )
        raise typer.Exit(1)
    finally:
        conn.close()


def _strip_seeded_bootstrap_password_or_exit(*, context: str) -> None:
    """Remove the seeded plaintext bootstrap password before a public re-exec.

    Version-independent protection: a re-exec'd child of ANY version (including an
    old studio-venv predating the pre-bind gate) then reads None instead of
    injecting the default credential into the public page. must_change_password
    stays set, so the login page still forces a change and the timer still arms.
    Removal IS the protection, so if it fails (locked file, read-only auth dir)
    fail closed rather than publish it.
    """
    bootstrap_file = STUDIO_HOME / "auth" / BOOTSTRAP_PASSWORD_FILE
    try:
        bootstrap_file.unlink(missing_ok = True)
    except OSError as exc:
        typer.echo(
            "Error: refusing to publish Unsloth on a public Cloudflare URL: "
            f"could not remove the seeded bootstrap password file ({exc}), so an "
            f"older Unsloth child could still serve the default credential ({context}). "
            "Delete it manually or change the admin password (run `unsloth studio` "
            "locally with a terminal attached, or `unsloth studio reset-password`), "
            "then retry.",
            err = True,
        )
        raise typer.Exit(1)


def _require_servable_frontend_or_exit(
    *, frontend: Optional[Path], api_only: bool, cloudflare: Optional[bool], host: str, secure: bool
) -> Optional[Path]:
    """Fail closed BEFORE the pre-exposure gate if a public UI launch has no
    login page to change the seeded password.

    The gate strips the seeded .bootstrap_password on a headless public launch,
    so if the child then cannot serve the login page the admin is locked out
    (must_change_password=1, no file, no UI) until `unsloth studio reset-password`.
    The login page is the ONLY in-band way to change the seeded password, so a
    public non-api-only launch must have a servable dist before the strip.

    Returns the dist to serve: a user-supplied --frontend (validated to contain
    index.html) or the auto-resolved built dist. Returns `frontend` unchanged for
    non-public or --api-only launches (no login page needed).
    """
    if api_only or not _should_prompt_password_change(
        cloudflare = cloudflare, host = host, secure = secure, api_only = api_only
    ):
        return frontend
    if frontend is not None:
        # A user-supplied dist is not vetted by _find_frontend_dist, so verify it
        # can serve the login page; else `--frontend /bad/path` bypasses the guard.
        if (Path(frontend) / "index.html").is_file():
            return frontend
        typer.echo(
            "Error: --frontend points at a directory with no index.html, so a "
            "public Unsloth launch would have no login page to change the seeded "
            "admin password. Point --frontend at a built dist, rebuild it (re-run "
            "install.sh), or use --api-only.",
            err = True,
        )
        raise typer.Exit(1)
    # _find_frontend_dist only returns a path that already contains index.html.
    resolved = _find_frontend_dist()
    if resolved is not None:
        return resolved
    typer.echo(
        "Error: the Unsloth frontend is not built, so a public launch would have "
        "no login page to change the seeded admin password. Build it (re-run "
        "install.sh), pass --frontend PATH to a built dist, or use --api-only.",
        err = True,
    )
    raise typer.Exit(1)


def _validate_inproc_backend_before_strip(
    *, cloudflare: Optional[bool], host: str, secure: bool, api_only: bool
) -> None:
    """In-venv (in-process) analogue of the re-exec launcher check.

    In-venv there is no re-exec, so the backend is imported in-process only AFTER
    the gate. On the headless public path the gate strips the seeded
    .bootstrap_password, so a broken venv that fails at import would leave
    must_change_password=1 with no password to log in. Import the backend up front
    on that path and exit cleanly if broken, before anything is stripped.
    Headless-only so an interactive prompt is not delayed behind the import.
    """
    if not _should_prompt_password_change(
        cloudflare = cloudflare, host = host, secure = secure, api_only = api_only
    ):
        return
    if _prompt_streams_interactive():
        return
    try:
        _load_run_module()
    except Exception as exc:
        typer.echo(
            f"Error: the Unsloth backend could not be loaded ({exc}); refusing to "
            "expose Unsloth publicly before it is confirmed runnable. Re-run: "
            "unsloth studio setup",
            err = True,
        )
        raise typer.Exit(1)


def _tunnel_binary_confirmed_unavailable() -> bool:
    """True only if cloudflared is provably unavailable (found nowhere on PATH or
    in the Unsloth cache AND the download failed), so the tunnel cannot start.

    Used on the --secure path (loopback bind, so the tunnel is the ONLY public
    exposure) to skip stripping the seeded recovery password before a public URL
    that will never come up. Loads the stdlib-only cloudflare_tunnel helper by
    file path so the check runs in the parent, before the strip.

    Returns False on ANY uncertainty: a possible credential leak outweighs a
    recoverable lockout, so the caller keeps the strip unless the tunnel is
    provably dead.
    """
    run_py = _find_run_py()
    if run_py is None:
        return False
    backend_dir = run_py.parent
    tunnel_py = backend_dir / "cloudflare_tunnel.py"
    if not tunnel_py.is_file():
        return False
    # ensure_cloudflared() lazily imports utils.paths.storage_roots to resolve the
    # Unsloth bin cache. The outer CLI hasn't added studio/backend to sys.path yet,
    # so that import would fail and return None (a false "unavailable" that wrongly
    # refuses --secure). Add the backend dir so the cache path resolves as in the child.
    added_backend_path = False
    try:
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
            added_backend_path = True
        spec = importlib.util.spec_from_file_location("studio.backend.cloudflare_tunnel", tunnel_py)
        if spec is None or spec.loader is None:
            return False
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.ensure_cloudflared() is None
    except Exception:
        return False
    finally:
        if added_backend_path:
            try:
                sys.path.remove(str(backend_dir))
            except ValueError:
                pass


def _child_self_suppresses(*, in_studio_venv: bool, child_run_py: Optional[Path]) -> bool:
    """True when the child that will serve Unsloth is provably THIS install's
    backend, whose pre-bind gate sets app.state.suppress_bootstrap_injection and
    so never serves the seeded credential publicly -- even with .bootstrap_password
    on disk. The parent-side strip is then unnecessary and can be skipped to avoid
    a lockout if the tunnel never comes up, keeping the file for LOCAL recovery.

    True iff we run in-process here, or the re-exec target is the outer install's
    own run.py (identity match). False on ANY doubt -- a studio-venv console script
    or a venv run.py that may predate the gate -- so the strip stays in force
    wherever an old child is possible.
    """
    if in_studio_venv:
        return True
    if child_run_py is None:
        return False
    try:
        outer_run_py = (_PACKAGE_ROOT / "studio" / "backend" / "run.py").resolve()
        return child_run_py.resolve() == outer_run_py
    except OSError:
        return False


def _enforce_password_change_before_exposure(
    *,
    cloudflare: Optional[bool],
    host: str,
    secure: bool,
    api_only: bool,
    child_self_suppresses: bool = False,
) -> None:
    """Force a terminal password change before the first public (tunnel) exposure.

    When the launch will start the tunnel and the admin still has its
    auto-generated bootstrap password, ask for a new one in the terminal (masked,
    confirmed) before any server or tunnel exists. Committing here, in the parent,
    keeps the password off argv/env and an older studio-venv child sees it
    immediately. Without a terminal, warn and fall back to the bootstrap shutdown
    timer (~1h, UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT).
    """
    if not _should_prompt_password_change(
        cloudflare = cloudflare, host = host, secure = secure, api_only = api_only
    ):
        return
    # Before public exposure we must PROVE the admin password is no longer the
    # seeded default. If we cannot (auth DB won't open, or a fresh admin cannot be
    # seeded + committed below), an old studio-venv child could regenerate a fresh
    # bootstrap credential and serve it; stripping a file we can't vouch for cannot
    # stop a regeneration. So those cases fail closed, as does a failure after the
    # user typed a new password.
    try:
        conn = _connect_auth_db()
    except (OSError, sqlite3.Error) as exc:
        # Cannot open the auth DB, so cannot confirm a committed admin exists.
        # Refuse rather than risk a child serving the default login; a transient
        # lock clears on retry.
        typer.echo(
            "Error: refusing to publish Unsloth on a public Cloudflare URL: could "
            f"not open the Unsloth auth database ({exc}) to confirm the admin "
            "password was changed. Retry (a transient database lock clears), or "
            "change the password first (run `unsloth studio` locally with a "
            "terminal attached, or `unsloth studio reset-password`).",
            err = True,
        )
        raise typer.Exit(1)
    try:
        try:
            _ensure_cli_default_admin(conn)
            # Persist a freshly seeded admin before we might re-exec: the INSERT is
            # otherwise uncommitted and rolls back on conn.close(). If the seed or
            # commit fails, no admin is committed, so a re-exec'd OLD child finds
            # none, regenerates a fresh bootstrap password + file, and serves THAT
            # -- stripping cannot stop a regeneration. Can't prove a committed
            # admin, so fail closed.
            conn.commit()
        except (OSError, sqlite3.Error) as exc:
            # Best-effort remove any half-written seed file (its row rolled back);
            # the launch is refused regardless.
            try:
                (STUDIO_HOME / "auth" / BOOTSTRAP_PASSWORD_FILE).unlink(missing_ok = True)
            except OSError:
                pass
            typer.echo(
                "Error: refusing to publish Unsloth on a public Cloudflare URL: could "
                f"not initialize the admin account ({exc}), so a re-exec'd Unsloth "
                "child could regenerate and serve a default credential. Retry (a "
                "transient database lock clears), or change the password first (run "
                "`unsloth studio` locally with a terminal attached, or `unsloth "
                "studio reset-password`).",
                err = True,
            )
            raise typer.Exit(1)
        try:
            row = conn.execute(
                "SELECT password_salt, password_hash, must_change_password "
                "FROM auth_user WHERE username = ?",
                (DEFAULT_ADMIN_USERNAME,),
            ).fetchone()
        except (OSError, sqlite3.Error) as exc:
            if child_self_suppresses:
                # Could not read must_change back, but the child is this install's
                # own backend and suppresses the injection, so nothing serves the
                # seeded credential; proceed without stripping.
                return
            # The admin is committed above, so an old child finds it and won't
            # regenerate; we just couldn't read must_change back. Strip the seeded
            # file so nothing serves it, failing closed if the strip itself fails.
            typer.echo(
                f"Warning: could not read the Unsloth admin state back ({exc}); "
                "removing the seeded bootstrap password before public exposure.",
                err = True,
            )
            _strip_seeded_bootstrap_password_or_exit(context = "auth DB row unreadable")
            return
        if row and _credential_undelivered(row[1]):
            # An earlier launch committed an auto-generated password and could not
            # print it, so it refused. must_change_password is 0 now, so the check
            # below would return and let the public child start under a credential
            # nobody has. Keep failing closed until the password is reset.
            typer.echo(
                "Error: refusing to publish Unsloth on a public Cloudflare URL: the "
                "admin password auto-generated by an earlier launch was committed but "
                "never displayed, so no one can log in. Reset it with `unsloth studio "
                "reset-password`, then relaunch.",
                err = True,
            )
            raise typer.Exit(1)
        if not row or not row[2]:
            return
        if not _prompt_streams_interactive():
            # No terminal for the interactive change and no --password supplied
            # (that path clears must_change and returns above).
            #
            # On --secure the loopback bind means the tunnel is the ONLY public
            # exposure: if cloudflared is provably unavailable no public URL can
            # ever come up, so refuse rather than rotate the recovery credential
            # for a launch that will not start. (A wildcard --cloudflare bind is
            # public regardless of the tunnel, so it still proceeds below.)
            if secure and _tunnel_binary_confirmed_unavailable():
                typer.echo(
                    "Error: refusing to expose Unsloth: the Cloudflare tunnel binary "
                    "(cloudflared) is unavailable and could not be downloaded, so no "
                    "public URL can start. The seeded bootstrap password is preserved "
                    "for recovery; fix connectivity and retry, or change the password "
                    "first (`unsloth studio` locally, or `unsloth studio "
                    "reset-password`).",
                    err = True,
                )
                raise typer.Exit(1)
            # Auto-generate a strong admin password and commit it via the normal
            # update path (clears must_change_password so the public child launches
            # cleanly, rotates the JWT secret, revokes refresh tokens, and deletes
            # the seeded bootstrap file). Mirrors run.py's headless gate. A child of
            # ANY version then reads must_change=0 with no seeded file, so it serves
            # a normal login page requiring the new password and never injects a
            # default credential. Surface it once here, before any re-exec, so the
            # secret never crosses to the child argv. child_self_suppresses is no
            # longer consulted here: rotating a real password protects every child.
            # Resolve the console stream that will surface the one-time credential
            # BEFORE rotating the seeded recovery password. On a non-interactive
            # launch stderr/stdout can be absent (a Windows pythonw/service wrapper)
            # or closed, in which case typer.echo(err=True) silently no-ops and the
            # only new credential is lost after the bootstrap password was deleted,
            # locking the operator out. Mirror run.py's _one_time_secret_stream
            # fail-closed preflight: with no usable console, refuse WITHOUT rotating
            # so the seeded bootstrap password stays intact for local recovery.
            out = _one_time_secret_console_stream()
            if out is None:
                typer.echo(
                    "Error: refusing to rotate the Unsloth admin password: no usable "
                    "console (stderr/stdout) to show the auto-generated credential, so "
                    "it would be lost. The seeded bootstrap password is preserved; "
                    "change the password first (`unsloth studio` locally with a "
                    "terminal attached, or `unsloth studio reset-password`).",
                    err = True,
                )
                raise typer.Exit(1)
            generated = secrets.token_urlsafe(24)
            if not _cli_update_password(
                conn, DEFAULT_ADMIN_USERNAME, generated, require_must_change = True
            ):
                # Lost the compare-and-set: a password was set (another Studio tab
                # finishing /change-password, a concurrent launch) between the
                # must_change read above and this write, so ours was never stored.
                # The account is off the seeded default, so launch with theirs and
                # never show a credential that would not authenticate.
                return
            # Delivery is post-commit: the seeded recovery password is already gone,
            # so a console that died since the preflight must not propagate its
            # write error. Retry the other console, and fail closed with a
            # secret-free message when neither accepts the banner.
            # Mark BEFORE the banner: between the commit above and a confirmed
            # write, this password lives only in memory, and the seeded recovery
            # credential is already gone. Read the committed hash back rather than
            # recomputing it, so the sentinel matches whatever actually landed.
            try:
                committed_row = conn.execute(
                    "SELECT password_hash FROM auth_user WHERE username = ?",
                    (DEFAULT_ADMIN_USERNAME,),
                ).fetchone()
                if committed_row:
                    _mark_credential_undelivered(committed_row[0])
            except (OSError, sqlite3.Error):
                pass
            if not _deliver_auto_generated_credentials(DEFAULT_ADMIN_USERNAME, generated, out = out):
                _log_secret_free_delivery_failure()
                raise typer.Exit(1)
            _clear_credential_undelivered()
            return
        password_salt, password_hash = row[0], row[1]

        def _is_current_password(candidate: str) -> bool:
            return hmac.compare_digest(
                _pbkdf2_hex(candidate, password_salt.encode("utf-8")), password_hash
            )

        typer.echo(
            "Unsloth Studio will be exposed on the public internet, so set a "
            "password now. Ctrl+C to abort.",
            err = True,
        )
        try:
            new_password = _password_prompt.prompt_new_password(_is_current_password)
        except (KeyboardInterrupt, EOFError):
            typer.echo(
                "\nError: password change aborted; refusing to expose Unsloth "
                "with the default admin password. Re-run and set a password, "
                "or launch without --secure/--cloudflare.",
                err = True,
            )
            raise typer.Exit(1)
        _cli_update_password(conn, DEFAULT_ADMIN_USERNAME, new_password)
        typer.echo(f"Password updated for '{DEFAULT_ADMIN_USERNAME}'.", err = True)
    finally:
        conn.close()


def _load_model_via_http(
    port: int,
    api_key: str,
    model: str,
    gguf_variant: Optional[str],
    max_seq_length: int,
    load_in_4bit: bool,
    gpu_memory_mode: Literal["auto", "manual"] = "auto",
    tensor_parallel: bool = False,
    speculative_type: Optional[SpeculativeType] = None,
    spec_draft_n_max: Optional[int] = None,
    llama_extra_args: Optional[List[str]] = None,
    timeout: int = 600,
) -> dict:
    """POST to ``/api/inference/load`` using the API key for auth."""
    import json
    import urllib.request
    import urllib.error

    from unsloth_cli._inference import raise_for_deferred_error, require_completed_padded_body

    payload: dict = {
        "model_path": model,
        "max_seq_length": max_seq_length,
        "load_in_4bit": load_in_4bit,
    }
    if gguf_variant:
        payload["gguf_variant"] = gguf_variant
    if gpu_memory_mode == "manual":
        payload["gpu_memory_mode"] = "manual"
        payload["gpu_layers"] = -1
    if tensor_parallel:
        payload["tensor_parallel"] = True
    if speculative_type is not None:
        payload["speculative_type"] = speculative_type
    if spec_draft_n_max is not None:
        payload["spec_draft_n_max"] = spec_draft_n_max
    if llama_extra_args:
        payload["llama_extra_args"] = list(llama_extra_args)

    data = json.dumps(payload).encode()
    url = f"http://127.0.0.1:{port}/api/inference/load"
    req = urllib.request.Request(
        url,
        data = data,
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout = timeout) as resp:
            try:
                body = json.loads(resp.read())
            except ValueError:
                body = None  # truncated padded reply; rejected below
        # A slow load commits its 200 before it finishes and pads the body, so a late
        # failure arrives in-band; raise it as the HTTPError this function already turns
        # into the RuntimeError the caller reports. A truncated body is no report at all.
        return require_completed_padded_body(url, raise_for_deferred_error(url, body))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors = "replace")
        raise RuntimeError(f"Model load failed (HTTP {exc.code}): {body}") from exc


def _format_context_length_line(load_result: dict) -> Optional[str]:
    value = load_result.get("context_length")
    if isinstance(value, bool):
        return None
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return None
    if value_int <= 0:
        return None
    return f"  Context length: {value_int} tokens"


# ── unsloth studio (server) ──────────────────────────────────────────


@studio_app.callback(invoke_without_command = True)
def studio_default(
    ctx: typer.Context,
    port: int = typer.Option(8888, "--port", "-p"),
    host: str = typer.Option("127.0.0.1", "--host", "-H"),
    frontend: Optional[Path] = typer.Option(None, "--frontend", "-f"),
    silent: bool = typer.Option(False, "--silent", "-q"),
    api_only: bool = typer.Option(
        False,
        "--api-only",
        help = "Run API server only, no frontend serving (for Tauri desktop app)",
    ),
    parallel: int = typer.Option(
        _PARALLEL_DEFAULT_PLAIN,
        "--parallel",
        "--n-parallel",
        min = _PARALLEL_MIN,
        max = _PARALLEL_MAX,
        help = (
            f"llama-server parallel decode slots ({_PARALLEL_MIN}..{_PARALLEL_MAX}). "
            f"Default {_PARALLEL_DEFAULT_PLAIN}. The Unsloth run settings "
            "(Parallel Slots) override it per load."
        ),
    ),
    cloudflare: Optional[bool] = typer.Option(
        None,
        "--cloudflare/--no-cloudflare",
        help = "Expose Unsloth on a PUBLIC internet URL via a free Cloudflare HTTPS "
        "tunnel, for non-api-only wildcard binds (0.0.0.0 or ::). Off by default; "
        "pass --cloudflare to enable it (--secure implies it). --no-cloudflare forces "
        "it off but does not change a raw wildcard bind.",
    ),
    secure: bool = typer.Option(
        False,
        "--secure/--no-secure",
        help = "Expose ONLY a Cloudflare HTTPS link: bind localhost and fail closed "
        "if the tunnel can't start. Without it, --no-secure also serves the raw "
        "0.0.0.0 port, which is reachable from anywhere on the network.",
    ),
    not_secure: bool = typer.Option(
        False,
        "--not-secure",
        hidden = True,
        help = "Deprecated alias for --no-secure.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help = "Log every API request, including the high-frequency polling that is "
        "deduplicated by default.",
    ),
    enable_tools: Optional[bool] = typer.Option(
        None,
        "--enable-tools/--disable-tools",
        help = "Force server-side tools (web search, code execution) on or off for "
        "every request. Default: no server-wide policy, so the per-chat UI toggle "
        "(the request's own enable_tools) decides; `unsloth studio run` is the "
        "launcher that defaults them on. /v1/messages takes the on direction per "
        "request (enable_tools) because it has no confirmation channel; the off "
        "direction still applies everywhere.",
    ),
    disable_dns_pinning: bool = typer.Option(
        False,
        "--disable-dns-pinning",
        help = "Send the hostname (not the validated IP) in web fetches that go through an "
        "explicitly configured HTTP(S)_PROXY, so the proxy can apply hostname policy and "
        "TLS interception. Direct fetches stay pinned to the validated IP.",
    ),
    password: str = typer.Option(
        "",
        "--password",
        help = "Set the INITIAL admin password non-interactively (headless setups), "
        "only when none is set yet. Also reads the UNSLOTH_STUDIO_PASSWORD env var, or "
        "`--password -` to read one line from stdin. A literal value is visible in the "
        "process list and shell history. Rotate later with `unsloth studio reset-password`.",
    ),
):
    """Launch the Unsloth Studio server."""
    # Back-compat: --not-secure is a deprecated alias for --no-secure.
    secure = _resolve_secure(secure, not_secure)
    # Runs before every subcommand (run/setup/update/...).
    _ensure_studio_env_exported()
    if ctx.invoked_subcommand is not None:
        # Typer doesn't forward parent options to subcommands, so
        # `unsloth studio --parallel N run ...` would silently drop N.
        if parallel != _PARALLEL_DEFAULT_PLAIN:
            typer.echo(
                f"Error: --parallel on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio "
                f"{ctx.invoked_subcommand}`, put the flag after the "
                f"subcommand: `unsloth studio {ctx.invoked_subcommand} "
                f"--parallel {parallel} ...`",
                err = True,
            )
            raise typer.Exit(2)
        # Same for --cloudflare/--no-cloudflare: it would not reach the subcommand.
        if cloudflare is not None:
            _cf_flag = "--cloudflare" if cloudflare else "--no-cloudflare"
            typer.echo(
                f"Error: {_cf_flag} on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio "
                f"{ctx.invoked_subcommand}`, put it after the subcommand: "
                f"`unsloth studio {ctx.invoked_subcommand} {_cf_flag} ...`",
                err = True,
            )
            raise typer.Exit(2)
        # Same for --secure: it would not reach the subcommand.
        if secure:
            typer.echo(
                f"Error: --secure on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio "
                f"{ctx.invoked_subcommand}`, put it after the subcommand: "
                f"`unsloth studio {ctx.invoked_subcommand} --secure ...`",
                err = True,
            )
            raise typer.Exit(2)
        # Same for --verbose: it would not reach the subcommand.
        if verbose:
            typer.echo(
                f"Error: --verbose on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio "
                f"{ctx.invoked_subcommand}`, put it after the subcommand: "
                f"`unsloth studio {ctx.invoked_subcommand} --verbose ...`",
                err = True,
            )
            raise typer.Exit(2)
        # Same for --enable-tools/--disable-tools: it would not reach the subcommand.
        if enable_tools is not None:
            _tool_flag = "--enable-tools" if enable_tools else "--disable-tools"
            typer.echo(
                f"Error: {_tool_flag} on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio "
                f"{ctx.invoked_subcommand}`, put it after the subcommand: "
                f"`unsloth studio {ctx.invoked_subcommand} {_tool_flag} ...`",
                err = True,
            )
            raise typer.Exit(2)
        if disable_dns_pinning:
            typer.echo(
                "Error: --disable-dns-pinning on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio {ctx.invoked_subcommand}`, "
                f"put it after the subcommand: `unsloth studio {ctx.invoked_subcommand} "
                "--disable-dns-pinning ...`",
                err = True,
            )
            raise typer.Exit(2)
        # Same for --api-only: dropping it here would silently serve the UI.
        if api_only:
            typer.echo(
                f"Error: --api-only on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio "
                f"{ctx.invoked_subcommand}`, put it after the subcommand: "
                f"`unsloth studio {ctx.invoked_subcommand} --api-only ...`",
                err = True,
            )
            raise typer.Exit(2)
        # Same for --password: it applies to the plain-server path only.
        if password:
            typer.echo(
                f"Error: --password on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio "
                f"{ctx.invoked_subcommand}`, put it after the subcommand: "
                f"`unsloth studio {ctx.invoked_subcommand} --password ...`",
                err = True,
            )
            raise typer.Exit(2)
        return

    runtime_gate_handoff = _studio_runtime_gate.consume_runtime_gate_handoff()
    _preserve_cloudflare_intent(cloudflare, secure)

    # --secure requires the tunnel; force a loopback bind.
    if secure:
        if cloudflare is False:
            typer.echo(
                "Error: --secure requires the Cloudflare tunnel; do not combine it "
                "with --no-cloudflare.",
                err = True,
            )
            raise typer.Exit(2)
        if host not in ("127.0.0.1", "localhost", "::1"):
            typer.echo(
                "Note: --secure ignores -H (it binds loopback and serves only "
                "through the Cloudflare tunnel). Drop --secure to bind "
                f"{host} directly, or keep --secure for a tunnel-only public link.",
                err = True,
            )
        host = "127.0.0.1"

    # --verbose restores the per-request access logs that are suppressed by
    # default (plain-server path; the `run` subcommand has its own --verbose).
    if verbose:
        _enable_verbose_access_logs()
    if disable_dns_pinning:
        os.environ["UNSLOTH_STUDIO_DISABLE_DNS_PINNING"] = "1"
    else:
        os.environ.setdefault("UNSLOTH_STUDIO_DISABLE_DNS_PINNING", "0")

    # Use the studio venv if present and not already in it. Resolve the child
    # launcher BEFORE the gate: a headless gate strips the seeded
    # .bootstrap_password, so aborting afterward (venv/run.py missing) would leave
    # must_change_password=1 with no password to log in.
    studio_venv_dir = STUDIO_HOME / "unsloth_studio"
    in_studio_venv = sys.prefix.startswith(str(studio_venv_dir))
    # Before any of the three launch paths below, and before the environment is handed
    # to a child: an override contradicting single-arch wheels makes every kernel launch
    # fail, and the installer's own unset cannot reach a launch it does not perform (#7331).
    _clear_hsa_override_before_launch(silent = silent)
    studio_python = run_py = None
    resolved_frontend = frontend
    if not in_studio_venv:
        studio_python = _studio_venv_python()
        run_py = _find_run_py()
        if not (studio_python and run_py):
            typer.echo("Unsloth Studio not set up. Run install.sh first.")
            raise typer.Exit(1)
        # A public UI launch must have a servable login page BEFORE the gate can
        # strip the seeded .bootstrap_password, or the child has no way to change
        # it. Also returns the resolved dist so the child serves a real build
        # regardless of where its __file__ lands (fixes the shadowed silent 404).
        resolved_frontend = _require_servable_frontend_or_exit(
            frontend = resolved_frontend,
            api_only = api_only,
            cloudflare = cloudflare,
            host = host,
            secure = secure,
        )
        # Non-public / api-only launches skip that validation but still forward an
        # explicitly resolved dist for the same silent-404 reason.
        if resolved_frontend is None and not api_only:
            resolved_frontend = _find_frontend_dist()
    else:
        # Already in the studio venv: no re-exec, served in-process below. On the
        # headless public path the gate strips the seeded .bootstrap_password, so
        # validate BOTH FIRST -- else a bad dist or broken venv fails only after
        # the strip (must_change_password=1, no password to log in). Frontend check
        # first (cheap); the backend import is headless-only so an interactive
        # prompt is not delayed behind it.
        resolved_frontend = _require_servable_frontend_or_exit(
            frontend = resolved_frontend,
            api_only = api_only,
            cloudflare = cloudflare,
            host = host,
            secure = secure,
        )
        _validate_inproc_backend_before_strip(
            cloudflare = cloudflare, host = host, secure = secure, api_only = api_only
        )

    # A supplied --password / UNSLOTH_STUDIO_PASSWORD / stdin sets the initial
    # admin password here in the parent, before the gate and any re-exec, so the
    # secret never reaches the child argv; strip the env var so a re-exec'd child
    # can't re-read it. The interactive gate below then no-ops.
    _apply_supplied_password_before_launch(_password_prompt.resolve_supplied_password(password))
    os.environ.pop(_password_prompt.SUPPLIED_PASSWORD_ENV, None)

    # Public (tunnel) exposure with the seeded default password: force a terminal
    # password change first, before any re-exec or server exists. The child is
    # self-suppressing when we serve in-process or re-exec this install's own
    # run.py (its pre-bind gate suppresses the injection), so the gate can skip
    # the destructive strip.
    _enforce_password_change_before_exposure(
        cloudflare = cloudflare,
        host = host,
        secure = secure,
        api_only = api_only,
        child_self_suppresses = _child_self_suppresses(
            in_studio_venv = in_studio_venv, child_run_py = run_py
        ),
    )

    if not in_studio_venv:
        if studio_python and run_py:
            if not silent:
                typer.echo("Launching Unsloth Studio... Please wait...")
            args = [
                str(studio_python),
                str(run_py),
                "--host",
                host,
                "--port",
                str(port),
                "--parallel",
                str(parallel),
            ]
            # Forward the frontend dist resolved before the gate (skipped in
            # --api-only, which serves no UI).
            if resolved_frontend is not None:
                args.extend(["--frontend", str(resolved_frontend)])
            if silent:
                args.append("--silent")
            if api_only:
                args.append("--api-only")
            # Forward polarity explicitly: _find_run_py can fall back to an older
            # run.py (--cloudflare defaulted on), so an unset default must not let a
            # mixed install silently re-enable the tunnel. --secure implies it, so
            # forward nothing then.
            if cloudflare is True:
                args.append("--cloudflare")
            elif not secure:
                args.append("--no-cloudflare")
            args.append("--secure" if secure else "--no-secure")
            # Forward an explicit tool policy; None -> run.py leaves it unset (tools on).
            if enable_tools is True:
                args.append("--enable-tools")
            elif enable_tools is False:
                args.append("--disable-tools")
            # On Windows os.execvp keeps the parent alive, so Ctrl+C
            # would orphan the child; use Popen+wait instead.
            if sys.platform == "win32":
                import subprocess as _sp

                # Hand our std handles to the child: without them CREATE_NO_WINDOW
                # gives the backend its own hidden console and `unsloth studio > log`
                # captures nothing -- the same trap noted at the setup.ps1 call below.
                # Omitting stdin does not withhold it (subprocess still fills it from
                # GetStdHandle); that would need stdin = DEVNULL.
                with _studio_runtime_launch_guard(inherited = runtime_gate_handoff):
                    proc = _sp.Popen(
                        args,
                        stdout = _stream_for_subprocess(sys.stdout),
                        stderr = _stream_for_subprocess(sys.stderr),
                        **_windows_hidden_subprocess_kwargs(),
                    )
                try:
                    rc = proc.wait()
                except KeyboardInterrupt:
                    # Child handles its own signal; let it finish.
                    rc = proc.wait()
                if rc != 0:
                    typer.echo(
                        f"\nError: Unsloth server exited unexpectedly (code {rc}).",
                        err = True,
                    )
                    typer.echo(
                        "Check the error above. If a package is missing, "
                        "re-run: unsloth studio setup",
                        err = True,
                    )
                raise typer.Exit(rc)
            else:
                os.execvp(str(studio_python), args)
        else:
            typer.echo("Unsloth Studio not set up. Run install.sh first.")
            raise typer.Exit(1)

    with _studio_deps.studio_backend_imports("unsloth studio"):
        run_mod = _load_run_module()
    run_server = run_mod.run_server

    if not silent:
        display_host = _display_host_for_bind(run_mod, host)
        typer.echo(f"Starting Unsloth Studio on http://{_url_host(display_host)}:{port}")

    run_kwargs = dict(
        host = host,
        port = port,
        silent = silent,
        api_only = api_only,
        llama_parallel_slots = parallel,
        cloudflare = cloudflare,
        secure = secure,
        enable_tools = enable_tools,
    )
    # Forward the frontend validated before the gate (in-venv path), so the
    # in-process server serves exactly the dist we vouched for.
    if resolved_frontend is not None:
        run_kwargs["frontend_path"] = resolved_frontend
    with _studio_runtime_launch_guard(inherited = runtime_gate_handoff):
        run_server(**run_kwargs)

    try:
        if run_mod._shutdown_event is not None:
            # Event.wait() with no timeout blocks at C-level on Linux
            # and swallows SIGINT; loop with a 1s timeout instead.
            while not run_mod._shutdown_event.is_set():
                run_mod._shutdown_event.wait(timeout = 1)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        run_mod._graceful_shutdown(run_mod._server)
        typer.echo("\nShutting down...")
    finally:
        getattr(run_mod, "_wait_for_server_shutdown", lambda: None)()


# ── unsloth studio run ───────────────────────────────────────────────


def _split_repo_variant(model_arg: str) -> tuple[str, Optional[str]]:
    """Split ``org/name:variant`` into ``(repo, variant)``; mirrors
    llama.cpp's ``-hf <repo>:<quant>``. Local paths, Windows drives,
    and ids without ``:`` pass through verbatim."""
    s = model_arg.strip()
    if not s:
        return s, None
    if s.startswith(("/", "./", "../", "~")) or s == ".":
        return s, None
    # Windows drive letter (e.g. "C:\path"): colon is a path separator.
    if len(s) >= 2 and s[1] == ":" and s[0].isalpha():
        return s, None
    if ":" not in s:
        return s, None
    repo, _, variant = s.rpartition(":")
    if not repo or not variant:
        return s, None
    # Quant labels never contain a slash; `foo:bar/baz` isn't repo:variant.
    if "/" in variant:
        return s, None
    return repo, variant


def _expand_attached_np_short() -> None:
    # Click clusters `-np8` as `-n -p 8` (-p = --port), dropping the parallel
    # value. Split to `-np <N>` so typer's alias matches. Stops at `--`;
    # accepts signed/junk forms so typer reports a clean error against `-np`.
    # Kept in lockstep with the backend `_flag_name` recogniser.
    i = 0
    while i < len(sys.argv):
        tok = sys.argv[i]
        if tok == "--":
            break
        if len(tok) > 3 and tok.startswith("-np") and tok[3] != "=":
            suffix = tok[3:]
            first_numeric = suffix[0].isdigit() or (
                len(suffix) > 1 and suffix[0] in {"-", "+"} and suffix[1].isdigit()
            )
            if first_numeric:
                sys.argv[i : i + 1] = ["-np", suffix]
                i += 2
                continue
        i += 1


def _consume_legacy_short_aliases(
    args: List[str], aliases: tuple[str, ...], current: Optional[str], canonical: str
) -> tuple[Optional[str], List[str]]:
    """Pop exact-match legacy shorts (`-m`/`-hfr`/`-f`) from args;
    leave clusters (`-mg`/`-fa`/...) for the llama-server tail. Inline
    `-x=value` form also accepted."""
    out: List[str] = []
    value = current
    i, n = 0, len(args)
    while i < n:
        tok = args[i]
        if tok == "--":  # end of options; tail is raw payload.
            out.extend(args[i:])
            break
        name, sep, inline = tok.partition("=")
        if name not in aliases:
            out.append(tok)
            i += 1
            continue
        if value is not None:
            raise typer.BadParameter(f"{name} conflicts with {canonical} already provided")
        if sep:
            if inline == "":  # `-m=` would become --model '' (Path('')='.').
                raise typer.BadParameter(f"{name} requires a non-empty value")
            value = inline
            i += 1
        elif i + 1 < n:
            nxt = args[i + 1]
            # `--long` is unambiguously a flag; single-dash `-x` may be a path.
            if nxt.startswith("--") and nxt != "--":
                raise typer.BadParameter(f"{name} expects a value but got the flag {nxt}")
            value = nxt
            i += 2
        else:
            raise typer.BadParameter(f"{name} requires a value")
    return value, out


# Help panels so `unsloth run --help` groups options instead of one long list.
_RUN_PANEL_MODEL = "Model"
_RUN_PANEL_SERVER = "Server & network"
_RUN_PANEL_TOOLS = "Tool calls"
_RUN_PANEL_SAMPLING = "Sampling"
_RUN_PANEL_ADVANCED = "Advanced"


@studio_app.command(
    context_settings = {
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def run(
    ctx: typer.Context,
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-hf",
        "--hf-repo",
        # `-m` / `-hfr` removed (Click would cluster `-mg`/`-md`/...).
        # Exact-match `-m`/`-hfr` still work via the legacy shim below.
        # `-hf` stays (multi-char shorts don't cluster).
        rich_help_panel = _RUN_PANEL_MODEL,
        help = (
            "Model path or HF repo. Accepts llama.cpp-style "
            "`org/repo:variant` syntax. `-hf` / `--hf-repo` match "
            "llama-server's spelling."
        ),
    ),
    gguf_variant: Optional[str] = typer.Option(
        None,
        "--gguf-variant",
        rich_help_panel = _RUN_PANEL_MODEL,
        help = "GGUF quant variant (e.g. UD-Q4_K_XL)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        rich_help_panel = _RUN_PANEL_ADVANCED,
        help = "Log every API request, including the high-frequency polling that is "
        "deduplicated by default.",
    ),
    max_seq_length: int = typer.Option(
        0,
        "--max-seq-length",
        "--context-length",
        rich_help_panel = _RUN_PANEL_MODEL,
        help = "Runtime context length in tokens (0 = model default for GGUF; 2048 for hub models)",
    ),
    gpu_memory_mode: Literal["auto", "manual"] = typer.Option(
        "auto",
        "--gpu-memory-mode",
        rich_help_panel = _RUN_PANEL_MODEL,
        help = (
            "GPU memory strategy for GGUF models. Auto lets Unsloth select GPUs "
            "and cap context to fit VRAM. Manual with default layers and context "
            "delegates placement and sizing to llama.cpp --fit."
        ),
    ),
    speculative_type: Optional[SpeculativeType] = typer.Option(
        None,
        "--speculative-type",
        rich_help_panel = _RUN_PANEL_MODEL,
        help = (
            "Speculative decoding mode for GGUF models. DSpark automatically uses a "
            "matching dspark-*.gguf sidecar when available. Default: unset (Unsloth auto)."
        ),
    ),
    spec_draft_n_max: Optional[int] = typer.Option(
        None,
        "--spec-draft-n-max",
        min = 1,
        max = 16,
        rich_help_panel = _RUN_PANEL_MODEL,
        help = "Maximum draft tokens per step for MTP or DSpark (1..16).",
    ),
    load_in_4bit: bool = typer.Option(
        True, "--load-in-4bit/--no-load-in-4bit", rich_help_panel = _RUN_PANEL_MODEL
    ),
    api_key_name: str = typer.Option(
        "cli",
        "--api-key-name",
        rich_help_panel = _RUN_PANEL_ADVANCED,
        help = "Label for the auto-generated API key",
    ),
    port: int = typer.Option(8888, "--port", "-p", rich_help_panel = _RUN_PANEL_SERVER),
    host: str = typer.Option("127.0.0.1", "--host", "-H", rich_help_panel = _RUN_PANEL_SERVER),
    # `-f` removed (clustered `-fa`/`-fit*`); studio_default keeps it.
    frontend: Optional[Path] = typer.Option(None, "--frontend", rich_help_panel = _RUN_PANEL_SERVER),
    api_only: bool = typer.Option(
        False,
        "--api-only",
        rich_help_panel = _RUN_PANEL_SERVER,
        help = "Serve only the API (no web UI), for a headless model server. "
        "Pairs with --secure to expose the API over the Cloudflare link alone.",
    ),
    silent: bool = typer.Option(False, "--silent", "-q", rich_help_panel = _RUN_PANEL_ADVANCED),
    enable_tools: Optional[bool] = typer.Option(
        None,
        "--enable-tools/--disable-tools",
        rich_help_panel = _RUN_PANEL_TOOLS,
        help = (
            "Force server-side tools (web search, code execution) on or off for "
            "every request. Default: on for every bind, with a request's own "
            "enable_tools: false (what the Unsloth UI sends) honored. /v1/messages "
            "takes the on direction per request (enable_tools) because it has no "
            "confirmation channel; the off direction still applies everywhere."
        ),
    ),
    disable_dns_pinning: bool = typer.Option(
        False,
        "--disable-dns-pinning",
        rich_help_panel = _RUN_PANEL_TOOLS,
        help = "Send the hostname (not the validated IP) in web fetches that go through an "
        "explicitly configured HTTP(S)_PROXY, so the proxy can apply hostname policy and "
        "TLS interception. Direct fetches stay pinned to the validated IP.",
    ),
    tool_call_healing: Optional[bool] = typer.Option(
        None,
        "--enable-tool-call-healing/--disable-tool-call-healing",
        rich_help_panel = _RUN_PANEL_TOOLS,
        help = (
            "Promote text-form tool calls (small GGUFs often emit <tool_call>...) "
            "back into structured calls on the client-tool passthrough. Default: on. "
            "An explicit --disable-tool-call-healing is an absolute server kill-switch."
        ),
    ),
    tool_call_nudging: Optional[bool] = typer.Option(
        None,
        "--enable-tool-call-nudging/--disable-tool-call-nudging",
        rich_help_panel = _RUN_PANEL_TOOLS,
        help = (
            "On the non-streaming client-tool passthrough, retry once with a short "
            "nudge when the model emitted a tool signal that healing could not repair. "
            "Default: on. No effect on streaming requests or the server-side agentic loop."
        ),
    ),
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        min = 0.0,
        max = 2.0,
        rich_help_panel = _RUN_PANEL_SAMPLING,
        help = (
            "Pin the sampling temperature for every request that omits it, overriding the "
            "model's recommended value. Default: unset (use the per-model recommendation)."
        ),
    ),
    top_p: Optional[float] = typer.Option(
        None,
        "--top-p",
        min = 0.0,
        max = 1.0,
        rich_help_panel = _RUN_PANEL_SAMPLING,
        help = "Pin top-p (nucleus) sampling. Default: unset (per-model recommendation).",
    ),
    top_k: Optional[int] = typer.Option(
        None,
        "--top-k",
        min = -1,
        max = 100,
        rich_help_panel = _RUN_PANEL_SAMPLING,
        help = "Pin top-k sampling. Default: unset (per-model recommendation).",
    ),
    min_p: Optional[float] = typer.Option(
        None,
        "--min-p",
        min = 0.0,
        max = 1.0,
        rich_help_panel = _RUN_PANEL_SAMPLING,
        help = "Pin min-p sampling threshold. Default: unset (per-model recommendation).",
    ),
    repetition_penalty: Optional[float] = typer.Option(
        None,
        "--repetition-penalty",
        min = 1.0,
        max = 2.0,
        rich_help_panel = _RUN_PANEL_SAMPLING,
        help = "Pin the repetition penalty. Default: unset (per-model recommendation).",
    ),
    presence_penalty: Optional[float] = typer.Option(
        None,
        "--presence-penalty",
        min = 0.0,
        max = 2.0,
        rich_help_panel = _RUN_PANEL_SAMPLING,
        help = "Pin the presence penalty. Default: unset (per-model recommendation).",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        rich_help_panel = _RUN_PANEL_ADVANCED,
        help = "Accepted for backward compatibility; the tool policy no longer prompts.",
    ),
    parallel: int = typer.Option(
        _PARALLEL_DEFAULT_RUN,
        "--parallel",
        "--n-parallel",
        "-np",
        min = _PARALLEL_MIN,
        max = _PARALLEL_MAX,
        rich_help_panel = _RUN_PANEL_SERVER,
        help = (
            "llama-server parallel decode slots. N requests share one "
            "loaded model; each slot gets ctx/N KV cache. Default "
            f"{_PARALLEL_DEFAULT_RUN} (pre-PR hardcoded value). The Unsloth "
            "run settings (Parallel Slots) can override it per load."
        ),
    ),
    cloudflare: Optional[bool] = typer.Option(
        None,
        "--cloudflare/--no-cloudflare",
        rich_help_panel = _RUN_PANEL_SERVER,
        help = "Expose Unsloth on a PUBLIC internet URL via a free Cloudflare HTTPS "
        "tunnel, for non-api-only wildcard binds (0.0.0.0 or ::). Off by default; "
        "pass --cloudflare to enable it (--secure implies it). --no-cloudflare forces "
        "it off but does not change a raw wildcard bind.",
    ),
    secure: bool = typer.Option(
        False,
        "--secure/--no-secure",
        rich_help_panel = _RUN_PANEL_SERVER,
        help = "Expose ONLY a Cloudflare HTTPS link: bind localhost and fail closed "
        "if the tunnel can't start. Without it, --no-secure also serves the raw "
        "0.0.0.0 port, which is reachable from anywhere on the network.",
    ),
    not_secure: bool = typer.Option(
        False,
        "--not-secure",
        hidden = True,
        help = "Deprecated alias for --no-secure.",
    ),
    tensor_parallel: bool = typer.Option(
        False,
        "--tensor-parallel/--no-tensor-parallel",
        rich_help_panel = _RUN_PANEL_MODEL,
        help = (
            "Split a GGUF across GPUs by tensor (--split-mode tensor) instead of "
            "by layer. Multi-GPU only (no effect on one GPU); dense models gain "
            "decode speed, MoE usually don't."
        ),
    ),
    start_api_key_marker: bool = typer.Option(
        False,
        "--start-api-key-marker",
        hidden = True,
        help = "Emit an early API key marker for the unsloth start parent process.",
    ),
    password: str = typer.Option(
        "",
        "--password",
        rich_help_panel = _RUN_PANEL_ADVANCED,
        help = "Set the INITIAL admin password non-interactively (headless setups), "
        "only when none is set yet. Also reads the UNSLOTH_STUDIO_PASSWORD env var, or "
        "`--password -` to read one line from stdin. A literal value is visible in the "
        "process list and shell history. Rotate later with `unsloth studio reset-password`.",
    ),
):
    """Start Unsloth, load a model, print an API key -- one-liner server.

    Unknown flags pass through to llama-server (GGUF only). Unsloth
    rejects managed flags with HTTP 400: model identity, network
    (--host/--port/--path/--api-prefix/--reuse-port), auth/TLS
    (--api-key/--ssl-*), single-model UI (--ui/--models-*/--webui),
    and parallel slots (use --parallel above). Full denylist in
    studio/backend/core/inference/llama_server_args.py. Other knobs
    (-c, -ngl, --jinja, --flash-attn, -t, ...) pass through and
    last-wins-override Unsloth's auto-set value.

    Example:
        unsloth studio run --model unsloth/Qwen3-1.7B-GGUF --gguf-variant UD-Q4_K_XL
        unsloth studio run --model unsloth/Qwen3-1.7B-GGUF --temperature 0.7 --seed 42 --parallel 8
        unsloth studio run --model some-model --chat-template-file /path/to/tpl.jinja
        unsloth studio run --model unsloth/Qwen3-27B-GGUF --gguf-variant Q8_0 --tensor-parallel
    """
    # A newer outer CLI can re-exec into an older Unsloth venv; pass this signal via
    # env so an older child ignores it instead of treating it as a llama-server arg.
    inherited_start_api_key_marker = _consume_start_api_key_marker_env()
    start_api_key_marker = start_api_key_marker or inherited_start_api_key_marker
    runtime_gate_handoff = _studio_runtime_gate.consume_runtime_gate_handoff()
    # The group callback returns before its own clear once a subcommand is named, and
    # `unsloth run` is bound straight here, so this path has to do it itself or
    # llama-server and the backend start with the contradicting override (#7331).
    _clear_hsa_override_before_launch(silent = bool(silent))

    # Back-compat: --not-secure is a deprecated alias for --no-secure.
    secure = _resolve_secure(secure, not_secure)
    _preserve_cloudflare_intent(cloudflare, secure)
    extra_llama_args: List[str] = list(ctx.args) if ctx.args else []

    # Tool-call healing/nudging are read from the env at backend import. Resolve here
    # (before any re-exec/import) so the in-venv child inherits the decision. When the
    # flag is omitted, respect a value the parent already set (e.g. `unsloth start`
    # forwards its choice via the env) and otherwise apply the default: healing on,
    # nudging on for a CLI-launched server.
    _healing_disabled = (
        os.environ.get("UNSLOTH_DISABLE_TOOL_CALL_HEALING") == "1"
        if tool_call_healing is None
        else not tool_call_healing
    )
    os.environ["UNSLOTH_DISABLE_TOOL_CALL_HEALING"] = "1" if _healing_disabled else "0"
    if tool_call_nudging is not None:
        os.environ["UNSLOTH_TOOL_CALL_NUDGE"] = "1" if tool_call_nudging else "0"
    elif "UNSLOTH_TOOL_CALL_NUDGE" not in os.environ:
        os.environ["UNSLOTH_TOOL_CALL_NUDGE"] = "1"

    # Sampling overrides: the backend resolver reads UNSLOTH_SAMPLING_* to hard-pin a field
    # (winning over both the client and the per-model recommendation). Only write a flag that
    # was set explicitly so an omitted flag inherits any value the parent forwarded (e.g.
    # `unsloth start`) and, when nothing is set, leaves the per-model recommendation in charge.
    for _sampling_env, _sampling_value in (
        ("UNSLOTH_SAMPLING_TEMPERATURE", temperature),
        ("UNSLOTH_SAMPLING_TOP_P", top_p),
        ("UNSLOTH_SAMPLING_TOP_K", top_k),
        ("UNSLOTH_SAMPLING_MIN_P", min_p),
        ("UNSLOTH_SAMPLING_REPETITION_PENALTY", repetition_penalty),
        ("UNSLOTH_SAMPLING_PRESENCE_PENALTY", presence_penalty),
    ):
        if _sampling_value is not None:
            os.environ[_sampling_env] = str(_sampling_value)

    # Set before any re-exec so the in-venv server inherits it via the env.
    # `run --verbose` used to pass through to llama-server (its own -v); keep
    # that by forwarding --log-verbose so we add Unsloth logs without dropping it.
    if verbose:
        _enable_verbose_access_logs()
        if not any(a in ("--verbose", "-v", "--log-verbose") for a in extra_llama_args):
            extra_llama_args.append("--log-verbose")
    if disable_dns_pinning:
        os.environ["UNSLOTH_STUDIO_DISABLE_DNS_PINNING"] = "1"
    else:
        os.environ.setdefault("UNSLOTH_STUDIO_DISABLE_DNS_PINNING", "0")

    # Promote legacy exact `-m`/`-hfr`/`-f` back into typer params;
    # clusters stay in extras.
    model, extra_llama_args = _consume_legacy_short_aliases(
        extra_llama_args,
        ("-m", "-hfr"),
        model,
        "--model",
    )
    legacy_frontend, extra_llama_args = _consume_legacy_short_aliases(
        extra_llama_args,
        ("-f",),
        str(frontend) if frontend is not None else None,
        "--frontend",
    )
    if legacy_frontend is not None and frontend is None:
        frontend = Path(legacy_frontend)

    if model is None:
        typer.echo(
            "Error: Missing option '--model' / '-hf' / '--hf-repo' "
            "(legacy aliases '-m' / '-hfr' are still accepted).",
            err = True,
        )
        raise typer.Exit(2)

    # 0. Parse llama.cpp `repo:variant` in --model; error if also paired
    # with --gguf-variant and they disagree.
    parsed_repo, embedded_variant = _split_repo_variant(model)
    if embedded_variant:
        if gguf_variant and gguf_variant != embedded_variant:
            typer.echo(
                f"Error: --model embeds variant '{embedded_variant}' but "
                f"--gguf-variant '{gguf_variant}' was also provided.",
                err = True,
            )
            raise typer.Exit(1)
        model = parsed_repo
        gguf_variant = gguf_variant or embedded_variant

    # --secure requires the tunnel; force a loopback bind so the raw port is never public.
    if secure:
        if cloudflare is False:
            typer.echo(
                "Error: --secure requires the Cloudflare tunnel; do not combine it "
                "with --no-cloudflare.",
                err = True,
            )
            raise typer.Exit(2)
        if host not in ("127.0.0.1", "localhost", "::1"):
            typer.echo(
                "Note: --secure ignores -H (it binds loopback and serves only "
                "through the Cloudflare tunnel). Drop --secure to bind "
                f"{host} directly, or keep --secure for a tunnel-only public link.",
                err = True,
            )
        host = "127.0.0.1"

    # Tool policy does not depend on the bind: tools default on everywhere
    # (--secure is a loopback tunnel; the operator owns a raw bind). With no flag
    # this stays None, so the default applies without becoming an override and a
    # request's own enable_tools: false is honored. Resolve here so the re-exec'd
    # child inherits the same decision.
    from unsloth_cli._tool_policy import is_external_host, resolve_tool_policy

    enable_tools = resolve_tool_policy(
        host = host,
        flag = enable_tools,
        yes = yes,
        silent = silent,
    )

    # 1. Re-exec into the studio venv (same pattern as studio_default). Resolve
    # the child launcher BEFORE the gate: a headless gate strips the seeded
    # .bootstrap_password, so aborting afterward (venv/entry point missing) would
    # leave must_change_password=1 with no password to log in.
    studio_venv_dir = STUDIO_HOME / "unsloth_studio"
    in_studio_venv = sys.prefix.startswith(str(studio_venv_dir))
    studio_bin = None
    resolved_frontend = frontend
    if not in_studio_venv:
        studio_python = _studio_venv_python()
        if not studio_python:
            typer.echo("Unsloth Studio not set up. Run install.sh first.")
            raise typer.Exit(1)
        # Re-exec via the studio venv's `unsloth` console-script. Windows ships it as
        # unsloth.exe, so the bare name is never a file there and `unsloth run` aborted
        # with "venv missing 'unsloth' entry point" on a perfectly good install.
        #
        # On Windows the file is no longer what gets launched (see the launch_head
        # below) and no longer the only thing that proves a CLI: quarantine deletes
        # the stub and leaves the environment able to run, so the installed package
        # answers for it.
        studio_bin = studio_python.parent / (
            "unsloth.exe" if platform.system() == "Windows" else "unsloth"
        )
        if not studio_bin.is_file() and not _managed_cli_package_present(studio_python):
            typer.echo("Unsloth venv missing 'unsloth' entry point. Re-run: unsloth studio setup")
            raise typer.Exit(1)
        # `run` serves the same Unsloth UI (unless --api-only); a public launch must
        # have a servable login page BEFORE the gate strips the seeded password, or
        # the child has no way to change it. Validate here and forward the resolved
        # dist so a shadowed child that can't self-resolve one still serves it.
        resolved_frontend = _require_servable_frontend_or_exit(
            frontend = frontend,
            api_only = api_only,
            cloudflare = cloudflare,
            host = host,
            secure = secure,
        )
    else:
        # In-venv (in-process) run: validate the servable frontend and importable
        # backend before the headless gate strips the seeded password. Frontend
        # check first (cheap); backend import is headless-only so a prompt isn't
        # delayed.
        resolved_frontend = _require_servable_frontend_or_exit(
            frontend = frontend,
            api_only = api_only,
            cloudflare = cloudflare,
            host = host,
            secure = secure,
        )
        _validate_inproc_backend_before_strip(
            cloudflare = cloudflare, host = host, secure = secure, api_only = api_only
        )

    # A supplied --password / UNSLOTH_STUDIO_PASSWORD / stdin sets the initial
    # admin password here in the parent, before the gate and any re-exec, so the
    # secret never reaches the child argv; strip the env var so a re-exec'd child
    # can't re-read it. The interactive gate below then no-ops.
    _apply_supplied_password_before_launch(_password_prompt.resolve_supplied_password(password))
    os.environ.pop(_password_prompt.SUPPLIED_PASSWORD_ENV, None)

    # Public (tunnel) exposure with the seeded default password: force a terminal
    # password change first, before any re-exec or server exists. The re-exec here
    # runs the studio venv's `unsloth` console script (a possibly-OLD child), so it
    # is NOT provably self-suppressing -- only the in-process case is, and the
    # strip stays in force otherwise.
    _enforce_password_change_before_exposure(
        cloudflare = cloudflare,
        host = host,
        secure = secure,
        api_only = api_only,
        child_self_suppresses = _child_self_suppresses(
            in_studio_venv = in_studio_venv, child_run_py = None
        ),
    )

    if not in_studio_venv:
        # Windows launches the child through the venv interpreter rather than the
        # console script it just validated: Application Control blocks the
        # generated unsloth.exe on some machines, and the signed python.exe beside
        # it is not blocked. POSIX keeps execing the script directly, which is what
        # the os.execvp below needs anyway.
        launch_head = (
            _managed_cli_argv(studio_python) if sys.platform == "win32" else [str(studio_bin)]
        )
        args = [
            *launch_head,
            "studio",
            "run",
            "--model",
            model,
            "--max-seq-length",
            str(max_seq_length),
            "--api-key-name",
            api_key_name,
            "--port",
            str(port),
            "--host",
            host,
        ]
        if gpu_memory_mode != "auto":
            args.extend(["--gpu-memory-mode", gpu_memory_mode])
        if gguf_variant:
            args.extend(["--gguf-variant", gguf_variant])
        if speculative_type is not None:
            args.extend(["--speculative-type", speculative_type])
        if spec_draft_n_max is not None:
            args.extend(["--spec-draft-n-max", str(spec_draft_n_max)])
        # Forward the explicit polarity; a future default flip on one
        # layer must not silently invert behaviour for the other.
        args.append("--load-in-4bit" if load_in_4bit else "--no-load-in-4bit")
        # Forward the frontend resolved before the gate, not just a user-supplied
        # one: the parent may have found a built dist the shadowed child cannot,
        # and stripping without forwarding it would abort the child at frontend
        # setup (lockout).
        if resolved_frontend is not None:
            args.extend(["--frontend", str(resolved_frontend)])
        if api_only:
            args.append("--api-only")
        if silent:
            args.append("--silent")
        # Forward the resolved tool policy so the child doesn't re-resolve. None
        # forwards neither flag: the child then leaves the policy unset too.
        if enable_tools is True:
            args.append("--enable-tools")
        elif enable_tools is False:
            args.append("--disable-tools")
        # Forward --yes only if the user passed it; resolution no longer prompts.
        if yes:
            args.append("--yes")
        # Typer claims --parallel outside ctx.args; without this the
        # child reverts to its default and silently drops the value.
        args.extend(["--parallel", str(parallel)])
        # Always forward explicit polarity: a mixed-version studio venv whose old
        # default was --cloudflare-on must not silently re-enable the tunnel.
        # --secure implies it, so forward nothing then.
        if cloudflare is True:
            args.append("--cloudflare")
        elif not secure:
            args.append("--no-cloudflare")
        args.append("--secure" if secure else "--no-secure")
        args.append("--tensor-parallel" if tensor_parallel else "--no-tensor-parallel")
        if verbose:
            args.append("--verbose")
        # llama-server pass-through extras → child ctx.args → load payload.
        if extra_llama_args:
            args.extend(extra_llama_args)

        if start_api_key_marker:
            os.environ[_START_API_KEY_MARKER_ENV] = "1"
        try:
            if sys.platform == "win32":
                with _studio_runtime_launch_guard(inherited = runtime_gate_handoff) as gate_held:
                    popen_kwargs = {}
                    if gate_held:
                        popen_kwargs["env"] = _studio_runtime_gate.runtime_gate_child_environment()
                    proc = subprocess.Popen(args, **popen_kwargs)
                try:
                    rc = proc.wait()
                except KeyboardInterrupt:
                    rc = proc.wait()
                raise typer.Exit(rc)
            else:
                os.execvp(str(studio_bin), args)
        finally:
            # execvp doesn't return on success; restore env after a Windows wait or a failed launch.
            os.environ.pop(_START_API_KEY_MARKER_ENV, None)

    # ── 2. Start server (always suppress built-in banner) ─────────────
    with _studio_deps.studio_backend_imports("unsloth studio"):
        run_mod = _load_run_module()
    run_server = run_mod.run_server

    # Match the route handlers' import path: run.py adds studio/backend/ to
    # sys.path, so they import as `state.tool_policy`. Set this before
    # run_server() starts uvicorn; once sockets are bound, routes can be hit.
    # run_server() applies the same pair; both calls are idempotent.
    from state.tool_policy import set_tool_policy, set_tool_policy_default

    set_tool_policy_default(True)
    set_tool_policy(enable_tools)

    run_kwargs = dict(
        host = host,
        port = port,
        silent = True,
        api_only = api_only,
        llama_parallel_slots = parallel,
        cloudflare = cloudflare,
        secure = secure,
        # Headless serving prints its own URL/API-key banner; the Tauri-only
        # TAURI_PORT line would corrupt that machine-parseable output.
        emit_tauri_port = False,
        # We read the bound port back below, so a fallback past another Unsloth is
        # safe here and keeps side-by-side model runs working.
        abort_if_own_studio = False,
    )
    # Forward the frontend validated before the gate (in-venv path).
    if resolved_frontend is not None:
        run_kwargs["frontend_path"] = resolved_frontend
    with _studio_runtime_launch_guard(inherited = runtime_gate_handoff):
        app = run_server(**run_kwargs)
    actual_port = getattr(app.state, "server_port", port) or port

    # Steps 3-5 can abort (health timeout, model-load error, or Ctrl+C during the
    # slow load); tear the server and its children (llama-server, cloudflared) down
    # on any abort so they never orphan.
    from studio.backend.run import _graceful_shutdown, _server

    try:
        # 3. Wait for server health.
        if not silent:
            typer.echo("Starting Unsloth Studio...")
        if not _wait_for_server(actual_port):
            typer.echo("Error: server did not become healthy within 30 seconds.", err = True)
            raise typer.Exit(1)

        # 4. Create API key in-process.
        api_key = _create_api_key_inprocess(api_key_name)
        if start_api_key_marker:
            # `unsloth start` reads this key from a private 0600 log to authenticate
            # download-progress polling; the normal `unsloth run` output is unchanged.
            typer.echo(f"UNSLOTH_START_API_KEY: {api_key}")

        # 5. Load model via HTTP.
        if not silent:
            typer.echo(f"Loading model: {model}...")
        try:
            result = _load_model_via_http(
                port = actual_port,
                api_key = api_key,
                model = model,
                gguf_variant = gguf_variant,
                max_seq_length = max_seq_length,
                load_in_4bit = load_in_4bit,
                gpu_memory_mode = gpu_memory_mode,
                tensor_parallel = tensor_parallel,
                speculative_type = speculative_type,
                spec_draft_n_max = spec_draft_n_max,
                llama_extra_args = extra_llama_args,
            )
        except RuntimeError as exc:
            typer.echo(f"Error: {exc}", err = True)
            raise typer.Exit(1)
    except BaseException:
        _graceful_shutdown(_server)
        getattr(run_mod, "_wait_for_server_shutdown", lambda: None)()
        raise

    loaded_model = result.get("model", model)
    display_variant = f" ({gguf_variant})" if gguf_variant else ""
    context_length_line = _format_context_length_line(result)

    # 6. Print banner.
    display_host = _display_host_for_bind(run_mod, host)
    base_url = f"http://{_url_host(display_host)}:{actual_port}"
    sdk_base_url = f"{base_url}/v1"
    # run_server started the tunnel during the silent run above (wildcard or --secure).
    _cf_url = getattr(app.state, "cloudflare_url", None)
    # --secure: examples must use the public tunnel URL, not the loopback address.
    if secure and _cf_url:
        sdk_base_url = f"{_cf_url}/v1"

    # Orange so the tool-policy notice stands out; printed under
    # --silent / --yes too so the policy is never invisible.
    _tool_notice_fg = (217, 119, 87)
    _is_external = is_external_host(host)
    if enable_tools is False:
        _tool_notice = "Server-side tools are DISABLED (--disable-tools)."
    elif secure:
        _tool_notice = (
            "Server-side tools are ENABLED, reachable via the authenticated "
            "Cloudflare HTTPS tunnel. Anyone with the API key can run code on "
            "this machine. Do not share the API key. Pass --disable-tools to turn off."
        )
    elif _is_external:
        _tool_notice = (
            "Server-side tools are ENABLED and this port is network-reachable. "
            "Anyone who can reach it with the API key can run code on this "
            "machine. Do not share the API key. Pass --disable-tools to turn off."
        )
    else:
        _tool_notice = (
            "Server-side tools are ENABLED for loopback. Pass --disable-tools to turn off."
        )

    if not silent:
        typer.echo("")
        typer.echo("=" * 56)
        if secure and _cf_url:
            typer.echo(f"  Unsloth Studio running (secure) at {_cf_url}")
            typer.echo(f"  On this machine only: {base_url}")
        else:
            typer.echo(f"  Unsloth Studio running at {base_url}")
            _emit_run_cloudflare_notice(run_mod, host, display_host, actual_port, secure)
        typer.echo(f"  Model loaded: {loaded_model}{display_variant}")
        if context_length_line:
            typer.echo(context_length_line)
        typer.echo(f"  API Key:      {api_key}")
        typer.echo("")
        typer.echo("  OpenAI / Anthropic SDK base URL:")
        typer.echo(f"    {sdk_base_url}")
        typer.echo("=" * 56)
        typer.secho(_tool_notice, fg = _tool_notice_fg, bold = True)
        typer.echo("")
        typer.echo("OpenAI Chat Completions:")
        typer.echo(f"  curl {sdk_base_url}/chat/completions \\")
        typer.echo(f'    -H "Authorization: Bearer {api_key}" \\')
        typer.echo('    -H "Content-Type: application/json" \\')
        typer.echo(
            """    -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'"""
        )
        typer.echo("")
        typer.echo("Anthropic Messages:")
        typer.echo(f"  curl {sdk_base_url}/messages \\")
        typer.echo(f'    -H "Authorization: Bearer {api_key}" \\')
        typer.echo('    -H "Content-Type: application/json" \\')
        typer.echo(
            """    -d '{"max_tokens": 256, "messages": [{"role": "user", "content": "Hello"}], "stream": true}'"""
        )
        typer.echo("")
        typer.echo("OpenAI Responses:")
        typer.echo(f"  curl {sdk_base_url}/responses \\")
        typer.echo(f'    -H "Authorization: Bearer {api_key}" \\')
        typer.echo('    -H "Content-Type: application/json" \\')
        typer.echo("""    -d '{"input": "Hello", "stream": true}'""")
        typer.echo("")
    else:
        # Silent still prints URL + API key + tool-status policy.
        if secure and _cf_url:
            typer.echo(f"URL:     {_cf_url}")
            typer.echo(f"Local:   {base_url}")
        else:
            typer.echo(f"URL:     {base_url}")
            _emit_run_cloudflare_notice(run_mod, host, display_host, actual_port, secure)
        if context_length_line:
            typer.echo(context_length_line.strip())
        typer.echo(f"API Key: {api_key}")
        typer.secho(_tool_notice, fg = _tool_notice_fg, bold = True)

    # 7. Wait for Ctrl+C.
    try:
        if run_mod._shutdown_event is not None:
            while not run_mod._shutdown_event.is_set():
                run_mod._shutdown_event.wait(timeout = 1)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        run_mod._graceful_shutdown(run_mod._server)
        typer.echo("\nShutting down...")
    finally:
        getattr(run_mod, "_wait_for_server_shutdown", lambda: None)()


# ── unsloth studio stop ───────────────────────────────────────────────

_PID_FILE = STUDIO_HOME / "studio.pid"
PID_FILE_GLOB = "studio-*.pid"


def _pid_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` exists.

    ``os.kill(pid, 0)`` raises OSError (WinError 87) for every pid on Windows,
    so use ``tasklist`` there and the signal-0 probe elsewhere.
    """
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/NH", "/FO", "CSV"],
                capture_output = True,
                text = True,
                timeout = 10,
            ).stdout
        except Exception:
            # Can't determine -- assume alive; taskkill no-ops if already gone.
            return True
        return f'"{int(pid)}"' in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _parse_pid_record(text: str) -> "tuple[int, float | None] | None":
    """Parse ``pid`` / optional ``create_time`` from PID file contents."""
    lines = text.splitlines()
    if not lines or not lines[0].strip().isdigit():
        return None
    try:
        # isdigit() is not enough: "²".isdigit() is True but int() rejects it.
        pid = int(lines[0].strip())
    except ValueError:
        return None
    # kill(0) signals our whole process group; kill(1) is init. Never either.
    if pid < 2:
        return None
    created = None
    if len(lines) > 1:
        try:
            created = float(lines[1].strip())
        except ValueError:
            created = None
    return pid, created


def _read_pid_record(path: Path) -> "tuple[int, float | None] | None":
    """Parse ``pid`` / optional ``create_time`` from a PID file."""
    try:
        text = path.read_text(encoding = "utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return _parse_pid_record(text)


def _unlink_quietly(path: Path) -> None:
    """Drop a record without letting one bad file end the loop.

    An undeletable record must not stop us reaching the other servers -- that is
    the orphan this command exists to prevent.
    """
    try:
        path.unlink(missing_ok = True)
    except OSError as e:
        typer.echo(f"Could not remove PID file {path.name}: {e}", err = True)


def _report_unreadable(paths: "list[Path]") -> None:
    """Say which servers we could not reach, since `stop` is about to exit 1."""
    names = ", ".join(sorted(p.name for p in paths))
    typer.echo(
        f"Could not read {len(paths)} PID file(s): {names}. A server recorded "
        f"there may still be running; re-run with permission to read "
        f"{STUDIO_HOME} to stop it.",
        err = True,
    )


def _pid_file_entries(
    unreadable: "list[Path] | None" = None,
) -> "list[tuple[int, list[float | None], list[Path]]]":
    """(pid, create_times, files) per recorded server, including the legacy studio.pid.

    Paths that could not be read are appended to `unreadable` when given, so the
    caller can tell "nothing is running" apart from "something is running and we
    could not see it".

    Grouped by PID: a server writes both its per-port file and studio.pid, and
    signalling twice would hit the SIG_DFL the first SIGTERM installs, hard-killing
    it mid-shutdown. Every recorded time is kept -- a stale file and a live server
    can share a PID, and the stale one must not veto the live one.
    """
    by_pid: "dict[int, tuple[list[float | None], list[Path]]]" = {}
    try:
        paths = sorted(STUDIO_HOME.glob(PID_FILE_GLOB)) + [_PID_FILE]
    except OSError:
        paths = [_PID_FILE]
    seen = set()
    for path in paths:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            text = path.read_text(encoding = "utf-8")
        except (OSError, UnicodeDecodeError) as e:
            # Unreadable is not the same as invalid. A root-owned record, or one
            # caught mid-write, still belongs to a live server, and deleting it
            # strands that server -- the bug this command exists to fix.
            typer.echo(f"Cannot read PID file {path.name}: {e}", err = True)
            if unreadable is not None:
                unreadable.append(path)
            continue
        record = _parse_pid_record(text)
        if record is None:
            typer.echo(f"Ignoring invalid PID file {path.name}")
            _unlink_quietly(path)
            continue
        pid, created = record
        created_times, files = by_pid.setdefault(pid, ([], []))
        created_times.append(created)
        files.append(path)
    return [(pid, times, files) for pid, (times, files) in by_pid.items()]


def _pid_is_studio_server(pid: int, created_times: "Sequence[float | None]" = ()) -> bool:
    """False only when a recorded start time proves this PID is a different process.

    Any recorded time matching is enough -- a stale record must not veto a live
    server that reused the PID. Records with no time at all (a legacy studio.pid,
    or a server started without psutil) cannot be checked, so they are trusted:
    the old `stop` signalled with no checks at all, and skipping a live server is
    the orphan bug this exists to fix.

    An untimed record sitting *alongside* a timed one carries no information, so
    it must not cancel the timed one either. Every current server writes both a
    timed per-port record and an untimed studio.pid, so letting the untimed half
    win made this check inert exactly where it matters and let `stop` SIGTERM an
    unrelated process that had inherited the PID.
    """
    known = [c for c in created_times if c is not None]
    if not known:
        return True
    try:
        import psutil
        actual = psutil.Process(pid).create_time()
    except Exception:
        return True
    return any(abs(actual - c) < 1.0 for c in known)


def _signal_stop(pid: int) -> "str | None":
    """SIGTERM (or taskkill) the pid. Returns an error string, or None on success."""
    import signal as _signal

    if pid < 2:
        return f"refusing to signal PID {pid}"
    try:
        if sys.platform == "win32":
            # /T also stops llama-server children, which otherwise keep GPU and port.
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check = True)
        else:
            os.kill(pid, _signal.SIGTERM)
    except ProcessLookupError:
        return None
    except Exception as e:
        return str(e)
    return None


@studio_app.command()
def stop():
    """Stop every running Unsloth Studio server for this STUDIO_HOME.

    The port fallback can leave more than one running, so stop them all.
    """
    unreadable: "list[Path]" = []
    entries = _pid_file_entries(unreadable)
    if not entries:
        if unreadable:
            # Reporting success here would be a lie: the records we could not
            # read are kept, and the servers behind them are still serving.
            _report_unreadable(unreadable)
            raise typer.Exit(1)
        typer.echo("No running Unsloth server found (no PID file).")
        raise typer.Exit(0)

    signalled, failed = [], []
    for pid, created_times, paths in entries:
        if not _pid_alive(pid) or not _pid_is_studio_server(pid, created_times):
            for path in paths:
                _unlink_quietly(path)
            continue
        error = _signal_stop(pid)
        if error is not None:
            failed.append((pid, error))
            typer.echo(f"Failed to stop Unsloth server (PID {pid}): {error}", err = True)
            continue
        typer.echo(f"Sent shutdown signal to Unsloth server (PID {pid}).")
        signalled.append((pid, paths))

    if not signalled and not failed:
        if unreadable:
            _report_unreadable(unreadable)
            raise typer.Exit(1)
        typer.echo("No running Unsloth server found (cleaned up stale PID files).")
        raise typer.Exit(0)

    pending = list(signalled)
    for _ in range(10):
        if not pending:
            break
        time.sleep(0.5)
        for entry in list(pending):
            pid, paths = entry
            if not _pid_alive(pid):
                for path in paths:
                    _unlink_quietly(path)
                pending.remove(entry)

    stopped = len(signalled) - len(pending)
    if stopped:
        typer.echo(f"Unsloth server{'s' if stopped > 1 else ''} stopped ({stopped}).")
    for pid, _paths in pending:
        typer.echo(f"Unsloth server (PID {pid}) is shutting down (may take a few seconds).")
    if unreadable:
        _report_unreadable(unreadable)
    if failed or unreadable:
        raise typer.Exit(1)


# ── unsloth studio setup / update ─────────────────────────────────────


def _wait_for_windows_setup_process(process) -> int:
    """Reap setup and its descendants before a runtime-gate owner can unwind."""

    try:
        return process.wait()
    except BaseException:
        if process.poll() is not None:
            raise
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin = subprocess.DEVNULL,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                check = False,
                **_windows_hidden_subprocess_kwargs(),
            )
        except BaseException:
            # taskkill interrupted or unavailable: hold the gate until setup
            # exits naturally rather than exposing a live mutator.
            pass
        while process.poll() is None:
            try:
                process.wait()
            except KeyboardInterrupt:
                continue
        raise


# -NoProfile below drops $PSDefaultParameterValues wholesale, and on a corporate host a profile
# entry such as 'Invoke-WebRequest:Proxy' may be the only route to the VC++ runtime and the uv
# installer setup.ps1 downloads. install.ps1 hands over the keys it kept; a standalone update
# has no installer above it, so the probe below asks a short-lived PowerShell that DOES load
# the profile.

# Markers unlikely in a banner, and fixed so both sides agree.
_PROXY_PROBE_BEGIN = "<<UNSLOTH_PROXY_DEFAULTS>>"
_PROXY_PROBE_END = "<</UNSLOTH_PROXY_DEFAULTS>>"

_PS_PROXY_PROBE = (
    "$ErrorActionPreference = 'SilentlyContinue'; "
    # Windows PowerShell 5.1 writes REDIRECTED output in the console code page while this
    # process decodes UTF-8, so a non-ASCII proxy value came back with replacement characters,
    # still parsed as JSON, and handed setup a proxy that does not work.
    "try { [Console]::OutputEncoding = "
    "New-Object System.Text.UTF8Encoding $false } catch { }; "
    "try { $OutputEncoding = [Console]::OutputEncoding } catch { }; "
    "$PSModuleAutoLoadingPreference = 'All'; "
    # Launched -NoProfile, with the caller's profiles dot-sourced by name instead, because left
    # to itself this process loads the CONSOLEHOST profile -- an unrelated host's for a caller
    # in the VS Code console. $PROFILE is fully populated under -NoProfile (the paths are
    # computed, not loaded), so naming them is exact.
    "$__unslothHostProfileName = $env:_UNSLOTH_PS_HOST_PROFILE; "
    # All-users first, PowerShell's own startup order: a machine-managed proxy lives in
    # AllUsersAllHosts on a domain-joined box, and the user's profile only overrides it by
    # running last.
    "try { $__unslothProfiles = @($PROFILE.AllUsersAllHosts); "
    # ONLY the caller's host profile, and only when the caller could be identified: sourcing
    # every Microsoft.*_profile.ps1 in the directory ran profiles for hosts nobody was using,
    # which can print, clobber $PSDefaultParameterValues, or exit before the record is written.
    "if ($__unslothHostProfileName) { "
    "$__unslothProfiles += (Join-Path (Split-Path -Parent $PROFILE.AllUsersCurrentHost) "
    "$__unslothHostProfileName) }; "
    "$__unslothProfiles += $PROFILE.AllUsersCurrentHost; "
    "$__unslothProfiles += $PROFILE.CurrentUserAllHosts; "
    # ADDED, never substituted: TERM_PROGRAM=vscode is set by every VS Code integrated terminal,
    # not only the PowerShell extension's own host, so a plain pwsh terminal there loads
    # Microsoft.PowerShell_profile.ps1 and substitution missed the proxy it actually has.
    "if ($__unslothHostProfileName) { "
    "$__unslothProfiles += (Join-Path (Split-Path -Parent $PROFILE.CurrentUserCurrentHost) "
    "$__unslothHostProfileName) }; "
    # Current-host profile LAST.
    "$__unslothProfiles += $PROFILE.CurrentUserCurrentHost; "
    "foreach ($__unslothProfile in ($__unslothProfiles | Select-Object -Unique)) { "
    "if ($__unslothProfile -and (Test-Path -LiteralPath $__unslothProfile -PathType Leaf)) { "
    "try { . $__unslothProfile } catch { } } } } catch { }; "
    # Re-pinned, because setting [Console]::OutputEncoding is an ordinary profile customization
    # that overrides the pin above. The parent decodes this stream as UTF-8, so a console left
    # on a legacy code page corrupts the framed record and the proxy URI with it.
    "try { [Console]::OutputEncoding = "
    "New-Object System.Text.UTF8Encoding $false } catch { }; "
    "try { $OutputEncoding = [Console]::OutputEncoding } catch { }; "
    "$out = @{}; "
    "foreach ($k in @($PSDefaultParameterValues.Keys)) { "
    "if ($k -is [string] -and [regex]::IsMatch($k, ':Proxy(Credential|UseDefaultCredentials)?$', "
    "[System.Text.RegularExpressions.RegexOptions]::IgnoreCase)) { "
    "$v = $PSDefaultParameterValues[$k]; "
    "if ($v -is [uri]) { $out[$k] = $v.AbsoluteUri } "
    "elseif ($v -is [string] -or $v -is [bool]) { $out[$k] = $v } "
    # A script block is the supported form for a DYNAMIC default -- e.g.
    # { [uri]$env:CORP_PROXY } -- evaluated per call by Invoke-WebRequest. Evaluate here and
    # serialize the RESULT: executable code must not cross the handoff.
    "elseif ($v -is [scriptblock]) { try { $r = & $v; "
    "if ($r -is [uri]) { $out[$k] = $r.AbsoluteUri } "
    "elseif ($r -is [string] -or $r -is [bool]) { $out[$k] = $r } } catch { } } } }; "
    # $out already holds copies, so the profile's table is now only a hazard:
    # ConvertTo-Json:AsArray = $true is a legitimate setting that turns this record into a JSON
    # array, which the reader rejects for not being a dictionary.
    "$PSDefaultParameterValues = @{}; "
    # FRAMED, not bare: the profile is free to print a banner or a MOTD, and that output
    # arriving ahead of the JSON made the parse throw. Module-qualified, as with the uv lookup:
    # an alias named ConvertTo-Json or Write-Output would otherwise reshape the frame.
    f"if ($out.Count -gt 0) {{ "
    f"Microsoft.PowerShell.Utility\\Write-Output '{_PROXY_PROBE_BEGIN}'; "
    f"$out | Microsoft.PowerShell.Utility\\ConvertTo-Json -Compress; "
    f"Microsoft.PowerShell.Utility\\Write-Output '{_PROXY_PROBE_END}' }}"
)


# What the CALLER's host names its own CurrentUserCurrentHost profile, when we can tell. An
# unidentifiable host gets no extra profile rather than someone else's.
_HOST_PROFILE_BY_TERM_PROGRAM = {"vscode": "Microsoft.VSCode_profile.ps1"}


def _caller_host_profile_name() -> Optional[str]:
    term_program = (os.environ.get("TERM_PROGRAM") or "").strip().casefold()
    return _HOST_PROFILE_BY_TERM_PROGRAM.get(term_program)


# Windows PowerShell's own module directory, under %SystemRoot%.
_WINDOWS_PS_MODULE_DIR = r"System32\WindowsPowerShell\v1.0\Modules"
_WINDOWS_PS_HOSTS = frozenset({"powershell.exe", "powershell", "powershell_ise.exe"})


def _fold_module_entry(entry: str) -> str:
    """A PSModulePath entry reduced to a comparable form (separator and case insensitive)."""
    return entry.replace("/", "\\").rstrip("\\").casefold()


def _windows_powershell_module_path(current: str) -> Optional[str]:
    """``current`` with Windows PowerShell's own module directory first, or None to leave it be.

    PowerShell 7 strips its module paths only when IT launches powershell.exe, so reached through
    this Python process the child keeps them in FRONT and resolves 5.1's own modules to the
    PowerShell 7 copies. A profile that imports one then throws while it is dot-sourced and the
    proxy it was going to publish is lost. Same repair as install.ps1, applied per probed host:
    pwsh needs none, because PowerShell 7 prefixes its own paths on startup."""
    root = os.environ.get("SystemRoot")
    if not root:
        return None
    own = root.rstrip("\\/") + "\\" + _WINDOWS_PS_MODULE_DIR
    entries = [entry for entry in current.split(";") if entry.strip()]
    if entries and _fold_module_entry(entries[0]) == _fold_module_entry(own):
        return None
    kept = [e for e in entries if _fold_module_entry(e) != _fold_module_entry(own)]
    return ";".join([own] + kept)


def _profile_probe_env(host: str = "") -> dict:
    """The probe child's environment: the caller's host profile when it is known, and the probed
    host's own module precedence."""
    env = dict(os.environ)
    name = _caller_host_profile_name()
    if name:
        env["_UNSLOTH_PS_HOST_PROFILE"] = name
    else:
        env.pop("_UNSLOTH_PS_HOST_PROFILE", None)
    if platform.system() == "Windows" and _fold_module_entry(host).rsplit("\\", 1)[-1] in (
        _WINDOWS_PS_HOSTS
    ):
        # os.environ upper-cases keys on Windows, so the copy is keyed PSMODULEPATH: reading
        # "PSModulePath" misses the caller's value and adds a second, case-differing entry.
        key = next((k for k in env if k.upper() == "PSMODULEPATH"), "PSModulePath")
        reordered = _windows_powershell_module_path(env.get(key, ""))
        if reordered:
            env[key] = reordered
    return env


def _framed_probe_record(stdout: str) -> Optional[str]:
    """The JSON between the markers, or None. Tolerates anything the profile printed."""
    start = stdout.find(_PROXY_PROBE_BEGIN)
    if start < 0:
        return None
    start += len(_PROXY_PROBE_BEGIN)
    end = stdout.find(_PROXY_PROBE_END, start)
    if end < 0:
        return None
    return stdout[start:end].strip() or None


def _profile_probe_hosts() -> list[str]:
    r"""The PowerShell hosts whose profile to ask, most likely caller first.

    The two editions keep SEPARATE profiles, so both are asked and the caller's edition goes
    first, its value winning on merge.

    The caller is inferred from PSModulePath, which every host exports: Windows PowerShell's
    points at ``...\WindowsPowerShell\v1.0\Modules`` and pwsh's at ``...\PowerShell\7\Modules``.
    By ORDER, not by absence, since a machine can have both trees on PSModulePath at once and
    each host puts its OWN module directory first. Neither present, or both at the same
    position: keep the default order rather than guess.
    """
    hosts = ["pwsh.exe", "powershell.exe"]
    module_path = os.environ.get("PSModulePath", "").lower()
    windows_at = module_path.find("windowspowershell")
    seven_at = module_path.find("powershell\\7")
    if windows_at >= 0 and (seven_at < 0 or windows_at < seven_at):
        hosts = ["powershell.exe", "pwsh.exe"]
    return [host for host in hosts if shutil.which(host)]


# Annotations below are quoted: this module has no `from __future__ import annotations`, and on
# Python 3.9 evaluating `str | list[str]` at def time raises TypeError, taking the CLI import
# with it.

# The whole profile probe's budget, shared across however many hosts are tried.
_PROFILE_PROBE_TIMEOUT_SECONDS = 20.0


def _probe_profile_proxy_defaults(powershell: "str | list[str]") -> Optional[str]:
    """The caller's profile proxy defaults as JSON, or None.

    install.ps1 hands these over in the environment, but a standalone `unsloth studio update`
    has no installer above it and a PowerShell variable does not reach this Python process. So
    ask for it, in a throwaway process whose only job is to print the table.

    Several hosts may be given: their answers are MERGED, earlier hosts winning per key.

    Best effort. A slow, interactive or broken profile costs one timeout and the child proceeds
    as it does today."""
    hosts = [powershell] if isinstance(powershell, str) else list(powershell)
    # ONE budget for the whole probe, not one per host: with both editions installed, two hung
    # profiles would otherwise stall every standalone update for twice the stated cost.
    deadline = time.monotonic() + _PROFILE_PROBE_TIMEOUT_SECONDS
    merged: dict = {}
    # $PSDefaultParameterValues keys are case-INSENSITIVE, so "Invoke-WebRequest:Proxy" and
    # "invoke-webrequest:proxy" are one entry to PowerShell and two to a Python dict. Only the
    # first spelling seen is handed on, or the prelude would replay both and let the
    # lower-priority host's land last.
    claimed: dict = {}
    seen_keys: set = set()
    for host in hosts:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            probe = subprocess.run(
                [
                    host,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    _PS_PROXY_PROBE,
                ],
                env = _profile_probe_env(host),
                capture_output = True,
                text = True,
                # text=True alone decodes with the locale codec and STRICT errors, so a UTF-8
                # profile banner on an ANSI console raised UnicodeDecodeError -- neither
                # OSError nor SubprocessError, so it escaped the handler below and took the
                # update down.
                encoding = "utf-8",
                errors = "replace",
                timeout = remaining,
                **_windows_hidden_subprocess_kwargs(),
            )
        except (OSError, subprocess.SubprocessError):
            continue
        payload = _framed_probe_record(probe.stdout or "")
        if not payload:
            continue
        try:
            # Validated, not trusted: the framing finds the record, this confirms it is one, so
            # nothing hands the child a string ConvertFrom-Json throws on.
            parsed = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(parsed, dict) or not parsed:
            continue
        # Per CMDLET, not per key: pairing the earlier host's Proxy with the later host's
        # ProxyUseDefaultCredentials would offer the user's Windows credentials to a proxy whose
        # own profile never asked for that. A cmdlet is claimed whole by the first host.
        for key, value in parsed.items():
            if not isinstance(key, str):
                continue
            cmdlet = key.split(":", 1)[0].casefold()
            folded = key.casefold()
            if _cmdlet_claimed_elsewhere(cmdlet, claimed, parsed):
                continue

            if folded in seen_keys:
                continue
            claimed[cmdlet] = parsed
            seen_keys.add(folded)
            merged[key] = value
    if not merged:
        return None
    return json.dumps(merged)


def _cmdlet_claimed_elsewhere(cmdlet: str, claimed: dict, source: object) -> bool:
    """Whether another host already owns the cmdlet family ``cmdlet`` belongs to.

    A literal comparison is not enough: the command half of a $PSDefaultParameterValues key may
    be a wildcard, and PowerShell applies such an entry to every cmdlet it matches. Overlap in
    either direction claims the family.
    """
    for owned, owner in claimed.items():
        if owner is source:
            continue
        if owned == cmdlet or _patterns_can_overlap(cmdlet, owned):
            return True
    return False


@functools.lru_cache(maxsize = 512)
def _patterns_can_overlap(left: str, right: str) -> bool:
    """Whether any command name matches BOTH wildcard patterns.

    Two patterns can share matches without either matching the other as a STRING (Invoke-Web* and
    *-WebRequest both apply to Invoke-WebRequest), so the languages are intersected rather than
    string-matched -- otherwise a host's Start-Bits* entry swallowed the other host's unrelated
    Invoke-Web* one. A character class is not decided, only assumed to overlap: that is the
    conservative answer, and no cmdlet name is written that way.
    """
    if "[" in left or "[" in right:
        return True

    @functools.lru_cache(maxsize = None)
    def walk(i: int, j: int) -> bool:
        # Both patterns are consumed in step, except at a '*', which may absorb one more
        # character from the other side or match nothing at all.
        if i == len(left):
            return all(char == "*" for char in right[j:])
        if j == len(right):
            return all(char == "*" for char in left[i:])
        here, there = left[i], right[j]
        if here == "*":
            return walk(i + 1, j) or walk(i, j + 1)
        if there == "*":
            return walk(i, j + 1) or walk(i + 1, j)
        if here == "?" or there == "?" or here == there:
            return walk(i + 1, j + 1)
        return False

    return walk(0, 0)


_PS_PROXY_DEFAULTS_PRELUDE = (
    "$__unslothProxyDefaults = $env:_UNSLOTH_PS_PROXY_DEFAULTS; "
    # Read once, then GONE from the environment: a profile proxy can carry credentials
    # (http://user:secret@proxy is the ordinary corporate form) and every native process setup
    # starts would otherwise inherit it. Needed only by this session's
    # $PSDefaultParameterValues, which is not inherited.
    "Remove-Item Env:_UNSLOTH_PS_PROXY_DEFAULTS -ErrorAction SilentlyContinue; "
    "if ($__unslothProxyDefaults) { try { "
    "(ConvertFrom-Json $__unslothProxyDefaults).PSObject.Properties | ForEach-Object { "
    "$PSDefaultParameterValues[$_.Name] = $_.Value } } catch { } }; "
)


def _run_setup_script(*, verbose: bool = False, repo_root: Optional[Path] = None) -> None:
    """Find and run the studio setup/update script."""
    script = _find_setup_script(repo_root)
    if not script:
        if repo_root is not None:
            name = "setup.ps1" if platform.system() == "Windows" else "setup.sh"
            typer.echo(f"Error: {repo_root} has no studio/{name}.", err = True)
            typer.echo("  --local needs a complete checkout: the setup script builds", err = True)
            typer.echo("  the frontend into the tree that is installed editable.", err = True)
        else:
            typer.echo("Error: Could not find setup script (setup.sh / setup.ps1).")
        raise typer.Exit(1)

    env = {**os.environ, "UNSLOTH_VERBOSE": "1"} if verbose else None

    if platform.system() == "Windows":
        # Resolved, not bare: the gate that runs immediately before this in setup() and update()
        # had to stop trusting PATH for exactly this reason (#9440), and the Popen below has no
        # OSError handler, so a bare name here just moves the same WinError 2 one frame later.
        powershell = _studio_runtime_gate.resolve_windows_powershell()
        powershell_args = [powershell]
        # PRESENCE, not truthiness: install.ps1 publishes this around the handoff and sets it to
        # "{}" when it found no proxy, so treating that as "nobody handed anything over" would
        # send an installer launch off to reload the profiles it deliberately discarded.
        if os.environ.get("_UNSLOTH_PS_PROXY_DEFAULTS") is None:
            probed = _probe_profile_proxy_defaults(_profile_probe_hosts() or [powershell])
            if probed:
                env = {**(env or os.environ), "_UNSLOTH_PS_PROXY_DEFAULTS": probed}
        # -NoProfile unconditionally, not just on the hidden branch: install.ps1 hands off to
        # exactly here from a console where stdout is a tty, so the hidden branch does not fire
        # and a profile that aliases uv or python would break setup.ps1.
        powershell_args.append("-NoProfile")
        if _should_hide_windows_subprocesses():
            powershell_args.extend(["-NoLogo", "-NonInteractive", "-WindowStyle", "Hidden"])
        # Use -Command + `*>&1` (not -File) so setup.ps1's Write-Host output
        # (Information stream #6) merges into stdout. -File drops it when
        # stdout is a pipe, e.g. `unsloth studio update --local 2>&1 | tee`.
        # Single-quote escaping handles paths containing apostrophes.
        script_pwsh_literal = str(script).replace("'", "''")
        powershell_args.extend(
            [
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                f"{_PS_PROXY_DEFAULTS_PRELUDE}& '{script_pwsh_literal}' *>&1",
            ]
        )
        # Explicitly hand std handles to the child so CI tee sees setup.ps1's
        # output. On Windows, subprocess.Popen defaults to close_fds=True
        # (bInheritHandles=False); combined with CREATE_NO_WINDOW the child
        # has no console and no inherited handles, so Write-Host writes to
        # nothing. Passing stdout/stderr makes Python mark the std handles
        # inheritable via PROC_THREAD_ATTRIBUTE_HANDLE_LIST. Empty update.log
        # on windows-latest CI was the smoking gun (runs 25533694490/25534292239).
        process = subprocess.Popen(
            powershell_args,
            env = env,
            stdin = _stream_for_subprocess(sys.stdin),
            stdout = _stream_for_subprocess(sys.stdout),
            stderr = _stream_for_subprocess(sys.stderr),
            **_windows_hidden_subprocess_kwargs(),
        )
        returncode = _wait_for_windows_setup_process(process)
    else:
        result = subprocess.run(["bash", str(script)], env = env)
        returncode = result.returncode

    if returncode != 0:
        raise typer.Exit(returncode)


# The refresh re-runs the installer with --shortcuts-only, fetched rather than shipped
# so a launcher fix reaches users without waiting for a release.
_INSTALLER_URL_BASH = "https://unsloth.ai/install.sh"
_INSTALLER_URL_PWSH = "https://unsloth.ai/install.ps1"
# unsloth.ai 301s to raw.githubusercontent.com, so both are in the chain. Anywhere
# else, or plain http, is refused rather than followed.
_INSTALLER_FETCH_HOSTS = frozenset({"unsloth.ai", "raw.githubusercontent.com"})
_INSTALLER_FETCH_TIMEOUT = 30
# install.sh is ~250KB; the cap just stops an unbounded body from being buffered.
_INSTALLER_MAX_BYTES = 8 * 1024 * 1024
# The flag this code passes, so an installer without it cannot serve the request.
# Internal names would be tighter but can be renamed in a perfectly good installer,
# and a false negative here skips every wheel-based refresh until new Python ships.
_INSTALLER_MARKERS = {
    "install.sh": (b"--shortcuts-only",),
    "install.ps1": (b"--shortcuts-only",),
}


def _is_allowed_installer_url(url: str) -> bool:
    """https on a known host. Applied to the first request and to every redirect."""
    split = urllib.parse.urlsplit(url)
    return split.scheme == "https" and split.hostname in _INSTALLER_FETCH_HOSTS


class _InstallerRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep the installer fetch on the unsloth.ai -> raw.githubusercontent chain."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _is_allowed_installer_url(newurl):
            raise urllib.error.URLError(f"refused installer redirect to {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _build_installer_opener() -> urllib.request.OpenerDirector:
    """A private opener, so the redirect chain can be checked before it is followed.

    urlopen() would instead use whatever install_opener() put in place. Carrying those
    handlers over was tried and abandoned: OpenerDirector.add_handler() assigns
    handler.parent, so sharing them repoints the installed opener at this one and
    breaks every later urlopen() in the process, and copying them does not help either
    because the default HTTPSHandler here already answers first. Proxy environment
    variables and the system trust store still work, since build_opener() sets up both.
    The residue is a machine whose proxy auth or CA lives in a programmatically
    installed opener: its refresh is skipped, which the update survives, rather than
    silently pulling the installer through an unchecked path.
    """
    return urllib.request.build_opener(_InstallerRedirectHandler)


def _looks_like_installer(body: Optional[bytes], installer_name: str) -> bool:
    """Cheap shape check before a fetched installer is executed.

    Not a trust check -- the hosts above are trusted. It stops a captive-portal page or
    an HTTP error body from being piped into bash when something in between answers.
    """
    if not body or len(body) < 512:
        return False
    head = body.lstrip()[:256].lower()
    # `<#` opens PowerShell comment-based help, which is a perfectly ordinary way for
    # install.ps1 to start, so match the actual markup rather than any leading "<".
    if not head.startswith(b"<#") and (
        head.startswith((b"<!doctype", b"<html", b"<head", b"<?xml", b"<body"))
        or b"<html" in head
        or b"<!doctype" in head
    ):
        return False
    return all(marker in body for marker in _INSTALLER_MARKERS[installer_name])


def _fetch_installer(installer_name: str, *, verbose: bool = False) -> Optional[bytes]:
    """Fetch install.sh / install.ps1, or None if nothing usable came back."""
    url = _INSTALLER_URL_PWSH if installer_name == "install.ps1" else _INSTALLER_URL_BASH
    if not _is_allowed_installer_url(url):
        typer.echo(f"  refresh-launcher  refusing to fetch {installer_name} from {url}")
        return None
    try:
        opener = _build_installer_opener()
        request = urllib.request.Request(url, headers = {"User-Agent": "unsloth-studio-update"})
        with opener.open(request, timeout = _INSTALLER_FETCH_TIMEOUT) as response:
            body = response.read(_INSTALLER_MAX_BYTES + 1)
            # read(amt) does not check Content-Length; only a further read() does,
            # raising IncompleteRead. Without it a transfer cut off mid-file still
            # carries the markers and would be piped into bash half-written.
            if len(body) <= _INSTALLER_MAX_BYTES:
                body += response.read()
    except (
        urllib.error.URLError,
        # IncompleteRead / a malformed proxy response raises at the HTTP framing layer,
        # which is neither URLError nor OSError, so it would abort an already-done update.
        http.client.HTTPException,
        TimeoutError,
        OSError,
        ValueError,
    ) as exc:
        typer.echo(f"  refresh-launcher  skipped: could not fetch {url} ({exc})")
        return None

    if len(body) > _INSTALLER_MAX_BYTES:
        typer.echo(f"  refresh-launcher  skipped: oversized {installer_name} response")
        return None
    if not _looks_like_installer(body, installer_name):
        typer.echo(f"  refresh-launcher  skipped: response is not {installer_name}")
        return None
    if verbose:
        typer.echo(f"  refresh-launcher  fetched {url} ({len(body)} bytes)")
    return body


def _installer_script_candidates(installer_name: str) -> List[Path]:
    """Source-tree installers, which outrank the network because `update --local` is
    testing its own installer, so fetching over the top of it would be wrong."""
    candidates: List[Path] = []
    local_repo = (os.environ.get("STUDIO_LOCAL_REPO") or "").strip()
    if local_repo:
        candidates.append(Path(local_repo).expanduser() / installer_name)
    # Clone or editable install: _PACKAGE_ROOT is the repo root.
    root = _PACKAGE_ROOT / installer_name
    if root not in candidates:
        candidates.append(root)
    return candidates


def _installers_on_disk(candidates: Sequence[Path]) -> List[Path]:
    """Every candidate that exists, not just the first.

    The pre-refactor loop probed and launched in one pass, so a candidate that could
    not be launched left the next one to try before the network was reached. Returning
    only the first match would quietly drop that second chance.
    """
    found: List[Path] = []
    for candidate in candidates:
        try:
            if candidate.is_file():
                found.append(candidate)
        except OSError:
            continue
    return found


def _refresh_desktop_shortcuts(*, verbose: bool = False) -> None:
    """Re-run installer with --shortcuts-only to refresh launchers post-update."""
    env = {**os.environ}
    if verbose:
        env["UNSLOTH_VERBOSE"] = "1"

    is_windows = platform.system() == "Windows"
    installer_name = "install.ps1" if is_windows else "install.sh"

    args = ["--shortcuts-only"]
    if verbose:
        args.append("--verbose")

    checkouts = _installers_on_disk(_installer_script_candidates(installer_name))

    if is_windows:
        ps_argv: List[str] = [_studio_runtime_gate.resolve_windows_powershell()]
        # -NoProfile unconditionally, as in _run_setup_script above: gating it on the hidden
        # branch left the visible console path, where a profile is exactly what IS loaded.
        ps_argv.append("-NoProfile")
        if _should_hide_windows_subprocesses():
            ps_argv.extend(["-NoLogo", "-NonInteractive", "-WindowStyle", "Hidden"])

        # Stops at the first candidate that launched; only an unlaunchable one moves on.
        if any(_run_installer_ps1(script, args, ps_argv, env) for script in checkouts):
            return
        fetched = _fetch_installer(installer_name, verbose = verbose)
        if fetched is not None:
            _run_fetched_installer_ps1(fetched, args, ps_argv, env)
        return

    if any(_run_installer_bash(script, args, env) for script in checkouts):
        return
    fetched = _fetch_installer(installer_name, verbose = verbose)
    if fetched is not None:
        _run_fetched_installer_bash(fetched, args, env)


def _run_installer_bash(script: Path, args: Sequence[str], env: dict) -> bool:
    """False when the interpreter could not be launched, so the caller can fall back.

    The pre-refactor candidate loop caught that OSError and carried on to the next
    candidate and then to the network, silently. Returning False keeps a machine that
    cannot spawn bash for the checkout on exactly that path instead of ending the
    refresh early.
    """
    try:
        result = subprocess.run(["bash", str(script), *args], env = env, check = False)
    except OSError:
        return False
    if result.returncode != 0:
        typer.echo(f"  refresh-launcher  {script.name} exited {result.returncode}")
    return True


def _run_fetched_installer_bash(installer: bytes, args: Sequence[str], env: dict) -> None:
    try:
        result = subprocess.run(["bash", "-s", "--", *args], input = installer, env = env, check = False)
    except OSError as exc:
        typer.echo(f"  refresh-launcher  skipped: bash exec failed ({exc})")
        return
    if result.returncode != 0:
        typer.echo(f"  refresh-launcher  fetched install.sh exited {result.returncode}")


def _run_installer_ps1(
    script: Path, args: Sequence[str], ps_argv: Sequence[str], env: dict
) -> bool:
    """False when powershell.exe could not be launched. See _run_installer_bash."""
    quoted = str(script).replace("'", "''")
    argv = list(ps_argv)
    argv.extend(["-ExecutionPolicy", "Bypass", "-Command", f"& '{quoted}' {' '.join(args)} *>&1"])
    try:
        result = subprocess.run(argv, env = env, check = False, **_windows_hidden_subprocess_kwargs())
    except OSError:
        return False
    if result.returncode != 0:
        typer.echo(f"  refresh-launcher  {script.name} exited {result.returncode}")
    return True


def _run_fetched_installer_ps1(
    installer: bytes, args: Sequence[str], ps_argv: Sequence[str], env: dict
) -> None:
    """Run a fetched install.ps1 from a tempfile.

    -File rather than `-Command -`: stdin is decoded with [Console]::InputEncoding
    (CP1252/OEM on most Windows boxes), which mangles install.ps1's box-drawing chars,
    while -File honours the BOM written below. The args go after the path so the
    installer's own `Install-UnslothStudio @args` at EOF receives them, which is why
    this no longer rewrites that line. The prefix gives AV/EDR engines (and anyone
    grepping temp) a clear identity.

    Creating and writing that file is its own failure mode: a full disk, a read-only or
    missing %TEMP%, or AV holding the handle. Those raise OSError, and the refresh runs
    after the package update has already succeeded, so they are reported and skipped
    rather than allowed to abort the command.
    """
    try:
        ps1_fd, ps1_path = tempfile.mkstemp(prefix = "unsloth-studio-refresh-", suffix = ".ps1")
    except OSError as exc:
        typer.echo(f"  refresh-launcher  skipped: could not create a temp script ({exc})")
        return
    try:
        try:
            with os.fdopen(ps1_fd, "wb") as fh:
                fh.write(b"\xef\xbb\xbf" + installer)
        except OSError as exc:
            typer.echo(f"  refresh-launcher  skipped: could not write the temp script ({exc})")
            return
        argv = list(ps_argv)
        argv.extend(["-ExecutionPolicy", "Bypass", "-File", ps1_path, *args])
        try:
            result = subprocess.run(
                argv, env = env, check = False, **_windows_hidden_subprocess_kwargs()
            )
        except OSError as exc:
            typer.echo(f"  refresh-launcher  skipped: powershell exec failed ({exc})")
            return
        if result.returncode != 0:
            typer.echo(f"  refresh-launcher  fetched install.ps1 exited {result.returncode}")
    finally:
        try:
            os.unlink(ps1_path)
        except OSError:
            pass


@studio_app.command(hidden = True)
def setup(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help = "Full pip/build output during setup for troubleshooting.",
    ),
):
    """Run Unsloth setup (called by install.ps1 / install.sh)."""
    runtime_gate_handoff = _studio_runtime_gate.consume_runtime_gate_handoff()
    with _studio_runtime_launch_guard(inherited = runtime_gate_handoff):
        _studio_runtime_gate.ensure_managed_environment_is_idle(STUDIO_HOME)
        # Duplicate-metadata repair can reinstall unsloth even when the
        # installer set SKIP_STUDIO_BASE. Free and preserve the running Windows
        # launcher exactly as the direct update path does.
        with _WindowsLauncherUpdateTransaction() as launcher_update:
            _run_setup_script(verbose = verbose)
            launcher_update.validate_launcher()


def _fail_if_install_damaged(package_name: str = "unsloth") -> None:
    """Refuse to call an update successful when the tree it produced is damaged.

    pip considers a distribution with intact metadata already satisfied, so an
    update reinstalls nothing when a package's FILES are damaged: it prints
    "Unsloth Studio Installed", exits 0, and the backend then dies at boot. That
    is the shape behind "just re-run the installer", and it is only actionable
    if the update says so.
    """
    if _studio_deps.running_outside_managed_venv((STUDIO_HOME / "unsloth_studio",)):
        # This CLI does not live in the venv the update just wrote, so its own
        # file list describes the wrong tree. Silence beats a wrong answer.
        return
    managed_names = (package_name, "unsloth-zoo")
    managed_conflicts = _studio_deps.installed_metadata_conflicts(names = managed_names)
    if managed_conflicts:
        typer.echo("", err = True)
        typer.echo("Update finished, but Unsloth package metadata is inconsistent:", err = True)
        for entry in managed_conflicts:
            typer.echo(f"  {entry}", err = True)
        typer.echo("", err = True)
        typer.echo("The file check cannot safely choose between these records.", err = True)
        typer.echo("The installer could not repair its managed package metadata.", err = True)
        typer.echo(
            "Recreate the managed environment before running the Unsloth installer again.", err = True
        )
        typer.echo("", err = True)
        typer.echo(
            "To update anyway without this check: unsloth studio update --no-verify", err = True
        )
        raise typer.Exit(code = 1)
    other_conflicts = _studio_deps.installed_metadata_conflicts(exclude_names = managed_names)
    if other_conflicts:
        typer.echo("", err = True)
        typer.echo("Warning: some other packages have duplicate metadata:", err = True)
        for entry in other_conflicts:
            typer.echo(f"  {entry}", err = True)
        typer.echo("", err = True)
        typer.echo("Unsloth skipped file verification for these packages.", err = True)
        typer.echo(
            "Reinstall the intended version from its original package source, or use a clean environment.",
            err = True,
        )
    damaged = _studio_deps.damaged_installed_files()
    if not damaged:
        return
    typer.echo("", err = True)
    typer.echo("Update finished, but some installed files are damaged:", err = True)
    for entry in damaged:
        typer.echo(f"  {entry}", err = True)
    typer.echo("", err = True)
    typer.echo("An update cannot repair these. pip sees intact package metadata and", err = True)
    typer.echo("reinstalls nothing, so Unsloth will keep failing to start. Reinstall", err = True)
    typer.echo("over the top:", err = True)
    # Carry a custom root into the command. The shim is a bare symlink and
    # _ensure_studio_env_exported only sets os.environ for this process, so the
    # shell that runs this line has no UNSLOTH_STUDIO_HOME: an unqualified
    # reinstall would build a fresh ~/.unsloth/studio and leave the damaged
    # install exactly as broken as it was.
    # And carry the recorded install mode. install.sh derives SKIP_TORCH only
    # from its own flag or UNSLOTH_NO_TORCH and passes that value into setup, so
    # a plain reinstall over a GGUF-only install downloads the whole PyTorch
    # stack. Only added when the record says True: recorded_no_torch() returns
    # None when nothing recorded the mode, and None must never be read as False.
    #
    # No root argument: the manifest and marker live in the VENV, not the
    # install root, and recorded_no_torch defaults to Path(sys.prefix). Passing
    # STUDIO_HOME would look one directory too high, find nothing, and silently
    # never fire. The early return above guarantees sys.prefix is that venv.
    no_torch = False
    try:
        _manifest = _studio_deps.load_install_manifest_module()
        no_torch = _manifest is not None and _manifest.recorded_no_torch() is True
    except Exception:
        no_torch = False
    if platform.system() == "Windows":
        prefix = ""
        if _STUDIO_HOME_IS_CUSTOM:
            prefix = "$env:UNSLOTH_STUDIO_HOME = '{}'; ".format(str(STUDIO_HOME).replace("'", "''"))
        if no_torch:
            prefix += "$env:UNSLOTH_NO_TORCH = '1'; "
        typer.echo(f"  {prefix}irm https://unsloth.ai/install.ps1 | iex", err = True)
    else:
        # The assignments go before `sh`, not before `curl`: that is the form
        # install.sh documents, and it is sh that reads them.
        env = ""
        if _STUDIO_HOME_IS_CUSTOM:
            env = f"UNSLOTH_STUDIO_HOME={shlex.quote(str(STUDIO_HOME))} "
        if no_torch:
            env += "UNSLOTH_NO_TORCH=1 "
        typer.echo(f"  curl -fsSL https://unsloth.ai/install.sh | {env}sh", err = True)
    typer.echo("", err = True)
    # The installer installs the current requirement sets; it does not prune or
    # reinstall anything outside them. So a package left over from an older
    # release, or added by hand, is not repaired by the command above and would
    # otherwise report the same damage forever. Say what to do in that case
    # rather than scoping the scan, which would risk passing over real damage.
    typer.echo("If a package above is still listed after that, the installer does not", err = True)
    typer.echo("manage it. Repair it directly, or remove it if nothing needs it:", err = True)
    # --no-deps: without it pip resolves the damaged package's dependency graph
    # and --force-reinstall would replace pinned runtime packages too, which can
    # swap the installed CUDA/ROCm torch build for a default one while repairing
    # an unrelated orphan. The installer's own targeted repairs pair the two for
    # the same reason. The interpreter path is quoted because a custom root may
    # contain spaces, and on Windows a quoted command needs the call operator.
    # <package>==<version> rather than a bare name: --force-reinstall reinstalls
    # even when the package is already up to date, so an unpinned name fetches
    # the newest release and silently upgrades an orphan whose consumers may
    # depend on the older one. --no-deps does not protect against that.
    _spec = "<package>==<installed version>"
    if platform.system() == "Windows":
        _py = str(Path(sys.executable)).replace("'", "''")
        typer.echo(f"  & '{_py}' -m pip install --force-reinstall --no-deps {_spec}", err = True)
    else:
        _py = shlex.quote(str(Path(sys.executable)))
        typer.echo(f"  {_py} -m pip install --force-reinstall --no-deps {_spec}", err = True)
    typer.echo("", err = True)
    typer.echo("To update anyway without this check: unsloth studio update --no-verify", err = True)
    raise typer.Exit(code = 1)


@studio_app.command()
def update(
    local: bool = typer.Option(False, "--local", help = "Install from local repo instead of PyPI"),
    package: str = typer.Option(
        "unsloth", "--package", help = "Package name to install/update (for testing)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help = "Full pip/build output during update for troubleshooting.",
    ),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help = "After updating, scan installed files for damage an update cannot repair.",
    ),
):
    """Update Unsloth Studio dependencies and rebuild."""
    # Re-export UNSLOTH_STUDIO_HOME for env-mode installs so the refresh
    # subprocess resolves the same install root the user originally chose.
    _ensure_studio_env_exported()
    # Ensure SKIP_STUDIO_BASE is not inherited from a parent install.ps1 session
    os.environ.pop("SKIP_STUDIO_BASE", None)
    os.environ["STUDIO_PACKAGE_NAME"] = package
    repo_root: Optional[Path] = None
    if local:
        os.environ["STUDIO_LOCAL_INSTALL"] = "1"
        # Pass the repo root explicitly so install_python_stack.py doesn't
        # have to guess from SCRIPT_DIR (which may be inside site-packages).
        # Deriving it from __file__ only holds while this CLI runs from a
        # checkout. Once an update has installed unsloth into the venv
        # non-editably, parents[2] IS site-packages, and uv rejects it with
        # "does not appear to be a Python project: neither 'setup.py' nor
        # 'pyproject.toml' found" -- which is what a second `update --local`
        # hit on Windows, where the first update replaces the editable install.
        # Absolutise the override: setup.sh does `cd "$SCRIPT_DIR"` before it
        # runs install_python_stack.py, so a relative STUDIO_LOCAL_REPO would
        # be re-resolved against studio/ (no pyproject.toml) and hand uv back
        # the very error this guard exists to replace. .strip()/.expanduser()
        # match the handling in _refresh_desktop_shortcuts.
        _explicit = (os.environ.get("STUDIO_LOCAL_REPO") or "").strip()
        repo_root = (
            Path(_explicit).expanduser().resolve()
            if _explicit
            else Path(__file__).resolve().parents[2]
        )
        if not (repo_root / "pyproject.toml").is_file():
            typer.echo("Error: --local needs an Unsloth checkout to install from.", err = True)
            typer.echo(f"  no pyproject.toml under: {repo_root}", err = True)
            typer.echo("  This CLI is running from an installed copy, not a source tree.", err = True)
            typer.echo("", err = True)
            typer.echo("  Point at a checkout:", err = True)
            if platform.system() == "Windows":
                # PowerShell has no `VAR=value command` prefix form: it parses
                # the assignment as a command name and fails to find it. This
                # guard fires on the Windows update path, so the POSIX spelling
                # would be unusable for most of the people who see it.
                typer.echo(
                    "    $env:STUDIO_LOCAL_REPO='C:\\path\\to\\unsloth'; "
                    "unsloth studio update --local",
                    err = True,
                )
            else:
                typer.echo(
                    "    STUDIO_LOCAL_REPO=/path/to/unsloth unsloth studio update --local",
                    err = True,
                )
            typer.echo("  Or update from PyPI:", err = True)
            typer.echo("    unsloth studio update", err = True)
            raise typer.Exit(2)
        os.environ["STUDIO_LOCAL_REPO"] = str(repo_root)
    else:
        os.environ["STUDIO_LOCAL_INSTALL"] = "0"
        os.environ.pop("STUDIO_LOCAL_REPO", None)
    # main gained a runtime gate around setup; this branch replaced the
    # rename-to-.deleteme helpers with the launcher transaction. Both apply:
    # the gate keeps a second Unsloth process off the venv, the transaction
    # keeps the launcher recoverable across the setup it wraps.
    runtime_gate_handoff = _studio_runtime_gate.consume_runtime_gate_handoff()
    with _studio_runtime_launch_guard(inherited = runtime_gate_handoff):
        _studio_runtime_gate.ensure_managed_environment_is_idle(STUDIO_HOME)
        with _WindowsLauncherUpdateTransaction() as launcher_update:
            _run_setup_script(verbose = verbose, repo_root = repo_root)
            # This deliberately runs even with --no-verify: the broad package scan
            # is optional, but a successful update must leave its own launcher usable.
            launcher_update.validate_launcher()
            if verify:
                _fail_if_install_damaged(package)
    # Tauri desktop owns its own bundle entries; skip CLI launcher refresh
    # so a Tauri-initiated update doesn't create duplicate shortcuts.
    if os.environ.get("UNSLOTH_TAURI_UPDATE") == "1":
        if verbose:
            typer.echo("  refresh-launcher  skipped (Tauri update)")
        return
    _refresh_desktop_shortcuts(verbose = verbose)


class _WindowsLauncherUpdateTransaction:
    """Keep the managed Windows launcher recoverable during a Python update."""

    _VERSION_TIMEOUT_SECONDS = 10
    # Sentinel rather than a message: _launcher_health_error matches on identity,
    # so it can never be confused with a real diagnostic that happens to read the
    # same way, and it never reaches a user.
    _POLICY_BLOCKED = "an Application Control policy blocked the launcher"
    # Absence is not corruption. Quarantine takes the unsigned stub and leaves the
    # environment intact, and nothing executes the stub any more, so the CLI can be
    # perfectly healthy without it. Kept apart from the PE-shape failure, which is
    # still a real one.
    _LAUNCHER_ABSENT = "the updated launcher is not on disk"
    _RESTORE_ATTEMPTS = 3

    def __init__(self) -> None:
        self.enabled = platform.system() == "Windows"
        self.launcher: Optional[Path] = None
        self.backup: Optional[Path] = None
        self.legacy_backup: Optional[Path] = None
        self.stale: Optional[Path] = None
        self.shim: Optional[Path] = None
        self.lock_path: Optional[Path] = None
        self._lock_file = None
        self._validated = False

    @staticmethod
    def _is_valid_pe(path: Path) -> bool:
        try:
            if not path.is_file() or path.stat().st_size < 2:
                return False
            with path.open("rb") as handle:
                return handle.read(2) == b"MZ"
        except OSError:
            return False

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        """Publish a sibling copy without exposing a partial destination."""
        fd, temporary_name = tempfile.mkstemp(
            prefix = f".{destination.name}.",
            suffix = ".tmp",
            dir = str(destination.parent),
        )
        temporary = Path(temporary_name)
        try:
            with source.open("rb") as source_handle, os.fdopen(fd, "wb") as target_handle:
                fd = -1
                while True:
                    chunk = source_handle.read(1024 * 1024)
                    if not chunk:
                        break
                    target_handle.write(chunk)
                target_handle.flush()
                os.fsync(target_handle.fileno())
            os.replace(temporary, destination)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                temporary.unlink(missing_ok = True)
            except OSError:
                pass

    def _acquire_lock(self) -> None:
        import msvcrt

        assert self.lock_path is not None
        try:
            self.lock_path.parent.mkdir(parents = True, exist_ok = True)
        except OSError:
            pass
        lock_file = self.lock_path.open("a+b")
        try:
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            lock_file.close()
            typer.echo(
                "Error: another Unsloth Studio update is already running for this environment.",
                err = True,
            )
            raise typer.Exit(1)
        self._lock_file = lock_file

    def _release_lock(self) -> None:
        if self._lock_file is None:
            return
        try:
            import msvcrt
            self._lock_file.seek(0)
            msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        finally:
            self._lock_file.close()
            self._lock_file = None

    def _recover_missing_launcher(self) -> None:
        assert self.launcher is not None
        # Validity, not existence: a truncated or quarantined launcher is just as
        # unusable, and the backup beside it can repair either.
        if self._is_valid_pe(self.launcher):
            return
        last_error: Optional[Tuple[Path, OSError]] = None
        for recovery in (self.backup, self.stale, self.legacy_backup, self.shim):
            if recovery is not None and self._is_valid_pe(recovery):
                try:
                    self._atomic_copy(recovery, self.launcher)
                except OSError as exc:
                    # Try the next copy. The header check and the copy open the file
                    # separately, so antivirus taking a candidate in between is a race
                    # this loop can lose without the others being unusable, and giving
                    # up on the first one turned a recoverable install into a failure.
                    last_error = (recovery, exc)
                    continue
                return
        if last_error is not None:
            recovery, exc = last_error
            typer.echo(
                f"Error: could not recover {self.launcher} from {recovery}: {exc}",
                err = True,
            )
            typer.echo(f"Manual recovery copy retained at: {recovery}", err = True)
            raise typer.Exit(1)

    @staticmethod
    def _files_match(left: Path, right: Path) -> bool:
        try:
            if left.stat().st_size != right.stat().st_size:
                return False
            with left.open("rb") as left_handle, right.open("rb") as right_handle:
                while True:
                    left_chunk = left_handle.read(1024 * 1024)
                    if left_chunk != right_handle.read(1024 * 1024):
                        return False
                    if not left_chunk:
                        return True
        except OSError:
            return False

    def _move_launcher_aside(self) -> None:
        """Free the canonical path so the installer can publish a replacement.

        uv only self-replaces its OWN executable, so it deletes a third-party
        console script outright and hard-errors when the file is in use, after
        which the pip fallback no-ops on the already-satisfied bare unsloth and
        the upgrade is silently skipped. Renaming a running image is allowed on
        Windows: verified on windows-latest that renaming a live console-script
        launcher succeeds and a replacement can then be written at the freed
        path. Non-fatal, since failing to move it aside costs only the upgrade.
        """
        assert self.launcher is not None and self.stale is not None
        if not self._is_valid_pe(self.launcher):
            return
        try:
            os.replace(self.launcher, self.stale)
        except OSError as exc:
            # Not fatal: an antivirus hold must not make the environment
            # unupdatable. But say what it costs, because uv cannot then replace
            # the launcher and the pip fallback drops --upgrade-package, so
            # unsloth is left at its old version while everything else updates.
            typer.echo(f"Warning: could not move the Unsloth launcher aside: {exc}", err = True)
            typer.echo(
                "  unsloth itself may not be upgraded. Close anything holding "
                f"{self.launcher} and re-run the update.",
                err = True,
            )

    def _retained_backup(self) -> Optional[Path]:
        """The backup, when it exists and is usable. Nothing to point users at otherwise."""
        if self.backup is not None and self._is_valid_pe(self.backup):
            return self.backup
        return None

    def _recovery_candidates(self) -> List[Path]:
        """Copies that could stand in for the launcher, best first.

        The backup is the last launcher known to run; the moved-aside copy is
        only this run's unvalidated canonical file; the legacy .deleteme and the
        PATH shim are what an install broken by the old updater still has. All
        of them are kept, because passing the two-byte header check does not
        make any one of them runnable and the next candidate has to be reachable.
        """
        seen: List[Path] = []
        for path in (self.backup, self.stale, self.legacy_backup, self.shim):
            if path is None or not self._is_valid_pe(path):
                continue
            if not any(os.path.normcase(str(path)) == os.path.normcase(str(p)) for p in seen):
                seen.append(path)
        return seen

    def _restore_from(self, source: Path) -> bool:
        assert self.launcher is not None
        # The common setup-failure case leaves the original executable exactly
        # where it was. Avoid replacing that running file on Windows when it
        # already is the source byte-for-byte.
        if self._is_valid_pe(self.launcher) and self._files_match(self.launcher, source):
            return True
        last_error: Optional[OSError] = None
        for attempt in range(self._RESTORE_ATTEMPTS):
            try:
                self._atomic_copy(source, self.launcher)
                return self._is_valid_pe(self.launcher)
            except OSError as exc:
                last_error = exc
                if attempt + 1 < self._RESTORE_ATTEMPTS:
                    time.sleep(0.1)
        if last_error is not None:
            typer.echo(f"Error: could not restore the Unsloth launcher: {last_error}", err = True)
        return False

    def _restore_runnable(self) -> bool:
        """Put back the first copy that actually runs.

        Passing the two-byte header check does not make a copy runnable, so a
        candidate that fails --version must not stop the next one being tried,
        and a launcher already in place and working must not be replaced by a
        candidate that is merely PE-shaped.

        Under Application Control that discrimination degrades to the PE-shape
        check, because no candidate can be told apart by running it: every
        --version attempt dies in CreateProcess. That is a real limitation and
        not a hidden one -- a PE-shaped but corrupt copy can be left in place --
        but it is also unavoidable, since the only way to tell a good stub from a
        corrupt one is to start it. It is confined to the case where the launcher
        was missing or truncated, since an intact one never reaches this loop:
        _launcher_health_error asks the interpreter and reports healthy.
        """
        if self._launcher_health_error() is None:
            return True
        candidates = self._recovery_candidates()
        for source in candidates:
            if self._restore_from(source) and self._launcher_health_error() is None:
                return True
        # Nothing ran. Leave the best candidate in place rather than whichever
        # one happened to be tried last.
        if candidates:
            self._restore_from(candidates[0])
        # No copy could be put back and started. A launcher that is simply gone,
        # or one the policy denies, is still not a broken CLI, so ask the
        # interpreter before giving up. Asked only here, after every candidate
        # has been tried, so a launcher that could have been recovered still is.
        return self._recovered_cli_health_error() is None

    def _launcher_runs_error(self) -> Optional[str]:
        """Whether THIS launcher file starts and answers --version.

        Deliberately about the file, not about the CLI: it is the only check that
        can catch a stub that is PE-shaped but corrupt, on every machine. Its
        caller decides what a policy denial means.

        The stub is probed first, and still is on a locked-down machine, which
        costs one denied-launch event per update there. That is a log entry, not
        a failure. Asking the interpreter instead would be quieter but blind: it
        cannot see a corrupt launcher on ANY machine, including the overwhelming
        majority that have no policy at all, which is the strictly worse trade.
        """
        assert self.launcher is not None
        if not self.launcher.exists():
            return self._LAUNCHER_ABSENT
        if not self._is_valid_pe(self.launcher):
            return "the updated launcher is not a non-empty PE file"
        try:
            result = subprocess.run(
                [str(self.launcher), "--version"],
                check = False,
                capture_output = True,
                timeout = self._VERSION_TIMEOUT_SECONDS,
                **_windows_hidden_subprocess_kwargs(),
            )
        except subprocess.TimeoutExpired:
            return f"the updated launcher timed out after {self._VERSION_TIMEOUT_SECONDS} seconds"
        except OSError as exc:
            if _is_application_control_block(exc):
                return self._POLICY_BLOCKED
            return f"the updated launcher could not run --version ({exc})"
        if result.returncode != 0:
            return f"the updated launcher returned {result.returncode} for --version"
        return None

    def _launcher_health_error(self) -> Optional[str]:
        """Whether the update left a working CLI -- the question that matters.

        Identical to _launcher_runs_error except when the launcher was denied by
        an Application Control policy. That denial happens before Python starts,
        so it says nothing about the update: the package can be perfectly
        healthy. Ask the signed interpreter beside it instead, or every update on
        such a machine reports failure and rolls a good install back (issue
        #8490). A launcher that fails to start for any other reason is the
        failure it always was.
        """
        error = self._launcher_runs_error()
        if error is self._POLICY_BLOCKED:
            return self._interpreter_health_error(error)
        return error

    def _recovered_cli_health_error(self) -> Optional[str]:
        """_launcher_health_error, once recovering the launcher has been ruled out.

        Absence is the difference. A missing launcher IS worth restoring, so
        _launcher_health_error keeps reporting it and validate_launcher goes on
        to put the previous one back. But quarantine deletes the unsigned stub
        rather than denying it, and no copy survives being restored either, so
        once every candidate has failed, absence says as little about the update
        as a policy denial does: ask the interpreter instead of failing a good
        update and rolling it back (issue #8490).
        """
        error = self._launcher_runs_error()
        if error is self._POLICY_BLOCKED or error is self._LAUNCHER_ABSENT:
            return self._interpreter_health_error(error)
        return error

    def _interpreter_health_error(self, reason: str) -> Optional[str]:
        """Health of the managed CLI when the launcher itself cannot be started.

        Answers the question validate_launcher actually has -- did the update
        leave a working CLI? -- on a machine where --version on the launcher can
        never succeed. Falls back to reporting the original block when there is
        no interpreter to ask.

        Isolated, alone among this module's managed invocations, because the
        launch it is predicting is itself isolated: build_update_command in
        studio/src-tauri/src/update.rs runs the desktop updater under
        Isolation::Isolated and clears PYTHONHOME/PYTHONPATH. Inheriting them
        here would let a foreign checkout on PYTHONPATH answer --version for a
        managed package the update actually broke, and validate_launcher would
        keep an update that the next desktop launch cannot start. -I implies -E,
        so it clears the same two variables the Rust side removes by hand.
        """
        assert self.launcher is not None
        python = self.launcher.parent / "python.exe"
        if not python.is_file():
            blocked = reason is self._POLICY_BLOCKED
            state = "is blocked by an Application Control policy" if blocked else "is missing"
            return (
                f"the updated launcher {state} and there is no managed interpreter "
                f"at {python} to ask instead"
            )
        try:
            result = subprocess.run(
                _managed_cli_argv(python, "--version", isolated = True),
                check = False,
                capture_output = True,
                # The import probe's ceiling, not the launcher's. --version on the
                # launcher is a process start; here it is a bare interpreter start
                # plus the whole CLI package import, which is exactly the work
                # _MANAGED_CLI_IMPORT_PROBE_TIMEOUT is generous for. Under the
                # antivirus scan that produced the quarantine this path exists to
                # survive, the launcher's 10 seconds would call a healthy update
                # broken and roll it back, once per recovery candidate.
                timeout = _MANAGED_CLI_IMPORT_PROBE_TIMEOUT,
                **_windows_hidden_subprocess_kwargs(),
            )
        except subprocess.TimeoutExpired:
            return (
                f"the managed Python CLI timed out after "
                f"{_MANAGED_CLI_IMPORT_PROBE_TIMEOUT} seconds"
            )
        except OSError as exc:
            return f"the managed Python CLI could not run --version ({exc})"
        if result.returncode != 0:
            return f"the managed Python CLI returned {result.returncode} for --version"
        return None

    @staticmethod
    def _managed_scripts_dir() -> Path:
        """Scripts dir of the venv setup actually updates.

        setup.ps1 installs into STUDIO_HOME/unsloth_studio, which is not this
        interpreter when a pip-installed or checkout CLI drives the update. Same
        distinction _studio_deps._managed_root draws for the damage scan.
        """
        managed = STUDIO_HOME / "unsloth_studio"
        if (managed / "pyvenv.cfg").is_file():
            try:
                foreign = managed.resolve() != Path(sys.prefix).resolve()
            except OSError:
                foreign = True
            if foreign:
                return managed / "Scripts"
        return Path(sys.executable).resolve().parent

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            scripts = self._managed_scripts_dir()
        except (OSError, RuntimeError) as exc:
            typer.echo(f"Error: could not resolve the managed Python environment: {exc}", err = True)
            raise typer.Exit(1)
        self.launcher = scripts / "unsloth.exe"
        self.backup = scripts / "unsloth.exe.update-backup"
        self.legacy_backup = scripts / "unsloth.exe.deleteme"
        # Under the Unsloth home, not the venv: setup.ps1 removes the whole
        # $VenvDir to rebuild a stale torch, and an open handle inside it makes
        # Windows refuse the recursive delete. One lock per Unsloth home is the
        # right grain anyway, since that is what names the managed venv.
        self.lock_path = STUDIO_HOME / "unsloth.exe.update-lock"
        # install.ps1 hardlinks this to the launcher, so it survives the old
        # updater's .deleteme unlink and is a valid recovery source.
        self.shim = STUDIO_HOME / "bin" / "unsloth.exe"
        self.stale = scripts / "unsloth.exe.update-stale"
        self._acquire_lock()
        try:
            self._recover_missing_launcher()
            if not self._is_valid_pe(self.launcher):
                # Warn, do not exit. The previous updater could leave an install
                # with no launcher and no .deleteme, and refusing here would stop
                # exactly those users from ever updating again. Setup may well
                # write a new launcher; validate_launcher still judges the result.
                typer.echo(
                    f"Warning: the managed Unsloth launcher is missing or invalid: {self.launcher}",
                    err = True,
                )
                typer.echo("Continuing; setup may reinstall it.", err = True)
                if self._retained_backup() is None:
                    self.backup = None
            elif self._retained_backup() is None:
                # Only write a backup when there is no usable one already. A
                # backup outlives __enter__ only when a previous run died before
                # validating, so it holds the last launcher known to run, while
                # the canonical file has passed nothing but the two-byte header
                # check. Overwriting it here destroyed the only recovery copy.
                try:
                    self._atomic_copy(self.launcher, self.backup)
                except OSError as exc:
                    # A backup is a safety net, not a precondition. Antivirus or a
                    # locked-down Scripts dir must not abort the update outright.
                    typer.echo(f"Warning: could not back up the Unsloth launcher: {exc}", err = True)
                    self.backup = None
            self._move_launcher_aside()
        except BaseException:
            self._release_lock()
            raise
        return self

    def validate_launcher(self) -> None:
        if not self.enabled:
            return
        # Whether setup published anything decides how a bad result is read, so
        # it has to be sampled before any restore puts a launcher back.
        published = self.launcher.exists()
        error = self._launcher_health_error()
        if error is not None:
            restored = self._restore_runnable()
            # Setup publishing nothing is the case this transaction exists for:
            # a no-op pip update leaves the freed path empty, and the old
            # updater then deleted its own .deleteme, leaving no launcher at
            # all. Putting the previous one back is success, not failure. A
            # launcher setup DID write and that cannot run is still a failure,
            # even though the previous one goes back.
            if published or not restored:
                typer.echo(f"Error: Unsloth Studio update failed because {error}.", err = True)
                if restored:
                    typer.echo("The previous launcher was restored.", err = True)
                elif self._retained_backup() is not None:
                    typer.echo(f"Manual recovery copy retained at: {self.backup}", err = True)
                raise typer.Exit(1)
        self._validated = True
        # Only once the launcher is actually back. A quarantined or locked stub can
        # be judged healthy through the interpreter while every attempt to restore
        # a copy failed, and deleting the copies there would throw away the only
        # material a later run could recover from. They are fixed names, so keeping
        # them costs nothing and the next update that does put a launcher back
        # clears them.
        if not self._is_valid_pe(self.launcher):
            return
        for orphan in (self.stale, self.backup, self.legacy_backup):
            if orphan is None:
                continue
            try:
                orphan.unlink(missing_ok = True)
            except OSError:
                pass

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            if self.enabled and exc_type is not None and not self._validated:
                if not self._restore_runnable() and self._retained_backup() is not None:
                    typer.echo(f"Manual recovery copy retained at: {self.backup}", err = True)
        finally:
            self._release_lock()
        return False


# ── unsloth studio reset-password ────────────────────────────────────


@studio_app.command("desktop-capabilities", hidden = True)
def desktop_capabilities(
    json_output: bool = typer.Option(
        False,
        "--json",
        help = "Emit machine-readable JSON.",
    ),
):
    state = _install_state()
    payload = {
        "desktop_protocol_version": 1,
        # 2 adds studio_install_ok; the desktop treats < 2 as stale rather than
        # guess at an absent field.
        "desktop_manageability_version": 2,
        "supports_provision_desktop_auth": True,
        "supports_api_only": True,
        "supports_desktop_backend_ownership": True,
        # Did the install finish and are the backend's boot deps still there.
        "studio_install_ok": bool(state["ok"]),
        "studio_install_reason": state["reason"],
        "version": "unknown",
    }
    try:
        from importlib.metadata import version as package_version
        payload["version"] = package_version("unsloth")
    except Exception:
        pass

    if json_output:
        typer.echo(json.dumps(payload, sort_keys = True))
        return

    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


@studio_app.command("verify-install")
def verify_install(
    json_output: bool = typer.Option(
        False,
        "--json",
        help = "Emit machine-readable JSON.",
    ),
):
    """Check that the Unsloth Studio dependency install completed.

    Exits 0 when complete, 1 otherwise. setup.sh / setup.ps1 use the exit code
    to decide whether the "already up to date" fast path may be taken.
    """
    state = _install_state()

    if json_output:
        typer.echo(json.dumps(state, sort_keys = True))
        raise typer.Exit(0 if state["ok"] else 1)

    if state["ok"]:
        typer.echo("Unsloth Studio install is complete.")
        raise typer.Exit(0)

    typer.echo(f"Unsloth Studio install is incomplete ({state['reason']}).")
    if state["missing"]:
        typer.echo(f"  missing packages: {', '.join(state['missing'])}")
    typer.echo("  repair with: unsloth studio update")
    raise typer.Exit(1)


@studio_app.command("provision-desktop-auth", hidden = True)
def provision_desktop_auth():
    """Create/repair desktop auth state for the local machine."""
    auth_dir = STUDIO_HOME / "auth"
    secret = _create_desktop_secret_in_cli()
    _write_auth_secret(auth_dir / DESKTOP_SECRET_FILE, secret)
    typer.echo("Desktop auth ready.")


@studio_app.command("reset-password")
def reset_password():
    """Reset the Unsloth admin password.

    Rotates the credential in place: a running Unsloth accepts the new password on
    its next request, so there is nothing to restart. Shared /p preview links are
    not revoked -- rotate those in Settings if the old password leaked.
    """
    new_password = _generate_reset_password()
    try:
        conn = _connect_auth_db()
    except (OSError, sqlite3.Error) as exc:
        typer.echo(
            f"Error: could not open the auth database ({exc}). Check that "
            f"{STUDIO_HOME / 'auth'} is writable; if auth.db itself is unreadable, stop "
            "Unsloth, delete it, and start again to re-seed.",
            err = True,
        )
        raise typer.Exit(1)

    try:
        _ensure_cli_default_admin(conn)
        _cli_update_password(conn, DEFAULT_ADMIN_USERNAME, new_password, revoke_api_keys = True)
    except (OSError, sqlite3.Error) as exc:
        typer.echo(f"Error: could not reset the password ({exc}).", err = True)
        raise typer.Exit(1)
    finally:
        conn.close()

    typer.echo(f"New password for '{DEFAULT_ADMIN_USERNAME}': {new_password}")
    typer.echo(
        "Sessions and API keys revoked. A running Unsloth takes it on the next request, "
        "though repeated failed logins can hold the rate limit shut for up to a minute."
    )
