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

from unsloth_cli import _studio_deps, _studio_runtime_gate, _studio_stage
from unsloth_cli._inference import SpeculativeType
from unsloth_cli.commands import _password_prompt

studio_app = typer.Typer(help = "Unsloth Studio commands.")


def _enable_verbose_access_logs() -> None:
    os.environ["UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS"] = "0"
    os.environ["UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS"] = "0"


# Root order: UNSLOTH_STUDIO_HOME, STUDIO_HOME, UNSLOTH_HOME/studio, sys.prefix,
# legacy ~/.unsloth/studio. Keep this aligned with storage_roots.studio_root().
# Shim markers mirror install.ps1 / uninstall.ps1 and are matched as bytes.
_CMD_SHIM_MARKERS = (b"unsloth-studio-managed-launcher", b"from unsloth_cli import app")
_CMD_SHIM_MAX_BYTES = 8192


def _looks_like_installer_managed_studio_home(candidate: Path) -> bool:
    """Sentinel (share/studio.conf, or our own bin/unsloth.cmd) so a dev venv named
    unsloth_studio is not taken for a custom root. It must be OUR shim: the dir is on PATH."""
    if (candidate / "share" / "studio.conf").is_file():
        return True
    if platform.system() != "Windows":
        return (candidate / "bin" / "unsloth").is_file()
    if (candidate / "bin" / "unsloth.exe").is_file():
        return True
    return _is_managed_cmd_shim(candidate / "bin" / "unsloth.cmd")


def _is_managed_cmd_shim(path: Path) -> bool:
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
    # Keeps the CLI on the same root as storage_roots.py; see test_unsloth_home_root_agreement.py.
    master = (os.environ.get("UNSLOTH_HOME") or "").strip()
    if master:
        try:
            candidate = Path(master).expanduser().resolve() / "studio"
        except (OSError, ValueError):
            candidate = Path(master).expanduser() / "studio"
        try:
            is_custom = candidate != (Path.home() / ".unsloth" / "studio").resolve()
        except (OSError, ValueError):
            is_custom = candidate != (Path.home() / ".unsloth" / "studio")
        return candidate, is_custom
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
    """Re-export UNSLOTH_STUDIO_HOME / UNSLOTH_LLAMA_CPP_PATH for custom roots only, per
    subcommand rather than at import, so unrelated importers see no env changes."""
    if not _STUDIO_HOME_IS_CUSTOM:
        return
    # Truthy-check, not setdefault: a blank UNSLOTH_STUDIO_HOME= must not win.
    if not os.environ.get("UNSLOTH_STUDIO_HOME"):
        os.environ["UNSLOTH_STUDIO_HOME"] = str(STUDIO_HOME)
    try:
        _legacy_studio = (Path.home() / ".unsloth" / "studio").resolve()
        _is_legacy = STUDIO_HOME.resolve() == _legacy_studio
    except (OSError, ValueError):
        _is_legacy = STUDIO_HOME == (Path.home() / ".unsloth" / "studio")
    # The runtimes are siblings of studio/, at the master root, so STUDIO_HOME/llama.cpp is one
    # level too deep. run.py keeps a non-blank value, so a wrong export here wins everywhere.
    _master = (os.environ.get("UNSLOTH_HOME") or "").strip()
    if _master:
        try:
            _llama_dir = Path(_master).expanduser().resolve() / "llama.cpp"
        except (OSError, ValueError):
            _llama_dir = Path(_master).expanduser() / "llama.cpp"
    elif _is_legacy:
        _llama_dir = Path.home() / ".unsloth" / "llama.cpp"
    else:
        _llama_dir = STUDIO_HOME / "llama.cpp"
    if not os.environ.get("UNSLOTH_LLAMA_CPP_PATH"):
        os.environ["UNSLOTH_LLAMA_CPP_PATH"] = str(_llama_dir)


BOOTSTRAP_PASSWORD_FILE = ".bootstrap_password"
DESKTOP_SECRET_FILE = ".desktop_secret"
# Cached raw CLI API key; the full name carries a digest: ".cli_api_key_cli_<digest>".
CLI_API_KEY_FILE_PREFIX = ".cli_api_key_"
DEFAULT_ADMIN_USERNAME = "unsloth"
DESKTOP_SECRET_PREFIX = "desktop-"
API_KEY_PBKDF2_SALT_KEY = "api_key_pbkdf2_salt"
DESKTOP_SECRET_HASH_KEY = "desktop_secret_hash"
DESKTOP_SECRET_CREATED_AT_KEY = "desktop_secret_created_at"
PBKDF2_ITERATIONS = 100_000
_START_API_KEY_MARKER_ENV = "_UNSLOTH_START_API_KEY_MARKER"
_CLOUDFLARE_INTENT_ENV = "_UNSLOTH_CLOUDFLARE_INTENT"


def _consume_start_api_key_marker_env() -> bool:
    return os.environ.pop(_START_API_KEY_MARKER_ENV, None) == "1"


def _preserve_cloudflare_intent(cloudflare: Optional[bool], secure: bool) -> None:
    if _CLOUDFLARE_INTENT_ENV in os.environ:
        return
    if secure or cloudflare is True:
        intent = "enabled"
    elif cloudflare is False:
        intent = "disabled"
    else:
        intent = "unset"
    os.environ[_CLOUDFLARE_INTENT_ENV] = intent


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _should_hide_windows_subprocesses() -> bool:
    if platform.system() != "Windows":
        return False
    try:
        return not sys.stdout.isatty()
    except (AttributeError, OSError, ValueError):
        return True


def _windows_hidden_subprocess_kwargs() -> dict[str, object]:
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


# Application Control denies the unsigned unsloth.exe, so managed invocations go through the interpreter (#8490). Byte-identical to process.rs and install.ps1.
_WINDOWS_CLI_ENTRYPOINT = (
    "import sys, os; sys.path[:1] = [x for x in sys.path[:1] if getattr(sys.flags, 'safe_path', False) or x not in ('', os.getcwd())]; "
    "sys.argv[0] = 'unsloth'; from unsloth_cli import app; sys.exit(app())"
)

# The trampoline's exact import (find_spec passes for an empty unsloth_cli), with the sys.path[0] scrub a parity test literal_evals.
_MANAGED_CLI_IMPORT_PROBE = (
    "import sys, os; sys.path[:1] = [x for x in sys.path[:1] if getattr(sys.flags, 'safe_path', False) or x not in ('', os.getcwd())]; "
    "from unsloth_cli import app; sys.exit(0)"
)

# Generous: cold interpreter start plus package import. A timeout means "no verdict", not failure.
_MANAGED_CLI_IMPORT_PROBE_TIMEOUT = 60

# ERROR_ACCESS_DISABLED_BY_POLICY, surfaced by Python as OSError.winerror.
_ERROR_ACCESS_DISABLED_BY_POLICY = 1260


def _managed_cli_argv(
    python: Path,
    *args: str,
    isolated: bool = False,
) -> List[str]:
    """argv that runs the managed `unsloth` CLI through *python*. -X utf8 precedes -I because -I
    implies -E. *isolated* mirrors Isolation in process.rs; the default inherits."""
    flags = ["-X", "utf8", "-I"] if isolated else ["-X", "utf8"]
    return [str(python), *flags, "-c", _WINDOWS_CLI_ENTRYPOINT, *args]


def _is_application_control_block(error: OSError) -> bool:
    """True when Windows refused to start a program by policy: nothing ran."""
    return getattr(error, "winerror", None) == _ERROR_ACCESS_DISABLED_BY_POLICY


@contextlib.contextmanager
def _studio_runtime_launch_guard(*, inherited: bool = False, wait: bool = False):
    guard = _studio_runtime_gate.studio_runtime_launch_guard(
        STUDIO_HOME,
        inherited = inherited,
        wait = wait,
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
    """Return *stream* if it has a real OS fd, else None: Windows will not inherit std handles
    unless passed explicitly, and a test-captured stream has no fd."""
    if stream is None:
        return None
    try:
        stream.fileno()
    except (AttributeError, OSError, ValueError):
        return None
    return stream


def _display_host_for_bind(run_mod, host: str) -> str:
    return run_mod._display_host_for_bind(host)


def _network_share_host_for_bind(run_mod, host: str) -> str:
    resolver = getattr(run_mod, "_network_share_host_for_bind", None)
    if resolver is None:
        return _display_host_for_bind(run_mod, host)
    return resolver(host)


def _loopback_bind_host_for(host: str) -> str:
    from unsloth_cli._tool_policy import wildcard_loopback_host
    return wildcard_loopback_host(host) or "127.0.0.1"


def _is_wildcard_bind(host: str) -> bool:
    from unsloth_cli._tool_policy import is_wildcard_host
    return is_wildcard_host(host)


def _openable_host_for_bind(run_mod, host: str) -> str:
    """Host to print in a URL: the LAN address when one resolves, else loopback. A
    wildcard bind must never be printed as-is; no browser can open it."""
    share_host = _network_share_host_for_bind(run_mod, host)
    if _is_wildcard_bind(share_host):
        return _loopback_bind_host_for(host)
    return share_host


def _require_bind_host(host: str) -> None:
    if isinstance(host, str) and host.strip():
        return
    typer.echo(
        "Error: --host cannot be empty; use 0.0.0.0 to bind every IPv4 interface.",
        err = True,
    )
    raise typer.Exit(2)


def _normalize_wildcard_bind_host(host: str) -> str:
    from unsloth_cli._tool_policy import normalize_wildcard_bind_host
    try:
        return normalize_wildcard_bind_host(host)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err = True)
        raise typer.Exit(2) from None


def _require_unambiguous_ephemeral_bind(host: str, port: int) -> None:
    if port != 0:
        return
    from unsloth_cli._tool_policy import resolved_bind_address_count

    if resolved_bind_address_count(host) <= 1:
        return
    typer.echo(
        "Error: --port 0 cannot be used when --host resolves to multiple bind "
        "addresses; choose an explicit port.",
        err = True,
    )
    raise typer.Exit(2)


def _url_host(host: str) -> str:
    url_host = host.replace("%", "%25")
    return (
        f"[{url_host}]"
        if ":" in url_host and not (url_host.startswith("[") and url_host.endswith("]"))
        else url_host
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
    if platform.system() == "Windows":
        p = STUDIO_HOME / "unsloth_studio" / "Scripts" / "python.exe"
    else:
        p = STUDIO_HOME / "unsloth_studio" / "bin" / "python"
    return p if p.is_file() else None


def _managed_cli_site_packages_layout(python: Path) -> bool:
    """On-disk hint that the venv still carries the CLI. Weaker than the import probe: an empty
    unsloth_cli/ or an orphaned dist-info passes here."""
    site_packages = python.parent.parent / "Lib" / "site-packages"
    if (site_packages / "unsloth_cli").is_dir():
        return True
    return any(site_packages.glob("unsloth-*.dist-info"))


def _managed_cli_package_present(python: Path) -> bool:
    """Whether the venv holding *python* can still import the package the CLI runs. Windows only,
    and asked of the interpreter, since this gate fronts the .bootstrap_password strip."""
    if platform.system() != "Windows":
        return False
    try:
        probe = subprocess.run(
            [str(python), "-X", "utf8", "-c", _MANAGED_CLI_IMPORT_PROBE],
            capture_output = True,
            timeout = _MANAGED_CLI_IMPORT_PROBE_TIMEOUT,
            # A non-interactive Windows launch must not flash a console window (#8490).
            **_windows_hidden_subprocess_kwargs(),
        )
    except subprocess.TimeoutExpired:
        # Timeout is not failure: the untimed re-exec would still come up, so fall back to the on-disk layout.
        return _managed_cli_site_packages_layout(python)
    except (OSError, subprocess.SubprocessError):
        # The re-exec runs that same interpreter. Fail closed: the caller strips .bootstrap_password first.
        return False
    return probe.returncode == 0


def _hsa_override_gfx_arch(value: Optional[str]) -> Optional[str]:
    """gfx arch named by an HSA_OVERRIDE_GFX_VERSION value, or None if unreadable. Kept in sync
    with _hsa_override_gfx_arch in studio/install_python_stack.py and install.sh."""
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
    """Whether the installed torch actually resolves through the ``rocm`` meta-package: the generic
    pytorch.org wheels vendor their own runtime, so after a switch it is orphaned."""
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
                    # Plain `rocm` and `rocm[libraries,devel]` both count; `rocm-sdk-core` is a component, not the arbiter.
                    if re.search(r"requires-dist:\s*rocm(?![-_a-z0-9])", line, re.IGNORECASE):
                        return True
                return False
    return False


def _installed_rocm_single_arch(venv_dir: Path) -> Optional[str]:
    """gfx arch the ROCm runtime in *venv_dir* ACTIVELY carries kernels for, or None. Read from the
    `rocm` meta-package: globbing for rocm_sdk_libraries_gfx* would read an ORPHAN. None also
    covers a MULTI-arch family such as gfx120x-all, which contradicts no override."""
    # Only a LIVE install while torch resolves through it: the generic pytorch.org wheels orphan `rocm`, which would then name the OLD family.
    if not _torch_requires_rocm_metapackage(venv_dir):
        return None
    _metadata: Optional[Path] = None
    for sp_pattern in ("lib/python*/site-packages", "Lib/site-packages"):
        for sp in venv_dir.glob(sp_pattern):
            for info in sp.glob("rocm-*.dist-info"):
                # rocm-sdk-* also start "rocm-"; only the bare meta-package arbitrates.
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
        return None
    _family = _families.pop()
    # Single ISA only: a gfx120x-all family contradicts no override.
    return _family if re.fullmatch(r"gfx[0-9a-f]+", _family) else None


def _clear_hsa_override_contradicting_install(venv_dir: Path) -> Optional[str]:
    """Drop an HSA_OVERRIDE_GFX_VERSION no installed kernel can satisfy (#7331): against per-gfx
    wheels it fails every launch, and install.sh's unset dies with the installer, so this is the
    chokepoint every path shares. Keyed on the INSTALL, never a hardware probe."""
    raw = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if not raw or platform.system() == "Windows":
        return None
    arch = _installed_rocm_single_arch(venv_dir)
    if arch is None:
        return None
    named = _hsa_override_gfx_arch(raw)
    if named is None or named == arch:
        return None
    os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)
    return arch


def _clear_hsa_override_before_launch(silent: bool = False) -> Optional[str]:
    """Run the #7331 spoof clear for whichever entry point is about to launch. Idempotent."""
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
    run_py = _PACKAGE_ROOT / "studio" / "backend" / "run.py"
    if run_py.is_file():
        return run_py
    for pattern in (
        "lib/python*/site-packages/studio/backend/run.py",
        "Lib/site-packages/studio/backend/run.py",
    ):
        for match in (STUDIO_HOME / "unsloth_studio").glob(pattern):
            return match
    return None


def _install_state(deep: bool = False) -> dict:
    """verify_install() result for this root; STUDIO_HOME is an extra search root so a CLI outside
    the managed venv still inspects the venv the desktop app launches."""
    return _studio_deps.install_state(
        extra_roots = (STUDIO_HOME / "unsloth_studio",),
        deep = deep,
    )


_RUN_MODULE = None


def _load_run_module():
    """Import studio.backend.run by file path: `studio update` can leave a partial
    site-packages/studio/backend/ tree that shadows an editable install."""
    global _RUN_MODULE
    if _RUN_MODULE is not None:
        return _RUN_MODULE

    run_py = _find_run_py()
    if run_py is None:
        raise ImportError("Could not find studio/backend/run.py. Re-run: unsloth studio setup")

    loaded = sys.modules.get("studio.backend.run")
    if loaded is not None:
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
    """Find studio/setup.sh or studio/setup.ps1, from any CWD. *repo_root* is the --local checkout
    and its script must win, since the editable install removes the tree the installed one builds into."""
    name = "setup.ps1" if platform.system() == "Windows" else "setup.sh"
    # No fallback: reaching for the installed copy's script is what this branch prevents.
    if repo_root is not None:
        s = repo_root / "studio" / name
        return s if s.is_file() else None
    s = _PACKAGE_ROOT / "studio" / name
    if s.is_file():
        return s
    for pattern in (
        f"lib/python*/site-packages/studio/{name}",
        f"Lib/site-packages/studio/{name}",
    ):
        for match in (STUDIO_HOME / "unsloth_studio").glob(pattern):
            return match
    return None


# Mirrored in studio/backend/run.py argparse and the backend denylist test; bump both.
_PARALLEL_MIN = 1
_PARALLEL_MAX = 64
_PARALLEL_DEFAULT_RUN = 4
# At 1 every extra chat queues behind the generating one. _slots_that_fit_on_gpu() may cut it back.
_PARALLEL_DEFAULT_PLAIN = 4


def _resolve_secure(secure: bool, not_secure: bool) -> bool:
    """Reconcile the deprecated --not-secure alias with --secure/--no-secure. Typer parses them as
    independent options, so last-wins ordering is restored from argv to match the backend."""
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
    """Yield repo roots from setuptools `__editable___*_finder.py` MAPPINGs naming `studio`, as the
    parent of the mapped package."""
    import ast
    import re

    for sp_pattern in ("lib/python*/site-packages", "Lib/site-packages"):
        for sp in venv_dir.glob(sp_pattern):
            for finder in sp.glob("__editable___*_finder.py"):
                try:
                    src = finder.read_text(encoding = "utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                # [^}]* still rejects nested dicts, which the setuptools template never emits.
                m = re.search(r"^MAPPING\s*(?::[^=]*)?=\s*(\{[^}]*\})", src, re.M | re.S)
                if not m:
                    continue
                try:
                    mapping = ast.literal_eval(m.group(1))
                except (SyntaxError, ValueError):
                    continue
                # literal_eval can return a set / list / None: the regex only captures `{...}`.
                if not isinstance(mapping, dict):
                    continue
                studio_pkg = mapping.get("studio")
                if studio_pkg:
                    yield Path(studio_pkg).parent


def _find_frontend_dist() -> Optional[Path]:
    """Locate a built `studio/frontend/dist`, or None so callers can fall back to --api-only.
    Editable source roots are probed too, so a shadowing `unsloth` on PATH cannot win."""
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


_direct_http_opener = None


def _direct_urlopen(request, timeout):
    global _direct_http_opener

    if _direct_http_opener is None:

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                raise urllib.error.HTTPError(
                    req.full_url, code, f"refusing redirect to {newurl}", headers, fp
                )

        _direct_http_opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirect(),
        )
    return _direct_http_opener.open(request, timeout = timeout)


def _wait_for_server(
    port: int,
    timeout: int = 30,
    request_host: str = "127.0.0.1",
) -> bool:
    import urllib.request
    import urllib.error

    url = f"http://{_url_host(request_host)}:{port}/api/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with _direct_urlopen(url, timeout = 2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError, ConnectionError):
            pass
        time.sleep(0.5)
    return False


def _cli_api_key_secret_path(name: str) -> Path:
    """Cache path for the raw API key named *name*.

    Identity is the digest; the stem is only so a human can tell the files apart.
    Sharing a stem would hand one label's credential to another (`foo/bar` vs
    `foo?bar`, past the 64-char cut, `cli` vs `CLI` on APFS/NTFS). ASCII-only
    keeps 64 chars at 64 bytes: NAME_MAX is 255 BYTES, so multibyte alnums made a
    282-byte name that failed to cache and re-minted every launch (#10595 again).
    """
    safe = "".join(
        ch if (ch.isascii() and ch.isalnum()) or ch in "-_" else "_" for ch in name
    ).strip("_")
    if not safe:
        safe = "cli"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return STUDIO_HOME / "auth" / f"{CLI_API_KEY_FILE_PREFIX}{safe[:64]}_{digest}"


def _read_cli_api_key_secret(name: str) -> str:
    try:
        return _cli_api_key_secret_path(name).read_text(encoding = "utf-8").strip()
    except (OSError, ValueError):
        return ""


def _create_api_key_inprocess(name: str) -> str:
    """Return a raw API key for *name*, minting only when the cached one is dead.

    Uses a direct storage call, bypassing the ``must_change_password`` gate that
    blocks HTTP POST /api/auth/api-keys on fresh installs."""
    storage = _load_backend_auth_storage()
    cached = _read_cli_api_key_secret(name)
    if cached and storage.validate_api_key_with_credential(cached, touch = False):
        return cached

    raw_key, _row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME,
        name = name,
    )
    # Best-effort: the key is already committed and the caller shuts the server
    # down on any exception, so raising here would kill a healthy launch and
    # re-mint on every retry. Same trade-off as start.py's _write_private_json.
    try:
        _write_auth_secret(_cli_api_key_secret_path(name), raw_key)
    except OSError as exc:
        typer.echo(
            f"Warning: could not cache the {name} API key ({exc}); this launch is "
            "unaffected, but the next one will create another key.",
            err = True,
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
        # newline pins LF: text mode writes CRLF and `$(cat ...)` leaves the CR on the credential.
        with os.fdopen(fd, "w", encoding = "utf-8", newline = "\n") as f:
            fd = -1
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
    # sqlite3.connect makes a new DB 0644 under a 022 umask; keep auth/ and auth.db private.
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


def _launch_publishes_tunnel(
    *, cloudflare: Optional[bool], host: str, secure: bool, api_only: bool
) -> bool:
    """Whether this launch will publish Unsloth through the Cloudflare tunnel. Kept separate from
    the wider _should_prompt_password_change, whose guards exist because a headless tunnel launch
    strips .bootstrap_password."""
    if secure:
        return True
    if cloudflare is not True:
        return False
    from unsloth_cli._tool_policy import is_wildcard_host

    return is_wildcard_host(host) and not api_only


def _bind_is_wildcard(host: str) -> bool:
    from unsloth_cli._tool_policy import is_wildcard_host
    return is_wildcard_host(host)


def _should_prompt_password_change(
    *, cloudflare: Optional[bool], host: str, secure: bool, api_only: bool
) -> bool:
    """Whether this launch puts Unsloth where someone else can reach it. is_external_host, NOT
    is_wildcard_host: the narrower test let an ordinary LAN bind reach an older child that served
    the seeded password. A raw bind counts only with a terminal attached, since downstream deletes
    .bootstrap_password and would break -H 0.0.0.0 containers."""
    if secure:
        return True
    if not host or api_only:
        return False
    from unsloth_cli._tool_policy import is_external_host, is_wildcard_host

    if cloudflare is True and is_wildcard_host(host):
        # A tunnel prompts regardless of the terminal; headless is handled downstream.
        return True
    return is_external_host(host) and _prompt_streams_interactive() and _prompt_owns_the_terminal()


def _prompt_streams_interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


# Wait for the FIRST keystroke only. 30s stays inside a default Docker HEALTHCHECK start period.
_UNATTENDED_PROMPT_SECONDS = 30.0


def _prompt_owns_the_terminal() -> bool:
    """Whether this process may DRIVE the terminal, not merely see one: a backgrounded job passes
    isatty() but read_masked's tcsetattr SIGTTOUs it and freezes the launch. True on any doubt."""
    try:
        return os.tcgetpgrp(sys.stdin.fileno()) == os.getpgrp()
    except (AttributeError, OSError, ValueError):
        return True


def _bootstrap_deadline_active() -> bool:
    """Whether the backend's bootstrap shutdown deadline will arm. Mirror of
    studio/backend/auth/bootstrap_timeout.py: malformed falls back to the 1h default, 0 disables."""
    raw = os.environ.get("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", "").strip()
    if not raw:
        return True
    try:
        return int(raw) > 0
    except ValueError:
        return True


# Set by the parent after an unanswered prompt so the re-exec'd child does not repeat the wait.
_UNATTENDED_PROMPT_DONE_ENV = "UNSLOTH_STUDIO_UNATTENDED_PROMPT_DONE"


def _deadline_sentence() -> str:
    """Say what will actually happen: BOOTSTRAP_TIMEOUT=0 arms no deadline, and promising one an
    operator acts on is worse than saying nothing."""
    if _bootstrap_deadline_active():
        return (
            "Unsloth shuts down after the bootstrap deadline "
            "(UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT, default 1h) unless the password "
            "is changed."
        )
    return (
        "The bootstrap shutdown deadline is DISABLED for this launch "
        "(UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT=0), so nothing will stop it serving "
        "that credential."
    )


def _generate_reset_password() -> str:
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
) -> None:
    """CLI mirror of backend update_password + change-password effects, in one transaction. File
    cleanup runs after commit, so it cannot roll back."""
    password_salt, password_hash = _hash_password(new_password)
    with conn:
        conn.execute(
            """
            UPDATE auth_user
            SET password_salt = ?, password_hash = ?, jwt_secret = ?, must_change_password = 0
            WHERE username = ?
            """,
            (password_salt, password_hash, secrets.token_urlsafe(64), username),
        )
        conn.execute("DELETE FROM refresh_tokens WHERE username = ?", (username,))
        conn.execute(
            "DELETE FROM app_secrets WHERE key IN (?, ?)",
            (DESKTOP_SECRET_HASH_KEY, DESKTOP_SECRET_CREATED_AT_KEY),
        )
        if revoke_api_keys:
            conn.execute("DELETE FROM api_keys")
    stale_files = [BOOTSTRAP_PASSWORD_FILE, DESKTOP_SECRET_FILE]
    if revoke_api_keys:
        # Reset only: the rows are gone, so each cached key is now plaintext for a
        # dead credential. An ordinary change keeps the rows, so its cache stays valid.
        try:
            stale_files += sorted(
                p.name for p in (STUDIO_HOME / "auth").glob(f"{CLI_API_KEY_FILE_PREFIX}*")
            )
        except OSError:
            pass
    for stale in stale_files:
        stale_path = STUDIO_HOME / "auth" / stale
        try:
            stale_path.unlink(missing_ok = True)
        except OSError as exc:
            # The hash is committed, so a failed unlink must not roll back, but a locked-yet-writable file must be truncated or its plaintext re-validates the credential.
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


def _apply_supplied_password_before_launch(supplied_password: "str | None") -> None:
    """Non-interactively set the INITIAL admin password before the server binds. Only ever the
    FIRST one: an override would be an auth bypass, and the parent runs it so the secret never
    crosses to child argv."""
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
        # Fail closed on any DB failure (typer.Exit from the branches above propagates).
        typer.echo(
            f"Error: --password could not update the Unsloth auth database ({exc}); not starting.",
            err = True,
        )
        raise typer.Exit(1)
    finally:
        conn.close()


def _strip_seeded_bootstrap_password_or_exit(*, context: str) -> None:
    """Remove the seeded plaintext bootstrap password before a public re-exec, so a child of ANY
    version reads None. Removal IS the protection, so a failed removal fails closed."""
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
    """Fail closed BEFORE the pre-exposure gate if a public UI launch has no login page: the gate
    strips .bootstrap_password, and the login page is the only in-band way to change it."""
    if api_only or not _launch_publishes_tunnel(
        cloudflare = cloudflare, host = host, secure = secure, api_only = api_only
    ):
        return frontend
    if frontend is not None:
        # A user-supplied dist is not vetted, so `--frontend /bad/path` would otherwise bypass the guard.
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
    """In-venv analogue of the re-exec launcher check: import the backend before the gate strips
    .bootstrap_password. Headless only, so no prompt waits on the import."""
    if not _launch_publishes_tunnel(
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
    """True only if cloudflared is provably unavailable, so --secure can keep the seeded recovery
    password. False on ANY uncertainty: a leak outweighs a recoverable lockout."""
    run_py = _find_run_py()
    if run_py is None:
        return False
    backend_dir = run_py.parent
    tunnel_py = backend_dir / "cloudflare_tunnel.py"
    if not tunnel_py.is_file():
        return False
    # ensure_cloudflared() imports utils.paths lazily; without studio/backend on sys.path it returns a false "unavailable" that wrongly refuses --secure.
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
    """True when the child serving Unsloth is provably THIS install's backend, which suppresses the
    seeded credential, so the strip can be skipped. False on ANY doubt."""
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
    """Force a terminal password change before the first public (tunnel) exposure. Committing in the
    parent keeps it off argv/env; without a terminal, fall back to the bootstrap shutdown timer."""
    if not _should_prompt_password_change(
        cloudflare = cloudflare, host = host, secure = secure, api_only = api_only
    ):
        return
    # Use the real predicate, not `cloudflare is True`: a non-secure tunnel only starts for a wildcard host.
    tunnel_will_start = _launch_publishes_tunnel(
        cloudflare = cloudflare, host = host, secure = secure, api_only = api_only
    )
    if not tunnel_will_start and os.environ.get(_UNATTENDED_PROMPT_DONE_ENV):
        # The outer CLI already waited out this terminal. Peeked, never popped (run.py consumes it); never for a tunnel, which fails closed.
        return
    if tunnel_will_start:
        exposure = "on a public Cloudflare URL"
    elif _bind_is_wildcard(host):
        exposure = "on every network interface"
    else:
        # A concrete bind listens on that address only, so "every network interface" is untrue.
        exposure = f"at {host}, which other machines on the network can reach"
    # Before public exposure we must PROVE the password is not the seeded default; an old child could regenerate one. Unprovable fails closed.
    try:
        conn = _connect_auth_db()
    except (OSError, sqlite3.Error) as exc:
        # Cannot confirm a committed admin exists; a transient lock clears on retry.
        typer.echo(
            f"Error: refusing to expose Unsloth {exposure}: could "
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
            # Persist a freshly seeded admin before any re-exec: uncommitted it rolls back on close and an OLD child would regenerate its own.
            conn.commit()
        except (OSError, sqlite3.Error) as exc:
            # Best-effort remove any half-written seed file; the launch is refused regardless.
            try:
                (STUDIO_HOME / "auth" / BOOTSTRAP_PASSWORD_FILE).unlink(missing_ok = True)
            except OSError:
                pass
            typer.echo(
                f"Error: refusing to expose Unsloth {exposure}: could "
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
                # The child suppresses the injection, so nothing serves the seeded credential.
                return
            # The admin is committed, so an old child will not regenerate; strip anyway and fail closed.
            typer.echo(
                f"Warning: could not read the Unsloth admin state back ({exc}); "
                "removing the seeded bootstrap password before public exposure.",
                err = True,
            )
            _strip_seeded_bootstrap_password_or_exit(context = "auth DB row unreadable")
            return
        if not row or not row[2]:
            return
        if not _prompt_streams_interactive():
            # Only proceed headless if the bootstrap deadline protects the launch: it never arms for api-only, and TIMEOUT=0 disables it.
            if api_only or not _bootstrap_deadline_active():
                typer.echo(
                    "Error: refusing to publish Unsloth on a public Cloudflare "
                    "URL: the default admin password was never changed, no "
                    "terminal is attached to change it here, and the bootstrap "
                    "shutdown deadline does not apply to this launch (api-only, "
                    "or UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT=0). Change the "
                    "password first (run `unsloth studio` locally and log in, "
                    "or re-run with a terminal attached), then retry.",
                    err = True,
                )
                raise typer.Exit(1)
            if child_self_suppresses:
                # The child is this install's own backend and never serves the seeded credential publicly, so skip the strip and keep the file for LOCAL recovery.
                typer.echo(
                    "Warning: Unsloth is being exposed publicly while the admin "
                    "account still uses its auto-generated bootstrap password. The "
                    "login page forces a change and the credential is never served "
                    "on the public page. Set a new password by running `unsloth "
                    "studio` locally with a terminal attached, or `unsloth studio "
                    "reset-password`; Unsloth shuts down after ~1h if the password "
                    "stays unchanged (UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT).",
                    err = True,
                )
                return
            # On --secure the bind is loopback, so with cloudflared provably unavailable stripping the only recovery credential would just lock the user out.
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
            # An OLD studio-venv child would serve the seeded credential from disk, so delete it here in the parent; must_change_password stays set.
            _strip_seeded_bootstrap_password_or_exit(context = "no terminal to change it")
            typer.echo(
                "Warning: Unsloth is being exposed publicly while the admin account "
                "still uses its auto-generated bootstrap password. The seeded password "
                "file has been removed so it is not served on the public page. Set a new "
                "password by running `unsloth studio` locally with a terminal attached, "
                "or `unsloth studio reset-password`; Unsloth shuts down after ~1h if the "
                "password stays unchanged (UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT).",
                err = True,
            )
            return
        password_salt, password_hash = row[0], row[1]

        def _is_current_password(candidate: str) -> bool:
            return hmac.compare_digest(
                _pbkdf2_hex(candidate, password_salt.encode("utf-8")), password_hash
            )

        # Ctrl+C aborts a TUNNEL launch only; on a raw bind it declines the prompt, because that launch worked before this gate existed.
        refusal = (
            "Ctrl+C to abort."
            if tunnel_will_start
            else "Ctrl+C to skip, and Unsloth starts with the auto-generated password."
        )
        typer.echo(
            f"Unsloth Studio will be reachable {exposure}, so set a password now. {refusal}",
            err = True,
        )
        try:
            new_password = _password_prompt.prompt_new_password(
                _is_current_password,
                # A raw bind must never block a launch that used to start: a detached pty passes every isatty test yet nobody types. A tunnel fails closed.
                first_key_timeout = None if tunnel_will_start else _UNATTENDED_PROMPT_SECONDS,
            )
        except _password_prompt.PromptUnattended:
            typer.echo(
                "Warning: no response at the terminal, so Unsloth is starting with "
                "the auto-generated admin password on a bind that is reachable from "
                f"the network. {_deadline_sentence()} Change it by logging in, or "
                "with `unsloth studio reset-password`.",
                err = True,
            )
            # Tell the child the terminal has been tried, or it waits its own deadline on the same pty and can trip a startup watchdog.
            os.environ[_UNATTENDED_PROMPT_DONE_ENV] = "1"
            return
        except (KeyboardInterrupt, EOFError):
            if tunnel_will_start:
                typer.echo(
                    "\nError: password change aborted; refusing to publish Unsloth "
                    "on a public URL with the default admin password. Re-run and "
                    "set a password, or launch without --secure/--cloudflare.",
                    err = True,
                )
                raise typer.Exit(1)
            # A raw bind is not a publication, so Ctrl+C returns it to pre-prompt behaviour, as run.py's gate does.
            typer.echo(
                "\nWarning: password change aborted, so Unsloth is starting with "
                "the auto-generated admin password on a bind that is reachable "
                f"from the network. {_deadline_sentence()} Change it by logging "
                "in, with `unsloth studio reset-password`, or by passing "
                "--password / UNSLOTH_STUDIO_PASSWORD.",
                err = True,
            )
            os.environ[_UNATTENDED_PROMPT_DONE_ENV] = "1"
            return
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
    request_host: str = "127.0.0.1",
) -> dict:
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
    url = f"http://{_url_host(request_host)}:{port}/api/inference/load"
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
        with _direct_urlopen(req, timeout = timeout) as resp:
            try:
                body = json.loads(resp.read())
            except ValueError:
                body = None
        # A slow load commits its 200 early and pads the body, so a late failure arrives in-band.
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
    # --not-secure is a deprecated alias for --no-secure.
    secure = _resolve_secure(secure, not_secure)
    _ensure_studio_env_exported()
    if ctx.invoked_subcommand is not None:
        # Typer does not forward parent options, so `unsloth studio --parallel N run` would drop N.
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
        if secure:
            typer.echo(
                f"Error: --secure on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio "
                f"{ctx.invoked_subcommand}`, put it after the subcommand: "
                f"`unsloth studio {ctx.invoked_subcommand} --secure ...`",
                err = True,
            )
            raise typer.Exit(2)
        if verbose:
            typer.echo(
                f"Error: --verbose on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio "
                f"{ctx.invoked_subcommand}`, put it after the subcommand: "
                f"`unsloth studio {ctx.invoked_subcommand} --verbose ...`",
                err = True,
            )
            raise typer.Exit(2)
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
        if api_only:
            typer.echo(
                f"Error: --api-only on `unsloth studio` applies to the "
                f"plain-server path only. For `unsloth studio "
                f"{ctx.invoked_subcommand}`, put it after the subcommand: "
                f"`unsloth studio {ctx.invoked_subcommand} --api-only ...`",
                err = True,
            )
            raise typer.Exit(2)
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

    _require_bind_host(host)
    runtime_gate_handoff = _studio_runtime_gate.consume_runtime_gate_handoff()
    runtime_gate_acquire = _studio_runtime_gate.consume_runtime_gate_acquire()
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

    host = _normalize_wildcard_bind_host(host)
    _require_unambiguous_ephemeral_bind(host, port)

    # --verbose restores the per-request access logs suppressed by default.
    if verbose:
        _enable_verbose_access_logs()
    if disable_dns_pinning:
        os.environ["UNSLOTH_STUDIO_DISABLE_DNS_PINNING"] = "1"
    else:
        os.environ.setdefault("UNSLOTH_STUDIO_DISABLE_DNS_PINNING", "0")

    # Resolve the child launcher BEFORE the gate: a headless gate strips the seeded password, so aborting afterwards leaves no way to log in.
    studio_venv_dir = STUDIO_HOME / "unsloth_studio"
    in_studio_venv = sys.prefix.startswith(str(studio_venv_dir))
    # Before the env reaches a child: an override contradicting single-arch wheels fails every kernel launch, and install.sh's unset cannot reach here (#7331).
    _clear_hsa_override_before_launch(silent = silent)
    studio_python = run_py = None
    resolved_frontend = frontend
    if not in_studio_venv:
        studio_python = _studio_venv_python()
        run_py = _find_run_py()
        if not (studio_python and run_py):
            typer.echo("Unsloth Studio not set up. Run install.sh first.")
            raise typer.Exit(1)
        # A public UI launch needs a servable login page before the gate strips the seeded password.
        resolved_frontend = _require_servable_frontend_or_exit(
            frontend = resolved_frontend,
            api_only = api_only,
            cloudflare = cloudflare,
            host = host,
            secure = secure,
        )
        # Non-public / api-only launches still forward a resolved dist, for the same silent 404.
        if resolved_frontend is None and not api_only:
            resolved_frontend = _find_frontend_dist()
    else:
        # In the studio venv there is no re-exec: validate frontend and backend BEFORE the headless gate strips the seeded password.
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

    _apply_supplied_password_before_launch(_password_prompt.resolve_supplied_password(password))
    os.environ.pop(_password_prompt.SUPPLIED_PASSWORD_ENV, None)

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
            if resolved_frontend is not None:
                args.extend(["--frontend", str(resolved_frontend)])
            if silent:
                args.append("--silent")
            if api_only:
                args.append("--api-only")
            # Explicit polarity: an older run.py defaults --cloudflare on, so an unset default must not re-enable the tunnel on a mixed install.
            if cloudflare is True:
                args.append("--cloudflare")
            elif not secure:
                args.append("--no-cloudflare")
            args.append("--secure" if secure else "--no-secure")
            if enable_tools is True:
                args.append("--enable-tools")
            elif enable_tools is False:
                args.append("--disable-tools")
            # On Windows os.execvp keeps the parent alive and Ctrl+C would orphan the child.
            if sys.platform == "win32":
                import subprocess as _sp

                # Without our std handles, CREATE_NO_WINDOW gives the backend a hidden console and `unsloth studio > log` captures nothing.
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

    with _studio_runtime_launch_guard(
        inherited = runtime_gate_handoff,
        wait = runtime_gate_acquire,
    ):
        with _studio_deps.studio_backend_imports("unsloth studio"):
            run_mod = _load_run_module()
        run_server = run_mod.run_server

        if not silent:
            launch_host = _openable_host_for_bind(run_mod, host)
            typer.echo(f"Starting Unsloth Studio on http://{_url_host(launch_host)}:{port}")

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
        if resolved_frontend is not None:
            run_kwargs["frontend_path"] = resolved_frontend
        run_server(**run_kwargs)

    try:
        if run_mod._shutdown_event is not None:
            # Event.wait() with no timeout blocks at C level on Linux and swallows SIGINT.
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


def _split_repo_variant(model_arg: str) -> tuple[str, Optional[str]]:
    """Split ``org/name:variant`` into ``(repo, variant)``; mirrors llama.cpp's
    ``-hf <repo>:<quant>``. Local paths, Windows drives and ids without ``:`` pass through."""
    s = model_arg.strip()
    if not s:
        return s, None
    if s.startswith(("/", "./", "../", "~")) or s == ".":
        return s, None
    # Windows drive letter: the colon is a path separator.
    if len(s) >= 2 and s[1] == ":" and s[0].isalpha():
        return s, None
    if ":" not in s:
        return s, None
    repo, _, variant = s.rpartition(":")
    if not repo or not variant:
        return s, None
    # Quant labels never contain a slash; `foo:bar/baz` is not repo:variant.
    if "/" in variant:
        return s, None
    return repo, variant


def _expand_attached_np_short() -> None:
    # Click clusters `-np8` as `-n -p 8`, dropping the value. Lockstep with the backend `_flag_name`.
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
    """Pop exact-match legacy shorts (`-m`/`-hfr`/`-f`) from args, leaving clusters
    (`-mg`/`-fa`) for the llama-server tail. Inline `-x=value` accepted."""
    out: List[str] = []
    value = current
    i, n = 0, len(args)
    while i < n:
        tok = args[i]
        if tok == "--":
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
            if inline == "":
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
        # `-m` / `-hfr` removed (Click clusters `-mg`/`-md`); the legacy shim still takes exact matches.
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
        help = "Label for the API key reused across runs",
    ),
    port: int = typer.Option(8888, "--port", "-p", rich_help_panel = _RUN_PANEL_SERVER),
    host: str = typer.Option("127.0.0.1", "--host", "-H", rich_help_panel = _RUN_PANEL_SERVER),
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
    # Passed via env, so an older re-exec target ignores it instead of treating it as a llama-server arg.
    inherited_start_api_key_marker = _consume_start_api_key_marker_env()
    start_api_key_marker = start_api_key_marker or inherited_start_api_key_marker
    runtime_gate_handoff = _studio_runtime_gate.consume_runtime_gate_handoff()
    # The group callback returns before its own clear once a subcommand is named, so this path must clear the override itself (#7331).
    _clear_hsa_override_before_launch(silent = bool(silent))

    secure = _resolve_secure(secure, not_secure)
    _preserve_cloudflare_intent(cloudflare, secure)
    extra_llama_args: List[str] = list(ctx.args) if ctx.args else []

    # Read from the env at backend import, so resolve before any re-exec; an omitted flag keeps a value the parent forwarded.
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

    # UNSLOTH_SAMPLING_* hard-pins a field over client and per-model recommendation, so only write flags set explicitly.
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

    # Set before any re-exec. --log-verbose keeps llama-server's own -v passthrough working.
    if verbose:
        _enable_verbose_access_logs()
        if not any(a in ("--verbose", "-v", "--log-verbose") for a in extra_llama_args):
            extra_llama_args.append("--log-verbose")
    if disable_dns_pinning:
        os.environ["UNSLOTH_STUDIO_DISABLE_DNS_PINNING"] = "1"
    else:
        os.environ.setdefault("UNSLOTH_STUDIO_DISABLE_DNS_PINNING", "0")

    # Promote legacy exact `-m`/`-hfr`/`-f` back into typer params; clusters stay in extras.
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

    # Parse llama.cpp `repo:variant`; error if it disagrees with --gguf-variant.
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

    _require_bind_host(host)

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

    host = _normalize_wildcard_bind_host(host)
    _require_unambiguous_ephemeral_bind(host, port)

    # Tool policy does not depend on the bind. None applies the default without becoming an override, so a request's enable_tools: false is honored.
    from unsloth_cli._tool_policy import is_external_host, resolve_tool_policy

    enable_tools = resolve_tool_policy(
        host = host,
        flag = enable_tools,
        yes = yes,
        silent = silent,
    )

    studio_venv_dir = STUDIO_HOME / "unsloth_studio"
    in_studio_venv = sys.prefix.startswith(str(studio_venv_dir))
    studio_bin = None
    resolved_frontend = frontend
    if not in_studio_venv:
        studio_python = _studio_venv_python()
        if not studio_python:
            typer.echo("Unsloth Studio not set up. Run install.sh first.")
            raise typer.Exit(1)
        # Re-exec via the studio venv's console script. On Windows quarantine deletes the stub from a working venv, so the package answers for it.
        studio_bin = studio_python.parent / (
            "unsloth.exe" if platform.system() == "Windows" else "unsloth"
        )
        if not studio_bin.is_file() and not _managed_cli_package_present(studio_python):
            typer.echo("Unsloth venv missing 'unsloth' entry point. Re-run: unsloth studio setup")
            raise typer.Exit(1)
        resolved_frontend = _require_servable_frontend_or_exit(
            frontend = frontend,
            api_only = api_only,
            cloudflare = cloudflare,
            host = host,
            secure = secure,
        )
    else:
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

    _apply_supplied_password_before_launch(_password_prompt.resolve_supplied_password(password))
    os.environ.pop(_password_prompt.SUPPLIED_PASSWORD_ENV, None)

    # Before any re-exec or server exists. This re-exec runs a possibly-OLD console script, so it is NOT provably self-suppressing.
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
        # Application Control blocks the generated unsloth.exe on some machines but not the signed python.exe beside it.
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
        # Explicit polarity: a future default flip on one layer must not invert the other.
        args.append("--load-in-4bit" if load_in_4bit else "--no-load-in-4bit")
        # Not just a user-supplied dist: the parent may have found a build the shadowed child cannot, and the child would abort (lockout).
        if resolved_frontend is not None:
            args.extend(["--frontend", str(resolved_frontend)])
        if api_only:
            args.append("--api-only")
        if silent:
            args.append("--silent")
        if enable_tools is True:
            args.append("--enable-tools")
        elif enable_tools is False:
            args.append("--disable-tools")
        if yes:
            args.append("--yes")
        # Typer claims --parallel outside ctx.args; without this the child reverts to its default.
        args.extend(["--parallel", str(parallel)])
        # Explicit polarity: a mixed-version studio venv whose old default was cloudflare-on must not re-enable the tunnel.
        if cloudflare is True:
            args.append("--cloudflare")
        elif not secure:
            args.append("--no-cloudflare")
        args.append("--secure" if secure else "--no-secure")
        args.append("--tensor-parallel" if tensor_parallel else "--no-tensor-parallel")
        if verbose:
            args.append("--verbose")
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
            os.environ.pop(_START_API_KEY_MARKER_ENV, None)

    with _studio_deps.studio_backend_imports("unsloth studio"):
        run_mod = _load_run_module()
    run_server = run_mod.run_server

    # Match the route handlers' import path and set it before uvicorn binds; run_server() applies the same pair, idempotently.
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
        emit_tauri_port = False,
        abort_if_own_studio = False,
    )
    if resolved_frontend is not None:
        run_kwargs["frontend_path"] = resolved_frontend
    with _studio_runtime_launch_guard(inherited = runtime_gate_handoff):
        app = run_server(**run_kwargs)
    actual_port = getattr(app.state, "server_port", port) or port

    from studio.backend.run import _graceful_shutdown, _server

    try:
        request_host = getattr(app.state, "server_request_host", None)
        if not isinstance(request_host, str) or not request_host:
            typer.echo("Error: server did not expose its bound address.", err = True)
            raise typer.Exit(1)
        if not silent:
            typer.echo("Starting Unsloth Studio...")
        if not _wait_for_server(actual_port, request_host = request_host):
            typer.echo("Error: server did not become healthy within 30 seconds.", err = True)
            raise typer.Exit(1)

        api_key = _create_api_key_inprocess(api_key_name)
        if start_api_key_marker:
            typer.echo(f"UNSLOTH_START_API_KEY: {api_key}")

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
                request_host = request_host,
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

    display_host = _display_host_for_bind(run_mod, host)
    base_host = _openable_host_for_bind(run_mod, host)
    base_url = f"http://{_url_host(base_host)}:{actual_port}"
    sdk_base_url = f"{base_url}/v1"
    _cf_url = getattr(app.state, "cloudflare_url", None)
    if secure and _cf_url:
        sdk_base_url = f"{_cf_url}/v1"

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


_PID_FILE = STUDIO_HOME / "studio.pid"
PID_FILE_GLOB = "studio-*.pid"


def _pid_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` exists. os.kill(pid, 0) raises WinError 87 for
    every pid on Windows, so use tasklist there."""
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/NH", "/FO", "CSV"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
            ).stdout
        except Exception:
            # Cannot determine; assume alive, taskkill no-ops if already gone.
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
    try:
        text = path.read_text(encoding = "utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return _parse_pid_record(text)


def _unlink_quietly(path: Path) -> None:
    """Drop a record without letting one bad file end the loop: an undeletable record must
    not stop us reaching the other servers."""
    try:
        path.unlink(missing_ok = True)
    except OSError as e:
        typer.echo(f"Could not remove PID file {path.name}: {e}", err = True)


def _report_unreadable(paths: "list[Path]") -> None:
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
    """(pid, create_times, files) per recorded server, including the legacy studio.pid. Grouped by
    PID: a second SIGTERM would hard-kill a server mid-shutdown. Every recorded time is kept, so a
    stale file cannot veto a live server."""
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
            # Unreadable is not invalid: a root-owned or mid-write record still belongs to a live server.
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
    """False only when a recorded start time proves this PID is a different process. Untimed records
    are trusted and must not cancel a timed one: every server writes both."""
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
            # Reporting success would be a lie: the servers behind the unreadable records are still serving.
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


def _wait_for_windows_setup_process(process) -> int:
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
            # taskkill interrupted or unavailable: hold the gate until setup exits naturally.
            pass
        while process.poll() is None:
            try:
                process.wait()
            except KeyboardInterrupt:
                continue
        raise


# -NoProfile drops $PSDefaultParameterValues, which may hold the only proxy out of a corporate host; a standalone update has no installer to hand it over.
_PROXY_PROBE_BEGIN = "<<UNSLOTH_PROXY_DEFAULTS>>"
_PROXY_PROBE_END = "<</UNSLOTH_PROXY_DEFAULTS>>"

_PS_PROXY_PROBE = (
    "$ErrorActionPreference = 'SilentlyContinue'; "
    # PowerShell 5.1 writes redirected output in the console code page, so a non-ASCII proxy value came back mangled but still parsed as JSON.
    "try { [Console]::OutputEncoding = "
    "New-Object System.Text.UTF8Encoding $false } catch { }; "
    "try { $OutputEncoding = [Console]::OutputEncoding } catch { }; "
    "$PSModuleAutoLoadingPreference = 'All'; "
    # Dot-sourced by name, or this process loads the CONSOLEHOST profile. $PROFILE is populated under -NoProfile.
    "$__unslothHostProfileName = $env:_UNSLOTH_PS_HOST_PROFILE; "
    # All-users first, PowerShell's own startup order, so a user profile overrides it by running last.
    "try { $__unslothProfiles = @($PROFILE.AllUsersAllHosts); "
    # ONLY the caller's host profile: other hosts' profiles can print, clobber the table, or exit.
    "if ($__unslothHostProfileName) { "
    "$__unslothProfiles += (Join-Path (Split-Path -Parent $PROFILE.AllUsersCurrentHost) "
    "$__unslothHostProfileName) }; "
    "$__unslothProfiles += $PROFILE.AllUsersCurrentHost; "
    "$__unslothProfiles += $PROFILE.CurrentUserAllHosts; "
    # ADDED, never substituted: TERM_PROGRAM=vscode is set by every VS Code terminal, so substitution missed a plain pwsh terminal's proxy.
    "if ($__unslothHostProfileName) { "
    "$__unslothProfiles += (Join-Path (Split-Path -Parent $PROFILE.CurrentUserCurrentHost) "
    "$__unslothHostProfileName) }; "
    "$__unslothProfiles += $PROFILE.CurrentUserCurrentHost; "
    "foreach ($__unslothProfile in ($__unslothProfiles | Select-Object -Unique)) { "
    "if ($__unslothProfile -and (Test-Path -LiteralPath $__unslothProfile -PathType Leaf)) { "
    "try { . $__unslothProfile } catch { } } } } catch { }; "
    # Re-pinned: a profile may set [Console]::OutputEncoding, and the parent decodes UTF-8.
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
    # A script block is the supported DYNAMIC default; serialize the RESULT, since executable code must not cross the handoff.
    "elseif ($v -is [scriptblock]) { try { $r = & $v; "
    "if ($r -is [uri]) { $out[$k] = $r.AbsoluteUri } "
    "elseif ($r -is [string] -or $r -is [bool]) { $out[$k] = $r } } catch { } } } }; "
    # $out holds copies, and ConvertTo-Json:AsArray = $true would make this record an array.
    "$PSDefaultParameterValues = @{}; "
    # FRAMED and module-qualified: a profile banner made the parse throw, and an alias could reshape the frame.
    f"if ($out.Count -gt 0) {{ "
    f"Microsoft.PowerShell.Utility\\Write-Output '{_PROXY_PROBE_BEGIN}'; "
    f"$out | Microsoft.PowerShell.Utility\\ConvertTo-Json -Compress; "
    f"Microsoft.PowerShell.Utility\\Write-Output '{_PROXY_PROBE_END}' }}"
)


# What the CALLER's host names its own profile; an unidentifiable host gets none.
_HOST_PROFILE_BY_TERM_PROGRAM = {"vscode": "Microsoft.VSCode_profile.ps1"}


def _caller_host_profile_name() -> Optional[str]:
    term_program = (os.environ.get("TERM_PROGRAM") or "").strip().casefold()
    return _HOST_PROFILE_BY_TERM_PROGRAM.get(term_program)


_WINDOWS_PS_MODULE_DIR = r"System32\WindowsPowerShell\v1.0\Modules"
_WINDOWS_PS_HOSTS = frozenset({"powershell.exe", "powershell", "powershell_ise.exe"})


def _fold_module_entry(entry: str) -> str:
    return entry.replace("/", "\\").rstrip("\\").casefold()


def _windows_powershell_module_path(current: str) -> Optional[str]:
    """``current`` with Windows PowerShell's own module directory first, or None to leave it be: a
    5.1 child started from Python would otherwise resolve its modules to the PowerShell 7 copies."""
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
        # os.environ upper-cases keys on Windows, so read PSMODULEPATH or a second, case-differing entry appears.
        key = next((k for k in env if k.upper() == "PSMODULEPATH"), "PSModulePath")
        reordered = _windows_powershell_module_path(env.get(key, ""))
        if reordered:
            env[key] = reordered
    return env


def _framed_probe_record(stdout: str) -> Optional[str]:
    start = stdout.find(_PROXY_PROBE_BEGIN)
    if start < 0:
        return None
    start += len(_PROXY_PROBE_BEGIN)
    end = stdout.find(_PROXY_PROBE_END, start)
    if end < 0:
        return None
    return stdout[start:end].strip() or None


def _profile_probe_hosts() -> list[str]:
    r"""The PowerShell hosts whose profile to ask, caller first. Inferred from PSModulePath by
    ORDER, not absence, since both trees can be present; ambiguous keeps the default order."""
    hosts = ["pwsh.exe", "powershell.exe"]
    module_path = os.environ.get("PSModulePath", "").lower()
    windows_at = module_path.find("windowspowershell")
    seven_at = module_path.find("powershell\\7")
    if windows_at >= 0 and (seven_at < 0 or windows_at < seven_at):
        hosts = ["powershell.exe", "pwsh.exe"]
    return [host for host in hosts if shutil.which(host)]


# Quoted annotations: Python 3.9 evaluates `str | list[str]` at def time and raises TypeError.

_PROFILE_PROBE_TIMEOUT_SECONDS = 20.0


def _probe_profile_proxy_defaults(powershell: "str | list[str]") -> Optional[str]:
    """The caller's profile proxy defaults as JSON, or None. Several hosts MERGE, earlier hosts
    winning per key. Best effort: a broken profile costs one timeout."""
    hosts = [powershell] if isinstance(powershell, str) else list(powershell)
    # ONE budget for the whole probe: two hung profiles must not double the stated cost.
    deadline = time.monotonic() + _PROFILE_PROBE_TIMEOUT_SECONDS
    merged: dict = {}
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
                # text=True decodes with the locale codec and STRICT errors, and that UnicodeDecodeError is neither OSError nor SubprocessError.
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
            parsed = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(parsed, dict) or not parsed:
            continue
        # Per CMDLET, not per key: pairing one host's Proxy with another's ProxyUseDefaultCredentials would offer Windows credentials to a proxy that never asked.
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
    """Whether another host already owns the cmdlet family ``cmdlet`` belongs to. The command"""
    for owned, owner in claimed.items():
        if owner is source:
            continue
        if owned == cmdlet or _patterns_can_overlap(cmdlet, owned):
            return True
    return False


@functools.lru_cache(maxsize = 512)
def _patterns_can_overlap(left: str, right: str) -> bool:
    """Whether any command name matches BOTH wildcard patterns: two can share matches without
    matching each other as strings, so the languages are intersected."""
    if "[" in left or "[" in right:
        return True

    @functools.lru_cache(maxsize = None)
    def walk(i: int, j: int) -> bool:
        # Both patterns are consumed in step, except at '*', which may absorb one more character or none.
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
    # Read once, then GONE: a profile proxy can carry credentials every native process would inherit.
    "Remove-Item Env:_UNSLOTH_PS_PROXY_DEFAULTS -ErrorAction SilentlyContinue; "
    "if ($__unslothProxyDefaults) { try { "
    "(ConvertFrom-Json $__unslothProxyDefaults).PSObject.Properties | ForEach-Object { "
    "$PSDefaultParameterValues[$_.Name] = $_.Value } } catch { } }; "
)


_UV_CACHE_BUCKETS = ("archive-", "builds-", "built-wheels-", "wheels-", "sdists-")
_UV_CACHE_METADATA_SUFFIXES = (".lock", ".msgpack", ".http", ".rev")


def _uv_cache_has_packages(cache_dir: Path) -> bool:
    """wheels-* is metadata only on uv 0.10, so counting any file reads a merely-resolved cache
    as warm. Same rule as install.sh:_configure_uv_cache."""
    try:
        buckets = [
            entry
            for entry in cache_dir.iterdir()
            if entry.name.startswith(_UV_CACHE_BUCKETS) and entry.is_dir()
        ]
    except (OSError, ValueError):
        return False
    for bucket in buckets:
        for _root, _dirs, files in os.walk(bucket):
            for name in files:
                if name in ("CACHEDIR.TAG", ".git", ".gitignore"):
                    continue
                if name.endswith(_UV_CACHE_METADATA_SUFFIXES):
                    continue
                return True
    return False


def _uv_platform_cache_dir() -> Optional[Path]:
    if platform.system() == "Windows":
        local_app_data = (os.environ.get("LOCALAPPDATA") or "").strip()
        return Path(local_app_data) / "uv" / "cache" if local_app_data else None
    xdg = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    if xdg:
        return Path(xdg) / "uv"
    home = (os.environ.get("HOME") or "").strip()
    return Path(home) / ".cache" / "uv" if home else None


# uv's boolish spelling. Anything outside it is a value uv refuses to run on.
_UV_TRUE = ("1", "true", "yes", "on")


def _uv_no_cache_requested() -> bool:
    """uv --no-cache caches in a temporary directory and discards it, outranks --cache-dir,
    and recording it would aim later updates at a cache that never existed."""
    return (os.environ.get("UV_NO_CACHE") or "").strip().lower() in _UV_TRUE


def _uv_default_cache_dir(cwd: Optional[Path] = None) -> Optional[Path]:
    """Asked of uv, not reconstructed, so uv.toml and UV_CONFIG_FILE count, and asked from where
    setup will ask it, since uv discovers uv.toml from its working directory."""
    uv = shutil.which("uv")
    if not uv:
        return _uv_platform_cache_dir()
    child_env = {key: value for key, value in os.environ.items() if key != "UV_CACHE_DIR"}
    try:
        result = subprocess.run(
            [uv, "cache", "dir"],
            cwd = str(cwd) if cwd is not None else None,
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            env = child_env,
            timeout = 30,
            **_windows_hidden_subprocess_kwargs(),
        )
    except Exception:
        return _uv_platform_cache_dir()
    if result.returncode != 0:
        # A malformed uv.toml beside the CALLER fails this, though setup.sh runs uv elsewhere.
        return _uv_platform_cache_dir()
    lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    if not lines:
        return _uv_platform_cache_dir()
    # uv answers a relative cache-dir against its own working directory. No expanduser: uv makes a literal "~" directory.
    probe_cwd = str(cwd) if cwd is not None else os.getcwd()
    working = os.environ.get("UV_WORKING_DIR")
    base = os.path.join(probe_cwd, working) if working else probe_cwd
    return Path(os.path.abspath(os.path.join(base, lines[-1])))


def _recorded_install_uv_cache() -> Optional[Path]:
    """The cache the installer used, as it recorded it. Content cannot tell a Studio cache the
    installer filled from one holding a single wheel the backend dropped there."""
    try:
        recorded = (STUDIO_HOME / "cache" / "uv-cache-dir").read_text(
            encoding = "utf-8-sig", errors = "surrogateescape"
        )
    except OSError:
        return None
    # One record, one trailing delimiter, path before it: splitting on lines would break a path containing a newline.
    if recorded.endswith("\n"):
        recorded = recorded[:-1]
    if recorded.endswith("\r"):
        recorded = recorded[:-1]
    if not recorded.strip():
        return None
    return Path(os.path.abspath(recorded))


def _backfill_uv_cache_marker(env: Optional[dict]) -> None:
    """Record the cache this update used, for installs whose installer never did; otherwise they
    stay on the content fallback, which goes stale once the backend drops a wheel in."""
    if (os.environ.get("UV_CACHE_DIR") or "").strip():
        return
    if _uv_no_cache_requested():
        return
    chosen = (env or {}).get("UV_CACHE_DIR")
    if not chosen:
        return
    live = _recorded_install_uv_cache()
    if live is not None and _uv_cache_has_packages(live):
        return
    stage_root = (os.environ.get(_studio_stage.STAGE_ROOT_ENV) or "").strip()
    if stage_root:
        # A 805-807 desktop shell ran the OLD CLI with --stage and it is running this setup inside its stage, so park the choice for its stage() to promote.
        marker = Path(stage_root) / _studio_stage.UV_CACHE_MARKER
    else:
        marker = STUDIO_HOME / "cache" / "uv-cache-dir"
    try:
        marker.parent.mkdir(parents = True, exist_ok = True)
        marker.unlink(missing_ok = True)
        # fsencode: an undecodable path arrives as surrogates and raises UnicodeEncodeError, not OSError.
        marker.write_bytes(os.fsencode(f"{chosen}\n"))
    except (OSError, ValueError):
        pass


def _with_studio_uv_cache(env: Optional[dict], cwd: Optional[Path] = None) -> Optional[dict]:
    """An update reached neither installer nor _setup_cache_env, so uv re-downloaded
    what the install had just fetched."""
    if (os.environ.get("UV_CACHE_DIR") or "").strip():
        return env
    if _uv_no_cache_requested():
        return env
    studio_cache = STUDIO_HOME / "cache" / "uv"
    recorded = _recorded_install_uv_cache()
    if recorded is not None and _uv_cache_has_packages(recorded):
        # Only while it holds something: a marker for an emptied cache loses to a warm one.
        return {**(env or os.environ), "UV_CACHE_DIR": str(recorded)}
    # No marker, and content cannot settle it: one on-demand wheel warms the Studio cache even in shared mode, so use uv's default.
    default_cache = _uv_default_cache_dir(cwd)
    if default_cache is not None and _uv_cache_has_packages(default_cache):
        return {**(env or os.environ), "UV_CACHE_DIR": str(default_cache)}
    return {**(env or os.environ), "UV_CACHE_DIR": str(studio_cache)}


def _run_setup_script(*, verbose: bool = False, repo_root: Optional[Path] = None) -> None:
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
    # Where setup runs uv from: setup.sh cds into its own directory, setup.ps1 keeps this cwd.
    setup_cwd = None if platform.system() == "Windows" else script.parent
    env = _with_studio_uv_cache(env, cwd = setup_cwd)

    if platform.system() == "Windows":
        # Resolved, not bare: PATH is not trusted here (#9440) and the Popen below has no OSError handler.
        powershell = _studio_runtime_gate.resolve_windows_powershell()
        powershell_args = [powershell]
        # PRESENCE, not truthiness: install.ps1 sets this to "{}" when it found no proxy, and reloading the profiles it discarded would undo that.
        if os.environ.get("_UNSLOTH_PS_PROXY_DEFAULTS") is None:
            probed = _probe_profile_proxy_defaults(_profile_probe_hosts() or [powershell])
            if probed:
                env = {**(env or os.environ), "_UNSLOTH_PS_PROXY_DEFAULTS": probed}
        # -NoProfile unconditionally: install.ps1 hands off from a tty console, and a profile aliasing uv or python would break setup.ps1.
        powershell_args.append("-NoProfile")
        if _should_hide_windows_subprocesses():
            # Match install.rs: avoid the Hidden/Bypass detection pair; CREATE_NO_WINDOW below already hides the console.
            powershell_args.extend(["-NoLogo", "-NonInteractive"])
        # -Command + `*>&1` (not -File) so setup.ps1's Write-Host output merges into stdout; -File drops it when stdout is a pipe. Single quotes are doubled for paths with apostrophes.
        script_pwsh_literal = str(script).replace("'", "''")
        powershell_args.extend(
            [
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                f"{_PS_PROXY_DEFAULTS_PRELUDE}& '{script_pwsh_literal}' *>&1",
            ]
        )
        # Popen defaults to close_fds=True on Windows, so with CREATE_NO_WINDOW the child has no console and Write-Host writes to nothing.
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
    _backfill_uv_cache_marker(env)


# Fetched rather than shipped, so a launcher fix reaches users without waiting for a release.
_INSTALLER_URL_BASH = "https://unsloth.ai/install.sh"
_INSTALLER_URL_PWSH = "https://unsloth.ai/install.ps1"
_INSTALLER_FETCH_HOSTS = frozenset({"unsloth.ai", "raw.githubusercontent.com"})
_INSTALLER_FETCH_TIMEOUT = 30
_INSTALLER_MAX_BYTES = 8 * 1024 * 1024
# The flag this code passes: internal names would be tighter but can be renamed, and a false negative skips every refresh.
_INSTALLER_MARKERS = {
    "install.sh": (b"--shortcuts-only",),
    "install.ps1": (b"--shortcuts-only",),
}


def _is_allowed_installer_url(url: str) -> bool:
    split = urllib.parse.urlsplit(url)
    return split.scheme == "https" and split.hostname in _INSTALLER_FETCH_HOSTS


class _InstallerRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _is_allowed_installer_url(newurl):
            raise urllib.error.URLError(f"refused installer redirect to {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _build_installer_opener() -> urllib.request.OpenerDirector:
    """A private opener, so the redirect chain can be checked before it is followed. The installed
    opener cannot be reused: add_handler assigns handler.parent and would repoint it at this one."""
    return urllib.request.build_opener(_InstallerRedirectHandler)


def _looks_like_installer(body: Optional[bytes], installer_name: str) -> bool:
    """Cheap shape check before a fetched installer is executed. Not a trust check: it stops a
    captive-portal page or an error body from being piped into bash."""
    if not body or len(body) < 512:
        return False
    head = body.lstrip()[:256].lower()
    # `<#` opens PowerShell comment-based help, an ordinary start for install.ps1.
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
            # read(amt) does not check Content-Length; only a further read() raises IncompleteRead on a truncated transfer.
            if len(body) <= _INSTALLER_MAX_BYTES:
                body += response.read()
    except (
        urllib.error.URLError,
        # Raised at the HTTP framing layer, which is neither URLError nor OSError.
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
    """Source-tree installers, which outrank the network because `update --local` is testing
    its own installer."""
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
    """Every candidate that exists, not just the first: an unlaunchable candidate must still
    leave the next one to try before the network is reached."""
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
        # -NoProfile unconditionally, as in _run_setup_script: the visible console path loads a profile.
        ps_argv.append("-NoProfile")
        if _should_hide_windows_subprocesses():
            # Avoid the same Hidden/Bypass detection pair as setup above;
            # both local and fetched runners set CREATE_NO_WINDOW.
            ps_argv.extend(["-NoLogo", "-NonInteractive"])

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
    """False when the interpreter could not be launched, so the caller can fall back to the
    next candidate and then the network instead of ending the refresh early."""
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
    """Run a fetched install.ps1 from a tempfile. -File rather than `-Command -`: stdin decoding
    mangles install.ps1's box-drawing chars, and args go after the path so `Install-UnslothStudio
    @args` receives them. A tempfile that cannot be written is reported and skipped."""
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
        # Duplicate-metadata repair can reinstall unsloth even under SKIP_STUDIO_BASE, so free the running Windows launcher.
        with _WindowsLauncherUpdateTransaction() as launcher_update:
            _run_setup_script(verbose = verbose)
            launcher_update.validate_launcher()


def _fail_if_install_damaged(package_name: str = "unsloth") -> None:
    """Refuse to call an update successful when the tree it produced is damaged: pip treats intact
    metadata as satisfied, so the update exits 0 and the backend dies at boot."""
    managed_venv = _studio_stage.runtime_root(STUDIO_HOME) / "unsloth_studio"
    if _studio_deps.running_outside_managed_venv((managed_venv,)):
        # This CLI is not in the venv the update wrote, so its file list describes the wrong tree.
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
    # Carry the custom root and the recorded install mode, or the reinstall builds a fresh ~/.unsloth/studio and pulls the whole PyTorch stack. No root argument: recorded_no_torch reads the VENV.
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
        # The assignments go before `sh`, not before `curl`: that is the form install.sh documents.
        env = ""
        if _STUDIO_HOME_IS_CUSTOM:
            env = f"UNSLOTH_STUDIO_HOME={shlex.quote(str(STUDIO_HOME))} "
        if no_torch:
            env += "UNSLOTH_NO_TORCH=1 "
        typer.echo(f"  curl -fsSL https://unsloth.ai/install.sh | {env}sh", err = True)
    typer.echo("", err = True)
    # The installer installs only the current requirement sets, so a leftover package is not repaired by the command above.
    typer.echo("If a package above is still listed after that, the installer does not", err = True)
    typer.echo("manage it. Repair it directly, or remove it if nothing needs it:", err = True)
    # --no-deps, or --force-reinstall could swap the installed CUDA/ROCm torch build; <package>==<version>, or it upgrades the orphan its consumers pinned.
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
    stage: bool = typer.Option(
        False,
        "--stage",
        hidden = True,
        help = "Accepted for 805-807 desktop shells, which get a refusal. Background staging is gone.",
    ),
):
    """Update Unsloth Studio dependencies and rebuild."""
    # Re-export UNSLOTH_STUDIO_HOME so the refresh subprocess resolves the same install root.
    _ensure_studio_env_exported()
    # `is True`, not truthiness: an in-process caller that omits it gets a truthy OptionInfo sentinel.
    if stage is True:
        _refuse_staged_update()
        return
    staging = _studio_stage.is_staging()
    # Do not inherit SKIP_STUDIO_BASE from a parent install.ps1 session.
    os.environ.pop("SKIP_STUDIO_BASE", None)
    os.environ["STUDIO_PACKAGE_NAME"] = package
    repo_root: Optional[Path] = None
    if local:
        os.environ["STUDIO_LOCAL_INSTALL"] = "1"
        # Explicit repo root: __file__ holds only from a checkout, and once unsloth is installed non-editably parents[2] IS site-packages, which uv rejects.
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
                # PowerShell has no `VAR=value command` prefix form, and this guard fires on the Windows path.
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
    # The gate keeps a second Unsloth process off the venv; the transaction keeps the launcher recoverable across setup.
    runtime_gate_handoff = _studio_runtime_gate.consume_runtime_gate_handoff()
    with _studio_runtime_launch_guard(inherited = runtime_gate_handoff or staging):
        if not staging:
            _studio_runtime_gate.ensure_managed_environment_is_idle(STUDIO_HOME)
        # Constructed after the idle scan (pinned by test_studio_runtime_gate).
        launcher_transaction = _WindowsLauncherUpdateTransaction()
        if staging:
            # A staged run writes no launcher.
            launcher_transaction.enabled = False
        with launcher_transaction as launcher_update:
            _run_setup_script(verbose = verbose, repo_root = repo_root)
            # Runs even with --no-verify: a successful update must leave its own launcher usable.
            launcher_update.validate_launcher()
            if verify:
                _fail_if_install_damaged(package)
    # Tauri desktop owns its own bundle entries; refreshing here would duplicate shortcuts.
    if staging or os.environ.get("UNSLOTH_TAURI_UPDATE") == "1":
        if verbose:
            typer.echo("  refresh-launcher  skipped (Tauri update)")
        return
    _refresh_desktop_shortcuts(verbose = verbose)


def _discard_orphaned_stage() -> None:
    """Take a leftover `.update-stage` out of an 805-807 shell's way: they map any stage directory
    to `stage` and re-spawn `--stage` forever. Renamed before deleting."""
    stage = STUDIO_HOME / _studio_stage.STAGE_DIR_NAME
    if not os.path.lexists(stage):
        return
    aside = STUDIO_HOME / f".update-rollback-stage-{os.getpid()}"
    try:
        os.replace(stage, aside)
    except OSError:
        shutil.rmtree(stage, ignore_errors = True)
    else:
        shutil.rmtree(aside, ignore_errors = True)


def _refuse_staged_update() -> None:
    """Fail fast for a 805-807 desktop shell asking this wheel to stage.

    The `.update-failed.json` marker is what makes their reconciler skip the same version.
    backend_version is a plain String there, so a failed lookup is spelled out rather than null;
    shell_version is an Option<String> and stays null. Nothing here touches the environment.
    """
    from importlib.metadata import version as package_version

    try:
        backend_version = package_version("unsloth")
    except Exception:
        backend_version = "unknown"
    if not isinstance(backend_version, str) or not backend_version:
        backend_version = "unknown"
    shell_version = (os.environ.get(_studio_stage.SHELL_VERSION_ENV) or "").strip() or None
    _discard_orphaned_stage()
    marker = STUDIO_HOME / ".update-failed.json"
    payload = (
        json.dumps({"backend_version": backend_version, "shell_version": shell_version}, indent = 2)
        + "\n"
    )
    try:
        marker.parent.mkdir(parents = True, exist_ok = True)
        temporary = marker.with_name(marker.name + f".{os.getpid()}.tmp")
        temporary.write_text(payload, encoding = "utf-8")
        os.replace(temporary, marker)
    except OSError:
        # A refusal the desktop can act on matters more than the marker.
        pass
    # stdout, not stderr: update.rs promotes a [TAURI:ERROR] line off the child's stdout.
    typer.echo("[TAURI:ERROR] background staging is no longer supported; run the standard update")
    raise typer.Exit(1)


class _WindowsLauncherUpdateTransaction:
    """Keep the managed Windows launcher recoverable during a Python update."""

    _VERSION_TIMEOUT_SECONDS = 10
    # Sentinel rather than a message: _launcher_health_error matches on identity, and it never reaches a user.
    _POLICY_BLOCKED = "an Application Control policy blocked the launcher"
    # Absence is not corruption: quarantine leaves a healthy environment, so keep it apart from the PE-shape failure.
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
        # Validity, not existence: a truncated or quarantined launcher is just as unusable.
        if self._is_valid_pe(self.launcher):
            return
        last_error: Optional[Tuple[Path, OSError]] = None
        for recovery in (self.backup, self.stale, self.legacy_backup, self.shim):
            if recovery is not None and self._is_valid_pe(recovery):
                try:
                    self._atomic_copy(recovery, self.launcher)
                except OSError as exc:
                    # Try the next copy: the header check and the copy open the file separately, so one candidate lost to antivirus must not fail the install.
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
        """Free the canonical path so the installer can publish a replacement: uv deletes a third-party
        console script outright and hard-errors when it is in use, after which the pip fallback
        no-ops. Non-fatal; it costs only the upgrade."""
        assert self.launcher is not None and self.stale is not None
        if not self._is_valid_pe(self.launcher):
            return
        try:
            os.replace(self.launcher, self.stale)
        except OSError as exc:
            # Not fatal, but say what it costs: the pip fallback drops --upgrade-package, leaving unsloth at its old version.
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
        """Copies that could stand in for the launcher, best first. All are kept: passing the header
        check does not make any of them runnable."""
        seen: List[Path] = []
        for path in (self.backup, self.stale, self.legacy_backup, self.shim):
            if path is None or not self._is_valid_pe(path):
                continue
            if not any(os.path.normcase(str(path)) == os.path.normcase(str(p)) for p in seen):
                seen.append(path)
        return seen

    def _restore_from(self, source: Path) -> bool:
        assert self.launcher is not None
        # The common setup-failure case leaves the original in place; do not replace a running Windows file with a byte-identical source.
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
        """Put back the first copy that actually runs. Under Application Control every --version dies in
        CreateProcess, so this degrades to the shape check."""
        if self._launcher_health_error() is None:
            return True
        candidates = self._recovery_candidates()
        for source in candidates:
            if self._restore_from(source) and self._launcher_health_error() is None:
                return True
        # Nothing ran; leave the best candidate rather than whichever was tried last.
        if candidates:
            self._restore_from(candidates[0])
        # Gone, or denied by policy, is still not a broken CLI. Asked only after every candidate.
        return self._recovered_cli_health_error() is None

    def _launcher_runs_error(self) -> Optional[str]:
        """Whether THIS launcher file starts and answers --version. About the file, not the CLI: the
        only check that catches a PE-shaped but corrupt stub."""
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
        """Whether the update left a working CLI. Identical to _launcher_runs_error except for an
        Application Control denial, which happens before Python starts: ask the signed interpreter,
        or every update on such a machine rolls a good install back (#8490)."""
        error = self._launcher_runs_error()
        if error is self._POLICY_BLOCKED:
            return self._interpreter_health_error(error)
        return error

    def _recovered_cli_health_error(self) -> Optional[str]:
        """_launcher_health_error, once recovery has been ruled out: a missing launcher IS worth
        restoring, but once every candidate failed, quarantine says as little as a denial (#8490)."""
        error = self._launcher_runs_error()
        if error is self._POLICY_BLOCKED or error is self._LAUNCHER_ABSENT:
            return self._interpreter_health_error(error)
        return error

    def _interpreter_health_error(self, reason: str) -> Optional[str]:
        """Health of the managed CLI when the launcher cannot be started. Isolated alone among this
        module's managed invocations, mirroring build_update_command's Isolation::Isolated."""
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
                # The import probe's ceiling, not the launcher's: this is an interpreter start plus the whole CLI import.
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
        """Scripts dir of the venv setup actually updates: STUDIO_HOME/unsloth_studio, which is not
        this interpreter when a pip-installed or checkout CLI drives the update."""
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
        # Under the Unsloth home, not the venv: setup.ps1 removes the whole $VenvDir, which Windows refuses while a handle inside is open.
        self.lock_path = STUDIO_HOME / "unsloth.exe.update-lock"
        # install.ps1 hardlinks this to the launcher, so it survives the old updater and is a valid recovery source.
        self.shim = STUDIO_HOME / "bin" / "unsloth.exe"
        self.stale = scripts / "unsloth.exe.update-stale"
        self._acquire_lock()
        try:
            self._recover_missing_launcher()
            if not self._is_valid_pe(self.launcher):
                # Warn, do not exit: the previous updater could leave no launcher and no .deleteme, and refusing would strand those users.
                typer.echo(
                    f"Warning: the managed Unsloth launcher is missing or invalid: {self.launcher}",
                    err = True,
                )
                typer.echo("Continuing; setup may reinstall it.", err = True)
                if self._retained_backup() is None:
                    self.backup = None
            elif self._retained_backup() is None:
                # Only when there is no usable backup: a surviving one holds the last launcher known to run.
                try:
                    self._atomic_copy(self.launcher, self.backup)
                except OSError as exc:
                    # A backup is a safety net, not a precondition.
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
        # Sampled before any restore, since whether setup published anything decides how a bad result reads.
        published = self.launcher.exists()
        error = self._launcher_health_error()
        if error is not None:
            restored = self._restore_runnable()
            # Setup publishing nothing is the case this exists for, so restoring is success; a launcher setup DID write that cannot run is a failure.
            if published or not restored:
                typer.echo(f"Error: Unsloth Studio update failed because {error}.", err = True)
                if restored:
                    typer.echo("The previous launcher was restored.", err = True)
                elif self._retained_backup() is not None:
                    typer.echo(f"Manual recovery copy retained at: {self.backup}", err = True)
                raise typer.Exit(1)
        self._validated = True
        # Only once the launcher is back: a quarantined stub can be judged healthy while every restore failed, and the copies are the only recovery material.
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
        # 2 adds studio_install_ok; the desktop treats < 2 as stale.
        "desktop_manageability_version": 2,
        "supports_provision_desktop_auth": True,
        "supports_api_only": True,
        "supports_desktop_backend_ownership": True,
        # Did the install finish and are the backend boot deps still there.
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

    Scans the installed files too, unlike `desktop-capabilities`: nothing times
    this one out.
    """
    state = _install_state(deep = True)

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
