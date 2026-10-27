# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import ntpath
import os
import platform
import re
import stat as stat_module
import sys
import threading
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable
import tempfile

from loggers import get_logger
from utils.account_context import current_account, is_owner_context
from utils.paths.path_utils import drop_appledouble_metadata, host_normalize_path
from utils.paths.private_tmp import private_tmp_dir, private_tmp_path

logger = get_logger(__name__)


def _infer_studio_home_from_venv() -> Path | None:
    """Needs a share/studio.conf or bin shim sentinel, so a dev venv named unsloth_studio is ignored."""
    try:
        prefix = Path(sys.prefix).resolve()
    except (OSError, ValueError):
        return None
    if prefix.name != "unsloth_studio":
        return None
    candidate = prefix.parent
    shim_name = "unsloth.exe" if os.name == "nt" else "unsloth"
    try:
        has_sentinel = (candidate / "share" / "studio.conf").is_file() or (
            candidate / "bin" / shim_name
        ).is_file()
    except OSError:
        return None
    if not has_sentinel:
        return None
    # In Docker sys.prefix is the app layer, not the volume: never adopt it as home.
    app_dir = os.environ.get("UNSLOTH_STUDIO_APP", "").strip()
    if app_dir:
        try:
            if candidate == Path(app_dir).resolve():
                return None
        except (OSError, ValueError):
            pass
    return candidate


def _resolved(value: str) -> Path:
    try:
        return Path(value).expanduser().resolve()
    except (OSError, ValueError):
        return Path(value).expanduser()


MASTER_ROOT_NOTE = ".unsloth-master-root"

_recorded_master_roots: dict[str, Path | None] = {}
_recorded_master_lock = threading.Lock()


def _studio_root_without_master() -> Path:
    """Studio root with the master root excluded, so the note reader never calls back into studio_root()."""
    override = (os.environ.get("UNSLOTH_STUDIO_HOME") or "").strip()
    if not override:
        override = (os.environ.get("STUDIO_HOME") or "").strip()
    if override:
        return _resolved(override)
    inferred = _infer_studio_home_from_venv()
    if inferred is not None:
        return inferred
    return Path.home() / ".unsloth" / "studio"


def _is_legacy_studio_tree(studio: Path) -> bool:
    """Keyed on the tree, not the value: a note naming ~/.unsloth/studio or any ancestor such as $HOME must never be used, since process.rs scrubs those."""
    try:
        return studio.resolve() == (Path.home() / ".unsloth" / "studio").resolve()
    except (OSError, RuntimeError, ValueError):
        return False


def _recorded_master_root() -> Path | None:
    """Master root setup recorded, so a one-off UNSLOTH_HOME does not move later launches; honoured only while it exists and contains studio."""
    studio = _studio_root_without_master()
    key = str(studio)
    with _recorded_master_lock:
        if key in _recorded_master_roots:
            return _recorded_master_roots[key]
    found: Path | None = None
    # A miss is cached; a failure to look (EACCES, EIO) is transient and is not.
    definitive = True
    try:
        recorded = (studio / "share" / MASTER_ROOT_NOTE).read_text(encoding = "utf-8").strip()
    except (FileNotFoundError, NotADirectoryError, ValueError, UnicodeDecodeError):
        recorded = ""
    except OSError:
        recorded = ""
        definitive = False
    if recorded:
        master = _resolved(recorded)
        try:
            here = studio.resolve()
            if (
                master.is_dir()
                and (here == master or master in here.parents)
                and not _is_legacy_studio_tree(studio)
            ):
                found = master
        except (OSError, ValueError):
            found = None
    if definitive:
        with _recorded_master_lock:
            _recorded_master_roots[key] = found
    return found


def forget_recorded_master_root() -> None:
    """Drop the cached note reading. For tests, and for an installer that writes the note into
    a tree this process has already looked at."""
    with _recorded_master_lock:
        _recorded_master_roots.clear()


def unsloth_home() -> Path | None:
    """The master root, or None. STUDIO_HOME is its studio/ child; llama.cpp, node and
    whisper.cpp are SIBLINGS of studio/, the layout setup.sh already gives UNSLOTH_HOME."""
    override = (os.environ.get("UNSLOTH_HOME") or "").strip()
    if override:
        return _resolved(override)
    return _recorded_master_root()


_PORTABLE_ON_VALUES = ("1", "true", "yes", "on")
_PORTABLE_OFF_VALUES = ("0", "false", "off", "no")

# portable_mode() runs on every cache-var lookup, so warn once per process, not once per call.
_warned_unrecognized_portable = False


def _warn_unrecognized_portable(raw: str) -> None:
    global _warned_unrecognized_portable
    if _warned_unrecognized_portable:
        return
    _warned_unrecognized_portable = True
    # An off value only declines to turn portable mode on; it does not veto the master root.
    logger.warning(
        "Ignoring UNSLOTH_PORTABLE=%r: expected one of %s to turn portable mode on, or one of "
        "%s to leave that choice to UNSLOTH_HOME, which turns it on when it names a master root.",
        raw,
        "/".join(_PORTABLE_ON_VALUES),
        "/".join(_PORTABLE_OFF_VALUES),
    )


def portable_mode() -> bool:
    """Whether this install keeps everything under one directory. Implied by UNSLOTH_HOME, and
    settable on its own so an existing install can opt in."""
    # Case-folded: FALSE must not read as on.
    raw = (os.environ.get("UNSLOTH_PORTABLE") or "").strip()
    value = raw.lower()
    if value in _PORTABLE_ON_VALUES:
        return True
    if value and value not in _PORTABLE_OFF_VALUES:
        _warn_unrecognized_portable(raw)
    # Unrecognized means no opinion, as an off value does: neither vetoes a real master root.
    return unsloth_home() is not None


# Warn once per distinct pair: studio_root() runs many times a request.
_warned_root_conflicts: set[tuple[str, str]] = set()


def _warn_root_conflict(resolved: Path, master: Path) -> None:
    key = (str(resolved), str(master))
    if key in _warned_root_conflicts:
        return
    _warned_root_conflicts.add(key)
    # Not fatal: failing here would break a resolver called at import time.
    logger.warning(
        "UNSLOTH_STUDIO_HOME (%s) is outside UNSLOTH_HOME (%s); this "
        "install is not self-contained.",
        resolved,
        master,
    )


def studio_root() -> Path:
    """UNSLOTH_STUDIO_HOME outranks all others, which only name the tree this directory sits in."""
    override = (os.environ.get("UNSLOTH_STUDIO_HOME") or "").strip()
    if not override:
        override = (os.environ.get("STUDIO_HOME") or "").strip()
    if override:
        resolved = _resolved(override)
        master = unsloth_home()
        # Path.parents excludes the path itself, so a flat layout would warn on every call.
        if master is not None and master != resolved and master not in resolved.parents:
            _warn_root_conflict(resolved, master)
        return resolved
    master = unsloth_home()
    if master is not None:
        return master / "studio"
    inferred = _infer_studio_home_from_venv()
    if inferred is not None:
        return inferred
    return Path.home() / ".unsloth" / "studio"


def workspace_root() -> Path:
    """Private persistent root of the acting account: owner keeps the historical install-root
    layout, others get ``accounts/<account_id>/``, keyed by id so a reused name inherits nothing."""
    root = studio_root()
    if is_owner_context():
        return root
    return root / "accounts" / current_account().account_id


def cache_root() -> Path:
    """Central cache dir for all studio downloads (models, datasets, etc.). Shared."""
    return studio_root() / "cache"


def llama_slot_cache_root() -> Path:
    """Dir llama-server saves/restores slot KV state in across idle unloads."""
    return cache_root() / "llama-slots"


def studio_bin_root() -> Path:
    """Dir for Unsloth-managed executables (the `unsloth` shim, downloaded tools like cloudflared)."""
    return studio_root() / "bin"


def account_path(relative: str) -> Path:
    """``workspace_root() / relative``, checked to really live inside the account's workspace: a
    directory swapped for a link into another account's tree would carry all its readers there."""
    path = workspace_root() / relative
    if not is_owner_context() and not within_account(path):
        raise ValueError(f"path escapes the account workspace: {path!s}")
    return path


def assets_root() -> Path:
    return account_path("assets")


def datasets_root() -> Path:
    return account_path("assets/datasets")


def dataset_uploads_root() -> Path:
    return account_path("assets/datasets/uploads")


def recipe_datasets_root() -> Path:
    return account_path("assets/datasets/recipes")


def outputs_root() -> Path:
    return account_path("outputs")


def exports_root() -> Path:
    return account_path("exports")


def auth_root() -> Path:
    return studio_root() / "auth"


def auth_db_path() -> Path:
    return auth_root() / "auth.db"


def studio_db_path() -> Path:
    return account_path("studio.db")


def rag_root() -> Path:
    """Root directory for retrieval-augmented-generation state (db + uploads)."""
    return account_path("rag")


def rag_db_path() -> Path:
    """SQLite file holding RAG documents, chunks, FTS5 + sqlite-vec indexes."""
    return rag_root() / "rag.db"


def rag_uploads_root() -> Path:
    """Directory where uploaded source documents are stored for ingestion."""
    return rag_root() / "uploads"


def _xdg_user_dir(key: str) -> Path | None:
    config = Path.home() / ".config" / "user-dirs.dirs"
    try:
        lines = config.read_text(encoding = "utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    prefix = f"{key}="
    for line in lines:
        line = line.strip()
        if not line.startswith(prefix):
            continue
        value = line[len(prefix) :].strip().strip('"')
        if not value:
            return None
        return Path(value.replace("$HOME", str(Path.home()))).expanduser()
    return None


def _documents_from_registry_value(value: object, expandable: bool) -> Path | None:
    """The Documents path a Windows shell-folder registry value names."""
    if not isinstance(value, str) or not value.strip():
        return None
    # REG_EXPAND_SZ is unexpanded; ntpath because %VAR% is Windows syntax.
    return Path(ntpath.expandvars(value) if expandable else value)


def _windows_documents_dir() -> Path | None:
    """Asks Windows for Documents: OneDrive's Known Folder Move can leave ~/Documents behind."""
    if os.name != "nt":
        return None
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
        ) as key:
            # "Personal" is the registry's name for Documents.
            value, kind = winreg.QueryValueEx(key, "Personal")
    except OSError:
        return None
    return _documents_from_registry_value(value, kind == winreg.REG_EXPAND_SZ)


def documents_root() -> Path:
    override = (os.environ.get("UNSLOTH_STUDIO_DOCUMENTS_HOME") or "").strip()
    if override:
        return Path(override).expanduser()
    return (
        _windows_documents_dir()
        or _xdg_user_dir("XDG_DOCUMENTS_DIR")
        or (Path.home() / "Documents")
    )


def shared_project_workspaces_root() -> Path:
    """Base every account's ``project_workspaces_root`` lives under; confinement hides it first."""
    override = (os.environ.get("UNSLOTH_STUDIO_PROJECTS_HOME") or "").strip()
    return Path(override).expanduser() if override else documents_root() / "Unsloth Studio"


def project_workspaces_root() -> Path:
    override = (os.environ.get("UNSLOTH_STUDIO_PROJECTS_HOME") or "").strip()
    base = shared_project_workspaces_root()
    if is_owner_context():
        return base if override else base / "Projects"
    return base / "Accounts" / current_account().account_id / "Projects"


def shared_tmp_root() -> Path:
    return private_tmp_path("unsloth-studio")


def tmp_root() -> Path:
    # Created and checked here: on a shared /tmp another account could own or plant the root.
    root = private_tmp_dir("unsloth-studio")
    if is_owner_context():
        return root
    return root / "accounts" / current_account().account_id


def seed_uploads_root() -> Path:
    return account_path("assets/datasets/seed-uploads")


def unstructured_seed_cache_root() -> Path:
    return tmp_root() / "unstructured-seed-cache"


def unstructured_uploads_root() -> Path:
    return account_path("assets/datasets/unstructured-uploads")


def oxc_validator_tmp_root() -> Path:
    return tmp_root() / "oxc-validator"


def tensorboard_root() -> Path:
    return account_path("runs")


def _mkdir(path: Path) -> Path:
    path.mkdir(parents = True, exist_ok = True)
    return path


class RetiredAccountError(RuntimeError):
    """Held across the rename-aside and every guarded directory creation; raised for a write to an account whose private roots are retired."""


root_retirement_lock = threading.RLock()


def external_account_sandbox_root() -> Path | None:
    """The managed account's tool sandbox when ``UNSLOTH_STUDIO_SANDBOX_HOME`` moves it out of the
    workspace. A private root like the others, so retirement and ``ensure_dir`` cover it."""
    override = (os.environ.get("UNSLOTH_STUDIO_SANDBOX_HOME") or "").strip()
    if is_owner_context() or not override:
        return None
    return (
        Path(os.path.abspath(os.path.expanduser(override)))
        / "accounts"
        / current_account().account_id
    )


def managed_account_roots() -> tuple[Path, ...]:
    """Every private root retirement renames aside for the acting managed account."""
    roots = [workspace_root(), project_workspaces_root(), tmp_root()]
    sandbox = external_account_sandbox_root()
    if sandbox is not None:
        roots.append(sandbox)
    return tuple(roots)


def _under_managed_workspace(path: Path) -> bool:
    """Lexically, whether *path* is inside one of the roots retirement renames aside."""
    if is_owner_context():
        return False
    try:
        absolute = Path(os.path.abspath(path))
        for root in managed_account_roots():
            try:
                absolute.relative_to(os.path.abspath(root))
                return True
            except ValueError:
                continue
    except (OSError, ValueError):
        return False
    return False


def ensure_dir(path: Path) -> Path:
    """Create *path*; inside a managed workspace this is retirement-aware for every caller."""
    if _under_managed_workspace(path):
        return ensure_account_dir(path)
    return _mkdir(path)


def ensure_account_dir(path: Path) -> Path:
    """``ensure_dir`` inside the acting account's workspace: refuse once the tombstone is set, or a
    finalizer outliving deletion recreates the renamed-aside roots. Locked against the rename."""
    with root_retirement_lock:
        if not is_owner_context():
            from core.training.account_jobs import account_is_retired

            # Existence is not proof of life: a late request can mkdir the workspace back.
            if account_is_retired():
                raise RetiredAccountError(
                    f"account has been deleted; refusing to recreate {path!s}"
                )
        return _mkdir(path)


def legacy_hf_cache_dir() -> Path:
    """Old Unsloth-specific HF hub cache, kept for backward-compat scans."""
    return cache_root() / "huggingface" / "hub"


def hf_default_cache_dir() -> Path:
    """Platform default HF hub cache, ignoring env overrides, so pre-Studio downloads are found."""
    return Path.home() / ".cache" / "huggingface" / "hub"


def _host_path(path: str | Path) -> Path:
    """Maps a drive-letter path from another tool's config through the WSL automount root before use."""
    return Path(host_normalize_path(str(path))).expanduser()


def _existing_dirs(candidates: Iterable[str | Path], *, resolve: bool) -> list[Path]:
    """Resolves paths only for containment checks; model ids keep the spelling the user configured."""
    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            expanded = _host_path(candidate)
            resolved = expanded.resolve()
            is_dir = expanded.is_dir()
        except (OSError, RuntimeError, ValueError):
            continue
        key = str(resolved)
        if key in seen or not is_dir:
            continue
        seen.add(key)
        out.append(resolved if resolve else expanded)
    return out


def _lmstudio_downloads_folder() -> str:
    """Reads settings with utf-8-sig, since LM Studio may write a BOM that breaks a plain utf-8 read."""
    settings_path = Path.home() / ".lmstudio" / "settings.json"
    if not settings_path.is_file():
        return ""
    try:
        settings = json.loads(settings_path.read_text(encoding = "utf-8-sig"))
        downloads = settings.get("downloadsFolder", "")
        # A number or list here is a corrupt file, not a path; str() would stat "123".
        return downloads if isinstance(downloads, str) else ""
    except Exception as exc:
        logger.debug("Ignoring unreadable LM Studio settings at %s: %s", settings_path, exc)
        return ""


def lmstudio_model_dirs() -> list[Path]:
    """Return LM Studio model directories that exist on disk."""
    candidates: list[str | Path] = []

    downloads = _lmstudio_downloads_folder()
    if downloads:
        candidates.append(downloads)

    candidates.append(Path.home() / ".lmstudio" / "models")
    # Legacy cache location.
    candidates.append(Path.home() / ".cache" / "lm-studio" / "models")

    return _existing_dirs(candidates, resolve = False)


def ollama_model_dirs() -> list[Path]:
    """Ollama dirs from OLLAMA_MODELS, user-level and system-wide defaults; only those that exist."""
    candidates: list[str | Path] = []
    ollama_env = os.environ.get("OLLAMA_MODELS")
    if ollama_env:
        candidates.append(ollama_env)
    candidates.append(Path.home() / ".ollama" / "models")
    candidates.append(Path("/usr/share/ollama/.ollama/models"))
    candidates.append(Path("/var/lib/ollama/.ollama/models"))
    return _existing_dirs(candidates, resolve = False)


def _hermes_native_home() -> Path:
    """Hermes' platform-native home, ignoring HERMES_HOME."""
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        return base / "hermes"
    return Path.home() / ".hermes"


def _hermes_root() -> Path:
    """HERMES_HOME under the native home means that home; profiles/<name> elsewhere means its root; anything else is the root (Docker, custom)."""
    env_home = os.environ.get("HERMES_HOME", "").strip()
    native = _hermes_native_home()
    if not env_home:
        return native
    env_path = Path(env_home)
    try:
        env_path.resolve().relative_to(native.resolve())
        return native
    except (OSError, ValueError):
        pass
    if env_path.parent.name == "profiles":
        return env_path.parent.parent
    return env_path


def hermes_model_dirs() -> list[Path]:
    """Scans the root models dir and the native home models dir: downloads are machine-scoped, and unsloth start hermes points HERMES_HOME at a throwaway dir."""
    return _existing_dirs(
        [_hermes_root() / "models", _hermes_native_home() / "models"],
        resolve = False,
    )


def _omlx_base_path() -> Path:
    """oMLX's data root, resolved as its own ``resolve_default_base_path`` does:
    ``OMLX_BASE_PATH`` > the macOS app's bootstrap file > ``~/.omlx``."""
    env_value = os.environ.get("OMLX_BASE_PATH", "").strip()
    if env_value:
        return Path(env_value).expanduser()
    bootstrap = Path.home() / "Library" / "Application Support" / "oMLX" / "base-path"
    try:
        raw = bootstrap.read_text(encoding = "utf-8").strip()
    except (OSError, UnicodeDecodeError):
        raw = ""
    return Path(raw).expanduser() if raw else Path.home() / ".omlx"


def _omlx_configured_dirs(base: Path) -> list[str]:
    """``model.model_dirs``, else the legacy ``model.model_dir``, from oMLX's settings.json."""
    settings_path = base / "settings.json"
    if not settings_path.is_file():
        return []
    try:
        settings = json.loads(settings_path.read_text(encoding = "utf-8-sig"))
        model_settings = settings.get("model") or {}
        configured = model_settings.get("model_dirs") or []
        if isinstance(configured, str):
            configured = [configured]
        if not configured and model_settings.get("model_dir"):
            configured = [model_settings["model_dir"]]
        return [d for d in configured if isinstance(d, str) and d]
    except Exception as exc:
        logger.debug("Ignoring unreadable oMLX settings at %s: %s", settings_path, exc)
        return []


def omlx_model_dirs() -> list[Path]:
    """Drops the LM Studio folder from the oMLX roots to avoid a double scan; keeps HF cache roots, whose models-- walk cannot see oMLX flat folders."""
    base = _omlx_base_path()
    # oMLX applies OMLX_MODEL_DIR (comma-separated) over settings.json.
    env_dirs = [d.strip() for d in os.environ.get("OMLX_MODEL_DIR", "").split(",") if d.strip()]
    candidates: list[str | Path] = env_dirs or _omlx_configured_dirs(base) or [base / "models"]
    lmstudio = set()
    for path in lmstudio_model_dirs():
        try:
            lmstudio.add(str(path.resolve()))
        except (OSError, RuntimeError, ValueError):
            continue
    out = []
    for path in _existing_dirs(candidates, resolve = False):
        try:
            if str(path.resolve()) in lmstudio:
                continue
        except (OSError, RuntimeError, ValueError):
            continue
        out.append(path)
    return out


def well_known_model_dirs() -> list[Path]:
    """Only existing dirs, so the UI never shows dead chips; LM Studio, Ollama and Hermes lead."""
    candidates: list[str | Path] = []
    candidates.extend(lmstudio_model_dirs())
    candidates.extend(ollama_model_dirs())
    candidates.extend(hermes_model_dirs())

    # HF hub cache root, separate from the explicit HF cache chip.
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    # Generic "my models" spots users drop things into.
    for name in ("models", "Models"):
        candidates.append(Path.home() / name)

    return _existing_dirs(candidates, resolve = True)


def _user_set_hf_home() -> bool:
    """Reads the import-time snapshot; initialize_hf_cache_environment fills a blank HF_HOME first."""
    try:
        from utils.hf_cache_settings import _EXPLICIT_CACHE_ENV
    except ImportError:
        return False
    return bool(_EXPLICIT_CACHE_ENV.get("HF_HOME"))


def _portable_cache_defaults(root: Path) -> dict[str, str]:
    """Cache vars move under the root only in portable mode; HF_HOME never moves, since credentials should not follow the cache onto a removable volume."""
    if not portable_mode():
        return {}
    if _user_set_hf_home():
        # Others derive from an explicit HF_HOME; pinning them would split the cache.
        return {"TORCH_HOME": str(root / "torch")}
    return {
        "HF_DATASETS_CACHE": str(root / "huggingface" / "datasets"),
        "HF_ASSETS_CACHE": str(root / "huggingface" / "assets"),
        # transformers adds this to sys.path at import; unset, remote-code modules land on the host.
        "HF_MODULES_CACHE": str(root / "huggingface" / "modules"),
        "TORCH_HOME": str(root / "torch"),
    }


def _triton_cache_defaults(root: Path) -> dict[str, str]:
    """Not TRITON_HOME, which would also move ~/.triton/override and silently drop kernel overrides."""
    if (os.environ.get("TRITON_HOME") or "").strip():
        # Whoever moved the whole tree meant the cache with it; TRITON_CACHE_DIR would outrank it.
        return {}
    return {
        "TRITON_CACHE_DIR": str(root / "triton"),
        # A sibling, not a child: dumps are asked for by hand and should outlive a cache wipe.
        "TRITON_DUMP_DIR": str(root / "triton-dump"),
    }


def _nothing_at(path: Path, *, ending: str = "") -> bool:
    """Positive absence proof only: Path.exists reads unreadable paths as missing; pass ending lowercase, since Windows glob matches case-insensitively."""
    try:
        if not ending:
            os.lstat(path)
            return False
        # lstat, not scandir: scandir follows links, hiding a redirected target.
        if stat_module.S_ISLNK(os.lstat(path).st_mode) or _is_reparse_point(path):
            return False
        with os.scandir(path) as entries:
            return not any(entry.name.lower().endswith(ending) for entry in entries)
    except FileNotFoundError:
        # Windows collapses NotADirectory/Permission into PATH_NOT_FOUND; do not trust absence alone.
        return _nothing_above(path)
    except (OSError, ValueError):
        # Could not look: declining a pin is cheaper than hiding config beneath one.
        return False


def _nothing_above(path: Path) -> bool:
    """A FileNotFoundError proves absence only when the nearest statable ancestor is a directory."""
    current = os.path.dirname(os.fspath(path))
    while current:
        try:
            info = os.stat(current)
        except FileNotFoundError:
            parent = os.path.dirname(current)
            if parent == current:
                return True
            current = parent
            continue
        except OSError:
            return False
        return stat_module.S_ISDIR(info.st_mode)
    return True


def _matplotlib_config_dir() -> Path | None:
    """Mirrors the matplotlib config dir (None if it falls back to temp); XDG_CONFIG_HOME goes before Path.home() since it works with no home."""
    if sys.platform.startswith(("linux", "freebsd")):
        base = (os.environ.get("XDG_CONFIG_HOME") or "").strip()
        if base:
            return Path(base) / "matplotlib"
    try:
        home = Path.home()
    except (OSError, RuntimeError):
        return None
    if sys.platform.startswith(("linux", "freebsd")):
        return home / ".config" / "matplotlib"
    if sys.platform == "win32":
        legacy = home / ".matplotlib"
        # Not is_dir(): unreadable ~/.matplotlib raises before 3.14 and reads absent from 3.14.
        if not _nothing_at(legacy):
            return legacy
        local_app_data = (os.environ.get("LOCALAPPDATA") or "").strip()
        return Path(local_app_data) / "matplotlib" if local_app_data else legacy
    return home / ".matplotlib"


def _matplotlib_defaults(root: Path) -> dict[str, str]:
    """MPLCONFIGDIR also moves config, so pin it only when the config dir holds no user files."""
    managed = root / "matplotlib"
    pinned = {"MPLCONFIGDIR": str(managed)}
    # Our own config first, or a later ~/.config/matplotlib flips the style across launches.
    if not (
        _nothing_at(managed / "matplotlibrc")
        and _nothing_at(managed / "stylelib", ending = ".mplstyle")
    ):
        return pinned
    config_dir = _matplotlib_config_dir()
    if config_dir is not None and not (
        _nothing_at(config_dir / "matplotlibrc")
        and _nothing_at(config_dir / "stylelib", ending = ".mplstyle")
    ):
        return {}
    return pinned


def _is_reparse_point(path: Path) -> bool:
    """Detects Windows junctions, which is_symlink misses; returns False without os.path.isjunction."""
    isjunction = getattr(os.path, "isjunction", None)
    if isjunction is None:
        return False
    try:
        return bool(isjunction(path))
    except (OSError, ValueError):
        return False


def _data_designer_in_use(home: Path) -> bool:
    """A home that cannot be listed counts as in use; an empty one created at first launch does not."""
    try:
        entries = list(home.iterdir())
    except FileNotFoundError:
        return False
    except (OSError, ValueError):
        return True
    for entry in entries:
        try:
            if entry.name != "managed-assets":
                return True
            # A link is a user redirect; is_dir() follows it and would discard the redirect.
            if entry.is_symlink() or _is_reparse_point(entry):
                return True
            if not entry.is_dir():
                return True
            if any(entry.iterdir()):
                return True
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            return True
    return False


def _data_designer_defaults(root: Path) -> dict[str, str]:
    """Not a cache: leave an existing ~/.data-designer alone; MANAGED_ASSETS_PATH follows the home and both are read at import time."""
    if (os.environ.get("DATA_DESIGNER_HOME") or "").strip():
        return {}
    home = root.parent / "data-designer"
    pinned = {
        "DATA_DESIGNER_HOME": str(home),
        "DATA_DESIGNER_MANAGED_ASSETS_PATH": str(home / "managed-assets"),
    }
    # Our populated home first, or a later ~/.data-designer would take over.
    if _data_designer_in_use(home):
        return pinned
    try:
        legacy = Path.home() / ".data-designer"
    except (OSError, RuntimeError):
        # No home, so the pin can hide nothing: data_designer's default is off the same call.
        return pinned
    return pinned if _nothing_at(legacy) else {}


def _path_safe(value: str) -> str:
    """A directory-name-safe rendering of a build field."""
    return re.sub(r"[^A-Za-z0-9.]+", "-", value)


def _torch_version_fields() -> dict[str, str]:
    """Reads torch/version.py as text, since torch may not be importable yet and annotations vary."""
    origin = getattr(importlib.util.find_spec("torch"), "origin", None)
    if not origin:
        return {}
    text = (Path(origin).parent / "version.py").read_text(encoding = "utf-8")
    found = re.findall(
        r"""^(__version__|debug|cuda|hip|xpu)\s*(?::[^=\n]+)?=\s*([^\s#]+)""",
        text,
        re.MULTILINE,
    )
    return {name: value.strip("'\"") for name, value in found}


def _torch_accelerator_tag(fields: dict[str, str]) -> str:
    """torch's own cu_str, widened to the runtimes it declines to name: cpp_extension picks 'cpu'
    whenever version.cuda is unset, filing a ROCm build beside a real CPU one. hip first, as
    torch main prioritises ROCm."""
    for field, prefix in (("hip", "rocm"), ("cuda", "cu"), ("xpu", "xpu")):
        value = fields.get(field)
        if not value or value == "None":
            continue
        # torch spells the CUDA version without dots: 12.8 -> cu128.
        return prefix + _path_safe(value.replace(".", "") if field == "cuda" else value)
    return "cpu"


def _torch_runtime_tag() -> str:
    """Tags builds by Python, platform and accelerator: torch tags only its default root, so a flat TORCH_EXTENSIONS_DIR would mix runtimes."""
    tag = f"py{sys.version_info.major}{sys.version_info.minor}{getattr(sys, 'abiflags', '')}"
    # Include arch: arm64 and Rosetta x86_64 pythons otherwise share ninja builds.
    tag += "_" + _path_safe(f"{sys.platform}-{platform.machine() or 'unknown'}")
    try:
        fields = _torch_version_fields()
    except (ImportError, OSError, ValueError, AttributeError):
        # No torch yet, or a half-built tree; the interpreter tag alone still isolates.
        return tag
    if not fields:
        return tag
    tag += "_" + _torch_accelerator_tag(fields)
    version = fields.get("__version__")
    if version:
        tag += "_" + _path_safe(version)
    if fields.get("debug") == "True":
        # A debug build keeps the soname of a release one but not its ABI.
        tag += "_debug"
    return tag


# Inductor's cpp_builder shlex-splits the g++ command, so paths with spaces/quotes break builds.
# toolchain_path_unparseable has the full character list.
_TOOLCHAIN_PATH_KEYS = frozenset(
    {
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "TRITON_CACHE_DIR",
        "TRITON_DUMP_DIR",
        "TRITON_HOME",
        "CUDA_CACHE_PATH",
    }
)


def _usable_dir(value: str) -> bool:
    """A real write probe: a read-only dir passes makedirs(exist_ok) yet fails every later write."""
    try:
        if not Path(value).is_dir():
            return False
    except (OSError, ValueError):
        return False
    try:
        handle, probe = tempfile.mkstemp(dir = value, prefix = ".unsloth-write-probe.")
    except (OSError, ValueError):
        return False
    # Guarded: an EIO/ENOSPC on close must not kill backend start for a yes/no probe.
    try:
        os.close(handle)
    except OSError:
        return False
    try:
        os.unlink(probe)
    except OSError:
        pass
    return True


def toolchain_path_unparseable(value: str) -> bool:
    """Whitespace or quotes, and on POSIX backslashes, break the unquoted path under shlex.split."""
    if any(ch.isspace() for ch in value):
        return True
    if "'" in value or '"' in value:
        return True
    return os.name != "nt" and "\\" in value


def _toolchain_unsafe(key: str, value: str) -> bool:
    """Whether pinning *key* to *value* would hand a compiler a path it cannot parse."""
    return key in _TOOLCHAIN_PATH_KEYS and toolchain_path_unparseable(value)


def _private_dir(path: str) -> bool:
    """Rejects a dir another local user could pre-create or rename (torch loads .so from it); parent checked first, via lstat, POSIX ownership only."""
    parent = Path(path).parent
    try:
        parent.mkdir(parents = True, exist_ok = True)
    except (OSError, ValueError):
        return False
    if not _holding_dir_is_safe(parent):
        return False
    try:
        os.mkdir(path, 0o700)
        return True
    except FileExistsError:
        pass
    except (OSError, ValueError):
        return False
    try:
        info = os.lstat(path)
    except (OSError, ValueError):
        return False
    if not stat_module.S_ISDIR(info.st_mode) or stat_module.S_ISLNK(info.st_mode):
        return False
    if os.name == "nt":
        return True
    if info.st_uid != os.geteuid():
        return False
    return not info.st_mode & (stat_module.S_IWGRP | stat_module.S_IWOTH)


def _windows_temp_root_is_private(parent: Path) -> bool:
    """Accepts only the default LOCALAPPDATA temp root, which Windows already ACLs per account."""
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return False
    try:
        return os.path.normcase(str(parent.resolve())) == os.path.normcase(
            str((Path(local) / "Temp").resolve())
        )
    except (OSError, ValueError):
        return False


def _dir_is_not_swappable(directory: Path) -> bool:
    """Whether an entry inside *directory* can be renamed away by another account."""
    try:
        info = os.stat(directory)
    except (OSError, ValueError):
        return False
    # Owner and mode checks are both required, neither substitutes: another account can chmod its own dir writable; root is trusted.
    if info.st_uid not in (0, os.geteuid()):
        return False
    # And the write bits, because owning a directory does not stop anyone else writing in it.
    if not info.st_mode & (stat_module.S_IWGRP | stat_module.S_IWOTH):
        return True
    # Sticky limits renames to owners, so a shared dir held by us or root is fine (/tmp).
    return bool(info.st_mode & stat_module.S_ISVTX)


def _holding_dir_is_safe(parent: Path) -> bool:
    """Checks every ancestor, both lexical and resolved, since any writable one enables a swap; symlinked roots are not refused (macOS /tmp)."""
    if os.name == "nt":
        return _windows_temp_root_is_private(parent)
    try:
        chains = (Path(os.path.abspath(parent)), parent.resolve())
    except (OSError, ValueError):
        return False
    for start in chains:
        current = start
        while True:
            if not _dir_is_not_swappable(current):
                return False
            if current.parent == current:
                break
            current = current.parent
    return True


def _parseable_toolchain_fallback(key: str, intended: str) -> str | None:
    """Keyed on the intended path and OS account so two accounts sharing an install get separate dirs; hex digest keeps the name parseable."""
    try:
        base = tempfile.gettempdir()
    except (OSError, ValueError):
        return None
    if not base or toolchain_path_unparseable(base):
        return None
    account = str(os.geteuid()) if hasattr(os, "geteuid") else (os.environ.get("USERNAME") or "")
    digest = hashlib.sha256(f"{account}\0{intended}".encode("utf-8", "replace")).hexdigest()[:12]
    candidate = str(Path(base) / f"unsloth-{key.lower().replace('_', '-')}-{digest}")
    # The join can still reintroduce one: gettempdir() is parseable but Path may normalise.
    return None if toolchain_path_unparseable(candidate) else candidate


def parseable_cache_fallback(key: str, intended: str) -> str | None:
    """Single entry point, so the diffusion cache gets the same answer; no shared fallback is published."""
    candidate = _parseable_toolchain_fallback(key, intended)
    if candidate is None:
        return None
    if not _private_dir(candidate) or not _usable_dir(candidate):
        return None
    return candidate


def _use_studio_uv_tools() -> None:
    """An install with a custom UNSLOTH_STUDIO_HOME keeps its own uv in <studio>/uv, off the user
    PATH, and its uv-managed Python in <studio>/python. Keyed on those directories, which only
    that installer creates, so nothing else opts in. Its uv goes first: the installer only places
    one there when PATH had none or one below its minimum."""
    root = studio_root()
    try:
        if root.resolve() == (Path.home() / ".unsloth" / "studio").resolve():
            return
    except (OSError, RuntimeError, ValueError):
        return
    if (root / "python").is_dir() and not (os.environ.get("UV_PYTHON_INSTALL_DIR") or "").strip():
        os.environ["UV_PYTHON_INSTALL_DIR"] = str(root / "python")
    uv_dir = root / "uv"
    if (uv_dir / ("uv.exe" if os.name == "nt" else "uv")).is_file():
        os.environ["PATH"] = _path_with_first(os.environ.get("PATH", ""), str(uv_dir))


def _path_with_first(path: str, first: str) -> str:
    rest = [entry for entry in path.split(os.pathsep) if entry and entry != first]
    return os.pathsep.join([first, *rest])


def _setup_cache_env() -> None:
    """Set cache env vars for HuggingFace, uv, and vLLM.

    Explicit Hugging Face environment variables take precedence over Unsloth's
    stored location. Unsloth seeds import-time variables once, while each later
    worker receives its own captured cache location.
    """
    _use_studio_uv_tools()
    root = cache_root()
    from utils.hf_cache_settings import initialize_hf_cache_environment

    initialize_hf_cache_environment()
    defaults: dict[str, str] = {
        "UV_CACHE_DIR": str(root / "uv"),
        "VLLM_CACHE_ROOT": str(root / "vllm"),
        # unsloth_zoo defaults to a cwd-relative name (user home on Windows). Set before
        # unsloth_zoo.compiler imports: it reads the value at import time.
        "UNSLOTH_COMPILE_LOCATION": str(root.parent / "compiled_cache"),
        # Regenerable and process-scoped; shared user data stays put except in portable mode.
        "TORCHINDUCTOR_CACHE_DIR": str(root / "torchinductor"),
        # Tagged: torch adds its ABI folder only when unset, so runtimes would share .so files.
        "TORCH_EXTENSIONS_DIR": str(root / "torch-extensions" / _torch_runtime_tag()),
        # NVIDIA's JIT compile cache; ~/.nv/ComputeCache otherwise.
        "CUDA_CACHE_PATH": str(root / "cuda"),
        "NUMBA_CACHE_DIR": str(root / "numba"),
    }
    defaults.update(_matplotlib_defaults(root))
    defaults.update(_triton_cache_defaults(root))
    defaults.update(_data_designer_defaults(root))
    defaults.update(_portable_cache_defaults(root))
    for key, value in defaults.items():
        # Blank counts as unset: KEY= would put an empty entry on sys.path.
        inherited = (os.environ.get(key) or "").strip()
        # Explicit value wins unless builders cannot parse it; Windows persists stale values, so refuse at use time (setup.ps1 checks provenance before deleting).
        if inherited and _toolchain_unsafe(key, inherited):
            # Seeded from the default so healed and clean installs share one directory.
            fallback = parseable_cache_fallback(key, value)
            logger.debug(
                "refusing inherited %s=%s: the C++ builders cannot paste it into a command line "
                "unquoted; using %s",
                key,
                inherited,
                fallback if fallback is not None else "no pin at all",
            )
            if fallback is not None:
                os.environ[key] = fallback
            else:
                os.environ.pop(key, None)
            continue
        if not inherited:
            if _toolchain_unsafe(key, value):
                # Unset is not safe: torch's default tempdir path keeps quotes and spaces from the login.
                fallback = parseable_cache_fallback(key, value)
                if fallback is not None:
                    logger.debug(
                        "%s holds a character the C++ builders cannot paste into a command "
                        "line unquoted; pinning %s to %s instead",
                        value,
                        key,
                        fallback,
                    )
                    os.environ[key] = fallback
                    continue
                logger.debug(
                    "leaving %s unset: %s holds a character the C++ builders cannot paste "
                    "into a command line unquoted, and the temporary directory is no better",
                    key,
                    value,
                )
                # Popped, not blanked: Inductor reads whitespace as a relative path.
                os.environ.pop(key, None)
                continue
            os.environ[key] = value
            # Best-effort: a non-writable custom HF_HOME must not crash startup
            try:
                created = True
                try:
                    Path(value).mkdir(parents = True, exist_ok = False)
                except FileExistsError:
                    created = False
                if key == "UNSLOTH_COMPILE_LOCATION" and created:
                    # The marker licenses cleanup rmtree, so write it only when this call made the dir.
                    from utils.cache_cleanup import CACHE_MARKER
                    (Path(value) / CACHE_MARKER).touch(exist_ok = True)
            except (OSError, ImportError):
                pass
            # A toolchain path we could not create is worse than none, since torch would fail every compile; applies only to toolchain keys, not data roots.
            if key in _TOOLCHAIN_PATH_KEYS and not _usable_dir(value):
                logger.debug("leaving %s unset: %s is not a usable directory", key, value)
                os.environ.pop(key, None)


def setup_cache_env() -> None:
    """Seeds cache env without creating studio dirs, for uvicorn main:app (skips run.py); must run before unsloth_zoo.compiler is imported."""
    _setup_cache_env()


def ensure_studio_directories() -> None:
    """Create all standard studio directories on startup."""
    for dir_fn in (
        studio_root,
        assets_root,
        datasets_root,
        dataset_uploads_root,
        recipe_datasets_root,
        unstructured_uploads_root,
        outputs_root,
        exports_root,
        auth_root,
        tensorboard_root,
    ):
        ensure_dir(dir_fn())
    _setup_cache_env()


def _clean_relative_path(path_value: str, *, strip_prefixes: tuple[str, ...] = ()) -> Path:
    path = Path(path_value).expanduser()
    parts = [part for part in path.parts if part not in ("", ".")]
    while parts and parts[0] in strip_prefixes:
        parts = parts[1:]
    return Path(*parts) if parts else Path()


def _has_parent_segment(raw: str, path: Path) -> bool:
    """On POSIX, Path misses backslash-separated .. segments, so Windows-style parsing is checked too."""
    if ".." in path.parts:
        return True
    if ".." in PureWindowsPath(raw).parts:
        return True
    return ".." in raw.replace("\\", "/").split("/")


def _is_absolute_user_path(path: Path) -> bool:
    expanded = str(path)
    if os.name == "nt":
        return path.is_absolute() and PureWindowsPath(expanded).is_absolute()
    return path.is_absolute() and PurePosixPath(expanded).is_absolute()


def _assert_contained(resolved: Path, root: Path) -> None:
    """Raise ValueError if ``resolved`` realpaths outside ``root``."""
    try:
        resolved_real = Path(os.path.realpath(resolved))
        root_real = Path(os.path.realpath(root))
    except OSError as exc:
        raise ValueError(f"path resolution failed: {exc}") from exc
    try:
        resolved_real.relative_to(root_real)
    except ValueError as exc:
        raise ValueError(
            f"path escapes root: {resolved!s} -> {resolved_real!s} is not under {root_real!s}"
        ) from exc


def within_account(path: Path) -> bool:
    if is_owner_context():
        return True
    try:
        real = Path(os.path.realpath(path))
    except OSError:
        return False
    for root in (workspace_root(), project_workspaces_root(), tmp_root()):
        try:
            real.relative_to(Path(os.path.realpath(root)))
            return True
        except ValueError:
            continue
    return False


def own_entry(path: Path) -> bool:
    return path.exists() and within_account(path)


def require_within_account(path: Path) -> Path:
    if not within_account(path):
        raise ValueError(f"path escapes the account workspace: {path!s}")
    return path


def resolve_under_root(
    path_value: str | None,
    *,
    root: Path,
    strip_prefixes: tuple[str, ...] = (),
) -> Path:
    """Absolute paths are accepted only if already under root, so re-resolving paths is idempotent."""
    if not path_value or not str(path_value).strip():
        return root

    raw = str(path_value).strip()
    if "\x00" in raw:
        raise ValueError("path may not contain null bytes")

    path = Path(raw).expanduser()
    if _has_parent_segment(raw, path):
        raise ValueError(f"path may not contain '..' segments: {raw!r}")

    if _is_absolute_user_path(path):
        _assert_contained(path, root)
        return path

    cleaned = _clean_relative_path(raw, strip_prefixes = strip_prefixes)
    candidate = root / cleaned
    _assert_contained(candidate, root)
    return candidate


def default_run_dir_name(model_name: str) -> str:
    # Local paths collapse to their final component so a source cannot escape outputs_root.
    raw = str(model_name or "").strip()
    is_path = (
        "\\" in raw
        or raw.startswith(("/", "~", "."))
        or os.path.isabs(raw)
        or (len(raw) >= 2 and raw[1] == ":")
    )
    base = PureWindowsPath(raw).name if is_path else raw.replace("/", "_")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)[:200].strip("._-")
    return base or "model"


def resolve_output_dir(path_value: str | None = None) -> Path:
    return resolve_under_root(
        path_value,
        root = outputs_root(),
        strip_prefixes = ("outputs",),
    )


def resolve_export_dir(path_value: str | None = None) -> Path:
    """Read-side export lookup, contained under exports_root(); writes use resolve_export_write_dir."""
    return resolve_under_root(
        path_value,
        root = exports_root(),
        strip_prefixes = ("exports",),
    )


def resolve_export_write_dir(path_value: str | None = None) -> Path:
    """Export write path: absolute paths pass through unchanged so a save can target another drive."""
    if not path_value or not str(path_value).strip():
        return exports_root()
    raw = str(path_value).strip()
    if "\x00" in raw:
        raise ValueError("path may not contain null bytes")
    path = Path(raw).expanduser()
    if _has_parent_segment(raw, path):
        raise ValueError(f"path may not contain '..' segments: {raw!r}")
    if _is_absolute_user_path(path):
        return require_within_account(path)
    return resolve_under_root(
        path_value,
        root = exports_root(),
        strip_prefixes = ("exports",),
    )


def resolve_tensorboard_dir(path_value: str | None = None) -> Path:
    return resolve_under_root(
        path_value,
        root = tensorboard_root(),
        strip_prefixes = ("runs", "tensorboard"),
    )


def dataset_files_in_dir(directory: Path) -> list[Path]:
    """Loadable dataset files for *directory*, preferring a ``parquet-files/`` export over the
    directory's own files. Raises ``ValueError`` when it holds no supported format."""
    parquet_dir = directory / "parquet-files"
    if not parquet_dir.exists():
        parquet_dir = directory
    parquet = drop_appledouble_metadata(sorted(parquet_dir.glob("*.parquet")))
    if parquet:
        return parquet
    files: list[Path] = []
    for ext in (".json", ".jsonl", ".csv", ".parquet"):
        files.extend(drop_appledouble_metadata(sorted(directory.glob(f"*{ext}"))))
    if not files:
        raise ValueError(f"No supported data files in directory: {directory}")
    return files


def resolve_dataset_path(path_value: str) -> Path:
    raw = str(path_value or "").strip()
    if "\x00" in raw:
        raise ValueError("dataset path may not contain null bytes")
    path = Path(raw).expanduser()
    if ".." in path.parts:
        raise ValueError(f"dataset path may not contain '..' segments: {raw!r}")
    if path.is_absolute():
        for root_fn in (datasets_root, dataset_uploads_root, recipe_datasets_root):
            try:
                _assert_contained(path, root_fn())
                return require_within_account(path)
            except ValueError:
                continue
        raise ValueError(f"dataset path must be relative or under a dataset root: {raw!r}")

    parts = [part for part in Path(path_value).parts if part not in ("", ".")]
    if parts[:2] == ["assets", "datasets"]:
        parts = parts[2:]
    if parts and parts[0] == "uploads":
        cleaned = Path(*parts[1:]) if len(parts) > 1 else Path()
        return require_within_account(dataset_uploads_root() / cleaned)
    if parts and parts[0] == "recipes":
        cleaned = Path(*parts[1:]) if len(parts) > 1 else Path()
        return require_within_account(recipe_datasets_root() / cleaned)

    cleaned = Path(*parts) if parts else Path()
    candidates = [
        dataset_uploads_root() / cleaned,
        recipe_datasets_root() / cleaned,
        datasets_root() / cleaned,
        dataset_uploads_root() / cleaned.name,
        recipe_datasets_root() / cleaned.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return require_within_account(candidate)
    return candidates[0]
