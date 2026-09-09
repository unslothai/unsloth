# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Size and empty the caches this install fills up, and nothing else.

Every deletable location is named by a KEY here, never by a path the caller
sent: the API takes ``uv`` or ``hf_hub``, and this module is the only thing that
turns that into a directory. A caller therefore cannot ask for "/" or "~", and
the answer to "may this be deleted" is decided from the resolved directory, not
from what was asked for.

The rules a purge obeys, in the order they are checked:

* The key must be one of ``CACHE_DEFINITIONS``.
* Its resolved root must be an absolute existing directory, at least two
  components below the filesystem anchor, and must be neither a symlink nor a
  Windows junction.
* The root must not be, or contain, anything in ``protected_paths()``: the
  studio home, studio.db, projects, models, datasets, outputs, exports, auth,
  the Hugging Face cache HOME (which holds the token), or a managed asset home
  such as DATA_DESIGNER_HOME.
* The root must not contain another key's root whose own clear is narrower than
  emptying it: an opt-in cache, or a pattern-limited one that keeps the files
  which are not cache. Either is only ever cleared when it is asked for by name.
* The root is emptied, never removed, so nothing recreates it at the wrong
  place with the wrong permissions.
* A symlink inside the root is unlinked, never followed, and a directory whose
  real path leaves the root is skipped rather than removed.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from loggers import get_logger

logger = get_logger(__name__)

# Groups the UI renders under one heading each.
GROUP_PACKAGES = "packages"
GROUP_COMPILE = "compile"
GROUP_MODELS = "models"


class CachePurgeRefused(RuntimeError):
    """A cache root failed a safety check, so nothing was deleted from it."""


@dataclass(frozen = True)
class CacheDefinition:
    key: str
    group: str
    # True when emptying this costs a re-download of something the user chose to
    # fetch. Never part of a bulk purge; the UI asks for it on its own.
    opt_in: bool
    resolve: Callable[[], list[Path]]
    # When set, only top-level entries matching one of these globs are measured
    # or deleted. For roots that hold configuration next to cache files.
    patterns: Optional[tuple[str, ...]] = None
    # Cleared through its own module rather than by emptying the directory.
    custom_purge: Optional[Callable[[], "PurgeOutcome"]] = None
    # Sized by its own rule, when a plain walk of the roots would not match what
    # a clear removes.
    custom_measure: Optional[Callable[[], "tuple[int, int]"]] = None


@dataclass
class PurgeOutcome:
    freed_bytes: int = 0
    removed_entries: int = 0
    errors: Optional[list[str]] = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


# ---------------------------------------------------------------------------
# Platform cache locations
# ---------------------------------------------------------------------------


def _home() -> Path:
    return Path.home()


def _env_dir(name: str) -> Optional[Path]:
    value = (os.environ.get(name) or "").strip()
    if not value:
        return None
    try:
        return Path(value).expanduser()
    except (OSError, RuntimeError, ValueError):
        return None


def _is_windows() -> bool:
    """The seam the path-layout branches below test on.

    ``os.name`` cannot be faked for them: pathlib reads it too, so patching it
    makes Path itself try to build a WindowsPath on a POSIX host.
    """
    return os.name == "nt"


def _local_app_data() -> Path:
    value = (os.environ.get("LOCALAPPDATA") or "").strip()
    if value:
        return Path(value)
    return _home() / "AppData" / "Local"


def _platform_cache_dir(name: str, *, windows_tail: str = "Cache") -> Path:
    """Where a tool that follows platform convention keeps *name*'s cache."""
    if _is_windows():
        return _local_app_data() / name / windows_tail
    if sys.platform == "darwin":
        return _home() / "Library" / "Caches" / name
    xdg = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    base = Path(xdg).expanduser() if xdg else _home() / ".cache"
    return base / name


def _first(*candidates: Optional[Path]) -> list[Path]:
    for candidate in candidates:
        if candidate is not None:
            return [candidate]
    return []


# ---------------------------------------------------------------------------
# Per-cache resolvers. Each returns the roots to measure, existing or not.
# ---------------------------------------------------------------------------


def _uv_dirs() -> list[Path]:
    return _first(_env_dir("UV_CACHE_DIR"), _platform_cache_dir("uv"))


_UNPROBED = object()
_pip_configured: object = _UNPROBED


def _pip_configured_dir() -> Optional[Path]:
    """pip's effective cache directory, asked of pip itself.

    ``cache-dir`` in pip.conf moves the cache, and the fallback pip commands in
    core/training/worker.py and utils/wheel_utils.py do not pass --isolated, so
    they honour it. Reimplementing pip's five config kinds, their per-platform
    and legacy locations, and the way PIP_CONFIG_FILE suppresses the user file
    would give this install a second, weaker answer; ``pip cache dir`` is the
    first-hand one.

    Probed once per process. A config file does not change under a running
    backend, and this sits on a read the Resources tab makes.
    """
    from utils.child_stdio import utf8_child_env

    global _pip_configured
    if _pip_configured is not _UNPROBED:
        return _pip_configured  # type: ignore[return-value]
    _pip_configured = None
    try:
        done = subprocess.run(
            [sys.executable, "-m", "pip", "cache", "dir"],
            capture_output = True,
            text = True,
            # The child picks its stdout encoding from the locale, which is the
            # ANSI codepage on Windows and ASCII under a C locale, so a path
            # with non-ASCII in it would come back mangled either way. Tell the
            # child to emit the UTF-8 this decodes, as the other spawns here do.
            encoding = "utf-8",
            errors = "replace",
            env = utf8_child_env(),
            timeout = 20,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logger.debug(f"Could not ask pip for its cache directory: {exc}")
        return None
    reported = done.stdout.strip() if done.returncode == 0 else ""
    if reported:
        _pip_configured = Path(reported).expanduser()
    return _pip_configured  # type: ignore[return-value]


def _pip_dirs() -> list[Path]:
    return _first(_env_dir("PIP_CACHE_DIR"), _pip_configured_dir(), _platform_cache_dir("pip"))


def _npm_dirs() -> list[Path]:
    # Only _cacache is the cache: ~/.npm also holds logs npm writes about
    # failures, which are the user's diagnostics rather than ours to drop.
    configured = _env_dir("npm_config_cache") or _env_dir("NPM_CONFIG_CACHE")
    if configured is not None:
        return [configured / "_cacache"]
    if _is_windows():
        return [_local_app_data() / "npm-cache" / "_cacache"]
    return [_home() / ".npm" / "_cacache"]


def _bun_dirs() -> list[Path]:
    return _first(_env_dir("BUN_INSTALL_CACHE_DIR"), _home() / ".bun" / "install" / "cache")


def _hf_paths():
    from utils.hf_cache_settings import get_hf_cache_paths
    return get_hf_cache_paths()


def _hf_homes() -> list[Path]:
    """Every cache home this install has been pointed at, for PROTECTION only.

    Never for deletion. The history behind it is appended to by
    ``PUT /api/settings/hugging-face-cache``, which an API key may call, while
    the purge endpoint refuses one. Resolving a purge root through it would let
    a caller that cannot delete choose what a later clear deletes, which is the
    one thing the key-not-path rule at the top of this module exists to stop.
    Protecting more locations than are cleared is safe in that direction.
    """
    from utils.hf_cache_settings import known_hf_cache_homes
    return list(known_hf_cache_homes())


def _hf_hub_dirs() -> list[Path]:
    return [_hf_paths().hub_cache]


def _hf_child_dirs(name: str, configured: Optional[Path]) -> list[Path]:
    # HF reads the variable INSTEAD of <home>/<name>, so these are alternatives.
    # The effective home, not the displayed one: an explicit HF_HUB_CACHE makes
    # the display home the hub's parent, which is somebody else's directory and
    # holds none of these children.
    from utils.hf_cache_settings import effective_cache_home
    return _first(configured, effective_cache_home() / name)


def _studio_datasets_fallback() -> Optional[Path]:
    """The Studio cache a dataset load retries in when HF's is not writable."""
    try:
        from utils.paths.storage_roots import cache_root
        return _safe_resolve(cache_root() / "hf-datasets")
    except Exception as exc:  # noqa: BLE001 - a broken root must not widen the list
        logger.debug(f"Could not resolve the Studio dataset fallback cache: {exc}")
        return None


def _hf_datasets_dirs() -> list[Path]:
    configured = _env_dir("HF_DATASETS_CACHE")
    fallback = _studio_datasets_fallback()
    if configured is not None and fallback is not None and _safe_resolve(configured) == fallback:
        # cache_safe._retry_in_studio_cache points HF_DATASETS_CACHE at that
        # cache for the length of one load, in THIS process, and load_dataset is
        # writing Arrow files and lock state there while it does. It is not one
        # of the caches this inventory offers, so a scoped override must not
        # turn it into a purge root under a load that is still running.
        configured = None
    return _hf_child_dirs("datasets", configured)


def _hf_assets_dirs() -> list[Path]:
    return _hf_child_dirs("assets", _env_dir("HF_ASSETS_CACHE"))


def _hf_xet_dirs() -> list[Path]:
    return [_hf_paths().xet_cache]


def _torch_inductor_dirs() -> list[Path]:
    configured = _env_dir("TORCHINDUCTOR_CACHE_DIR")
    if configured is not None:
        return [configured]
    import getpass
    import re
    import tempfile

    # torch/_inductor/runtime/cache_dir_utils.py, followed exactly: getpass.getuser
    # reads LOGNAME/USER/LNAME/USERNAME and then the pwd account name, so a
    # container with none of them set puts the cache at torchinductor_root while
    # a bare uid would look at torchinductor_0 and find nothing.
    try:
        user = getpass.getuser()
    except (KeyError, ModuleNotFoundError, OSError):
        try:
            user = f"uid_{os.getuid()}"
        except AttributeError:
            user = "unknown_user"
    sanitized = re.sub(r'[\\/:*?"<>|]', "_", user)
    return [Path(tempfile.gettempdir()) / f"torchinductor_{sanitized}"]


def _torch_extensions_dirs() -> list[Path]:
    return _first(_env_dir("TORCH_EXTENSIONS_DIR"), _platform_cache_dir("torch_extensions"))


def _triton_dirs() -> list[Path]:
    return _first(_env_dir("TRITON_CACHE_DIR"), _home() / ".triton" / "cache")


def _cuda_dirs() -> list[Path]:
    configured = _env_dir("CUDA_CACHE_PATH")
    if configured is not None:
        return [configured]
    # The CUDA programming guide's defaults, which follow no shared convention:
    # ROAMING AppData on Windows, Application Support on macOS, ~/.nv elsewhere.
    if _is_windows():
        roaming = (os.environ.get("APPDATA") or "").strip()
        base = Path(roaming) if roaming else _home() / "AppData" / "Roaming"
        return [base / "NVIDIA" / "ComputeCache"]
    if sys.platform == "darwin":
        return [_home() / "Library" / "Application Support" / "NVIDIA" / "ComputeCache"]
    return [_home() / ".nv" / "ComputeCache"]


def _numba_dirs() -> list[Path]:
    # Unset, numba tries __pycache__ next to the source it compiles, which is
    # not a directory of ours to empty. When that is not writable, which is any
    # install owned by another account, UserWideCacheLocator falls back to
    # AppDirs(appname = "numba", appauthor = False).user_cache_dir, and that one
    # is numba's alone.
    return _first(_env_dir("NUMBA_CACHE_DIR"), _platform_cache_dir("numba"))


def _matplotlib_dirs() -> list[Path]:
    """matplotlib.get_cachedir()'s own rules, which are XDG on Linux only.

    _get_config_or_cache_dir takes the XDG branch for linux and freebsd and
    otherwise uses ~/.matplotlib, with Windows preferring %LOCALAPPDATA%\\matplotlib
    when that legacy directory does not already exist. The platform convention
    helper agrees on none of that off Linux.
    """
    configured = _env_dir("MPLCONFIGDIR")
    if configured is not None:
        return [configured]
    legacy = _home() / ".matplotlib"
    if _is_windows():
        if legacy.is_dir():
            return [legacy]
        local = (os.environ.get("LOCALAPPDATA") or "").strip()
        return [Path(local) / "matplotlib"] if local else [legacy]
    if sys.platform == "darwin":
        return [legacy]
    xdg = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    base = Path(xdg).expanduser() if xdg else _home() / ".cache"
    return [base / "matplotlib"]


MATPLOTLIB_PATTERNS = ("fontlist-*.json", "tex.cache", "ttfcache")


def _matplotlib_patterns() -> Optional[tuple[str, ...]]:
    # Only the XDG branch gives matplotlib a cache dir of its own. MPLCONFIGDIR
    # and the ~/.matplotlib default are both get_configdir() as well, so
    # matplotlibrc sits beside the font list and only the cache entries may go.
    merged = _env_dir("MPLCONFIGDIR") is not None or _is_windows() or sys.platform == "darwin"
    return MATPLOTLIB_PATTERNS if merged else None


def _vllm_dirs() -> list[Path]:
    # vllm/envs.py: XDG_CACHE_HOME or ~/.cache, then "vllm", on every platform.
    # The platform helper agrees on Linux and diverges on macOS and Windows.
    xdg = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    base = Path(xdg).expanduser() if xdg else _home() / ".cache"
    return _first(_env_dir("VLLM_CACHE_ROOT"), base / "vllm")


def _unsloth_compiled_dirs() -> list[Path]:
    from utils.cache_cleanup import _cleanable_cache_dirs
    return [directory for directory, _dedicated in _cleanable_cache_dirs()]


def _measure_unsloth_compiled() -> tuple[int, int]:
    """Size only what a clear would actually remove.

    A directory Unsloth created goes whole; one it merely wrote into keeps
    everything but the generated modules, so counting all of it would promise
    space the clear cannot free.
    """
    from utils.cache_cleanup import _cleanable_cache_dirs

    total = 0
    entries = 0
    for directory, dedicated in _cleanable_cache_dirs():
        size, count = _measure_root(
            directory, patterns = None if dedicated else _GENERATED_MODULE_PATTERNS
        )
        total += size
        entries += count
    return total, entries


def _purge_unsloth_compiled() -> PurgeOutcome:
    """Clear the compiled cache through the module that owns it.

    cache_cleanup already knows which of these directories Unsloth created and
    which merely hold files it generated, and it serializes against a sibling
    backend that may be compiling right now. Re-deriving either here would give
    this install a second, weaker answer.
    """
    from utils.cache_cleanup import (
        LOCK_BUSY,
        _cleanable_cache_dirs,
        clear_unsloth_compiled_cache,
        compiled_cache_lock,
    )

    outcome = PurgeOutcome()
    with compiled_cache_lock() as lock_state:
        if lock_state == LOCK_BUSY:
            outcome.errors.append(
                "Another Unsloth backend is using the compiled cache, so it was left in place."
            )
            return outcome
        # clear_unsloth_compiled_cache is all-or-nothing across the directories
        # it owns, so one of them failing the guard stops the whole clear rather
        # than letting the others carry it through.
        protected = protected_paths()
        trees = protected_trees()
        keep = sheltered_roots("unsloth_compiled")
        for directory, dedicated in _cleanable_cache_dirs():
            if not dedicated:
                # Only generated module files are touched there, never the
                # directory's other contents.
                continue
            try:
                assert_purgeable_root(directory, protected = protected, trees = trees, keep = keep)
            except CachePurgeRefused as exc:
                outcome.errors.append(str(exc))
                return outcome
        before, entries = _measure_unsloth_compiled()
        clear_unsloth_compiled_cache()
        after, remaining = _measure_unsloth_compiled()
    outcome.freed_bytes = max(0, before - after)
    outcome.removed_entries = max(0, entries - remaining)
    if after > 0:
        # clear_unsloth_compiled_cache swallows every unlink and rmtree failure,
        # so a read-only directory or a locked file leaves the cache in place and
        # reports nothing. Bytes rather than entries: a dedicated cache is
        # recreated holding an empty marker file, which is not a leftover.
        outcome.errors.append(
            "Part of the compiled cache could not be removed and is still in place."
        )
    return outcome


_GENERATED_MODULE_PATTERNS = ("unsloth_compiled_module_*.py",)


CACHE_DEFINITIONS: tuple[CacheDefinition, ...] = (
    CacheDefinition("uv", GROUP_PACKAGES, False, _uv_dirs),
    CacheDefinition("pip", GROUP_PACKAGES, False, _pip_dirs),
    CacheDefinition("npm", GROUP_PACKAGES, False, _npm_dirs),
    CacheDefinition("bun", GROUP_PACKAGES, False, _bun_dirs),
    CacheDefinition("torch_inductor", GROUP_COMPILE, False, _torch_inductor_dirs),
    CacheDefinition("torch_extensions", GROUP_COMPILE, False, _torch_extensions_dirs),
    CacheDefinition("triton", GROUP_COMPILE, False, _triton_dirs),
    CacheDefinition("cuda", GROUP_COMPILE, False, _cuda_dirs),
    CacheDefinition("numba", GROUP_COMPILE, False, _numba_dirs),
    CacheDefinition("matplotlib", GROUP_COMPILE, False, _matplotlib_dirs),
    CacheDefinition("vllm", GROUP_COMPILE, False, _vllm_dirs),
    # Opt-in, not swept up by a bulk purge. The others cost a recompile or a
    # re-download; this one can break a job that is running RIGHT NOW. A training
    # or inference worker imports generated modules from here lazily, well after
    # it started, and compiled_cache_lock only serialises this against a sibling
    # BACKEND, not against a spawned worker holding no lock. Clearing it under a
    # running job forfeits hours of compute, which is not a trade to make on the
    # user's behalf from a button labelled "free up space".
    CacheDefinition(
        "unsloth_compiled",
        GROUP_COMPILE,
        True,
        _unsloth_compiled_dirs,
        custom_purge = _purge_unsloth_compiled,
        custom_measure = _measure_unsloth_compiled,
    ),
    CacheDefinition("hf_xet", GROUP_MODELS, False, _hf_xet_dirs),
    CacheDefinition("hf_assets", GROUP_MODELS, False, _hf_assets_dirs),
    # Both cost a re-download of something the user asked for, so neither is
    # ever swept up by a bulk purge.
    CacheDefinition("hf_datasets", GROUP_MODELS, True, _hf_datasets_dirs),
    CacheDefinition("hf_hub", GROUP_MODELS, True, _hf_hub_dirs),
)

CACHE_KEYS: tuple[str, ...] = tuple(definition.key for definition in CACHE_DEFINITIONS)

_DEFINITIONS_BY_KEY = {definition.key: definition for definition in CACHE_DEFINITIONS}

# One purge at a time per process: two concurrent sweeps of one root race each
# other into "file disappeared" errors that look like failures.
_purge_lock = threading.Lock()


def definition_for(key: str) -> CacheDefinition:
    try:
        return _DEFINITIONS_BY_KEY[key]
    except KeyError:
        raise ValueError(f"Unknown cache key: {key!r}") from None


# ---------------------------------------------------------------------------
# What may never be deleted
# ---------------------------------------------------------------------------


def _safe_resolve(path: Path) -> Optional[Path]:
    try:
        return Path(os.path.realpath(str(Path(path).expanduser())))
    except (OSError, RuntimeError, ValueError):
        return None


def _is_junction(path: Path | str) -> bool:
    """True for a Windows directory junction or volume mount point.

    A junction is the same hazard as a symlink and does not answer to the same
    test: since 3.8 only IO_REPARSE_TAG_SYMLINK sets S_IFLNK, so is_symlink()
    is False for a junction while realpath() still follows it to its target.
    Without this, UV_CACHE_DIR pointed at a junction would have the TARGET
    emptied. os.path.isjunction arrived in 3.12 and this package supports 3.9,
    so the reparse tag is read directly when it is missing.
    """
    isjunction = getattr(os.path, "isjunction", None)
    if isjunction is not None:
        try:
            return bool(isjunction(path))
        except (OSError, ValueError):
            return False
    if os.name != "nt":
        return False
    try:
        tag = getattr(os.lstat(path), "st_reparse_tag", 0)
    except (OSError, ValueError, AttributeError):
        return False
    return tag == getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)


def protected_paths() -> set[Path]:
    """Locations a purge must never delete, nor delete anything containing them.

    User data, not caches: the databases, the token, projects, datasets,
    outputs, exports, and the managed asset homes. The Hugging Face cache HOME
    is here while its ``hub`` child is purgeable, because the home also holds
    the access token.
    """
    from utils.paths.storage_roots import (
        assets_root,
        auth_db_path,
        auth_root,
        cache_root,
        datasets_root,
        documents_root,
        exports_root,
        outputs_root,
        project_workspaces_root,
        rag_root,
        studio_db_path,
        studio_root,
        tensorboard_root,
    )

    candidates: list[Path] = [
        Path.home(),
        studio_root(),
        studio_db_path(),
        auth_root(),
        auth_db_path(),
        assets_root(),
        datasets_root(),
        outputs_root(),
        exports_root(),
        rag_root(),
        tensorboard_root(),
        documents_root(),
        project_workspaces_root(),
        # Not a cache root of ours as a whole: it is the parent the per-tool
        # cache directories sit under, plus llama.cpp slot state.
        cache_root(),
    ]
    for name in ("DATA_DESIGNER_HOME", "UNSLOTH_STUDIO_PROJECTS_HOME", "UNSLOTH_STUDIO_HOME"):
        configured = _env_dir(name)
        if configured is not None:
            candidates.append(configured)
    try:
        for home in _hf_homes():
            candidates.append(home)
            # HF writes the access token here.
            candidates.append(home / "token")
        paths = _hf_paths()
        # cache_home is what Settings DISPLAYS, and an explicit HF_HUB_CACHE whose
        # basename is not "hub" (HF_HUB_CACHE=/mnt/hf-cache) makes that the hub
        # directory itself. Protecting it there would mark the model cache
        # permanently unclearable while protecting no credential: the token lives
        # in the HF home, which the loop above covers on its own.
        if _safe_resolve(paths.cache_home) != _safe_resolve(paths.hub_cache):
            candidates.append(paths.cache_home)
    except Exception as exc:  # noqa: BLE001 - a broken HF setting must not widen the allow-list
        logger.debug(f"Could not resolve the Hugging Face cache homes: {exc}")
        candidates.append(Path.home() / ".cache" / "huggingface")
    hf_home = _env_dir("HF_HOME")
    if hf_home is not None:
        candidates.append(hf_home)
        candidates.append(hf_home / "token")

    return _resolved_set(candidates)


def protected_trees() -> set[Path]:
    """Directories nothing inside may be deleted, however a variable is pointed.

    ``protected_paths`` stops a root that IS or CONTAINS user data; this stops a
    root BENEATH it, which is what an inherited ``TRITON_CACHE_DIR`` pointing
    into the projects folder would be. The studio home and the Hugging Face
    cache home are deliberately absent: real caches live inside both.
    """
    from utils.paths.storage_roots import (
        assets_root,
        auth_root,
        datasets_root,
        documents_root,
        exports_root,
        outputs_root,
        project_workspaces_root,
        rag_root,
        tensorboard_root,
    )

    candidates: list[Path] = [
        assets_root(),
        datasets_root(),
        outputs_root(),
        exports_root(),
        auth_root(),
        rag_root(),
        tensorboard_root(),
        project_workspaces_root(),
        # The real Documents folder. No tool keeps a cache under it, and a
        # variable pointed at ~/Documents/anything would otherwise empty it.
        documents_root(),
    ]
    configured = _env_dir("DATA_DESIGNER_HOME")
    if configured is not None:
        candidates.append(configured)
    return _resolved_set(candidates)


def _resolved_set(candidates: Iterable[Path]) -> set[Path]:
    resolved_paths: set[Path] = set()
    for candidate in candidates:
        resolved = _safe_resolve(candidate)
        if resolved is not None:
            resolved_paths.add(resolved)
    return resolved_paths


def _is_within(child: Path, parent: Path) -> bool:
    """True when *child* is *parent* or sits under it, comparing real paths."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def sheltered_roots(exclude_key: Optional[str] = None) -> dict[Path, str]:
    """Resolved roots another key's clear must not empty, mapped to why.

    Two kinds, and the reason is the same either way: their own clear is
    narrower than emptying the directory, so a clear that swallows them whole
    takes something that clear was written to keep.

    An opt-in cache costs a re-download, or a running job, so it is never swept
    up by a bulk clear. A pattern-limited root holds configuration next to the
    cache files (``MPLCONFIGDIR`` keeps matplotlibrc beside the font list), and
    only the matching entries are ever removed from it.

    Nothing stops a variable from putting either INSIDE another cache
    (``MPLCONFIGDIR=/cache/uv/matplotlib`` under ``UV_CACHE_DIR=/cache/uv``),
    and emptying the outer root would then delete it anyway. The key being
    cleared is excluded, so asking for a cache by name still clears it.
    """
    roots: dict[Path, str] = {}
    for definition in CACHE_DEFINITIONS:
        if definition.key == exclude_key:
            continue
        if definition.opt_in:
            reason = "the opt-in cache"
        elif _patterns_for(definition) is not None:
            reason = f"the {definition.key} cache"
        else:
            continue
        for root in _resolve_roots(definition):
            resolved = _safe_resolve(root)
            if resolved is not None:
                roots.setdefault(resolved, reason)
    return roots


def assert_purgeable_root(
    root: Path,
    *,
    protected: Optional[set[Path]] = None,
    trees: Optional[set[Path]] = None,
    keep: Optional[dict[Path, str]] = None,
) -> Path:
    """Return the real path of *root*, or raise if emptying it is not allowed.

    The single gate every deletion in this module goes through. It answers from
    the resolved directory, so a cache variable pointed at a symlink cannot
    smuggle in a target that would fail these checks.
    """
    protected = protected_paths() if protected is None else protected
    trees = protected_trees() if trees is None else trees
    raw = Path(root)
    if not raw.is_absolute():
        raise CachePurgeRefused(f"Cache root is not an absolute path: {raw}")
    # The link itself, not its target: emptying through a link deletes files at
    # a location nobody named, and unlinking it would remove the user's link.
    if raw.is_symlink():
        raise CachePurgeRefused(f"Cache root is a symlink: {raw}")
    if _is_junction(raw):
        raise CachePurgeRefused(f"Cache root is a junction: {raw}")
    resolved = _safe_resolve(raw)
    if resolved is None:
        raise CachePurgeRefused(f"Cache root cannot be resolved: {raw}")
    if len(resolved.parts) < 3:
        # "/", "/home", "C:\\", "C:\\Users": nothing that shallow is a cache.
        raise CachePurgeRefused(f"Cache root is too close to the filesystem root: {resolved}")
    if not resolved.is_dir():
        raise CachePurgeRefused(f"Cache root is not a directory: {resolved}")
    for reserved in protected:
        if resolved == reserved:
            raise CachePurgeRefused(f"Refusing to delete a protected location: {resolved}")
        if _is_within(reserved, resolved):
            raise CachePurgeRefused(
                f"Refusing to empty {resolved}: it contains the protected location {reserved}"
            )
    for tree in trees:
        if _is_within(resolved, tree):
            raise CachePurgeRefused(
                f"Refusing to empty {resolved}: it sits inside the protected folder {tree}"
            )
    for reserved, kind in ({} if keep is None else keep).items():
        if resolved == reserved or _is_within(reserved, resolved):
            raise CachePurgeRefused(
                f"Refusing to empty {resolved}: it holds {kind} {reserved}, "
                "which is only ever cleared when it is asked for by name"
            )
    return resolved


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def _matching(name: str, patterns: Optional[Iterable[str]]) -> bool:
    if patterns is None:
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _entry_size(entry: os.DirEntry, seen: set) -> int:
    try:
        stat = entry.stat(follow_symlinks = False)
    except OSError:
        return 0
    if stat.st_nlink > 1:
        # The Hugging Face cache hardlinks blobs on filesystems without symlink
        # support, so the same bytes appear under several names.
        key = (stat.st_dev, stat.st_ino)
        if key in seen:
            return 0
        seen.add(key)
    return int(stat.st_size)


def _descendable(entry: os.DirEntry) -> bool:
    """True for a directory entry a size walk may descend into.

    ``is_dir`` is True for a Windows junction while ``is_symlink`` is not, so
    without this the walk sizes the junction's target instead of the cache, and a
    junction back to an ancestor never terminates. Free on POSIX, where both
    ``os.path.isjunction`` and the fallback answer without a syscall.
    """
    return not _is_junction(entry.path)


def _measure_tree(path: Path, seen: set) -> tuple[int, int]:
    """Bytes and entry count under *path*, never following a symlink out."""
    total = 0
    count = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    count += 1
                    if entry.is_symlink():
                        # A link's own bytes are its target's, counted where the
                        # target lives (HF snapshots link into blobs).
                        continue
                    if entry.is_dir(follow_symlinks = False):
                        if _descendable(entry):
                            stack.append(Path(entry.path))
                        continue
                    total += _entry_size(entry, seen)
        except OSError:
            continue
    return total, count


def _measure_root(root: Path, *, patterns: Optional[Iterable[str]] = None) -> tuple[int, int]:
    """Bytes and top-level entry count for one cache root."""
    seen: set = set()
    total = 0
    entries = 0
    try:
        if Path(root).is_symlink() or _is_junction(root) or not Path(root).is_dir():
            return 0, 0
    except OSError:
        return 0, 0
    try:
        with os.scandir(root) as scan:
            for entry in scan:
                if not _matching(entry.name, patterns):
                    continue
                entries += 1
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks = False):
                    if _descendable(entry):
                        size, _ = _measure_tree(Path(entry.path), seen)
                        total += size
                    continue
                total += _entry_size(entry, seen)
    except OSError as exc:
        logger.debug(f"Could not size {root}: {exc}")
    return total, entries


def _resolve_roots(definition: CacheDefinition) -> list[Path]:
    """Existing, deduplicated roots for one definition, in resolver order."""
    try:
        candidates = definition.resolve()
    except Exception as exc:  # noqa: BLE001 - one broken resolver must not break the report
        logger.debug(f"Could not resolve the {definition.key} cache: {exc}")
        return []
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            path = Path(candidate).expanduser()
            if not path.is_dir():
                continue
        except OSError:
            continue
        resolved = _safe_resolve(path)
        key = os.path.normcase(str(resolved if resolved is not None else path))
        if key in seen:
            continue
        seen.add(key)
        roots.append(path)
    return roots


def describe_cache(definition: CacheDefinition) -> dict:
    """Size one cache and say whether it may be purged, without deleting."""
    patterns = _patterns_for(definition)
    roots = _resolve_roots(definition)
    purgeable = True
    blocked_reason: Optional[str] = None
    # Gate first, then measure. A root the gate refuses (UV_CACHE_DIR=/ or the
    # home directory) would otherwise be walked recursively on every open of the
    # Resources tab before the refusal is reached, and none of those bytes can be
    # reclaimed anyway.
    measurable = list(roots)
    if roots and definition.custom_purge is None:
        protected = protected_paths()
        trees = protected_trees()
        keep = sheltered_roots(definition.key)
        measurable = []
        for root in roots:
            try:
                assert_purgeable_root(root, protected = protected, trees = trees, keep = keep)
            except CachePurgeRefused as exc:
                if purgeable:
                    purgeable = False
                    blocked_reason = str(exc)
                continue
            measurable.append(root)
    if definition.custom_measure is not None:
        total, entries = definition.custom_measure()
    else:
        total = 0
        entries = 0
        for root in measurable:
            size, count = _measure_root(root, patterns = patterns)
            total += size
            entries += count
    return {
        "key": definition.key,
        "group": definition.group,
        "opt_in": definition.opt_in,
        "paths": [str(root) for root in roots],
        "size_bytes": total,
        "entry_count": entries,
        "present": bool(roots),
        "purgeable": bool(roots) and purgeable,
        "blocked_reason": blocked_reason,
    }


def _patterns_for(definition: CacheDefinition) -> Optional[tuple[str, ...]]:
    if definition.key == "matplotlib":
        return _matplotlib_patterns()
    if definition.key == "unsloth_compiled":
        return None
    return definition.patterns


# Sizing walks every file in every cache, which is seconds on a large uv or
# triton cache, so a repeat read inside this window reuses the last answer.
# A purge drops the entry it touched, so a size is never stale in the direction
# that would offer space that is already gone.
_INVENTORY_TTL_SECONDS = 60.0
_size_cache: dict[str, tuple[float, dict]] = {}
# Bumped by every invalidation, so a walk that began before one cannot store its
# answer after it. Dropping the entry is not enough on its own: a forced scan in
# one tab starts before a purge in another, finishes after it, and would install
# the pre-purge size for the rest of the window.
_size_epochs: dict[str, int] = {}
_size_cache_lock = threading.Lock()


def invalidate_cache_size(key: str) -> None:
    with _size_cache_lock:
        _size_cache.pop(key, None)
        _size_epochs[key] = _size_epochs.get(key, 0) + 1


def _described(definition: CacheDefinition, *, refresh: bool) -> dict:
    now = time.monotonic()
    with _size_cache_lock:
        began_at = _size_epochs.get(definition.key, 0)
        remembered = None if refresh else _size_cache.get(definition.key)
    if remembered is not None and now - remembered[0] < _INVENTORY_TTL_SECONDS:
        return remembered[1]
    entry = describe_cache(definition)
    with _size_cache_lock:
        if _size_epochs.get(definition.key, 0) == began_at:
            _size_cache[definition.key] = (time.monotonic(), entry)
    return entry


def cache_inventory(*, refresh: bool = False) -> dict:
    """Every known cache, its size, and the free space on the studio volume."""
    entries = [_described(definition, refresh = refresh) for definition in CACHE_DEFINITIONS]
    total = sum(entry["size_bytes"] for entry in entries if entry["present"])
    reclaimable = sum(
        entry["size_bytes"]
        for entry in entries
        if entry["present"] and entry["purgeable"] and not entry["opt_in"]
    )
    return {
        "caches": entries,
        "total_bytes": total,
        "reclaimable_bytes": reclaimable,
        "free_bytes": _free_bytes(),
        "total_disk_bytes": _total_disk_bytes(),
    }


def _usage():
    from utils.paths.storage_roots import studio_root
    for candidate in (studio_root(), Path.home(), Path(os.path.abspath(os.sep))):
        try:
            return shutil.disk_usage(candidate)
        except OSError:
            continue
    return None


def _free_bytes() -> Optional[int]:
    usage = _usage()
    return int(usage.free) if usage is not None else None


def _total_disk_bytes() -> Optional[int]:
    usage = _usage()
    return int(usage.total) if usage is not None else None


# ---------------------------------------------------------------------------
# Purging
# ---------------------------------------------------------------------------


def _remove_entry(entry: os.DirEntry, root: Path, outcome: PurgeOutcome, seen: set) -> None:
    path = Path(entry.path)
    try:
        if entry.is_symlink():
            # Unlink the link, never what it points at. This is the escape a
            # cache directory can be made to contain.
            size = 0
            os.unlink(path)
        elif entry.is_dir(follow_symlinks = False):
            real = _safe_resolve(path)
            if real is None or not _is_within(real, root):
                outcome.errors.append(f"Skipped {path}: it resolves outside the cache root")
                return
            size, _ = _measure_tree(path, seen)
            # rmtree refuses a symlinked directory and does not follow links it
            # finds inside, so the walk cannot leave *root*.
            shutil.rmtree(path)
        else:
            size = _entry_size(entry, seen)
            os.unlink(path)
    except OSError as exc:
        outcome.errors.append(f"Could not remove {path}: {exc}")
        return
    outcome.freed_bytes += size
    outcome.removed_entries += 1


def empty_cache_root(
    root: Path,
    *,
    patterns: Optional[Iterable[str]] = None,
    protected: Optional[set[Path]] = None,
    trees: Optional[set[Path]] = None,
    keep: Optional[set[Path]] = None,
) -> PurgeOutcome:
    """Delete the contents of one cache root, leaving the root itself in place."""
    outcome = PurgeOutcome()
    resolved = assert_purgeable_root(root, protected = protected, trees = trees, keep = keep)
    seen: set = set()
    try:
        with os.scandir(resolved) as scan:
            entries = list(scan)
    except OSError as exc:
        outcome.errors.append(f"Could not read {resolved}: {exc}")
        return outcome
    for entry in entries:
        if not _matching(entry.name, patterns):
            continue
        _remove_entry(entry, resolved, outcome, seen)
    return outcome


def purge_cache(key: str) -> dict:
    """Empty one cache by key. Never raises for a refusal; it reports it."""
    definition = definition_for(key)
    if definition.custom_purge is not None:
        outcome = definition.custom_purge()
        return _purge_result(definition, outcome)
    outcome = PurgeOutcome()
    patterns = _patterns_for(definition)
    protected = protected_paths()
    trees = protected_trees()
    keep = sheltered_roots(definition.key)
    for root in _resolve_roots(definition):
        try:
            root_outcome = empty_cache_root(
                root, patterns = patterns, protected = protected, trees = trees, keep = keep
            )
        except CachePurgeRefused as exc:
            logger.warning(f"Refusing to purge the {key} cache at {root}: {exc}")
            outcome.errors.append(str(exc))
            continue
        outcome.freed_bytes += root_outcome.freed_bytes
        outcome.removed_entries += root_outcome.removed_entries
        outcome.errors.extend(root_outcome.errors)
    return _purge_result(definition, outcome)


def _purge_result(definition: CacheDefinition, outcome: PurgeOutcome) -> dict:
    return {
        "key": definition.key,
        "freed_bytes": outcome.freed_bytes,
        "removed_entries": outcome.removed_entries,
        "errors": list(outcome.errors),
    }


# Emptying either of these removes the repositories the Hub inventory reports,
# so it is one of the app-driven mutations that scan is invalidated on. Without
# it the Hub and the model picker keep listing deleted models for the scan's TTL,
# and an immediate refetch is served the pre-purge answer.
_HF_SCANNED_KEYS = frozenset({"hf_hub", "hf_datasets"})


def _invalidate_hf_scans() -> None:
    try:
        from hub.utils.inventory_scan import invalidate_hf_cache_scans
    except ImportError as exc:
        logger.debug(f"Could not invalidate the Hugging Face scans: {exc}")
        return
    invalidate_hf_cache_scans()


def purge_caches(keys: Iterable[str]) -> dict:
    """Empty each named cache, then report what the inventory looks like after.

    Unknown keys raise before anything is deleted, so a request that names one
    cache this build does not have changes nothing at all.
    """
    requested = [str(key) for key in keys]
    if not requested:
        raise ValueError("Choose at least one cache to clear.")
    definitions = [definition_for(key) for key in dict.fromkeys(requested)]
    results = []
    with _purge_lock:
        for definition in definitions:
            logger.info(f"Clearing the {definition.key} cache")
            results.append(purge_cache(definition.key))
            # Its remembered size is what it held a moment ago, not what it
            # holds now.
            invalidate_cache_size(definition.key)
    if any(definition.key in _HF_SCANNED_KEYS for definition in definitions):
        _invalidate_hf_scans()
    return {
        "results": results,
        "freed_bytes": sum(result["freed_bytes"] for result in results),
        "inventory": cache_inventory(),
    }
