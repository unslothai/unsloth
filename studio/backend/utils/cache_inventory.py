# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Size and empty the caches this install fills up, and nothing else.

The API takes a KEY, never a path, and this module is the only thing that turns
one into a directory, so "may this be deleted" is answered from the resolved
directory rather than from what was asked for. The rules, in order:

* The key is one of ``CACHE_DEFINITIONS``.
* Its root is an absolute existing directory, at least two components below the
  filesystem anchor, and neither a symlink nor a Windows junction.
* It is not, and does not contain, anything in ``protected_paths()``.
* It does not sit inside anything in ``protected_trees()``.
* It does not contain another key's root whose own clear is narrower than
  emptying it (``sheltered_roots``).
* It is emptied, never removed, so nothing recreates it with the wrong
  permissions.
* A symlink inside it is unlinked rather than followed, and a directory whose
  real path leaves it is skipped.
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
    custom_purge: Optional[Callable[[], "PurgeOutcome"]] = None
    # Sized by its own rule, when a walk would not match what a clear removes.
    custom_measure: Optional[Callable[[], "tuple[int, int]"]] = None


@dataclass
class PurgeOutcome:
    freed_bytes: int = 0
    removed_entries: int = 0
    errors: Optional[list[str]] = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


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
    # A seam the path-layout branches can be tested on: patching os.name is not
    # one, because pathlib reads it and Path would build a WindowsPath on POSIX.
    return os.name == "nt"


def _local_app_data() -> Path:
    value = (os.environ.get("LOCALAPPDATA") or "").strip()
    if value:
        return Path(value)
    return _home() / "AppData" / "Local"


def _platform_cache_dir(name: str, *, windows_tail: str = "Cache") -> Path:
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


_UV_CACHE_MARKER = "uv-cache-dir"


def _recorded_uv_cache() -> Optional[Path]:
    """The uv cache the installer recorded, which updates keep filling.

    Parsed as unsloth_cli's _with_studio_uv_cache parses it: one record, one
    trailing delimiter, and no expanduser, since uv makes a literal "~" dir.
    """
    try:
        from utils.paths.storage_roots import cache_root
        recorded = (cache_root() / _UV_CACHE_MARKER).read_text(
            encoding = "utf-8-sig", errors = "surrogateescape"
        )
    except (OSError, ValueError, RuntimeError):
        return None
    recorded = recorded.removesuffix("\n").removesuffix("\r")
    if not recorded.strip():
        return None
    try:
        return Path(os.path.abspath(recorded))
    except (OSError, ValueError):
        return None


def _uv_dirs() -> list[Path]:
    # Both: _setup_cache_env seeds one, the installer may have recorded another,
    # and either can be the multi-gigabyte one.
    roots = _first(_env_dir("UV_CACHE_DIR"), _platform_cache_dir("uv"))
    recorded = _recorded_uv_cache()
    if recorded is not None:
        roots.append(recorded)
    return roots


# One answer per tool per process; a config file does not change under a running
# backend. The lock spans the probe, not just the store: recording the miss
# first let a second cold request read it as a finished failure and show the
# fallback path, while its Clear resolved the configured one.
_probed_cache_dirs: dict[str, Optional[Path]] = {}
_probe_lock = threading.Lock()


def _probe_tool_cache_dir(name: str, command: list[str]) -> Optional[Path]:
    """Ask a package manager where its own cache is.

    ``cache-dir`` in pip.conf and ``cache`` in .npmrc move it, no environment
    variable carries either, and Studio's invocations of both honour them.
    Reimplementing pip's five config kinds or npm's chain would drift.
    """
    from utils.child_stdio import utf8_child_env
    with _probe_lock:
        if name in _probed_cache_dirs:
            return _probed_cache_dirs[name]
        answer: Optional[Path] = None
        try:
            done = subprocess.run(
                command,
                capture_output = True,
                text = True,
                # The child would otherwise pick the locale encoding, the ANSI
                # codepage on Windows, and mangle a non-ASCII path.
                encoding = "utf-8",
                errors = "replace",
                env = utf8_child_env(),
                timeout = 20,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            logger.debug(f"Could not ask {name} for its cache directory: {exc}")
        else:
            reported = done.stdout.strip() if done.returncode == 0 else ""
            if reported:
                candidate = Path(reported).expanduser()
                # npm prints "undefined" rather than failing with no answer.
                if candidate.is_absolute():
                    answer = candidate
        _probed_cache_dirs[name] = answer
        return answer


def _pip_dirs() -> list[Path]:
    return _first(
        _env_dir("PIP_CACHE_DIR"),
        _probe_tool_cache_dir("pip", [sys.executable, "-m", "pip", "cache", "dir"]),
        _platform_cache_dir("pip"),
    )


def _npm_configured_dir() -> Optional[Path]:
    # Any npm on PATH will do: .npmrc is per user, not per install.
    npm = shutil.which("npm")
    return None if npm is None else _probe_tool_cache_dir("npm", [npm, "config", "get", "cache"])


# The package cache and the one npx installs into, which Studio fills whenever
# it launches an MCP server. Not the rest of ~/.npm: _logs is the user's.
_NPM_CACHE_CHILDREN = ("_cacache", "_npx")


def _npm_dirs() -> list[Path]:
    configured = _env_dir("npm_config_cache") or _env_dir("NPM_CONFIG_CACHE")
    if configured is None:
        configured = _npm_configured_dir()
    if configured is None:
        configured = _local_app_data() / "npm-cache" if _is_windows() else _home() / ".npm"
    return [configured / child for child in _NPM_CACHE_CHILDREN]


def _bun_dirs() -> list[Path]:
    return _first(_env_dir("BUN_INSTALL_CACHE_DIR"), _home() / ".bun" / "install" / "cache")


def _hf_paths():
    from utils.hf_cache_settings import get_hf_cache_paths
    return get_hf_cache_paths()


def _hf_homes() -> list[Path]:
    """Every cache home this install has been pointed at, for PROTECTION only.

    Never for deletion: an API key may append to that history through
    ``PUT /api/settings/hugging-face-cache`` while the purge route refuses one,
    so resolving a purge root through it would let a caller that cannot delete
    choose what a later clear takes. Protecting extra locations is safe.
    """
    from utils.hf_cache_settings import known_hf_cache_homes
    return list(known_hf_cache_homes())


def _hf_hub_dirs() -> list[Path]:
    return [_hf_paths().hub_cache]


def _hf_child_dirs(name: str, configured: Optional[Path]) -> list[Path]:
    # HF reads the variable INSTEAD of <home>/<name>, so these are alternatives,
    # and the home is the effective one rather than the one Settings displays.
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
        # cache_safe._retry_in_studio_cache points the variable there for the
        # length of one load, in THIS process, while load_dataset writes Arrow
        # files and lock state into it. Not a cache this inventory offers.
        configured = None
    return _hf_child_dirs("datasets", configured)


def _hf_assets_dirs() -> list[Path]:
    return _hf_child_dirs("assets", _env_dir("HF_ASSETS_CACHE"))


def _hf_xet_dirs() -> list[Path]:
    return [_hf_paths().xet_cache]


def _diffusion_compile_root() -> Optional[Path]:
    try:
        from core.inference.diffusion_compile_cache import cache_root
        return _safe_resolve(cache_root())
    except Exception as exc:  # noqa: BLE001 - a broken import must not widen the list
        logger.debug(f"Could not resolve the diffusion compile cache: {exc}")
        return None


def _torch_inductor_dirs() -> list[Path]:
    configured = _env_dir("TORCHINDUCTOR_CACHE_DIR")
    diffusion = _diffusion_compile_root()
    resolved = None if configured is None else _safe_resolve(configured)
    if configured is not None and diffusion is not None and resolved is not None:
        if _is_within(resolved, diffusion):
            # diffusion_compile_cache.begin() repoints this at <key>/inductor
            # while a model is resident and restores it on unload, so a row that
            # followed it would name one directory and the clear take another.
            configured = None
    if configured is not None:
        return [configured]
    import getpass
    import re
    import tempfile

    # torch/_inductor/runtime/cache_dir_utils.py, followed exactly. getpass also
    # reads LOGNAME and the pwd name, so a container with no USER puts the cache
    # at torchinductor_root while a bare uid would look at torchinductor_0.
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
    # The CUDA guide's defaults, which follow no shared convention: ROAMING
    # AppData on Windows, Application Support on macOS, ~/.nv elsewhere.
    if _is_windows():
        roaming = (os.environ.get("APPDATA") or "").strip()
        base = Path(roaming) if roaming else _home() / "AppData" / "Roaming"
        return [base / "NVIDIA" / "ComputeCache"]
    if sys.platform == "darwin":
        return [_home() / "Library" / "Application Support" / "NVIDIA" / "ComputeCache"]
    return [_home() / ".nv" / "ComputeCache"]


def _numba_dirs() -> list[Path]:
    # __pycache__ next to the source is not ours to empty, but when it is not
    # writable numba's UserWideCacheLocator falls back to
    # AppDirs("numba", appauthor = False).user_cache_dir, which is numba's alone.
    return _first(_env_dir("NUMBA_CACHE_DIR"), _platform_cache_dir("numba"))


def _matplotlib_dirs() -> list[Path]:
    """get_cachedir()'s own rules: _get_config_or_cache_dir takes the XDG branch
    for linux and freebsd only, and otherwise ~/.matplotlib, with win32
    preferring %LOCALAPPDATA%\\matplotlib when the legacy dir is absent."""
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
    # Only the XDG branch is a cache dir of its own. MPLCONFIGDIR and the
    # ~/.matplotlib default are get_configdir() too, so matplotlibrc sits beside
    # the font list and only the cache entries may go.
    merged = _env_dir("MPLCONFIGDIR") is not None or _is_windows() or sys.platform == "darwin"
    return MATPLOTLIB_PATTERNS if merged else None


def _vllm_dirs() -> list[Path]:
    # vllm/envs.py: XDG_CACHE_HOME or ~/.cache, then "vllm", on every platform,
    # which the platform helper matches on Linux only.
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
        # clear_unsloth_compiled_cache swallows every failure, so a read-only
        # directory looks identical to a clean clear. Bytes and not entries: a
        # dedicated cache is recreated holding an empty marker file.
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
    # Opt-in. The others cost a recompile or a re-download; this one can break a
    # job running RIGHT NOW, because a worker imports generated modules from here
    # lazily and compiled_cache_lock only serialises against a sibling BACKEND.
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


def _safe_resolve(path: Path) -> Optional[Path]:
    try:
        return Path(os.path.realpath(str(Path(path).expanduser())))
    except (OSError, RuntimeError, ValueError):
        return None


def _is_junction(path: Path | str) -> bool:
    """True for a Windows directory junction or volume mount point.

    The same hazard as a symlink and not the same test: since 3.8 only
    IO_REPARSE_TAG_SYMLINK sets S_IFLNK, so is_symlink() is False for a junction
    while realpath() follows it, and UV_CACHE_DIR pointed at one would have the
    TARGET emptied. os.path.isjunction is 3.12+ and this package supports 3.9.
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
        studio_bin_root,
        studio_db_path,
        studio_root,
        tensorboard_root,
        tmp_root,
    )

    candidates: list[Path] = [
        Path.home(),
        studio_root(),
        studio_db_path(),
        # The shim and the managed executables. No cache belongs there.
        studio_bin_root(),
        auth_root(),
        auth_db_path(),
        assets_root(),
        datasets_root(),
        outputs_root(),
        exports_root(),
        rag_root(),
        tensorboard_root(),
        # Not a cache: decoding and training read files back from it.
        tmp_root(),
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
            candidates.append(home / "token")
        paths = _hf_paths()
        # cache_home is what Settings DISPLAYS, and HF_HUB_CACHE=/mnt/hf-cache
        # makes it the hub directory itself. Protecting it there would make the
        # model cache permanently unclearable and no credential safer.
        if _safe_resolve(paths.cache_home) != _safe_resolve(paths.hub_cache):
            candidates.append(paths.cache_home)
    except Exception as exc:  # noqa: BLE001 - a broken HF setting must not widen the allow-list
        logger.debug(f"Could not resolve the Hugging Face cache homes: {exc}")
        candidates.append(Path.home() / ".cache" / "huggingface")
    hf_home = _env_dir("HF_HOME")
    if hf_home is not None:
        candidates.append(hf_home)
        candidates.append(hf_home / "token")
    # Read INSTEAD of <home>/token, and it can name a file inside a cache.
    token_path = _env_dir("HF_TOKEN_PATH")
    if token_path is not None:
        candidates.append(token_path)

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
        studio_bin_root,
        tensorboard_root,
        tmp_root,
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
        # Descendants of the studio home and of the temp dir are deliberately
        # allowed, since the caches live there, so these need naming on their own.
        documents_root(),
        studio_bin_root(),
        tmp_root(),
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
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def sheltered_roots(exclude_key: Optional[str] = None) -> dict[Path, str]:
    """Resolved roots another key's clear must not empty, mapped to why.

    An opt-in cache and a pattern-limited one for the same reason: their own
    clear is narrower than emptying the directory, so swallowing them whole
    takes what that clear was written to keep. Nothing stops a variable from
    putting either inside another cache (``MPLCONFIGDIR=/cache/uv/matplotlib``
    under ``UV_CACHE_DIR=/cache/uv``). The key being cleared is excluded.
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

    The one gate every deletion here goes through, answering from the resolved
    directory so a variable pointed at a symlink cannot smuggle in a target.
    """
    protected = protected_paths() if protected is None else protected
    trees = protected_trees() if trees is None else trees
    raw = Path(root)
    if not raw.is_absolute():
        raise CachePurgeRefused(f"Cache root is not an absolute path: {raw}")
    # The link itself, not its target: emptying through one deletes files at a
    # location nobody named, and unlinking it would take the user's link.
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
    # is_dir is True for a junction and is_symlink is not, so descending would
    # size its target and a junction to an ancestor would never terminate. Free
    # on POSIX, where isjunction and the fallback answer without a syscall.
    return not _is_junction(entry.path)


def _measure_tree(path: Path, seen: set) -> tuple[int, int]:
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
    # Gate first, then measure: a root the gate refuses would otherwise be walked
    # recursively on every open of the tab, for bytes nothing can reclaim.
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
                if blocked_reason is None:
                    blocked_reason = str(exc)
                continue
            measurable.append(root)
        # What a clear will do: purge_cache skips a refused root and carries on,
        # so one bad root does not put the others out of reach.
        purgeable = bool(measurable)
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


# A walk is seconds on a large uv or triton cache, so a repeat read inside this
# window reuses the last answer.
_INVENTORY_TTL_SECONDS = 60.0
_size_cache: dict[str, tuple[float, dict]] = {}
# Bumped by every invalidation, so a walk that began before one cannot store its
# answer after it. Dropping the entry is not enough: a scan started in one tab
# before a purge in another would install the pre-purge size for the window.
_size_epochs: dict[str, int] = {}
_size_cache_lock = threading.Lock()


def invalidate_cache_size(key: str) -> None:
    with _size_cache_lock:
        _size_cache.pop(key, None)
        _size_epochs[key] = _size_epochs.get(key, 0) + 1


_size_flights: dict[str, threading.Lock] = {}
_size_flights_lock = threading.Lock()


def _flight_for(key: str) -> threading.Lock:
    with _size_flights_lock:
        return _size_flights.setdefault(key, threading.Lock())


def _described(definition: CacheDefinition, *, refresh: bool) -> dict:
    started = time.monotonic()
    with _size_cache_lock:
        remembered = None if refresh else _size_cache.get(definition.key)
    if remembered is not None and started - remembered[0] < _INVENTORY_TTL_SECONDS:
        return remembered[1]
    # One walk per cache at a time: simultaneous misses would each pay a cold
    # walk for the same answer, and two open tabs are enough to cause it.
    with _flight_for(definition.key):
        with _size_cache_lock:
            began_at = _size_epochs.get(definition.key, 0)
            remembered = _size_cache.get(definition.key)
        # One that finished during the wait began after this call did, so it is
        # fresh enough for it whether or not it asked to force.
        if remembered is not None and remembered[0] >= started:
            return remembered[1]
        entry = describe_cache(definition)
        with _size_cache_lock:
            if _size_epochs.get(definition.key, 0) == began_at:
                _size_cache[definition.key] = (time.monotonic(), entry)
        return entry


def cache_inventory(*, refresh: bool = False) -> dict:
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


def _remove_entry(entry: os.DirEntry, root: Path, outcome: PurgeOutcome, seen: set) -> None:
    path = Path(entry.path)
    try:
        if entry.is_symlink():
            # The link, never its target: this is the escape a cache can hold.
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


# Which registries write into each cache. Both for the hub and xet roots: a
# dataset download snapshot_downloads its datasets-- entries into the same hub
# tree, and each claim of either kind carries the xet cache it fetches chunks to.
_DOWNLOAD_REGISTRIES = {
    "hf_hub": ("models", "datasets"),
    "hf_xet": ("models", "datasets"),
    "hf_datasets": ("datasets",),
}

_PURGE_BUSY = "Cancel the active downloads before clearing this cache."


def _reserve_downloads(key: str) -> tuple[list, Optional[str]]:
    """Hold every download registry that writes into this cache, or say why not.

    A reservation and not a look, for the reason the per-repository deletes call
    begin_delete: a worker can claim between a check and the rmtree. A registry
    this cannot reach does not block the purge, or an import would kill the button.
    """
    kinds = _DOWNLOAD_REGISTRIES.get(key)
    if not kinds:
        return [], None
    try:
        from hub.utils.download_registry import get_datasets_registry, get_models_registry
        registries = [
            get_models_registry() if kind == "models" else get_datasets_registry() for kind in kinds
        ]
    except Exception as exc:  # noqa: BLE001 - a broken registry must not block a purge
        logger.debug(f"Could not reach the download registries for {key}: {exc}")
        return [], None
    reserved: list = []
    for registry in registries:
        try:
            granted = registry.begin_cache_purge()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Could not reserve a download registry for {key}: {exc}")
            continue
        if not granted:
            _release_downloads(reserved)
            return [], _PURGE_BUSY
        reserved.append(registry)
    return reserved, None


def _release_downloads(reserved: Iterable) -> None:
    for registry in reserved:
        try:
            registry.end_cache_purge()
        except Exception as exc:  # noqa: BLE001 - never leave a purge holding one
            logger.warning(f"Could not release a download registry: {exc}")


def purge_cache(key: str) -> dict:
    """Empty one cache by key. Never raises for a refusal; it reports it."""
    definition = definition_for(key)
    reserved, busy = _reserve_downloads(key)
    if busy is not None:
        logger.warning(f"Refusing to purge the {key} cache: {busy}")
        return _purge_result(definition, PurgeOutcome(errors = [busy]))
    try:
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
    finally:
        _release_downloads(reserved)


def _purge_result(definition: CacheDefinition, outcome: PurgeOutcome) -> dict:
    return {
        "key": definition.key,
        "freed_bytes": outcome.freed_bytes,
        "removed_entries": outcome.removed_entries,
        "errors": list(outcome.errors),
    }


# Emptying either removes the repositories the Hub inventory reports, so it is
# one of the app-driven mutations inventory_scan says it is invalidated on.
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
            invalidate_cache_size(definition.key)
    if any(definition.key in _HF_SCANNED_KEYS for definition in definitions):
        _invalidate_hf_scans()
    return {
        "results": results,
        "freed_bytes": sum(result["freed_bytes"] for result in results),
        "inventory": cache_inventory(),
    }
