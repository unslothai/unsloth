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
    # True when emptying costs a user-chosen re-download; never in a bulk purge.
    opt_in: bool
    resolve: Callable[[], list[Path]]
    # Only top-level entries matching these globs are measured or deleted, for config-holding roots.
    patterns: Optional[tuple[str, ...]] = None
    custom_purge: Optional[Callable[[], "PurgeOutcome"]] = None
    # Sized by its own rule, when a walk would not match what a clear removes.
    custom_measure: Optional[Callable[[], "tuple[int, int]"]] = None
    # Its own gate, for a cache whose roots the standard one cannot check.
    custom_block: Optional[Callable[[], Optional[str]]] = None


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
    # Patching os.name is not a seam: pathlib would build WindowsPath on POSIX.
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
    """Read the way unsloth_cli parses it: no expanduser, since uv makes a literal ~ directory."""
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
    # Both: _setup_cache_env seeds one, the installer may record another; either can be multi-gigabyte.
    roots = _first(_env_dir("UV_CACHE_DIR"), _platform_cache_dir("uv"))
    recorded = _recorded_uv_cache()
    if recorded is not None:
        roots.append(recorded)
    return roots


# The lock spans the probe, so a cold request never reads a pending miss as failure.
_probed_cache_dirs: dict[str, Optional[Path]] = {}
_probe_lock = threading.Lock()


def _probe_tool_cache_dir(name: str, command: list[str]) -> Optional[Path]:
    """Asks the tool itself, since pip.conf and .npmrc can move the cache with no environment variable."""
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
                # Else the locale encoding (ANSI codepage on Windows) mangles non-ASCII paths.
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


# Not the rest of ~/.npm: _logs is the user's.
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
    """For protection only: an API key can append to this history, so it must never pick a purge root."""
    from utils.hf_cache_settings import known_hf_cache_homes
    return list(known_hf_cache_homes())


def _hf_hub_dirs() -> list[Path]:
    return [_hf_paths().hub_cache]


def _hf_child_dirs(name: str, configured: Optional[Path]) -> list[Path]:
    # HF reads the variable instead of <home>/<name>, so these are alternatives; the home is the effective one, not the one Settings displays.
    from utils.hf_cache_settings import effective_cache_home
    return _first(configured, effective_cache_home() / name)


def _studio_datasets_fallback() -> Optional[Path]:
    """The Studio cache a dataset load retries in when HF's is not writable."""
    try:
        from utils.paths.storage_roots import cache_root
        return _safe_resolve(cache_root() / "hf-datasets")
    except Exception as exc:  # noqa: BLE001 - a broken root must not widen the list
        logger.debug(f"Could not resolve the Unsloth dataset fallback cache: {exc}")
        return None


def _hf_datasets_dirs() -> list[Path]:
    configured = _env_dir("HF_DATASETS_CACHE")
    fallback = _studio_datasets_fallback()
    if configured is not None and fallback is not None and _safe_resolve(configured) == fallback:
        # cache_safe points this at a Studio cache during one load; not offered here.
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
            # diffusion_compile_cache repoints this while a model is resident, so a row following it would clear a different directory than it names.
            configured = None
    if configured is not None:
        return [configured]
    import getpass
    import re
    import tempfile

    # Mirrors torch cache_dir_utils exactly; uses getpass, not a bare uid, since a container with no USER puts the cache at another directory.
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
    configured = _env_dir("TORCH_EXTENSIONS_DIR")
    if configured is not None:
        return [configured]
    if _is_windows():
        # torch/_appdirs defaults appauthor to the app name, so the name appears twice.
        return [_local_app_data() / "torch_extensions" / "torch_extensions" / "Cache"]
    return [_platform_cache_dir("torch_extensions")]


def _triton_dirs() -> list[Path]:
    return _first(_env_dir("TRITON_CACHE_DIR"), _home() / ".triton" / "cache")


def _cuda_dirs() -> list[Path]:
    configured = _env_dir("CUDA_CACHE_PATH")
    if configured is not None:
        return [configured]
    # CUDA's defaults: ROAMING AppData (Windows), Application Support (macOS), ~/.nv elsewhere.
    if _is_windows():
        roaming = (os.environ.get("APPDATA") or "").strip()
        base = Path(roaming) if roaming else _home() / "AppData" / "Roaming"
        return [base / "NVIDIA" / "ComputeCache"]
    if sys.platform == "darwin":
        return [_home() / "Library" / "Application Support" / "NVIDIA" / "ComputeCache"]
    return [_home() / ".nv" / "ComputeCache"]


def _numba_dirs() -> list[Path]:
    # __pycache__ is not ours to empty; numba's fallback user_cache_dir (AppDirs) is numba's alone.
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
    # Only the XDG branch is cache-only; other config dirs also hold matplotlibrc.
    merged = _env_dir("MPLCONFIGDIR") is not None or _is_windows() or sys.platform == "darwin"
    return MATPLOTLIB_PATTERNS if merged else None


def _vllm_dirs() -> list[Path]:
    # vllm/envs.py: XDG_CACHE_HOME or ~/.cache, then vllm, on every platform (Linux-only in the helper).
    xdg = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    base = Path(xdg).expanduser() if xdg else _home() / ".cache"
    return _first(_env_dir("VLLM_CACHE_ROOT"), base / "vllm")


def _unsloth_compiled_dirs() -> list[Path]:
    from utils.cache_cleanup import _cleanable_cache_dirs
    return [directory for directory, _dedicated in _cleanable_cache_dirs()]


def _measure_unsloth_compiled() -> tuple[int, int]:
    """Counts only what a clear removes: directories Unsloth made whole, the rest only generated modules."""
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


def _compiled_cache_refusal() -> Optional[str]:
    """Its own gate: the clear is all-or-nothing across the directories it owns, so one refusal stops it."""
    from utils.cache_cleanup import _cleanable_cache_dirs

    protected = protected_paths()
    trees = protected_trees()
    keep = sheltered_roots("unsloth_compiled")
    for directory, dedicated in _cleanable_cache_dirs():
        # Only the generated module files are ever touched in a shared one.
        if not dedicated:
            continue
        try:
            assert_purgeable_root(directory, protected = protected, trees = trees, keep = keep)
        except CachePurgeRefused as exc:
            return str(exc)
    return None


def _purge_unsloth_compiled() -> PurgeOutcome:
    """Delegates to cache_cleanup, which knows what Unsloth created and serializes with sibling backends."""
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
        refusal = _compiled_cache_refusal()
        if refusal is not None:
            outcome.errors.append(refusal)
            return outcome
        before, entries = _measure_unsloth_compiled()
        clear_unsloth_compiled_cache()
        after, remaining = _measure_unsloth_compiled()
    outcome.freed_bytes = max(0, before - after)
    outcome.removed_entries = max(0, entries - remaining)
    if after > 0:
        # Failures are swallowed: check remaining bytes, not entries, as a dedicated cache keeps a marker.
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
    # Opt-in: workers import generated modules from here lazily mid-job, so clearing can break a running job; the compile lock covers only sibling backends.
    CacheDefinition(
        "unsloth_compiled",
        GROUP_COMPILE,
        True,
        _unsloth_compiled_dirs,
        custom_purge = _purge_unsloth_compiled,
        custom_measure = _measure_unsloth_compiled,
        custom_block = _compiled_cache_refusal,
    ),
    CacheDefinition("hf_xet", GROUP_MODELS, False, _hf_xet_dirs),
    CacheDefinition("hf_assets", GROUP_MODELS, False, _hf_assets_dirs),
    CacheDefinition("hf_datasets", GROUP_MODELS, True, _hf_datasets_dirs),
    CacheDefinition("hf_hub", GROUP_MODELS, True, _hf_hub_dirs),
)

CACHE_KEYS: tuple[str, ...] = tuple(definition.key for definition in CACHE_DEFINITIONS)

_DEFINITIONS_BY_KEY = {definition.key: definition for definition in CACHE_DEFINITIONS}

# Concurrent sweeps of one root race into spurious errors.
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
    """On Windows is_symlink() is False for a junction yet realpath follows it, so check separately; os.path.isjunction needs 3.12 and the package supports 3.9."""
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
    """User data, never purged; the Hugging Face home stays protected because it holds the access token."""
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
        # Not a cache root of ours as a whole: the parent of per-tool cache dirs, plus llama.cpp slot state.
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
        # With HF_HUB_CACHE set, cache_home can be the hub dir itself; protecting it would make the model cache permanently unclearable.
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
    """Refuses a cache root beneath user data, e.g. TRITON_CACHE_DIR inside projects; studio and HF cache homes are absent since real caches live in both."""
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
        # Studio home and temp dir descendants are allowed (caches live there), so name these.
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
    """Roots another key's clear must not empty: emptying one takes what its narrower clear keeps."""
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
    """Judges the resolved directory, so a variable pointed at a symlink cannot smuggle in a target."""
    protected = protected_paths() if protected is None else protected
    trees = protected_trees() if trees is None else trees
    raw = Path(root)
    if not raw.is_absolute():
        raise CachePurgeRefused(f"Cache root is not an absolute path: {raw}")
    # Refuse the link itself, not its target: emptying through it would delete files at a location nobody named.
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


class LinkLedger:
    """Bytes a removal would actually free, counted per inode rather than per name.

    One link frees its bytes when it goes. Several free nothing unless every one of them goes
    too, and that is not a corner case: uv's default link mode on Windows is hardlink, so an
    installed environment's files ARE links into the uv cache. The Hugging Face cache does the
    same for blobs on filesystems without symlinks, but there both links live inside the cache.

    So links are tallied as they are met and a multi-link inode contributes only once the number
    seen inside the roots being measured reaches st_nlink. Anything still linked from outside is
    left out, which understates rather than promises space that unlinking will not return.
    """

    def __init__(self) -> None:
        self._single = 0
        self._multi: dict[tuple[int, int], list[int]] = {}

    def add(self, stat) -> None:
        if stat.st_nlink <= 1:
            self._single += int(stat.st_size)
            return
        record = self._multi.get((stat.st_dev, stat.st_ino))
        if record is None:
            self._multi[(stat.st_dev, stat.st_ino)] = [int(stat.st_size), int(stat.st_nlink), 1]
        else:
            record[2] += 1

    def freeable_bytes(self) -> int:
        return self._single + sum(size for size, links, met in self._multi.values() if met >= links)


def _record_entry(entry: os.DirEntry, ledger: LinkLedger) -> None:
    try:
        stat = entry.stat(follow_symlinks = False)
    except OSError:
        return
    ledger.add(stat)


def _descendable(entry: os.DirEntry) -> bool:
    # is_dir is True for a junction, so descending could loop forever.
    return not _is_junction(entry.path)


def _measure_tree(path: Path, ledger: LinkLedger) -> int:
    """Size is left to the ledger: a multi-link inode's cost is only known after every root is walked."""
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
                    _record_entry(entry, ledger)
        except OSError:
            continue
    return count


def _measure_root(root: Path, *, patterns: Optional[Iterable[str]] = None) -> tuple[int, int]:
    ledger = LinkLedger()
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
                        _measure_tree(Path(entry.path), ledger)
                    continue
                _record_entry(entry, ledger)
    except OSError as exc:
        logger.debug(f"Could not size {root}: {exc}")
    return ledger.freeable_bytes(), entries


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
    # Gate first so refused roots are never walked.
    measurable = list(roots)
    if roots and definition.custom_block is not None:
        blocked_reason = definition.custom_block()
        purgeable = blocked_reason is None
    elif roots and definition.custom_purge is None:
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
        # purge_cache skips a refused root and carries on, so one bad root does not block the others.
        purgeable = bool(measurable)
    # A LINKED cache is never offered, since removing it breaks the install; checked after root refusals.
    if blocked_reason is None:
        blocked_reason = _link_mode_refusal(definition.key)
        if blocked_reason is not None:
            purgeable = False
    if definition.custom_measure is not None:
        total, entries = (0, 0) if blocked_reason is not None else definition.custom_measure()
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


# Walks take seconds on large uv/triton caches, so repeat reads within the TTL reuse the last answer.
_INVENTORY_TTL_SECONDS = 60.0
# Tuple of (walk start, store time, entry): TTL uses store time, forced reads compare walk start.
_size_cache: dict[str, tuple[float, float, dict]] = {}
# Bumped per invalidation so a walk begun earlier cannot store a stale size.
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
    if remembered is not None and started - remembered[1] < _INVENTORY_TTL_SECONDS:
        return remembered[2]
    # One walk per cache at a time, or simultaneous misses (e.g. two open tabs) each pay a cold walk.
    with _flight_for(definition.key):
        with _size_cache_lock:
            epoch = _size_epochs.get(definition.key, 0)
            remembered = _size_cache.get(definition.key)
        # A forced read needs a walk that began after it asked.
        if remembered is not None:
            if refresh:
                if remembered[0] >= started:
                    return remembered[2]
            elif time.monotonic() - remembered[1] < _INVENTORY_TTL_SECONDS:
                return remembered[2]
        begun = time.monotonic()
        entry = describe_cache(definition)
        with _size_cache_lock:
            if _size_epochs.get(definition.key, 0) == epoch:
                _size_cache[definition.key] = (begun, time.monotonic(), entry)
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


def _remove_entry(
    entry: os.DirEntry, root: Path, outcome: PurgeOutcome, ledger: LinkLedger, survivors: LinkLedger
) -> None:
    """Freed bytes come from the ledger after the root: an inode still linked from outside frees nothing."""
    path = Path(entry.path)
    try:
        if entry.is_symlink():
            # The link, never its target: this is the escape a cache can hold.
            os.unlink(path)
        elif entry.is_dir(follow_symlinks = False):
            real = _safe_resolve(path)
            if real is None or not _is_within(real, root):
                outcome.errors.append(f"Skipped {path}: it resolves outside the cache root")
                return
            _measure_tree(path, ledger)
            # rmtree refuses a symlinked dir and does not follow inner links, so it stays in root.
            try:
                shutil.rmtree(path)
            except OSError:
                # Survivors are subtracted so freed bytes never count files still on disk.
                _measure_tree(path, survivors)
                raise
        else:
            _record_entry(entry, ledger)
            os.unlink(path)
    except OSError as exc:
        outcome.errors.append(f"Could not remove {path}: {exc}")
        return
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
    ledger = LinkLedger()
    # What a failed removal left behind, subtracted from the ledger below.
    survivors = LinkLedger()
    try:
        with os.scandir(resolved) as scan:
            entries = list(scan)
    except OSError as exc:
        outcome.errors.append(f"Could not read {resolved}: {exc}")
        return outcome
    for entry in entries:
        if not _matching(entry.name, patterns):
            continue
        _remove_entry(entry, resolved, outcome, ledger, survivors)
    # Clamp at zero: a file that grew between walks must not make a partial purge report negative bytes.
    outcome.freed_bytes += max(0, ledger.freeable_bytes() - survivors.freeable_bytes())
    return outcome


# Hub and xet roots are written by models and datasets; dataset snapshots land in the same hub tree.
_DOWNLOAD_REGISTRIES = {
    "hf_hub": ("models", "datasets"),
    "hf_xet": ("models", "datasets"),
    "hf_datasets": ("datasets",),
}

_PURGE_BUSY = "Cancel the active downloads before clearing this cache."


def _reserve_downloads(key: str) -> tuple[list, Optional[str]]:
    """Reserves rather than checks, since a worker could claim the cache before the rmtree; a registry that cannot be reached must not block the purge."""
    kinds = _DOWNLOAD_REGISTRIES.get(key)
    if not kinds:
        return [], None
    try:
        from hub.services.datasets import downloads as dataset_downloads
        from hub.utils.download_registry import get_models_registry

        # The service, not the singleton: managed installs have one registry per account, and the service keeps new ones from appearing mid-purge.
        registries = [
            get_models_registry() if kind == "models" else dataset_downloads for kind in kinds
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


# Spawned trainers read these via HF_HUB_CACHE and HF_XET_CACHE outside every registry here, so clearing could cost a run.
_WORKER_SENSITIVE_KEYS = frozenset({"hf_hub", "hf_xet", "hf_datasets"})


def _any_training_active() -> bool:
    """Either trainer. They are tracked separately and both spawn a worker that
    loads from the Hugging Face cache directly, the LLM one through
    snapshot_download and the diffusion one through from_pretrained."""
    active = False
    try:
        from core.training import get_training_backend
        active = bool(get_training_backend().is_training_active())
    except Exception as exc:  # noqa: BLE001 - a broken import must not block a purge
        logger.debug(f"Could not read the training state: {exc}")
    if active:
        return True
    try:
        from core.training.diffusion_training_service import get_diffusion_training_service
        return bool(get_diffusion_training_service().is_active())
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"Could not read the diffusion training state: {exc}")
        return False


def _uv_symlink_refusal() -> Optional[str]:
    """Refuse the uv cache when UV_LINK_MODE=symlink or site-packages entries link into it, since clearing it would break installed packages."""
    if (os.environ.get("UV_LINK_MODE") or "").strip().lower() == "symlink":
        return "uv is set to link packages from this cache; clearing it would break them."
    try:
        import sysconfig
        site_packages = sysconfig.get_paths().get("purelib")
    except Exception as exc:  # noqa: BLE001 - no site-packages means nothing linked from here
        logger.debug(f"Could not locate site-packages for the uv link check: {exc}")
        return None
    if not site_packages:
        return None
    roots = [root for root in _uv_dirs() if root is not None]
    if not roots:
        return None
    try:
        with os.scandir(site_packages) as entries:
            for entry in entries:
                if not entry.is_symlink():
                    continue
                target = _safe_resolve(Path(entry.path))
                if target is None:
                    continue
                if any(_is_within(target, _safe_resolve(root) or root) for root in roots):
                    return (
                        "The installed packages are symlinked into this cache; "
                        "clearing it would break them."
                    )
    except OSError as exc:
        logger.debug(f"Could not read site-packages for the uv link check: {exc}")
    return None


def _link_mode_refusal(key: str) -> Optional[str]:
    return _uv_symlink_refusal() if key == "uv" else None


def _training_refusal(key: str) -> Optional[str]:
    if key not in _WORKER_SENSITIVE_KEYS:
        return None
    return "Stop the training run before clearing this cache." if _any_training_active() else None


def _inference_refusal(key: str) -> Optional[str]:
    """Refuses a model-cache clear while a backend holds weights (sd.cpp re-reads them per generation); fails closed if the check raises."""
    if key not in _WORKER_SENSITIVE_KEYS:
        return None
    try:
        from hub.services.models.deletion import any_model_load_blocks_cache_clear
    except Exception as exc:  # noqa: BLE001 - no guard module means no backend to hold anything
        logger.debug(f"Could not import the inference load-state guard: {exc}")
        return None
    try:
        return any_model_load_blocks_cache_clear()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not verify the inference load state; refusing the clear: {exc}")
        return "Could not verify whether a model is loaded. Try again in a moment."


def purge_cache(key: str) -> dict:
    """Empty one cache by key. Never raises for a refusal; it reports it."""
    definition = definition_for(key)
    training = _training_refusal(key)
    if training is not None:
        logger.warning(f"Refusing to purge the {key} cache: {training}")
        return _purge_result(definition, PurgeOutcome(errors = [training]))
    inference = _inference_refusal(key)
    if inference is not None:
        logger.warning(f"Refusing to purge the {key} cache: {inference}")
        return _purge_result(definition, PurgeOutcome(errors = [inference]))
    linked = _link_mode_refusal(key)
    if linked is not None:
        logger.warning(f"Refusing to purge the {key} cache: {linked}")
        return _purge_result(definition, PurgeOutcome(errors = [linked]))
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
            if definition.key == "hf_hub":
                # audio.cpp's link farm beside the hub hardlinks its blobs, so they would stay on disk.
                from core.inference.audio_cpp_files import prune_link_farm
                prune_link_farm(root)
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


# Emptying either removes the repos the Hub inventory reports, so it invalidates inventory_scan.
_HF_SCANNED_KEYS = frozenset({"hf_hub", "hf_datasets"})
# Caches under the configured HF root; they describe another directory once that root changes.
_HF_ROOTED_KEYS = frozenset({"hf_hub", "hf_xet", "hf_datasets"})


def _invalidate_hf_scans() -> None:
    try:
        from hub.utils.inventory_scan import invalidate_hf_cache_scans
    except ImportError as exc:
        logger.debug(f"Could not invalidate the Hugging Face scans: {exc}")
        return
    invalidate_hf_cache_scans()


def invalidate_hf_rooted_sizes() -> None:
    """Memo is keyed by cache key, not path, so a moved Models Folder leaves stale sizes for the TTL."""
    for key in _HF_ROOTED_KEYS:
        invalidate_cache_size(key)


def purge_caches(keys: Iterable[str]) -> dict:
    """Unknown keys raise before any deletion, so a request naming one unknown cache changes nothing."""
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
