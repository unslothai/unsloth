# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


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
    opt_in: bool
    resolve: Callable[[], list[Path]]
    patterns: Optional[tuple[str, ...]] = None
    custom_purge: Optional[Callable[[], "PurgeOutcome"]] = None
    custom_measure: Optional[Callable[[], "tuple[int, int]"]] = None
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
    "Match the CLI: uv treats ~ literally in the recorded path."
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
    roots = _first(_env_dir("UV_CACHE_DIR"), _platform_cache_dir("uv"))
    recorded = _recorded_uv_cache()
    if recorded is not None:
        roots.append(recorded)
    return roots


# The lock spans the probe, so a cold request never reads a pending miss as failure.
_probed_cache_dirs: dict[str, Optional[Path]] = {}
_probe_lock = threading.Lock()


def _probe_tool_cache_dir(name: str, command: list[str]) -> Optional[Path]:
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
    npm = shutil.which("npm")
    return None if npm is None else _probe_tool_cache_dir("npm", [npm, "config", "get", "cache"])


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
    "History is API-writable and must never select purge roots."
    from utils.hf_cache_settings import known_hf_cache_homes
    return list(known_hf_cache_homes())


def _hf_hub_dirs() -> list[Path]:
    return [_hf_paths().hub_cache]


def _hf_child_dirs(name: str, configured: Optional[Path]) -> list[Path]:
    from utils.hf_cache_settings import effective_cache_home
    return _first(configured, effective_cache_home() / name)


def _studio_datasets_fallback() -> Optional[Path]:
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
        return [_local_app_data() / "torch_extensions" / "torch_extensions" / "Cache"]
    return [_platform_cache_dir("torch_extensions")]


def _triton_dirs() -> list[Path]:
    return _first(_env_dir("TRITON_CACHE_DIR"), _home() / ".triton" / "cache")


def _cuda_dirs() -> list[Path]:
    configured = _env_dir("CUDA_CACHE_PATH")
    if configured is not None:
        return [configured]
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
    xdg = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    base = Path(xdg).expanduser() if xdg else _home() / ".cache"
    return _first(_env_dir("VLLM_CACHE_ROOT"), base / "vllm")


def _unsloth_compiled_dirs() -> list[Path]:
    from utils.cache_cleanup import _cleanable_cache_dirs
    return [directory for directory, _dedicated in _cleanable_cache_dirs()]


def _measure_unsloth_compiled() -> tuple[int, int]:
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
    from utils.cache_cleanup import _cleanable_cache_dirs

    protected = protected_paths()
    trees = protected_trees()
    keep = sheltered_roots("unsloth_compiled")
    for directory, dedicated in _cleanable_cache_dirs():
        if not dedicated:
            continue
        try:
            assert_purgeable_root(directory, protected = protected, trees = trees, keep = keep)
        except CachePurgeRefused as exc:
            return str(exc)
    return None


def _purge_unsloth_compiled() -> PurgeOutcome:
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
    "Windows junctions escape is_symlink(); Python 3.9 needs the fallback."
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
        studio_bin_root(),
        auth_root(),
        auth_db_path(),
        assets_root(),
        datasets_root(),
        outputs_root(),
        exports_root(),
        rag_root(),
        tensorboard_root(),
        tmp_root(),
        documents_root(),
        project_workspaces_root(),
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
    "Count an inode only when all its hardlinks are removed."

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

    def add_single_link_bytes(self, size: int) -> None:
        self._single += size

    def freeable_bytes(self) -> int:
        return self._single + sum(size for size, links, met in self._multi.values() if met >= links)


class _OccupancyLedger:
    def __init__(self) -> None:
        self._seen: set[tuple] = set()
        self.bytes = 0
        self.by_device: dict[int, int] = {}
        self.device_paths: dict[int, Path] = {}

    def add(self, info, path: Path | str) -> None:
        # Some network filesystems report st_ino 0: identify by path instead.
        identity = (info.st_dev, info.st_ino) if info.st_ino else (info.st_dev, str(path))
        if identity in self._seen:
            return
        self._seen.add(identity)
        size = int(info.st_size)
        self.bytes += size
        self.by_device[info.st_dev] = self.by_device.get(info.st_dev, 0) + size
        self.device_paths.setdefault(info.st_dev, Path(path).parent)


def _record_entry(
    entry: os.DirEntry,
    ledger: LinkLedger,
    occupancy: Optional[_OccupancyLedger] = None,
) -> None:
    try:
        # Windows DirEntry.stat zeroes st_dev, st_ino and st_nlink.
        stat = (
            os.stat(entry.path, follow_symlinks = False)
            if occupancy is not None and os.name == "nt"
            else entry.stat(follow_symlinks = False)
        )
    except OSError:
        return
    ledger.add(stat)
    if occupancy is not None:
        occupancy.add(stat, entry.path)


def _descendable(entry: os.DirEntry) -> bool:
    # is_dir is True for a junction, so descending could loop forever.
    return not _is_junction(entry.path)


def _walk_into(
    stack: list,
    ledger: LinkLedger,
    occupancy: Optional[_OccupancyLedger] = None,
    *,
    excluded_roots: Optional[set[Path]] = None,
) -> int:
    "Keep paths as strings: uv/triton trees can contain millions of files."
    count = 0
    single = 0
    add = ledger.add
    pop = stack.pop
    push = stack.append
    try:
        while stack:
            current = pop()
            if excluded_roots and Path(current) in excluded_roots:
                continue
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        count += 1
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks = False):
                            if _descendable(entry):
                                push(entry.path)
                            continue
                        if occupancy is not None:
                            _record_entry(entry, ledger, occupancy)
                            continue
                        try:
                            found = entry.stat(follow_symlinks = False)
                        except OSError:
                            continue
                        if found.st_nlink <= 1:
                            single += found.st_size
                        else:
                            add(found)
            except OSError:
                continue
    finally:
        ledger.add_single_link_bytes(single)
    return count


def _measure_tree(
    path: Path,
    ledger: LinkLedger,
    occupancy: Optional[_OccupancyLedger] = None,
) -> int:
    return _walk_into([os.fspath(path)], ledger, occupancy)


def _measure_root(
    root: Path,
    *,
    patterns: Optional[Iterable[str]] = None,
    ledger: Optional[LinkLedger] = None,
    occupancy: Optional[_OccupancyLedger] = None,
    excluded_roots: Optional[set[Path]] = None,
) -> tuple[int, int]:
    ledger = LinkLedger() if ledger is None else ledger
    entries = 0
    try:
        if Path(root).is_symlink() or _is_junction(root) or not Path(root).is_dir():
            return 0, 0
    except OSError:
        return 0, 0
    subdirs: list = []
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
                        subdirs.append(entry.path)
                    continue
                _record_entry(entry, ledger, occupancy)
    except OSError as exc:
        logger.debug(f"Could not size {root}: {exc}")
    subdirs.reverse()
    _walk_into(subdirs, ledger, occupancy, excluded_roots = excluded_roots)
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


def _measurement_roots(definition: CacheDefinition) -> list[Path]:
    "Remembered HF locations count storage without widening a purge."
    roots = _resolve_roots(definition)
    if definition.key == "hf_hub":
        from utils.hf_cache_settings import known_hf_hub_caches
        roots.extend(known_hf_hub_caches())
    elif definition.key in {"hf_datasets", "hf_xet", "hf_assets"}:
        child = definition.key.removeprefix("hf_")
        roots.extend(home / child for home in _hf_homes())
    unique: list[Path] = []
    protected = protected_paths()
    for root in roots:
        resolved = _safe_resolve(root)
        if resolved is None or resolved in unique:
            continue
        try:
            assert_purgeable_root(root, protected = protected, trees = set(), keep = {})
        except CachePurgeRefused:
            continue
        if any(_is_within(resolved, previous) for previous in unique):
            continue
        unique = [previous for previous in unique if not _is_within(previous, resolved)]
        unique.append(resolved)
    return unique


def _nested_active_roots(definition: CacheDefinition, measured: list[Path]) -> list[Path]:
    protected = protected_paths()
    nested: list[Path] = []
    for root in _resolve_roots(definition):
        resolved = _safe_resolve(root)
        if resolved is None or resolved in measured or resolved in nested:
            continue
        if not any(_is_within(resolved, parent) for parent in measured):
            continue
        try:
            assert_purgeable_root(root, protected = protected, trees = set(), keep = {})
        except CachePurgeRefused:
            continue
        nested.append(resolved)
    return nested


def _system_disk_device():
    try:
        return Path(os.path.abspath(os.sep)).stat().st_dev
    except OSError:
        return None


def _system_disk_occupancy(occupancy: _OccupancyLedger) -> int:
    from utils.system_disk import _SAME_POOL_SLACK

    device = _system_disk_device()
    total = occupancy.by_device.get(device, 0)
    if device is None:
        return 0
    try:
        system = shutil.disk_usage(os.path.abspath(os.sep))
    except (OSError, ValueError):
        return total
    for other, size in occupancy.by_device.items():
        if other == device:
            continue
        # As models_disk_usage: APFS Data / btrfs subvolumes share the pool under another st_dev.
        try:
            usage = shutil.disk_usage(occupancy.device_paths[other])
        except (OSError, ValueError):
            continue
        if usage.total == system.total and abs(usage.free - system.free) < _SAME_POOL_SLACK:
            total += size
    return total


def _model_cache_action(root: Path, entry: os.DirEntry) -> dict:
    refused = {"delete_allowed": False}
    try:
        if (
            not entry.name.startswith("models--")
            or entry.is_symlink()
            or _is_junction(entry.path)
            or not entry.is_dir(follow_symlinks = False)
        ):
            return refused
        from hub.utils.paths import is_valid_repo_id
        from hub.utils.hf_cache_state import scoped_delete_root

        repo_id = entry.name[len("models--") :].replace("--", "/")
        if not is_valid_repo_id(repo_id):
            return refused
        candidate = _safe_resolve(Path(entry.path))
        if candidate is None or candidate.parent != root or candidate.name != entry.name:
            return refused
        if scoped_delete_root("model", repo_id, str(candidate)) != root:
            return refused
        return {
            "repo_id": repo_id,
            "repo_type": "model",
            "cache_path": str(candidate),
            "delete_allowed": True,
        }
    except (ImportError, OSError, ValueError, RuntimeError):
        return refused


def cache_breakdown(key: str) -> dict:
    if key not in {"hf_hub", "hf_datasets", "hf_xet", "hf_assets"}:
        raise ValueError("Only Hugging Face caches have a directory breakdown.")
    rows = []
    occupancy = _OccupancyLedger()
    definition = definition_for(key)
    measured = _measurement_roots(definition)
    nested = _nested_active_roots(definition, measured)
    excluded_roots = set(nested)
    for root in nested + measured:
        try:
            with os.scandir(root) as scan:
                for entry in scan:
                    if entry.is_symlink() or _is_junction(entry.path):
                        continue
                    if nested and root not in nested and _safe_resolve(Path(entry.path)) in nested:
                        continue
                    before = occupancy.bytes
                    if entry.is_dir(follow_symlinks = False):
                        _measure_root(
                            Path(entry.path),
                            occupancy = occupancy,
                            excluded_roots = excluded_roots if root not in nested else None,
                        )
                    else:
                        info = (
                            os.stat(entry.path, follow_symlinks = False)
                            if os.name == "nt"
                            else entry.stat(follow_symlinks = False)
                        )
                        occupancy.add(info, entry.path)
                    size = occupancy.bytes - before
                    action = (
                        _model_cache_action(root, entry)
                        if key == "hf_hub"
                        else {"delete_allowed": False}
                    )
                    rows.append(
                        {"name": entry.name, "path": entry.path, "size_bytes": size, **action}
                    )
        except OSError:
            continue
    return {"entries": sorted(rows, key = lambda row: (-row["size_bytes"], row["path"]))}


def describe_cache(definition: CacheDefinition) -> dict:
    patterns = _patterns_for(definition)
    roots = _resolve_roots(definition)
    purgeable = True
    blocked_reason: Optional[str] = None
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
        purgeable = bool(measurable)
    if blocked_reason is None:
        blocked_reason = _link_mode_refusal(definition.key)
        if blocked_reason is not None:
            purgeable = False
    measured_paths = roots
    system_disk_bytes = None
    if definition.key in {"hf_hub", "hf_datasets", "hf_xet", "hf_assets"}:
        measured_paths = _measurement_roots(definition)
        entries = 0
        clear_entries = 0
        occupancy = _OccupancyLedger()
        clear_ledger = LinkLedger()
        clear_roots = {_safe_resolve(root) for root in measurable} if purgeable else set()
        measured_keys = {_safe_resolve(root) for root in measured_paths}
        nested = {root for root in clear_roots if root is not None and root not in measured_keys}
        # Share one stat observation between both ledgers, even while downloads grow.
        for root in measurable if purgeable else []:
            if _safe_resolve(root) not in nested:
                continue
            _, count = _measure_root(
                root, patterns = patterns, ledger = clear_ledger, occupancy = occupancy
            )
            clear_entries += count
        for root in measured_paths:
            clearable = _safe_resolve(root) in clear_roots
            _, count = _measure_root(
                root,
                patterns = patterns,
                occupancy = occupancy,
                ledger = clear_ledger if clearable else None,
                excluded_roots = nested,
            )
            entries += count
            if clearable:
                clear_entries += count
        total = occupancy.bytes
        clear_size = clear_ledger.freeable_bytes()
        system_disk_bytes = _system_disk_occupancy(occupancy)
    else:
        if definition.custom_measure is not None:
            total, entries = (0, 0) if blocked_reason is not None else definition.custom_measure()
        else:
            total = entries = 0
            for root in measurable:
                size, count = _measure_root(root, patterns = patterns)
                total += size
                entries += count
        clear_size = total
        clear_entries = entries
    return {
        "key": definition.key,
        "group": definition.group,
        "opt_in": definition.opt_in,
        "paths": [str(root) for root in roots],
        "measured_paths": [str(root) for root in measured_paths],
        "clear_size_bytes": clear_size,
        "clear_entry_count": clear_entries,
        "system_disk_bytes": system_disk_bytes,
        "size_bytes": total,
        "entry_count": entries,
        "present": bool(roots or measured_paths),
        "purgeable": bool(roots) and purgeable,
        "blocked_reason": blocked_reason,
    }


def _patterns_for(definition: CacheDefinition) -> Optional[tuple[str, ...]]:
    if definition.key == "matplotlib":
        return _matplotlib_patterns()
    if definition.key == "unsloth_compiled":
        return None
    return definition.patterns


_INVENTORY_TTL_SECONDS = 60.0
_BACKGROUND_WALK_SECONDS = 2.0
# Tuple of (walk start, store time, entry, walk seconds): TTL uses store time, forced reads compare walk start.
_size_cache: dict[str, tuple[float, float, dict, float]] = {}
# Bumped per invalidation so a walk begun earlier cannot store a stale size.
_size_epochs: dict[str, int] = {}
_size_cache_lock = threading.Lock()


def invalidate_cache_size(key: str) -> None:
    with _size_cache_lock:
        _size_cache.pop(key, None)
        _size_epochs[key] = _size_epochs.get(key, 0) + 1


_size_flights: dict[str, threading.Lock] = {}
_size_flights_lock = threading.Lock()
# Background walks run one at a time: two stat-bound walks in one process only trade the GIL.
_background_walk_lock = threading.Lock()
_background_pending: set = set()


def _flight_for(key: str) -> threading.Lock:
    with _size_flights_lock:
        return _size_flights.setdefault(key, threading.Lock())


def _walk_and_remember(definition: CacheDefinition, epoch: int) -> dict:
    begun = time.monotonic()
    entry = describe_cache(definition)
    finished = time.monotonic()
    with _size_cache_lock:
        if _size_epochs.get(definition.key, 0) == epoch:
            _size_cache[definition.key] = (begun, finished, entry, finished - begun)
    return entry


def _refresh_in_background(definition: CacheDefinition) -> bool:
    key = definition.key
    with _size_cache_lock:
        if key in _background_pending:
            return True
        _background_pending.add(key)

    def walk() -> None:
        try:
            with _background_walk_lock:
                flight = _flight_for(key)
                if not flight.acquire(blocking = False):
                    return
                try:
                    with _size_cache_lock:
                        epoch = _size_epochs.get(key, 0)
                        remembered = _size_cache.get(key)
                    if (
                        remembered is None
                        or time.monotonic() - remembered[1] < _INVENTORY_TTL_SECONDS
                    ):
                        return
                    _walk_and_remember(definition, epoch)
                finally:
                    flight.release()
        except Exception as exc:  # noqa: BLE001 - a failed refresh leaves the last size in place
            logger.debug(f"Could not refresh the {key} cache size: {exc}")
        finally:
            with _size_cache_lock:
                _background_pending.discard(key)

    try:
        threading.Thread(target = walk, name = f"cache-size-{key}", daemon = True).start()
    except Exception as exc:  # noqa: BLE001 - e.g. no thread left; the next read walks in the foreground
        with _size_cache_lock:
            _background_pending.discard(key)
            remembered = _size_cache.get(key)
            if (
                remembered is not None
                and time.monotonic() - remembered[1] >= _INVENTORY_TTL_SECONDS
            ):
                del _size_cache[key]
        logger.debug(f"Could not start a {key} cache size refresh: {exc}")
        return False
    return True


def _described(
    definition: CacheDefinition,
    *,
    refresh: bool,
    deferred: Optional[list] = None,
    asked_at: Optional[float] = None,
) -> dict:
    started = time.monotonic()
    asked = started if asked_at is None else asked_at
    with _size_cache_lock:
        remembered = None if refresh else _size_cache.get(definition.key)
    if remembered is not None:
        if started - remembered[1] < _INVENTORY_TTL_SECONDS:
            return remembered[2]
        if remembered[3] >= _BACKGROUND_WALK_SECONDS:
            if deferred is not None:
                deferred.append(definition)
                return remembered[2]
            if _refresh_in_background(definition):
                return remembered[2]
    with _flight_for(definition.key):
        with _size_cache_lock:
            epoch = _size_epochs.get(definition.key, 0)
            remembered = _size_cache.get(definition.key)
        # A forced read needs a walk that began after it asked.
        if remembered is not None:
            if refresh:
                if remembered[0] >= asked:
                    return remembered[2]
            elif time.monotonic() - remembered[1] < _INVENTORY_TTL_SECONDS:
                return remembered[2]
        return _walk_and_remember(definition, epoch)


def cache_inventory(*, refresh: bool = False) -> dict:
    deferred: list = []
    asked_at = time.monotonic()
    entries = [
        _described(definition, refresh = refresh, deferred = deferred, asked_at = asked_at)
        for definition in CACHE_DEFINITIONS
    ]
    for definition in deferred:
        _refresh_in_background(definition)
    total = sum(entry["size_bytes"] for entry in entries if entry["present"])
    reclaimable = sum(
        entry.get("clear_size_bytes", entry["size_bytes"])
        for entry in entries
        if entry["present"] and entry["purgeable"] and not entry["opt_in"]
    )
    return {
        "caches": entries,
        "total_bytes": total,
        "reclaimable_bytes": reclaimable,
        "free_bytes": _free_bytes(),
        "total_disk_bytes": _total_disk_bytes(),
        "refreshing": bool(deferred),
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
    path = Path(entry.path)
    try:
        if entry.is_symlink():
            os.unlink(path)
        elif entry.is_dir(follow_symlinks = False):
            real = _safe_resolve(path)
            if real is None or not _is_within(real, root):
                outcome.errors.append(f"Skipped {path}: it resolves outside the cache root")
                return
            _measure_tree(path, ledger)
            try:
                shutil.rmtree(path)
            except OSError:
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


_DOWNLOAD_REGISTRIES = {
    "hf_hub": ("models", "datasets"),
    "hf_xet": ("models", "datasets"),
    "hf_datasets": ("datasets",),
}

_PURGE_BUSY = "Cancel the active downloads before clearing this cache."


def _reserve_downloads(key: str) -> tuple[list, Optional[str]]:
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
    "sd.cpp re-reads resident weights on every generation."
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


_HF_SCANNED_KEYS = frozenset({"hf_hub", "hf_datasets"})
_HF_ROOTED_KEYS = frozenset({"hf_hub", "hf_xet", "hf_datasets"})


def _invalidate_hf_scans() -> None:
    try:
        from hub.utils.inventory_scan import invalidate_hf_cache_scans
    except ImportError as exc:
        logger.debug(f"Could not invalidate the Hugging Face scans: {exc}")
        return
    invalidate_hf_cache_scans(sizes = False)


def invalidate_hf_rooted_sizes() -> None:
    "Memo keys omit paths, so moving models must invalidate cached sizes."
    for key in _HF_ROOTED_KEYS:
        invalidate_cache_size(key)


def purge_caches(keys: Iterable[str]) -> dict:
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
