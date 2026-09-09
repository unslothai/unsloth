# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib.util
import json
import ntpath
import os
import platform
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable
import tempfile

from loggers import get_logger
from utils.paths.path_utils import drop_appledouble_metadata, host_normalize_path

logger = get_logger(__name__)


def _infer_studio_home_from_venv() -> Path | None:
    """Return parent of sys.prefix as STUDIO_HOME when running from an
    installer-managed unsloth_studio venv. Sentinel-gated (share/studio.conf
    or bin shim) so a dev venv named unsloth_studio isn't misidentified.
    """
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
    if has_sentinel:
        return candidate
    return None


def _resolved(value: str) -> Path:
    try:
        return Path(value).expanduser().resolve()
    except (OSError, ValueError):
        return Path(value).expanduser()


def unsloth_home() -> Path | None:
    """The master root, or None. STUDIO_HOME is its studio/ child; llama.cpp, node and
    whisper.cpp are SIBLINGS of studio/, the layout setup.sh already gives UNSLOTH_HOME."""
    override = (os.environ.get("UNSLOTH_HOME") or "").strip()
    return _resolved(override) if override else None


_PORTABLE_ON_VALUES = ("1", "true", "yes", "on")
_PORTABLE_OFF_VALUES = ("0", "false", "off", "no")

# portable_mode() runs on every cache-var lookup, so warn once per process, not once per call.
_warned_unrecognized_portable = False


def _warn_unrecognized_portable(raw: str) -> None:
    global _warned_unrecognized_portable
    if _warned_unrecognized_portable:
        return
    _warned_unrecognized_portable = True
    logger.warning(
        "Ignoring UNSLOTH_PORTABLE=%r: expected one of %s to turn portable mode "
        "on, or one of %s to leave it off.",
        raw,
        "/".join(_PORTABLE_ON_VALUES),
        "/".join(_PORTABLE_OFF_VALUES),
    )


def portable_mode() -> bool:
    """Whether this install keeps everything under one directory. Implied by UNSLOTH_HOME, and
    settable on its own so an existing install can opt in."""
    # Case-folded: UNSLOTH_PORTABLE=FALSE read as "on" would move the caches out from under a
    # user who asked for the opposite.
    raw = (os.environ.get("UNSLOTH_PORTABLE") or "").strip()
    value = raw.lower()
    if value in _PORTABLE_ON_VALUES:
        return True
    if value and value not in _PORTABLE_OFF_VALUES:
        _warn_unrecognized_portable(raw)
    # Unrecognized means no opinion, exactly as an off value does: neither vetoes a real
    # portable install, whose root is what makes it portable.
    return unsloth_home() is not None


# Warned once per distinct pair rather than per call: studio_root() runs many times a request,
# and both variables can be rebound inside one process.
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
    """Unsloth install root.

    Priority: UNSLOTH_STUDIO_HOME, then STUDIO_HOME alias, then UNSLOTH_HOME's studio/ child, then
    sys.prefix inference, then legacy ~/.unsloth/studio. UNSLOTH_STUDIO_HOME outranks both: it
    names this exact directory, while the others only name the tree it sits in.
    """
    override = (os.environ.get("UNSLOTH_STUDIO_HOME") or "").strip()
    if not override:
        override = (os.environ.get("STUDIO_HOME") or "").strip()
    if override:
        resolved = _resolved(override)
        master = unsloth_home()
        # Path.parents excludes the path itself, so a flat layout (both naming one root) would
        # otherwise warn on every call.
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


def cache_root() -> Path:
    """Central cache dir for all studio downloads (models, datasets, etc.)."""
    return studio_root() / "cache"


def llama_slot_cache_root() -> Path:
    """Dir llama-server saves/restores slot KV state in across idle unloads."""
    return cache_root() / "llama-slots"


def studio_bin_root() -> Path:
    """Dir for Unsloth-managed executables (the `unsloth` shim, downloaded tools like cloudflared)."""
    return studio_root() / "bin"


def assets_root() -> Path:
    return studio_root() / "assets"


def datasets_root() -> Path:
    return assets_root() / "datasets"


def dataset_uploads_root() -> Path:
    return datasets_root() / "uploads"


def recipe_datasets_root() -> Path:
    return datasets_root() / "recipes"


def outputs_root() -> Path:
    return studio_root() / "outputs"


def exports_root() -> Path:
    return studio_root() / "exports"


def auth_root() -> Path:
    return studio_root() / "auth"


def auth_db_path() -> Path:
    return auth_root() / "auth.db"


def studio_db_path() -> Path:
    return studio_root() / "studio.db"


def rag_root() -> Path:
    """Root directory for retrieval-augmented-generation state (db + uploads)."""
    return studio_root() / "rag"


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
    # REG_EXPAND_SZ stores it unexpanded, e.g. %USERPROFILE%\Documents. ntpath
    # rather than os.path: %VAR% is Windows syntax, which posixpath leaves as-is.
    return Path(ntpath.expandvars(value) if expandable else value)


def _windows_documents_dir() -> Path | None:
    """Windows' own Documents folder, wherever the user moved it.

    OneDrive's Known Folder Move repoints Documents at the synced copy and
    leaves ~/Documents behind, so that guess writes to the wrong place or to a
    folder that is not there at all.
    """
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


def project_workspaces_root() -> Path:
    override = (os.environ.get("UNSLOTH_STUDIO_PROJECTS_HOME") or "").strip()
    if override:
        return Path(override).expanduser()
    return documents_root() / "Unsloth Studio" / "Projects"


def tmp_root() -> Path:
    return Path(tempfile.gettempdir()) / "unsloth-studio"


def seed_uploads_root() -> Path:
    return datasets_root() / "seed-uploads"


def unstructured_seed_cache_root() -> Path:
    return tmp_root() / "unstructured-seed-cache"


def unstructured_uploads_root() -> Path:
    return datasets_root() / "unstructured-uploads"


def oxc_validator_tmp_root() -> Path:
    return tmp_root() / "oxc-validator"


def tensorboard_root() -> Path:
    return studio_root() / "runs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents = True, exist_ok = True)
    return path


def legacy_hf_cache_dir() -> Path:
    """Old Unsloth-specific HF hub cache, kept for backward-compat scans."""
    return cache_root() / "huggingface" / "hub"


def hf_default_cache_dir() -> Path:
    """Platform default HuggingFace hub cache (ignoring env overrides).

    Where HF caches when no ``HF_HUB_CACHE`` / ``HF_HOME`` is set. Scanned
    so models downloaded *before* installing Unsloth Studio are discovered.
    """
    return Path.home() / ".cache" / "huggingface" / "hub"


def _host_path(path: str | Path) -> Path:
    """Expand a configured path into one this process can stat.

    A drive-letter path from another tool's config means nothing to a WSL process
    until it is mapped under the automount root.
    """
    return Path(host_normalize_path(str(path))).expanduser()


def _existing_dirs(candidates: Iterable[str | Path], *, resolve: bool) -> list[Path]:
    """Host-translate *candidates*, drop non-directories, dedupe by real path.

    *resolve* picks the return shape: ``well_known_model_dirs`` feeds a containment
    check and needs real paths, while the per-tool lists feed model ids and must keep
    the spelling the user configured.
    """
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
    """Custom models folder from LM Studio's settings.json, or "" if unset.

    utf-8-sig: LM Studio may write this file with a BOM, which a plain utf-8 read turns
    into a JSONDecodeError that used to be swallowed, dropping the folder (#9748).
    """
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
    """Return Ollama model directories that exist on disk.

    User-level plus the common system-wide install paths
    (https://github.com/ollama/ollama/issues/733).
    """
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
    """The Hermes root a download hangs off, mirroring its own resolution.

    HERMES_HOME under the native home (the normal and profile layouts) still
    means the native home; a ``<root>/profiles/<name>`` path elsewhere means
    ``<root>``; anything else IS the root (Docker / custom deployments).
    """
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
    """Return Hermes model directories that exist on disk.

    Hermes Desktop's one-click GGUF downloads land in ``<root>/models``. That is
    machine-scoped upstream, never profile-scoped -- a 20 GB GGUF is a machine
    asset and every profile shares the one server that runs it -- so it hangs off
    the root, not off HERMES_HOME when that names a profile.

    The native root is scanned as well, because ``unsloth start hermes`` points
    HERMES_HOME at a throwaway session dir while the user's real downloads stay
    under the native home; scanning only the resolved root would lose them for the
    duration of a session Studio launched itself.
    """
    return _existing_dirs(
        [_hermes_root() / "models", _hermes_native_home() / "models"],
        resolve = False,
    )


def well_known_model_dirs() -> list[Path]:
    """Return directories commonly used by other local LLM tools.

    Backs the folder browser's quick-pick chips. Returns only paths that
    exist on disk, so the UI never shows dead chips. Order reflects rough
    likelihood of models being there -- LM Studio, Ollama and Hermes first,
    then generic fallbacks.
    """
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
    """Whether HF_HOME was set by the user rather than seeded by Unsloth.

    initialize_hf_cache_environment fills a blank HF_HOME before this runs, so hf_cache_settings'
    import-time snapshot is the only record of who chose it.
    """
    try:
        from utils.hf_cache_settings import _EXPLICIT_CACHE_ENV
    except ImportError:
        return False
    return bool(_EXPLICIT_CACHE_ENV.get("HF_HOME"))


def _portable_cache_defaults(root: Path) -> dict[str, str]:
    """Cache vars that only move under the root in portable mode, since they hold shared user data
    or large re-downloads. The hub and xet caches are handled inside hf_cache_settings, which must
    agree with the Settings UI. HF_HOME never moves: credentials should not follow a cache onto a
    removable volume.
    """
    if not portable_mode():
        return {}
    if _user_set_hf_home():
        # Assets, datasets and modules derive from an explicit HF_HOME, so pinning them here
        # would split one deliberately chosen cache across two volumes.
        return {"TORCH_HOME": str(root / "torch")}
    return {
        "HF_DATASETS_CACHE": str(root / "huggingface" / "datasets"),
        # These two derive from <HF_HOME>, which stays on the host, so they are the HF roots
        # that would otherwise still write outside the volume.
        "HF_ASSETS_CACHE": str(root / "huggingface" / "assets"),
        # transformers.utils.hub reads this at import and appends it to sys.path, so a
        # trust_remote_code load leaves generated modules on the host without it.
        "HF_MODULES_CACHE": str(root / "huggingface" / "modules"),
        "TORCH_HOME": str(root / "torch"),
    }


def _triton_cache_defaults(root: Path) -> dict[str, str]:
    """Triton's regenerable directories, named one at a time.

    TRITON_HOME would move all three at once (triton-lang/triton#4265), and that is why it is not
    used: ~/.triton/override holds hand-written kernels, so moving their parent makes a
    TRITON_KERNEL_OVERRIDE=1 run silently fall back to the compiler's own output. The dedicated
    variables outrank the derivation (triton/knobs.py cache_knobs).
    """
    if (os.environ.get("TRITON_HOME") or "").strip():
        # Whoever moved the whole tree meant the cache with it; TRITON_CACHE_DIR would outrank it.
        return {}
    return {
        "TRITON_CACHE_DIR": str(root / "triton"),
        # A sibling, not a child: dumps are asked for by hand and should outlive a cache wipe.
        "TRITON_DUMP_DIR": str(root / "triton-dump"),
    }


def _nothing_at(path: Path, *, ending: str = "") -> bool:
    """Whether *path* positively holds nothing, the only state that licenses a pin.

    Not Path.exists/is_dir/glob: they report ENOTDIR, ELOOP, EBADF (and from 3.14 EACCES and EIO)
    as absence, so a directory we merely cannot inspect would read as empty and the redirect would
    hide the user's files. os.lstat and os.scandir raise on all of it. lstat rather than stat: a
    dangling symlink is still something the user put there. *ending* asks instead whether the
    DIRECTORY holds no entry with that suffix; pass it lowercase, as Windows glob matched
    case-insensitively.
    """
    try:
        if not ending:
            os.lstat(path)
            return False
        with os.scandir(path) as entries:
            return not any(entry.name.lower().endswith(ending) for entry in entries)
    except FileNotFoundError:
        return True
    except (OSError, ValueError):
        # Could not look. Declining a pin costs a shared cache directory; taking one wrongly
        # hides the configuration or datasets underneath it.
        return False


def _matplotlib_config_dir() -> Path | None:
    """Where matplotlib reads matplotlibrc and stylelib/ from when MPLCONFIGDIR is unset, or None
    when this machine has no such directory. Mirrors _get_config_or_cache_dir: XDG config base on
    Linux/FreeBSD, %LOCALAPPDATA% on Windows but keeping a pre-existing ~/.matplotlib there.

    The Windows branch is matplotlib 3.11's; the pinned 3.10.9 sends every non-XDG platform to
    ~/.matplotlib. The disagreement is one-way and safe: an rc under %LOCALAPPDATA% that 3.10.9
    would ignore only costs us the pin, and it never hides a file matplotlib does read.

    None means matplotlib falls back to a temporary directory, so a pin can strand nothing.
    """
    # XDG_CONFIG_HOME ahead of Path.home(), as _get_xdg_config_dir does: an install that sets it
    # has a config dir even with no resolvable home.
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
        # Not is_dir(): an unreadable ~/.matplotlib raises before 3.14 and reads as absent from
        # 3.14, which sends the probe to a %LOCALAPPDATA% that is empty on exactly the machines
        # where the old directory holds the rc.
        if not _nothing_at(legacy):
            return legacy
        local_app_data = (os.environ.get("LOCALAPPDATA") or "").strip()
        return Path(local_app_data) / "matplotlib" if local_app_data else legacy
    return home / ".matplotlib"


def _matplotlib_defaults(root: Path) -> dict[str, str]:
    """MPLCONFIGDIR, unless matplotlib's own directory holds user configuration.

    The one variable moves the CONFIG directory as well as the cache, so pinning it would drop a
    user matplotlibrc and every custom style, silently changing the loss plots
    core/training/training.py draws. matplotlib creates the directory on import, so contents
    decide, not existence.
    """
    config_dir = _matplotlib_config_dir()
    if config_dir is not None and not (
        _nothing_at(config_dir / "matplotlibrc")
        and _nothing_at(config_dir / "stylelib", ending = ".mplstyle")
    ):
        return {}
    return {"MPLCONFIGDIR": str(root / "matplotlib")}


def _data_designer_in_use(home: Path) -> bool:
    """Whether the managed Data Designer home holds work worth keeping.

    _setup_cache_env creates this directory and its managed-assets child on the first launch, so
    existence alone would pin a home nobody has written to. Same rule as _nothing_at read for
    contents: a home we merely cannot list still counts as in use, since dropping the pin would
    hide the recipes here behind a re-seeded ~/.data-designer.
    """
    try:
        entries = list(home.iterdir())
    except FileNotFoundError:
        return False
    except (OSError, ValueError):
        return True
    for entry in entries:
        try:
            if entry.name != "managed-assets" or not entry.is_dir():
                return True
            if any(entry.iterdir()):
                return True
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            return True
    return False


def _data_designer_defaults(root: Path) -> dict[str, str]:
    """Data Designer's home, unless the user already has one.

    Not a cache: repointing an existing ~/.data-designer hides its yaml configs and multi-GB
    parquet behind a re-seeded default. MANAGED_ASSETS_PATH derives from the home, so it is only
    ours to set when the home is. data_designer.config.utils.constants reads both at IMPORT time.
    """
    if (os.environ.get("DATA_DESIGNER_HOME") or "").strip():
        return {}
    home = root.parent / "data-designer"
    pinned = {
        "DATA_DESIGNER_HOME": str(home),
        "DATA_DESIGNER_MANAGED_ASSETS_PATH": str(home / "managed-assets"),
    }
    # The legacy probe re-runs every launch, so on its own it would hand the recipes written here
    # to a ~/.data-designer created later by a standalone run. Our own populated home is the
    # record of the first choice.
    if _data_designer_in_use(home):
        return pinned
    try:
        legacy = Path.home() / ".data-designer"
    except (OSError, RuntimeError):
        # No home, so this pin can hide nothing: data_designer's own default is equally
        # unavailable, off the same call.
        return pinned
    return pinned if _nothing_at(legacy) else {}


def _path_safe(value: str) -> str:
    """A directory-name-safe rendering of a build field."""
    return re.sub(r"[^A-Za-z0-9.]+", "-", value)


def _torch_version_fields() -> dict[str, str]:
    """The build identity torch.version exposes, read without importing torch.

    Runs on a startup path that executes before torch exists in a fresh venv, so it stays a file
    read. torch/version.py assigns only literals, but whether it annotates them
    (``cuda: Optional[str] = ...``) varies by release, hence a regex rather than ast or exec.
    """
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
    """Name the extension cache after the runtime that builds into it.

    torch.utils.cpp_extension._get_build_directory appends a ``py<ver>_<accelerator>`` folder to
    the DEFAULT root only, never to a TORCH_EXTENSIONS_DIR we supply, so pinning a flat path drops
    the isolation that keeps a py313/cu128 build from being loaded by a py312/cu126 one.

    The accelerator comes from the generated cuda/hip fields rather than a local segment of
    __version__: conda-forge's CPU and CUDA packages of one release carry the same __version__.
    __version__ stays in the tag too, since local segments such as +cpu.cxx11.abi mark ABI splits
    no other field records.
    """
    tag = f"py{sys.version_info.major}{sys.version_info.minor}{getattr(sys, 'abiflags', '')}"
    # Architecture too, which torch's own py<ver>_<cu_str> naming omits: an arm64 python and a
    # Rosetta x86_64 python on one Mac agree on every other field, so ninja reads the other's
    # build as up to date. Same shape for a $HOME an aarch64 and an x86_64 host both mount.
    tag += "_" + _path_safe(f"{sys.platform}-{platform.machine() or 'unknown'}")
    try:
        fields = _torch_version_fields()
    except (ImportError, OSError, ValueError, AttributeError):
        # No torch yet, or a half-built source tree. The interpreter tag alone still isolates
        # more than the flat path it replaces.
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


def _setup_cache_env() -> None:
    """Set cache env vars for HuggingFace, uv, and vLLM.

    Explicit Hugging Face environment variables take precedence over Unsloth's
    stored location. Unsloth seeds import-time variables once, while each later
    worker receives its own captured cache location.
    """
    root = cache_root()
    from utils.hf_cache_settings import initialize_hf_cache_environment

    initialize_hf_cache_environment()
    defaults: dict[str, str] = {
        "UV_CACHE_DIR": str(root / "uv"),
        "VLLM_CACHE_ROOT": str(root / "vllm"),
        # unsloth_zoo defaults this to a bare relative name.
        # It resolves against the CWD and the Windows launcher runs Unsloth with WorkingDirectory=%USERPROFILE%, so the
        # cache landed in the user home. Must be set before unsloth_zoo.compiler imports: it reads the value at import
        # time and puts it on sys.path.
        "UNSLOTH_COMPILE_LOCATION": str(root.parent / "compiled_cache"),
        # Regenerable and process-scoped. Shared user data (HF hub cache, torch.hub checkpoints)
        # stays where the other tools look, except in portable mode.
        "TORCHINDUCTOR_CACHE_DIR": str(root / "torchinductor"),
        # Tagged: torch only inserts its own ABI folder when TORCH_EXTENSIONS_DIR is unset, so a
        # flat pin would let two runtimes sharing this root import each other's .so.
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
        # Blank counts as unset: an inherited KEY= would otherwise pin the cache to "", which puts an empty entry on
        # sys.path and sends the compiler to the system temp directory instead.
        if not (os.environ.get(key) or "").strip():
            os.environ[key] = value
            # Best-effort: a non-writable custom HF_HOME must not crash startup
            try:
                created = True
                try:
                    Path(value).mkdir(parents = True, exist_ok = False)
                except FileExistsError:
                    created = False
                if key == "UNSLOTH_COMPILE_LOCATION" and created:
                    # Marks the directory as ours, so the cleanup can delete
                    # from it without inferring that from its contents. Only when
                    # this call made it: the marker is what licenses an rmtree.
                    from utils.cache_cleanup import CACHE_MARKER
                    (Path(value) / CACHE_MARKER).touch(exist_ok = True)
            except (OSError, ImportError):
                pass


def setup_cache_env() -> None:
    """Seed the cache env vars without creating every studio directory.

    For `uvicorn main:app`, which bypasses run.py and so never reaches
    ensure_studio_directories, but still has to pin UNSLOTH_COMPILE_LOCATION
    before unsloth_zoo.compiler is imported.
    """
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
    """Return true when a user path contains a parent-directory segment.

    On POSIX, ``Path("E:\\foo\\..\\bar")`` treats backslashes as normal
    characters, so check both the host parser and Windows-style parsing.
    """
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


def resolve_under_root(
    path_value: str | None,
    *,
    root: Path,
    strip_prefixes: tuple[str, ...] = (),
) -> Path:
    """Resolve ``path_value`` and assert the result is under ``root``.

    Absolutes are accepted only if already contained (so pre-resolved
    internal paths re-enter idempotently); schemas reject absolutes upstream.
    """
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
    # Repo ids keep their namespace while local paths collapse to their final component, so an absolute source cannot
    # escape outputs_root; length-capped to the filesystem name limit.
    # Repo ids keep their namespace (org/model -> org_model).
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
    """Resolve an export directory — contained under exports_root().

    Used by scan/read endpoints. Use :func:`resolve_export_write_dir`
    for the export write path where absolute paths are accepted.
    """
    return resolve_under_root(
        path_value,
        root = exports_root(),
        strip_prefixes = ("exports",),
    )


def resolve_export_write_dir(path_value: str | None = None) -> Path:
    """Resolve an export save directory — accepts absolute paths.

    Unlike :func:`resolve_export_dir`, this function passes absolute
    paths through as-is so users can target a different drive when
    their Unsloth install lives on a constrained system volume
    (see :gh-issue:`6082`). Used only by the export write path.
    """
    if not path_value or not str(path_value).strip():
        return exports_root()
    raw = str(path_value).strip()
    if "\x00" in raw:
        raise ValueError("path may not contain null bytes")
    path = Path(raw).expanduser()
    if _has_parent_segment(raw, path):
        raise ValueError(f"path may not contain '..' segments: {raw!r}")
    if _is_absolute_user_path(path):
        return path
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
                return path
            except ValueError:
                continue
        raise ValueError(f"dataset path must be relative or under a dataset root: {raw!r}")

    parts = [part for part in Path(path_value).parts if part not in ("", ".")]
    if parts[:2] == ["assets", "datasets"]:
        parts = parts[2:]
    if parts and parts[0] == "uploads":
        cleaned = Path(*parts[1:]) if len(parts) > 1 else Path()
        return dataset_uploads_root() / cleaned
    if parts and parts[0] == "recipes":
        cleaned = Path(*parts[1:]) if len(parts) > 1 else Path()
        return recipe_datasets_root() / cleaned

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
            return candidate
    return candidates[0]
