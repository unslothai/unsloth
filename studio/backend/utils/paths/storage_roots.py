# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile


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


def studio_root() -> Path:
    """Unsloth install root.

    Priority: UNSLOTH_STUDIO_HOME, then STUDIO_HOME alias, then sys.prefix
    inference, then legacy ~/.unsloth/studio. UNSLOTH_STUDIO_HOME wins if
    both are set (specific signal beats generic alias).
    """
    override = (os.environ.get("UNSLOTH_STUDIO_HOME") or "").strip()
    if not override:
        override = (os.environ.get("STUDIO_HOME") or "").strip()
    if override:
        try:
            return Path(override).expanduser().resolve()
        except (OSError, ValueError):
            return Path(override).expanduser()
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
    except OSError:
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


def documents_root() -> Path:
    override = (os.environ.get("UNSLOTH_STUDIO_DOCUMENTS_HOME") or "").strip()
    if override:
        return Path(override).expanduser()
    return _xdg_user_dir("XDG_DOCUMENTS_DIR") or (Path.home() / "Documents")


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


def lmstudio_model_dirs() -> list[Path]:
    """Return LM Studio model directories that exist on disk."""
    dirs: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path) -> None:
        resolved = p.resolve()
        if resolved not in seen and p.is_dir():
            seen.add(resolved)
            dirs.append(p)

    # LM Studio settings.json custom downloads folder
    settings_path = Path.home() / ".lmstudio" / "settings.json"
    if settings_path.is_file():
        try:
            with open(settings_path) as f:
                settings = json.load(f)
            downloads = settings.get("downloadsFolder", "")
            if downloads:
                _add(Path(downloads).expanduser())
        except Exception:
            pass

    # LM Studio default models directory (all platforms)
    _add(Path.home() / ".lmstudio" / "models")

    # Legacy LM Studio cache location
    _add(Path.home() / ".cache" / "lm-studio" / "models")

    return dirs


def well_known_model_dirs() -> list[Path]:
    """Return directories commonly used by other local LLM tools.

    Backs the folder browser's quick-pick chips. Returns only paths that
    exist on disk, so the UI never shows dead chips. Order reflects rough
    likelihood of models being there -- LM Studio and Ollama first, then
    generic fallbacks.
    """
    candidates: list[Path] = []

    # LM Studio (reuses the logic above, including settings.json override)
    candidates.extend(lmstudio_model_dirs())

    # Ollama -- user-level and common system-wide install paths
    # (https://github.com/ollama/ollama/issues/733).
    ollama_env = os.environ.get("OLLAMA_MODELS")
    if ollama_env:
        candidates.append(Path(ollama_env).expanduser())
    candidates.append(Path.home() / ".ollama" / "models")
    candidates.append(Path("/usr/share/ollama/.ollama/models"))
    candidates.append(Path("/var/lib/ollama/.ollama/models"))

    # HF hub cache root (separate from the explicit HF cache chip)
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    # Generic "my models" spots users drop things into
    for name in ("models", "Models"):
        candidates.append(Path.home() / name)

    # Dedupe preserving order; keep only extant dirs
    out: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        try:
            resolved = str(p.resolve())
        except OSError:
            continue
        if resolved in seen:
            continue
        if Path(resolved).is_dir():
            seen.add(resolved)
            out.append(Path(resolved))
    return out


def _setup_cache_env() -> None:
    """Set cache env vars for HuggingFace, uv, and vLLM.

    Explicit Hugging Face environment variables take precedence over Studio's
    stored location. Studio seeds import-time variables once, while each later
    worker receives its own captured cache location.
    """
    root = cache_root()
    from utils.hf_cache_settings import initialize_hf_cache_environment

    initialize_hf_cache_environment()
    defaults: dict[str, str] = {
        "UV_CACHE_DIR": str(root / "uv"),
        "VLLM_CACHE_ROOT": str(root / "vllm"),
    }
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            # Best-effort: a non-writable custom HF_HOME must not crash startup;
            # HF surfaces a clear error at download time instead.
            try:
                Path(value).mkdir(parents = True, exist_ok = True)
            except OSError:
                pass


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
            f"path escapes root: {resolved!s} -> {resolved_real!s} " f"is not under {root_real!s}"
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
    # Folder-safe run name for an auto-created output dir. Repo ids keep their
    # namespace (org/model -> org_model); local paths (incl. G:\dir\model)
    # collapse to their final component so an absolute source can't escape
    # outputs_root. Length-capped to stay under the filesystem name limit.
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
