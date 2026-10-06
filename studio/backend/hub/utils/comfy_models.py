# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ComfyUI models-folder discovery and loose single-file diffusion checkpoints.

A ComfyUI install keeps its weights by role under ``models/``: denoisers in ``diffusion_models/``
(``unet/`` on older installs) and ``checkpoints/``, text encoders in ``text_encoders/`` (``clip/``)
and VAEs in ``vae/``. Each of those folders holds many loose ``.safetensors`` files side by side,
none with a ``config.json``, so the generic scan (which lists folders and loose GGUFs) showed only
the GGUFs. A user registering a ComfyUI root, or its ``models/`` folder, as a scan folder gets the
denoiser folders scanned, with every loose checkpoint listed as its own row when its header says it
is a diffusion model. ``extra_model_paths.yaml`` beside the root adds more folders.

Text-encoder and VAE folders are only resolved here (``ComfyLayout.text_encoder_dirs`` /
``vae_dirs``), for a split-layout loader to use; nothing loads from them yet.

Pure filesystem helpers, no torch: the scanners call these on every listing.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from utils.paths.path_utils import is_appledouble_metadata

# ComfyUI folder names per role. ``unet`` and ``clip`` are the legacy names of ``diffusion_models``
# and ``text_encoders``; ComfyUI still reads both, so a user may have either.
DIT_FOLDERS = ("diffusion_models", "unet", "checkpoints")
TEXT_ENCODER_FOLDERS = ("text_encoders", "clip")
VAE_FOLDERS = ("vae",)
_ROLE_FOLDERS = DIT_FOLDERS + TEXT_ENCODER_FOLDERS + VAE_FOLDERS
# Folders only a ComfyUI ``models/`` dir has; one of them beside a denoiser folder is the tell.
_COMFY_ONLY_FOLDERS = frozenset(
    {
        *_ROLE_FOLDERS,
        "loras",
        "clip_vision",
        "controlnet",
        "upscale_models",
        "embeddings",
        "style_models",
        "model_patches",
        "audio_encoders",
        "latent_upscale_models",
    }
)
EXTRA_MODEL_PATHS_FILE = "extra_model_paths.yaml"
# The yaml is user-authored config, not a model: never read a large file on the listing path.
_MAX_EXTRA_PATHS_BYTES = 256 * 1024

# Transformers / diffusers shard names: a piece of a sharded model, never a whole checkpoint.
_SHARD_RE = re.compile(r"-\d{3,}-of-\d{3,}(?:\.[^.]+)?\.safetensors$", re.IGNORECASE)


@dataclass(frozen = True)
class ComfyLayout:
    """The role folders of one ComfyUI install, existing directories only, deduplicated."""

    models_dir: Optional[Path]
    dit_dirs: tuple[Path, ...] = ()
    text_encoder_dirs: tuple[Path, ...] = ()
    vae_dirs: tuple[Path, ...] = ()


def _is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _is_comfy_models_dir(path: Path) -> bool:
    """A ``models/`` folder: at least one denoiser folder plus one other ComfyUI-only folder."""
    try:
        names = {entry.name for entry in os.scandir(path) if entry.is_dir()}
    except OSError:
        return False
    if not names & set(DIT_FOLDERS):
        return False
    return len(names & _COMFY_ONLY_FOLDERS) >= 2


def _split_paths(value) -> list[str]:
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, (list, tuple)):
        return [
            str(v).strip() for v in value if isinstance(v, (str, os.PathLike)) and str(v).strip()
        ]
    return []


def read_extra_model_paths(yaml_path: Path) -> dict[str, list[Path]]:
    """``{folder role: [paths]}`` from a ComfyUI ``extra_model_paths.yaml``. Never raises.

    Each top-level section may set ``base_path`` (``~`` and ``$VARS`` expanded, relative to the
    yaml's folder); every other key names a model folder role and holds one path or a ``|`` block
    of one path per line, joined onto ``base_path`` or, without one, resolved against the yaml's
    folder. ``is_default`` is ComfyUI's own ordering flag and is skipped. Only the roles Studio
    uses are kept."""
    out: dict[str, list[Path]] = {}
    try:
        if yaml_path.stat().st_size > _MAX_EXTRA_PATHS_BYTES:
            return out
        import yaml
        with open(yaml_path, "r", encoding = "utf-8-sig") as handle:
            config = yaml.safe_load(handle)
    except Exception:  # noqa: BLE001 - a broken user yaml must not break the listing
        return out
    if not isinstance(config, dict):
        return out
    yaml_dir = yaml_path.resolve().parent
    for section in config.values():
        if not isinstance(section, dict):
            continue
        base: Optional[Path] = None
        raw_base = section.get("base_path")
        if isinstance(raw_base, str) and raw_base.strip():
            base = Path(os.path.expandvars(os.path.expanduser(raw_base.strip())))
            if not base.is_absolute():
                base = yaml_dir / base
        for role, value in section.items():
            if role in ("base_path", "is_default") or role not in _ROLE_FOLDERS:
                continue
            for entry in _split_paths(value):
                path = Path(os.path.expandvars(os.path.expanduser(entry)))
                if not path.is_absolute():
                    path = (base or yaml_dir) / path
                out.setdefault(role, []).append(Path(os.path.normpath(path)))
    return out


def _dedupe_dirs(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        if not _is_dir(path):
            continue
        try:
            key = os.path.normcase(os.path.realpath(path))
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return tuple(out)


def _allowed(path: Path) -> bool:
    """Paths a yaml names are user input: keep them out of the same system folders a scan folder
    may not be registered under."""
    try:
        from hub.storage.scan_folders import is_denied_system_path
        return not is_denied_system_path(os.path.realpath(path))
    except Exception:  # noqa: BLE001
        return True


def comfy_layout(folder: Path) -> Optional[ComfyLayout]:
    """The ComfyUI layout ``folder`` is the root of, or the ``models/`` dir of, else None.

    A root is a folder whose ``models/`` passes the models-dir test, or one carrying an
    ``extra_model_paths.yaml``; the yaml's folders are added to the ``models/`` ones."""
    folder = Path(folder)
    models_dir: Optional[Path] = None
    yaml_path: Optional[Path] = None
    if _is_comfy_models_dir(folder):
        models_dir = folder
        candidate = folder.parent / EXTRA_MODEL_PATHS_FILE
        if candidate.is_file():
            yaml_path = candidate
    else:
        if _is_dir(folder / "models") and _is_comfy_models_dir(folder / "models"):
            models_dir = folder / "models"
        candidate = folder / EXTRA_MODEL_PATHS_FILE
        try:
            if candidate.is_file():
                yaml_path = candidate
        except OSError:
            yaml_path = None
    if models_dir is None and yaml_path is None:
        return None
    extra = read_extra_model_paths(yaml_path) if yaml_path is not None else {}

    def _role(names: tuple[str, ...]) -> tuple[Path, ...]:
        local = [models_dir / name for name in names] if models_dir is not None else []
        listed = [p for name in names for p in extra.get(name, ()) if _allowed(p)]
        return _dedupe_dirs([*local, *listed])

    layout = ComfyLayout(
        models_dir = models_dir,
        dit_dirs = _role(DIT_FOLDERS),
        text_encoder_dirs = _role(TEXT_ENCODER_FOLDERS),
        vae_dirs = _role(VAE_FOLDERS),
    )
    if not (layout.dit_dirs or layout.text_encoder_dirs or layout.vae_dirs):
        return None
    return layout


_MAX_SUBDIR_DEPTH = 3
_MAX_SUBDIRS = 200


def _plain_subdirs(root: Path) -> list[Path]:
    """Sub-folders of a denoiser folder, as ComfyUI also reads them (``diffusion_models/wan/x``):
    bounded, no hidden folders, no symlinked folders, not into a folder that is itself one model
    (a ``config.json`` / ``model_index.json`` beside its weights)."""
    out: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack and len(out) < _MAX_SUBDIRS:
        current, depth = stack.pop()
        if depth >= _MAX_SUBDIR_DEPTH:
            continue
        try:
            with os.scandir(current) as entries:
                children = sorted(
                    Path(e.path)
                    for e in entries
                    if not e.name.startswith(".") and e.is_dir(follow_symlinks = False)
                )
        except OSError:
            continue
        for child in children:
            if any((child / m).exists() for m in ("config.json", "model_index.json")):
                continue
            out.append(child)
            stack.append((child, depth + 1))
    return out[:_MAX_SUBDIRS]


def comfy_dit_scan_roots(folder: Path) -> tuple[Path, ...]:
    """The denoiser folders to scan for a registered scan folder that is a ComfyUI root or
    ``models/`` dir (with their plain sub-folders); empty for any other folder."""
    layout = comfy_layout(folder)
    if layout is None:
        return ()
    roots: list[Path] = []
    for root in layout.dit_dirs:
        roots.append(root)
        roots.extend(_plain_subdirs(root))
    return tuple(roots)


def is_loose_checkpoint_candidate(path: Path) -> bool:
    """A loose ``.safetensors`` that could be a whole single-file checkpoint by its name: not a
    shard, not a PEFT adapter, not macOS metadata."""
    name = path.name
    lower = name.lower()
    if not lower.endswith(".safetensors") or is_appledouble_metadata(path):
        return False
    if _SHARD_RE.search(lower) or lower in ("adapter_model.safetensors",):
        return False
    return True


def loose_diffusion_checkpoints(folder: Path, *, entry_limit: Optional[int] = None) -> list[Path]:
    """Loose ``.safetensors`` files directly in ``folder`` that the header check offers as a
    diffusion model (text encoders, VAEs and LoRAs are not; an unclassifiable header falls back to
    the file name), sorted by name.

    Only for a folder holding no ``config.json`` / ``adapter_config.json`` / ``model_index.json``:
    such a folder is one model and is listed whole. Never raises."""
    try:
        for marker in (
            "config.json",
            "adapter_config.json",
            "model_index.json",
            "modular_model_index.json",
        ):
            if (folder / marker).exists():
                return []
        files: list[Path] = []
        with os.scandir(folder) as entries:
            for index, entry in enumerate(entries, start = 1):
                if entry_limit is not None and index > entry_limit:
                    break
                path = Path(entry.path)
                try:
                    if is_loose_checkpoint_candidate(path) and entry.is_file():
                        files.append(path)
                except OSError:
                    continue
    except OSError:
        return []
    if not files:
        return []
    offered = []
    for path in sorted(files):
        try:
            if _offer_loose_checkpoint(path):
                offered.append(path)
        except Exception:  # noqa: BLE001 - an unreadable header is not a listing failure
            continue
    return offered


def _name_detects_family(name: str) -> bool:
    from core.inference.diffusion_families import detect_family
    from core.inference.video_families import detect_video_family
    return detect_family(name) is not None or detect_video_family(name) is not None


def _offer_loose_checkpoint(path: Path) -> bool:
    """The header decides: a DiT of a supported family is offered, a DiT of no supported family and
    a text encoder / VAE / LoRA / ControlNet never are. A header that classifies as neither (unreadable, unknown layout) falls back to the name, so a
    stray ``foo.safetensors`` beside the models is not offered as one."""
    from core.inference import diffusion_content

    if not diffusion_content.offer_as_dit(str(path)):
        return False
    info = diffusion_content.inspect_checkpoint(str(path))
    if info.role == diffusion_content.ROLE_DIT:
        # a DiT of an architecture Studio has no family for is not a loadable pick
        return bool(info.family)
    return _name_detects_family(path.name)
