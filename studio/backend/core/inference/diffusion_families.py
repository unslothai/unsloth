# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pure helpers for diffusion model identification.

No torch/diffusers imports here: everything in this module is a pure function of
its string/path arguments so it can be unit-tested without the heavy runtime.

A diffusion checkpoint published as a single-file GGUF only carries the
transformer weights; the matching VAE / text encoders / scheduler come from a
companion ``diffusers`` base repo. ``DiffusionFamily`` maps a checkpoint to the
diffusers classes and base repo needed to assemble the full pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Optional


@dataclass(frozen = True)
class DiffusionFamily:
    name: str
    pipeline_class: str
    transformer_class: str
    base_repo: str
    # Extra lowercased substrings (besides ``name``) that map a repo id here.
    aliases: tuple[str, ...] = field(default_factory = tuple)


# MVP: Z-Image-Turbo only. Its single-file GGUF is the transformer (Lumina2 DiT);
# the VAE / text-encoder / scheduler come from the base diffusers repo. More
# families (FLUX, Qwen-Image) can be appended here when we ship them.
_FAMILIES: tuple[DiffusionFamily, ...] = (
    DiffusionFamily(
        name = "z-image",
        pipeline_class = "ZImagePipeline",
        transformer_class = "ZImageTransformer2DModel",
        base_repo = "Tongyi-MAI/Z-Image-Turbo",
        aliases = ("z-image-turbo", "zimage", "z_image"),
    ),
)


def detect_family(repo_id: str, override: Optional[str] = None) -> Optional[DiffusionFamily]:
    """Resolve a ``DiffusionFamily`` from a repo id, or an explicit override.

    ``override`` matches a family ``name`` or alias exactly; otherwise the repo
    id is scanned for the first family whose name/alias appears in it. Returns
    ``None`` when nothing matches so the caller can raise a clean error.
    """
    if override:
        key = override.strip().lower()
        for fam in _FAMILIES:
            if key == fam.name or key in fam.aliases:
                return fam
        return None
    needle = repo_id.lower()
    for fam in _FAMILIES:
        if fam.name in needle or any(alias in needle for alias in fam.aliases):
            return fam
    return None


def resolve_base_repo(fam: DiffusionFamily, base_repo: Optional[str]) -> str:
    """The companion diffusers repo: caller-supplied if given, else the family default."""
    base = (base_repo or "").strip()
    return base or fam.base_repo


def resolve_local_gguf_child(repo_root: Path, gguf_filename: str) -> Path:
    """Resolve ``gguf_filename`` to a file under ``repo_root``, rejecting escapes.

    ``gguf_filename`` is user-supplied, so an absolute path or a ``..`` segment
    could otherwise read outside the local repo directory.
    """
    if (
        Path(gguf_filename).is_absolute()
        or PurePosixPath(gguf_filename).is_absolute()
        or gguf_filename.startswith(("/", "\\"))
        or "\\" in gguf_filename
    ):
        raise ValueError("gguf_filename must be a relative path inside the repo.")
    rel = PurePosixPath(gguf_filename)
    if any(part in ("", ".", "..") for part in rel.parts):
        raise ValueError("gguf_filename must not contain '', '.', or '..' segments.")
    # Resolve symlinks before the containment check: the guards above stop
    # lexical escapes, but a symlink inside the repo could still point outside it.
    repo_real = repo_root.resolve()
    child = repo_root.joinpath(*rel.parts).resolve()
    if child != repo_real and repo_real not in child.parents:
        raise ValueError("gguf_filename must resolve to a file inside the repo.")
    if not child.exists():
        raise FileNotFoundError(f"'{gguf_filename}' not found under {repo_root}.")
    return child


def supported_families() -> list[dict]:
    """Name + base repo for each known family (for status / UI listing)."""
    return [{"name": fam.name, "base_repo": fam.base_repo} for fam in _FAMILIES]
