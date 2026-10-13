# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-family policy data carried on ``DiffusionFamily.profile`` / ``VideoFamily.profile``.

Stdlib only (the transformer-quant smoke-probe child imports its consumers). Everything a new family needs beyond its
registry row lives here as data; the consumer modules (transformer quant, text-encoder quant, untiled VAE decode,
preview, step skip, trusted repos) derive their lookup tables from the registries through ``profile_table`` and keep
their historical names and lookup semantics.

The quant fields are baked into PUBLISHED prequant artifacts and validated at load (``diffusion_prequant.
_validate_checkpoint`` requires an artifact's exclude tokens to be a superset of ``exclude_tokens_for_scheme``), so a
change to ``int8_exclude_tokens`` / ``int8_convrot`` / ``nvfp4_zero_row_tokens`` on a family with a hosted checkpoint
needs a rebuild and republish; tests/test_diffusion_family_data_freeze.py pins every resolved value.

Scheme names are spelled as plain strings ("int8", "fp8", "mxfp8", "nvfp4") to keep this module import-free; they
equal ``diffusion_transformer_quant.TQ_*``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen = True)
class AutoPrefer:
    """A family's head of the transformer-quant ``auto`` order, ahead of the arch ladder; off below ``floor`` or on
    consumer GPUs (unless ``consumer_ok``), and ``gated`` / ``backend`` need a passing NVFP4 gate record / backend for
    THIS device."""

    floor: tuple[int, int]
    schemes: tuple[str, ...]
    consumer_ok: bool = False
    gated: bool = False
    backend: Optional[str] = None


@dataclass(frozen = True)
class ConvRotSpec:
    """int8 ConvRot: ``group`` Hadamard size and the rotated Linear fqn ``suffixes``. Not an exclusion, so plain int8
    artifacts still validate. ``filename`` / ``repo`` name the hosted rotated build (only ``repo`` gets the rotated
    name prepended), ``variant_repos`` the variant repos that host one under their own name, ``default_on`` whether
    int8 rotates unless ``UNSLOTH_DIFFUSION_INT8_CONVROT`` turns it off (the rest are opt-in with ``=1``). ``group`` 0
    means no runtime rotation: the entry only names a hosted build whose rotation is baked into the file."""

    group: int
    suffixes: tuple[str, ...]
    filename: Optional[str] = None
    repo: Optional[str] = None
    variant_repos: tuple[str, ...] = ()
    default_on: bool = False


@dataclass(frozen = True)
class UntiledVae:
    """Untiled video VAE decode peak model (video_vae_untiled). Either per latent pixel with an RGB output term
    (``bytes_per_latent_pixel`` + ``output_scale`` = (spatial, temporal)), for a causal decoder that decodes one latent
    frame at a time, or per output frame x latent pixel (``bytes_per_frame_latent_pixel`` + ``temporal_scale``) for a
    decoder holding every frame. Bytes are fp16-decoder bytes; the caller scales by element size."""

    bytes_per_latent_pixel: Optional[float] = None
    output_scale: Optional[tuple[int, int]] = None
    bytes_per_frame_latent_pixel: Optional[float] = None
    temporal_scale: Optional[int] = None


@dataclass(frozen = True)
class FamilyProfile:
    # int8 name tokens left dense on top of the generic M = 1 tokens (``_int_mm`` needs M > 16, or a measured quality
    # lever). Baked into published int8 artifacts.
    int8_exclude_tokens: tuple[str, ...] = ()
    # int8 name tokens whose small-M Linears get activation rows PADDED to the ``_int_mm`` floor instead of excluded.
    int8_pad_tokens: tuple[str, ...] = ()
    int8_convrot: Optional[ConvRotSpec] = None
    # nvfp4 Linears reachable with an EMPTY activation (zero-row guard after quantization).
    nvfp4_zero_row_tokens: tuple[str, ...] = ()
    # Transformer-quant schemes ruled out at the model level (explicit requests too); a passing NVFP4 gate record
    # lifts an nvfp4 deny per base.
    scheme_deny: frozenset[str] = frozenset()
    # Extra schemes denied for TRAINING on top of ``scheme_deny`` (nvfp4 is denied for training everywhere).
    train_scheme_deny: frozenset[str] = frozenset()
    auto_prefer: Optional[AutoPrefer] = None
    # Reason auto keeps bf16 while bf16 fits (INT8 still picked once bf16 would offload); None = not this family.
    auto_bf16_when_resident: Optional[str] = None
    # Video: prequant DiT schemes AUTO seeds even where bf16 fits resident (else auto keeps resident bf16).
    auto_prequant_when_resident: frozenset[str] = frozenset()
    # Text-encoder int8 keep-bf16 schedule (skip_first, skip_last) decoder blocks; None = int8 falls back to fp8.
    te_int8_skip: Optional[tuple[int, int]] = None
    untiled_vae: Optional[UntiledVae] = None
    # diffusion_preview_factors.FACTORS key for the live latent preview; None = no preview.
    preview_factors: Optional[str] = None
    # Measured auto static step skip, keyed by the lowercased UPSTREAM repo id so unmeasured siblings never match:
    # (repo, ((tier or "steps", value), ...)). See diffusion_cache.AUTO_STATIC_SKIP.
    auto_static_skip: tuple[tuple[str, tuple[tuple[str, int], ...]], ...] = ()
    # Lowercased vendor repos trusted for a full (non-GGUF) load on top of the ones the registry row already names
    # (base_repo, named_variant_bases, train_base_repos, prequant_variant_repos bases). Exact match, safetensors-only,
    # no remote code.
    trusted_repos: tuple[str, ...] = ()
    # Extra exact family strings the transformer-quant tables (exclude / pad / ConvRot / zero-row / denies / auto
    # preferences) answer to, besides ``name``: a builder spelling a family by a version name ("ltx-2.3"). Not general
    # alias resolution, which would hand a family's artifact policy to every alias substring.
    quant_names: tuple[str, ...] = ()
    # Compile the denoiser with static shapes (token counts depend only on (w, h, frames): one recompile per shape,
    # ~1.3x faster); read by the diffusion_autoopt ``ltx_static_compile`` family hook.
    compile_static: bool = False
    # Install the LTX-2 STG compile adapter after a successful compile (``ltx_stg_compile_adapter`` family hook).
    stg_compile_adapter: bool = False


DEFAULT_PROFILE = FamilyProfile()


def _families() -> tuple[Any, ...]:
    from .diffusion_families import _FAMILIES as image
    from .video_families import _FAMILIES as video
    return (*image, *video)


def profile_table(attr: str, *, quant: bool = False) -> dict[str, Any]:
    """``{family name: profile.<attr>}`` over both registries, for families that set it (not None / empty).
    ``quant`` also keys each value under the profile's ``quant_names``."""
    table: dict[str, Any] = {}
    for fam in _families():
        profile = getattr(fam, "profile", DEFAULT_PROFILE)
        value = getattr(profile, attr)
        if value is None or value == () or value == frozenset():
            continue
        table[fam.name] = value
        if quant:
            for name in profile.quant_names:
                table.setdefault(name, value)
    return table


def profile_for(family: Any) -> FamilyProfile:
    """``family.profile``, else the registry row's profile for its exact ``name`` (a stand-in family object), else the
    empty profile."""
    profile = getattr(family, "profile", None)
    if isinstance(profile, FamilyProfile):
        return profile
    name = getattr(family, "name", None)
    if name:
        for fam in _families():
            if fam.name == name:
                return fam.profile
    return DEFAULT_PROFILE


def trusted_repo_ids(families: Any) -> frozenset[str]:
    """Lowercased non-``unsloth/`` repo ids ``families`` name as their own bases, plus each profile's
    ``trusted_repos``. ``unsloth/`` ids are trusted by prefix elsewhere, so they are left out here."""
    ids: set[str] = set()
    for fam in families:
        named = [fam.base_repo]
        named += list(getattr(fam, "named_variant_bases", ()) or ())
        named += list(getattr(fam, "train_base_repos", ()) or ())
        named += [row[0] for row in getattr(fam, "prequant_variant_repos", ()) or ()]
        named += list(getattr(fam, "profile", DEFAULT_PROFILE).trusted_repos)
        ids.update(r.strip().lower() for r in named if r)
    return frozenset(r for r in ids if not r.startswith("unsloth/"))
