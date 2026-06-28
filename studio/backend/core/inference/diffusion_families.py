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
    # Pipeline kwarg carrying the guidance value. Most use "guidance_scale";
    # Qwen-Image's distilled guidance is off, so its real CFG is "true_cfg_scale".
    cfg_kwarg: str = "guidance_scale"
    # Extra lowercased substrings (besides ``name``) that map a repo id here.
    aliases: tuple[str, ...] = field(default_factory = tuple)
    # True for families whose activations overflow float16's finite range
    # (~6.5e4) and produce inf -> NaN latents -> a black image. The backend
    # promotes a resolved float16 to float32 for these at load time.
    fp16_incompatible: bool = False
    # Set False only for a family whose denoiser block does not compile cleanly with
    # regional torch.compile. Now consulted on the GGUF path too (compile runs on the
    # GGUF transformer); all current families compile, so this stays True.
    supports_torch_compile: bool = True
    # Optional pre-quantized transformer checkpoints, as (scheme, repo_id) pairs (a
    # hashable mapping). When the fast transformer_quant path resolves a scheme with a
    # hosted checkpoint, the loader fetches the already-quantized weights instead of
    # materialising the dense bf16 transformer on the GPU (much lower load VRAM + a
    # smaller download). Empty until checkpoints are hosted -> behaviour is unchanged.
    prequant_repos: tuple[tuple[str, str], ...] = field(default_factory = tuple)
    # Native (stable-diffusion.cpp) single-file assets, used only when the no-GPU
    # sd.cpp engine is selected (CPU / Apple). The transformer GGUF is shared with
    # the diffusers path; sd-cli additionally needs a single-file VAE and text
    # encoder(s), because the diffusers base repo ships those sharded and sd-cli
    # cannot read that layout. Each asset is a hashable (repo_id, filename) the
    # backend fetches with hf_hub_download. ``sd_cpp_text_encoders`` carries a
    # trailing SdCppModelFiles field name (clip_l / t5xxl / llm / qwen2vl / clip_g)
    # so the backend maps each file onto the right sd-cli flag. Empty -> the family
    # has no native mapping and the sd.cpp route falls back to diffusers.
    sd_cpp_vae: Optional[tuple[str, str]] = None
    # VAE latent-format override for sd-cli (--vae-format): "flux2" for the FLUX.2
    # autoencoder, None (auto) otherwise.
    sd_cpp_vae_format: Optional[str] = None
    sd_cpp_text_encoders: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)


# Keyed by architecture, not per model variant: a checkpoint's specific base repo
# is read from its HF base_model tag at load time, so one entry covers Turbo/full,
# schnell/dev, etc. base_repo here is only a fallback. Only archs whose diffusers
# transformer supports from_single_file load here (ERNIE-Image does not, yet).
# FLUX.2-dev and FLUX.2-klein-9B are left out only because their base diffusers
# repos are gated; the open klein-4B base stands in for the klein family below.
_FAMILIES: tuple[DiffusionFamily, ...] = (
    DiffusionFamily(
        name = "flux.1",
        pipeline_class = "FluxPipeline",
        transformer_class = "FluxTransformer2DModel",
        base_repo = "black-forest-labs/FLUX.1-schnell",
        aliases = ("flux1", "flux-1"),
        sd_cpp_vae = ("black-forest-labs/FLUX.1-schnell", "ae.safetensors"),
        sd_cpp_text_encoders = (
            ("comfyanonymous/flux_text_encoders", "clip_l.safetensors", "clip_l"),
            ("comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors", "t5xxl"),
        ),
    ),
    # FLUX.2-klein is a distinct pipeline (Flux2KleinPipeline) with a Qwen3 text
    # encoder, not the Mistral-based Flux2Pipeline; it must precede a generic
    # flux match. The base Flux2Pipeline (FLUX.2-dev) is gated, so it's omitted.
    DiffusionFamily(
        name = "flux.2-klein",
        pipeline_class = "Flux2KleinPipeline",
        transformer_class = "Flux2Transformer2DModel",
        base_repo = "black-forest-labs/FLUX.2-klein-4B",
        aliases = ("flux2-klein",),
        # FLUX.2 uses a distinct 32-channel autoencoder; sd-cli needs the latent
        # format override. The single-file VAE ships in Comfy-Org/flux2-dev (the
        # klein-4B repo only has a sharded diffusers VAE). Shares Qwen3-4B with z-image.
        sd_cpp_vae = ("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"),
        sd_cpp_vae_format = "flux2",
        sd_cpp_text_encoders = (
            ("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors", "llm"),
        ),
    ),
    DiffusionFamily(
        name = "qwen-image",
        pipeline_class = "QwenImagePipeline",
        transformer_class = "QwenImageTransformer2DModel",
        base_repo = "Qwen/Qwen-Image",
        cfg_kwarg = "true_cfg_scale",
        aliases = ("qwen_image", "qwenimage"),
        sd_cpp_vae = ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors"),
        # The Qwen2.5-VL text encoder as a Q4_K_M GGUF keeps the CPU RAM win (the
        # bf16 safetensors encoder is ~15 GB). sd-cli's --qwen2vl is an alias of --llm.
        sd_cpp_text_encoders = (
            ("unsloth/Qwen2.5-VL-7B-Instruct-GGUF", "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", "qwen2vl"),
        ),
    ),
    DiffusionFamily(
        name = "z-image",
        pipeline_class = "ZImagePipeline",
        transformer_class = "ZImageTransformer2DModel",
        base_repo = "Tongyi-MAI/Z-Image-Turbo",
        aliases = ("zimage", "z_image"),
        # Z-Image's MLP down-projections peak near 9e5, which overflows float16.
        fp16_incompatible = True,
        sd_cpp_vae = ("Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors"),
        sd_cpp_text_encoders = (
            ("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors", "llm"),
        ),
    ),
)

# Editing / inpaint checkpoints share an arch keyword but need a different
# pipeline and an input image, which this text-to-image backend doesn't drive.
_EDIT_KEYWORDS = ("edit", "kontext", "inpaint")


def detect_family(repo_id: str, override: Optional[str] = None) -> Optional[DiffusionFamily]:
    """Resolve a ``DiffusionFamily`` from a repo id, or an explicit override.

    ``override`` matches a family ``name`` or alias exactly; otherwise the repo
    id is scanned for the first family whose name/alias appears in it. Image
    editing checkpoints are rejected (None) since this backend is text-to-image.
    """
    if override:
        key = override.strip().lower()
        for fam in _FAMILIES:
            if key == fam.name or key in fam.aliases:
                return fam
        return None
    needle = repo_id.lower()
    if any(kw in needle for kw in _EDIT_KEYWORDS):
        return None
    for fam in _FAMILIES:
        if fam.name in needle or any(alias in needle for alias in fam.aliases):
            return fam
    return None


def resolve_base_repo(fam: DiffusionFamily, base_repo: Optional[str]) -> str:
    """The companion diffusers repo: caller-supplied if given, else the family fallback."""
    base = (base_repo or "").strip()
    return base or fam.base_repo


def family_prequant_repo(fam: DiffusionFamily, scheme: str) -> Optional[str]:
    """The hosted pre-quantized transformer repo for ``scheme`` in this family, or None."""
    for entry_scheme, repo_id in fam.prequant_repos:
        if entry_scheme == scheme:
            return repo_id
    return None


def family_sd_cpp_supported(fam: DiffusionFamily) -> bool:
    """True when the family has the single-file VAE + text-encoder mapping the
    native sd.cpp engine needs. A family without it can only run on diffusers, so
    the no-GPU route falls back rather than routing to sd-cli."""
    return bool(fam.sd_cpp_vae and fam.sd_cpp_text_encoders)


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
