# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load a *pre-cast* text encoder instead of downloading the dense one and casting it.

The runtime ``text_encoder_quant=fp8`` path (``diffusion_precision._cast_fp8``) downloads
the full bf16 text encoder and layerwise-casts it in place on every load. For the
heavyweight encoders (LTX's Gemma3-12B ~49 GB fp32, FLUX.2-dev's Mistral-24B ~48 GB,
Qwen-Image's Qwen2.5-VL ~16.6 GB) that download dominates a fresh machine's load. When the
encoder was already cast and saved (``scripts/build_te_prequant_checkpoint.py``), this loads
the ~half-size fp8-storage state dict directly: meta-init the encoder skeleton,
``load_state_dict(assign=True)``, then install the SAME layerwise upcast hooks the runtime
cast uses. The layerwise cast is a deterministic storage transform, so the loaded encoder is
bit-identical to dense-load-then-cast by construction.

v1 covers the layerwise ``fp8`` STORAGE scheme only: its state dict is plain tensors
(``torch.load(weights_only = True)``, no pickle execution), and cast-on-load equals
load-of-cast exactly. The dynamic-compute schemes (fp8_dynamic / int8 / nvfp4) build
torchao subclass wrappers at runtime and int8 keys off per-family keep-bf16 schedules, so
their artifacts are deliberately NOT hosted; the metadata layout leaves room to add them.

Best-effort and lazily imported: a missing / mismatched / unreadable checkpoint returns
None and the caller falls back to the dense download + cast. Inert with nothing configured.
"""

from __future__ import annotations

import re
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Iterable, Optional

# Reuse the DiT module's operator allowlist for local paths: one env var, one policy.
from .diffusion_prequant import (
    ALLOW_LOCAL_PREQUANT_PATH_ENV,
    _local_prequant_path_allowed,
    _same_base_model,
)

# torch.save dict layout tag; bump on an on-disk change so old/foreign artifacts are rejected.
TE_PREQUANT_FORMAT = "unsloth_prequant_text_encoder_state_dict_v1"

# fp8: layerwise storage cast (families with te_prequant_repos). int8: ConvRot weight-only, TE_INT8_CONVROT_FILES only.
TE_PREQUANT_SCHEMES = ("fp8", "int8")

# Distinct from TE_PREQUANT_FORMAT so an fp8-only build refuses the file instead of loading int8 into bf16 Linears.
TE_PREQUANT_FORMAT_INT8_CONVROT = "unsloth_prequant_text_encoder_int8_convrot_v1"

# family -> {component: (repo_id, filename)}. ComfyUI's qwen3vl_8b_int8_convrot scheme: decoder projections int8 in the
# group-256 Hadamard basis + fp32 per-row scale; vision tower, embedding, norms bf16; no lm_head. Built by
# scripts/build_te_int8_convrot_checkpoint.py.
_T5_INT8 = ("unsloth/FLUX.1-schnell-FP8", "FLUX.1-schnell-text_encoder_2-INT8-ConvRot.safetensors")
_QWEN25VL_INT8 = ("unsloth/Qwen-Image-FP8", "Qwen-Image-text_encoder-INT8-ConvRot.safetensors")

TE_INT8_CONVROT_FILES: dict[str, dict[str, tuple[str, ...]]] = {
    "qwen-image-2.1": {
        "text_encoder": (
            "unsloth/Qwen-Image-2.1-FP8",
            "Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors",
        ),
    },
    # Qwen3-4B (Qwen3ForCausalLM, ``model.layers``), stock weights: the same build as above on a text-only Qwen3. The
    # distilled and base 4B repos ship identical bytes, so one file serves both.
    "flux.2-klein": {
        "text_encoder": (
            "unsloth/FLUX.2-klein-4B-FP8",
            "FLUX.2-klein-4B-text_encoder-INT8-ConvRot.safetensors",
        ),
    },
    # The rest are built with ``--load transformers`` from the class the pipeline instantiates, so their keys are that
    # class's runtime state dict. One file per distinct encoder; byte-identical encoders share it through
    # ``_TE_EQUIVALENT_BASES``. T5-XXL (FLUX.1 schnell / dev / Krea / Kontext): all seven projections per block int8.
    "flux.1": {"text_encoder_2": _T5_INT8},
    "flux.1-kontext": {"text_encoder_2": _T5_INT8},
    # Mistral-Small-3.2-24B (the Mistral3 conditional-generation class): the Pixtral vision tower and projector stay bf16.
    "flux.2-dev": {
        "text_encoder": (
            "unsloth/FLUX.2-dev-FP8",
            "FLUX.2-dev-text_encoder-INT8-ConvRot.safetensors",
        ),
    },
    # Qwen2.5-VL-7B, shared by every Qwen-Image release and HunyuanImage-2.1; the vision tower stays bf16 (edit runs
    # it).
    "qwen-image": {"text_encoder": _QWEN25VL_INT8},
    "qwen-image-edit": {"text_encoder": _QWEN25VL_INT8},
    "qwen-image-layered": {"text_encoder": _QWEN25VL_INT8},
    "hunyuanimage-2.1": {"text_encoder": _QWEN25VL_INT8},
    # Z-Image's own Qwen3-4B, as the bare Qwen3Model the pipeline builds
    "z-image": {
        "text_encoder": (
            "unsloth/Z-Image-Turbo-FP8",
            "Z-Image-Turbo-text_encoder-INT8-ConvRot.safetensors",
        ),
    },
    # Krea's Qwen3-VL-4B, as Qwen3VLModel
    "krea-2": {
        "text_encoder": (
            "unsloth/Krea-2-Turbo-FP8",
            "Krea-2-Turbo-text_encoder-INT8-ConvRot.safetensors",
        ),
    },
    # HiDream-I1: text_encoder_3 is FLUX.1's T5-XXL (the third element names the component the file was built as), and
    # text_encoder_4 the Llama-3.1-8B-Instruct mirror, assembled by diffusion_hidream.hidream_te4_kwargs.
    "hidream-i1": {
        "text_encoder_3": (*_T5_INT8, "text_encoder_2"),
        "text_encoder_4": (
            "unsloth/HiDream-I1-Full-FP8",
            "HiDream-I1-Full-text_encoder_4-INT8-ConvRot.safetensors",
        ),
    },
    # Ideogram 4's Qwen3-VL-8B (Qwen3VLModel), from the vendor fp8 shards dequantized the way the dense load does
    "ideogram-4": {
        "text_encoder": (
            "unsloth/ideogram-4-FP8",
            "ideogram-4-text_encoder-INT8-ConvRot.safetensors",
        ),
    },
    # Gemma-2-2B (Gemma2Model), stored fp32 on the Hub
    "lumina-2": {
        "text_encoder": (
            "unsloth/Lumina-Image-2.0-FP8",
            "Lumina-Image-2.0-text_encoder-INT8-ConvRot.safetensors",
        ),
    },
    # UMT5-XXL, one file for all four Wan bases (an equivalence group below): every self-attention and gated-FFN
    # projection of encoder.block int8 in the group-256 Hadamard basis, the shared embedding and norms dense. Opt-in
    # (text_encoder_quant=int8); the family default is the fp8 file beside it, as ComfyUI's Wan templates.
    **{
        name: {
            "text_encoder": (
                "unsloth/Wan2.2-TI2V-5B-FP8",
                "Wan2.2-TI2V-5B-text_encoder-INT8-ConvRot.safetensors",
            ),
        }
        for name in ("wan2.2-ti2v-5b", "wan2.2-t2v-a14b", "wan2.1-t2v-1.3b", "wan2.1-t2v-14b")
    },
    # Qwen3-VL-4B, the same build as BFL's qwen3vl_4b_int8_convrot except that embed_tokens stays bf16: every int8
    # projection matches BFL's byte for byte. The base and distilled repos carry the same bytes (one encoder).
    "flux.3-klein": {
        "text_encoder": (
            "unsloth/flux-3-image-klein-FP8",
            "flux-3-image-klein-text_encoder-INT8-ConvRot.safetensors",
        ),
    },
    "flux.3-klein-base": {
        "text_encoder": (
            "unsloth/flux-3-image-klein-base-FP8",
            "flux-3-image-klein-base-text_encoder-INT8-ConvRot.safetensors",
        ),
    },
    # MOSS-Image Qwen3.5-9B: MLP + full-attention Linears int8; gated-delta, embeddings, vision stay bf16.
    "moss-image": {
        "text_encoder": (
            "unsloth/MOSS-Image-SFT-FP8",
            "MOSS-Image-SFT-text_encoder-INT8-ConvRot.safetensors",
        ),
    },
}

# family -> lowercased variant base -> {component: (repo_id, filename)}, for a variant whose encoder is a different
# model from the family base's (FLUX.2 klein 9B: Qwen3-8B, not Qwen3-4B). Looked up by the base the family view
# carries (``te_prequant_sources_for_base`` swaps it in), so the family base keeps the table above.
TE_INT8_CONVROT_VARIANT_FILES: dict[str, dict[str, dict[str, tuple[str, str]]]] = {
    "flux.2-klein": {
        "black-forest-labs/flux.2-klein-9b": {
            "text_encoder": (
                "unsloth/FLUX.2-klein-9B-FP8",
                "FLUX.2-klein-9B-text_encoder-INT8-ConvRot.safetensors",
            ),
        },
    },
}

# Module names of the Linears an int8 ConvRot file stores int8: decoder q/k/v/o + gate/up/down of a Qwen / Llama /
# Mistral / Gemma stack (``...layers.N.``; vision towers name their blocks ``blocks`` / ``attention`` and stay bf16)
# and T5's SelfAttention q/k/v/o + DenseReluDense wi_0/wi_1/wo. The builder quantizes exactly these and the loader
# refuses a file that leaves one of them dense.
TE_INT8_CONVROT_LINEAR = re.compile(
    r"(?:^|\.)layers\.\d+\.(?:self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)$"
    r"|(?:^|\.)block\.\d+\.layer\.\d+\.(?:SelfAttention\.[qkvo]|DenseReluDense\.(?:wi_0|wi_1|wo))$"
)

# True while a load runs whose text-encoder scheme was picked automatically (unset / "auto"). Nobody asked for that
# scheme, so a hosted file that cannot be loaded never fails the load: the dense encoder is loaded instead (logged). An
# explicit request keeps the strict refusal. Set by the image loader around each load (``auto_te_request``).
_AUTO_TE_REQUEST: ContextVar[bool] = ContextVar("unsloth_auto_te_request", default = False)


@contextmanager
def auto_te_request(auto: bool):
    token = _AUTO_TE_REQUEST.set(bool(auto))
    try:
        yield
    finally:
        _AUTO_TE_REQUEST.reset(token)


def te_request_is_auto() -> bool:
    return _AUTO_TE_REQUEST.get()


def _picked_automatically(mode: Any) -> bool:
    from .diffusion_precision import te_quant_picked_automatically
    return te_quant_picked_automatically(mode)


# Families whose encoder class has no lm_head at all (UMT5EncoderModel), so the int8 file needs no lm_head trim and the
# UNSLOTH_TE_KEEP_LM_HEAD kill switch has nothing to keep.
_TE_INT8_HEADLESS_FAMILIES = frozenset(
    {"wan2.2-ti2v-5b", "wan2.2-t2v-a14b", "wan2.1-t2v-1.3b", "wan2.1-t2v-14b"}
)

# The scheme the loaded FILE carried ("fp8" / "int8"), so quantize_text_encoders never re-casts an int8 encoder.
TE_PREQUANT_SCHEME_ATTR = "_unsloth_te_prequant_scheme"

# ``<root>/<owner>/<repo>/<filename>`` used before the Hub (os.pathsep-separated roots), as for the DiT pre-quants.
TE_PREQUANT_MIRROR_ENV = "UNSLOTH_DIFFUSION_PREQUANT_MIRROR"


def quantize_int8_convrot_weight(weight: Any, group_size: int = 256) -> tuple[Any, Any]:
    """``(int8 weight, float32 [out, 1] scale)`` of a Linear weight in the ConvRot basis: ``W_rot = W @ blockdiag(H).T``
    in float32, then symmetric per-output-channel absmax / 127 with round-half-to-even. The builder's quantizer, here so
    the artifact and its tests share one definition; ``Int8ConvRotLinear`` is its inverse."""
    import torch

    from .diffusion_convrot import build_convrot_hadamard

    out_features, in_features = weight.shape
    if in_features % group_size:
        raise ValueError(
            f"in_features {in_features} is not divisible by the ConvRot group {group_size}"
        )
    h = build_convrot_hadamard(group_size, device = weight.device, dtype = torch.float32)
    rotated = (
        weight.float().reshape(out_features, in_features // group_size, group_size) @ h.T
    ).reshape(out_features, in_features)
    scale = (rotated.abs().amax(dim = 1, keepdim = True) / 127.0).clamp_min(1e-12)
    return torch.round(rotated / scale).clamp(-127, 127).to(torch.int8), scale


def family_te_int8_convrot(fam: Any, component: str) -> Optional[tuple[str, ...]]:
    """``(repo_id, filename)`` of the family's hosted int8 ConvRot encoder for ``component``, or None. A family view
    carrying a variant base with its own encoder (``TE_INT8_CONVROT_VARIANT_FILES``) gets that variant's file only."""
    name = str(getattr(fam, "name", fam) or "").strip().lower()
    base = str(getattr(fam, "base_repo", "") or "").strip().lower()
    variant = TE_INT8_CONVROT_VARIANT_FILES.get(name, {}).get(base)
    if variant is not None:
        return variant.get(component)
    return TE_INT8_CONVROT_FILES.get(name, {}).get(component)


def te_prequant_mirror_path(repo_id: Optional[str], name: Optional[str]) -> Optional[str]:
    """``<mirror>/<repo_id>/<name>`` when a configured mirror holds that file, else None. Never raises."""
    import os

    raw = (os.environ.get(TE_PREQUANT_MIRROR_ENV) or "").strip()
    if not raw or not repo_id or not name:
        return None
    parts = [*str(repo_id).split("/"), *str(name).split("/")]
    if any(p in ("", ".", "..") or "\\" in p for p in parts):
        return None
    for root in raw.split(os.pathsep):
        root = root.strip()
        if not root:
            continue
        try:
            base = os.path.realpath(os.path.expanduser(root))
            candidate = os.path.realpath(os.path.join(base, *parts))
        except Exception:  # noqa: BLE001 - a bad root is simply not a mirror
            continue
        if candidate.startswith(base.rstrip(os.sep) + os.sep) and os.path.isfile(candidate):
            return candidate
    return None


# Components the pipeline-assembly injection covers; text_encoder_4 lives in diffusion_hidream.py.
TE_PREQUANT_COMPONENTS = ("text_encoder", "text_encoder_2", "text_encoder_3")

# Pre-cast fp8 size vs bf16; dense embeddings/norms keep it above 0.5.
# 0.65 is the measured max rounded up: overestimating is the safe direction.
TE_PREQUANT_BUDGET_SCALE = 0.65
# The int8 ConvRot encoder alone (no fp8 fallback to cover): Qwen-Image-2.1's is 9,349,769,248 bytes against
# 17,534,339,488 for the dense encoder, 0.533, rounded up.
TE_INT8_CONVROT_BUDGET_SCALE = 0.56
# Per family where the default under-states: an encoder whose bf16 embedding table is a large share of it keeps more
# bytes dense. Lumina-2's Gemma-2-2B file is 3,207,382,688 bytes against the 5.23 GB bf16 equivalent of its fp32 Hub
# copy (0.613), Krea-2's Qwen3-VL-4B 5,247,054,376 against 8.88 GB dense (0.591); both rounded up.
# FLUX.3's Qwen3-VL-4B file is 5,247,076,320 bytes against 8,875,719,344 dense (0.591), rounded up the same way.
# Wan's UMT5 keeps its 256k-row embedding dense: its int8 file is 6,735,404,944 bytes against 11,361,851,208 dense,
# 0.593, rounded up the same way.
# HunyuanImage-2.1 loads Qwen-Image's 8,974,915,168-byte file against a 15.5 GB table price (0.579), rounded up.
_TE_INT8_CONVROT_BUDGET_SCALES = {
    "hunyuanimage-2.1": 0.60,
    "lumina-2": 0.62,
    "krea-2": 0.60,
    "flux.3-klein": 0.60,
    "flux.3-klein-base": 0.60,
    # MOSS-Image's 11,488,171,720-byte file against its 15.9 GB table price (0.723), rounded up.
    "moss-image": 0.73,
    **{
        name: 0.62
        for name in ("wan2.2-ti2v-5b", "wan2.2-t2v-a14b", "wan2.1-t2v-1.3b", "wan2.1-t2v-14b")
    },
}


def te_int8_convrot_budget_scale(fam: Any) -> float:
    """Fraction of the bf16 encoder the family's int8 ConvRot file occupies, for memory budgeting."""
    name = str(getattr(fam, "name", fam) or "").strip().lower()
    return _TE_INT8_CONVROT_BUDGET_SCALES.get(name, TE_INT8_CONVROT_BUDGET_SCALE)


def te_prequant_budget_scale(
    fam: Any, *, te_quant_mode: Optional[str], target: Any, base: str
) -> float:
    """Scale to apply to a family's bf16 text-encoder size when budgeting memory for this pick:
    ``TE_PREQUANT_BUDGET_SCALE`` when the load takes its encoder PRE-CAST from a hosted fp8
    checkpoint, else 1.0.

    Keyed on ``te_prequant_sources`` -- the same pure resolver the download plan and the load
    itself use, so a budget can never disagree with them about what gets loaded -- and NOT on
    ``text_encoder_quant`` alone. That distinction is the conservative one: the runtime cast
    (``quantize_text_encoders``) runs *after* pipeline assembly has already materialised the
    dense encoder, so its steady state is fp8 but its peak is bf16, and the peak is what a
    budget has to cover. Only the pre-cast path is fp8-sized end to end.

    Best-effort like the rest of this module: anything unresolvable returns 1.0, i.e. today's
    bf16 budget."""
    if getattr(fam, "name", None) == "ltx-2.5":
        from .video_ltx25 import ltx25_te_budget_scale, ltx25_te_prequant_sources

        # Only the hosted file is int8-sized end to end: the at-load fallback materializes the dense encoder first.
        if not ltx25_te_prequant_sources(te_quant_mode, base):
            return 1.0
        components = getattr(fam, "bf16_components_gb", None) or (0.0, 0.0)
        return ltx25_te_budget_scale(te_quant_mode, components[1])
    try:
        sources = te_prequant_sources_for_base(
            fam,
            base,
            te_quant_mode = te_quant_mode,
            target = target,
        )
    except Exception:  # noqa: BLE001 -- an unresolvable pre-cast just means the dense encoder
        return 1.0
    if not sources:
        return 1.0
    if _auto_pick(te_quant_mode) and not te_prequant_sources_confirmed(sources):
        # An automatic default whose hosted file is not confirmed loads the DENSE encoder, so budget that.
        return 1.0
    scale = (
        te_int8_convrot_budget_scale(fam)
        if _int8_convrot_only(te_quant_mode, target) or auto_int8_takes_file_only(te_quant_mode)
        else TE_PREQUANT_BUDGET_SCALE
    )
    return _with_dense_siblings(fam, base, te_quant_mode, scale)


# GB of each family's bf16 text-encoder term (diffusion_auto_policy._FAMILY_BF16_GB) held by companions no hosted file
# replaces: FLUX.1's CLIP-L, HunyuanImage-2.1's ByT5, HiDream's CLIP-L + CLIP-G. They stay bf16 under a hosted int8
# file and under an automatic pick (an explicit fp8 still casts them, as before), so the scale must not shrink them.
_TE_DENSE_SIBLING_GB = {
    "flux.1": 0.25,
    "flux.1-kontext": 0.25,
    "hunyuanimage-2.1": 0.8,
    "hidream-i1": 3.3,
}


def te_dense_sibling_gb(fam: Any, te_quant_mode: Any) -> float:
    """GB of the family's text-encoder term that stays dense bf16 beside a hosted pre-cast file for this pick."""
    try:
        from .diffusion_precision import TE_QUANT_INT8, normalize_te_quant
        if normalize_te_quant(te_quant_mode) != TE_QUANT_INT8 and not _auto_pick(te_quant_mode):
            return 0.0
    except Exception:  # noqa: BLE001
        return 0.0
    return _TE_DENSE_SIBLING_GB.get(str(getattr(fam, "name", "") or "").strip().lower(), 0.0)


def _with_dense_siblings(fam: Any, base: Optional[str], te_quant_mode: Any, scale: float) -> float:
    """``scale`` applied to the hosted share of the encoder term only; the dense siblings keep their full size."""
    sibling = te_dense_sibling_gb(fam, te_quant_mode)
    if sibling <= 0.0:
        return scale
    try:
        from .diffusion_auto_policy import family_bf16_components_gb
        table = family_bf16_components_gb(fam, base)
        total = float(table[1]) if table else 0.0
    except Exception:  # noqa: BLE001
        total = 0.0
    if total <= sibling:
        return scale
    return (sibling + (total - sibling) * scale) / total


def explicit_int8_hosted_components(
    fam: Any,
    base: Optional[str],
    target: Any,
    *,
    supplied: Iterable[str] = (),
    local_files_only: Optional[bool] = None,
) -> dict[str, bool]:
    """``{component: takes its hosted int8 ConvRot file}`` for every encoder ``fam`` hosts one for that no supplied
    encoder replaces (HiDream's text_encoder_4 included): the device runs the file, the lm_head trim is on (else the
    resolver puts fp8 first), the base matches the file's, and the file is confirmed (on disk, or listed by the
    Hub)."""
    name = str(getattr(fam, "name", "") or "").strip().lower()
    out: dict[str, bool] = {}
    for component in TE_INT8_CONVROT_FILES.get(name, {}):
        if component in set(supplied or ()):
            # A supplied encoder replaces the hosted file in this slot, so the slot does not take int8.
            out[component] = False
            continue
        try:
            standalone: dict[str, str] = {}
            if component == "text_encoder_4":
                from .diffusion_hidream import HIDREAM_LLAMA_REPO
                standalone = {"text_encoder_4": HIDREAM_LLAMA_REPO}
            # TE4 is matched against its own standalone repo, the rest against the selected base.
            component_base = standalone.get(component) or base or getattr(fam, "base_repo", "")
            got = te_prequant_sources_for_base(
                fam,
                str(component_base or ""),
                te_quant_mode = "int8",
                target = target,
                components = (component,),
                **({"standalone_component_bases": standalone} if standalone else {}),
            ).get(component)
            out[component] = bool(
                got is not None
                and not (got.kind == "repo" and _is_fp8_te_name(got.filename))
                and te_prequant_sources_confirmed(
                    {component: replace(got, fallback_filenames = ())},
                    local_files_only = local_files_only,
                )
            )
        except Exception:  # noqa: BLE001 - unanswerable: not granted
            out[component] = False
    return out


def explicit_int8_takes_hosted_file(
    fam: Any,
    base: Optional[str],
    target: Any,
    *,
    supplied: Iterable[str] = (),
    local_files_only: Optional[bool] = None,
) -> bool:
    """Whether an explicit int8 request on ``fam`` / ``base`` will load the hosted int8 ConvRot file for EVERY
    encoder the family hosts one for (see ``explicit_int8_hosted_components``), with none replaced by a supplied
    encoder. Preflight, precision and the HiDream TE4 gate all ask this, so none of them grants the plain-tensor int8
    path to a load that will not take it."""
    name = str(getattr(fam, "name", "") or "").strip().lower()
    if not TE_INT8_CONVROT_FILES.get(name):
        return False
    takes = explicit_int8_hosted_components(
        fam, base, target, supplied = supplied, local_files_only = local_files_only
    )
    return len(takes) == len(TE_INT8_CONVROT_FILES[name]) and all(takes.values())


def explicit_int8_unavailable_reason(
    fam: Any,
    base: Optional[str],
    target: Any,
    *,
    supplied: Iterable[str] = (),
    local_files_only: Optional[bool] = None,
) -> Optional[str]:
    """Why an explicit int8 request cannot take the int8 ConvRot file(s) ``fam`` names, or None when it can (or the
    family names none). Names each unconfirmed file so an int8-only host (Apple Silicon) reports the missing file.
    Never raises."""
    try:
        name = str(getattr(fam, "name", "") or "").strip().lower()
        named = []
        for component in TE_INT8_CONVROT_FILES.get(name, {}):
            if component in set(supplied or ()):
                continue
            standalone: dict[str, str] = {}
            if component == "text_encoder_4":
                from .diffusion_hidream import HIDREAM_LLAMA_REPO
                standalone = {"text_encoder_4": HIDREAM_LLAMA_REPO}
            component_base = standalone.get(component) or base or getattr(fam, "base_repo", "")
            got = te_prequant_sources_for_base(
                fam,
                str(component_base or ""),
                te_quant_mode = "int8",
                target = target,
                components = (component,),
                **({"standalone_component_bases": standalone} if standalone else {}),
            ).get(component)
            # Only a missing int8 FILE; other refusals carry their own reasons.
            if got is None or got.kind != "repo" or _is_fp8_te_name(got.filename):
                continue
            if not te_prequant_sources_confirmed(
                {component: replace(got, fallback_filenames = ())},
                local_files_only = local_files_only,
            ):
                named.append(f"{got.location}/{got.filename}")
        if not named:
            return None
        offline = bool(local_files_only) or _hub_offline()
        where = (
            "is not in the local cache (offline load)" if offline else "is not published on the Hub"
        )
        return f"the hosted int8 ConvRot text encoder {', '.join(named)} for {getattr(fam, 'name', fam)} {where}"
    except Exception:  # noqa: BLE001 - unanswerable: no reason to add
        return None


def _auto_pick(te_quant_mode: Any) -> bool:
    try:
        from .diffusion_precision import te_quant_picked_automatically
        return te_quant_picked_automatically(te_quant_mode)
    except Exception:  # noqa: BLE001
        return False


# How long a Hub-listing verdict is reused, so the several budget calls of one load ask the Hub once.
_CONFIRMED_TTL_S = 300.0
_confirmed_cache: dict[tuple, tuple[float, bool]] = {}


def _hub_offline() -> bool:
    try:
        from huggingface_hub import constants
        return bool(constants.HF_HUB_OFFLINE)
    except Exception:  # noqa: BLE001
        return False


def te_prequant_sources_confirmed(
    sources: dict[str, "TePrequantSource"], *, local_files_only: Optional[bool] = None
) -> bool:
    """Whether every hosted pre-cast file in ``sources`` is CONFIRMED: on disk (path, mirror, Hub cache), or listed
    by the Hub. Offline, or when the Hub cannot answer, only what is on disk counts. An automatic pick budgets the
    dense encoder unless this holds, since that is what its load falls back to."""
    import os
    import time

    pending: dict[str, "TePrequantSource"] = {}
    for component, source in sources.items():
        kind = getattr(source, "kind", None)
        if kind == "path":
            if not os.path.isfile(os.path.expanduser(str(source.location))):
                return False
            continue
        if kind != "repo" or not getattr(source, "filename", None):
            return False
        names = [n for n in te_candidate_filenames(source) if te_candidate_is_readable(n)]
        if not any(_held_locally(source.location, n, None) for n in names):
            pending[component] = source
    if not pending:
        return True
    if local_files_only is None:
        # Inside a load the prequant layer records the load's own local_files_only.
        try:
            from .diffusion_prequant import _LOCAL_FILES_ONLY
            local_files_only = bool(_LOCAL_FILES_ONLY.get())
        except Exception:  # noqa: BLE001
            local_files_only = False
    if local_files_only or _hub_offline():
        # A load that may not fetch can only use what is on disk.
        return False
    key = tuple(
        sorted((c, s.location, tuple(te_candidate_filenames(s))) for c, s in pending.items())
    )
    now = time.monotonic()
    hit = _confirmed_cache.get(key)
    if hit is not None and now - hit[0] < _CONFIRMED_TTL_S:
        return hit[1]
    try:
        from huggingface_hub import HfApi
        found = te_prequant_hub_files(pending, HfApi())
        confirmed = all(c in found for c in pending)
    except Exception:  # noqa: BLE001 - an unanswerable Hub confirms nothing
        confirmed = False
    _confirmed_cache[key] = (now, confirmed)
    return confirmed


def auto_int8_takes_file_only(te_quant_mode: Any) -> bool:
    """Whether this pick is an AUTOMATIC int8 one, which loads the hosted int8 ConvRot file or the dense encoder and
    never the fp8 file behind it (``te_prequant_sources`` drops it). Its pre-cast budget is the int8 file's own
    fraction, not ``TE_PREQUANT_BUDGET_SCALE``: that ceiling covers an fp8 file this pick never opens, and pricing it
    oversized every auto load by up to 1.6 GB (Qwen-Image-2.1: 0.65 against the file's 0.533 of 17.5 GB)."""
    try:
        from .diffusion_precision import TE_QUANT_INT8, normalize_te_quant
        return normalize_te_quant(te_quant_mode) == TE_QUANT_INT8 and _auto_pick(te_quant_mode)
    except Exception:  # noqa: BLE001
        return False


def _int8_convrot_only(te_quant_mode: Optional[str], target: Any) -> bool:
    """Whether this pick can ONLY take the int8 ConvRot file (no fp8 fallback): Apple Silicon."""
    try:
        from .diffusion_precision import TE_QUANT_FP8, normalize_te_quant, te_quant_supported
        return (
            normalize_te_quant(te_quant_mode) == "int8"
            and int8_convrot_te_runs_on(target)
            and not te_quant_supported(target, TE_QUANT_FP8)
        )
    except Exception:  # noqa: BLE001
        return False


# Bases with VERIFIED byte-identical encoder weights share one artifact; any other base is refused.
_TE_EQUIVALENT_BASES: tuple[frozenset[str], ...] = (
    # Qwen-Image-2512 shares Qwen2.5-VL-7B's byte-identical shards; refusing it pulls 16.6 GB unused.
    frozenset(
        {
            "qwen/qwen-image",
            "qwen/qwen-image-2512",
            # The edit and layered releases republish the same four shards (sha256 compared 2026-10-16).
            "qwen/qwen-image-edit",
            "qwen/qwen-image-edit-2509",
            "qwen/qwen-image-edit-2511",
            "qwen/qwen-image-layered",
            "hunyuanvideo-community/hunyuanimage-2.1-diffusers",
        }
    ),
    # Qwen3-VL-8B: one file vs four shards, but all 750 tensors byte-identical (compared 2026-10-09).
    frozenset(
        {
            "qwen/qwen-image-2.1",
            "qwen/qwen-image-2.1-turbo",
        }
    ),
    # UMT5-XXL: Wan2.2 TI2V-5B and T2V-A14B ship byte-identical bf16 text_encoder/ shards, Wan2.1 T2V-1.3B and 14B
    # byte-identical fp32 ones, and every Wan2.1 fp32 tensor rounds to the Wan2.2 bf16 one bit for bit (all 242,
    # compared 2026-10-16), so a bf16 load of either gives bit-identical prompt embeddings.
    frozenset(
        {
            "wan-ai/wan2.2-ti2v-5b-diffusers",
            "wan-ai/wan2.2-t2v-a14b-diffusers",
            "wan-ai/wan2.1-t2v-1.3b-diffusers",
            "wan-ai/wan2.1-t2v-14b-diffusers",
        }
    ),
    # Qwen2.5-VL-7B text tower (Qwen2_5_VLTextModel): the HunyuanVideo-1.5 480p and 720p repacks ship identical
    # text_encoder/ sha256 (compared 2026-10-16). Kept apart from the Qwen-Image group, whose file is the full VLM.
    frozenset(
        {
            "hunyuanvideo-community/hunyuanvideo-1.5-diffusers-480p_t2v",
            "hunyuanvideo-community/hunyuanvideo-1.5-diffusers-720p_t2v",
        }
    ),
    # Qwen3-VL-4B: the FLUX.3 Klein distilled and base text_encoder/ shards carry identical sha256 (2026-10-16)
    frozenset(
        {
            "black-forest-labs/flux-3-image-klein",
            "black-forest-labs/flux-3-image-klein-base",
        }
    ),
    # Qwen3-VL-4B: identical sha256 across the Krea-2 pair (compared 2026-08-25). The tensors are Qwen3-VL-4B-Instruct's
    # (all 713 match, 2026-10-22), but packed as one file without the ``model.`` prefix, so the sha256 differs from
    # FLUX.3 Klein's two shards and the pair stays its own group.
    frozenset(
        {
            "krea/krea-2-turbo",
            "krea/krea-2-raw",
        }
    ),
    # T5-XXL: 2 shards, 9,524,648,584 bytes, identical sha256 across every FLUX.1 release (text_encoder_2; Kontext
    # re-verified 2026-10-16) and HiDream-I1 Full / Dev / Fast, which ship the same bytes as text_encoder_3 (the
    # family's te_prequant_repos row names the hosted file's component). The CLIP-L files differ between the two, which
    # is safe only because no CLIP is hosted: a group vouches for the components that have hosted files.
    frozenset(
        {
            "black-forest-labs/flux.1-schnell",
            "black-forest-labs/flux.1-dev",
            "black-forest-labs/flux.1-krea-dev",
            "black-forest-labs/flux.1-kontext-dev",
            "hidream-ai/hidream-i1-full",
            "hidream-ai/hidream-i1-dev",
            "hidream-ai/hidream-i1-fast",
        }
    ),
    # Z-Image's own Qwen3-4B (not stock, unlike FLUX.2 klein 4B's): identical sha256 across Turbo and base
    frozenset(
        {
            "tongyi-mai/z-image-turbo",
            "tongyi-mai/z-image",
        }
    ),
    # Stock Qwen3-4B (= unsloth/Qwen3-4B): the distilled and base FLUX.2 klein 4B repos ship identical shards
    frozenset(
        {
            "black-forest-labs/flux.2-klein-4b",
            "black-forest-labs/flux.2-klein-base-4b",
        }
    ),
    # Stock Qwen3-8B (= unsloth/Qwen3-8B): likewise for the 9B pair (compared 2026-10-16)
    frozenset(
        {
            "black-forest-labs/flux.2-klein-9b",
            "black-forest-labs/flux.2-klein-base-9b",
        }
    ),
    # MOSS-Image SFT and Base share identical text_encoder shards (LFS sha256).
    frozenset(
        {
            "openmoss-team/moss-image-sft",
            "openmoss-team/moss-image-base",
        }
    ),
)


def te_base_equivalent(ckpt_base: str, base: str) -> bool:
    """True when the checkpoint's baked base and the loading base carry byte-identical
    weights for the component: the same repo (``_same_base_model``) or a verified
    equivalence group above."""
    if _same_base_model(ckpt_base, base):
        return True
    # Groups hold UPSTREAM ids: an unnormalised mirror id would be refused.
    from .diffusion_families import canonical_base

    a, b = canonical_base(ckpt_base).lower(), canonical_base(base).lower()
    return any(a in group and b in group for group in _TE_EQUIVALENT_BASES)


@dataclass(frozen = True)
class TePrequantSource:
    """Where a pre-cast text-encoder checkpoint lives. ``kind`` is "path" (a local file) or
    "repo" (Hub repo id in ``location`` + ``filename``)."""

    kind: str
    location: str
    filename: Optional[str] = None
    # Ordered fallback names after ``filename`` if the repo lacks it; only the "repo" kind uses them.
    fallback_filenames: tuple = ()
    # The component the hosted FILE was built as, when another family loads the same bytes under a different slot
    # (HiDream's text_encoder_3 is FLUX.1's text_encoder_2). None = the component it is loaded into.
    component: Optional[str] = None


def int8_convrot_te_runs_on(target: Any) -> bool:
    """MPS runs the int8 ConvRot encoder without fp8: plain tensors, a Hadamard matmul and ``F.linear``."""
    return getattr(target, "device", None) == "mps"


def hosted_int8_convrot_te_runs_on(fam: Any, target: Any) -> bool:
    """Whether an explicit int8 encoder request on ``fam`` loads the hosted int8 ConvRot file on ``target`` without the
    fp8 fallback: the family hosts the file (and the lm_head trim it needs is on) and the device runs it as plain
    tensors. Pure: no IO."""
    try:
        if not int8_convrot_te_runs_on(target):
            return False
        hosted = family_te_int8_convrot(fam, "text_encoder")
        source = resolve_te_prequant_source(fam, "text_encoder", "int8")
        return (
            hosted is not None
            and source is not None
            and source.kind == "repo"
            and source.filename == hosted[1]
        )
    except Exception:  # noqa: BLE001 -- unresolvable means the runtime-cast rules apply
        return False


def _is_fp8_te_name(name: Optional[str]) -> bool:
    return bool(name) and "fp8" in str(name).lower()


def te_prequant_repo_stem(repo_id: str, component: str, scheme: str) -> str:
    """The extensionless checkpoint name for ``(component, scheme)`` in ``repo_id``: hosted repos
    are named <Model>-FP8 (or -INT8 / -quantized) and carry <Model>-<component>-<SCHEME> files,
    e.g. unsloth/LTX-2-FP8 -> LTX-2-text_encoder-FP8."""
    model = repo_id.rsplit("/", 1)[-1]
    for suffix in ("-fp8", "-int8", "-quantized"):
        if model.lower().endswith(suffix):
            model = model[: -len(suffix)]
            break
    return f"{model}-{component}-{scheme.upper()}"


def te_prequant_repo_filenames(repo_id: str, component: str, scheme: str) -> tuple:
    """Both .safetensors and .pt are live and the name must be requested; safetensors is tried first."""
    # Lazy import: prequant_safetensors pulls torchao, which some hosts lack.
    from .prequant_safetensors import SAFETENSORS_SUFFIX

    stem = te_prequant_repo_stem(repo_id, component, scheme)
    return (f"{stem}{SAFETENSORS_SUFFIX}", f"{stem}.pt")


def te_prequant_companion_names(repo_id: str, families: Iterable[Any]) -> set[str]:
    """Lowercased root file names a pre-cast text encoder can take in ``repo_id``: every ``te_prequant_repos`` row
    (three fields, or four with the component the file was built as), every variant row, and the concrete int8
    ConvRot files. Cache cleanup reads these as companions, not as an installed checkpoint."""
    pairs: set[tuple[str, str]] = set()
    for fam in families:
        for row in getattr(fam, "te_prequant_repos", ()) or ():
            if len(row) >= 3:
                pairs.add((str(row[3] if len(row) > 3 else row[1]), str(row[0])))
        for row in getattr(fam, "te_prequant_variant_repos", ()) or ():
            if len(row) >= 4:
                pairs.add((str(row[2]), str(row[1])))
    pairs |= {(c, s) for c, _s in pairs for s in ("fp8", "int8")}
    names = {n.lower() for c, s in pairs for n in te_prequant_repo_filenames(repo_id, c, s)}
    key = str(repo_id or "").strip().lower()
    entries = [e for comps in TE_INT8_CONVROT_FILES.values() for e in comps.values()]
    entries += [
        e
        for bases in TE_INT8_CONVROT_VARIANT_FILES.values()
        for comps in bases.values()
        for e in comps.values()
    ]
    for entry in entries:
        if len(entry) >= 2 and str(entry[0]).strip().lower() == key:
            names.add(str(entry[1]).lower())
    return names


def te_prequant_repo_filename(repo_id: str, component: str, scheme: str) -> str:
    """The preferred checkpoint filename for ``(component, scheme)`` in ``repo_id``."""
    return te_prequant_repo_filenames(repo_id, component, scheme)[0]


def te_candidate_filenames(source: Any) -> tuple:
    """Shared by the plan and resolver so they cannot disagree on which artifact a source means."""
    names = (
        getattr(source, "filename", None),
        *(getattr(source, "fallback_filenames", None) or ()),
    )
    return tuple(n for n in names if n)


def te_candidate_is_readable(name: Optional[str]) -> bool:
    """Not the DiT's torchao check: a plain-tensor .pt always reads; safetensors needs no torchao."""
    if not name:
        return False
    from .prequant_safetensors import is_safetensors_checkpoint, plain_safetensors_supported

    return plain_safetensors_supported() if is_safetensors_checkpoint(name) else True


def family_te_prequant_repo(fam: Any, scheme: str, component: str) -> Optional[str]:
    """The hosted pre-cast encoder repo for ``(scheme, component)`` in this family, or None.

    Reads the family's ``te_prequant_repos`` (scheme, component, repo_id) triples; the field
    is optional on both DiffusionFamily and VideoFamily, so one resolver serves both loaders.
    """
    entry = _family_te_prequant_entry(fam, scheme, component)
    return entry[0] if entry else None


def _family_te_prequant_entry(fam: Any, scheme: str, component: str) -> Optional[tuple[str, str]]:
    """``(repo_id, hosted_component)`` of the row for ``(scheme, component)``. A row may carry a fourth element naming
    the component the hosted file was built as (another family's slot for the same bytes)."""
    from .diffusion_nvfp4_flag import nvfp4_blocked

    if nvfp4_blocked(scheme):
        return None
    for entry in getattr(fam, "te_prequant_repos", ()) or ():
        try:
            entry_scheme, entry_component, repo_id, *rest = entry
        except Exception:  # noqa: BLE001 - a malformed entry must not break the load
            continue
        if len(rest) > 1 or not isinstance(repo_id, str):
            continue
        if entry_scheme == scheme and entry_component == component:
            return repo_id, (rest[0] if rest else component)
    return None


def resolve_te_prequant_source(
    fam: Any,
    component: str,
    scheme: str,
    *,
    path_override: Optional[str] = None,
) -> Optional[TePrequantSource]:
    """Order: local path_override, then the family's hosted repo, else None (download dense and cast)."""
    if scheme not in TE_PREQUANT_SCHEMES:
        return None
    override = (path_override or "").strip()
    if override:
        return TePrequantSource(kind = "path", location = override, filename = None)
    if scheme == "int8":
        # int8 file first, then the fp8 names in the same repo, so a miss takes fp8 rather than the dense shards.
        hosted = family_te_int8_convrot(fam, component)
        if hosted is None:
            return None
        repo_id, filename, *hosted_component = hosted
        fp8 = resolve_te_prequant_source(fam, component, "fp8")
        from .diffusion_text_encoder_trim import family_trims_lm_head

        family = str(getattr(fam, "name", "") or "").strip().lower()
        if family not in _TE_INT8_HEADLESS_FAMILIES and not family_trims_lm_head(family):
            # The int8 file has no lm_head; with the head kept (UNSLOTH_TE_KEEP_LM_HEAD) only the fp8 one can serve.
            return fp8
        fallback = ()
        if (
            fp8 is not None
            and fp8.kind == "repo"
            and fp8.location.lower() == repo_id.lower()
            and not _family_has_torchao_int8(fam)
        ):
            # A family with a keep-bf16 int8 schedule (Qwen-Image and its releases, FLUX.2-dev) keeps the on-the-fly
            # torchao int8 as the fallback for a missing file, as before the file existed; fp8 comes after that.
            fallback = te_candidate_filenames(fp8)
        return TePrequantSource(
            kind = "repo",
            location = repo_id,
            filename = filename,
            fallback_filenames = tuple(n for n in fallback if n != filename),
            component = hosted_component[0] if hosted_component else None,
        )
    entry = _family_te_prequant_entry(fam, scheme, component)
    if entry:
        repo_id, hosted_component = entry
        names = te_prequant_repo_filenames(repo_id, hosted_component, scheme)
        return TePrequantSource(
            kind = "repo",
            location = repo_id,
            filename = names[0],
            fallback_filenames = names[1:],
            component = hosted_component if hosted_component != component else None,
        )
    return None


def _family_has_torchao_int8(fam: Any) -> bool:
    try:
        from .diffusion_precision import _TE_INT8_SKIP
        return _TE_INT8_SKIP.get(str(getattr(fam, "name", "") or "").lower()) is not None
    except Exception:  # noqa: BLE001
        return False


def te_prequant_sources(
    fam: Any,
    *,
    te_quant_mode: Optional[str],
    target: Any,
    components: Iterable[str] = TE_PREQUANT_COMPONENTS,
) -> dict[str, TePrequantSource]:
    """Shares the download plan's resolver, so the plan and load agree on which dense encoders to fetch."""
    try:
        from . import diffusion_precision as precision
        from .diffusion_precision import (
            TE_QUANT_FP8,
            normalize_te_quant,
            te_quant_supported,
        )

        from .diffusion_precision import te_quant_picked_automatically

        auto = te_quant_picked_automatically(te_quant_mode)
        mode = normalize_te_quant(te_quant_mode)
        if mode not in TE_PREQUANT_SCHEMES:
            return {}
        family = getattr(fam, "name", None)
        # TE deny table lives on the video branch precision module; resolve lazily.
        denied = getattr(precision, "_te_family_denied", None)
        if callable(denied) and denied(family, mode):
            return {}
        # int8 falls back to the fp8 file, so it needs fp8; MPS runs the int8 file alone and drops the fp8 names.
        fp8_runs = te_quant_supported(target, TE_QUANT_FP8)
        if not fp8_runs and not (mode == "int8" and int8_convrot_te_runs_on(target)):
            return {}
        sources: dict[str, TePrequantSource] = {}
        for component in components:
            source = resolve_te_prequant_source(fam, component, mode)
            if source is not None and auto and mode == "int8":
                # An automatic int8 pick takes the int8 file or the dense encoder, never the fp8 file behind it, nor
                # the fp8 file put first by a kept lm_head (UNSLOTH_TE_KEEP_LM_HEAD).
                if source.kind == "repo" and _is_fp8_te_name(source.filename):
                    continue
                source = replace(source, fallback_filenames = ())
            if source is not None and not fp8_runs:
                if source.kind == "repo" and _is_fp8_te_name(source.filename):
                    continue
                source = replace(source, fallback_filenames = ())
            if source is not None:
                sources[component] = source
        return sources
    except Exception:  # noqa: BLE001 -- fall back to the dense encoder
        return {}


def te_prequant_sources_for_base(
    fam: Any,
    base: str,
    *,
    te_quant_mode: Optional[str],
    target: Any,
    components: Iterable[str] = TE_PREQUANT_COMPONENTS,
    standalone_component_bases: Optional[dict[str, str]] = None,
) -> dict[str, TePrequantSource]:
    """Pre-cast sources whose registered encoder base matches the selected base.

    A family match alone is insufficient for a custom pipeline. The hosted checkpoint was
    built from the family's registered base, so selecting it for another base would download
    the artifact before checkpoint metadata could reject it. Components loaded from their own
    standalone repos may map both sides of this comparison through
    ``standalone_component_bases``.

    A variant base whose encoder is a different model (``te_prequant_variant_repos``) resolves
    against that variant's rows and files only, never the family base's.

    An automatic pick (unset request) never swaps a hosted encoder into a load from a local folder: the folder name
    matching the upstream repo says nothing about the weights inside it, which may be a fine-tune. That load keeps
    the folder's own encoder, as it did before the family had an automatic default. An explicit scheme is unchanged.
    """
    auto_local = _auto_pick(te_quant_mode) and _is_local_base(base)
    fam = te_variant_family_view(fam, base)
    sources = te_prequant_sources(
        fam,
        te_quant_mode = te_quant_mode,
        target = target,
        components = components,
    )
    registered_base = str(getattr(fam, "base_repo", "") or "")
    standalone = standalone_component_bases or {}
    compatible: dict[str, TePrequantSource] = {}
    for component, source in sources.items():
        checkpoint_base = standalone.get(component) or registered_base
        selected_base = standalone.get(component) or base
        if auto_local and component not in standalone:
            continue
        if checkpoint_base and selected_base and te_base_equivalent(checkpoint_base, selected_base):
            compatible[component] = source
    return compatible


def _is_local_base(base: Optional[str]) -> bool:
    """Whether ``base`` is a folder on disk rather than a Hub id (the loaders' own rule)."""
    if not base:
        return False
    try:
        from .diffusion_families import _is_local_path
        return bool(_is_local_path(str(base)))
    except Exception:  # noqa: BLE001
        return False


def te_variant_family_view(fam: Any, base: Optional[str]) -> Any:
    """``fam`` as seen by a load of ``base``: unchanged unless ``base`` is (equivalent to) a variant listed in the
    family's ``te_prequant_variant_repos``, in which case a copy whose ``base_repo`` is that variant and whose
    ``te_prequant_repos`` are that variant's rows. ``family_te_int8_convrot`` keys its variant table on the same base,
    so the copy can never resolve the family base's encoder for a variant that ships another model."""
    rows = getattr(fam, "te_prequant_variant_repos", ()) or ()
    if not rows or not base:
        return fam
    registered = str(getattr(fam, "base_repo", "") or "")
    if registered and te_base_equivalent(registered, base):
        return fam
    for variant in dict.fromkeys(str(row[0]) for row in rows if row):
        if te_base_equivalent(variant, base):
            import dataclasses
            import types

            own = tuple(tuple(row[1:]) for row in rows if row and str(row[0]) == variant)
            try:
                return dataclasses.replace(fam, base_repo = variant, te_prequant_repos = own)
            except Exception:  # noqa: BLE001 - not a dataclass: no pre-cast encoder rather than the wrong one
                return types.SimpleNamespace(
                    name = getattr(fam, "name", None), base_repo = variant, te_prequant_repos = own
                )
    return fam


# Weight files a dense encoder folder holds. Everything else (config.json, the shard index, tokenizer JSON) is kept
# when the pre-cast checkpoint replaces the weights: the pre-cast loader still meta-inits from the base repo component
# config.
_TE_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth", ".pt", ".msgpack", ".h5")


def is_prequant_covered_weight(rfilename: str, components: Iterable[str]) -> bool:
    """True when ``rfilename`` is a dense weight shard of one of ``components`` -- i.e. a file
    a pre-cast checkpoint makes unnecessary to download."""
    lowered = rfilename.lower()
    if not lowered.endswith(_TE_WEIGHT_SUFFIXES):
        return False
    return any(rfilename.startswith(f"{component}/") for component in components)


def load_prequant_text_encoder(
    base: str,
    component: str,
    source: TePrequantSource,
    *,
    dtype: Any,
    hf_token: Optional[str] = None,
    scheme: str = "fp8",
    logger: Any = None,
    config_subfolder: Optional[str] = None,
    config_overrides: Optional[dict] = None,
    local_files_only: bool = False,
    trim_lm_head: bool = False,
    failures: Optional[list] = None,
    config_source: Optional[str] = None,
) -> Optional[Any]:
    """Load the pre-cast text encoder described by ``source`` (on CPU, for pipeline
    assembly to place), with the layerwise upcast hooks already installed.

    Returns the encoder, or None on any problem (missing / mismatched / unreadable
    checkpoint) so the caller falls back to the dense download + cast. Best-effort:
    never raises for an unavailable artifact.

    ``config_subfolder`` overrides where the encoder config lives in ``base`` (default:
    the component name; "" means the repo root, for encoders assembled from a separate
    standalone repo like HiDream's Llama TE4). ``config_overrides`` sets config fields
    the pipeline's assembly normally passes to ``from_pretrained`` (forward-behaviour
    flags only; the state dict is unaffected by them).
    ``trim_lm_head`` builds the encoder without its untied ``lm_head`` and never reads that tensor
    (``diffusion_text_encoder_trim``). ``failures`` collects the error text of a load that raised.
    ``config_source`` is where the encoder config is read when it is not ``base``: the staged directory the
    pipeline itself assembles from, which a cache-only load may have resolved under another cache root. ``base``
    stays the canonical id the checkpoint is validated against."""
    try:
        if source.kind == "path" and not _local_prequant_path_allowed(source.location):
            _warn(
                logger,
                f"{scheme}:{component}:path",
                RuntimeError(
                    "request-supplied local pre-cast path refused; set "
                    f"{ALLOW_LOCAL_PREQUANT_PATH_ENV} to an allowlisted directory "
                    "containing trusted checkpoints to permit it",
                ),
            )
            return None

        import os

        from utils.hf_cache_settings import active_hf_hub_cache

        cache_dir = active_hf_hub_cache()
        path = _resolve_checkpoint_path(
            source,
            hf_token,
            cache_dir = cache_dir,
            local_files_only = local_files_only,
            logger = logger,
        )
        if path is None:
            return None

        import torch

        from .prequant_safetensors import is_safetensors_checkpoint, load_plain_prequant_safetensors

        # weights_only=True: no pickle runs; a torchao-subclass scheme needs a format bump and an allowlist.
        from .diffusion_text_encoder_trim import (
            LM_HEAD_KEY,
            class_trims_lm_head,
            config_ties_lm_head,
            trim_text_encoder,
        )

        te_class = None
        skip = ()
        if is_safetensors_checkpoint(path):
            if trim_lm_head:
                te_class = _safetensors_te_class(path)
            skip = (LM_HEAD_KEY,) if trim_lm_head and class_trims_lm_head(te_class) else ()
            ckpt = load_plain_prequant_safetensors(path, skip_names = skip)
        else:
            ckpt = torch.load(path, weights_only = True, map_location = "cpu")
        # By format tag: an int8 request accepts its fp8 fallback; an fp8 request never accepts an int8 file.
        file_scheme = (
            "int8"
            if isinstance(ckpt, dict) and ckpt.get("format") == TE_PREQUANT_FORMAT_INT8_CONVROT
            else "fp8"
        )
        if file_scheme != scheme and not (scheme == "int8" and file_scheme == "fp8"):
            _warn(logger, scheme, ValueError(f"checkpoint scheme {file_scheme!r} != {scheme!r}"))
            return None
        if not _validate_checkpoint(
            ckpt, file_scheme, getattr(source, "component", None) or component, base, logger
        ):
            return None
        state_dict = ckpt["state_dict"]
        te_class = (ckpt.get("metadata") or {}).get("te_class")
        trim = trim_lm_head and class_trims_lm_head(te_class)

        import transformers

        encoder_cls = getattr(transformers, str(te_class), None)
        if encoder_cls is None:
            _warn(
                logger,
                f"{scheme}:{component}",
                ValueError(f"checkpoint te_class {te_class!r} not found in transformers"),
            )
            return None
        subfolder = component if config_subfolder is None else config_subfolder
        config_kwargs: dict[str, Any] = {
            "token": hf_token,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        if subfolder:
            config_kwargs["subfolder"] = subfolder
        from .diffusion_families import pinned_revision_kwargs

        config_from = config_source or base
        config = transformers.AutoConfig.from_pretrained(
            config_from, **config_kwargs, **pinned_revision_kwargs(config_from)
        )
        # Krea-2 ships transformers-5.x configs whose rope lives under rope_parameters; the runtime component loader
        # remaps it for a 4.x runtime, and the meta-init here must match or the rebuilt encoder forwards with a broken
        # rope.
        from .diffusion_krea2 import remap_rope_parameters

        remap_rope_parameters(getattr(config, "text_config", config))
        for key, value in (config_overrides or {}).items():
            setattr(config, key, value)
        if file_scheme == "int8":
            encoder = _build_int8_convrot_encoder(
                encoder_cls, config, state_dict, dtype = dtype, trim = trim
            )
            setattr(encoder, TE_PREQUANT_SCHEME_ATTR, "int8")
            if logger is not None:
                logger.info(
                    "diffusion.te_prequant: loaded %s int8 ConvRot weight-only checkpoint (%s)",
                    component,
                    path
                    if source.kind == "path"
                    else f"{source.location}/{os.path.basename(path)}",
                )
            return encoder
        if trim and config_ties_lm_head(config):
            # Tied: nothing to drop.
            trim = False
            if LM_HEAD_KEY in skip:
                state_dict[LM_HEAD_KEY] = _read_safetensors_tensor(path, LM_HEAD_KEY)
        if trim:
            state_dict.pop(LM_HEAD_KEY, None)
        from accelerate import init_empty_weights

        with init_empty_weights():
            encoder = encoder_cls(config)
        if trim:
            trim_text_encoder(encoder)
        _fill_tied_aliases(encoder, state_dict)
        encoder.load_state_dict(state_dict, strict = True, assign = True)
        if _has_meta_tensors(encoder):
            # Non-persistent buffers stay on meta: rebuild on CPU, then re-assign cast weights.
            encoder = encoder_cls(config)
            if trim:
                trim_text_encoder(encoder)
            encoder.load_state_dict(state_dict, strict = True, assign = True)
        # assign=True unties tied weights (e.g. Qwen3 lm_head); re-tie them.
        tie = getattr(encoder, "tie_weights", None)
        if callable(tie):
            tie()
        encoder.eval()

        # Install the runtime upcast hooks, else fp8 weights meet bf16 activations.
        from .diffusion_precision import _cast_fp8

        class _Target:
            pass

        target = _Target()
        target.dtype = dtype
        _cast_fp8(encoder, target)
        setattr(encoder, TE_PREQUANT_SCHEME_ATTR, "fp8")
        if logger is not None:
            # The FILE's scheme, not the request's (an fp8 fallback used to log int8).
            logger.info(
                "diffusion.te_prequant: loaded %s fp8 checkpoint (%s)%s",
                component,
                path if source.kind == "path" else f"{source.location}/{os.path.basename(path)}",
                "; the int8 request took the fp8 file because the int8 ConvRot file is not available"
                if scheme == "int8"
                else "",
            )
        return encoder
    except Exception as exc:  # noqa: BLE001 - fall back to the dense download + cast
        _warn(logger, f"{scheme}:{component}:{source.kind}", exc)
        if failures is not None:
            # text only: the exception's traceback would pin this frame's checkpoint through the fp8 retry
            failures.append(str(exc))
        return None


def supplied_component_pipe_kwargs(
    base: str,
    *,
    dtype: Any,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
    family: Optional[str] = None,
    logger: Any = None,
) -> dict[str, Any]:
    """Supplied text-encoder / VAE modules by component. Never swallows a failure: the base repo's weights for
    these components were not downloaded, so there is nothing to fall back to."""
    from .diffusion_comfy_components import active_component_overrides, load_override_modules

    overrides = active_component_overrides()
    if overrides is None or not overrides.files:
        return {}
    from utils.hf_cache_settings import active_hf_hub_cache

    return load_override_modules(
        overrides,
        base = base,
        dtype = dtype,
        hf_token = hf_token,
        local_files_only = local_files_only,
        family = family,
        cache_dir = active_hf_hub_cache(),
        logger = logger if logger is not None else _module_logger(),
    )


def _module_logger() -> Any:
    try:
        from loggers import get_logger
        return get_logger(__name__)
    except Exception:  # noqa: BLE001
        return None


def te_prequant_pipe_kwargs(
    fam: Any,
    base: str,
    *,
    te_quant_mode: Optional[str],
    target: Any,
    dtype: Any,
    hf_token: Optional[str] = None,
    logger: Any = None,
    local_files_only: bool = False,
    dense_source: Optional[str] = None,
    config_source: Optional[str] = None,
) -> dict[str, Any]:
    supplied = supplied_component_pipe_kwargs(
        base,
        dtype = dtype,
        hf_token = hf_token,
        local_files_only = local_files_only,
        family = getattr(fam, "name", None),
        logger = logger,
    )
    injected = _te_prequant_pipe_kwargs(
        fam,
        base,
        te_quant_mode = te_quant_mode,
        target = target,
        dtype = dtype,
        hf_token = hf_token,
        logger = logger,
        local_files_only = local_files_only,
        skip_components = tuple(supplied),
        dense_source = dense_source,
        config_source = config_source,
    )
    injected.update(supplied)
    return injected


def _te_prequant_pipe_kwargs(
    fam: Any,
    base: str,
    *,
    te_quant_mode: Optional[str],
    target: Any,
    dtype: Any,
    hf_token: Optional[str] = None,
    logger: Any = None,
    local_files_only: bool = False,
    skip_components: tuple = (),
    dense_source: Optional[str] = None,
    config_source: Optional[str] = None,
) -> dict[str, Any]:
    """Gated like the runtime cast, so injection never engages where quantize_text_encoders would not."""
    unavailable: list[str] = []
    unbudgeted: list[str] = []
    built_failed: dict[str, str] = {}
    refusal: Optional[str] = None
    failures: list = []
    cause = ""
    try:
        from .diffusion_precision import TE_QUANT_FP8, normalize_te_quant, te_quant_supported
        from .diffusion_text_encoder_trim import family_trims_lm_head

        sources = te_prequant_sources_for_base(
            fam,
            base,
            te_quant_mode = te_quant_mode,
            target = target,
        )
        # Non-empty only for a hosted scheme (see te_prequant_sources' gate).
        mode = normalize_te_quant(te_quant_mode) or TE_QUANT_FP8
        injected: dict[str, Any] = {}
        for component, source in sources.items():
            if component in skip_components:
                continue
            failures.clear()
            trim = component == "text_encoder" and family_trims_lm_head(getattr(fam, "name", None))
            encoder = load_prequant_text_encoder(
                base,
                component,
                source,
                dtype = dtype,
                hf_token = hf_token,
                scheme = mode,
                logger = logger,
                local_files_only = local_files_only,
                trim_lm_head = trim,
                failures = failures,
                config_source = config_source,
            )
            fp8_names = tuple(
                n for n in te_candidate_filenames(source) if n != getattr(source, "filename", None)
            )
            if (
                encoder is None
                and mode == "int8"
                and not _picked_automatically(te_quant_mode)
                and source.kind == "repo"
                and fp8_names
                and te_quant_supported(target, TE_QUANT_FP8)
                and _held_locally(source.location, source.filename, hf_token)
            ):
                # The int8 file resolved but was refused: the plan dropped the dense shards, so take the fp8 names.
                encoder = load_prequant_text_encoder(
                    base,
                    component,
                    replace(
                        source,
                        filename = fp8_names[0],
                        fallback_filenames = fp8_names[1:],
                    ),
                    dtype = dtype,
                    hf_token = hf_token,
                    scheme = TE_QUANT_FP8,
                    logger = logger,
                    local_files_only = local_files_only,
                    trim_lm_head = trim,
                    failures = failures,
                    config_source = config_source,
                )
            if (
                encoder is None
                and _picked_automatically(te_quant_mode)
                and te_prequant_sources_confirmed(
                    {component: source}, local_files_only = local_files_only
                )
            ):
                # The file is there but did not build: the plan priced it, not the dense encoder.
                built_failed[component] = getattr(source, "filename", None) or str(source.location)
            if encoder is not None:
                injected[component] = encoder
            elif int8_convrot_te_runs_on(target):
                # The MPS plan priced this checkpoint, not the dense shards beside it.
                unbudgeted.append(component)
                cause = f" ({failures[-1]})" if failures else cause
            elif (dense_source or component in built_failed) and not _dense_component_cached(
                dense_source or base, component
            ):
                # Online too for a confirmed automatic file that did not build: the plan left the dense shards out and
                # priced the file, so fetching the dense encoder now would overrun it.
                unavailable.append(component)
                cause = f" ({failures[-1]})" if failures else cause
        if not unavailable and not unbudgeted:
            if built_failed and logger is not None:
                # Dense shards already cached off MPS: the cached-plan path priced max(file, scanned dense), so the
                # dense encoder is inside the plan.
                logger.warning(
                    "diffusion.te_prequant: automatic %s %s for %s is present but could not be loaded%s; loading the "
                    "cached dense encoder, which the memory plan covered",
                    mode,
                    ", ".join(built_failed.values()),
                    base,
                    cause,
                )
            return injected
        uncovered = [c for c in unbudgeted + unavailable if c in built_failed]
        if uncovered:
            # A confirmed automatic file that fails to build: the memory plan was sized for it (te_prequant_budget_scale,
            # _precast_text_encoder_mib), and the dense encoder is not covered (not cached, or MPS, where the plan never
            # takes the dense shards). Loading it anyway would overrun the plan after the resident model was evicted.
            names = ", ".join(f"{built_failed[c]} ({c})" for c in uncovered)
            refusal = (
                f"The hosted pre-quantized text encoder {names} for {base} is present but could not be "
                f"loaded{cause}. This load was sized for it, so the larger dense encoder is not loaded in its "
                "place. Set Text encoder precision to Dense (bf16) to load the dense encoder instead, or delete "
                "the cached file so it is downloaded again."
            )
        elif _picked_automatically(te_quant_mode) and _fetch_dense_components(
            dense_source or base,
            unavailable + unbudgeted,
            hf_token = hf_token,
            local_files_only = local_files_only,
            logger = logger,
        ):
            # Nobody asked for this scheme: the dense encoder loads in its place rather than failing the load.
            if logger is not None:
                logger.warning(
                    "diffusion.te_prequant: automatic %s %s for %s could not be loaded%s; loading the dense encoder",
                    mode,
                    ", ".join(unbudgeted + unavailable),
                    base,
                    cause,
                )
            return injected
    except Exception as exc:  # noqa: BLE001 - injection is an optimisation, never a blocker
        _warn(logger, "pipe_kwargs", exc)
        return {}
    if refusal is not None:
        raise RuntimeError(refusal)
    if unbudgeted:
        raise RuntimeError(
            f"The pre-quantized {', '.join(unbudgeted)} for {base} could not be loaded{cause}. The "
            "memory check sized this load for it, so the larger dense encoder is not loaded in its "
            "place. Set Text encoder precision to Dense (bf16) to load the dense encoder instead."
        )
    raise RuntimeError(
        f"The pre-quantized {', '.join(unavailable)} for {base} could not be loaded{cause}. Its dense "
        "weights were not downloaded because the pre-quantized copy replaces them. If memory ran out "
        "(os error 1455 means the Windows page file is too small), close other apps or enlarge the page "
        "file and load again, or set Text encoder precision to Dense (bf16)."
    )


def _fetch_dense_components(
    source: str,
    components: list,
    *,
    hf_token: Optional[str],
    local_files_only: bool,
    logger: Any = None,
) -> bool:
    """Bring the dense shards of ``components`` the plan left out back into ``source`` (a Hub cache snapshot or repo
    id) so assembly can load them. True when there is nothing to fetch or the fetch worked; False offline, for a local
    folder that is not a cache snapshot, or on any failure."""
    if not components:
        return True
    try:
        if all(_dense_component_cached(source, c) for c in components):
            return True
    except Exception:  # noqa: BLE001 - unknown: try the fetch
        pass
    if local_files_only:
        return False
    try:
        import os

        from huggingface_hub import snapshot_download

        from utils.hf_cache_settings import active_hf_hub_cache

        repo_id, revision, cache_dir = source, None, active_hf_hub_cache()
        if os.path.isdir(str(source)):
            parts = os.path.normpath(str(source)).split(os.sep)
            if len(parts) < 3 or parts[-2] != "snapshots" or not parts[-3].startswith("models--"):
                return False
            repo_id = parts[-3][len("models--") :].replace("--", "/")
            # Into the SAME snapshot assembly opens: its cache root (the secondary one included) and its revision, so
            # a staged snapshot gains the shards rather than a new one appearing elsewhere.
            revision, cache_dir = parts[-1], os.sep.join(parts[:-3]) or os.sep
        fetched = snapshot_download(
            repo_id,
            allow_patterns = [f"{c}/*" for c in components],
            token = hf_token,
            cache_dir = cache_dir,
            revision = revision,
        )
        if revision is not None and os.path.realpath(str(fetched)) != os.path.realpath(str(source)):
            return False
        return True
    except Exception as exc:  # noqa: BLE001 - the strict refusal below still explains the load
        _warn(logger, "dense_fallback_fetch", exc)
        return False


def _held_locally(repo_id: str, name: Optional[str], hf_token: Optional[str]) -> bool:
    """Whether ``repo_id/name`` is on this machine, without touching the network: the operator's mirror, or either
    cache root the loader reuses (the live one and huggingface_hub's import-time one, as _download_checkpoint_name)."""
    try:
        if te_prequant_mirror_path(repo_id, name) is not None:
            return True
        import types

        from utils.hf_cache_settings import active_hf_hub_cache

        from .diffusion_prequant import _cached_in_root

        source = types.SimpleNamespace(kind = "repo", location = repo_id, filename = name)
        return any(
            _cached_in_root(source, root, name) is not None
            for root in (active_hf_hub_cache(), None)
        )
    except Exception:  # noqa: BLE001 - not cached, or unanswerable
        return False


def _dense_component_cached(source: str, component: str) -> bool:
    """Whether ``component``'s dense shards are on disk for ``source`` (a folder, or a repo id in either cache root)."""
    from .media_locality import _component_present, _hosted_component_cached

    if _hosted_component_cached(source, component, "", ""):
        return True
    import os

    if os.path.isdir(str(source)):
        return False
    try:
        from pathlib import Path

        from huggingface_hub import constants

        repo_dir = Path(constants.HF_HUB_CACHE) / f"models--{str(source).replace('/', '--')}"
        ref = (repo_dir / "refs" / "main").read_text(encoding = "utf-8").strip()
        snapshot = repo_dir / "snapshots" / ref
        return bool(ref) and snapshot.is_dir() and _component_present(snapshot / component)
    except Exception:  # noqa: BLE001 - not cached there
        return False


def _build_int8_convrot_encoder(
    encoder_cls: Any, config: Any, state_dict: dict, *, dtype: Any, trim: bool
) -> Any:
    """Every Linear with a ``<name>.weight_scale`` becomes MiniMax-H3's ``Int8ConvRotLinear``, the rest loads dense in
    ``dtype``. Plain tensors, so ``Module.to()`` and group offloading work. Raises on any mismatch."""
    import torch
    from accelerate import init_empty_weights

    from .diffusion_text_encoder_trim import (
        LM_HEAD_KEY,
        _no_logits_head_class,
        config_ties_lm_head,
        is_trimmed,
        trim_text_encoder,
    )
    from .video_minimax_h3_te import (
        H3_TE_CONVROT_GROUP,
        _int8_convrot_linear_class,
        _meta_tensor_names,
        _validate_comfy_quant,
    )

    if LM_HEAD_KEY in state_dict:
        raise ValueError("the int8 ConvRot encoder unexpectedly carries an lm_head")
    scale_suffix, quant_suffix = ".weight_scale", ".comfy_quant"
    quantized = sorted(k[: -len(scale_suffix)] for k in state_dict if k.endswith(scale_suffix))
    if not quantized:
        raise ValueError("no int8 ConvRot projections in this checkpoint")
    # include_buffers=False: rotary inv_freq buffers are built for real; parameters arrive via assign=True.
    with init_empty_weights(include_buffers = False):
        encoder = encoder_cls(config)
    # An encoder-only class (UMT5EncoderModel) has no head at all; one that has a head needs the trim, as the file
    # carries no lm_head and a kept one would load random.
    if getattr(encoder, "lm_head", None) is not None and not trim:
        raise ValueError("the int8 ConvRot encoder carries no lm_head and needs the lm_head trim")
    if trim:
        trim_text_encoder(encoder)
    if (
        getattr(encoder, "lm_head", None) is not None
        and not is_trimmed(encoder)
        and config_ties_lm_head(config)
    ):
        # A tied head (FLUX.3's Qwen3-VL-4B) is kept by the trim, but this file has no lm_head key for it and the
        # family reads hidden states only, so drop it here: a meta head would fail the strict load below.
        head = encoder.lm_head
        encoder.lm_head = _no_logits_head_class()(head.in_features, head.out_features)
    linear_cls = _int8_convrot_linear_class()
    for prefix in quantized:
        blob = state_dict.pop(prefix + quant_suffix, None)
        if blob is None:
            raise ValueError(f"{prefix}: quantized weight with no {quant_suffix} metadata")
        _validate_comfy_quant(blob, prefix)
        parent_path, _, leaf = prefix.rpartition(".")
        parent = encoder.get_submodule(parent_path)
        existing = getattr(parent, leaf)
        weight = state_dict[prefix + ".weight"]
        if not isinstance(existing, torch.nn.Linear) or weight.dtype != torch.int8:
            raise ValueError(f"{prefix}: not an int8 weight for a Linear")
        if (existing.out_features, existing.in_features) != tuple(weight.shape):
            raise ValueError(
                f"{prefix}: checkpoint shape {tuple(weight.shape)} != model "
                f"{(existing.out_features, existing.in_features)}"
            )
        setattr(
            parent,
            leaf,
            linear_cls(
                existing.in_features,
                existing.out_features,
                bias = (prefix + ".bias") in state_dict,
                group_size = H3_TE_CONVROT_GROUP,
            ),
        )
    if any(k.endswith(quant_suffix) for k in state_dict):
        raise ValueError("quant metadata for a projection with no scale")
    # The encoder's own fp32-kept modules (T5's ``wo``, which overflows in low precision): a float32 tensor stored for
    # one stays float32, as from_pretrained keeps it.
    keep_fp32 = tuple(str(m) for m in (getattr(encoder, "_keep_in_fp32_modules", None) or ()))

    def _kept_fp32(key: str) -> bool:
        return any(token in key.split(".") for token in keep_fp32)

    # Storage dtypes stay (int8 payload, float32 scales); every dense float tensor follows the compute dtype.
    for key, tensor in list(state_dict.items()):
        if (
            tensor.is_floating_point()
            and not key.endswith(scale_suffix)
            and dtype is not None
            and tensor.dtype != dtype
            and not (tensor.dtype == torch.float32 and _kept_fp32(key))
        ):
            state_dict[key] = tensor.to(dtype)
    # A dense block projection would load as bf16 and outgrow the budget this path is priced at (the encoder's own
    # fp32-kept modules, UMT5's ``wo``, stay dense by design).
    dense = [
        n
        for n, m in encoder.named_modules()
        if isinstance(m, torch.nn.Linear) and TE_INT8_CONVROT_LINEAR.search(n) and not _kept_fp32(n)
    ]
    if dense:
        raise ValueError(
            f"{len(dense)} decoder projection(s) are not quantized in this artifact, e.g. {sorted(dense)[0]}"
        )
    _fill_tied_aliases(encoder, state_dict)
    encoder.load_state_dict(state_dict, strict = True, assign = True)
    # assign=True gives a tied pair separate tensors; re-tie a headless encoder (T5 / UMT5's encoder.embed_tokens ->
    # shared), as the fp8 path does. One with a (trimmed) head is left as loaded: its head is a stand-in with no weight
    # to tie.
    tie = getattr(encoder, "tie_weights", None)
    if callable(tie) and getattr(encoder, "lm_head", None) is None:
        tie()
    stranded = _meta_tensor_names(encoder)
    if stranded:
        raise ValueError(f"{len(stranded)} tensor(s) still on the meta device, e.g. {stranded[0]}")
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def _fill_tied_aliases(encoder: Any, state_dict: dict) -> None:
    """Give a key the file leaves out because the model ties it to another one the source of that tie.

    A tied embedding is one tensor under two names (UMT5's ``encoder.embed_tokens.weight`` is ``shared.weight``), and a
    file that stores it once (the dense shards themselves, an int8 build, a converter run with --drop-tied-copies)
    would otherwise fail the strict load. Only ties the MODEL declares: transformers 5's ``_tied_weights_keys`` map,
    or one parameter registered under two names (4.x shares the module). ``lm_head`` ties are left alone, because
    whether a head is tied depends on the config and the callers handle it themselves."""
    pairs: list = []
    tied = getattr(encoder, "_tied_weights_keys", None)
    if isinstance(tied, dict):
        pairs.extend((str(t), str(s)) for t, s in tied.items())
    try:
        names: dict = {}
        for name, param in encoder.named_parameters(remove_duplicate = False):
            names.setdefault(id(param), []).append(name)
        for group in names.values():
            pairs.extend((t, s) for t in group for s in group if t != s)
    except Exception:  # noqa: BLE001 - no parameter walk, no shared-module ties to fill
        pass
    for target, source in pairs:
        if target.endswith("lm_head.weight"):
            continue
        if target not in state_dict and source in state_dict:
            state_dict[target] = state_dict[source]


def _safetensors_te_class(path: str) -> Optional[str]:
    try:
        import json

        from safetensors import safe_open

        from .prequant_safetensors import UNSLOTH_METADATA_KEY

        with safe_open(path, framework = "pt", device = "cpu") as handle:
            raw = handle.metadata() or {}
        metadata = json.loads(raw.get(UNSLOTH_METADATA_KEY) or "{}")
        value = metadata.get("te_class") if isinstance(metadata, dict) else None
        return str(value) if value else None
    except Exception:  # noqa: BLE001 - unreadable header: read every tensor, trim after
        return None


def _read_safetensors_tensor(path: str, name: str) -> Any:
    from safetensors import safe_open
    with safe_open(path, framework = "pt", device = "cpu") as handle:
        return handle.get_tensor(name)


def _resolve_checkpoint_path(
    source: TePrequantSource,
    hf_token: Optional[str],
    *,
    cache_dir: str,
    local_files_only: bool = False,
    logger: Any = None,
) -> Optional[str]:
    """The local file path for ``source``, downloading from the Hub if needed; None if absent."""
    if source.kind == "path":
        import os
        expanded = os.path.expanduser(source.location)
        return expanded if os.path.isfile(expanded) else None
    if source.kind == "repo":
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

        # Online only a real 404 advances: LocalEntryNotFoundError subclasses EntryNotFoundError
        # and also means an unreachable Hub. Offline, a cache miss is the only verdict.
        miss = (
            (EntryNotFoundError, LocalEntryNotFoundError)
            if local_files_only
            else (EntryNotFoundError,)
        )
        names = [n for n in te_candidate_filenames(source) if te_candidate_is_readable(n)]
        # Mirror for every name before the Hub for any, so the order is the same with or without one.
        for name in names:
            mirrored = te_prequant_mirror_path(source.location, name)
            if mirrored is not None:
                return mirrored
        last: Optional[Exception] = None
        from .diffusion_prequant import _first_mirrored

        # The operator's mirror answers before the Hub is asked for any name (and so also offline).
        mirrored = _first_mirrored(source.location, names, te_candidate_is_readable)
        if mirrored is not None:
            return mirrored
        from .diffusion_prequant import prefer_cached_pickle_twins

        names = prefer_cached_pickle_twins(
            source.location,
            names,
            readable = te_candidate_is_readable,
            cache_dir = cache_dir,
            logger = logger,
            roots = tuple(dict.fromkeys((cache_dir, None))),  # what _download_checkpoint_name reuses
        )
        from .diffusion_prequant import _download_checkpoint_name, explain_container_choice

        for index, name in enumerate(names):
            try:
                path = _download_checkpoint_name(
                    source,
                    name,
                    hf_token,
                    cache_dir,
                    propagate_missing = index < len(names) - 1,
                    local_files_only = local_files_only,
                )
                explain_container_choice(
                    source.location,
                    name,
                    te_candidate_filenames(source),
                    names,
                    readable = te_candidate_is_readable,
                    logger = logger,
                )
                return path
            except LocalEntryNotFoundError:
                # Hub unreachable: take a later name already cached, else re-raise.
                if not local_files_only:
                    unreachable = sys.exc_info()[1]
                    for cached_name in names[names.index(name) + 1 :]:
                        try:
                            # Both cache roots, as for any other name.
                            return _download_checkpoint_name(
                                source,
                                cached_name,
                                hf_token,
                                cache_dir,
                                propagate_missing = False,
                                local_files_only = True,
                            )
                        except (EntryNotFoundError, LocalEntryNotFoundError):
                            continue
                    raise unreachable
                last = sys.exc_info()[1]
                continue
            except miss as exc:
                # Name not in repo: try the next; anything else (auth, corrupt cache) surfaces as itself.
                last = exc
                continue
        if last is not None:
            raise last
    return None


def _validate_checkpoint(ckpt: Any, scheme: str, component: str, base: str, logger: Any) -> bool:
    """Checks format, scheme, component and base only; torch and transformers versions are not enforced."""
    expected_format = TE_PREQUANT_FORMAT_INT8_CONVROT if scheme == "int8" else TE_PREQUANT_FORMAT
    if not isinstance(ckpt, dict) or ckpt.get("format") != expected_format:
        _warn(logger, scheme, ValueError("unrecognised pre-cast text-encoder checkpoint format"))
        return False
    if "state_dict" not in ckpt:
        _warn(logger, scheme, ValueError("pre-cast checkpoint has no state_dict"))
        return False
    meta = ckpt.get("metadata") or {}
    if meta.get("scheme") != scheme:
        _warn(logger, scheme, ValueError(f"checkpoint scheme {meta.get('scheme')!r} != {scheme!r}"))
        return False
    if meta.get("component") != component:
        _warn(
            logger,
            scheme,
            ValueError(f"checkpoint component {meta.get('component')!r} != {component!r}"),
        )
        return False
    ckpt_base = meta.get("base_model_id")
    if base:
        # Builder always records base_model_id; refuse one that omits it (wrong-base keys load).
        if not ckpt_base:
            _warn(
                logger,
                scheme,
                ValueError(
                    f"checkpoint metadata missing base_model_id; refusing for base {base!r}"
                ),
            )
            return False
        if not te_base_equivalent(ckpt_base, base):
            _warn(logger, scheme, ValueError(f"checkpoint base {ckpt_base!r} != {base!r}"))
            return False
    return True


def te_prequant_unmirrored(
    repo_id: Optional[str], files: list[tuple[str, int]]
) -> list[tuple[str, int]]:
    """``files`` minus those a configured mirror serves: the loader reads those in place, never from the Hub."""
    return [(name, size) for name, size in files if te_prequant_mirror_path(repo_id, name) is None]


def te_prequant_hub_files(
    sources: dict[str, "TePrequantSource"],
    api: Any,
    logger: Any = None,
) -> dict[str, list[tuple[str, int]]]:
    """Only components listed here may drop dense weights; a missing artifact keeps its dense encoder."""
    found: dict[str, list[tuple[str, int]]] = {}
    for component, source in sources.items():
        if getattr(source, "kind", None) != "repo" or not getattr(source, "filename", None):
            continue
        mirrored = next(
            (
                (name, te_prequant_mirror_path(source.location, name))
                for name in te_candidate_filenames(source)
                if te_candidate_is_readable(name)
                and te_prequant_mirror_path(source.location, name) is not None
            ),
            None,
        )
        if mirrored is not None:
            import os
            found[component] = [(mirrored[0], os.path.getsize(mirrored[1]))]
            continue
        try:
            from .hub_reachability import always_asked

            # Picks the encoder: always asked, only bounded.
            info = always_asked(api).model_info(source.location, files_metadata = True)
        except Exception as exc:  # noqa: BLE001 -- unavailable pre-cast means the dense encoder
            _warn(logger, f"hub_files:{source.location}", exc)
            continue
        sizes = {s.rfilename: int(getattr(s, "size", 0) or 0) for s in (info.siblings or [])}
        # First candidate the repo holds and this install can open, in resolver order.
        from .diffusion_prequant import prefer_cached_pickle_twins

        ordered = prefer_cached_pickle_twins(
            source.location, te_candidate_filenames(source), readable = te_candidate_is_readable
        )
        for name in ordered:
            if name in sizes and te_candidate_is_readable(name):
                found[component] = [(name, sizes[name])]
                break
    return found


def _has_meta_tensors(module: Any) -> bool:
    """True if any parameter or buffer is still on the meta device after loading."""
    from itertools import chain
    try:
        return any(
            getattr(t, "is_meta", False) for t in chain(module.parameters(), module.buffers())
        )
    except Exception:  # noqa: BLE001
        return False


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.te_prequant: %s failed: %s", what, exc)
