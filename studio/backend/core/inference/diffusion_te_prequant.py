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
        "text_encoder": ("unsloth/FLUX.2-dev-FP8", "FLUX.2-dev-text_encoder-INT8-ConvRot.safetensors"),
    },
    # Qwen2.5-VL-7B, shared by every Qwen-Image release and HunyuanImage-2.1; the vision tower stays bf16 (edit runs
    # it).
    "qwen-image": {"text_encoder": _QWEN25VL_INT8},
    "qwen-image-edit": {"text_encoder": _QWEN25VL_INT8},
    "qwen-image-layered": {"text_encoder": _QWEN25VL_INT8},
    "hunyuanimage-2.1": {"text_encoder": _QWEN25VL_INT8},
    # Z-Image's own Qwen3-4B, as the bare Qwen3Model the pipeline builds
    "z-image": {
        "text_encoder": ("unsloth/Z-Image-Turbo-FP8", "Z-Image-Turbo-text_encoder-INT8-ConvRot.safetensors"),
    },
    # Krea's Qwen3-VL-4B, as Qwen3VLModel
    "krea-2": {
        "text_encoder": ("unsloth/Krea-2-Turbo-FP8", "Krea-2-Turbo-text_encoder-INT8-ConvRot.safetensors"),
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
        "text_encoder": ("unsloth/ideogram-4-FP8", "ideogram-4-text_encoder-INT8-ConvRot.safetensors"),
    },
    # Gemma-2-2B (Gemma2Model), stored fp32 on the Hub
    "lumina-2": {
        "text_encoder": ("unsloth/Lumina-Image-2.0-FP8", "Lumina-Image-2.0-text_encoder-INT8-ConvRot.safetensors"),
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


# Components the pipeline-assembly injection covers (text_encoder_4 is family-assembled separately, see
# diffusion_hidream.py).
TE_PREQUANT_COMPONENTS = ("text_encoder", "text_encoder_2", "text_encoder_3")

# Fraction of a bf16 text encoder a PRE-CAST fp8 checkpoint occupies, for memory budgeting. fp8 storage is one byte
# per parameter against bf16's two, so the floor is 0.5, but the cast deliberately keeps modules dense: nn.Embedding
# tables, the norms in DEFAULT_SKIP_MODULES_PATTERN, the encoder's own _keep_in_fp32_modules (T5's ``wo``) and an
# lm_head tied to the input embedding. Measured from Hub file metadata over every published artifact (2026-08-07) as
# hosted checkpoint bytes over the bf16-EQUIVALENT dense bytes of the same component, the ratios run 0.514 (FLUX.2-dev
# Mistral-24B) to 0.620 (FLUX.1-schnell T5-XXL); the small encoders sit highest, because their embedding tables are a
# large share of the parameters and stay dense. 0.65 is the observed maximum rounded up, so this OVER-states every
# measured encoder rather than under-stating any: an under-estimate is the expensive direction, since it lets an
# oversized load through to the OS killer.
TE_PREQUANT_BUDGET_SCALE = 0.65
# The int8 ConvRot encoder alone (no fp8 fallback to cover): Qwen-Image-2.1's is 9,349,769,248 bytes against
# 17,534,339,488 for the dense encoder, 0.533, rounded up.
TE_INT8_CONVROT_BUDGET_SCALE = 0.56
# Per family where the default under-states: an encoder whose bf16 embedding table is a large share of it keeps more
# bytes dense. Lumina-2's Gemma-2-2B file is 3,207,382,688 bytes against the 5.23 GB bf16 equivalent of its fp32 Hub
# copy (0.613), Krea-2's Qwen3-VL-4B 5,247,054,376 against 8.88 GB dense (0.591); both rounded up.
_TE_INT8_CONVROT_BUDGET_SCALES = {"lumina-2": 0.62, "krea-2": 0.60}


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
        from .video_ltx25 import ltx25_te_budget_scale
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
    if _int8_convrot_only(te_quant_mode, target):
        return te_int8_convrot_budget_scale(fam)
    return TE_PREQUANT_BUDGET_SCALE


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


# Bases whose text-encoder weights are VERIFIED byte-identical (every shard LFS sha256 compared on 2026-07-18), so one
# hosted artifact serves them all. The validator accepts a base_model_id from the same group; anything else keeps the
# strict refusal.
_TE_EQUIVALENT_BASES: tuple[frozenset[str], ...] = (
    # Qwen2.5-VL-7B text encoder: 4 shards, 16,584,414,544 bytes, identical sha256 set. Qwen-Image-2512 republishes
    # the same four shards (re-verified 2026-08-25); refusing it here would pull 16.6 GB of dense encoder the load
    # never opens.
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
    # Qwen3-VL-4B (Krea's own weights, not FLUX.3's): identical sha256 across the Krea-2 pair (compared 2026-08-25)
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
)


def te_base_equivalent(ckpt_base: str, base: str) -> bool:
    """True when the checkpoint's baked base and the loading base carry byte-identical
    weights for the component: the same repo (``_same_base_model``) or a verified
    equivalence group above."""
    if _same_base_model(ckpt_base, base):
        return True
    # The groups hold UPSTREAM ids: an unnormalised mirror id is a different string, so it would be refused and the
    # pre-cast encoder dropped for a dense pull.
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
    # Names to try after ``filename``, in order, when the repo does not carry it. Only the "repo"
    # kind uses this: a local path either exists or it does not.
    fallback_filenames: tuple = ()
    # The component the hosted FILE was built as, when another family loads the same bytes under a different slot
    # (HiDream's text_encoder_3 is FLUX.1's text_encoder_2). None = the component it is loaded into.
    component: Optional[str] = None


def int8_convrot_te_runs_on(target: Any) -> bool:
    """MPS runs the int8 ConvRot encoder without fp8: plain tensors, a Hadamard matmul and ``F.linear``."""
    return getattr(target, "device", None) == "mps"


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
    """Candidate filenames for ``(component, scheme)``, safetensors first.

    Both extensions are live. The reader handles either, but the NAME has to be asked for, and a
    single hardcoded extension is why a hosted safetensors encoder was unreachable: the resolver
    requested ``.pt``, the Hub returned 404 and the loader fell back to the dense encoder without
    saying anything, which costs a user the whole point of the artifact (17.5 GB instead of 9.4 GB
    on Qwen-Image-2.1). Preference order rather than a registry entry, so a repo that swaps its
    encoder to safetensors is picked up with no code change, and every repo still hosting a
    ``.pt`` (Qwen-Image, LTX-2 and the rest) keeps resolving exactly as before.
    """
    # Imported here, not at module scope: prequant_safetensors pulls torchao in, and this module is
    # imported during pipeline assembly on hosts that may not have it.
    from .prequant_safetensors import SAFETENSORS_SUFFIX

    stem = te_prequant_repo_stem(repo_id, component, scheme)
    return (f"{stem}{SAFETENSORS_SUFFIX}", f"{stem}.pt")


def te_prequant_repo_filename(repo_id: str, component: str, scheme: str) -> str:
    """The preferred checkpoint filename for ``(component, scheme)`` in ``repo_id``."""
    return te_prequant_repo_filenames(repo_id, component, scheme)[0]


def te_candidate_filenames(source: Any) -> tuple:
    """``source``'s names, best first, for anything SHAPED like a source.

    One accessor so the download PLAN and the resolver cannot disagree about which artifact a
    source means. They did: the resolver learned the chain while every consumer kept matching
    ``filename`` alone, so the moment the preferred name became a safetensors spelling no repo
    hosting a ``.pt`` was recognised by the plan, its dense encoder went back into the pull, and
    the loader fetched the ``.pt`` on top of it. Planners also pass lightweight stand-ins, so this
    reads defensively rather than touching the dataclass.
    """
    names = (
        getattr(source, "filename", None),
        *(getattr(source, "fallback_filenames", None) or ()),
    )
    return tuple(n for n in names if n)


def te_candidate_is_readable(name: Optional[str]) -> bool:
    """Whether this install can open a pre-cast encoder artifact called ``name``.

    NOT the transformer's ``restricted_prequant_load_supported``: this state dict is plain
    tensors, read under a bare ``weights_only`` load with no constructor allowlist, so a ``.pt``
    is always readable and asking the DiT's question would refuse one on every install whose
    torchao lacks some DiT scheme's constructors. The safetensors container needs safetensors, not torchao.
    """
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
    """Resolve where the pre-cast checkpoint for ``(fam, component, scheme)`` comes from.

    Priority: (1) explicit local ``path_override``; (2) the family's hosted repo entry;
    (3) None -> no pre-cast artifact, caller downloads dense and casts. Pure: no IO."""
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

        if not family_trims_lm_head(getattr(fam, "name", None)):
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
    """``{component: source}`` for every text encoder this pick would load PRE-CAST rather
    than dense; ``{}`` when none apply.

    ``components`` defaults to the generic pipeline injection set; callers that assemble an
    additional component separately may request it explicitly.

    Pure (no IO, no ``torch.load``) and gated exactly like ``te_prequant_pipe_kwargs``
    below, which calls it. Download planning uses the same resolver so a plan can never
    disagree with the load about which dense encoders are still needed -- staging the dense
    encoder for a pre-cast load wastes tens of GB (LTX's Gemma3 is ~49 GB), and dropping one
    the load actually wants costs a surprise mid-load pull."""
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
        # The per-family TE deny table ships on the video branch precision module (the image branch has no denials), so
        # resolve it lazily and one module serves both.
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
                # An automatic int8 pick takes the int8 file or the dense encoder, never the fp8 file behind it.
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
    """
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
        if checkpoint_base and selected_base and te_base_equivalent(checkpoint_base, selected_base):
            compatible[component] = source
    return compatible


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
    (``diffusion_text_encoder_trim``). ``failures`` collects the error text of a load that raised."""
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

        # The layerwise-fp8 state dict is plain tensors, so weights_only=True suffices and no pickle code runs even for
        # a local path. A future torchao-subclass scheme needs a format bump AND the DiT module's allowlist.
        # A ``.safetensors`` artifact is read through the plain-tensor reader instead, which returns the same dict shape, so
        # ``_validate_checkpoint`` and everything after it are unchanged. Dispatch is on the extension the resolver
        # asked the Hub for, never on sniffing the bytes.
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
        config = transformers.AutoConfig.from_pretrained(base, **config_kwargs)
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
        encoder.load_state_dict(state_dict, strict = True, assign = True)
        if _has_meta_tensors(encoder):
            # Non-persistent buffers (built in __init__, absent from the state dict) stay on meta. Rebuild on CPU so
            # they hold real values, then re-assign the cast weights.
            encoder = encoder_cls(config)
            if trim:
                trim_text_encoder(encoder)
            encoder.load_state_dict(state_dict, strict = True, assign = True)
        # assign=True swaps in SEPARATE tensors for tied weights (the saved dict carries a copy per key), untying e.g.
        # Qwen3's lm_head from embed_tokens and defeating _cast_fp8's tied-projection skip. Re-tie to the
        # builder-identical structure; a no-op when untied.
        tie = getattr(encoder, "tie_weights", None)
        if callable(tie):
            tie()
        encoder.eval()

        # Install the SAME upcast hooks the runtime cast applies. The weight cast inside is idempotent, so this only
        # arms the per-layer upcast; without it the fp8 storage weights would meet bf16 activations at the first
        # forward.
        from .diffusion_precision import _cast_fp8

        class _Target:
            pass

        target = _Target()
        target.dtype = dtype
        _cast_fp8(encoder, target)
        setattr(encoder, TE_PREQUANT_SCHEME_ATTR, "fp8")
        if logger is not None:
            logger.info(
                "diffusion.te_prequant: loaded %s %s checkpoint (%s)",
                component,
                scheme,
                source.kind,
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
) -> dict[str, Any]:
    """Component overrides for pipeline assembly: ``{<component>: <pre-cast encoder>}``
    for every ``TE_PREQUANT_COMPONENTS`` attr the family hosts a pre-cast checkpoint for
    (e.g. flux.1 hosts its T5-XXL as ``text_encoder_2``); ``{}`` when none resolve
    (assembly loads dense as today).

    Gated exactly like the runtime cast (mode normalized, device-supported, family not
    denied), so injection can never engage where ``quantize_text_encoders`` would not.
    The later ``quantize_text_encoders`` call re-applies the cast idempotently and keeps
    status reporting truthful.

    ``dense_source`` is the local directory or repo id assembly reads without fetching (None when it can
    still fetch). A pre-cast encoder that fails to load with no dense weights there raises: the plan left
    those shards out of the prefetch, so assembly would die on a missing shard instead."""
    unavailable: list[str] = []
    unbudgeted: list[str] = []
    failures: list = []
    cause = ""
    try:
        from .diffusion_precision import TE_QUANT_FP8, normalize_te_quant, te_quant_supported
        from .diffusion_text_encoder_trim import family_trims_lm_head
        from .media_locality import _hosted_component_cached

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
                )
            if encoder is not None:
                injected[component] = encoder
            elif int8_convrot_te_runs_on(target):
                # The MPS plan priced this checkpoint, not the dense shards beside it.
                unbudgeted.append(component)
                cause = f" ({failures[-1]})" if failures else cause
            elif dense_source and not _hosted_component_cached(dense_source, component, "", ""):
                unavailable.append(component)
                cause = f" ({failures[-1]})" if failures else cause
        if not unavailable and not unbudgeted:
            return injected
        if _picked_automatically(te_quant_mode) and _fetch_dense_components(
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
    source: str, components: list, *, hf_token: Optional[str], local_files_only: bool, logger: Any = None
) -> bool:
    """Bring the dense shards of ``components`` the plan left out back into ``source`` (a Hub cache snapshot or repo
    id) so assembly can load them. True when there is nothing to fetch or the fetch worked; False offline, for a local
    folder that is not a cache snapshot, or on any failure."""
    if not components:
        return True
    try:
        from .media_locality import _hosted_component_cached

        if all(_hosted_component_cached(source, c, "", "") for c in components):
            return True
    except Exception:  # noqa: BLE001 - unknown: try the fetch
        pass
    if local_files_only:
        return False
    try:
        import os

        from huggingface_hub import snapshot_download

        from utils.hf_cache_settings import active_hf_hub_cache

        repo_id = source
        if os.path.isdir(str(source)):
            parts = os.path.normpath(str(source)).split(os.sep)
            if len(parts) < 3 or parts[-2] != "snapshots" or not parts[-3].startswith("models--"):
                return False
            repo_id = parts[-3][len("models--") :].replace("--", "/")
        snapshot_download(
            repo_id,
            allow_patterns = [f"{c}/*" for c in components],
            token = hf_token,
            cache_dir = active_hf_hub_cache(),
        )
        return True
    except Exception as exc:  # noqa: BLE001 - the strict refusal below still explains the load
        _warn(logger, "dense_fallback_fetch", exc)
        return False


def _held_locally(repo_id: str, name: Optional[str], hf_token: Optional[str]) -> bool:
    """Whether ``repo_id/name`` is on this machine (mirror or Hub cache), without touching the network."""
    try:
        if te_prequant_mirror_path(repo_id, name) is not None:
            return True
        from huggingface_hub import hf_hub_download

        from utils.hf_cache_settings import active_hf_hub_cache

        hf_hub_download(
            repo_id = repo_id,
            filename = name,
            token = hf_token,
            cache_dir = active_hf_hub_cache(),
            local_files_only = True,
        )
        return True
    except Exception:  # noqa: BLE001 - not cached, or unanswerable
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
    if getattr(encoder, "lm_head", None) is not None and not trim:
        # The file has no lm_head; a build that keeps one would load it random. A headless encoder class (T5,
        # Gemma2Model, Qwen3Model, Qwen3VLModel) has nothing to drop.
        raise ValueError("the int8 ConvRot encoder carries no lm_head and needs the lm_head trim")
    if trim:
        trim_text_encoder(encoder)
    if getattr(encoder, "lm_head", None) is not None and not is_trimmed(encoder) and config_ties_lm_head(config):
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
    # Storage dtypes stay (int8 payload, float32 scales); every dense float tensor follows the compute dtype.
    for key, tensor in list(state_dict.items()):
        if (
            tensor.is_floating_point()
            and not key.endswith(scale_suffix)
            and dtype is not None
            and tensor.dtype != dtype
        ):
            state_dict[key] = tensor.to(dtype)
    # A dense decoder projection would load as bf16 and outgrow the budget this path is priced at.
    dense = [
        n
        for n, m in encoder.named_modules()
        if isinstance(m, torch.nn.Linear) and TE_INT8_CONVROT_LINEAR.search(n)
    ]
    if dense:
        raise ValueError(
            f"{len(dense)} decoder projection(s) are not quantized in this artifact, e.g. {sorted(dense)[0]}"
        )
    encoder.load_state_dict(state_dict, strict = True, assign = True)
    # assign=True untied a tied embedding pair (T5's shared / encoder.embed_tokens); re-tie it, as the fp8 path does.
    tie = getattr(encoder, "tie_weights", None)
    if callable(tie):
        tie()
    stranded = _meta_tensor_names(encoder)
    if stranded:
        raise ValueError(f"{len(stranded)} tensor(s) still on the meta device, e.g. {stranded[0]}")
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


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

        # Which exception means "this NAME is absent" depends on the mode, and the two are not
        # interchangeable. huggingface_hub documents LocalEntryNotFoundError as "not on the disk
        # when network is disabled OR UNAVAILABLE (connection issue). The entry may exist on the
        # Hub", and it SUBCLASSES EntryNotFoundError, so catching the base online would swallow an
        # unreachable Hub, spend a second full attempt on the next name, and report that one's
        # error instead of the connection failure that actually happened. Online, only a real 404
        # advances; offline, a cache miss is the only verdict there is.
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
                # Online, the Hub is unreachable (not a missing name): take a later name already cached (fp8
                # cached before the int8 file was published), else re-raise rather than blame the next one.
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
                # This name is not in this repo. Try the next extension rather than giving up:
                # only "no candidate exists" is a real miss, and anything else (auth, a corrupt
                # cache) must still surface as itself.
                last = exc
                continue
        if last is not None:
            raise last
    return None


def _validate_checkpoint(ckpt: Any, scheme: str, component: str, base: str, logger: Any) -> bool:
    """Reject a checkpoint that is the wrong format / scheme / component / base model.

    ``te_class`` presence is checked by the caller (it resolves the class); torch /
    transformers versions are recorded by the builder for forensics but not enforced (the
    fp8 storage cast is version-stable plain-tensor data)."""
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
        # Keys matching a different base can load strict=True and encode prompts with the wrong weights. The builder
        # always records base_model_id, so refuse one that omits it.
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
    """``{component: [(rfilename, size)]}`` for every hosted pre-cast checkpoint that really
    resolves on the Hub.

    Only a component listed here may have its dense weights dropped from a plan or a prefetch:
    an unpublished / gated / renamed artifact keeps its dense encoder, exactly as the load's own
    fallback does. Checked per source so one missing repo cannot sink the whole plan. A local
    path override is already on disk and is never staged."""
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
            info = api.model_info(source.location, files_metadata = True)
        except Exception as exc:  # noqa: BLE001 -- unavailable pre-cast means the dense encoder
            _warn(logger, f"hub_files:{source.location}", exc)
            continue
        sizes = {s.rfilename: int(getattr(s, "size", 0) or 0) for s in (info.siblings or [])}
        # The first candidate the repo HOLDS and this install can OPEN, in the resolver's own
        # order, so the bytes counted here are the bytes that will actually be fetched. Matching
        # the primary name alone reported every .pt repo as having no pre-cast encoder at all the
        # moment safetensors became the preferred spelling.
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
