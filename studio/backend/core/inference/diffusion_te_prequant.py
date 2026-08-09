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

from dataclasses import dataclass
from typing import Any, Iterable, Optional

# Reuse the DiT module's operator allowlist for local paths: one env var, one policy.
from .diffusion_prequant import (
    ALLOW_LOCAL_PREQUANT_PATH_ENV,
    _local_prequant_path_allowed,
    _same_base_model,
)

# torch.save dict layout tag; bump on an on-disk change so old/foreign artifacts are rejected.
TE_PREQUANT_FORMAT = "unsloth_prequant_text_encoder_state_dict_v1"

# The one scheme hosted in v1 (see module docstring).
TE_PREQUANT_SCHEMES = ("fp8",)

# Components the pipeline-assembly injection covers (text_encoder_4 is family-assembled separately, see diffusion_hidream.py).
TE_PREQUANT_COMPONENTS = ("text_encoder", "text_encoder_2", "text_encoder_3")

# Fraction of a bf16 text encoder a PRE-CAST fp8 checkpoint occupies, for memory budgeting.
#
# fp8 storage is one byte per parameter against bf16's two, so the floor is 0.5, but the cast
# deliberately keeps modules dense: nn.Embedding tables, the norms in
# DEFAULT_SKIP_MODULES_PATTERN, the encoder's own _keep_in_fp32_modules (T5's ``wo``) and an
# lm_head tied to the input embedding. Measured from Hub file metadata over every published
# artifact (2026-08-07), as hosted checkpoint bytes over the bf16-EQUIVALENT dense bytes of the
# same component (an fp32-stored encoder is halved first, since the pipeline loads it bf16):
#
#   FLUX.2-dev       text_encoder    Mistral-24B    24,683,130,873 / 48,022,800,560 = 0.514
#   HiDream-I1-Full  text_encoder_4  Llama-3.1-8B    8,555,963,320 / 16,060,556,376 = 0.533
#   Qwen-Image       text_encoder    Qwen2.5-VL-7B   8,839,210,073 / 16,584,414,544 = 0.533
#   LTX-2            text_encoder    Gemma3-12B     13,205,302,695 / 24,374,720,836 = 0.542
#   Krea-2-Turbo     text_encoder    Qwen3-4B        4,831,262,424 /  8,875,715,136 = 0.544
#   Z-Image-Turbo    text_encoder    Qwen3-4B        4,411,751,967 /  8,044,982,000 = 0.548
#   Lumina-Image-2.0 text_encoder    Gemma2-2B       3,204,501,909 /  5,228,699,608 = 0.613
#   FLUX.1-schnell   text_encoder_2  T5-XXL          5,900,818,800 /  9,524,648,584 = 0.620
#
# The small encoders sit highest: their embedding tables are a large share of the parameters and
# stay dense. 0.65 is the observed maximum rounded up, so this OVER-states every measured
# encoder rather than under-stating any. It is a memory budget, and an under-estimate is the
# expensive direction: it lets an oversized load through to the OS killer.
TE_PREQUANT_BUDGET_SCALE = 0.65


def te_prequant_budget_scale(fam: Any, *, te_quant_mode: Optional[str], target: Any) -> float:
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
    try:
        sources = te_prequant_sources(fam, te_quant_mode = te_quant_mode, target = target)
    except Exception:  # noqa: BLE001 -- an unresolvable pre-cast just means the dense encoder
        return 1.0
    return TE_PREQUANT_BUDGET_SCALE if sources else 1.0

# Bases whose text-encoder weights are VERIFIED byte-identical (every shard LFS sha256 compared on 2026-07-18), so one hosted artifact serves them all. The validator accepts a base_model_id from the same group; anything else keeps the strict refusal.
_TE_EQUIVALENT_BASES: tuple[frozenset[str], ...] = (
    # Qwen2.5-VL-7B text encoder: 4 shards, 16,584,414,544 bytes, identical sha256 set.
    frozenset(
        {
            "qwen/qwen-image",
            "hunyuanvideo-community/hunyuanimage-2.1-diffusers",
        }
    ),
    # T5-XXL (text_encoder_2): 2 shards, 9,524,648,584 bytes, identical sha256 across every FLUX.1 release; HiDream-I1 ships the same bytes as text_encoder_3 (cross-component mapping is not wired yet).
    frozenset(
        {
            "black-forest-labs/flux.1-schnell",
            "black-forest-labs/flux.1-dev",
            "black-forest-labs/flux.1-krea-dev",
            "hidream-ai/hidream-i1-full",
        }
    ),
)


def te_base_equivalent(ckpt_base: str, base: str) -> bool:
    """True when the checkpoint's baked base and the loading base carry byte-identical
    weights for the component: the same repo (``_same_base_model``) or a verified
    equivalence group above."""
    if _same_base_model(ckpt_base, base):
        return True
    # The groups hold UPSTREAM ids: an unnormalised mirror id is a different string, so it would be
    # refused and the pre-cast encoder dropped for a dense pull.
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


def te_prequant_repo_filename(repo_id: str, component: str, scheme: str) -> str:
    """The checkpoint filename for ``(component, scheme)`` in ``repo_id``: hosted repos are
    named <Model>-FP8 (or -INT8 / -quantized) and carry <Model>-<component>-<SCHEME>.pt
    files, e.g. unsloth/LTX-2-FP8 -> LTX-2-text_encoder-FP8.pt."""
    model = repo_id.rsplit("/", 1)[-1]
    for suffix in ("-fp8", "-int8", "-quantized"):
        if model.lower().endswith(suffix):
            model = model[: -len(suffix)]
            break
    return f"{model}-{component}-{scheme.upper()}.pt"


def family_te_prequant_repo(fam: Any, scheme: str, component: str) -> Optional[str]:
    """The hosted pre-cast encoder repo for ``(scheme, component)`` in this family, or None.

    Reads the family's ``te_prequant_repos`` (scheme, component, repo_id) triples; the field
    is optional on both DiffusionFamily and VideoFamily, so one resolver serves both loaders.
    """
    for entry in getattr(fam, "te_prequant_repos", ()) or ():
        try:
            entry_scheme, entry_component, repo_id = entry
        except Exception:  # noqa: BLE001 — a malformed entry must not break the load
            continue
        if entry_scheme == scheme and entry_component == component:
            return repo_id
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
    repo_id = family_te_prequant_repo(fam, scheme, component)
    if repo_id:
        return TePrequantSource(
            kind = "repo",
            location = repo_id,
            filename = te_prequant_repo_filename(repo_id, component, scheme),
        )
    return None


def te_prequant_sources(
    fam: Any, *, te_quant_mode: Optional[str], target: Any
) -> dict[str, TePrequantSource]:
    """``{component: source}`` for every text encoder this pick would load PRE-CAST rather
    than dense; ``{}`` when none apply.

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

        mode = normalize_te_quant(te_quant_mode)
        if mode != TE_QUANT_FP8:
            return {}
        family = getattr(fam, "name", None)
        # The per-family TE deny table ships on the video branch precision module (the image branch has no denials), so resolve it lazily and one module serves both.
        denied = getattr(precision, "_te_family_denied", None)
        if callable(denied) and denied(family, mode):
            return {}
        if not te_quant_supported(target, mode):
            return {}
        sources: dict[str, TePrequantSource] = {}
        for component in TE_PREQUANT_COMPONENTS:
            source = resolve_te_prequant_source(fam, component, mode)
            if source is not None:
                sources[component] = source
        return sources
    except Exception:  # noqa: BLE001 — an unresolvable pre-cast just means the dense encoder
        return {}


# Weight files a dense encoder folder holds. Everything else (config.json, the shard index, tokenizer JSON) is kept when the pre-cast checkpoint replaces the weights: the pre-cast loader still meta-inits from the base repo component config.
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
    flags only; the state dict is unaffected by them)."""
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

        path = _resolve_checkpoint_path(source, hf_token)
        if path is None:
            return None

        import torch

        # The layerwise-fp8 state dict is plain tensors, so weights_only=True suffices and no pickle code runs even for a local path. A future torchao-subclass scheme needs a format bump AND the DiT module's allowlist.
        ckpt = torch.load(path, weights_only = True, map_location = "cpu")
        if not _validate_checkpoint(ckpt, scheme, component, base, logger):
            return None
        state_dict = ckpt["state_dict"]
        te_class = (ckpt.get("metadata") or {}).get("te_class")

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
        config_kwargs: dict[str, Any] = {"token": hf_token}
        if subfolder:
            config_kwargs["subfolder"] = subfolder
        config = transformers.AutoConfig.from_pretrained(base, **config_kwargs)
        # Krea-2 ships transformers-5.x configs whose rope lives under rope_parameters; the runtime component loader remaps it for a 4.x runtime, and the meta-init here must match or the rebuilt encoder forwards with a broken rope.
        from .diffusion_krea2 import remap_rope_parameters

        remap_rope_parameters(getattr(config, "text_config", config))
        for key, value in (config_overrides or {}).items():
            setattr(config, key, value)
        from accelerate import init_empty_weights

        with init_empty_weights():
            encoder = encoder_cls(config)
        # assign=True swaps in the loaded tensors rather than copying into meta; strict=True since the saved dict is the full state dict of the same class.
        encoder.load_state_dict(state_dict, strict = True, assign = True)
        if _has_meta_tensors(encoder):
            # Non-persistent buffers (built in __init__, absent from the state dict) stay on meta. Rebuild on CPU so they hold real values, then re-assign the cast weights.
            encoder = encoder_cls(config)
            encoder.load_state_dict(state_dict, strict = True, assign = True)
        # assign=True swaps in SEPARATE tensors for tied weights (the saved dict carries a copy per key), untying e.g. Qwen3's lm_head from embed_tokens and defeating _cast_fp8's tied-projection skip. Re-tie to the builder-identical structure; a no-op when untied.
        tie = getattr(encoder, "tie_weights", None)
        if callable(tie):
            tie()
        encoder.eval()

        # Install the SAME upcast hooks the runtime cast applies. The weight cast inside is idempotent, so this only arms the per-layer upcast; without it the fp8 storage weights would meet bf16 activations at the first forward.
        from .diffusion_precision import _cast_fp8

        class _Target:
            pass

        target = _Target()
        target.dtype = dtype
        _cast_fp8(encoder, target)
        if logger is not None:
            logger.info(
                "diffusion.te_prequant: loaded %s %s checkpoint (%s)",
                component,
                scheme,
                source.kind,
            )
        return encoder
    except Exception as exc:  # noqa: BLE001 — fall back to the dense download + cast
        _warn(logger, f"{scheme}:{component}:{source.kind}", exc)
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
) -> dict[str, Any]:
    """Component overrides for pipeline assembly: ``{<component>: <pre-cast encoder>}``
    for every ``TE_PREQUANT_COMPONENTS`` attr the family hosts a pre-cast checkpoint for
    (e.g. flux.1 hosts its T5-XXL as ``text_encoder_2``); ``{}`` when none resolve
    (assembly loads dense as today).

    Gated exactly like the runtime cast (mode normalized, device-supported, family not
    denied), so injection can never engage where ``quantize_text_encoders`` would not.
    The later ``quantize_text_encoders`` call re-applies the cast idempotently and keeps
    status reporting truthful."""
    try:
        from .diffusion_precision import TE_QUANT_FP8

        sources = te_prequant_sources(fam, te_quant_mode = te_quant_mode, target = target)
        # Non-empty only for the one hosted scheme (see te_prequant_sources' gate).
        mode = TE_QUANT_FP8
        injected: dict[str, Any] = {}
        for component, source in sources.items():
            encoder = load_prequant_text_encoder(
                base,
                component,
                source,
                dtype = dtype,
                hf_token = hf_token,
                scheme = mode,
                logger = logger,
            )
            if encoder is not None:
                injected[component] = encoder
        return injected
    except Exception as exc:  # noqa: BLE001 — injection is an optimisation, never a blocker
        _warn(logger, "pipe_kwargs", exc)
        return {}


def _resolve_checkpoint_path(source: TePrequantSource, hf_token: Optional[str]) -> Optional[str]:
    """The local file path for ``source``, downloading from the Hub if needed; None if absent."""
    if source.kind == "path":
        import os
        expanded = os.path.expanduser(source.location)
        return expanded if os.path.isfile(expanded) else None
    if source.kind == "repo":
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id = source.location, filename = source.filename, token = hf_token)
    return None


def _validate_checkpoint(ckpt: Any, scheme: str, component: str, base: str, logger: Any) -> bool:
    """Reject a checkpoint that is the wrong format / scheme / component / base model.

    ``te_class`` presence is checked by the caller (it resolves the class); torch /
    transformers versions are recorded by the builder for forensics but not enforced (the
    fp8 storage cast is version-stable plain-tensor data)."""
    if not isinstance(ckpt, dict) or ckpt.get("format") != TE_PREQUANT_FORMAT:
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
        # Keys matching a different base can load strict=True and encode prompts with the wrong weights. The builder always records base_model_id, so refuse one that omits it.
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
        try:
            info = api.model_info(source.location, files_metadata = True)
        except Exception as exc:  # noqa: BLE001 -- unavailable pre-cast means the dense encoder
            _warn(logger, f"hub_files:{source.location}", exc)
            continue
        files = [
            (s.rfilename, int(getattr(s, "size", 0) or 0))
            for s in (info.siblings or [])
            if s.rfilename == source.filename
        ]
        if files:
            found[component] = files
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
