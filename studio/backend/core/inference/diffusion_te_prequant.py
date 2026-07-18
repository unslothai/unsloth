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
from typing import Any, Optional

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

        # The layerwise-fp8 state dict is plain tensors (fp8 storage for cast leaves, the
        # original dtype for skipped modules), so weights_only=True suffices: no pickle code
        # runs even for a local-path artifact. A future torchao-subclass scheme needs a
        # format bump AND weights_only=False behind the same allowlist as the DiT module.
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
        for key, value in (config_overrides or {}).items():
            setattr(config, key, value)
        from accelerate import init_empty_weights

        with init_empty_weights():
            encoder = encoder_cls(config)
        # assign=True swaps in the loaded tensors rather than copying into meta; strict=True
        # since the saved dict is the full state dict of the same class.
        encoder.load_state_dict(state_dict, strict = True, assign = True)
        if _has_meta_tensors(encoder):
            # Non-persistent buffers (built in __init__, absent from the state dict) stay on
            # meta. Rebuild on CPU so they hold real values, then re-assign the cast weights.
            encoder = encoder_cls(config)
            encoder.load_state_dict(state_dict, strict = True, assign = True)
        encoder.eval()

        # Install the SAME upcast hooks the runtime cast applies. The weight cast inside is
        # idempotent (fp8 -> fp8), so this only arms the per-layer upcast; without it the
        # fp8 storage weights would meet bf16 activations at the first forward. A hook
        # failure means the encoder cannot run; fall back to the dense path.
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
    """Component overrides for pipeline assembly: ``{"text_encoder": <pre-cast encoder>}``
    when the requested TE quant is layerwise fp8 and this family hosts a pre-cast
    checkpoint for its primary encoder; ``{}`` otherwise (assembly loads dense as today).

    Gated exactly like the runtime cast (mode normalized, device-supported, family not
    denied), so injection can never engage where ``quantize_text_encoders`` would not.
    The later ``quantize_text_encoders`` call re-applies the cast idempotently and keeps
    status reporting truthful."""
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
        # The per-family TE deny table ships on the video branch's precision module; the
        # image branch has no denials. Resolve lazily so one module serves both.
        denied = getattr(precision, "_te_family_denied", None)
        if callable(denied) and denied(family, mode):
            return {}
        if not te_quant_supported(target, mode):
            return {}
        source = resolve_te_prequant_source(fam, "text_encoder", mode)
        if source is None:
            return {}
        encoder = load_prequant_text_encoder(
            base,
            "text_encoder",
            source,
            dtype = dtype,
            hf_token = hf_token,
            scheme = mode,
            logger = logger,
        )
        if encoder is None:
            return {}
        return {"text_encoder": encoder}
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

        return hf_hub_download(
            repo_id = source.location, filename = source.filename, token = hf_token
        )
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
        # Keys matching a different base can load strict=True and encode prompts with the
        # wrong weights. The builder always records base_model_id; refuse one that omits it.
        if not ckpt_base:
            _warn(
                logger,
                scheme,
                ValueError(
                    f"checkpoint metadata missing base_model_id; refusing for base {base!r}"
                ),
            )
            return False
        if not _same_base_model(ckpt_base, base):
            _warn(logger, scheme, ValueError(f"checkpoint base {ckpt_base!r} != {base!r}"))
            return False
    return True


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
