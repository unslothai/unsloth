# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load a *pre-quantized* transformer instead of quantising a dense one on the GPU.

The runtime transformer_quant path loads the dense bf16 transformer and ``quantize_``s it
in place, materialising the full bf16 weights on the GPU first (~2x the GGUF peak, plus the
full bf16 download). When a transformer was already quantised and saved
(``scripts/build_prequant_checkpoint.py``), this loads those weights directly: build the
skeleton on ``meta`` (``init_empty_weights`` + ``from_config``), ``load_state_dict
(assign=True)`` the quantized state dict (subclass tensors assigned, not copied, so dense
bf16 never touches the GPU), then move to device.

Measured (B200, Z-Image fp8): GPU load peak 12.9 -> 6.3 GB, download 12 -> 6.28 GB, output
bit-identical (LPIPS 0.0). The checkpoint carries the same scheme + ``min_features`` as the
runtime path, so the result matches quantising on the fly.

Best-effort and lazily imported: a missing / mismatched / unreadable checkpoint returns None
and the caller falls back to dense-quantise (then GGUF). Inert with nothing configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# torch.save dict layout tag; bump on an on-disk change so old/foreign artifacts are rejected.
PREQUANT_FORMAT = "unsloth_prequant_transformer_state_dict_v1"

# Loading ends in ``torch.load(weights_only=False)``, which executes pickle code. A hosted
# repo checkpoint is first-party; a ``kind == "path"`` can come from a request's
# ``transformer_prequant_path``, so unpickling it is RCE. A request-supplied path is
# unpickled ONLY when it resolves inside an operator-configured ALLOWLIST of directories; a
# bare on/off toggle is never a wildcard. The hosted-repo path is unaffected.
ALLOW_LOCAL_PREQUANT_PATH_ENV = "UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH"

_PREQUANT_TOGGLE_TOKENS = {"1", "true", "yes", "on", "0", "false", "no", "off"}


def _allowed_prequant_roots() -> list:
    """Operator-allowlisted directories whose pre-quant checkpoints may be unpickled.

    ``UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH`` = one or more dirs (``os.pathsep``-separated). A
    bare truthy/falsey toggle is ignored: it must name a directory, so no "allow all" mode."""
    import os

    raw = (os.environ.get(ALLOW_LOCAL_PREQUANT_PATH_ENV) or "").strip()
    if not raw:
        return []
    roots = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part or part.lower() in _PREQUANT_TOGGLE_TOKENS:
            continue  # a bare on/off value is not a directory
        try:
            roots.append(os.path.realpath(os.path.expanduser(part)))
        except Exception:  # noqa: BLE001 — a bad entry is simply not allowlisted
            continue
    return roots


def _local_prequant_path_allowed(path: str) -> bool:
    """True only when ``path`` resolves inside an allowlisted directory. ``realpath`` first
    so a symlink cannot point an allowlisted name at a file outside the allowed roots."""
    import os

    roots = _allowed_prequant_roots()
    if not roots:
        return False
    try:
        real = os.path.realpath(os.path.expanduser(path))
    except Exception:  # noqa: BLE001
        return False
    return any(real == r or real.startswith(r + os.sep) for r in roots)


def local_prequant_path_ready(path: str) -> bool:
    """True only when a local pre-quant path would actually load: inside an allowlisted root
    AND the file is present. The auto-policy planner checks this before budgeting the small
    prequant plan, so it never skips the dense shards for a path the loader will refuse
    (which would evict the resident pipeline then rebuild dense under an undersized plan ->
    OOM)."""
    import os

    if not _local_prequant_path_allowed(path):
        return False
    return os.path.isfile(os.path.expanduser(path))


@dataclass(frozen = True)
class PrequantSource:
    """Where a pre-quantized checkpoint lives. ``kind`` is "path" (a local file) or "repo"
    (Hub repo id in ``location`` + ``filename``; ``fallback_filename`` is tried when the
    primary name is absent, covering repos still on the legacy transformer_<scheme>.pt)."""

    kind: str
    location: str
    filename: Optional[str] = None
    fallback_filename: Optional[str] = None


def prequant_filename(scheme: str) -> str:
    """The legacy checkpoint filename for ``scheme`` inside a Hub repo."""
    return f"transformer_{scheme}.pt"


def prequant_repo_filename(repo_id: str, scheme: str) -> str:
    """The model-name checkpoint filename for ``scheme`` in ``repo_id``: the hosted repos are
    named <Model>-FP8 (or -INT8 / -quantized) and carry <Model>-<SCHEME>.pt files, e.g.
    unsloth/Z-Image-Turbo-FP8 -> Z-Image-Turbo-INT8.pt / Z-Image-Turbo-FP8.pt."""
    model = repo_id.rsplit("/", 1)[-1]
    for suffix in ("-fp8", "-int8", "-quantized"):
        if model.lower().endswith(suffix):
            model = model[: -len(suffix)]
            break
    return f"{model}-{scheme.upper()}.pt"


def resolve_prequant_source(
    fam: Any,
    scheme: str,
    *,
    path_override: Optional[str] = None,
) -> Optional[PrequantSource]:
    """Resolve where the checkpoint for ``(fam, scheme)`` comes from.

    Priority: (1) explicit local ``path_override``; (2) the family's hosted repo for
    ``scheme``; (3) None -> no pre-quant, caller quantises dense. Pure: no IO, no torch.
    """
    override = (path_override or "").strip()
    if override:
        return PrequantSource(kind = "path", location = override, filename = None)
    try:
        from .diffusion_families import family_prequant_repo
        repo_id = family_prequant_repo(fam, scheme)
    except Exception:  # noqa: BLE001 — a bad family object must not break the load
        repo_id = None
    if repo_id:
        return PrequantSource(
            kind = "repo",
            location = repo_id,
            filename = prequant_repo_filename(repo_id, scheme),
            fallback_filename = prequant_filename(scheme),
        )
    return None


def usable_prequant_source(
    fam: Any,
    scheme: str,
    *,
    path_override: Optional[str] = None,
) -> Optional[PrequantSource]:
    """``resolve_prequant_source``, but a local path counts only when the loader would
    accept it: inside the allowlist AND present on disk. Otherwise resolves to None so
    memory planning falls back to dense-fit checks up front, instead of the loader refusing
    the path only after the resident pipeline was evicted and dense bf16 materialises under
    a plan that never budgeted for it (evict-then-OOM). Hosted-repo sources are unaffected."""
    src = resolve_prequant_source(fam, scheme, path_override = path_override)
    if src is not None and src.kind == "path" and not local_prequant_path_ready(src.location):
        return None
    return src


def load_prequantized_transformer(
    transformer_cls: Any,
    base: str,
    source: PrequantSource,
    *,
    device: str,
    dtype: Any,
    hf_token: Optional[str] = None,
    scheme: str,
    min_features: Optional[int] = None,
    fast_accum: Optional[bool] = None,
    logger: Any = None,
) -> Optional[Any]:
    """Load the pre-quantized transformer described by ``source`` onto ``device``.

    Returns the placed transformer, or None on any problem (missing / mismatched /
    unreadable checkpoint, or unsupported meta-init) so the caller falls back to
    dense-quantise. Best-effort: never raises for an unavailable artifact.
    """
    try:
        # weights_only=False executes pickle code, so a local path is unpickled ONLY when
        # allowlisted. The hosted family repo is first-party and always allowed.
        if source.kind == "path" and not _local_prequant_path_allowed(source.location):
            _warn(
                logger,
                f"{scheme}:path",
                RuntimeError(
                    "request-supplied local pre-quant path refused (unpickling an arbitrary "
                    f"file is unsafe); set {ALLOW_LOCAL_PREQUANT_PATH_ENV} to an allowlisted "
                    "directory containing trusted checkpoints to permit it",
                ),
            )
            return None

        path = _resolve_checkpoint_path(source, hf_token)
        if path is None:
            return None

        import torch

        # torchao weight subclasses aren't safetensors-serializable, so the checkpoint is a
        # torch.save pickle; weights_only=False rebuilds those subclasses. Local path gated
        # above; repo branch is first-party.
        ckpt = torch.load(path, weights_only = False, map_location = "cpu")
        if not _validate_checkpoint(
            ckpt, scheme, base, logger, min_features = min_features, fast_accum = fast_accum
        ):
            return None
        state_dict = ckpt["state_dict"]

        config = transformer_cls.load_config(base, subfolder = "transformer", token = hf_token)
        from accelerate import init_empty_weights

        with init_empty_weights():
            transformer = transformer_cls.from_config(config)
        # assign=True swaps in the loaded tensors rather than copying into meta (a copy into
        # meta is a no-op); strict=True since the saved dict is the full state dict of the
        # same class.
        transformer.load_state_dict(state_dict, strict = True, assign = True)
        if _has_meta_tensors(transformer):
            # Non-persistent buffers (built in __init__, absent from the state dict) stay on
            # meta. Rebuild on CPU so they hold real values, then re-assign the quantized
            # weights; dense bf16 lives in CPU RAM only, the GPU gets just the quant footprint.
            transformer = transformer_cls.from_config(config)
            transformer.load_state_dict(state_dict, strict = True, assign = True)

        transformer = transformer.to(device)
        # from_config starts in TRAIN mode; the dense/GGUF paths use from_pretrained, which
        # returns an eval()'d module. Match that so train/eval-sensitive layers (e.g.
        # dropout) can't make prequant inference diverge from the other paths.
        try:
            transformer.eval()
        except Exception:  # noqa: BLE001 — eval() is best-effort
            pass
        try:  # diagnostic marker, mirrors the runtime-quant path
            transformer._unsloth_runtime_quant = scheme
        except Exception:  # noqa: BLE001 — marker is best-effort
            pass
        if logger is not None:
            logger.info(
                "diffusion.prequant: loaded %s checkpoint (%s) onto %s",
                scheme,
                source.kind,
                device,
            )
        return transformer
    except Exception as exc:  # noqa: BLE001 — fall back to the dense-quantise path
        _warn(logger, f"{scheme}:{source.kind}", exc)
        return None


def _resolve_checkpoint_path(source: PrequantSource, hf_token: Optional[str]) -> Optional[str]:
    """The local file path for ``source``, downloading from the Hub if needed; None if absent."""
    if source.kind == "path":
        import os

        # Expand ~ (the allowlist gate already did), else os.path.isfile sees a literal "~".
        expanded = os.path.expanduser(source.location)
        return expanded if os.path.isfile(expanded) else None
    if source.kind == "repo":
        from huggingface_hub import hf_hub_download
        try:
            from huggingface_hub.errors import EntryNotFoundError
        except Exception:  # noqa: BLE001 — older hub layouts; fall back to a private marker
            class EntryNotFoundError(Exception):  # type: ignore[no-redef]
                pass
        try:
            return hf_hub_download(repo_id = source.location, filename = source.filename, token = hf_token)
        except EntryNotFoundError:
            if not source.fallback_filename or source.fallback_filename == source.filename:
                raise
            return hf_hub_download(
                repo_id = source.location, filename = source.fallback_filename, token = hf_token
            )
    return None


def _validate_checkpoint(
    ckpt: Any,
    scheme: str,
    base: str,
    logger: Any,
    min_features: Optional[int] = None,
    fast_accum: Optional[bool] = None,
) -> bool:
    """Reject a checkpoint that is the wrong format / scheme / base model / filter.

    ``min_features`` (when given) is the runtime Linear-feature threshold: a different
    ``--min-features`` quantises a different set of Linears, so assign=True would silently
    install a mismatched model while status still reports the scheme. Reject it.

    ``fast_accum`` (fp8 only): when the caller forces it and the checkpoint baked a different
    value, the loaded kernels would ignore the request, so reject and let the dense path
    honor it. A checkpoint predating a metadata field (absent) is accepted for back-compat."""
    if not isinstance(ckpt, dict) or ckpt.get("format") != PREQUANT_FORMAT:
        _warn(logger, scheme, ValueError("unrecognised pre-quant checkpoint format"))
        return False
    if "state_dict" not in ckpt:
        _warn(logger, scheme, ValueError("pre-quant checkpoint has no state_dict"))
        return False
    meta = ckpt.get("metadata") or {}
    if meta.get("scheme") != scheme:
        _warn(logger, scheme, ValueError(f"checkpoint scheme {meta.get('scheme')!r} != {scheme!r}"))
        return False
    # fp8 REQUIRES per-row granularity (per-tensor collapses outlier-heavy DiTs to noise). An
    # old checkpoint omits ``fp8_granularity`` or records non-per-row; reject so the loader
    # re-quantises instead of installing a broken fp8 transformer.
    from .diffusion_transformer_quant import FP8_GRANULARITY, TQ_FP8

    if scheme == TQ_FP8 and meta.get("fp8_granularity") != FP8_GRANULARITY:
        _warn(
            logger,
            scheme,
            ValueError(
                f"fp8 checkpoint granularity {meta.get('fp8_granularity')!r} != "
                f"{FP8_GRANULARITY!r} (stale per-tensor artifact); rebuild it"
            ),
        )
        return False
    ckpt_base = meta.get("base_model_id")
    if base:
        # Keys matching a different base can load strict=True and generate from the wrong
        # weights. Our builder always records base_model_id, so one that omits it against a
        # requested base is untrustworthy -- refuse it.
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
    if min_features is not None:
        ckpt_min = meta.get("min_features")
        if ckpt_min is not None and int(ckpt_min) != int(min_features):
            _warn(
                logger,
                scheme,
                ValueError(f"checkpoint min_features {ckpt_min!r} != runtime {min_features!r}"),
            )
            return False
    # The int8 exclusion set is scheme-derived, but a future change to the token list would
    # leave old checkpoints with a stale baked set that passes scheme+min_features then
    # crashes at the first denoise. Reject a recorded mismatch; absent is accepted.
    ckpt_excludes = meta.get("exclude_name_tokens")
    if ckpt_excludes is not None:
        from .diffusion_transformer_quant import exclude_tokens_for_scheme

        # The exclude set derives from scheme AND family; use the recorded family so an artifact
        # baked under an older token list (e.g. a Qwen int8 checkpoint from before the
        # text-stream exclude) is rejected and re-quantised, not loaded crashing.
        expected = tuple(exclude_tokens_for_scheme(scheme, meta.get("family")))
        if tuple(ckpt_excludes) != expected:
            _warn(
                logger,
                scheme,
                ValueError(
                    f"checkpoint exclude_name_tokens {tuple(ckpt_excludes)!r} != {expected!r}"
                ),
            )
            return False
    # require_bf16 (skip non-bf16 Linears) is scheme-pinned (fp8/mxfp8 need bf16; nvfp4/int8
    # take fp32). Recording and verifying it guards against a future _REQUIRE_BF16_SCHEMES
    # change loading an old-filter checkpoint (different quantised layer set). Absent accepted.
    ckpt_require_bf16 = meta.get("require_bf16")
    if ckpt_require_bf16 is not None:
        from .diffusion_transformer_quant import _REQUIRE_BF16_SCHEMES
        expected_require_bf16 = scheme in _REQUIRE_BF16_SCHEMES
        if bool(ckpt_require_bf16) != expected_require_bf16:
            _warn(
                logger,
                scheme,
                ValueError(
                    f"checkpoint require_bf16 {bool(ckpt_require_bf16)!r} != {expected_require_bf16!r}"
                ),
            )
            return False
    # fp8 fast-accum is baked into the saved kernels; only enforce when the caller forces it.
    if fast_accum is not None:
        ckpt_fa = meta.get("fast_accum")
        if ckpt_fa is not None and bool(ckpt_fa) != bool(fast_accum):
            _warn(
                logger,
                scheme,
                ValueError(f"checkpoint fast_accum {ckpt_fa!r} != requested {bool(fast_accum)!r}"),
            )
            return False
    return True


def _same_base_model(a: str, b: str) -> bool:
    """Tolerant base-model id compare: exact, or same final path/repo segment (e.g.
    ``/models/Z-Image-Turbo`` vs ``Tongyi-MAI/Z-Image-Turbo``)."""

    def _tail(x: str) -> str:
        return x.replace("\\", "/").rstrip("/").split("/")[-1].lower()

    return a == b or _tail(a) == _tail(b)


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
        logger.warning("diffusion.prequant: %s failed: %s", what, exc)
