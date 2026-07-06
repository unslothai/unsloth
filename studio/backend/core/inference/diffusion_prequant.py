# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load a *pre-quantized* transformer instead of quantising a dense one on the GPU.

The opt-in fast transformer_quant path (see ``diffusion_transformer_quant.py``) loads
the dense bf16 transformer and torchao-``quantize_``s it in place. That materialises the
full bf16 weights on the GPU before quantising, so the load peak is ~2x the GGUF's and it
pulls the full bf16 download. When a transformer has already been quantised once and saved
(``scripts/build_prequant_checkpoint.py``), this module loads those weights directly:

  1. build the transformer skeleton on the ``meta`` device (no storage) via
     ``accelerate.init_empty_weights`` + ``from_config``;
  2. ``load_state_dict(assign=True)`` the quantized state dict (the torchao weight subclass
     tensors are assigned in, not copied), so the dense bf16 never touches the GPU;
  3. move to the device.

Measured (B200, Z-Image fp8): transformer GPU load peak 12.9 -> 6.3 GB, download 12 ->
6.28 GB, output bit-identical (LPIPS 0.0). The checkpoint carries the exact same scheme +
``min_features`` as the runtime path, so the result is identical to quantising on the fly.

Best-effort and lazily imported throughout: a missing / mismatched / unreadable checkpoint
returns None and the caller falls back to the dense-quantise path (and then to GGUF). All
behaviour is gated on a configured source -- with nothing configured this module is inert.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# torch.save dict layout this module reads (and the build script writes). Bumped if the
# on-disk structure changes so an old/foreign artifact is rejected rather than mis-loaded.
PREQUANT_FORMAT = "unsloth_prequant_transformer_state_dict_v1"

# Loading a checkpoint ends in ``torch.load(weights_only=False)``, which executes arbitrary
# code embedded in the pickle. A hosted family *repo* checkpoint is first-party and trusted,
# but a ``source.kind == "path"`` can originate from the ``transformer_prequant_path`` field
# of a load request -- i.e. an authenticated API caller naming an arbitrary local file.
# Unpickling that is remote code execution, so a request-supplied path is unpickled ONLY when
# it resolves inside an operator-configured ALLOWLIST of directories. A bare on/off toggle is
# deliberately NOT accepted as a wildcard: enabling local checkpoints for one trusted
# directory must never also permit unpickling any other path a request happens to name. The
# trusted hosted-repo path is unaffected.
ALLOW_LOCAL_PREQUANT_PATH_ENV = "UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH"

_PREQUANT_TOGGLE_TOKENS = {"1", "true", "yes", "on", "0", "false", "no", "off"}


def _allowed_prequant_roots() -> list:
    """Operator-allowlisted directories whose pre-quant checkpoints may be unpickled.

    Set ``UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH`` to one or more directories (separated by
    ``os.pathsep``). A bare truthy/falsey toggle is ignored on purpose -- it must name a
    directory, so there is no "allow everything" mode."""
    import os

    raw = (os.environ.get(ALLOW_LOCAL_PREQUANT_PATH_ENV) or "").strip()
    if not raw:
        return []
    roots = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part or part.lower() in _PREQUANT_TOGGLE_TOKENS:
            continue  # a bare on/off value is not a directory -> never a wildcard allow
        try:
            roots.append(os.path.realpath(os.path.expanduser(part)))
        except Exception:  # noqa: BLE001 — a bad entry is simply not allowlisted
            continue
    return roots


def _local_prequant_path_allowed(path: str) -> bool:
    """True only when ``path`` resolves inside an operator-allowlisted directory; an
    arbitrary request-supplied path is never unpickled. ``realpath`` first so a symlink
    cannot point an allowlisted name at a file outside the allowed roots."""
    import os

    roots = _allowed_prequant_roots()
    if not roots:
        return False
    try:
        real = os.path.realpath(os.path.expanduser(path))
    except Exception:  # noqa: BLE001
        return False
    return any(real == r or real.startswith(r + os.sep) for r in roots)


@dataclass(frozen = True)
class PrequantSource:
    """Where a pre-quantized transformer checkpoint lives. ``kind`` is "path" (a local
    file) or "repo" (a Hub repo id in ``location`` + ``filename`` inside it)."""

    kind: str
    location: str
    filename: Optional[str] = None


def prequant_filename(scheme: str) -> str:
    """The conventional checkpoint filename for ``scheme`` inside a Hub repo."""
    return f"transformer_{scheme}.pt"


def resolve_prequant_source(
    fam: Any,
    scheme: str,
    *,
    path_override: Optional[str] = None,
) -> Optional[PrequantSource]:
    """Resolve where the pre-quantized checkpoint for ``(fam, scheme)`` should come from.

    Priority: (1) an explicit local ``path_override`` (testing / power users); (2) the
    family's hosted repo for ``scheme``; (3) None -> no pre-quant, caller quantises dense.
    Pure: no IO, no torch -- it only decides the source, the loader fetches it.
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
        return PrequantSource(kind = "repo", location = repo_id, filename = prequant_filename(scheme))
    return None


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

    Returns the placed, already-quantized transformer, or None on any problem (missing /
    mismatched / unreadable checkpoint, or a meta-init the class does not support) so the
    caller falls back to the dense-quantise path. Best-effort: never raises for an
    ordinary unavailable artifact.
    """
    try:
        # weights_only=False (required below) executes pickle code, so a caller-supplied
        # local path is unpickled ONLY when it resolves inside an operator-allowlisted
        # directory. The hosted family repo is first-party and always allowed.
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

        # torchao weight subclasses are not safetensors-serializable, so the checkpoint is
        # a torch.save pickle. weights_only=False is required to rebuild those subclasses.
        # The local-path branch is gated above; the repo branch is a first-party artifact.
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
        # assign=True swaps in the loaded (quantized) tensors rather than copying into the
        # meta tensors (a copy into meta is a no-op); strict=True since the saved state
        # dict is the full state dict of the same class (non-persistent buffers excluded).
        transformer.load_state_dict(state_dict, strict = True, assign = True)
        if _has_meta_tensors(transformer):
            # A class with non-persistent buffers (computed in __init__, absent from the
            # state dict) leaves those on meta. Rebuild on CPU so the buffers hold their
            # real values, then re-assign the quantized weights. The dense bf16 lives in
            # CPU RAM only -- the GPU still receives just the quantized footprint.
            transformer = transformer_cls.from_config(config)
            transformer.load_state_dict(state_dict, strict = True, assign = True)

        transformer = transformer.to(device)
        # Built via from_config (not from_pretrained), so it starts in TRAIN mode; the
        # dense and GGUF paths load through from_pretrained, which diffusers documents as
        # returning an eval()'d module. Match that here so any train/eval-sensitive layer
        # (e.g. dropout) can't make prequant inference nondeterministic or diverge from
        # the other load paths.
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

        # Expand ~ once: the allowlist gate (_local_prequant_path_allowed) already
        # expands it, so a "~/..." path that passed the gate must be expanded here too
        # or os.path.isfile() sees the literal "~" and silently skips a real checkpoint.
        expanded = os.path.expanduser(source.location)
        return expanded if os.path.isfile(expanded) else None
    if source.kind == "repo":
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id = source.location, filename = source.filename, token = hf_token)
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

    ``min_features`` (when given) is the runtime Linear-feature threshold: a checkpoint
    built with a different ``--min-features`` quantises a different set of Linear layers,
    so ``load_state_dict(assign=True)`` would silently install a model that does not match
    what the dense path produces while status still reports the requested scheme. Reject it.

    ``fast_accum`` (fp8 only) is the runtime accumulate choice: when the caller forces it
    explicitly (not None) and the checkpoint recorded a different baked value, the loaded
    fp8 kernels would ignore the request while status still reports fp8, so reject and let
    the dense path honor it. A checkpoint that predates the metadata (field absent) is
    accepted unchanged for backward compatibility."""
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
    # fp8 REQUIRES per-row granularity (per-tensor collapses outlier-heavy DiTs to noise). A
    # checkpoint built before that fix carries the old per-tensor layout and either omits
    # ``fp8_granularity`` or records something other than per-row; reject it so the loader
    # falls back to rebuilding/re-quantising instead of installing a broken fp8 transformer.
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
        # A checkpoint whose keys happen to match a different base can load strict=True and
        # then generate from the wrong weights while status reports the requested scheme.
        # Our builder always records base_model_id, so a checkpoint that omits it against a
        # requested base is untrustworthy -- refuse rather than silently accept it.
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
    # The int8 exclusion set (M=1 modulation / conditioning-embedder projections) is derived
    # from the scheme, but a future change to that token list would leave older checkpoints
    # with a stale baked set that still passes scheme+min_features and then crashes at the
    # first denoise step. When the checkpoint records the set, reject a mismatch; absent
    # (older artifact) is accepted since scheme+min_features already pin today's filter.
    ckpt_excludes = meta.get("exclude_name_tokens")
    if ckpt_excludes is not None:
        from .diffusion_transformer_quant import exclude_tokens_for_scheme
        expected = tuple(exclude_tokens_for_scheme(scheme))
        if tuple(ckpt_excludes) != expected:
            _warn(
                logger,
                scheme,
                ValueError(
                    f"checkpoint exclude_name_tokens {tuple(ckpt_excludes)!r} != {expected!r}"
                ),
            )
            return False
    # require_bf16 (skip non-bf16 Linears) is the bf16-weight gate, pinned by the scheme (fp8 and
    # mxfp8 assert a bf16 weight; nvfp4 / int8 quantise fp32 fine). Like exclude_name_tokens it is
    # pinned by the scheme today, but recording and verifying it guards against a future
    # _REQUIRE_BF16_SCHEMES change silently loading a checkpoint built under the old filter (it would
    # carry a different quantised layer set). Absent (older artifact) is accepted since scheme already
    # pins today's gate.
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
    """Tolerant compare of two base-model ids: an exact match, or the same final
    path/repo segment (so a local path or a fork id matches the canonical repo, e.g.
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
