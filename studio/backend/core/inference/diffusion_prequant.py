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

# Loading ends in ``torch.load(weights_only=False)``, which executes pickle code. A hosted repo checkpoint is first-party;
# a ``kind == "path"`` can come from a request, so it is unpickled ONLY inside an operator-configured directory ALLOWLIST.
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


def prequant_subfolder_prefix(subfolder: Optional[str]) -> str:
    """``"prequant/"`` for a non-empty subfolder, ``""`` otherwise.

    Always a literal forward slash, never ``os.path.join``: these strings are Hub REPO paths, and
    the ``\\`` a Windows join would produce matches no repo entry and misses the hub cache lookup
    entirely, so a Windows box would silently download the dense denoiser instead. Any slashes the
    caller wrote around (or inside) the value are normalised for the same reason."""
    cleaned = (subfolder or "").strip().replace("\\", "/").strip("/")
    return f"{cleaned}/" if cleaned else ""


def resolve_prequant_source(
    fam: Any,
    scheme: str,
    *,
    path_override: Optional[str] = None,
    base_repo: Optional[str] = None,
    subfolder: str = "",
) -> Optional[PrequantSource]:
    """Resolve where the checkpoint for ``(fam, scheme)`` comes from.

    Priority: (1) explicit local ``path_override``; (2) the family's hosted repo for
    ``scheme`` (variant-specific when ``base_repo`` names a base with its own baked
    checkpoint); (3) None -> no pre-quant, caller quantises dense. Pure: no IO, no torch.

    ``subfolder`` is the directory inside that repo holding the checkpoint (default: the repo
    root, where every image-side repo keeps it). It prefixes BOTH names, primary and fallback,
    since a repo that nests one nests the other. Everything downstream -- ``try_to_load_from_cache``
    and ``hf_hub_download`` alike -- takes a ``/``-containing filename as-is, so only the name
    BUILDER needs to know.
    """
    override = (path_override or "").strip()
    if override:
        return PrequantSource(kind = "path", location = override, filename = None)
    try:
        from .diffusion_families import family_prequant_repo
        repo_id = family_prequant_repo(fam, scheme, base_repo = base_repo)
    except Exception:  # noqa: BLE001 — a bad family object must not break the load
        repo_id = None
    if repo_id:
        prefix = prequant_subfolder_prefix(subfolder)
        return PrequantSource(
            kind = "repo",
            location = repo_id,
            filename = prefix + prequant_repo_filename(repo_id, scheme),
            fallback_filename = prefix + prequant_filename(scheme),
        )
    return None


def usable_prequant_source(
    fam: Any,
    scheme: str,
    *,
    path_override: Optional[str] = None,
    base_repo: Optional[str] = None,
    subfolder: str = "",
) -> Optional[PrequantSource]:
    """``resolve_prequant_source``, but a local path counts only when the loader would
    accept it: inside the allowlist AND present on disk. Otherwise resolves to None so
    memory planning falls back to dense-fit checks up front, instead of the loader refusing
    the path only after the resident pipeline was evicted and dense bf16 materialises under
    a plan that never budgeted for it (evict-then-OOM). Hosted-repo sources are unaffected."""
    # ``subfolder`` is forwarded only when it is actually set, so the default call stays BYTE
    # IDENTICAL to the one this function has always made. Callers substitute ``resolve_prequant_source``
    # (the auto-policy planner's probe is exercised that way), and a stub written against the older
    # signature would raise TypeError on an unexpected keyword -- swallowed upstream as "no prequant
    # available", which silently sizes the plan for the dense build instead.
    extra = {"subfolder": subfolder} if subfolder else {}
    src = resolve_prequant_source(
        fam, scheme, path_override = path_override, base_repo = base_repo, **extra
    )
    if src is not None and src.kind == "path" and not local_prequant_path_ready(src.location):
        return None
    return src


def cached_checkpoint_path(source: Any, *, cache_dir: Optional[str] = None) -> Optional[str]:
    """The path of a hosted (``kind == "repo"``) checkpoint ALREADY in the local Hub cache.

    A pure lookup (a refs read plus a stat, no network), so memory planning can ask on every pick.
    Only the PRIMARY ``filename`` counts: a cached ``fallback_filename`` (the legacy artifact) must
    not short-circuit it, or a stale name stays pinned once the repo ships the real one, so a
    fallback-only cache reads as "this would have to download" and the GGUF simply runs.

    Both cache roots are searched: Studio pins the LIVE cache setting while an unpinned
    ``hf_hub_download`` falls back to huggingface_hub's import-time constant. Never raises."""
    for root in (cache_dir, None) if cache_dir else (None,):
        hit = _cached_in_root(source, root)
        if hit is not None:
            return hit
    return None


def _cached_in_root(
    source: Any,
    root: Optional[str],
    name: Optional[str] = None,
) -> Optional[str]:
    """One checkpoint name's path inside ONE cache root, or None. Defaults to the primary name; the
    resolver passes ``fallback_filename`` once the primary turns out to be absent. Never raises."""
    if source is None or getattr(source, "kind", None) != "repo":
        return None
    name = name or getattr(source, "filename", None)
    if not name:
        return None
    try:
        import os

        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001 — no cache API to ask: treat as not cached
        return None
    try:
        hit = try_to_load_from_cache(source.location, name, cache_dir = root)
    except Exception:  # noqa: BLE001 — a malformed cache entry is not a hit
        return None
    # A str is the cached path; a miss is None and a known-absent file is a sentinel object.
    return hit if isinstance(hit, str) and os.path.isfile(hit) else None


def prequant_checkpoint_cached(source: Any, *, cache_dir: Optional[str] = None) -> bool:
    """True when ``source`` resolves from the cache, i.e. enabling prequant costs no download."""
    return cached_checkpoint_path(source, cache_dir = cache_dir) is not None


def _pin_kernel_preference(state_dict: Any, logger: Any = None) -> int:
    """Force every loaded fp8 weight onto the plain-torch kernel, matching the local path.

    `_fp8_config` pins `KernelPreference.TORCH` when it BUILDS a config, because AUTO silently
    switches to the MSLK kernel wherever an mslk package is importable (sm90+). A hosted
    checkpoint escapes that pin entirely: the preference is serialized on each Float8Tensor, and
    every published one carries AUTO. Restoring it re-arms the exact kernel the pin exists to
    avoid, and `mslk.f8f8bf16_rowwise` has no fake impl, so the first COMPILED generate dies with
    "Operator does not support running with fake tensors" -- an HTTP 500 on the default speed
    mode, reachable the moment the pre-quant repos are readable.

    Safe to rewrite in place: the preference selects a matmul kernel, it is not weight data, so
    the tensors stay bit-identical and the checkpoint's own sha256 still describes them. The
    plain-torch path is also the faster one compiled (an opaque extern call blocks inductor
    quantize fusion), so this costs nothing.
    """
    try:
        from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
    except Exception:  # noqa: BLE001 -- enum moved or absent: leave the checkpoint as saved
        return 0
    pinned = 0
    for t in state_dict.values():
        if getattr(t, "kernel_preference", None) not in (None, KernelPreference.TORCH):
            try:
                t.kernel_preference = KernelPreference.TORCH
                pinned += 1
            except Exception:  # noqa: BLE001 -- frozen subclass: nothing else to try
                pass
    if pinned and logger is not None:
        logger.info("diffusion.prequant: pinned %d weights to the plain-torch fp8 kernel", pinned)
    return pinned


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
    cache_dir: Optional[str] = None,
    prepare_model: Optional[Any] = None,
    logger: Any = None,
) -> Optional[Any]:
    """Load the pre-quantized transformer described by ``source`` onto ``device``.

    ``cache_dir`` is the live Hub cache root, as every other loader call pins it: unset, a fetch
    lands under huggingface_hub's import-time constant, so a mid-session cache change re-downloads
    into a root Studio no longer reads.

    ``prepare_model`` (optional) is called as ``prepare_model(transformer, metadata)`` on the
    freshly built skeleton, AFTER ``from_config`` and BEFORE ``load_state_dict``. That window is
    the only one where a family can reshape the module to match how the checkpoint was baked (a
    swapped submodule, a patched attention class): earlier there is no module, and later
    ``strict=True`` has already rejected the mismatch. It gets the checkpoint's own metadata so it
    can key on what was baked rather than on today's defaults. A raising callback falls out to the
    outer handler below, i.e. a warning and a dense fallback, never a failed load.

    Returns the placed transformer, or None on any problem (missing / mismatched /
    unreadable checkpoint, or unsupported meta-init) so the caller falls back to
    dense-quantise. Best-effort: never raises for an unavailable artifact.
    """
    try:
        # weights_only=False executes pickle code, so a local path is unpickled ONLY when allowlisted; the hosted family repo is first-party.
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

        path = _resolve_checkpoint_path(source, hf_token, cache_dir)
        if path is None:
            return None

        import torch

        # torchao weight subclasses are not safetensors-serializable, so the checkpoint is a torch.save pickle and weights_only=False rebuilds them. Local paths gated above.
        ckpt = torch.load(path, weights_only = False, map_location = "cpu")
        if not _validate_checkpoint(
            ckpt, scheme, base, logger, min_features = min_features, fast_accum = fast_accum
        ):
            return None
        state_dict = ckpt["state_dict"]
        _pin_kernel_preference(state_dict, logger)

        # Read from the root that actually supplied the checkpoint: after a mid-session cache change
        # the pinned root may be gone or read-only, and load_config's raise is swallowed below into
        # a None return, silently dropping a prequant whose checkpoint is cached and already loaded.
        config = _load_transformer_config(transformer_cls, base, hf_token, cache_dir, path)
        from accelerate import init_empty_weights

        metadata = ckpt.get("metadata") or {}
        with init_empty_weights():
            transformer = transformer_cls.from_config(config)
        if prepare_model is not None:
            prepare_model(transformer, metadata)
        # assign=True swaps in the loaded tensors instead of copying into meta (a no-op); strict=True since the saved dict is the full state dict of the same class.
        transformer.load_state_dict(state_dict, strict = True, assign = True)
        if _has_meta_tensors(transformer):
            # Non-persistent buffers (built in __init__, absent from the state dict) stay on meta. Rebuild on CPU so they hold real values, then re-assign the quantized weights; dense bf16 never reaches the GPU.
            transformer = transformer_cls.from_config(config)
            # The retry REPLACES the module, so the hook has to run again: skipping it here would
            # load the same state dict into a differently shaped model, and this branch is the one
            # families with non-persistent buffers always take -- the mismatch would be the norm,
            # not the corner case, and strict=True would surface it as a bare key error.
            if prepare_model is not None:
                prepare_model(transformer, metadata)
            transformer.load_state_dict(state_dict, strict = True, assign = True)

        transformer = transformer.to(device)
        # Same small-M row padding the runtime quantise path applies, and for the same reason: a
        # checkpoint built under the current exclusion set QUANTISES the family's small-M linears,
        # so without the wrappers they would raise inside _int_mm the moment the compiled scope
        # reaches them. After load_state_dict, since wrapping reparents the Linears; after .to()
        # so the granularity probe reads the device tensors the GEMM will actually see.
        from .diffusion_transformer_quant import apply_small_m_padding

        apply_small_m_padding(transformer, scheme, metadata.get("family"), logger = logger)
        # from_config starts in TRAIN mode while the dense/GGUF paths use from_pretrained (eval()'d). Match it so train/eval-sensitive layers cannot make prequant inference diverge.
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


def _entry_not_found_errors() -> tuple:
    """``(EntryNotFoundError, LocalEntryNotFoundError)`` for both huggingface_hub majors.

    On 1.x the base splits into a remote 404 and ``LocalEntryNotFoundError`` (no copy in this root,
    no network); on BOTH majors local subclasses the base, so catch it first where they differ.
    Private markers on an unexpected layout are raised by nothing, keeping today's behaviour."""
    try:
        from huggingface_hub.errors import EntryNotFoundError
    except Exception:  # noqa: BLE001 — older/newer hub layouts

        class EntryNotFoundError(Exception):  # type: ignore[no-redef]
            pass

    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except Exception:  # noqa: BLE001

        class LocalEntryNotFoundError(EntryNotFoundError):  # type: ignore[no-redef]
            pass

    return EntryNotFoundError, LocalEntryNotFoundError


def _download_checkpoint_name(
    source: PrequantSource,
    name: str,
    hf_token: Optional[str],
    cache_dir: Optional[str],
    *,
    propagate_missing: bool,
) -> str:
    """Download ONE checkpoint filename, reusing a copy that sits under the other cache root.

    Pinned to ``cache_dir``, hf_hub_download would not look there and would re-fetch multiple GB, so
    re-run it THROUGH that root rather than return the raw path: the blob is reused after one HEAD,
    a republished checkpoint is picked up rather than pinned stale, and offline still resolves off
    the cached pointer. ``propagate_missing`` says another filename is still to be tried, so a
    remote 404 for THIS one must reach the caller's fallback branch; swallowing it would return the
    stale other-root copy of a name the repo no longer publishes. A local cache miss is not that
    verdict, and with no name left to try neither is a 404: both keep the copy already found."""
    from huggingface_hub import hf_hub_download

    EntryNotFoundError, LocalEntryNotFoundError = _entry_not_found_errors()

    if cache_dir is not None and _cached_in_root(source, cache_dir, name) is None:
        elsewhere = _cached_in_root(source, None, name)
        if elsewhere is not None:
            try:
                return hf_hub_download(
                    repo_id = source.location,
                    filename = name,
                    token = hf_token,
                    cache_dir = None,
                )
            except LocalEntryNotFoundError:  # offline with the copy right there: use it
                return elsewhere
            except EntryNotFoundError:
                if not propagate_missing:
                    return elsewhere
                raise
            except Exception:  # noqa: BLE001 — revalidation is a bonus, never a new failure
                return elsewhere
    return hf_hub_download(
        repo_id = source.location,
        filename = name,
        token = hf_token,
        cache_dir = cache_dir,
    )


def _resolve_checkpoint_path(
    source: PrequantSource,
    hf_token: Optional[str],
    cache_dir: Optional[str] = None,
) -> Optional[str]:
    """The local file path for ``source``, downloading from the Hub if needed; None if absent."""
    if source.kind == "path":
        import os

        # Expand ~ (the allowlist gate already did), else os.path.isfile sees a literal "~".
        expanded = os.path.expanduser(source.location)
        return expanded if os.path.isfile(expanded) else None
    if source.kind == "repo":
        EntryNotFoundError, _ = _entry_not_found_errors()
        has_fallback = (
            bool(source.fallback_filename) and source.fallback_filename != source.filename
        )
        try:
            return _download_checkpoint_name(
                source,
                source.filename,
                hf_token,
                cache_dir,
                propagate_missing = has_fallback,
            )
        except EntryNotFoundError:
            if not has_fallback:
                raise
            # Primary genuinely absent: same other-root treatment, with nothing left after it.
            return _download_checkpoint_name(
                source,
                source.fallback_filename,
                hf_token,
                cache_dir,
                propagate_missing = False,
            )
    return None


def _config_cache_roots(checkpoint_path: str, cache_dir: Optional[str]) -> tuple:
    """Cache roots to read the transformer config from, the checkpoint's OWN root first.

    ``_resolve_checkpoint_path`` may answer from huggingface_hub's import-time root even when Studio
    pins its live one, so pinning the config to the live root alone misses in exactly the
    cache-moved/offline case the checkpoint lookup just accepted, and load_config's raise is
    swallowed into a None return. The other root is still tried second."""
    if cache_dir is None:
        return (None,)
    import os

    try:
        # normcase before comparing: on Windows C:\Users vs c:\users would read as "not under the
        # live root" and silently reverse the order below.
        root = os.path.normcase(os.path.realpath(cache_dir))
        real = os.path.normcase(os.path.realpath(checkpoint_path))
        under_live = real == root or real.startswith(root + os.sep)
    except Exception:  # noqa: BLE001 — an unresolvable path keeps today's order
        under_live = True
    return (cache_dir, None) if under_live else (None, cache_dir)


def _load_transformer_config(
    transformer_cls: Any,
    base: str,
    hf_token: Optional[str],
    cache_dir: Optional[str],
    checkpoint_path: str,
) -> Any:
    """``transformer_cls.load_config`` against the checkpoint's cache root, then the other one."""
    last: Optional[BaseException] = None
    for root in _config_cache_roots(checkpoint_path, cache_dir):
        try:
            return transformer_cls.load_config(
                base, subfolder = "transformer", token = hf_token, cache_dir = root
            )
        except Exception as exc:  # noqa: BLE001 — try the other root before giving up
            last = exc
    raise last  # type: ignore[misc]


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
    # fp8 REQUIRES per-row granularity (per-tensor collapses outlier-heavy DiTs to noise). An old checkpoint omits ``fp8_granularity`` or records non-per-row, so reject and let the loader re-quantise.
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
        # Keys matching a different base can load strict=True and generate from the wrong weights. Our builder always records base_model_id, so one omitting it against a requested base is untrustworthy.
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
    # The int8 exclusion set is scheme-derived, so a token-list change would leave old checkpoints with a stale baked set that passes scheme+min_features then crashes at the first denoise. Reject a recorded mismatch; absent is accepted.
    ckpt_excludes = meta.get("exclude_name_tokens")
    if ckpt_excludes is not None:
        from .diffusion_transformer_quant import exclude_tokens_for_scheme

        # The exclude set derives from scheme AND family, so use the recorded family: an artifact baked under an older token list is rejected and re-quantised, not loaded crashing.
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
    # require_bf16 (skip non-bf16 Linears) is scheme-pinned; recording and verifying it stops a future _REQUIRE_BF16_SCHEMES change loading an old-filter checkpoint. Absent accepted.
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
    ``/models/Z-Image-Turbo`` vs ``Tongyi-MAI/Z-Image-Turbo``).

    Both sides normalise through ``canonical_base`` first, so a mirror id in a baked
    ``base_model_id`` check cannot refuse the checkpoint and send the load down the multi-GB dense
    download. Today's mirrors keep the repo name, so the tail compare would cover them, but this
    must not depend on that.
    """
    from .diffusion_families import canonical_base

    a, b = canonical_base(a), canonical_base(b)

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
