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

torchao's weight subclasses are not safetensors-serializable, so the artifact is a torch.save
pickle -- read under ``weights_only`` plus the constructor ALLOWLIST below, never as a free one.
It is a mutable remote file reached by loads that never asked for a scheme (auto resolves an unset
precision to a hosted checkpoint), so "first-party repo" cannot stand in for that restriction.

Best-effort and lazily imported: a missing / mismatched / unreadable checkpoint returns None
and the caller falls back to dense-quantise (then GGUF). Inert with nothing configured.
"""

from __future__ import annotations

import threading as _threading
from dataclasses import dataclass
from typing import Any, Optional

# torch.save dict layout tag; bump on an on-disk change so old/foreign artifacts are rejected.
PREQUANT_FORMAT = "unsloth_prequant_transformer_state_dict_v1"

# v2 is v1 plus an ACTIVATION ROTATION (see ``diffusion_convrot``): the weights are stored in a
# rotated basis and are wrong unless the loader rotates the activations to match. That is the one
# on-disk change a released Unsloth cannot ignore safely -- an old build reading a rotated artifact
# as v1 would load it clean, raise nothing and render quietly wrong pixels forever -- so it gets a
# tag old builds refuse outright, and the load drops to dense instead. Strictly a biconditional:
# a v2 artifact MUST declare a rotation and a v1 artifact must NOT, both checked below, so neither
# a hand-edited tag nor a builder that forgot one half can produce something that loads.
PREQUANT_FORMAT_ROTATED = "unsloth_prequant_transformer_state_dict_v2"

PREQUANT_FORMATS = (PREQUANT_FORMAT, PREQUANT_FORMAT_ROTATED)


def prequant_format_for(metadata: Any) -> str:
    """The on-disk format tag an offline builder should stamp for ``metadata``."""
    from .diffusion_convrot import declares_rotation
    return PREQUANT_FORMAT_ROTATED if declares_rotation(metadata) else PREQUANT_FORMAT


# A request-supplied ``kind == "path"`` is read ONLY inside an operator-configured directory
# ALLOWLIST: an arbitrary path is an arbitrary MODEL. Not a code-execution gate -- the load is
# weights_only either way.
ALLOW_LOCAL_PREQUANT_PATH_ENV = "UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH"

# The constructors a pre-quant checkpoint's pickle may name, on top of what ``weights_only``
# already permits (storages, dtypes, ``_rebuild_*``, ``OrderedDict``, ``torch.device``,
# ``_get_layout``). Surveyed across every hosted checkpoint Unsloth resolves (image + video, fp8 +
# int8, rotated and not) this is the complete set, so the load runs ``weights_only = True`` and a
# checkpoint naming anything else is refused before one opcode of it executes, hosted or local.
#
# Registered under the name the PICKLE records, which for a re-exported class is not the class's
# own ``__module__`` (``torchao.quantization.Float8Tensor`` really lives in
# ``...quantize_.workflows.float8.float8_tensor``), so both spellings are listed. Names a given
# torchao lacks are skipped rather than raised: the set spans every release
# ``install_python_stack`` pins (0.14, 0.16, 0.17) and an absent class could not have produced a
# loadable checkpoint here anyway.
#
# Adding a scheme means adding its constructors here; forgetting warns and falls back to
# dense-quantise, never a silent unpickle.
_PREQUANT_SAFE_GLOBALS: tuple[tuple[str, str], ...] = (
    # int8: AffineQuantizedTensor + its plain layout, wrapped for dynamic activation quant.
    ("torchao.dtypes.affine_quantized_tensor", "AffineQuantizedTensor"),
    ("torchao.dtypes.uintx.plain_layout", "PlainAQTTensorImpl"),
    ("torchao.dtypes.utils", "PlainLayout"),
    ("torchao.quantization.linear_activation_quantized_tensor", "LinearActivationQuantizedTensor"),
    ("torchao.quantization.quant_api", "_int8_symm_per_token_reduced_range_quant"),
    ("torchao.quantization.quant_primitives", "ZeroPointDomain"),
    ("torchao.quantization.quant_primitives", "MappingType"),
    # fp8: the newer tensor subclass, its per-row granularity and its kernel/mm options.
    ("torchao.quantization", "Float8Tensor"),
    ("torchao.quantization.quantize_.workflows.float8.float8_tensor", "Float8Tensor"),
    (
        "torchao.quantization.quantize_.workflows.float8.float8_tensor",
        "QuantizeTensorToFloat8Kwargs",
    ),
    ("torchao.quantization.quantize_.common.kernel_preference", "KernelPreference"),
    ("torchao.quantization.granularity", "PerRow"),
    ("torchao.quantization.granularity", "PerTensor"),
    ("torchao.float8.inference", "Float8MMConfig"),
    # mxfp8 / nvfp4: no hosted checkpoint uses these, but they are TQ_SCHEMES that
    # scripts/build_prequant_checkpoint.py bakes, so a LOCAL override can be either. torchao only
    # registers them on import of the prototype package, which nothing on this path imports.
    ("torchao.prototype.mx_formats.mx_tensor", "MXTensor"),
    ("torchao.prototype.mx_formats.mx_tensor", "QuantizeTensorToMXKwargs"),
    ("torchao.prototype.mx_formats.config", "ScaleCalculationMode"),
    ("torchao.prototype.mx_formats.nvfp4_tensor", "NVFP4Tensor"),
    ("torchao.prototype.mx_formats.nvfp4_tensor", "QuantizeTensorToNVFP4Kwargs"),
    # The version string torch.save stamps into the subclass state: not in torch's default set,
    # and without it every torchao checkpoint refuses to load.
    ("torch.torch_version", "TorchVersion"),
)


def _prequant_safe_globals() -> list:
    """``(object, pickled name)`` pairs to register; names this torchao lacks are skipped."""
    import importlib

    pairs = []
    for module, name in _PREQUANT_SAFE_GLOBALS:
        try:
            obj = getattr(importlib.import_module(module), name)
        except Exception:  # noqa: BLE001 -- a name this release does not ship is not allowed
            continue
        pairs.append((obj, f"{module}.{name}"))
    return pairs


_SAFE_GLOBALS_LOCK = _threading.Lock()
_SAFE_GLOBALS_REGISTERED: Optional[bool] = None
# Filled in by the registration: which of the names above this install actually resolved.
_RESOLVED_SAFE_GLOBALS: set = set()

# What a checkpoint of each scheme actually NAMES, read off the artifacts with pickletools rather
# than assumed: every hosted repo the family tables list, plus a local bake of each scheme for the
# two nothing hosts. Only these are required, so dropping an unused name does not fail a scheme.
_SCHEME_REQUIRED_GLOBALS: dict = {
    "int8": frozenset(
        {
            "torchao.dtypes.affine_quantized_tensor.AffineQuantizedTensor",
            "torchao.dtypes.uintx.plain_layout.PlainAQTTensorImpl",
            "torchao.dtypes.utils.PlainLayout",
            "torchao.quantization.linear_activation_quantized_tensor."
            "LinearActivationQuantizedTensor",
            "torchao.quantization.quant_api._int8_symm_per_token_reduced_range_quant",
            "torchao.quantization.quant_primitives.ZeroPointDomain",
            "torch.torch_version.TorchVersion",
        }
    ),
    "fp8": frozenset(
        {
            # The ALIAS spelling, which is what the fp8 pickles record.
            "torchao.quantization.Float8Tensor",
            "torchao.quantization.quantize_.workflows.float8.float8_tensor."
            "QuantizeTensorToFloat8Kwargs",
            "torchao.quantization.quantize_.common.kernel_preference.KernelPreference",
            "torchao.quantization.granularity.PerRow",
            "torchao.float8.inference.Float8MMConfig",
            "torch.torch_version.TorchVersion",
        }
    ),
    "mxfp8": frozenset(
        {
            "torchao.prototype.mx_formats.mx_tensor.MXTensor",
            "torchao.prototype.mx_formats.mx_tensor.QuantizeTensorToMXKwargs",
            "torchao.prototype.mx_formats.config.ScaleCalculationMode",
            "torchao.quantization.quantize_.common.kernel_preference.KernelPreference",
        }
    ),
    "nvfp4": frozenset(
        {
            "torchao.prototype.mx_formats.nvfp4_tensor.NVFP4Tensor",
            "torchao.prototype.mx_formats.nvfp4_tensor.QuantizeTensorToNVFP4Kwargs",
        }
    ),
}


def _tuple_safe_globals_supported() -> bool:
    """Whether this torch's ``add_safe_globals`` understands ``(object, name)`` pairs (2.6+).

    Asked by VERSION rather than by trying it: 2.4/2.5 accept the pairs silently and only fail
    later, in ``_get_user_allowed_globals``, which reads ``f.__module__`` off every entry of a
    PROCESS-WIDE list -- so a tuple left there breaks every other weights_only load in Unsloth.
    Nothing is registered unless the answer here is yes."""
    try:
        import torch
        parts = str(torch.__version__).split("+")[0].split(".")
        return (int(parts[0]), int(parts[1])) >= (2, 6)
    except Exception:  # noqa: BLE001 -- an unreadable version is not a supported one
        return False


def _register_prequant_safe_globals() -> bool:
    """Register the allowlist ONCE, process-wide and permanently. True when the load can run.

    Not the ``safe_globals`` context manager, deliberately: it adds on entry and REMOVES on exit
    against a process-wide table, so two overlapping reads (a download-plan probe beside a load;
    both arrive on the route's thread pool) let whichever finishes first strip the allowlist out
    from under the other's ``torch.load``, failing a good checkpoint and dropping it to dense.
    Adding once and never removing has no such window.

    The widening this costs is small and bounded: other ``weights_only`` loads in the process
    also accept these torch/torchao tensor constructors, which build tensors and nothing else. A
    pickle naming ANY global is still refused.

    Registration takes ``(object, name)`` pairs so a re-exported class is registered under the
    name the pickle records, and that form is version-checked BEFORE anything is registered (see
    ``_tuple_safe_globals_supported``). Below 2.6 nothing is registered and
    ``restricted_prequant_load_supported`` tells planning to stop offering pre-quant sources at
    all. Answered once and memoised, including the failure."""
    global _SAFE_GLOBALS_REGISTERED

    if _SAFE_GLOBALS_REGISTERED is not None:
        return _SAFE_GLOBALS_REGISTERED
    with _SAFE_GLOBALS_LOCK:
        if _SAFE_GLOBALS_REGISTERED is not None:
            return _SAFE_GLOBALS_REGISTERED
        ok = False
        try:
            from core._torchao_stub import is_stubbed

            import torch

            add = getattr(torch.serialization, "add_safe_globals", None)
            # A STUBBED torchao (Windows ROCm) fabricates a class for every name asked of it, so
            # the allowlist would register fakes and answer yes for an install that cannot
            # rebuild a single quantized tensor.
            if add is not None and not is_stubbed("torchao") and _tuple_safe_globals_supported():
                pairs = _prequant_safe_globals()
                resolved = {name for _obj, name in pairs}
                # "Some entries resolved" is not "a checkpoint can be opened". The floor is what
                # EVERY artifact needs whatever its scheme: the version stamp plus at least one
                # real torchao tensor class. Per-SCHEME completeness is asked separately, by the
                # caller that knows which scheme it is about to plan for.
                if "torch.torch_version.TorchVersion" in resolved and any(
                    name.startswith("torchao.") for name in resolved
                ):
                    add(pairs)
                    _RESOLVED_SAFE_GLOBALS.update(resolved)
                    # The same derivation the unpickler runs, so a form this torch cannot express
                    # fails here rather than under a load a plan was already sized on.
                    try:
                        torch._weights_only_unpickler._get_user_allowed_globals()
                    except AttributeError:  # noqa: BLE001 -- private; absence is not a failure
                        pass
                    ok = True
        except Exception:  # noqa: BLE001 -- no allowlist means no restricted load, never a raise
            ok = False
        _SAFE_GLOBALS_REGISTERED = ok
        return ok


def restricted_prequant_load_supported(scheme: Optional[str] = None) -> bool:
    """Whether this install can read a pre-quant checkpoint, for ``scheme`` when one is named.

    Without the allowlist there is no safe way to open a pre-quant pickle and the loader refuses.
    Planning has to ask the same question BEFORE it sizes the load: a plan that counts on a 6 GB
    artifact, drops the dense shards and evicts the resident pipeline has nothing left when the
    refusal arrives. ``usable_prequant_source`` therefore answers None here, hosted and local
    alike, which is the same answer the loader will give.

    PER SCHEME, because the schemes do not share constructors and torchao does not retire them
    together: ``AffineQuantizedTensor`` and its layout carry every int8 checkpoint and are
    already deprecated upstream (pytorch/ao#2752), so a release that drops them while keeping
    ``Float8Tensor`` leaves fp8 loadable and int8 not. An unknown or unnamed scheme gets the
    floor answer the registration itself already checked."""
    if not _register_prequant_safe_globals():
        return False
    required = _SCHEME_REQUIRED_GLOBALS.get((scheme or "").strip().lower())
    return True if required is None else required <= _RESOLVED_SAFE_GLOBALS


def _torch_load_prequant(path: str, **kwargs: Any) -> Any:
    """``torch.load`` a pre-quant checkpoint under the allowlist above.

    ``weights_only = True`` is the whole point: a pickle that may name any global is remote code
    execution the moment the artifact is not the one that was published. Everything the format
    legitimately needs is allowlisted, so the restriction costs nothing and a mutated artifact
    raises ``UnpicklingError`` into the caller's dense fallback instead of running. A torch that
    cannot express the allowlist is refused outright, never reopened unrestricted."""
    import torch

    if not _register_prequant_safe_globals():
        raise RuntimeError(
            "this torch cannot register the pre-quant constructor allowlist (needs "
            "torch.serialization.add_safe_globals with (object, name) support, i.e. >= 2.6), so "
            "a pre-quant checkpoint cannot be deserialized without allowing arbitrary pickle "
            "globals"
        )
    return torch.load(path, weights_only = True, **kwargs)


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
    base_repo: Optional[str] = None,
    task: Optional[str] = None,
) -> Optional[PrequantSource]:
    """Resolve where the checkpoint for ``(fam, scheme)`` comes from.

    Priority: (1) explicit local ``path_override``; (2) the family's hosted repo for
    ``scheme`` (variant-specific when ``base_repo`` names a base with its own baked
    checkpoint); (3) None -> no pre-quant, caller quantises dense. Pure: no IO, no torch.

    ``task`` names the workflow / denoiser PARTITION the load is bringing up, for the families
    that host more than one under a single repo and scheme (MiniMax-H3's keyframe and reference
    denoisers). It only ever selects a more specific filename: unset, or set to a task the family
    declares nothing for, resolves exactly what it resolved before.

    Both names are repo-ROOT names. Every hosted prequant repo, image and video alike, keeps its
    checkpoints at the root, so there is no directory to prepend; a repo that nested them would
    404 on the primary AND on the fallback and the load would silently fall back to dense.
    """
    override = (path_override or "").strip()
    if override:
        return PrequantSource(kind = "path", location = override, filename = None)
    preferred = None
    agnostic = None
    try:
        from .diffusion_families import family_prequant_filename, family_prequant_repo

        repo_id = family_prequant_repo(fam, scheme, base_repo = base_repo)
        preferred = family_prequant_filename(fam, scheme, task = task)
        # What the same call would have resolved WITHOUT a task, which is what decides whether a
        # fallback is safe below. Skipped when no task was asked for, since then the two are the
        # same lookup.
        agnostic = family_prequant_filename(fam, scheme) if task else preferred
    except Exception:  # noqa: BLE001 — a bad family object must not break the load
        repo_id = None
    if repo_id:
        derived = prequant_repo_filename(repo_id, scheme)
        # A family may name a SECOND artifact for the same repo and scheme (today: MiniMax-H3's
        # rotated INT8 denoiser). It becomes the primary and the derived name becomes the
        # fallback, so a build that knows the new name gets it and every older build keeps
        # resolving the artifact it already understands. Without an override nothing changes: the
        # derived name is primary and the legacy transformer_<scheme>.pt is the fallback.
        #
        # A TASK-SPECIFIC name gets NO fallback. The other artifacts in the repo are the same
        # family, the same scheme and the same base, so every check the loader makes would pass
        # on them -- the fallback would quietly install another partition's denoiser and generate
        # from the wrong weights, which is precisely what naming the artifact per task prevents.
        # Absent is better than wrong here: no artifact means the released bfloat16 denoiser.
        task_specific = preferred is not None and preferred != agnostic
        return PrequantSource(
            kind = "repo",
            location = repo_id,
            filename = preferred or derived,
            fallback_filename = (
                None if task_specific else (derived if preferred else prequant_filename(scheme))
            ),
        )
    return None


_LOCAL_PREQUANT_SCHEME: dict[tuple[str, int, int], Optional[str]] = {}


def local_prequant_scheme(path: str) -> Optional[str]:
    """The scheme a local pre-quant checkpoint records, or None when it cannot be read.

    ``resolve_prequant_source`` hands back a ``path`` source for ANY override, whatever scheme was
    asked for: the file is never inspected. That is fine when the caller named the scheme, but
    under ``auto`` the ladder picks one and an override baked for a different scheme then reads as
    an available pre-quant. Planning skips staging the dense transformer, the loader reaches the
    same ``metadata.scheme`` check that runs at load time, refuses the file, and with no dense
    fallback the pick silently drops to GGUF.

    Cheap despite the file size: ``mmap`` plus ``map_location = "meta"`` maps the storages instead
    of reading them, so only the pickle structure is parsed (~1s on a 34 GB checkpoint). Cached on
    (path, mtime, size) because the auto ladder asks once per candidate scheme. Read under the
    same allowlisted ``weights_only`` load the loader uses, so probing a file that turns out not
    to be a checkpoint cannot execute anything either."""
    import os

    try:
        real = os.path.expanduser(path)
        st = os.stat(real)
        # Nanoseconds, not int(st_mtime): an atomic swap for a same-sized artifact inside the same
        # second would otherwise reuse the previous scheme for the life of the process, and int8
        # and fp8 checkpoints of one model are exactly that shape.
        key = (real, st.st_mtime_ns, int(st.st_size))
    except Exception:  # noqa: BLE001 -- unreadable is "unknown", handled by the caller
        return None
    if key in _LOCAL_PREQUANT_SCHEME:
        return _LOCAL_PREQUANT_SCHEME[key]
    scheme: Optional[str] = None
    try:
        obj = _torch_load_prequant(real, map_location = "meta", mmap = True)
        if isinstance(obj, dict) and obj.get("format") in PREQUANT_FORMATS:
            recorded = (obj.get("metadata") or {}).get("scheme")
            scheme = str(recorded) if recorded else None
    except Exception:  # noqa: BLE001 -- a checkpoint we cannot parse is "unknown", never a match
        scheme = None
    _LOCAL_PREQUANT_SCHEME[key] = scheme
    return scheme


def usable_prequant_source(
    fam: Any,
    scheme: str,
    *,
    path_override: Optional[str] = None,
    base_repo: Optional[str] = None,
) -> Optional[PrequantSource]:
    """``resolve_prequant_source``, but a local path counts only when the loader would
    accept it: inside the allowlist AND present on disk AND baked for THIS scheme. Otherwise
    resolves to None so memory planning falls back to dense-fit checks up front, instead of the
    loader refusing the path only after the resident pipeline was evicted and dense bf16
    materialises under a plan that never budgeted for it (evict-then-OOM). Hosted-repo sources are
    unaffected.

    The scheme check matters most under ``auto``, which picks a scheme the user never named: an
    int8 override must not read as an available fp8 pre-quant just because the file exists. A
    checkpoint whose scheme cannot be read is treated as not usable, matching every other unknown
    here, since the loader would reject it too.

    An install that cannot restrict the load has no usable source AT ALL, hosted included: the
    loader refuses every checkpoint there, and a plan that had already dropped the dense shards
    for one would find that out after the eviction."""
    if not restricted_prequant_load_supported(scheme):
        return None
    src = resolve_prequant_source(fam, scheme, path_override = path_override, base_repo = base_repo)
    if src is not None and src.kind == "path":
        if not local_prequant_path_ready(src.location):
            return None
        if local_prequant_scheme(src.location) != scheme:
            return None
    return src


def cached_checkpoint_path(source: Any, *, cache_dir: Optional[str] = None) -> Optional[str]:
    """The path of a hosted (``kind == "repo"``) checkpoint ALREADY in the local Hub cache.

    A pure lookup (a refs read plus a stat, no network), so memory planning can ask on every pick.
    Only the PRIMARY ``filename`` counts: a cached ``fallback_filename`` (the legacy artifact) must
    not short-circuit it, or a stale name stays pinned once the repo ships the real one, so a
    fallback-only cache reads as "this would have to download" and the GGUF simply runs.

    Both cache roots are searched: Unsloth pins the LIVE cache setting while an unpinned
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
    config_subfolder: str = "transformer",
    local_files_only: bool = False,
    logger: Any = None,
) -> Optional[Any]:
    """Load the pre-quantized transformer described by ``source`` onto ``device``.

    ``cache_dir`` is the live Hub cache root, as every other loader call pins it: unset, a fetch
    lands under huggingface_hub's import-time constant, so a mid-session cache change re-downloads
    into a root Unsloth no longer reads.

    ``config_subfolder`` is where the DENOISER CONFIG lives inside ``base``, defaulting to the
    universal ``transformer``. A family hosting several denoiser partitions in one repo overrides
    it with the one this checkpoint belongs to (MiniMax-H3's ``transformer_ref``): the scoped
    download stages only that partition, so reading the config from the other one would send an
    otherwise fully staged load back to the Hub.

    ``prepare_model`` (optional) is called as ``prepare_model(transformer, metadata)`` on the
    freshly built skeleton, AFTER ``from_config`` and BEFORE ``load_state_dict``. That window is
    the only one where a family can reshape the module to match how the checkpoint was baked (a
    swapped submodule, a patched attention class): earlier there is no module, and later
    ``strict=True`` has already rejected the mismatch. It gets the checkpoint's own metadata so it
    can key on what was baked rather than on today's defaults. A raising callback falls out to the
    outer handler below, i.e. a warning and a dense fallback, never a failed load.

    A checkpoint that declares an ACTIVATION ROTATION (``diffusion_convrot``) has the matching
    online half installed here, on exactly the fqns it records. That is unconditional and central
    rather than a family opt-in, because the one failure mode worth designing against is the
    silent one: rotated weights met by unrotated activations render wrong pixels and raise
    nothing.

    Returns the placed transformer, or None on any problem (missing / mismatched /
    unreadable checkpoint, unsupported meta-init, or a rotation this build cannot apply exactly)
    so the caller falls back to dense-quantise. Best-effort: never raises for an unavailable
    artifact.
    """
    try:
        # A request-supplied local path names arbitrary WEIGHTS, a different question from the
        # deserialization one below: allowlisted or not, the file is read weights_only.
        if source.kind == "path" and not _local_prequant_path_allowed(source.location):
            _warn(
                logger,
                f"{scheme}:path",
                RuntimeError(
                    "request-supplied local pre-quant path refused (loading arbitrary weights "
                    f"into the served model); set {ALLOW_LOCAL_PREQUANT_PATH_ENV} to an "
                    "allowlisted directory containing trusted checkpoints to permit it",
                ),
            )
            return None

        path = _resolve_checkpoint_path(
            source, hf_token, cache_dir, local_files_only = local_files_only
        )
        if path is None:
            return None

        # A torch.save pickle, deserialized under the constructor ALLOWLIST above and never as a
        # free-running one. First-party hosting is no reason to execute whatever bytes arrive: the
        # artifact is mutable, fetched over the network, and reached by loads that never asked for
        # one (auto resolves an unset precision to a hosted checkpoint), so a mutated file must
        # fail to load rather than run.
        ckpt = _torch_load_prequant(path, map_location = "cpu")
        if not _validate_checkpoint(
            ckpt, scheme, base, logger, min_features = min_features, fast_accum = fast_accum
        ):
            return None
        state_dict = ckpt["state_dict"]
        _pin_kernel_preference(state_dict, logger)

        # Read from the root that actually supplied the checkpoint: after a mid-session cache change
        # the pinned root may be gone or read-only, and load_config's raise is swallowed below into
        # a None return, silently dropping a prequant whose checkpoint is cached and already loaded.
        config = _load_transformer_config(
            transformer_cls,
            base,
            hf_token,
            cache_dir,
            path,
            config_subfolder,
            local_files_only = local_files_only,
        )
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

        # The ONLINE half of an activation rotation, applied here rather than in a family's
        # ``prepare_model`` hook so that no route can load a rotated checkpoint without it: the
        # offline half is already baked into the weights that were just assigned, and a rotated
        # weight met by an unrotated activation renders plausible garbage with nothing to catch.
        # A no-op for every artifact that declares no rotation, and a RAISE (caught below into the
        # dense fallback) for one this build cannot honour exactly. After load_state_dict because
        # the meta retry above rebuilds the module; before apply_small_m_padding because padding
        # reparents the Linears and the recorded fqns name the unwrapped tree.
        from .diffusion_convrot import apply_activation_rotation

        apply_activation_rotation(transformer, metadata, logger = logger)

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
    local_files_only: bool = False,
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
                    local_files_only = local_files_only,
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
        local_files_only = local_files_only,
    )


def _resolve_checkpoint_path(
    source: PrequantSource,
    hf_token: Optional[str],
    cache_dir: Optional[str] = None,
    *,
    local_files_only: bool = False,
) -> Optional[str]:
    """The local file path for ``source``, downloading from the Hub if needed; None if absent.

    ``local_files_only`` is the caller's promise that this load may not fetch anything, so a cache
    miss answers None and the build falls back rather than pulling several GB nobody asked for."""
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
                local_files_only = local_files_only,
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
                local_files_only = local_files_only,
            )
    return None


def _config_cache_roots(checkpoint_path: str, cache_dir: Optional[str]) -> tuple:
    """Cache roots to read the transformer config from, the checkpoint's OWN root first.

    ``_resolve_checkpoint_path`` may answer from huggingface_hub's import-time root even when Unsloth
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
    subfolder: str = "transformer",
    *,
    local_files_only: bool = False,
) -> Any:
    """``transformer_cls.load_config`` against the checkpoint's cache root, then the other one.

    The config is a few KB, but it is still a Hub fetch, and a load that promised to reach nothing
    has to keep that promise for the small files too."""
    last: Optional[BaseException] = None
    for root in _config_cache_roots(checkpoint_path, cache_dir):
        try:
            return transformer_cls.load_config(
                base,
                subfolder = subfolder,
                token = hf_token,
                cache_dir = root,
                local_files_only = local_files_only,
            )
        except Exception as exc:  # noqa: BLE001 — try the other root before giving up
            last = exc
    raise last  # type: ignore[misc]


def _fp8_activation_floor_present(state_dict: Any, logger: Any) -> bool:
    """True unless some fp8 tensor was quantised with no activation lower bound.

    Only the first quantised tensor is inspected: the builder applies one config to the whole
    module, so the floor is uniform. A state dict with no fp8 tensor at all is left to the other
    checks (an empty or wrong-scheme artifact is their business, not this one)."""
    from .diffusion_transformer_quant import TQ_FP8

    try:
        items = state_dict.items() if hasattr(state_dict, "items") else ()
        for name, tensor in items:
            kwargs = getattr(tensor, "act_quant_kwargs", None)
            if kwargs is None:
                continue
            if getattr(kwargs, "hp_value_lb", None):
                return True
            _warn(
                logger,
                TQ_FP8,
                ValueError(
                    f"fp8 checkpoint has no activation scale floor on {name!r} "
                    "(built before activation_value_lb); a zero activation row renders black. "
                    "Rebuild it"
                ),
            )
            return False
    except Exception:  # noqa: BLE001 -- an unreadable state dict is the other checks' problem
        return True
    return True


def _validate_activation_rotation(ckpt_format: Any, meta: Any, scheme: str, logger: Any) -> bool:
    """Reject a checkpoint whose activation rotation this build cannot honour EXACTLY.

    Three ways an artifact and a loader can disagree about the rotation, and all three end in the
    same place -- weights in a rotated basis multiplied by unrotated activations, which is finite,
    raises nothing, and renders quietly wrong -- so all three are refused here rather than
    discovered later:

      * the artifact declares a rotation and is tagged v1. Only v2 makes an Unsloth too old for this
        code refuse it, so a v1 tag on rotated weights is a hazard to every OTHER build, and the
        builder that produced it is not one to trust about anything else in the file;
      * the artifact is tagged v2 and declares none. Nothing here would rotate, and the tag says
        something was meant to;
      * the rotation is declared but its contract does not parse (an unknown kind, a group that is
        not a power of 4, an absent or malformed fqn list).

    Refusing costs a dense fallback: slower and bigger, never wrong."""
    from .diffusion_convrot import declares_rotation, rotation_metadata_error

    rotated = declares_rotation(meta)
    tagged = ckpt_format == PREQUANT_FORMAT_ROTATED
    if rotated != tagged:
        _warn(
            logger,
            scheme,
            ValueError(
                f"checkpoint format {ckpt_format!r} and its activation rotation disagree "
                f"(declares a rotation: {rotated}); a rotated checkpoint must be tagged "
                f"{PREQUANT_FORMAT_ROTATED!r} so older builds refuse it instead of running it "
                "unrotated"
            ),
        )
        return False
    problem = rotation_metadata_error(meta)
    if problem:
        _warn(logger, scheme, ValueError(problem))
        return False
    return True


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
    if not isinstance(ckpt, dict) or ckpt.get("format") not in PREQUANT_FORMATS:
        _warn(logger, scheme, ValueError("unrecognised pre-quant checkpoint format"))
        return False
    if "state_dict" not in ckpt:
        _warn(logger, scheme, ValueError("pre-quant checkpoint has no state_dict"))
        return False
    meta = ckpt.get("metadata") or {}
    if not _validate_activation_rotation(ckpt.get("format"), meta, scheme, logger):
        return False
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
    # fp8 also REQUIRES the activation scale floor, and this is checked on the loaded TENSORS, not
    # on metadata. torchao's per-row activation quantiser divides by each row's amax, so a zero row
    # (qwen's text stream emits them) gives scale 0 and NaN qdata unless activation_value_lb floors
    # it. That floor is serialised per tensor as act_quant_kwargs.hp_value_lb, so an artifact built
    # before the fix stays broken however it is loaded, and it predates any metadata field we could
    # stamp -- and "absent is accepted for back-compat", the convention every check above follows,
    # is exactly wrong here. Reading the tensors is fail-closed and needs no format bump.
    if scheme == TQ_FP8 and not _fp8_activation_floor_present(ckpt.get("state_dict"), logger):
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


def pin_prequantized_module(
    manager: Any,
    module: Any,
    device: Any,
    *,
    logger: Any = None,
    label: str = "pre-quantized denoiser",
) -> bool:
    """Keep a module resident on ``device``, out of a ComponentsManager's rotation.

    ``ComponentsManager.enable_auto_cpu_offload`` parks every component on the CPU and moves each
    one onto the accelerator inside its own ``pre_forward``, i.e. from within the block that is
    already executing. A torchao-quantized module does not survive that move: the device change
    reaches ``return_and_correct_aliasing``, which tries to alias a CPU storage to an accelerator
    tensor and raises ``Attempted to set the storage of a tensor on device "cuda:0" to a storage
    on different device "cpu"``, and MiniMax-H3's denoise loop dies on its first step. Moving the
    same module at load time, outside any executing block, works -- so the fix is to place it once
    here and take it out of the rotation rather than to move it per forward.

    That is also what a pre-quantized denoiser is for: the hosted H3 checkpoint is ~20 GB against
    66.3 GB dense, so keeping it resident is the saving being spent. The other components keep
    their hooks, and the strategy sizes its decisions from live free memory, so the encoder and
    the VAEs still offload around it.

    For a torchao module that placement is REQUIRED, for the reason above. A caller may also pin a
    plain dense module, where it is an optimisation instead: a module that moves per forward cannot
    be regionally compiled either, since the onload hooks wrap the forward the graph would replace.
    That caller owns the fit check (this function sizes nothing) and passes its own ``label``.

    Returns True when the module was pinned. Best-effort on the hook surgery: if the manager does
    not look the way this expects, the module is still placed on ``device`` and False is returned,
    which is the behaviour before pinning existed.
    """
    hooks = list(getattr(manager, "model_hooks", None) or ())
    target = next((hook for hook in hooks if getattr(hook, "model", None) is module), None)
    pinned = False
    if target is not None:
        try:
            # Drop the accelerate hook so no pre_forward/offload ever moves this module again ...
            target.remove()
            # ... and unlist it, so another component's pre_forward cannot pick it as the thing to
            # evict (which would move it to the CPU with no hook left to bring it back).
            for hook in hooks:
                others = getattr(getattr(hook, "hook", None), "other_hooks", None)
                if others:
                    hook.hook.other_hooks = [item for item in others if item is not target]
            manager.model_hooks = [hook for hook in hooks if hook is not target]
            pinned = True
        except Exception as exc:  # noqa: BLE001 -- placement below still has to happen
            _warn(logger, "pin:hook", exc)
    module.to(device)
    if logger is not None:
        logger.info(
            "diffusion.prequant: %s pinned on %s (offload rotation: %s)",
            label,
            device,
            "removed" if pinned else "unchanged",
        )
    return pinned


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
