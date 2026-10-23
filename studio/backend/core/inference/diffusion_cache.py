# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in step caching for the diffusion transformer (First-Block-Cache).

Once a DiT's trajectory settles its output changes little across steps, so most of the
transformer can be reused. FBCache computes the first block and, if its residual barely changed
from the previous step (within ``threshold``), skips the rest and reuses their cached output.
diffusers ships it natively (``transformer.enable_cache(FirstBlockCacheConfig(...))``).

Measured on Flux.1-dev (28 steps, 1024px, B200): ~1.4x on top of torch.compile (2.83 -> 2.03 s)
at LPIPS ~0.08 -- deep inside the speed-for-quality bar.

Explicit ``fbcache`` works on every tier; unset / ``auto`` engages only on ``max`` with 20+ steps.
It composes with torch.compile only at ``fullgraph=False`` (the cache's compiler-disabled decision
is a graph break), which the speed layer switches to automatically. Best-effort: an incompatible
model is caught and the load proceeds uncached. torch / diffusers imported lazily.
"""

from __future__ import annotations

from typing import Any, Optional

TC_OFF = "off"
TC_AUTO = "auto"
TC_FBCACHE = "fbcache"
# Fixed-schedule step skip (diffusion_step_skip.py); explicit, or auto for the measured families below.
TC_STATIC = "static"
TC_MODES = (TC_FBCACHE, TC_STATIC)

# Quantised transformers shift the residual distribution, so need a higher threshold.
DEFAULT_FBCACHE_THRESHOLD = 0.08
QUANT_FBCACHE_THRESHOLD = 0.12

# Auto step-count bar: FBCache's win scales with step count, so auto engages only at 20+ steps ("dev" schedules
# qualify, distilled turbo never does).
FBCACHE_MIN_STEPS = 20

# == diffusion_speed.SPEED_MAX, spelled out to keep this module import-free.
AUTO_STEP_CACHE_TIER = "max"

# == diffusion_speed.REDUCTION_FILTER_OPTION, spelled out to keep this module import-free.
_REDUCTION_FILTER_OPTION = "test_configs.force_filter_reduction_configs"


_TIER_DEFAULT = "default"
_TIER_MAX = "max"


# Per-model interval on each tier ("default" also covers max), measured vs the no-skip render (PR #12652): default
# needs mean LPIPS <= 0.05, max <= 0.10. "steps" = the step count it was measured at: fewer steps compute every step.
# Keyed by the UPSTREAM repo id so distilled / unmeasured siblings never match.
# The rows are ``FamilyProfile.auto_static_skip`` on the registry; the registries are torch-free, so this stays light.
def _auto_static_skip_table() -> dict:
    from .family_profile import profile_table
    return {
        repo.strip().lower(): dict(entry)
        for rows in profile_table("auto_static_skip").values()
        for repo, entry in rows
    }


AUTO_STATIC_SKIP: dict = _auto_static_skip_table()

AUTO_STATIC_SKIP_TIERS = (_TIER_MAX,)

# Load gate on the model's default steps (keeps distilled siblings out); explicit "static" keeps STATIC_MIN_STEPS (12).
AUTO_STATIC_MIN_STEPS = 20

# Kill switch: auto never picks static (explicit requests still honoured).
ENV_AUTO_STEP_SKIP = "UNSLOTH_DIFFUSION_AUTO_STEP_SKIP"


def auto_step_skip_disabled(env: Optional[dict] = None) -> bool:
    import os
    raw = (
        str((os.environ if env is None else env).get(ENV_AUTO_STEP_SKIP, "") or "").strip().lower()
    )
    return raw in ("0", "false", "off", "no", "none")


def auto_static_skip_entry(*identifiers: Optional[str]) -> Optional[dict]:
    """The AUTO_STATIC_SKIP row for the first identifier (repo id, then resolved base) naming a listed checkpoint."""
    from .diffusion_families import canonical_base

    for identifier in identifiers:
        if not identifier:
            continue
        entry = AUTO_STATIC_SKIP.get(canonical_base(str(identifier)).strip().lower())
        if entry is not None:
            return entry
    return None


def skip_tier(requested: Optional[str], effective: Optional[str]) -> Optional[str]:
    """The tier the skip policy reads: an explicit off / eager stays lossless even when quant forces a compile."""
    asked = str(requested or "").strip().lower()
    return asked if asked in ("off", "eager") else effective


def auto_static_skip_plan(
    identifiers: Any,
    speed_mode: Optional[str],
    default_steps: Optional[int],
    env: Optional[dict] = None,
) -> Optional[dict]:
    """Settings for an auto static skip (``{"every": n, "min_steps": m}``), or None when auto must not pick it.

    ``speed_mode`` is the tier the user ASKED for (an eager downgrade forced by offload does not change what the
    skip costs in quality). Only ``max`` skips automatically: the skip is lossy (mean LPIPS up to 0.05 on the
    ``default`` interval, 0.10 on ``max``, PR #12652) and ComfyUI never skips steps by default, so ``default``, an
    unset speed and the deferred profile render every step. An explicit ``static`` request is still honoured."""
    if auto_step_skip_disabled(env):
        return None
    if isinstance(identifiers, str) or identifiers is None:
        identifiers = (identifiers,)
    entry = auto_static_skip_entry(*identifiers)
    if not entry or speed_mode not in AUTO_STATIC_SKIP_TIERS:
        return None
    every = entry.get(_TIER_MAX)
    try:
        steps_ok = default_steps is not None and int(default_steps) >= AUTO_STATIC_MIN_STEPS
    except (TypeError, ValueError):
        steps_ok = False
    if not every or not steps_ok:
        return None
    return {
        "every": int(every),
        "min_steps": max(AUTO_STATIC_MIN_STEPS, int(entry.get("steps") or 0)),
    }


def auto_step_cache_allowed(speed_mode: Optional[str]) -> bool:
    """Takes the EFFECTIVE speed tier."""
    return speed_mode == AUTO_STEP_CACHE_TIER


def resolve_auto_step_cache(
    speed_mode: Optional[str],
    default_steps: int,
    *,
    static_plan: Optional[dict] = None,
) -> Optional[str]:
    """``static_plan`` (from auto_static_skip_plan) wins: a family measured for the fixed schedule takes it on every
    tier the plan allows, FBCache stays the max-tier fallback for the rest."""
    if static_plan:
        return TC_STATIC
    if auto_step_cache_allowed(speed_mode) and int(default_steps) >= FBCACHE_MIN_STEPS:
        return TC_FBCACHE
    return None


def cache_breaks_graph(mode: Optional[str]) -> bool:
    """Whether an engaged mode decides inside the forward (FBCache), costing fullgraph and the CUDA graph."""
    return bool(mode) and mode != TC_STATIC


def normalize_transformer_cache(value: Optional[str]) -> Optional[str]:
    """Lower/strip a cache mode; None / "" / "none" / "off" -> None, "auto" -> TC_AUTO (loader
    decides from step count). Raises ValueError for an unsupported value."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized or normalized in ("none", "off"):
        return None
    if normalized == TC_AUTO:
        return TC_AUTO
    if normalized not in TC_MODES:
        raise ValueError(
            f"Unsupported transformer_cache '{value}'. Use one of: off, auto, "
            f"{', '.join(TC_MODES)}."
        )
    return normalized


# (hidden_states idx, encoder idx or None) per block class; missing entry silently disables caching.
_UNREGISTERED_BLOCK_METADATA: dict = {
    (
        "diffusers.models.transformers.transformer_qwenimage21",
        "QwenImage21TransformerBlock",
    ): (0, None),
}


def register_unregistered_transformer_blocks(logger: Any = None) -> tuple:
    """Fills FBCache metadata for unregistered blocks; never overwrites, since upstream is authoritative."""
    added: list = []
    try:
        from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
    except Exception:  # noqa: BLE001 - an older diffusers has no registry to fill
        return ()
    for (module_name, class_name), (
        hidden_index,
        encoder_index,
    ) in _UNREGISTERED_BLOCK_METADATA.items():
        try:
            import importlib
            block_cls = getattr(importlib.import_module(module_name), class_name, None)
        except Exception:  # noqa: BLE001 - this diffusers does not ship the family
            continue
        if block_cls is None:
            continue
        try:
            TransformerBlockRegistry.get(block_cls)
            continue  # already known, ours would be a downgrade
        except Exception:  # noqa: BLE001 - "not registered" is the case we are here for
            pass
        try:
            TransformerBlockRegistry.register(
                model_class = block_cls,
                metadata = TransformerBlockMetadata(
                    return_hidden_states_index = hidden_index,
                    return_encoder_hidden_states_index = encoder_index,
                ),
            )
            added.append(class_name)
        except Exception as exc:  # noqa: BLE001 - registration is an optimisation, never a gate
            if logger is not None:
                logger.debug("could not register %s for step caching: %s", class_name, exc)
    return tuple(added)


def _invalidate_child_registry_cache(transformer: Any) -> None:
    """Clears the cached child-registry list, otherwise a cached forward dies: No context is set."""
    registry = getattr(transformer, "_diffusers_hook", None)
    if registry is not None and getattr(registry, "_child_registries_cache", None) is not None:
        try:
            registry._child_registries_cache = None
        except Exception:  # noqa: BLE001 -- diffusers internals moved; leave as-is
            pass


# diffusers cache hook names whose compute branch we re-point at a compiled inner forward (leader = measuring first
# block, block = the rest); both share the fn_ref layout.
_CACHE_HOOK_NAMES = (
    "mag_cache_leader_block_hook",
    "mag_cache_block_hook",
    "fbc_leader_block_hook",
    "fbc_block_hook",
)


def _compile_hooked_block_inners(transformer: Any, logger: Any = None) -> int:
    """Compiles computed steps of cache-hooked blocks, skip stays eager; stacked hook chains left alone."""
    try:
        import torch
    except Exception:  # noqa: BLE001 -- no torch, nothing to arm
        return 0
    armed = 0
    try:
        for module in transformer.modules():
            registry = getattr(module, "_diffusers_hook", None)
            if registry is None or getattr(module, "_compiled_call_impl", None) is None:
                continue
            hooks = getattr(registry, "hooks", None) or {}
            for name in _CACHE_HOOK_NAMES:
                hook = hooks.get(name)
                fn_ref = getattr(hook, "fn_ref", None) if hook is not None else None
                orig = getattr(fn_ref, "original_forward", None)
                if orig is None or getattr(hook, "_unsloth_orig_inner", None) is not None:
                    continue
                if getattr(orig, "__self__", None) is None:
                    continue  # not a plain bound method; arming would miss the block
                # fullgraph False, dynamic True: the cache decision graph-breaks; Dynamo caches per code object.
                dynamic = None if getattr(transformer, "_unsloth_auto_dynamic", False) else True
                # Same reduction-config filter as the blocks (diffusion_speed.pin_reduction_configs).
                block_kwargs = getattr(transformer, "_unsloth_regional_compile_kwargs", None) or {}
                pinned = (block_kwargs.get("options") or {}).get(_REDUCTION_FILTER_OPTION)
                extra = {"options": {_REDUCTION_FILTER_OPTION: True}} if pinned else {}
                compiled = torch.compile(orig, fullgraph = False, dynamic = dynamic, **extra)
                # Same runtime fallback as the block's own compile: a lowering failure on the first computed step
                # restores the eager inner instead of failing the render.
                guard = getattr(transformer, "_unsloth_compile_guard", None)
                if guard is not None:

                    def restore(ref: Any = fn_ref, inner: Any = orig) -> None:
                        ref.original_forward = inner

                    guard.restores.append(restore)
                    compiled = guard.wrap(compiled, orig, transformer)
                fn_ref.original_forward = compiled
                hook._unsloth_orig_inner = orig
                armed += 1
    except Exception as exc:  # noqa: BLE001 -- best-effort: the cache still works eager
        _warn(logger, "cache-hook inner compile", exc)
        return armed
    if armed and logger is not None:
        logger.info(
            "diffusion.cache: %d cache-hooked block(s) armed with compiled inner forwards",
            armed,
        )
    return armed


def _unhook_first_block_cache(transformer: Any) -> bool:
    """Removes FBCache hooks directly; False means cannot verify, so the caller must keep the marker."""
    # Another CacheMixin cache (MagCache/PAB) is not ours to tear down; config class matched by name.
    config = getattr(transformer, "_cache_config", None)
    if config is not None and type(config).__name__ != "FirstBlockCacheConfig":
        return False
    try:
        from diffusers.hooks import HookRegistry
        from diffusers.hooks.first_block_cache import _FBC_BLOCK_HOOK, _FBC_LEADER_BLOCK_HOOK
    except Exception:  # noqa: BLE001 - private names; an unknown layout means we cannot verify
        return False
    try:
        registry = HookRegistry.check_if_exists_or_initialize(transformer)
        registry.remove_hook(_FBC_LEADER_BLOCK_HOOK, recurse = True)
        registry.remove_hook(_FBC_BLOCK_HOOK, recurse = True)
        # Left dangling by a partial engage, and it is what diffusers keys every later call on.
        transformer._cache_config = None
    except Exception:  # noqa: BLE001
        return False
    return True


def _first_block_cache_is_hooked(transformer: Any) -> bool:
    """Checked before engaging, so a failure can tell our half-finished install from another cache."""
    try:
        from diffusers.hooks.first_block_cache import _FBC_BLOCK_HOOK, _FBC_LEADER_BLOCK_HOOK
        names = (_FBC_LEADER_BLOCK_HOOK, _FBC_BLOCK_HOOK)
        for module in transformer.modules():
            registry = getattr(module, "_diffusers_hook", None)
            hooks = getattr(registry, "hooks", None) or {} if registry is not None else {}
            if any(name in hooks for name in names):
                return True
    except Exception:  # noqa: BLE001 - private names, or not a torch module: cannot tell
        return False
    return False


def _restore_hooked_block_inners(transformer: Any) -> None:
    """Must run before disable_cache, or a stale compiled wrapper is left on the uncached forward path."""
    try:
        modules = list(transformer.modules())
    except Exception:  # noqa: BLE001 -- not a torch module (tests/fakes): nothing armed
        return
    for module in modules:
        registry = getattr(module, "_diffusers_hook", None)
        if registry is None:
            continue
        hooks = getattr(registry, "hooks", None) or {}
        for name in _CACHE_HOOK_NAMES:
            hook = hooks.get(name)
            orig = getattr(hook, "_unsloth_orig_inner", None) if hook is not None else None
            if orig is None:
                continue
            try:
                hook.fn_ref.original_forward = orig
                hook._unsloth_orig_inner = None
            except Exception:  # noqa: BLE001 -- per-hook best-effort
                pass


def _pipeline_opens_cache_context(pipe: Any) -> bool:
    """Reads __call__ source for a cache_context block, since FBCache needs one; False when unreadable."""
    import inspect

    call = getattr(pipe, "__call__", None)
    if call is None:
        return False
    try:
        src = inspect.getsource(call)
    except (OSError, TypeError):
        return False
    # Match the call `cache_context(` (the paren avoids a false positive on prose).
    return "cache_context(" in src


def _transformer_blocks_registered(transformer: Any, logger: Any = None) -> bool:
    """True when the registry cannot be read, so enable_cache stays the judge."""
    try:
        import torch
        from diffusers.hooks._common import _ALL_TRANSFORMER_BLOCK_IDENTIFIERS
        from diffusers.hooks._helpers import TransformerBlockRegistry
    except Exception:  # noqa: BLE001 - no registry to ask
        return True
    named_children = getattr(transformer, "named_children", None)
    if not callable(named_children):
        return True
    register_unregistered_transformer_blocks(logger)
    _register_layer_split_blocks(transformer)
    blocks = [
        getattr(block, "_orig_mod", block)
        for name, child in named_children()
        if name in _ALL_TRANSFORMER_BLOCK_IDENTIFIERS and isinstance(child, torch.nn.ModuleList)
        for block in child
    ]
    # FBCache needs a head and a tail block.
    if len(blocks) < 2:
        return False
    try:
        for cls in {type(block) for block in blocks}:
            TransformerBlockRegistry.get(cls)
    except Exception:  # noqa: BLE001 - unregistered, or the registry itself failed to load
        return False
    return True


def step_cache_supported(pipe: Any, *, logger: Any = None) -> bool:
    """Mirrors apply_step_cache's refusals for an auto request, without engaging the cache."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is None or not callable(getattr(transformer, "enable_cache", None)):
        return False
    if not _pipeline_opens_cache_context(pipe):
        return False
    if _reuses_prefix_kv(pipe, transformer):
        return False
    return _transformer_blocks_registered(transformer, logger)


def install_fbcache_length_guard() -> bool:
    """FBCache recomputes on a sequence-length change instead of raising; process-wide and idempotent."""
    try:
        from diffusers.hooks.first_block_cache import FBCHeadBlockHook
    except Exception:  # noqa: BLE001 - no FBCache in this diffusers
        return False
    original = getattr(FBCHeadBlockHook, "_should_compute_remaining_blocks", None)
    if original is None:
        return False
    if getattr(original, "_unsloth_length_guard", False):
        return True
    import torch

    @torch.compiler.disable
    def _should_compute_remaining_blocks(self, hidden_states_residual):
        state = self.state_manager.get_state()
        previous = state.head_block_residual
        if previous is not None and previous.shape != hidden_states_residual.shape:
            return True
        tail = state.tail_block_residuals
        if tail is not None and getattr(tail[0], "shape", None) not in (
            None,
            hidden_states_residual.shape,
        ):
            return True
        return original(self, hidden_states_residual)

    _should_compute_remaining_blocks._unsloth_length_guard = True
    FBCHeadBlockHook._should_compute_remaining_blocks = _should_compute_remaining_blocks
    return True


def _register_layer_split_blocks(transformer: Any) -> bool:
    """Register a layer split's block classes (same-named subclasses of the real blocks) with their base's metadata,
    so the cache finds them. A copy, never the base's own object: ``register`` writes ``_cls`` into it. True when
    every split block class is registered."""
    try:
        import dataclasses

        from diffusers.hooks._helpers import TransformerBlockRegistry
    except Exception:  # noqa: BLE001 - no registry: enable_cache decides
        return True
    ok = True
    for cls in {type(m) for m in transformer.modules()}:
        base = cls.__dict__.get("_unsloth_layer_split_base")
        if base is None:
            continue
        try:
            TransformerBlockRegistry.get(cls)
            continue
        except Exception:  # noqa: BLE001 - not registered yet
            pass
        try:
            meta = TransformerBlockRegistry.get(base)
        except Exception:  # noqa: BLE001 - the base is not a cacheable block either
            continue
        try:
            TransformerBlockRegistry.register(
                model_class = cls,
                metadata = dataclasses.replace(meta, _cls = None, _cached_parameter_indices = None),
            )
        except Exception:  # noqa: BLE001
            ok = False
    return ok


def _tensors_to(obj: Any, device: Any) -> Any:
    if isinstance(obj, (list, tuple)):
        return type(obj)(_tensors_to(x, device) for x in obj)
    to = getattr(obj, "to", None)
    return to(device) if callable(to) else obj


def _first_device(obj: Any) -> Any:
    if isinstance(obj, (list, tuple)):
        for x in obj:
            dev = _first_device(x)
            if dev is not None:
                return dev
        return None
    return getattr(obj, "device", None)


def install_fbcache_split_guard() -> bool:
    """Make First-Block-Cache's tail hook work when the tail block runs on another card than the head block (a layer
    split): the head block's output is copied to the tail's card for the residual, and the residual goes back to the
    head's card, where the head hook adds it on a cached step. Copies are exact, so the cache decides and renders as
    on one card. Calls whose tensors share a card take the original path unchanged. Process-wide and idempotent;
    must run before ``enable_cache`` (diffusers binds ``new_forward`` when it registers the hook). False when this
    diffusers has no FBCache block hook."""
    try:
        from diffusers.hooks.first_block_cache import FBCBlockHook
    except Exception:  # noqa: BLE001 - no FBCache in this diffusers
        return False
    install_fbcache_mixed_stream_tail()
    original = getattr(FBCBlockHook, "new_forward", None)
    if original is None:
        return False
    if getattr(original, "_unsloth_split_guard", False):
        return True

    def new_forward(self, module, *args, **kwargs):
        if not getattr(self, "is_tail", False):
            return original(self, module, *args, **kwargs)
        state = self.state_manager.get_state()
        head = state.head_block_output
        home = _first_device(head)
        try:
            here = self._metadata._get_parameter_from_args_kwargs(
                "hidden_states", args, kwargs
            ).device
        except Exception:  # noqa: BLE001 - the original raises the same way
            here = None
        if not state.should_compute or home is None or here is None or here == home:
            return original(self, module, *args, **kwargs)
        state.head_block_output = _tensors_to(head, here)
        try:
            out = original(self, module, *args, **kwargs)
        finally:
            state.head_block_output = head
        residuals = state.tail_block_residuals
        if residuals is not None:
            state.tail_block_residuals = tuple(
                r.to(home) if r is not None else None for r in residuals
            )
        return out

    new_forward._unsloth_split_guard = True
    new_forward._unsloth_mixed_stream_tail = getattr(original, "_unsloth_mixed_stream_tail", False)
    FBCBlockHook.new_forward = new_forward
    return True


def _split_mixed_stream_tail_residuals(tail_hidden: Any, head_output: Any) -> Optional[tuple]:
    """Tail residuals for a dual-stream head and a single-stream tail, or None when unmappable.

    The tail ran over ``cat([encoder, hidden], dim=1)``; splitting at the text length gives
    residuals shaped like the head's ``[hidden, encoder]`` outputs, which the skip branch adds to.
    """
    head_hidden, head_encoder = head_output[0], head_output[1]
    if head_encoder is None or getattr(tail_hidden, "ndim", 0) < 2:
        return None
    text_len = head_encoder.shape[1]
    if (
        tail_hidden.shape[1] != text_len + head_hidden.shape[1]
        or tail_hidden.shape[:1] != head_hidden.shape[:1]
        or tail_hidden.shape[2:] != head_hidden.shape[2:]
        or head_encoder.shape[2:] != head_hidden.shape[2:]
    ):
        return None
    return (
        tail_hidden[:, text_len:] - head_hidden,
        tail_hidden[:, :text_len] - head_encoder,
    )


def _rebind_fbcache_block_hooks(transformer: Any) -> None:
    """Point already-registered FBCache block hooks at the current ``FBCBlockHook.new_forward``.

    ``HookRegistry.register_hook`` keeps ``partial(hook.new_forward, module)``, the method as it was
    at registration, so hooks installed before ``install_fbcache_mixed_stream_tail`` (an FBCache we
    adopt instead of re-enabling) would still run the stock tail and crash FLUX.2. Best-effort."""
    try:
        import functools

        from diffusers.hooks.first_block_cache import _FBC_BLOCK_HOOK, FBCBlockHook
    except Exception:  # noqa: BLE001 - no FBCache in this diffusers
        return
    current = getattr(FBCBlockHook, "new_forward", None)
    if current is None:
        return
    for module in transformer.modules():
        registry = getattr(module, "_diffusers_hook", None)
        hooks = getattr(registry, "hooks", None) if registry is not None else None
        hook = hooks.get(_FBC_BLOCK_HOOK) if isinstance(hooks, dict) else None
        fn_ref = getattr(hook, "fn_ref", None) if hook is not None else None
        bound = getattr(fn_ref, "forward", None)
        if not isinstance(hook, FBCBlockHook) or not isinstance(bound, functools.partial):
            continue
        if getattr(bound.func, "__func__", None) is current:
            continue
        try:
            fn_ref.forward = functools.update_wrapper(
                functools.partial(hook.new_forward, *bound.args), hook.new_forward
            )
        except Exception:  # noqa: BLE001 - diffusers internals moved; leave this hook as it is
            pass


def install_fbcache_mixed_stream_tail() -> bool:
    """Make First-Block-Cache work when the head block is dual-stream and the tail single-stream.

    diffusers' tail hook picks its branch from the TAIL's output type, so on FLUX.2 (dual-stream
    head, single-stream tail) it subtracts the head's ``[hidden, encoder]`` list from a tensor and
    crashes on the first step. Only that combination takes the new path; every other family calls
    the original hook unchanged. An unmappable tail drops the head residual so the next step
    computes in full. Process-wide and idempotent; False when diffusers has no FBCache block hook."""
    try:
        from diffusers.hooks.first_block_cache import FBCBlockHook
    except Exception:  # noqa: BLE001 - no FBCache in this diffusers
        return False
    original = getattr(FBCBlockHook, "new_forward", None)
    if original is None:
        return False
    if getattr(original, "_unsloth_mixed_stream_tail", False):
        return True

    def new_forward(self, module, *args, **kwargs):
        metadata = self._metadata
        if (
            not self.is_tail
            or getattr(metadata, "return_encoder_hidden_states_index", 0) is not None
        ):
            return original(self, module, *args, **kwargs)
        shared_state = self.state_manager.get_state()
        head_output = shared_state.head_block_output
        if not shared_state.should_compute or not isinstance(head_output, (list, tuple)):
            return original(self, module, *args, **kwargs)
        output = self.fn_ref.original_forward(*args, **kwargs)
        tail_hidden = (
            output[metadata.return_hidden_states_index]
            if isinstance(output, (list, tuple))
            else output
        )
        residuals = _split_mixed_stream_tail_residuals(tail_hidden, head_output)
        if residuals is None:
            # No skip may use these residuals: with no head residual the next step computes.
            shared_state.head_block_residual = None
        shared_state.tail_block_residuals = residuals
        return output

    new_forward._unsloth_mixed_stream_tail = True
    new_forward._unsloth_original = original
    FBCBlockHook.new_forward = new_forward
    return True


def _reuses_prefix_kv(pipe: Any, transformer: Any) -> bool:
    """True when forward takes kv_cache_mode (prefix-KV reuse); errs toward True, costing only speed."""
    import inspect

    forward = getattr(transformer, "forward", None)
    if forward is None:
        return False
    try:
        if "kv_cache_mode" not in inspect.signature(forward).parameters:
            return False
    except (TypeError, ValueError):  # not introspectable: assume no prefix reuse
        return False
    call = getattr(pipe, "__call__", None)
    if call is None:
        return False
    try:
        src = inspect.getsource(call)
    except (OSError, TypeError):
        # Unknown is the crashing side, so treat it as driven.
        return True
    return "kv_cache_mode" in src


def apply_step_cache(
    pipe: Any,
    *,
    mode: Optional[str],
    threshold: Optional[float] = None,
    quant_active: bool = False,
    length_changes_ok: bool = False,
    logger: Any = None,
) -> Optional[str]:
    """Returns mode engaged or None; length_changes_ok only on explicit request (prefix-KV skips more)."""
    mode = normalize_transformer_cache(mode)
    if mode is None or mode == TC_AUTO:
        # AUTO is resolved by the loader before this; treat a stray auto as off.
        return None
    if mode == TC_STATIC:
        _warn(logger, mode, RuntimeError("static step skip is not supported on this backend"))
        return None
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return None
    from .diffusion_layer_split import is_layer_split

    if is_layer_split(transformer) and not (
        install_fbcache_split_guard() and _register_layer_split_blocks(transformer)
    ):
        # without the guard the tail hook subtracts the head block's output (primary) from its own (another card)
        _warn(logger, mode, RuntimeError("denoiser is split across cards; running uncached"))
        return None
    # Before enable_cache, which is what raises on a block class the registry has never seen.
    register_unregistered_transformer_blocks(logger)
    # FLUX.2's dual-stream head + single-stream tail crashes the stock tail hook on the first step.
    install_fbcache_mixed_stream_tail()
    thr = (
        threshold
        if threshold is not None
        else (QUANT_FBCACHE_THRESHOLD if quant_active else DEFAULT_FBCACHE_THRESHOLD)
    )
    # The lower-level hook installs on non-CacheMixin transformers and crashes generation.
    enable_cache = getattr(transformer, "enable_cache", None)
    if not callable(enable_cache):
        _warn(logger, mode, RuntimeError("transformer has no cache_context (not a CacheMixin)"))
        return None
    # The hook raises "No context is set" unless the pipeline opens cache_context.
    if not _pipeline_opens_cache_context(pipe):
        _warn(
            logger, mode, RuntimeError("pipeline __call__ opens no cache_context; running uncached")
        )
        return None
    # Prefix-KV shortens the sequence after step 1: refuse before enable_cache, or it fails at step 2.
    if _reuses_prefix_kv(pipe, transformer) and not (
        length_changes_ok and install_fbcache_length_guard()
    ):
        _warn(
            logger,
            mode,
            RuntimeError(
                "transformer reuses a prefix KV cache, so the block sequence length changes "
                "after the first step; running uncached"
            ),
        )
        return None
    # enable_cache raises when already enabled; a redundant call falls into recovery and loses the cache.
    if getattr(transformer, "is_cache_enabled", False):
        prior = getattr(transformer, "_unsloth_step_cache", None)
        live = getattr(transformer, "_cache_config", None)
        # Only the live config authorises the no-op, never our marker.
        if (
            type(live).__name__ == "FirstBlockCacheConfig"
            and getattr(live, "threshold", None) == thr
        ):
            # May not be a cache we installed, so the post-enable integration cannot be assumed
            # done. Both idempotent; skipping the first strands the child-registry list.
            _invalidate_child_registry_cache(transformer)
            _rebind_fbcache_block_hooks(transformer)
            _compile_hooked_block_inners(transformer, logger)
            try:
                transformer._unsloth_step_cache = f"{mode}@{thr}"
            except Exception:  # noqa: BLE001 - marker is best-effort
                pass
            return mode
        try:
            _restore_hooked_block_inners(transformer)
            transformer.disable_cache()
        except Exception as exc:  # noqa: BLE001 - cannot re-configure -> finish the teardown ourselves
            # disable_cache can raise between hook removals, leaving is_cache_enabled True.
            removed = _unhook_first_block_cache(transformer)
            if not removed:
                # Cannot verify, so report what was last known engaged and leave the marker alone.
                _warn(logger, mode, exc)
                return prior.split("@")[0] if isinstance(prior, str) else None
            # Fully unhooked, so drop the stale marker and engage at the new settings.
            try:
                transformer._unsloth_step_cache = None
            except Exception:  # noqa: BLE001 - marker is best-effort
                pass
    # Before the engage: afterwards partial and foreign installs look identical.
    hooked_before = _first_block_cache_is_hooked(transformer)
    try:
        try:
            from diffusers import FirstBlockCacheConfig
        except ImportError:  # older diffusers exports it only from diffusers.hooks
            from diffusers.hooks import FirstBlockCacheConfig

        config = FirstBlockCacheConfig(threshold = thr)
        enable_cache(config)
        # Stale child-registry list would hide the new hooks; must follow every enable_cache.
        _invalidate_child_registry_cache(transformer)
        # The stock merge block, stripped line by line with blanks dropped; its four pieces must keep their order, so the match is exact, not a needle search
        _compile_hooked_block_inners(transformer, logger)
        try:
            transformer._unsloth_step_cache = f"{mode}@{thr}"
        except Exception:  # noqa: BLE001 - marker is best-effort
            pass
        if logger is not None:
            logger.info("diffusion.cache: %s engaged (threshold=%s)", mode, thr)
        return mode
    except Exception as exc:  # noqa: BLE001 - incompatible model -> run uncached
        if hooked_before:
            # Low-level FBCache install: register_hook refused the duplicate, so hooks are intact.
            _invalidate_child_registry_cache(transformer)
            _rebind_fbcache_block_hooks(transformer)
            _compile_hooked_block_inners(transformer, logger)
            # Marker before success: it keeps the CUDA graph wrapper eager over live hooks.
            try:
                transformer._unsloth_step_cache = mode
            except Exception:  # noqa: BLE001 - marker is best-effort
                pass
            _warn(logger, mode, exc)
            return mode
        # enable_cache can fail part-hooked; restore armed compiled inners FIRST
        _restore_hooked_block_inners(transformer)
        disabled = True
        try:
            transformer.disable_cache()
        except Exception:  # noqa: BLE001 - _unhook_first_block_cache below is what actually decides
            disabled = False
        # enable_cache assigns _cache_config LAST, so a partial raise reads as not caching.
        del disabled
        removed = _unhook_first_block_cache(transformer)
        # Marker tracks the hooks, not intent: clear only once they are known gone.
        if removed:
            try:
                transformer._unsloth_step_cache = None
            except Exception:  # noqa: BLE001 - marker is best-effort
                pass
        _warn(logger, mode, exc)
        return None


def effective_denoise_steps(steps: int, strength: Optional[float]) -> int:
    """Steps actually denoised under strength, floored (not rounded); auto step-cache keys on this."""
    s = int(steps)
    if strength is None or float(strength) >= 1.0:
        return s
    return max(1, min(int(s * float(strength)), s))


def effective_request_strength(
    request_strength: Optional[float],
    has_init_image: bool,
    pipe_accepts_strength: bool,
    pipe_default_strength: Any,
) -> Optional[float]:
    """The strength the pipe applies, including its own signature default when the request omits it."""
    if not (has_init_image and pipe_accepts_strength):
        return None
    if request_strength is not None:
        return request_strength
    return pipe_default_strength if isinstance(pipe_default_strength, (int, float)) else None


def auto_step_cache_mode(steps: int) -> Optional[str]:
    """The cache mode an AUTO decision runs for ``steps`` (before the runner check): FBCache at
    ``FBCACHE_MIN_STEPS``+, else uncached. The single rule ``maybe_toggle_step_cache`` applies."""
    return TC_FBCACHE if int(steps) >= FBCACHE_MIN_STEPS else None


def maybe_toggle_step_cache(
    pipe: Any,
    *,
    steps: int,
    quant_active: bool = False,
    threshold: Optional[float] = None,
    logger: Any = None,
) -> Optional[str]:
    """Generation-time enable/disable for an AUTO cache decision, keyed on the step count: engage
    FBCache at ``FBCACHE_MIN_STEPS``+, else uncached. Idempotent (``_unsloth_step_cache`` marker),
    so per-generation calls are cheap. Only the auto path calls this. Returns the active mode."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return None
    engaged = getattr(transformer, "_unsloth_step_cache", None)
    want = auto_step_cache_mode(steps) is not None
    if want and not engaged:
        return apply_step_cache(
            pipe,
            mode = TC_FBCACHE,
            threshold = threshold,
            quant_active = quant_active,
            logger = logger,
        )
    if not want and engaged:
        disable_cache = getattr(transformer, "disable_cache", None)
        if callable(disable_cache):
            # Read before teardown: disable_cache clears it.
            ours = (
                type(getattr(transformer, "_cache_config", None)).__name__
                == "FirstBlockCacheConfig"
            )
            try:
                # Restore before the hooks splice original_forward back, so compiled wrappers do not leak.
                _restore_hooked_block_inners(transformer)
                disable_cache()
                # disable_cache removes nothing for an adopted low-level cache (_cache_config None).
                if not _unhook_first_block_cache(transformer) and not ours:
                    _warn(
                        logger,
                        "fbcache disable",
                        RuntimeError("cache hooks could not be verified removed"),
                    )
                    return TC_FBCACHE
                transformer._unsloth_step_cache = None
                if logger is not None:
                    logger.info(
                        "diffusion.cache: fbcache disengaged (auto: %s steps < %s)",
                        steps,
                        FBCACHE_MIN_STEPS,
                    )
                return None
            except Exception as exc:  # noqa: BLE001 - keep the cache rather than crash
                _warn(logger, "fbcache disable", exc)
        return TC_FBCACHE
    return TC_FBCACHE if engaged else None


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.cache: %s unavailable (%s); running uncached", what, exc)
