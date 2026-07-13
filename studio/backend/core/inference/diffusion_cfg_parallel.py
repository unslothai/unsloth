# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dual-GPU CFG branch parallelism for guider-driven video pipelines.

A CFG denoise step runs the SAME DiT twice (once per guidance branch) and the diffusers
modular pipelines (HunyuanVideo-1.5) run those branches sequentially, consuming both
predictions only after the loop in ``self.guider(...)``. On a multi-GPU host the second
GPU is idle, so routing one branch to a full DiT replica there almost halves the denoise
wall time: measured on 2x B200 (HunyuanVideo-1.5-720p, 1280x720/33f/30 steps, trim + cuDNN
+ regional compile) 1.77x e2e uncached and 1.72x with the auto MagCache -- and the output
is BIT-IDENTICAL to the single-GPU run (final-latent max abs diff 0.0, byte-identical
frames), because the replica runs the identical checkpoint/kernels on an identical device.

Mechanism (no pipeline fork): a proxy replaces ``pipe.transformer``. The pipeline opens
``transformer.cache_context(name)`` before each branch, telling the proxy WHICH branch is
dispatched; ``pred_cond`` runs on the replica via a persistent worker thread (returning a
placeholder immediately), the other runs on the primary, and a patched ``guider.forward``
resolves the placeholder -- issuing the replica -> primary copy only after BOTH branches'
kernels are queued, so the copy is the only cross-device sync. Thread dispatch (not async
CUDA) is required because the MagCache hooks' ``.item()`` skip decisions block the CPU.

Accuracy policy: output is BIT-IDENTICAL only when both branches run the same kernels --
the EAGER stack, verified to the byte (cache on or off; cudnn.benchmark either way). Under
regional compile the two devices' separately compiled artifacts differ by ~1 bf16 ulp/step
(a reduction kernel with a different summation order), and a 30-step trajectory amplifies
it chaotically (mean ~12/255 per-pixel delta, composition/luma preserved). So AUTO engages
only on an eager-tier load, where eager+parallel (~0.85 s/step) even beats the compiled
sequential default (~1.09 s/step) at better accuracy; the compiled stack needs an explicit
"on" accepting the fp-noise divergence for the 1.72x. Generations that may (re)compile
either module (first run, cache toggle, shape change) are dispatched inline: dynamo patches
``Module.__call__`` process-wide, so a concurrent replica compile corrupts or crashes.

Two process-global thread-safety landmines this module also handles:
- diffusers' ``_native_cudnn`` backend enters the process-global ``sdpa_kernel(...)``
  context per call; two threads racing its save/restore can leave the process stuck
  cudnn-only (the VAE's mask-carrying SDPA then dies "No available kernel"). While engaged,
  the backend is swapped for a direct ``aten._scaled_dot_product_cudnn_attention`` call
  (the exact kernel the composite dispatches to, so numerics are unchanged), restored on
  teardown.
- the inline-dispatch compile serialization above.

Best-effort throughout: any gate/build failure leaves the pipe single-device, reported in
the resolved record. torch / diffusers imported lazily.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from typing import Any, Optional

CFG_PARALLEL_OFF = "off"
CFG_PARALLEL_AUTO = "auto"
CFG_PARALLEL_ON = "on"
CFG_PARALLEL_MODES = (CFG_PARALLEL_OFF, CFG_PARALLEL_AUTO, CFG_PARALLEL_ON)

# Families the AUTO policy may engage on: the guider-driven HunyuanVideo-1.5 pipelines,
# where bit-identity and the 1.7x win were measured. An explicit "on" skips this list.
_CFG_PARALLEL_FAMILY_ALLOW = frozenset({"hunyuanvideo-1.5", "hunyuanvideo-1.5-720p"})

# Secondary-device VRAM bar: replica weight bytes plus activation/workspace headroom
# (measured replica-device peak 17.4 GB for the 15.6 GB bf16 HV15 DiT at 720p/33f, ~1.8 GB
# activations; the margin also covers the CUDA context + cudnn workspaces). A co-tenant can
# grab the GPU between check and load, so the build stays best-effort regardless.
_REPLICA_HEADROOM_BYTES = int(4.5 * (1 << 30))


def normalize_cfg_parallel(value: Optional[str]) -> str:
    """Lower/strip a requested cfg_parallel mode; None / "" -> auto (the gate decides).
    Raises ValueError for an unsupported value."""
    if value is None:
        return CFG_PARALLEL_AUTO
    normalized = str(value).strip().lower()
    if not normalized or normalized == CFG_PARALLEL_AUTO:
        return CFG_PARALLEL_AUTO
    if normalized in ("none",):
        return CFG_PARALLEL_OFF
    if normalized not in CFG_PARALLEL_MODES:
        raise ValueError(
            f"Unsupported cfg_parallel '{value}'. Use one of: {', '.join(CFG_PARALLEL_MODES)}."
        )
    return normalized


class _PendingPred:
    """Placeholder for a branch prediction still being produced by the worker thread. The
    pipeline stores ``self.transformer(...)[0]`` per branch and reads it only in the guider
    combine, so ``(pending,)`` satisfies the unwrap; the patched guider forward resolves it,
    issuing the cross-device copy from the MAIN thread so it is stream-ordered after the
    other branch's kernels."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.value: Any = None
        self.error: Optional[BaseException] = None

    def resolve(self, device: Any) -> Any:
        self.event.wait()
        if self.error is not None:
            raise self.error
        v = self.value
        return v.to(device, non_blocking = True) if v.device != device else v


class _ReplicaView:
    """Present the replica as ``pipe.transformer`` to the lever helpers (trim / attention
    backend / regional compile), like video.py's ``_SecondDiTView`` for an MoE second
    expert. Everything else reads through to the real pipe."""

    def __init__(self, pipe: Any, replica: Any) -> None:
        object.__setattr__(self, "_pipe", pipe)
        object.__setattr__(self, "transformer", replica)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_pipe"), name)


class CFGParallelProxy:
    """Stands in for ``pipe.transformer``: routes ``pred_cond`` to the replica DiT on the
    secondary device while the other branch runs on the primary; every other attribute
    delegates to the primary. Cache mutations fan out to BOTH modules so per-branch cache
    state matches the single-GPU run.

    ``enabled`` / ``dispatch`` are resolved per generation by ``plan_generation``; with
    ``enabled`` False the proxy is a pure passthrough (the sequential path)."""

    def __init__(
        self,
        primary: Any,
        replica: Any,
        guider: Any,
        *,
        compiled: bool,
        explicit_on: bool,
        device_match: bool = True,
        logger: Any = None,
    ) -> None:
        import torch

        self._primary = primary
        self._replica = replica
        self._guider = guider
        self._logger = logger
        self._compiled = bool(compiled)
        self._explicit_on = bool(explicit_on)
        # False when the replica sits on a DIFFERENT GPU arch than the primary (explicit-on
        # only; auto declines at the gate): eager kernels differ across archs, so lossless
        # must never be reported for the pair.
        self._device_match = bool(device_match)
        self._ctx: Optional[str] = None
        self.enabled = False
        self.dispatch = "inline"
        # Replica cache state fell out of sync (an enable/disable half-failed): stop routing
        # to it -- the sequential passthrough is always correct.
        self._broken = False
        # A generation only "settles" its (shape, steps, cache) key once it COMPLETES; until
        # then every dispatch stays inline so a (re)compile of either module is serialized
        # (dynamo patches Module.__call__ process-wide during tracing).
        self._settled_key: Optional[tuple] = None
        self._pending_key: Optional[tuple] = None
        # id-keyed cache for constant-per-generation inputs (text embeds are ~200 MB;
        # re-copying them every step would waste PCIe/NVLink time).
        self._const_cache: dict = {}
        self._p_dev = next(primary.parameters()).device
        self._r_dev = next(replica.parameters()).device
        self._jobs: queue.Queue = queue.Queue()
        self._worker = threading.Thread(
            target = self._worker_loop, daemon = True, name = "cfg-parallel-replica"
        )
        self._worker.start()
        # The guider combines on the primary device; resolving the replica's pending
        # branch THERE puts the replica -> primary copy after both branches' queued
        # kernels (the only cross-device sync per step).
        self._orig_guider_forward = guider.forward
        p_dev = self._p_dev

        def _resolve(pred: Any) -> Any:
            if isinstance(pred, _PendingPred):
                return pred.resolve(p_dev)
            if isinstance(pred, torch.Tensor) and pred.device != p_dev:
                return pred.to(p_dev, non_blocking = True)
            return pred

        orig_forward = self._orig_guider_forward

        def _device_homogenising_forward(
            pred_cond,
            pred_uncond = None,
            **kw,
        ):
            return orig_forward(_resolve(pred_cond), _resolve(pred_uncond), **kw)

        guider.forward = _device_homogenising_forward

    # ── delegation ────────────────────────────────────────────────────────────
    def __getattr__(self, name: str) -> Any:
        primary = self.__dict__.get("_primary")
        if primary is None:
            raise AttributeError(name)
        return getattr(primary, name)

    def modules(self) -> list:
        """Both modules' submodules, so helpers walking ``transformer.modules()`` (the
        cache-hook inner arming) reach the replica's blocks too -- otherwise its computed
        steps run eager and the slower branch erases the parallel win."""
        mods = list(self._primary.modules())
        try:
            mods += list(self._replica.modules())
        except Exception:  # noqa: BLE001 -- a torch-less fake in tests
            pass
        return mods

    # ── cache fan-out (keeps per-branch cache state identical to single-GPU) ──
    def enable_cache(self, config: Any) -> None:
        # The primary needs no explicit invalidation: the caller (apply_step_cache) runs
        # _invalidate_child_registry_cache on the object it enabled, and __getattr__
        # forwards _diffusers_hook to _primary. Only the replica is unreachable through
        # delegation, hence its explicit _invalidate_registry.
        self._primary.enable_cache(config)
        try:
            self._replica.enable_cache(config)
            _invalidate_registry(self._replica)
            self._broken = False
        except Exception:
            # A half-cached pair skips differently per branch; re-raise so the caller's
            # best-effort path disables BOTH (disable_cache fans out).
            self._broken = True
            raise

    def disable_cache(self) -> None:
        # Removal must be transactional: if the primary's disable_cache raised while it ran
        # outside this guard, the replica was never cleaned and _broken stayed False, so a
        # half-removed pair kept routing. Attempt BOTH, record every failure, and only then
        # decide _broken -- any failure disables routing and surfaces so the caller reloads.
        failures: list[tuple[str, Exception]] = []
        for name, module in (("primary", self._primary), ("replica", self._replica)):
            try:
                module.disable_cache()
            except Exception as exc:  # noqa: BLE001 -- collect, don't skip the other branch
                failures.append((name, exc))
                _warn(self._logger, f"cfg-parallel {name} disable_cache", exc)
        self._broken = bool(failures)
        if failures:
            details = "; ".join(f"{name}: {exc}" for name, exc in failures)
            raise RuntimeError(
                f"CFG-parallel cache removal failed ({details}); parallel routing is "
                "disabled until the model is reloaded"
            )

    def _reset_stateful_cache(self, *args: Any, **kwargs: Any) -> None:
        for module in (self._primary, self._replica):
            reset = getattr(module, "_reset_stateful_cache", None)
            if callable(reset):
                try:
                    reset(*args, **kwargs)
                except Exception:  # noqa: BLE001 -- reset is best-effort
                    pass

    @contextlib.contextmanager
    def cache_context(self, name: str):
        self._ctx = name
        try:
            if self.enabled and self.dispatch == "inline":
                # Thread dispatch enters the replica's context inside the worker instead,
                # so it stays open across the whole worker-side forward.
                with self._primary.cache_context(name), self._replica.cache_context(name):
                    yield
            else:
                with self._primary.cache_context(name):
                    yield
        finally:
            self._ctx = None

    # ── per-generation policy ─────────────────────────────────────────────────
    def plan_generation(
        self, *, cache_engaged: bool, steps: int, width: int, height: int, frames: int
    ) -> dict:
        """Resolve routing + dispatch for the next generation (call AFTER the cache toggle
        so the engaged state is current). Returns the plan for logging."""
        # Prompt/conditioning constants are reusable only WITHIN one generation (the same tensor
        # objects flow through every denoise step); a new generation brings new ids. Clearing here
        # releases the previous generation's replica-side copies (text embeds ~200 MiB each, held on
        # BOTH GPUs) up front instead of pinning them until the 16-entry churn cap or teardown, and
        # also cleans up after a cancelled/failed run that never reached note_generation_done().
        self._const_cache.clear()
        # The engaged-cache marker may live on the proxy (post-install toggle) or the
        # primary (pre-install engage); the delegating getattr covers both.
        marker = getattr(self, "_unsloth_step_cache", None)
        key = (int(steps), int(width), int(height), int(frames), str(marker))
        # Bit-identity (2x B200): the eager stack is byte-identical (cache on or off), while
        # ANY compiled stack drifts ~1 ulp/step across the devices' separate artifacts,
        # amplified over the trajectory -- cache state doesn't change that, so identity keys
        # on the KERNELS alone and only explicit "on" accepts compiled drift. A replica on a
        # DIFFERENT arch (explicit-on only) runs different eager kernels, never lossless.
        lossless = not self._compiled and self._device_match
        cfg_active = getattr(self._guider, "num_conditions", 2) > 1
        self.enabled = cfg_active and not self._broken and (lossless or self._explicit_on)
        self.dispatch = "thread" if key == self._settled_key else "inline"
        self._pending_key = key
        plan = {
            "enabled": self.enabled,
            "dispatch": self.dispatch,
            "lossless": lossless,
            "cache_engaged": bool(cache_engaged),
        }
        if self._logger is not None:
            self._logger.info("diffusion.cfg_parallel: plan %s", plan)
        return plan

    def note_generation_done(self) -> None:
        """Commit the settled key after a COMPLETED generation; a cancelled/failed one keeps
        the next dispatch inline (its compiles may not have finished). A DISABLED run
        (guidance near 1) never routed the replica, so its key must not unlock thread
        dispatch either -- the replica's first compile for that shape would otherwise run
        concurrently with the primary next time, the exact race inline dispatch serializes."""
        if self.enabled:
            self._settled_key = self._pending_key

    # ── the branch router ─────────────────────────────────────────────────────
    def _move(self, v: Any) -> Any:
        import torch

        if not isinstance(v, torch.Tensor):
            return v
        hit = self._const_cache.get(id(v))
        if hit is not None and hit[0] is v:
            return hit[1]
        moved = v.to(self._r_dev, non_blocking = True)
        if v.numel() * v.element_size() >= (1 << 20):
            if len(self._const_cache) > 16:
                self._const_cache.clear()  # new-latents ids churn; constants re-promote
            self._const_cache[id(v)] = (v, moved)
        return moved

    def _worker_loop(self) -> None:
        import torch
        while True:
            job = self._jobs.get()
            if job is None:  # shutdown sentinel
                return
            ctx_name, args, kwargs, pending = job
            try:
                with torch.inference_mode(), self._replica.cache_context(ctx_name):
                    pending.value = self._replica(*args, **kwargs)[0]
            except BaseException as exc:  # noqa: BLE001 -- surfaced at resolve()
                pending.error = exc
            finally:
                pending.event.set()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.enabled and self._ctx == "pred_cond":
            # Input copies issued from the MAIN thread: their source-stream event is
            # recorded BEFORE the other branch's kernels are queued on the primary.
            args = tuple(self._move(a) for a in args)
            kwargs = {k: self._move(v) for k, v in kwargs.items()}
            if self.dispatch == "thread":
                pending = _PendingPred()
                self._jobs.put((self._ctx, args, kwargs, pending))
                return (pending,)
            return self._replica(*args, **kwargs)
        return self._primary(*args, **kwargs)

    def shutdown(self) -> None:
        try:
            self._jobs.put(None)
            self._worker.join(timeout = 5.0)
        except Exception:  # noqa: BLE001 -- daemon thread; process exit reaps it
            pass


# ── thread-safe cuDNN attention (scoped to an engaged proxy) ──────────────────────
_CUDNN_PATCH_STATE: dict = {}


def _install_threadsafe_cudnn_attention(logger: Any = None) -> bool:
    """Swap diffusers' ``_native_cudnn`` backend for a direct aten call (see module note).
    Idempotent; returns True when the patch is (already) installed."""
    if _CUDNN_PATCH_STATE:
        return True
    try:
        import torch
        from diffusers.models import attention_dispatch as ad

        orig = ad._native_cudnn_attention

        def _threadsafe_cudnn_attention(
            query,
            key,
            value,
            attn_mask = None,
            dropout_p = 0.0,
            is_causal = False,
            scale = None,
            enable_gqa = False,
            return_lse = False,
            _parallel_config = None,
        ):
            # Fall back to the stock path for options the direct op doesn't cover; the
            # video DiT hot path never takes them.
            if _parallel_config is not None or return_lse or enable_gqa or dropout_p:
                return orig(
                    query,
                    key,
                    value,
                    attn_mask = attn_mask,
                    dropout_p = dropout_p,
                    is_causal = is_causal,
                    scale = scale,
                    enable_gqa = enable_gqa,
                    return_lse = return_lse,
                    _parallel_config = _parallel_config,
                )
            # F.scaled_dot_product_attention (what the stock backend calls) treats a boolean
            # mask as "True participates" and converts it to an ADDITIVE bias internally; the
            # lower-level cuDNN op takes that bias directly, so a bool mask passed straight
            # through is misread for any partial (non-all-True) mask. Convert to match SDPA.
            if attn_mask is not None and attn_mask.dtype == torch.bool:
                attn_mask = torch.zeros_like(attn_mask, dtype = query.dtype).masked_fill_(
                    ~attn_mask, float("-inf")
                )
            q, k, v = (x.permute(0, 2, 1, 3).contiguous() for x in (query, key, value))
            out = torch.ops.aten._scaled_dot_product_cudnn_attention(
                q, k, v, attn_mask, False, 0.0, is_causal, False, scale = scale
            )[0]
            return out.permute(0, 2, 1, 3)

        ad._native_cudnn_attention = _threadsafe_cudnn_attention
        patched_keys = []
        for key, fn in list(getattr(ad._AttentionBackendRegistry, "_backends", {}).items()):
            if fn is orig:
                ad._AttentionBackendRegistry._backends[key] = _threadsafe_cudnn_attention
                patched_keys.append(key)
        _CUDNN_PATCH_STATE.update(module = ad, orig = orig, keys = patched_keys)
        if logger is not None:
            logger.info("diffusion.cfg_parallel: thread-safe cudnn attention installed")
        return True
    except Exception as exc:  # noqa: BLE001 -- without it, parallel dispatch is unsafe
        _warn(logger, "thread-safe cudnn attention", exc)
        return False


def _restore_threadsafe_cudnn_attention() -> None:
    if not _CUDNN_PATCH_STATE:
        return
    try:
        ad = _CUDNN_PATCH_STATE["module"]
        orig = _CUDNN_PATCH_STATE["orig"]
        ad._native_cudnn_attention = orig
        for key in _CUDNN_PATCH_STATE["keys"]:
            ad._AttentionBackendRegistry._backends[key] = orig
    except Exception:  # noqa: BLE001 -- best-effort restore
        pass
    _CUDNN_PATCH_STATE.clear()


# ── gate + build ──────────────────────────────────────────────────────────────────
def _device_identity(idx: int) -> Optional[tuple]:
    """(device name, compute capability) for CUDA device ``idx``, or None when the props
    cannot be queried (stubbed/old torch) -- identity is then unknown and the check stays
    best-effort rather than blocking the engage."""
    import torch
    try:
        return (
            str(torch.cuda.get_device_name(idx)),
            tuple(torch.cuda.get_device_capability(idx)),
        )
    except Exception:  # noqa: BLE001 -- unqueryable props: identity unknown
        return None


def _pick_secondary_device(
    primary_index: int, *, min_free_bytes: int = 0
) -> tuple[Optional[int], int, bool]:
    """(secondary CUDA device != primary, its free bytes, identity-match flag).

    Bit-identity needs the SAME kernels on both branches, and eager kernel selection is
    arch-dependent, so the picker prefers a device whose (name, capability) MATCH the primary's;
    only if none matches does it fall back to a mismatched one (so explicit ``on`` can still
    engage, lossy). An unqueryable identity counts as a match (best-effort).

    But a MATCHING device that cannot actually hold the replica is useless: rank a device that
    fits ``min_free_bytes`` above one that does not FIRST, so a viable heterogeneous GPU wins over
    an identical GPU too small for the replica (explicit ``on`` would otherwise fail with an
    insufficient-memory gate while a usable device sat idle). Among devices in the same viability
    tier, prefer the identity match, then the most free."""
    import torch

    primary_id = _device_identity(primary_index)
    best, best_free, best_match = None, -1, False
    best_key: tuple = (False, False, -1)
    for idx in range(torch.cuda.device_count()):
        if idx == primary_index:
            continue
        try:
            free, _total = torch.cuda.mem_get_info(idx)
        except Exception:  # noqa: BLE001 -- device unqueryable: skip it
            continue
        candidate_id = _device_identity(idx)
        match = primary_id is None or candidate_id is None or candidate_id == primary_id
        key = (free >= min_free_bytes, match, free)
        if best is None or key > best_key:
            best, best_free, best_match, best_key = idx, free, match, key
    return best, best_free, best_match


def maybe_enable_cfg_parallel(
    pipe: Any,
    fam: Any,
    *,
    requested: Optional[str],
    kind: str,
    transformer_source: Optional[str],
    hf_token: Optional[str],
    dtype: Any,
    quant_engaged: Optional[str],
    offload_active: bool,
    compiled: bool,
    attention_backend: Optional[str],
    speed_active: bool,
    speed_mode: Optional[str] = None,
    logger: Any = None,
) -> tuple[Optional[CFGParallelProxy], str]:
    """Gate, build and install the CFG-parallel proxy on ``pipe``. Returns
    ``(proxy, reason)`` when engaged or ``(None, reason)`` naming the failed gate.
    Best-effort: never raises; a miss leaves the pipe untouched."""
    mode = normalize_cfg_parallel(requested)
    if mode == CFG_PARALLEL_OFF:
        return None, "disabled by request"
    explicit_on = mode == CFG_PARALLEL_ON
    if not explicit_on and not speed_active:
        # Speed=off is the reference contract: every auto speed lever resolves off, and CFG
        # parallel is a speed lever (and reserves a second GPU). So auto never engages it;
        # an explicit cfg_parallel=on stays honored as a deliberate override.
        return None, "speed=off keeps auto CFG parallel off; request cfg_parallel=on to override"
    fam_name = str(getattr(fam, "name", "") or "").strip().lower()
    if not explicit_on and fam_name not in _CFG_PARALLEL_FAMILY_ALLOW:
        return None, "family not in the measured allowlist"
    if not explicit_on and compiled:
        # Cross-device compiled artifacts differ by ~1 ulp/step, amplified over the
        # trajectory: only the eager tier is bit-identical, so auto won't spend a replica's
        # VRAM on a stack it would never route.
        return None, (
            "compiled stack diverges across devices (~1 ulp/step, amplified over the "
            "trajectory); auto parallelises only the eager tier -- request "
            "cfg_parallel=on to accept fp-noise divergence for the ~1.7x"
        )
    if not bool(getattr(fam, "guidance_via_guider", False)):
        return None, "pipeline is not guider-driven (no per-branch cache_context)"
    if kind != "pipeline":
        return None, f"'{kind}' load has no clean second transformer source"
    if quant_engaged:
        return None, f"quantized DiT ({quant_engaged}) replica is unvalidated"
    if offload_active:
        return None, "offload plan moves the DiT; a pinned replica would defeat it"
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None, "torch unavailable"
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return None, "needs 2+ CUDA devices"
    primary = getattr(pipe, "transformer", None)
    if primary is None:
        return None, "pipe has no transformer"
    # Branch routing keys off the pipeline's per-branch cache_context calls.
    from .diffusion_cache import _ensure_block_metadata_registered, _pipeline_opens_cache_context

    if not _pipeline_opens_cache_context(pipe):
        return None, "pipeline opens no cache_context (branches are not identifiable)"
    try:
        p_dev = next(primary.parameters()).device
        if p_dev.type != "cuda":
            return None, f"primary DiT is on {p_dev.type}, not cuda"
        weight_bytes = sum(p.numel() * p.element_size() for p in primary.parameters())
        primary_index = p_dev.index or 0
        need = weight_bytes + _REPLICA_HEADROOM_BYTES
        # Filter by the replica's memory need FIRST so a viable heterogeneous GPU is preferred
        # over an identical GPU too small to hold the replica.
        secondary, free, device_match = _pick_secondary_device(primary_index, min_free_bytes = need)
        if secondary is None:
            return None, "no queryable secondary CUDA device"
        if not device_match:
            # A different GPU arch runs different eager kernels, so the byte-identity AUTO
            # promises cannot hold. Auto declines; explicit "on" proceeds but is downgraded
            # to lossy (plan_generation reports lossless=False) with a warning.
            primary_name = (_device_identity(primary_index) or ("unknown",))[0]
            secondary_name = (_device_identity(secondary) or ("unknown",))[0]
            if not explicit_on:
                return None, (
                    f"secondary cuda:{secondary} ({secondary_name}) is a different device "
                    f"than the primary ({primary_name}); eager kernels are arch-dependent, "
                    "so bit-identity cannot hold -- request cfg_parallel=on to accept the "
                    "divergence"
                )
            if logger is not None:
                logger.warning(
                    "diffusion.cfg_parallel: replica device cuda:%d (%s) differs from the "
                    "primary (%s); explicit on proceeds but the output is NOT bit-identical "
                    "to the single-GPU run (lossless=False)",
                    secondary,
                    secondary_name,
                    primary_name,
                )
        if free < need:
            return None, (
                f"secondary cuda:{secondary} has {free / 2**30:.1f} GiB free, "
                f"needs {need / 2**30:.1f} GiB for the DiT replica"
            )
    except Exception as exc:  # noqa: BLE001 -- any probe failure: stay single-device
        _warn(logger, "cfg-parallel gating", exc)
        return None, "device probe failed"

    # ── build the replica and mirror the primary's levers ──
    try:
        replica = (
            type(primary)
            .from_pretrained(
                transformer_source,
                subfolder = "transformer",
                torch_dtype = dtype,
                token = hf_token or None,
            )
            .to(f"cuda:{secondary}")
        )
        replica.eval()
    except Exception as exc:  # noqa: BLE001 -- download/VRAM race: stay single-device
        _warn(logger, "cfg-parallel replica load", exc)
        return None, "replica load failed"
    try:
        from .diffusion_attention import (
            apply_attention_backend,
            attention_backend_supported_on_device,
            install_hunyuan_attention_trim,
        )

        view = _ReplicaView(pipe, replica)
        if speed_active:
            install_hunyuan_attention_trim(view, fam, logger = logger)
        if attention_backend is not None:
            # The backend was arch-gated against the PRIMARY device; re-validate it on the
            # replica's (possibly heterogeneous) GPU before installing it -- FA3 is SM90-only,
            # FA4 needs SM100, cuDNN needs Ampere+, so a mismatched replica would set fine then
            # crash on its first attention kernel. Unsupported -> pin native on the replica.
            replica_backend = attention_backend
            if not attention_backend_supported_on_device(replica_backend, secondary):
                if logger is not None:
                    logger.warning(
                        "diffusion.cfg_parallel: attention backend %r resolved for the primary "
                        "is unsupported on the replica cuda:%d (different arch); pinning native "
                        "there",
                        replica_backend,
                        secondary,
                    )
                replica_backend = None
            apply_attention_backend(view, replica_backend, logger = logger)
        if compiled:
            from .diffusion_speed import SPEED_MAX, _compile_repeated_blocks

            # Mirror the primary's tier: under speed=max the primary compiles max-autotune +
            # fuses QKV, so the replica must too or it becomes the slower branch and throttles
            # the whole parallel run. A cache may engage/toggle on this DiT, so fullgraph stays
            # off like the loader.
            max_speed = str(speed_mode) == SPEED_MAX
            _compile_repeated_blocks(view, logger, max_autotune = max_speed, cache_active = True)
            if max_speed:
                # Fuse the REPLICA's QKV projections directly: _fuse_qkv(view) would resolve the
                # pipe-level fuse_qkv_projections through _ReplicaView delegation and re-fuse the
                # PRIMARY instead, leaving the replica unfused.
                fuse = getattr(replica, "fuse_qkv_projections", None)
                if callable(fuse):
                    try:
                        fuse()
                    except Exception as exc:  # noqa: BLE001 -- optimisation only
                        _warn(logger, "cfg-parallel replica fuse_qkv", exc)
        if not _install_threadsafe_cudnn_attention(logger):
            raise RuntimeError("thread-safe attention patch failed")
        # The proxy's class name hides the transformer's from the metadata probe, so
        # register while the real class is still visible.
        _ensure_block_metadata_registered(primary, logger)
        guider = getattr(pipe, "guider", None)
        if guider is None or not callable(getattr(guider, "forward", None)):
            raise RuntimeError("pipe has no patchable guider")
        proxy = CFGParallelProxy(
            primary,
            replica,
            guider,
            compiled = compiled,
            explicit_on = explicit_on,
            device_match = device_match,
            logger = logger,
        )
        pipe.transformer = proxy
    except Exception as exc:  # noqa: BLE001 -- roll the replica back; stay single-device
        _warn(logger, "cfg-parallel install", exc)
        try:
            del replica
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        # No proxy was committed, so _teardown_state never runs for this install: a
        # failure AFTER the cuDNN patch landed (e.g. no patchable guider) must undo it
        # here or every later single-device generation keeps the direct aten replacement.
        # No-op when the patch never installed.
        _restore_threadsafe_cudnn_attention()
        return None, "replica install failed"
    if logger is not None:
        logger.info(
            "diffusion.cfg_parallel: engaged (replica on cuda:%d, %.1f GiB weights)",
            secondary,
            weight_bytes / 2**30,
        )
    return proxy, f"engaged: DiT replica on cuda:{secondary}"


def teardown_cfg_parallel(
    pipe: Any,
    proxy: Any,
    logger: Any = None,
) -> None:
    """Restore the pipe to its single-device shape and free the replica's VRAM.
    Safe to call with a half-built or foreign object; never raises."""
    try:
        primary = getattr(proxy, "_primary", None)
        if primary is not None and getattr(pipe, "transformer", None) is proxy:
            pipe.transformer = primary
        guider = getattr(proxy, "_guider", None)
        orig_fwd = getattr(proxy, "_orig_guider_forward", None)
        if guider is not None and orig_fwd is not None:
            guider.forward = orig_fwd
        shutdown = getattr(proxy, "shutdown", None)
        if callable(shutdown):
            shutdown()
        if getattr(proxy, "_replica", None) is not None:
            proxy._replica = None
        proxy._const_cache = {}
        _restore_threadsafe_cudnn_attention()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if logger is not None:
            logger.info("diffusion.cfg_parallel: torn down (replica freed)")
    except Exception as exc:  # noqa: BLE001 -- teardown is best-effort
        _warn(logger, "cfg-parallel teardown", exc)


def _invalidate_registry(module: Any) -> None:
    from .diffusion_cache import _invalidate_child_registry_cache
    _invalidate_child_registry_cache(module)


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning(
            "diffusion.cfg_parallel: %s unavailable (%s); running single-device", what, exc
        )
