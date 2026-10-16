# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: GPU allocation, load fit and GPU selection."""

import math
import os
import sys
import time

from pathlib import Path
from typing import (
    Callable,
    Iterable,
    Mapping,
    Optional,
)
from core.inference.llama_server_args import (
    force_pageable_load,
    resolve_effective_memory_state,
)
from utils.native_path_leases import child_env_without_native_path_secret

from .binary_runtime import (
    _HOST_RAM_HEADROOM_MIB,
    _PAGEABLE_OVERRIDE_NOTE,
    _PAGEABLE_OVERRIDE_TEXT_ONLY_NOTICE,
    _llama_lib_dir,
    _loader_path_var,
    _native_linux_system_rocm_lib_dirs,
    _prepend_loader_dir,
    _resolve_llama_binary,
    _wsl_system_rocm_lib_dirs,
)
from .draft_args import (
    _pass_through_overrides_placement,
    _sidecar_adapter_bytes,
)
from .launch_flags import _kv_offload_from_args
from .log import logger
from .memory_policy import (
    _active_vram_fraction,
    _kv_bytes_per_elem,
    _vram_usable_mib,
)

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _GpuFitMixin:
    def _page_an_oversized_unmapped_load(
        self, cmd: Iterable[str], env: Optional[dict], oversized
    ) -> tuple[list, Optional[str]]:
        """Rewrite a launch that is BOTH oversized and non-pageable into a pageable one.

        Returns ``(cmd, note)``: the argv to launch, and the sentence to append to
        whatever warning is emitted, or None when nothing was overridden and ``cmd``
        comes back unchanged.

        The host-RAM shortfall is survivable only because the weights are mmap'd: the
        whole premise is that the spill pages in from disk as the model runs. Modes
        "none" and "mlock" have no such fallback -- upstream sets use_mmap for
        mmap/mmap+mlock/auto and nothing else (llama-model-loader.cpp), so those two
        read the weights into a buffer llama.cpp allocates. The same shortfall is then
        an allocation failure or an OOM kill during loading, not slow generation.
        Reached by ``--load-mode``, by the deprecated ``--no-mmap`` / ``--no-direct-io``
        spellings, and by the LLAMA_ARG_* twins llama.cpp reads before argv, so the
        EFFECTIVE state is what decides, not the tokens Unsloth happens to have emitted.

        This never blocks a load, so the unmapped request is OVERRIDDEN rather than
        refused: the pageable equivalent goes in (a lock is kept, as mmap+mlock) so the
        load the user asked for can actually finish. Only on a real shortfall -- an
        unmapped load that fits is exactly what was asked for and is left byte for byte
        as written. Said out loud, since a silent override is its own surprise: in the
        returned note when a warning is emitted, and in the log either way, so a load
        whose warning was silenced is still traceable.

        One helper for every launch that can discover a shortfall (the APU preflight,
        the discrete host guard, the arch-crash retry), because the decision belongs to
        the CONDITION and not to any one warning: a site that only recorded, or a
        warning the environment suppressed, would otherwise skip the remap and hand the
        child a load it cannot complete. ``oversized`` is the per-site shortfall
        predicate, called only once the cheap non-pageable test has passed, so a
        comfortable launch pays for no pricing.
        """
        # The caller's own list back on every no-op path, not a rebuilt copy: an
        # untouched launch must stay byte for byte, and the same token objects, as
        # whatever built it.
        original = cmd if isinstance(cmd, list) else list(cmd or ())
        _mlock_now, _reserves_ram_now = resolve_effective_memory_state(original, env)
        if not _reserves_ram_now:
            return original, None
        try:
            if not (oversized() if callable(oversized) else oversized):
                return original, None
        except Exception:
            # Advisory pricing: an unreadable host or GGUF leaves the launch alone.
            logger.debug("Could not price the load for a pageable override.", exc_info = True)
            return original, None
        paged, overridden = force_pageable_load(original, env)
        if not overridden:
            return original, None
        logger.warning(
            "Overriding the unmapped load mode (%s) for an oversized model: it would "
            "allocate the whole model in host RAM instead of paging it in.",
            ", ".join(overridden),
        )
        return paged, _PAGEABLE_OVERRIDE_NOTE

    @property
    def last_load_warning(self) -> Optional[str]:
        """Advisory notice from the most recent load, or None. Read by the route."""
        notices = (self._variant_fallback_warning, self._last_load_warning)
        return " ".join(notice for notice in notices if notice) or None

    def _begin_load_warnings(self) -> None:
        """Drop the previous load's memory notice, at the point a load commits to
        replacing the resident server. Without it a warned load followed by a
        comfortable one would report the first load's notice against the second.

        Called from the Phase 1 teardown, not from the top of ``load_model``: an
        attempt rejected before that point leaves the old child serving, and its
        advisory has to survive with it (the route still answers ``already_loaded``
        from that child). The arch-crash retry calls it again for the same reason in
        reverse -- the placement everything was priced against is the one that just
        died."""
        self._last_load_warning = None
        self.hub_access_warning = None
        # Same lifetime, same reason: the advice describes the placement the dying
        # child was priced against and must not be reported against its replacement.
        self._last_carveout_advice = None

    def _record_load_warning(self, message: Optional[str]) -> None:
        """Log an advisory memory notice and keep it for the route to hand back.

        Never raises: these replace what used to be a refusal, and an oversized GGUF
        pages from disk instead of failing, so the load is the user's call. First notice
        wins, since the earliest priced the whole pool and later ones see a narrowed
        selection.
        """
        if not message:
            return
        logger.warning(message)
        if self._last_load_warning is None:
            self._last_load_warning = message

    def _amend_load_warning(self, note: Optional[str]) -> None:
        """Append to the notice already recorded for this load, if there is one.

        The pageable-load override can be settled after an earlier guard has already
        recorded its warning: the APU preflight runs long before ``cmd`` and ``env``
        exist, which is where the override has to be applied. First notice wins, so
        that earlier text is what the route hands back, and appending is what keeps the
        override named in the message the user actually sees. A load whose warning was
        suppressed has nothing to amend and keeps the log line as its only record.
        """
        if note and self._last_load_warning:
            self._last_load_warning += note

    def _reprice_after_dropping_pinned_projector(
        self,
        *,
        apu_msg: Optional[str],
        host_msg: Optional[str],
        model_size: Optional[int],
        pinned_bytes: int,
        avail_mib: Optional[int],
        part: str = "APU",
    ) -> None:
        """Re-price the APU RAM advisory once a text-only retry drops a CPU-pinned
        vision projector.

        The APU preflight charges ``model_size + _mmproj_pinned_bytes``, because a
        projector pinned to the CPU loads into the same system RAM as the weights. When
        that projector then fails to start and the retry strips ``--mmproj``, the child
        that came up healthy never read those bytes: the recorded shortfall describes a
        load nothing is running. Priced again without them, so the notice matches what
        the serving process actually holds -- rewritten to the smaller need if the
        weights alone still do not fit, handed to the discrete guard's notice if that
        one also found a shortfall, and otherwise dropped.

        What is NOT undone is the pageable-load override. It rewrote the argv and, in
        ``force_pageable_load``, the shared child env in place, and the child was
        spawned from both long before this runs, so the server IS memory-mapped and
        nothing here can change that; reverting it would need yet another respawn of a
        process that just came up healthy, on the recovery rung that exists because this
        model has already failed to start once. So an override that the reprice shows
        was not needed is reported rather than reverted: whatever note the message
        carried is kept, and a load whose only remaining news is the override says that
        and nothing else.

        A no-op unless the projector was actually charged and the APU advisory is
        actually what the route would hand back -- an opt-out that silenced it, or any
        other notice that first-notice-wins kept, is left exactly as it is.
        """
        if not apu_msg or pinned_bytes <= 0 or model_size is None or avail_mib is None:
            return
        current = self._last_load_warning
        if not current or not current.startswith(apu_msg):
            return
        # Whatever _amend_load_warning appended (the pageable-override note), carried
        # over rather than rebuilt, so a reprice cannot describe the override in
        # different words than the launch site did.
        suffix = current[len(apu_msg) :]
        # ``avail_mib`` is the figure the PREFLIGHT read, carried down rather than taken
        # again here. By this point the text-only child is healthy and holding the
        # weights, so a fresh reading is the pool minus the very model being priced:
        # it would charge ``model_size`` against ``avail - model_size`` and report a
        # shortfall for a load that demonstrably just started. Same pool, one term
        # removed, is the only comparison that answers the question being asked.
        repriced = self._apu_ram_shortfall_message(model_size, avail_mib, part = part)
        if repriced:
            self._last_load_warning = repriced + suffix
        elif host_msg:
            # The discrete guard priced the same launch and lost first-notice-wins.
            # It reads the model path off the argv and never charged the projector,
            # so dropping the projector leaves its verdict standing.
            self._last_load_warning = host_msg + suffix
        elif suffix.strip():
            self._last_load_warning = _PAGEABLE_OVERRIDE_TEXT_ONLY_NOTICE
        else:
            self._last_load_warning = None
        logger.info(
            "Text-only fallback dropped the CPU-pinned vision projector (%.0f MiB); "
            "re-priced the host memory advisory against the weights alone.",
            pinned_bytes / (1024 * 1024),
        )

    def _reprice_after_cpu_only_fallback(
        self,
        *,
        host_msg: Optional[str],
        cpu_cmd: Iterable[str],
        env: Optional[Mapping[str, str]],
        avail_mib: Optional[int],
        pageable_note: Optional[str] = None,
    ) -> None:
        """Re-price the host memory advisory once a Vulkan crash is replayed on CPU.

        The replay launches with ``--gpu-layers 0 --device none``, so no VRAM is
        credited to it at all: the child that serves the session holds the WHOLE model
        in system RAM and pages it from disk. Two things follow, and the preflight
        notice gets neither right.

        A load that was warned about its spill was warned about the spill of the GPU
        placement that just died. The replay's spill is the whole model, which is
        strictly larger, so the recorded figure understates what the running child
        actually pages. And a load that FIT in VRAM was never warned at all, yet it is
        the one the user most needs told: it now runs entirely from host RAM on a host
        that may not hold it, and ``memory_warning`` was null for it.

        Priced against ``avail_mib``, the figure the PREFLIGHT read, for the same reason
        the projector reprice carries its own snapshot down: this runs after
        ``_spawn_and_wait`` reports the CPU child healthy, so a fresh reading is the
        pool minus the model being priced and would report a shortfall for a load that
        demonstrably just started.

        Called only where the replay is known healthy, never where it is chosen. A
        failed replay is not terminal on every rung -- the drafter replay reports "not
        recovered" and the caller tries the drafterless argv next -- so pricing at the
        decision would describe a placement about to be abandoned.

        The pageable-load override is NOT re-decided here, and nothing claims it was
        reverted. ``force_pageable_load`` rewrote the argv and the shared child env in
        place before the first spawn, and ``_cpu_isolated_replay`` strips placement
        alone, so the replay inherits both and the CPU child really is memory-mapped.
        Whatever note the preflight message carried is carried over verbatim onto the
        re-priced text rather than rebuilt, so a reprice cannot describe the override in
        different words than the launch site did.

        ``pageable_note`` is the override ``_prepare_cpu_fallback_launch`` decided on
        the REPLAY, which the preflight message cannot have carried: a launch that fit
        in VRAM was priced as pageable-enough and left the user's unmapped mode alone,
        and only the replay's own footprint (no VRAM credited) turns it into a full
        host allocation. Both notes cannot be set at once -- if the preflight had
        already overridden, the replay it built from is mapped and the second pricing
        finds nothing to rewrite -- so this appends rather than replaces. Priced against
        the same ``avail_mib`` the rewrite used, so the advisory cannot describe an
        unmapped load the same pool reading has just made mapped.

        A no-op unless the notice the route would hand back is the one this path can
        speak for: an opt-out that silenced it (the re-pricing call honours it too), or
        any other notice first-notice-wins kept, is left exactly as it is. The note
        follows the message: an override on a silenced load stays in the log alone,
        exactly as it does on the main launch path.

        The carve-out advice is dropped rather than re-priced: this replay runs
        ``--gpu-layers 0 --device none``, so no allocation holds any of the weights.
        Here rather than at either call site, since both reach this same state.
        """
        self._last_carveout_advice = None
        repriced = self._launch_host_shortfall_message(
            cpu_cmd,
            (),
            env,
            child_has_no_gpu = True,
            avail_mib = avail_mib,
        )
        note = pageable_note or ""
        current = self._last_load_warning
        if current is None:
            # The model fit in VRAM, so nothing warned; on CPU it fits nowhere but disk.
            self._record_load_warning((repriced + note) if repriced else None)
            return
        if not host_msg or not current.startswith(host_msg) or not repriced:
            # Nothing here can be re-priced, but an override still changed the child:
            # say so on whatever notice first-notice-wins kept.
            self._amend_load_warning(pageable_note)
            return
        self._last_load_warning = repriced + current[len(host_msg) :] + note
        logger.info(
            "Re-priced the host memory advisory for the CPU replay: it runs with no "
            "GPU, so the whole model is charged to system RAM."
        )

    def _fits_without_paging(
        self,
        footprint_bytes: Optional[int],
        detected_gpus: Iterable[tuple],
        *,
        gpu_indices: Optional[Iterable[int]] = None,
        shared_gpu_ids: Iterable[int] = (),
        host_only_bytes: int = 0,
        vram_margin_mib: float = 0.0,
        avail_mib: Optional[int] = None,
        headroom_mib: int = _HOST_RAM_HEADROOM_MIB,
    ) -> Optional[bool]:
        """Whether the whole load fits in VRAM, or in VRAM plus host RAM (else False;
        None when it cannot be told). This is the ``--load-mode`` question: mmap earns
        its keep by demand-paging a footprint the machine cannot hold, so a load that
        fits never needs it.

        ``footprint_bytes`` is weights + KV + scratch, not weights alone. The refusal
        guards next door deliberately price weights only, because understating there
        can only fail to refuse a load that would have worked. Here the bias runs the
        other way: understating would claim a fit that isn't one and hand the load a
        loader that cannot page, so every term the launch knows about is charged.

        Shared iGPU memory is not added on top of host RAM (it IS host RAM), so those
        devices are dropped from the VRAM term and their weights are carried by the RAM
        term, the same split ``_launch_host_shortfall_message`` uses. ``gpu_indices``
        narrows the VRAM term to the devices this launch actually pins; unpinned cards
        hold nothing for it. Unreadable RAM answers None rather than guessing, which
        keeps the caller on llama.cpp's own default.

        ``host_only_bytes`` is the part of ``footprint_bytes`` that CANNOT be satisfied
        out of VRAM however much is free -- a drafter the launch pinned to the CPU is
        the case -- so it is subtracted from what VRAM may cover and added to the RAM
        requirement instead. Pooling it would let free VRAM pay for bytes that can only
        ever be allocated in host RAM.

        ``vram_margin_mib`` is the free VRAM llama.cpp's own fitter keeps PER DEVICE
        and will not allocate into (``--fit-target``, default 1024 MiB, "target margin
        per device for --fit"; common/arg.cpp, common/common.h). Under ``--fit on``
        the child spills weights to host RAM rather than dip into it, so those bytes
        are not credit this launch may spend. Zero on the proved path, which emits
        ``-ngl -1 --fit off`` and runs no fitter at all.
        """
        if not footprint_bytes or footprint_bytes <= 0:
            return None
        host_only = max(0, int(host_only_bytes or 0))
        shared = set(shared_gpu_ids or ())
        pinned = None if gpu_indices is None else {int(idx) for idx in gpu_indices}
        margin_mib = max(0.0, float(vram_margin_mib or 0.0))
        free_vram_mib = sum(
            max(0, row[1] - margin_mib)
            for row in (detected_gpus or ())
            if row[0] not in shared and (pinned is None or row[0] in pinned)
        )
        free_vram_bytes = int(free_vram_mib * 1024 * 1024)
        # VRAM covers the GPU-eligible part; its spill plus the host-only bytes hit RAM.
        spill = max(0, max(0, footprint_bytes - host_only) - free_vram_bytes) + host_only
        if spill <= 0:
            return True
        avail = self._available_system_memory_mib() if avail_mib is None else avail_mib
        if avail is None:
            return None
        # Same headroom the host-offload refusal keeps free: a RAM filled to the last
        # byte is the paging case this is trying to avoid.
        return spill <= max(0, avail - headroom_mib) * 1024 * 1024

    def _fit_derived_load_mode(
        self,
        *,
        model_size: Optional[int],
        mmproj_pinned_bytes: int = 0,
        kv_cache_bytes: int = 0,
        kv_sized: bool = True,
        mtp_bytes: int = 0,
        mtp_unsized: bool = False,
        host_only_bytes: int = 0,
        compute_buffer_flat: int = 0,
        compute_buffer_ctx: int = 0,
        pipeline_overhead_bytes: int = 0,
        soft_overhead: int = 0,
        gpus: Optional[Iterable[tuple]] = None,
        gpu_indices: Optional[Iterable[int]] = None,
        shared_gpu_ids: Iterable[int] = (),
        fit_margin_mib: float = 0.0,
        is_vulkan_backend: bool = False,
        avail_mib: Optional[int] = None,
        extra_args: Optional[Iterable[str]] = None,
        env: Optional[Mapping[str, str]] = None,
        lazy_read_bytes: int = 0,
        announce: bool = True,
    ) -> Optional[str]:
        """``"none"`` when the fit proves the load needs no demand paging, else None.

        mmap exists to demand-page a footprint the machine cannot hold, and llama.cpp
        builds its async pinned-buffer upload path ONLY when mmap is off
        (llama-model-loader.cpp, ``upload_backend``), so a load that fits wants
        ``none``: same residency, faster load, no model-sized mapping left behind.
        Anything that does not fit, or that cannot be priced, keeps llama.cpp's own
        ``auto``, where the mapping is what stops an over-committed load from being
        killed outright.

        The footprint is charged WHOLE, weights and KV and scratch, because
        understating it would claim a fit that is not there and hand the load a loader
        that cannot page. Every term that cannot be read abstains for the same reason:
        unreadable weights, an unsized KV (the estimator returns 0 without the dims,
        and a 0 KV term on a long-context load is exactly the missing term) and a
        drafter engaged but unsized, which the placement itself covers with a flat
        cushion rather than a number.

        ``host_only_bytes`` is the slice of that footprint no amount of free VRAM can
        hold (a drafter pinned to the CPU with ``-ngld 0``): counted in the total, but
        charged to host RAM alone, because free VRAM cannot pay for an allocation the
        child only ever makes on the host. ``kv_cache_bytes`` joins it under an
        effective ``--no-kv-offload``, which puts the WHOLE cache on the host
        whatever the layer placement says (llama-kv-cache.cpp defaults each layer's
        buffer type to the CPU one and upgrades it only inside ``if (offload)``).

        Apple Silicon abstains outright. Metal is the one backend where
        ``buffer_from_host_ptr`` IS supported, so mmap wraps the weights in a Metal
        buffer in place, zero copy and file backed; ``none`` would allocate the same
        bytes anonymously, and an evictable 30 GB mapping on a 32 GB machine is the
        difference between working and being jetsammed. The host figure is the wrong
        ceiling there anyway (see ``_apple_metal_memory_budget_bytes``), so a fit read
        off ``MemAvailable`` would not be one worth acting on.

        ``extra_args`` is the pass-through block as the child will really get it (the
        caller strips what a ``gpu_ids`` pin owns first), and it voids the VRAM term.
        Those tokens are appended AFTER this launch's own placement flags and
        llama.cpp is last-wins (common/arg.cpp assigns ``n_gpu_layers`` / ``devices``
        on every occurrence), so a hand-typed ``-ngl``, a ``--device`` naming no GPU
        or a CPU tensor override runs weights out of host RAM that the planner just
        credited to a card. The placement this priced is then not the one that
        launches, and ``none`` on a footprint host RAM cannot hold is an OOM where
        ``auto`` would have demand-paged. Dropping the VRAM credit rather than
        abstaining keeps the answer for the case that is still provable: a load host
        RAM holds WHOLE is safe wherever those flags end up putting it. The same
        predicates the Model Memory host-residency gate uses, so the two cannot drift.

        ``env`` is the environment the child will really get (``os.environ`` when the
        caller passes none). Every one of those pass-through overrides has an env twin
        llama.cpp reads BEFORE argv, so classifying on the tokens alone would miss an
        inherited ``LLAMA_ARG_DEVICE=none``, which no automatic launch clears (only an
        explicit ``gpu_ids`` pin does). A ``--device`` naming FEWER devices than this
        fit credits voids the VRAM term for the same reason a CPU-valued one does: the
        cards it leaves out hold nothing for this launch. ``LLAMA_ARG_N_GPU_LAYERS``
        is the sharpest of them: the fitting path emits no ``-ngl`` for it to lose
        to, so an inherited count is the whole layer policy (see
        ``_env_fixes_gpu_layers``).

        ``fit_margin_mib`` is what llama.cpp's fitter keeps free per device when the
        fitter really runs, i.e. VRAM this load may not spend; see
        ``_fits_without_paging``. The caller answers that from the EFFECTIVE flags
        rather than from its own plan: the extras land after this launch's ``--fit``
        and llama.cpp is last-wins, ``-ngl -1`` is llama.cpp's own default so an
        enabled fitter is free to lower it (common/fit.cpp aborts only on a count the
        user really set), and a pass-through ``--fit-target`` raises the margin above
        anything ``_ctx_integrity_flags`` emitted.

        ``lazy_read_bytes`` (``_lazy_read_host_bytes``) leaves the footprint: it stays
        mmap'd and pages through the headroom. ``announce`` False silences a what-if call.
        """
        from utils.hardware import is_apple_silicon

        if not model_size or not kv_sized or mtp_unsized or is_apple_silicon():
            return None
        # Pass-through adapters are resident weights model_size (the base GGUF alone)
        # never carried. A TERM, not a placement override, so an unreadable one abstains.
        sidecar_bytes = _sidecar_adapter_bytes(extra_args)
        if sidecar_bytes is None:
            return None
        rows = list(gpus or ())
        source_env = os.environ if env is None else env
        _credited = len(list(gpu_indices)) if gpu_indices is not None else len(rows)
        if _pass_through_overrides_placement(
            extra_args,
            source_env,
            _credited,
            is_vulkan_backend = is_vulkan_backend,
            gpu_indices = gpu_indices,
        ):
            logger.info(
                "Load mode: pass-through arguments override the placement this fit "
                "priced; charging the whole load to host RAM."
            )
            rows, gpu_indices = [], None
        footprint = (
            model_size
            # Pass-through LoRA / control vectors, allocated alongside the base tensors.
            + max(0, sidecar_bytes)
            # A CPU-pinned projector left model_size but is still resident, and this
            # is a footprint rather than a VRAM budget.
            + max(0, mmproj_pinned_bytes)
            + max(0, kv_cache_bytes)
            # Carries the drafter's own weights and KV.
            + max(0, mtp_bytes)
            # Resident, but only ever in host RAM; charged to the RAM term alone below.
            + max(0, host_only_bytes)
            + max(0, compute_buffer_flat)
            + max(0, compute_buffer_ctx)
            # A layer split puts a fixed CUDA context and scratch on EVERY device, and
            # the lumps above cover one, so the rest is a term of its own (the same one
            # _subset_model_size reserves). 0 on a single GPU.
            + max(0, pipeline_overhead_bytes)
            + max(0, soft_overhead)
        )
        _lazy_discount = max(0, int(lazy_read_bytes or 0))
        if _lazy_discount:
            footprint = max(1, footprint - _lazy_discount)
        # Unified memory is ONE pool: an APU's free VRAM IS the host RAM the spill
        # comes out of, so adding both fits a model twice into memory that holds it
        # once. shared_gpu_ids already names Vulkan iGPUs; the ROCm APUs it cannot see
        # are added here. Both end up priced against host RAM alone.
        shared = set(shared_gpu_ids or ())
        pinned = list(gpu_indices) if gpu_indices is not None else [idx for idx, _free in rows]
        if rows and not is_vulkan_backend and self._amd_apu_wants_unified_memory(pinned):
            # Per device, not blanket: the gate above is the cheap "is ANY of them
            # unified" question, and a discrete card next to an APU still has a pool of
            # its own, so re-ask per row rather than forfeit a fit that is really there.
            shared |= {idx for idx, _free in rows if self._amd_apu_wants_unified_memory([idx])}
        # Integrated CUDA SoCs (Jetson, DGX Spark) too. Cache only: _get_gpu_memory filled it.
        _integrated = LlamaCppBackend._INTEGRATED_CUDA_IDS.get(
            LlamaCppBackend._integrated_cuda_mask_key(), ()
        )
        shared |= {idx for idx, _free in rows if idx in _integrated}
        # --no-kv-offload puts the WHOLE cache in host RAM whatever the layer placement
        # says (llama-kv-cache.cpp upgrades a layer's buffer type only inside
        # `if (offload)`), so free VRAM may not pay for it. Resolved off the same
        # effective views the placement overrides above use.
        _kv_on_host = not _kv_offload_from_args(extra_args, env = source_env)
        if self._fits_without_paging(
            footprint,
            rows,
            gpu_indices = gpu_indices,
            shared_gpu_ids = shared,
            # A CPU-pinned projector is host-resident: counted once in the footprint
            # above, and charged here to the term free VRAM may NOT pay for, so surplus
            # VRAM cannot cover it and answer True without consulting RAM.
            host_only_bytes = (
                max(0, host_only_bytes)
                + max(0, mmproj_pinned_bytes)
                # Same split as the pinned projector: counted once above, charged to RAM.
                + (max(0, kv_cache_bytes) if _kv_on_host else 0)
            ),
            vram_margin_mib = fit_margin_mib,
            avail_mib = avail_mib,
        ):
            if announce:
                logger.info(
                    "Load mode: the whole load (%.1f GB) fits without paging, using "
                    "--load-mode %s%s.",
                    footprint / (1024**3),
                    self._FIT_LOAD_MODE,
                    (
                        f" ({_lazy_discount / (1024**3):.1f} GB of lazily read tables stay "
                        "on disk)"
                        if _lazy_discount
                        else ""
                    ),
                )
            return self._FIT_LOAD_MODE
        return None

    # What a proven fit picks. One name, because the choice is a measurement.
    # Load-to-ready on a B200 over local NVMe, median of 3, full offload, b10360:
    #
    #   Qwen3-4B Q4_K_M (2.5 GB)          cold   warm
    #     mmap (no flag)                  1.81   1.42
    #     none                            2.04   1.49
    #     dio                             2.05   1.69
    #   Qwen3-30B-A3B UD-Q4_K_XL (17 GB)  cold   warm
    #     mmap (no flag)                  5.43   2.81
    #     none                            5.49   3.21
    #     dio                             2.26   2.30
    #
    # So "none" is NOT the fast one here: it drops mmap and unlocks llama.cpp's async
    # pinned-buffer uploader, but still pays the page-cache copy, which on this storage
    # costs more than the uploader wins. "dio" takes that uploader AND skips the cache,
    # and is 2.4x faster cold on the model big enough for loading to matter, which is
    # upstream's guidance too. Kept at "none" because that is what was asked for; this
    # constant is the whole switch. dio has no pre-enum spelling, so a build without
    # --load-mode would emit nothing rather than a legacy flag.
    _FIT_LOAD_MODE = "none"

    METAL_CTX_OVERCOMMIT_ENV = "UNSLOTH_ALLOW_METAL_CTX_OVERCOMMIT"

    @staticmethod
    def _metal_context_overcommit_message(
        requested_ctx: int,
        max_available_ctx: int,
        cache_type_kv: Optional[str] = None,
        *,
        nothing_fits: bool = False,
        wired_limit: bool = False,
    ) -> Optional[str]:
        """Refusal when a hand-set context exceeds what unified memory holds (else None).

        The Metal placement branch already works out the largest context that fits, but
        only Auto was moved to it: an explicit request was passed through verbatim, on
        the theory that "--fit on" is a backstop. It is one, just not a trustworthy one
        here. llama.cpp will reduce an explicit context (common.h: fit_params_min_ctx
        defaults to 4096, and only "-c 0" disables the reduction outright), but it
        decides from the free memory ggml-metal reports, off the device's
        recommendedMaxWorkingSetSize. That is a property of the machine, not the moment:
        it knows nothing of Unsloth's own resident gigabyte or two, of whatever else the
        user has open, or of the iogpu wired limit that is the figure actually being
        blown. _apple_metal_memory_budget_bytes exists for exactly that gap, and takes
        min(device ceiling, psutil available) instead.

        So an optimistic estimate leaves the request alone and the launch over-commits
        wired memory. Wired pages are not reclaimable, so Jetsam cannot step in: the
        driver faults and the machine panics rather than the load failing the way it
        would on a discrete GPU (r/unsloth, M1 Max 32 GB, Qwen3.8-27B-UD-Q4_K_XL,
        panicked twice on a hand-set context, and never on Auto).

        Refusing rather than silently clamping is deliberate: a context that quietly
        became a quarter of what was asked for is its own support thread.

        Unlike ``_apu_ram_shortfall_message`` this prices the context, not just the
        weights. That helper leaves KV out because context auto-reduces on its path; here
        the explicit request is exactly what does not auto-reduce, so the KV and compute
        buffers it implies are the bytes that do the damage.

        Callers pass a ceiling measured with a real KV estimate. The 4096 floor the
        branch falls back to when KV cannot be sized is a guess, and refusing against it
        would block contexts that load fine today.

        ``wired_limit`` marks a ceiling priced under the GPU wired headroom, which Auto does not
        size from, so the advice leaves Auto out.

        UNSLOTH_ALLOW_METAL_CTX_OVERCOMMIT=1 abstains, matching the host-offload opt-out,
        though the failure mode it re-enables is the whole machine rather than one app.
        """
        if requested_ctx <= 0:
            return None
        # nothing_fits carries its own verdict, and is the one case with no positive
        # ceiling: the fit priced its smallest context and even that did not fit.
        if not nothing_fits:
            if max_available_ctx <= 0:
                return None
            if requested_ctx <= max_available_ctx:
                return None
        # the user's own opt-out, so read the real environment, not the curated child env
        if os.environ.get(LlamaCppBackend.METAL_CTX_OVERCOMMIT_ENV, "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            logger.info(
                "%s set: launching at a context unified memory is not sized for.",
                LlamaCppBackend.METAL_CTX_OVERCOMMIT_ENV,
            )
            return None
        kv_hint = (
            " Setting the KV cache to q8_0 roughly halves what the context costs."
            if (cache_type_kv or "f16").strip().lower() in ("f16", "fp16", "")
            else ""
        )
        auto_advice = "" if wired_limit else "leave it on Auto, "
        if nothing_fits:
            return (
                "No context fits in this Mac's unified memory with this model. The weights "
                "fit, but they leave too little for even the smallest context, so a load at "
                f"any length would over-commit. The GPU and the rest of the system share one "
                f"pool here, so there is nothing to offload to, and the load would bring the "
                f"machine down instead of reporting an error. Use a smaller or more quantized "
                f"GGUF, or free memory."
                f"{kv_hint} Set "
                f"{LlamaCppBackend.METAL_CTX_OVERCOMMIT_ENV}=1 to load it anyway."
            )
        return (
            f"A context of {requested_ctx:,} tokens does not fit in this Mac's unified "
            f"memory with this model. The largest that fits is {max_available_ctx:,} "
            "tokens. The GPU and the rest of the system share one pool here, so there is "
            "nothing to offload to, and the load would bring the machine down instead of "
            f"reporting an error. Lower the context to {max_available_ctx:,} or less, "
            f"{auto_advice}or use a more quantized GGUF."
            f"{kv_hint} Set "
            f"{LlamaCppBackend.METAL_CTX_OVERCOMMIT_ENV}=1 to load it anyway."
        )

    @staticmethod
    def _metal_context_pressure_message(
        requested_ctx: int, need_mib: float, free_budget_mib: float
    ) -> str:
        # need up, free down, so the printed pair cannot round into a tie
        need_gb = math.ceil(need_mib / 1024)
        free_gb = math.floor(max(0.0, free_budget_mib) / 1024)
        return (
            f"A context of {requested_ctx:,} tokens needs about {need_gb} GB of unified "
            f"memory, more than the {free_gb} GB Studio budgets from the memory free right "
            "now. It fits under the GPU's wired memory limit, so it is loading anyway, but "
            "macOS may have to compress or swap other apps to make room and generation may "
            "slow down. Lower the context or free memory to avoid that."
        )

    @staticmethod
    def _cuda_context_overcommit_notice(
        requested_ctx: int,
        max_available_ctx: int,
        cache_type_kv: Optional[str] = None,
        *,
        quantised_ctx_fits: bool = False,
    ) -> Optional[str]:
        """Advisory when a hand-set context exceeds what a discrete GPU holds (else None).

        Metal refuses this (shared wired pool panics the machine); a discrete GPU offloads
        layers to the CPU instead, so only warn.
        ``quantised_ctx_fits`` is the caller's q8_0 pricing verdict, not a guess.
        """
        if requested_ctx <= 0 or max_available_ctx <= 0:
            return None
        if requested_ctx <= max_available_ctx:
            return None
        kv_hint = ""
        if quantised_ctx_fits and (cache_type_kv or "f16").strip().lower() in (
            "f16",
            "fp16",
            "",
        ):
            kv_hint = " Setting the KV cache to q8_0 makes this context fit without shortening it."
        # Fires only after a --fit on or spill-plan hand-off: planned CPU offload on every OS.
        cause = (
            "The GPU cannot hold it, so layers will be moved to the CPU and generation "
            f"will be slower. Lower the context to {max_available_ctx:,} or less, or "
            "leave it on Auto."
        )
        return (
            f"A context of {requested_ctx:,} tokens does not fit in this GPU's memory with "
            f"this model. The largest that fits is {max_available_ctx:,} tokens. "
            f"{cause}{kv_hint}"
        )

    @staticmethod
    def _unmeasured_context_notice(
        requested_ctx: int, cache_type_kv: Optional[str] = None
    ) -> Optional[str]:
        """Advisory for a hand-set context with no KV estimate behind it: the short-context
        fallbacks used to cut a 262,144-token request to 4,096 and publish it (#9653). Not a
        refusal, which on a guess blocks live loads."""
        if requested_ctx <= 0:
            return None
        kv_hint = (
            " Setting the KV cache to q8_0 roughly halves what the context costs."
            if (cache_type_kv or "f16").strip().lower() in ("f16", "fp16", "")
            else ""
        )
        return (
            f"This model's GGUF does not carry the attention dimensions needed to size "
            f"the KV cache, so the {requested_ctx:,}-token context was launched as "
            f"requested without being checked against available memory. The reported "
            f"maximum context is that request, not a measured ceiling. If the load fails "
            f"or the machine runs short of memory, lower the context or leave it on "
            f"Auto.{kv_hint}"
        )

    # Skip the wait when the last kill is older than this; the driver has
    # already reclaimed the prior process's allocations.
    _VRAM_SETTLE_WINDOW_S: float = 15.0

    @staticmethod
    def _wait_for_vram_settle(
        max_wait: float = 2.0,
        interval: float = 0.25,
        tolerance_mib: int = 256,
        since_kill: float = 0.0,
    ) -> None:
        """Poll ``_get_gpu_free_memory`` until free VRAM stabilises.

        The driver reclaims a dead process's allocations asynchronously, so
        sampling free memory in the kill-to-spawn window reads artificially low
        and pushes GPU selection toward needless CPU offload (the Apply-reload
        OOM bare-shell launches never see).

        Short-circuits on cold start, stale kill (older than
        ``_VRAM_SETTLE_WINDOW_S``), CPU-only hosts, probe exceptions, and GPU-set
        changes. ``max_wait`` bounds wall-clock time so a wedged ``nvidia-smi``
        can't extend the reload.
        """
        now = time.monotonic()
        if since_kill <= 0.0:
            return
        if now - since_kill > LlamaCppBackend._VRAM_SETTLE_WINDOW_S:
            return
        deadline = now + max_wait

        def _probe_or_none():
            if time.monotonic() >= deadline:
                return None
            try:
                from utils.hardware import gpu_query

                # Consecutive samples are compared: a cached one would read as settled.
                with gpu_query.fresh_reads():
                    return LlamaCppBackend._get_gpu_free_memory()
            except Exception:
                return None

        prev = _probe_or_none()
        if prev is None or not prev:
            return
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            # Clip the nap so a near-zero ``max_wait`` is respected.
            time.sleep(min(interval, remaining))
            curr = _probe_or_none()
            if curr is None or not curr or len(curr) != len(prev):
                return
            prev_map = dict(prev)
            stable = True
            for idx, free in curr:
                if idx not in prev_map:
                    stable = False
                    break
                prev_free = prev_map[idx]
                # Adaptive: 2% of the larger sample dominates the 256 MiB floor.
                per_gpu_tol = max(tolerance_mib, int(max(free, prev_free) * 0.02))
                if abs(free - prev_free) >= per_gpu_tol:
                    stable = False
                    break
            if stable:
                return
            prev = curr

    # Free-VRAM fraction at which Unsloth pins the GPU directly instead of
    # deferring to ``--fit on``. 3% headroom: the compute buffer is now modelled in
    # the fit, so this only guards fragmentation + multi-GPU per-device CUDA context
    # (~2-3%); kept >= 3% as a floor (0.90 dropped 91-94% fits to CPU offload, #5106).
    _GPU_PIN_VRAM_FRACTION = 0.97

    # Fallback per-device tensor-mode compute buffer (MiB), used only when GGUF
    # dims are unavailable so _estimate_compute_buffer_bytes (the primary, derived
    # path) returns 0.
    _TENSOR_PARALLEL_BUFFER_RESERVE_MIB = 5120

    # Fixed per-device overhead on every GPU of a LAYER split (CUDA context +
    # scratch), beyond the conserved slot-scaling buffer. ~0.9 GB/device measured
    # (Qwen3.6-27B, b9625), independent of --parallel; reserved per extra GPU so a
    # tight layer split can't advertise a context that OOMs at load.
    _PIPELINE_PER_DEVICE_OVERHEAD_MIB = 1024

    # V cache types that llama.cpp can run WITHOUT flash attention. Only the V
    # axis has the dependency: a quantized V cache (q8_0/q4_0/q4_1/q5_0/q5_1/
    # iq4_nl) aborts init with "V cache quantization requires flash_attn", while
    # a quantized K cache runs fine without FA. So the flash-attn-off crash-
    # recovery fallback must reset a quantized V cache to f16 before it can
    # launch (and leaves K alone). These three are the only non-quantized types.
    _NON_QUANTIZED_KV_TYPES = frozenset({"f16", "bf16", "f32"})

    # Main-model placement settings that Manual mode owns. They must not leak
    # from Unsloth's parent environment into llama-server and silently override
    # the command assembled from the current request. Draft-model placement is
    # intentionally separate and remains available to speculative decoding.
    _MANUAL_PLACEMENT_ENV_VARS = (
        "LLAMA_ARG_CPU_MOE",
        "LLAMA_ARG_N_CPU_MOE",
        "LLAMA_ARG_N_GPU_LAYERS",
        "LLAMA_ARG_TENSOR_SPLIT",
        "LLAMA_ARG_FIT",
        "LLAMA_ARG_FIT_TARGET",
        "LLAMA_ARG_FIT_CTX",
    )

    # Eligibility must reject every inherited placement value replay removes.
    _CPU_FALLBACK_MAIN_PLACEMENT_ENV_VARS = (
        *_MANUAL_PLACEMENT_ENV_VARS,
        "LLAMA_ARG_DEVICE",
        "LLAMA_ARG_MAIN_GPU",
        "LLAMA_ARG_SPLIT_MODE",
        "LLAMA_ARG_OVERRIDE_TENSOR",
    )

    # (binary, mtime, model, kv types) that aborted on --split-mode tensor this
    # process (#6415 geometry limit, e.g. MQA n_head_kv=1). Model-keyed so one
    # model's abort doesn't skip tensor for others; tensor is tried by default,
    # recorded only on a real abort.
    # KV-type-keyed too: since ggml-org/llama.cpp#23792 the type reaches the tensor
    # child rather than being rewritten to f16, so an abort can be specific to one
    # type. iq4_nl asserts on b10441 where q4_0 runs on the same binary and model
    # (ggml-org/llama.cpp#27116); keyed only by (binary, model) that would disable
    # tensor mode for the types that work.
    _tensor_split_abort_keys: set[tuple[str, int, str, tuple[str, str]]] = set()

    # (binary, mtime) that refused a quantized KV cache under --split-mode tensor.
    # Coarser than _tensor_split_abort_keys on purpose: the pre-b9455 rejection is a
    # missing capability of the BINARY, not of one model or type, so keying it per
    # model+pair would pay another doomed full load for each new model and each
    # q8_0 -> q4_0 switch. f16 still splits on such a binary, so this is consulted
    # only when an axis is actually quantized.
    _tensor_quant_kv_unsupported_binaries: set[tuple[str, int]] = set()

    # Binary dirs already reported by _warn_missing_windows_cuda_runtime. The env is rebuilt
    # for every launch and every --list-devices probe, so one line per binary is enough.
    _missing_cuda_runtime_warned: set[str] = set()

    @classmethod
    def _binary_key(cls, binary: Optional[str]) -> Optional[tuple[str, int]]:
        """(path, mtime_ns); ns mtime re-probes a same-second binary swap."""
        if not binary:
            return None
        try:
            mtime = Path(binary).stat().st_mtime_ns
        except OSError:
            mtime = 0
        return (binary, mtime)

    @classmethod
    def _record_tensor_quant_kv_unsupported(cls, binary: Optional[str]) -> None:
        """Remember a binary that cannot run a quantized KV cache in tensor mode."""
        key = cls._binary_key(binary)
        if key is not None:
            cls._tensor_quant_kv_unsupported_binaries.add(key)

    @classmethod
    def _tensor_quant_kv_unsupported_binary(
        cls,
        binary: Optional[str],
        cache_types: tuple[str, str] = ("f16", "f16"),
    ) -> bool:
        """True if this binary already refused a quantized KV cache in tensor mode
        AND this launch asks for one. A non-quantized pair is unaffected."""
        if not any(_kv_bytes_per_elem(t) < _kv_bytes_per_elem("f16") for t in cache_types):
            return False
        key = cls._binary_key(binary)
        return key is not None and key in cls._tensor_quant_kv_unsupported_binaries

    @classmethod
    def _tensor_split_cache_key(
        cls,
        binary: Optional[str],
        model: Optional[str],
        cache_types: tuple[str, str] = ("f16", "f16"),
    ) -> Optional[tuple[str, int, str, tuple[str, str]]]:
        """(path, mtime_ns, model, (type_k, type_v)) key; ns mtime re-probes a
        same-second binary swap. ``cache_types`` defaults to llama.cpp's own
        default pair, which is what a caller that sets no type gets."""
        if not binary or not model:
            return None
        try:
            mtime = Path(binary).stat().st_mtime_ns
        except OSError:
            mtime = 0
        k, v = cache_types
        return (binary, mtime, model, (k.strip().lower(), v.strip().lower()))

    @classmethod
    def _tensor_split_aborts(
        cls,
        binary: Optional[str],
        model: Optional[str],
        cache_types: tuple[str, str] = ("f16", "f16"),
    ) -> bool:
        """True if this (binary, model, KV types) aborted on --split-mode tensor
        this session. A different cache type is a different question, so it is
        tried rather than assumed broken."""
        key = cls._tensor_split_cache_key(binary, model, cache_types)
        return key is not None and key in cls._tensor_split_abort_keys

    @classmethod
    def _record_tensor_split_abort(
        cls,
        binary: Optional[str],
        model: Optional[str],
        cache_types: tuple[str, str] = ("f16", "f16"),
    ) -> None:
        """Remember a (binary, model, KV types) that aborts on --split-mode tensor."""
        key = cls._tensor_split_cache_key(binary, model, cache_types)
        if key is not None:
            cls._tensor_split_abort_keys.add(key)

    # Keyed like the tensor latch: binary mtime drops entries after an update.
    _sched_reserve_abort_keys: set[tuple] = set()

    @classmethod
    def _sched_reserve_aborts(cls, binary: Optional[str], model: Optional[str]) -> bool:
        key = cls._tensor_split_cache_key(binary, model)
        return key is not None and key in cls._sched_reserve_abort_keys

    @classmethod
    def _record_sched_reserve_abort(cls, binary: Optional[str], model: Optional[str]) -> None:
        key = cls._tensor_split_cache_key(binary, model)
        if key is not None:
            cls._sched_reserve_abort_keys.add(key)

    @classmethod
    def _forget_sched_reserve_abort(cls, binary: Optional[str], model: Optional[str]) -> None:
        key = cls._tensor_split_cache_key(binary, model)
        if key is not None:
            cls._sched_reserve_abort_keys.discard(key)

    @staticmethod
    def _windows_pip_nvidia_dll_dirs(prefix: str) -> list[str]:
        """Return DLL dirs from pip-installed CUDA wheels under
        ``<prefix>/Lib/site-packages/`` so llama-server.exe can load
        ``cudart64_X.dll`` / ``cublas64_X.dll`` without a system CUDA toolkit.
        Mirrors the Linux ``nvidia/cu*/lib`` LD_LIBRARY_PATH block, covering the
        Windows wheel layouts seen in the wild:
          * ``nvidia/<pkg>/bin`` -- legacy modular wheels.
          * ``nvidia/<pkg>/bin/x86_64`` and ``.../bin/x64`` -- CUDA 13 layout
            for unsuffixed packages (#5106).
          * ``nvidia/<pkg>/Library/bin`` (and arch subdirs) -- conda repacks.
          * ``torch/lib`` -- PyTorch's CUDA-bundled wheel can ship
            ``cudart64_*.dll`` here; mirrors install_llama_prebuilt.py.

        Walks with ``Path.iterdir`` not ``glob.glob`` so it's safe against
        Windows paths containing ``[`` or ``]`` (valid in usernames)."""
        site_packages = Path(prefix) / "Lib" / "site-packages"
        out: list[str] = []
        seen: set[str] = set()

        def _add(path: Path) -> None:
            if not path.is_dir():
                return
            key = os.path.normcase(os.path.abspath(str(path)))
            if key in seen:
                return
            seen.add(key)
            out.append(str(path))

        nvidia_root = site_packages / "nvidia"
        if nvidia_root.is_dir():
            for pkg_dir in nvidia_root.iterdir():
                if not pkg_dir.is_dir():
                    continue
                # Arch-specific subdirs first so the explicit cudart64_X.dll
                # location wins over an empty sibling ``bin``.
                for sub in (
                    pkg_dir / "bin" / "x86_64",
                    pkg_dir / "bin" / "x64",
                    pkg_dir / "bin",
                    pkg_dir / "Library" / "bin" / "x86_64",
                    pkg_dir / "Library" / "bin" / "x64",
                    pkg_dir / "Library" / "bin",
                ):
                    _add(sub)
        _add(site_packages / "torch" / "lib")
        return out

    @classmethod
    def _warn_missing_windows_cuda_runtime(cls, binary_dir: str, path_dirs: list[str]) -> None:
        """Say so when a CUDA llama-server has no cudart to load. Diagnostic only.

        The CUDA prebuilt links ``cudart64_*.dll`` / ``cublas64_*.dll`` and takes them
        from the managed venv -- ``torch/lib`` or the ``nvidia/*`` wheels, per
        _windows_pip_nvidia_dll_dirs. A 2.11.0+cpu torch ships neither, so ggml cannot
        load its CUDA backend and ``llama-server.exe --list-devices`` prints
        ``Available devices: (none)`` while UNSLOTH_PREBUILT_INFO.json still says
        ``backend cuda`` (#8473, HF discussion 87). Today that is entirely silent.

        Changes nothing about the launch: the process still starts, still falls back to
        CPU, and a custom build with the DLLs somewhere else is not second-guessed.
        Never raises -- a diagnostic must not be able to stop a load.
        """
        try:
            if binary_dir in cls._missing_cuda_runtime_warned:
                return
            # Same identification _installed_ggml_backends uses: the official prebuilts are
            # single-backend, so the ggml CUDA lib beside llama-server IS the build.
            ggml_cuda = os.path.join(binary_dir, "ggml-cuda.dll")
            if not os.path.isfile(ggml_cuda):
                return
            for directory in path_dirs:
                try:
                    names = os.listdir(directory)
                except OSError:
                    continue
                if any(name.lower().startswith("cudart64_") for name in names):
                    return
            cls._missing_cuda_runtime_warned.add(binary_dir)
            logger.warning(
                "llama.cpp is the CUDA build (%s) but no cudart64_*.dll was found on its "
                "DLL search path. The CUDA ggml backend will not load and llama-server "
                "will report no devices. This is what a CPU-only PyTorch in the managed "
                "environment looks like; repair the installation to restore GPU support.",
                ggml_cuda,
            )
        except Exception as e:
            logger.debug(f"CUDA runtime DLL diagnostic failed: {e}")

    @staticmethod
    def _build_windows_path_dirs(binary_dir: str, prefix: str, cuda_path: str) -> list[str]:
        """Ordered PATH entries prepended so llama-server.exe resolves cudart /
        cublas DLLs: binary_dir, pip nvidia wheels, CUDA_PATH/bin, .../bin/x64.
        Extracted so test_windows_gpu_detection_mock tests the real logic. #5106."""
        path_dirs = [binary_dir]
        path_dirs.extend(LlamaCppBackend._windows_pip_nvidia_dll_dirs(prefix))
        if cuda_path:
            cuda_bin = os.path.join(cuda_path, "bin")
            if os.path.isdir(cuda_bin):
                path_dirs.append(cuda_bin)
            cuda_bin_x64 = os.path.join(cuda_path, "bin", "x64")
            if os.path.isdir(cuda_bin_x64):
                path_dirs.append(cuda_bin_x64)
        return path_dirs

    @staticmethod
    def _llama_server_env_for_binary(
        binary: str, *, use_system_rocm: bool = True
    ) -> dict[str, str]:
        """Build a subprocess env that lets llama-server resolve native libs.

        ``use_system_rocm=False`` skips the native-Linux system ROCm prepend.
        WSL's librocdxg prepend is independent and is not gated by this flag.
        """
        env = child_env_without_native_path_secret()
        # Every llama-server child inherits os.environ, so an off-meaning
        # GGML_CUDA_P2P is stripped HERE, not at one call site: chat, the STT
        # sidecar and the embedding probe all build from this, and any path that
        # skips it keeps the #10613 bug (presence, not value).
        LlamaCppBackend._sanitize_p2p_env(env)
        # _llama_lib_dir resolves the llama-server symlink to the real build/bin.
        binary_dir = str(_llama_lib_dir(binary))

        if sys.platform == "win32":
            # Ordering: see _build_windows_path_dirs. #5106.
            path_dirs = LlamaCppBackend._build_windows_path_dirs(
                binary_dir,
                sys.prefix,
                os.environ.get("CUDA_PATH", ""),
            )
            existing_path = env.get("PATH", "")
            env["PATH"] = ";".join(path_dirs) + ";" + existing_path
            # Warn against the FULL search path, inherited entries included: a hand-installed CUDA
            # toolkit puts cudart64_*.dll on PATH without the venv or CUDA_PATH knowing, and warning
            # on the prepended directories alone told working custom setups to repair a fine install.
            _full_search_path = path_dirs + [d for d in existing_path.split(";") if d]
            LlamaCppBackend._warn_missing_windows_cuda_runtime(binary_dir, _full_search_path)

            # ROCm: the prebuilt bundles rocblas.dll but NOT the Tensile
            # kernel files (rocblas/library/*.dat + *.hsaco); the DLL searches
            # <binary_dir>/rocblas/library/ which doesn't exist.
            _hip_path = os.environ.get("HIP_PATH", os.environ.get("ROCM_PATH", ""))
            if _hip_path:
                _rocblas_lib = os.path.join(_hip_path, "bin", "rocblas", "library")
                if os.path.isdir(_rocblas_lib):
                    env.setdefault("ROCBLAS_TENSILE_LIBPATH", _rocblas_lib)
        elif sys.platform == "darwin":
            # dyld ignores LD_LIBRARY_PATH, so the Linux branch below left the
            # child with no search path at all, while the installer's own
            # validation sets DYLD_LIBRARY_PATH and so never saw it (#8566).
            # A healthy prebuilt resolves via @loader_path and does not need
            # this; one with wrong install names dies in dyld without it.
            _dyld = _loader_path_var()
            env[_dyld] = _prepend_loader_dir(env.get(_dyld, ""), binary_dir)
        else:
            # Linux: LD_LIBRARY_PATH for shared libs next to the binary plus
            # CUDA runtime libs (libcudart, libcublas, etc.)
            import platform

            lib_dirs = []
            # WSL: system HIP before the bundle's (which segfaults on /dev/dxg).
            lib_dirs.extend(_wsl_system_rocm_lib_dirs())
            if lib_dirs:
                env.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
            # Native Linux AMD: system ROCm libs before the bundle's HIP runtime,
            # which can be incompatible with the host amdkfd driver (#7233).
            if use_system_rocm and not LlamaCppBackend._prefers_bundle_only_rocm(binary):
                lib_dirs.extend(_native_linux_system_rocm_lib_dirs(binary_dir))
            lib_dirs.append(binary_dir)
            _arch = platform.machine()  # x86_64, aarch64, etc.

            # Pip-installed nvidia CUDA runtime libs. The prebuilt binary links
            # libcudart.so.13 / libcublas.so.13 which live here, not in
            # /usr/local/cuda.
            import glob as _glob

            # glob.escape: a prefix with [brackets] is otherwise read as a pattern.
            _site = os.path.join(_glob.escape(sys.prefix), "lib", "python*", "site-packages")
            _pip_dirs = []
            for _nv_pattern in [
                os.path.join(_site, "nvidia", _sub, "lib") for _sub in ("cu*", "cudnn", "nvjitlink")
            ] + [os.path.join(_site, "torch", "lib")]:
                for _nv_dir in _glob.glob(_nv_pattern):
                    if os.path.isdir(_nv_dir):
                        _pip_dirs.append(_nv_dir)

            _system_dirs = []
            for cuda_lib in [
                "/usr/local/cuda/lib64",
                f"/usr/local/cuda/targets/{_arch}-linux/lib",
                # Fallback CUDA compat paths (e.g. binary built with CUDA 12
                # where default /usr/local/cuda is CUDA 13+).
                "/usr/local/cuda-12/lib64",
                "/usr/local/cuda-12.8/lib64",
                f"/usr/local/cuda-12/targets/{_arch}-linux/lib",
                f"/usr/local/cuda-12.8/targets/{_arch}-linux/lib",
            ]:
                if os.path.isdir(cuda_lib):
                    _system_dirs.append(cuda_lib)
            from utils.tegra import TEGRA_LIB_DIR, is_tegra

            if is_tegra():
                # Jetson: JetPack's CUDA first; the generic pip builds load but crash on unified memory (#4862).
                _tegra = [TEGRA_LIB_DIR] if os.path.isdir(TEGRA_LIB_DIR) else []
                lib_dirs.extend(_tegra + _system_dirs + _pip_dirs)
            else:
                lib_dirs.extend(_pip_dirs + _system_dirs)

            # Vendored dirs go last: rescue only, never displace a runtime already found.
            from utils.llama_cpp_freshness import read_install_marker
            from utils.prebuilt.runtime_libs import vendored_cuda_runtime_dirs

            marker_binary = str(_resolve_llama_binary(binary))
            vendored_cuda_dirs = vendored_cuda_runtime_dirs(read_install_marker(marker_binary))
            existing_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(
                path for path in [*lib_dirs, existing_ld, *vendored_cuda_dirs] if path
            )

        return env

    @classmethod
    def _clear_manual_placement_env(cls, env: dict[str, str]) -> None:
        """Remove inherited main-model placement owned by Manual mode."""
        for name in cls._MANUAL_PLACEMENT_ENV_VARS:
            env.pop(name, None)

    @classmethod
    def _clear_device_placement_env(cls, env: dict[str, str]) -> None:
        """Remove inherited device placement owned by an explicit ``gpu_ids``."""
        for name in ("LLAMA_ARG_DEVICE", "LLAMA_ARG_MAIN_GPU"):
            env.pop(name, None)

    @classmethod
    def _clear_split_placement_env(cls, env: dict[str, str]) -> None:
        """Remove an inherited split mode / ratio when the arch gate drops ours.

        These are the env spelling of --split-mode / --tensor-split, so stripping the
        tokens alone leaves the inherited value in force. It matters more here: a
        tensor split aborts a child with no visible device, and a narrowed set
        re-indexes the survivors so an inherited positional ratio lands on the wrong
        card."""
        for name in ("LLAMA_ARG_SPLIT_MODE", "LLAMA_ARG_TENSOR_SPLIT"):
            env.pop(name, None)

    @staticmethod
    def _select_gpus(
        model_size_bytes: int,
        gpus: list[tuple[int, int]],
        usable_fraction: Optional[float] = None,
        total_by_idx: Optional[dict[int, int]] = None,
        per_device_overhead_bytes: int = 0,
        min_gpus: int = 1,
        shared: frozenset = frozenset(),
    ) -> tuple[Optional[list[int]], bool]:
        """Pick GPU(s) for a model from estimated VRAM and free memory.

        ``shared``: GPUs another loaded model runs on. A model that fits one card alone
        takes a card outside it when one fits, so two models don't share compute.

        ``min_gpus`` (default 1, capped at ``len(gpus)``) keeps a downgraded
        tensor/multi-GPU request spread instead of collapsing to one card.

        ``model_size_bytes`` should include weights and estimated KV cache.
        ``usable_fraction`` (default ``_GPU_PIN_VRAM_FRACTION``) provides
        headroom for compute buffers, CUDA context, and other runtime
        overhead; callers lower it when MTP reserves VRAM for a draft model.
        ``total_by_idx`` (index -> total MiB) makes the headroom an ABSOLUTE
        ``(1 - fraction) * total`` per GPU instead of a fraction of free.
        ``per_device_overhead_bytes`` is the fixed layer-split cost per GPU beyond
        the first; a k-GPU pin must hold ``model + (k-1) * overhead`` or it can OOM
        a device after -ngl -1 (no --fit fallback). Single-GPU adds none.

        Returns (gpu_indices, use_fit):
          - ([1], False)       fits on 1 GPU at the headroom threshold
          - ([1, 2], False)    needs 2 GPUs
          - (None, True)       too large, let --fit handle it
        """
        if not gpus:
            return None, True

        min_gpus = max(1, min(min_gpus, len(gpus)))
        model_size_mib = model_size_bytes / (1024 * 1024)
        if usable_fraction is None:
            usable_fraction = _active_vram_fraction()
        overhead_mib = per_device_overhead_bytes / (1024 * 1024)

        # Per-GPU usable budget: free minus the reserve when total is known, else
        # the legacy free*frac (also covers a total-0 two-column probe).
        _sysmem_fallback = LlamaCppBackend._sysmem_fallback_risk()

        def _usable(idx: int, free_mib: int) -> float:
            t = total_by_idx.get(idx, 0) if total_by_idx else 0
            usable = _vram_usable_mib(
                free_mib, t, usable_fraction, sysmem_fallback = _sysmem_fallback
            )
            return max(0.0, usable) if t > 0 else usable

        # Rank by usable budget (free - reserve), not raw free: a more-used large
        # card can have less usable room than a less-used small one.
        ranked = sorted(gpus, key = lambda g: _usable(g[0], g[1]), reverse = True)

        # Cap a downgraded multi-GPU request to the usable count so it doesn't pull
        # in a near-full card to hit min_gpus. No-op for the default min_gpus == 1.
        usable_count = sum(1 for idx, free_mib in ranked if _usable(idx, free_mib) > overhead_mib)
        min_gpus = max(1, min(min_gpus, usable_count or 1))

        # Try 1 GPU at the usable-VRAM threshold (only when one device is allowed).
        if min_gpus <= 1 and _usable(ranked[0][0], ranked[0][1]) >= model_size_mib:
            for idx, free_mib in ranked:
                if idx not in shared and _usable(idx, free_mib) >= model_size_mib:
                    return [idx], False
            return [ranked[0][0]], False

        # Try N GPUs (most-free first); each past the first adds per-device overhead.
        # Require at least min_gpus devices before accepting a fit.
        cumulative = 0.0
        selected = []
        for idx, free_mib in ranked:
            selected.append(idx)
            cumulative += _usable(idx, free_mib)
            if (
                len(selected) >= min_gpus
                and cumulative >= model_size_mib + (len(selected) - 1) * overhead_mib
            ):
                return sorted(selected), False

        # Too large even for all GPUs; let --fit handle it
        logger.debug(
            "Model does not fit in available GPU memory, falling back to --fit",
            model_size_mib = round(model_size_mib, 2),
            ranked_gpus = ranked,
        )
        return None, True

    @staticmethod
    def _select_gpus_split_aware(
        model_size_bytes: int,
        gpus: list[tuple[int, int]],
        usable_fraction: Optional[float] = None,
        total_by_idx: Optional[dict[int, int]] = None,
        per_device_overhead_bytes: int = 0,
        min_gpus: int = 1,
        split_extra_bytes: int = 0,
        shared: frozenset = frozenset(),
    ) -> tuple[Optional[list[int]], bool]:
        """``_select_gpus``, re-checked at the multi-device context-compute rate.

        The device count comes out of the footprint, so the first pass must price the
        compute buffer at the single-device rate -- but a layer split replicates a BIGGER
        one per card (see ``_CTX_COMPUTE_SPLIT_MULT``). ``split_extra_bytes`` is that
        per-device step (0 when llama.cpp declines pipeline parallelism): charge it and
        re-select once the count is known to be > 1, exact in one retry since the step is
        count-independent. The bigger overhead can cut ``_select_gpus``'s usable-card
        count to one, and a lone card is not a split, so a retry landing below two
        devices is re-priced without the delta. Still no fit returns (None, True) ->
        --fit on, i.e. CPU offload rather than a pin that OOMs at launch."""
        gpu_indices, use_fit = LlamaCppBackend._select_gpus(
            model_size_bytes,
            gpus,
            usable_fraction = usable_fraction,
            total_by_idx = total_by_idx,
            per_device_overhead_bytes = per_device_overhead_bytes,
            min_gpus = min_gpus,
            shared = shared,
        )
        if use_fit or split_extra_bytes <= 0 or not gpu_indices or len(gpu_indices) < 2:
            return gpu_indices, use_fit
        retry_indices, retry_fit = LlamaCppBackend._select_gpus(
            model_size_bytes + split_extra_bytes,
            gpus,
            usable_fraction = usable_fraction,
            total_by_idx = total_by_idx,
            per_device_overhead_bytes = per_device_overhead_bytes + split_extra_bytes,
            min_gpus = min_gpus,
        )
        if not retry_fit and retry_indices and len(retry_indices) >= 2:
            return retry_indices, retry_fit
        single, single_fit = LlamaCppBackend._select_gpus(
            model_size_bytes,
            gpus,
            usable_fraction = usable_fraction,
            total_by_idx = total_by_idx,
            per_device_overhead_bytes = per_device_overhead_bytes,
            min_gpus = 1,
            shared = shared,
        )
        if not single_fit and single is not None and len(single) == 1:
            return single, False
        return retry_indices, retry_fit

    @staticmethod
    def _every_gpu_holds_reserve(usable_mib: Iterable[float], reserve_bytes: int) -> bool:
        """Whether every card in a layer-split subset can hold the reserve it replicates
        (flat per-device overhead + its own context-compute copy). A pooled budget cannot
        see this: in the sum a roomy card's spare VRAM covers a nearly full card's copy,
        and the small card OOMs at launch. Bites when the subset loop cannot start at
        one GPU (``_auto_min_gpus`` >= 2, set by every tensor -> layer downgrade);
        starting at one, the failed n-1 subset already bounds the smallest card from
        below by the reserve."""
        reserve_mib = reserve_bytes / (1024 * 1024)
        values = list(usable_mib)
        return bool(values) and all(usable >= reserve_mib for usable in values)

    @staticmethod
    def _cap_ctx_to_per_device_reserve(
        ctx: int,
        usable_mib: Iterable[float],
        reserve_bytes_fn: Callable[[int], int],
        min_ctx: int = 8192,
    ) -> int:
        """Largest context in [min_ctx, ctx] whose reserve fits every selected GPU.

        Bisect because reserves are monotone but may include fixed costs. Use the
        same MiB comparison as _every_gpu_holds_reserve. Return 0 if min_ctx fails."""
        values = list(usable_mib)
        if not values or ctx < min_ctx:
            return 0
        smallest = min(values)
        lo, hi, best = min_ctx, ctx, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if reserve_bytes_fn(mid) / (1024 * 1024) <= smallest:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        best = (best // 256) * 256  # same alignment _fit_context_to_vram uses
        return best if best >= min_ctx else 0
