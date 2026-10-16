# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: lifecycle, tensor spill planning."""

import os

from typing import (
    Iterable,
    Mapping,
    Optional,
)
from core.inference.llama_server_args import (
    _DEVICE_FLAGS,
    _GPU_LAYER_FLAGS,
    fit_target_margin_in,
)

from .batch_args import (
    _extra_args_n_parallel,
    _extra_args_split_mode,
    _extra_args_tensor_split,
)
from .binary_runtime import _HOST_RAM_HEADROOM_MIB
from .draft_args import (
    _extra_args_device,
    _extra_args_draft_device_pin,
    _sidecar_adapter_bytes,
    _without_subsequence,
)
from .launch_flags import (
    _args_place_tensors_on_cpu,
    _env_places_tensors_on_cpu,
    _kv_offload_from_args,
    _spilled_decode_threads,
)
from .log import logger
from .memory_policy import (
    _TENSOR_SPLIT_FLAGS,
    _extra_args_set_any_flag,
)


class _TensorSpillMixin:
    def _planned_tensor_spill(
        self,
        inputs: Optional[dict],
        *,
        extra_args: Optional[Iterable[str]] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> "Optional[SpillPlan]":
        """A ``-ot`` spill plan for a load Unsloth could not fit, or None to abstain.

        Abstaining reproduces today's ``--fit on`` byte for byte, so every
        uncertain input abstains rather than guessing. That is the whole safety
        argument: the only behaviour that changes is the case where a plan was
        produced, and a plan is only produced when the numbers are known.

        The planner must NOT run when anything else owns placement. Pricing a
        placement the child will not actually get is the single bug class behind
        most of the review traffic on the load-mode work, and the cheap way to
        avoid it is to decline rather than to model the override.
        """
        from core.inference.offload_cost_model import HostProfile
        from core.inference.offload_planner import (
            PlanOptions,
            plan_placement,
            smart_offload_enabled,
        )

        source_env = os.environ if env is None else env
        if not smart_offload_enabled(source_env) or not inputs:
            return None

        # Unified memory: spilling moves bytes inside one pool, so the trade does
        # not exist. Metal also keeps mmap zero copy via buffer_from_host_ptr.
        from utils.hardware import is_apple_silicon

        if is_apple_silicon():
            return None

        # An integrated CUDA SoC (Jetson, DGX Spark) is unified memory too, and
        # nothing else here sees it: _amd_apu_wants_unified_memory answers only for
        # ROCm, and shared_gpu_ids is populated only on Vulkan, which reports total
        # 0 for an iGPU. CUDA states it directly (cudaDeviceProp::integrated, torch's
        # is_integrated), and Studio's diffusion policy already classifies such a
        # device as unified memory for exactly this reason -- CPU offload frees
        # nothing there (diffusion_memory.py:292-294). A spill would count one pool
        # twice and then pin the result with --fit off.
        if self._integrated_cuda_unified_memory(inputs.get("gpu_indices")):
            logger.debug("Tensor spill: declined, an integrated CUDA device is unified memory")
            return None

        # Someone else owns the placement -- decline. Each of these re-places the
        # model out from under a plan made here: -ot / --cpu-moe (or its env twin)
        # already moves tensors, a layer count overrides -ngl -1, a device list
        # changes which cards the resident half was sized against.
        if (
            _args_place_tensors_on_cpu(extra_args)
            or _env_places_tensors_on_cpu(source_env)
            or _extra_args_set_any_flag(extra_args, _GPU_LAYER_FLAGS)
            or str(source_env.get("LLAMA_ARG_N_GPU_LAYERS", "")).strip()
            or _extra_args_set_any_flag(extra_args, _DEVICE_FLAGS)
            or str(source_env.get("LLAMA_ARG_DEVICE", "")).strip()
            # A pinned DRAFT device is placement too and is not in _DEVICE_FLAGS.
            # Its reserve rides in extra_resident_bytes, which
            # _per_device_shortfall books onto device 0, while llama.cpp puts the
            # drafter on the pinned card -- approving a footprint for the wrong
            # device, then pinning it with --fit off. cpu/none are not a pin.
            or _extra_args_draft_device_pin(extra_args)
            # -sm row and -sm tensor split WITHIN a tensor rather than handing out
            # whole rows of layers, so the row model below does not describe them.
            # `tensor` is a real accepted value (common/arg.cpp:2762-2776) and goes
            # further than `row`: llama.cpp collapses every card into ONE meta
            # device (llama.cpp:157-215), so per-device budgets are meaningless,
            # and llama.cpp's own fitter refuses it (common/fit.cpp:182-183).
            # `none` and `layer` ARE describable and are handled after this gate.
            or (_extra_args_split_mode(extra_args, source_env) in ("row", "tensor"))
            # An INHERITED -sm none may never reach the child: on a layer-split
            # launch load_model pops LLAMA_ARG_SPLIT_MODE (plus the paired
            # LLAMA_ARG_TENSOR_SPLIT) for any non-"layer" mode
            # (llama_cpp.py:20452-20458). That runs AFTER this and its predicate is
            # not visible here, so budgeting the single main GPU below would plan
            # against a mode the child never sees and skip the per-device check
            # entirely (it short-circuits on one device). argv is the only spelling
            # that provably survives to the child.
            or (
                _extra_args_split_mode(extra_args, source_env) == "none"
                and _extra_args_split_mode(extra_args, {}) != "none"
            )
            # -mg is declined above as a device flag; its env twin is not a flag
            # and reaches the child anyway. Under -sm none llama.cpp keeps
            # devices[main_gpu] (llama.cpp:288-299), so budgeting the first
            # retained card would price the wrong GPU.
            or str(source_env.get("LLAMA_ARG_MAIN_GPU", "")).strip()
            # ANY explicit --fit, not just an enabling one. A retry revokes the
            # plan by appending "--fit on", and extras are appended BEFORE that,
            # so planning over a user's "--fit off" would let the revocation
            # silently overturn it. Same test _fit_off_retry_eligible uses.
            or any(
                str(a) in ("-fit", "--fit") or str(a).startswith(("-fit=", "--fit="))
                for a in (extra_args or ())
            )
            or str(source_env.get("LLAMA_ARG_FIT", "")).strip()
            # --rpc devices are REMOTE and go at the FRONT of the list
            # (llama.cpp:275), so the free-memory row split hands the first slice
            # of layers to a card Studio never detected. Every number here comes
            # from the local `gpus` snapshot, so the plan would credit local VRAM
            # for layers that went elsewhere and pin it with --fit off.
            or (_extra_args_device(extra_args, {"--rpc"}) or "").strip()
            or str(source_env.get("LLAMA_ARG_RPC", "")).strip()
            # --fit-target is VRAM the user told the fitter it may not spend
            # (common/fit.cpp:282-288). Emitting --fit off skips
            # common_params_fit_impl and drops the margin silently, while this
            # planner spends it. Decline, like any other explicit --fit.
            or fit_target_margin_in(extra_args, source_env) is not None
        ):
            logger.debug("Tensor spill: declined, pass-through arguments own the placement")
            return None

        # Slots are SIZING, not placement, and these numbers were priced at Unsloth's
        # own --parallel. A pass-through is appended after it and llama.cpp is
        # last-wins, so a larger one grows the attention cache and the recurrent
        # state under a deficit computed for the smaller count. Only larger matters:
        # a smaller value leaves the plan over-reserved, the safe direction.
        priced_parallel = int(inputs.get("n_parallel") or 1)
        override_parallel = _extra_args_n_parallel(extra_args, source_env)
        if override_parallel is not None and override_parallel > priced_parallel:
            logger.debug(
                "Tensor spill: declined, --parallel %d overrides the %d slots this was priced at",
                override_parallel,
                priced_parallel,
            )
            return None

        model_size = int(inputs.get("model_size") or 0)
        kv_cache_bytes = int(inputs.get("kv_cache_bytes") or 0)
        rows = list(inputs.get("gpus") or ())
        if not model_size or not kv_cache_bytes or not rows:
            return None
        if not self._can_estimate_kv():
            return None

        layout = self._tensor_spill_layout(inputs.get("model_path"))
        if layout is None or not layout.complete:
            return None

        # Per-device VRAM the planner may credit. SHARED rows are dropped, not
        # summed: an APU or iGPU reports host RAM as VRAM, so crediting it and then
        # spilling into the same RAM counts one pool twice.
        pinned = inputs.get("gpu_indices")
        shared = set(inputs.get("shared_gpu_ids") or ())
        pinned_set = None if pinned is None else set(pinned)
        # Dropping a shared device from the BUDGET only helps if the child stops
        # using it, and nothing here can promise that: the auto-Vulkan arm pins
        # every DETECTED gpu, iGPU included, and runs after this snapshot (so
        # gpu_indices reads None here); with no --device the child sees every card
        # anyway. llama.cpp splits rows by free memory and an iGPU's "free" is the
        # whole host pool, so it draws the MAJORITY of the layers and their caches
        # -- the opposite of this planner's premise. Decline instead.
        if shared and (pinned_set is None or (shared & pinned_set)):
            logger.debug(
                "Tensor spill: declined, a shared-memory GPU stays in the child's device list"
            )
            return None
        # The USABLE budget Unsloth's fit tested, not raw free: it carries the
        # VRAM-budget fraction and the per-card reserve floor. Falling back to free
        # keeps a caller that supplies no map working.
        usable_mib = inputs.get("gpu_usable_mib") or {}
        kept = [
            (idx, free_mib)
            for idx, free_mib in rows
            if idx not in shared and (pinned_set is None or idx in pinned_set)
        ]
        # -sm none is not a reason to decline, it is a device list of one:
        # llama.cpp keeps only devices[main_gpu] (llama.cpp:288-299), the
        # single-GPU case this planner has the most evidence for. -mg and
        # LLAMA_ARG_MAIN_GPU are declined above, so main_gpu is llama.cpp's default
        # of 0 -- the first kept row. ARGV only, deliberately: the env spelling is
        # declined above because load_model may strip it before launch, so it does
        # not describe the child's device list.
        if _extra_args_split_mode(extra_args, {}) == "none" and kept:
            kept = kept[:1]
        vram_per_device = [
            int(usable_mib.get(idx, free_mib) * 1024 * 1024) for idx, free_mib in kept
        ]
        # llama.cpp sizes its row ranges from RAW free VRAM (llama-model.cpp:1433),
        # so the split must be modelled on the raw numbers. The budget subtracts a
        # reserve that depends on each card's TOTAL, preserving the ratios only when
        # every card has the same free/total. Same filter and order as above, so
        # index i means the same device in both lists.
        split_weights = [int(free_mib * 1024 * 1024) for _idx, free_mib in kept]
        # -ts REPLACES that free-memory split rather than skewing it: llama.cpp
        # copies tensor_split into `splits` verbatim and prefix-sums it
        # (llama-model.cpp:1436-1447), so it IS the row-ownership weight, not a
        # reason to decline. A share of 0 draws no rows, which _device_slots
        # already models, and the length must match `kept` or the two lists stop
        # describing the same devices.
        # Read the env the CHILD will get: load_model pops LLAMA_ARG_TENSOR_SPLIT
        # together with a non-layer LLAMA_ARG_SPLIT_MODE (llama_cpp.py:20505-20511),
        # so an inherited -ts riding alongside such a mode never reaches the child
        # -- proving row ownership against the scrubbed shares approves the wrong
        # device and pins it with --fit off. argv is untouched by that
        # reconciliation, so it is always read.
        _inherited_sm = str(source_env.get("LLAMA_ARG_SPLIT_MODE", "")).strip().lower()
        _ts_env = source_env if _inherited_sm in ("", "layer") else {}
        _ts = _extra_args_tensor_split(extra_args, _ts_env)
        if _ts is None and (
            _extra_args_set_any_flag(extra_args, _TENSOR_SPLIT_FLAGS)
            or str(_ts_env.get("LLAMA_ARG_TENSOR_SPLIT", "")).strip()
        ):
            # A -ts we could not read is NOT "no -ts": the child still gets one,
            # and free-VRAM proportions under it are the shortfall this avoids.
            logger.debug("Tensor spill: declined, -ts is present but could not be parsed")
            return None
        if _ts is not None:
            if len(_ts) < len(kept):
                logger.debug(
                    "Tensor spill: declined, -ts names %d shares for %d devices",
                    len(_ts),
                    len(kept),
                )
                return None
            split_weights = _ts[: len(kept)]
        if not vram_per_device:
            return None
        avail_mib = self._available_system_memory_mib()
        if avail_mib is None:
            return None

        # A separate drafter is a second MODEL, and an UNPINNED one is placed like
        # any other: common_base_params_to_speculative copies the draft device list
        # verbatim (common/speculative.cpp:2319-2331), and an empty one drops
        # llama_prepare_model_devices into its default enumeration of every visible
        # GPU (llama.cpp:184-276), so the drafter's weights AND its context follow
        # the free-memory row split onto every card. A PINNED one already declined
        # above. Its reserve rides in extra_resident_bytes, which
        # _per_device_shortfall books entirely on device 0, so every other card is
        # under-booked by its share -- and --fit off means common/fit.cpp never runs
        # to catch the overflow. Nothing here says how llama.cpp will split a model
        # this layout knows nothing about, so decline, like the pass-through adapter
        # below.
        if inputs.get("separate_draft_on_gpu") and len(vram_per_device) > 1:
            logger.debug(
                "Tensor spill: declined, a separate drafter's bytes cannot be placed "
                "across %d devices",
                len(vram_per_device),
            )
            return None

        # The trailing nextn/MTP blocks the layout dropped, charged back when a
        # draft will engage. Unsloth's budget already assumed they were paid for (an
        # embedded head contributes 0 to _mtp_draft_weights because its weights sit
        # inside model_size), but the planner rebuilds from the tensor table, where
        # they are gone -- leaving the deficit short by the whole MTP head.
        extra_gpu_bytes = int(inputs.get("extra_gpu_bytes") or 0)
        if inputs.get("mtp_will_engage") and layout.excluded_block_bytes:
            extra_gpu_bytes += layout.excluded_block_bytes

        # The compute buffer, on the same terms as the fit that sent us here:
        # ctx_compute is per device, compute_buffer_flat is one lump charged once
        # against the pool and booked onto device 0, where it lives.
        extra_gpu_bytes += int(inputs.get("compute_buffer_flat") or 0)

        # A projector Studio never resolved but the child loads anyway. arg.cpp
        # applies LLAMA_ARG_MMPROJ before argv (common/arg.cpp:780-802) and a
        # LLAMA_ARG_MMPROJ_URL download overwrites even the --mmproj this launch
        # emits (common/arg.cpp:500-503, :632-634), and the projector is GPU
        # resident unless --no-mmproj-offload says otherwise. extra_gpu_bytes above
        # carries only the RESOLVED projector, so those bytes sit in no term here
        # and the plan pins the shortfall with --fit off. The seam hands over the
        # same numbers the load-mode fit used, including its abstain for a URL or
        # an unreadable path.
        if inputs.get("env_mmproj_unsized"):
            logger.debug("Tensor spill: declined, an inherited projector could not be sized")
            return None
        extra_gpu_bytes += int(inputs.get("env_mmproj_bytes") or 0)

        # Pass-through --lora / --lora-scaled / --control-vector weights, GPU
        # resident and in no other term here. Both land on the base layer's buffer
        # type (llama-adapter.cpp:335, :67) and neither mmaps: the file is staged
        # with ggml_backend_tensor_set (:407), so the bytes are resident whatever
        # --load-mode says, and `model_size` is the base GGUF alone. Same treatment
        # _fits_without_paging gives them: a term, and an unreadable file abstains.
        sidecar_bytes = _sidecar_adapter_bytes(extra_args)
        if sidecar_bytes is None:
            logger.debug("Tensor spill: declined, a pass-through adapter could not be sized")
            return None
        # ...but a flat term is only right on ONE device. extra_resident_bytes is
        # charged entirely to device 0 (_per_device_shortfall), while a LoRA tensor
        # is allocated on its BASE tensor's buffer (llama-adapter.cpp:335) and a
        # control-vector row on model.select_buft(il) (:67) -- so on a layer split
        # they follow the layers onto every card. Booking them all on the main GPU
        # under-books the others, and --fit off means common/fit.cpp never runs to
        # catch the overflow. The layout is built from the base GGUF alone and holds
        # no adapter tensor table, so the per-device share cannot be derived here.
        # Decline, like every other placement this planner cannot price.
        if sidecar_bytes and len(vram_per_device) > 1:
            logger.debug(
                "Tensor spill: declined, a pass-through adapter's bytes cannot be placed "
                "across %d devices",
                len(vram_per_device),
            )
            return None
        extra_gpu_bytes += sidecar_bytes

        decode_threads = _spilled_decode_threads(
            inputs.get("n_threads"),
            extra_args,
        )
        if decode_threads is None:
            logger.debug("Tensor spill: declined, decode CPU settings cannot be priced")
            return None

        return plan_placement(
            layout,
            vram_per_device,
            avail_mib * 1024 * 1024,
            int(inputs.get("n_ctx") or 0),
            split_weights_per_device = split_weights,
            kv_layer_weights = list(inputs.get("kv_layer_weights") or ()),
            opts = PlanOptions(
                overhead_bytes_per_device = (
                    int(inputs.get("soft_overhead") or 0)
                    + int(inputs.get("ctx_compute_per_device") or 0)
                ),
                # Charged for the devices AFTER the first, matching every other
                # use of this constant in this file (:18664, :18922, and the
                # `n > 1` guards at :18761 and :19186). Added to the flat
                # per-device term it also came off device 0, which withheld a GiB
                # of a single card that no split was ever going to allocate --
                # phantom deficit the planner covered by spilling real blocks.
                pipeline_overhead_bytes = self._PIPELINE_PER_DEVICE_OVERHEAD_MIB * 1024 * 1024,
                # Module-level, unlike the line above, so not reachable through self.
                host_ram_headroom_bytes = _HOST_RAM_HEADROOM_MIB * 1024 * 1024,
                # -nkvo is not a reason to decline: it moves the cache and the
                # recurrent state OUT of VRAM, so the deficit is smaller, not
                # larger.
                kv_on_host = not _kv_offload_from_args(extra_args, env = source_env),
                # GPU-resident allocations the layout cannot see, since it is built
                # from the target GGUF alone: the vision projector and the MTP draft
                # reserve. Both are in the footprint that produced the use_fit
                # verdict, so omitting them makes the deficit too small.
                extra_resident_bytes = extra_gpu_bytes,
                # Spilled decode runs on the CPU backend (ggml migrates an op to the
                # GPU only at batch >= 32, and decode is batch 1), so the penalty
                # tracks core count. Read the real one, not a default.
                host = HostProfile(
                    threads = decode_threads,
                    # Wired, not defaulted. On a unified-memory APU the credited
                    # "VRAM" IS system RAM, so an -ot spill frees no device memory
                    # and only buys the CPU backend's slower read path. The planner
                    # abstains on those, but only if told: left at False it would
                    # plan a spill on every Strix Halo (gfx1151), Strix Point
                    # (gfx1150), Krackan (gfx1152) and Phoenix iGPU. Unsloth already
                    # classifies these from the driver's integrated flag, then
                    # gcnArchName, then the device-name substrings.
                    unified_memory = self._amd_apu_wants_unified_memory(inputs.get("gpu_indices")),
                ),
            ),
            # The cache Unsloth already priced for THIS launch, byte-accurately:
            # cache dtype, SWA windows, MLA latents, unified vs per-slot streams,
            # cell padding and flash-attention V padding. layout.kv_bytes() is a
            # bare f16 GQA product with none of those, so on an MLA or f32-cache
            # load it lands well UNDER the real cache. A floor, not a replacement,
            # so the layout still wins where it is the more conservative.
            kv_bytes_floor = kv_cache_bytes,
        )

    @staticmethod
    def _spill_plan_flags_for(plan: "Optional[SpillPlan]") -> "list[str]":
        """The argv tokens for ``plan``, or ``[]`` when it must not be emitted.

        The emptiness test is the point. ``Plan`` is a dataclass with no
        ``__bool__``/``__len__``, so EVERY instance is truthy -- including the
        ones ``plan_placement`` returns to report that it could not place the
        load (``insufficient``), that it abstained (incomplete layout, unified
        memory, no creditable VRAM), or that the load already fits. Emitting for
        those would append ``-ngl -1 --fit off`` with no tensor moved to the
        host, i.e. pin every layer to the GPU and switch the fitter off on
        exactly the loads the planner just said do not fit. Only a plan that
        actually spills something earns the flags; everything else falls through
        to ``--fit on``, which is what this arm did before the planner existed.

        One flag per pattern: llama-server ACCUMULATES repeated ``-ot``
        (common/arg.cpp:2657 push_backs into the shared override vector). Do not
        join with ";" -- that is llama-bench's separator; llama-server splits an
        ``-ot`` value on ",".

        ``--fit off`` is REQUIRED, not tidiness. The fitter aborts only on an
        n_gpu_layers the USER set (common/fit.cpp:377) and -1 is llama.cpp's own
        default (llama-model.cpp:2453), so ``-ngl -1`` does not hold it off: left
        on, it would re-place layers over this plan and put the cache back on the
        host. ``-ngl -1`` keeps every layer assigned to a GPU so the cache stays
        there (llama-kv-cache.cpp:215); the ``-ot`` patterns move only the named
        weights.
        """
        if plan is None or not plan.spills_anything:
            return []
        tokens = [tok for pat in plan.ot_patterns for tok in ("-ot", f"{pat}=CPU")]
        if not tokens:
            return []
        return ["-ngl", "-1", "--fit", "off", *tokens]

    def _drop_tensor_spill(self, run_cmd: "list[str]", why: str) -> "list[str]":
        """Take the spill plan back out of an argv and restore ``--fit on``.

        Every retry that re-places the model or resizes the cache invalidates the
        arithmetic the plan was built on: a no-flash retry changes the KV dtype, an
        arch fallback changes the device set, a projector move changes the resident
        floor. Rather than re-plan on a path that is already recovering from a
        crash, hand the placement back to llama.cpp, which is what this arm did
        before the planner existed.

        The record is deliberately NOT cleared: the CPU fallback respawns later
        from an argv that may still carry these tokens.
        """
        if not self._spill_plan_flags:
            return run_cmd
        stripped = _without_subsequence(run_cmd, self._spill_plan_flags)
        if stripped == run_cmd:
            return run_cmd
        logger.info("Tensor spill: dropping the plan for the %s retry; %s", why, "using --fit on")
        return [*stripped, "--fit", "on"]

    def _tensor_spill_layout(
        self,
        model_path: "Optional[str]",
        *,
        all_shards: bool = False,
    ) -> "Optional[ModelLayout]":
        """The GGUF's placement buckets, read once per path and cached.

        Per BLOCK, not per bucket total, so the planner can spill the minimum set
        that closes the deficit rather than the whole FFN. That matters: the spill
        penalty is linear in bytes (measured 5.544 ms/GiB across 16/32/48/65
        blocks, intercept -0.10 ms), so every byte spilled beyond the deficit is
        paid on every token for nothing.

        Any read failure returns None and the whole feature abstains, leaving the
        argv byte-identical to today. A guess here would be priced as a fit.
        """
        from core.inference.offload_layout import layout_from_gguf

        if not model_path:
            return None
        # Keyed on file IDENTITY, not name: the backend outlives a load, so
        # re-downloading or re-quantising to the same path would otherwise plan
        # against the OLD tensor table and understate the deficit. Same
        # (size, mtime_ns) stat identity _slot_launch_fingerprint uses.
        try:
            st = os.stat(model_path)
            key = (model_path, st.st_size, st.st_mtime_ns, all_shards)
        except OSError:
            key = (model_path, None, None, all_shards)
        cached = getattr(self, "_spill_layout_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            layout = (
                layout_from_gguf(model_path, all_shards = True)
                if all_shards
                else layout_from_gguf(model_path)
            )
        except Exception as e:  # unreadable, truncated, or an arch we cannot bucket
            logger.debug("Tensor spill: cannot read layout from %s (%s)", model_path, e)
            layout = None
        self._spill_layout_cache = (key, layout)
        return layout

    @staticmethod
    def _fit_off_retry_eligible(cmd: "list[str]", use_fit: bool) -> bool:
        """Whether a llama-server startup crash may be retried with --fit off.

        Only when Unsloth's own VRAM math placed the model (use_fit=False)
        and nothing on the command line set the fit mode explicitly
        (-fit / --fit, space- or equals-form). --fit-ctx / --fit-target /
        -fitc / -fitt tune the fit step but do not select the mode, so
        they do not block the retry.
        """
        if use_fit:
            return False
        for a in cmd:
            if a in ("-fit", "--fit") or a.startswith(("-fit=", "--fit=")):
                return False
        return True
