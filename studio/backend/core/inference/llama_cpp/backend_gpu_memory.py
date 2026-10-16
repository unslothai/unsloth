# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: GPU allocation, memory probes and host budgets."""

import json
import math
import os
import re
import subprocess
import sys

from pathlib import Path
from typing import (
    Callable,
    Collection,
    Iterable,
    Mapping,
    Optional,
)
from core.inference.llama_server_args import (
    _GPU_LAYER_FLAGS,
    apply_load_mode_policy,
    apply_model_memory_policy,
    fit_is_effectively_on,
    fit_is_enabled_in,
    parse_gpu_layers_override,
    resolve_effective_load_state,
    scrub_memory_env,
    strip_shadowing_flags,
)
from utils import host_memory as _host_memory
from utils.child_stdio import utf8_child_env
from utils.native_path_leases import child_env_without_native_path_secret
from utils.paths.path_utils import _is_wsl
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)

from .binary_runtime import (
    _CARVEOUT_ADVICE_HOST_FRACTION,
    _CARVEOUT_ADVICE_MIN_HOST_GB,
    _CARVEOUT_NOMINAL_SLACK_GB,
    _CGROUP_ROOT,
    _GGML_GPU_BACKEND_RE,
    _HOST_RAM_HEADROOM_MIB,
    _IGPU_HOST_RESERVE_MIB,
    _PROC_SELF_CGROUP,
    _apply_igpu_host_reserve_mib,
    _lib_dir_has_ggml_backend,
    _llama_lib_dir,
    _load_probe_memo_get,
    _load_probe_memo_put,
    _loader_path_var,
)
from .draft_args import (
    _device_selection_is_cpu,
    _extra_args_device,
    _extra_args_main_device,
    _parse_metal_memory_breakdown,
    _sidecar_adapter_paths,
    _split_mode_confines_to_one_device,
)
from .launch_flags import (
    _args_place_tensors_on_cpu,
    _env_fixes_gpu_layers,
    _env_places_tensors_on_cpu,
)
from .launch_support import _metal_capable_host
from .log import logger
from .memory_policy import (
    _AMD_SMI_TOTAL_TOLERANCE,
    _APPLE_UNIFIED_MEMORY_FRACTION,
    _APPLE_WIRED_CEILING_FRACTION,
    _FIT_CONTROL_ENV_VARS,
    _FIT_CONTROL_FLAGS,
    _FIT_MIN_CTX,
    _LLAMA_FIT_MIN_CTX,
    _TENSOR_SPLIT_FLAGS,
    _extra_args_set_any_flag,
)

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _GpuMemoryMixin:
    @staticmethod
    def _backend_lacks_gpu_lib(binary: Optional[str] = None) -> bool:
        """Detect a split-library build proven to be CPU-only.

        Unknown and static layouts return False to avoid rejecting custom GPU builds.
        """
        binary = binary or LlamaCppBackend._find_llama_server_binary()
        if not binary:
            return False
        backends = LlamaCppBackend._installed_ggml_backends(binary)
        return bool(backends.intersection({"cpu", "base"})) and not backends.intersection(
            {"cuda", "hip", "vulkan"}
        )

    def is_vulkan_build(self) -> bool:
        """Whether the resolved llama-server uses Vulkan ordinals."""
        try:
            return self._is_vulkan_backend(self._find_llama_server_binary())
        except Exception:
            return False

    def _gpu_ids_own_placement(self, gpu_ids) -> bool:
        """Whether a ``gpu_ids`` pick overrules a user ``--device`` / ``--main-gpu``.

        One predicate, not eleven copies: launch, fit classifier, cache tuning and
        reload comparator disagreeing is how the argv and the already-loaded check
        drift apart. Letting a full-set pick relinquish to a pass-through was tried
        and withdrawn; the inherited-mask reorder already supplies that ordering.
        """
        return gpu_ids is not None

    @staticmethod
    def _strip_device_extra_args(extra_args):
        """Drop device and main-GPU flags so gpu_ids remains authoritative."""
        return strip_shadowing_flags(
            extra_args,
            strip_context = False,
            strip_cache = False,
            strip_spec = False,
            strip_template = False,
            strip_split_mode = False,
            strip_device = True,
        )

    # A concrete per-device gfx arch: "gfx" plus digits, with the trailing hex letter
    # real parts carry (gfx90a, gfx90c), which a digits-only pattern would reject.
    # Deliberately does NOT match generic code objects (gfx11-generic) or family
    # labels (gfx110X); no device reports either, and the gate fails open on them.
    _CONCRETE_GFX_ARCH = re.compile(r"^gfx[0-9][0-9a-f]{2,4}$")

    @staticmethod
    def _installed_llama_gfx_archs(binary: Optional[str] = None) -> Optional[frozenset]:
        """Concrete gfx archs the installed ROCm prebuilt was built for
        (``mapped_targets`` in UNSLOTH_PREBUILT_INFO.json). None when unknown (no
        marker, non-ROCm bundle), so callers fail open and keep every device
        (#7624). Never raises."""
        try:
            from utils.llama_cpp_freshness import read_install_marker

            marker = read_install_marker(binary or LlamaCppBackend._find_llama_server_binary())
            if not marker:
                return None
            targets = marker.get("mapped_targets")
            if not isinstance(targets, list):
                return None
            archs = frozenset(
                str(t).split(":")[0].strip().lower() for t in targets if str(t).strip()
            )
            if not archs:
                return None
            # Exact-set membership against what the device reports, so it only means
            # anything while every token is concrete. One token no device can report
            # (generic code object, family label, malformed marker) matches nothing,
            # dropping EVERY GPU and silently forcing CPU, worse than not gating.
            # mapped_targets is remote data versioned independently of this code, so
            # an unrecognised list is "unknown" and fails open, all or nothing:
            # dropping just the odd token would under-report the build's coverage.
            unknown = sorted(a for a in archs if not LlamaCppBackend._CONCRETE_GFX_ARCH.match(a))
            if unknown:
                logger.warning(
                    f"install marker mapped_targets has non-concrete arch tokens "
                    f"{unknown}; skipping the GPU arch gate"
                )
                return None
            return archs
        except Exception as e:
            logger.debug(f"install marker arch read failed: {e}")
            return None

    @staticmethod
    def _installed_llama_cuda_sms(binary: Optional[str] = None) -> Optional[frozenset]:
        """SM targets the installed CUDA prebuilt was built for (``supported_sms``
        in UNSLOTH_PREBUILT_INFO.json). None when unknown (no marker, older
        install, non-CUDA bundle), so callers fail open. Never raises."""
        try:
            from utils.llama_cpp_freshness import read_install_marker

            marker = read_install_marker(binary or LlamaCppBackend._find_llama_server_binary())
            if not marker:
                return None
            sms = marker.get("supported_sms")
            if not isinstance(sms, list) or not sms:
                return None
            if not all(str(s).strip().isdigit() for s in sms):
                return None
            return frozenset(int(s) for s in sms)
        except Exception as e:
            logger.debug(f"install marker supported_sms read failed: {e}")
            return None

    @staticmethod
    def _cuda_compute_caps() -> dict:
        """Physical id -> SM (90 for sm_90) via nvidia-smi, honoring
        CUDA_VISIBLE_DEVICES; {} when unknown. Never raises."""
        try:
            from utils.hardware import gpu_query

            result = gpu_query.run_nvidia_smi(
                ["nvidia-smi", "--query-gpu=index,compute_cap", "--format=csv,noheader"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                env = child_env_without_native_path_secret(),
                **_windows_hidden_subprocess_kwargs(),
            )
            if result.returncode != 0:
                return {}
            allowed = LlamaCppBackend._visible_devices_mask("CUDA_VISIBLE_DEVICES")
            if allowed is not None and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
                # Numeric mask entries are CUDA ordinals; nvidia-smi reports PCI
                # physical indices. Without a shared order, fail open.
                return {}
            caps = {}
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    idx = int(parts[0])
                    major, minor = parts[1].split(".")
                    sm = int(major) * 10 + int(minor)
                except ValueError:
                    continue
                if allowed is not None and idx not in allowed:
                    continue
                caps[idx] = sm
            return caps
        except Exception as e:
            logger.debug(f"compute_cap probe failed: {e}")
            return {}

    @classmethod
    def _cuda_sm_gate_error(
        cls,
        binary: Optional[str] = None,
        gpu_indices: Optional[Iterable[int]] = None,
    ) -> Optional[str]:
        """Error message when every visible GPU is OLDER than the oldest arch the
        installed CUDA bundle was built for, or None to proceed. Fails open on
        unknown coverage or caps.

        ``gpu_indices`` (physical ids the child is pinned to) tightens "every
        visible" to "any pinned": the child places work on each pinned card, so one
        too-old card aborts the load even beside a supported one.

        Only the too-old direction is broken. The bundles compile plain arch
        numbers, so each fatbin carries PTX beside its cubins, and the driver JITs
        that PTX forward onto a newer card; ggml_cuda_highest_compiled_arch picks
        the highest compiled arch <= the device and only aborts ("not compiled with
        any CUDA arch <= N") when the device is below every one of them. So an
        exact-SM test would refuse working installs: the legacy PTX-only bundle
        (sm_50-61) drives an sm_86/sm_89 host at full speed. Missing native cubins
        cost a one-time JIT, not a launch. The installer's exact-membership check
        is a different question -- which bundle is BEST to install -- not whether
        the installed one can run."""
        # Resolved once so the marker read and the remedy name the same binary.
        binary = binary or cls._find_llama_server_binary()
        supported = cls._installed_llama_cuda_sms(binary)
        if supported is None:
            return None
        caps = cls._cuda_compute_caps()
        oldest = min(supported)
        if gpu_indices is not None:
            pinned = {int(i) for i in gpu_indices}
            caps = {idx: sm for idx, sm in caps.items() if idx in pinned and sm < oldest}
        elif any(sm >= oldest for sm in caps.values()):
            return None
        if not caps:
            return None
        present = ", ".join(f"GPU {idx} is sm_{sm}" for idx, sm in sorted(caps.items()))
        # The updater cannot replace a pinned LLAMA_SERVER_PATH or a llama-server
        # found on PATH, so the remedy follows the binary's provenance.
        return (
            f"The installed llama.cpp build only has GPU code for sm_{oldest} and "
            f"newer, but {present} -- it was likely installed on a machine with a "
            f"newer GPU, so {cls._runtime_remedy(binary)}."
        )

    @classmethod
    def _cuda_sm_gate_keep(
        cls, binary: Optional[str], gpus: list[tuple[int, int, int]]
    ) -> list[tuple[int, int, int]]:
        """``gpus`` minus the CUDA cards older than the installed bundle's oldest
        arch, so auto placement cannot pin one beside a supported card (RTX 4090 +
        GTX 1060 under an sm_75+ build). Unchanged on unknown coverage or caps, and
        when no card survives: the gate then refuses with its own message."""
        supported = cls._installed_llama_cuda_sms(binary)
        if supported is None or not gpus:
            return gpus
        caps = cls._cuda_compute_caps()
        oldest = min(supported)
        kept = [g for g in gpus if caps.get(g[0], oldest) >= oldest]
        if not kept or len(kept) == len(gpus):
            return gpus
        for idx, _free, _total in gpus:
            if caps.get(idx, oldest) < oldest:
                logger.warning(
                    f"Skipping GPU {idx} (sm_{caps[idx]}): installed llama.cpp "
                    f"build only has GPU code for sm_{oldest} and newer"
                )
        return kept

    @classmethod
    def _arch_gate_survivors(cls, binary: Optional[str] = None) -> list[int]:
        """Physical ids the ROCm arch gate leaves, or [] when there is nothing to
        mask: unknown coverage, a non-ROCm host, a build covering every card, or an
        inherited mask whose ids we cannot name back to the child.

        For callers that must PIN the answer, not just rank by it: the child
        enumerates every ROCm agent, and that HSA enumeration is what dies on an
        uncovered arch, so a mixed host (unsupported iGPU + supported dGPU) passes
        the gate and still crashes unless the survivors are masked in. Lazy, and
        never raises."""
        try:
            # Coverage first: a cached marker read, where _host_torch_is_rocm imports
            # torch (~1.3s cold). Non-ROCm bundles record no mapped_targets, so they
            # answer [] without that import.
            if cls._installed_llama_gfx_archs(binary) is None:
                return []  # unknown coverage: fail open, same as the probe
            if not cls._host_torch_is_rocm():
                return []
            if cls._visibility_mask_is_unmappable():
                # A UUID/MIG mask leaves ordinals unmapped, so pinning would swap it
                # for numbers the runtime resolves against the whole host, possibly
                # re-exposing the dropped card. Fail open (pre-#7624).
                logger.info(
                    "Not pinning the arch gate's survivors: the inherited GPU "
                    "visibility mask is not index based, so they cannot be named "
                    "to the child."
                )
                return []
            gated = [
                idx for idx, _free, _total in cls._get_gpu_memory(binary, for_llama_server = True)
            ]
            everything = [idx for idx, _free, _total in cls._get_gpu_memory(binary)]
            return gated if gated and len(gated) < len(everything) else []
        except Exception as e:
            logger.debug(f"arch gate survivor probe failed: {e}")
            return []

    @staticmethod
    def _get_gpu_free_memory(
        binary: Optional[str] = None, *, for_llama_server: bool = False
    ) -> list[tuple[int, int]]:
        """Free memory per GPU as ``(gpu_index, free_mib)`` sorted by index; empty
        if no supported GPU is reachable. Thin wrapper over ``_get_gpu_memory``,
        which also documents ``for_llama_server`` (forwarded verbatim)."""
        return [
            (idx, free)
            for idx, free, _total in LlamaCppBackend._get_gpu_memory(
                binary, for_llama_server = for_llama_server
            )
        ]

    @staticmethod
    def _metal_zero_ctx_floor(
        effective_ctx: int,
        auto_fit: bool,
        caller_owns_budget: bool,
        native_ctx: Optional[int],
        max_available_ctx: Optional[int] = None,
        *,
        fitter_runs: bool = True,
    ) -> int:
        """Context to start at when Metal would otherwise be sent "-c 0", else 0.

        llama.cpp reads "-c 0" as fit_params_min_ctx = UINT32_MAX, pinning the
        model's full native length and disabling --fit's reduction. Two paths
        reach the command builder with a zero context after the Apple cap has
        been skipped or thrown away:

          * the cap is guarded on ``effective_ctx > 0``, so a GGUF whose
            metadata carries no context length is never capped;
          * the ``except Exception`` around GPU selection restores the original
            request, 0 when context is on Auto, discarding a context the cap had
            already computed.

        Either way the child over-commits unified memory, surfacing as
        llama-server's "Compute error." at decode (#5118, #6529). Floor to the
        same ``_FIT_MIN_CTX`` the cap falls back to without a KV estimate.

        ``fitter_runs`` is whether the child's own --fit step will actually run, and
        both the exemption and the floor VALUE depend on it. _FIT_MIN_CTX sits above
        llama.cpp's own ``fit_params_min_ctx`` on purpose: 8192 is defensible on a Mac
        only because it is a request the fitter can still reduce, down to 4096. Turn
        the fitter off and that backstop is gone, and this arm would hand a machine
        with room for 4096 but not 8192 a context nothing can bring down -- a startup
        failure or a "Compute error." at decode, the very thing the floor exists to
        prevent. So a launch with no fitter is floored at ``_LLAMA_FIT_MIN_CTX``, the
        number the fitter would have reduced to, rather than at the ceiling that
        assumed one. Conservative on purpose: --fit off is a flag the user typed, and
        overriding it to keep our own invented floor would be Unsloth taking back a
        setting the user asked for, on the path where they most likely asked for it
        because the fitter misbehaves on their host.

        Exemptions: no Metal budget (0 off Apple Silicon, so this is inert
        there); ``auto_fit``, which omits -c so --fit sizes it, and which is
        therefore honoured only while ``fitter_runs`` -- the conjunction is taken
        here rather than trusted to the caller, since without it the exemption hands
        the child no -c and no fitter, which is llama.cpp's native length; and
        ``caller_owns_budget``, a manual mode with a fixed layer count where the
        user owns memory management. That last is read off the REQUEST, because
        the paravirtual CPU pin rewrites every placement to manual/0 first and
        would make a plain Auto request look caller-owned.
        """
        if effective_ctx > 0 or (auto_fit and fitter_runs) or caller_owns_budget:
            return 0
        if not LlamaCppBackend._apple_metal_memory_budget_bytes():
            return 0
        floor = _FIT_MIN_CTX if fitter_runs else _LLAMA_FIT_MIN_CTX
        # Never above a ceiling the cap already worked out: on the exception path
        # max_available_ctx survives the reset, and its KV-based answer can sit
        # below the floor. Floating back up would re-create the over-commit.
        return min(floor, native_ctx or floor, max_available_ctx or floor)

    @staticmethod
    def _metal_drops_zero_ctx_override(
        ctx_override: Optional[int], caller_owns_budget: bool
    ) -> bool:
        """Whether a pass-through "-c 0" must be dropped before it reaches Metal.

        User extras are appended after Unsloth's own -c and llama.cpp is
        last-wins, so a zero override outlives both the Apple cap and the floor
        above and re-pins the native length. Only a zero; a positive -c stays
        honored. Inert off Apple Silicon like the floor, but unlike the floor it
        also fires in Auto-layers, where the command omits -c precisely so --fit
        sizes the context. ``caller_owns_budget`` is exempt, read off the request
        for the reason the floor documents.
        """
        if ctx_override != 0 or caller_owns_budget:
            return False
        return bool(LlamaCppBackend._apple_metal_memory_budget_bytes())

    @staticmethod
    def _apple_metal_memory_budget_bytes() -> int:
        """Unified-memory budget for GGUF context fitting on Apple Silicon.

        No GPU is enumerated on Metal, so the context would default to native and
        over-commit unified memory ("Compute error." at decode, #5118/#6529). Use a
        fraction of the Metal working set, else total RAM; 0 off Apple Silicon or
        when unresolvable, so callers skip the cap.

        Both of those inputs describe the MACHINE, not the moment. MLX's
        max_recommended_working_set_size is a static device property, and
        virtual_memory().total obviously is, so on a 16 GB Mac the budget came
        out around 9 GB whether the machine was idle or already holding several
        gigabytes. Unsloth's own idle footprint alone is over a gigabyte once the
        warm thread has imported torch and, on Apple Silicon, MLX. The fit then
        sized a context against headroom that was not there, and llama-server
        died in KV or compute allocation.

        So take whichever is smaller: the device ceiling, or what is actually
        available right now. The ceiling still applies on a machine with plenty
        free, which is the case the working-set number was chosen for.

        _APPLE_UNIFIED_MEMORY_FRACTION then earns a second job it did not have
        over a static ceiling: available is a snapshot, and llama-server spends
        seconds loading weights, so another app can take memory inside that
        window. Budgeting all of it would re-create the failure this avoids.
        """
        from utils.hardware import is_apple_silicon

        if not is_apple_silicon():
            return 0
        from utils.hardware.hardware import _metal_recommended_working_set_bytes

        rec_bytes = _metal_recommended_working_set_bytes()
        vm = None
        try:
            import psutil
            vm = psutil.virtual_memory()
        except Exception:
            vm = None
        if rec_bytes <= 0:
            if vm is None:
                return 0
            try:
                rec_bytes = int(vm.total)
            except Exception:
                return 0
        # available, not free: psutil's macOS free subtracts speculative pages and
        # leaves out the inactive queue, so it reads far below what a new allocation
        # can get. available is inactive + free (psutil arch/osx/mem.c), which adds
        # that reclaimable cache back. An estimate either way, since it counts dirty
        # inactive pages that cost a compression to reclaim and ignores the
        # compressor entirely. Only ever lowers the budget.
        try:
            available = int(getattr(vm, "available", 0) or 0) if vm is not None else 0
        except Exception:
            available = 0
        if available > 0:
            rec_bytes = min(rec_bytes, available)
        return int(rec_bytes * _APPLE_UNIFIED_MEMORY_FRACTION)

    def _metal_measured_model_mib(
        self, binary: Optional[str], model_path: Optional[str]
    ) -> Optional[tuple[int, int, int]]:
        """Measure full-offload model buffers with the sibling ``llama-fit-params``.

        Discounted launches pin the same placement. Results are cached by probe and model
        identity; a missing, failed, or unreadable probe returns None.
        """
        if not binary or not model_path:
            return None
        try:
            probe = _llama_lib_dir(binary) / "llama-fit-params"
            probe_stat = probe.stat()
            model_stat = os.stat(model_path)
        except OSError:
            return None
        key = (
            str(probe),
            probe_stat.st_size,
            probe_stat.st_mtime_ns,
            model_path,
            model_stat.st_size,
            model_stat.st_mtime_ns,
        )
        cached = getattr(self, "_metal_model_mib_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        measured: Optional[tuple[int, int, int]] = None
        try:
            env = self._llama_server_env_for_binary(str(binary))
            for name in tuple(env):
                if name.startswith("LLAMA_ARG_"):
                    env.pop(name, None)
            result = subprocess.run(
                [str(probe), "-m", model_path, "-ngl", "999", "-c", "512", "-lv", "4"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                # A cold Metal shader compile lands here instead of at the launch.
                timeout = 120,
                check = False,
                env = env,
                **_windows_hidden_subprocess_kwargs(),
            )
            if result.returncode == 0:
                measured = _parse_metal_memory_breakdown(
                    (result.stdout or "") + "\n" + (result.stderr or "")
                )
        except Exception as e:
            logger.debug(f"llama-fit-params memory probe failed: {e}")
        self._metal_model_mib_cache = (key, measured)
        return measured

    def _metal_demand_paged_embedding_bytes(
        self,
        model_path: Optional[str],
        extra_args: Optional[Iterable[str]],
        *,
        binary: Optional[str],
        requested_load_mode: Optional[str],
        supports_load_mode: bool,
        settings: tuple[bool, bool],
        layers_fixed: bool,
        mtp_may_engage: bool,
        env: Optional[Mapping[str, str]] = None,
    ) -> int:
        """Return mmap embedding bytes proven absent from the Metal working set.

        Input embeddings live on the CPU but can still fall inside a Metal file mapping.
        Fail closed unless the probe matches the launch placement and the loader leaves the
        embeddings pageable. CPU fallback weights remain charged.
        """
        source_env = os.environ if env is None else env
        if (
            layers_fixed
            or _extra_args_set_any_flag(extra_args, _GPU_LAYER_FLAGS)
            or _env_fixes_gpu_layers(source_env)
            or _device_selection_is_cpu(extra_args, source_env)
            or _args_place_tensors_on_cpu(extra_args)
            or _env_places_tensors_on_cpu(source_env)
            or _extra_args_set_any_flag(extra_args, _FIT_CONTROL_FLAGS)
            or any(source_env.get(name) for name in _FIT_CONTROL_ENV_VARS)
            # Extras are appended after the pin, while inherited values are overridden by it.
            or fit_is_enabled_in(extra_args)
            # A pass-through adapter is resident on the base tensor's buffer
            # (llama-adapter.cpp:335, :67) and neither the probe nor model_size carries it,
            # and nothing on the Apple path charges it the way _fits_without_paging does
            # off Metal. Spending the discount would hand that budget to the KV cache.
            or _sidecar_adapter_paths(extra_args)
        ):
            return 0
        env_view = dict(source_env)
        scrub_memory_env(env_view, settings)
        managed, extras = apply_model_memory_policy(
            extra_args,
            supports_load_mode = supports_load_mode,
            weights_in_host_memory = True,
            gpu_offload_confirmed = False,
            env = env_view,
            settings = settings,
        )
        selected, extras = apply_load_mode_policy(
            extras,
            supports_load_mode = supports_load_mode,
            weights_in_host_memory = True,
            requested_load_mode = requested_load_mode,
            settings = settings,
        )
        mlock, reserves_ram, direct_io = resolve_effective_load_state(
            [*managed, *selected, *extras], env_view
        )
        if mlock or reserves_ram or direct_io:
            return 0
        layout = self._tensor_spill_layout(model_path, all_shards = True)
        if layout is None or not layout.complete:
            return 0
        embeddings = int(layout.token_embd_bytes)
        if embeddings <= 0 or (mtp_may_engage and layout.has_excluded_blocks):
            return 0
        measured = self._metal_measured_model_mib(binary, model_path)
        if measured is None:
            logger.info(
                "Metal fit: could not measure how llama.cpp maps this model (llama-fit-params "
                "missing or unreadable), so its %.2f GB of input embeddings stay charged.",
                embeddings / (1024**3),
            )
            return 0
        metal_mib, host_mib, rows = measured
        mib = 1024 * 1024
        charged = (metal_mib + rows) * mib + max(0, host_mib * mib - embeddings)
        # TENSOR_SKIP returns before the tensor exists (llama-model-loader.cpp:1125-1131), so a
        # skipped block reaches no buffer and no table row. Those bytes are in the file and not
        # in the probe, which is the signature the embeddings are being recognised by.
        mapped = int(layout.tensor_bytes) - int(layout.excluded_block_bytes)
        unmapped = min(embeddings, max(0, mapped - charged))
        if not unmapped:
            logger.info(
                "Metal fit: llama.cpp maps this model's %.2f GB of input embeddings into Metal, "
                "so they stay charged. A llama.cpp update may remove that.",
                embeddings / (1024**3),
            )
        return unmapped

    @staticmethod
    def _apple_metal_wired_ceiling_bytes() -> int:
        """Wired GPU memory a Metal load may still take, system-wide; 0 if unknown or used up."""
        from utils.hardware import is_apple_silicon

        if not is_apple_silicon():
            return 0
        try:
            probe = subprocess.run(
                ["sysctl", "-n", "iogpu.wired_limit_mb"],
                capture_output = True,
                text = True,
                timeout = 5,
                encoding = "utf-8",
                errors = "replace",
            )
            cap_bytes = int(probe.stdout.strip()) * 1024 * 1024
        except Exception:
            return 0
        # 0 means the kernel default, Metal's recommended working set.
        if cap_bytes <= 0:
            from utils.hardware.hardware import _metal_recommended_working_set_bytes
            cap_bytes = _metal_recommended_working_set_bytes()
        if cap_bytes <= 0:
            return 0
        from utils.hardware.hardware import _read_apple_gpu_stats

        in_use = _read_apple_gpu_stats().get("vram_used_bytes")
        if in_use is None:
            return 0
        return max(0, int((cap_bytes - in_use) * _APPLE_WIRED_CEILING_FRACTION))

    @staticmethod
    def _metal_fit_target_floor_mib() -> float:
        """Smallest ``--fit-target`` that keeps llama.cpp's Metal fit inside the wired headroom.

        ggml-metal reports free as recommendedMaxWorkingSetSize minus this process's own
        allocations (ggml-metal-device.m, ggml_metal_device_get_memory), blind to the GPU
        memory WindowServer and other apps already hold, so --fit packs to within 1 GiB of
        the static ceiling and over-commits the wired limit (#9279). 0 when either side is
        unreadable, which leaves llama.cpp's own margin.
        """
        headroom = LlamaCppBackend._apple_metal_wired_ceiling_bytes()
        if headroom <= 0:
            return 0.0
        from utils.hardware.hardware import _metal_recommended_working_set_bytes

        seen = _metal_recommended_working_set_bytes()
        if seen <= 0:
            return 0.0
        return max(0.0, (seen - headroom) / (1024 * 1024))

    @staticmethod
    def _rocm_arch_gate_keep(
        binary: Optional[str], torch_mod, for_llama_server: bool
    ) -> Callable[[int], bool]:
        """Predicate over PHYSICAL ids: False for a device the installed ROCm
        prebuilt has no kernels for.

        llama-server placement only, so the free-memory rank cannot pick a GPU that
        binary cannot run (#7624: an iGPU reporting shared RAM outranks the dGPU and
        the child dies with "device kernel image is invalid"). Keeps every device off
        that path, off ROCm, on unknown coverage, and for a device with no reported
        arch. Shared by both branches of ``_get_gpu_memory`` so the amd-smi and torch
        paths cannot gate differently.
        """
        if not for_llama_server or not LlamaCppBackend._torch_is_rocm(torch_mod):
            return lambda _idx: True
        supported_archs = LlamaCppBackend._installed_llama_gfx_archs(binary)
        if supported_archs is None:
            return lambda _idx: True
        arch_by_id = LlamaCppBackend._rocm_arch_by_physical_id()

        def _keep(idx: int) -> bool:
            _base = arch_by_id.get(idx, "")
            if _base and _base not in supported_archs:
                logger.warning(
                    f"Skipping GPU {idx} ({_base}): installed llama.cpp "
                    f"prebuilt only covers {sorted(supported_archs)}"
                )
                return False
            return True

        return _keep

    @staticmethod
    def _rocm_hip_is_reachable() -> bool:
        """Whether HIP can actually open a device on this ROCm host.

        amd-smi reads the driver over sysfs and libdrm, HIP needs ``/dev/kfd`` and a
        matching HSA runtime, so the two disagree on a container given only
        ``/dev/dri`` and on a torch built against another ROCm: amd-smi lists the
        card and ``hipGetDeviceCount`` returns 0. The llama-server child is a HIP
        process too, so answering there places a model on a device nothing can open.

        ``is_available()`` is that ``hipGetDeviceCount``, not ``_lazy_init``, so it
        creates no primary context; ``_rocm_unified_memory_gpu_ids`` already calls it
        further down the same probe. False when torch is missing or unreadable.
        """
        try:
            import torch
            return bool(hasattr(torch, "cuda") and torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def _rocm_hip_device_count() -> int:
        """How many devices HIP can open here, or 0 when torch cannot say.

        ``hipGetDeviceCount``, the same call ``_rocm_hip_is_reachable`` asks for a
        yes/no, so it costs no primary context either. This is the size of the
        inventory the llama-server child will see; amd-smi reads sysfs and answers
        for the whole host regardless, so the two disagree wherever something
        outside the visibility variables filters the device set (a device-cgroup
        container given one ``/dev/dri/renderD*`` node, ``GPU_DEVICE_ORDINAL``, a
        card amd-smi could not size).
        """
        try:
            import torch
            if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                return 0
            return int(torch.cuda.device_count())
        except Exception:
            return 0

    @staticmethod
    def _gpu_device_ordinal_active() -> bool:
        """Whether ``GPU_DEVICE_ORDINAL`` is filtering this process's devices.

        ROCm's fourth visibility variable, and the one nothing here reads:
        ``_active_gpu_visibility_mask`` knows the HIP, ROCR and CUDA names only, so
        ``_resolve_visible_physical_ids`` answers None ("no mask") under it. AMD
        documents it at the ROCclr layer as "devices indices exposed to OpenCL and
        HIP applications", while clr itself reads it only on the non-HIP branch
        (``rocclr/device/rocm/rocdevice.cpp`` picks HIP_VISIBLE_DEVICES /
        CUDA_VISIBLE_DEVICES when ``amd::IS_HIP``), so the two disagree on whether
        HIP honours it. Treat it as a mask either way: being wrong here costs the
        host nothing but this branch's context saving, and being wrong the other way
        offers a hidden card for placement. ``utils/hardware/hardware.py::
        _rocm_visibility_mask_active`` gates the system-wide VRAM overlay on the same
        four names. Whitespace is not a filter, matching that helper."""
        return bool(os.environ.get("GPU_DEVICE_ORDINAL", "").strip())

    @staticmethod
    def _rocm_visibility_masks_are_stacked() -> bool:
        """Whether a ROCr mask AND a HIP-layer mask are both filtering this process.

        The two sit at different layers and therefore COMPOSE: ROCR_VISIBLE_DEVICES
        filters and renumbers the agent list inside the ROCr runtime
        (``ROCR-Runtime/core/inc/amd_filter_device.h::RvdFilter``), and clr's
        ``Device::init`` then indexes ``HIP_VISIBLE_DEVICES`` (or its
        ``CUDA_VISIBLE_DEVICES`` twin) into ``gpu_agents_``, the vector
        ``hsa_iterate_agents`` just returned -- so the HIP ordinals count positions in
        what ROCr left, not physical devices. ``ROCR_VISIBLE_DEVICES=1,2`` with
        ``HIP_VISIBLE_DEVICES=0`` opens physical GPU 1.

        ``_active_gpu_visibility_mask`` returns the single highest-precedence value,
        which is the right answer for one layer and cannot express two: it reads
        ``[0]`` for that example, and on a host whose cards have the same total the
        inventory and total-memory cross-checks agree with it. Callers that would map
        those ids onto amd-smi's host-wide device list must decline instead. Composing
        the masks here is not an option: the physical->ROCr mapping is exactly what
        ROCr renumbered away, and guessing it wrong offers a hidden card.

        HIP and CUDA together are ONE layer, not two -- clr reads whichever is set
        first and never both -- so they do not stack. Windows has no ROCr layer at
        all, so a stray ROCR variable filters nothing there, matching
        ``_active_gpu_visibility_mask`` and ``_emit_child_gpu_visibility``. Set is
        set: an empty ROCr mask hides every agent rather than nothing."""
        if sys.platform == "win32":
            return False
        if os.environ.get("ROCR_VISIBLE_DEVICES") is None:
            return False
        return (
            os.environ.get("HIP_VISIBLE_DEVICES") is not None
            or os.environ.get("CUDA_VISIBLE_DEVICES") is not None
        )

    @staticmethod
    def _rocm_total_memory_mib_by_physical_id() -> dict[int, int]:
        """Total memory HIP reports per PHYSICAL device id, honoring the active ROCm
        mask, for cross-checking amd-smi's totals.

        Read through ``get_device_properties`` like ``_rocm_arch_by_physical_id``, so
        it costs no primary context (``mem_get_info`` is the call that does).
        Devices torch cannot describe are omitted and callers fail open on them;
        empty off ROCm and on any error."""
        totals: dict[int, int] = {}
        try:
            import torch

            if not LlamaCppBackend._torch_is_rocm(torch):
                return totals
            if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                return totals
            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()
            for ordinal in range(torch.cuda.device_count()):
                try:
                    total_bytes = int(
                        getattr(torch.cuda.get_device_properties(ordinal), "total_memory", 0) or 0
                    )
                except Exception:
                    continue
                if total_bytes <= 0:
                    continue
                pid = (
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
                totals[pid] = total_bytes // (1024 * 1024)
        except Exception:
            return totals
        return totals

    @staticmethod
    def _amd_smi_hip_id_map(
        enumerated: list[int], mask: Optional[list[int]]
    ) -> Optional[dict[int, int]]:
        """{amd-smi gpu id: HIP id} for every enumerated device, else None.

        amd-smi numbers its devices in discovery order and HIP numbers them by KFD
        node id, so the two agree on most hosts and not on all of them (MI350X in
        SPX/NPS1 among others). Every id this probe returns is fed back as a HIP
        ordinal -- matched against the visibility mask, then written into the child's
        HIP_VISIBLE_DEVICES -- so an untranslated amd-smi id pins the wrong card.

        None means "not provable", and the branch then defers to torch, which
        enumerates in HIP's space by construction.
        """
        if len(enumerated) == 1 and mask in (None, [0]):
            # One card on the host, so HIP can only be looking at that card and its
            # id can only be 0. Keeps the single-GPU ROCm desktop, which is most of
            # them, on amd-smi even where the CLI predates the mapping below.
            return {enumerated[0]: 0}
        from utils.hardware.amd import get_hip_id_by_gpu_index

        mapping = get_hip_id_by_gpu_index()
        if mapping is None or not all(_g in mapping for _g in enumerated):
            logger.debug(
                "amd-smi VRAM probe deferring to torch: no amd-smi to HIP id mapping "
                "for this host (amd-smi list -e needs ROCm 6.4.0+)"
            )
            return None
        return mapping

    @staticmethod
    def _get_gpu_memory_amd_smi(
        binary: Optional[str] = None, *, for_llama_server: bool = False
    ) -> list[tuple[int, int, int]]:
        """ROCm free/total VRAM per PHYSICAL (HIP) gpu id via amd-smi, else ``[]``.

        Same index space and arch gate as the torch branch of ``_get_gpu_memory``,
        read from a child process instead. Reading VRAM through
        ``torch.cuda.mem_get_info`` creates a HIP primary context that the backend
        never gives back (~700 MiB measured), and the GGUF models this sizes run in a
        llama-server CHILD, so that VRAM is lost for the life of the process to
        answer a question a subprocess can answer.

        ``[]`` off ROCm and whenever amd-smi cannot answer, so the caller falls
        through to torch exactly as before. A gate that drops every card answers
        ``[]`` too, and torch then gates identically to the same empty list, so the
        one context this still costs is on the arm already headed for the CPU.
        A visible unified-memory APU answers ``[]`` as well: amd-smi sees only its
        dedicated VRAM carve-out, not the GTT pool the torch branch measures. So does
        anything it can only half answer: a subset is a non-empty list the caller
        takes as the whole host, which would drop a device from the count tensor
        parallelism is decided on.

        The inventory has to be HIP's, since the child is a HIP process: the answer
        is declined when the device set is filtered by something the mask cannot see
        (``GPU_DEVICE_ORDINAL``, a device-cgroup container), when amd-smi and HIP
        count different numbers of devices, and when they report totals far enough
        apart to be describing different memory scopes.
        """
        try:
            import torch
            if not LlamaCppBackend._torch_is_rocm(torch):
                return []
        except Exception:
            return []
        if not LlamaCppBackend._rocm_hip_is_reachable():
            # amd-smi would still list the hardware, but HIP cannot open it and
            # neither can the llama-server child. Torch answers [] here, as it did
            # before this branch existed, and the load goes to CPU.
            logger.debug(
                "amd-smi VRAM probe deferring to torch: this ROCm torch cannot open "
                "a device, so amd-smi's inventory is not usable by llama-server"
            )
            return []
        try:
            from utils.hardware.amd import get_gpu_vram_report

            if LlamaCppBackend._visibility_mask_is_unmappable():
                # A mask IS set but names devices by UUID, so it cannot be applied
                # here. amd-smi ignores the mask itself (it reads KFD directly), so
                # answering would offer cards the parent deliberately hid. Torch sees
                # only the masked set, so let it answer.
                logger.debug(
                    "amd-smi VRAM probe deferring to torch: the GPU visibility mask "
                    "is not index based, so amd-smi's device list cannot be filtered"
                )
                return []
            if LlamaCppBackend._gpu_device_ordinal_active():
                # _resolve_visible_physical_ids does not read this one, so "no mask"
                # below would be a lie wherever it does filter, and every hidden card
                # would be offered for ranking.
                logger.debug(
                    "amd-smi VRAM probe deferring to torch: GPU_DEVICE_ORDINAL is "
                    "set, so amd-smi's device list cannot be filtered"
                )
                return []
            if LlamaCppBackend._rocm_visibility_masks_are_stacked():
                # A ROCr mask under a HIP mask re-indexes it, and the resolved value
                # is the HIP one alone -- ordinals into the set ROCr left, not the
                # physical ids amd-smi answers for. Every later check agrees with the
                # wrong ids on a host whose cards match, so this is the only place it
                # can be caught.
                logger.debug(
                    "amd-smi VRAM probe deferring to torch: ROCR_VISIBLE_DEVICES is "
                    "stacked under a HIP-layer mask, so the visible ids are positions "
                    "in the ROCr-filtered set rather than physical devices"
                )
                return []
            mask = LlamaCppBackend._resolve_visible_physical_ids()
            if mask is not None and not mask:
                return []  # every device hidden
            vram, enumerated = get_gpu_vram_report()
            if not enumerated:
                return []
            # amd-smi ids are not HIP ids; translate before anything compares them
            # against the mask or hands them to the child.
            hip_by_gpu = LlamaCppBackend._amd_smi_hip_id_map(enumerated, mask)
            if hip_by_gpu is None:
                return []
            vram = {hip_by_gpu[_g]: _v for _g, _v in vram.items()}
            # amd-smi enumerates every card regardless of the mask, so drop the ones
            # this process cannot see. No mask means all of them.
            visible = set(mask) if mask is not None else set(hip_by_gpu.values())
            if not visible.issubset(vram):
                # A device that is visible and unmeasured: all or nothing.
                logger.debug(
                    "amd-smi VRAM probe deferring to torch: no usable VRAM reading "
                    f"for visible GPU(s) {sorted(visible - set(vram))}"
                )
                return []
            hip_count = LlamaCppBackend._rocm_hip_device_count()
            if len(visible) != hip_count:
                # This inventory has to BE the child's. amd-smi answers for the whole
                # host off sysfs, so anything filtering devices outside the
                # visibility variables leaves it larger than HIP's: a device-cgroup
                # container given one /dev/dri/renderD* node and no env var (see
                # test_overlay_skips_device_cgroup_filtered_container) is the usual
                # one. Smaller means amd-smi omitted a card HIP can open, which the
                # single-GPU shortcut in _amd_smi_hip_id_map would otherwise read as
                # a one-card host. Either way the numbers are not comparable.
                logger.debug(
                    "amd-smi VRAM probe deferring to torch: amd-smi reports "
                    f"{len(visible)} visible GPU(s) and HIP opens {hip_count}"
                )
                return []
            hip_totals = LlamaCppBackend._rocm_total_memory_mib_by_physical_id()
            disagreeing = sorted(
                _idx
                for _idx in visible
                if _idx in hip_totals
                and abs(vram[_idx][1] - hip_totals[_idx])
                > _AMD_SMI_TOTAL_TOLERANCE * hip_totals[_idx]
            )
            if disagreeing:
                # Two memory scopes, not two readings of one pool. The case that
                # matters is an APU _rocm_classify_unified_memory does not name (a
                # gfx1103 Phoenix wheel with no is_integrated flag): amd-smi sees the
                # BIOS carve-out and HIP the GTT pool, so accepting amd-smi's figure
                # would cut context or push the model to CPU. A partitioned MI300
                # reads the other way round. hardware.py::
                # _rocm_system_wide_vram_by_index gates its own overlay on the same
                # comparison. Devices torch cannot describe are absent from
                # hip_totals and fail open, so an unreadable properties call cannot
                # disable this branch wholesale.
                logger.debug(
                    "amd-smi VRAM probe deferring to torch: amd-smi and HIP report "
                    f"different total memory for GPU(s) {disagreeing}"
                )
                return []
            unified_ids = LlamaCppBackend._rocm_unified_memory_gpu_ids()
            if unified_ids:
                # amd-smi reports only the dedicated VRAM carve-out on a
                # unified-memory APU, while HIP reports the far larger GTT pool the
                # models actually run in: on a 128GiB Strix Halo with an 8GiB BIOS
                # carve-out amd-smi says 8GiB and torch says ~100GiB. Handing the
                # GGUF fitter the slice would cut context or push the model to CPU.
                # utils/hardware/hardware.py::_apply_unified_memory_correction
                # adopts torch's total over amd-smi's for the System tab for the
                # same reason. Defer the WHOLE host rather than mix two memory
                # scopes: dropping only the APU would remove it from placement, and
                # the two branches of _get_gpu_memory must not disagree.
                logger.debug(
                    "amd-smi VRAM probe deferring to torch: unified-memory APU(s) "
                    f"{sorted(unified_ids)} report only their dedicated VRAM slice"
                )
                return []
            arch_keeps = LlamaCppBackend._rocm_arch_gate_keep(binary, torch, for_llama_server)
            gpus: list[tuple[int, int, int]] = []
            for idx in sorted(visible):
                if not arch_keeps(idx):
                    continue
                free_mib, total_mib = vram[idx]
                gpus.append((idx, free_mib, total_mib))
            return gpus
        except Exception as e:
            logger.debug(f"amd-smi GPU probe failed: {e}")
            return []

    @staticmethod
    def _explain_empty_gpu_probe(binary: Optional[str] = None) -> str:
        """Why ``_get_gpu_memory`` came back empty, as one short phrase.

        Six unrelated causes share that one empty list -- no torch, no usable device, a
        stale visibility mask, the #7624 arch gate, amd-smi declining, or no GPU at all --
        and they need different fixes. No ``mem_get_info`` call, so this never creates the
        HIP context the amd-smi branch avoids. Never raises, never returns "".
        """
        try:
            binary = binary or LlamaCppBackend._find_llama_server_binary()
            if _metal_capable_host():
                # Same check as the load site: an Intel Mac wants its real reason.
                return "this probe reads CUDA and HIP only; Apple Silicon offloads through Metal"
            # Before the backend branches, because it is the reason underneath BOTH: a
            # render node this user cannot open leaves HIP with no device and the Vulkan
            # loader with nothing to enumerate, and "the Vulkan probe reported no device"
            # then sends the user after a driver that is fine (#10466). A Vulkan binary is
            # asked only about the render node, never /dev/kfd, and keeps its own reason.
            # Only of a build that can drive an AMD card: _is_vulkan_backend answers which
            # backend the install defers to, so a CUDA-plus-Vulkan build counts as CUDA. An
            # install this probe cannot read stays eligible, so a detection miss does not
            # lose the #10466 host.
            _is_vulkan = LlamaCppBackend._is_vulkan_backend(binary)
            _backends = LlamaCppBackend._installed_ggml_backends(binary)
            _amd_capable = not LlamaCppBackend._backend_lacks_gpu_lib(binary) and (
                _is_vulkan or "hip" in _backends or not _backends
            )

            # The ordinal space the three device lists index, so an entry can be checked
            # against something rather than only read. None on any host whose KFD topology
            # cannot be read, which is the direction that leaves a selector alone.
            try:
                from utils.hardware.amd import amd_kfd_gpu_node_count
                _amd_gpu_count = amd_kfd_gpu_node_count()
            except Exception:  # noqa: BLE001
                _amd_gpu_count = None

            def _post_rocr_device_count() -> "int | None":
                # ROCr filters the physical list FIRST and renumbers what survives, and
                # the HIP layer indexes those (_rocm_visibility_masks_are_stacked), so a
                # HIP ordinal is judged against the post-ROCr count: on two GPUs with
                # ROCR_VISIBLE_DEVICES=0 one survives, and HIP ordinal 1 hides everything
                # where the physical count of 2 reads it as harmless. None when the
                # survivors cannot be counted, which leaves the HIP selector alone.
                if not _rocr_filters:
                    return _amd_gpu_count
                _raw = (os.environ.get("ROCR_VISIBLE_DEVICES") or "").strip()
                if not _raw or not _amd_gpu_count:
                    return None
                # ROCr's own rule, from RvdFilter in core/inc/amd_filter_device.h: it
                # surfaces "tokens that are Legal and NOT Terminating", an index terminates
                # when it "lies outside the interval [0 - (numGpuDevices - 1)]" OR "maps to
                # a device that has been previously selected", and a token is Illegal when
                # it "can't be evaluated into an instance of Device UUID or Enumeration
                # Index". Every ending is a PREFIX of known length, including a repeated
                # ordinal ("0,0" surfaces one device) and an empty token. Only a UUID is
                # unknowable here: the KFD count is an ordinal space.
                _survivors = 0
                _selected: "set[int]" = set()
                for _entry in _raw.split(","):
                    _entry = _entry.strip()
                    if not _entry.isdigit():
                        # AMD documents the UUID form as the literal "GPU-XX"; anything
                        # else that is not an index is Illegal to ROCr too, so it ends the
                        # list at a length this does know.
                        return None if _entry.lower().startswith("gpu-") else _survivors
                    _idx = int(_entry)
                    if _idx >= _amd_gpu_count or _idx in _selected:
                        return _survivors
                    _selected.add(_idx)
                    _survivors += 1
                return _survivors

            def _hides_every_device(
                value: str,
                count: "int | None" = None,
                *,
                strict: bool = False,
            ) -> bool:
                # CUDA and HIP read the list left to right and stop at the first entry
                # that names no device, so a value that is empty, or whose FIRST entry
                # is empty or negative, exposes nothing; HIP_VISIBLE_DEVICES=0 still
                # exposes GPU 0.
                first = value.split(",")[0].strip()
                if first == "" or first.startswith("-"):
                    return True
                # Nor does a token have to LOOK like a number to end the list. clr takes
                # `index = atoi(str_id)` and rejects the token unless `str_id` is that
                # index written back out, so HIP_VISIBLE_DEVICES=garbage (and 0x1, and 00)
                # terminates on the FIRST token, exactly as -1 does. Asked only for the
                # clr-layer variables: ROCr's illegal-token rule is a separate parser, in
                # _is_an_illegal_rocr_selector, and a UUID token is left to
                # _cannot_be_resolved since resolving it needs the agents.
                if strict and not first.lower().startswith("gpu-"):
                    try:
                        _index = int(first)
                    except ValueError:
                        return True
                    if str(_index) != first:
                        return True
                # An entry that looks valid can still name nothing: the list stops at the
                # first index no device answers to, so HIP_VISIBLE_DEVICES=3 on a one-GPU
                # host exposes zero devices, which is the empty probe being explained. Only
                # ordinals, and only against a count actually read: reading an unknown
                # count as a bound would call every selector here a blocker.
                _bound = _amd_gpu_count if count is None else count
                if not _bound or not first.isdigit():
                    return False
                return int(first) >= _bound

            def _cannot_be_resolved(value: str) -> bool:
                # ROCr and clr both resolve a UUID ("0,GPU-4b2c...") as well as an
                # ordinal (rocdevice.cpp matches "GPU-" against HSA_AMD_AGENT_INFO_UUID).
                # Nothing here can match one, so it is unresolved rather than judged:
                # calling it a blocker invents a fault. Every other non-index is Illegal,
                # which _is_an_illegal_rocr_selector decides.
                first = value.split(",")[0].strip()
                return first.lower().startswith("gpu-")

            def _vk_selects_no_device(value: str) -> bool:
                # ggml stops at the first token with no integer prefix, so if the FIRST
                # lacks one device_indices stays empty. A sign counts as a prefix: "-1"
                # extracts and wraps, which throws rather than selecting nothing.
                _first = value.replace(",", " ").split()
                if not _first:
                    return True
                _token = _first[0]
                _digits = _token[1:] if _token[:1] in ("+", "-") else _token
                return not _digits[:1].isdigit()

            def _vk_ordinal_always_throws(value: str) -> bool:
                # A negative ordinal is the one out-of-range value decidable WITHOUT the
                # raw device count, because it wraps past every possible one: ggml reads
                # `size_t tmp; while (ss >> tmp)` (ggml-vulkan.cpp, ggml_vk_instance_init),
                # strtoull accepts the sign, and "-1" becomes 2**64-1, so it always throws
                # "Invalid Vulkan device index". No group membership repairs that, so it is
                # a blocker like an Illegal ROCr token. Extraction walks the list, so a
                # negative one throws wherever it sits provided every token before it still
                # extracts; "-0" wraps to 0 and is in range.
                for _token in value.replace(",", " ").split():
                    _signed = _token[:1] in ("+", "-")
                    _digits = _token[1:] if _signed else _token
                    if not _digits[:1].isdigit():
                        return False  # extraction stops here, so nothing after it is read
                    if _token[:1] == "-" and _digits.lstrip("0"):
                        return True
                return False

            def _is_an_illegal_rocr_selector(value: str) -> bool:
                # ROCr's filter (ROCR-Runtime, core/inc/amd_filter_device.h) calls a token
                # Illegal when it "can't be evaluated into an instance of Device UUID or
                # Enumeration Index", and an Illegal token terminates the list -- so an
                # illegal FIRST token leaves zero survivors, exactly as an out-of-range
                # ordinal does, and _post_rocr_device_count already counts it that way.
                # Reported as a definite blocker rather than as something to check only if
                # the group change fails, since no membership makes the runtime enumerate
                # a device again.
                first = value.split(",")[0].strip()
                return (
                    bool(first)
                    and not first.startswith("-")
                    and not first.isdigit()
                    and not first.lower().startswith("gpu-")
                )

            # Which of the four this host actually reads, per variable rather than one rule
            # for all. The runtime being explained is HIP, so clr's precedence holds:
            # rocdevice.cpp reads HIP_VISIBLE_DEVICES when its FIRST BYTE is not NUL and
            # CUDA_VISIBLE_DEVICES otherwise, so an empty CUDA mask behind a valid HIP one
            # is never consulted -- while an empty HIP mask does not win, clr's flag
            # defaulting to "" and unable to tell it from unset. ROCr sits BELOW that layer
            # and composes rather than defers (_rocm_visibility_masks_are_stacked), so an
            # empty ROCr mask does blind the runtime; Windows has no ROCr layer at all.
            # _active_gpu_visibility_mask is deliberately not used: it gates the same chain
            # on torch being a ROCm build, which is the wrong question for a HIP
            # llama-server sitting beside the CPU torch wheel this host tends to have.
            _hip_layer_var = (
                "HIP_VISIBLE_DEVICES"
                if os.environ.get("HIP_VISIBLE_DEVICES", "")
                else "CUDA_VISIBLE_DEVICES"
            )
            _rocr_filters = (
                sys.platform != "win32" and os.environ.get("ROCR_VISIBLE_DEVICES") is not None
            )
            _ordinal_filters = LlamaCppBackend._gpu_device_ordinal_active()

            _post_rocr_count = _post_rocr_device_count()

            masks = []
            blocking = []
            unresolved = []
            for var in (
                "CUDA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "GPU_DEVICE_ORDINAL",
            ):
                raw = os.environ.get(var)
                if raw is None:
                    continue
                phrase = f"{var}={raw!r}" if raw.strip() else f"{var} is empty"
                masks.append(phrase)
                if var == "GPU_DEVICE_ORDINAL":
                    _consulted = _ordinal_filters
                elif var == "ROCR_VISIBLE_DEVICES":
                    _consulted = _rocr_filters
                else:
                    _consulted = var == _hip_layer_var
                # A Vulkan build reads none of the four, so none of them blocks it.
                if _is_vulkan or not _consulted:
                    continue
                # ROCr indexes the physical list, the HIP layer indexes ROCr's survivors.
                _rocr = var == "ROCR_VISIBLE_DEVICES"
                _bound = _amd_gpu_count if _rocr else _post_rocr_count
                if _hides_every_device(raw, _bound, strict = not _rocr) or (
                    _rocr and _is_an_illegal_rocr_selector(raw)
                ):
                    blocking.append(phrase)
                elif _cannot_be_resolved(raw):
                    unresolved.append(phrase)
            # The four above are the HIP/CUDA selectors, which a Vulkan build reads none of.
            # GGML_VK_VISIBLE_DEVICES is the one it DOES read and _run_vulkan_probe passes
            # it through deliberately, so it is the only selector that can empty a Vulkan
            # probe. Only the two ends are decidable here: ggml_vk_instance_init reads
            # ordinals with `ss >> tmp` against the RAW vkEnumeratePhysicalDevices list,
            # before CPU devices are dropped and ICDs deduplicated, so this process does not
            # have the bound. A first token with no integer prefix (the empty string
            # included) selects nothing, and a NEGATIVE ordinal wraps past every possible
            # bound and always throws; a merely large positive one needs the bound to judge,
            # so it stays unresolved. Anything else is reported as unresolved.
            if _is_vulkan:
                _vk_raw = os.environ.get("GGML_VK_VISIBLE_DEVICES")
                if _vk_raw is not None:
                    _vk_phrase = (
                        f"GGML_VK_VISIBLE_DEVICES={_vk_raw!r}"
                        if _vk_raw.strip()
                        else "GGML_VK_VISIBLE_DEVICES is empty"
                    )
                    masks.append(_vk_phrase)
                    if _vk_selects_no_device(_vk_raw) or _vk_ordinal_always_throws(_vk_raw):
                        blocking.append(_vk_phrase)
                    else:
                        unresolved.append(_vk_phrase)

            mask_note = f" ({', '.join(masks)})" if masks else ""

            node_hint = None
            if _amd_capable:
                try:
                    from utils.hardware.amd import amd_node_permission_hint
                    node_hint = amd_node_permission_hint(needs_kfd = not _is_vulkan)
                except Exception:  # noqa: BLE001
                    node_hint = None

            def _closed_nodes_block_the_runtime() -> bool:
                # A host this cannot read answers True, which keeps the closed node as the
                # stated reason -- what this returned before the sibling check existed.
                try:
                    from utils.hardware.amd import amd_closed_nodes_block_the_runtime
                    return amd_closed_nodes_block_the_runtime(needs_kfd = not _is_vulkan)
                except Exception:  # noqa: BLE001
                    return True

            def _another_vendor_has_an_open_node() -> bool:
                # Vulkan enumerates any vendor, so an open Intel or NVIDIA render node is a
                # complete path for THIS binary and the closed AMD one cannot be the whole
                # story. Asked only for Vulkan: HIP needs /dev/kfd and an AMD render node,
                # which no other vendor's node substitutes for. False on a host this cannot
                # read, which keeps the behaviour it had before the check existed.
                if not _is_vulkan:
                    return False
                # Which drivers the loader would actually load decides this: a loader that
                # can only load AMD never opens the other vendor's driver, so its open node
                # is no path. One that can load that vendor is the opposite case and must
                # NOT suppress.
                try:
                    from utils.hardware.amd import (
                        a_non_amd_render_node_is_open,
                        the_vulkan_loader_can_only_load_amd,
                    )
                    if the_vulkan_loader_can_only_load_amd():
                        return False
                    return a_non_amd_render_node_is_open()
                except Exception:  # noqa: BLE001
                    return False

            # The closed node when it does NOT explain the empty probe, appended to whatever
            # reason does rather than returned in place of it.
            _second_finding = ""
            # Sufficient on its own, so it rides on the PRIMARY reason, not on what
            # gets demoted.
            _loader_finding = ""

            def _the_vulkan_loader_has_no_driver() -> bool:
                try:
                    from utils.hardware.amd import the_vulkan_loader_has_no_usable_driver
                    return the_vulkan_loader_has_no_usable_driver()
                except Exception:  # noqa: BLE001
                    return False

            def _the_loader_override_to_blame() -> "str | None":
                # Which of the three causes it is, since only two are repaired by
                # installing anything.
                try:
                    from utils.hardware.amd import the_vulkan_loader_override_to_blame
                    return the_vulkan_loader_override_to_blame()
                except Exception:  # noqa: BLE001
                    return None

            def _reason(text: str) -> str:
                return f"{text}{_loader_finding}{_second_finding}"

            if node_hint:
                # A mask hides devices whatever the node permissions are, so a host with
                # both needs both fixes. Only a mask that can hide EVERY device counts as a
                # second blocker: a valid selector beside a closed node is not why the probe
                # came back empty. mask_note below still lists all four -- there it
                # annotates what torch was looking at rather than claiming a repair.
                if blocking:
                    node_hint = (
                        f"{node_hint} A device visibility mask is also in force "
                        f"({', '.join(blocking)}), which the groups do not clear."
                    )
                elif unresolved:
                    node_hint = (
                        f"{node_hint} {', '.join(unresolved)} names a device this cannot "
                        f"resolve, so whether it also hides the card is unknown; check it "
                        f"if the groups do not help."
                    )
                # A closed node explains an empty probe only when it is one the runtime would
                # have used. On a multi-AMD host a render node can be shut while a sibling is
                # open, and the runtime then had a complete path and enumerated nothing
                # anyway, so the closed one is a SECOND finding. Asked per backend, since HIP
                # also needs /dev/kfd and that node has no sibling.
                _node_is_why = (
                    _closed_nodes_block_the_runtime() and not _another_vendor_has_an_open_node()
                )
                # A blocker the node repair cannot clear: manifests were found and none
                # is loadable, so the probe stays empty however the node is owned. Kept out
                # of node_hint, which may be demoted below: folding it in filed the one
                # diagnosis that always holds under "not why the probe is empty". "also"
                # only where the node repair really does precede it.
                _also = "also " if _node_is_why else ""
                if _is_vulkan and _the_vulkan_loader_has_no_driver():
                    # The repair depends on WHY. A filter that disables every manifest, and
                    # a forced list pointing at paths that do not resolve, are environment
                    # settings that reinstalling a driver leaves exactly as they were. Named
                    # rather than described, so the user has something to unset.
                    _override = _the_loader_override_to_blame()
                    if _override:
                        _loader_finding = (
                            f" The Vulkan loader {_also}has no driver it can load "
                            f"here, and what leaves it with none is {_override}: clear or "
                            f"correct that, since reinstalling the driver does not change "
                            f"an environment override."
                        )
                    else:
                        _loader_finding = (
                            f" The Vulkan loader {_also}has no driver it can load "
                            f"here: every ICD manifest it would read is missing its library "
                            f"or is 32-bit, so reinstall the Vulkan driver as well."
                        )
                if _node_is_why:
                    return f"{node_hint}{_loader_finding}"
                _second_finding = f" Separately, and not why the probe is empty: {node_hint}"
            if _is_vulkan:
                return _reason("the Vulkan probe reported no device")

            try:
                import torch
            except Exception:  # noqa: BLE001
                return _reason(f"torch is not importable, so no GPU could be enumerated{mask_note}")
            if not hasattr(torch, "cuda") or not torch.cuda.is_available():
                return _reason(f"torch reports no usable CUDA or HIP device{mask_note}")
            # Counting devices does not create a context; reading their memory would.
            count = torch.cuda.device_count()
            if not count:
                return _reason(f"torch enumerated 0 devices{mask_note}")

            if LlamaCppBackend._torch_is_rocm(torch):
                coverage = LlamaCppBackend._installed_llama_gfx_archs(binary)
                if coverage:
                    present = sorted(set(LlamaCppBackend._rocm_arch_by_physical_id().values()))
                    if present and not (set(present) & set(coverage)):
                        return _reason(
                            f"the installed llama.cpp build covers {sorted(coverage)} but this "
                            f"host has {present}, so the arch gate dropped every device"
                        )
                return _reason(
                    f"torch sees {count} ROCm device(s) but the probe returned none, so "
                    f"amd-smi and the torch fallback both declined{mask_note}"
                )
            return _reason(f"torch sees {count} device(s) but the probe returned none{mask_note}")
        except Exception as e:  # noqa: BLE001 -- diagnostics must not break a load
            return f"the reason could not be determined ({type(e).__name__})"

    @staticmethod
    def _nvidia_probe_script() -> Optional[Path]:
        """studio/nvidia_probe.py, the installers' NVML / CUDA driver reader (UNSLOTH_NVIDIA_PROBE wins)."""
        from utils.prebuilt.update_flow import find_installer_script
        return find_installer_script(env_var = "UNSLOTH_NVIDIA_PROBE", script_name = "nvidia_probe.py")

    @staticmethod
    def _nvml_rows_visible(rows: list[dict]) -> list[dict]:
        """The NVML rows CUDA_VISIBLE_DEVICES permits, in mask order: an index or a GPU-/MIG-
        uuid prefix per entry, as the CUDA runtime reads it. An entry naming no single device
        ends the mask there: "0,2,-1,1" exposes 0 and 2, "-1" alone hides every GPU. A GPU in
        MIG mode is one slice to CUDA (the first, when the mask does not name one), so its
        first slice row stands in for the parent's whole-card memory, and a second slice of
        the same card is dropped: the rows are keyed by physical index, and one slice per
        card is what the driver exposed before R570 and what the child is pinned to."""
        raw = os.environ.get("CUDA_VISIBLE_DEVICES")
        parents = [r for r in rows if not r.get("mig")]

        def as_cuda_sees(r: dict) -> dict:
            if r.get("mig"):
                return r  # a slice the mask named stays that slice
            slices = [
                s for s in rows if s.get("mig") and str(s.get("index")) == str(r.get("index"))
            ]
            return slices[0] if slices else r

        tokens = [t.strip() for t in (raw or "").split(",") if t.strip()]
        picked: list[dict] = []
        if raw is None:
            picked = [as_cuda_sees(r) for r in parents]
        for token in tokens:
            if token.isdigit():
                hits = [r for r in parents if str(r.get("index")) == token]
            elif token.startswith("GPU-"):
                hits = [r for r in parents if str(r.get("uuid", "")).startswith(token)]
            elif token.startswith("MIG-"):
                # A MIG- entry names a slice row the probe lists under its parent.
                hits = [
                    r for r in rows if r.get("mig") and str(r.get("uuid", "")).startswith(token)
                ]
            else:
                hits = []
            if len(hits) != 1:
                break
            picked.extend(
                r
                for r in map(as_cuda_sees, hits)
                if str(r.get("index")) not in {str(p.get("index")) for p in picked}
            )
        return picked

    @staticmethod
    def _uuid_by_index(rows: list[dict]) -> dict[int, str]:
        """The uuid each visible index stands for: every row when the mask named rows by
        uuid, else only a slice standing in for its MIG parent (a slice has no index of its
        own). Derived from the rows and the mask on each call, so it belongs to no probe."""
        tokens = [t.strip() for t in (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")]
        by_uuid = any(t and not t.isdigit() for t in tokens)
        return {
            int(r["index"]): str(r["uuid"])
            for r in LlamaCppBackend._nvml_rows_visible(rows)
            if str(r.get("index", "")).isdigit() and (by_uuid or r.get("mig"))
        }

    @staticmethod
    def _child_visibility_for(gpu_indices) -> str:
        """The mask for these indices: the inherited uuids when the mask named the rows by
        uuid (a MIG slice must stay a MIG- entry), else the indices themselves."""
        ids = [int(i) for i in gpu_indices]
        by_index = LlamaCppBackend._uuid_by_index(LlamaCppBackend._NVML_ROWS)
        if by_index and all(i in by_index for i in ids):
            return ",".join(by_index[i] for i in ids)
        return ",".join(str(i) for i in ids)

    @staticmethod
    def _get_gpu_memory_nvml() -> list[tuple[int, int, int]]:
        """Free and total memory per NVIDIA GPU from NVML (studio/nvidia_probe.py in a child
        with a deadline), for a host whose nvidia-smi is absent, stale or hung: read as "no
        GPU", it pinned the embedding server to the CPU. Physical indices, masked like the
        nvidia-smi rows; a visible row without a memory reading voids the answer, so the
        torch probe decides."""
        if sys.platform == "darwin" or os.environ.get("UNSLOTH_NVIDIA_LIBRARY_PROBE", "1") == "0":
            return []
        script = LlamaCppBackend._nvidia_probe_script()
        if script is None:
            return []
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(script), "--json"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                env = utf8_child_env(child_env_without_native_path_secret()),
                **_windows_hidden_subprocess_kwargs(),
            )
            payload = json.loads(result.stdout or "null")
        except Exception as e:
            logger.debug(f"NVML probe failed: {e}")
            return []
        if not isinstance(payload, dict) or payload.get("source") != "nvml":
            return []
        LlamaCppBackend._NVML_ROWS = list(payload.get("devices") or [])
        gpus: list[tuple[int, int, int]] = []
        for row in LlamaCppBackend._nvml_rows_visible(LlamaCppBackend._NVML_ROWS):
            try:
                idx = int(row["index"])
                free_mib = int(row.get("memory_free_mib") or 0)
                total_mib = int(row.get("memory_total_mib") or 0)
            except (KeyError, TypeError, ValueError):
                return []
            if total_mib <= 0:
                # The probe writes a failed reading as total 0. One visible GPU without a
                # reading voids the answer: a partial list would place and split across
                # fewer GPUs than the child enumerates. A full card (free 0) is a reading.
                return []
            gpus.append((idx, free_mib, total_mib))
        gpus.sort(key = lambda g: g[0])
        return gpus

    @staticmethod
    def _get_gpu_memory(
        binary: Optional[str] = None, *, for_llama_server: bool = False
    ) -> list[tuple[int, int, int]]:
        """Query free AND total memory per GPU.

        Order: ``nvidia-smi`` (NVIDIA, respects ``CUDA_VISIBLE_DEVICES``), then
        ``amd-smi`` (ROCm, the same answer without a HIP context), then
        ``torch.cuda.mem_get_info``, a universal fallback that works on ROCm too
        (HIP reuses the ``torch.cuda.*`` namespace). The fallback covers #5106
        (nvidia-smi returned [] on AMD) and hosts with no smi tool on PATH; it is
        last because it costs this process a permanent CUDA/HIP context, and kept
        because callers read ``[]`` as "no GPU" and would silently drop to CPU.

        On a Vulkan build the ggml Vulkan probe is authoritative, so the indices are
        ggml's compact Vulkan ordinals (what ``--device Vulkan<i>`` selects). It
        reports ``total`` for discrete cards and 0 for an iGPU (shared RAM) so the
        fit falls back to free*frac. The torch branch treats an AMD unified-memory
        APU the same way by arch name, ROCm having no iGPU flag of its own.

        ``for_llama_server`` opts into the ROCm arch gate: devices whose gfx arch is
        not in the prebuilt's built-arch list are dropped, so auto placement cannot
        pick a GPU that binary has no kernels for (#7624). Off by default because
        this probe also backs ``core/rag/embeddings.py::_resolve_auto``, where such a
        device is usually still fine under PyTorch and filtering it would push
        sentence-transformers onto the CPU for no reason. Pass it only where a
        llama-server process is about to be placed.

        Returns (gpu_index, free_mib, total_mib) sorted by index; empty if no
        supported GPU is reachable."""
        binary = binary or LlamaCppBackend._find_llama_server_binary()
        if LlamaCppBackend._is_vulkan_backend(binary):
            return LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
        # ── NVIDIA via nvidia-smi ────────────────────────────────────
        unsized_ids: list[int] = []
        try:
            from utils.hardware import gpu_query

            # Placement / fit: never a cached reading. A hung CLI raises so the MIG-aware NVML branch runs.
            result = gpu_query.run_nvidia_smi(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                env = child_env_without_native_path_secret(),
                **_windows_hidden_subprocess_kwargs(),
            )
            if result.returncode == 0:
                allowed = LlamaCppBackend._visible_devices_mask("CUDA_VISIBLE_DEVICES")
                gpus: list[tuple[int, int, int]] = []
                for line in result.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 2:
                        continue
                    # Index and free required; skip a bad line rather than abandon
                    # the probe to the torch fallback.
                    try:
                        idx = int(parts[0])
                        free_mib = int(parts[1])
                    except ValueError:
                        # Jetson answers "[N/A]": no dedicated VRAM (#4862).
                        if parts[0].isdigit() and (allowed is None or int(parts[0]) in allowed):
                            unsized_ids.append(int(parts[0]))
                        continue
                    # Total parsed separately: a two-column line or a non-integer
                    # total ("N/A" on MIG/vGPU) keeps the GPU at total 0 (fit uses
                    # the free*frac fallback) instead of dropping it.
                    total_mib = 0
                    if len(parts) >= 3 and parts[2]:
                        try:
                            total_mib = int(parts[2])
                        except ValueError:
                            total_mib = 0
                    if allowed is not None and idx not in allowed:
                        continue
                    gpus.append((idx, free_mib, total_mib))
                # Match the docstring's sort-by-id guarantee (driver order isn't).
                gpus.sort(key = lambda g: g[0])
                if gpus:
                    LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES = True
                    if for_llama_server:
                        gpus = LlamaCppBackend._cuda_sm_gate_keep(binary, gpus)
                    # The CLI row is the carve-out, not the pool. Discrete rows unchanged.
                    return LlamaCppBackend._widen_integrated_cuda_rows(gpus)
        except Exception as e:
            logger.debug(f"nvidia-smi probe failed: {e}")

        # ── NVIDIA via NVML, when nvidia-smi is absent, stale or hung ────
        nvml_gpus = LlamaCppBackend._get_gpu_memory_nvml()
        if nvml_gpus:
            LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES = True
            if for_llama_server:
                nvml_gpus = LlamaCppBackend._cuda_sm_gate_keep(binary, nvml_gpus)
            # nvidia-smi IS NVML: same carve-out, same PCI indexing, same correction.
            return LlamaCppBackend._widen_integrated_cuda_rows(nvml_gpus)

        # ── AMD ROCm via amd-smi ─────────────────────────────────────
        rocm_gpus = LlamaCppBackend._get_gpu_memory_amd_smi(
            binary, for_llama_server = for_llama_server
        )
        if rocm_gpus:
            return rocm_gpus

        # ── Torch fallback (covers AMD ROCm and missing nvidia-smi) ──
        gpus = LlamaCppBackend._get_gpu_memory_torch(binary, for_llama_server)
        if gpus or not (for_llama_server and unsized_ids):
            return gpus
        return LlamaCppBackend._tegra_unified_memory_rows(unsized_ids)

    @staticmethod
    def _tegra_unified_memory_rows(gpu_ids: list[int]) -> list[tuple[int, int, int]]:
        """Price a Jetson's GPU from system RAM when nvidia-smi answers ``[N/A]`` and
        torch cannot see it (a generic aarch64 wheel), so the context is still capped
        (#4862). llama-server only: it runs on JetPack's CUDA, torch-backed callers do not."""
        from utils.tegra import is_tegra

        if not is_tegra():
            return []
        avail = LlamaCppBackend._available_system_memory_mib()
        if avail is None:
            return []
        cgroup_mib = LlamaCppBackend._cgroup_available_memory_mib()
        # Same policy as the torch integrated branch: a cgroup-bound pool publishes no total.
        total = (
            0
            if cgroup_mib is not None and cgroup_mib <= avail
            else LlamaCppBackend._total_system_memory_mib() or 0
        )
        free = _apply_igpu_host_reserve_mib(avail, True)
        LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES = True
        # Torch never saw these, so record them as integrated or the pool is credited twice.
        LlamaCppBackend._INTEGRATED_CUDA_IDS[LlamaCppBackend._integrated_cuda_mask_key()] = set(
            gpu_ids
        )
        return [(idx, free, total) for idx in sorted(gpu_ids)]

    @staticmethod
    def _get_gpu_memory_torch(
        binary: Optional[str], for_llama_server: bool
    ) -> list[tuple[int, int, int]]:
        """The torch arm of ``_get_gpu_memory``."""
        try:
            import torch

            if not hasattr(torch, "cuda") or not torch.cuda.is_available():
                return []
            if not hasattr(torch.cuda, "mem_get_info"):
                return []
            # CUDA ordinals from here on, not nvidia-smi PCI indices.
            LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES = False
            # torch.cuda enumerates GPUs RELATIVE to the visibility mask. We
            # feed these IDs back into the subprocess as CVD, so visible ordinals
            # must be translated to physical indices first; otherwise CVD=2,3
            # gets rewritten to 0,1 and targets the wrong GPUs.
            # Match utils/hardware/hardware.py::_get_parent_visible_gpu_spec:
            # treat an empty mask (HIP_VISIBLE_DEVICES="") as "no GPUs" rather
            # than falling through. ``or`` would coerce "" to the wrong source.
            # Empty mask (CVD="") yields an empty list -> no GPUs, consistent
            # with the nvidia-smi path.
            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()
            # Shared-pool APU: same as the Vulkan iGPU path. Hold back the host
            # margin, and report total 0 since that "total" is system RAM.
            unified_ids = LlamaCppBackend._rocm_unified_memory_gpu_ids()
            # An integrated CUDA SoC shares one pool too, and its free reading is wrong
            # in the OPPOSITE direction to ROCm's -- see below.
            integrated_ids = LlamaCppBackend._integrated_cuda_gpu_ids()
            # Same #7624 arch gate the amd-smi branch applies, from the one helper.
            arch_keeps = LlamaCppBackend._rocm_arch_gate_keep(binary, torch, for_llama_server)
            # How many devices the shared pool is about to be divided between. Read
            # once, from the same set the loop tests, so the count and the per-device
            # decision cannot disagree and produce a division by zero.
            shared_cuda_count = sum(1 for _i in integrated_ids if arch_keeps(_i))
            gpus = []
            # Windows ROCm's free reading is an over-report on discrete cards too,
            # not only on the shared pool handled below (#8403). It is capped
            # against this process's own allocator, so VRAM a resident torch model
            # holds in the backend process stops being offered to llama.cpp slots.
            from utils.hardware import trusted_mem_get_info

            for ordinal in range(torch.cuda.device_count()):
                free_bytes, total_bytes = trusted_mem_get_info(ordinal)
                idx = (
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
                if not arch_keeps(idx):
                    continue
                shared = idx in unified_ids
                integrated = idx in integrated_ids
                cgroup_bound = False
                raw_mib = free_bytes // (1024 * 1024)
                total_mib = total_bytes // (1024 * 1024)
                if shared:
                    # ROCm's free is unreliable on a shared pool (Windows HIP
                    # reports free==total, #7072), and system RAM is the real
                    # ceiling there, so cap before taking the reserve.
                    avail = LlamaCppBackend._available_system_memory_mib()
                    if avail is not None:
                        raw_mib = min(raw_mib, avail)
                elif integrated:
                    # cudaMemGetInfo's free half here is MemFree, which counts the
                    # page cache as used, so a GGUF's own download or mmap collapses it
                    # and the context is fitted against its own cached bytes (#9889).
                    # MemAvailable prices that reclaimable cache back in.
                    avail = LlamaCppBackend._available_system_memory_mib()
                    # A zero total is a probe that could not size the pool, not a pool
                    # of zero: capping against it would take the device to nothing.
                    if avail is not None and total_mib > 0:
                        raw_mib = min(total_mib, max(raw_mib, avail))
                    # Again as a CEILING: `avail` is cgroup-capped already, but as a
                    # lower bound that is thrown away whenever host-wide MemFree is
                    # larger, the normal case in a container. Allocations here are
                    # charged to the cgroup, so an oversized fit dies at memory.max.
                    # `<=` below, not `<`: _available_system_memory_mib is itself
                    # cgroup-capped, so it hands back exactly the remainder whenever
                    # MemAvailable exceeds it, which makes equality the ordinary result
                    # on a constrained Spark rather than a coincidence.
                    cgroup_mib = LlamaCppBackend._cgroup_available_memory_mib()
                    if cgroup_mib is not None and cgroup_mib <= raw_mib:
                        raw_mib = cgroup_mib
                        # ...and then the pool is the container's, not the device's.
                        # _vram_usable_mib takes its occupancy reserve as
                        # (1 - frac) * total, so a host-wide total against a container's
                        # free reading reserves memory this process cannot reach: a
                        # 16 GiB cgroup on a 121 GiB Spark loses most of its budget to
                        # a card it does not have. Publishing no total is the same
                        # answer the ROCm shared pool gives, and prices the fit off the
                        # free reading, which IS the container's ceiling.
                        cgroup_bound = True
                    # One pool, however many integrated devices draw on it. Every
                    # figure above is a whole-pool figure, so handing it to each
                    # device in turn lets a caller that sums across cards commit the
                    # same bytes twice. Share it instead. Division by 1 on every
                    # shipping part, since no product pairs two integrated SoCs.
                    if shared_cuda_count > 1:
                        raw_mib //= shared_cuda_count
                        total_mib //= shared_cuda_count
                free_mib = _apply_igpu_host_reserve_mib(raw_mib, shared or integrated)
                if free_mib < raw_mib:
                    logger.info(
                        f"{'CUDA' if integrated else 'ROCm'} device {idx} is a "
                        f"unified-memory {'SoC' if integrated else 'APU'} sharing system "
                        f"RAM; reserving {raw_mib - free_mib}MiB host headroom "
                        f"({raw_mib}->{free_mib}MiB usable)"
                    )
                # ROCm publishes 0 because that "total" is system RAM of unknown
                # scope. An integrated part's total IS the pool, so it is a real
                # ceiling; zeroing it would drop the fit to free*frac.
                gpus.append((idx, free_mib, 0 if shared or cgroup_bound else total_mib))
            # Match the nvidia-smi path's docstring guarantee of sorted-by-id.
            return sorted(gpus, key = lambda g: g[0])
        except Exception as e:
            logger.debug(f"torch GPU probe failed: {e}")
            return []

    @staticmethod
    def _vulkan_auto_gpu_memory(gpus: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        """Prefer discrete devices for automatic Vulkan placement."""
        discrete = [gpu for gpu in gpus if gpu[2] > 0]
        return discrete or gpus

    @staticmethod
    def _run_vulkan_probe(binary: Optional[str] = None) -> list[dict]:
        """Run ``_vulkan_probe.py`` and parse its per-device lines.

        Returns raw (uncapped) rows sorted by index:
        ``{"index", "free_mib", "total_mib", "is_igpu", "name"}``. The index is
        ggml's compact Vulkan ordinal -- the one the registry names
        ``Vulkan<index>`` and load_model pins with ``--device``, NOT the raw
        ``GGML_VK_VISIBLE_DEVICES`` space. A user-set ``GGML_VK_VISIBLE_DEVICES``
        is honored by ggml (passed through), so the list already reflects it.
        ``name`` is ggml's device description; "" from an older 4-column probe.
        [] when no Vulkan build or device is reachable.
        """
        binary = binary or LlamaCppBackend._find_llama_server_binary()
        if not binary:
            return []
        # Memoised per load: a 15s subprocess now asked for from several places. Cleared
        # by _load_probe_memo_scope, so a device change between loads is never cached.
        _memo = _load_probe_memo_get("vulkan", binary)
        if _memo is not None:
            return _memo
        binary_dir = _llama_lib_dir(binary)
        if not _lib_dir_has_ggml_backend(binary_dir, "vulkan"):
            _load_probe_memo_put("vulkan", binary, [])
            return []

        env = child_env_without_native_path_secret()
        # Let ggml apply GGML_VK_VISIBLE_DEVICES; Python sees compact ordinals.
        if sys.platform != "win32":
            # Let the loader resolve sibling ggml libs next to the binary.
            # _loader_path_var keeps macOS off LD_LIBRARY_PATH, which dyld
            # ignores (#8566).
            _loader_var = _loader_path_var()
            existing_ld = env.get(_loader_var, "")
            env[_loader_var] = (
                f"{binary_dir}{os.pathsep}{existing_ld}" if existing_ld else str(binary_dir)
            )
        probe_script = Path(__file__).resolve().parent.with_name("_vulkan_probe.py")
        try:
            # UTF-8 to match the probe's stdout reconfigure: device names can be
            # non-ASCII, and the platform-default decode (cp1252) could throw.
            result = subprocess.run(
                [sys.executable, str(probe_script), str(binary_dir)],
                capture_output = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 15,
                env = utf8_child_env(env),
                **_windows_hidden_subprocess_kwargs(),
            )
            if result.returncode != 0:
                logger.debug(
                    f"vulkan GPU probe exited {result.returncode}: {result.stderr.strip()}"
                )
                _load_probe_memo_put("vulkan", binary, [])
                return []
        except Exception as e:
            logger.debug(f"vulkan GPU probe failed: {e}")
            # Memoised too: an unanswered probe comes straight back here, so an uncached
            # timeout is paid twice over and again per rung.
            _load_probe_memo_put("vulkan", binary, [])
            return []

        rows: list[dict] = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t")
            # 4 columns from an older probe (no name); 5 with the name column.
            if len(parts) not in (4, 5, 6):
                continue
            try:
                rows.append(
                    {
                        "index": int(parts[0]),
                        "free_mib": int(parts[1]) // (1024 * 1024),
                        "is_igpu": parts[2] == "1",
                        "total_mib": int(parts[3]) // (1024 * 1024),
                        "name": parts[4].strip() if len(parts) >= 5 else "",
                        # Absent on an older staged probe, so it reads unknown and declines.
                        "type_known": len(parts) >= 6 and parts[5].strip() == "1",
                    }
                )
            except ValueError:
                continue
        rows.sort(key = lambda r: r["index"])
        _load_probe_memo_put("vulkan", binary, rows)
        return rows

    @staticmethod
    def vulkan_device_inventory(binary: Optional[str] = None) -> list[dict]:
        """UI-facing Vulkan device list: the devices llama-server will actually
        use, with real totals (an iGPU keeps its shared-RAM total here -- the
        caller labels it, unlike the fit which zeroes it). Same rows as
        ``_run_vulkan_probe``; names fall back to ``Vulkan<i>``.
        """
        rows = LlamaCppBackend._run_vulkan_probe(binary)
        for row in rows:
            if not row["name"]:
                row["name"] = f"Vulkan{row['index']}"
        return rows

    @staticmethod
    def _get_gpu_free_memory_vulkan(binary: Optional[str] = None) -> list[tuple[int, int, int]]:
        """Query free (and total) VRAM per device via the bundled ggml Vulkan backend.

        Fit-oriented view of ``_run_vulkan_probe``: returns (device_index,
        free_mib, total_mib) sorted by index. iGPUs leave a host-RAM margin (see
        ``_apply_igpu_host_reserve_mib``) and report total 0; discrete cards pass
        their real total through. [] when no Vulkan build or device is reachable.
        """
        gpus: list[tuple[int, int, int]] = []
        for row in LlamaCppBackend._run_vulkan_probe(binary):
            idx, free_mib, is_igpu = row["index"], row["free_mib"], row["is_igpu"]
            # iGPU "total" is shared RAM, not a VRAM budget -> keep 0 so the
            # fit stays on free*frac (the host reserve below is its
            # headroom); a discrete card passes its real total through.
            total_mib = 0 if is_igpu else row["total_mib"]
            capped = _apply_igpu_host_reserve_mib(free_mib, is_igpu)
            if capped < free_mib:
                logger.info(
                    f"Vulkan device VK{idx} is an integrated GPU sharing system "
                    f"RAM; reserving {free_mib - capped}MiB host headroom "
                    f"({free_mib}->{capped}MiB usable)"
                )
            gpus.append((idx, capped, total_mib))
        if gpus:
            logger.info(
                "Vulkan GPU memory detected: "
                + ", ".join(f"VK{idx}={free}MiB" for idx, free, _total in gpus)
            )
        return gpus

    @staticmethod
    def _igpu_backed_free_mib(free_mib: int, headroom_mib: int = 0) -> int:
        """Bound a raw iGPU free reading by host RAM plus any firmware carve-out.

        Vulkan combines OS-visible RAM with carve-outs. Memory above ``MemTotal``
        is a conservative floor for the free carve-out; add ``MemAvailable`` for
        the host-backed share after its required headroom. WSL cannot use this
        inference because the VM and adapter report different memory scopes.
        """
        available = LlamaCppBackend._available_system_memory_mib()
        total_ram = LlamaCppBackend._total_system_memory_mib()
        if available is None or total_ram is None:
            return free_mib
        unseen = 0 if _is_wsl() else max(0, free_mib - total_ram)
        host_backed = max(0, available - max(0, headroom_mib))
        return min(free_mib, unseen + host_backed)

    @staticmethod
    def _igpu_backed_usable_mib(usable_mib: int) -> int:
        """Bound a post-reserve iGPU budget for shared-heap admission.

        Placement keeps the Vulkan reading so an existing full-offload plan is
        not demoted to the fitter. The host guard alone reconstructs the raw
        reading, bounds its host-backed share with the whole-system headroom,
        and caps that against the already-reserved placement budget. A zero
        budget stays zero because the raw value below the reserve is no longer
        recoverable and could not admit weights either way.
        """
        if usable_mib <= 0:
            return 0
        raw_free_mib = usable_mib + _IGPU_HOST_RESERVE_MIB
        backed_mib = LlamaCppBackend._igpu_backed_free_mib(
            raw_free_mib, headroom_mib = _HOST_RAM_HEADROOM_MIB
        )
        return min(usable_mib, backed_mib)

    @staticmethod
    def _cgroup_memory_budgets() -> list[tuple[int, int]]:
        """Memory this process can still charge to an enforcing cgroup.

        Delegates to ``utils.host_memory``; empty = no finite limit.
        """
        return _host_memory.cgroup_memory_budgets(_CGROUP_ROOT, _PROC_SELF_CGROUP)

    @staticmethod
    def _cgroup_available_memory_mib() -> Optional[int]:
        """What this process can still charge, i.e. the tightest REMAINDER."""
        return _host_memory.cgroup_headroom_mib(LlamaCppBackend._cgroup_memory_budgets())

    @staticmethod
    def _cgroup_memory_limit_mib() -> Optional[int]:
        """The CAPACITY an enforcing cgroup allows, i.e. the tightest LIMIT.

        Not interchangeable with the remainder above, which shrinks as the container
        fills and would price against memory that is merely busy rather than absent."""
        return _host_memory.cgroup_limit_mib(LlamaCppBackend._cgroup_memory_budgets())

    @staticmethod
    def _available_system_memory_mib() -> Optional[int]:
        """Available system RAM in MiB (psutil, then /proc/meminfo), or None if
        neither is readable, capped by this process's cgroup remainder. On a
        unified-memory APU this, not the ROCm-reported VRAM, is the real ceiling:
        the weights load into shared system RAM."""
        return _host_memory.usable_mib(
            _host_memory.system_available_mib(), LlamaCppBackend._cgroup_available_memory_mib()
        )

    @staticmethod
    def _total_system_memory_mib() -> Optional[int]:
        """Total system RAM in MiB (psutil, then /proc/meminfo), or None if neither is
        readable. The ceiling on what ``MemAvailable`` can ever become, so a preflight that
        runs before the resident model and pipeline are torn down can price against it
        without charging for host memory that is about to come back."""
        return _host_memory.system_total_mib()

    @staticmethod
    def _host_memory_capacity_mib() -> Optional[int]:
        """Host RAM this process may actually charge, in MiB.

        MemTotal narrowed by an enforcing cgroup, whose limit a container's own
        /proc/meminfo does not show. The LIMIT, never the remainder, which shrinks as
        the container fills and would price against memory that is merely busy.
        """
        return _host_memory.usable_mib(
            LlamaCppBackend._total_system_memory_mib(), LlamaCppBackend._cgroup_memory_limit_mib()
        )

    _ARGV_MODEL = frozenset({"-m", "--model"})
    _ARGV_RPC = frozenset({"--rpc"})

    @staticmethod
    def _binary_ships_no_gpu_backend(
        binary: Optional[str] = None, env: Optional[Mapping[str, str]] = None
    ) -> bool:
        """Whether a split-library build beside ``binary`` carries no GPU backend at all.

        Stricter than ``_backend_lacks_gpu_lib``, which reads cuda, hip and vulkan only:
        a SYCL, MUSA, CANN or OpenCL build offloads too, and pricing its weights against
        host RAM would refuse a load the accelerator can hold. A static or unrecognised
        layout, a directory this cannot read, and a GGML_BACKEND_PATH pointing the child
        at plugins elsewhere all answer False so a GPU build keeps its VRAM credit.
        """
        binary = binary or LlamaCppBackend._find_llama_server_binary()
        if not binary:
            return False
        source = os.environ if env is None else env
        if str(source.get("GGML_BACKEND_PATH", "") or "").strip():
            return False
        try:
            files = tuple(path.name for path in _llama_lib_dir(binary).iterdir() if path.is_file())
        except OSError:
            return False
        cpu_stem = "ggml-cpu" if sys.platform == "win32" else "libggml-cpu"
        base_stem = "ggml-base" if sys.platform == "win32" else "libggml-base"
        split_library = any(name.startswith((cpu_stem, base_stem)) for name in files)
        return split_library and not any(_GGML_GPU_BACKEND_RE.match(name) for name in files)

    def _argv_offloads_every_layer(
        self,
        argv: Iterable[str],
        env: Optional[Mapping[str, str]] = None,
    ) -> bool:
        """Whether the final command puts every model layer on a GPU.

        Unknown placement is treated as partial offload so unverified GPU memory
        is never credited.
        """
        args = [str(a) for a in argv or ()]
        if _device_selection_is_cpu(args, env):
            return False
        if _args_place_tensors_on_cpu(args) or _env_places_tensors_on_cpu(env):
            return False
        # The fitter owns placement unless it is explicitly disabled.
        if fit_is_effectively_on(args, env):
            return False
        try:
            requested = parse_gpu_layers_override(args)
        except ValueError:
            return False
        if requested is None:
            return False
        # -1 means every layer; otherwise the count must exceed the block count.
        n_layers = self.n_layers
        return requested == -1 or (bool(n_layers) and requested > n_layers)

    def _launch_forces_full_offload(
        self,
        argv: Iterable[str],
        env: Optional[Mapping[str, str]] = None,
    ) -> bool:
        """Whether the child puts every layer on a GPU whatever the fitter says. The
        fitter cannot lower a count the user fixed (common/fit.cpp "n_gpu_layers
        already set by user", downgraded to a warning), so a count above the block
        count stands under ``--fit on``, as does no count at all with the fitter off
        (llama.cpp's default is ``-1``)."""
        args = [str(a) for a in argv or ()]
        if self._argv_offloads_every_layer(args, env):
            return True
        if _device_selection_is_cpu(args, env):
            return False
        if _args_place_tensors_on_cpu(args) or _env_places_tensors_on_cpu(env):
            return False
        try:
            requested = parse_gpu_layers_override(args)
        except ValueError:
            return False
        if requested is None and env and _env_fixes_gpu_layers(env):
            raw = str(env.get("LLAMA_ARG_N_GPU_LAYERS", "")).strip().lower()
            if raw == "all":
                return True
            try:
                requested = int(raw)
            except ValueError:
                return False
        if requested is None:
            return not fit_is_effectively_on(args, env)
        n_layers = self.n_layers
        if not n_layers:
            return False
        return requested > n_layers

    @staticmethod
    def _rows_the_child_can_reach(detected_gpus, pinned_ids) -> list:
        """``detected_gpus`` narrowed to the cards THIS launch pinned the child to.

        ``_launch_host_shortfall_message`` sums free VRAM over every row it is handed,
        which is right for an advisory about the host and wrong for a decision about
        one launch: a child pinned to a subset cannot spend the VRAM on the cards it
        cannot see, so crediting them prices the spill too low and can leave an
        unmapped oversized load unrewritten, which is an OOM rather than a slow load.

        ``pinned_ids`` is the narrowing this load performed, not the visibility mask
        in ``env``. An inherited CUDA_VISIBLE_DEVICES is already applied: the probe
        only ever reported the cards it left visible, so re-applying it here would
        subtract the same pin twice and price a single-card host as having none.
        None means nothing narrowed the launch, which is not the same as an empty
        pin, and physical ids match ``row[0]``.
        """
        rows = list(detected_gpus or ())
        if pinned_ids is None:
            return rows
        visible = set(pinned_ids)
        return [row for row in rows if row[0] in visible]

    def _shared_heap_budget(
        self,
        gpus: Iterable[tuple],
        shared_gpu_ids: Collection[int],
        model_bytes: int,
        argv: list[str],
        env: Optional[Mapping[str, str]] = None,
    ) -> tuple[int, int]:
        """Return reachable shared Vulkan memory and the bytes it must hold.

        Shared iGPU rows overlap, so only the largest pool is credited. The pool
        must be reachable by an unambiguous full offload -- a device pin that
        names it, or no pin on a host where llama.cpp finds no discrete card --
        and only selected discrete cards reduce the bytes assigned to it.
        """
        rows = [(row[0], max(0, row[1])) for row in gpus]
        pinned = _extra_args_main_device(argv)
        if pinned is None and env:
            pinned = env.get("LLAMA_ARG_DEVICE")
        if pinned is None:
            reachable = {idx for idx, _free in rows}
            # Unpinned, llama.cpp builds the device list itself and appends
            # integrated GPUs only when it found no discrete one, so a shared
            # heap beside a discrete card receives nothing.
            if any(idx not in shared_gpu_ids for idx in reachable):
                reachable -= set(shared_gpu_ids)
        else:
            # Match the ggml registry names emitted by _vulkan_pin_args.
            names = {device.strip().lower() for device in str(pinned).split(",")}
            reachable = {idx for idx, _free in rows if f"vulkan{idx}" in names}
        pool = 0
        if (
            shared_gpu_ids
            and self._argv_offloads_every_layer(argv, env)
            # Tensor-split positions cannot be mapped reliably to probe indices.
            and not _extra_args_set_any_flag(argv, _TENSOR_SPLIT_FLAGS)
            and not str((env or {}).get("LLAMA_ARG_TENSOR_SPLIT") or "").strip()
            # With multiple devices, split-mode none does not identify the recipient.
            and not (len(reachable) > 1 and _split_mode_confines_to_one_device(argv, env))
        ):
            budgets = []
            for idx, free in rows:
                if idx not in shared_gpu_ids or idx not in reachable:
                    continue
                backed = self._igpu_backed_usable_mib(free)
                if backed < free:
                    logger.info(
                        f"Vulkan device VK{idx} offers {free}MiB to placement; "
                        f"host admission credits {backed}MiB"
                    )
                budgets.append(backed)
            pool = max(budgets, default = 0)
        on_cards = sum(free for idx, free in rows if idx not in shared_gpu_ids and idx in reachable)
        return pool, model_bytes - on_cards * 1024 * 1024

    def _launch_host_shortfall_message(
        self,
        cmd: Iterable[str],
        detected_gpus: Iterable[tuple],
        env: Optional[Mapping[str, str]] = None,
        *,
        child_has_no_gpu: bool = False,
        avail_mib: Optional[int] = None,
        shared_gpu_ids: Iterable[int] = (),
        honor_opt_out: bool = True,
    ) -> Optional[str]:
        """Warning when the weights alone do not fit in free VRAM plus available RAM.

        Advisory only. Nothing here blocks a load: callers log this and launch anyway.

        Weights only, against the whole free pool, is a strict lower bound on what the
        launch must hold: the KV cache, projector, drafter and compute buffers all add
        to it, and a layer or device pin only narrows the VRAM actually reachable. So
        every term this leaves out moves the estimate down, never up, and no missing
        term can turn a quiet load into a warned one. That is what keeps the check
        a floor with no placement modelling to keep in step with llama.cpp.

        Read from the finished argv, so the model path is the one the child opens. An
        unsized model abstains rather than guessing, and so does an empty GPU pool,
        which means the probe threw. ``child_has_no_gpu`` is the launch reporting a
        placement it already knows reaches no card, rather than one it could not read:
        there the whole model is host-resident and takes no VRAM credit.

        UNSLOTH_ALLOW_HOST_OFFLOAD=1 abstains outright. DEPRECATED: it was the opt-out
        from the old refusal, so all it does now is silence the warning. Still honoured
        so existing scripts and docs keep working.

        ``honor_opt_out=False`` prices the shortfall with that variable ignored, for a
        caller that needs the FACT rather than the warning. Silencing the message must
        never disable the pageable-load override, which is what lets an oversized
        unmapped load finish at all: see ``_page_an_oversized_unmapped_load``.

        ``avail_mib`` overrides the host figure for a caller that runs before the resident
        owners are released; the launch reads what is available now. ``shared_gpu_ids``
        names Vulkan iGPUs whose reported free memory overlaps the host pool. Either
        reading can hold the remaining weights, but they must never be added together.
        """
        argv = [str(a) for a in cmd or ()]
        if not argv:
            return None
        model_path = _extra_args_device(argv, self._ARGV_MODEL)
        if not model_path:
            return None
        if honor_opt_out and self._host_offload_warning_opted_out():
            logger.info("UNSLOTH_ALLOW_HOST_OFFLOAD set: skipping the host-RAM preflight.")
            return None
        # rpc places layers on remote devices this cannot size, in either spelling
        _env = os.environ if env is None else env
        if (_extra_args_device(argv, self._ARGV_RPC) or "").strip() or str(
            _env.get("LLAMA_ARG_RPC", "") or ""
        ).strip():
            return None
        try:
            model_bytes = self._get_gguf_size_bytes(model_path)
        except Exception:
            return None
        gpus = [] if child_has_no_gpu else list(detected_gpus or ())
        # _get_gpu_memory swallows a failed probe as [], so an empty pool it did not
        # vouch for cannot be told from a host that has no gpu at all
        if not model_bytes or (not gpus and not child_has_no_gpu):
            return None
        shared = set(shared_gpu_ids or ())
        # Same lazy-read discount as the fit, or this rewrite undoes its pinned verdict.
        try:
            model_bytes = max(
                1,
                model_bytes
                - self._lazy_read_host_bytes(
                    model_path,
                    caps = self.probe_server_capabilities(argv[0]),
                    extra_args = argv,
                    env = _env,
                    auto_resolves_off = self._lazy_auto_resolves_off(
                        gpu_indices = [row[0] for row in gpus] if gpus else None,
                        detected_gpus = gpus,
                        shared_gpu_ids = shared,
                        is_vulkan_backend = bool(shared) or self._is_vulkan_backend(argv[0]),
                        probe_integrated_cuda = False,
                    ),
                ),
            )
        except Exception as e:  # noqa: BLE001 -- unverified discount: price the whole file
            logger.debug("Lazy-read discount skipped in the host preflight: %s", e)
        free_vram_mib = sum(max(0, row[1]) for row in gpus if row[0] not in shared)
        heap_free_mib, heap_bytes = self._shared_heap_budget(gpus, shared, model_bytes, argv, _env)
        offload_bytes = model_bytes - free_vram_mib * 1024 * 1024
        # A carve-out can satisfy the dedicated-VRAM shortfall. Skip card-resident
        # loads so a cgroup budget is never applied to a nonpositive need.
        if 0 < offload_bytes and heap_bytes <= heap_free_mib * 1024 * 1024:
            # A finite cgroup still limits host-backed GPU allocations.
            cgroup_mib = self._cgroup_available_memory_mib()
            if cgroup_mib is None:
                return None
            return self._apu_ram_shortfall_message(heap_bytes, cgroup_mib)
        return self._host_offload_shortfall_message(
            offload_bytes,
            self._available_system_memory_mib() if avail_mib is None else avail_mib,
        )

    @staticmethod
    def _apu_ram_shortfall_message(
        model_size_bytes: int,
        avail_mib: Optional[int],
        headroom_mib: int = _HOST_RAM_HEADROOM_MIB,
        *,
        part: str = "APU",
    ) -> Optional[str]:
        """On a unified-memory part, return a user-facing WARNING when the weights
        do not fit in available system RAM (else None). Weights only: KV/context
        auto-reduce, so counting them too would warn about loads that are fine.
        None avail (unknown RAM) never warns.

        Advisory, never a refusal: the load goes ahead and llama.cpp reports what
        actually happens rather than Studio pre-empting a failure it predicted.

        ``part`` names the hardware. An integrated CUDA SoC has the same shortfall and
        is not an APU, and the WSL hint cannot apply to a Jetson or a DGX Spark.
        """
        if avail_mib is None:
            return None
        need_mib = model_size_bytes / (1024 * 1024)
        if need_mib <= avail_mib - headroom_mib:
            return None
        free_hint = " (on WSL, raise the memory limit in .wslconfig)" if part == "APU" else ""
        return (
            f"This model needs about {need_mib / 1024:.0f} GB but only about "
            f"{avail_mib / 1024:.0f} GB of memory is available. On a unified-memory "
            f"{part} the weights load into system RAM, so the OS may stop the load. "
            "Loading anyway. If it does not complete, use a smaller or more "
            f"quantized GGUF, or free memory{free_hint}."
        )

    @staticmethod
    def _igpu_dedicated_memory_bytes(
        gpu_indices = None, *, ordinals_are_vulkan = False
    ) -> Optional[int]:
        """Memory dedicated to the selected integrated GPU, in bytes, or ``None``.

        Two readings, since no single source covers every platform: the DirectX
        registry (Windows, no vendor runtime needed, already parsed for the GPU
        inventory) and ``_rocm_selected_pool_mib`` (Linux, needs a ROCm torch,
        reports the carve-out as the device's total memory).

        ``None`` whenever the reading would be a guess; every caller treats absence
        as "say nothing".

        The registry goes first despite being the less authoritative answer, because
        it is the cheap one: a few ``winreg`` queries against a torch import plus a
        ``get_device_properties`` per device, which this file documents as leaking a
        ~700 MiB primary context. Off Windows it returns nothing instantly.
        """
        # Every adapter the registry lists, not only those with a readable allocation:
        # the count IS the attribution test. Filtering the unreadable ones out hid a
        # shared APU, leaving a discrete Radeon looking like the only candidate and its
        # fixed VRAM quoted as the APU's carve-out.
        answers: list[Optional[dict]] = []
        try:
            from utils.hardware.hardware import (
                _AMD_PCI_VENDOR_ID,
                _INTEL_PCI_VENDOR_ID,
                _windows_amd_adapter_records_by_luid,
            )
            for vendor_id in (_AMD_PCI_VENDOR_ID, _INTEL_PCI_VENDOR_ID):
                answers.append(
                    _windows_amd_adapter_records_by_luid(vendor_id, distinguish_failure = True)
                )
        except Exception:
            answers = []
        if any(answer is not None for answer in answers):
            # A vendor answered, so this is Windows. A vendor that could not be read
            # leaves the inventory incomplete, and an incomplete inventory cannot call
            # the adapter it did see the only one, so fail closed.
            if any(answer is None for answer in answers):
                return None
            adapters = sum(len(answer) for answer in answers)
            # AMD only, though Intel is counted. On an APU the DirectX value is the
            # firmware carve-out; on Intel UMA it is a small dedicated block beside
            # memory handed out dynamically, so quoting it would advise a setting that
            # may not exist and promise residency it cannot deliver.
            amd_sizes = [
                int(record["dedicated_memory_bytes"])
                for record in (answers[0] or {}).values()
                if record.get("dedicated_memory_bytes")
            ]
            if adapters == 1 and len(amd_sizes) == 1 and amd_sizes[0] > 0:
                return amd_sizes[0]
            if adapters:
                # Two adapters need the inventory's LUID-to-device join; attributing
                # the wrong one would advise about the wrong GPU. An APU paired with a
                # discrete Radeon lands here, and silence is right.
                return None
        if ordinals_are_vulkan:
            # No Linux reading for a Vulkan launch, for two reasons. gpu_indices holds
            # VULKAN ordinals while _rocm_selected_pool_mib compares PHYSICAL HIP ids,
            # and nothing in the Vulkan inventory carries a HIP id to join on, so the
            # reading could land on a device this launch never touches. And it is not
            # free: it creates a HIP primary context in THIS process (~800 MiB) out of
            # the very pool it would then call too small.
            return None
        pool_mib = LlamaCppBackend._rocm_selected_pool_mib(gpu_indices)
        return int(pool_mib) * 1024 * 1024 if pool_mib and pool_mib > 0 else None

    @staticmethod
    def _igpu_carveout_ladder_gb(cap_gb: float) -> list[int]:
        """Plausible dedicated-GPU-memory sizes up to ``cap_gb``, ascending.

        Firmware and driver panels offer a menu, not a slider, built from powers of
        two and their halves (…16, 24, 32, 48, 64, 96, 128…). Generated rather than
        tabulated so an unusually large machine still gets a suggestion; the caller
        picks the smallest entry that fits, so an entry the user's firmware lacks is
        a recommendation one notch off, not a wrong one.
        """
        # A `while` on the model-load path: the caller's try/except catches a raise,
        # not a hang. A non-finite cap makes the condition permanently true, so reject
        # it before the loop.
        if not isinstance(cap_gb, (int, float)) or not math.isfinite(cap_gb):
            return []
        rungs: set[int] = set()
        # From 1 GB, not 4: an APU on its automatic setting reports a few hundred
        # megabytes, and starting at 4 took two more gigabytes from the host than the
        # smallest-setting-that-fits rule needed.
        step = 1
        while step <= cap_gb:
            rungs.add(step)
            if step * 1.5 <= cap_gb:
                rungs.add(int(step * 1.5))
            step *= 2
        return sorted(rungs)

    @staticmethod
    def _igpu_carveout_advice(
        model_size_bytes: Optional[int],
        carve_out_bytes: Optional[int],
        host_total_bytes: Optional[int],
        *,
        is_igpu: bool,
        min_host_gb: int = _CARVEOUT_ADVICE_MIN_HOST_GB,
        host_fraction: float = _CARVEOUT_ADVICE_HOST_FRACTION,
        nominal_slack_gb: float = _CARVEOUT_NOMINAL_SLACK_GB,
    ) -> Optional[dict]:
        """Advice payload when an integrated GPU's dedicated memory is too small to
        hold this model's weights, else ``None``.

        Weights spilling out of the allocation run from shared system memory, which
        is markedly slower. Raising the allocation is a firmware/driver-panel setting
        only the user can make, so this only ever advises.

        Silent on a discrete GPU (``is_igpu`` false): that allocation is fixed
        silicon. Nothing is hardcoded to one machine -- the ceiling comes from what
        this host has, so a 32 GB laptop and a 512 GB workstation are each sized to
        themselves, and a model too large for ANY allocation here gets no advice.
        """
        if not is_igpu:
            return None
        # Strictly positive, not merely truthy: -1 is truthy, so a bare falsiness test
        # would carry a nonsense driver reading into confident wrong advice.
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for value in (model_size_bytes, carve_out_bytes, host_total_bytes)
        ):
            return None
        if model_size_bytes <= carve_out_bytes:
            return None  # already fits: nothing to advise

        gb = float(1024**3)
        # Windows subtracts the carve-out from the RAM it reports, so the machine has
        # both; reading one under-counts by exactly the amount this advice is about.
        machine_gb = (host_total_bytes + carve_out_bytes) / gb
        need_gb = model_size_bytes / gb
        current_gb = carve_out_bytes / gb

        # Whatever is suggested, the rest of the system still has to run.
        reserve_gb = max(float(min_host_gb), machine_gb * host_fraction)
        cap_gb = machine_gb - reserve_gb
        if need_gb > cap_gb:
            # No allocation this machine can offer holds the weights, so advice would
            # be something the user cannot act on.
            return None

        suggested = next(
            (rung for rung in LlamaCppBackend._igpu_carveout_ladder_gb(cap_gb) if rung >= need_gb),
            None,
        )
        # Not `<= current_gb`: the reading is the pool the driver kept, so the rung the
        # user is already on sits just above it. See _CARVEOUT_NOMINAL_SLACK_GB.
        if suggested is None or suggested <= current_gb + nominal_slack_gb:
            return None

        return {
            "current_gb": round(current_gb, 1),
            "needed_gb": round(need_gb, 1),
            "suggested_gb": int(suggested),
            "machine_gb": round(machine_gb, 1),
            "host_left_gb": round(machine_gb - suggested, 1),
        }

    def _record_carveout_advice(
        self,
        gpu_indices,
        need_bytes,
        *,
        is_vulkan_backend = False,
        shared_gpu_ids = None,
        detected_gpus = None,
        target_unknown = False,
        forced_cpu = False,
    ) -> None:
        """Work out whether this load is worth advising about, and stash the result.

        Advisory only, and never raises: the launch decides whether to spill the
        weights into shared memory, this only says whether the user could stop the
        spill existing at all, and an advisory must not break a model load.

        Tests are ordered cheapest first because this runs on every load, including
        the Vulkan ones that skip the managed-memory branch above. The allocation
        reading is arithmetic over a registry query; the integrated-GPU probe behind
        it imports torch and reads device properties, so the common load whose model
        fits pays only the cheap half.

        ``forced_cpu``: the architecture gate emptied the pool and the env block below
        masks every device away, so no allocation would hold a single weight. Priced
        before that mask is written, so without this it would offer exactly that.

        ``target_unknown``: the cache tuning's test, borrowed. With no ``gpu_ids`` a
        user ``--device`` (or ``LLAMA_ARG_DEVICE``) survives into the child and wins
        last-wins over the generated pin, so the placement this would advise about is
        not the one the child gets. Decline rather than re-derive it from argv.
        """
        self._last_carveout_advice = None
        try:
            if not need_bytes or target_unknown or forced_cpu:
                return
            # Gated HERE, not beside the ROCm gate below: on Vulkan gpu_indices holds
            # VULKAN ORDINALS, and _amd_apu_wants_unified_memory would read them as
            # physical HIP ids, which on a mixed host advises about an integrated GPU
            # the model is not using or hides advice that was valid. This branch is
            # also free (a set test against the planner's shared_gpu_ids, no torch), so
            # a dGPU-only Vulkan launch returns before the allocation reading.
            if is_vulkan_backend and not self._offload_target_shares_system_memory(
                is_vulkan_backend = True,
                shared_gpu_ids = shared_gpu_ids,
                detected_gpus = detected_gpus,
                gpu_indices = gpu_indices,
            ):
                return
            carve_out = self._igpu_dedicated_memory_bytes(
                gpu_indices, ordinals_are_vulkan = is_vulkan_backend
            )
            if not carve_out or need_bytes <= carve_out:
                return  # fits, or nothing to compare it against
            total_mib = self._total_system_memory_mib()
            advice = self._igpu_carveout_advice(
                need_bytes,
                carve_out,
                int(total_mib) * 1024 * 1024 if total_mib else None,
                is_igpu = True,
            )
            if advice is None:
                return
            # Only now, with a shortfall confirmed and a followable suggestion, is the
            # probe worth paying for. AMD only: a CUDA integrated part has no readable
            # allocation on either branch above, so carve_out is already None and we
            # returned, and calling _integrated_cuda_unified_memory anyway would create
            # a CUDA primary context per device for nothing. Not on Vulkan either: that
            # launch was classified above, in the index space it actually uses.
            if not is_vulkan_backend and not self._amd_apu_wants_unified_memory(gpu_indices):
                return
            # Asked last, so the common path never pays a database round trip.
            from utils.igpu_carveout_notice_settings import notice_already_dismissed

            if notice_already_dismissed(advice.get("current_gb")):
                return
            advice["message"] = self._igpu_carveout_advice_message(advice)
            self._last_carveout_advice = advice
            logger.info(
                "Integrated GPU has %.0f GB dedicated but this model needs about "
                "%.0f GB; suggesting %d GB.",
                advice["current_gb"],
                advice["needed_gb"],
                advice["suggested_gb"],
            )
        except Exception:
            logger.debug("Carve-out advice declined", exc_info = True)
            self._last_carveout_advice = None

    @property
    def last_carveout_advice(self) -> Optional[dict]:
        """Advice from the most recent load, or None. Read by the route."""
        return getattr(self, "_last_carveout_advice", None)

    @staticmethod
    def _fmt_gb(value: float) -> str:
        """A GB quantity as the user should read it: whole numbers above 10 GB, one
        decimal below, so an APU's few-hundred-megabyte automatic allocation does not
        print as "only about 0 GB is allocated" and read like a bug.
        """
        if value < 10:
            return f"{value:.1f}".rstrip("0").rstrip(".")
        return f"{value:.0f}"

    @staticmethod
    def _igpu_carveout_advice_message(advice: dict) -> str:
        """The advice as user-facing prose: two sentences, because it is a toast.

        Names no vendor, menu or key: the control is firmware on one machine and a
        driver panel on the next, and a confident wrong instruction costs the user
        more than a neutral one.

        Length is a correctness constraint, not a preference, for the reason
        xet_progress_notice.ts records: a toast tall enough to cover the controls
        under it takes them away for as long as it is up. So it carries the four
        numbers that make the advice actionable -- needed, allocated, suggested, left
        for the system -- and stops.
        """
        fmt = LlamaCppBackend._fmt_gb
        return (
            f"Weights need about {fmt(advice['needed_gb'])} GB but only "
            f"{fmt(advice['current_gb'])} GB is allocated to the integrated GPU, so the rest "
            f"runs from slower shared memory. Raising it to {advice['suggested_gb']} GB in "
            "your firmware or GPU control panel leaves about "
            f"{fmt(advice['host_left_gb'])} GB for the system."
        )

    @staticmethod
    def _host_offload_shortfall_message(
        offload_bytes: int,
        avail_mib: Optional[int],
        headroom_mib: int = _HOST_RAM_HEADROOM_MIB,
    ) -> Optional[str]:
        """On a discrete GPU, return a user-facing WARNING when the part of a load
        that misses VRAM does not fit in available system RAM (else None). Priced
        against free VRAM with no margin subtracted, so it under-states the spill and
        only an unambiguous shortfall warns. None avail (unknown RAM), and anything
        VRAM-resident, never warn.

        Advisory, never a refusal. The spill is mmap'd, so the weights page in from disk
        rather than failing: off fast storage that is a deliberate way to run a quant
        larger than the machine's fast memory, and this check cannot tell how fast the
        backing store is. It reports the cost; the user decides.
        """
        if offload_bytes <= 0 or avail_mib is None:
            return None
        need_mib = offload_bytes / (1024 * 1024)
        if need_mib <= avail_mib - headroom_mib:
            return None
        # need up, usable down, so the printed pair cannot round into a tie
        need_gb = math.ceil(need_mib / 1024)
        usable_gb = math.floor(max(0, avail_mib - headroom_mib) / 1024)
        return (
            f"About {need_gb} GB of this model does not fit in GPU memory and would run "
            f"from system RAM. Only about {avail_mib / 1024:.0f} GB is available and "
            f"{headroom_mib / 1024:.0f} GB of that is kept free for the rest of the "
            f"system, leaving about {usable_gb} GB usable. The weights are memory-mapped, "
            "so the machine pages them in from disk as it runs. Loading anyway: expect "
            "slow generation, and slower still on a hard disk than on an SSD. A smaller "
            "or more quantized GGUF, or more free memory, would run faster."
        )

    @staticmethod
    def _host_offload_warning_opted_out() -> bool:
        """Whether UNSLOTH_ALLOW_HOST_OFFLOAD silences the host-RAM warning.

        The user's own opt-out, so it reads the REAL environment rather than the
        curated child env. DEPRECATED and warning-scoped: it was the escape from a
        refusal that no longer exists, so all it may do is suppress a message. It must
        never reach a decision that changes what the child runs.
        """
        return os.environ.get("UNSLOTH_ALLOW_HOST_OFFLOAD", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
