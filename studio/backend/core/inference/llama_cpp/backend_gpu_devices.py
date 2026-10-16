# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: GPU allocation, device enumeration and visibility."""

import bisect
import ctypes
import math
import os
import re
import struct
import subprocess
import sys
import threading

from pathlib import Path
from typing import (
    Callable,
    Iterable,
    Mapping,
    Optional,
)
from core.inference.llama_server_args import (
    _GPU_LAYER_FLAGS,
    fit_is_effectively_on,
    fit_is_enabled_in,
    parse_gpu_layers_override,
    resolve_effective_load_state,
)
from utils.child_stdio import utf8_child_env
from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)

from .binary_runtime import (
    _DEVICE_VISIBILITY_ENV,
    _MISSING,
    _NEEDS_ROCM_CLASSIFICATION,
    _SELF_EVIDENTLY_DISCRETE,
    _apply_igpu_host_reserve_mib,
    _llama_lib_dir,
    _load_probe_memo_get,
    _load_probe_memo_put,
    _parse_listed_devices,
)
from .draft_args import (
    _CPU_DEVICE_VALUES,
    _count_subsequence,
    _device_selection_is_cpu,
    _extra_args_device,
    _extra_args_main_device,
    _split_mode_confines_to_one_device,
    _without_subsequence,
)
from .gguf_cache import _SHARD_FULL_RE
from .launch_flags import (
    _args_place_tensors_on_cpu,
    _env_places_tensors_on_cpu,
)
from .log import logger
from .memory_policy import _extra_args_set_any_flag
from .moe_cache import (
    _LAZY_READ_TENSORS,
    _NEXTN_SKIPPED_WITHOUT_MTP_ARCHS,
    _effective_lazy_mode,
    _lazy_read_bytes,
)

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _GpuDevicesMixin:
    # ── GPU allocation ────────────────────────────────────────────

    @staticmethod
    def _get_gguf_size_bytes(model_path: str) -> int:
        """Total GGUF size in bytes, including split shards."""
        main = Path(model_path)
        total = main.stat().st_size

        # Check for split shards (e.g. model-00001-of-00003.gguf)
        m = _SHARD_FULL_RE.match(main.name)
        if m:
            prefix, _, num_total = m.group(1), m.group(2), m.group(3)
            sibling_pat = re.compile(
                r"^" + re.escape(prefix) + r"-\d{5}-of-" + re.escape(num_total) + r"\.gguf$",
                re.IGNORECASE,
            )
            for sibling in main.parent.iterdir():
                if sibling != main and sibling_pat.match(sibling.name):
                    total += sibling.stat().st_size

        return total

    @staticmethod
    def _gguf_scan_tensor_bytes(
        path: str, also: Optional[Callable[[str], bool]] = None
    ) -> "tuple[Optional[str], dict[str, int], int]":
        """(architecture, lazy-read candidate bytes by name, routed expert bytes) from one
        GGUF header. Streamed: ``GGUFReader`` materialises the vocabulary under the load
        lock. A quant type gguf predates is sized from the next tensor's offset. Names
        ``also`` accepts are kept by name too, routed experts included."""
        from gguf.constants import GGML_QUANT_SIZES
        from core.inference.offload_layout import _BLOCK_RE, _MOE_EXPERT_RE

        wanted = {name.encode() for names in _LAZY_READ_TENSORS.values() for name in names}
        architecture: Optional[str] = None
        alignment = 32  # GGUF's default when general.alignment is absent
        named: dict[str, int] = {}
        experts = 0
        offsets: list[int] = []
        unsized: list[tuple[str, int]] = []
        with open(path, "rb") as f:
            if struct.unpack("<I", f.read(4))[0] != 0x46554747:  # b"GGUF"
                return None, {}, 0
            # Version 1 used 32-bit counts and lengths, which this layout misreads.
            if struct.unpack("<I", f.read(4))[0] < 2:
                return None, {}, 0
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            for _ in range(n_kv):
                key = f.read(struct.unpack("<Q", f.read(8))[0])
                value_type = struct.unpack("<I", f.read(4))[0]
                if key == b"general.architecture" and value_type == 8:
                    raw = f.read(struct.unpack("<Q", f.read(8))[0])
                    architecture = raw.decode("utf-8", "replace")
                    continue
                if key == b"general.alignment" and value_type == 4:
                    alignment = max(1, struct.unpack("<I", f.read(4))[0])
                    continue
                LlamaCppBackend._gguf_skip_value(f, value_type)
            for _ in range(n_tensors):
                raw_name = f.read(struct.unpack("<Q", f.read(8))[0])
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
                ggml_type = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]
                offsets.append(offset)
                name = raw_name.decode("utf-8", "replace")
                block_match = _BLOCK_RE.match(name)
                is_expert = bool(block_match and _MOE_EXPERT_RE.match(block_match.group(2)))
                kept = also is not None and also(name)
                if not is_expert and raw_name not in wanted and not kept:
                    continue
                block = GGML_QUANT_SIZES.get(ggml_type)
                if block is None:
                    unsized.append((name, offset))
                    continue
                size = math.prod(dims) // block[0] * block[1]
                if is_expert:
                    experts += size
                if not is_expert or kept:
                    named[name] = size
            if unsized:
                data_start = -(-f.tell() // alignment) * alignment
                data_end = max(0, os.path.getsize(path) - data_start)
                ordered = sorted(offsets)
                for name, offset in unsized:
                    later = ordered[bisect.bisect_right(ordered, offset) :]
                    size = max(0, (later[0] if later else data_end) - offset)
                    block_match = _BLOCK_RE.match(name)
                    is_expert = bool(block_match and _MOE_EXPERT_RE.match(block_match.group(2)))
                    if is_expert:
                        experts += size
                    if not is_expert or (also is not None and also(name)):
                        named[name] = size
        return architecture, named, experts

    def _gguf_tensor_scan(
        self, model_path: Optional[str]
    ) -> "Optional[tuple[Optional[str], dict[str, int], int]]":
        """``_gguf_scan_tensor_bytes`` over every shard, cached per file identity; None
        when a shard is missing or unreadable."""
        from core.inference.offload_layout import split_shard_paths

        if not model_path:
            return None
        paths = split_shard_paths(model_path) or [model_path]
        try:
            key = tuple((p, st.st_size, st.st_mtime_ns) for p in paths for st in (os.stat(p),))
        except OSError:
            return None
        cached = getattr(self, "_gguf_tensor_scan_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            architecture: Optional[str] = None
            named: dict[str, int] = {}
            experts = 0
            for p in paths:
                arch, part, part_experts = self._gguf_scan_tensor_bytes(p)
                # Shard 1 carries the metadata.
                architecture = architecture or arch
                for name, size in part.items():
                    named[name] = named.get(name, 0) + size
                experts += part_experts
            result = (architecture, named, experts)
        except Exception as e:  # unreadable or truncated: discount nothing
            logger.debug("GGUF tensor scan failed for %s (%s)", model_path, e)
            result = None
        self._gguf_tensor_scan_cache = (key, result)
        return result

    def _unloaded_nextn_bytes(self, model_path: Optional[str]) -> int:
        """Bytes of the embedded NextN/MTP blocks llama.cpp leaves unloaded without
        ``--spec-type draft-mtp`` (nextn_flags marks them TENSOR_SKIP unless load_mtp).
        Only for architectures measured to skip them; 0 when unknown or unreadable."""
        from core.inference.offload_layout import _BLOCK_RE, split_shard_paths

        nextn = int(getattr(self, "_nextn_predict_layers", None) or 0)
        n_blocks = int(getattr(self, "_n_layers", None) or 0)
        if (
            not model_path
            or nextn <= 0
            or n_blocks <= nextn
            or getattr(self, "_architecture", None) not in _NEXTN_SKIPPED_WITHOUT_MTP_ARCHS
        ):
            return 0
        first = n_blocks - nextn

        def in_head(name: str) -> bool:
            match = _BLOCK_RE.match(name)
            return bool(match) and int(match.group(1)) >= first

        try:
            return sum(
                size
                for p in (split_shard_paths(model_path) or [model_path])
                for name, size in self._gguf_scan_tensor_bytes(p, also = in_head)[1].items()
                if in_head(name)
            )
        except Exception as e:  # missing shard or truncated header: discount nothing
            logger.debug("NextN block scan failed for %s (%s)", model_path, e)
            return 0

    def _lazy_read_host_bytes(
        self,
        model_path: Optional[str],
        *,
        caps: Optional[Mapping[str, object]],
        extra_args: Optional[Iterable[str]] = None,
        env: Optional[Mapping[str, str]] = None,
        auto_resolves_off: bool = False,
    ) -> int:
        """Bytes llama.cpp leaves mmap'd from disk under every load mode (``_LAZY_READ_TENSORS``);
        zero without lazy-read support, under ``off``, or on an unreadable header.
        ``auto_resolves_off``: llama.cpp turns ``auto`` into ``off`` for these devices."""
        if not model_path or not (caps or {}).get("supports_lazy_mode"):
            return 0
        mode = _effective_lazy_mode(
            extra_args, env, flag = str((caps or {}).get("lazy_mode_flag") or "--lazy-mode")
        )
        if mode == "off" or (mode == "auto" and auto_resolves_off):
            return 0
        scan = self._gguf_tensor_scan(model_path)
        if scan is None:
            return 0
        architecture, named, _experts = scan
        return _lazy_read_bytes(architecture, named, mode)

    def _lazy_auto_resolves_off(
        self,
        *,
        gpu_indices: Optional[Iterable[int]],
        detected_gpus: Optional[Iterable[tuple]],
        shared_gpu_ids: Optional[Iterable[int]],
        is_vulkan_backend: bool,
        probe_integrated_cuda: bool = True,
    ) -> bool:
        """Whether llama.cpp resolves ``--lazy-mode auto`` to ``off`` for this child: a model
        device without mmap support (ggml-org/llama.cpp#28160), i.e. every selected device
        integrated, since the default list holds iGPUs only when no discrete GPU is seen.
        ``probe_integrated_cuda`` False uses only a cached SoC answer (no new context)."""
        rows = [(row[0], row[1]) for row in (detected_gpus or ())]
        if self._offload_target_shares_system_memory(
            is_vulkan_backend = is_vulkan_backend,
            shared_gpu_ids = shared_gpu_ids,
            detected_gpus = rows,
            gpu_indices = gpu_indices,
        ):
            return True
        if is_vulkan_backend:
            return False
        if not probe_integrated_cuda and not self._integrated_cuda_probe_is_free():
            return False
        return self._integrated_cuda_selection_is_all_shared(
            list(gpu_indices) if gpu_indices else None
        )

    @staticmethod
    def _installed_ggml_backends(binary: Optional[str] = None) -> frozenset[str]:
        """Backend libraries shipped beside llama-server."""
        binary = binary or LlamaCppBackend._find_llama_server_binary()
        if not binary:
            return frozenset()
        try:
            files = tuple(path.name for path in _llama_lib_dir(binary).iterdir() if path.is_file())
        except OSError:
            return frozenset()
        backends = set()
        for backend in ("base", "cpu", "cuda", "hip", "vulkan"):
            stem = f"ggml-{backend}.dll" if sys.platform == "win32" else f"libggml-{backend}.so"
            if any(name == stem or name.startswith(stem + ".") for name in files):
                backends.add(backend)
        return frozenset(backends)

    def _record_memory_state(
        self,
        argv,
        env = None,
    ) -> None:
        """Record the ``(mlock, reserves_ram)`` a launch really runs with, and
        whether it streams, from ONE parse. A dozen paths record a state; a
        second parse at each of them would drift from this one."""
        mlock, reserves_ram, direct_io = resolve_effective_load_state(argv, env)
        self._memory_state = (mlock, reserves_ram)
        self._memory_direct_io = direct_io

    def _drop_managed_dio(
        self,
        argv,
        reason: str,
        clear_record: bool = True,
    ):
        """Take the no-reserve DirectIO pair back out of an argv whose placement
        is no longer the confirmed full offload it was chosen for.

        ``use_mmap`` covers mmap / mmap+mlock / auto only, so under dio every
        layer llama.cpp leaves on the CPU is read into an allocated buffer
        instead of a mapping the kernel can page: the reservation the setting
        exists to avoid. Only Unsloth's own tokens; a user's --load-mode is
        theirs, exactly as with the fit's mode.

        ``clear_record`` False when the caller strips a COPY and `cmd` still
        carries the pair, which is the arm the fit's own mode already takes for
        the same reason: the arch-crash and CPU rungs respawn from `cmd`, and
        forgetting the tokens here leaves their own strip a no-op on an argv that
        still has them. The applicable flag goes either way, because it describes
        the child about to spawn, and the `_mem_policy_for_cmd` snapshot restores
        `cmd`'s value for the rung that goes back to it.
        """
        if not self._memory_dio_flags:
            self._memory_dio_applicable = False
            return argv
        # The record says a pair exists SOMEWHERE, not in THIS argv: a rung that strips a
        # copy keeps it for `cmd`, so strip only while this argv holds more than the user wrote.
        if _count_subsequence(list(argv), self._memory_dio_flags) <= _count_subsequence(
            list(getattr(self, "_memory_dio_user_tokens", []) or []), self._memory_dio_flags
        ):
            if clear_record:
                self._memory_dio_flags = []
            self._memory_dio_applicable = False
            return argv
        stripped = _without_subsequence(argv, self._memory_dio_flags)
        if clear_record:
            self._memory_dio_flags = []
        self._memory_dio_applicable = False
        # The pair may have been the only mark, and a child equal to an unmanaged one must
        # not be torn down when the toggles go off.
        self._memory_policy_active = self._memory_policy_extras_touched
        logger.info("Model Memory: dropping the managed --load-mode dio; %s", reason)
        return stripped

    @staticmethod
    def _vulkan_offload_is_discrete(binary: Optional[str], gpu_indices = None) -> bool:
        """True only when the probe ANSWERED and every device in play is discrete.

        ``_vulkan_targets_are_igpus`` folds "probe failed" into "not an iGPU",
        which is the safe direction for its own caller (that one only skips a
        page-lock) and the wrong one here: an iGPU's VRAM is system RAM, so
        confirming an offload we could not read hands DirectIO to weights that
        are really host-backed. Absence of an answer declines.
        """
        try:
            rows = LlamaCppBackend._run_vulkan_probe(binary)
        except Exception:
            return False
        if not rows:
            return False
        wanted = set(gpu_indices) if gpu_indices else None
        selected = [r for r in rows if wanted is None or r["index"] in wanted]
        if not selected:
            return False
        # EVERY ordinal, not merely some: a partial intersection confirmed an unclassified one.
        if wanted is not None and wanted - {r["index"] for r in selected}:
            return False
        # The probe reports all-False when the type query fails, so "not integrated" and
        # "unread" share a value. Right for the page-lock caller, wrong here.
        if not all(r.get("type_known") for r in selected):
            return False
        return not any(r["is_igpu"] for r in selected)

    @classmethod
    def _gpu_offload_confirmed(
        cls,
        binary: Optional[str],
        env: Optional[Mapping[str, str]],
        gpu_indices,
        host_resident: bool,
        dio_possible: bool,
        extra_args = None,
    ) -> bool:
        """Whether this launch really puts the weights on a discrete GPU.

        The one predicate behind managed DirectIO. Confirming wrongly is worse than
        not acting: DirectIO over host-resident weights replaces a pageable mapping
        with a model-sized allocated buffer. Two questions, and anything unanswered
        declines:

        1. Will these devices exist for the child? ``--list-devices`` is the loader's
           own verdict, so a missing runtime, a disabled plugin, a wrong-vendor plugin
           and a ``GGML_BACKEND_PATH`` plugin all resolve without reading a filename.
        2. Is the device discrete? ggml does not report it. Vulkan has the separate
           probe, CUDA and ROCm are classified upstream by `_weights_in_host_memory`,
           and a SYCL or OpenCL id can be an integrated GPU nothing here recognises.

        ``dio_possible`` is the platform-and-build gate, NOT the toggle: off Windows
        no answer can reach a flag, so the probes are not worth spawning. Gating on
        the toggle would be wrong, because the verdict is recorded as this launch's
        placement and compared against a later save.
        """
        if not dio_possible or host_resident:
            return False
        devices = cls._enumerated_gpu_devices(binary, env)
        # A pass-through `--device` is appended last and decides placement, so the
        # auto-selected ordinals are not what runs. Its values are ggml ids.
        override = cls._effective_device_ids(extra_args, env)
        if override is not None:
            if any(d.lower() in _CPU_DEVICE_VALUES for d in override):
                return False
            listed = {d.lower() for d in (devices or [])}
            if not override or not {d.lower() for d in override} <= listed:
                return False
            gpu_indices = [
                cls._device_ordinal(d) for d in override if cls._device_ordinal(d) is not None
            ]
            devices = [d for d in (devices or []) if d.lower() in {o.lower() for o in override}]
        if not cls._offload_devices_are_live(devices, gpu_indices):
            return False
        # EVERY selected device: returning the first kind's verdict confirmed a mixed set
        # that still had weights on an integrated GPU.
        selected = cls._selected_devices(devices, gpu_indices)
        backends = {cls._device_backend(d) for d in selected}
        if backends - {"vulkan"} - _SELF_EVIDENTLY_DISCRETE - _NEEDS_ROCM_CLASSIFICATION:
            return False
        # An AMD id only counts once something has actually looked at the device.
        if (backends & _NEEDS_ROCM_CLASSIFICATION) and not cls._rocm_classification_answered():
            return False
        if "vulkan" in backends:
            # Only the VULKAN ordinals: a CUDA target's 0 otherwise satisfies Vulkan row 0.
            vulkan_ordinals = [
                cls._device_ordinal(d) for d in selected if cls._device_backend(d) == "vulkan"
            ]
            if any(o is None for o in vulkan_ordinals):
                return False
            return cls._vulkan_offload_is_discrete(binary, vulkan_ordinals)
        return True

    @staticmethod
    def _device_ordinal(device: str) -> Optional[int]:
        """The trailing ordinal of a ggml device id, or None."""
        match = re.search(r"(\d+)$", device)
        return int(match.group(1)) if match else None

    @staticmethod
    def _compact_ordinals(
        gpu_indices,
        env: Optional[Mapping[str, str]],
        is_vulkan = False,
    ):
        """Physical ordinals translated into the child's COMPACT space.

        A mask reindexes survivors from 0, so the adapter this launch calls physical 1
        is ``CUDA0`` to the child, and comparing the two rejected a narrowed launch that
        was fully offloaded. The mask lists survivors in order, so a physical id's
        position in it IS its compact ordinal. Unchanged with no mask, or one we cannot
        map.
        """
        if not gpu_indices or not env or is_vulkan:
            return gpu_indices
        # GGML_VK_VISIBLE_DEVICES is deliberately absent: ggml passes it through, so a
        # Vulkan ordinal is ALREADY compact and translating it again picked another row.
        for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            raw = env.get(name)
            if not raw or not str(raw).strip():
                continue
            try:
                order = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
            except ValueError:
                return gpu_indices
            if not order:
                return gpu_indices
            try:
                mapped = [order.index(int(i)) for i in gpu_indices if int(i) in order]
            except (TypeError, ValueError):
                return gpu_indices
            return mapped or gpu_indices
        return gpu_indices

    @staticmethod
    def _physical_ordinals(
        compact,
        env: Optional[Mapping[str, str]],
        is_vulkan = False,
    ):
        """The inverse of `_compact_ordinals`: compact ordinals back to physical ids.

        `--device` names compact ids, but `_amd_apu_wants_unified_memory` is documented
        to take PHYSICAL ones, so a selection that arrives compact has to be mapped
        back before residency can classify it.
        """
        if not compact or not env or is_vulkan:
            return compact
        for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            raw = env.get(name)
            if not raw or not str(raw).strip():
                continue
            try:
                order = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
            except ValueError:
                return compact
            if not order:
                return compact
            try:
                mapped = [order[int(i)] for i in compact if 0 <= int(i) < len(order)]
            except (TypeError, ValueError):
                return compact
            return mapped or compact
        return compact

    @classmethod
    def _effective_main_gpu(cls, extra_args, env) -> Optional[int]:
        """The sole device a ``--split-mode none`` launch puts every weight on.

        llama.cpp keeps only ``devices[main_gpu]``, and ``--main-gpu`` can name a device
        OUTSIDE the automatic plan, so it replaces the selection rather than narrowing
        it. Missing that confirmed a discrete card while the child put the weights on a
        shared-memory iGPU.
        """
        if not _split_mode_confines_to_one_device(extra_args, env):
            return None
        value = _extra_args_device(extra_args, {"--main-gpu", "-mg"})
        if value is None and env:
            value = env.get("LLAMA_ARG_MAIN_GPU")
        if value is None:
            return None
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    @classmethod
    def _effective_gpu_selection(
        cls,
        binary,
        env,
        gpu_indices,
        extra_args,
        dio_possible = True,
        is_vulkan = False,
    ):
        """The ordinals the child will really use.

        A pass-through ``--device`` is appended last, so it beats the auto selection.
        Resolved once and handed to every consumer, because the host-residency verdict
        and the confirmation must be about the SAME devices: computed apart, an override
        onto a unified-memory APU was priced against the discrete card it replaced.

        Returns ``(physical, compact)``, because the consumers need the SAME selection
        in DIFFERENT spaces and collapsing them broke one or the other:
        `_amd_apu_wants_unified_memory` is documented to take physical ids, while
        `--list-devices` reports the child's compact ordinals.

        Unchanged when there is no override, when it names no GPU, or when it names a
        device the build never enumerated; the confirmation declines those on their own
        terms, and answering them here would hide the reason.
        """
        # Gated like the confirmation: with no dio decision to follow, the enumeration
        # cannot change an outcome and is not worth a subprocess.
        if not dio_possible:
            return gpu_indices, gpu_indices
        override = cls._effective_device_ids(extra_args, env)
        if override and not any(d.lower() in _CPU_DEVICE_VALUES for d in override):
            listed = {d.lower() for d in (cls._enumerated_gpu_devices(binary, env) or [])}
            if {d.lower() for d in override} <= listed:
                # `--device` names compact ids, so physical is the derived form.
                ordinals = [cls._device_ordinal(d) for d in override]
                compact = [o for o in ordinals if o is not None]
                if not compact:
                    return gpu_indices, cls._compact_ordinals(gpu_indices, env, is_vulkan)
                return cls._physical_ordinals(compact, env, is_vulkan), compact
            return gpu_indices, cls._compact_ordinals(gpu_indices, env, is_vulkan)
        # Otherwise PHYSICAL, and `--main-gpu` under `-sm none` replaces the selection.
        main_gpu = cls._effective_main_gpu(extra_args, env)
        if main_gpu is not None:
            # `--main-gpu` indexes the FILTERED list, so it too arrives compact.
            return cls._physical_ordinals([main_gpu], env, is_vulkan), [main_gpu]
        return gpu_indices, cls._compact_ordinals(gpu_indices, env, is_vulkan)

    @classmethod
    def _effective_device_ids(cls, extra_args, env) -> Optional[list[str]]:
        """The explicit ``--device`` selection, or None when there is none.

        Same precedence llama.cpp uses and `_device_selection_is_cpu` already mirrors:
        the last argv value, then ``LLAMA_ARG_DEVICE``. The values are ggml device ids,
        the same namespace ``--list-devices`` prints, so they compare directly.
        """
        value = _extra_args_main_device(extra_args)
        if value is None and env:
            value = env.get("LLAMA_ARG_DEVICE")
        if value is None:
            return None
        return [d.strip() for d in str(value).split(",") if d.strip()]

    @staticmethod
    def _selected_devices(devices: Optional[list[str]], gpu_indices) -> list[str]:
        """The enumerated GPU devices this launch will use. ggml ids are
        ``<Backend><ordinal>``; the CPU device is not a placement target, and with no
        pin every GPU device is in play."""
        if not devices:
            return []
        gpu = [d for d in devices if not d.upper().startswith("CPU")]
        if not gpu_indices:
            return gpu
        wanted = {int(i) for i in gpu_indices}
        picked = []
        for device in gpu:
            match = re.search(r"(\d+)$", device)
            if match and int(match.group(1)) in wanted:
                picked.append(device)
        return picked

    @classmethod
    def _offload_devices_are_live(cls, devices: Optional[list[str]], gpu_indices) -> bool:
        """Whether every device this launch pins was enumerated by the build.

        Replaces "do the host's GPUs exist": a device the host reports but the child
        cannot open is the case that confirmed an offload which never happened.

        By ORDINAL COVERAGE, not device count: a multi-backend build can enumerate the
        same ordinal twice (``CUDA0`` and ``Vulkan0``), and counting ids rejected a
        pinned launch whose devices were all live.
        """
        selected = cls._selected_devices(devices, gpu_indices)
        if not selected:
            return False
        if not gpu_indices:
            return True
        covered = {cls._device_ordinal(d) for d in selected}
        return {int(i) for i in gpu_indices} <= covered

    @staticmethod
    def _device_backend(device: str) -> str:
        """The ggml backend behind a device id, lowercased. What `_is_vulkan_backend`
        and the plugin-root scan approximated from filenames: the build reports what it
        actually loaded, so an external ``GGML_BACKEND_PATH`` plugin names itself too."""
        return re.sub(r"\d+$", "", device).strip().lower()

    @classmethod
    def _devices_are_vulkan(cls, devices: Optional[list[str]], gpu_indices) -> bool:
        """Whether any device in play is a Vulkan one, read off the build's own ids."""
        return any(
            cls._device_backend(d) == "vulkan" for d in cls._selected_devices(devices, gpu_indices)
        )

    @classmethod
    def _enumerated_gpu_devices(
        cls,
        binary: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> Optional[list[str]]:
        """The ggml device ids ``--list-devices`` reports, or None for no usable answer.

        Tri-state like ``sd_cpp_accelerator_device_verdict``: ``[]`` is the build
        saying it has no devices, None is an older build rejecting the flag, a timeout
        or a crash. Both decline, but empty is a real answer and is not folded in.
        """
        binary = binary or cls._find_llama_server_binary()
        if not binary:
            return None
        key = (binary, cls._device_visibility_key(env))
        memo = _load_probe_memo_get("devices", key)
        if memo is not None:
            return None if memo is _MISSING else memo
        devices = cls._run_list_devices(binary, env)
        _load_probe_memo_put("devices", key, _MISSING if devices is None else devices)
        return devices

    @staticmethod
    def _device_visibility_key(env: Optional[Mapping[str, str]]):
        """What this environment lets the child see, as a memo key. A rung that masks
        an adapter has a different answer coming, and without this a narrowing onto a
        usable discrete GPU kept the original set's failure."""
        source = os.environ if env is None else env
        return tuple((name, source.get(name)) for name in _DEVICE_VISIBILITY_ENV)

    @staticmethod
    def _run_list_devices(
        binary: str, env: Optional[Mapping[str, str]] = None
    ) -> Optional[list[str]]:
        """``--list-devices`` once, parsed. The flag prints and exits, so this leaves no
        server behind; it is on the argv denylist for that reason, which does not apply
        to Unsloth calling it deliberately."""
        try:
            probe_env = LlamaCppBackend._llama_server_env_for_binary(binary)
        except Exception:
            probe_env = child_env_without_native_path_secret()
        # The probe must see what the CHILD sees: on Windows only
        # `_llama_server_env_for_binary` puts the managed venv's CUDA runtime on PATH, so
        # the caller's REMOVALS are replayed rather than its whole mapping.
        if env is not None:
            # BOTH directions: removals alone dropped a narrowing rung's new mask.
            for name in _DEVICE_VISIBILITY_ENV:
                if name in env:
                    probe_env[name] = env[name]
                else:
                    probe_env.pop(name, None)
            for name in list(probe_env):
                if name not in env and name in os.environ:
                    probe_env.pop(name, None)
        try:
            result = subprocess.run(
                [binary, "--list-devices"],
                capture_output = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 30,
                env = utf8_child_env(probe_env),
                **_windows_hidden_subprocess_kwargs(),
            )
        except Exception as e:
            logger.debug(f"llama-server --list-devices failed: {e}")
            return None
        if result.returncode != 0:
            logger.debug(f"llama-server --list-devices exited {result.returncode}")
            return None
        return _parse_listed_devices(result.stdout)

    @staticmethod
    def _is_vulkan_backend(binary: Optional[str] = None) -> bool:
        """True if the installed llama.cpp build is Vulkan-only.

        The official prebuilts are single-backend, so the Vulkan ggml lib next
        to llama-server identifies a Vulkan build. Keeps the free-memory probe
        and GPU pin in ggml's Vulkan device-index space. For a custom
        multi-backend build with a CUDA or HIP ggml lib alongside Vulkan, defer
        to that backend (torch-usable, better-understood probe/pin).
        """
        backends = LlamaCppBackend._installed_ggml_backends(binary)
        return "vulkan" in backends and not backends.intersection({"cuda", "hip"})

    _SYSMEM_FALLBACK_RISK: dict[Optional[str], bool] = {}

    @staticmethod
    def _sysmem_fallback_risk(binary: Optional[str] = None) -> bool:
        """True when an over-budget allocation would be absorbed by host RAM, not refused.

        Windows + CUDA build only. Discreteness needs no probe: integrated CUDA parts are
        Linux-only, and get_device_properties would pin a ~700 MiB CUDA context.
        """
        if sys.platform != "win32":
            return False
        if binary is None:
            try:
                binary = LlamaCppBackend._find_llama_server_binary()
            except Exception:  # noqa: BLE001 - classify with whatever the lib lookup resolves
                binary = None
        # Keyed by resolved path and build revision: runtimes switch, updates swap in place.
        key = LlamaCppBackend._binary_revision(binary) or binary
        cached = LlamaCppBackend._SYSMEM_FALLBACK_RISK.get(key)
        if cached is not None:
            return cached
        try:
            backends = LlamaCppBackend._installed_ggml_backends(binary)
        except Exception as e:  # noqa: BLE001 - an unreadable lib dir is not a load failure
            logger.debug("sysmem-fallback classification failed, keeping the base budget: %s", e)
            return False  # not cached: a transient read error must not pin False forever
        if not backends:
            return False  # an unreadable dir scans empty; same rule, not cached
        risk = "cuda" in backends
        LlamaCppBackend._SYSMEM_FALLBACK_RISK[key] = risk
        return risk

    @staticmethod
    def _active_gpu_visibility_mask() -> Optional[str]:
        """The raw visibility mask in force (HIP, then ROCR, then CUDA on ROCm; CUDA
        otherwise), or None when unset. Split out so a caller can tell "no mask" from
        "a mask I cannot map to physical ids"."""
        try:
            import torch

            # Same ROCm detection as _emit_child_gpu_visibility (AMD SDK wheels leave
            # version.hip unset but encode "rocm" in __version__). If the two
            # disagree an inherited ROCR mask reads back as "no mask" and the child's
            # new pin re-exposes the GPU it was hiding.
            is_rocm = LlamaCppBackend._torch_is_rocm(torch)
        except Exception:
            is_rocm = False
        if is_rocm:
            hip_v = os.environ.get("HIP_VISIBLE_DEVICES")
            # ROCR_VISIBLE_DEVICES is Linux-only: Windows HIP has no ROCr layer, so a
            # stray ROCR var there masks nothing and must not be read as the
            # ordinal->physical mapping (mirrors _emit_child_gpu_visibility).
            rocr_v = None if sys.platform == "win32" else os.environ.get("ROCR_VISIBLE_DEVICES")
            cvd = (
                hip_v
                if hip_v is not None
                else rocr_v
                if rocr_v is not None
                else os.environ.get("CUDA_VISIBLE_DEVICES")
            )
        else:
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        return cvd

    @staticmethod
    def _resolve_visible_physical_ids() -> Optional[list[int]]:
        """Physical GPU ids behind the active visibility mask (HIP/ROCR/CUDA on
        ROCm, CUDA otherwise). None when unset, [] for an empty mask. Shared by the
        APU / datacenter / free-memory probes so they agree on the mapping."""
        cvd = LlamaCppBackend._active_gpu_visibility_mask()
        if cvd is None:
            return None
        try:
            return [int(x.strip()) for x in cvd.split(",") if x.strip()]
        except ValueError:
            pass
        # A resolvable CUDA UUID mask, else the GPU auto-pick escapes the mask onto hidden cards (#8873).
        try:
            from utils.hardware.hardware import _get_parent_visible_gpu_spec
            spec = _get_parent_visible_gpu_spec()
        except Exception:
            return None
        return spec["numeric_ids"] if spec.get("raw") == cvd.strip() else None

    @staticmethod
    def _visibility_mask_is_unmappable() -> bool:
        """True when a mask IS set but does not resolve to physical ids: ROCr and CUDA
        accept UUID tokens (ROCR_VISIBLE_DEVICES="0,GPU-DEADBEEF..."), CUDA also MIG
        ids. Torch ordinals cannot be mapped back, so re-emitting them numerically
        would REPLACE the mask with ids the runtime resolves against the whole host,
        handing the child cards the parent hid, including one a gate just dropped.
        Callers that would pin fail open. False when unset and for an empty mask."""
        return (
            LlamaCppBackend._active_gpu_visibility_mask() is not None
            and LlamaCppBackend._resolve_visible_physical_ids() is None
        )

    @staticmethod
    def _emit_child_gpu_visibility(
        env: dict,
        pinned: str,
        *,
        prefer_rocr: bool = False,
    ) -> None:
        """Write the child's GPU visibility mask: CUDA, plus a ROCm mirror on AMD
        (masking only CUDA_VISIBLE_DEVICES leaves an AMD child seeing every GPU).

        Default: HIP_VISIBLE_DEVICES, clearing any inherited ROCR mask so the two
        cannot stack (ROCR re-indexes from 0, a non-zero HIP pin then points out of
        range, HIP sees 0 devices, and llama.cpp falls back to CPU).

        prefer_rocr masks at the ROCr/HSA layer instead (clearing HIP): a HIP mask
        filters only AFTER the HSA runtime enumerates every agent, and that
        enumeration segfaults on a GPU the build has no kernels for (gfx1036 iGPU
        under a gfx103X prebuilt) before llama-server logs a line.

        The CPU sentinel ("-1") keeps the HIP mask and leaves an inherited ROCR mask
        alone: HIP sees zero devices either way, so the anti-stacking clear buys
        nothing while dropping the mask re-exposes hidden agents to that same
        enumeration. Windows keeps the HIP mask too, ROCR being Linux-only."""
        env["CUDA_VISIBLE_DEVICES"] = pinned
        if pinned == "-1":
            # An inherited LLAMA_ARG_DEVICE / LLAMA_ARG_MAIN_GPU naming a device that
            # no longer enumerates makes llama.cpp exit rather than use our CPU.
            LlamaCppBackend._clear_device_placement_env(env)
        try:
            import torch as _torch

            # torch.version.hip is set on ROCm, None on CUDA; AMD SDK wheels may
            # leave it unset but encode "rocm" in __version__ (mirrors detect_hardware).
            if (
                getattr(_torch.version, "hip", None) is not None
                or "rocm" in getattr(_torch, "__version__", "").lower()
            ):
                if prefer_rocr and pinned != "-1" and sys.platform != "win32":
                    env["ROCR_VISIBLE_DEVICES"] = pinned
                    env.pop("HIP_VISIBLE_DEVICES", None)
                    # ROCR re-indexes the visible agents from 0, and with HIP
                    # cleared HIP honours CUDA_VISIBLE_DEVICES -- so it must carry
                    # the post-ROCR ordinals (0..N-1), not the physical ids, else a
                    # non-zero pick points out of range and HIP sees 0 devices (the
                    # same stacking the default path avoids by clearing ROCR).
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(
                        str(i) for i in range(len(pinned.split(",")))
                    )
                else:
                    env["HIP_VISIBLE_DEVICES"] = pinned
                    if pinned != "-1":
                        env.pop("ROCR_VISIBLE_DEVICES", None)
        except Exception as e:
            logger.debug("Failed to set ROCm visibility env vars for child: %s", e)

    @staticmethod
    def _unmasked_child_gpu_physical_ids() -> Optional[tuple[int, ...]]:
        from utils.hardware import get_parent_visible_gpu_ids

        parent_gpu_ids = get_parent_visible_gpu_ids()
        if len(parent_gpu_ids) == 1:
            return (int(parent_gpu_ids[0]),)
        return None

    @staticmethod
    def _inherited_child_gpu_order(pinned_ids: Iterable[int]) -> Optional[list[int]]:
        """``pinned_ids`` in the parent mask's order, or None to keep them as given.

        A numeric visibility mask defines enumeration ORDER as well as membership
        (CUDA_VISIBLE_DEVICES=1,0 makes physical 1 the child's device 0), and every
        producer of gpu_indices sorts ascending, so re-emitting it verbatim silently
        reverses a deliberate reordering and the user has no way to place a card
        first. Same rule _pin_visible_gpu_order_for_split applies to a manual
        --tensor-split, which is where it was already spelled out.

        Only when the mask covers the whole pinned set: a partial mask cannot say
        where the ids it omits belong. None when the order is already what we would
        emit, so callers can treat non-None as "this changed something".
        """
        inherited = LlamaCppBackend._resolve_visible_physical_ids()
        if not inherited:
            return None
        wanted = [int(i) for i in pinned_ids]
        ordered = [i for i in inherited if i in set(wanted)]
        if sorted(ordered) != sorted(wanted) or ordered == wanted:
            return None
        return ordered

    @staticmethod
    def _repoint_emitted_tensor_split(
        cmd: list, old_order: list[int], new_order: list[int]
    ) -> bool:
        """Move Unsloth's own ``--tensor-split`` shares onto the reordered devices.

        _plan_tensor_parallel returns the shares positional over gpu_indices, so a
        reordered mask without this hands the roomier card's share to the smaller
        one. Returns False when the shares cannot be moved, which vetoes the whole
        reorder: emitting the mask alone would be the same bug.
        """
        try:
            at = len(cmd) - 1 - cmd[::-1].index("--tensor-split")
        except ValueError:
            return True  # no shares emitted, nothing positional to carry
        if at + 1 >= len(cmd):
            return False
        shares = cmd[at + 1].split(",")
        if len(shares) != len(old_order):
            return False
        by_id = dict(zip(old_order, shares))
        cmd[at + 1] = ",".join(by_id[i] for i in new_order)
        return True

    @staticmethod
    def _pin_visible_gpu_order_for_split(env: dict) -> Optional[tuple[int, ...]]:
        """Pin the child's GPU enumeration to the picker's order for a manual
        ``--tensor-split`` across the whole visible set. CUDA's default
        FASTEST_FIRST enumeration applies the shares to the wrong cards on
        heterogeneous hosts (#5025), and CUDA_DEVICE_ORDER only fixes the
        numbering base: an inherited numeric visibility mask ALSO defines
        enumeration order, so a reordered parent mask (CUDA_VISIBLE_DEVICES=3,1)
        would still hand the shares to the wrong cards. The UI built the split
        positionally over get_backend_visible_gpu_info's device list (ascending
        physical via nvidia-smi, inherited mask order on the torch fallback), so
        re-emit the same set in that report order -- not an assumed ascending
        sort. The visible set itself never changes. No mask, an empty mask, or a
        UUID/MIG mask (which resolves to None) is left alone -- the multi-GPU
        controls are hidden for the latter."""
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        inherited = LlamaCppBackend._resolve_visible_physical_ids()
        if not inherited:
            return None
        order = None
        try:
            from utils.hardware import get_backend_visible_gpu_info
            info = get_backend_visible_gpu_info()
            if info.get("available") and info.get("index_kind") == "physical":
                reported = [d["index"] for d in info.get("devices", [])]
                if sorted(reported) == sorted(inherited):
                    order = reported
        except Exception as e:
            logger.debug("Could not read reported GPU order for split pin: %s", e)
        if order is None:
            order = sorted(inherited)
        # Re-emit at the layer that produced the mapping. A parent masked only
        # via ROCR_VISIBLE_DEVICES hides agents at the driver layer, and the
        # default HIP re-emission clears that mask -- HSA then enumerates every
        # agent again and can segfault at startup on an unsupported GPU the
        # parent was hiding (the crash prefer_rocr exists to avoid). Linux-only,
        # mirroring _resolve_visible_physical_ids: on Windows a stray ROCR var
        # is dead and was not the mapping's source.
        prefer_rocr = (
            sys.platform != "win32"
            and env.get("HIP_VISIBLE_DEVICES") is None
            and env.get("ROCR_VISIBLE_DEVICES") is not None
        )
        LlamaCppBackend._emit_child_gpu_visibility(
            env, ",".join(str(i) for i in order), prefer_rocr = prefer_rocr
        )
        return tuple(int(i) for i in order)

    @staticmethod
    def _vulkan_probe_answered(binary: Optional[str]) -> bool:
        """Whether the Vulkan device probe produced rows at all.

        `_vulkan_targets_are_igpus` folds "no answer" into "not integrated", which is
        right for the page-lock question it was written for and not something the
        bookkeeping can read as a measurement.
        """
        try:
            return bool(LlamaCppBackend._run_vulkan_probe(binary))
        except Exception:
            return False

    @staticmethod
    def _vulkan_targets_are_igpus(binary: Optional[str], gpu_indices = None) -> bool:
        """True when any Vulkan device in play is integrated.

        An iGPU's reported "VRAM" is shared system RAM (see
        ``_apply_igpu_host_reserve_mib``), so a model "fully offloaded" onto one
        is still backed by pageable host memory and is worth page-locking. Any,
        not every: a mixed selection splits weights onto the iGPU too, and those
        pages are as evictable as if it were the only device. An unreadable probe
        answers False.
        """
        try:
            rows = LlamaCppBackend._run_vulkan_probe(binary)
        except Exception:
            return False
        if not rows:
            return False
        wanted = set(gpu_indices) if gpu_indices else None
        selected = [r for r in rows if wanted is None or r["index"] in wanted]
        return any(r["is_igpu"] for r in selected)

    def _weights_in_host_memory(
        self,
        *,
        fully_gpu_offloaded: bool,
        gpu_memory_mode: Optional[str],
        gpu_layers: Optional[int],
        extra_args: Optional[Iterable[str]],
        gpu_indices = None,
        is_vulkan_backend: bool = False,
        binary: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        probe_vulkan: bool = True,
        fit_active: bool = False,
    ) -> bool:
        """True when the weights will sit in pageable host RAM, so mlock helps.

        Only the Model Memory gate uses this, and only to SKIP page-locking, so
        it errs towards True: that is the pre-existing behaviour.

        ``probe_vulkan`` False skips the Vulkan probe, the one subprocess in
        here, for a bookkeeping-only call; that keeps the conservative True.
        """
        from utils.hardware import is_apple_silicon

        # A CPU placement extra keeps weights in RAM whatever the layer count,
        # so it applies to the auto path too and cannot be short-circuited past.
        # Inherited env does the same and outlives any token stripping.
        if _args_place_tensors_on_cpu(extra_args):
            return True
        if _env_places_tensors_on_cpu(env):
            return True
        # A --device that names no GPU beats any layer count: llama.cpp has
        # nowhere to offload to, so the whole model stays in host RAM.
        if _device_selection_is_cpu(extra_args, env):
            return True
        # fully_gpu_offloaded predicts OUR "-ngl -1 --fit off", but auto keeps the
        # user's extras and appends them after, and llama.cpp is last-wins, so an
        # -ngl 0 lands the model in RAM. Only a fit that is switched ON voids the
        # prediction: "--fit off" is what we already pass, and a disabled fitter
        # cannot move anything to the CPU.
        if _extra_args_set_any_flag(extra_args, _GPU_LAYER_FLAGS) or fit_is_enabled_in(extra_args):
            fully_gpu_offloaded = False
        all_on_gpu = fully_gpu_offloaded or self._offloads_every_layer(
            gpu_memory_mode = gpu_memory_mode,
            gpu_layers = gpu_layers,
            extra_args = extra_args,
            fit_active = fit_active,
        )
        if not all_on_gpu:
            return True
        # Unified memory: the "VRAM" is host RAM, so the weights stay pageable
        # however fully they are offloaded.
        if is_apple_silicon():
            return True
        if is_vulkan_backend:
            # gpu_indices are Vulkan ordinals here, which the ROCm APU helper
            # would read as physical ids and answer for the wrong device. The
            # Vulkan probe owns device type; unprobed keeps the True answer.
            return not probe_vulkan or self._vulkan_targets_are_igpus(binary, gpu_indices)
        return self._amd_apu_wants_unified_memory(gpu_indices)

    def _offloads_every_layer(
        self,
        *,
        gpu_memory_mode: Optional[str],
        gpu_layers: Optional[int],
        extra_args: Optional[Iterable[str]],
        fit_active: bool = False,
    ) -> bool:
        """True when every layer lands on the GPU by a route ``fully_gpu_offloaded``
        does not cover: manual mode at its full-offload maximum, or a user -ngl.

        Only the Model Memory gate uses this, and only to SKIP page-locking, so
        every unknown answers False and keeps the pre-existing behaviour: an
        unreadable block count, CPU-side experts, a tensor override that places
        weights by name, or an unparseable -ngl.
        """
        n_layers = self.n_layers
        if not n_layers:
            return False
        if self._n_cpu_moe:
            return False
        # The parsed predicate, not flag presence: --n-cpu-moe 0 places nothing,
        # and answering False here would pin host RAM for an all-GPU launch.
        if _args_place_tensors_on_cpu(extra_args):
            return False
        requested: Optional[int] = None
        if gpu_memory_mode == "manual" and gpu_layers is not None and gpu_layers >= 0:
            # Manual owns offload and strips -ngl from the extras, so the field
            # is what the child gets. The picker's maximum is n_layers + 1.
            requested = gpu_layers
        else:
            try:
                requested = parse_gpu_layers_override(extra_args)
            except ValueError:
                return False
        if requested is None:
            return False
        # -1 is "all layers"; llama.cpp needs a count ABOVE the block count to
        # cover the output layer too, which is what the picker's maximum sends.
        # -1 is also llama.cpp's DEFAULT, so an active fitter is free to lower it
        # (common/fit.cpp aborts only on a count the user really set); a concrete
        # count stands, which is why only this arm is gated.
        if requested == -1:
            return not fit_active
        return requested > n_layers

    def _partially_offloads_layers(
        self,
        args: Optional[Iterable[str]],
        env: Optional[Mapping[str, str]] = None,
        fit_implies_partial: bool = True,
    ) -> bool:
        """Whether the effective main-model layer policy is partial GPU offload.

        A concrete layer count owns placement even when ``--fit on`` is present.
        Otherwise ``-1`` (or an omitted count) is partial only while the fitter may
        lower it. Zero is CPU-only rather than partial offload, and keeps CPU MTP,
        and so is a ``--device`` selection naming no GPU: llama.cpp then runs the
        model on the CPU whatever the layer count and the fitter say.

        ``fit_implies_partial`` is what an active fitter is allowed to mean. In Auto
        it is evidence: the planner ran and could not prove the model fits, which is
        why the fitter is on. In Manual mode the planner is skipped by design, so
        ``--fit on`` is a default rather than a verdict and says nothing about
        placement -- pass False there and let a concrete count answer instead.
        """
        n_layers = self.n_layers
        if not n_layers:
            return False
        try:
            requested = parse_gpu_layers_override(args)
        except ValueError:
            return False
        source_env = os.environ if env is None else env
        if _device_selection_is_cpu(args, source_env):
            return False
        if requested is None:
            raw = source_env.get("LLAMA_ARG_N_GPU_LAYERS")
            if raw is not None and raw.strip():
                try:
                    requested = int(raw)
                except ValueError:
                    requested = None
        if requested == 0:
            return False
        if requested is not None and requested > 0:
            # llama.cpp needs a count above block_count to include the output
            # layer, matching _offloads_every_layer.
            return requested <= n_layers
        return fit_implies_partial and fit_is_effectively_on(args, source_env)

    @staticmethod
    def _torch_is_rocm(torch) -> bool:
        """Whether this torch is a ROCm build. AMD SDK wheels leave
        ``version.hip`` unset and only encode "rocm" in ``__version__``."""
        return (
            getattr(torch.version, "hip", None) is not None
            or "rocm" in getattr(torch, "__version__", "").lower()
        )

    @staticmethod
    def _host_torch_is_rocm() -> bool:
        """Whether torch here is a ROCm build, without the caller importing torch.
        False when torch is missing or unreadable. Cheap guard so a CUDA or
        CPU-only host does not pay for ROCm-only work (#7624)."""
        try:
            import torch
            return LlamaCppBackend._torch_is_rocm(torch)
        except Exception:
            return False

    @staticmethod
    def _rocm_classification_answered() -> bool:
        """Whether `_rocm_unified_memory_gpu_ids` could actually look at the devices.

        Kept separate from it because the two questions have opposite safe answers:
        skipping a page-lock may treat "no APU" and "could not tell" alike, choosing
        a loader may not. DirectIO over a unified-memory APU replaces a pageable
        mapping with a model-sized allocated buffer.

        Answering it means reading every device, not finding the machinery to read
        them with, so this repeats the classify call rather than trusting a count.
        """
        try:
            import torch

            if not LlamaCppBackend._torch_is_rocm(torch):
                return False
            if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                return False
            from core.training.worker import _rocm_classify_unified_memory

            count = torch.cuda.device_count()
            if count <= 0:
                return False
            # Actually classify them. An enumerable device is not a classified one: the
            # loop below skips whatever it cannot read with `except: continue`, so that
            # device is simply absent from the unified set, which reads as discrete.
            # Every VISIBLE device, not just the selected ones, because mapping a ggml
            # ordinal onto a torch one is a translation this does not own.
            for ordinal in range(count):
                _rocm_classify_unified_memory(torch.cuda.get_device_properties(ordinal))
            return True
        except Exception:
            return False

    @staticmethod
    def _rocm_unified_memory_gpu_ids() -> set[int]:
        """PHYSICAL ids of visible ROCm GPUs whose "VRAM" is shared system RAM.

        Delegates to the training worker's classifier so both paths agree on what
        an APU is: it reads the driver's own ``is_integrated`` first, then the
        arch-name spellings, then the Radeon name table, because AMD SDK wheels
        may populate none of the arch attributes. Empty off ROCm and on error, so
        every caller keeps its discrete-GPU default.
        """
        try:
            import torch

            if not LlamaCppBackend._torch_is_rocm(torch):
                return set()
            if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                return set()
            from core.training.worker import _rocm_classify_unified_memory

            # Map visible ordinal -> physical id via the active ROCm mask (HIP,
            # then ROCR, then CUDA), mirroring _get_gpu_memory's ROCm branch.
            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()
            unified: set[int] = set()
            for ordinal in range(torch.cuda.device_count()):
                try:
                    _arch, is_unified = _rocm_classify_unified_memory(
                        torch.cuda.get_device_properties(ordinal)
                    )
                except Exception:
                    continue
                if not is_unified:
                    continue
                unified.add(
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
            return unified
        except Exception:
            return set()

    @staticmethod
    def _rocm_selected_pool_mib(gpu_indices = None) -> Optional[int]:
        """Return selected ROCm APUs' combined carve-out in MiB.

        Return ``None`` unless every requested device is present, readable, and a
        unified-memory APU. The ggml setting is process-wide, so mixed selections
        fail closed.
        """
        try:
            import torch

            if not LlamaCppBackend._torch_is_rocm(torch):
                return None
            if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                return None
            from core.training.worker import _rocm_classify_unified_memory

            wanted = None if gpu_indices is None else set(gpu_indices)
            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()
            total_mib = 0
            seen: set[int] = set()
            for ordinal in range(torch.cuda.device_count()):
                pid = (
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
                if wanted is not None and pid not in wanted:
                    continue
                try:
                    props = torch.cuda.get_device_properties(ordinal)
                    _arch, is_unified = _rocm_classify_unified_memory(props)
                except Exception:
                    return None
                if not is_unified:
                    return None
                total = int(getattr(props, "total_memory", 0) or 0)
                if total <= 0:
                    return None
                total_mib += total // (1024 * 1024)
                seen.add(pid)
            if not seen or (wanted is not None and seen != wanted):
                return None
            return total_mib
        except Exception:
            return None

    @staticmethod
    def _unified_memory_would_help(gpu_indices = None, need_bytes = None) -> bool:
        """Whether managed allocation is worth taking for the selected APUs.

        Host RAM must be the larger pool, against the carve-out's TOTAL (over-asking
        the carve-out fails cleanly, over-asking host RAM is an OOM kill), AND the
        weights must not fit that carve-out: managed pages fault in k_set_rows on
        Linux ROCm gfx1151 (HF Qwen3.8-Flash-Next-GGUF discussion 30, #10330).
        """
        try:
            pool_mib = LlamaCppBackend._rocm_selected_pool_mib(gpu_indices)
            if not pool_mib:
                return False
            host_mib = LlamaCppBackend._available_system_memory_mib()
            if not host_mib:
                return False
            if int(host_mib) <= int(pool_mib):
                return False
            if need_bytes is None:
                return False
            return int(need_bytes) > int(pool_mib) * 1024 * 1024
        except Exception:
            return False

    @staticmethod
    def _rocm_arch_by_physical_id() -> dict[int, str]:
        """Base gfx arch (lowercased, ``:xnack`` stripped) per PHYSICAL device id,
        honoring the active ROCm visibility mask (HIP, then ROCR, then CUDA).

        Read through the training worker's classifier so the alternate spellings
        AMD SDK / Radeon wheels use (``gcn_arch_name``, ``arch_name``,
        ``gfx_arch_name``) are covered as well as canonical ``gcnArchName``;
        reading only the canonical name would leave the map empty on those wheels
        and silently fail the gate open. Devices torch cannot describe are omitted;
        empty off ROCm and on any error."""
        arch_by_id: dict[int, str] = {}
        try:
            import torch

            if not LlamaCppBackend._torch_is_rocm(torch):
                return arch_by_id
            if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                return arch_by_id
            from core.training.worker import _rocm_classify_unified_memory

            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()
            for ordinal in range(torch.cuda.device_count()):
                try:
                    _arch, _is_unified = _rocm_classify_unified_memory(
                        torch.cuda.get_device_properties(ordinal)
                    )
                except Exception:
                    continue
                _base = (_arch or "").split(":")[0].strip().lower()
                if not _base:
                    continue
                pid = (
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
                arch_by_id[pid] = _base
        except Exception:
            return arch_by_id
        return arch_by_id

    @staticmethod
    def _offload_target_shares_system_memory(
        *, is_vulkan_backend: bool, shared_gpu_ids, detected_gpus, gpu_indices
    ) -> bool:
        """True when EVERY device this launch offloads to reports host RAM as its VRAM.

        An iGPU or a unified-memory APU has no separate pool and no bus between the
        two: a buffer llama.cpp calls "host RAM" and one it calls "VRAM" are the same
        physical memory. Tuning that exists to avoid moving bytes across PCI-E
        therefore buys nothing there, and any recovery path it disables is pure loss.

        Fails closed. A mixed selection, an unreadable inventory or a launch that
        pins nothing all answer False, so the caller keeps the behaviour it has.
        The two index spaces are kept apart the way the rest of this file keeps
        them: Vulkan ordinals against shared_gpu_ids, physical HIP/CUDA ids
        against the unified-memory sets.
        """
        devices = (
            list(gpu_indices) if gpu_indices else [idx for idx, _free in (detected_gpus or ())]
        )
        if not devices:
            return False
        if is_vulkan_backend:
            shared = set(shared_gpu_ids or ())
            return bool(shared) and all(idx in shared for idx in devices)
        # ROCm only, deliberately. The one caller is the Windows full-offload branch and
        # every integrated CUDA part is Linux-only, so _integrated_cuda_gpu_ids() could
        # only answer False here -- at the price of a get_device_properties() call, which
        # leaks a ~700 MiB primary context (see _get_gpu_memory). The ROCm helper answers
        # empty on a CUDA torch without touching the device.
        try:
            unified = LlamaCppBackend._rocm_unified_memory_gpu_ids()
        except Exception:
            return False
        return bool(unified) and all(idx in unified for idx in devices)

    @staticmethod
    def _amd_apu_wants_unified_memory(gpu_indices = None) -> bool:
        """True only for AMD unified-memory APUs, where GGML_CUDA_ENABLE_UNIFIED_MEMORY
        lets llama.cpp use shared system RAM (it hurts discrete GPUs). gpu_indices
        (PHYSICAL ids) scopes the check, so a dGPU on a mixed host is not treated as
        unified-memory; None means every visible GPU."""
        # Guarded like the rest of this family: a bad gpu_indices (not iterable,
        # unhashable members) answers False rather than failing the load.
        try:
            unified = LlamaCppBackend._rocm_unified_memory_gpu_ids()
            if gpu_indices is None:
                return bool(unified)
            return any(_i in unified for _i in gpu_indices)
        except Exception:
            return False

    # PHYSICAL integrated ids per visibility mask. Integratedness cannot change under
    # a fixed mask, and the mask decides which devices the answer is about.
    _INTEGRATED_CUDA_IDS: dict[tuple, set[int]] = {}

    @staticmethod
    def _integrated_cuda_mask_key() -> tuple:
        return tuple(
            os.environ.get(name)
            for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
        )

    @staticmethod
    def _integrated_cuda_probe_is_free() -> bool:
        """True when ``_integrated_cuda_gpu_ids()`` costs no NEW CUDA context.

        The launch preflight runs AFTER the VRAM budget was taken, so a probe there
        would price a tightly fitted child against a stale budget. So the preflight
        never probes: it reads an answer only if one is cached for this mask.
        ``_get_gpu_memory`` fills that cache before reading any free/total figure, which
        puts the cost inside the snapshot.

        BOTH its arms fill it now. The torch arm always did, and that was assumed to be
        the arm an integrated SoC takes, because nvidia-smi answers ``[N/A]`` on a DGX
        Spark and the CLI probe parses nothing. A Windows RTX Spark N1X answers 8128 MiB
        instead -- a real number, scoped to the dedicated carve-out -- so the CLI arm
        wins there and an integrated host reached the preflight with nothing cached.
        The probe reads ``get_device_properties`` only, which creates no primary context
        (measured at 0 MiB on an N1X, against 116 MiB for ``mem_get_info``).
        """
        return LlamaCppBackend._integrated_cuda_mask_key() in LlamaCppBackend._INTEGRATED_CUDA_IDS

    @staticmethod
    def _integrated_cuda_gpu_ids() -> set[int]:
        """PHYSICAL ids of visible CUDA GPUs whose "VRAM" is shared system RAM.

        Jetson and DGX Spark class parts set cudaDeviceProp::integrated, which torch
        surfaces as ``is_integrated``; the diffusion memory policy reads the same flag
        (diffusion_memory.py:292-294). ROCm is excluded because it reuses
        ``torch.cuda.*`` and ``_rocm_unified_memory_gpu_ids`` already answers for it.
        Empty off CUDA and on error, so every caller keeps its discrete-GPU default.
        """
        cached = LlamaCppBackend._INTEGRATED_CUDA_IDS.get(
            LlamaCppBackend._integrated_cuda_mask_key()
        )
        if cached is not None:
            # Settled: a second pass over the devices could learn nothing.
            return set(cached)
        try:
            import torch

            if LlamaCppBackend._torch_is_rocm(torch):
                return set()
            if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                return set()
            # Same ordinal -> physical mapping the ROCm twin uses, so a masked host
            # (CUDA_VISIBLE_DEVICES=2) does not answer for the wrong card.
            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()
            integrated: set[int] = set()
            # A device that did not answer leaves this incomplete: still returned, so
            # an unqueryable card keeps its discrete default, but not remembered, since
            # a caller reading it as settled would retry the query that failed.
            complete = True
            for ordinal in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(ordinal)
                except Exception:
                    complete = False
                    continue
                # Both spellings: the attribute was `integrated` before torch renamed
                # it, and an old wheel exposing neither reads discrete.
                if not (
                    getattr(props, "is_integrated", False) or getattr(props, "integrated", False)
                ):
                    continue
                integrated.add(
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
            # Cached for the preflight, but only a pass that reached every device: a
            # torch that raised says nothing about the hardware.
            if complete:
                LlamaCppBackend._INTEGRATED_CUDA_IDS[
                    LlamaCppBackend._integrated_cuda_mask_key()
                ] = integrated
            return integrated
        except Exception:
            return set()

    # {physical id: pool total MiB} per visibility mask, filled beside _INTEGRATED_CUDA_IDS.
    _INTEGRATED_CUDA_POOL_MIB: dict[tuple, dict[int, int]] = {}

    @staticmethod
    def _integrated_cuda_pool_total_mib() -> dict[int, int]:
        """Pool size in MiB for each visible integrated CUDA device.

        ``props.total_memory`` is what an integrated part can actually allocate, and it
        is the figure nvidia-smi contradicts on a Windows RTX Spark N1X: 46477 MiB here
        against the 8128 MiB carve-out the CLI reports. Read from the device list, so
        it costs no primary context (measured at 0 MiB, against 116 MiB for
        mem_get_info on the same part) and is cached per mask like the flag itself.

        Empty off CUDA, on ROCm and on any error, so callers keep the CLI's total.
        """
        key = LlamaCppBackend._integrated_cuda_mask_key()
        cached = LlamaCppBackend._INTEGRATED_CUDA_POOL_MIB.get(key)
        if cached is not None:
            return dict(cached)
        integrated = LlamaCppBackend._integrated_cuda_gpu_ids()
        if not integrated:
            return {}
        try:
            import torch

            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()
            totals: dict[int, int] = {}
            complete = True
            for ordinal in range(torch.cuda.device_count()):
                idx = (
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
                if idx not in integrated:
                    continue
                try:
                    props = torch.cuda.get_device_properties(ordinal)
                    totals[idx] = int(props.total_memory) // (1024 * 1024)
                except Exception:
                    complete = False
            if complete:
                LlamaCppBackend._INTEGRATED_CUDA_POOL_MIB[key] = totals
            return totals
        except Exception:
            return {}

    @staticmethod
    def _widen_integrated_cuda_rows(gpus: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        """Re-price nvidia-smi rows for an integrated CUDA SoC against its shared pool.

        The torch arm of ``_get_gpu_memory`` already sizes an integrated part this way,
        but it is only reached when the CLI answers nothing, which is the DGX Spark
        shape (memory.total reads ``[N/A]`` there) and not the Windows RTX Spark N1X
        shape, where the CLI answers 8128 MiB of a 46477 MiB budget and the nvidia-smi
        arm returns first. A model was told it needed "7.9 GB of a 6.9 GB budget" on a
        machine with 45 GiB. So the same policy is applied to a CLI answer that came
        back scoped to the carve-out.

        Widening only: a total is replaced solely by a LARGER one, and the free half is
        floored at the reading it replaces, so no device is offered less than the CLI
        already vouched for. A discrete host gets its rows back unchanged, having paid
        one classification that is cached for the life of the process.
        """
        try:
            integrated = LlamaCppBackend._integrated_cuda_gpu_ids()
            if not integrated or not gpus:
                return gpus
            # These rows are nvidia-smi indices; `_integrated_cuda_gpu_ids` answers in
            # CUDA's, which CUDA_DEVICE_ORDER defaults to FASTEST_FIRST, pinning only
            # device 0 (docs.nvidia.com/cuda/cuda-programming-guide, environment
            # variables). An unmappable mask is worse: no physical ids AND, since
            # `_visible_devices_mask` cannot parse it either, unfiltered CLI rows. Join
            # only when both halves are mappable, as `_cuda_join_is_unsafe` does.
            order = (os.environ.get("CUDA_DEVICE_ORDER") or "").strip().upper()
            raw_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
            mask_unset = raw_mask is None or not raw_mask.strip()
            mappable = mask_unset or LlamaCppBackend._resolve_visible_physical_ids() is not None
            # No single-row exemption: `gpus` is what SURVIVED parsing (the N1X's own
            # NPU row does not), so it is not a device count. main.py:19 sets PCI_BUS_ID,
            # so every Spark, Jetson and N1X still qualifies below.
            if not (mappable and order == "PCI_BUS_ID"):
                logger.debug(
                    "Not widening integrated CUDA rows: CUDA_DEVICE_ORDER is %r and "
                    "the mask %s, so a CUDA id cannot be joined to an nvidia-smi index "
                    "here.",
                    order or "unset (FASTEST_FIRST)",
                    "resolves to no physical ids" if not mappable else "is mappable",
                )
                return gpus
            totals = LlamaCppBackend._integrated_cuda_pool_total_mib()
            avail = LlamaCppBackend._available_system_memory_mib()
            cgroup_mib = LlamaCppBackend._cgroup_available_memory_mib()
            # One pool, however many integrated devices draw on it: the same reason the
            # torch arm divides. No shipping product pairs two integrated SoCs, so this
            # is a division by 1 everywhere it currently runs.
            shared_count = max(1, sum(1 for idx, _f, _t in gpus if idx in integrated))
            out: list[tuple[int, int, int]] = []
            for idx, free_mib, total_mib in gpus:
                if idx not in integrated:
                    out.append((idx, free_mib, total_mib))
                    continue
                cli_total_mib = total_mib
                pool_mib = totals.get(idx) or 0
                if pool_mib > total_mib:
                    total_mib = pool_mib
                raw_mib = free_mib
                # MemAvailable, as the torch arm uses: both free readings undercount here
                # (#9889). A zero total means unsized, so it never becomes a cap.
                if avail is not None and total_mib > 0:
                    raw_mib = min(total_mib, max(raw_mib, avail))
                cgroup_bound = False
                if cgroup_mib is not None and cgroup_mib <= raw_mib:
                    # Charged to the cgroup, so publish no total and let the fit price
                    # off the free reading, which IS the container's ceiling.
                    raw_mib = cgroup_mib
                    cgroup_bound = True
                if shared_count > 1:
                    raw_mib //= shared_count
                    total_mib //= shared_count
                capped = _apply_igpu_host_reserve_mib(raw_mib, True)
                # Never below what the driver already vouched for. NOT under a cgroup:
                # allocations are charged to memory.max, so the floor would hand the fit
                # planner a budget the kernel kills.
                if not cgroup_bound:
                    capped = max(capped, free_mib // shared_count)
                # Only when something moved: _get_gpu_memory is uncached and
                # _wait_for_vram_settle polls it every 0.25 s.
                if capped != free_mib or total_mib != cli_total_mib:
                    logger.info(
                        f"CUDA device {idx} is a unified-memory SoC sharing system RAM; "
                        f"sizing it against the shared pool "
                        f"({free_mib}->{capped}MiB free, "
                        f"{pool_mib or total_mib}MiB pool)"
                    )
                out.append((idx, capped, 0 if cgroup_bound else total_mib))
            return sorted(out, key = lambda g: g[0])
        except Exception as e:
            logger.debug(f"integrated CUDA pool sizing failed: {e}")
            return gpus

    @staticmethod
    def _integrated_cuda_unified_memory(gpu_indices = None) -> bool:
        """True when a credited CUDA device is an integrated (unified-memory) SoC.

        Scoped by gpu_indices (PHYSICAL ids) like its ROCm twin, so a dGPU on a
        mixed host is not treated as unified; None means every visible GPU. Any
        integrated device is enough: one shared pool in the credited set is already
        the double count.
        """
        try:
            integrated = LlamaCppBackend._integrated_cuda_gpu_ids()
            if gpu_indices is None:
                return bool(integrated)
            return any(_i in integrated for _i in gpu_indices)
        except Exception:
            return False

    @staticmethod
    def _integrated_cuda_selection_is_all_shared(gpu_indices = None) -> bool:
        """True when EVERY credited CUDA device is an integrated SoC.

        Stricter than ``_integrated_cuda_unified_memory``, which answers for ANY, and
        that difference matters here: this guard charges the WHOLE model against system
        RAM, and layers placed on a discrete card in a mixed selection never touch the
        shared pool. Answering for any would call an otherwise fitting load oversized,
        warn about it, and override an explicitly unmapped load to use mmap.
        """
        try:
            integrated = LlamaCppBackend._integrated_cuda_gpu_ids()
            if not integrated:
                return False
            if gpu_indices is not None:
                return bool(gpu_indices) and all(_i in integrated for _i in gpu_indices)
            # Nothing pinned, so the credited set is every visible device.
            visible = LlamaCppBackend._resolve_visible_physical_ids()
            if visible is None:
                import torch
                visible = list(range(torch.cuda.device_count()))
            return bool(visible) and all(_i in integrated for _i in visible)
        except Exception:
            return False

    # "Off" spellings ggml itself ignores (it tests presence); we honour them.
    _UNIFIED_MEMORY_OFF = frozenset({"", "0", "false", "no", "off"})

    @staticmethod
    def _unified_memory_opted_out(env = None) -> bool:
        """True when GGML_CUDA_ENABLE_UNIFIED_MEMORY must be ABSENT from this launch.

        ggml tests presence, not value, so `=0` is not an off switch (#8651). Both
        opt-outs (UNSLOTH_DISABLE_UNIFIED_MEMORY=1, mirroring UNSLOTH_DISABLE_DC_TUNING,
        or a falsy GGML_CUDA_ENABLE_UNIFIED_MEMORY) must end in the name unset.
        Fails open (False) so a bad env never blocks a load.
        """
        try:
            source = os.environ if env is None else env
            if str(source.get("UNSLOTH_DISABLE_UNIFIED_MEMORY", "")).strip() == "1":
                return True
            value = source.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY")
            if value is None:
                return False
            return str(value).strip().lower() in LlamaCppBackend._UNIFIED_MEMORY_OFF
        except Exception:
            return False

    @staticmethod
    def _unified_memory_opted_in(env = None) -> bool:
        """True when UNSLOTH_ENABLE_UNIFIED_MEMORY=1 asks for managed allocation even
        though the weights fit. Exact "1"; the opt-out wins over it."""
        try:
            source = os.environ if env is None else env
            return str(source.get("UNSLOTH_ENABLE_UNIFIED_MEMORY", "")).strip() == "1"
        except Exception:
            return False

    @staticmethod
    def _unified_memory_for_launch(
        gpu_indices,
        need_bytes,
        *,
        opted_in = False,
    ) -> bool:
        """The launch-time decision behind GGML_CUDA_ENABLE_UNIFIED_MEMORY. It is
        process-wide and hurts discrete cards, so a mixed selection is always False."""
        if opted_in and LlamaCppBackend._rocm_selected_pool_mib(gpu_indices) is not None:
            return True
        return LlamaCppBackend._unified_memory_would_help(gpu_indices, need_bytes = need_bytes)

    # Datacenter / professional NVIDIA parts that get the FP32-accum tuning.
    # Whole-word (\b) so "a100" does not fire on "RTX A1000".
    _DATACENTER_GPU_RE = re.compile(
        r"\b(?:a100|a30|h100|h200|h800|gh200|b200|b100|b300|gb200|gb300|"
        r"l40s?|l4|rtx pro 6000|rtx 6000 ada)\b"
    )

    # The NVLink-capable subset. The rest (RTX 6000 Ada/PRO, L40/L40S, L4) have no
    # NVLink connector, so peer copies cross PCIe, where a translating IOMMU on
    # bare metal discards them silently while CUDA reports success (#10613).
    # NECESSARY, never SUFFICIENT: _p2p_veto_reason also demands a positive NV#
    # link, since a name cannot see an absent bridge.
    _NVLINK_FABRIC_GPU_RE = re.compile(
        r"\b(?:a100|a30|h100|h200|h800|gh200|b200|b100|b300|gb200|gb300)\b"
    )

    @staticmethod
    def _all_selected_gpus_match(pattern, gpu_indices = None) -> bool:
        """True iff every selected NVIDIA GPU's name matches ``pattern``. False for
        consumer GeForce / ROCm / CPU / errors and for a mixed selection.

        gpu_indices are PHYSICAL ids (see _get_gpu_free_memory) but
        get_device_properties wants mask-relative ordinals, so names are keyed by
        physical id via CUDA_VISIBLE_DEVICES; otherwise a masked host would drop
        the tuning or probe the wrong GPU."""
        try:
            import torch

            if getattr(torch.version, "hip", None) is not None:
                return False  # ROCm reuses torch.cuda.*; not a CUDA part
            if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
                return False
            count = torch.cuda.device_count()

            # Mirror _get_gpu_free_memory: map visible ordinal -> physical id via
            # CUDA_VISIBLE_DEVICES; unset/unparsable leaves physical id == ordinal.
            physical_ids = LlamaCppBackend._resolve_visible_physical_ids()

            names_by_id: dict[int, str] = {}
            for ordinal in range(count):
                try:
                    name = (torch.cuda.get_device_properties(ordinal).name or "").lower()
                except Exception:
                    continue
                pid = (
                    physical_ids[ordinal]
                    if physical_ids is not None and ordinal < len(physical_ids)
                    else ordinal
                )
                names_by_id[pid] = name

            indices = list(gpu_indices) if gpu_indices else list(names_by_id)
            saw = False
            for _i in indices:
                name = names_by_id.get(_i)
                if name is None:
                    continue  # not visible -> skip (fail conservative)
                saw = True
                if not pattern.search(name):
                    return False
            return saw
        except Exception:
            return False

    @staticmethod
    def _is_datacenter_gpu(gpu_indices = None) -> bool:
        """True iff every selected GPU qualifies for the FP32-accum tuning. Mask
        handling: _all_selected_gpus_match. Peer-copy safety: _p2p_veto_reason."""
        return LlamaCppBackend._all_selected_gpus_match(
            LlamaCppBackend._DATACENTER_GPU_RE, gpu_indices
        )

    # `nvidia-smi topo -m` vocabulary: NV# is a bonded set of # NVLinks; SYS /
    # NODE / PHB / PXB / PIX all traverse PCIe and are unsafe. The header is
    # ANSI-underlined, so escapes are stripped before parsing.
    _TOPO_GPU_LABEL_RE = re.compile(r"^GPU(\d+)$")
    _TOPO_NVLINK_RE = re.compile(r"^NV\d+$")
    _ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

    # One probe per process, not per launch. None = not yet probed; otherwise a
    # 1-tuple whose value may itself be None ("probed and unavailable").
    _NVLINK_TOPO_CACHE = None
    # Bumped on refresh so a slow prime cannot publish over a newer invalidation.
    _NVLINK_TOPO_GENERATION = 0
    _NVLINK_TOPO_LOCK = threading.Lock()

    # Its own vocabulary: a fabricated NV18 could be misread as a link count.
    _NVML_NVLINK_LABEL = "NVLINK"
    _NVML_NO_NVLINK_LABEL = "NO-NVLINK"

    # nvml.h. Only scalars and opaque handles cross, so no struct layout to get wrong.
    _NVML_SUCCESS = 0
    _NVML_P2P_CAPS_INDEX_NVLINK = 2
    _NVML_P2P_STATUS_OK = 0
    # 1..5 are the documented denials; UNKNOWN (6) and anything past the enum are not.
    _NVML_P2P_STATUS_DEFINITE_NEGATIVES = frozenset({1, 2, 3, 4, 5})
    _NVML_FEATURE_ENABLED = 1
    # Link enumeration stops at the first error; this only bounds a runaway loop.
    _NVML_MAX_LINKS = 64

    @classmethod
    def _matrix_is_nvml(cls, matrix: dict) -> bool:
        """Whether this matrix came from NVML, i.e. whether nvidia-smi is known to
        have answered."""
        return any(
            v in (cls._NVML_NVLINK_LABEL, cls._NVML_NO_NVLINK_LABEL) for v in matrix.values()
        )

    @classmethod
    def _label_is_nvlink(cls, label: str) -> bool:
        """The one place that decides whether a label means NVLink, so the
        selected-pair check and the whole-box check cannot drift apart."""
        return bool(cls._TOPO_NVLINK_RE.match(label)) or label == cls._NVML_NVLINK_LABEL

    @staticmethod
    def _nvml_library():
        """The driver's own NVML, via ctypes: present exactly when nvidia-smi is, and
        no Python dependency (nvidia-ml-py is NOT installed in a Studio env). None
        when it cannot be loaded."""
        if os.name == "nt":
            # nvml.dll can sit in NVSMI\ rather than a DLL search path, as nvidia-smi.exe
            # does (see _nvidia_smi_executable in utils/hardware/nvidia.py).
            names = ["nvml.dll"]
            for root in (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432")):
                if root:
                    names.append(os.path.join(root, "NVIDIA Corporation", "NVSMI", "nvml.dll"))
            windir = os.environ.get("SystemRoot")
            if windir:
                names.append(os.path.join(windir, "System32", "nvml.dll"))
        else:
            names = ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        for name in names:
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        return None

    # ctypes has no subprocess timeout and a wedged driver blocks nvmlInit_v2 forever.
    _NVML_PROBE_TIMEOUT_SECONDS = 10

    @classmethod
    def _probe_nvml_nvlink_topology(cls) -> Optional[dict]:
        """_probe_nvml_nvlink_topology_inner under a wall-clock bound. A stalled NVML
        call cannot be cancelled, so the thread is abandoned as a daemon and the caller
        falls through to the shell-out, which has a timeout of its own."""
        box = {}

        def _run():
            box["matrix"] = cls._probe_nvml_nvlink_topology_inner()

        worker = threading.Thread(target = _run, daemon = True, name = "nvml-nvlink-probe")
        worker.start()
        worker.join(cls._NVML_PROBE_TIMEOUT_SECONDS)
        if worker.is_alive():
            logger.debug(
                f"NVML NVLink probe still running after "
                f"{cls._NVML_PROBE_TIMEOUT_SECONDS}s; abandoning it"
            )
            return None
        return box.get("matrix")

    @classmethod
    def _probe_nvml_nvlink_topology_inner(cls) -> Optional[dict]:
        """Same matrix as _probe_nvlink_topology, from NVML instead of a shell-out:
        ~85 ms against ~1200 ms on an 8x B200. Keyed by the NVML device index, which
        is the nvidia-smi index (both enumerate in PCI order and both ignore
        CUDA_VISIBLE_DEVICES, verified on 8x B200). None when the answer is not
        conclusive, which sends the caller to the shell-out.

        Evidence is per PAIR: an active link whose remote endpoint is an NVSwitch
        proves only that the GPU is attached to a switch, and switch fabrics can be
        partitioned, so "every GPU has a switch link" must NOT be read as "every pair
        is reachable"."""
        lib = cls._nvml_library()
        if lib is None:
            return None
        initialised = False
        try:
            try:
                get_status = lib.nvmlDeviceGetP2PStatus
                get_count = lib.nvmlDeviceGetCount_v2
                get_handle = lib.nvmlDeviceGetHandleByIndex_v2
                get_link_state = lib.nvmlDeviceGetNvLinkState
                nvml_init = lib.nvmlInit_v2
            except AttributeError as e:
                # An older driver missing these still has the working shell-out.
                logger.debug(f"NVML lacks a required symbol: {e}")
                return None
            for fn in (get_status, get_count, get_handle, get_link_state, nvml_init):
                fn.restype = ctypes.c_int
            if nvml_init() != cls._NVML_SUCCESS:
                return None
            initialised = True

            count = ctypes.c_uint()
            if get_count(ctypes.byref(count)) != cls._NVML_SUCCESS:
                return None
            if count.value < 2:
                return None  # peer copies need two devices; nothing to say

            handles = []
            for i in range(count.value):
                handle = ctypes.c_void_p()
                if get_handle(ctypes.c_uint(i), ctypes.byref(handle)) != cls._NVML_SUCCESS:
                    return None  # partial inventory: never approve the rest
                handles.append(handle)

            # Per-device active links, used only to contradict a positive.
            active_links = []
            for handle in handles:
                seen = 0
                for link in range(cls._NVML_MAX_LINKS):
                    state = ctypes.c_uint()
                    rc = get_link_state(handle, ctypes.c_uint(link), ctypes.byref(state))
                    if rc != cls._NVML_SUCCESS:
                        break  # past this device's last link, or unsupported
                    if state.value == cls._NVML_FEATURE_ENABLED:
                        seen += 1
                active_links.append(seen)

            matrix: dict = {}
            for a in range(count.value):
                for b in range(count.value):
                    if a == b:
                        continue
                    status = ctypes.c_uint()
                    rc = get_status(
                        handles[a],
                        handles[b],
                        ctypes.c_uint(cls._NVML_P2P_CAPS_INDEX_NVLINK),
                        ctypes.byref(status),
                    )
                    if rc != cls._NVML_SUCCESS:
                        return None  # unknown for one pair is unknown for all
                    linked = status.value == cls._NVML_P2P_STATUS_OK
                    if not linked and status.value not in cls._NVML_P2P_STATUS_DEFINITE_NEGATIVES:
                        # NO-NVLINK would read as conclusive and rob `topo -m` of the say.
                        logger.debug(
                            f"NVML P2P status {status.value} for GPU {a}<->{b} is not a "
                            "definite answer; treating the topology as unreadable"
                        )
                        return None
                    # No live NVLink on an endpoint means the pair cannot be NVLinked.
                    # An OK on a PCIe-only box becomes unknown, not a false positive (#10613).
                    if linked and not (active_links[a] and active_links[b]):
                        logger.debug(
                            f"NVML says GPU {a}<->{b} is NVLink but they report "
                            f"{active_links[a]}/{active_links[b]} active links; "
                            "treating the topology as unreadable"
                        )
                        return None
                    matrix[(a, b)] = cls._NVML_NVLINK_LABEL if linked else cls._NVML_NO_NVLINK_LABEL

            # By key, not count: right-sized with wrong keys passes a length check.
            expected = {(a, b) for a in range(count.value) for b in range(count.value) if a != b}
            if set(matrix) != expected:
                return None
            return matrix
        except Exception as e:
            logger.debug(f"NVML NVLink probe failed: {e}")
            return None
        finally:
            if initialised:
                try:
                    lib.nvmlShutdown()
                except Exception:
                    pass

    @staticmethod
    def _probe_nvlink_topology() -> Optional[dict]:
        """Parse ``nvidia-smi topo -m`` into {(i, j): "NV18" | "NODE" | ...}, keyed
        by the *nvidia-smi* index (PCI order, NOT the CUDA ordinal), both orders
        present. None when the topology cannot be established. Never raises.

        The table also carries NIC rows/columns, affinity columns and a Legend, so
        the GPU columns are the leading run of GPU<n> header labels and only GPU<n>
        rows are read."""
        try:
            from utils.hardware import gpu_query

            result = gpu_query.run_nvidia_smi(
                ["nvidia-smi", "topo", "-m"],
                cache = False,
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                env = child_env_without_native_path_secret(),
                **_windows_hidden_subprocess_kwargs(),
            )
            if result.returncode != 0:
                return None
            columns: list[int] = []
            matrix: dict = {}
            for raw in result.stdout.splitlines():
                line = LlamaCppBackend._ANSI_ESCAPE_RE.sub("", raw)
                if line.strip().lower().startswith("legend"):
                    break
                tokens = line.split()
                if not tokens:
                    continue
                if not columns:
                    # Header: the leading run of GPU<n> labels names the columns.
                    for token in tokens:
                        m = LlamaCppBackend._TOPO_GPU_LABEL_RE.match(token)
                        if m is None:
                            break
                        columns.append(int(m.group(1)))
                    # Data rows lead with GPU<n> too, so one column is more likely
                    # a missing header than a real table: keep looking.
                    if len(columns) < 2:
                        columns = []
                    continue
                row = LlamaCppBackend._TOPO_GPU_LABEL_RE.match(tokens[0])
                if row is None:
                    continue  # NIC row, or the blank line before the legend
                source = int(row.group(1))
                labels = tokens[1 : 1 + len(columns)]
                if len(labels) != len(columns):
                    return None  # truncated row: refuse to guess
                for col, label in zip(columns, labels):
                    if col != source:
                        matrix[(source, col)] = label.upper()
            # Every directed pair the header promised, or nothing. Row-truncated
            # output still exits 0 and parses cleanly, and the rows that arrived can
            # be uniformly NV#, which reads as "the whole box is NVLinked" and would
            # enable P2P on cards whose links were never seen (#10613).
            expected = {(a, b) for a in columns for b in columns if a != b}
            if set(matrix) != expected:
                logger.debug(
                    f"nvidia-smi topo -m: {len(matrix)} pairs for {len(columns)} "
                    f"GPUs, expected {len(expected)}; treating as unreadable"
                )
                return None
            return matrix or None
        except Exception as e:
            logger.debug(f"nvidia-smi topo probe failed: {e}")
            return None

    @classmethod
    def _probe_interconnect_matrix(cls) -> Optional[dict]:
        """NVML first, `nvidia-smi topo -m` second, nothing third. Both answer in the
        same index space and shape; NVML is ~14x faster, so the shell-out is only paid
        on drivers too old to answer.

        UNSLOTH_P2P_TOPO_CROSSCHECK=1 runs both and logs any disagreement, to qualify
        the fast path on fabrics not available here (PCIe-only, partially bridged); an
        NVML-positive / topo-negative pair is the result that would matter."""
        matrix = cls._probe_nvml_nvlink_topology()
        if matrix is not None and os.environ.get("UNSLOTH_P2P_TOPO_CROSSCHECK") == "1":
            topo = cls._probe_nvlink_topology()
            if topo is not None:
                nvml_yes = {k for k, v in matrix.items() if cls._label_is_nvlink(v)}
                topo_yes = {k for k, v in topo.items() if cls._label_is_nvlink(v)}
                if nvml_yes != topo_yes:
                    logger.warning(
                        "P2P topology cross-check disagreement: NVML-only pairs "
                        f"{sorted(nvml_yes - topo_yes)}, topo-only pairs "
                        f"{sorted(topo_yes - nvml_yes)}. Using topo -m. Please "
                        "report this with `nvidia-smi topo -m` output (#10613)."
                    )
                    return topo
        if matrix is not None:
            return matrix
        return cls._probe_nvlink_topology()

    @classmethod
    def prime_nvlink_topology(cls) -> None:
        """Warm the cache with NVML only, for a speculative caller off the load path.

        Deliberately skips the `nvidia-smi` fallback. A prime is an optimisation, so
        it may pay the cheap path and nothing else: spawning a subprocess that can run
        for its full timeout, on a background thread, perturbs whatever else shares
        the process. If NVML cannot answer the cache stays cold and the load path does
        the slow probe inline, exactly as it would have without a prime.

        Publishes only a success, for the reason cache_failure documents below: a miss
        cached this early would outlive the condition that caused it."""
        if cls._NVLINK_TOPO_CACHE is not None:
            return
        if os.environ.get("UNSLOTH_P2P_TOPO_CROSSCHECK") == "1":
            # The cross-check lives in _probe_interconnect_matrix, which only the load
            # path reaches, and the warm stage always primes first: publishing here
            # would retire the diagnostic. Cold cache costs the inline slow probe.
            return
        matrix = cls._probe_nvml_nvlink_topology()
        if matrix is None:
            return
        with cls._NVLINK_TOPO_LOCK:
            if cls._NVLINK_TOPO_CACHE is None:
                cls._NVLINK_TOPO_CACHE = (matrix,)

    @classmethod
    def _nvlink_topology(
        cls,
        refresh = False,
        cache_failure = True,
    ) -> Optional[dict]:
        """_probe_interconnect_matrix, cached for the life of the process.

        The probe runs outside the lock (slow, and a duplicate pass is harmless), but
        publication is guarded: a background prime and a model load can both be in
        flight, and the generation check stops a slow winner from overwriting a newer
        invalidation."""
        if not refresh and cls._NVLINK_TOPO_CACHE is not None:
            return cls._NVLINK_TOPO_CACHE[0]
        with cls._NVLINK_TOPO_LOCK:
            generation = cls._NVLINK_TOPO_GENERATION
            if refresh:
                cls._NVLINK_TOPO_GENERATION += 1
                generation = cls._NVLINK_TOPO_GENERATION
            elif cls._NVLINK_TOPO_CACHE is not None:
                return cls._NVLINK_TOPO_CACHE[0]
        probed = cls._probe_interconnect_matrix()
        with cls._NVLINK_TOPO_LOCK:
            if generation == cls._NVLINK_TOPO_GENERATION:
                if probed is None and not cache_failure:
                    # A miss probed mid driver init would keep P2P off for the process.
                    return None
                existing = cls._NVLINK_TOPO_CACHE
                if probed is None and existing is not None and existing[0] is not None:
                    # The generation only guards a refresh, so without this a racing
                    # transient failure would erase a concurrent pass's good matrix.
                    return existing[0]
                cls._NVLINK_TOPO_CACHE = (probed,)
                return probed
            # A refresh started mid-pass owns the answer; discard this one as stale.
            cached = cls._NVLINK_TOPO_CACHE
        return probed if cached is None else cached[0]

    @staticmethod
    def _running_virtualized() -> bool:
        """True when this kernel runs under a hypervisor. CUDA supports peer copies
        via VM pass-through but NOT on bare metal behind a translating IOMMU, so the
        IOMMU veto applies only off this path. False when unknown (no flags line on
        non-x86), which keeps the veto in play."""
        try:
            with open("/proc/cpuinfo", "r", encoding = "utf-8", errors = "replace") as f:
                for line in f:
                    if line.startswith("flags") and ":" in line:
                        return "hypervisor" in line.split(":", 1)[1].split()
        except Exception:
            pass
        return False

    # One directory per IOMMU group, each with a `type`. Overridable for tests.
    _IOMMU_GROUPS_ROOT = "/sys/kernel/iommu_groups"

    # Whether _get_gpu_memory ids are nvidia-smi PCI indices (True) or torch CUDA
    # ordinals (False, the fallback); None until a probe has run. They coincide
    # only under PCI_BUS_ID, so a caller pinning the child's device order has to
    # know which it holds (#10613).
    _GPU_IDS_ARE_PCI_INDICES = None
    # index -> inherited uuid, when CUDA_VISIBLE_DEVICES named the NVML rows by uuid.
    _NVML_ROWS: list = []  # the last NVML inventory; _child_visibility_for reads it

    # Boot-time property: read once, not per launch (the #10613 host has 175
    # groups). Only the default root is cached; an explicit root (tests) re-reads.
    _IOMMU_CACHE = None

    @classmethod
    def _iommu_is_translating_cached(cls) -> Optional[bool]:
        if cls._IOMMU_CACHE is None:
            try:
                cls._IOMMU_CACHE = (cls._iommu_is_translating(),)
            except Exception as e:
                logger.debug(f"IOMMU probe failed: {e}")
                cls._IOMMU_CACHE = (None,)
        return cls._IOMMU_CACHE[0]

    @staticmethod
    def _iommu_is_translating(root = None) -> Optional[bool]:
        """Whether the IOMMU translates DMA: True for DMA / DMA-FQ groups, False
        when every group is identity/passthrough or there are none, None when the
        groups exist but none is readable. Never raises.

        CUDA C++ Programming Guide, "IOMMU on Linux": bare-metal PCIe peer-to-peer
        is unsupported while the IOMMU translates, and the dropped write looks like
        a successful copy rather than an error (#10613). Read per group, not per
        GPU: a host with ANY translating group is not one to infer P2P safety on,
        and resolving each GPU's PCI address needs a third index-space join."""
        root = root or LlamaCppBackend._IOMMU_GROUPS_ROOT
        try:
            groups = os.listdir(root)
        except Exception:
            # Absent = nothing translates; present but unreadable = unknown.
            return None if os.path.isdir(root) else False
        if not groups:
            return False  # IOMMU compiled in but off
        kinds = []
        for group in groups:
            try:
                with open(
                    os.path.join(root, group, "type"),
                    "r",
                    encoding = "utf-8",
                    errors = "replace",
                ) as f:
                    kinds.append(f.read().strip().upper())
            except OSError:
                continue  # pre-5.x kernels have no `type`; other groups may
        if not kinds:
            return None
        return any(not kind.startswith("IDENTITY") for kind in kinds)

    @classmethod
    def _p2p_veto_reason(
        cls,
        gpu_indices = None,
        launch_order_pinned = False,
        ids_are_pci_indices = None,
    ) -> Optional[str]:
        """Fail-closed wrapper around _p2p_veto_reason_inner: anything unexpected
        reads as "do not set P2P", since losing the tuning beats an exception
        escaping into load_model and failing the load."""
        try:
            return cls._p2p_veto_reason_inner(gpu_indices, launch_order_pinned, ids_are_pci_indices)
        except Exception as e:
            logger.debug(f"peer-fabric check failed: {e}")
            return f"the peer-fabric check could not complete ({type(e).__name__})"

    @classmethod
    def _p2p_veto_reason_inner(
        cls,
        gpu_indices = None,
        launch_order_pinned = False,
        ids_are_pci_indices = None,
    ) -> Optional[str]:
        """Why GGML_CUDA_P2P must NOT be set for this selection, or None once a
        working NVLink fabric is confirmed for every selected pair. Fails CLOSED on
        every unknown.

        Before #10613 the flag came from the device NAME alone, which cannot see an
        absent NVLink bridge: 2x RTX 6000 Ada matched the allowlist and every model
        returned garbage. The driver is no better, since cudaDeviceCanAccessPeer
        and `nvidia-smi topo -p2p w` both answered OK on that host, so the only
        evidence accepted here is a positive NV# link.

        Index spaces: gpu_indices are physical ids, the matrix is keyed by
        nvidia-smi index, and the two coincide only under
        CUDA_DEVICE_ORDER=PCI_BUS_ID (CUDA otherwise enumerates FASTEST_FIRST).
        Rather than lose the tuning on every host that never set it, an inexact
        mapping demands a uniformly NV# matrix, which holds under any permutation.

        Overrides come first: UNSLOTH_DISABLE_DC_P2P=1 drops P2P while KEEPING
        FP32-accum (all-or-nothing UNSLOTH_DISABLE_DC_TUNING would discard a tuning
        that is not implicated), UNSLOTH_FORCE_DC_P2P=1 opts back in for a real but
        unparsable fabric -- check it with scripts/p2p_integrity_probe.py."""
        if os.environ.get("UNSLOTH_DISABLE_DC_P2P") == "1":
            return "disabled by UNSLOTH_DISABLE_DC_P2P=1"
        if os.environ.get("UNSLOTH_FORCE_DC_P2P") == "1":
            return None

        def _pcie(reason: str) -> str:
            """Append the IOMMU diagnosis to a veto that implies a PCIe peer path.
            On bare-metal Linux with a translating IOMMU such a copy is not merely
            slow but unsupported, the write silently discarded. Hangs off EVERY such
            veto, not just the topology one: the #10613 reporter's 2x RTX 6000 Ada
            fails the NAME gate first and would otherwise never learn why its output
            was garbled."""
            if cls._iommu_is_translating_cached() and not cls._running_virtualized():
                reason += (
                    "; bare-metal Linux with a translating IOMMU, where CUDA "
                    "does not support PCIe peer copies at all"
                )
            return reason

        if not cls._all_selected_gpus_match(cls._NVLINK_FABRIC_GPU_RE, gpu_indices):
            return _pcie("no NVLink-capable part in the selection, so peer copies would cross PCIe")

        matrix = cls._nvlink_topology()
        if not matrix:
            return "neither NVML nor nvidia-smi topo -m gave a usable interconnect matrix"

        gpu_ids = sorted({i for pair in matrix for i in pair})

        # gpu_indices are ALREADY nvidia-smi indices (_get_gpu_memory reads
        # `--query-gpu=index`, the same enumeration as `topo -m`), so they index the
        # matrix VERBATIM and remapping them as CUDA ordinals is wrong: under
        # FASTEST_FIRST that would turn a PCIe-crossing selection into an
        # NVLinked-looking one and enable the flag this gate exists to withhold. A
        # usable matrix means nvidia-smi answered, so the selection came from its
        # branch too.
        if gpu_indices is not None:
            # Verifying these ids means something only if the CHILD resolves the
            # same cards. Without a PCI_BUS_ID pin it reads them in FASTEST_FIRST
            # order, so [0,1] here can be [0,2] there: refuse rather than confirm
            # NVLink for a pair that is not the one about to run (#10613).
            # Unless the whole box is NVLinked, where the mapping cannot matter and
            # vetoing would lose P2P on every auto-fit NVSwitch launch (#6098).
            uniformly_nvlinked = all(cls._label_is_nvlink(v) for v in matrix.values())
            if not (
                launch_order_pinned
                or os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
                or uniformly_nvlinked
            ):
                return _pcie(
                    "the child's device order is not pinned to PCI_BUS_ID and the "
                    "box is not uniformly NVLinked, so the verified GPUs may not be "
                    "the ones it runs on; set CUDA_DEVICE_ORDER=PCI_BUS_ID to "
                    "enable P2P here"
                )
            # A usable `topo -m` proved nvidia-smi enumerated the selection; NVML does
            # not, and torch ordinals index this matrix wrongly under FASTEST_FIRST (#10613).
            pci_ids = (
                cls._GPU_IDS_ARE_PCI_INDICES is True
                if ids_are_pci_indices is None
                else ids_are_pci_indices
            )
            # As above: every mapping agrees on a uniform box, and stray ids still veto.
            if cls._matrix_is_nvml(matrix) and not pci_ids and not uniformly_nvlinked:
                return _pcie(
                    "the GPU selection came from torch rather than nvidia-smi, so "
                    "its ids are CUDA ordinals and cannot be matched against the "
                    "NVML interconnect matrix"
                )
            selected = sorted(set(gpu_indices))
        else:
            # No explicit selection: the child uses every VISIBLE GPU. nvidia-smi
            # ignores CUDA_VISIBLE_DEVICES, so the matrix covers devices the child
            # never touches, and filtering it to the mask keeps a clean NVLinked
            # pair from being vetoed by a PCIe edge to a hidden device.
            # Sound only when mask and matrix share an index space: numeric mask
            # entries are CUDA ordinals, the matrix is PCI indices, and under
            # FASTEST_FIRST mask "0,1" can mean physical 0,2, which could confirm
            # NV# for a pair that is not the one in use (_cuda_compute_caps declines
            # the same mapping). Unmappable means check the whole box.
            visible = cls._resolve_visible_physical_ids()
            mappable = os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"
            selected = (
                [i for i in gpu_ids if i in set(visible)]
                if visible is not None and mappable
                else list(gpu_ids)
            )

        if len(selected) < 2:
            return "fewer than two GPUs resolved in the interconnect matrix"

        for a in selected:
            for b in selected:
                if a == b:
                    continue
                label = matrix.get((a, b))
                if label is None:
                    return f"GPU {a} and GPU {b} are absent from the interconnect matrix"
                if not cls._label_is_nvlink(label):
                    return _pcie(f"GPU {a} to GPU {b} is {label}, not NVLink")
        return None

    @staticmethod
    def _effective_gpu_count(gpu_indices = None) -> int:
        """GPUs llama-server will use: len(selection), else the visible CUDA
        device count (None = every visible GPU). 0 on error so multi-GPU tuning
        stays off when the count is unknown."""
        if gpu_indices is not None:
            return len(gpu_indices)
        try:
            import torch
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                return torch.cuda.device_count()
        except Exception:
            return 0
        return 0

    # Values a user writes to mean "off". ggml tests GGML_CUDA_P2P for PRESENCE, not
    # value, so passing any of these through ENABLES what they turned off; only
    # absence is off. Same trap as GGML_CUDA_ENABLE_UNIFIED_MEMORY (#8651).
    _FALSY_ENV_VALUES = frozenset({"", "0", "false", "off", "no"})

    # Property of the host, not the model, so warn once rather than per launch.
    _warned_no_nvlink = False

    # The loaded template refuses an empty render. Reset on every load.
    _empty_chat_render_refused = False

    @staticmethod
    def _sanitize_p2p_env(env: dict) -> Optional[str]:
        """Drop an inherited GGML_CUDA_P2P that must not reach the child; return the
        removed value for logging, else None. ggml tests the variable for PRESENCE
        and not value, so both cases mean OFF and neither is honoured upstream: a
        falsy value ("0", "false", "off", ""), or UNSLOTH_DISABLE_DC_P2P=1, where
        suppressing our own default is not enough because an inherited truthy value
        would sail past it (#10613).

        Belongs at the call site on EVERY launch path: env is inherited from
        os.environ, and a consumer-GPU user who set the flag off by hand never
        reaches the datacenter gate at all."""
        value = env.get("GGML_CUDA_P2P")
        if value is None:
            return None
        if (
            str(value).strip().lower() in LlamaCppBackend._FALSY_ENV_VALUES
            or os.environ.get("UNSLOTH_DISABLE_DC_P2P") == "1"
        ):
            del env["GGML_CUDA_P2P"]
            logger.info(
                "Dropped inherited GGML_CUDA_P2P=%r from the llama-server "
                "environment: llama.cpp tests this variable for presence, not "
                "value, so passing it through would have ENABLED peer copies "
                "(#10613)",
                value,
            )
            return value
        return None

    @staticmethod
    def _p2p_user_opted_out() -> bool:
        """Whether the USER asked for peer copies off. Read from os.environ, not a
        child env _sanitize_p2p_env may already have cleaned, so the intent survives
        the variable and the default cannot put it back."""
        return (
            os.environ.get("UNSLOTH_DISABLE_DC_P2P") == "1"
            or str(os.environ.get("GGML_CUDA_P2P", "1")).strip().lower()
            in LlamaCppBackend._FALSY_ENV_VALUES
        )

    @staticmethod
    def _apply_datacenter_env(
        env: dict,
        gpu_indices = None,
        p2p_opted_out = False,
        launch_order_pinned = False,
        ids_are_pci_indices = None,
    ) -> bool:
        """Inject DC llama.cpp tuning into env in place via setdefault (user values
        win); return whether the box qualified. Only datacenter NVIDIA parts qualify
        (consumer/ROCm/CPU/error are a no-op); opt out with
        UNSLOTH_DISABLE_DC_TUNING=1. Any qualifying GPU gets
        GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F (FP32 accum: ~0% cost on B200).

        GGML_CUDA_P2P (+33-51% pp tensor-split on B200) additionally needs a
        CONFIRMED NVLink fabric across the selection (_p2p_veto_reason): without one
        the peer copy is silently discarded and every model emits garbage (#10613).
        CUDA_SCALE_LAUNCH_QUEUES=4x (+8-16% pipeline split) is deliberately NOT
        gated with it: it sizes a command buffer, moves no data across the bus, and
        measured clean in isolation on the affected host.

        p2p_opted_out records that the call site already removed a falsy
        user-supplied GGML_CUDA_P2P (_sanitize_p2p_env), so the default must not put
        it straight back."""
        # Defence in depth, idempotent: called directly with a falsy value,
        # setdefault would keep "0" and log "GGML_CUDA_P2P=0 (user)", which reads as
        # off to a human and ON to ggml (#10613).
        if LlamaCppBackend._sanitize_p2p_env(env) is not None:
            p2p_opted_out = True
        if os.environ.get("UNSLOTH_DISABLE_DC_TUNING") == "1":
            return False
        if not LlamaCppBackend._is_datacenter_gpu(gpu_indices):
            return False

        def _apply(name: str, value: str) -> str:
            supplied = name in env
            env.setdefault(name, value)
            return f"{name}={env[name]}" + (" (user)" if supplied else "")

        applied = [_apply("GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F", "1")]
        if LlamaCppBackend._effective_gpu_count(gpu_indices) > 1:
            # Moves no data between GPUs, so it is not gated on the fabric.
            applied.append(_apply("CUDA_SCALE_LAUNCH_QUEUES", "4x"))
            # Asked for off: skip the probe rather than pay up to its 10s timeout to
            # decide something the user already decided.
            veto = (
                "GGML_CUDA_P2P was turned off in the environment"
                if p2p_opted_out
                else LlamaCppBackend._p2p_veto_reason(
                    gpu_indices, launch_order_pinned, ids_are_pci_indices
                )
            )
            if veto is None:
                applied.append(_apply("GGML_CUDA_P2P", "1"))
            elif "GGML_CUDA_P2P" in env:
                # Truthy and user-supplied: their value stands, but it steers
                # straight into the #10613 failure.
                logger.warning(
                    "GGML_CUDA_P2P is set in the environment (%s), so peer copies stay "
                    "ON despite: %s. Unset it entirely -- not =0, which reads as ON "
                    "upstream -- if model output is garbled (#10613).",
                    env["GGML_CUDA_P2P"],
                    veto,
                )
            elif p2p_opted_out or os.environ.get("UNSLOTH_DISABLE_DC_P2P") == "1":
                # Asked for off and got off. The fabric was never inspected, so
                # calling it unconfirmed would warn about corruption on a box that
                # may be perfectly healthy.
                logger.info("GGML_CUDA_P2P left unset: %s", veto)
            elif not LlamaCppBackend._warned_no_nvlink:
                LlamaCppBackend._warned_no_nvlink = True
                logger.warning(
                    "Multi-GPU data-center box without a confirmed NVLink fabric: %s. "
                    "Leaving GGML_CUDA_P2P unset. A PCIe peer copy can be discarded "
                    "silently while CUDA still reports success, which surfaces as "
                    "garbled output rather than an error (#10613). Verify with "
                    "scripts/p2p_integrity_probe.py; if it passes, set "
                    "UNSLOTH_FORCE_DC_P2P=1 to opt back in.",
                    veto,
                )
        logger.info(
            "Data-center GPU detected: applied DC llama.cpp env tuning (%s)",
            ", ".join(applied),
        )
        return True

    @staticmethod
    def _visible_devices_mask(env_name: str) -> Optional[set[int]]:
        """Physical indices a ``*_VISIBLE_DEVICES`` mask permits, or None if unset.

        ``if x.strip()`` filters trailing-comma masks ("0,1,"); an empty mask
        ("") yields an empty set (all devices hidden), distinct from an unset
        var (None, no mask). Used by the nvidia-smi probe.
        """
        raw = os.environ.get(env_name)
        if raw is None:
            return None
        try:
            return set(int(x.strip()) for x in raw.split(",") if x.strip())
        except ValueError:
            if (
                env_name != "CUDA_VISIBLE_DEVICES"
                or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            ):
                return None
        from utils.hardware import nvidia

        ids = nvidia.resolve_uuid_mask(raw.strip())
        return set(ids) if ids is not None else None

    @staticmethod
    def _vulkan_pin_args(gpu_indices: Optional[Iterable[int]]) -> list[str]:
        """``--device Vulkan<i>,...`` to pin a Vulkan launch to selected GPUs.

        The indices are ggml's compact Vulkan ordinals (as _get_gpu_free_memory
        reports and the registry names ``Vulkan<i>``). Pin by that name, NOT via
        GGML_VK_VISIBLE_DEVICES: ggml parses that env var in the raw
        vkEnumeratePhysicalDevices space (before dropping CPU/llvmpipe devices
        and deduplicating ICDs), so a compact ordinal there could select a
        different physical device or the CPU rasterizer.
        """
        if not gpu_indices:
            return []
        return ["--device", ",".join(f"Vulkan{i}" for i in gpu_indices)]
