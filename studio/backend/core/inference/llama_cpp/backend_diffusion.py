# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: diffusion runner (DiffusionGemma)."""

import os
import re
import subprocess
import sys
import threading
import time

from pathlib import Path
from typing import (
    Callable,
    List,
    Literal,
    Optional,
)
from utils.child_stdio import utf8_child_env
from utils.gpu_memory_events import invalidate_gpu_memory as _invalidate_gpu_memory
from utils.log_retention import prune_log_dir
from utils.native_path_leases import child_env_without_native_path_secret
from utils.process_lifetime import child_popen_kwargs as _child_popen_kwargs
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)

from .launch_flags import (
    _DIFFUSION_FIT_CTX,
    _DIFFUSION_RUNNER_OVERHEAD_MIB,
    _diffusion_auto_ngl,
    _diffusion_manual_ngl,
    _shim_supports_ngl,
    _visual_server_splits_gpus,
)
from .log import logger
from .memory_policy import (
    _active_vram_fraction,
    _vram_usable_mib,
)
from .placement_args import _metal_device_is_paravirtual
from .server_log import _PARAVIRTUAL_DIFFUSION_NO_NGL_ERROR
from .swa_cache import _swa_cache_path


class _DiffusionRunnerMixin:
    # ── Diffusion runner (DiffusionGemma) ──

    def _find_diffusion_assets(self) -> Optional[tuple[list, str, Optional[str]]]:
        """Resolve how to launch the DiffusionGemma runner: (shim argv prefix,
        visual-server binary, optional extra PYTHONPATH dir for the file override).

        Shim: UNSLOTH_DG_SHIM (a .py file) first, else the installed
        unsloth_zoo.diffusion_studio.shim. Binary: DG_VISUAL_BIN first, else
        alongside llama-server. Returns None if neither can be found.
        """
        import importlib.util

        # Visual-server binary: env override, else next to llama-server or in the
        # install's build/bin (where the prebuilt/installer puts it). .exe on Windows.
        visual_bin = os.environ.get("DG_VISUAL_BIN")
        if not visual_bin:
            name = "llama-diffusion-gemma-visual-server" + (".exe" if os.name == "nt" else "")
            # include_denied: a transiently locked llama-server still pins the
            # install dir so the adjacent visual-server can be found
            base = self._find_llama_server_binary(include_denied = True)
            if base:
                base_dir = Path(base).parent
                for cand in (
                    base_dir / name,
                    base_dir / "build" / "bin" / name,
                    base_dir / "build" / "bin" / "Release" / name,
                ):
                    if cand.is_file():
                        visual_bin = str(cand)
                        break
        if not (visual_bin and Path(visual_bin).is_file()):
            return None

        # Shim: a file override (its dir goes on PYTHONPATH), else the zoo package via -m.
        shim_file = os.environ.get("UNSLOTH_DG_SHIM")
        if shim_file and Path(shim_file).is_file():
            return ([sys.executable, shim_file], visual_bin, str(Path(shim_file).parent))

        # Find the installed shim without importing the heavy unsloth_zoo package
        # (find_spec on the top-level package does not run its __init__).
        try:
            spec = importlib.util.find_spec("unsloth_zoo")
        except Exception:
            spec = None
        if spec is not None and spec.submodule_search_locations:
            pkg_dir = Path(list(spec.submodule_search_locations)[0])
            if (pkg_dir / "diffusion_studio" / "shim.py").is_file():
                return (
                    [sys.executable, "-m", "unsloth_zoo.diffusion_studio.shim"],
                    visual_bin,
                    None,
                )

        return None

    def diffusion_split_supported(self) -> bool:
        """Whether a diffusion launch right now would honour --ngl.

        Mirrors the launch gate for the training guard: a split the launcher will drop
        must be guarded as GPU-resident. False when no shim resolves or the probe fails,
        matching the launcher's own fallback."""
        try:
            assets = self._find_diffusion_assets()
        except Exception:
            return False
        if assets is None:
            return False
        shim_cmd, _visual_bin, _extra_pythonpath = assets
        return _shim_supports_ngl(shim_cmd)

    @staticmethod
    def _diffusion_gpu_arg(
        gpu_ids: Optional[List[int]],
        *,
        cpu_only: bool = False,
        force_cpu: bool = False,
    ) -> str:
        """Device token passed to the diffusion visual-server child.

        The visual engine replaces its child's CUDA visibility mask with this
        token, so an unpinned load must carry forward the first token from the
        parent's mask rather than turning a parent-relative ordinal into a new
        physical selection.
        """
        # force_cpu alone outranks the picker: zero GPU layers place nothing, and a real
        # device would let the child hold a context while _gpu_offload_active reports
        # False, making the training VRAM coordinator skip the unload. cpu_only stays
        # *below* the picker: an explicit pick wins on a host whose GPU torch cannot see.
        if force_cpu:
            return ""
        if gpu_ids:
            return str(sorted(gpu_ids)[0])
        if cpu_only:
            return ""
        if "DG_GPU" in os.environ:
            return os.environ["DG_GPU"]
        parent_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        if parent_mask:
            first = next((token.strip() for token in parent_mask.split(",") if token.strip()), "")
            if first and first != "-1":
                return first
        return "0"

    @classmethod
    def _diffusion_fit_gpus(
        cls,
        model_size_bytes: int,
        gpu_ids: Optional[List[int]],
        gpu: str,
        visual_bin: Optional[str],
    ) -> Optional[List[int]]:
        """Physical GPUs to layer-split a diffusion GGUF over, or None to keep ``gpu`` alone.

        The visual server loads with llama.cpp's default layer split over every device in
        its mask, so widening the mask is the whole change. Only a GGUF that does not fit
        ``gpu`` moves: alone it overflows the card (a hang on Windows, where it spills into
        shared memory, #6347). The pool is the explicit pick, else every visible GPU; when
        no subset holds it, None leaves the single-device GPU / CPU layer fit in charge.
        Each device is charged the runner's own context on top of its weights, the same fit
        test as _diffusion_auto_ngl, so the two never disagree on whether ``gpu`` suffices.
        Shared with the training guard, so both budget the devices the runner will use. A
        runner that cannot split keeps today's single device."""
        if not gpu.isdigit() or not model_size_bytes or not _visual_server_splits_gpus(visual_bin):
            return None
        # Unpinned: only ids the child resolves the same way under PCI order (no inherited
        # mask to reinterpret, ids from nvidia-smi / NVML), as for llama-server auto fits.
        if not gpu_ids and ("DG_GPU" in os.environ or "CUDA_VISIBLE_DEVICES" in os.environ):
            return None
        try:
            binary = cls._find_llama_server_binary()
            # Vulkan ordinals are not the CUDA ids the runner's mask takes.
            if binary and cls._is_vulkan_backend(binary):
                return None
            memory = cls._get_gpu_memory(binary)
        except Exception as e:
            logger.debug(f"DiffusionGemma GPU fit skipped: {e}")
            return None
        if not gpu_ids and cls._GPU_IDS_ARE_PCI_INDICES is not True:
            return None
        pool = set(int(i) for i in gpu_ids) if gpu_ids else None
        gpus = [(idx, free) for idx, free, _ in memory if pool is None or idx in pool]
        total_by_idx = {idx: total for idx, _, total in memory}
        first = [(idx, free) for idx, free in gpus if idx == int(gpu)]
        if len(gpus) < 2 or not first:
            return None
        overhead = int(_DIFFUSION_RUNNER_OVERHEAD_MIB * 1024 * 1024)
        need = model_size_bytes + overhead
        if cls._select_gpus(need, first, total_by_idx = total_by_idx)[0]:
            return None
        picked, _ = cls._select_gpus(
            need, gpus, total_by_idx = total_by_idx, per_device_overhead_bytes = overhead
        )
        if not picked or sorted(picked) == [int(gpu)]:
            return None
        return sorted(picked)

    def _diffusion_fit_ngl(
        self, model_path: str, gpu: str, maxtok: int, *, pinned: bool
    ) -> Optional[int]:
        """Auto-mode ``--ngl`` for the runner's device ``gpu``, or None to keep its default."""
        # An NGL in the environment is the zoo runner's own override: leave it in charge.
        if not gpu.isdigit() or "NGL" in os.environ:
            return None
        try:
            binary = self._find_llama_server_binary()
            # Vulkan ordinals are not the CUDA ids the runner's mask takes.
            if binary and self._is_vulkan_backend(binary):
                return None
            rows = self._get_gpu_memory(binary)
            file_bytes = self._get_gguf_size_bytes(model_path)
            sysmem_fallback = self._sysmem_fallback_risk(binary)
        except Exception as e:
            logger.debug(f"DiffusionGemma layer fit skipped: {e}")
            return None
        row = next((r for r in rows if r[0] == int(gpu)), None)
        # Unpinned, the child's "0" is CUDA's fastest-first device unless PCI order is set,
        # not nvidia-smi's index 0: size against the roomiest card so no load that fits
        # today is cut down.
        if not pinned and len(rows) > 1 and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
            row = max(
                rows,
                key = lambda r: _vram_usable_mib(
                    r[1], r[2], _active_vram_fraction(), sysmem_fallback = sysmem_fallback
                ),
            )
        if row is None:
            return None
        return _diffusion_auto_ngl(
            file_bytes,
            self._n_layers,
            self._n_heads,
            maxtok,
            row[1],
            row[2],
            sysmem_fallback = sysmem_fallback,
        )

    def _start_diffusion_server(
        self,
        *,
        model_path: str,
        gguf_path: Optional[str],
        hf_repo: Optional[str],
        hf_variant: Optional[str],
        model_identifier: str,
        n_ctx: int,
        extra_args: Optional[List[str]],
        gpu_ids: Optional[List[int]] = None,
        gpu_memory_mode: Literal["auto", "manual"] = "auto",
        gpu_layers: int = -1,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """Launch the OpenAI-compat diffusion shim (which drives the on-device
        visual decoder) and wait for health. Presents the same /v1 + /health
        interface as llama-server, so the rest of Unsloth is unchanged.

        Manual mode forwards gpu_layers as --ngl so a GGUF larger than VRAM can be
        split across GPU and RAM; auto keeps every layer on the GPU unless the GGUF
        cannot fit it."""
        assets = self._find_diffusion_assets()
        if assets is None:
            raise RuntimeError(
                "DiffusionGemma runner not found. Install unsloth_zoo (which ships "
                "unsloth_zoo.diffusion_studio.shim) or set UNSLOTH_DG_SHIM to a shim "
                "file, and provide the visual-server binary via DG_VISUAL_BIN or next "
                "to llama-server in the install tree."
            )
        shim_cmd, visual_bin, extra_pythonpath = assets
        self._diffusion_visual_bin = visual_bin

        self._kill_process()
        self._port = self._find_free_port()
        # Auto-size (0): the visual server probes the largest context that fits this GPU's VRAM
        # (capped at the training context). An explicit in-range n_ctx overrides it.
        maxtok = n_ctx if (n_ctx and 0 < n_ctx <= 65536) else 0
        # Manual mode: honour the picked split. Without it the runner pins every layer
        # to GPU and a GGUF larger than VRAM dies in cudaMalloc even at 0 layers
        # (#7574). An older shim has no --ngl, so drop it rather than die in argparse.
        manual_ngl = _diffusion_manual_ngl(gpu_memory_mode, gpu_layers)
        if manual_ngl is not None and not _shim_supports_ngl(shim_cmd):
            # Dropping a zero-layer split on virtualised Metal undoes the CPU pin:
            # cpu_only below is torch.cuda only, so it reads 0 on a Mac and the empty
            # --gpu token still lets the runner use Metal. Only the split keeps weights
            # off it, so refuse rather than serve possibly corrupt output.
            if manual_ngl == 0 and _metal_device_is_paravirtual():
                raise ValueError(_PARAVIRTUAL_DIFFUSION_NO_NGL_ERROR)
            logger.warning(
                "DiffusionGemma: ignoring the %d GPU-layer split; the installed "
                "unsloth_zoo shim has no --ngl. Upgrade unsloth_zoo to set the split.",
                manual_ngl,
            )
            manual_ngl = None
        # No visible CUDA GPU: a genuine CPU host, or a GPU host masked with
        # CUDA_VISIBLE_DEVICES="" to force CPU serving. Keep the visual-server child
        # CPU-masked (empty --gpu) so the shim does not re-expose GPU 0 via its default.
        cpu_only = self._effective_gpu_count() == 0
        # Stronger than cpu_only and deliberately not derived from it: _effective_gpu_count
        # is torch.cuda only, so it reads 0 on Metal, Vulkan, Windows-HIP or Intel XPU hosts
        # the visual server will happily use. Only an explicit zero-layer split keeps the
        # weights off every backend, and a real --gpu token there would let the child hold a
        # CUDA context while _gpu_offload_active reports False, which the coordinator trusts.
        holds_no_gpu = manual_ngl == 0
        # Honor the GPU picker first: start from the lowest selected GPU. With no pick,
        # fall back to the CPU-only mask, else DG_GPU / 0. A manual split already says
        # where the layers go; otherwise a GGUF too big for that one device is
        # layer-split over the pool (#6347).
        gpu = self._diffusion_gpu_arg(gpu_ids, cpu_only = cpu_only, force_cpu = holds_no_gpu)
        # Auto: a GGUF too large for that device is layer-split over the GPU pool when the
        # runner supports it (#6347), else gets the GPU / CPU split that fits it (#6456).
        fit_ngl = None
        fit_gpus = None
        if manual_ngl is None:
            fit_ngl = self._diffusion_fit_ngl(model_path, gpu, maxtok, pinned = bool(gpu_ids))
            try:
                fit_gpus = self._diffusion_fit_gpus(
                    self._get_gguf_size_bytes(model_path),
                    gpu_ids,
                    gpu,
                    visual_bin,
                )
            except OSError as e:
                logger.debug(f"DiffusionGemma GPU fit skipped: {e}")
            # The probe runs under the load lock and can be slow: honour a Stop that came in.
            if cancelled is not None and cancelled():
                logger.info("Load cancelled before diffusion server start")
                self._health_wait_cancelled = True
                return False
            if fit_gpus:
                fit_ngl = None
                gpu = ",".join(str(i) for i in fit_gpus)
                logger.info(
                    f"DiffusionGemma does not fit one GPU; splitting layers over GPUs {gpu}"
                )
            if fit_ngl is not None and not _shim_supports_ngl(shim_cmd):
                logger.warning(
                    "DiffusionGemma does not fit GPU %s and the installed unsloth_zoo shim "
                    "has no --ngl to split it. Run 'unsloth studio update'.",
                    gpu,
                )
                fit_ngl = None
            # Pin the context the split was sized for. Left to auto-size, the runner's second
            # probe pass sizes against host RAM, which on Windows the driver backs with
            # shared memory instead of refusing.
            if fit_ngl is not None and not 0 < maxtok <= 8192:
                maxtok = _DIFFUSION_FIT_CTX
            # Nothing fits: run on the CPU like a manual zero-layer split, not on a
            # GPU context the card may not have room for.
            if fit_ngl == 0:
                holds_no_gpu = True
                gpu = self._diffusion_gpu_arg(gpu_ids, cpu_only = cpu_only, force_cpu = True)

        cmd = list(shim_cmd) + [
            "--gguf",
            model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
            "--gpu",
            gpu,
            "--maxtok",
            str(maxtok),
        ]
        if manual_ngl is not None:
            cmd += ["--ngl", str(manual_ngl)]
            # Block diffusion is compute-bound, so CPU-side layers cost far more than on an
            # autoregressive model (~6 tok/s vs ~31 for non-diffusion Gemma). Make the
            # tradeoff visible instead of swapping an OOM for a slow model.
            logger.warning(
                "DiffusionGemma: %d GPU layers requested. Layers left on CPU are much "
                "slower for block diffusion than for a regular model; prefer a smaller "
                "quant over a large CPU split if throughput matters.",
                manual_ngl,
            )
        elif fit_ngl is not None:
            cmd += ["--ngl", str(fit_ngl)]
            logger.warning(
                "DiffusionGemma does not fit GPU %s: keeping %d of %s layers on it and the "
                "rest on CPU, which is much slower for block diffusion. A smaller quant "
                "that fits the GPU is faster.",
                gpu,
                fit_ngl,
                self._n_layers,
            )

        env = child_env_without_native_path_secret()
        # `python -m unsloth_zoo.diffusion_studio.shim` imports unsloth_zoo, which
        # refuses to load unless UNSLOTH_IS_PRESENT is set (normally by `import
        # unsloth`). The shim never imports unsloth, so set it here as unsloth does.
        env["UNSLOTH_IS_PRESENT"] = "1"
        # The shim's `import unsloth_zoo` aborts in get_device_type() ("needs a GPU")
        # when no accelerator is visible, even though it only drives the CPU
        # visual-server binary and does no torch GPU work. Allow the CPU device so the
        # runner starts; the visual server still runs on the CPU llama.cpp build.
        if cpu_only:
            env.setdefault("UNSLOTH_ALLOW_CPU", "1")
        env["DG_VISUAL_BIN"] = visual_bin
        env["DG_GPU"] = gpu
        if gpu_ids or fit_gpus:
            # The visual server remasks via CUDA_VISIBLE_DEVICES=<gpu>; pin PCI
            # order (as the llama-server path does) so the picked physical id maps
            # to the GPU the picker showed, not CUDA's default fastest-first order.
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # The file-override shim imports its sibling visual_engine; put its dir on PYTHONPATH.
        # (The zoo-package shim is an installed module and needs no PYTHONPATH change.)
        if extra_pythonpath:
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                (extra_pythonpath + os.pathsep + existing) if existing else extra_pythonpath
            )

        logger.info(f"Starting DiffusionGemma runner: {' '.join(cmd)}")
        self._stdout_lines = []
        self._llama_log_fh = None
        self._llama_log_path = None
        try:
            log_dir = _swa_cache_path().parent / "logs" / "diffusion-server"
            log_dir.mkdir(parents = True, exist_ok = True)
            self._llama_log_path = log_dir / f"diffusion-{int(time.time())}-port-{self._port}.log"
            self._llama_log_fh = open(self._llama_log_path, "w", encoding = "utf-8", buffering = 1)
            # After the open, so the cap counts this file and can never delete it.
            prune_log_dir(log_dir, "diffusion-*.log", protect = self._llama_log_path)
            logger.info(f"diffusion runner stdout/stderr -> {self._llama_log_path}")
        except (OSError, UnicodeDecodeError) as e:
            logger.debug(f"Could not open diffusion runner log file: {e}")

        # The shim (and its visual server) die with this backend process, so a
        # Unsloth crash/restart never orphans a GPU process.
        # Own Popen, and no parent-death backstop on every platform, so a runner
        # started after the shutdown sweep outlives it.
        with self._spawn_lock:
            if self._spawn_is_stale():
                logger.info("app is shutting down; not starting the diffusion runner")
                self._close_attempt_log()
                self._health_wait_cancelled = True
                return False
            _spawned = subprocess.Popen(
                cmd,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                env = utf8_child_env(env),
                # Deliberately NOT start_new_session, as with the component
                # installer: the desktop stops this backend by signalling its
                # process group and force-kills it after five seconds, so a session
                # of its own would leave the shim and the visual server holding the
                # GPU until the next launch sweeps them.
                **_windows_hidden_subprocess_kwargs(),
                **_child_popen_kwargs(),
            )
            self._process = _spawned
            _invalidate_gpu_memory("llama-server spawned")
            # macOS has no parent-death signal, so only this record reaps a runner
            # holding the GPU. Under the lock: adopting after the sweep re-adds a
            # pid it just forgot.
            try:
                from utils.process_lifetime import adopt_pid
                adopt_pid(_spawned.pid)
            except Exception as e:
                logger.debug(f"Could not track diffusion runner for lifetime sweep: {e}")
        # Same post-adoption recheck as the llama-server spawns: this backend can be
        # helper-owned, which run.py's singleton teardown never marks, so a latch set
        # after the in-lock check would otherwise leave the shim and the visual server
        # alive for the whole health wait below.
        if self._spawn_is_stale():
            logger.info("shutdown began during the spawn; killing the new diffusion runner")
            self._kill_process()
            self._close_attempt_log()
            self._health_wait_cancelled = True
            return False
        self._stdout_thread = threading.Thread(
            target = self._drain_stdout, daemon = True, name = "diffusion-stdout"
        )
        self._stdout_thread.start()

        # Publish state before the health wait (mirrors the llama-server path).
        self._gguf_path = model_path
        self._gguf_load_identity = self._gguf_load_source_identity(model_path)
        self._hf_repo = hf_repo
        self._is_vision = False
        # Clear the prior GGUF's toggle too: a diffusion model must not report the
        # last model's vision state.
        self._disable_vision = False
        self._vision_disabled_by_user = False
        self._is_audio = False  # clear any prior TTS/audio model's routing flag
        self._model_identifier = model_identifier
        self._cache_type_kv = None
        self._reasoning_budget = -1
        self._reasoning_budget_message = ""
        self._requested_reasoning_budget = -1
        self._requested_reasoning_budget_message = ""
        self._swa_full = False
        self._kv_cache_unified = False
        self._memory_state = None
        self._memory_direct_io = None
        self._memory_dio_applicable = False
        self._memory_dio_flags = []
        self._memory_dio_user_tokens = []
        self._memory_policy_active = False
        self._memory_policy_extras_touched = False
        self._memory_mlock_applicable = True
        self._fit_load_mode_flags = []
        self._memory_pending_launch = None
        self._n_ubatch = self._DEFAULT_N_UBATCH
        self._requested_n_batch = None
        self._requested_n_ubatch = None
        # The diffusion runner builds its own command and passes none of these,
        # so a previous GGUF's values must not be reported against it.
        self._requested_load_mode = None
        self._requested_spec_draft_cache_type = None
        self._requested_ctx_checkpoints = None
        self._requested_cache_ram = None
        self._idle_slot_clearing_active = False
        self._flash_attn_enabled = True
        self._effective_cache_types = ("f16", "f16")
        self._requested_cache_types = ("f16", "f16")
        self._kv_cache_context_total = None
        # False means "confirmed to hold no VRAM" and makes the training coordinator skip
        # the unload, so it is claimed only for a zero-layer split.
        self._gpu_offload_active = not holds_no_gpu
        # Diffusion uses only the layer split of the llama.cpp GPU-memory knobs: record what
        # was applied (so /load, /status and reload dedup agree with the child) and reset the
        # rest so a previous GGUF's manual settings are not reported.
        self._gpu_memory_mode = gpu_memory_mode if manual_ngl is not None else "auto"
        self._gpu_layers = manual_ngl if manual_ngl is not None else -1
        # What was ASKED for, not always what was applied (no --ngl downgrades it above).
        # /status reports reality via _gpu_layers, but the dedup guards must compare like
        # with like or the same request reloads on every /load.
        self._diffusion_requested_ngl = _diffusion_manual_ngl(gpu_memory_mode, gpu_layers)
        self._n_cpu_moe = 0
        self._tensor_split = None
        self._auto_tensor_split = None
        self._auto_tensor_split_emitted = None
        self._spill_plan_flags = []
        # Diffusion is never tensor-parallel; clear any state left by a prior TP
        # chat load (load_model phase 1 only kills the process, it doesn't run
        # the unload reset) so /status doesn't misreport TP and an identical
        # re-Apply doesn't reload against stale tensor-parallel state.
        self._tensor_parallel = False
        # Same reason, sharper consequence: a diffusion runner DOES hold VRAM, so a
        # leftover arch-gate flag would answer holds_no_vram True and let the arbiter
        # run a competing workload into its memory (#7624).
        self._arch_gate_forced_cpu = False
        # Record the devices the runner got: the lowest selected GPU, or the split above.
        # Unused members of a pick that fits one device are dropped, or /status misreports
        # placement and the dedup reloads. Clears any explicit pin from a prior chat load.
        # A zero-layer split on virtualised Metal masks the child's devices outright, so
        # it pins nothing: record no effective pin, or /status names a device the runner
        # never used and clearing that pick reloads the same CPU runner.
        _pv_diffusion_cpu = manual_ngl == 0 and _metal_device_is_paravirtual()
        _pinned = (fit_gpus or [sorted(gpu_ids)[0]]) if gpu_ids else None
        self._gpu_ids = _pinned if not _pv_diffusion_cpu else None
        # The frontend prefers requested_gpu_ids when hydrating the picker.
        self._requested_gpu_ids = (
            sorted(int(i) for i in gpu_ids) if gpu_ids and fit_gpus else _pinned
        )
        if hf_variant:
            self._hf_variant = hf_variant
        elif gguf_path:
            try:
                from utils.models.model_config import _extract_quant_label
                self._hf_variant = _extract_quant_label(gguf_path)
            except Exception:
                self._hf_variant = None
        else:
            self._hf_variant = None
        # Provisional until the server reports the budget it resolved (auto-size picks it from VRAM).
        self._effective_context_length = maxtok or self._context_length
        self._max_context_length = self._context_length or maxtok or None
        self._vram_fit_context_length = None

        healthy = self._wait_for_health(timeout = self._HEALTH_STALL_TIMEOUT_S, cancelled = cancelled)
        if healthy:
            if not self._publish_healthy():
                # A teardown between the probe and this commit is already killing
                # the runner; publishing would advertise a server that is gone.
                self._kill_process()
                return False
            self._gpu_offload_active = not holds_no_gpu
            if extra_args is not None:
                self._extra_args = list(extra_args)
                # No spec block to drop here, so launched IS requested; rebind so a
                # prior GGUF load's record cannot outlive it.
                self._requested_extra_args = list(extra_args)
                self._extra_args_source = (model_identifier, hf_variant)
            # The visual server logs "MAXTOK=<N>" with the context budget it actually resolved
            # (auto-sized to VRAM). Read it back so the UI context bar shows the real budget.
            chosen = maxtok
            try:
                for _ln in reversed(self._stdout_lines):
                    _m = re.search(r"MAXTOK=(\d+)", _ln)
                    if _m:
                        chosen = int(_m.group(1))
                        break
            except Exception:
                pass
            if chosen and chosen > 0:
                self._effective_context_length = chosen
                self._max_context_length = chosen
            self._requested_n_ctx = int(n_ctx)
            if cancelled is not None and cancelled():
                logger.info("DiffusionGemma start cancelled after it became healthy")
                self._healthy = False
                self._kill_process()
                return False
        else:
            self._healthy = False
            if getattr(self, "_health_wait_cancelled", False):
                # An automatic switch cancels without unloading, so nothing else reaps
                # the shim and the visual server: they would keep loading and holding
                # memory until some later teardown.
                logger.info("DiffusionGemma start cancelled; stopping the runner")
                self._kill_process()
            else:
                logger.error("DiffusionGemma runner failed to become healthy")
        return healthy
