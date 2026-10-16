# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: lifecycle, crash recovery and health wait."""

import httpx
import threading
import time

from dataclasses import replace
from typing import (
    Callable,
    Optional,
)
from core.inference.llama_server_args import strip_shadowing_flags

from .launch_flags import (
    _child_spec_env,
    _extra_args_requests_mtp,
)
from .log import logger
from .memory_policy import (
    _LLAMA_FIT_TARGET_DEFAULT_MIB,
    _VRAM_FLOOR_RESERVE_MIB,
    _extra_args_set_spec_type,
)
from .spec_args import _canonicalize_spec_mode

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _HealthMixin:
    def _maybe_recover_from_mtp_crash(self, exc: Optional[BaseException] = None) -> bool:
        """Schedule one background reload without MTP after a mid-generation death.

        An MTP launch can pass /health and the load-time fallback, then abort on
        a later decode (MTP+tensor flash-attn, Gemma 4 on Apple Silicon #13253).
        Not a persistent ban: a fresh load re-tries MTP. Returns True if scheduled.
        """
        # Cheap async-safe gate: only our live MTP launch, not cancelled, with a
        # snapshot to replay.
        if self._cancel_event.is_set():
            return False
        if not self._mtp_runtime_fallback_active:
            return False
        # Read before claiming: a raise after the claim strands the flag, and nothing
        # else clears it, blocking every later respawn.
        snapshot = self._last_load_intent
        proc = self._process
        if snapshot is None or proc is None:
            return False
        # Still serving: a transient error, so leave the caller its normal retry.
        if proc.poll() is None and self._server_socket_is_open():
            return False
        # Single-flight: the first failure claims the reload.
        with self._mtp_runtime_fallback_lock:
            if self._mtp_runtime_fallback_in_progress:
                return False
            self._mtp_runtime_fallback_in_progress = True

        def _recover():
            try:
                # Confirm the process really exited (the error can arrive a beat
                # early) so a transient stream error can't disable MTP.
                deadline = time.monotonic() + 5.0
                while proc.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.1)
                if proc.poll() is None:
                    logger.debug("Generation error but llama-server is alive; keeping MTP.")
                    return
                logger.warning(
                    "llama-server exited mid-generation with MTP speculative "
                    "decoding (%s); reloading without speculative decoding.",
                    type(exc).__name__ if exc is not None else "server exited",
                )
                # Re-check under the load lock (RLock allows the nested
                # load_model) so a newer load isn't clobbered by this stale replay.
                requested_mode = snapshot.speculative_type
                with self._serial_load_lock:
                    with self._lock:
                        if self._cancel_event.is_set():
                            logger.info("MTP-crash reload skipped: load was cancelled/unloaded.")
                            return
                        if self._process is not proc:
                            logger.info("MTP-crash reload skipped: a newer load is already active.")
                            return
                        if self._last_load_intent != snapshot:
                            logger.info("MTP-crash reload skipped: load settings changed.")
                            return
                        unload_epoch = self._unload_epoch
                    # Appending --spec-default cannot drop MTP: llama.cpp appends spec
                    # types rather than replacing, so the extras have to lose theirs.
                    # Once they do, Unsloth owns the block and the launch scrubs the
                    # inherited env for the replay too.
                    _ea = list(snapshot.extra_args or ())
                    fallback_extra_args = snapshot.extra_args
                    if _extra_args_requests_mtp(_ea, env = _child_spec_env(_ea)):
                        fallback_extra_args = tuple(
                            strip_shadowing_flags(
                                _ea,
                                strip_context = False,
                                strip_cache = False,
                                strip_spec = True,
                                strip_template = False,
                                strip_split_mode = False,
                            )
                        )
                    fallback = replace(
                        snapshot,
                        speculative_type = "off",
                        extra_args = fallback_extra_args,
                    )
                    started = bool(self.load_model(fallback))
                    if not started:
                        logger.info("MTP-crash reload stopped before the replacement was ready.")
                        return
                    undo_reload = False
                    with self._lock:
                        if self._cancel_event.is_set() or self._unload_epoch != unload_epoch:
                            undo_reload = True
                        else:
                            # Same reason the mode is restored below: the replay launched a
                            # stripped list but the caller keeps sending the original, so a
                            # comparator seeing only the stripped one would reload the very MTP
                            # configuration that just crashed.
                            if fallback_extra_args is not snapshot.extra_args:
                                self._requested_extra_args = (
                                    self._strip_device_extra_args(_ea)
                                    if self._gpu_ids_own_placement(snapshot.gpu_ids)
                                    else list(_ea)
                                )
                            # Restore the requested mode + reason load_model("off") cleared,
                            # so /status shows the user's mode + note (like the startup fallback).
                            # None when the extras own --spec-type, as the duplicate-load match expects.
                            self._requested_spec_mode = (
                                None
                                if _extra_args_set_spec_type(_ea)
                                else _canonicalize_spec_mode(requested_mode)
                            )
                            self._spec_fallback_reason = "runtime_error"
                    if undo_reload:
                        logger.info(
                            "MTP-crash reload undone: the model was unloaded during reload."
                        )
                        self.unload_model()
                        return
                logger.info("Reloaded without MTP after the mid-generation crash.")
            except Exception as e:
                logger.error(f"Reload without MTP failed: {e}")
            finally:
                with self._mtp_runtime_fallback_lock:
                    self._mtp_runtime_fallback_in_progress = False

        try:
            threading.Thread(target = _recover, daemon = True, name = "mtp-crash-reload").start()
        except RuntimeError as exc:
            # Release the claim: a reload that never started would block respawn forever.
            with self._mtp_runtime_fallback_lock:
                self._mtp_runtime_fallback_in_progress = False
            logger.error(f"Could not start the MTP-crash reload: {exc}")
            return False
        return True

    def _start_mtp_crash_watchdog(self) -> None:
        """Background poll that recovers on an MTP crash even when no request
        observes it (direct proxy endpoints, or nothing in flight).

        Armed only for a live MTP launch; the no-MTP reload disarms it, so it
        can't loop.
        """
        if not self._mtp_runtime_fallback_active:
            return
        proc = self._process
        if proc is None:
            return
        # Replace any prior watchdog (loads are serialised, so at most one).
        self._stop_mtp_crash_watchdog()
        stop = threading.Event()
        self._mtp_watchdog_stop = stop

        def _watch():
            # Exit on stop or process death. _kill_process sets stop before
            # terminating, so re-check it: only a real crash (stop unset) recovers.
            while not stop.wait(1.0):
                if proc.poll() is not None:
                    if not stop.is_set():
                        self._maybe_recover_from_mtp_crash()
                    return

        t = threading.Thread(target = _watch, daemon = True, name = "mtp-crash-watchdog")
        self._mtp_watchdog_thread = t
        t.start()

    def _stop_mtp_crash_watchdog(self) -> None:
        """Signal the crash watchdog to exit; called before any deliberate kill."""
        stop = getattr(self, "_mtp_watchdog_stop", None)
        if stop is not None:
            stop.set()
        self._mtp_watchdog_thread = None

    # Extend the health deadline while startup work continues.
    _HEALTH_STALL_TIMEOUT_S = 600.0
    _STARTUP_PROGRESS_SAMPLE_S = 5.0
    # Ignore activity from the health probes themselves.
    _STARTUP_PROGRESS_MIN_BYTES = 1 << 20
    _STARTUP_PROGRESS_MIN_PAGE_FAULTS = 64
    _STARTUP_PROGRESS_MIN_CPU_FRACTION = 0.05

    @staticmethod
    def _page_faults(member, memory) -> int:
        # mmap loading can progress through page faults while RSS stays flat.
        for field in ("pfaults", "num_page_faults"):  # macOS, Windows
            if hasattr(memory, field):
                return getattr(memory, field)
        try:
            with open(f"/proc/{member.pid}/stat", "rb") as f:
                fields = f.read().rsplit(b")", 1)[1].split()
            return int(fields[7]) + int(fields[9])  # minflt + majflt
        except (OSError, IndexError, ValueError):
            return 0

    @classmethod
    def _startup_work_sample(cls, process) -> Optional[tuple[float, int, int, int]]:
        """Return cumulative (CPU, RSS, bytes read, page faults) for the process tree."""
        pid = getattr(process, "pid", None)
        if not isinstance(pid, int):
            return None
        try:
            import psutil
            root = psutil.Process(pid)
            members = [root, *root.children(recursive = True)]
        except Exception:
            return None
        cpu = 0.0
        rss = read = faults = 0
        for member in members:
            try:
                with member.oneshot():
                    times = member.cpu_times()
                    memory = member.memory_info()
                    try:
                        io = member.io_counters()
                    except (AttributeError, psutil.AccessDenied):
                        io = None  # macOS has no per-process I/O counters
                member_faults = cls._page_faults(member, memory)
            except Exception:
                if member is root:
                    return None
                continue
            cpu += times.user + times.system
            rss += memory.rss
            faults += member_faults
            if io is not None:
                read += getattr(io, "read_chars", io.read_bytes)  # Windows has no read_chars
        return (cpu, rss, read, faults)

    @classmethod
    def _made_startup_progress(
        cls, before: tuple[float, int, int, int], after: tuple[float, int, int, int], elapsed: float
    ) -> bool:
        # RSS is a high-water mark so evicted pages are not counted twice.
        return (
            after[1] - before[1] >= cls._STARTUP_PROGRESS_MIN_BYTES
            or after[2] - before[2] >= cls._STARTUP_PROGRESS_MIN_BYTES
            or after[3] - before[3] >= cls._STARTUP_PROGRESS_MIN_PAGE_FAULTS
            or after[0] - before[0] >= cls._STARTUP_PROGRESS_MIN_CPU_FRACTION * elapsed
        )

    def _wait_for_health(
        self,
        timeout: float = 120.0,
        interval: float = 0.5,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """Poll /health until success, exit, cancellation, or a startup stall."""
        deadline = time.monotonic() + timeout
        work_sample = None
        work_sampled_at = None
        work_checked_at = None
        url = f"{self.base_url}/health"
        health_probe_event = getattr(self, "_health_probe_event", None)
        if health_probe_event is None:  # Backward-compatible with lightweight test doubles.
            health_probe_event = self._health_probe_event = threading.Event()

        if cancelled is None:
            # A scoped /load carries its own cancel event, so load_model passes its own
            # predicate; fall back to the unload flag for callers that have none.
            cancel_event = getattr(self, "_cancel_event", None)
            cancelled = cancel_event.is_set if cancel_event is not None else None

        # Why this wait ended, for callers that must tell a cancel apart from a crash:
        # a cancel landing during CPU-fallback staging is not a cancelled wait.
        self._health_wait_cancelled = False
        # No teardown reset here: it would erase one that landed between this load's
        # spawn and this line. _torn_down_process is matched by identity instead.
        process = None  # the child this wait last looked at, read again after the loop

        while (now := time.monotonic()) < deadline:
            # Durable, unlike the per-process marker below: once teardown begins,
            # every later iteration sees it.
            if getattr(self, "_shutting_down", False):
                logger.info("llama-server was torn down while waiting for it to become healthy")
                self._health_wait_cancelled = True
                return False
            # unload_model() blocks on self._lock, which the load holds across this wait.
            if cancelled is not None and cancelled():
                logger.info("llama-server startup cancelled before it became healthy")
                self._health_wait_cancelled = True
                return False
            # Cleared before probing so output during the request stays latched for
            # the fallback wait below.
            health_probe_event.clear()
            # Read once: shutdown clears the reference from another thread (#10353).
            process = self._process
            if process is None:
                logger.info("llama-server was torn down while waiting for it to become healthy")
                # Terminal like a cancel: the caller must not read the cleared
                # reference, nor respawn a server shutdown just killed.
                self._health_wait_cancelled = True
                return False
            # Process crashed?
            if process.poll() is not None:
                # A teardown holds the reference across its waits, so THIS child
                # exiting under it is deliberate.
                if getattr(self, "_torn_down_process", None) is process:
                    logger.info("llama-server was torn down while waiting for it to become healthy")
                    self._health_wait_cancelled = True
                    return False
                # Let the drain thread collect final output.
                if self._stdout_thread is not None:
                    self._stdout_thread.join(timeout = 2)
                output = "\n".join(self._stdout_lines[-50:])
                # Keep the TAIL: crash details (abort reason, ROCm/CUDA error
                # text) print last, after the long startup banner. Head
                # truncation has cut off exactly the diagnostic line before.
                _log_hint = (
                    f" Full log: {self._llama_log_path}"
                    if getattr(self, "_llama_log_path", None)
                    else ""
                )
                logger.error(
                    f"llama-server exited with code {process.returncode}. "
                    f"Output (tail): {output[-2000:]}{_log_hint}"
                )
                return False

            if work_checked_at is None or now - work_checked_at >= self._STARTUP_PROGRESS_SAMPLE_S:
                work_checked_at = now
                sample = self._startup_work_sample(process)
                if sample is not None:
                    if work_sample is not None and self._made_startup_progress(
                        work_sample, sample, now - work_sampled_at
                    ):
                        deadline = now + timeout
                    peak = sample[1] if work_sample is None else max(work_sample[1], sample[1])
                    work_sample = (sample[0], peak, sample[2], sample[3])
                    work_sampled_at = now

            try:
                # trust_env=False: skip ambient HTTP(S)_PROXY, which if it 503s
                # for 127.0.0.1 loops the probe until timeout and hangs load.
                resp = httpx.get(url, timeout = 2.0, trust_env = False)
                if resp.status_code == 200:
                    # The probe blocks for up to 2s; a cancel landing inside that window
                    # would otherwise publish the model the user just asked us to drop.
                    if cancelled is not None and cancelled():
                        logger.info("llama-server became healthy after the load was cancelled")
                        self._health_wait_cancelled = True
                        return False
                    # A 200 arriving as shutdown began must not publish _healthy.
                    if getattr(self, "_shutting_down", False):
                        logger.info("llama-server became healthy while the app was shutting down")
                        self._health_wait_cancelled = True
                        return False
                    return True
            except (
                httpx.ConnectError,
                httpx.TimeoutException,
                # ReadError covers TCP RST mid-read while still binding the port
                # (Windows: WinError 10054); the crash branch catches real exits.
                httpx.ReadError,
                httpx.RemoteProtocolError,
                httpx.WriteError,
            ):
                pass

            health_probe_event.wait(interval)

        if cancelled is not None and cancelled():
            logger.info("llama-server startup cancelled at the health-check deadline")
            self._health_wait_cancelled = True
            return False

        # The deadline is the other way out of the loop, so it asks too -- and about
        # both signals, since _kill_process sets _shutting_down on entry but
        # _torn_down_process only after collecting descendants.
        if getattr(self, "_shutting_down", False) or (
            process is not None and getattr(self, "_torn_down_process", None) is process
        ):
            logger.info("llama-server was torn down while waiting for it to become healthy")
            self._health_wait_cancelled = True
            return False

        # Leave a marker so _classify_llama_start_failure tells a live but
        # never-healthy load (too large, or a proxy hijacking the loopback
        # probe) apart from a bad GGUF (#5740).
        marker = f"llama-server health check timed out: no startup progress for {timeout:g}s"
        self._stdout_lines.append(marker)
        logger.error(marker)
        return False

    @staticmethod
    def _fit_target_margin_mib(
        *,
        auto_fit: bool,
        fit_target_delta_mib: float = 0.0,
        supports_fit_target: bool = True,
        floor_mib: float = 0.0,
    ) -> float:
        """Free VRAM per device llama.cpp's fitter keeps for itself under ``--fit on``.

        One reader for the flag and for the load-mode fit, so the margin the child is
        told to keep and the margin the fit refuses to spend cannot drift. A server
        too old for ``--fit-target`` still keeps llama.cpp's own 1024 MiB default
        (common/common.h, ``fit_params_target``), which is also what the legacy path
        deliberately stays silent about. ``floor_mib`` is a margin the host needs kept
        whatever the budget says (the Metal wired headroom).
        """
        if not supports_fit_target:
            return _LLAMA_FIT_TARGET_DEFAULT_MIB
        base = _VRAM_FLOOR_RESERVE_MIB if auto_fit else _LLAMA_FIT_TARGET_DEFAULT_MIB
        return max(_VRAM_FLOOR_RESERVE_MIB, base + fit_target_delta_mib, floor_mib)

    @staticmethod
    def _ctx_integrity_flags(
        n_parallel: int,
        use_fit: bool,
        auto_fit: bool,
        requested_ctx: int,
        effective_ctx: int,
        caps: dict,
        *,
        fit_target_delta_mib: float = 0.0,
        fit_target_floor_mib: float = 0.0,
    ) -> list[str]:
        """Flags that keep the per-request window equal to the advertised ctx.

        Explicit ``--parallel`` disables llama-server's auto-slots
        ``--kv-unified`` default, silently splitting ``-c`` into per-slot
        windows of ``-c / N``; restore the shared pool so one request can use
        the full context. With ``--fit on``, ``--fit-ctx`` floors the fit step
        at an explicitly requested ctx so it offloads or fails instead of
        silently shrinking the window. The 8192 auto-floor applies only under
        Manual + Auto (``auto_fit``), which omits ``-c``: on the legacy auto path
        ``-c 0`` already pins the native window and ``--fit-ctx 8192`` would
        override it down to 8192. ``--fit-target`` starts from the tighter 512 MiB
        margin there and from llama.cpp's own default on the legacy path, moved by
        whatever the VRAM budget asks for either way (``fit_target_delta_mib``)
        and never below the 512 MiB floor.
        """
        flags: list[str] = []
        if n_parallel > 1 and caps.get("supports_kv_unified"):
            flags.append("--kv-unified")
        if use_fit and caps.get("supports_fit_ctx"):
            if requested_ctx > 0 and effective_ctx > 0:
                # Floor the fit step at the explicitly requested ctx.
                flags.extend(["--fit-ctx", str(effective_ctx)])
            elif auto_fit:
                # Manual + Auto omits -c, so floor at 8192 so --fit doesn't
                # shrink the window below a usable size.
                flags.extend(["--fit-ctx", "8192"])
        if use_fit and caps.get("supports_fit_target"):
            # Each path keeps its own starting margin: the tighter 512 under Manual
            # + Auto, so llama.cpp packs more on before spilling to system RAM, and
            # llama.cpp's own 1024 on the legacy path, which passes nothing today.
            # The budget moves that margin from where the path already sits, so the
            # default changes neither, and a raise stops at the same 512 floor every
            # other reserve here respects.
            _target = LlamaCppBackend._fit_target_margin_mib(
                auto_fit = auto_fit,
                fit_target_delta_mib = fit_target_delta_mib,
                floor_mib = fit_target_floor_mib,
            )
            # The legacy path stays silent when it would only restate llama.cpp's
            # own default, so an untouched slider leaves the command unchanged.
            if auto_fit or _target != _LLAMA_FIT_TARGET_DEFAULT_MIB:
                flags.extend(["--fit-target", str(int(_target))])
        return flags

    def _query_server_props(self) -> Optional[dict]:
        """llama-server's ``/props``, or None when it cannot be read."""
        url = f"{self.base_url}/props"
        try:
            # /props is not one of llama-server's public endpoints, so with the
            # child's --api-key an unauthenticated read 401s: the context readback silently keeps
            # the requested -c, and video reads as unsupported on a model that
            # supports it. None when there is no child key, which is httpx's
            # default and what every other call site here relies on.
            resp = httpx.get(url, headers = self._auth_headers, timeout = 5.0, trust_env = False)
            if resp.status_code != 200:
                return None
            props = resp.json()
            return props if isinstance(props, dict) else None
        except Exception:
            return None

    def _query_server_n_ctx(self) -> Optional[int]:
        """Per-slot context llama-server actually allocated, from ``/props``.

        The memory-fit step or ``--parallel`` slot split can leave this below
        the requested ``-c``; requests are validated against this value.

        Records the declared modalities on the way past: video input depends on
        build flags and ffmpeg, neither visible from the GGUF.
        """
        props = self._query_server_props()
        if props is None:
            return None
        modalities = props.get("modalities")
        if isinstance(modalities, dict):
            self._has_video_input = bool(modalities.get("video"))
        settings = props.get("default_generation_settings") or {}
        n_ctx = settings.get("n_ctx")
        return int(n_ctx) if n_ctx else None

    def _reconcile_effective_ctx_with_server(self, requested_n_ctx: int = 0) -> None:
        """Adopt the server's real ``n_ctx`` within an explicit requested ceiling.

        Keeps ``context_length`` (load response, status route, passthrough
        ``max_tokens`` ceiling) honest; clients sized to the requested value
        would otherwise hit ``exceed_context_size_error`` 400s early. A trailing
        pass-through ``--ctx-size`` can also make the server larger than Studio's
        pre-launch VRAM-fit estimate; publish that confirmed allocation without
        advertising beyond the user's resolved explicit request.

        ``requested_n_ctx`` is the EXPLICIT pass-through override (0 when absent),
        not ``resolve_requested_ctx``'s field fallback: only a flag emitted after
        Studio's own ``-c`` can allocate past the fit, so gating on the flag keeps
        llama.cpp's context padding from being published as an override that does
        not exist.

        ``_max_context_length`` deliberately stays put: /props confirms what was
        ALLOCATED, not that it fits VRAM without spilling, so the "may use system
        RAM" warning is right and ``max_context_length < context_length`` is legal.
        """
        actual_n_ctx = self._query_server_n_ctx()
        if not actual_n_ctx or actual_n_ctx <= 0:
            return
        slots = 1 if self._kv_cache_unified else self.effective_parallel_slots
        self._kv_cache_context_total = actual_n_ctx * slots
        effective_n_ctx = self._effective_context_length
        if not effective_n_ctx:
            # Same ceiling as the raise below, or a missing seed would advertise
            # past the explicit request.
            self._effective_context_length = (
                min(actual_n_ctx, requested_n_ctx) if requested_n_ctx > 0 else actual_n_ctx
            )
            return
        if actual_n_ctx < effective_n_ctx:
            logger.warning(
                "llama-server allocated a smaller per-request context than "
                f"requested ({effective_n_ctx} -> {actual_n_ctx}; "
                "memory fit or --parallel slot split); clients must treat "
                f"{actual_n_ctx} as the real context window."
            )
            self._effective_context_length = actual_n_ctx
            return
        if requested_n_ctx > effective_n_ctx:
            confirmed_n_ctx = min(actual_n_ctx, requested_n_ctx)
            if confirmed_n_ctx > effective_n_ctx:
                logger.info(
                    "llama-server confirmed a larger per-request context from "
                    f"the explicit override ({effective_n_ctx} -> {confirmed_n_ctx})."
                )
                self._effective_context_length = confirmed_n_ctx
