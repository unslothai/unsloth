# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: lifecycle, unload, process teardown and orphan cleanup."""

import contextlib
import logging
import os
import signal
import socket
import subprocess
import sys
import time

from pathlib import Path
from typing import Optional
from utils.gpu_memory_events import invalidates_gpu_memory as _invalidates_gpu_memory
from utils.process_lifetime import is_signalable_pid as _is_signalable_pid

from .binary_runtime import _is_external_link
from .core_types import _serving_backends_snapshot
from .launch_support import (
    _PROC_ROOT,
    _QUIT_SIGNAL_EXIT_GRACE_S,
    _SIGKILL_REAP_TIMEOUT_S,
    _expanded_user_path,
)
from .log import logger

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _ShutdownMixin:
    def load_cancelled(self) -> bool:
        """True if a load was cancelled (e.g. via unload/_cancel_event) and not
        yet consumed by the next load_model. Lets the tensor->layer fallback
        avoid restarting a load the user just cancelled."""
        return self._cancel_event.is_set()

    @_invalidates_gpu_memory("llama.cpp unload")
    def unload_model(self) -> bool:
        """Terminate the subprocess and cancel any in-flight download."""
        self._cancel_event.set()
        with self._lock:
            self._unload_epoch += 1
            # Read before the kill clears it: callers unload from a finally either way.
            _was_resident = self._process is not None
            self._kill_process()
            self._cleanup_cpu_fallback_runtime()
            if self._api_key_file is not None:
                with contextlib.suppress(OSError):
                    os.unlink(self._api_key_file)
                self._api_key_file = None
            # The one unload line: routes/inference.py logged a second, differently named.
            if _was_resident:
                logger.info(f"Unloaded GGUF model: {self._model_identifier}")
            self._model_identifier = None
            self._gguf_path = None
            self._gguf_load_identity = None
            self._hf_repo = None
            self._mtp_draft_path = None
            self._mtp_draft_suppressed_path = None
            self._mtp_draft_suppressed_reason = None
            self._spec_fallback_reason = None

            self._mmproj_fallback_reason = None
            self._capability_probe_inconclusive = False
            self._spec_drafter_kind = None
            self._dspark_sidecar_absent = False
            self._dflash_sidecar_absent = False
            self._dflash_retry_needed = False
            self._launch_binary_revision = ()
            self._cpu_fallback_reason = None
            self._last_load_intent = None
            self._mtp_runtime_fallback_active = False
            self._hf_variant = None
            self._is_vision = False
            self._disable_vision = False
            self._vision_disabled_by_user = False
            self._is_audio = False
            self._audio_type = None
            self._audio_probed = False
            self._has_audio_input = False
            self._has_video_input = False
            self._mmproj_has_audio = False
            self._mmproj_accepts_image = True
            self._port = None
            self._healthy = False
            self._context_length = None
            self._effective_context_length = None
            self._max_context_length = None
            self._vram_fit_context_length = None
            self._reset_effective_parallel_slots()
            self._slot_save_dir = None
            self._slot_save_binary = None
            self._slot_loaded_identity = None
            self._prompt_cache_disabled = False
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
            self._memory_pending_launch = None
            self._vram_fraction_pending = None
            self._loaded_by_user_action = False
            self._n_ubatch = self._DEFAULT_N_UBATCH
            self._flash_attn_enabled = True
            self._effective_cache_types = ("f16", "f16")
            self._requested_cache_types = ("f16", "f16")
            self._kv_cache_context_total = None
            self._chat_template = None
            self._markup_tokens = []
            self._markup_profile = None
            self._chat_template_override = None
            self._empty_chat_render_refused = False
            self._supports_reasoning = False
            self._reasoning_always_on = False
            self._reasoning_budget = -1
            self._reasoning_budget_message = ""
            self._requested_reasoning_budget = -1
            self._requested_reasoning_budget_message = ""
            self._reasoning_style = "enable_thinking"
            self._reasoning_effort_levels = []
            self._reasoning_default = True
            self._supports_preserve_thinking = False
            self._preserve_thinking_default = False
            self._supports_tools = False
            self._cache_type_kv = None
            # GPU-pin state describes the active runner only; clear it so an explicit
            # pin never leaks into the next (or diffusion) runner.
            self._gpu_ids = None
            self._requested_gpu_ids = None
            self._tensor_parallel = False
            self._gpu_memory_mode = "auto"
            self._gpu_layers = -1
            self._n_cpu_moe = 0
            self._tensor_split = None
            self._auto_tensor_split = None
            self._auto_tensor_split_emitted = None
            self._arch_gate_forced_cpu = False
            self._layer_preserves_tensor_intent = False
            self._speculative_type = None
            self._requested_spec_mode = None
            self._spec_draft_n_max = None
            self._n_layers = None
            self._n_experts = None
            self._leading_dense_block_count = None
            self._n_kv_heads = None
            self._n_kv_heads_by_layer = None
            self._n_heads = None
            self._embedding_length = None
            self._pooling_type = None
            self._kv_key_length = None
            self._kv_value_length = None
            self._sliding_window = None
            self._sliding_window_pattern = None
            self._full_attention_interval = None
            self._kv_lora_rank = None
            self._key_length_mla = None
            self._kv_key_length_swa = None
            self._kv_value_length_swa = None
            self._ssm_inner_size = None
            self._ssm_state_size = None
            self._ssm_group_count = None
            self._ssm_conv_kernel = None
            self._kda_head_dim = None
            self._shared_kv_layers = None
            self._nextn_predict_layers = None
            # Clean up temp chat template file.
            if hasattr(self, "_chat_template_file") and self._chat_template_file:
                try:
                    os.unlink(self._chat_template_file.name)
                except Exception:
                    pass
                self._chat_template_file = None
            self._unload_audio_codec()
            return True

    @staticmethod
    def _leading_process_group(pid):
        """The pid's own process group, when it leads one. None otherwise.

        `not pid` rejects 0 and None but not 1, and `getpgid(1) == 1`, so init
        reads as a group leader here and the caller below would then broadcast
        SIGKILL to every process the user owns. The pid comes from a live Popen
        today so it cannot be 1, but the floor is a line and the consequence of
        losing that property later is not recoverable.
        """
        if not _is_signalable_pid(pid):
            return None
        if os.name != "posix" or not hasattr(os, "getpgid"):
            return None
        try:
            pgid = os.getpgid(pid)
        except OSError:
            return None
        return pgid if pgid == pid else None

    @staticmethod
    def _quit_signal_grace_left(pid) -> Optional[float]:
        if not _is_signalable_pid(pid) or not hasattr(os, "getpgid"):
            return None
        from utils.process_lifetime import seconds_since_quit_signal

        elapsed = seconds_since_quit_signal()
        try:
            if elapsed is None or os.getpgid(pid) != os.getpgrp():
                return None
        except OSError:
            return None
        return max(0.0, _QUIT_SIGNAL_EXIT_GRACE_S - elapsed)

    def _still_listening(self) -> bool:
        port = getattr(self, "_port", None)
        if not port:
            return True
        try:
            with socket.create_connection(("127.0.0.1", port), timeout = 0.5):
                return True
        except ConnectionRefusedError:
            return False
        except OSError:
            return True

    @staticmethod
    def _kill_process_group(pgid):
        """Take down what the leader left behind, if anything is still there."""
        if not _is_signalable_pid(pgid):
            return  # killpg(1) is kill(-1); killpg(0) is our own group
        if not hasattr(os, "killpg"):
            return
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass

    @staticmethod
    def _collect_descendants(pid):
        """`(children, known)` for the kill below.

        `known` is False when the walk could not be made at all: no Toolhelp snapshot, or
        a root whose own identity cannot be read. An empty list then means "not
        enumerable", not "none", and the difference decides whether this unload may drop
        the record and the pidfile. Treating it as "none" terminates the leader, finds no
        survivors and deletes the only handles on whatever the failed walk did not list.
        """
        try:
            from utils.process_lifetime import collect_descendants_known
            return collect_descendants_known(pid)
        except Exception:
            # The import or the walk itself failed, which says nothing about the tree.
            return [], False

    @staticmethod
    def _tree_kill_surviving_process(pid) -> bool:
        """Last resort for a server still alive after terminate. True when it is gone.

        ``owner_verified``: this is reached holding the Popen that spawned the pid, so
        ownership is not in question even when the start time behind the lifetime
        record can no longer be read.
        """
        if not _is_signalable_pid(pid):
            return False
        try:
            from utils.process_lifetime import confirm_pid_exited, terminate_pid
        except Exception:
            return False
        try:
            terminate_pid(pid, timeout = 5.0, owner_verified = True)
        except Exception as e:
            logger.warning(f"Could not terminate surviving llama-server {pid}: {e}")
        try:
            # Not a bare `pid_is_running` read: every kill terminate_pid just made is
            # asynchronous, and asking on the line after one answers its own latency. The
            # two arms below it settle internally now, so this is belt and braces, but a
            # read that decides whether to delete the last handle on a live server is worth
            # spelling the same way everywhere.
            gone = confirm_pid_exited(pid)
        except Exception:
            return False
        if not gone:
            logger.warning(
                f"llama-server {pid} is still running after a tree kill; "
                "keeping its record so the next launch can reap it"
            )
        return gone

    @staticmethod
    def _confirm_group_kill_landed(pid) -> bool:
        """Did the killpg above actually take the leader with it. True when it is gone.

        The cheap half of `_tree_kill_surviving_process`, for the POSIX case where the
        expensive half is a second killpg over a group this teardown has already SIGKILLed.
        No signal is sent from here: the only open question is whether the one already sent
        has finished landing, and kills are asynchronous, so it is answered with a bounded
        poll rather than a single read taken microseconds after the signal.

        False keeps the lifetime record and the pidfile, exactly as a failed tree kill does,
        so a server the group kill could not reach is still reapable by the next launch.
        """
        if not _is_signalable_pid(pid):
            return False
        try:
            from utils.process_lifetime import confirm_pid_exited
        except Exception:
            return False
        try:
            gone = confirm_pid_exited(pid)
        except Exception:
            return False
        if not gone:
            logger.warning(
                f"llama-server {pid} is still running after its process group was killed; "
                "keeping its record so the next launch can reap it"
            )
        return gone

    @staticmethod
    def _terminate_descendants(collected):
        """The diffusion shim's visual server, and anything else it started.

        Returns the survivors as `(pid, identity)`: the ones still running when the sweep
        gave up, each with the creation-time identity the sweep verified. A forced kill can
        simply fail (access denied, a protected process, an uninterruptible driver ioctl),
        and reading the attempt as the outcome is what lets the caller delete the record
        and the pidfile out from under a worker that is still holding the GPU.
        Each survivor is adopted WITH that identity, so it gets a lifetime record of its
        own rather than depending on the leader's: the leader is usually gone by now, and
        on Windows there is no process group standing in for it. The identity travels
        because a survivor can exit between the sweep's last liveness check and the adopt,
        and adopting the bare number would then record a stranger and, where a job object
        is active, put it in a job that kills its members when the app closes.
        """
        if not collected:
            return []
        try:
            from utils.process_lifetime import adopt_pid, terminate_descendants
            survivors = terminate_descendants(collected, timeout = 5.0)
        except Exception as e:
            logger.debug(f"Could not terminate server descendants: {e}")
            # Unknown, not none. The pids were collected, so they can still be named, and
            # naming them is the whole point of this return value.
            return list(collected)
        for pid, identity in survivors:
            try:
                # from_snapshot: these pids came out of a walk, so a None identity here means
                # the collector could not READ one, not that this process has just spawned
                # the child. Capturing one now would record whatever holds the number at this
                # moment, which is the recycled stranger the identity check exists to keep
                # out -- and where a job object is active, put it in a job that kills its
                # members when the app closes.
                adopt_pid(pid, identity, from_snapshot = True)
            except Exception:
                pass
        if survivors:
            logger.warning(
                f"llama-server descendants still running after the sweep: "
                f"{[pid for pid, _ in survivors]}; "
                "recorded so the next launch can reap them"
            )
        return survivors

    def _publish_healthy(self) -> bool:
        """Commit _healthy under the spawn lock, or refuse if this load is stale.

        Under the lock the teardown mark is set with, so only two orders exist:
        publish then teardown (which clears _healthy), or teardown then a refused
        publish.
        """
        with self._spawn_lock:
            if self._spawn_is_stale():
                logger.info("load no longer belongs to this lifecycle; not publishing it healthy")
                return False
            self._healthy = True
            self._planned_vram_mib = dict(getattr(self, "_pending_plan_mib", {}))
            return True

    def _close_attempt_log(
        self,
        *,
        reason: str = "closed",
        exit_code: "Optional[int]" = None,
    ) -> None:
        """Close the per-attempt tee log opened just before a spawn.

        A refusal publishes no process and _kill_process returns early when there is
        none, so nothing else closes it: the next attempt leaks the descriptor and,
        on Windows, holds the file lock an update needs.
        """
        self._llama_log_tee_failed = False
        fh = getattr(self, "_llama_log_fh", None)
        if fh is None:
            return
        if exit_code is None:
            try:
                exit_code = self._process.poll()
            except Exception:
                pass
        try:
            fh.write(f"[studio] attempt end reason={reason} exit_code={exit_code}\n")
        except Exception:
            pass
        try:
            fh.close()
        except Exception:
            pass
        self._llama_log_fh = None
        try:
            logger.info(
                f"attempt log closed reason={reason} exit_code={exit_code} "
                f"path={getattr(self, '_llama_log_path', None)}"
            )
        except Exception:
            # The logger raises once stdout is closed; teardown must not.
            pass

    def _begin_server_lifecycle(self) -> None:
        """Clear shutdown state so a restarted server can launch again.

        The backend is a module singleton and an embedded host may call
        run_server() more than once in one process, so "shutting down" is scoped
        to a lifecycle rather than to the interpreter. Called from run_server
        before anything can spawn.

        """
        # _teardown_lock first, so this waits for an in-progress kill rather than
        # clearing the flag underneath it. Same order as _kill_process.
        with self._teardown_lock:
            with self._spawn_lock:
                self._shutting_down = False
                self._torn_down_process = None

    def _spawn_is_stale(self) -> bool:
        """Whether this load may no longer spawn. Caller holds _spawn_lock."""
        if getattr(self, "_shutting_down", False):
            return True
        # Per-instance state only covers the singleton run.py tears down. A helper or
        # advisor load builds its own backend (hub/utils/llm_assist.py,
        # utils/datasets/llm_assist.py), which nothing marks, so without this it would
        # still spawn a server after the sweep. Read second: the attribute is cheaper
        # and answers for the instance that actually gets torn down.
        from utils.process_lifetime import is_process_shutting_down

        return is_process_shutting_down()

    @_invalidates_gpu_memory("llama.cpp stop")
    def _kill_process(self, *, teardown: bool = False):
        """Terminate the subprocess if running.

        ``teardown`` marks an app-level stop (shutdown, atexit) rather than the
        retry ladder reaping a child it is about to replace: only the former may
        end an in-flight health wait.

        A teardown holds _teardown_lock for the WHOLE kill, because the terminate and
        wait below keep reading self._process and finally clear it: a lifecycle
        reopened mid-kill would have its new child dropped or terminated here, so
        _begin_server_lifecycle takes the same lock and waits.

        _spawn_lock is taken only long enough to set the flag, NOT across the kill.
        A spawn arriving mid-teardown then reads the flag and refuses in microseconds
        instead of queuing behind a SIGTERM/SIGKILL escalation that can run for
        seconds; the load thread it belongs to is one shutdown is already waiting on.
        Marked above the early return, since a quit during a download still has to be
        recorded.
        """
        if teardown:
            # Process-wide as well as per-instance: the atexit teardown reaches here
            # without going through run.py, and the backends a helper load builds for
            # itself are only ever covered by the shared latch.
            from utils.process_lifetime import mark_process_shutting_down

            mark_process_shutting_down()
            with self._teardown_lock:
                with self._spawn_lock:
                    self._shutting_down = True
                self._kill_process_body(teardown = True)
            return
        self._kill_process_body(teardown = False)

    def _kill_process_body(self, *, teardown: bool):
        """The kill itself. Caller holds _teardown_lock when ``teardown``."""
        # Stop the watchdog before a deliberate kill so a planned reload/unload
        # isn't seen as a crash; a real crash never routes through here.
        self._stop_mtp_crash_watchdog()
        self._reset_effective_parallel_slots()
        # Cleared before the early return, not in the finally below, so a stale split never
        # outlives its runner even when there was no process to kill.
        self._diffusion_requested_ngl = None
        self._child_gpu_physical_ids = None
        if self._process is None:
            self._planned_vram_mib = {}
            return
        # Not every _process is a Popen: tests stand one in to mean "a server is
        # loaded" without spawning anything. Terminating what cannot be terminated
        # is not an error worth reporting, but everything else still has to happen,
        # so this skips only the signalling and falls through to the finally rather
        # than returning early and duplicating part of it.
        terminable = hasattr(self._process, "terminate")
        if not terminable:
            logger.debug("no terminable llama-server process to kill; clearing state")
        # Both read before the terminate below: getpgid stops answering once the
        # wait reaps the leader, and the shim's children are reparented the
        # moment it exits. This is the only stop the desktop shutdown waits for,
        # so the visual server has to be named while that link still exists.
        _pid = getattr(self._process, "pid", None)
        _pgid = self._leading_process_group(_pid)
        _descendants, _descendants_known = self._collect_descendants(_pid)
        _exited_first = getattr(self._process, "poll", lambda: None)() is not None
        if teardown:
            # Before the signal, and as the process itself: the reference stays set
            # across the waits below, and only identity says which child a teardown
            # landing between a spawn and its wait referred to.
            self._torn_down_process = self._process
        try:
            if terminable:
                _grace = self._quit_signal_grace_left(_pid)
                if _grace is not None:
                    # The quit signal already reached it; a second one aborts llama-server on Metal.
                    try:
                        self._process.wait(timeout = _grace)
                    except subprocess.TimeoutExpired:
                        # Still listening: our signal never reached it (a stop sent to Studio alone).
                        if not self._still_listening():
                            self._process.kill()
                            self._process.wait(timeout = _SIGKILL_REAP_TIMEOUT_S)
                self._process.terminate()
                self._process.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            # Signal first, report after. logger here is a structlog PrintLogger
            # writing straight to stdout, so a closed stream raises out of the
            # warning; doing it first meant SIGKILL was skipped and the server
            # left running with the reference about to be dropped below.
            self._process.kill()
            # Between the two, deliberately. After kill() so the signal never
            # depends on a log write, and before this wait() because a second
            # TimeoutExpired raised in here is not caught by the handler it is
            # raised from -- it escapes, and a truly unkillable server would then
            # be reported by nothing at all.
            logger.warning("llama-server did not exit on SIGTERM, sent SIGKILL")
            self._process.wait(timeout = _SIGKILL_REAP_TIMEOUT_S)
        except Exception as e:
            logger.warning(f"Error killing llama-server process: {e}")
        finally:
            self._kill_process_group(_pgid)
            _surviving_descendants = self._terminate_descendants(_descendants)
            # getattr: teardown must tolerate a partially-built backend (failed
            # __init__ or a __new__-built instance), as with _llama_log_fh below.
            if getattr(self, "_stats_logger", None) is not None:
                self._stats_logger.stop()
                self._stats_logger = None
            # Drop it from the lifetime record only once it is confirmed gone.
            # A server that survived a failed kill has to stay recorded, or the
            # next startup sweep cannot reap it. The record stores a start-time
            # identity, so a recycled pid is never signalled either way.
            _killed_pid = getattr(self._process, "pid", None)
            _exit_code = getattr(self._process, "poll", lambda: None)()
            _exited = _exit_code is not None
            # The tree kill below tells the reaper the owner is known, which waives the
            # "cannot prove this pid is still our child" refusal. That waiver is only
            # true while we hold the handle that spawned the pid, so it is spent on a
            # real child and nothing else: a stand-in _process carrying a pid and no
            # poll() (tests use one to mean "a server is loaded", and one of them holds
            # this process's own pid) reads as "still running" through no fault of its
            # own, and signalling that number would take down whoever holds it now.
            _owns_child = terminable and callable(getattr(self._process, "poll", None))
            if _owns_child and _killed_pid is not None and not _exited:
                if _pgid is not None and hasattr(os, "killpg"):
                    # A real process group that `_kill_process_group` could actually
                    # signal, which only POSIX ever reports here (_leading_process_group
                    # answers None everywhere else, and that function's own no-op
                    # condition is repeated so this cannot skip a kill that never
                    # happened). killpg
                    # SIGKILL went to this exact group eleven lines above, and the tree
                    # kill's POSIX arm is killpg SIGTERM, a poll loop, then killpg
                    # SIGKILL over the same group: no reach this teardown does not
                    # already have, for a full /proc walk and up to five seconds more
                    # with _teardown_lock held. Measured at +560 ms of lock hold on the
                    # survivor path, which is time a lifecycle reopening behind the lock
                    # spends blocked.
                    #
                    # What the call did buy is the read-back, and that is kept. killpg is
                    # asynchronous like every other kill here, so asking on the next line
                    # answers "not finished yet" and keeps the record and the pidfile for
                    # a server that is already gone.
                    _exited = self._confirm_group_kill_landed(_killed_pid)
                else:
                    # The terminate above is all Popen offers, and on Windows that is
                    # TerminateProcess on the leader alone: a server that ignored it, or
                    # that the escalation could not reach, is still holding the model's
                    # mapping. taskkill /T /F is the only handle left on it there, and
                    # nothing on this path used to reach for it (#9790).
                    _exited = self._tree_kill_surviving_process(_killed_pid)
            # A descendant the sweep could not kill keeps the leader's record too. It was
            # adopted under its own pid above, but the record here is what the pidfile and
            # the next launch's reap are keyed on, and dropping it while something of this
            # server is still running is the state that leaves a worker on the GPU with
            # nothing naming it.
            if _surviving_descendants:
                _exited = False
            # And a tree that could not be enumerated is not a tree with nothing in it.
            # The sweep above ran over whatever the failed walk returned, so "no survivors"
            # here is a statement about a list that was never built. Keep the record: a
            # leaked worker the next launch can reap costs a stale pidfile, and dropping it
            # costs the only name anything has for a process still holding the GPU.
            if not _descendants_known and _pid is not None:
                logger.warning(
                    "llama-server descendants could not be enumerated before the kill; "
                    "keeping its record so the next launch can reap anything left"
                )
                _exited = False
            if _killed_pid is not None and _exited:
                try:
                    from utils.process_lifetime import forget_pid
                    forget_pid(_killed_pid)
                except Exception:
                    pass
            self._process = None
            self._planned_vram_mib = {}
            # Same rule as the lifetime record above: the pidfile is the next launch's
            # only handle on a server that outlived this kill, so it is removed once the
            # exit is confirmed and not merely attempted. Without a child handle there is
            # no exit to confirm and nothing was signalled, so the pidfile is dropped as
            # it always was rather than kept forever by a stand-in that cannot answer.
            if getattr(self, "_owns_pidfile", True) and (
                _killed_pid is None or _exited or not _owns_child
            ):
                self._clear_server_pid()
            # Clear healthy so a /load during the replacement's warm-up can't
            # short-circuit against the previous server's health (#5401).
            self._healthy = False
            # Reset to unknown so the training guard treats the next (still
            # loading) server as VRAM-resident rather than reading the killed
            # server's stale zero-offload flag until the health probe reclassifies.
            self._gpu_offload_active = None
            self._gpu_offload_layers = None
            self._offload_overridden = False
            self._gpu_backend_unavailable = False
            # Drives _wait_for_vram_settle in the next load_model; set in finally
            # so both in-process and frontend Apply paths record the kill.
            self._last_kill_monotonic = time.monotonic()
            stdout_thread = getattr(self, "_stdout_thread", None)
            if stdout_thread is not None:
                stdout_thread.join(timeout = 2)
                self._stdout_thread = None
            self._close_attempt_log(
                reason = "exited" if _exited_first else "killed", exit_code = _exit_code
            )

    def _other_planned_vram_mib(self) -> dict[int, int]:
        """MiB per GPU the other serving backends' running servers hold."""
        held: dict[int, int] = {}
        serving = _serving_backends_snapshot()
        if self not in serving:
            return held
        for backend in serving:
            if backend is self or not getattr(backend, "is_active", False):
                continue
            for idx, mib in getattr(backend, "_planned_vram_mib", {}).items():
                held[idx] = held.get(idx, 0) + mib
        return held

    def _note_server_pid(self, pid: int) -> None:
        if getattr(self, "_owns_pidfile", True):
            self._record_server_pid(pid)
            return
        try:
            from utils.process_lifetime import adopt_pid
            adopt_pid(pid)
        except Exception as e:
            logger.debug(f"Could not track llama-server for lifetime sweep: {e}")

    @staticmethod
    def _server_pidfile_path() -> Optional[Path]:
        """Pidfile recording the live llama-server PID, under the active studio root
        (per-root, so concurrent Unsloth instances with distinct UNSLOTH_STUDIO_HOME stay
        isolated, mirroring the reaper's custom-root isolation)."""
        try:
            from utils.paths.storage_roots import studio_root  # noqa: WPS433
            return studio_root() / "llama-server.pid"
        except Exception:
            return None

    @classmethod
    def _record_server_pid(cls, pid: int) -> None:
        """Best-effort record of the spawned llama-server PID for orphan reaping.

        Stores ``pid:starttime`` so a later startup can reject a PID that has
        since been recycled to a different process (see ``_pid_start_identity``).
        A bare ``pid`` (no identity) is still accepted on read for compatibility.
        """
        # Track it generically too: the pidfile holds one server, while the
        # process-lifetime record covers every child and is what the startup
        # sweep reads where there is no parent-death signal (macOS).
        try:
            from utils.process_lifetime import adopt_pid
            adopt_pid(pid)
        except Exception as e:
            logger.debug(f"Could not track llama-server for lifetime sweep: {e}")
        path = cls._server_pidfile_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents = True, exist_ok = True)
            path.write_text(f"{pid}:{cls._pid_start_identity(pid)}", encoding = "utf-8")
        except Exception as e:
            logger.debug(f"Could not write llama-server pidfile: {e}")

    @classmethod
    def _clear_server_pid(cls) -> None:
        """Best-effort removal of the llama-server pidfile."""
        path = cls._server_pidfile_path()
        if path is None:
            return
        try:
            path.unlink(missing_ok = True)
        except Exception as e:
            logger.debug(f"Could not remove llama-server pidfile: {e}")

    @staticmethod
    def _pid_is_llama_server(pid: int) -> bool:
        """True only if pid is a live process whose binary is a llama-server. Guards
        against PID reuse before killing a recorded orphan; returns False on any
        uncertainty so an unrelated process is never killed."""
        try:
            import psutil
            try:
                proc = psutil.Process(pid)
                if (proc.name() or "").lower().startswith("llama-server"):
                    return True
                return Path(proc.exe() or "").name.lower().startswith("llama-server")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                return False
        except ImportError:
            pass
        if sys.platform != "linux":
            return False
        try:
            if Path(os.readlink(f"/proc/{pid}/exe")).name.lower().startswith("llama-server"):
                return True
        except OSError:
            pass
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                tokens = fh.read().split(b"\x00")
            first = tokens[0].decode("utf-8", "replace") if tokens else ""
            return Path(first).name.lower().startswith("llama-server")
        except OSError:
            return False

    @staticmethod
    def _pid_start_identity(pid: int) -> str:
        """Stable per-PID identity (process start time) guarding against PID reuse.

        Returns a token string, or "" when it cannot be determined (the caller
        then falls back to the llama-server name check only)."""
        try:
            import psutil
            try:
                return str(psutil.Process(pid).create_time())
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                return ""
        except ImportError:
            pass
        if sys.platform == "linux":
            try:
                with open(f"/proc/{pid}/stat", "rb") as fh:
                    data = fh.read()
                # field 22 (starttime), counted from after the ")" that closes comm.
                return data[data.rfind(b")") + 2 :].split()[19].decode()
            except (OSError, IndexError):
                return ""
        return ""

    @staticmethod
    def _pid_parent_is_alive(pid: int) -> bool:
        """True if the recorded server's parent is still running, i.e. the server is
        NOT orphaned. Lets the cross-session reap kill only a true orphan (parent
        gone) and never a live server owned by a running Unsloth, regardless of which
        process performs the sweep. Biased toward "alive" on uncertainty so a live
        server is never mistakenly reaped."""
        try:
            import psutil

            try:
                ppid = psutil.Process(pid).ppid()
            except psutil.NoSuchProcess:
                return False  # the recorded server itself is gone
            except psutil.Error:
                return True  # cannot tell -- never risk killing a live server
            if ppid <= 1:
                return False  # reparented to init -> orphan
            return psutil.pid_exists(ppid)
        except ImportError:
            pass
        if sys.platform == "linux":
            try:
                with open(f"/proc/{pid}/stat", "rb") as fh:
                    data = fh.read()
                ppid = int(data[data.rfind(b")") + 2 :].split()[1])
            except (OSError, IndexError, ValueError):
                return False
            if ppid <= 1:
                return False
            return Path(f"/proc/{ppid}").exists()
        return False

    @staticmethod
    def _unlink_pidfile(path: Path) -> None:
        """Best-effort removal of a resolved pidfile path."""
        try:
            path.unlink(missing_ok = True)
        except Exception:
            pass

    @classmethod
    def _reap_recorded_pid(cls) -> int:
        """Kill the exact llama-server PID recorded at spawn, but only when it is a
        genuine orphan -- its parent (the Unsloth that spawned it) is gone. This is
        the cross-session backstop the parent-death reaper (Job Object /
        PR_SET_PDEATHSIG) cannot cover: an orphan left by an already-dead Unsloth
        (macOS, a best-effort failure, or a pre-existing orphan). Path-independent,
        so it also catches an orphan the install-root match would miss.

        A live server whose parent is still running is never reaped, so constructing
        a second backend in-process (the helper / advisor paths each build a
        LlamaCppBackend) cannot kill the active chat server. A recorded PID that has
        been recycled to a different process is rejected by the start-time identity
        and the llama-server name check, so unrelated user processes are never
        touched. SIGKILL falls back to SIGTERM on Windows, where os.kill maps it to
        TerminateProcess and SIGKILL is undefined."""
        path = cls._server_pidfile_path()
        if path is None or not path.exists():
            return 0

        pid = -1
        identity = ""
        try:
            pid_str, _, identity = path.read_text(encoding = "utf-8").strip().partition(":")
            pid = int(pid_str)
        except Exception:
            pid = -1

        # `pid <= 0` let pid 1 through. The cmdline check further down is what
        # actually keeps this honest and init cannot pass it, but the reaper in
        # process_lifetime had a start-time identity check that was just as
        # convincing and a record naming pid 1 still went through it, so the
        # cheap bound goes in front of the clever one here too.
        if not _is_signalable_pid(pid):
            cls._unlink_pidfile(path)  # garbage record
            return 0
        if pid == os.getpid():
            return 0  # never our own pid; leave the record alone

        if cls._pid_parent_is_alive(pid):
            # Live server with a running parent -> not an orphan; keep the record so
            # a later startup can still reap it if that parent later dies abnormally.
            return 0

        # Parent is gone: candidate orphan. Reject a PID recycled to something else.
        if identity and cls._pid_start_identity(pid) != identity:
            cls._unlink_pidfile(path)
            return 0

        killed = 0
        if cls._pid_is_llama_server(pid):
            try:
                os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
                killed = 1
                logger.info(f"Killed orphaned llama-server from pidfile (pid={pid})")
            except (ProcessLookupError, PermissionError):
                pass
            except Exception as e:
                logger.debug(f"Could not kill recorded llama-server pid {pid}: {e}")
        cls._unlink_pidfile(path)
        return killed

    @staticmethod
    def _kill_orphaned_servers() -> int:
        """Kill orphaned llama-server processes started by studio.

        Only kills processes whose resolved binary lives under a known
        Unsloth install dir (or matches an exact env-var override), to avoid
        terminating unrelated llama-server instances. Mirrors every location
        _find_llama_server_binary() can return, so orphans from any
        supported install path are cleaned up.

        Reads /proc directly on Linux and uses psutil elsewhere (macOS, Windows).
        The pgrep fallback the psutil-less path used to take is gone: /proc is
        what that fallback read anyway, and psutil is a hard dependency, so it
        was unreachable from both directions.

        Candidates are collected first and killed afterwards, so an enumeration
        that fails part way kills nothing rather than some. The parent check
        therefore runs after enumeration rather than during it; a parent that
        exits in between makes the process a genuine orphan, and the start-time
        check below still refuses to signal a PID that has been reused.

        Returns the count of processes killed; callers arm the VRAM-settle
        wait on a positive count.
        """
        # Cross-session backstop first: reap the exact PID we recorded at spawn,
        # but only if it is a true orphan whose parent is gone (so a helper backend
        # built while a chat server is live can never kill it). The root-gated
        # enumeration below stays as a fallback.
        killed = LlamaCppBackend._reap_recorded_pid()
        try:
            # -- Build the ownership allowlist --------------------------------
            # exact_binaries -- env var overrides (exact path match).
            # install_roots  -- Unsloth-owned dir trees (binary must be under one).
            install_roots: list[Path] = []

            # Env-mode custom root (mirrors _find_llama_server_binary).
            _resolved_sr, _is_legacy = LlamaCppBackend._resolved_studio_root_and_is_legacy()
            _is_custom_root = not _is_legacy
            if _is_custom_root:
                install_roots.append(_resolved_sr / "llama.cpp")

            # Primary install dir (default mode only). Env-mode skips this so a
            # custom-root Unsloth can't kill a default-install Unsloth's server.
            if not _is_custom_root:
                install_roots.append(Path.home() / ".unsloth" / "llama.cpp")

            # Legacy in-tree build dirs (older setup.sh)
            project_root = Path(__file__).resolve().parents[5]
            install_roots.append(project_root / "llama.cpp")

            # Legacy: extracted binary
            install_roots.append(project_root / "bin")

            # UNSLOTH_LLAMA_CPP_PATH env var (custom install dir)
            custom_dir = os.environ.get("UNSLOTH_LLAMA_CPP_PATH")
            if custom_dir:
                # expanduser to match _find_llama_server_binary: the allowlist has
                # to name the tree discovery actually spawns from, or a server
                # started out of an expanded ~ is one this refuses to clean up.
                install_roots.append(_expanded_user_path(custom_dir))

            # LLAMA_SERVER_PATH env var (exact binary path)
            exact_binaries: list[Path] = []
            env_binary = os.environ.get("LLAMA_SERVER_PATH")
            if env_binary:
                try:
                    exact_binaries.append(Path(env_binary).resolve())
                except OSError:
                    pass

            # Resolve all roots so is_relative_to works reliably.
            resolved_roots: list[Path] = []
            for root in install_roots:
                try:
                    # A --with-llama-cpp-dir local link (symlink/junction)
                    # resolves into the user's own checkout. Adding it would let
                    # us treat the user's externally-launched llama-server as our
                    # orphan and kill it, so leave such roots out of the
                    # allowlist (we forgo orphan-reaping for local-link installs).
                    if _is_external_link(root):
                        continue
                    resolved_roots.append(root.resolve())
                except OSError:
                    pass

            my_pid = os.getpid()

            # -- Enumerate candidate processes ---------------------------------
            # (pid, resolved binary, kill callable) per llama-server-looking process;
            # ownership, orphan and kill handling is shared below so the paths agree.
            candidates = []

            def _looks_like_server(name):
                return bool(name) and name.lower().startswith("llama-server")

            def _stat_start_time(line):
                right = line.rfind(b")")
                fields = line[right + 2 :].split()
                return fields[19] if len(fields) > 19 else None

            def _proc_start_time(proc_dir):
                """Field 22 of /proc/<pid>/stat, in jiffies since boot."""
                try:
                    with open(f"{proc_dir}/stat", "rb") as fh:
                        line = fh.read()
                except OSError:
                    return None
                right = line.rfind(b")")
                if right < 0:
                    return None
                # Fields after comm start at field 3, so starttime is index 19.
                fields = line[right + 2 :].split()
                return fields[19] if len(fields) > 19 else None

            def _make_signal_killer(proc_dir, pid, start_time):
                def _kill():
                    # psutil.Process.kill() refuses to signal a reused PID, so re-check
                    # identity here too: starttime never changes for a process and a reused
                    # PID gets a later one, so a mismatch means this is not our process.
                    if start_time is None or _proc_start_time(proc_dir) != start_time:
                        return False
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        return False
                    return True

                return _kill

            # /proc is Linux only; a missing one falls back to psutil rather than finding
            # nothing. Tests point _PROC_ROOT elsewhere to drive either path.
            proc_root = _PROC_ROOT
            if sys.platform == "linux" and os.path.isdir(proc_root):
                # psutil.process_iter(["pid", "name", "exe"]) reads /proc/<pid>/stat several
                # times per process and resolves exe for every one before the name rejects
                # almost all. This sweep runs on every backend start and every temporary
                # LlamaCppBackend(), so read comm once and resolve exe only on a name match.
                def _proc_exe(proc_dir):
                    try:
                        target = os.readlink(f"{proc_dir}/exe")
                    except OSError:
                        return None
                    if not target:
                        return None
                    # The kernel appends " (deleted)" once the binary is replaced or removed,
                    # which is what an upgrade's orphan looks like. psutil strips the marker,
                    # so strip it here too or that orphan stops being recognised as ours.
                    if target.endswith(" (deleted)") and not os.path.exists(target):
                        target = target[: -len(" (deleted)")]
                    try:
                        return Path(target).resolve()
                    except OSError:
                        return None

                with os.scandir(proc_root) as entries:
                    for entry in entries:
                        if not entry.name.isdigit():
                            continue
                        try:
                            pid = int(entry.name)
                        except ValueError:
                            continue
                        if pid == my_pid:
                            continue
                        try:
                            with open(f"{entry.path}/stat", "rb") as fh:
                                stat_line = fh.read()
                        except OSError:
                            continue  # the process exited mid-scan
                        left = stat_line.find(b"(")
                        right = stat_line.rfind(b")")
                        if left < 0 or right <= left:
                            continue
                        # comm is truncated to 15 bytes, which still leaves the 12-byte
                        # "llama-server" prefix intact, so this selects the names psutil does.
                        name = stat_line[left + 1 : right].decode("utf-8", "replace")
                        if not _looks_like_server(name):
                            continue
                        exe_path = _proc_exe(entry.path)
                        if exe_path is None:
                            continue
                        candidates.append(
                            (
                                pid,
                                exe_path,
                                _make_signal_killer(entry.path, pid, _stat_start_time(stat_line)),
                            )
                        )
            else:
                try:
                    import psutil
                except ImportError:
                    psutil = None

                if psutil is not None:

                    def _make_psutil_killer(psutil_mod, proc):
                        def _kill():
                            try:
                                proc.kill()
                            except (
                                psutil_mod.NoSuchProcess,
                                psutil_mod.AccessDenied,
                                psutil_mod.ZombieProcess,
                            ):
                                return False
                            return True

                        return _kill

                    for proc in psutil.process_iter(["pid", "name", "exe"]):
                        try:
                            pid = proc.info["pid"]
                            if pid == my_pid:
                                continue
                            if not _looks_like_server(proc.info.get("name") or ""):
                                continue
                            exe = proc.info.get("exe")
                            if not exe:
                                continue
                            candidates.append(
                                (pid, Path(exe).resolve(), _make_psutil_killer(psutil, proc))
                            )
                        except (
                            psutil.NoSuchProcess,
                            psutil.AccessDenied,
                            psutil.ZombieProcess,
                        ):
                            pass
                else:
                    # Unreachable while the Linux branch above exists; kept so the
                    # psutil-less path stays correct if that changes.
                    return killed

            # -- Ownership check, orphan check, kill ---------------------------
            for pid, binary, kill in candidates:
                # Floored here rather than in each enumerator: both the /proc scan
                # and the psutil scan feed this loop, and a third one added later
                # would too. Not hypothetical for pid 1 either, since #7894
                # established Unsloth can run as a container entrypoint, and a
                # container whose entrypoint is llama-server puts a process this
                # sweep recognises at pid 1. Killing it takes the container down.
                if not _is_signalable_pid(pid):
                    continue
                # Ownership: exact match OR binary under a known root.
                is_ours = binary in exact_binaries or any(
                    binary.is_relative_to(root) for root in resolved_roots
                )
                if not is_ours:
                    continue

                # A live parent means a running Unsloth (or the user's shell) still owns it.
                if LlamaCppBackend._pid_parent_is_alive(pid):
                    continue

                if kill():
                    killed += 1
                    logger.info(f"Killed orphaned llama-server process (pid={pid})")
        except Exception:
            logger.warning("Error during orphan server cleanup", exc_info = True)
        return killed

    def _cleanup(self):
        """atexit handler to ensure llama-server is terminated (a teardown).

        Nothing here may report a failure through the logging machinery. By the
        time atexit runs, the streams the handlers write to can already be closed,
        and a write then fails -- which is how this was found, as several unrelated
        tracebacks printed after the test summary.

        Two different mechanisms, because there are two kinds of logger in play.
        This module's own logger is a structlog PrintLogger writing straight to
        stdout (loggers/config.py), which has no error handling of its own and does
        not consult raiseExceptions at all, so only the except below covers it.
        raiseExceptions covers the stdlib loggers other libraries install, which
        report a broken handler by printing their own traceback instead of raising.
        Neither one alone is enough.
        """
        raise_exceptions = logging.raiseExceptions
        logging.raiseExceptions = False
        try:
            self._kill_process(teardown = True)
            # TemporaryDirectory's exit hook runs first and cannot delete a staged
            # runtime whose server is alive (Windows locks the exe). Retry post-kill.
            self._cleanup_cpu_fallback_runtime()
        except Exception:
            # atexit swallows this anyway, and there is nowhere left to report it.
            pass
        finally:
            logging.raiseExceptions = raise_exceptions
