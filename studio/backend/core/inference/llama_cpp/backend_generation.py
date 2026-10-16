# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: message building and generation."""

import contextlib
import httpx
import json
import re
import socket
import subprocess
import sys
import threading
import time

from typing import (
    Callable,
    Generator,
    Optional,
    Union,
)
from core.inference.context_window import (
    estimate_message_tokens_without_unpriced_media,
    messages_without_unpriced_media,
)
from core.inference.stream_errors import stream_error_from_chunk
from core.inference.tool_call_parser import (
    parse_tool_calls_from_text as _shared_parse_tool_calls_from_text,
)
from utils.first_token_timeout import (
    finish_first_token_wait,
    socket_timeout,
    start_first_token_wait,
)

from .compaction import (
    _DEFAULT_FIRST_TOKEN_TIMEOUT_S,
    _DEFAULT_MAX_TOKENS_FLOOR,
    _archive_and_recall,
    _backend_supports_tools,
    _boundary_metadata,
    _can_reset_epoch,
    _compaction_fit_kwargs,
    _conversation_recall_reserve,
    _fit_with_instruction_pins,
    _keeps_compaction_boundary,
    _records_boundary,
    _retrieval_budget,
    _sticky_compaction_details,
)
from .core_types import (
    _DEVICE_LOST_LOG_GRACE_S,
    _LlamaStreamCancelled,
    _apply_seeded_llama_request,
    _is_gpu_device_lost,
)
from .launch_support import _local_ssl_context
from .log import logger
from .timings import (
    _backfill_usage_from_timings,
    _llama_chunk_has_generated_output,
    _perf_callback_wants_timings,
    _report_live_llama_timings,
)
from .tool_loop_text import (
    _DEFAULT_STREAM_STALL_TIMEOUT_S,
    _RESPAWN_REAP_GRACE_S,
    _finalize_reasoning_only_cumulative,
    _quote_cut_event,
)

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _GenerationMixin:
    # ── Message building (OpenAI format) ──────────────────────────

    @staticmethod
    def _parse_tool_calls_from_text(
        content: str,
        *,
        allow_incomplete: bool = True,
        enabled_tool_names: Optional[set] = None,
    ) -> list[dict]:
        """Wrapper around the shared parser; ``enabled_tool_names`` gates the markerless bare-JSON form."""
        return _shared_parse_tool_calls_from_text(
            content,
            allow_incomplete = allow_incomplete,
            enabled_tool_names = enabled_tool_names,
        )

    @staticmethod
    def _build_openai_messages(messages: list[dict], image_b64: Optional[str] = None) -> list[dict]:
        """Build OpenAI-format messages, optionally injecting an image_url part
        into the last user message for vision models. As-is if no image."""
        if not image_b64:
            return messages

        # Convert the last user message to multimodal content parts
        result = [msg.copy() for msg in messages]
        last_user_idx = None
        for i, msg in enumerate(result):
            if msg["role"] == "user":
                last_user_idx = i

        if last_user_idx is not None:
            text_content = result[last_user_idx].get("content", "")
            result[last_user_idx]["content"] = [
                {"type": "text", "text": text_content},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                    },
                },
            ]

        return result

    # ── Generation (proxy to llama-server) ────────────────────────

    @contextlib.contextmanager
    def _open_stream(self, url: str, payload: dict, cancel_event):
        """Open a streaming POST to llama-server, retrying through prefill, and
        yield ``(response, first_token_deadline)`` once a 200 lands. Owns the
        httpx.Client + auth headers for the stream's lifetime; raises
        RuntimeError on a non-200. Shared scaffold for the streaming consumers,
        which differ only in how they parse the SSE body."""
        from core.inference.mcp_images import prepare_image_turn_boundaries

        if "messages" in payload:
            payload = {
                **payload,
                "messages": prepare_image_turn_boundaries(
                    payload["messages"],
                    getattr(self, "_chat_template_override", None)
                    or getattr(self, "_chat_template", None),
                ),
            }
        stream_timeout = httpx.Timeout(connect = 10, read = 0.5, write = 10, pool = 10)
        with httpx.Client(
            timeout = stream_timeout,
            limits = httpx.Limits(max_keepalive_connections = 0),
            trust_env = False,
            verify = _local_ssl_context(),
        ) as client:
            first_token_deadline = start_first_token_wait()
            with self._stream_with_retry(
                client,
                url,
                payload,
                cancel_event,
                headers = self._auth_headers,
                first_token_deadline = first_token_deadline,
            ) as response:
                if response.status_code != 200:
                    error_body = response.read().decode()
                    raise RuntimeError(
                        f"llama-server returned {response.status_code}: {error_body}"
                    )
                yield response, first_token_deadline

    @staticmethod
    def _sse_event_payload(event: str) -> Optional[dict]:
        """Decode one complete SSE event's ``data:`` payload, or None."""
        payload_lines = [
            line[5:].lstrip() for line in event.splitlines() if line.startswith("data:")
        ]
        if not payload_lines:
            return None
        try:
            data = json.loads("\n".join(payload_lines))
        except (TypeError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _sse_event_prefill_progress(event: str) -> Optional[float]:
        """Return the reported processed prompt tokens, if present."""
        data = LlamaCppBackend._sse_event_payload(event)
        if data is None:
            return None
        progress = data.get("prompt_progress")
        if not isinstance(progress, dict):
            return None
        try:
            return float(progress.get("processed"))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _sse_event_has_generated_output(event: str) -> bool:
        """Return true when a complete SSE event carries model-generated output."""
        data = LlamaCppBackend._sse_event_payload(event)
        if data is None:
            return False
        return _llama_chunk_has_generated_output(data)

    @staticmethod
    def _iter_text_cancellable(
        response: "httpx.Response",
        cancel_event: Optional[threading.Event] = None,
        stall_timeout_s: float = _DEFAULT_STREAM_STALL_TIMEOUT_S,
        first_token_deadline: Optional[float] = None,
        post_first_chunk_read_timeout_s: Optional[float] = _DEFAULT_STREAM_STALL_TIMEOUT_S,
    ) -> Generator[str, None, None]:
        """Iterate a stream while polling cancel and stall timeouts."""
        text_iter = response.iter_text()
        if first_token_deadline is None:
            first_token_deadline = start_first_token_wait()
        last_chunk_at: Optional[float] = None
        last_prefill_progress: Optional[float] = None
        prefill_sse_buffer = ""
        while True:
            if cancel_event is not None and cancel_event.is_set():
                response.close()
                return
            try:
                if last_chunk_at is None:
                    remaining_s = first_token_deadline - time.monotonic()
                    if remaining_s <= 0:
                        raise httpx.ReadTimeout("The model did not produce a first token in time.")
                    LlamaCppBackend._set_stream_read_timeout(response, socket_timeout(remaining_s))
                chunk = next(text_iter)
                starts_output = last_chunk_at is not None
                if chunk and last_chunk_at is None:
                    prefill_sse_buffer += chunk
                    while match := re.search(r"(?:\r\n|\r|\n){2}", prefill_sse_buffer):
                        event = prefill_sse_buffer[: match.start()]
                        prefill_sse_buffer = prefill_sse_buffer[match.end() :]
                        if LlamaCppBackend._sse_event_has_generated_output(event):
                            starts_output = True
                            prefill_sse_buffer = ""
                            break
                        # Renew only on increasing progress so repeated events still time out.
                        processed = LlamaCppBackend._sse_event_prefill_progress(event)
                        if processed is not None and (
                            last_prefill_progress is None or processed > last_prefill_progress
                        ):
                            last_prefill_progress = processed
                            first_token_deadline = start_first_token_wait()
                if chunk and starts_output:
                    finish_first_token_wait()
                    if last_chunk_at is None and post_first_chunk_read_timeout_s is not None:
                        LlamaCppBackend._set_stream_read_timeout(
                            response,
                            post_first_chunk_read_timeout_s,
                        )
                    last_chunk_at = time.monotonic()
                yield chunk
            except StopIteration:
                return
            except httpx.ReadTimeout:
                now = time.monotonic()
                if last_chunk_at is None:
                    if now >= first_token_deadline:
                        raise
                elif now - last_chunk_at >= stall_timeout_s:
                    raise httpx.ReadTimeout("The model stopped producing tokens mid-response.")
                continue

    @staticmethod
    def _set_stream_read_timeout(
        response: "httpx.Response", read_timeout_s: Optional[float]
    ) -> None:
        """Lower only post-header stream reads; keep prefill timeout long."""
        try:
            timeout_ext = response.request.extensions.get("timeout")
            if isinstance(timeout_ext, dict):
                timeout_ext["read"] = read_timeout_s
        except Exception:
            logger.debug("Could not lower response read timeout", exc_info = True)

    @staticmethod
    def _shutdown_active_httpx_sockets(client: "httpx.Client") -> None:
        """Best-effort interrupt for a sync httpx read blocked in recv(), whether
        parked before headers (prefill) or mid-stream."""
        try:
            pool = getattr(getattr(client, "_transport", None), "_pool", None)
            connections = list(getattr(pool, "_connections", []) or [])
            for connection in connections:
                inner = getattr(connection, "_connection", None)
                stream = getattr(inner, "_network_stream", None)
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    continue
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
        except Exception:
            logger.debug("Could not shutdown active httpx socket", exc_info = True)
        try:
            client.close()
        except Exception:
            logger.debug("Could not close httpx client", exc_info = True)

    @staticmethod
    def _install_cancel_aware_read(
        client: "httpx.Client",
        cancel_event: threading.Event,
        response: Optional["httpx.Response"] = None,
        poll_s: float = 0.2,
    ) -> None:
        """Wrap the httpcore stream so the reader interrupts its own blocked recv() on cancel.

        A cross-thread socket shutdown wakes a parked recv() on POSIX but not on
        Windows (Winsock), so read in short slices and poll cancel_event between them
        (plain or TLS); slice timeouts are swallowed so a slow-but-alive stream survives.
        httpcore snapshots request.extensions["timeout"]["read"] once at body start, so
        given ``response`` we re-read the live value per call to honor the post-first-token
        stall timeout instead of the long prefill timeout."""
        import httpcore

        def _live_read_timeout(snapshot: Optional[float]) -> Optional[float]:
            if response is None:
                return snapshot
            try:
                ext = response.request.extensions.get("timeout")
                if isinstance(ext, dict) and "read" in ext:
                    value = ext["read"]
                    # None is a live value too: an unlimited first-token wait has no read bound.
                    if value is None or isinstance(value, (int, float)):
                        return value
            except Exception:
                pass
            return snapshot

        try:
            pool = getattr(getattr(client, "_transport", None), "_pool", None)
            for connection in list(getattr(pool, "_connections", []) or []):
                inner = getattr(connection, "_connection", None)
                stream = getattr(inner, "_network_stream", None)
                if stream is None or getattr(stream, "_unsloth_cancel_wrapped", False):
                    continue
                orig_read = stream.read

                def read(
                    max_bytes,
                    timeout = None,
                    _orig = orig_read,
                ):
                    effective = _live_read_timeout(timeout)
                    deadline = None if effective is None else time.monotonic() + effective
                    while True:
                        if cancel_event.is_set():
                            raise httpcore.ReadError("stream cancelled by user")
                        if deadline is None:
                            step = poll_s
                        else:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise httpcore.ReadTimeout("read operation timed out")
                            step = min(poll_s, remaining)
                        try:
                            return _orig(max_bytes, timeout = step)
                        except httpcore.ReadTimeout:
                            if deadline is not None and time.monotonic() >= deadline:
                                raise
                            continue  # slow but alive: keep reading

                stream.read = read
                stream._unsloth_cancel_wrapped = True
        except Exception:
            logger.debug("Could not install cancel-aware read", exc_info = True)

    @staticmethod
    @contextlib.contextmanager
    def _stream_with_retry(
        client: "httpx.Client",
        url: str,
        payload: dict,
        cancel_event: Optional[threading.Event] = None,
        headers: Optional[dict] = None,
        first_token_deadline: Optional[float] = None,
    ):
        """Open one streaming POST and let cancel interrupt prefill or reads."""
        if cancel_event is not None and cancel_event.is_set():
            raise _LlamaStreamCancelled

        _cancel_closed = threading.Event()
        _response_ref: list = [None]

        def _cancel_watcher():
            while not _cancel_closed.is_set():
                if cancel_event.wait(timeout = 0.3):
                    while not _cancel_closed.is_set():
                        r = _response_ref[0]
                        try:
                            # response.close() can't wake a read already blocked in
                            # recv(); only a socket shutdown does, so shut down first.
                            LlamaCppBackend._shutdown_active_httpx_sockets(client)
                            if r is not None:
                                r.close()
                            return
                        except Exception as e:
                            logger.debug(f"Error closing request in cancel watcher: {e}")
                        _cancel_closed.wait(timeout = 0.1)
                    return

        watcher = None
        if cancel_event is not None:
            watcher = threading.Thread(target = _cancel_watcher, daemon = True, name = "prefill-cancel")
            watcher.start()

        try:
            if first_token_deadline is None:
                first_token_deadline = start_first_token_wait()
            prefill_read_timeout = max(0.1, first_token_deadline - time.monotonic())
            if sys.platform == "win32":
                # Stop cannot wake a Winsock recv parked before headers, so that wait keeps the default bound.
                prefill_read_timeout = min(prefill_read_timeout, _DEFAULT_FIRST_TOKEN_TIMEOUT_S)
            prefill_timeout = httpx.Timeout(
                connect = 30,
                read = socket_timeout(prefill_read_timeout),
                write = 10,
                pool = 10,
            )
            with client.stream(
                "POST",
                url,
                json = payload,
                timeout = prefill_timeout,
                headers = headers,
            ) as response:
                _response_ref[0] = response
                # Portable mid-stream cancel: the reader polls cancel itself, so Stop
                # interrupts a stalled read where the watcher's Windows socket shutdown
                # does not. Installed without a cancel event too: only this wrapper
                # honors the live stall timeout over the one latched for the header wait.
                LlamaCppBackend._install_cancel_aware_read(
                    client, cancel_event or threading.Event(), response
                )
                if cancel_event is not None and cancel_event.is_set():
                    raise _LlamaStreamCancelled
                yield response
                return
        except (httpx.RequestError, RuntimeError):
            # Response was closed by the cancel watcher
            if cancel_event is not None and cancel_event.is_set():
                raise _LlamaStreamCancelled
            raise
        finally:
            _cancel_closed.set()

    def _server_socket_is_open(self, timeout_s: float = 0.15) -> bool:
        """True if anything still accepts on the server port.

        The listening socket dies with the process, so this tells a live server
        from a dead one without waiting for the child to become reapable.
        """
        port = self._port
        if not port:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout = timeout_s):
                return True
        except OSError:
            return False

    def _respawn_if_dead(self) -> bool:
        """Relaunch the llama-server if its process has exited.

        A loaded chat model can be SIGKILL'd mid-session (usually GPU/RAM pressure
        from a training run on the same box), leaving a defunct process while
        ``is_loaded`` still reads True. Replay the last ``load_model`` call to
        recover, returning True once healthy. Serialised on ``_respawn_lock`` so
        many generations hitting the dead server trigger at most one reload.
        """
        # Read outside the lock so a queued caller can tell the replacement from the child
        # its own error came from; otherwise each burns the grace wait below, and that
        # sleep is held under the lock, so the waits serialise.
        served_by = self._process
        with self._respawn_lock:
            proc = self._process
            if proc is None:
                return False
            if self._cancel_event.is_set():
                # unload_model sets this before it kills, so the child can still be
                # accepting. Reporting it healthy would aim the retry at a server
                # that is deliberately going away.
                return False
            if proc is not served_by:
                # Replaced while we queued: this child never served our request.
                return self._healthy
            if proc.poll() is None:
                # Still serving, so the error was transient. Charging it the grace below
                # would cost a second per caller, serialised under this lock.
                if self._server_socket_is_open():
                    return self._healthy
                # A closing server can beat its own exit status: calling it alive returns
                # the stale _healthy and spends the retry on the corpse.
                deadline = time.monotonic() + _RESPAWN_REAP_GRACE_S
                while proc.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
            if proc.poll() is None:
                # Alive: either a concurrent caller already respawned it (healthy), or
                # this connection error wasn't a dead server.
                return self._healthy
            with self._mtp_runtime_fallback_lock:
                if self._mtp_runtime_fallback_in_progress:
                    # An MTP-free reload owns this corpse; replaying the old kwargs
                    # restarts the crashing config and aborts that reload.
                    logger.info("Respawn skipped: an MTP-free reload is already recovering.")
                    return False
            # The RLock lets the load_model below re-enter it.
            with self._serial_load_lock:
                if self._process is not proc:
                    logger.info("Respawn skipped: a newer load is already active.")
                    return self._healthy
                # Snapshot under _lock, the one unload_model holds, so a teardown is
                # either wholly before us (flag set) or wholly after (epoch bumped).
                # _serial_load_lock alone would not exclude it: unload never takes it.
                with self._lock:
                    if self._cancel_event.is_set():
                        logger.info("Respawn skipped: the model was unloaded.")
                        return False
                    intent = self._last_load_intent
                    if intent is None:
                        return False
                    epoch = self._unload_epoch
                    self._healthy = False
                logger.warning(
                    f"llama-server for '{self._model_identifier}' exited "
                    f"(code {proc.returncode}); respawning to recover the session"
                )
                try:
                    started = bool(self.load_model(intent))
                except Exception as exc:
                    logger.error(f"Failed to respawn llama-server: {exc}")
                    return False
                if started and self._unload_epoch != epoch:
                    # An unload landed mid-reload. load_model cleared _cancel_event on
                    # the way in, so the epoch is the only surviving evidence; undo the
                    # replacement rather than leave a model the user stopped running.
                    logger.info("Respawn undone: the model was unloaded during the reload.")
                    self.unload_model()
                    return False
                return started

    def _retire_device_lost_server(self, served_by) -> None:
        """Kill the llama-server that hit a lost GPU device, if it is still current (#11453).

        It survives VK_ERROR_DEVICE_LOST but fails every later request; a dead child is
        what _respawn_if_dead already recovers.
        """
        with self._respawn_lock:
            if (
                served_by is None
                or served_by is not self._process
                or served_by.poll() is not None
                or self._cancel_event.is_set()
            ):
                return
            # Else the MTP watchdog reads this kill as an MTP crash.
            self._stop_mtp_crash_watchdog()
            try:
                served_by.kill()
                served_by.wait(timeout = 10)
            except (OSError, subprocess.TimeoutExpired):
                pass
        try:
            logger.warning(
                f"llama-server for '{self._model_identifier}' lost its GPU device "
                "(the graphics driver reset it); restarted it. On Intel Arc with "
                "Vulkan, GGML_VK_DISABLE_COOPMAT=1 may avoid the reset."
            )
        except Exception:
            # The logger raises once stdout is closed; recovery must not.
            pass

    def _retire_device_lost_server_soon(self, served_by) -> None:
        """Retire after a device loss in the server log, for the passthrough routes.

        The grace lets a chat request that hit the same error retire and retry first.
        """

        def _retire():
            time.sleep(_DEVICE_LOST_LOG_GRACE_S)
            self._retire_device_lost_server(served_by)

        try:
            threading.Thread(target = _retire, daemon = True, name = "llama-device-lost").start()
        except RuntimeError:
            pass

    @contextlib.contextmanager
    def _open_chat_stream_with_respawn_retry(
        self,
        payload: dict,
        cancel_event,
        on_respawn: Optional[Callable[[], None]] = None,
    ):
        """Open a chat stream, respawning a dead llama-server once before streaming.

        Retry only when opening the response fails: once it is open a consumer may
        already have emitted content or tool events, so a replay could duplicate
        output and side effects. ``base_url`` is resolved per attempt because a
        respawn may pick a new port. The budget is one retry per model request, not
        per chat turn, so a long tool loop never discards a completed tool.

        A child dying after the accept but before the headers surfaces as
        ReadError/WriteError/RemoteProtocolError rather than ConnectError, and which
        one differs per OS. llama-server flushes its 200 at slot start, so that window
        is an upload still in flight or a request behind busy slots; a death during
        decode arrives with the response open and is not replayed. Timeouts are
        excluded: the server is slow, not dead, and a replay would spend the
        first-token budget twice.
        """
        for attempt in range(2):
            response_opened = False
            served_by = getattr(self, "_process", None)
            try:
                url = f"{self.base_url}/v1/chat/completions"
                with self._open_stream(url, payload, cancel_event) as opened:
                    response_opened = True
                    yield opened
                    return
            except RuntimeError as exc:
                if not _is_gpu_device_lost(exc):
                    raise
                # Retry only before the 200; once opened the consumer may have emitted output.
                self._retire_device_lost_server(served_by)
                if response_opened or attempt > 0 or not self._respawn_if_dead():
                    raise
                logger.warning(
                    "llama-server lost its GPU device; restarted it and retrying the generation"
                )
                if on_respawn is not None:
                    on_respawn()
                continue
            except (httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                if response_opened:
                    raise
                if self._maybe_recover_from_mtp_crash(exc):
                    raise RuntimeError("Lost connection to llama-server") from exc
                if attempt == 0 and self._respawn_if_dead():
                    logger.warning(
                        "llama-server was unreachable; respawned it and retrying the generation"
                    )
                    if on_respawn is not None:
                        on_respawn()
                    continue
                raise

    @staticmethod
    def decode_slot_from_chunk(chunk) -> Optional[int]:
        """The llama-server slot a stream chunk was decoded on, None if it does not say.

        Only the final result carries ``__verbose``, and only when asked for.
        """
        if not isinstance(chunk, dict):
            return None
        verbose = chunk.get("__verbose")
        if not isinstance(verbose, dict):
            return None
        slot = verbose.get("id_slot")
        if type(slot) is not int or slot < 0:
            return None
        return slot

    def release_idle_chat_slot(self, base_url: str, slot: int) -> bool:
        """Erase one completed round's cached context, True once the engine says so.

        Declines unless the round decoded on this server, which a reload renumbers, and
        unless --slot-save-path is in play, which the endpoint requires. An erase aimed at
        a slot since handed to another request is deferred rather than dropped, so even a
        timeout can cost that chat its prefix.
        """
        if (
            not self._slot_save_dir
            or base_url != self.base_url
            or type(slot) is not int
            or slot < 0
        ):
            return False
        try:
            response = httpx.post(
                f"{base_url}/slots/{slot}",
                params = {"action": "erase"},
                headers = self._auth_headers,
                timeout = 2.0,
                trust_env = False,
            )
            response.raise_for_status()
            result = response.json()
            return (
                result.get("id_slot") == slot
                and type(result.get("n_erased")) is int
                and result["n_erased"] >= 0
            )
        except Exception:
            logger.debug("Approval cache reclamation failed; retaining reservation", exc_info = True)
            return False

    def generate_chat_completion(
        self,
        messages: list[dict],
        image_b64: Optional[str] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.01,
        max_tokens: Optional[int] = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[dict] = None,
        stop: Optional[list[str]] = None,
        cancel_event: Optional[threading.Event] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
        replay_preserve_thinking: bool = True,
        continue_final_message: bool = False,
        seed: Optional[int] = None,
        promote_reasoning_only: bool = True,
        perf_callback: Optional[Callable[[dict], None]] = None,
        reasoning_provenance: Optional[dict] = None,
        context_overflow: Optional[str] = None,
        context_policy: Optional[str] = None,
        compaction_headroom_ratio: Optional[float] = None,
        thread_id: Optional[str] = None,
        tools_withheld: bool = False,
        thinking_budget_tokens: Optional[int] = None,
        _allow_respawn_retry: bool = True,
        compact_context: bool = False,
    ) -> Generator[Union[str, dict], None, None]:
        """
        Send a chat completion to llama-server and stream tokens back.

        Uses /v1/chat/completions -- llama-server applies the chat template
        and handles vision (multimodal image_url parts) natively.

        Yields cumulative text (matching InferenceBackend's convention).
        """
        if not self.is_loaded:
            raise RuntimeError("llama-server is not loaded")
        if replay_preserve_thinking:
            preserve_thinking = self._replay_preserve_thinking(
                preserve_thinking, messages, enable_thinking, reasoning_effort
            )

        from core.inference.chat_template_helpers import (
            neutralize_control_markup_in_messages,
            trailing_assistant_resume_kind,
        )

        openai_messages = self._build_openai_messages(messages, image_b64)
        _resume_kind = (
            trailing_assistant_resume_kind(openai_messages) if continue_final_message else None
        )
        continue_final_message = _resume_kind is not None
        resumed_reasoning = (
            openai_messages[-1]["reasoning_content"] if _resume_kind == "reasoning_content" else ""
        )

        payload = {
            # llama-server applies the chat template itself (#7066).
            "messages": neutralize_control_markup_in_messages(
                openai_messages, None, self.markup_profile
            ),
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k >= 0 else 0,
            "min_p": min_p,
            "repeat_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        retry_messages = messages
        retry_image_b64 = image_b64
        retry_max_tokens = max_tokens
        retry_context_overflow = context_overflow
        retry_preflight_context_length = None
        # Progress events let advancing prefills renew the first-token deadline.
        payload["return_progress"] = True
        if _perf_callback_wants_timings(perf_callback):
            payload["timings_per_token"] = True
        if logit_bias:
            payload["logit_bias"] = logit_bias
        # Per-request enable_thinking / reasoning_effort / preserve_thinking
        _reasoning_kw = self._request_reasoning_kwargs(
            enable_thinking, reasoning_effort, preserve_thinking
        )
        if _reasoning_kw is not None:
            payload["chat_template_kwargs"] = _reasoning_kw
        if thinking_budget_tokens is not None:
            payload["thinking_budget_tokens"] = thinking_budget_tokens
        if continue_final_message:
            # llama-server applies the template; it rejects both flags set true.
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False
        # Default cap to the model context when known.
        payload["max_tokens"] = (
            max_tokens
            if max_tokens is not None
            else (self._effective_context_length or _DEFAULT_MAX_TOKENS_FLOOR)
        )
        if context_overflow == "truncate_oldest" and self._effective_context_length:
            try:
                _before_fit = openai_messages
                # One value for both the boundary read and the fit: a boundary the fit
                # would not consume as an epoch must not be discarded as if it would.
                _can_reset = _can_reset_epoch(
                    thread_id,
                    _backend_supports_tools(self),
                    tools_withheld = tools_withheld,
                )
                _sticky, _sticky_is_checkpoint, _sticky_manual = _sticky_compaction_details(
                    thread_id,
                    _before_fit,
                    context_policy = context_policy,
                    can_reset = _can_reset,
                    compaction_headroom_ratio = compaction_headroom_ratio,
                )
                openai_messages, truncation = _fit_with_instruction_pins(
                    openai_messages,
                    context_length = self._effective_context_length,
                    max_tokens = payload["max_tokens"],
                    count_tokens = lambda fitted: self.count_chat_tokens(
                        neutralize_control_markup_in_messages(
                            messages_without_unpriced_media(fitted), None, self.markup_profile
                        ),
                        None,
                        None,
                        strict = True,
                        chat_template_kwargs = _reasoning_kw,
                        continue_final_message = continue_final_message,
                        should_abort = lambda: bool(cancel_event and cancel_event.is_set()),
                    ),
                    estimate_message = estimate_message_tokens_without_unpriced_media,
                    reserve_tokens = _conversation_recall_reserve(thread_id),
                    sticky_dropped = _sticky,
                    sticky_is_checkpoint = _sticky_is_checkpoint,
                    sticky_manual = _sticky_manual,
                    force_reset = compact_context,
                    keeps_boundary = _keeps_compaction_boundary(thread_id),
                    can_reset = _can_reset,
                    recall_offered = False,
                    **_compaction_fit_kwargs(context_policy, compaction_headroom_ratio),
                )
                if truncation:
                    # Inline, not a forged tool exchange: this path sends no tools array,
                    # and strict templates reject a tool role with no catalogue.
                    _recalled = _archive_and_recall(
                        openai_messages,
                        _before_fit,
                        branch_messages = _before_fit,
                        thread_id = thread_id,
                        style = "inline",
                        # Checkpoint mode recalls once, on the turn that reset; rolling has no
                        # such key and keeps recalling on every turn that evicted anything.
                        force_recall = bool(truncation.get("checkpoint_started", True)),
                        # A rescued refusal may evict messages; archive them without recall.
                        recall_done = not truncation["fits"],
                        # This single-shot answer never returns to the prompt.
                        recall_budget_tokens = _retrieval_budget(
                            self._effective_context_length,
                            payload["max_tokens"],
                            truncation.get("prompt_tokens_after") or 0,
                        ),
                        count_tokens = lambda fitted: self.count_chat_tokens(
                            neutralize_control_markup_in_messages(
                                messages_without_unpriced_media(fitted),
                                None,
                                self.markup_profile,
                            ),
                            None,
                            None,
                            strict = True,
                            chat_template_kwargs = _reasoning_kw,
                            continue_final_message = continue_final_message,
                        ),
                    )
                    openai_messages = _recalled["conversation"]
                    truncation = {**truncation, **_recalled["counts"]}
                if truncation and _records_boundary(truncation):
                    truncation = {
                        **truncation,
                        **_boundary_metadata(
                            openai_messages, _before_fit, compaction_headroom_ratio
                        ),
                    }
                payload["messages"] = neutralize_control_markup_in_messages(
                    openai_messages, None, self.markup_profile
                )
                # Reuse the fitted request on respawn; re-running the preflight would
                # emit the same truncation event twice.
                retry_messages = openai_messages
                retry_image_b64 = None
                retry_max_tokens = payload["max_tokens"]
                retry_context_overflow = None
                retry_preflight_context_length = self._effective_context_length
                # `fits` False too: it carries the diagnosis the client needs to explain
                # WHY. Otherwise the user only sees llama-server's error, which reports
                # the whole conversation's size and advises shortening it even when the
                # single message just sent is the part that does not fit.
                if truncation:
                    yield {"type": "context_truncated", **truncation}
            except Exception as exc:
                logger.warning("Could not preflight the rolling context window: %s", exc)
        if stop:
            payload["stop"] = stop
        _apply_seeded_llama_request(payload, seed)
        payload["stream_options"] = {"include_usage": True}

        served_by = getattr(self, "_process", None)
        url = f"{self.base_url}/v1/chat/completions"
        cumulative = ""
        in_thinking = False
        _stream_done = False
        _metadata_usage = None
        _metadata_timings = None
        _metadata_finish_reason = None

        def _replay_on_replacement_server():
            context_overflow_ = retry_context_overflow
            max_tokens_ = retry_max_tokens
            if (
                retry_preflight_context_length is not None
                and retry_preflight_context_length != self._effective_context_length
            ):
                # Refit the compacted prompt against the replacement server's window;
                # any event now reports only additional evictions.
                context_overflow_ = context_overflow
                if max_tokens is None:
                    max_tokens_ = None
            yield from self.generate_chat_completion(
                retry_messages,
                image_b64 = retry_image_b64,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                min_p = min_p,
                max_tokens = max_tokens_,
                repetition_penalty = repetition_penalty,
                presence_penalty = presence_penalty,
                frequency_penalty = frequency_penalty,
                logit_bias = logit_bias,
                stop = stop,
                cancel_event = cancel_event,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
                continue_final_message = continue_final_message,
                seed = seed,
                promote_reasoning_only = promote_reasoning_only,
                perf_callback = perf_callback,
                reasoning_provenance = reasoning_provenance,
                context_overflow = context_overflow_,
                context_policy = context_policy,
                compaction_headroom_ratio = compaction_headroom_ratio,
                # The retry refits for the replacement window and can evict more than
                # the first attempt did. Without the thread those extra turns are
                # archived nowhere and no reserve or boundary applies, on the one path
                # that deliberately compacts again.
                thread_id = thread_id,
                # The retry refits, so it must be told the same about this request's
                # tools as the first attempt was.
                tools_withheld = tools_withheld,
                thinking_budget_tokens = thinking_budget_tokens,
                compact_context = compact_context,
                _allow_respawn_retry = False,
            )

        try:
            with self._open_stream(url, payload, cancel_event) as (
                response,
                first_token_deadline,
            ):
                buffer = ""
                has_content_tokens = False
                # Track content separately: a literal `</think>` can appear in the answer.
                answer_text = ""
                reasoning_text = ""
                _prov_entry = None
                for raw_chunk in self._iter_text_cancellable(
                    response,
                    cancel_event,
                    first_token_deadline = first_token_deadline,
                ):
                    buffer += raw_chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if not line:
                            continue
                        if line == "data: [DONE]":
                            if in_thinking:
                                if has_content_tokens:
                                    # Real thinking + content: close the tag
                                    cumulative += "</think>"
                                    _prov_entry = None
                                    yield cumulative
                                else:
                                    # Only reasoning_content, no content:
                                    # model put its whole reply in reasoning
                                    # (e.g. Qwen3 always-think). Show it as
                                    # the main response, not a thinking block.
                                    cumulative = _finalize_reasoning_only_cumulative(
                                        cumulative,
                                        reasoning_text,
                                        _metadata_finish_reason,
                                        promote_reasoning_only,
                                        resumed_reasoning,
                                    )
                                    _prov_entry = None
                                    yield cumulative
                            _stream_done = True
                            break  # exit inner while
                        if not line.startswith("data: "):
                            continue

                        try:
                            data = json.loads(line[6:])
                            _report_live_llama_timings(perf_callback, data)
                            # Diffusion frame (per-step canvas) from the shim: forward untouched so
                            # the frontend renders it in place. No assistant text, so it never enters
                            # the cumulative content.
                            if data.get("type") == "diffusion_frame":
                                yield data
                                continue
                            # Capture server timings/usage from final chunks.
                            _chunk_timings = data.get("timings")
                            if _chunk_timings:
                                _metadata_timings = _chunk_timings
                            _chunk_usage = data.get("usage")
                            if _chunk_usage:
                                _metadata_usage = _chunk_usage
                            # An error chunk carries no choices, so without this the loop
                            # ignored it and the reply ended with no finish_reason and no
                            # incomplete stamp: to the user, a mid-sentence stop for no
                            # stated reason. Raise so the failure is surfaced.
                            _stream_error = stream_error_from_chunk(data)
                            if _stream_error is not None:
                                raise _stream_error
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                _fr = choices[0].get("finish_reason")
                                if _fr:
                                    _metadata_finish_reason = _fr

                                # Reasoning/thinking tokens: llama-server
                                # sends these as "reasoning_content"; wrap
                                # in <think> tags for the frontend parser.
                                reasoning = delta.get("reasoning_content", "")
                                if reasoning:
                                    reasoning_text += reasoning
                                    if not in_thinking:
                                        # Provenance for think-aware consumers: a
                                        # leading <think> we open here is genuine
                                        # reasoning, unlike literal tags in content.
                                        if reasoning_provenance is not None and not cumulative:
                                            reasoning_provenance["wrapped"] = (
                                                reasoning_provenance.get("wrapped", 0) + 1
                                            )
                                            _prov_entry = {"len": 0}
                                            reasoning_provenance.setdefault("wraps", []).append(
                                                _prov_entry
                                            )
                                        cumulative += "<think>"
                                        in_thinking = True
                                    cumulative += reasoning
                                    if _prov_entry is not None:
                                        _prov_entry["len"] += len(reasoning)
                                    yield cumulative

                                token = delta.get("content", "")
                                if token:
                                    has_content_tokens = True
                                    answer_text += token
                                    if in_thinking:
                                        cumulative += "</think>"
                                        in_thinking = False
                                        _prov_entry = None
                                    cumulative += token
                                    yield cumulative
                        except json.JSONDecodeError:
                            logger.debug(f"Skipping malformed SSE line: {line[:100]}")
                    if _stream_done:
                        break  # exit outer for
                _cut = _quote_cut_event(
                    reasoning_text,
                    answer_text,
                    _metadata_finish_reason,
                    promote_reasoning_only,
                )
                if _cut is not None:
                    yield _cut
                if _metadata_usage or _metadata_timings or _metadata_finish_reason:
                    _metadata_usage = _backfill_usage_from_timings(
                        _metadata_usage, _metadata_timings
                    )
                    yield {
                        "type": "metadata",
                        # Never None: a finish-only metadata event (no usage,
                        # no timings) would otherwise crash consumers that do
                        # usage.get(...) on the non-streaming paths.
                        "usage": _metadata_usage or {},
                        "timings": _metadata_timings,
                        "finish_reason": _metadata_finish_reason,
                    }

        except _LlamaStreamCancelled:
            return
        except httpx.ConnectError as e:
            # Server already down. If this was an MTP crash, recover by
            # reloading without MTP (scheduled in the background) and fail this
            # request. Otherwise the server was likely SIGKILL'd by GPU pressure
            # from a concurrent training run: respawn the same config and retry the
            # generation once (bounded by the private flag, no duplicate output).
            if self._maybe_recover_from_mtp_crash(e):
                raise RuntimeError("Lost connection to llama-server")
            if _allow_respawn_retry and not cumulative and self._respawn_if_dead():
                logger.warning(
                    "llama-server was unreachable; respawned it and retrying the generation"
                )
                yield from _replay_on_replacement_server()
                return
            raise RuntimeError("Lost connection to llama-server")
        except Exception as e:
            if cancel_event is not None and cancel_event.is_set():
                return
            if _is_gpu_device_lost(e):
                self._retire_device_lost_server(served_by)
                if _allow_respawn_retry and not cumulative and self._respawn_if_dead():
                    logger.warning(
                        "llama-server lost its GPU device; restarted it and retrying the generation"
                    )
                    yield from _replay_on_replacement_server()
                    return
                raise
            # Died mid-generation: recover MTP, re-raise unchanged for this request.
            self._maybe_recover_from_mtp_crash(e)
            raise
