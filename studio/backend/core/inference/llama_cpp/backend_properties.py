# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: properties."""

import httpx
import math

from dataclasses import replace
from pathlib import Path
from typing import (
    List,
    Optional,
)
from core.inference.llama_server_args import (
    _effective_tensor_parallel,
    _tensor_parallel_matches_loaded,
    resolve_reasoning_budget,
    resolve_reasoning_budget_message,
)

from .core_types import (
    GgufLoadIntent,
    _EMPTY_THINK_PROBE_LOCK,
    _SPEC_KIND_CAPABILITY,
)
from .draft_args import _normalized_load_mode
from .gguf_cache import (
    _SHARD_RE,
    _mtp_drafter_loads_standalone,
)
from .launch_flags import (
    _diffusion_manual_ngl,
    _swa_full_from_args_or_env,
)
from .launch_support import _local_ssl_context
from .log import logger
from .memory_policy import (
    _extra_args_set_spec_type,
    _planned_main_cache_types,
)
from .placement_args import _metal_device_is_paravirtual
from .reasoning import (
    _REASONING_EFFORT_CLAMPS_LOGGED,
    _clamp_reasoning_effort_to_levels,
    _coerce_reasoning_effort,
)
from .server_log import _GPU_MODEL_BUFFER_RE
from .spec_args import _canonicalize_spec_mode

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _PropertiesMixin:
    # ── Properties ────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._process is not None and self._healthy

    @property
    def is_active(self) -> bool:
        """True if a llama-server process exists (loading or loaded)."""
        return self._process is not None

    def reclaimable_gpu_memory_gb(self) -> Optional[dict[int, float]]:
        """Stable GPU model buffers the disposable llama-server will release.

        The startup log sizes only resident model buffers, so this is a
        conservative lower bound: KV/compute buffers remain charged.  Unlike a
        live process-memory sample, the value cannot race a short generation
        that grows or clears its cache between the free-memory and owner reads.
        """
        process = self._process
        if process is None or process.poll() is not None:
            return None
        physical_ids = list(getattr(self, "_child_gpu_physical_ids", None) or ())
        if not physical_ids:
            return None
        reported: dict[int, float] = {}
        for line in list(self._stdout_lines):
            match = _GPU_MODEL_BUFFER_RE.search(line)
            if match is None:
                continue
            ordinal = int(match.group(1))
            if ordinal >= len(physical_ids):
                return None
            physical = int(physical_ids[ordinal])
            reported[physical] = reported.get(physical, 0.0) + float(match.group(2)) / 1024.0
        if self._process is not process or process.poll() is not None:
            return None
        return reported or None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def _auth_headers(self) -> "Optional[dict[str, str]]":
        """Bearer header for the child's per-launch --api-key, else None (key disabled)."""
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None

    @property
    def model_identifier(self) -> Optional[str]:
        return self._model_identifier

    @property
    def is_vision(self) -> bool:
        """Whether this model takes image input; ``_is_vision`` only records that a
        projector was attached, which happens for audio input too."""
        return self._is_vision and self._mmproj_accepts_image

    @property
    def is_diffusion(self) -> bool:
        """True when the loaded GGUF is a block-diffusion model (DiffusionGemma)."""
        return self._is_diffusion

    @property
    def swa_full(self) -> bool:
        """Whether the active llama-server received full-size SWA mode."""
        return self._swa_full

    @property
    def hf_variant(self) -> Optional[str]:
        return self._hf_variant

    @property
    def gguf_path(self) -> Optional[str]:
        return self._gguf_path

    @property
    def hf_repo(self) -> Optional[str]:
        """HF repo of the loaded model, or None for local/native file loads."""
        return self._hf_repo

    @property
    def mtp_draft_path(self) -> Optional[str]:
        return self._mtp_draft_path

    @property
    def mtp_draft_suppressed_path(self) -> Optional[str]:
        """Drafter the last load resolved and then dropped on purpose, else None.
        Duplicate-load checks accept it alongside mtp_draft_path so a drafter still on
        disk does not read as a settings change."""
        return self._mtp_draft_suppressed_path

    @property
    def spec_fallback_reason(self) -> Optional[str]:
        """Why MTP was disabled on the last MTP-requesting load, else None."""
        return self._spec_fallback_reason

    @property
    def mmproj_fallback_reason(self) -> Optional[str]:
        """How the active model recovered from an mmproj startup failure, else None."""
        return self._mmproj_fallback_reason

    def _binary_changed_since_launch(self) -> bool:
        """Whether a different llama-server is installed than the live one was launched
        from. False when either side is unreadable, so an install still in flight is not
        read as a finished one; the next Apply asks again."""
        # Same path space as the launch recorded: _binary_revision keys on the
        # path string, and on macOS the launch resolves a managed entrypoint to
        # its target, so comparing an unresolved discovery path against it would
        # differ for the same unchanged file and reload the model on every Apply.
        return self._binary_changed_since_revision(self._launch_binary_revision)

    def _binary_changed_since_revision(self, revision: tuple) -> bool:
        """Whether the current selection differs from a captured binary."""
        # No captured revision is no baseline, and discovery cannot recover one.
        if not revision:
            return False
        current = self._binary_revision(
            self._exec_path_for_launch(self._find_llama_server_binary())
        )
        if not current:
            return False
        return current != revision

    def spec_binary_fallback_can_retry(self) -> bool:
        """Whether the binary has since gained what the last load stood down for.

        The UI tells the user to run `unsloth studio update`, which leaves the request
        identical, so the duplicate-load comparators would otherwise dedup the very
        reload that update was meant to enable.

        Ask about the capability the drafter kind needs, not the reason: every kind
        records "binary_no_mtp", so asking about MTP after a DSpark stand-down says yes
        on a build that never gained draft-dspark. "binary_outdated" is an unknown
        architecture, which no flag advertises, so that one compares the file.
        """
        if self._spec_fallback_reason == "binary_outdated":
            return self._binary_changed_since_launch()
        if self._spec_fallback_reason != "binary_no_mtp":
            return False
        if self._launch_binary_revision and not self._binary_changed_since_launch():
            # An untouched file advertises nothing new, and this runs on every
            # duplicate-load check where the probe below is a subprocess.
            return False
        capability = _SPEC_KIND_CAPABILITY.get(self._spec_drafter_kind or "")
        if capability is None:
            return False
        try:
            return bool(self.probe_server_capabilities().get(capability))
        except Exception as exc:
            logger.debug("Could not recheck the speculative capability: %s", exc)
            return False

    @property
    def extra_args(self) -> Optional[List[str]]:
        """Extra llama-server flags from the last load (a copy). None =
        never set, [] = explicitly cleared. Used by the route for
        inheritance."""
        return list(self._extra_args) if self._extra_args is not None else None

    @property
    def requested_extra_args(self) -> Optional[List[str]]:
        """Extras the last load was INVOKED with (a copy), before any launch-time
        rewrite; falls back to the launched list when nothing was rewritten. Keeps a
        rewrite the caller cannot see (the virtualised-Metal drafter drop) from reading
        as a settings change in the duplicate-load comparators."""
        stored = (
            self._requested_extra_args
            if self._requested_extra_args is not None
            else self._extra_args
        )
        return list(stored) if stored is not None else None

    @property
    def requested_n_ctx(self) -> int:
        """n_ctx the last load was invoked with (not the effective cap).
        0 means Auto. Used by the route to detect Auto-vs-explicit flips."""
        return self._requested_n_ctx

    @property
    def reasoning_budget(self) -> int:
        return self._reasoning_budget

    @property
    def reasoning_budget_message(self) -> str:
        return self._reasoning_budget_message

    @property
    def extra_args_source(self) -> Optional[tuple[str, Optional[str]]]:
        """(model_identifier, hf_variant) the stored extra_args came from.
        ``None`` if no extras have ever been recorded. Used by the route
        to refuse cross-model inheritance (#5401)."""
        return self._extra_args_source

    @property
    def context_length(self) -> Optional[int]:
        """Return the effective context length the server is running at."""
        return self._effective_context_length or self._context_length

    @property
    def effective_parallel_slots(self) -> int:
        """Return the serving-slot count the active llama-server actually uses."""
        try:
            slots = int(getattr(self, "_effective_parallel_slots", 1))
        except (TypeError, ValueError):
            slots = 1
        return max(1, slots)

    @property
    def offloaded_layers(self) -> Optional[int]:
        """Layers llama.cpp actually put on a GPU, when its log said so."""
        return self._gpu_offload_layers[0] if self._gpu_offload_layers else None

    @property
    def offload_total_layers(self) -> Optional[int]:
        return self._gpu_offload_layers[1] if self._gpu_offload_layers else None

    @property
    def gpu_backend_unavailable(self) -> bool:
        """Studio found a GPU for this load but llama.cpp reported none."""
        return self._gpu_backend_unavailable

    @property
    def offload_overridden(self) -> bool:
        """Whether the user's own extras (Auto mode keeps -ngl) chose this split."""
        return self._offload_overridden

    @property
    def requested_parallel_slots(self) -> int:
        """--parallel the last load asked for, before any fit-time reduction.
        The reload dedupe compares requested-vs-requested (like requested_n_ctx);
        the effective count would reload forever after a fitter reduction."""
        try:
            slots = int(getattr(self, "_requested_n_parallel", 1))
        except (TypeError, ValueError):
            slots = 1
        return max(1, slots)

    @property
    def requested_n_batch(self) -> Optional[int]:
        """--batch-size the last load asked for; None means llama.cpp default."""
        return self._requested_n_batch

    @property
    def requested_n_ubatch(self) -> Optional[int]:
        """--ubatch-size the last load asked for; None means llama.cpp default."""
        return self._requested_n_ubatch

    @property
    def requested_load_mode(self) -> Optional[str]:
        """--load-mode the last load asked for; None means llama.cpp's own auto.
        The Model Memory settings may have replaced it, which the settings route
        reports; this stays the caller's intent so the dedupe compares like with
        like."""
        return self._requested_load_mode

    @property
    def requested_spec_draft_cache_type(self) -> Optional[str]:
        """Draft KV cache dtype the last load asked for; None means f16."""
        return self._requested_spec_draft_cache_type

    @property
    def requested_ctx_checkpoints(self) -> Optional[int]:
        """--ctx-checkpoints the last load asked for; None means the default 32."""
        return self._requested_ctx_checkpoints

    @property
    def requested_cache_ram(self) -> Optional[int]:
        """--cache-ram (MiB) the last load asked for; None means the default 8192."""
        return self._requested_cache_ram

    @property
    def idle_slot_clearing_active(self) -> bool:
        """Whether an idle slot's KV cells actually come back on this server.

        Under ``--kv-unified`` an idle slot keeps its cells until ``prompt_clear()``,
        which llama-server runs only under ``--cache-idle-slots``. False means releasing
        a slot frees nothing, so nothing may be handed to another caller on the strength
        of it. False until a load has spawned, which is the safe answer.
        """
        return self._idle_slot_clearing_active

    @property
    def max_context_length(self) -> Optional[int]:
        """Return the largest context that fits on this hardware at load time.

        The UI's "safe zone" warning threshold: the ``_fit_context_to_vram``
        binary-search cap for the best GPU subset, or the 4096 fallback if the
        weights exceed 90% of every subset. The slider ceiling is
        ``native_context_length``; dragging above this triggers the warning.
        """
        return self._max_context_length or self._context_length

    @property
    def vram_fit_context_length(self) -> Optional[int]:
        """Largest context the load-time fit priced inside GPU / Metal memory, else None."""
        # A child wholly on CPU (--device none, Vulkan crash fallback, zero layers) holds no VRAM.
        if (
            self._gpu_offload_active is False
            or self._cpu_fallback_reason
            or self._arch_gate_forced_cpu
            or (self._gpu_memory_mode == "manual" and self._gpu_layers == 0)
        ):
            return None
        return self._vram_fit_context_length

    @property
    def native_context_length(self) -> Optional[int]:
        """Return the model's native context length from GGUF metadata."""
        return self._context_length

    def _commit_effective_parallel_slots(self, n_parallel: int) -> None:
        try:
            slots = int(n_parallel)
        except (TypeError, ValueError):
            slots = 1
        self._effective_parallel_slots = max(1, slots)

    def _reset_effective_parallel_slots(self) -> None:
        self._effective_parallel_slots = 1
        # Cleared with the effective count so a stale value can't skew the dedupe.
        self._requested_n_parallel = 1
        self._slots_cut_for_drafter = False
        self._requested_n_batch = None
        self._requested_n_ubatch = None
        self._requested_load_mode = None
        self._requested_spec_draft_cache_type = None
        self._requested_ctx_checkpoints = None
        self._requested_cache_ram = None
        self._idle_slot_clearing_active = False

    @staticmethod
    def _read_rss_bytes(pid: int) -> Optional[int]:
        """Resident set size of ``pid`` in bytes, from /proc/<pid>/status (Linux).
        0 when the status has no VmRSS line (zombie / kernel thread); None where
        /proc is unavailable (macOS/Windows) or the value is unreadable."""
        try:
            with open(f"/proc/{pid}/status", "r", encoding = "utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        # IndexError guards a "VmRSS:" line with no value column.
                        return int(line.split()[1]) * 1024  # kB -> bytes
        except (FileNotFoundError, PermissionError, ValueError, IndexError, OSError):
            return None
        return 0  # readable but no VmRSS line

    def load_progress(self) -> Optional[dict]:
        """Return live model-load progress, or None if not loading.

        During warm-up llama-server mmaps weight shards into page cache before
        pushing layers to VRAM, a window where status only reports ``loading``
        and the UI spinner looks stuck for minutes on large MoEs. Samples
        ``/proc/<pid>/status VmRSS`` against the sum of GGUF shard sizes for a
        real progress bar. Returns ``None`` when no load is in flight.

        Shape::

            {
                "phase": "mmap" | "ready",
                "bytes_loaded": int,   # VmRSS of the llama-server
                "bytes_total":  int,   # sum of shard file sizes
                "fraction": float,     # bytes_loaded / bytes_total, 0..1
            }

        Linux-only; returns ``None`` where ``/proc/<pid>/status`` is unavailable.
        """
        proc = self._process
        if proc is None:
            return None
        pid = proc.pid
        if pid is None:
            return None

        # Sum shard sizes (primary + any extras alongside).
        bytes_total = 0
        gguf_path = self._gguf_path
        if gguf_path:
            primary = Path(gguf_path)
            try:
                if primary.is_file():
                    bytes_total += primary.stat().st_size
            except OSError:
                pass
            # Extra shards share the primary's prefix before the shard index.
            try:
                parent = primary.parent
                stem = primary.name
                m = _SHARD_RE.match(stem)
                prefix = m.group(1) if m else None
                if prefix and parent.is_dir():
                    prefix_lower = prefix.lower()
                    for sibling in parent.iterdir():
                        if (
                            sibling.is_file()
                            and sibling.name.lower().startswith(prefix_lower)
                            and sibling.name != stem
                            and sibling.suffix.lower() == ".gguf"
                        ):
                            try:
                                bytes_total += sibling.stat().st_size
                            except OSError:
                                pass
            except OSError:
                pass

        # VmRSS of the llama-server; None where /proc is unavailable.
        bytes_loaded = LlamaCppBackend._read_rss_bytes(pid)
        if bytes_loaded is None:
            return None

        # RSS climbs as weights page in, then drops once -ngl offloads them to
        # VRAM and the mmap pages are freed. Hold a per-process high-water mark
        # so the bar never regresses to ~8% mid-load (#5740).
        hwm_pid, hwm = getattr(self, "_load_rss_hwm", (None, 0))
        hwm = bytes_loaded if hwm_pid != pid else max(hwm, bytes_loaded)
        self._load_rss_hwm = (pid, hwm)
        bytes_loaded = hwm

        phase = "ready" if self._healthy else "mmap"
        fraction = 0.0
        if bytes_total > 0:
            fraction = min(1.0, bytes_loaded / bytes_total)
        # Once llama-server is healthy the load is complete by definition. With
        # layers offloaded to VRAM (-ngl) the process releases the mmap'd weight
        # pages, so VmRSS sinks back well below the shard total; the raw RSS
        # fraction would then report a partial (~8%) load indefinitely and freeze
        # a fraction-driven progress bar even though the model is ready (#5740).
        if self._healthy:
            if bytes_total > 0:
                bytes_loaded = bytes_total
            fraction = 1.0
        return {
            "phase": phase,
            "bytes_loaded": bytes_loaded,
            "bytes_total": bytes_total,
            "fraction": round(fraction, 4),
        }

    @property
    def chat_template(self) -> Optional[str]:
        return self._chat_template

    @property
    def markup_profile(self):
        """This model's own structural markers, from its GGUF template and vocabulary.

        None when neither could be read, which falls back to the curated patterns: a model
        we cannot profile stays fully swept rather than unprotected (#7066)."""
        # getattr, not attribute access: a lightweight backend double in a test never runs
        # __init__, and an unprofiled model must fall back rather than raise.
        cached = getattr(self, "_markup_profile", None)
        if cached is None:
            from core.inference.chat_template_helpers import model_markup
            cached = (
                model_markup(
                    getattr(self, "_chat_template_override", None)
                    or getattr(self, "_chat_template", None),
                    getattr(self, "_markup_tokens", None),
                ),
            )
            try:
                self._markup_profile = cached
            except AttributeError:
                pass
        return cached[0]

    @property
    def chat_template_override(self) -> Optional[str]:
        return self._chat_template_override

    def _effective_chat_template(self, chat_template_override: Optional[str]) -> Optional[str]:
        """The template to launch with: the caller's override, else a repaired copy of the
        GGUF's own when llama-server's Jinja cannot parse it.

        The repair stays out of ``_chat_template_override``, which the reload check
        compares against the request's intent -- a value no request sends would reload the
        model on every Apply.
        """
        if chat_template_override:
            return chat_template_override
        from core.inference.chat_template_helpers import repair_numeric_member_access

        repaired = repair_numeric_member_access(self._chat_template)
        if repaired:
            logger.warning(
                "The GGUF's chat template indexes with numeric member access, which "
                "llama-server's Jinja rejects; launching with a repaired copy"
            )
        return repaired

    @property
    def supports_reasoning(self) -> bool:
        return self._supports_reasoning

    @property
    def reasoning_always_on(self) -> bool:
        return self._reasoning_always_on

    @property
    def reasoning_style(self) -> str:
        return self._reasoning_style

    @property
    def reasoning_effort_levels(self) -> list:
        """Discrete reasoning_effort levels the template offers; empty for the plain low/medium/high ladder."""
        return self._reasoning_effort_levels

    @property
    def supports_preserve_thinking(self) -> bool:
        return self._supports_preserve_thinking

    @property
    def preserve_thinking_default(self) -> bool:
        return self._preserve_thinking_default

    @property
    def reasoning_default(self) -> bool:
        return self._reasoning_default

    def _reasoning_kwargs(
        self,
        enable_thinking: bool,
        effort: Optional[str] = None,
    ) -> dict:
        if self._reasoning_style == "enable_thinking_effort":
            # GLM-5.2-style: enable_thinking is the on/off gate; when on, leave
            # the template's default effort (max) in place.
            return {"enable_thinking": enable_thinking}
        if self._reasoning_style == "reasoning_effort":
            return _coerce_reasoning_effort(
                getattr(self, "_architecture", None),
                {"reasoning_effort": effort or ("high" if enable_thinking else "low")},
            )
        return {"enable_thinking": enable_thinking}

    def _replay_preserve_thinking(
        self,
        preserve_thinking: Optional[bool],
        messages,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Optional[bool]:
        """True for a thinking-off follow-up with no reasoning in its history, where the template proves
        that only preserve_thinking replays each answer's empty think block as generated."""
        # Omitted reads as off; the probe proves what replays.
        if preserve_thinking is True or not getattr(self, "_supports_preserve_thinking", False):
            return preserve_thinking
        mode = self._request_reasoning_kwargs(enable_thinking, reasoning_effort, None) or {}
        thinking = mode.get("enable_thinking")
        if thinking is None and enable_thinking is None and reasoning_effort is None:
            # Agents rarely send either: llama-server then renders the load-time default.
            thinking = getattr(self, "_reasoning_default", None)
        if thinking is not False:
            return preserve_thinking
        answered = False
        for message in messages or ():
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            answered = True
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                return preserve_thinking
            # Inline reasoning counts too.
            content = message.get("content")
            parts = content if isinstance(content, list) else [content]
            if any(
                "</think>" in (part.get("text") if isinstance(part, dict) else part)
                for part in parts
                if isinstance(part, str)
                or (isinstance(part, dict) and isinstance(part.get("text"), str))
            ):
                return preserve_thinking
        if not answered:
            return preserve_thinking
        return True if self._empty_think_replays_with_preserve() else preserve_thinking

    def _empty_think_probe_key(self) -> tuple:
        return (
            getattr(self, "base_url", None),
            id(getattr(self, "_chat_template", None)),
            id(getattr(self, "_chat_template_override", None)),
        )

    def _empty_think_replays_with_preserve(self) -> bool:
        """Probed once per server: generation prompt + answer prefixes the next render only when kept."""
        key = self._empty_think_probe_key()
        cached = getattr(self, "_empty_think_preserve_probe", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        with _EMPTY_THINK_PROBE_LOCK:
            cached = getattr(self, "_empty_think_preserve_probe", None)
            if cached is not None and cached[0] == key:
                return cached[1]
            return self._probe_empty_think_replay(key)

    def _probe_empty_think_replay(self, key: tuple) -> bool:
        result = False
        try:
            off = dict(self._request_reasoning_kwargs(False, None, None) or {})
            answer = "Studio replay probe answer."
            first = [{"role": "user", "content": "Studio replay probe question."}]
            history = first + [
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "Studio replay probe follow-up."},
            ]
            with httpx.Client(
                timeout = 10,
                headers = self._auth_headers,
                trust_env = False,
                verify = _local_ssl_context(),
            ) as client:
                refused = []

                def _render(msgs, preserve):
                    r = client.post(
                        f"{self.base_url}/apply-template",
                        json = {
                            "messages": msgs,
                            "chat_template_kwargs": {**off, "preserve_thinking": preserve},
                        },
                    )
                    # A 4xx is the template's answer, cached; 408 / 429 mean busy.
                    if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
                        refused.append(r.status_code)
                    prompt = r.json().get("prompt") if r.status_code == 200 else None
                    return prompt if isinstance(prompt, str) else None

                generated = _render(first, False)
                kept = _render(history, True)
                dropped = _render(history, False)
            if generated is None or kept is None or dropped is None:
                if refused:
                    self._empty_think_preserve_probe = (key, False)
                return False
            result = kept.startswith(generated + answer) and not dropped.startswith(
                generated + answer
            )
        except Exception:  # noqa: BLE001 -- an unanswerable probe keeps the caller's setting
            return False
        self._empty_think_preserve_probe = (key, result)
        return result

    def _request_reasoning_kwargs(
        self,
        enable_thinking: Optional[bool],
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
    ) -> Optional[dict]:
        """Build chat_template_kwargs from per-request reasoning fields.

        Merges the active model's reasoning style (``enable_thinking`` or
        ``reasoning_effort``) plus the independent ``preserve_thinking``
        kwarg when the template supports it.
        """
        kwargs: dict = {}
        # Always-on reasoning models hardcode <think> tags and don't consume
        # enable_thinking / reasoning_effort -- skip.
        if self._supports_reasoning and not self._reasoning_always_on:
            if self._reasoning_style == "enable_thinking_effort":
                # GLM-5.2-style: enable_thinking gates thinking on/off, and the
                # reasoning_effort level (e.g. 'high' | 'max') is only meaningful
                # while thinking is on. Disabling is enable_thinking=false; a raw
                # API caller can also disable via the OpenAI-style
                # reasoning_effort="none" sentinel. We never coerce off into a
                # 'low' effort the way gpt-oss does (those models genuinely
                # cannot disable).
                thinking_off = enable_thinking is False or reasoning_effort == "none"
                # Clamp unsupported levels to avoid template defaults or errors.
                effort = (
                    None
                    if thinking_off
                    else _clamp_reasoning_effort_to_levels(
                        reasoning_effort, self._reasoning_effort_levels
                    )
                )
                if effort is not None and effort != reasoning_effort:
                    key = (tuple(self._reasoning_effort_levels), reasoning_effort)
                    if key not in _REASONING_EFFORT_CLAMPS_LOGGED:
                        _REASONING_EFFORT_CLAMPS_LOGGED.add(key)
                        logger.info(
                            f"reasoning_effort '{reasoning_effort}' is not offered by this "
                            f"template {list(self._reasoning_effort_levels)}; using '{effort}'"
                        )
                # A named effort level implies thinking on, so emit enable_thinking
                # even if the caller sent only reasoning_effort (else the template
                # defaults it off and the requested level never renders).
                if enable_thinking is not None or reasoning_effort == "none" or effort is not None:
                    kwargs["enable_thinking"] = not thinking_off
                if effort is not None:
                    kwargs["reasoning_effort"] = effort
            elif self._reasoning_style == "reasoning_effort":
                _levels = getattr(self, "_reasoning_effort_levels", None) or ()
                if reasoning_effort in ("none", "low", "medium", "high") or (
                    reasoning_effort in _levels
                ):
                    kwargs["reasoning_effort"] = reasoning_effort
                elif reasoning_effort == "minimal":
                    kwargs["reasoning_effort"] = "low"
                elif enable_thinking is not None:
                    kwargs["reasoning_effort"] = "high" if enable_thinking else "low"
            else:
                if enable_thinking is not None:
                    kwargs["enable_thinking"] = enable_thinking
        if self._supports_preserve_thinking and preserve_thinking is not None:
            kwargs["preserve_thinking"] = preserve_thinking
        _coerce_reasoning_effort(getattr(self, "_architecture", None), kwargs)
        return kwargs or None

    @property
    def supports_tools(self) -> bool:
        # DiffusionGemma serves via the visual runner, whose live per-step canvas
        # frames are dropped by the agentic tool loop; never route it through tools.
        if self._is_diffusion:
            return False
        return self._supports_tools

    @property
    def supports_tool_passthrough(self) -> bool:
        # supports_tools is forced off for DiffusionGemma (its agentic loop drops the
        # per-step canvas frames), but client passthrough skips that loop, so it uses
        # the real _supports_tools.
        return self._supports_tools

    @property
    def cache_type_kv(self) -> Optional[str]:
        return self._cache_type_kv

    @property
    def tensor_parallel(self) -> bool:
        """Whether --split-mode tensor is active on the loaded server."""
        return self._tensor_parallel

    @property
    def gpu_memory_mode(self) -> str:
        """Active GPU memory strategy: 'auto' or 'manual' (gpu_layers < 0 = Auto/--fit, >= 0 = pinned)."""
        return self._gpu_memory_mode

    @property
    def gpu_layers(self) -> int:
        """Requested --gpu-layers for manual mode (-1 when not manual)."""
        return self._gpu_layers

    @property
    def holds_no_vram(self) -> bool:
        """Whether the resident server is a confirmed zero-VRAM launch, so the GPU
        arbiter can leave an image/video pipeline alone when re-asserting ownership.

        True for a manual zero-offload load whose argv carried no GPU companion, pin
        or tensor mode (``_gpu_offload_active`` None or True keeps the normal path),
        and for an arch-gated launch masked onto the CPU by the "-1" sentinel
        (#7624). The latter arrives through an AUTOMATIC request, so demanding manual
        mode would answer False for the case routes/inference.py:8206 documents and
        leave a VRAM-less server holding the CHAT claim, unloadable by an owner it
        never competed with."""
        if self._arch_gate_forced_cpu:
            return True
        # A diffusion runner reports offload False only when its GPU token was masked: a
        # manual zero split, or an Auto fit that found no room on the card.
        if self._is_diffusion and self._gpu_offload_active is False:
            return True
        return (
            self._gpu_memory_mode == "manual"
            and self._gpu_layers == 0
            and self._gpu_offload_active is False
        )

    @property
    def diffusion_requested_ngl(self) -> Optional[int]:
        """The diffusion split the live runner was ASKED for, or None for the default.

        Differs from ``gpu_layers`` only when the shim could not take the split: that
        reports what is running, this keeps the ask so a repeat still dedupes."""
        return getattr(self, "_diffusion_requested_ngl", None)

    @property
    def n_cpu_moe(self) -> int:
        """MoE expert layers manual mode kept on CPU (--n-cpu-moe); 0 = none."""
        return self._n_cpu_moe

    @property
    def tensor_split(self) -> Optional[List[float]]:
        """Relative model share per GPU (--tensor-split) for the active load.

        Manual mode returns the requested ratio; auto tensor-parallel mode
        returns the normalized ratio that was actually emitted, or None when
        the planner left the split to llama.cpp's default."""
        if self._tensor_split is not None:
            return self._tensor_split
        if self._auto_tensor_split_emitted is not None:
            return list(self._auto_tensor_split_emitted)
        return None

    @property
    def gpu_ids(self) -> Optional[List[int]]:
        """User-picked physical GPU indices, or None for automatic selection."""
        return self._gpu_ids

    @property
    def requested_gpu_ids(self) -> Optional[List[int]]:
        """RAW requested GPU pin (before the fit narrowed it), or None for auto.
        gpu_ids echoes the EFFECTIVE pin for /status."""
        return self._requested_gpu_ids

    def matches_gpu_ids(self, gpu_ids: Optional[List[int]]) -> bool:
        """Whether a requested pin is already satisfied by the active runner.

        A regular GGUF load may narrow the requested placement pool to the
        smallest fitting subset. Accept both the original request and the
        effective status-echoed subset so either can round-trip without a
        needless reload. Diffusion records the lowest device of a pick that fits it,
        else the whole pick it split over, so either form matches.
        """
        if self._is_diffusion:
            full = sorted(int(x) for x in gpu_ids) if gpu_ids else None
            # Raw pick accepted too: a forced-CPU runner records no effective pin, so
            # without it re-sending the same selection would reload every time.
            held = ((self._gpu_ids or None), (self._requested_gpu_ids or None))
            return (full[:1] if full else None) in held or full in held

        requested = [int(x) for x in gpu_ids] if gpu_ids else None
        raw = self._requested_gpu_ids or None
        effective = self._gpu_ids or None
        return requested == raw or requested == effective

    def _adopt_widened_pin(self, pin_ids: List[int]) -> None:
        """Record a companion-widened explicit pin as the effective one (#12467).

        Else /status and dedupe keep the pre-widen pin and a reload drops the companion
        card. A launch that recorded none (forced CPU) stays None.
        """
        if self._gpu_ids is not None:
            self._gpu_ids = [int(i) for i in pin_ids]

    def _record_matching_gpu_request(self, gpu_ids: Optional[List[int]]) -> None:
        """Adopt the caller's explicit pool after a full already-loaded match.

        Matching an effective subset avoids a reload, but the incoming request
        is still the user's latest placement intent. Record it so status and a
        later reload do not restore GPUs the user just removed.
        """
        if self._is_diffusion:
            full = sorted(int(x) for x in gpu_ids) if gpu_ids else None
            # Keep the whole pick when the fit moved off its lowest GPU (a split, or one
            # higher card), or a later pick of that lowest GPU would match and not reload.
            moved = bool(self._gpu_ids) and bool(full) and self._gpu_ids != full[:1]
            self._requested_gpu_ids = full if moved else (full[:1] if full else None)
        else:
            self._requested_gpu_ids = [int(x) for x in gpu_ids] if gpu_ids else None
        if self._last_load_intent is not None:
            self._last_load_intent = replace(
                self._last_load_intent,
                gpu_ids = tuple(self._requested_gpu_ids) if self._requested_gpu_ids else None,
            )

    @property
    def last_load_intent(self) -> Optional[GgufLoadIntent]:
        return self._last_load_intent

    def _runtime_matches_intent(
        self, intent: GgufLoadIntent, effective_extra_args: Optional[list[str]]
    ) -> bool:
        """Whether active runtime settings satisfy one resolved caller intent."""

        if intent.force_reload:
            return False
        if self._requested_n_ctx != int(intent.n_ctx):
            return False
        if not self._is_diffusion and self._requested_n_parallel != max(1, int(intent.n_parallel)):
            return False
        # The same count chosen rather than defaulted is one Auto may not cut.
        if (
            not self._is_diffusion
            and self._slots_cut_for_drafter
            and not intent.n_parallel_from_default
        ):
            return False
        if not self._is_diffusion and (
            self._requested_n_batch != intent.n_batch or self._requested_n_ubatch != intent.n_ubatch
        ):
            return False
        # Same rule for the tuning group: requested against requested, so a server
        # launched with a different value is a reload. A blank value is the llama.cpp
        # default, so two loads that asked for nothing still match, while clearing a
        # knob relaunches.
        if not self._is_diffusion and (
            _normalized_load_mode(self._requested_load_mode)
            != _normalized_load_mode(intent.load_mode)
            or self._requested_spec_draft_cache_type != intent.spec_draft_cache_type
            or self._requested_ctx_checkpoints != intent.ctx_checkpoints
            or self._requested_cache_ram != intent.cache_ram
        ):
            return False

        # The KV cache comparison waits for _invoked_extras below: the scalar
        # self._cache_type_kv alone cannot describe this launch.

        # Toggling vision changes which files the child opens, so it cannot be
        # satisfied by a live server that was launched the other way. Not on the
        # diffusion path: _start_diffusion_server ignores the switch and records it
        # False, so comparing it there makes every identical repeat request a
        # mismatch and tears down a runtime that was already the one asked for.
        if not self._is_diffusion and bool(self._disable_vision) != bool(intent.disable_vision):
            return False

        extra_args = list(effective_extra_args) if effective_extra_args is not None else None
        # Request against request. The live EFFECTIVE value can carry an inherited
        # LLAMA_ARG_THINK_BUDGET* that no request can express, so comparing against it would
        # tear down a healthy server on every load and never converge.
        if not self._is_diffusion and (
            self._requested_reasoning_budget
            != resolve_reasoning_budget(extra_args, intent.reasoning_budget)
            or self._requested_reasoning_budget_message
            != resolve_reasoning_budget_message(extra_args, intent.reasoning_budget_message)
        ):
            return False
        # A request omitting the extras field inherits the LAUNCHED list; anything else is
        # the caller's own. A launch-time rewrite (drafter drop, MTP crash replay) makes
        # the two differ, and the equality below wants what the load was INVOKED with.
        # Judged on the flag, not on value equality: a caller that deliberately sends the
        # stripped list is clearing the failed drafter and must not read as inheriting it.
        _invoked_extras = self.requested_extra_args if intent.extra_args_inherited else extra_args
        # Requested against requested, per axis, the same rule the tuning group above
        # uses. The scalar self._cache_type_kv holds only what Unsloth emitted as a
        # managed flag, so a cache set through extras or the env records None on one
        # side and a type on the other, and an identical repeat /load tears down a
        # healthy server. _requested_cache_types is what the live server was ASKED for,
        # so two equal requests match even when the launch rewrote the cache underneath
        # them. Before ggml-org/llama.cpp#23792 the tensor gate hid this by rewriting
        # the cache away; a layer load has always had it.
        if self._requested_cache_types != _planned_main_cache_types(
            intent.cache_type_kv, _invoked_extras
        ):
            return False
        # A virtualised Metal device never launches a tensor split (CPU-pinned, and
        # --split-mode is overridden with the default layer split), so judge the
        # normalized request there: an extras `-sm tensor`, which the pin deliberately
        # leaves in place, would otherwise never match.
        if _metal_device_is_paravirtual():
            if self._tensor_parallel:
                return False
        elif not _tensor_parallel_matches_loaded(
            extra_args, intent.tensor_parallel, self._tensor_parallel
        ):
            # A mode the gate normalised away is this request normalized, not a
            # different one. Guarded on a RECORDED drop, so a launch that dropped
            # nothing cannot excuse a live tensor server against a layer request.
            if not (self._arch_gate_dropped_tensor_parallel and not self._tensor_parallel):
                return False
        if (
            self._layer_preserves_tensor_intent
            and not _effective_tensor_parallel(extra_args, intent.tensor_parallel)
            and not intent.preserve_multi_gpu_on_layer
        ):
            return False

        if self._is_diffusion:
            requested_ngl = _diffusion_manual_ngl(intent.gpu_memory_mode, intent.gpu_layers)
            if requested_ngl != self.diffusion_requested_ngl:
                return False
            # A split dropped by an older shim still matches until the shim gains
            # --ngl support, at which point the same request should apply it.
            if (
                requested_ngl is not None
                and self._gpu_layers != requested_ngl
                and self.diffusion_split_supported()
            ):
                return False
        else:
            if self._swa_full != _swa_full_from_args_or_env(extra_args):
                return False
            if self._gpu_memory_mode != intent.gpu_memory_mode:
                return False
            if intent.gpu_memory_mode == "manual" and (
                self._gpu_layers != intent.gpu_layers
                or (
                    intent.gpu_layers >= 0
                    and (
                        self._n_cpu_moe != intent.n_cpu_moe
                        or (
                            (tuple(self._tensor_split) if self._tensor_split else None)
                            != (intent.tensor_split or None)
                            # Same rule as the mode above: a dropped ratio is this
                            # request normalized. Guarded on a RECORDED drop, never
                            # None == None. Any other ratio reloads.
                            and not (
                                self._arch_gate_dropped_tensor_split is not None
                                and self._arch_gate_dropped_tensor_split
                                == (intent.tensor_split or None)
                            )
                        )
                    )
                )
            ):
                return False
            # Auto tensor-parallel mode can still carry a user ratio, which the
            # planner's own split does not describe. Two otherwise identical auto
            # requests with different ratios must not reuse the same server --
            # and two with the SAME ratio, or with none at all, must.
            if (
                intent.gpu_memory_mode != "manual"
                and self._tensor_parallel
                and self._auto_tensor_split != self._auto_split_fingerprint(intent.tensor_split)
            ):
                return False

        if not self.matches_gpu_ids(list(intent.gpu_ids) if intent.gpu_ids else None):
            return False

        spec_owned_by_extra_args = _extra_args_set_spec_type(extra_args)
        speculative_type = (
            None
            if spec_owned_by_extra_args
            else _canonicalize_spec_mode(intent.speculative_type) or "auto"
        )
        backend_spec = self._requested_spec_mode
        if spec_owned_by_extra_args:
            if backend_spec is not None:
                return False
        elif speculative_type != (backend_spec or "auto"):
            return False
        # Reload so the next Apply retries the drafter fetch. Worth it when the
        # repo publishes a drafter and the fetch failed; not when it publishes
        # none, which for DSpark is the permanent state of every repo but one and
        # would relaunch an identical server forever.
        if (
            intent.gguf_path is None
            and self._spec_fallback_reason == "drafter_not_found"
            # The fetch did find it and the load dropped it; refetching finds it again.
            and self._mtp_draft_suppressed_path is None
            and speculative_type in ("auto", "mtp", "mtp+ngram", "dspark", "dflash")
            and not (self._spec_drafter_kind == "dspark" and self._dspark_sidecar_absent)
            # DFlash asks through _dflash_retry_needed below, which is set only for
            # retryable failures. A permanent listing error records no answer, so
            # _dflash_sidecar_absent stays False and this arm relaunched forever.
            and self._spec_drafter_kind != "dflash"
            and not spec_owned_by_extra_args
        ):
            return False
        # The stand-down the UI asks the user to fix by updating llama.cpp. That leaves
        # the request identical, so without this the repaired load never happens.
        if (
            speculative_type in ("auto", "mtp", "mtp+ngram", "dspark", "dflash", "ngram")
            and self.spec_binary_fallback_can_retry()
        ):
            return False
        # Under Auto a failed fetch records no fallback reason, so the branch above
        # cannot see it. A repo with no sidecar is _dflash_sidecar_absent's business.
        if self._dflash_retry_needed and speculative_type in ("auto", "dflash"):
            return False
        # Retry a runtime the probe degraded, once the probe starts answering (#8317).
        # The retry window is worth nothing if Apply keeps deduping against the degraded
        # runtime, since nothing would re-probe. The slot clamp is the trap:
        # _requested_n_parallel stores the ASK, so a clamped runtime compares equal and
        # would never be revisited. One reload, not a loop -- it records the conclusive
        # probe and clears this flag, so a build that really lacks the capability
        # re-derives the same runtime and dedupes after. The probe is cached, so it is
        # free on the only path that reaches it.
        if (
            self._capability_probe_inconclusive
            and not self._is_diffusion
            and not self.probe_server_capabilities().get("mtp_probe_inconclusive")
        ):
            return False

        compared_draft_n_max = self._spec_draft_n_max
        if self._spec_fallback_reason == "runtime_error" and self._last_load_intent is not None:
            # The MTP-free recovery clears the runtime value but retains the
            # user's prior value in its intent. Only a changed value should retry MTP.
            compared_draft_n_max = self._last_load_intent.spec_draft_n_max
        draft_depth_matters = (
            self._speculative_type in ("draft-mtp", "draft-dspark", "draft-dflash")
            # Auto's Hybrid Mamba partial-offload stand-down engaged nothing, so the
            # types above cannot see it. The depth priced the rollback copies that
            # made the placement partial, so a change can re-enable MTP.
            or self._spec_fallback_reason in ("runtime_error", "mtp_partial_offload")
        )
        if draft_depth_matters:
            requested_draft_n_max = intent.spec_draft_n_max
            if (requested_draft_n_max is None) != (compared_draft_n_max is None):
                return False
            if requested_draft_n_max is not None and requested_draft_n_max != compared_draft_n_max:
                return False
        if (self._chat_template_override or None) != (intent.chat_template_override or None):
            return False

        if tuple(_invoked_extras or ()) != tuple(self.requested_extra_args or ()):
            return False
        if (intent.gguf_path is not None or intent.compare_mtp_draft) and speculative_type in (
            "auto",
            "mtp",
            "mtp+ngram",
            "dspark",
            "dflash",
        ):
            try:
                # Auto counts as dspark/dflash once it resolved that way: the launch
                # stored that sidecar, so comparing the MTP field (None for these
                # repos) against it would reload a healthy server on every Apply.
                # _speculative_type is reset to "default" when a drafter fails to
                # start and the drafterless retry succeeds, but the launch still
                # recorded the sidecar it resolved. Keying the comparison off it
                # would then compare the intent's (empty) MTP path against that
                # recorded sidecar and reload a healthy server on every Apply.
                # _spec_drafter_kind survives the fallback, so it decides here.
                _compare_dspark = speculative_type == "dspark" or (
                    speculative_type == "auto"
                    and (
                        self._speculative_type == "draft-dspark"
                        or self._spec_drafter_kind == "dspark"
                    )
                )
                _compare_dflash = speculative_type == "dflash" or (
                    speculative_type == "auto"
                    and (
                        self._speculative_type == "draft-dflash"
                        or self._spec_drafter_kind == "dflash"
                    )
                )
                if _compare_dspark:
                    intent_draft = intent.dspark_draft_path
                elif _compare_dflash:
                    intent_draft = intent.dflash_draft_path
                else:
                    intent_draft = intent.mtp_draft_path
                requested_draft = Path(intent_draft).resolve() if intent_draft else None
                # A drafter the last load dropped on purpose counts as launched here:
                # the file is still there, so comparing it against the stored None
                # would reload the same drafter-free server forever.
                _loaded = self._mtp_draft_path or self._mtp_draft_suppressed_path
                loaded_draft = Path(_loaded).resolve() if _loaded else None
            except OSError:
                return False
            if requested_draft != loaded_draft:
                return False
            # Repair a rejected sidecar in place and a path-only comparison would keep the
            # drafter-free server. Only the unloadable drop is re-asked: the paravirtual
            # one is a property of the build, so the same bytes drop again. Cached on
            # (path, mtime, size), so only a repaired file is reparsed.
            if (
                self._mtp_draft_suppressed_reason == "unloadable"
                and self._mtp_draft_suppressed_path
                and _mtp_drafter_loads_standalone(self._mtp_draft_suppressed_path)
            ):
                return False
        return True

    @property
    def n_layers(self) -> Optional[int]:
        """Model layer count (GGUF block_count), or None if unknown."""
        return self._n_layers

    @property
    def n_moe_layers(self) -> int:
        """Number of MoE expert layers (the --n-cpu-moe ceiling), 0 if not MoE.

        block_count minus the leading dense layers (which carry no experts):
        --n-cpu-moe counts from layer 0, so those dense layers are no-ops.
        """
        if not self._n_experts or not self._n_layers:
            return 0
        return max(0, self._n_layers - (self._leading_dense_block_count or 0))

    _EMBEDDING_POOLING_TYPES = frozenset({1, 2, 3})  # llama_pooling_type MEAN, CLS, LAST

    @property
    def is_embedding_gguf(self) -> bool:
        """Whether the loaded GGUF is an embedding model, from its pooling type.

        Only MEAN, CLS and LAST pool into the one vector per sequence that
        `/v1/embeddings` returns. NONE (0) pools nothing, and RANK (4) is a reranker
        head: llama_context sizes its pooled buffer to n_cls_out while llama-server's
        send_embedding reads n_embd_out floats out of it.

        When the header omits pooling_type (common on dedicated encoder arches such as
        nomic-bert), fall back to architecture and model-name hints so llama-server still
        launches with --embedding (#9128).
        """
        pooling = getattr(self, "_pooling_type", None)
        if pooling is not None:
            return pooling in self._EMBEDDING_POOLING_TYPES
        path = getattr(self, "_gguf_path", None)
        if not path:
            return False
        from utils.models.gguf_metadata import is_gguf_embedding_model

        return is_gguf_embedding_model(
            path,
            model_identifier = self._model_identifier,
            architecture = self._architecture,
        )

    @staticmethod
    def _resolve_cpu_moe_flag(
        n_cpu_moe: int, n_moe_layers: int, leading_dense: int
    ) -> Optional[int]:
        """The --n-cpu-moe value (absolute first-N layers), or None to omit it.

        Clamps the requested count to the model's MoE layers, then offsets past
        the leading dense layers (--n-cpu-moe counts from layer 0). Returns None
        for nothing-to-offload (0 requested) or a non-MoE model.
        """
        if n_cpu_moe <= 0 or n_moe_layers <= 0:
            return None
        return leading_dense + min(n_cpu_moe, n_moe_layers)

    @staticmethod
    def _sanitize_tensor_split(tensor_split: Optional[List[float]]) -> List[float]:
        """Per-GPU shares with negative and non-finite entries clamped to 0.

        A direct caller's negative entry would launch a placement different
        from the ratio the UI showed, and inf would pass a plain ``> 0`` total
        gate and emit ``--tensor-split inf,...``. Returns [] for input that
        can't be read as floats (the length gate at the call site then drops
        the split).
        """
        try:
            return [
                x if math.isfinite(x) and x > 0.0 else 0.0 for x in (float(v) for v in tensor_split)
            ]
        except (TypeError, ValueError, OverflowError):
            return []

    @staticmethod
    def _format_tensor_split(values: List[float]) -> str:
        """``--tensor-split`` text for a sanitized ratio, in plain decimal.

        Fixed point rather than ``%g``, which renders a perfectly legal
        ``1000000,1`` as ``1e+06,1``. ``std::stof`` reads that, but the argv a
        user is asked to paste into a bug report should say what they asked
        for, and a whole-number ratio should come back out as the ``3,1`` they
        typed.

        A ratio of very small numbers would round to ``0,0``, which llama.cpp
        normalizes by dividing by a zero total. Only that case is rescaled, by
        the largest weight -- free, because llama.cpp normalizes the list
        anyway, so ``3,1`` and ``75,25`` are already the same instruction.
        """

        def _render(items):
            return [f"{x:.6f}".rstrip("0").rstrip(".") or "0" for x in items]

        rendered = _render(values)
        if values and all(part == "0" for part in rendered):
            peak = max(values)
            if peak > 0:
                rendered = _render(v / peak for v in values)
        return ",".join(rendered)

    @staticmethod
    def _auto_split_fingerprint(tensor_split: Optional[List[float]]) -> Optional[tuple[float, ...]]:
        """What an auto-mode load ASKED for, as a comparable ratio.

        Requested against requested, the same rule the tuning group in
        ``_runtime_matches_intent`` uses, and it has to be: what auto mode
        EMITS is not in the caller's units. The planner's own split is a list
        of per-device MiB budgets, and a user ratio the budget check declined
        is emitted as nothing at all -- so comparing either against the
        caller's ``tensor_split`` mismatches on every repeat of a request that
        has not changed, and reloads a multi-gigabyte model forever.

        Normalized because llama.cpp normalizes: ``-ts 3,1`` and ``-ts 75,25``
        are the same instruction, so they are the same intent here. A ratio
        that sanitizes to nothing is None, which is what an absent ratio is.
        """
        if not tensor_split:
            return None
        values = LlamaCppBackend._sanitize_tensor_split(tensor_split)
        total = sum(values)
        if not values or total <= 0:
            return None
        return tuple(round(v / total, 6) for v in values)

    @property
    def layer_preserves_tensor_intent(self) -> bool:
        """True when a downgraded tensor request kept this layer load multi-GPU."""
        return self._layer_preserves_tensor_intent

    @property
    def speculative_type(self) -> Optional[str]:
        return self._speculative_type

    @property
    def spec_drafter_kind(self) -> Optional[str]:
        """Which drafter the resolution was about: ``mtp``, ``dspark`` or ``dflash``.

        Distinct from ``requested_spec_mode``, which stays ``auto`` when Auto
        resolves the kind itself, and from ``speculative_type``, which reads
        ``default`` once a fallback fires. The UI needs the kind to name the
        right recovery in a fallback notice.
        """
        return self._spec_drafter_kind

    @property
    def requested_spec_mode(self) -> Optional[str]:
        """Canonical UI-facing mode the user requested (see field doc)."""
        return self._requested_spec_mode

    @property
    def spec_draft_n_max(self) -> Optional[int]:
        """User --spec-draft-n-max override active on the load, or None when
        the platform default (6 GPU / 3 CPU) is in effect."""
        return self._spec_draft_n_max
