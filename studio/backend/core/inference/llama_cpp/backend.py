# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The composed LlamaCppBackend class."""

import atexit
import subprocess
import threading
import time

from pathlib import Path
from typing import (
    List,
    Optional,
)

from .backend_binary import _BinaryDiscoveryMixin
from .backend_capabilities import _CapabilityProbeMixin
from .backend_diffusion import _DiffusionRunnerMixin
from .backend_download import _DownloadMixin
from .backend_generation import _GenerationMixin
from .backend_gpu_devices import _GpuDevicesMixin
from .backend_gpu_fit import _GpuFitMixin
from .backend_gpu_memory import _GpuMemoryMixin
from .backend_health import _HealthMixin
from .backend_kv_cache import _KvCacheEstimateMixin
from .backend_launch_retries import _LaunchRetryMixin
from .backend_load import _LoadMixin
from .backend_preflight import _PreflightMixin
from .backend_process_io import _ProcessIoMixin
from .backend_properties import _PropertiesMixin
from .backend_shutdown import _ShutdownMixin
from .backend_slots import _SlotsMixin
from .backend_spill import _TensorSpillMixin
from .backend_start_failures import _StartFailureMixin
from .backend_tokens import _TokenCountingMixin
from .backend_tool_loop import _ToolLoopMixin
from .core_types import (
    GgufLoadIntent,
    _CpuFallbackRuntime,
)


class LlamaCppBackend(
    _PropertiesMixin,
    _BinaryDiscoveryMixin,
    _CapabilityProbeMixin,
    _GpuDevicesMixin,
    _GpuMemoryMixin,
    _GpuFitMixin,
    _KvCacheEstimateMixin,
    _DownloadMixin,
    _ProcessIoMixin,
    _DiffusionRunnerMixin,
    _PreflightMixin,
    _StartFailureMixin,
    _LaunchRetryMixin,
    _LoadMixin,
    _ShutdownMixin,
    _TensorSpillMixin,
    _SlotsMixin,
    _HealthMixin,
    _GenerationMixin,
    _ToolLoopMixin,
    _TokenCountingMixin,
):
    """Manages a llama-server subprocess for GGUF model inference.

    Lifecycle:
        1. load_model(): start llama-server with the GGUF file
        2. generate_chat_completion(): proxy to /v1/chat/completions, stream back
        3. unload_model(): terminate the subprocess
    """

    # Held across "is a teardown running?" and publishing the child, never across a
    # health wait. On the class: doubles built with __new__ never run __init__.
    _spawn_lock = threading.Lock()

    # Held across a whole teardown, so a lifecycle cannot reopen mid-kill. Separate
    # from _spawn_lock so that long hold does not also block a spawn, which only
    # needs to read the flag. Order is always _teardown_lock then _spawn_lock.
    _teardown_lock = threading.Lock()

    def __init__(self, *, manages_processes: bool = True):
        """``manages_processes = False`` builds an INERT probe.

        The constructor is not free: it reaps orphaned llama-servers, walking /proc,
        resolving each candidate's exe and signalling the ones it recognises, then
        registers an atexit handler holding this instance for the life of the process.
        Both are right for the backend that owns a child, neither for a probe that only
        reads a header -- and since the memory-estimate route those helpers run on every
        settings change. Measured on one estimate: five constructions, five /proc scans,
        five atexit handlers, never released, so fifty estimates left 250 of them
        holding 250 backends.

        Default True, so every existing caller keeps today's behaviour.
        """
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        # Advisory memory notice from the load in flight, handed back on LoadResponse.
        # Reset by _begin_load_warnings so one load's notice is never reported against
        # the next.
        self._last_load_warning: Optional[str] = None
        # Same, for a quant fallback: the download fills the pair, the launch publishes it.
        self._variant_fallback_warning: Optional[str] = None
        # The route's Hub notice for the running model, so /status matches the load response.
        self.hub_access_warning: Optional[str] = None
        self._pending_variant_fallback: Optional[tuple[str, str]] = None
        # Set per launch by _record_carveout_advice; None on nearly every load.
        self._last_carveout_advice: Optional[dict] = None
        self._model_identifier: Optional[str] = None
        self._gguf_path: Optional[str] = None
        # Snapshot of the exact file(s) handed to the resident process. A local
        # link can be atomically retargeted while `_gguf_path` stays unchanged.
        self._gguf_load_identity: Optional[tuple] = None
        self._hf_repo: Optional[str] = None
        # Separate MTP drafter launched with the current model; reload-dedup
        # key so a drafter that appears next to the weights forces a reload.
        self._mtp_draft_path: Optional[str] = None
        # Drafter this load resolved then suppressed: unloadable sidecar, or virtualised
        # Metal with no draft-layer flag. The caller keeps detecting it on disk, so
        # without this record every repeat Apply tears down a healthy server.
        self._mtp_draft_suppressed_path: Optional[str] = None
        # WHICH drop it was, "unloadable" or "paravirtual": only the unloadable one is a
        # verdict on the CONTENTS, so only it reloads on repair and renames the reason.
        self._mtp_draft_suppressed_reason: Optional[str] = None
        # Why MTP was disabled on the last load that asked for it (auto on an
        # MTP model, or forced mtp / mtp+ngram), else None. Drives the "update
        # llama.cpp" hint in the UI. "binary_no_mtp" / "binary_outdated" ->
        # a newer prebuilt would help; "runtime_error" -> it may not.
        self._spec_fallback_reason: Optional[str] = None

        # How a requested multimodal projector recovered at startup. `cpu_offload`
        # keeps vision; the other values accompany a deliberate text-only retry.
        self._mmproj_fallback_reason: Optional[str] = None
        # Whether THIS load ran against a probe that did not answer, rather than one
        # that answered "no". An inconclusive probe reports every capability absent, so it
        # silently degrades speculative decoding, the DSpark sidecar and the --kv-unified
        # slot clamp alike. Separate from _spec_fallback_reason, whose UI banner stays
        # suppressed for an unresolved probe, so a reload can still be judged worth
        # retrying (#8317).
        self._capability_probe_inconclusive: bool = False
        self._spec_drafter_kind: Optional[str] = None
        self._dspark_sidecar_absent: bool = False
        self._dflash_sidecar_absent: bool = False
        # Sidecar fetch lost to a blip, not to the repo lacking one
        # (_dflash_sidecar_absent). Worth one more Apply: under Auto nothing else
        # records that the load ended up with no drafter.
        self._dflash_retry_needed: bool = False
        # Which llama-server file this load ran. `unsloth studio update` replaces it
        # in place, so a stand-down blamed on the binary can spot a real update.
        self._launch_binary_revision: tuple = ()
        # Binary a load has selected but not necessarily spawned yet. None means
        # no load is pending; the settings route compares a populated revision
        # before is_active becomes true so a concurrent path save cannot hide a
        # required reload.
        self._binary_revision_pending: Optional[tuple] = None
        # Set after an auto-Vulkan crash recovers with all devices disabled.
        self._cpu_fallback_reason: Optional[str] = None
        self._cpu_fallback_runtime: Optional[_CpuFallbackRuntime] = None
        self._pending_cpu_fallback_cleanups: list = []
        self._hf_variant: Optional[str] = None
        self._is_vision: bool = False
        # Vision deliberately left unloaded by the caller. Tracked so a reload that
        # flips the toggle cannot be satisfied by the live server, which opened a
        # different set of files, and echoed to the client so the switch can reseed
        # from the load it actually ran.
        self._disable_vision: bool = False
        # Narrower than the raw request: image input is off because the user asked,
        # not because the model has no usable projector. Only a model that HAS one
        # can have had it switched off, and the client needs the distinction to point
        # at the switch rather than at a missing file.
        self._vision_disabled_by_user: bool = False
        # Block-diffusion model (e.g. DiffusionGemma): served by the diffusion
        # runner, not llama-server. Set from the GGUF architecture at load.
        self._architecture: Optional[str] = None
        self._is_diffusion: bool = False
        # Whole header walked? Separates "declares no architecture" from "unreadable".
        self._gguf_header_parsed: bool = False
        self._gguf_split_index: Optional[int] = None
        self._diffusion_visual_bin: Optional[str] = None
        self._healthy = False
        self._load_rss_hwm = (None, 0)  # (pid, peak VmRSS) for load_progress
        self._stats_logger = None  # vLLM-style engine-stats poller, set on load
        # Set by _classify_gpu_offload after _wait_for_health.
        self._gpu_offload_active: Optional[bool] = None
        # (offloaded, total) layers llama.cpp reported for this load, when it said.
        self._gpu_offload_layers: Optional[tuple[int, int]] = None
        # Whether the user's own extras pinned the layer split for this load.
        self._offload_overridden: bool = False
        # Studio saw a GPU for this load but llama.cpp's device table did not.
        self._gpu_backend_unavailable: bool = False
        # Diffusion only: the split the running child was asked for.
        self._diffusion_requested_ngl: Optional[int] = None
        self._context_length: Optional[int] = None
        self._effective_context_length: Optional[int] = None
        self._max_context_length: Optional[int] = None
        # Ceiling a fit priced inside the GPU / Metal budget; None for an anchor or floor.
        self._vram_fit_context_length: Optional[int] = None
        self._effective_parallel_slots: int = 1
        # --parallel the last load asked for, before any fit-time reduction.
        self._requested_n_parallel: int = 1
        self._slots_cut_for_drafter: bool = False
        # Budget the running child was sized against; None before any load, so the
        # settings route reports a stale budget instead of nagging on every save.
        self._vram_fraction_launched: Optional[float] = None
        # Budget a load has committed to but not published, covering planning ->
        # Popen as _memory_pending_launch does for Model Memory placement.
        self._vram_fraction_pending: Optional[float] = None
        # --batch-size / --ubatch-size the last load asked for; none = defaults or extras / env
        self._requested_n_batch: Optional[int] = None
        self._requested_n_ubatch: Optional[int] = None
        # The expert-spill micro-batch raise: the pair before it, and the argv tokens.
        self._moe_spill_batch_restore: Optional[tuple[Optional[int], Optional[int]]] = None
        self._moe_spill_batch_tokens: Optional[tuple[list[str], list[str]]] = None
        self._moe_cache_flags: List[str] = []
        # The tuning group the last load asked for; none = defaults, or left to
        # extras / env. What was REQUESTED, not what ran: Model Memory can replace
        # the load mode, and Windows full-offload tuning owns the two cache knobs.
        self._requested_load_mode: Optional[str] = None
        self._requested_spec_draft_cache_type: Optional[str] = None
        self._requested_ctx_checkpoints: Optional[int] = None
        self._requested_cache_ram: Optional[int] = None
        self._idle_slot_clearing_active: bool = False
        self._chat_template: Optional[str] = None
        self._markup_tokens: list = []
        self._markup_profile = None
        self._chat_template_override: Optional[str] = None
        self._supports_reasoning: bool = False
        self._reasoning_always_on: bool = False
        self._reasoning_budget: int = -1
        self._reasoning_budget_message: str = ""
        # What the load asked for, before the environment. See _runtime_matches_intent.
        self._requested_reasoning_budget: int = -1
        self._requested_reasoning_budget_message: str = ""
        self._reasoning_style: str = "enable_thinking"
        self._reasoning_effort_levels: list = []
        self._supports_preserve_thinking: bool = False
        self._preserve_thinking_default: bool = False
        self._supports_tools: bool = False
        self._cache_type_kv: Optional[str] = None
        # Whether --split-mode tensor was applied on the active load.
        self._tensor_parallel: bool = False
        # GPU memory strategy applied on the active load ("auto"/"manual").
        self._gpu_memory_mode: str = "auto"
        # Manual-mode load options (echoed back so the UI round-trips them).
        self._gpu_layers: int = -1
        # MoE expert layers to keep on CPU (--n-cpu-moe); 0 = none.
        self._n_cpu_moe: int = 0
        # Relative model share per GPU (--tensor-split), in GPU order; None =
        # default (llama.cpp splits by free VRAM).
        self._tensor_split: Optional[List[float]] = None
        # Same ratio, but for auto tensor-parallel mode. The planner may emit a
        # weighted split or fall back to a user-supplied ratio; recording it lets
        # _runtime_matches_intent detect a changed ratio and reload.
        self._auto_tensor_split: Optional[tuple[float, ...]] = None
        # The actual split that was emitted to llama-server in auto mode, used for
        # /status reporting. Manual mode stores this in _tensor_split.
        self._auto_tensor_split_emitted: Optional[tuple[float, ...]] = None
        # Tokens the tensor-spill plan added to this argv, so a retry that re-places
        # the model can strip them with _without_subsequence. Empty when it abstained.
        self._spill_plan_flags: List[str] = []
        # Ratio the arch gate dropped from this argv, so the duplicate-load check
        # knows the live server already IS this request normalized (#7624).
        self._arch_gate_dropped_tensor_split: Optional[tuple[float, ...]] = None
        # Same for the split MODE: the gate drops --split-mode tensor when it forces
        # CPU or leaves one survivor, and the request still asks for it.
        self._arch_gate_dropped_tensor_parallel: bool = False
        # Gate masked every device away, so an AUTOMATIC request became a zero-VRAM
        # launch (#7624), which is why holds_no_vram cannot demand manual mode.
        self._arch_gate_forced_cpu: bool = False
        # User-picked physical GPU indices (None = automatic selection).
        self._gpu_ids: Optional[List[int]] = None
        # Physical GPU ids in the exact ordinal order exposed to the current
        # llama-server child.  Automatic placement deliberately leaves
        # ``_gpu_ids`` unset, so reclaim accounting must not infer this mapping
        # from status state after a fit/arch gate narrowed the child mask.
        self._child_gpu_physical_ids: Optional[tuple[int, ...]] = None
        # RAW requested GPU pin, before the fit narrowed it. self._gpu_ids records the
        # EFFECTIVE (fit-narrowed) pin for /status; dedupe compares this raw value so a
        # [0, 1] narrowed to [0] and re-sent as [0, 1] still matches (#7239).
        self._requested_gpu_ids: Optional[List[int]] = None
        # Layer load kept multi-GPU only to honor a downgraded tensor request, so a
        # later explicit tensor-off reloads instead of deduping to it (#6659).
        self._layer_preserves_tensor_intent: bool = False
        self._reasoning_default: bool = True
        self._speculative_type: Optional[str] = None
        # Canonical UI-facing mode the user requested
        # (auto/mtp/ngram/mtp+ngram/off/ngram-simple). Round-tripped through the
        # status API so the dropdown reflects the picked mode, not the resolved
        # flag set (auto on a 27B MTP GGUF resolves to draft-mtp but reads "Auto").
        self._requested_spec_mode: Optional[str] = None
        # User --spec-draft-n-max override (None = platform default).
        self._spec_draft_n_max: Optional[int] = None
        # KV-cache estimation fields (populated by _read_gguf_metadata)
        self._n_layers: Optional[int] = None
        # MoE metadata (populated by _read_gguf_metadata): expert count (>0 =
        # MoE) and leading dense-layer count (offsets --n-cpu-moe, which counts
        # from layer 0). See the n_moe_layers property.
        self._n_experts: Optional[int] = None
        self._leading_dense_block_count: Optional[int] = None
        self._n_kv_heads: Optional[int] = None
        self._n_kv_heads_by_layer: Optional[list[int]] = None
        self._n_heads: Optional[int] = None
        self._embedding_length: Optional[int] = None
        # llama_pooling_type from the GGUF; present and non-zero marks an embedding model.
        self._pooling_type: Optional[int] = None
        # For the compute-graph buffer estimate; vocab from the tokens array len.
        # feed_forward_length is the widest layer when the GGUF stores one per layer.
        self._feed_forward_length: Optional[int] = None
        self._feed_forward_length_by_layer: Optional[list[int]] = None
        self._expert_used_count: Optional[int] = None
        self._expert_feed_forward_length: Optional[int] = None
        self._expert_shared_feed_forward_length: Optional[int] = None
        self._expert_shared_count: Optional[int] = None
        # Nemotron 3.5: routed experts run in this latent width, not n_embd.
        self._moe_latent_size: Optional[int] = None
        # Gemma 3n / Gemma 4 per-layer input embedding width.
        self._embedding_length_per_layer_input: Optional[int] = None
        self._vocab_size: Optional[int] = None
        # Set from reserves_micro_batch_outputs() by the callers that know the binary.
        self._reserves_micro_batch_outputs: bool = False
        self._resumes_thoughts: bool = True
        # Architecture-aware KV fields for 5-path estimation
        self._kv_key_length: Optional[int] = None
        self._kv_value_length: Optional[int] = None
        self._sliding_window: Optional[int] = None
        self._sliding_window_pattern: Optional[list[bool]] = None
        self._full_attention_interval: Optional[int] = None
        self._kv_lora_rank: Optional[int] = None
        self._key_length_mla: Optional[int] = None
        self._kv_key_length_swa: Optional[int] = None
        self._kv_value_length_swa: Optional[int] = None
        self._ssm_inner_size: Optional[int] = None
        self._ssm_state_size: Optional[int] = None
        self._ssm_group_count: Optional[int] = None
        self._ssm_conv_kernel: Optional[int] = None
        self._kda_head_dim: Optional[int] = None
        # Last N layers reuse earlier layers' KV and don't allocate their own
        # cache (Gemma 3n / Gemma 4: <arch>.attention.shared_kv_layers).
        self._shared_kv_layers: Optional[int] = None
        # MTP head count (llama.cpp #22673); >0 enables --spec-type draft-mtp.
        self._nextn_predict_layers: Optional[int] = None
        self._lock = threading.Lock()
        # Wraps load_model() end-to-end so concurrent loads serialise and never
        # coexist as two llama-server processes (#5401). RLock so MTP-crash
        # recovery can re-acquire it for its nested load_model.
        self._serial_load_lock = threading.RLock()
        # Serialises mid-session respawns so many generations hitting a killed
        # server trigger at most one reload (see _respawn_if_dead).
        self._respawn_lock = threading.Lock()
        # Bumped by every unload. load_model clears _cancel_event, so a respawn that
        # raced an unload needs a signal that survives the clear (see _respawn_if_dead).
        self._unload_epoch = 0
        # Set by the in-app updater while it swaps prebuilt binaries; load_model()
        # rejects fast so no server starts from a half-swapped binary.
        self._llama_update_in_progress = False
        # Last extra_args / requested n_ctx, preserved across unload so the chat
        # UI's /unload+/load Apply path can inherit them (#5401).
        # ``_extra_args_source`` records the (model_identifier, hf_variant) the
        # stored args came from so the route can refuse cross-model inheritance.
        self._extra_args: Optional[List[str]] = None
        # Extras as REQUESTED, before a launch-time rewrite (the virtualised-Metal
        # drafter drop strips the spec group), so the duplicate-load comparators can
        # compare requested-vs-requested like _requested_n_ctx does.
        self._requested_extra_args: Optional[List[str]] = None
        self._extra_args_source: Optional[tuple[str, Optional[str]]] = None
        self._requested_n_ctx: int = 0
        # Last healthy caller intent for crash recovery. Memory-only and never logged.
        self._last_load_intent: Optional[GgufLoadIntent] = None
        self._mtp_runtime_fallback_lock = threading.Lock()
        self._mtp_runtime_fallback_in_progress = False
        # Background watchdog so an MTP crash recovers even when no request
        # observes it (direct proxy endpoints, or nothing in flight).
        self._mtp_watchdog_thread: Optional[threading.Thread] = None
        self._mtp_watchdog_stop = threading.Event()
        # True when the launch actually runs MTP (Unsloth- or user/env-driven);
        # gates the watchdog and recovery so pass-through MTP is covered.
        self._mtp_runtime_fallback_active = False
        self._stdout_lines: list[str] = []
        self._stdout_thread: Optional[threading.Thread] = None
        # Wakes the health probe on subprocess readiness/exit, so a ready or crashed
        # server is not hidden behind the 500 ms fallback poll.
        self._health_probe_event = threading.Event()
        # llama-server tee log (see _drain_stdout / _kill_process).
        self._llama_log_fh = None
        self._llama_log_path: Optional[Path] = None
        self._cancel_event = threading.Event()
        self._api_key: Optional[str] = None
        self._api_key_file: Optional[Path] = None
        self._slot_save_dir: Optional[str] = None
        self._slot_save_binary: Optional[tuple[str, int]] = None
        # (gguf_identity, launch_fingerprint) snapshotted at load, so a later slot
        # save can tell whether the model files were swapped on disk since load.
        self._slot_loaded_identity: Optional[tuple] = None
        self._prompt_cache_disabled: bool = False
        self._swa_full: bool = False
        self._kv_cache_unified: bool = False
        # (mlock, reserves_ram) the running process was launched with, or None
        # when this policy does not govern it (the diffusion runner has no
        # load-mode). _memory_policy_active records that the launch differed
        # from an unmanaged one, so disabling the toggles can undo it.
        self._memory_state: Optional[tuple[bool, bool]] = None
        # mmap and dio share the pair above, so without this they are indistinguishable.
        self._memory_direct_io: Optional[bool] = None
        # The same chain one toggle apart; against _memory_direct_io that is "would
        # reloading change the loader".
        self._memory_dio_applicable: bool = False
        # The managed tokens, so a rung that gives up the offload can take them back out.
        self._memory_dio_flags: list[str] = []
        # Compared against by the strip, so it can never remove a hand-typed pair.
        self._memory_dio_user_tokens: list[str] = []
        self._memory_policy_active: bool = False
        # What still marks the launch with the managed flags removed (scrubbed env, vetoed
        # extra), so a withdrawing rung can spot a child now equal to an unmanaged one.
        self._memory_policy_extras_touched: bool = False
        # False when a launch is fully offloaded to a discrete GPU, where
        # page-locking host RAM is skipped on purpose.
        self._memory_mlock_applicable: bool = True
        # The "--load-mode none" tokens the FIT emitted, so paths that replace the
        # placement can take them back out. Empty for a mode the user asked for.
        self._fit_load_mode_flags: list[str] = []
        # The pair a launch in flight is committed to, None when none is. ONE attribute:
        # a separate marker and snapshot read out of step answered a save with
        # reload_required=false about a child already committed to the pre-save flags.
        self._memory_pending_launch: Optional[tuple[bool, bool]] = None
        # True when the resident model came from an explicit UI load rather than
        # the OpenAI API, so the idle unload can be scoped to API-loaded models.
        # Not on GgufLoadIntent: that is compared for equality to detect
        # duplicate loads, and a provenance field there would force a reload.
        self._loaded_by_user_action: bool = False
        self._n_ubatch: int = self._DEFAULT_N_UBATCH
        self._flash_attn_enabled: bool = True
        self._effective_cache_types: tuple[str, str] = ("f16", "f16")
        # The pair this load ASKED for, before any launch-time rewrite.
        self._requested_cache_types: tuple[str, str] = ("f16", "f16")
        # Total KV allocation context across all slots. _effective_context_length
        # becomes the per-slot request limit after /props reconciliation.
        self._kv_cache_context_total: Optional[int] = None
        # True once a probe has completed; cleared on transient failure.
        self._is_audio: bool = False
        self._audio_type: Optional[str] = None
        self._audio_probed: bool = False
        self._codec_failure = threading.local()
        # Audio INPUT capability (distinct from _is_audio, which is TTS output).
        self._has_audio_input: bool = False
        # Video INPUT capability, from llama-server's /props modalities. True
        # only when the mmproj, the build and ffmpeg all line up, none of which
        # the GGUF alone can tell us.
        self._has_video_input: bool = False
        self._mmproj_has_audio: bool = False  # clip.has_audio_encoder, set at load
        # clip.projector_type, set at load: the family whose image-token ceiling KV
        # admission sizes an image against.
        self._mmproj_projector_type: Optional[str] = None
        # clip.has_vision_encoder, set at load; True keeps an undeclared projector capable.
        self._mmproj_accepts_image: bool = True
        # Monotonic timestamp set in _kill_process; read by load_model
        # to decide whether to wait for the VRAM reclaim to finish.
        self._last_kill_monotonic: float = 0.0
        self._planned_vram_mib: dict[int, int] = {}
        self._pending_plan_mib: dict[int, int] = {}

        if manages_processes:
            _reaped = self._kill_orphaned_servers()
            if _reaped:
                # Reaped VRAM frees lazily; arm the settle wait so the first load
                # waits before ranking GPUs by free memory.
                self._last_kill_monotonic = time.monotonic()
            atexit.register(self._cleanup)
