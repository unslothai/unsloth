# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LlamaCppBackend: KV cache VRAM estimation."""

from typing import (
    Callable,
    Iterable,
    Mapping,
    Optional,
)
from core.inference.llama_server_args import (
    ctx_checkpoints_within_host_budget,
    parse_ctx_checkpoints_override,
)

from .launch_flags import (
    _DEFAULT_LLAMA_N_UBATCH,
    _SSM_EVERY_LAYER_ARCHS,
    _SSM_FFN_FREE_LAYER_ARCHS,
    _TARGET_KV_EXCLUDES_NEXTN_ARCHS,
    _env_ctx_checkpoints_override,
)
from .log import logger
from .memory_policy import (
    _ASSUMED_MAX_VOCAB,
    _MTP_VRAM_RESERVE_FRAC,
    _active_vram_fraction,
    _arch_mtp_skips_target_kv_copy,
    _kv_bytes_per_elem,
    _kv_cache_cell_layout,
    _pad_kv_cells,
    _vram_usable_mib,
)
from .placement_args import ctx_checkpoints_default_for_caps

# `LlamaCppBackend` is bound here by the package __init__ once the class exists, so the
# `LlamaCppBackend.<name>` references below resolve exactly as they did in one module.


class _KvCacheEstimateMixin:
    # ── KV cache VRAM estimation ─────────────────────────────────────

    def _can_estimate_kv(self) -> bool:
        """True if we have enough GGUF metadata to estimate KV cache size."""
        if self._n_layers is None:
            return False
        # MLA: kv_lora_rank suffices (K-only cache).
        if self._kv_lora_rank is not None:
            return True
        # New-style: need explicit key AND value dimensions.
        if self._kv_key_length is not None and self._kv_value_length is not None:
            return True
        # Legacy: need embedding_length + a head count (scalar or per-layer).
        return self._embedding_length is not None and (
            self._n_kv_heads is not None
            or self._n_heads is not None
            or self._n_kv_heads_by_layer is not None
        )

    def _kv_layer_weights(
        self,
        n_ctx: int,
        *,
        swa_full: bool = False,
        n_parallel: int = 1,
        kv_unified: bool = True,
        n_ubatch: Optional[int] = None,
        flash_attn: bool = True,
        cache_type_kv: Optional[str] = None,
    ) -> "list[int]":
        """Relative KV size per layer, or [] when the shape is uniform/unknown.

        Only the RATIOS matter: the planner scales this to the byte-accurate total
        it already has, so it never becomes a second, competing estimate of the
        cache size. It exists because the total says nothing about WHERE the big
        caches land, which on a multi-device split is the whole question -- a
        sliding-window layer holds a window's worth of cells while a
        full-attention layer holds the entire context, ~100x more at 128K against
        a 1K window (llama-kv-cache-iswa.cpp:69-104).

        The cell geometry MUST be the one the total was priced at, so the launch
        knobs that change it travel too: ``--swa-full`` collapses the two cache
        sizes into one, and the compact SWA allowance is per-slot plus a
        micro-batch and padded (_estimate_kv_cache_bytes path 3), not a bare
        ``min(window, n_ctx)``. Pricing a ratio the total does not have moves cache
        bytes onto the wrong card, a per-device shortfall under the ``--fit off``
        the plan pins.

        Trailing layers that reuse an earlier layer's cache get weight 0: Gemma 3n
        / Gemma 4 set ``n_layer_kv_from_start = n_layer_all - shared_kv_layers``
        and ``llama_hparams::has_kv`` is false past it (llama-hparams.cpp:275-279),
        so they allocate nothing. Giving them weight would spread the same total
        over layers that hold no cache.

        Only ever describes _estimate_kv_cache_bytes PATH 3, which picks its path
        before it looks at the window. The earlier paths price a different
        quantity: path 1 (MLA) caches one compressed K latent per layer with NO V
        and no window/full distinction, path 2 (hybrid recurrent) caches only 1 in
        full_attention_interval layers. A window-shaped vector against either is a
        different model of the cache, not an approximation, so answer [] and let
        the planner abstain. Reachable, not hypothetical: dots3note reads
        KV_LORA_RANK and ATTENTION_SLIDING_WINDOW(_PATTERN) in the same loader.

        Omits the per-slot context-checkpoint extra deliberately: llama.cpp's
        checkpoints are HOST buffers (common/common.cpp:2282-2297), so no VRAM
        total this is scaled against contains them either.
        """
        pattern = self._sliding_window_pattern
        n_layers = self._n_layers or 0
        if not pattern or n_layers <= 0:
            return []
        # Mirror _estimate_kv_cache_bytes' branch order: anything it returns before
        # path 3 is a shape this vector cannot describe.
        if self._kv_lora_rank is not None:
            return []
        if self._ssm_inner_size is not None and self._full_attention_interval is not None:
            return []
        swa = self._sliding_window or 0
        if swa <= 0:
            return []
        n_kv = self._n_kv_heads or self._n_heads or 1
        key_len = self._kv_key_length or 0
        val_len = self._kv_value_length or 0
        if not key_len or not val_len:
            return []
        key_len_swa = self._kv_key_length_swa or key_len
        val_len_swa = self._kv_value_length_swa or val_len
        # Same floor as _estimate_kv_cache_bytes: a bad GGUF cannot zero the vector.
        n_layers_kv = max(1, n_layers - (self._shared_kv_layers or 0))
        # Same cell layout the byte total was computed with.
        slots, streams, cells_per_stream = _kv_cache_cell_layout(n_ctx, n_parallel, kv_unified)
        full_cells = max(1, cells_per_stream * streams)
        if swa_full:
            swa_cells = full_cells
        else:
            ubatch = max(0, int(self._DEFAULT_N_UBATCH if n_ubatch is None else n_ubatch))
            swa_limit = int(swa) * (slots if kv_unified else 1) + ubatch
            swa_cells = max(1, _pad_kv_cells(min(cells_per_stream, swa_limit)) * streams)
        # With flash attention OFF llama.cpp cannot use a ragged V cache: every
        # layer's V is padded to hparams.n_embd_v_gqa_max() over the WHOLE model,
        # which is what _estimate_kv_cache_bytes charges (_max_kv_value_width). The
        # V half goes constant while K stays per-layer, so an unpadded vector
        # prices a ratio the total does not have. bpe_v is floored at f16 for a quantised
        # cache, and with V constant that asymmetry moves the ratio too, so carry both
        # rather than cancelling one.
        bpe_k = _kv_bytes_per_elem(cache_type_kv)
        bpe_v = bpe_k if flash_attn else max(bpe_k, _kv_bytes_per_elem("f16"))
        padded_v_width = None if flash_attn else self._max_kv_value_width(val_len, val_len_swa)
        weights: list[int] = []
        for layer_idx in range(n_layers):
            if layer_idx >= n_layers_kv:
                weights.append(0)
                continue
            layer_n_kv = self._kv_heads_for_layer(layer_idx, n_kv)
            is_swa = layer_idx < len(pattern) and bool(pattern[layer_idx])
            key_bytes = layer_n_kv * (key_len_swa if is_swa else key_len) * bpe_k
            value_bytes = (
                layer_n_kv * (val_len_swa if is_swa else val_len)
                if padded_v_width is None
                else padded_v_width
            ) * bpe_v
            width = key_bytes + value_bytes
            weights.append(int(width * (swa_cells if is_swa else full_cells)))
        return weights if any(weights) else []

    def _kv_heads_for_layer(self, layer_idx: int, fallback: int) -> int:
        if self._n_kv_heads_by_layer is not None and layer_idx < len(self._n_kv_heads_by_layer):
            return self._n_kv_heads_by_layer[layer_idx]
        return fallback

    def _recurrent_state_bytes(self, n_parallel: int = 1) -> int:
        """VRAM for the conv + recurrent state of linear-attention layers.

        Hybrids pair an MLA/attention cache with llama_memory_recurrent, which
        allocates per recurrent layer and per sequence, always f32, offloaded
        beside the KV cache. Unlike KV this does not scale with context, so it
        is the whole reservation at short contexts and the bisection in
        _fit_context_to_vram cannot shrink it away.

        Sizes follow llama_hparams::n_embd_r/n_embd_s. This helper covers the
        per-layer KDA layout; Hybrid Mamba uses its architecture-level interval
        and group count in _mamba_recurrent_state_bytes.
        """
        # Lightweight probes may omit fields used only by other architectures.
        heads_by_layer = getattr(self, "_n_kv_heads_by_layer", None)
        n_layers_raw = getattr(self, "_n_layers", None)
        kda_head_dim = getattr(self, "_kda_head_dim", None)
        if not heads_by_layer or not n_layers_raw or not kda_head_dim:
            return 0
        n_recurrent = sum(1 for i in range(self._n_layers) if self._kv_heads_for_layer(i, 1) == 0)
        if n_recurrent == 0:
            return 0
        head_dim, n_head = self._kda_head_dim, self._n_heads or 1
        d_conv = self._ssm_conv_kernel or 4
        n_embd_r = 3 * max(0, d_conv - 1) * n_head * head_dim
        n_embd_s = head_dim * head_dim * n_head
        return int(n_recurrent * (n_embd_r + n_embd_s) * 4 * max(1, n_parallel))

    def _rollback_state_bytes(self, n_parallel: int = 1) -> int:
        """Size one target-context rollback snapshot for either recurrent family."""
        return self._mamba_recurrent_state_bytes(n_parallel) or self._recurrent_state_bytes(
            n_parallel
        )

    def _ctx_checkpoint_bytes(
        self,
        cache_type_kv: Optional[str] = None,
        *,
        swa_full: bool = False,
        flash_attn: bool = True,
    ) -> int:
        """Host bytes of one slot's context checkpoint: the recurrent state, else the SWA window."""
        recurrent = self._rollback_state_bytes(1)
        if recurrent > 0:
            return recurrent

        def kv(checkpoints: int) -> int:
            return self._estimate_kv_cache_bytes(
                1,
                cache_type_kv,
                swa_full = swa_full,
                ctx_checkpoints = checkpoints,
                flash_attn = flash_attn,
            )

        return max(0, kv(1) - kv(0))

    def _bounded_ctx_checkpoints(
        self,
        n_parallel: int,
        server_caps: Mapping[str, object],
        extra_args: Optional[Iterable[str]] = None,
        env: Optional[Mapping[str, str]] = None,
        *,
        cache_type_kv: Optional[str] = None,
        swa_full: bool = False,
        flash_attn: bool = True,
    ) -> Optional[int]:
        """Return an automatic context-checkpoint cap, or None to preserve the argv."""
        flag = server_caps.get("ctx_checkpoints_flag")
        if not flag:
            return None
        # Extras and LLAMA_ARG_CTX_CHECKPOINTS are the operator's; emitting overrules them.
        if parse_ctx_checkpoints_override(extra_args) is not None:
            return None
        if _env_ctx_checkpoints_override(env) is not None:
            return None
        per_checkpoint = self._ctx_checkpoint_bytes(
            cache_type_kv, swa_full = swa_full, flash_attn = flash_attn
        )
        if per_checkpoint <= 0:
            return None
        total_ram_mib = self._host_memory_capacity_mib()
        # This build's own default: capping against 32 would raise a build that keeps 3.
        upstream_default = ctx_checkpoints_default_for_caps(server_caps)
        bounded = ctx_checkpoints_within_host_budget(
            per_checkpoint,
            n_parallel,
            (total_ram_mib * 1024 * 1024) if total_ram_mib else None,
            upstream_default = upstream_default,
        )
        if bounded >= upstream_default:
            return None
        logger.info(
            "Capping llama-server context checkpoints at %d per slot (this build's default %d): "
            "each one snapshots %.1f MiB of this model's state, so the default "
            "would hold %.1f GiB of host RAM across %d slot(s).",
            bounded,
            upstream_default,
            per_checkpoint / (1024**2),
            upstream_default * per_checkpoint * max(1, n_parallel) / (1024**3),
            max(1, n_parallel),
        )
        return bounded

    def _target_kv_excludes_nextn(self) -> bool:
        """Whether this model's TARGET KV cache skips the embedded MTP blocks.

        See _TARGET_KV_EXCLUDES_NEXTN_ARCHS for why this is per-architecture
        rather than a property of the nextn key.

        A hybrid recurrent header answers yes without consulting that list. Every
        hybrid Mamba architecture llama.cpp accepts a nextn key from -- Qwen3-Next,
        Qwen3.5, Qwen3.5-MoE, Nemotron-H -- is filtered at llama-model.cpp:2289,
        and the recurrent half is sized on n_layer() regardless
        (llama-memory-recurrent.cpp:29), so the ssm dims are sufficient evidence
        and a GGUF whose header names an arch this build has never heard of still
        gets the right answer. Everything else must be named, so an unreadable or
        unknown architecture keeps every block.
        """
        if not self._nextn_predict_layers:
            return False
        if self._ssm_inner_size is not None and self._full_attention_interval is not None:
            return True
        arch = getattr(self, "_architecture", None)
        return bool(arch) and str(arch).strip().lower() in _TARGET_KV_EXCLUDES_NEXTN_ARCHS

    def _ssm_recurrent_layer_count(self, n_layers: int) -> int:
        """SSM layers of a header with no attention interval, by llama.cpp's is_recr rules."""
        arch = getattr(self, "_architecture", None)
        if arch in _SSM_EVERY_LAYER_ARCHS:
            return n_layers
        heads = getattr(self, "_n_kv_heads_by_layer", None)
        if not heads or getattr(self, "_kda_head_dim", None):
            return 0
        ffn = (
            getattr(self, "_feed_forward_length_by_layer", None)
            if arch in _SSM_FFN_FREE_LAYER_ARCHS
            else None
        ) or []
        return sum(
            1
            for i, n_kv in enumerate(heads[:n_layers])
            if n_kv == 0 and not (i < len(ffn) and ffn[i] > 0)
        )

    def _mamba_recurrent_state_bytes(
        self,
        n_parallel: int = 1,
        n_rs_seq: int = 0,
    ) -> int:
        """VRAM for Hybrid Mamba conv/SSM state, including rollback copies.

        llama.cpp allocates one f32 recurrent state per parallel sequence. MTP
        additionally keeps ``--spec-draft-n-max`` rollback states per sequence
        while verifying draft tokens, so its allocation is ``1 + n_rs_seq``
        times the base state.
        """
        n_layers_raw = getattr(self, "_n_layers", None)
        d_inner_raw = getattr(self, "_ssm_inner_size", None)
        d_state_raw = getattr(self, "_ssm_state_size", None)
        n_group_raw = getattr(self, "_ssm_group_count", None)
        d_conv_raw = getattr(self, "_ssm_conv_kernel", None)
        fai_raw = getattr(self, "_full_attention_interval", None)
        # Mamba1 (Jamba, Mamba) writes no group count; llama.cpp reads it as 0.
        if not all(
            value is not None
            for value in (
                n_layers_raw,
                d_inner_raw,
                d_state_raw,
                d_conv_raw,
            )
        ) or (fai_raw is not None and n_group_raw is None):
            return 0
        # Excludes the embedded MTP blocks, as _estimate_kv_cache_bytes does and
        # for the same reason: llama.cpp keeps them out of the target's n_layer.
        n_layers = max(
            0,
            int(n_layers_raw or 0) - int(getattr(self, "_nextn_predict_layers", None) or 0),
        )
        if fai_raw is not None:
            fai = int(fai_raw or 0)
            n_attn = -(-n_layers // fai) if fai > 0 else n_layers
            n_recurrent = max(0, n_layers - n_attn)
        else:
            n_recurrent = self._ssm_recurrent_layer_count(n_layers)
        if n_recurrent == 0:
            return 0
        d_inner = int(d_inner_raw or 0)
        d_state = int(d_state_raw or 0)
        n_group = int(n_group_raw or 0)
        d_conv = int(d_conv_raw or 0)
        n_embd_r = max(0, d_conv - 1) * (d_inner + 2 * n_group * d_state)
        n_embd_s = d_state * d_inner
        return int(
            n_recurrent * (n_embd_r + n_embd_s) * 4 * max(1, n_parallel) * max(1, 1 + n_rs_seq)
        )

    def _legacy_head_dim(self) -> int:
        """Head-dim fallback for GGUFs without explicit key/value dims. Reached
        only via the legacy branch of _can_estimate_kv(), so _embedding_length
        is non-None here."""
        return self._embedding_length // self._n_heads if self._n_heads else 128  # type: ignore[operator]

    def _max_kv_value_width(
        self,
        default_len: int,
        swa_len: Optional[int] = None,
    ) -> int:
        """llama.cpp's hparams.n_embd_v_gqa_max() over every model layer."""
        n_layers = self._n_layers or 1
        n_kv = self._n_kv_heads or self._n_heads or 1
        if self._sliding_window_pattern is None:
            max_len = max(default_len, swa_len or default_len)
            return max(
                self._kv_heads_for_layer(layer_idx, n_kv) * max_len for layer_idx in range(n_layers)
            )
        return max(
            self._kv_heads_for_layer(layer_idx, n_kv)
            * (
                (swa_len or default_len)
                if (
                    layer_idx < len(self._sliding_window_pattern)
                    and self._sliding_window_pattern[layer_idx]
                )
                else default_len
            )
            for layer_idx in range(n_layers)
        )

    def _estimate_kv_cache_bytes(
        self,
        n_ctx: int,
        cache_type_kv: Optional[str] = None,
        *,
        swa_full: bool = False,
        n_parallel: int = 1,
        kv_unified: bool = True,
        n_ubatch: Optional[int] = None,
        ctx_checkpoints: int = 0,
        flash_attn: bool = True,
    ) -> int:
        """Estimate KV cache VRAM for a given context length.

        5-path architecture-aware estimation:
          1. MLA      -- compressed KV latent + RoPE, K-only (no separate V)
          2. Hybrid   -- only attention layers need KV (Mamba layers don't)
          3. SWA      -- sliding-window layers use compact or full cache cells
          4. GQA      -- standard full KV with explicit key/value dimensions
          5. Legacy   -- fallback using embed // n_heads

        Server-flag knobs (mirror llama-server's CLI):
          swa_full        -- --swa-full: SWA layers cache full n_ctx (path 3->4).
          n_parallel      -- --parallel slots: controls per-slot stream padding.
          kv_unified      -- --kv-unified: one shared stream vs one per slot.
          n_ubatch        -- --ubatch-size: SWA cache's processing headroom.
          ctx_checkpoints -- --ctx-checkpoints: N snapshots per slot. An SWA
                             model snapshots its window; a hybrid recurrent one
                             snapshots its WHOLE recurrent state (see path 2).
                             They live in host RAM, never VRAM.
          flash_attn      -- False pads variable-width V tensors to the model max.

        Returns 0 if metadata is insufficient.
        """
        if not self._can_estimate_kv() or n_ctx <= 0:
            return 0

        # GGUF block_count includes embedded MTP blocks, and llama.cpp keeps
        # those out of the target context for SOME architectures only -- see
        # _TARGET_KV_EXCLUDES_NEXTN_ARCHS. Where it does, their KV lives in the
        # draft context and _mtp_draft_kv_bytes prices it separately; where it
        # does not, the target cache still allocates them and subtracting here
        # would under-reserve.
        n_layers = self._n_layers  # type: ignore[assignment]
        if self._target_kv_excludes_nextn():
            n_layers = max(1, n_layers - (self._nextn_predict_layers or 0))
        # Gemma 3n / Gemma 4 reuse earlier KV in the last ``shared_kv_layers``
        # blocks (no cache). Floor at 1 so a bad GGUF can't zero out KV.
        shared = self._shared_kv_layers or 0
        n_layers_kv = max(1, n_layers - shared)
        n_kv = self._n_kv_heads or self._n_heads or 1  # type: ignore[assignment]

        # Bytes per element depends on KV cache quantization
        bpe_k = _kv_bytes_per_elem(cache_type_kv)
        # The automatic FA-off retry rewrites an invalid quantized V cache to
        # f16. Pricing that viable retry here avoids under-reserving it.
        bpe_v = bpe_k if flash_attn else max(bpe_k, _kv_bytes_per_elem("f16"))

        slots, streams, cells_per_stream = _kv_cache_cell_layout(n_ctx, n_parallel, kv_unified)
        total_cells = cells_per_stream * streams
        ubatch = max(
            0,
            int(self._DEFAULT_N_UBATCH if n_ubatch is None else n_ubatch),
        )

        # A hybrid snapshot is the whole recurrent state, not a context-scaled window.
        recurrent_checkpoints = slots * max(0, ctx_checkpoints) * self._rollback_state_bytes(1)

        # Path 1: MLA (DeepSeek-V2/V3, GLM-4.7, GLM-5, Kimi-K2.5)
        # One compressed KV latent per token/layer (shared across heads); V is
        # reconstructed from it, no separate V cache. key_length = kv_lora_rank
        # + rope_dim. MLA GGUFs set head_count_kv=1; default to 1 if absent to
        # avoid falling back to n_heads (e.g. 128 for DeepSeek-V3) which 128x's.
        if self._kv_lora_rank is not None:
            n_kv_mla = self._n_kv_heads or 1
            rope_dim = self._key_length_mla or 64
            key_len = self._kv_key_length or (self._kv_lora_rank + rope_dim)
            # Hybrid MLA (Kimi-K3): head_count_kv is a per-layer array whose KDA
            # linear-attention layers are 0 and hold a constant-size state instead
            # of a growing cache. Only the non-zero layers allocate KV, so counting
            # every block overstates it by the hybrid ratio (93 blocks vs 24 real
            # ones is 3.9x, or 78 GiB of phantom cache at 1M context) and shrinks
            # the auto-fit context for no reason. Uniform MLA keeps n_layers_kv.
            n_mla_layers = n_layers_kv
            if self._n_kv_heads_by_layer:
                attn = sum(
                    1 for i in range(n_layers_kv) if self._kv_heads_for_layer(i, n_kv_mla) > 0
                )
                n_mla_layers = max(1, attn)
            return (
                int(n_mla_layers * total_cells * n_kv_mla * key_len * bpe_k)
                + self._recurrent_state_bytes(n_parallel)
                + recurrent_checkpoints
            )

        key_len = self._kv_key_length
        val_len = self._kv_value_length

        # Path 2: Hybrid Mamba/Attention (Qwen3.5-27B, Qwen3.5-35B-A3B)
        # Only 1 in N layers is attention; the rest are Mamba (no KV cache).
        if self._ssm_inner_size is not None and self._full_attention_interval is not None:
            fai = self._full_attention_interval
            n_attn = -(-n_layers // fai) if fai > 0 else n_layers  # ceiling division
            recurrent = self._mamba_recurrent_state_bytes(n_parallel)
            if key_len is not None and val_len is not None:
                v_width = n_kv * val_len if flash_attn else self._max_kv_value_width(val_len)
                return (
                    int(n_attn * total_cells * (n_kv * key_len * bpe_k + v_width * bpe_v))
                    + recurrent
                    + recurrent_checkpoints
                )
            head_dim = self._legacy_head_dim()
            return (
                int(n_attn * total_cells * n_kv * 2 * head_dim * bpe_k)
                + recurrent
                + recurrent_checkpoints
            )

        # Path 3: Sliding window (Gemma 2/3/3n/4, gpt-oss, Cohere2 ...). Pattern
        # from the resolver; if absent, falls through to the legacy 1/4-global
        # heuristic. --parallel N accounting (verified against llama-server):
        # non-SWA cells total n_ctx across streams. Compact SWA adds one processing
        # micro-batch to the window allowance and pads to 256 cells; unified mode
        # holds all slots in one stream, while non-unified mode has one stream per
        # slot. --swa-full expands SWA to each stream's full context.
        if (
            self._sliding_window is not None
            and self._sliding_window > 0
            and key_len is not None
            and val_len is not None
        ):
            swa = self._sliding_window
            if swa_full:
                swa_cells_total = total_cells
            else:
                swa_limit = swa * (slots if kv_unified else 1) + ubatch
                swa_cells_per_stream = min(cells_per_stream, swa_limit)
                swa_cells_per_stream = _pad_kv_cells(swa_cells_per_stream)
                swa_cells_total = swa_cells_per_stream * streams
            key_len_swa = self._kv_key_length_swa or key_len
            val_len_swa = self._kv_value_length_swa or val_len
            padded_v_width = None if flash_attn else self._max_kv_value_width(val_len, val_len_swa)
            if self._sliding_window_pattern is not None:
                global_bytes = 0.0
                swa_bytes = 0.0
                checkpoint_extra_per_slot = 0.0
                # Only layers that allocate their own KV; trailing shared layers
                # reuse earlier caches.
                for layer_idx in range(n_layers_kv):
                    layer_n_kv = self._kv_heads_for_layer(layer_idx, n_kv)
                    is_swa = (
                        layer_idx < len(self._sliding_window_pattern)
                        and self._sliding_window_pattern[layer_idx]
                    )
                    layer_key_bytes = layer_n_kv * (key_len_swa if is_swa else key_len) * bpe_k
                    layer_value_bytes = (
                        layer_n_kv * (val_len_swa if is_swa else val_len)
                        if padded_v_width is None
                        else padded_v_width
                    ) * bpe_v
                    layer_kv_bytes = layer_key_bytes + layer_value_bytes
                    if is_swa:
                        swa_bytes += swa_cells_total * layer_kv_bytes
                        if ctx_checkpoints > 0 and not swa_full:
                            checkpoint_extra_per_slot += ctx_checkpoints * swa * layer_kv_bytes
                    else:
                        global_bytes += total_cells * layer_kv_bytes
                return int(global_bytes + swa_bytes + slots * checkpoint_extra_per_slot)
            n_global = max(1, n_layers_kv // 4)
            n_swa = n_layers_kv - n_global
            global_v_width = n_kv * val_len if padded_v_width is None else padded_v_width
            swa_v_width = n_kv * val_len_swa if padded_v_width is None else padded_v_width
            kv_per_token = n_kv * key_len * bpe_k + global_v_width * bpe_v
            kv_per_token_swa = n_kv * key_len_swa * bpe_k + swa_v_width * bpe_v
            global_bytes = n_global * total_cells * kv_per_token
            swa_bytes = n_swa * swa_cells_total * kv_per_token_swa
            checkpoint_extra_per_slot = (
                ctx_checkpoints * n_swa * swa * kv_per_token_swa
                if ctx_checkpoints > 0 and not swa_full
                else 0.0
            )
            return int(global_bytes + swa_bytes + slots * checkpoint_extra_per_slot)

        ssm_state = self._mamba_recurrent_state_bytes(n_parallel) + recurrent_checkpoints

        # Path 4: Standard GQA with explicit key/value dimensions
        if key_len is not None and val_len is not None:
            padded_v_width = None if flash_attn else self._max_kv_value_width(val_len)
            bytes_per_cell = 0.0
            for layer_idx in range(n_layers_kv):
                layer_n_kv = self._kv_heads_for_layer(layer_idx, n_kv)
                v_width = layer_n_kv * val_len if padded_v_width is None else padded_v_width
                bytes_per_cell += layer_n_kv * key_len * bpe_k + v_width * bpe_v
            return int(total_cells * bytes_per_cell) + ssm_state

        # Path 5: Legacy fallback (old GGUFs without explicit dimensions)
        head_dim = self._legacy_head_dim()
        # llama.cpp defaults head_count_kv to head_count, so a pure SSM (head_count 0) has no KV.
        layer_default = self._n_kv_heads if self._n_kv_heads is not None else (self._n_heads or 0)
        heads = [self._kv_heads_for_layer(i, layer_default) for i in range(n_layers_kv)]
        # Without flash attention V is padded to the widest layer, as on Path 4.
        v_heads = sum(heads) if flash_attn else len(heads) * max(heads)
        return int((sum(heads) * bpe_k + v_heads * bpe_v) * head_dim * total_cells) + ssm_state

    def _draft_backend_for(self, drafter_path: str) -> Optional["LlamaCppBackend"]:
        """Lightweight backend with a drafter GGUF's metadata, to size its own KV
        via _estimate_kv_cache_bytes. Cached per path; None if unreadable."""
        cache = getattr(self, "_draft_backend_cache", None)
        if cache is not None and cache[0] == drafter_path:
            return cache[1]
        db: Optional[LlamaCppBackend] = None
        try:
            db = LlamaCppBackend.__new__(LlamaCppBackend)
            for attr in (
                "_context_length",
                "_n_layers",
                "_n_kv_heads",
                "_n_heads",
                "_embedding_length",
                "_kv_key_length",
                "_kv_value_length",
                "_kv_lora_rank",
                "_sliding_window",
                "_sliding_window_pattern",
                "_ssm_inner_size",
                "_ssm_state_size",
                "_ssm_group_count",
                "_ssm_conv_kernel",
                "_full_attention_interval",
                "_key_length_mla",
                "_n_kv_heads_by_layer",
                "_kv_key_length_swa",
                "_kv_value_length_swa",
                "_shared_kv_layers",
                "_nextn_predict_layers",
            ):
                setattr(db, attr, None)
            db._model_identifier = "mtp-draft"
            db._read_gguf_metadata(drafter_path)
        except Exception as e:  # unreadable drafter -> caller falls back
            logger.debug(f"Could not read drafter GGUF for MTP budget: {e}")
            db = None
        self._draft_backend_cache = (drafter_path, db)
        return db

    def _mtp_draft_kv_bytes(
        self,
        n_ctx: int,
        *,
        drafter_path: Optional[str] = None,
        draft_cache_type_k: Optional[str] = None,
        draft_cache_type_v: Optional[str] = None,
        n_parallel: int = 1,
        swa_full: bool = False,
        kv_unified: bool = True,
        n_ubatch: Optional[int] = None,
        flash_attn: bool = True,
    ) -> Optional[int]:
        """Draft KV cache bytes at n_ctx, sized from GGUF dims (K and V types are
        independent). Separate drafter (Gemma): its own KV via _estimate_kv_cache_bytes
        at the heavier type. Embedded head (Qwen): nextn_predict_layers attention
        layers from the main dims. None when dims are missing (flat fallback)."""
        if n_ctx <= 0:
            return None
        bpe_k = _kv_bytes_per_elem(draft_cache_type_k)
        bpe_v = _kv_bytes_per_elem(draft_cache_type_v)
        if drafter_path:
            db = self._draft_backend_for(drafter_path)
            if db is None or not db._can_estimate_kv():
                return None
            # Gemma 4 assistant layers share the target context's final global
            # and SWA KV tensors, so only the drafter weights add memory.
            if getattr(db, "_architecture", None) == "gemma4-assistant":
                return 0
            heavier = draft_cache_type_k if bpe_k >= bpe_v else draft_cache_type_v
            # The drafter uses the main model's slot and stream layout, so its
            # compact SWA and per-stream padding must follow the same settings.
            kv = db._estimate_kv_cache_bytes(
                n_ctx,
                heavier,
                n_parallel = n_parallel,
                swa_full = swa_full,
                kv_unified = kv_unified,
                n_ubatch = n_ubatch,
                flash_attn = flash_attn,
            )
            return kv if kv > 0 else None
        nextn = self._nextn_predict_layers or 0
        n_kv = self._n_kv_heads or self._n_heads
        k_len = self._kv_key_length
        v_len = self._kv_value_length
        if not (nextn and n_kv and k_len and v_len):
            return None
        # The embedded MTP head is one draft layer, so a quantized draft KV can't
        # amortize its overhead and fits *less* context than f16 (llama.cpp#24102).
        # Floor it at f16: a quantized override is priced as f16, f32 keeps its 4
        # bytes. The separate-drafter branch is multi-layer, so it keeps its type.
        f16_bpe = _kv_bytes_per_elem("f16")
        bpe_k = max(bpe_k, f16_bpe)
        bpe_v = max(bpe_v, f16_bpe)
        _, streams, cells_per_stream = _kv_cache_cell_layout(n_ctx, n_parallel, kv_unified)
        v_width = n_kv * v_len
        if not flash_attn:
            v_width = self._max_kv_value_width(
                v_len,
                self._kv_value_length_swa,
            )
        return int(nextn * (n_kv * k_len * bpe_k + v_width * bpe_v) * cells_per_stream * streams)

    def _cpu_resident_draft_bytes(
        self,
        n_ctx: int,
        *,
        drafter_path: Optional[str],
        draft_cache_type_k: Optional[str] = None,
        draft_cache_type_v: Optional[str] = None,
        n_parallel: int = 1,
        swa_full: bool = False,
        kv_unified: bool = True,
        n_ubatch: Optional[int] = None,
        flash_attn: bool = True,
    ) -> Optional[int]:
        """Host-RAM footprint of a SEPARATE drafter pinned to the CPU (``-ngld 0``).

        The VRAM budget drops such a drafter, and rightly: none of it is on the
        GPU. The load-mode fit asks the other question -- is the whole load
        resident without demand paging -- and there the bytes have not gone
        anywhere, they have moved to host RAM. llama.cpp carries the process-wide
        load mode into the draft model: ``common_base_params_to_speculative``
        starts from ``result = params`` and overrides only devices/model/
        n_gpu_layers, and ``--load-mode none`` clears ``use_mmap``
        (llama-model-loader.cpp), so the draft GGUF becomes a real anonymous
        allocation rather than an evictable file mapping. Charging it is the same
        reasoning that keeps a CPU-pinned projector in the footprint.

        0 when no separate drafter is CPU-pinned. ``None`` when the drafter is
        there but cannot be priced, so the caller abstains instead of claiming a
        fit that is short a whole draft GGUF.
        """
        if not drafter_path:
            return 0
        try:
            weights = self._get_gguf_size_bytes(drafter_path)
        except Exception:
            weights = 0
        if weights <= 0:
            return None
        kv = self._mtp_draft_kv_bytes(
            n_ctx,
            drafter_path = drafter_path,
            draft_cache_type_k = draft_cache_type_k,
            draft_cache_type_v = draft_cache_type_v,
            n_parallel = n_parallel,
            swa_full = swa_full,
            kv_unified = kv_unified,
            n_ubatch = n_ubatch,
            flash_attn = flash_attn,
        )
        if kv is None:
            return None
        # The drafter has a llama_context of its own, so its decode graph is allocated
        # wherever it runs -- here, the host. _soft_overhead charges that constant only
        # while _mtp_reserves_gpu, which is False for exactly this placement.
        return int(weights) + max(0, int(kv)) + self._MTP_DRAFT_COMPUTE_BYTES

    def _estimate_mtp_overhead_bytes(
        self,
        n_ctx: int,
        *,
        spec_draft_n_max: int = 0,
        draft_cache_type_k: Optional[str] = None,
        draft_cache_type_v: Optional[str] = None,
        drafter_path: Optional[str] = None,
        draft_weights_bytes: int = 0,
        n_parallel: int = 1,
        mtp_keeps_target_ctx: bool = True,
        target_rollback: bool = True,
        swa_full: bool = False,
        kv_unified: bool = True,
        n_ubatch: Optional[int] = None,
        flash_attn: bool = True,
    ) -> Optional[int]:
        """MTP draft reserve at ``n_ctx`` = draft KV (grows with ctx) + separate-
        drafter weights + target verification state. MLA duplicates the target
        KV context, while Hybrid Mamba adds recurrent rollback copies. The other
        verify buffers ride in the ctx-fit headroom (no tuned constant). None when
        the draft KV can't be sized (caller keeps the flat fallback).
        ``draft_weights_bytes`` is the drafter file size (0 for embedded).
        ``mtp_keeps_target_ctx`` is True for MTP draft modes (which keep the
        duplicated target context) and False for separate-drafter spec modes
        (draft-simple/draft-eagle3), which do not. ``target_rollback`` is the
        wider set: every draft-model type llama.cpp gives the target n_rs_seq for
        (draft-mtp/eagle3/dflash/dspark), which is all of them but draft-simple.
        Both default to charging, so an unsure caller over-reserves."""
        draft_kv = self._mtp_draft_kv_bytes(
            n_ctx,
            drafter_path = drafter_path,
            draft_cache_type_k = draft_cache_type_k,
            draft_cache_type_v = draft_cache_type_v,
            n_parallel = n_parallel,
            swa_full = swa_full,
            kv_unified = kv_unified,
            n_ubatch = n_ubatch,
            flash_attn = flash_attn,
        )
        weights = max(0, draft_weights_bytes)
        # MLA models (GLM-5.x, DeepSeek, Kimi-K2) under MTP keep a *second* full copy
        # of the target model's KV context for draft verification -- llama.cpp's
        # `ctx_tgt=yes` -- allocated at f16 regardless of the main cache type. It is
        # ~the main KV again and dwarfs the embedded draft head (GLM-5.2 @ 1M ctx:
        # a ~2 GiB head next to a ~89 GiB target copy), so omitting it lets auto-fit
        # pick a context that fits on paper but OOMs cublasCreate at the first
        # decode. Gated on both MLA (kv_lora_rank present) and the engaged mode
        # actually being MTP: non-MLA MTP (Qwen/Gemma) keeps no such copy, and the
        # separate-drafter spec modes (draft-simple/draft-eagle3) load a small
        # distinct drafter with its own KV -- already counted in draft_kv/weights --
        # rather than duplicating the target, so they must not be charged for it.
        # Third gate: a NextN-only MTP context allocates no copy, and charging one
        # trips drafter_no_vram, losing the MTP being reserved for.
        target_ctx_copy = 0
        if (
            mtp_keeps_target_ctx
            and self._kv_lora_rank is not None
            and not _arch_mtp_skips_target_kv_copy(getattr(self, "_architecture", None))
        ):
            target_ctx_copy = self._estimate_kv_cache_bytes(
                n_ctx,
                "f16",
                n_parallel = n_parallel,
                swa_full = swa_full,
                kv_unified = kv_unified,
                n_ubatch = n_ubatch,
                flash_attn = flash_attn,
            )
        # Hybrid Mamba targets keep one recurrent state in the normal target
        # context, counted by _estimate_kv_cache_bytes. Verification adds one
        # rollback copy per drafted token, in the TARGET context, so a separate
        # drafter file pays it exactly like the embedded head: ``target_rollback``
        # decides, not ``drafter_path`` (see _TARGET_ROLLBACK_SPEC_TYPES). The
        # dominant hidden cost on Qwen3.5/3.8 at multiple parallel slots.
        target_recurrent_copies = 0
        if target_rollback and spec_draft_n_max > 0:
            target_recurrent_copies = self._rollback_state_bytes(n_parallel) * spec_draft_n_max
        spec_compute = self._mtp_draft_compute_bytes(
            n_ctx,
            drafter_path = drafter_path,
            n_ubatch = n_ubatch,
            n_parallel = n_parallel,
            spec_draft_n_max = spec_draft_n_max,
        )
        if draft_kv is None:
            # KV unsized (exotic/remote drafter): still reserve known weights + any
            # MLA target copy so a large config can't launch over budget (the small
            # unsized draft KV rides in the cushion). Nothing known -> None, so the
            # caller keeps the flat fallback.
            total = weights + target_ctx_copy + target_recurrent_copies
            return total + spec_compute if total > 0 else None
        return draft_kv + weights + target_ctx_copy + target_recurrent_copies + spec_compute

    def _mtp_draft_compute_bytes(
        self,
        n_ctx: int,
        *,
        drafter_path: Optional[str] = None,
        n_ubatch: Optional[int] = None,
        n_parallel: int = 1,
        spec_draft_n_max: int = 0,
    ) -> int:
        """Draft mask and activations beyond the separately charged floor, plus
        target verification rows. Use the drafter's dimensions when available,
        otherwise the target's dimensions for an embedded head."""
        verify_rows = self._spec_verify_rows_bytes(n_parallel, spec_draft_n_max)
        if n_ctx <= 0:
            return verify_rows
        ub = max(1, int(self._DEFAULT_N_UBATCH if n_ubatch is None else n_ubatch))
        width = self._compute_activation_width()
        db = self._draft_backend_for(drafter_path) if drafter_path else None
        if db is not None:
            draft_width = getattr(db, "_compute_activation_width", lambda: 0)()
            if draft_width > 0:
                width = draft_width + 2 * (getattr(self, "_embedding_length", None) or 0)
        activations = int(width * ub * 4 * self._COMPUTE_BUFFER_SAFETY)
        spec = verify_rows + max(0, ub * 2 * n_ctx + activations - self._MTP_DRAFT_COMPUTE_BYTES)
        if getattr(db, "_architecture", None) == "dflash":
            spec = max(spec, self._DFLASH_DRAFT_COMPUTE_BYTES)
        return spec

    def _spec_verify_rows_bytes(self, n_parallel: int, spec_draft_n_max: int) -> int:
        """Target output rows for verified draft tokens, wherever the drafter runs."""
        return (
            max(1, n_parallel)
            * max(0, spec_draft_n_max)
            * self._SPEC_VERIFY_ROW_COPIES
            * (getattr(self, "_vocab_size", None) or _ASSUMED_MAX_VOCAB)
            * 4
        )

    def _mtp_reserve_note(
        self,
        reserve_bytes: int,
        *,
        n_ctx: int,
        n_parallel: int,
        n_ubatch: Optional[int],
        n_max: Optional[int],
        target_rollback: bool,
        flat_fallback: bool,
        reprice: Optional[Callable[[int, int], int]] = None,
    ) -> str:
        """The MTP reserve line, naming only the dimensions that move the number.

        Which ones bite is per-model, so ``reprice`` (the estimator the reserve came
        from) is asked rather than re-deriving its branches; without it every
        dimension is named, as before. n_max is not one of its arguments, so that one
        stays structural. A None micro-batch is rendered as llama.cpp's default, since
        "ubatch None" names no parameter.
        """
        ubatch = self._DEFAULT_N_UBATCH if n_ubatch is None else n_ubatch

        def _moves(*candidates: tuple[int, int]) -> bool:
            if reprice is None:
                return True
            try:
                # The estimator's own value, not the total: a flat fallback reports 0.
                base = reprice(n_parallel, ubatch)
                for slots, ub in candidates:
                    if reprice(slots, ub) != base:
                        return True
            except Exception:
                # Cannot answer says nothing about the dependency: keep the name.
                return True
            return False

        # Every slot count DIVIDING the padded total reproduces it: at 12288 cells 2, 3
        # and 4 price the same, only 5 moves. The last candidate exceeds it, so cannot.
        _cells = _pad_kv_cells(max(1, n_ctx)) // 256
        slots_named = _moves(
            (n_parallel + 1, ubatch),
            (n_parallel + 2, ubatch),
            (max(n_parallel + 3, n_parallel * 2), ubatch),
            (max(n_parallel + 3, _cells + 1), ubatch),
        )
        # Same padding, via the compact-SWA window: a step of one bucket always crosses.
        ubatch_named = _moves(
            (n_parallel, ubatch * 2),
            (n_parallel, max(1, ubatch // 2)),
            (n_parallel, ubatch + 256),
        )
        scales_with_n_max = bool(
            target_rollback and n_max and self._rollback_state_bytes(n_parallel) > 0
        )
        return (
            f"MTP reserve: {reserve_bytes / (1024**3):.2f} GB "
            f"(draft KV @ {n_ctx}"
            + (f" x {n_parallel} slots" if slots_named else "")
            + (f", ubatch {ubatch}" if ubatch_named else "")
            + (f", n_max {n_max}" if scales_with_n_max else "")
            + (", flat-frac fallback" if flat_fallback else "")
            + "), "
        )

    _DEFAULT_N_UBATCH = _DEFAULT_LLAMA_N_UBATCH
    _COMPUTE_BUFFER_SAFETY = 1.10  # margin on the modelled activation width
    # Soft VRAM the modeled terms omit; charged to the fit budget on tight tiers (#6682).
    _CUDA_CONTEXT_RESERVE_BYTES = 320 * 1024 * 1024  # CUDA ctx + cuBLAS workspace (~330 MiB)
    _MMPROJ_VRAM_SAFETY = 1.4  # mmproj worst-case buffer vs file size (runtime ~1.3x)
    # Context the projector's residency is priced at. The placement loop shrinks the
    # context to this floor long before it would spill a layer, so this is the length
    # at which "does it fit" means "does every layer stay on the GPU".
    _MMPROJ_FIT_FLOOR_CTX = 8192
    _MTP_DRAFT_COMPUTE_BYTES = 224 * 1024 * 1024  # MTP draft decode graph beyond its KV
    # A DFlash/DSpark draft context re-reserves a flat block graph once the drafter initialises,
    # independent of n_ctx. Measured on llama.cpp b11642, draft graph + target verify graph:
    # Nemotron 3.5 Bolt DSpark (block 8) 528.5 + 44 MiB, LiLiCorr (block 8, SWA 1024) 307 + 60 MiB.
    # The _MTP_DRAFT_COMPUTE_BYTES floor is charged beside this, so the floor is the excess over it.
    _DFLASH_DRAFT_COMPUTE_BYTES = 349 * 1024 * 1024
    # Draft depth to budget when the extras own the spec block and the probe cannot
    # read the build's own default off its --help. Shipped values are 3
    # (common_params_speculative_draft::n_max) and 16; take the deeper, since the
    # Hybrid Mamba rollback copies scale by this and under-reserving them OOMs the
    # load while over-reserving only costs context. Zero cost on any other model,
    # where the recurrent state is nil.
    _UNKNOWN_SPEC_DRAFT_N_MAX = 16
    # Vocabulary-wide float32 copies per verified draft token (logits and sampling).
    _SPEC_VERIFY_ROW_COPIES = 3
    # Pipeline parallelism uses GGML_SCHED_MAX_COPIES input copies per device.
    # This applies only when _pipeline_parallel_disabled_by_args is false.
    _CTX_COMPUTE_SPLIT_MULT = 4
    # DeepSeek-V4 (deepseek4): its lightning indexer + sparse attention reserve a large
    # context-scaling compute buffer the rates above miss (present even with an f16
    # cache). Measured on UD-Q4_K_XL (ub=512): ~2 GiB at 16k -> ~65.5 GiB at 1M. Without
    # it auto-fit commits the full 1M train context, OOMs the reserve, and spills to CPU.
    _DSV4_CTX_COMPUTE_FLAT_BYTES = 2 * 1024**3  # ctx-independent indexer scratch
    _DSV4_CTX_COMPUTE_BYTES_PER_TOK = 72000  # per token at ub=512 (~72 GiB at 1M)

    # Inkling (inkling): with the reserve fix in the bundled llama.cpp (full-cache
    # reserve context reports the whole cache as position-contiguous, so the
    # worst-case graph is the banded flash path instead of the dense-bias
    # fallback), the ctx-linear compute term is just the banded KQ-mask cont:
    # measured on UD-IQ1_S (ub=512) 64K -> 1M total-VRAM slope of 50.6 KiB/tok,
    # of which 45,056 B/tok is KV -> ~5.6 KiB/tok compute. 8192 adds ~1.5x
    # headroom. (Pre-fix builds reserved the dense fallback at ~402 KiB/tok and
    # could not load large contexts at all.)
    _INKLING_CTX_COMPUTE_BYTES_PER_TOK = 8192  # per token at ub=512
    # Dense relative-bias fallback rate (quantized KV cache disables the banded
    # path): measured 402.5 GiB reserve at 1M ctx pre-fix, ~402 KiB per token.
    _INKLING_CTX_COMPUTE_DENSE_BYTES_PER_TOK = 402470  # per token at ub=512

    def _compute_activation_width(self) -> int:
        """Float32 activations the compute graph keeps live per micro-batch token.

        Take the widest dense, expert or recurrent block, then add per-layer
        embeddings. The 12 * n_embd floor covers attention and residual buffers.
        Return 0 without embedding dimensions.
        """

        # getattr: a backend built with __new__ carries only the dims it was given.
        def dim(name: str) -> int:
            return getattr(self, name, None) or 0

        n_embd = dim("_embedding_length")
        if n_embd <= 0:
            return 0

        widths = [12 * n_embd]
        if dim("_feed_forward_length") > 0:
            widths.append(4 * dim("_feed_forward_length"))
        n_used = dim("_expert_used_count")
        # Granite 4.0 H writes its expert width as feed_forward_length.
        n_ff_exp = dim("_expert_feed_forward_length") or (
            dim("_feed_forward_length") if dim("_n_experts") else 0
        )
        if n_used > 0 and n_ff_exp > 0:
            shared = dim("_expert_shared_feed_forward_length") or (
                n_ff_exp * dim("_expert_shared_count")
            )
            # Nemotron 3.5 routes experts through a moe_latent_size projection.
            n_embd_exp = dim("_moe_latent_size") or n_embd
            widths.append(n_used * (2 * n_embd_exp + 3 * n_ff_exp) + 3 * shared)
        inner = dim("_ssm_inner_size")
        if inner > 0:
            conv_in = inner + 2 * (dim("_ssm_group_count") or 1) * dim("_ssm_state_size")
            widths.append(4 * conv_in)
        per_layer_input = dim("_embedding_length_per_layer_input") * dim("_n_layers")
        return max(widths) + per_layer_input

    def _estimate_compute_buffer_bytes(
        self,
        *,
        n_ubatch: Optional[int] = None,
        n_parallel: int = 1,
        per_device_tensor: bool = False,
        vocab_ceiling: Optional[int] = None,
    ) -> int:
        """Per-device compute-graph buffer (bytes) that does not grow with context.

        Includes activations, output rows, sliding-window masks and recurrent state.
        Since b9460 (#23861), output rows are capped at min(ubatch, slots); older
        builds reserve ubatch rows. Extra verification rows belong to the MTP reserve.

        Tensor splits use the same buffer per device, so per_device_tensor does not
        change the result. Return 0 for missing dimensions; vocab_ceiling supplies
        a fallback vocabulary size."""
        width = self._compute_activation_width()
        n_vocab = self._vocab_size or vocab_ceiling or 0
        if width <= 0 or n_vocab <= 0:
            return 0
        ub = max(
            1,
            int(self._DEFAULT_N_UBATCH if n_ubatch is None else n_ubatch),
        )
        par = max(1, int(n_parallel))
        activations = width * ub * 4
        if self.is_embedding_gguf:
            outputs = (self._embedding_length or 0) * ub * 4
        elif getattr(self, "_reserves_micro_batch_outputs", False):
            # Before ggml-org/llama.cpp#23861: a row for every micro-batch token, once.
            # Measured on b9415 Vulkan: Gemma 4 E2B 515 MiB at ubatch 512, 137 on b10909.
            outputs = n_vocab * ub * 4
        else:
            outputs = n_vocab * min(ub, par) * 4
        swa_mask = 0
        sliding_window = self._effective_sliding_window()
        if sliding_window > 0:
            swa_mask = _pad_kv_cells(sliding_window * par + ub) * ub * 2
        # A recurrent block works on every slot's state at once: Qwen3.5 4B's buffer
        # grows 19.5 MiB from one slot to eight at the same micro-batch.
        inner = getattr(self, "_ssm_inner_size", None) or 0
        recurrent = 0
        if inner > 0:
            state = getattr(self, "_ssm_state_size", None) or 0
            conv_in = inner + 2 * (getattr(self, "_ssm_group_count", None) or 1) * state
            recurrent = par * (inner * state + 3 * conv_in) * 4
        return int((activations + outputs + recurrent) * self._COMPUTE_BUFFER_SAFETY + swa_mask)

    def _effective_sliding_window(self) -> int:
        """The sliding window when some layer really slides, else 0. Phi-4-mini
        declares a 262144 window over a 131072 context and no pattern, which bounds
        nothing."""
        window = getattr(self, "_sliding_window", None) or 0
        if window <= 0:
            return 0
        pattern = getattr(self, "_sliding_window_pattern", None)
        if pattern is not None:
            return window if any(pattern) else 0
        context = getattr(self, "_context_length", None) or 0
        return window if not context or window < context else 0

    def _attention_dequant_width(self) -> int:
        """K+V elements per cell copied to f16 from a quantized cache.

        Use the widest global layer: sliding-window layers do not scale with
        context. MLA uses compressed K only. Fall back to n_embd without head dims."""
        n_heads = getattr(self, "_n_heads", None)
        n_kv = getattr(self, "_n_kv_heads", None) or n_heads
        kv_lora_rank = getattr(self, "_kv_lora_rank", None)
        key_len = getattr(self, "_kv_key_length", None)
        val_len = getattr(self, "_kv_value_length", None)
        if kv_lora_rank is not None:
            rope_dim = getattr(self, "_key_length_mla", None) or 64
            return (getattr(self, "_n_kv_heads", None) or 1) * (key_len or kv_lora_rank + rope_dim)
        if key_len is None or val_len is None:
            if not (n_heads and n_kv):
                return getattr(self, "_embedding_length", None) or 0
            return 2 * n_kv * self._legacy_head_dim()
        by_layer = getattr(self, "_n_kv_heads_by_layer", None)
        pattern = (
            getattr(self, "_sliding_window_pattern", None)
            if getattr(self, "_sliding_window", None)
            else None
        )
        widest = 0
        for i, heads in enumerate(by_layer or ()):
            if not (pattern is not None and i < len(pattern) and pattern[i]):
                widest = max(widest, heads)
        return (widest or n_kv or 1) * (key_len + val_len)

    def _compute_buffer_ctx_bytes(
        self,
        n_ctx: int,
        n_ubatch: Optional[int] = None,
        cache_type_kv: Optional[str] = None,
        *,
        layer_split: bool = False,
        flash_attn: bool = True,
        n_parallel: int = 1,
    ) -> int:
        """Per-device context-dependent scratch, added to the flat compute buffer.

        Includes the KQ mask and quantized-cache dequantization. Without flash
        attention, also charge float32 scores and a float32 mask. Layer splits add
        scheduler copies, with sliding-window masks sized by n_parallel; tensor
        splits pass layer_split=False. DeepSeek and Inkling use separate rates.

        None cache type means f16; callers resolve environment overrides.
        Return 0 without n_embd or a positive context."""
        n_embd = self._embedding_length or 0
        if n_embd <= 0 or n_ctx <= 0:
            return 0
        ub = max(
            1,
            int(self._DEFAULT_N_UBATCH if n_ubatch is None else n_ubatch),
        )
        # One KQ mask copy ([n_kv, n_ubatch]), n_embd-independent: f16 under flash
        # attention, f32 without it (build_attn_inp_kq_mask). Every rate below is a
        # single-GPU total containing exactly one, so a split adds the extra copies.
        mask_rate = ub * (2 if flash_attn else 4)
        split_extra = (self._CTX_COMPUTE_SPLIT_MULT - 1) * mask_rate if layer_split else 0
        if getattr(self, "_architecture", None) == "deepseek4":
            # DSV4 indexer/CSA buffer (see constants): flat + linear, ub-scaled. Fires
            # for any KV type -- the indexer scratch is present even with an f16 cache.
            # No split_extra, unlike every other arch: 72000 already clears the measured
            # 65.5 KiB/tok by 4928 B/tok, more than the 4608 the mask copies cost.
            ub_scale = ub / self._DEFAULT_N_UBATCH
            return int(
                self._DSV4_CTX_COMPUTE_FLAT_BYTES
                + self._DSV4_CTX_COMPUTE_BYTES_PER_TOK * n_ctx * ub_scale
            )
        if getattr(self, "_architecture", None) == "inkling":
            ub_scale = ub / self._DEFAULT_N_UBATCH
            # The fused banded path requires an f32/f16/bf16 KV cache. A quantized
            # cache forces the dense relative-bias fallback, whose compute buffer is
            # [n_kv, ub, n_head] f32 (~402 KiB per context token at ub=512); size the
            # fit for that so a q8_0 cache gets a small honest context instead of an
            # unloadable one that crash-loops the server.
            if cache_type_kv and _kv_bytes_per_elem(cache_type_kv) < 2.0:
                rate = self._INKLING_CTX_COMPUTE_DENSE_BYTES_PER_TOK
            else:
                # Banded flash path (see constants): linear, ub-scaled.
                rate = self._INKLING_CTX_COMPUTE_BYTES_PER_TOK
            # Both paths allocate the mask, and 8192 has only ~1.5x headroom over the
            # 5.6 KiB/tok banded measurement -- a split alone would overrun it.
            return int(rate * n_ctx * ub_scale + split_extra * n_ctx)
        per_tok = mask_rate
        if _kv_bytes_per_elem(cache_type_kv) < 2.0:
            # Quantized cache: the f16 K/V copy, independent of the micro-batch.
            per_tok += self._attention_dequant_width() * 2
        if not flash_attn:
            n_head = getattr(self, "_n_heads", None) or max(1, n_embd // 64)
            per_tok += ub * n_head * 4
            if getattr(self, "_kv_lora_rank", None) is not None:
                # MLA decompresses a float32 copy of the latent cache to attend over it:
                # GLM-4.7-Flash grows 44 KiB per cell at ubatch 512, 2 KiB past the scores.
                per_tok += self._attention_dequant_width() * 4
        # Scheduler copies replicate masks and hidden states, but not dequant scratch.
        # Sliding-window copies use the same per-slot cell count as the flat buffer.
        split_fixed = 0
        if layer_split:
            split_fixed = self._CTX_COMPUTE_SPLIT_MULT * n_embd * ub * 4
            sliding_window = self._effective_sliding_window()
            if sliding_window > 0:
                swa_cells = _pad_kv_cells(sliding_window * max(1, int(n_parallel)) + ub)
                split_fixed += (self._CTX_COMPUTE_SPLIT_MULT - 1) * swa_cells * mask_rate
        return int((per_tok + split_extra) * n_ctx + split_fixed)

    def _slots_that_fit_on_gpu(
        self,
        n_parallel: int,
        effective_ctx: int,
        gpus: list[tuple[int, int]],
        total_by_idx: Optional[dict[int, int]],
        base_footprint_bytes: int,
        cache_type_kv: Optional[str],
        pin_fraction: float,
        per_device_overhead_bytes: int,
        min_gpus: int,
        n_ubatch: Optional[int] = None,
        swa_full: bool = False,
        kv_unified: bool = True,
        flash_attn: bool = True,
        split_extra_bytes: int = 0,
        split_extra_for_slots: Optional[Callable[[int], int]] = None,
        ubatch_for_slots: Optional[Callable[[int], Optional[int]]] = None,
        mtp_bytes_for_slots: Optional[Callable[[int, Optional[int]], int]] = None,
        ctx_checkpoints: int = 0,
        include_requested: bool = False,
        exact: bool = False,
    ) -> tuple[Optional[list[int]], bool, int]:
        """Largest serving-slot count that fits fully on GPU, so Unsloth keeps the model on
        GPU (-ngl -1) instead of --fit on, which offloads layers to host and collapses decode
        ~3x (oobabooga #6718). The search runs over [1, n_parallel) by default, or over
        [1, n_parallel] when ``include_requested`` is set. ``base_footprint_bytes`` is the
        slot-independent footprint (weights + soft overhead + context-linear compute,
        minus the folded compute buffer); each candidate re-adds the slot-sized compute buffer
        and KV, then re-selects GPUs like the explicit-context path -- ``split_extra_bytes``
        included, so a multi-GPU candidate is re-checked at the split rate. Returns
        (gpu_indices, use_fit=False, slots) for the largest fitting count, else (None, True,
        n_parallel). Unit-testable with synthetic VRAM maps.
        ``exact`` stops after the first candidate, for a caller that accepts only
        ``n_parallel`` itself and would re-price every lower count just to discard it.
        ``ctx_checkpoints`` mirrors --ctx-checkpoints: N host-RAM snapshots per slot,
        passed only when the GPUs share that pool (unified memory), else 0.
        ``ubatch_for_slots`` re-derives the micro-batch per candidate: the emitted
        --batch-size is raised to max(slots, 2), and llama.cpp caps the micro-batch
        against it, so a batch below the requested slot count shrinks as the candidates
        do. Holding it at the requested count made a fitting candidate look too big and
        dropped the load to --fit offload.
        ``mtp_bytes_for_slots(slots, ubatch)`` does the same for the MTP reserve, which is
        NOT slot-independent: compact SWA scales its window allowance by the slot count
        under kv_unified and adds one micro-batch, and an MLA target with recurrent (KDA)
        layers charges per slot. It takes the candidate micro-batch as well as the slot
        count, since a reduced candidate lowers the batch floor and so the ubatch too;
        pricing the reserve at the requested pair over-charged every candidate, so one that
        fits could be rejected and the load kept --fit. ``split_extra_for_slots(slots)``
        re-prices the split step, whose sliding-window masks also scale with slots."""
        for slots in range(n_parallel if include_requested else n_parallel - 1, 0, -1):
            _ub = ubatch_for_slots(slots) if ubatch_for_slots else n_ubatch
            cb = self._estimate_compute_buffer_bytes(
                n_ubatch = _ub, n_parallel = slots, per_device_tensor = False
            )
            if cb <= 0:
                cb = self._TENSOR_PARALLEL_BUFFER_RESERVE_MIB * 1024 * 1024
            total = (
                base_footprint_bytes
                + cb
                + (mtp_bytes_for_slots(slots, _ub) if mtp_bytes_for_slots else 0)
                + self._estimate_kv_cache_bytes(
                    effective_ctx,
                    cache_type_kv,
                    n_parallel = slots,
                    swa_full = swa_full,
                    kv_unified = kv_unified,
                    n_ubatch = _ub,
                    ctx_checkpoints = ctx_checkpoints,
                    flash_attn = flash_attn,
                )
            )
            gpu_indices, use_fit = self._select_gpus_split_aware(
                total,
                gpus,
                usable_fraction = pin_fraction,
                total_by_idx = total_by_idx,
                per_device_overhead_bytes = per_device_overhead_bytes,
                min_gpus = min_gpus,
                split_extra_bytes = (
                    split_extra_for_slots(slots) if split_extra_for_slots else split_extra_bytes
                ),
            )
            if not use_fit:
                return gpu_indices, False, slots
            if exact:
                break
        return None, True, n_parallel

    def _fit_context_to_vram(
        self,
        requested_ctx: int,
        available_mib: int,
        model_size_bytes: int,
        cache_type_kv: Optional[str] = None,
        min_ctx: int = 8192,
        *,
        swa_full: bool = False,
        n_parallel: int = 1,
        kv_unified: bool = True,
        n_ubatch: Optional[int] = None,
        ctx_checkpoints: int = 0,
        flash_attn: bool = True,
        kv_on_gpu: bool = True,
        mtp_engaged: bool = False,
        # True only for an absolute budget already summed across cards.
        pooled: bool = False,
        mtp_overhead_fn: Optional[Callable[[int], int]] = None,
        compute_ctx_bytes_fn: Optional[Callable[[int], int]] = None,
        budget_frac: Optional[float] = None,
        total_mib: Optional[int] = None,
    ) -> int:
        """Return the largest context length that fits in GPU VRAM.

        Budget caps occupancy at ``_CTX_FIT_VRAM_FRACTION`` of the card: an
        absolute ``free - (1 - frac) * total`` when ``total_mib`` is given, else
        ``free * frac``. Weights alone over budget returns ``requested_ctx``.

        ``kv_on_gpu`` mirrors ``--kv-offload`` (default on); when False the KV
        cache lives in CPU RAM and the requested context is honored verbatim.
        Other keyword args mirror ``_estimate_kv_cache_bytes``; pass ``ctx_checkpoints``
        only against a unified-memory budget, since the snapshots live in host RAM.

        ``mtp_engaged`` reserves extra VRAM for the MTP draft model's KV cache +
        compute buffers, else tight tiers (e.g. 32 GB) spill to a slower path.
        """
        if not self._can_estimate_kv():
            logger.debug(
                "Skipping context fit because KV cache metadata is unavailable",
                requested_ctx = requested_ctx,
                available_mib = available_mib,
            )
            return requested_ctx

        # KV lives off-GPU: no VRAM accounting needed for the cache itself.
        if not kv_on_gpu:
            return requested_ctx

        kv_kwargs = dict(
            swa_full = swa_full,
            n_parallel = n_parallel,
            kv_unified = kv_unified,
            n_ubatch = n_ubatch,
            ctx_checkpoints = ctx_checkpoints,
            flash_attn = flash_attn,
        )

        # byte-accurate mtp_overhead_fn supersedes the flat fraction (the fallback
        # when dims can't size the draft KV); callers may override budget_frac.
        if budget_frac is None:
            flat_mtp = mtp_engaged and mtp_overhead_fn is None
            budget_frac = _active_vram_fraction() - (_MTP_VRAM_RESERVE_FRAC if flat_mtp else 0.0)
        # Absolute reserve off total when known, else fraction-of-free; clamp >=0.
        budget_mib = _vram_usable_mib(
            available_mib,
            total_mib or 0,
            budget_frac,
            pooled = pooled,
            sysmem_fallback = LlamaCppBackend._sysmem_fallback_risk(),
        )
        if total_mib is not None and total_mib > 0:
            budget_mib = max(0.0, budget_mib)
        budget_bytes = budget_mib * 1024 * 1024
        model_footprint = model_size_bytes

        def _mtp_at(ctx: int) -> int:
            return mtp_overhead_fn(ctx) if mtp_overhead_fn is not None else 0

        def _cc_at(ctx: int) -> int:
            # Context-linear compute-buffer growth (flash-attn KQ mask + scratch);
            # the flat term in model_footprint only covers ctx -> 0.
            return compute_ctx_bytes_fn(ctx) if compute_ctx_bytes_fn is not None else 0

        # Already fits?
        kv = self._estimate_kv_cache_bytes(requested_ctx, cache_type_kv, **kv_kwargs)
        if model_footprint + kv + _mtp_at(requested_ctx) + _cc_at(requested_ctx) <= budget_bytes:
            return requested_ctx

        # Weights + compute buffer alone exceed budget -- reducing ctx can't help.
        if model_footprint >= budget_bytes:
            logger.debug(
                "Model footprint exceeds GPU budget before KV cache",
                requested_ctx = requested_ctx,
                available_mib = available_mib,
                model_size_gb = round(model_footprint / (1024**3), 2),
            )
            return requested_ctx

        # Binary search for max context that fits (KV + MTP draft reserve at that ctx)
        remaining = budget_bytes - model_footprint
        effective_min = min(min_ctx, requested_ctx)
        lo, hi = effective_min, requested_ctx
        best = effective_min
        while lo <= hi:
            mid = (lo + hi) // 2
            kv = self._estimate_kv_cache_bytes(mid, cache_type_kv, **kv_kwargs)
            if kv + _mtp_at(mid) + _cc_at(mid) <= remaining:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        # Round down to nearest 256 for alignment, never above requested_ctx
        best = (best // 256) * 256
        best = max(effective_min, best)
        best = min(best, requested_ctx)
        return best
