// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PerModelConfig } from "@/features/model-picker";
import { normalizeMlxKvQuant } from "@/features/model-picker/model-config/per-model-config";

import {
  parseGpuLayersOverride,
  resolveTensorParallel,
  stripManagedOffloadFlags,
} from "./llama-extra-args-normalize";
import type { GpuIndexKind } from "@/hooks/gpu-selection";

import type { InferenceStatusResponse } from "../types/api";

type ResidentRuntime = Pick<
  InferenceStatusResponse,
  | "engine"
  | "engine_parallelism"
  | "engine_precision"
  | "context_length"
  | "requested_context_length"
  | "cache_type_kv"
  | "mlx_kv_quant_requested"
  | "mlx_int8_prefill_requested"
  | "speculative_type"
  | "spec_draft_n_max"
  | "requested_parallel_slots"
  | "requested_n_batch"
  | "requested_n_ubatch"
  | "requested_load_mode"
  | "requested_spec_draft_cache_type"
  | "requested_ctx_checkpoints"
  | "reasoning_budget"
  | "reasoning_budget_message"
  | "requested_reasoning_budget"
  | "requested_reasoning_budget_message"
  | "requested_cache_ram"
  | "tensor_parallel"
  | "disable_vision"
  | "chat_template_override"
  | "requested_llama_extra_args"
  | "gpu_memory_mode"
  | "gpu_layers"
  | "n_cpu_moe"
  | "requested_gpu_ids"
  | "gpu_ids"
  | "is_gguf"
  | "is_diffusion"
  | "diffusion_requested_ngl"
  | "diffusion_split_supported"
  | "tensor_parallel_dropped_by_arch_gate"
  | "gpu_placement_paravirtual"
  | "tensor_split"
  | "cpu_fallback_reason"
  | "spec_fallback_reason"
>;

function sameList(
  left: readonly (string | number)[] | null | undefined,
  right: readonly (string | number)[] | null | undefined,
): boolean {
  const a = left ?? [];
  const b = right ?? [];
  return a.length === b.length && a.every((item, index) => item === b[index]);
}

/** Order matters: position decides which card is used first. */
function sameGpuPlacement(
  left: readonly number[] | null | undefined,
  right: readonly number[] | null | undefined,
): boolean {
  return sameList(left ?? [], right ?? []);
}

/** Unset values the applier fills in; required so callers cannot forget them. */
export type StandingConfigDefaults = {
  speculativeType: string | null;
  gpuMemoryMode: "auto" | "manual";
  gpuLayers: number;
  nCpuMoe: number;
  /** Compare the reconciled pick, not raw saved ids (physical vs Vulkan numbering). */
  reconcileGpuIds: (
    ids: number[] | null,
    savedIndexKind: GpuIndexKind | null | undefined,
  ) => number[] | null;
  defaultEngineGpuIds?: number[];
  /** Unset length is not 0: a GGUF re-pick resolves to the resident context. */
  resolveContextLength: (customContextLength: number | null) => number;
  /** Server default for an unset `--parallel`; null when unreadable. */
  parallelSlots: number | null;
  /** The applier clears splitRatio, so any remembered config asks for the default. */
  splitRatio: number[] | null;
  /** Passed in rather than imported to keep this module a React-free leaf. */
  normalizeSpeculative: (value: string | null | undefined) => string | null;
};

type SettingCheck = {
  placement?: true;
  mlxComparable?: true;
  ggufPlacement?: true;
  diffusionOnly?: true;
  /** Chat-only settings; status nulls them for diffusion runtimes. */
  chatOnly?: true;
  pinned: (config: PerModelConfig) => boolean;
  agrees: (
    config: PerModelConfig,
    status: ResidentRuntime,
    standing: StandingConfigDefaults,
  ) => boolean;
};

/** Fallbacks the backend retries on an identical load (see `_runtime_matches_intent`). */
const RETRYABLE_SPEC_FALLBACKS = new Set([
  "drafter_not_found",
  "drafter_unloadable",
  "binary_no_mtp",
  "binary_outdated",
]);

const BINARY_SPEC_FALLBACKS = new Set(["binary_no_mtp", "binary_outdated"]);

const SPECULATIVE_MODES = new Set([
  "auto",
  "mtp",
  "mtp+ngram",
  "dspark",
  "dflash",
]);

/** Whether an identical `/load` would repair degraded speculative decoding. Coarser than backend. */
export function residentSpeculativeNeedsRepair(
  status: Pick<
    InferenceStatusResponse,
    | "spec_fallback_reason"
    | "spec_fallback_binary_changed"
    | "spec_probe_retry_pending"
    | "spec_dflash_retry_pending"
    | "spec_dspark_sidecar_absent"
    | "spec_drafter_kind"
  >,
  resolvedSpeculativeType: string | null,
  sendsGgufPath = false,
): boolean {
  const mode = resolvedSpeculativeType ?? "auto";
  // These arms record no fallback reason, so the reason check below cannot see them.
  if (status.spec_probe_retry_pending === true) {
    return true;
  }
  if (
    status.spec_dflash_retry_pending === true &&
    (mode === "auto" || mode === "dflash")
  ) {
    return true;
  }
  // ngram-mod runs no drafter, but a binary_outdated stand-down is repaired by an update.
  const modeCanRepair =
    SPECULATIVE_MODES.has(mode) ||
    (mode === "ngram" && status.spec_fallback_reason === "binary_outdated");
  if (
    !RETRYABLE_SPEC_FALLBACKS.has(status.spec_fallback_reason ?? "") ||
    !modeCanRepair
  ) {
    return false;
  }
  // DFlash and absent DSpark sidecars are not transient, so they are excluded.
  if (status.spec_fallback_reason === "drafter_not_found") {
    // Backend arm requires `intent.gguf_path is None`, so a standalone file dedupes.
    if (sendsGgufPath) {
      return false;
    }
    if (status.spec_drafter_kind === "dflash") {
      return false;
    }
    if (
      status.spec_drafter_kind === "dspark" &&
      status.spec_dspark_sidecar_absent === true
    ) {
      return false;
    }
  }
  // Only an explicit false settles it; older backends keep the coarser answer.
  return !(
    BINARY_SPEC_FALLBACKS.has(status.spec_fallback_reason ?? "") &&
    status.spec_fallback_binary_changed === false
  );
}

const requestedGpuMemoryMode = (
  config: PerModelConfig,
  standing: StandingConfigDefaults,
): "auto" | "manual" => config.gpuMemoryMode ?? standing.gpuMemoryMode;

const cleanTemplate = (value: string | null | undefined): string | null =>
  value?.trim() ? value : null;

const SETTING_CHECKS: SettingCheck[] = [
  {
    // Resolved, not raw: unset is Auto, sent as 0 cross-model or resident ctx on a re-pick.
    pinned: () => true,
    agrees: (c, s, standing) =>
      standing.resolveContextLength(c.customContextLength ?? null) ===
      (s.requested_context_length ?? 0),
  },
  {
    pinned: () => true,
    agrees: (c, s) => (c.kvCacheDtype ?? null) === (s.cache_type_kv ?? null),
  },
  {
    mlxComparable: true,
    pinned: () => true,
    agrees: (c, s) =>
      (c.mlxKvQuant ?? null) === normalizeMlxKvQuant(s.mlx_kv_quant_requested),
  },
  {
    mlxComparable: true,
    pinned: () => true,
    agrees: (c, s) =>
      Boolean(c.mlxInt8Prefill) === (s.mlx_int8_prefill_requested === true),
  },
  {
    // Always pinned: unset resolves to the standing preference.
    pinned: () => true,
    agrees: (c, s, standing) =>
      (standing.normalizeSpeculative(c.speculativeType) ??
        standing.speculativeType) ===
      (standing.normalizeSpeculative(s.speculative_type) ??
        standing.speculativeType),
  },
  {
    // Pinned: the backend reloads on a null-vs-explicit flip.
    pinned: () => true,
    agrees: (c, s) =>
      // After MTP-free recovery null means unknown, not default; let /load decide.
      s.spec_fallback_reason !== "runtime_error" &&
      (c.specDraftNMax ?? null) === (s.spec_draft_n_max ?? null),
  },
  {
    chatOnly: true,
    mlxComparable: true,
    pinned: () => true,
    agrees: (c, s, standing) =>
      (c.nParallel ?? standing.parallelSlots) ===
      (s.requested_parallel_slots ?? standing.parallelSlots),
  },
  {
    chatOnly: true,
    pinned: () => true,
    agrees: (c, s) => (c.nBatch ?? null) === (s.requested_n_batch ?? null),
  },
  {
    chatOnly: true,
    pinned: () => true,
    agrees: (c, s) => (c.nUbatch ?? null) === (s.requested_n_ubatch ?? null),
  },
  {
    chatOnly: true,
    pinned: () => true,
    agrees: (c, s) => (c.loadMode ?? null) === (s.requested_load_mode ?? null),
  },
  {
    chatOnly: true,
    // Always pinned so clearing back to the f16 default still relaunches.
    pinned: () => true,
    agrees: (c, s) =>
      (c.specDraftCacheDtype ?? null) ===
      (s.requested_spec_draft_cache_type ?? null),
  },
  {
    chatOnly: true,
    pinned: () => true,
    agrees: (c, s) =>
      (c.ctxCheckpoints ?? null) === (s.requested_ctx_checkpoints ?? null),
  },
  {
    chatOnly: true,
    pinned: () => true,
    agrees: (c, s) => (c.cacheRam ?? null) === (s.requested_cache_ram ?? null),
  },
  {
    // Status echoes the effective budget, so env-shaped servers read as a reload (safe side).
    chatOnly: true,
    pinned: () => true,
    agrees: (c, s) =>
      (c.reasoningBudget ?? -1) ===
      (s.requested_reasoning_budget ?? s.reasoning_budget ?? -1),
  },
  {
    chatOnly: true,
    pinned: () => true,
    agrees: (c, s) =>
      (c.reasoningBudgetMessage ?? "") ===
      (s.requested_reasoning_budget_message ??
        s.reasoning_budget_message ??
        ""),
  },
  {
    pinned: () => true,
    agrees: (c, s) =>
      resolveTensorParallel(c.llamaExtraArgs, c.tensorParallel) ===
        (s.tensor_parallel ?? false) ||
      s.gpu_placement_paravirtual === true ||
      // A split the architecture gate normalized away still matches.
      (resolveTensorParallel(c.llamaExtraArgs, c.tensorParallel) &&
        s.tensor_parallel !== true &&
        s.tensor_parallel_dropped_by_arch_gate === true),
  },
  {
    // Backend reloads on mismatch, so adopting would keep projector VRAM spent.
    chatOnly: true,
    pinned: () => true,
    agrees: (c, s) => c.disableVision === (s.disable_vision ?? false),
  },
  {
    mlxComparable: true,
    pinned: () => true,
    agrees: (c, s) =>
      cleanTemplate(c.chatTemplateOverride) ===
      cleanTemplate(s.chat_template_override),
  },
  {
    // undefined: never read. null: cleared, agrees only with no pass-through args.
    chatOnly: true,
    pinned: (c) => c.llamaExtraArgs !== undefined,
    agrees: (c, s, standing) =>
      sameList(
        requestedGpuMemoryMode(c, standing) === "manual"
          ? stripManagedOffloadFlags(c.llamaExtraArgs)
          : c.llamaExtraArgs,
        s.requested_llama_extra_args,
      ),
  },
  {
    placement: true,
    ggufPlacement: true,
    pinned: () => true,
    agrees: (c, s, standing) =>
      (c.gpuMemoryMode ?? standing.gpuMemoryMode) ===
      (s.gpu_memory_mode ?? standing.gpuMemoryMode),
  },
  {
    // Only under Manual: under Auto the fitter chooses offload.
    placement: true,
    ggufPlacement: true,
    pinned: () => true,
    agrees: (c, s, standing) =>
      requestedGpuMemoryMode(c, standing) !== "manual" ||
      requestedGpuLayers(c, standing) === (s.gpu_layers ?? standing.gpuLayers),
  },
  {
    // Same guard as the backend: hidden nCpuMoe under Auto layers must be ignored.
    placement: true,
    ggufPlacement: true,
    pinned: () => true,
    agrees: (c, s, standing) =>
      requestedGpuMemoryMode(c, standing) !== "manual" ||
      requestedGpuLayers(c, standing) < 0 ||
      (c.nCpuMoe ?? standing.nCpuMoe) === (s.n_cpu_moe ?? standing.nCpuMoe),
  },
  {
    // Unset sends Automatic, so it does not adopt a server pinned to a chosen GPU.
    placement: true,
    pinned: () => true,
    agrees: (c, s, standing) => {
      const reconciled = standing.reconcileGpuIds(
        c.selectedGpuIds ?? null,
        c.selectedGpuIndexKind,
      );
      // The diffusion runner uses one device, its lowest id.
      const pick =
        s.is_diffusion === true && reconciled?.length
          ? [Math.min(...reconciled)]
          : reconciled;
      if (sameGpuPlacement(pick, s.requested_gpu_ids)) {
        return true;
      }
      // Fitting may narrow to a subset; an absent echo is no placement, not Automatic.
      return Boolean(s.gpu_ids?.length) && sameGpuPlacement(pick, s.gpu_ids);
    },
  },
  {
    // An auto-mode resident reports the planner's own split, so it agrees unless the store holds a
    // ratio (kept across a pending Manual-to-Auto edit, and the load sends it).
    placement: true,
    ggufPlacement: true,
    pinned: () => true,
    agrees: (_c, s, standing) =>
      (s.gpu_memory_mode === "auto" && standing.splitRatio == null) ||
      sameList(standing.splitRatio, s.tensor_split),
  },
  {
    // A managed override the backend would reject must not be folded into no override.
    pinned: (c) => c.llamaExtraArgs !== undefined,
    agrees: (c, _s, standing) =>
      requestedGpuMemoryMode(c, standing) !== "manual" ||
      parseGpuLayersOverride(c.llamaExtraArgs).kind !== "invalid",
  },
  {
    // Diffusion compares _diffusion_manual_ngl in place of the placement fields.
    diffusionOnly: true,
    pinned: () => true,
    agrees: (c, s, standing) => {
      const ngl = diffusionManualNgl(c, standing);
      if (ngl !== (s.diffusion_requested_ngl ?? null)) {
        return false;
      }
      return !(
        ngl !== null &&
        s.gpu_layers !== ngl &&
        s.diffusion_split_supported === true
      );
    },
  },
];

/** Under manual a pass-through `-ngl` is copied into the field before comparison. */
function requestedGpuLayers(
  config: PerModelConfig,
  standing: StandingConfigDefaults,
): number {
  const override =
    requestedGpuMemoryMode(config, standing) === "manual"
      ? parseGpuLayersOverride(config.llamaExtraArgs)
      : { kind: "absent" as const };
  return override.kind === "value"
    ? override.layers
    : (config.gpuLayers ?? standing.gpuLayers);
}

function diffusionManualNgl(
  config: PerModelConfig,
  standing: StandingConfigDefaults,
): number | null {
  const layers = requestedGpuLayers(config, standing);
  return requestedGpuMemoryMode(config, standing) === "manual" && layers >= 0
    ? layers
    : null;
}

/** Mirrors `_cpu_fallback_request_eligible` minus its environment terms. */
function cpuFallbackPlacementPreserved(
  config: PerModelConfig,
  status: ResidentRuntime,
  standing: StandingConfigDefaults,
): boolean {
  if (status.cpu_fallback_reason !== "vulkan_startup_crash") {
    return false;
  }
  const mode = config.gpuMemoryMode ?? standing.gpuMemoryMode;
  const layers = config.gpuLayers ?? standing.gpuLayers;
  return (
    (mode === "auto" || (mode === "manual" && layers === 0)) &&
    !standing.reconcileGpuIds(
      config.selectedGpuIds ?? null,
      config.selectedGpuIndexKind,
    )?.length &&
    !config.tensorParallel &&
    !standing.splitRatio?.length &&
    (config.nCpuMoe ?? standing.nCpuMoe) === 0 &&
    !config.llamaExtraArgs?.length
  );
}

/** Whether the resident load already runs this pick's settings. Biased to report differences.
 *  maxSeqLength is client-side and not compared. */
export function residentRuntimeMatchesConfig(
  status: ResidentRuntime,
  config: PerModelConfig | null | undefined,
  standing: StandingConfigDefaults,
): boolean {
  // No config means the load reads the live runtime, which was hydrated from the resident.
  if (!config) {
    return true;
  }
  if ((status.engine ?? "auto") !== (config.engine ?? "auto")) return false;
  if (status.engine === "vllm" || status.engine === "sglang") {
    if (
      (status.engine_precision ?? "auto") !== (config.enginePrecision ?? "auto") ||
      (status.engine_parallelism ?? "tensor") !== (config.engineParallelism ?? "tensor")
    )
      return false;
    if (
      config.maxSeqLength != null &&
      config.maxSeqLength > 0 &&
      config.maxSeqLength !==
        (status.requested_context_length ?? status.context_length)
    ) {
      return false;
    }
    const requested = standing.reconcileGpuIds(
      config.selectedGpuIds ?? null,
      config.selectedGpuIndexKind,
    ) ?? standing.defaultEngineGpuIds ?? [0];
    if (
      !sameGpuPlacement(
        requested,
        status.requested_gpu_ids ?? status.gpu_ids ?? [0],
      )
    ) {
      return false;
    }
  }
  const placementPreserved =
    // Paravirtual Metal pins every GGUF to CPU, so placement cannot differ.
    status.gpu_placement_paravirtual === true ||
    cpuFallbackPlacementPreserved(config, status, standing);
  // Diffusion takes no --parallel, batch sizes or pass-through args.
  const diffusion = status.is_diffusion === true;
  return SETTING_CHECKS.every(
    (check) =>
      (status.is_gguf === false && !check.mlxComparable) ||
      (diffusion && check.chatOnly) ||
      (diffusion && check.ggufPlacement) ||
      (!diffusion && check.diffusionOnly) ||
      (check.placement && placementPreserved) ||
      !check.pinned(config) ||
      check.agrees(config, status, standing),
  );
}

