// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PerModelConfig } from "@/features/model-picker";
import type { GpuIndexKind } from "@/hooks/gpu-selection";

import type { InferenceStatusResponse } from "../types/api";

/** The resident load's own invocation, as `/api/inference/status` echoes it. */
type ResidentRuntime = Pick<
  InferenceStatusResponse,
  | "requested_context_length"
  | "cache_type_kv"
  | "mlx_kv_bits_requested"
  | "speculative_type"
  | "spec_draft_n_max"
  | "requested_parallel_slots"
  | "requested_n_batch"
  | "requested_n_ubatch"
  | "tensor_parallel"
  | "chat_template_override"
  | "requested_llama_extra_args"
  | "gpu_memory_mode"
  | "gpu_layers"
  | "n_cpu_moe"
  | "requested_gpu_ids"
  | "tensor_split"
  | "cpu_fallback_reason"
>;

function sameList(
  left: readonly (string | number)[] | null | undefined,
  right: readonly (string | number)[] | null | undefined,
): boolean {
  const a = left ?? [];
  const b = right ?? [];
  return a.length === b.length && a.every((item, index) => item === b[index]);
}

/** Placement is a set, not an order: the backend narrows and reorders it at fit time. */
function sameGpuSet(
  left: readonly number[] | null | undefined,
  right: readonly number[] | null | undefined,
): boolean {
  const sort = (ids: readonly number[]) => [...ids].sort((a, b) => a - b);
  return sameList(sort(left ?? []), sort(right ?? []));
}

/**
 * What a config field resolves to when left unset.
 *
 * These four are not per-model, so omitting them is not silence: the applier fills them
 * from a standing preference or a constant (`applyPerModelConfigToRuntime`), and the load
 * sends that. The caller supplies them rather than this module reading the store, so the
 * comparison stays a pure function of its inputs. Required, not optional: a resolver a
 * caller can forget is one that silently keeps a runtime the user did not ask for.
 */
export type StandingConfigDefaults = {
  /** `readPersistedSpeculativeType()`, already normalized. */
  speculativeType: string | null;
  /** `readPersistedGpuMemoryMode()`. */
  gpuMemoryMode: "auto" | "manual";
  /** `GPU_LAYERS_AUTO`. */
  gpuLayers: number;
  /** The applier's own constant, 0. */
  nCpuMoe: number;
  /**
   * `reconcilePersistedGpuIds`, with the device cache already warm and the resident model's
   * diffusion flag bound. `performLoad` sends the reconciled pick, not the saved one: a
   * selection saved in another index namespace, or naming GPUs that are gone, becomes
   * Automatic before `/load`. Comparing the raw ids would let a saved physical `[1]` adopt a
   * server pinned to Vulkan device 1.
   */
  reconcileGpuIds: (
    ids: number[] | null,
    savedIndexKind: GpuIndexKind | null | undefined,
  ) => number[] | null;
  /**
   * `resolveLoadMaxSeqLength` bound to the inputs `performLoad` gives it, so the comparison
   * is against the n_ctx the load would send. An unset length is not simply 0: for a GGUF
   * re-pick it resolves to the resident context, and only otherwise to 0, which is Auto.
   */
  resolveContextLength: (customContextLength: number | null) => number;
  /**
   * `splitRatio` as the store holds it now, which is what the load sends. Never a config
   * field: `applyPerModelConfigToRuntime` clears it, so applying any remembered config
   * asks for the default distribution rather than the resident custom one.
   */
  splitRatio: number[] | null;
  /**
   * `normalizeSpeculativeType`, passed rather than imported. It lives on the chat runtime
   * store, which reaches React, and this module is deliberately a leaf so the node suite
   * can drive it; copying its mapping here (comma-chained legacy echoes and all) would be
   * a second copy free to drift from the one the load actually uses.
   */
  normalizeSpeculative: (value: string | null | undefined) => string | null;
};

/**
 * One setting the resident load can disagree about. `pinned` is whether the config has an
 * opinion at all, `agrees` whether the running server satisfies it. The pair survives for
 * `llamaExtraArgs`, the only field the applier leaves unresolved.
 */
type SettingCheck = {
  /** Placement, which the backend rewrites wholesale on a preserved CPU fallback. */
  placement?: true;
  pinned: (config: PerModelConfig) => boolean;
  agrees: (
    config: PerModelConfig,
    status: ResidentRuntime,
    standing: StandingConfigDefaults,
  ) => boolean;
};

/**
 * Fallback reasons the backend retries on an IDENTICAL next load, taken from the arms of
 * `LlamaCppBackend._runtime_matches_intent` that return False to force the repair: a drafter
 * fetch that can be attempted again, and the stand-down the UI asks the user to fix by
 * updating llama.cpp. The rest are not here on purpose. "drafter_no_vram" and
 * "mla_mtp_disabled" are Auto-mode policy, and "runtime_error" only reopens when the draft
 * count changes, which the settings comparison already sees; treating them as repairable
 * would prompt to stop running chats on every re-pick and repair nothing.
 */
const RETRYABLE_SPEC_FALLBACKS = new Set([
  "drafter_not_found",
  "binary_no_mtp",
  "binary_outdated",
]);

/** The modes the backend's retry arms are guarded on; anything else asked for no drafter. */
const SPECULATIVE_MODES = new Set([
  "auto",
  "mtp",
  "mtp+ngram",
  "dspark",
  "dflash",
]);

/**
 * Whether the resident load's speculative decoding is degraded in a way the next identical
 * `/load` would repair.
 *
 * This is the one place where sending a request the runtime already satisfies is not a
 * no-op, so it is the one reason to decline the shortcut on settings that match. The
 * decision stays coarser than the backend's: the status echoes the fallback reason but not
 * `_dflash_retry_needed` or an inconclusive capability probe, so a load that would dedupe
 * on the far side can still be sent. That costs one round trip through `already_loaded`,
 * which returns before any teardown.
 */
export function residentSpeculativeNeedsRepair(
  status: Pick<InferenceStatusResponse, "spec_fallback_reason">,
  resolvedSpeculativeType: string | null,
): boolean {
  return (
    RETRYABLE_SPEC_FALLBACKS.has(status.spec_fallback_reason ?? "") &&
    SPECULATIVE_MODES.has(resolvedSpeculativeType ?? "auto")
  );
}

const cleanTemplate = (value: string | null | undefined): string | null =>
  value?.trim() ? value : null;

/**
 * Mirrors the fields `_runtime_matches_intent` reloads for, plus the MLX pair
 * `_mlx_runtime_settings_match` compares.
 *
 * Nearly every check is unconditionally pinned. A config reaches here only after
 * `applyModelLoadConfigToRuntime` wrote it over the runtime store (`chat-page.tsx:3242`,
 * `hub-page.tsx:1329`), and that applier resolves each of these with `?? null`, so the
 * snapshot `performLoad` takes reads null rather than inheriting the resident value: an
 * unset field asks for the default, not for whatever is running. Only `llamaExtraArgs` is
 * genuinely optional, being the one field with no `?? null` fallback.
 */
const SETTING_CHECKS: SettingCheck[] = [
  {
    // Resolved, not compared raw: an unset length is Auto, which the load sends as 0 for a
    // cross-model GGUF pick and as the resident context when re-picking the same one.
    // Reading null as "no opinion" against a status echoing either number was a reload.
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
    pinned: () => true,
    agrees: (c, s) =>
      (c.mlxKvBits ?? null) === (s.mlx_kv_bits_requested ?? null),
  },
  {
    // Always pinned: an unset mode resolves to the standing preference, and the load sends
    // it. Reading it as silence let a pick asking for "off" adopt a resident MTP runtime.
    pinned: () => true,
    agrees: (c, s, standing) =>
      (standing.normalizeSpeculative(c.speculativeType) ??
        standing.speculativeType) ===
      (standing.normalizeSpeculative(s.speculative_type) ??
        standing.speculativeType),
  },
  {
    pinned: () => true,
    agrees: (c, s) =>
      (c.specDraftNMax ?? null) === (s.spec_draft_n_max ?? null),
  },
  {
    pinned: () => true,
    agrees: (c, s) =>
      (c.nParallel ?? null) === (s.requested_parallel_slots ?? null),
  },
  {
    pinned: () => true,
    agrees: (c, s) => (c.nBatch ?? null) === (s.requested_n_batch ?? null),
  },
  {
    pinned: () => true,
    agrees: (c, s) => (c.nUbatch ?? null) === (s.requested_n_ubatch ?? null),
  },
  {
    // Not nullable, so it always has an opinion; a status omitting it ran without.
    pinned: () => true,
    agrees: (c, s) => c.tensorParallel === (s.tensor_parallel ?? false),
  },
  {
    // Blank-trimmed on both ends: the applier and the load both send "" as null.
    pinned: () => true,
    agrees: (c, s) =>
      cleanTemplate(c.chatTemplateOverride) ===
      cleanTemplate(s.chat_template_override),
  },
  {
    // undefined: never read the stored value. null: the user cleared the box, which agrees
    // only with a load invoked with no pass-through args.
    pinned: (c) => c.llamaExtraArgs !== undefined,
    agrees: (c, s) => sameList(c.llamaExtraArgs, s.requested_llama_extra_args),
  },
  {
    // Standing preference, like the speculative mode above.
    placement: true,
    pinned: () => true,
    agrees: (c, s, standing) =>
      (c.gpuMemoryMode ?? standing.gpuMemoryMode) ===
      (s.gpu_memory_mode ?? standing.gpuMemoryMode),
  },
  {
    // Resolves to GPU_LAYERS_AUTO rather than to a preference, but the load still sends it.
    placement: true,
    pinned: () => true,
    agrees: (c, s, standing) =>
      (c.gpuLayers ?? standing.gpuLayers) ===
      (s.gpu_layers ?? standing.gpuLayers),
  },
  {
    placement: true,
    pinned: () => true,
    agrees: (c, s, standing) =>
      (c.nCpuMoe ?? standing.nCpuMoe) === (s.n_cpu_moe ?? standing.nCpuMoe),
  },
  {
    // Absent resolves to a null selection in the applier, and a null selection is sent as
    // Automatic, so this is pinned like the rest: an unset pick does not adopt a server
    // that was placed on a chosen GPU. Reconciled first, since that is what /load sends.
    placement: true,
    pinned: () => true,
    agrees: (c, s, standing) =>
      sameGpuSet(
        standing.reconcileGpuIds(
          c.selectedGpuIds ?? null,
          c.selectedGpuIndexKind,
        ),
        s.requested_gpu_ids,
      ),
  },
  {
    // The split is placement the config cannot carry: the applier clears splitRatio, so a
    // remembered config asks for the default distribution while a resident manual load may
    // be running a custom one. Omitting it kept that custom split with nothing saying so.
    placement: true,
    pinned: () => true,
    agrees: (_c, s, standing) => sameList(standing.splitRatio, s.tensor_split),
  },
];

/**
 * Whether the backend would rewrite this request into the resident CPU fallback.
 *
 * `adopt_load_intent_if_matched` runs `_preserve_cpu_fallback_intent` first, so after a
 * Vulkan startup crash an Auto request becomes the resident manual/zero-layer intent and
 * dedupes. Comparing placement literally rejected it and raised the prompt this PR removes.
 *
 * Mirrors `_cpu_fallback_request_eligible`, minus its environment terms: the resident
 * server already fell back under this same env, and a request carrying its own placement
 * args is excluded here rather than guessed at.
 */
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

/**
 * Whether the resident load already runs the settings this pick would ask for.
 *
 * Identity is not the whole of a load: `LlamaCppBackend` reuses a running server only when
 * the request also agrees on context, KV dtype, slots, batch sizes, placement, speculative
 * mode, chat template and pass-through args. An unset field is still an opinion, since the
 * applier resolves it before the load reads it: unset asks for the default, not for whatever
 * is running. What keeps the common case working is the other door: a model the user never
 * configured arrives with no config at all, and a model they did configure carries what
 * `currentRuntimePerModelConfig` wrote from the load that made it resident.
 *
 * The bias is one-sided on purpose. "Differs" costs one reload, which is what happened
 * before any of this existed; a wrong "matches" leaves the user on settings they did not
 * ask for, with the panel rolled back to the resident model so nothing says so.
 *
 * `maxSeqLength` is not compared: a client-side generation cap no status echoes, and it
 * never reaches llama-server's invocation.
 */
export function residentRuntimeMatchesConfig(
  status: ResidentRuntime,
  config: PerModelConfig | null | undefined,
  standing: StandingConfigDefaults,
): boolean {
  // No config at all is not the same as a config that pins nothing: with none, the load
  // path reads the live runtime, which was hydrated from the resident model, so there is
  // nothing that could differ.
  if (!config) {
    return true;
  }
  const placementPreserved = cpuFallbackPlacementPreserved(
    config,
    status,
    standing,
  );
  return SETTING_CHECKS.every(
    (check) =>
      (check.placement && placementPreserved) ||
      !check.pinned(config) ||
      check.agrees(config, status, standing),
  );
}
