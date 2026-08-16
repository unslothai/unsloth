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
    pinned: () => true,
    agrees: (c, s) =>
      (c.customContextLength ?? null) === (s.requested_context_length ?? null),
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
    pinned: () => true,
    agrees: (c, s, standing) =>
      (c.gpuMemoryMode ?? standing.gpuMemoryMode) ===
      (s.gpu_memory_mode ?? standing.gpuMemoryMode),
  },
  {
    // Resolves to GPU_LAYERS_AUTO rather than to a preference, but the load still sends it.
    pinned: () => true,
    agrees: (c, s, standing) =>
      (c.gpuLayers ?? standing.gpuLayers) ===
      (s.gpu_layers ?? standing.gpuLayers),
  },
  {
    pinned: () => true,
    agrees: (c, s, standing) =>
      (c.nCpuMoe ?? standing.nCpuMoe) === (s.n_cpu_moe ?? standing.nCpuMoe),
  },
  {
    // Absent resolves to a null selection in the applier, and a null selection is sent as
    // Automatic, so this is pinned like the rest: an unset pick does not adopt a server
    // that was placed on a chosen GPU. Reconciled first, since that is what /load sends.
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
];

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
  return SETTING_CHECKS.every(
    (check) => !check.pinned(config) || check.agrees(config, status, standing),
  );
}
