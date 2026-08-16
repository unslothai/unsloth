// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PerModelConfig } from "@/features/model-picker";

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
   * `normalizeSpeculativeType`, passed rather than imported. It lives on the chat runtime
   * store, which reaches React, and this module is deliberately a leaf so the node suite
   * can drive it; copying its mapping here (comma-chained legacy echoes and all) would be
   * a second copy free to drift from the one the load actually uses.
   */
  normalizeSpeculative: (value: string | null | undefined) => string | null;
};

/**
 * One setting the resident load can disagree about. `pinned` is whether the config has an
 * opinion at all, `agrees` whether the running server satisfies it. Keeping them apart is
 * the point: an unset field must not read as a demand for the default, except the four
 * above, which the applier always resolves and so are always pinned.
 */
type SettingCheck = {
  pinned: (config: PerModelConfig) => boolean;
  agrees: (
    config: PerModelConfig,
    status: ResidentRuntime,
    standing: StandingConfigDefaults,
  ) => boolean;
};

const set = (value: unknown): boolean => value != null;

/** Mirrors the fields `_runtime_matches_intent` reloads for, plus the MLX pair
 * `_mlx_runtime_settings_match` compares. */
const SETTING_CHECKS: SettingCheck[] = [
  {
    pinned: (c) => set(c.customContextLength),
    agrees: (c, s) =>
      c.customContextLength === (s.requested_context_length ?? null),
  },
  {
    pinned: (c) => set(c.kvCacheDtype),
    agrees: (c, s) => c.kvCacheDtype === (s.cache_type_kv ?? null),
  },
  {
    pinned: (c) => set(c.mlxKvBits),
    agrees: (c, s) => c.mlxKvBits === (s.mlx_kv_bits_requested ?? null),
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
    pinned: (c) => set(c.specDraftNMax),
    agrees: (c, s) => c.specDraftNMax === (s.spec_draft_n_max ?? null),
  },
  {
    pinned: (c) => set(c.nParallel),
    agrees: (c, s) => c.nParallel === (s.requested_parallel_slots ?? null),
  },
  {
    pinned: (c) => set(c.nBatch),
    agrees: (c, s) => c.nBatch === (s.requested_n_batch ?? null),
  },
  {
    pinned: (c) => set(c.nUbatch),
    agrees: (c, s) => c.nUbatch === (s.requested_n_ubatch ?? null),
  },
  {
    // Not nullable, so it always has an opinion; a status omitting it ran without.
    pinned: () => true,
    agrees: (c, s) => c.tensorParallel === (s.tensor_parallel ?? false),
  },
  {
    pinned: (c) => set(c.chatTemplateOverride),
    agrees: (c, s) =>
      c.chatTemplateOverride === (s.chat_template_override ?? null),
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
    // null or absent is Automatic, which pins nothing and so agrees with any placement.
    pinned: (c) => set(c.selectedGpuIds),
    agrees: (c, s) => sameGpuSet(c.selectedGpuIds, s.requested_gpu_ids),
  },
];

/**
 * Whether the resident load already runs the settings this pick would ask for.
 *
 * Identity is not the whole of a load: `LlamaCppBackend` reuses a running server only when
 * the request also agrees on context, KV dtype, slots, batch sizes, placement, speculative
 * mode, chat template and pass-through args. An unset field expresses no opinion and agrees
 * with whatever is running, which is what keeps the common case (no config at all) working.
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
