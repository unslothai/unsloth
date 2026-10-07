// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Server mirror of per-model config (otherwise browser-only) so API auto-switch loads get it.

import { authFetch } from "@/features/auth";
import type { GpuIndexKind } from "@/hooks/gpu-selection";
import { readFastApiError } from "@/lib/format-fastapi-error";
import {
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
  splitQuantSuffix,
} from "../model-config/model-identity";
import {
  DEFAULT_PER_MODEL_CONFIG,
  normalizeMlxKvQuant,
  type PerModelConfig,
  deletePerModelConfigsForOverrideKeys,
  normalizePerModelConfig,
} from "../model-config/per-model-config";

const OVERRIDES_URL = "/api/settings/openai-auto-switch/overrides";

export interface ApiModelOverride {
  engine_parallelism?: "tensor" | "pipeline" | "data";
  engine_precision?: "auto" | "bf16" | "fp16" | "int4" | "int8" | "fp8";
  engine?: "auto" | "vllm" | "sglang";
  // biome-ignore lint/style/useNamingConvention: API schema
  llama_extra_args?: string[];
  // biome-ignore lint/style/useNamingConvention: API schema
  max_seq_length?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  custom_context_length?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  kv_cache_dtype?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  mlx_kv_quant?: string;
  mlx_kv_bits?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  mlx_int8_prefill?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  speculative_type?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  spec_draft_n_max?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  n_parallel?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  reasoning_budget?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  reasoning_budget_message?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  n_batch?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  n_ubatch?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  load_mode?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  spec_draft_cache_type?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  ctx_checkpoints?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  cache_ram?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  tensor_parallel?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  disable_vision?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  chat_template_override?: string;
  // biome-ignore lint/style/useNamingConvention: API schema
  gpu_memory_mode?: "auto" | "manual";
  // biome-ignore lint/style/useNamingConvention: API schema
  gpu_layers?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  n_cpu_moe?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  gpu_ids?: number[];
  // Absent means "physical", all an older row could mean.
  // biome-ignore lint/style/useNamingConvention: API schema
  gpu_index_kind?: GpuIndexKind;
}

export type ApiModelOverrides = Record<string, ApiModelOverride>;

/** `repo:VARIANT` so two quants of one repo keep separate configs. */
export function modelOverrideKey(
  modelId: string,
  ggufVariant?: string | null,
): string {
  return ggufVariant ? `${modelId}:${ggufVariant}` : modelId;
}

/** Must match the backend's _fold_case_insensitive_path; POSIX paths are never folded. */
const WINDOWS_DRIVE_PATH = /^[a-zA-Z]:[\\/]/;
const WSL_DRIVE_PATH = /^\/mnt\/[a-zA-Z](\/|$)/;

function foldCaseInsensitivePath(key: string): string | null {
  const slashed = key.replace(/\\/g, "/");
  let minimum: number;
  if (WINDOWS_DRIVE_PATH.test(key)) {
    minimum = 3;
  } else if (slashed.startsWith("//")) {
    minimum = 2;
  } else if (WSL_DRIVE_PATH.test(slashed)) {
    minimum = 6;
  } else {
    return null;
  }
  let trimmed = slashed;
  while (trimmed.length > minimum && trimmed.endsWith("/")) {
    trimmed = trimmed.slice(0, -1);
  }
  return trimmed.toLowerCase();
}

export function foldOverrideKey(key: string): string {
  const path = foldCaseInsensitivePath(key);
  if (path !== null) {
    return path;
  }
  // A colon is legal in a POSIX filename; mirrors the backend's split_quant_suffix.
  const split = splitQuantSuffix(key);
  const id = split ? split[0] : key;
  const quant = split ? `:${split[1].toLowerCase()}` : "";
  // The browser lowercases the quant before storing; a repo id folds whole.
  return id.startsWith("/") ? `${id}${quant}` : `${id.toLowerCase()}${quant}`;
}

/** Keeps an explicit [] (tombstone) apart from an absent field; collapsing them leaked args. */
export type ResolvedExtraArgs = {
  tokens: string[];
  explicit: boolean;
};

function resolvedFrom(entry: ApiModelOverride): ResolvedExtraArgs {
  const tokens = entry.llama_extra_args;
  return { tokens: tokens ?? [], explicit: Array.isArray(tokens) };
}

function presentOverride(
  value: ApiModelOverride | undefined | null,
): ApiModelOverride | null {
  return value && Object.keys(value).length > 0 ? value : null;
}

export function resolveStoredOverride(
  overrides: ApiModelOverrides,
  keys: readonly string[],
): ApiModelOverride | null {
  // Mirrors the server: most specific key first, whole entries, stop at the first non-empty one.
  const folded = new Map<string, ApiModelOverride | null>();
  for (const [key, value] of Object.entries(overrides)) {
    if (!presentOverride(value)) {
      continue;
    }
    const foldedKey = foldOverrideKey(key);
    // Ambiguous folds resolve to nothing, as resolve_model_override_key does.
    folded.set(foldedKey, folded.has(foldedKey) ? null : value);
  }
  for (const key of keys) {
    const exact = presentOverride(overrides[key]);
    if (exact) {
      return exact;
    }
    const match = folded.get(foldOverrideKey(key));
    if (match) {
      return match;
    }
  }
  return null;
}

export function resolveStoredExtraArgs(
  overrides: ApiModelOverrides,
  keys: readonly string[],
): ResolvedExtraArgs {
  const resolved = resolveStoredOverride(overrides, keys);
  return resolved ? resolvedFrom(resolved) : { tokens: [], explicit: false };
}

export async function fetchModelOverrides(): Promise<ApiModelOverrides> {
  const res = await authFetch(OVERRIDES_URL);
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load saved model settings"),
    );
  }
  const body = (await res.json()) as { overrides?: ApiModelOverrides };
  return body.overrides ?? {};
}

/** Asked of the server: Python's casefold and ambiguity rules are only approximable here. */
export async function fetchLoadModelOverride(
  loadId: string,
  aliasId: string,
  ggufVariant?: string | null,
  fallbackKeys: readonly string[] = [],
): Promise<ApiModelOverride | null> {
  const query = new URLSearchParams({ model_id: loadId, alias_id: aliasId });
  if (ggufVariant) {
    query.set("gguf_variant", ggufVariant);
  }
  const res = await authFetch(`${OVERRIDES_URL}?${query.toString()}`);
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load saved model settings"),
    );
  }
  const body = (await res.json()) as {
    overrides?: ApiModelOverrides;
    resolved?: ApiModelOverride | null;
    // biome-ignore lint/style/useNamingConvention: API schema
    resolved_key?: string | null;
  };
  if (body.resolved !== undefined) {
    return presentOverride(body.resolved);
  }
  // An older backend returns the whole map and needs the keys to look under.
  const derived =
    fallbackKeys.length > 0
      ? fallbackKeys
      : [
          modelOverrideKey(loadId, ggufVariant),
          modelOverrideKey(aliasId, ggufVariant),
          loadId,
          aliasId,
        ].filter((key, index, all) => all.indexOf(key) === index);
  return resolveStoredOverride(body.overrides ?? {}, derived);
}

export async function fetchLoadExtraArgs(
  loadId: string,
  aliasId: string,
  ggufVariant?: string | null,
  fallbackKeys: readonly string[] = [],
): Promise<ResolvedExtraArgs> {
  const resolved = await fetchLoadModelOverride(
    loadId,
    aliasId,
    ggufVariant,
    fallbackKeys,
  );
  // An explicit [] is a cleared box; omitting it lets /load carry the resident model's args over.
  return resolvedFrom(resolved ?? {});
}

/** llama-server arguments only apply to GGUF loads. */
export function panelOverrideRow(
  override: ApiModelOverride | null,
  isGguf: boolean,
): ApiModelOverride | null {
  if (!override || isGguf) {
    return override;
  }
  return presentOverride(
    Object.fromEntries(
      Object.entries(override).filter(([key]) => key !== "llama_extra_args"),
    ),
  );
}

/**
 * The row is authoritative only for fields it carries; absent fields fall back to
 * `localConfig`, since an absent field is not evidence of a default.
 */
export function fromApiOverride(
  override: ApiModelOverride,
  localConfig?: PerModelConfig,
): PerModelConfig {
  const local = localConfig ?? DEFAULT_PER_MODEL_CONFIG;
  const extraArgs = Array.isArray(override.llama_extra_args)
    ? override.llama_extra_args
    : local.llamaExtraArgs;
  // A row without ids says nothing about placement, so the local pin keeps its namespace.
  const serverGpuIds = override.gpu_ids?.length ? override.gpu_ids : null;
  // The context pin spans two fields; a row stating either owns both, or it loads at two lengths.
  const serverStatesPin =
    override.custom_context_length != null || override.max_seq_length != null;
  const serverStatesKvQuant =
    "mlx_kv_quant" in override || "mlx_kv_bits" in override;
  const normalized = normalizePerModelConfig({
    ...DEFAULT_PER_MODEL_CONFIG,
    engine: override.engine ?? "auto",
    engineParallelism: override.engine_parallelism ?? local.engineParallelism ?? "tensor",
    enginePrecision:
      override.engine_precision ?? local.enginePrecision ?? "auto",
    customContextLength: serverStatesPin
      ? (override.custom_context_length ?? null)
      : local.customContextLength,
    maxSeqLength: serverStatesPin
      ? (override.max_seq_length ?? null)
      : local.maxSeqLength,
    kvCacheDtype: override.kv_cache_dtype ?? local.kvCacheDtype,
    mlxKvQuant: serverStatesKvQuant
      ? normalizeMlxKvQuant(override.mlx_kv_quant, override.mlx_kv_bits)
      : (local.mlxKvQuant ?? null),
    speculativeType: override.speculative_type ?? local.speculativeType,
    specDraftNMax: override.spec_draft_n_max ?? local.specDraftNMax,
    specDraftCacheDtype:
      override.spec_draft_cache_type ?? local.specDraftCacheDtype,
    nParallel: override.n_parallel ?? local.nParallel,
    reasoningBudget: override.reasoning_budget ?? local.reasoningBudget,
    reasoningBudgetMessage:
      override.reasoning_budget_message ?? local.reasoningBudgetMessage,
    nBatch: override.n_batch ?? local.nBatch,
    nUbatch: override.n_ubatch ?? local.nUbatch,
    loadMode: override.load_mode ?? local.loadMode,
    ctxCheckpoints: override.ctx_checkpoints ?? local.ctxCheckpoints,
    cacheRam: override.cache_ram ?? local.cacheRam,
    // Both are stored only when true, so an absent one is a gap.
    tensorParallel: override.tensor_parallel ?? local.tensorParallel,
    disableVision: override.disable_vision ?? local.disableVision,
    mlxInt8Prefill: override.mlx_int8_prefill ?? local.mlxInt8Prefill,
    chatTemplateOverride:
      override.chat_template_override ?? local.chatTemplateOverride,
    llamaExtraArgs: extraArgs,
    gpuMemoryMode: override.gpu_memory_mode ?? local.gpuMemoryMode,
    gpuLayers: override.gpu_layers ?? local.gpuLayers,
    nCpuMoe: override.n_cpu_moe ?? local.nCpuMoe,
    selectedGpuIds: serverGpuIds ?? local.selectedGpuIds ?? null,
    selectedGpuIndexKind: serverGpuIds
      ? (override.gpu_index_kind ?? "physical")
      : (local.selectedGpuIndexKind ?? null),
  });
  // The server uses [] as a tombstone, so keep it rather than collapsing to null.
  if (Array.isArray(extraArgs)) {
    normalized.llamaExtraArgs = [...extraArgs];
  }
  return normalized;
}

/** Only user-set fields are sent: absent means app default. `null` clears the entry. */
export function toApiOverride(config: PerModelConfig | null): ApiModelOverride {
  if (!config) {
    return {};
  }
  // Engine fields are always sent: the server keeps a stored engine when the field is absent.
  const payload: ApiModelOverride = {
    engine: config.engine ?? "auto",
    engine_precision: config.enginePrecision ?? "auto",
    engine_parallelism: config.engineParallelism ?? "tensor",
  };
  if (config.maxSeqLength && config.maxSeqLength > 0) {
    payload.max_seq_length = config.maxSeqLength;
  }
  if (config.customContextLength && config.customContextLength > 0) {
    payload.custom_context_length = config.customContextLength;
  }
  if (config.kvCacheDtype) {
    payload.kv_cache_dtype = config.kvCacheDtype;
  }
  // Must travel with kv_cache_dtype, or an API load runs the MLX model at full precision.
  if (config.mlxKvQuant) {
    payload.mlx_kv_quant = config.mlxKvQuant;
  }
  if (config.speculativeType) {
    payload.speculative_type = config.speculativeType;
  }
  if (config.specDraftNMax && config.specDraftNMax > 0) {
    payload.spec_draft_n_max = config.specDraftNMax;
  }
  if (config.nParallel && config.nParallel > 0) {
    payload.n_parallel = config.nParallel;
  }
  if (config.reasoningBudget !== -1) {
    payload.reasoning_budget = config.reasoningBudget;
  }
  if (config.reasoningBudgetMessage) {
    payload.reasoning_budget_message = config.reasoningBudgetMessage;
  }
  if (config.nBatch && config.nBatch > 0) {
    payload.n_batch = config.nBatch;
  }
  if (config.nUbatch && config.nUbatch > 0) {
    payload.n_ubatch = config.nUbatch;
  }
  // cacheRam is compared with null since 0 and -1 both mean something.
  if (config.loadMode) {
    payload.load_mode = config.loadMode;
  }
  if (config.specDraftCacheDtype) {
    payload.spec_draft_cache_type = config.specDraftCacheDtype;
  }
  if (config.ctxCheckpoints != null && config.ctxCheckpoints >= 0) {
    payload.ctx_checkpoints = config.ctxCheckpoints;
  }
  if (config.cacheRam != null) {
    payload.cache_ram = config.cacheRam;
  }
  if (config.tensorParallel) {
    payload.tensor_parallel = true;
  }
  if (config.disableVision) {
    payload.disable_vision = true;
  }
  if (config.mlxInt8Prefill) {
    payload.mlx_int8_prefill = true;
  }
  if (config.chatTemplateOverride?.trim()) {
    payload.chat_template_override = config.chatTemplateOverride;
  }
  // Here absent means preserve, not default: omit undefined, send [] for a cleared box.
  if (config.llamaExtraArgs !== undefined) {
    payload.llama_extra_args = config.llamaExtraArgs ?? [];
  }
  if (config.gpuMemoryMode === "manual") {
    payload.gpu_memory_mode = "manual";
  }
  if (typeof config.gpuLayers === "number" && config.gpuLayers >= 0) {
    payload.gpu_layers = config.gpuLayers;
  }
  if (typeof config.nCpuMoe === "number" && config.nCpuMoe > 0) {
    payload.n_cpu_moe = config.nCpuMoe;
  }
  // The pin travels with its index namespace; reconcileGpuSelection drops it on a mismatch.
  const gpuIndexKind = config.selectedGpuIndexKind ?? "physical";
  if (config.selectedGpuIds && config.selectedGpuIds.length > 0) {
    payload.gpu_ids = config.selectedGpuIds;
    if (gpuIndexKind !== "physical") {
      payload.gpu_index_kind = gpuIndexKind;
    }
  }
  return payload;
}

// One in-flight write per model so an older response cannot resurrect a replaced entry.
const writesByKey = new Map<string, Promise<ModelOverrideWriteResult>>();

export interface ModelOverrideWriteResult {
  overrides: ApiModelOverrides;
  removedKeys: string[];
}

export interface PutModelOverrideOptions {
  /** Fill only missing fields, so another tab's save is not overwritten by an older copy. */
  fillAbsentFields?: boolean;
  /** Evicting for the storage budget is not a forget: keep `llama_extra_args`. */
  keepLaunchFlags?: boolean;
  resetReasoningBudget?: boolean;
  resetReasoningBudgetMessage?: boolean;
}

export async function putModelOverride(
  modelId: string,
  ggufVariant: string | null | undefined,
  config: PerModelConfig | null,
  options?: PutModelOverrideOptions,
): Promise<ModelOverrideWriteResult> {
  // Keyed by folded identity, or two casings open two queues and race.
  const key = modelOverrideKey(
    normalizeModelIdentity(modelId),
    normalizeGgufVariantIdentity(ggufVariant),
  );
  // Chain on the settled tail: a failed write must not cancel the next one.
  const previous = writesByKey.get(key) ?? Promise.resolve();
  const write = previous
    .catch(() => {})
    .then(() => sendModelOverride(modelId, ggufVariant, config, options));
  writesByKey.set(key, write);
  try {
    return await write;
  } finally {
    // Only the last writer clears the slot, so a queue still building keeps order.
    if (writesByKey.get(key) === write) {
      writesByKey.delete(key);
    }
  }
}

async function sendModelOverride(
  modelId: string,
  ggufVariant: string | null | undefined,
  config: PerModelConfig | null,
  options?: PutModelOverrideOptions,
): Promise<ModelOverrideWriteResult> {
  const res = await authFetch(OVERRIDES_URL, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // biome-ignore lint/style/useNamingConvention: API schema
      model_id: modelOverrideKey(modelId, ggufVariant),
      // Tells the backend an omission is a clear, not an older client; older backends ignore it.
      // biome-ignore lint/style/useNamingConvention: API schema
      mirrors_server_tuning: true,
      // biome-ignore lint/style/useNamingConvention: API schema
      mirrors_reasoning_budget: true,
      ...(options?.fillAbsentFields
        ? // biome-ignore lint/style/useNamingConvention: API schema
          { fill_absent_fields: true }
        : {}),
      // An all-default save is shape-identical to a forget, so say which; guessing wrong wipes flags.
      remove: config === null && !options?.keepLaunchFlags,
      // Omitted args are preserved by the backend; a forget sends an explicit [].
      ...(config === null && !options?.keepLaunchFlags
        ? // biome-ignore lint/style/useNamingConvention: API schema
          { llama_extra_args: [] }
        : {}),
      ...toApiOverride(config),
      // Reset markers remove legacy passthrough flags; fill-only migration must never delete.
      ...(options?.resetReasoningBudget && config?.reasoningBudget === -1
        ? {
            // biome-ignore lint/style/useNamingConvention: API schema
            reasoning_budget: -1,
          }
        : {}),
      ...(options?.resetReasoningBudgetMessage && config?.reasoningBudgetMessage === ""
        ? {
            // biome-ignore lint/style/useNamingConvention: API schema
            reasoning_budget_message: "",
          }
        : {}),
    }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to save model settings for the API"),
    );
  }
  const body = (await res.json()) as {
    overrides?: ApiModelOverrides;
    // biome-ignore lint/style/useNamingConvention: API schema
    removed_keys?: unknown;
  };
  return {
    overrides: body.overrides ?? {},
    removedKeys: Array.isArray(body.removed_keys)
      ? body.removed_keys.filter(
          (key): key is string => typeof key === "string",
        )
      : [],
  };
}

/** Best-effort: the localStorage write already happened, so failures are only logged. */
export function syncModelOverride(
  modelId: string,
  ggufVariant: string | null | undefined,
  config: PerModelConfig | null,
  options?: PutModelOverrideOptions,
): void {
  void putModelOverride(modelId, ggufVariant, config, options)
    .then(({ removedKeys }) => {
      if (!deletePerModelConfigsForOverrideKeys(removedKeys)) {
        console.warn(
          "Forgot model settings on the server, but this browser kept its own copy.",
        );
      }
    })
    .catch((error: unknown) => {
      console.warn(
        "Failed to mirror model settings to the server; an API load of this model will use defaults.",
        error,
      );
    });
}
