// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Server-side mirror of the per-model config. ../model-config/per-model-config.ts lives in
// browser localStorage, so an API auto-switch load (no browser in the loop) came up with none of
// the user's settings. routes/inference.py reads this map and rebuilds the picker's LoadRequest.

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import {
  normalizeGgufVariantIdentity,
  normalizeModelIdentity,
} from "../model-config/model-identity";
import type { PerModelConfig } from "../model-config/per-model-config";

const OVERRIDES_URL = "/api/settings/openai-auto-switch/overrides";

/** One model's stored launch config, as the backend persists it. */
export interface ApiModelOverride {
  // biome-ignore lint/style/useNamingConvention: API schema
  llama_extra_args?: string[];
  // biome-ignore lint/style/useNamingConvention: API schema
  max_seq_length?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  custom_context_length?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  kv_cache_dtype?: string;
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
  tensor_parallel?: boolean;
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
}

export type ApiModelOverrides = Record<string, ApiModelOverride>;

/**
 * The key one model's config is stored under: the `repo:VARIANT` form an OpenAI
 * request names a quant by, so two quants of one repo keep separate configs and the
 * backend matches the requested name directly. Bare id when there is no variant.
 */
export function modelOverrideKey(
  modelId: string,
  ggufVariant?: string | null,
): string {
  return ggufVariant ? `${modelId}:${ggufVariant}` : modelId;
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

/**
 * Translate the UI's per-model config into the backend's schema. Only fields the user set are
 * sent: an absent field reads as "app default", so nulls would pin defaults and stop the model
 * following later global changes. A `null` config clears the entry. Exported so the backfill
 * can compare it against the stored entry.
 */
export function toApiOverride(config: PerModelConfig | null): ApiModelOverride {
  if (!config) {
    return {};
  }
  const payload: ApiModelOverride = {};
  if (config.maxSeqLength && config.maxSeqLength > 0) {
    payload.max_seq_length = config.maxSeqLength;
  }
  if (config.customContextLength && config.customContextLength > 0) {
    payload.custom_context_length = config.customContextLength;
  }
  if (config.kvCacheDtype) {
    payload.kv_cache_dtype = config.kvCacheDtype;
  }
  if (config.speculativeType) {
    payload.speculative_type = config.speculativeType;
  }
  if (config.specDraftNMax && config.specDraftNMax > 0) {
    payload.spec_draft_n_max = config.specDraftNMax;
  }
  // Blank follows the server-wide --parallel default, which is the app default here.
  if (config.nParallel && config.nParallel > 0) {
    payload.n_parallel = config.nParallel;
  }
  if (config.reasoningBudget !== -1) {
    payload.reasoning_budget = config.reasoningBudget;
  }
  if (config.reasoningBudgetMessage) {
    payload.reasoning_budget_message = config.reasoningBudgetMessage;
  }
  if (config.tensorParallel) {
    payload.tensor_parallel = true;
  }
  if (config.chatTemplateOverride?.trim()) {
    payload.chat_template_override = config.chatTemplateOverride;
  }
  // Only "manual" is a real override; "auto" is the follow-the-global default.
  if (config.gpuMemoryMode === "manual") {
    payload.gpu_memory_mode = "manual";
  }
  // gpuLayers < 0 is Auto, which is also the default.
  if (typeof config.gpuLayers === "number" && config.gpuLayers >= 0) {
    payload.gpu_layers = config.gpuLayers;
  }
  if (typeof config.nCpuMoe === "number" && config.nCpuMoe > 0) {
    payload.n_cpu_moe = config.nCpuMoe;
  }
  // Only a physical pin travels. The same integers are a Vulkan ordinal under Vulkan and a
  // CUDA/ROCm index elsewhere, and the override carries no namespace, so after a backend change
  // the server would pin a different device with ids its availability check accepts. An absent
  // kind predates the field, which was physical-only. Dropping the pin leaves auto placement.
  const gpuIndexKind = config.selectedGpuIndexKind ?? "physical";
  if (
    config.selectedGpuIds &&
    config.selectedGpuIds.length > 0 &&
    gpuIndexKind === "physical"
  ) {
    payload.gpu_ids = config.selectedGpuIds;
  }
  return payload;
}

// One in-flight write per model, so writes commit in issue order: otherwise the older
// response can land last and resurrect the entry the newer one replaced. Models overlap.
const writesByKey = new Map<string, Promise<void>>();

export interface PutModelOverrideOptions {
  /**
   * Fill in only what is missing: every value already on the server stays as it is. The backfill
   * reads the map once then writes each model in turn, so another tab's save would be overwritten
   * by this browser's older copy. Field level, so a legacy entry gains the browser-only fields.
   */
  fillAbsentFields?: boolean;
  /**
   * Clear the fields this UI mirrors but leave the server's own launch flags alone. Evicting a
   * local entry for the storage budget is not a forget, so it must not take `llama_extra_args`
   * the page can neither show nor restore. The route drops the row once nothing is left in it.
   */
  keepLaunchFlags?: boolean;
  /** Remove a legacy passthrough value only after this control was explicitly reset. */
  resetReasoningBudget?: boolean;
  resetReasoningBudgetMessage?: boolean;
}

export async function putModelOverride(
  modelId: string,
  ggufVariant: string | null | undefined,
  config: PerModelConfig | null,
  options?: PutModelOverrideOptions,
): Promise<void> {
  // Keyed by the folded identity: the backend resolves a legacy casing and the
  // normalized one to one row, so raw strings would open two queues and race again.
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
    await write;
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
): Promise<void> {
  const res = await authFetch(OVERRIDES_URL, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // biome-ignore lint/style/useNamingConvention: API schema
      model_id: modelOverrideKey(modelId, ggufVariant),
      // Only sent when set, so an older backend is not handed an unknown key every save.
      ...(options?.fillAbsentFields
        ? // biome-ignore lint/style/useNamingConvention: API schema
          { fill_absent_fields: true }
        : {}),
      // Say which operation this is: an all-default save carries no fields, shape-identical
      // to a forget, and guessing wrong wipes flags the UI cannot show or restore.
      remove: config === null && !options?.keepLaunchFlags,
      // Flags have no UI control, so the backend preserves them when omitted; a forget
      // means all of it, so that path sends an explicit [].
      ...(config === null && !options?.keepLaunchFlags
        ? // biome-ignore lint/style/useNamingConvention: API schema
          { llama_extra_args: [] }
        : {}),
      ...toApiOverride(config),
      // Write-only reset markers let the backend remove legacy passthrough flags
      // shadowing these controls. Fill-only migration must never delete stored flags.
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
}

/**
 * Mirror a per-model config save to the backend without blocking the UI. Best-effort: the
 * localStorage write already happened, so a failed sync must not fail the save. Logged, not
 * toasted: an API load falls back to defaults until the next save.
 */
export function syncModelOverride(
  modelId: string,
  ggufVariant: string | null | undefined,
  config: PerModelConfig | null,
  options?: PutModelOverrideOptions,
): void {
  void putModelOverride(modelId, ggufVariant, config, options).catch(
    (error: unknown) => {
      console.warn(
        "Failed to mirror model settings to the server; an API load of this model will use defaults.",
        error,
      );
    },
  );
}
