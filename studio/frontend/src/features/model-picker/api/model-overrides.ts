// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Server-side mirror of the per-model config.
//
// The config in ../model-config/per-model-config.ts lives in browser localStorage,
// so it only applied to loads the browser made, and an API auto-switch load (no
// browser in the loop) came up with none of the user's settings. Mirroring every
// save to the backend's override map closes that gap: routes/inference.py reads it
// and rebuilds the same LoadRequest the picker would have sent.

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
 * Translate the UI's per-model config into the backend's schema.
 *
 * Only fields the user set are sent: the backend reads an absent field as "app
 * default", so nulls would pin defaults and stop the model following later global
 * changes. A `null` config means "no saved settings", which clears the entry.
 */
function toApiOverride(config: PerModelConfig | null): ApiModelOverride {
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
  if (config.selectedGpuIds && config.selectedGpuIds.length > 0) {
    payload.gpu_ids = config.selectedGpuIds;
  }
  return payload;
}

// One in-flight write per model, so writes for a model commit in issue order.
// Otherwise saving twice quickly, or saving during the one-time backfill, races:
// the older response can land last and resurrect the entry the newer one meant to
// replace. Different models still overlap.
const writesByKey = new Map<string, Promise<void>>();

export async function putModelOverride(
  modelId: string,
  ggufVariant: string | null | undefined,
  config: PerModelConfig | null,
): Promise<void> {
  // Keyed by the folded identity, not the literal spelling: the backfill sends a
  // legacy casing and a UI save the normalized one, and the backend resolves both
  // to one row, so raw strings would open two queues and race again.
  const key = modelOverrideKey(
    normalizeModelIdentity(modelId),
    normalizeGgufVariantIdentity(ggufVariant),
  );
  // Chain on the settled tail: a failed write must not cancel the next one.
  const previous = writesByKey.get(key) ?? Promise.resolve();
  const write = previous
    .catch(() => {})
    .then(() => sendModelOverride(modelId, ggufVariant, config));
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
): Promise<void> {
  const res = await authFetch(OVERRIDES_URL, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // biome-ignore lint/style/useNamingConvention: API schema
      model_id: modelOverrideKey(modelId, ggufVariant),
      // Say which operation this is: an all-default save carries no fields, which is
      // shape-identical to "forget this model", and guessing wrong wipes launch flags
      // the UI cannot show or restore.
      remove: config === null,
      // Launch flags have no UI control, so the backend preserves them when omitted.
      // Forgetting means forgetting all of it, so that path sends an explicit [].
      // biome-ignore lint/style/useNamingConvention: API schema
      ...(config === null ? { llama_extra_args: [] } : {}),
      ...toApiOverride(config),
    }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to save model settings for the API"),
    );
  }
}

/**
 * Mirror a per-model config save to the backend without blocking the UI.
 *
 * Best-effort: the localStorage write is this browser's source of truth and has
 * already happened, so a failed sync must not fail the save or interrupt a load.
 * Logged rather than toasted: an API load of this model just falls back to app
 * defaults until the next successful save.
 */
export function syncModelOverride(
  modelId: string,
  ggufVariant: string | null | undefined,
  config: PerModelConfig | null,
): void {
  void putModelOverride(modelId, ggufVariant, config).catch(
    (error: unknown) => {
      console.warn(
        "Failed to mirror model settings to the server; an API load of this model will use defaults.",
        error,
      );
    },
  );
}
