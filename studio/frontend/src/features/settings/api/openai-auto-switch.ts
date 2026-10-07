// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type OpenAIAutoSwitchSettings = {
  enabled: boolean;
  autoUnloadIdleSeconds: number;
  defaultEnabled: boolean;
  // True when idle unload will really run (e.g. UNSLOTH_MODEL_IDLE_TTL set with the toggle off).
  idleUnloadActive: boolean;
  autoUnloadKeepKv: boolean;
  // Stored independently of `enabled` but gated on it.
  autoDownloadModel: boolean;
  autoUnloadApiOnly: boolean;
  // Separate from the chat TTL and off by default.
  mediaAutoUnloadIdleSeconds: number;
  // Lets the UI say a veto (residency) is holding a saved TTL off.
  mediaIdleUnloadActive: boolean;
  mediaAutoSwitchModel: boolean;
};

type ApiOpenAIAutoSwitchSettings = {
  enabled: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_unload_idle_seconds: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_enabled: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  idle_unload_active?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_unload_keep_kv?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_download_model?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_unload_api_only?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  media_auto_unload_idle_seconds?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  media_idle_unload_active?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  media_auto_switch_model?: boolean;
};

let cachedSettings: OpenAIAutoSwitchSettings | null = null;
let inFlightSettings: Promise<OpenAIAutoSwitchSettings> | null = null;
// A caller arriving after an invalidation must not adopt a request issued before it.
let inFlightGeneration = -1;

function fromApi(
  settings: ApiOpenAIAutoSwitchSettings,
): OpenAIAutoSwitchSettings {
  return {
    enabled: settings.enabled,
    autoUnloadIdleSeconds: settings.auto_unload_idle_seconds,
    defaultEnabled: settings.default_enabled,
    idleUnloadActive: settings.idle_unload_active ?? false,
    autoUnloadKeepKv: settings.auto_unload_keep_kv ?? true,
    autoDownloadModel: settings.auto_download_model ?? false,
    autoUnloadApiOnly: settings.auto_unload_api_only ?? false,
    mediaAutoUnloadIdleSeconds: settings.media_auto_unload_idle_seconds ?? 0,
    mediaIdleUnloadActive: settings.media_idle_unload_active ?? false,
    mediaAutoSwitchModel: settings.media_auto_switch_model ?? false,
  };
}

async function fetchOpenAIAutoSwitchSettings(): Promise<OpenAIAutoSwitchSettings> {
  const res = await authFetch("/api/settings/openai-auto-switch");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load model auto-switch settings"),
    );
  }
  return fromApi(await res.json());
}

// A response in flight across an invalidation must not refill the cache.
let cacheGeneration = 0;

function cacheSettings(settings: OpenAIAutoSwitchSettings, generation: number) {
  if (generation === cacheGeneration) {
    cachedSettings = settings;
  }
  return settings;
}

/** Called by the Model Memory endpoint, since residency changes `idleUnloadActive`. */
export function invalidateOpenAIAutoSwitchSettings() {
  cachedSettings = null;
  cacheGeneration += 1;
}

// Retries converge quickly; the bound only stops a write storm from spinning.
const MAX_REREADS = 3;

function startRead(generation: number) {
  const read = fetchOpenAIAutoSwitchSettings().finally(() => {
    if (inFlightGeneration === generation) {
      inFlightSettings = null;
    }
  });
  inFlightSettings = read;
  inFlightGeneration = generation;
  return read;
}

export async function loadOpenAIAutoSwitchSettings() {
  let settings: OpenAIAutoSwitchSettings | null = null;
  for (let attempt = 0; attempt < MAX_REREADS; attempt += 1) {
    if (cachedSettings) {
      return cachedSettings;
    }
    const generation = cacheGeneration;
    settings = await (inFlightSettings && inFlightGeneration === generation
      ? inFlightSettings
      : startRead(generation));
    if (generation === cacheGeneration) {
      return cacheSettings(settings, generation);
    }
    // The response predates the write and callers apply it directly, so refetch instead.
  }
  return settings as OpenAIAutoSwitchSettings;
}

/** `enabled` is always sent; omitted fields keep their stored value. */
export type OpenAIAutoSwitchUpdate = {
  enabled: boolean;
  autoUnloadIdleSeconds?: number;
  autoUnloadKeepKv?: boolean;
  autoDownloadModel?: boolean;
  autoUnloadApiOnly?: boolean;
  mediaAutoUnloadIdleSeconds?: number;
  mediaAutoSwitchModel?: boolean;
};

const UPDATE_KEYS = {
  autoUnloadIdleSeconds: "auto_unload_idle_seconds",
  autoUnloadKeepKv: "auto_unload_keep_kv",
  autoDownloadModel: "auto_download_model",
  autoUnloadApiOnly: "auto_unload_api_only",
  mediaAutoUnloadIdleSeconds: "media_auto_unload_idle_seconds",
  mediaAutoSwitchModel: "media_auto_switch_model",
} as const;

export async function updateOpenAIAutoSwitchSettings(
  update: OpenAIAutoSwitchUpdate,
): Promise<OpenAIAutoSwitchSettings> {
  const body: Record<string, unknown> = { enabled: update.enabled };
  for (const [field, key] of Object.entries(UPDATE_KEYS)) {
    const value = update[field as keyof typeof UPDATE_KEYS];
    if (value !== undefined) {
      body[key] = value;
    }
  }
  // Read before the request: a residency write landing mid-flight makes our own reply stale.
  const generation = cacheGeneration;
  const res = await authFetch("/api/settings/openai-auto-switch", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(
        res,
        "Failed to update model auto-switch settings",
      ),
    );
  }
  const settings = fromApi(await res.json());
  if (generation !== cacheGeneration) {
    // A Model Memory write landed mid-PUT; caching this reply would pin a stale idleUnloadActive.
    return loadOpenAIAutoSwitchSettings();
  }
  return cacheSettings(settings, generation);
}
