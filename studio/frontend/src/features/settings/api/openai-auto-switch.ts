// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type OpenAIAutoSwitchSettings = {
  enabled: boolean;
  autoUnloadIdleSeconds: number;
  defaultEnabled: boolean;
  // True when the idle-unload loop will actually unload (e.g. enabled via the
  // UNSLOTH_MODEL_IDLE_TTL env var even while the toggle is off).
  idleUnloadActive: boolean;
  // Persist the KV cache to disk on idle unload and restore it on reload.
  autoUnloadKeepKv: boolean;
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
};

let cachedSettings: OpenAIAutoSwitchSettings | null = null;
let inFlightSettings: Promise<OpenAIAutoSwitchSettings> | null = null;

function fromApi(
  settings: ApiOpenAIAutoSwitchSettings,
): OpenAIAutoSwitchSettings {
  return {
    enabled: settings.enabled,
    autoUnloadIdleSeconds: settings.auto_unload_idle_seconds,
    defaultEnabled: settings.default_enabled,
    idleUnloadActive: settings.idle_unload_active ?? false,
    autoUnloadKeepKv: settings.auto_unload_keep_kv ?? true,
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

function cacheSettings(settings: OpenAIAutoSwitchSettings) {
  cachedSettings = settings;
  return settings;
}

export async function loadOpenAIAutoSwitchSettings() {
  if (cachedSettings) {
    return cachedSettings;
  }
  inFlightSettings ??= fetchOpenAIAutoSwitchSettings()
    .then(cacheSettings)
    .finally(() => {
      inFlightSettings = null;
    });
  return inFlightSettings;
}

export async function updateOpenAIAutoSwitchSettings(
  enabled: boolean,
  autoUnloadIdleSeconds?: number,
  autoUnloadKeepKv?: boolean,
): Promise<OpenAIAutoSwitchSettings> {
  const res = await authFetch("/api/settings/openai-auto-switch", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      enabled,
      // Omitted fields keep their stored value.
      ...(autoUnloadIdleSeconds === undefined
        ? {}
        : // biome-ignore lint/style/useNamingConvention: API schema
          { auto_unload_idle_seconds: autoUnloadIdleSeconds }),
      ...(autoUnloadKeepKv === undefined
        ? {}
        : // biome-ignore lint/style/useNamingConvention: API schema
          { auto_unload_keep_kv: autoUnloadKeepKv }),
    }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(
        res,
        "Failed to update model auto-switch settings",
      ),
    );
  }
  return cacheSettings(fromApi(await res.json()));
}
