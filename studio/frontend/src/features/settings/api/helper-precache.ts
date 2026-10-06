// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

const HELPER_PRECACHE_EVENT = "unsloth-helper-precache-change";

export type HelperPrecacheSettings = {
  enabled: boolean;
  defaultEnabled: boolean;
  disabledByEnv: boolean;
};

type ApiHelperPrecacheSettings = {
  enabled: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_enabled: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  disabled_by_env: boolean;
};

let cachedHelperPrecache: HelperPrecacheSettings | null = null;
let inFlightHelperPrecache: Promise<HelperPrecacheSettings> | null = null;

export function subscribeHelperPrecacheSettings(
  listener: (settings: HelperPrecacheSettings) => void,
) {
  const handleChange = (event: Event) => {
    listener((event as CustomEvent<HelperPrecacheSettings>).detail);
  };
  window.addEventListener(HELPER_PRECACHE_EVENT, handleChange);
  return () => window.removeEventListener(HELPER_PRECACHE_EVENT, handleChange);
}

function fromApi(settings: ApiHelperPrecacheSettings): HelperPrecacheSettings {
  return {
    enabled: settings.enabled,
    defaultEnabled: settings.default_enabled,
    disabledByEnv: settings.disabled_by_env,
  };
}

function cacheHelperPrecache(settings: HelperPrecacheSettings) {
  cachedHelperPrecache = settings;
  window.dispatchEvent(
    new CustomEvent(HELPER_PRECACHE_EVENT, { detail: settings }),
  );
  return settings;
}

async function fetchHelperPrecacheSettings(): Promise<HelperPrecacheSettings> {
  const res = await authFetch("/api/settings/helper-precache");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load Helper LLM settings"),
    );
  }
  return fromApi(await res.json());
}

export async function loadHelperPrecacheSettings() {
  if (cachedHelperPrecache) {
    return cachedHelperPrecache;
  }
  inFlightHelperPrecache ??= fetchHelperPrecacheSettings()
    .then(cacheHelperPrecache)
    .finally(() => {
      inFlightHelperPrecache = null;
    });
  return inFlightHelperPrecache;
}

export async function updateHelperPrecacheSettings(
  enabled: boolean,
): Promise<HelperPrecacheSettings> {
  const res = await authFetch("/api/settings/helper-precache", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to update Helper LLM settings"),
    );
  }
  return cacheHelperPrecache(fromApi(await res.json()));
}
