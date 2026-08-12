// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

const NO_CHAT_HISTORY_EVENT = "unsloth-no-chat-history-change";

export type NoChatHistorySettings = {
  enabled: boolean;
  defaultEnabled: boolean;
  forcedByEnv: boolean;
};

type ApiNoChatHistorySettings = {
  enabled: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_enabled: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  forced_by_env: boolean;
};

let cachedNoChatHistory: NoChatHistorySettings | null = null;
let inFlightNoChatHistory: Promise<NoChatHistorySettings> | null = null;

export function subscribeNoChatHistorySettings(
  listener: (settings: NoChatHistorySettings) => void,
) {
  const handleChange = (event: Event) => {
    listener((event as CustomEvent<NoChatHistorySettings>).detail);
  };
  window.addEventListener(NO_CHAT_HISTORY_EVENT, handleChange);
  return () => window.removeEventListener(NO_CHAT_HISTORY_EVENT, handleChange);
}

function fromApi(settings: ApiNoChatHistorySettings): NoChatHistorySettings {
  return {
    enabled: settings.enabled,
    defaultEnabled: settings.default_enabled,
    forcedByEnv: settings.forced_by_env,
  };
}

function cacheNoChatHistory(settings: NoChatHistorySettings) {
  cachedNoChatHistory = settings;
  window.dispatchEvent(
    new CustomEvent(NO_CHAT_HISTORY_EVENT, { detail: settings }),
  );
  return settings;
}

async function fetchNoChatHistorySettings(): Promise<NoChatHistorySettings> {
  const res = await authFetch("/api/settings/no-chat-history");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load chat history settings"),
    );
  }
  return fromApi(await res.json());
}

export async function loadNoChatHistorySettings({ force = false } = {}) {
  if (cachedNoChatHistory && !force) {
    return cachedNoChatHistory;
  }
  inFlightNoChatHistory ??= fetchNoChatHistorySettings()
    .then(cacheNoChatHistory)
    .finally(() => {
      inFlightNoChatHistory = null;
    });
  return inFlightNoChatHistory;
}

export async function updateNoChatHistorySettings(
  enabled: boolean,
): Promise<NoChatHistorySettings> {
  const res = await authFetch("/api/settings/no-chat-history", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to update chat history settings"),
    );
  }
  return cacheNoChatHistory(fromApi(await res.json()));
}

export function getCachedNoChatHistoryEnabled(): boolean {
  return cachedNoChatHistory?.enabled ?? false;
}
