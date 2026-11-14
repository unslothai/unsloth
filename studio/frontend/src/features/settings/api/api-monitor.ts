// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

type ApiMonitorSettings = {
  enabled: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  effective_enabled: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  disabled_by_env: boolean;
};

/** Read the installation-wide API recording preference. */
export async function loadApiMonitorSettings(): Promise<ApiMonitorSettings> {
  const response = await authFetch("/api/settings/api-monitor");
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Failed to load API monitor settings"),
    );
  }
  return response.json();
}

/** Save the owner's preference for recording new API activity. */
export async function updateApiMonitorSettings(
  enabled: boolean,
): Promise<ApiMonitorSettings> {
  const response = await authFetch("/api/settings/api-monitor", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Failed to save API monitor settings"),
    );
  }
  return response.json();
}
