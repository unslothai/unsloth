// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

const MCP_ACCESS_PATH = "/api/settings/mcp-access";

export type McpAccessSettings = {
  enabled: boolean;
  /** UNSLOTH_STUDIO_ENABLE_MCP=1 forces this on and disables the switch */
  forcedByEnv: boolean;
  /** backend-reported URL used as the snippet fallback */
  url: string;
};

type ApiMcpAccessSettings = {
  enabled: boolean;
  forced_by_env: boolean;
  url: string;
};

export function mcpAccessFromApi(
  settings: ApiMcpAccessSettings,
): McpAccessSettings {
  return {
    enabled: settings.enabled,
    forcedByEnv: settings.forced_by_env,
    url: settings.url,
  };
}

async function request(
  fallback: string,
  init?: RequestInit,
): Promise<McpAccessSettings> {
  const res = await authFetch(MCP_ACCESS_PATH, init);
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallback));
  }
  return mcpAccessFromApi(await res.json());
}

export function loadMcpAccess(): Promise<McpAccessSettings> {
  return request("Failed to load agent access settings");
}

export function updateMcpAccess(enabled: boolean): Promise<McpAccessSettings> {
  return request("Failed to update agent access settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
}
