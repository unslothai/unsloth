// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

const MCP_ACCESS_PATH = "/api/settings/mcp-access";

export type McpAccessSettings = {
  enabled: boolean;
  /** UNSLOTH_STUDIO_ENABLE_MCP=1 turns it on and the switch cannot change it */
  forcedByEnv: boolean;
  /** what the server saw as its own address; display only, the snippet builds its own */
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

export async function loadMcpAccess(): Promise<McpAccessSettings> {
  const res = await authFetch(MCP_ACCESS_PATH);
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load agent access settings"),
    );
  }
  return mcpAccessFromApi(await res.json());
}

export async function updateMcpAccess(
  enabled: boolean,
): Promise<McpAccessSettings> {
  const res = await authFetch(MCP_ACCESS_PATH, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to update agent access settings"),
    );
  }
  return mcpAccessFromApi(await res.json());
}
