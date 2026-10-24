// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { DesktopAppModelSettings } from "../components/desktop-app-run-settings-state";

const BASE = "/api/settings/claude-desktop";

// Mirrors routes/claude_desktop.py ClaudeDesktopStatus. The backend owns the switch: it runs the
// gateway Claude talks to, and switches Claude back on Disconnect, on quit and after a crash.
export type ClaudeDesktopStatus = {
  supported: boolean;
  installed: boolean;
  app_path: string | null;
  download_url: string;
  connected: boolean;
  // "cli" while `unsloth start claude --app` in a terminal holds Claude Desktop.
  held_by: string | null;
  held_by_pid: number | null;
  running: boolean;
  model: string | null;
  // Claude slot id → Studio model id, or FOLLOW_LOADED (":loaded") to follow the loaded model.
  mappings: Record<string, string>;
  // Claude's five model slots, in the order Claude lists them.
  routes: { id: string; name: string }[];
  // The request came straight from this computer (the desktop app, or a local browser).
  local: boolean;
};

export type ClaudeDesktopConnectStatus =
  | "connected"
  | "pending_restart"
  | "needs_restart";

export type ClaudeDesktopConnectResult = {
  status: ClaudeDesktopConnectStatus;
  claude: ClaudeDesktopStatus;
};

export type ClaudeDesktopDisconnectResult = {
  notes: string[];
  claude: ClaudeDesktopStatus;
};

// A backend without the route (an older Studio) or a non-owner session: hide the card.
export class ClaudeDesktopUnavailable extends Error {}

export async function fetchClaudeDesktopStatus(): Promise<ClaudeDesktopStatus> {
  const res = await authFetch(BASE);
  if (res.status === 404 || res.status === 403) {
    throw new ClaudeDesktopUnavailable(String(res.status));
  }
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load Claude Desktop status"),
    );
  }
  return (await res.json()) as ClaudeDesktopStatus;
}

/** restart: undefined asks first (status "needs_restart" changes nothing), true restarts an open
 *  Claude, false switches it for its next launch. */
export async function connectClaudeDesktop(
  restart?: boolean,
): Promise<ClaudeDesktopConnectResult> {
  const res = await authFetch(`${BASE}/connect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ restart: restart ?? null }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to connect Claude Desktop"),
    );
  }
  return (await res.json()) as ClaudeDesktopConnectResult;
}

export async function disconnectClaudeDesktop(): Promise<ClaudeDesktopDisconnectResult> {
  const res = await authFetch(`${BASE}/disconnect`, { method: "POST" });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to disconnect Claude Desktop"),
    );
  }
  return (await res.json()) as ClaudeDesktopDisconnectResult;
}

// One row of Studio's chat catalog (GET /models): loaded models first, then every downloaded one.
export type DesktopAppModel = {
  id: string;
  loaded: boolean;
  quant: string | null;
};

export async function fetchClaudeDesktopModels(): Promise<DesktopAppModel[]> {
  const res = await authFetch(`${BASE}/models`);
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Failed to load models"));
  }
  return (await res.json()) as DesktopAppModel[];
}

export type ClaudeDesktopMappingsResult = {
  status: "saved" | ClaudeDesktopConnectStatus;
  claude: ClaudeDesktopStatus;
};

/** Save which Studio model each Claude slot uses. restart works as for connect: undefined asks
 *  first when a connected Claude is open, true restarts (or starts) it, false waits. */
export async function setClaudeDesktopMappings(
  mappings: Record<string, string>,
  restart?: boolean,
): Promise<ClaudeDesktopMappingsResult> {
  const res = await authFetch(`${BASE}/mappings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mappings, restart: restart ?? null }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to save the Claude model choices"),
    );
  }
  return (await res.json()) as ClaudeDesktopMappingsResult;
}


/** Where a model's shared run settings live and what bounds them (trained context, MTP heads,
 *  the memory guard). Shared by the Claude Desktop and Codex Desktop cards. */
export async function fetchDesktopAppModelSettings(
  model: string,
): Promise<DesktopAppModelSettings> {
  const res = await authFetch(
    `${BASE}/model-settings?model=${encodeURIComponent(model)}`,
  );
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Failed to read the model's settings"));
  }
  return await res.json();
}

export type { DesktopAppModelSettings };
