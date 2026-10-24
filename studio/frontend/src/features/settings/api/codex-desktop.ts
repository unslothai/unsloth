// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { DesktopAppModel } from "./claude-desktop";

const BASE = "/api/settings/codex-desktop";

// Mirrors routes/codex_desktop.py CodexDesktopStatus. Like Claude Desktop, the backend owns the
// switch and puts ~/.codex/config.toml back on Disconnect, on quit and after a crash.
export type CodexDesktopStatus = {
  supported: boolean;
  installed: boolean;
  app_path: string | null;
  download_url: string;
  connected: boolean;
  // "cli" while `unsloth start codex --app` in a terminal holds the Codex app.
  held_by: string | null;
  held_by_pid: number | null;
  running: boolean;
  // Codex's default model, and every model in its catalog, while connected.
  model: string | null;
  models: string[];
  // What Studio has loaded now, and the saved list (empty follows the loaded model).
  loaded_model: string | null;
  selected_models: string[];
  // How many models the list holds, as `ollama launch codex-app`.
  max_models: number;
  local: boolean;
};

export type CodexDesktopResult = {
  status: "connected" | "pending_restart" | "needs_restart" | "saved";
  codex: CodexDesktopStatus;
};

export type CodexDesktopDisconnectResult = {
  status: "disconnected" | "needs_restart";
  notes: string[];
  codex: CodexDesktopStatus;
};

export class CodexDesktopUnavailable extends Error {}

async function call<T>(
  path: string,
  method: string,
  body: unknown,
  fallback: string,
): Promise<T> {
  const res = await authFetch(`${BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, fallback));
  }
  return (await res.json()) as T;
}

export async function fetchCodexDesktopStatus(): Promise<CodexDesktopStatus> {
  const res = await authFetch(BASE);
  if (res.status === 404 || res.status === 403) {
    throw new CodexDesktopUnavailable(String(res.status));
  }
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load Codex Desktop status"),
    );
  }
  return (await res.json()) as CodexDesktopStatus;
}

export async function fetchCodexDesktopModels(): Promise<DesktopAppModel[]> {
  const res = await authFetch(`${BASE}/models`);
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Failed to load models"));
  }
  return (await res.json()) as DesktopAppModel[];
}

/** restart: undefined asks first when Codex is open, true restarts it, false switches it for its
 *  next launch. models: catalog ids (the first is Codex's default), [] to follow the loaded
 *  model, undefined for the saved list. */
export function connectCodexDesktop(
  restart?: boolean,
  models?: string[],
): Promise<CodexDesktopResult> {
  return call(
    "/connect",
    "POST",
    { restart: restart ?? null, models: models ?? null },
    "Failed to connect Codex Desktop",
  );
}

export function setCodexDesktopModels(
  models: string[],
  restart?: boolean,
): Promise<CodexDesktopResult> {
  return call(
    "/models",
    "PUT",
    { models, restart: restart ?? null },
    "Failed to change the Codex models",
  );
}

export function disconnectCodexDesktop(
  restart?: boolean,
): Promise<CodexDesktopDisconnectResult> {
  return call(
    "/disconnect",
    "POST",
    { restart: restart ?? null },
    "Failed to disconnect Codex Desktop",
  );
}
