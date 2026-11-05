// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

import { SettingsRouteAbsentError } from "./settings-route-absent";

const ROUTE = "/api/settings/trusted-tool-folders";

/** One folder per line; blank lines dropped. The backend resolves and validates each one. */
export function parseTrustedFolders(text: string): string[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

export async function loadTrustedToolFolders(fallbackMessage: string): Promise<string[]> {
  const res = await authFetch(ROUTE);
  // An older backend: hide the row instead of showing an unactionable error.
  if (res.status === 404) throw new SettingsRouteAbsentError(ROUTE);
  if (!res.ok) throw new Error(await readFastApiError(res, fallbackMessage));
  return ((await res.json()) as { folders: string[] }).folders;
}

export async function updateTrustedToolFolders(
  folders: string[],
  fallbackMessage: string,
): Promise<string[]> {
  const res = await authFetch(ROUTE, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ folders }),
  });
  if (!res.ok) throw new Error(await readFastApiError(res, fallbackMessage));
  return ((await res.json()) as { folders: string[] }).folders;
}
