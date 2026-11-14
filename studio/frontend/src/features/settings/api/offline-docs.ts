// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export interface OfflineDocsSettings {
  path: string | null;
  autoUpdate: boolean;
  file: string | null;
  updatedAt: string | null;
  release: string | null;
  currentRelease: string;
  error: string | null;
  refreshing: boolean;
}

async function call(
  path: string,
  fallback: string,
  init?: RequestInit,
): Promise<OfflineDocsSettings> {
  const response = await authFetch(`/api/settings/offline-docs${path}`, init);
  if (!response.ok) {
    throw new Error(await readFastApiError(response, fallback));
  }
  const v = await response.json();
  return {
    path: v.path,
    autoUpdate: v.auto_update,
    file: v.file,
    updatedAt: v.updated_at,
    release: v.release,
    currentRelease: v.current_release,
    error: v.error,
    refreshing: v.refreshing,
  };
}

export const loadOfflineDocsSettings = () =>
  call("", "Failed to load the offline docs settings");

/** Omitted fields stay as they are; `path: null` forgets the folder. */
export const updateOfflineDocsSettings = (update: {
  path?: string | null;
  autoUpdate?: boolean;
}) =>
  call("", "Failed to update the offline docs settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: update.path, auto_update: update.autoUpdate }),
  });

export const refreshOfflineDocs = () =>
  call("/refresh", "Failed to download the Unsloth docs", { method: "POST" });
