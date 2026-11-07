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

interface ApiOfflineDocsSettings {
  path: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_update: boolean;
  file: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  updated_at: string | null;
  release: string | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  current_release: string;
  error: string | null;
  refreshing: boolean;
}

function fromApi(value: ApiOfflineDocsSettings): OfflineDocsSettings {
  return {
    path: value.path,
    autoUpdate: value.auto_update,
    file: value.file,
    updatedAt: value.updated_at,
    release: value.release,
    currentRelease: value.current_release,
    error: value.error,
    refreshing: value.refreshing,
  };
}

export async function loadOfflineDocsSettings(): Promise<OfflineDocsSettings> {
  const response = await authFetch("/api/settings/offline-docs");
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Failed to load the offline docs settings"),
    );
  }
  return fromApi(await response.json());
}

/** Omitted fields stay as they are; `path: null` forgets the folder. */
export async function updateOfflineDocsSettings(update: {
  path?: string | null;
  autoUpdate?: boolean;
}): Promise<OfflineDocsSettings> {
  const body: Record<string, unknown> = {};
  if (update.path !== undefined) body.path = update.path;
  if (update.autoUpdate !== undefined) body.auto_update = update.autoUpdate;
  const response = await authFetch("/api/settings/offline-docs", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Failed to update the offline docs settings"),
    );
  }
  return fromApi(await response.json());
}

export async function refreshOfflineDocs(): Promise<OfflineDocsSettings> {
  const response = await authFetch("/api/settings/offline-docs/refresh", {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Failed to download the Unsloth docs"),
    );
  }
  return fromApi(await response.json());
}
