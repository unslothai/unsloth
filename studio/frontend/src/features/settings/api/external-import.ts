// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export const EXTERNAL_IMPORT_SOURCES = ["cursor", "claude", "codex"] as const;

export type ExternalImportSource = (typeof EXTERNAL_IMPORT_SOURCES)[number];

export const EXTERNAL_IMPORT_LABELS: Record<ExternalImportSource, string> = {
  cursor: "Cursor",
  claude: "Claude Code",
  codex: "Codex",
};

export type ExternalImportStatus = {
  available: boolean;
  chats: number;
};

export type ExternalImportResult = {
  newChats: number;
  messages: number;
  warnings: string[];
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await authFetch(path, init);
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "External import failed"));
  }
  return res.json();
}

export function loadExternalImportStatus(
  source: ExternalImportSource,
): Promise<ExternalImportStatus> {
  return request(`/api/import/${source}/status`);
}

export async function importExternalChats(
  source: ExternalImportSource,
): Promise<ExternalImportResult> {
  const result = await request<{
    new_chats: number;
    messages: number;
    warnings: string[];
  }>(`/api/import/${source}`, { method: "POST" });
  return {
    newChats: result.new_chats,
    messages: result.messages,
    warnings: result.warnings,
  };
}
