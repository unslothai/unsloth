// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

/** What Cursor has on this computer, so the settings row can offer it or stay hidden. */
export type CursorImportStatus = {
  available: boolean;
  projects: number;
  chats: number;
};

export type CursorImportResult = {
  projects: number;
  chats: number;
  /** Conversations Studio had not seen before. Zero when everything was already imported. */
  newChats: number;
  messages: number;
  skipped: number;
  warnings: string[];
};

type ApiCursorImportResult = {
  projects: number;
  chats: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  new_chats: number;
  messages: number;
  skipped: number;
  warnings: string[];
};

export async function loadCursorImportStatus(): Promise<CursorImportStatus> {
  const res = await authFetch("/api/import/cursor/status");
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Failed to read Cursor data"));
  }
  return res.json();
}

/** Import every Cursor conversation. Reads hundreds of files, so it can take a while. */
export async function importCursorChats(): Promise<CursorImportResult> {
  const res = await authFetch("/api/import/cursor", { method: "POST" });
  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Import from Cursor failed"));
  }
  const result: ApiCursorImportResult = await res.json();
  return {
    projects: result.projects,
    chats: result.chats,
    newChats: result.new_chats,
    messages: result.messages,
    skipped: result.skipped,
    warnings: result.warnings,
  };
}
