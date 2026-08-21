// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

/** What Claude Code has on this computer, so the settings row can offer it or stay hidden. */
export type ClaudeImportStatus = {
  available: boolean;
  projects: number;
  chats: number;
};

export type ClaudeImportResult = {
  projects: number;
  chats: number;
  /** Conversations Studio had not seen before. Zero when everything was already imported. */
  newChats: number;
  messages: number;
  skipped: number;
  warnings: string[];
};

type ApiClaudeImportResult = {
  projects: number;
  chats: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  new_chats: number;
  messages: number;
  skipped: number;
  warnings: string[];
};

export async function loadClaudeImportStatus(): Promise<ClaudeImportStatus> {
  const res = await authFetch("/api/import/claude/status");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to read Claude Code data"),
    );
  }
  return res.json();
}

/** Import every Claude Code conversation. Reads the session files, so it can take a while. */
export async function importClaudeChats(): Promise<ClaudeImportResult> {
  const res = await authFetch("/api/import/claude", { method: "POST" });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Import from Claude Code failed"),
    );
  }
  const result: ApiClaudeImportResult = await res.json();
  return {
    projects: result.projects,
    chats: result.chats,
    newChats: result.new_chats,
    messages: result.messages,
    skipped: result.skipped,
    warnings: result.warnings,
  };
}
