// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type ExternalImportSource = "cursor" | "claude";

export const EXTERNAL_IMPORT_LABELS: Record<ExternalImportSource, string> = {
  cursor: "Cursor",
  claude: "Claude Code",
};

export type ExternalImportStatus = {
  available: boolean;
  projects: number;
  chats: number;
};

export type ExternalImportResult = {
  projects: number;
  chats: number;
  newChats: number;
  messages: number;
  skipped: number;
  warnings: string[];
};

type ApiExternalImportResult = Omit<ExternalImportResult, "newChats"> & {
  // biome-ignore lint/style/useNamingConvention: API schema
  new_chats: number;
};

export async function loadExternalImportStatus(
  source: ExternalImportSource,
): Promise<ExternalImportStatus> {
  const res = await authFetch(`/api/import/${source}/status`);
  if (!res.ok) {
    throw new Error(
      await readFastApiError(
        res,
        `Failed to read ${EXTERNAL_IMPORT_LABELS[source]} data`,
      ),
    );
  }
  return res.json();
}

export async function importExternalChats(
  source: ExternalImportSource,
): Promise<ExternalImportResult> {
  const res = await authFetch(`/api/import/${source}`, { method: "POST" });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(
        res,
        `Import from ${EXTERNAL_IMPORT_LABELS[source]} failed`,
      ),
    );
  }
  const { new_chats: newChats, ...rest }: ApiExternalImportResult =
    await res.json();
  return { ...rest, newChats };
}
