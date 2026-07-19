// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { buildStoredChatExport } from "./chat-history-storage";

export const buildChatExport = buildStoredChatExport;

function triggerJsonDownload(data: unknown, filename: string): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export async function downloadChatExport(): Promise<void> {
  const data = await buildChatExport();
  triggerJsonDownload(
    data,
    `unsloth-chats-${new Date().toISOString().slice(0, 10)}.json`,
  );
}

// Minimal views over the `unknown[]` export fields we filter on.
type ExportThreadView = {
  id?: string;
  archived?: boolean;
  projectId?: string | null;
};
type ExportMessageView = { threadId?: string };
type ExportProjectView = { id?: string };

// Same backup shape as downloadChatExport, restricted to archived threads,
// their messages and their projects. Returns the archived thread count.
export async function buildArchivedChatExport(): Promise<{
  data: Awaited<ReturnType<typeof buildChatExport>>;
  archivedCount: number;
}> {
  const full = await buildChatExport();
  const archivedThreads = (full.threads as ExportThreadView[]).filter(
    (thread) => thread.archived === true,
  );
  const archivedThreadIds = new Set(
    archivedThreads
      .map((thread) => thread.id)
      .filter((id): id is string => typeof id === "string"),
  );
  const messages = (full.messages as ExportMessageView[]).filter(
    (message) =>
      typeof message.threadId === "string" &&
      archivedThreadIds.has(message.threadId),
  );
  const referencedProjectIds = new Set(
    archivedThreads
      .map((thread) => thread.projectId)
      .filter((id): id is string => typeof id === "string"),
  );
  const projects = (full.projects as ExportProjectView[] | undefined)?.filter(
    (project) =>
      typeof project.id === "string" && referencedProjectIds.has(project.id),
  );
  return {
    data: {
      ...full,
      threadCount: archivedThreads.length,
      projects: projects ?? [],
      threads: archivedThreads as unknown[],
      messages: messages as unknown[],
    },
    archivedCount: archivedThreads.length,
  };
}

// Download only the archived chats. Returns how many were exported.
export async function downloadArchivedChatExport(): Promise<number> {
  const { data, archivedCount } = await buildArchivedChatExport();
  triggerJsonDownload(
    data,
    `unsloth-archived-chats-${new Date().toISOString().slice(0, 10)}.json`,
  );
  return archivedCount;
}
