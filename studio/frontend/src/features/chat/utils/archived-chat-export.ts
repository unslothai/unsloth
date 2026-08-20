// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Minimal views over the `unknown[]` export fields we filter on.
type ExportThreadView = {
  id?: string;
  archived?: boolean;
  projectId?: string | null;
};
type ExportMessageView = { threadId?: string };
type ExportProjectView = { id?: string };

// Full chat-export backup shape, kept structural so the pure filter below
// stays decoupled from the storage layer that produces it.
export interface ChatExportData {
  exportedAt?: string;
  version?: number;
  threadCount: number;
  projects?: unknown[];
  threads: unknown[];
  messages: unknown[];
}

// Restrict a full chat export to archived threads, their messages and the
// projects those threads belong to. Pure: never mutates the input, and keeps
// the original thread/message objects so the backup re-imports unchanged.
export function filterArchivedChatExport<T extends ChatExportData>(
  full: T,
): { data: T; archivedCount: number } {
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
