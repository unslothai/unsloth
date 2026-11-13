// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { pickNativeChatImport } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { useSidebarOrganizationStore } from "../stores/sidebar-organization-store";
import {
  type ImportSource,
  fileImportSource,
  importConversationsFromSource,
  nativeImportSource,
} from "./chat-import";

export const CHAT_IMPORT_ACCEPT = ".json,.jsonl,.ndjson,.csv,.md,.markdown";

export interface ChatImportTarget {
  projectId: string | null;
  sectionId?: string;
  name?: string;
}

let running = false;

export async function runChatImport(source: ImportSource, target: ChatImportTarget): Promise<void> {
  running = true;
  const toastId = toast.loading("Importing chats...");
  try {
    const threadIds: string[] = [];
    const { imported, failed } = await importConversationsFromSource(
      source,
      target.projectId,
      {
        onSaved: (threadId) => threadIds.push(threadId),
        onProgress: ({ imported: done, bytesRead, totalBytes }) => {
          const percent = totalBytes
            ? Math.min(100, Math.round((bytesRead / totalBytes) * 100))
            : 0;
          toast.loading(`Importing chats: ${done} so far (${percent}%)...`, { id: toastId });
        },
      },
    );
    if (imported === 0 && failed === 0) {
      toast.info("No conversations found in file.", { id: toastId });
      return;
    }
    if (imported === 0) {
      toast.error("Import failed.", {
        id: toastId,
        description: `${failed} conversation${failed === 1 ? "" : "s"} could not be saved.`,
      });
      return;
    }
    if (target.sectionId) {
      useSidebarOrganizationStore.getState().setChatsSection([...new Set(threadIds)], target.sectionId);
    }
    const dest = target.name ?? "Recents";
    toast.success(
      failed > 0
        ? `Imported ${imported} conversation${imported === 1 ? "" : "s"} to ${dest}; ${failed} could not be saved.`
        : `Imported ${imported} conversation${imported === 1 ? "" : "s"} to ${dest}.`,
      { id: toastId },
    );
  } catch (error) {
    toast.error("Import failed.", {
      id: toastId,
      description: error instanceof Error ? error.message : undefined,
    });
  } finally {
    running = false;
  }
}

export function isChatImportRunning(): boolean {
  return running;
}

export async function pickAndImportChats(target: ChatImportTarget): Promise<void> {
  if (running) return;
  if (!isTauri) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = CHAT_IMPORT_ACCEPT;
    input.onchange = () => {
      const file = input.files?.[0];
      if (file) void runChatImport(fileImportSource(file), target);
    };
    input.click();
    return;
  }
  try {
    const selected = await pickNativeChatImport();
    if (selected) await runChatImport(nativeImportSource(selected), target);
  } catch (error) {
    toast.error("Import failed.", {
      description: error instanceof Error ? error.message : String(error),
    });
  }
}
