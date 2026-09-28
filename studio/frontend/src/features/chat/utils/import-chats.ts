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

export const CHAT_IMPORT_ACCEPT = ".json,.jsonl,.ndjson,.csv";

/** Where imported chats land: a project (null for none), optionally filed in a section. */
export interface ChatImportTarget {
  projectId: string | null;
  sectionId?: string;
  /** Named in the toast; "Recents" when there is no project or section. */
  name?: string;
}

// One import at a time, whichever menu started it.
let running = false;

/** Imports a file, counting up in a toast; the chats are filed in the target's section. */
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
      // Nothing was created, so however the count is phrased this is a failure.
      toast.error("Import failed.", {
        id: toastId,
        description: `${failed} conversation${failed === 1 ? "" : "s"} could not be saved.`,
      });
      return;
    }
    if (target.sectionId) {
      useSidebarOrganizationStore.getState().setChatsSection(threadIds, target.sectionId);
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

/** Asks for a file (the native picker in the desktop app), then imports it into the target. */
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
