// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Shared by sidebar and Projects rows: two copies of the folder probe drifted.

import { sandboxHasFiles } from "@/components/assistant-ui/sandbox-reveal";

import {
  CONVERSATION_MARKDOWN_FORMAT,
  CONVERSATION_MARKDOWN_LABEL,
} from "../utils/conversation-markdown";
import { allRecordedSandboxSessionIds } from "../utils/recorded-sandbox-session";
import { liveThreadBranch } from "../utils/live-thread-head";
import { forkChatThread } from "../api/chat-api";
import { settleThreadScopedSettingsForCopy } from "../stores/chat-runtime-store";
import type { SidebarItem } from "../hooks/use-chat-sidebar-items";
import {
  exportConversationCsv,
  exportConversationMarkdown,
  exportConversationMessagesJsonl,
  exportConversationRawJsonl,
  exportConversationShareGPT,
} from "../prompt-storage/prompt-storage-dialog";
import { listStoredChatMessages } from "../utils/chat-history-storage";

export type ConversationExportFormat =
  | "raw-jsonl"
  | "messages-jsonl"
  | "csv"
  | "sharegpt-jsonl"
  | typeof CONVERSATION_MARKDOWN_FORMAT;

/** Built on call: reading the barrel's label at module load throws on the import cycle. */
export function chatExportOptions(): Array<{
  label: string;
  format: ConversationExportFormat;
}> {
  return [
    { label: "Training JSONL", format: "raw-jsonl" },
    { label: "Message JSONL", format: "messages-jsonl" },
    { label: "CSV", format: "csv" },
    { label: "ShareGPT JSONL", format: "sharegpt-jsonl" },
    { label: CONVERSATION_MARKDOWN_LABEL, format: CONVERSATION_MARKDOWN_FORMAT },
  ];
}

export async function exportConversationByFormat(
  threadId: string,
  format: ConversationExportFormat,
): Promise<void> {
  switch (format) {
    case "raw-jsonl":
      return exportConversationRawJsonl(threadId);
    case "messages-jsonl":
      return exportConversationMessagesJsonl(threadId);
    case "csv":
      return exportConversationCsv(threadId);
    case "sharegpt-jsonl":
      return exportConversationShareGPT(threadId);
    case CONVERSATION_MARKDOWN_FORMAT:
      return exportConversationMarkdown(threadId);
    default: {
      const unhandled: never = format;
      throw new Error(`Unhandled export format: ${String(unhandled)}`);
    }
  }
}

export function getSidebarItemThreadIds(item: SidebarItem): string[] {
  return item.threadIds?.length ? item.threadIds : [item.id];
}

export function canForkChatRow(item: SidebarItem): boolean {
  return item.type === "single";
}

/** Settles settings first: the fork copies settings_json, so a debounced edit would be lost. */
export async function forkChatRow(item: SidebarItem) {
  const messageId = liveThreadBranch(item.id)?.at(-1);
  await settleThreadScopedSettingsForCopy(item.id);
  try {
    // closed chats use the transaction-selected tip; open chats keep the branch visible at invocation.
    return await forkChatThread(item.id, {
      messageId,
      newThreadId: crypto.randomUUID(),
      createdAt: Date.now(),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "";
    if (message.includes("still generating")) throw forkRefused();
    throw error;
  }
}

function forkRefused(): Error {
  return Object.assign(
    new Error("This chat is still generating. Fork it once it finishes."),
    { unslothForkRefused: true },
  );
}

export async function recordedSandboxSessionIds(
  ids: string[],
): Promise<string[]> {
  const recorded: string[] = [];
  // Every id, not the latest: a chat moved between projects wrote to two folders. Sequential, not
  // Promise.all: app-sidebar's export contract forbids a concurrent await here.
  for (const threadId of ids) {
    recorded.push(
      ...allRecordedSandboxSessionIds(await listStoredChatMessages(threadId)),
    );
  }
  return [...new Set(recorded)];
}

/**
 * The folders this chat's files are in: its tool results' sessions, plus the thread sandbox
 * probed on disk for chats too old to record one. A union, not a fallback; the project workspace
 * is never probed since every chat shares it.
 */
export async function sandboxSessionIdsHolding(
  ids: string[],
): Promise<string[]> {
  const recorded = await recordedSandboxSessionIds(ids);
  // Thread folders only: a shared project sandbox is no evidence this chat wrote there.
  const held: string[] = [];
  for (const candidate of ids) {
    if (recorded.includes(candidate)) continue;
    if (await sandboxHasFiles(candidate)) held.push(candidate);
  }
  return [...new Set([...recorded, ...held])];
}
