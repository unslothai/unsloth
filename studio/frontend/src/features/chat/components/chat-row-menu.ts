// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What a chat row's menu is made of, wherever the row is drawn: the sidebar's rows and the
// Projects page's. Kept in one place because the folder probe below drifted once when two callers
// carried their own.

import { sandboxHasFiles } from "@/components/assistant-ui/sandbox-reveal";

import {
  CONVERSATION_MARKDOWN_FORMAT,
  CONVERSATION_MARKDOWN_LABEL,
} from "../utils/conversation-markdown";
import { allRecordedSandboxSessionIds } from "../utils/recorded-sandbox-session";
import type { SidebarItem } from "../hooks/use-chat-sidebar-items";
import { listStoredChatMessages } from "../utils/chat-history-storage";

export type ConversationExportFormat =
  | "raw-jsonl"
  | "messages-jsonl"
  | "csv"
  | "sharegpt-jsonl"
  | typeof CONVERSATION_MARKDOWN_FORMAT;

/** Built on call, not at module scope: the markdown label arrives through the feature barrel,
 *  and reading it while this module loads throws if the cycle re-enters first. */
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
  const exports = await import("../prompt-storage/prompt-storage-dialog");
  switch (format) {
    case "raw-jsonl":
      return exports.exportConversationRawJsonl(threadId);
    case "messages-jsonl":
      return exports.exportConversationMessagesJsonl(threadId);
    case "csv":
      return exports.exportConversationCsv(threadId);
    case "sharegpt-jsonl":
      return exports.exportConversationShareGPT(threadId);
    case CONVERSATION_MARKDOWN_FORMAT:
      return exports.exportConversationMarkdown(threadId);
    default: {
      // Exhaustive: a new format is a build error, not a menu item that does nothing.
      const unhandled: never = format;
      throw new Error(`Unhandled export format: ${String(unhandled)}`);
    }
  }
}

/** The threads behind a row: a comparison is two, everything else is itself. */
export function getSidebarItemThreadIds(item: SidebarItem): string[] {
  return item.threadIds?.length ? item.threadIds : [item.id];
}

/** A comparison is two threads with no single tip to fork from. */
export function canForkChatRow(item: SidebarItem): boolean {
  return item.type === "single";
}

/**
 * Forks a chat from its last message, which is what the thread's own Fork does from the message
 * it is on. Settings are settled first: the fork copies `settings_json` in its own transaction,
 * so an edit still in the debounce would be left out and the copy would open on the older modes.
 */
export async function forkChatRow(item: SidebarItem) {
  const { forkChatThread } = await import("../api/chat-api");
  const { settleThreadScopedSettingsForCopy } = await import(
    "../stores/chat-runtime-store"
  );
  const messages = await listStoredChatMessages(item.id);
  const last = messages[messages.length - 1];
  if (!last) throw new Error("This chat has no messages to fork.");
  await settleThreadScopedSettingsForCopy(item.id);
  return forkChatThread(item.id, {
    messageId: last.id,
    newThreadId: crypto.randomUUID(),
    createdAt: Date.now(),
  });
}

/** The sandbox sessions this chat's stored tool results name, if any. */
export async function recordedSandboxSessionIds(
  ids: string[],
): Promise<string[]> {
  const recorded: string[] = [];
  // Every id a thread names, not just its latest: one chat that ran a tool, moved between projects
  // and ran another wrote to two folders on its own, and the newest would answer for both. One at a
  // time, not Promise.all: app-sidebar's export contract forbids a concurrent await on this path.
  for (const threadId of ids) {
    recorded.push(
      ...allRecordedSandboxSessionIds(await listStoredChatMessages(threadId)),
    );
  }
  return [...new Set(recorded)];
}

/**
 * The folders this chat's files are actually in: what its tool results name, or, for a chat old
 * enough that they name nothing, what is on disk. Chats stored before results carried a session
 * recorded nothing, so one that ran loose and has since joined a project would be answered with
 * the project workspace; its thread sandbox is the only other candidate, and files there are this
 * chat's. The project workspace is not probed, because it belongs to every chat alike.
 *
 * A union rather than a fallback: one recorded id is not evidence that the others are recorded
 * too, and taking it alone would answer for both folders while hiding the older. Shared by "Open
 * chat folder" and "Copy session id", which drifted apart once, the copy path skipping the probe
 * and reporting success on a folder the chat had never written to.
 */
export async function sandboxSessionIdsHolding(
  ids: string[],
): Promise<string[]> {
  const recorded = await recordedSandboxSessionIds(ids);
  // Thread folders only. A project sandbox is shared by every chat in the project, so files there are
  // no evidence that THIS chat wrote them, and counting one would report a second folder for any chat
  // that joined a project someone else had used. Both callers already fall back to the folder
  // membership gives them when nothing here names one.
  const held: string[] = [];
  for (const candidate of ids) {
    // Already named, so there is nothing a probe could add.
    if (recorded.includes(candidate)) continue;
    if (await sandboxHasFiles(candidate)) held.push(candidate);
  }
  return [...new Set([...recorded, ...held])];
}
