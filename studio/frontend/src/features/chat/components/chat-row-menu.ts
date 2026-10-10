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
import { liveThreadBranch } from "../utils/live-thread-head";
import {
  forkChatThread,
  getInferenceStatus,
  streamChatCompletions,
  updateChatThread,
} from "../api/chat-api";
import { parseExternalModelId } from "../external-providers";
import { normalizeModelIdentity } from "../../hub/lib/model-identity";
import {
  settleThreadScopedSettingsForCopy,
  useChatRuntimeStore,
} from "../stores/chat-runtime-store";
import type { SidebarItem } from "../hooks/use-chat-sidebar-items";
import {
  exportConversationCsv,
  exportConversationMarkdown,
  exportConversationMessagesJsonl,
  exportConversationRawJsonl,
  exportConversationShareGPT,
} from "../prompt-storage/prompt-storage-dialog";
import {
  getStoredChatThread,
  listStoredChatMessages,
  listStoredChatThreads,
} from "../utils/chat-history-storage";
import { savedBranchHead } from "../utils/branch-head";
import { orderByParentChain } from "../utils/message-order";
import {
  buildTitleRefreshRequest,
  heuristicChatTitle,
  titleFromStream,
  titleRefreshExcerpt,
} from "../utils/chat-title";

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

/** Tagged, so the caller says this rather than reporting a failure that did not happen. */
function forkRefused(): Error {
  return Object.assign(
    new Error("This chat is still generating. Fork it once it finishes."),
    { unslothForkRefused: true },
  );
}

export type RegenerateTitleOutcome =
  | "renamed"
  | "unchanged"
  | "empty"
  | "busy"
  | "failed";

const regeneratingTitles = new Set<string>();
// Past this, the title is picked from the messages instead.
const TITLE_MODEL_WAIT_MS = 15_000;

async function titleFromModel(checkpoint: string, excerpt: string): Promise<string | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TITLE_MODEL_WAIT_MS);
  try {
    const request = await buildTitleRefreshRequest(checkpoint, excerpt);
    if (!request) return null;
    return await titleFromStream(streamChatCompletions(request, controller.signal));
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/** A local model is asked only while it is serving: a request naming an idle-unloaded one makes the
 *  backend load it again, multi-GB for a few words. */
async function titleModelServing(checkpoint: string): Promise<boolean> {
  if (parseExternalModelId(checkpoint) !== null) return true;
  try {
    const status = await getInferenceStatus();
    if (!status.active_model || status.is_audio || status.is_diffusion) return false;
    const want = normalizeModelIdentity(checkpoint);
    return [status.model_identifier, status.active_model, ...(status.serving_checkpoints ?? [])].some(
      (id) => !!id && (id === checkpoint || normalizeModelIdentity(id) === want),
    );
  } catch {
    return false;
  }
}

/** Uses the selected model, never the one that answered, and only while it is serving; else the
 *  messages. A comparison's panes share user turns, so one is read. */
export async function regenerateChatTitle(
  item: SidebarItem,
): Promise<RegenerateTitleOutcome> {
  if (regeneratingTitles.has(item.id)) return "busy";
  regeneratingTitles.add(item.id);
  try {
    const threadId = getSidebarItemThreadIds(item)[0];
    const liveBranch = liveThreadBranch(threadId);
    const { params, modelLoading } = useChatRuntimeStore.getState();
    const checkpoint = !modelLoading ? params.checkpoint : "";
    const [startTitle, raw, serving] = await Promise.all([
      getStoredChatThread(threadId).then((thread) => thread?.title),
      listStoredChatMessages(threadId),
      checkpoint ? titleModelServing(checkpoint) : false,
    ]);
    // The branch on screen, else the one reopening the chat shows, as the exports read it.
    const storedIds = new Set(raw.map((m) => m.id));
    const headId = liveBranch?.length
      ? ([...liveBranch].reverse().find((id) => storedIds.has(id)) ?? null)
      : savedBranchHead(threadId, raw);
    const branch = raw.some((m) => m.parentId != null)
      ? orderByParentChain(raw, { includeSiblings: false, headId })
      : raw;
    const excerpt = titleRefreshExcerpt(branch);
    if (!excerpt) return "empty";
    const title =
      (serving ? await titleFromModel(checkpoint, excerpt) : null) ?? heuristicChatTitle(branch);
    if (!title) return "empty";
    if (title === startTitle) return "unchanged";
    const ids =
      item.type === "single"
        ? [threadId]
        : [...new Set((await listStoredChatThreads({ pairId: item.id, includeArchived: true })).map((t) => t.id))];
    try {
      // Guarded: a rename that lands while the title is generated wins (409).
      await Promise.all(
        ids.map((id) =>
          updateChatThread(id, { title }, startTitle === undefined ? {} : { expectedTitle: startTitle }),
        ),
      );
    } catch {
      return (await getStoredChatThread(threadId))?.title !== startTitle ? "unchanged" : "failed";
    }
    return "renamed";
  } catch {
    return "failed";
  } finally {
    regeneratingTitles.delete(item.id);
  }
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
