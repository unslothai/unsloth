// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AUTH_SESSION_CLEARED_EVENT,
  AUTH_SESSION_MARK_KEY,
  AUTH_TOKEN_KEY,
  getAuthSessionEpoch,
} from "@/features/auth";
import { useEffect, useRef, useState } from "react";
import {
  CHAT_HISTORY_REVISION_KEY,
  CHAT_HISTORY_UPDATED_EVENT,
  batchListChatMessages,
} from "../api/chat-api";
import { splitMcpImages } from "../api/mcp-images";
import { isMcpUiToolResult } from "../mcp-apps/mcp-ui";
import type { MessageRecord } from "../types";
import { isCoalescedHistoryEvent } from "../utils/chat-history-revision";
import {
  listStoredChatMessages,
  listStoredChatThreads,
} from "../utils/chat-history-storage";
import {
  chatSearchHadRows,
  forgetChatSearchHasRows,
  rememberChatSearchHasRows,
} from "../utils/chat-search-history-hint";
import {
  formatMcpToolName,
  mcpServerFromProvenance,
  mcpToolFromProvenance,
} from "../utils/mcp-tool-name";
import { attachmentsPastedText } from "../utils/pasted-text.ts";

export interface ChatSearchItem {
  type: "single" | "compare";
  id: string;
  title: string;
  userSearchText: string;
  // Prebuilt so filtering never re-lowercases per keystroke.
  searchText: string;
  createdAt: number;
  updatedAt?: number;
  projectId?: string | null;
  isFork?: boolean;
}

// Messages are indexed for the newest threads only; older chats match by title.
const THREAD_LIMIT = 200;
const SEARCH_REBUILD_DEBOUNCE_MS = 300;
// Past the dialog's 180ms exit, so releasing rows never lands mid-animation.
const ROW_RELEASE_DELAY_MS = 300;

const BINARY_KEY = /b64|base64|^(images?|audio|video)$/i;

function searchableText(value: unknown, depth = 0, toolName?: string): string {
  if (typeof value === "string") {
    let text = splitMcpImages(value).text;
    const cut = text.indexOf("\n__IMAGES__:");
    if (cut !== -1) text = text.slice(0, cut);
    return text
      .replace(/data:[^;,\s]+;base64,[A-Za-z0-9+/=]+/g, " ")
      .replace(/[A-Za-z0-9+/]{120,}={0,2}/g, " ");
  }
  if (value == null || depth > 4) return "";
  if (Array.isArray(value)) {
    return value.map((v) => searchableText(v, depth + 1)).join(" ");
  }
  if (typeof value === "object") {
    // Index a widget result by what was shown, not its up-to-1MB UI seed.
    if (depth === 0 && isMcpUiToolResult(value, toolName ?? "")) {
      return searchableText(value.text, 1);
    }
    const out: string[] = [];
    for (const [k, v] of Object.entries(value)) {
      if (!BINARY_KEY.test(k)) out.push(searchableText(v, depth + 1));
    }
    return out.join(" ");
  }
  return "";
}

function extractText(message: MessageRecord): string {
  const content = message.content;
  const pasted = attachmentsPastedText(message.attachments);
  if (!Array.isArray(content)) return pasted;
  const parts: string[] = [];
  if (pasted) parts.push(pasted);
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const p = part as Record<string, unknown>;
    if (
      (p.type === "text" || p.type === "reasoning") &&
      typeof p.text === "string"
    ) {
      parts.push(p.text);
    } else if (p.type === "thinking") {
      const t = typeof p.thinking === "string" ? p.thinking : p.text;
      if (typeof t === "string") parts.push(t);
    } else if (p.type === "tool-call") {
      if (typeof p.toolName === "string") parts.push(p.toolName);
      const mcpServer = mcpServerFromProvenance(p.provenance);
      if (mcpServer) {
        parts.push(mcpServer);
        const label =
          typeof p.toolName === "string"
            ? formatMcpToolName(
                p.toolName,
                mcpServer,
                mcpToolFromProvenance(p.provenance),
              )
            : null;
        if (label) parts.push(label);
      }
      const args = searchableText(
        typeof p.argsText === "string" ? p.argsText : p.args,
      );
      if (args) parts.push(args);
      const result = searchableText(
        p.result,
        0,
        typeof p.toolName === "string" ? p.toolName : undefined,
      );
      if (result) parts.push(result);
    } else if (p.type === "source") {
      for (const v of [p.title, p.url])
        if (typeof v === "string") parts.push(v);
    }
  }
  return parts.join(" ").replace(/\s+/g, " ").trim();
}

interface ChatSearchIndexBuild {
  items: ChatSearchItem[];
  complete: boolean;
}

// Exported for the bare-node cache harness.
export async function buildChatSearchIndex(): Promise<ChatSearchIndexBuild> {
  const active = await listStoredChatThreads({ includeArchived: false });

  const itemThreadIds = new Map<
    string,
    {
      item: Omit<ChatSearchItem, "searchText" | "userSearchText">;
      threadIds: string[];
    }
  >();
  const seenPairs = new Set<string>();

  for (const t of active) {
    if (t.pairId) {
      if (seenPairs.has(t.pairId)) {
        const existing = itemThreadIds.get(t.pairId);
        if (existing) {
          existing.threadIds.push(t.id);
          existing.item.updatedAt = Math.max(
            existing.item.updatedAt ?? 0,
            t.updatedAt ?? t.createdAt,
          );
        }
        continue;
      }
      seenPairs.add(t.pairId);
      itemThreadIds.set(t.pairId, {
        item: {
          type: "compare",
          id: t.pairId,
          title: t.title,
          createdAt: t.createdAt,
          updatedAt: t.updatedAt ?? t.createdAt,
          projectId: t.projectId ?? null,
        },
        threadIds: [t.id],
      });
    } else {
      itemThreadIds.set(t.id, {
        item: {
          type: "single",
          id: t.id,
          title: t.title,
          createdAt: t.createdAt,
          updatedAt: t.updatedAt ?? t.createdAt,
          projectId: t.projectId ?? null,
          isFork: Boolean(t.forkedFromThreadId),
        },
        threadIds: [t.id],
      });
    }
  }

  const loadedThreadIds = active.slice(0, THREAD_LIMIT).map((t) => t.id);
  let messagesByThread = await batchListChatMessages(loadedThreadIds).catch(
    () => new Map<string, MessageRecord[]>(),
  );
  let complete = true;

  // Legacy-only chats may predate server import; fill only missing ids via the legacy path.
  const missingThreadIds = loadedThreadIds.filter(
    (threadId) => !messagesByThread.has(threadId),
  );
  if (missingThreadIds.length > 0) {
    const legacyEntries = await Promise.all(
      missingThreadIds.map(
        async (threadId) =>
          [
            threadId,
            await listStoredChatMessages(threadId).catch(() => {
              complete = false;
              return [];
            }),
          ] as const,
      ),
    );
    messagesByThread = new Map(messagesByThread);
    for (const [threadId, messages] of legacyEntries) {
      messagesByThread.set(threadId, messages);
    }
  }

  const results: ChatSearchItem[] = [];
  for (const { item, threadIds } of itemThreadIds.values()) {
    const merged: MessageRecord[] = [];
    for (const tid of threadIds) {
      const arr = messagesByThread.get(tid);
      if (arr) merged.push(...arr);
    }
    if (
      merged.length === 0 &&
      threadIds.every((tid) => messagesByThread.has(tid))
    ) {
      continue;
    }
    merged.sort((a, b) => b.createdAt - a.createdAt);

    const userParts: string[] = [item.title];
    const allParts: string[] = [item.title];
    for (const m of merged) {
      const text = extractText(m);
      if (!text) continue;
      allParts.push(text);
      if (m.role === "user") userParts.push(text);
    }
    const userSearchText = userParts.join(" ").toLowerCase();
    const searchText = allParts.join(" ").toLowerCase();
    results.push({ ...item, userSearchText, searchText });
  }

  results.sort((a, b) => b.createdAt - a.createdAt);
  return { items: results, complete };
}

// Size cap in chars: tool-heavy histories would otherwise hold tens of MB behind a closed dialog.
const MAX_CACHED_SEARCH_TEXT_CHARS = 4_000_000;

let cachedIndex: ChatSearchItem[] | null = null;
let cachedIndexEpoch = -1;

// Scoped to the auth session: a second account must never see the previous user's chats.
function readCachedIndex(): ChatSearchItem[] | null {
  if (cachedIndexEpoch !== getAuthSessionEpoch()) {
    // -1 means nothing cached yet.
    if (cachedIndexEpoch !== -1) forgetChatSearchHasRows();
    cachedIndex = null;
    cachedIndexEpoch = getAuthSessionEpoch();
  }
  return cachedIndex;
}

function cachedSearchTextChars(items: ChatSearchItem[]): number {
  let total = 0;
  for (const item of items) total += item.searchText.length;
  return total;
}

export function writeCachedIndex(next: ChatSearchItem[] | null): void {
  cachedIndexEpoch = getAuthSessionEpoch();
  cachedIndex =
    next !== null && cachedSearchTextChars(next) > MAX_CACHED_SEARCH_TEXT_CHARS
      ? null
      : next;
  // An invalidation keeps a ROWS answer, but an EMPTY one may be about to gain a chat.
  if (next !== null) rememberChatSearchHasRows(next.length > 0);
  else if (chatSearchHadRows() === false) forgetChatSearchHasRows();
}

// A partial build is not an answer about whether history has rows.
export function publishChatSearchBuild(
  build: ChatSearchIndexBuild,
): ChatSearchItem[] {
  if (build.complete) {
    writeCachedIndex(build.items);
  } else {
    cachedIndexEpoch = getAuthSessionEpoch();
    cachedIndex = null;
    if (build.items.length > 0) rememberChatSearchHasRows(true);
    else forgetChatSearchHasRows();
  }
  return build.items;
}

// Stream chunks may join a structural refresh deadline but must not keep moving it.
export function shouldPostponeSearchRebuild(
  structuralRebuildPending: boolean,
  event: Event,
): boolean {
  return !(structuralRebuildPending && isCoalescedHistoryEvent(event));
}

const SEARCH_SESSION_CHANGED_EVENT = "unsloth-chat-search-session-changed";

// Other tabs and API clients change history without notifying this document.
if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    // An account switch elsewhere arrives only as a storage write.
    if (
      event.key === AUTH_SESSION_MARK_KEY ||
      (event.key === AUTH_TOKEN_KEY && event.newValue === null)
    ) {
      writeCachedIndex(null);
      forgetChatSearchHasRows();
      window.dispatchEvent(new Event(CHAT_HISTORY_UPDATED_EVENT));
      window.dispatchEvent(new Event(SEARCH_SESSION_CHANGED_EVENT));
      return;
    }
    if (event.key !== CHAT_HISTORY_REVISION_KEY) return;
    writeCachedIndex(null);
    window.dispatchEvent(new Event(CHAT_HISTORY_UPDATED_EVENT));
  });
  // The hint outlives the page, so logout clears it for the next account.
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    forgetChatSearchHasRows();
  });
}

// Readable during render so the dialog sizes before its opening paint. null = unknown.
export function chatSearchIndexHasRows(): boolean | null {
  const cached = readCachedIndex();
  if (cached !== null) return cached.length > 0;
  return chatSearchHadRows();
}

export function useChatSearchIndex(enabled: boolean): {
  items: ChatSearchItem[];
  loading: boolean;
} {
  const [items, setItems] = useState<ChatSearchItem[]>(
    () => readCachedIndex() ?? [],
  );
  const [loading, setLoading] = useState(false);
  const requestSeqRef = useRef(0);

  // Discard in the opening render: an effect runs after commit, painting stale rows first.
  const [wasEnabled, setWasEnabled] = useState(enabled);
  if (enabled !== wasEnabled) {
    setWasEnabled(enabled);
    if (enabled && readCachedIndex() === null) {
      if (items.length > 0) setItems([]);
      if (!loading) setLoading(true);
    }
  }

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      // Release after the exit animation, not in the closing render.
      let release: ReturnType<typeof setTimeout> | null = null;
      const scheduleRelease = () => {
        // Never postponed: a stream invalidates per chunk.
        if (release !== null) return;
        release = setTimeout(() => {
          release = null;
          if (readCachedIndex() !== null) return;
          setItems((prev) => (prev.length > 0 ? [] : prev));
        }, ROW_RELEASE_DELAY_MS);
      };
      scheduleRelease();
      // Drop only the cache: clearing state re-renders per streaming chunk.
      const invalidate = () => {
        writeCachedIndex(null);
        scheduleRelease();
      };
      window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, invalidate);
      return () => {
        if (release !== null) clearTimeout(release);
        window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, invalidate);
      };
    }
    let cancelled = false;
    let debounceTimer: ReturnType<typeof setTimeout> | null = null;
    let rebuildPending = false;
    let structuralRebuildPending = false;

    const run = () => {
      const seq = ++requestSeqRef.current;
      const epoch = getAuthSessionEpoch();
      if (readCachedIndex() === null) setLoading(true);
      buildChatSearchIndex()
        .then((build) => {
          if (cancelled || seq !== requestSeqRef.current) return;
          if (epoch !== getAuthSessionEpoch()) return;
          const result = publishChatSearchBuild(build);
          // A build older than the history event does not satisfy it.
          if (debounceTimer === null) {
            rebuildPending = false;
            structuralRebuildPending = false;
          }
          setItems(result);
        })
        .catch(() => {
          if (cancelled || seq !== requestSeqRef.current) return;
          // A failed rebuild must not keep the stale snapshot, which may list deleted chats.
          if (rebuildPending) writeCachedIndex(null);
          setItems(readCachedIndex() ?? []);
        })
        .finally(() => {
          if (cancelled || seq !== requestSeqRef.current) return;
          setLoading(false);
        });
    };

    const scheduleRebuild = (event: Event) => {
      rebuildPending = true;
      const structural = !isCoalescedHistoryEvent(event);
      if (structural) {
        structuralRebuildPending = true;
        requestSeqRef.current += 1;
      }
      if (
        debounceTimer !== null &&
        !shouldPostponeSearchRebuild(structuralRebuildPending, event)
      ) {
        return;
      }
      if (debounceTimer !== null) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        debounceTimer = null;
        structuralRebuildPending = false;
        if (!cancelled) run();
      }, SEARCH_REBUILD_DEBOUNCE_MS);
    };

    // Rows belong to the previous account: drop them and retire any in-flight build.
    const onSessionChanged = () => {
      if (cancelled) return;
      if (debounceTimer !== null) {
        clearTimeout(debounceTimer);
        debounceTimer = null;
      }
      setItems([]);
      run();
    };

    run();
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, scheduleRebuild);
    window.addEventListener(SEARCH_SESSION_CHANGED_EVENT, onSessionChanged);
    return () => {
      cancelled = true;
      if (debounceTimer !== null) clearTimeout(debounceTimer);
      // Closing cancels a queued rebuild, so drop the stale snapshot; rows stay for the exit.
      if (rebuildPending) writeCachedIndex(null);
      window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, scheduleRebuild);
      window.removeEventListener(
        SEARCH_SESSION_CHANGED_EVENT,
        onSessionChanged,
      );
    };
  }, [enabled]);

  return { items, loading };
}
