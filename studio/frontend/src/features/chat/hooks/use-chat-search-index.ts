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
} from "../utils/mcp-tool-name";
import { attachmentsPastedText } from "../utils/pasted-text.ts";

export interface ChatSearchItem {
  type: "single" | "compare";
  id: string;
  title: string;
  // Lowercased title + user messages only (short); searched first.
  userSearchText: string;
  // Lowercased title plus every message; the fallback when user text matches nothing.
  // Prebuilt so filtering never re-lowercases per keystroke.
  searchText: string;
  createdAt: number;
  projectId?: string | null;
}

const THREAD_LIMIT = 200;
const SEARCH_REBUILD_DEBOUNCE_MS = 300;
// Past the dialog's 180ms exit, so releasing uncached rows never lands mid-animation.
const ROW_RELEASE_DELAY_MS = 300;

// Keys whose values are base64 image/audio payloads, not searchable text.
const BINARY_KEY = /b64|base64|^(images?|audio|video)$/i;

// Drop a trailing __MCP_IMAGES__ envelope only when it is the valid JSON image array the
// backend appended, so tool text merely mentioning the marker stays searchable.
function stripMcpImageSuffix(value: string): string {
  const marker = "\n__MCP_IMAGES__:";
  const idx = value.lastIndexOf(marker);
  if (idx === -1) return value;
  try {
    const images: unknown = JSON.parse(value.slice(idx + marker.length));
    if (
      Array.isArray(images) &&
      images.length > 0 &&
      images.every(
        (img) =>
          typeof img === "object" &&
          img !== null &&
          typeof (img as Record<string, unknown>).data === "string" &&
          typeof (img as Record<string, unknown>).mimeType === "string",
      )
    ) {
      return value.slice(0, idx);
    }
  } catch {
    // Not a valid envelope; leave the text intact.
  }
  return value;
}

// Readable text from tool args/results, dropping base64 image/audio blobs so they never
// bloat the index.
function searchableText(value: unknown, depth = 0): string {
  if (typeof value === "string") {
    let text = stripMcpImageSuffix(value);
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
    const out: string[] = [];
    for (const [k, v] of Object.entries(value)) {
      if (!BINARY_KEY.test(k)) out.push(searchableText(v, depth + 1));
    }
    return out.join(" ");
  }
  return "";
}

// Pull searchable text from a message: text, reasoning, tool calls, cited sources and
// pasted bodies.
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
        // Index the rendered "Server · tool" label too, so pasting it matches.
        const label =
          typeof p.toolName === "string"
            ? formatMcpToolName(p.toolName, mcpServer)
            : null;
        if (label) parts.push(label);
      }
      const args = searchableText(
        typeof p.argsText === "string" ? p.argsText : p.args,
      );
      if (args) parts.push(args);
      const result = searchableText(p.result);
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

// Exported for the bare-node cache harness: it must prove a failed read is not
// indistinguishable from a completed empty history.
export async function buildChatSearchIndex(): Promise<ChatSearchIndexBuild> {
  const active = (
    await listStoredChatThreads({ includeArchived: false })
  ).slice(0, THREAD_LIMIT);

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
        if (existing) existing.threadIds.push(t.id);
        continue;
      }
      seenPairs.add(t.pairId);
      itemThreadIds.set(t.pairId, {
        item: {
          type: "compare",
          id: t.pairId,
          title: t.title,
          createdAt: t.createdAt,
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
          projectId: t.projectId ?? null,
        },
        threadIds: [t.id],
      });
    }
  }

  const allThreadIds = Array.from(itemThreadIds.values()).flatMap(
    (e) => e.threadIds,
  );
  let messagesByThread = await batchListChatMessages(allThreadIds).catch(
    () => new Map<string, MessageRecord[]>(),
  );
  let complete = true;

  // Legacy-only chats can exist before server-side history import finishes. Fill only the
  // missing ids via the legacy path instead of one request per thread up front.
  const missingThreadIds = allThreadIds.filter(
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
    if (merged.length === 0) {
      continue;
    }
    merged.sort((a, b) => b.createdAt - a.createdAt);

    // Two tiers: user messages (short, searched first) and the full conversation incl. tool
    // calls, used when user text matches nothing.
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

// THREAD_LIMIT bounds rows, not bytes: a tool-heavy history would otherwise hold tens of
// megabytes behind a closed dialog. Past this the index is rebuilt on each open.
const MAX_CACHED_SEARCH_TEXT_CHARS = 4_000_000;

// Last built index, kept across opens so reopening paints the previous rows at once and
// revalidates in place instead of collapsing to the empty state.
let cachedIndex: ChatSearchItem[] | null = null;
let cachedIndexEpoch = -1;

// Scoped to the auth session: a web logout only navigates, and a second account must never
// open onto the previous user's chats.
function readCachedIndex(): ChatSearchItem[] | null {
  if (cachedIndexEpoch !== getAuthSessionEpoch()) {
    // The next account's history is unknown, so the previous one's hint cannot size it. -1 is
    // "nothing cached yet" and leaves this account's alone.
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

// Exported for tests, which drive the real bookkeeping rather than a stand-in.
export function writeCachedIndex(next: ChatSearchItem[] | null): void {
  cachedIndexEpoch = getAuthSessionEpoch();
  cachedIndex =
    next !== null && cachedSearchTextChars(next) > MAX_CACHED_SEARCH_TEXT_CHARS
      ? null
      : next;
  // A build answers outright. An invalidation only says the history changed: a remembered ROWS
  // answer still holds, an EMPTY one may be about to gain its first chat.
  if (next !== null) rememberChatSearchHasRows(next.length > 0);
  else if (chatSearchHadRows() === false) forgetChatSearchHasRows();
}

// A partial build is useful for this open but is not an answer about whether the history has
// rows. Recording [] after every message read failed would open the next page load
// compact and then grow mid-animation once connectivity recovered.
export function publishChatSearchBuild(
  build: ChatSearchIndexBuild,
): ChatSearchItem[] {
  if (build.complete) {
    writeCachedIndex(build.items);
  } else {
    cachedIndexEpoch = getAuthSessionEpoch();
    cachedIndex = null;
    // An incomplete build cannot preserve a previous empty answer: partial rows already disprove
    // it, while no rows only mean the history could not be read. A positive partial answer is
    // safe and keeps the next open from starting compact.
    if (build.items.length > 0) rememberChatSearchHasRows(true);
    else forgetChatSearchHasRows();
  }
  return build.items;
}

// Once a structural refresh has a deadline, stream chunks may join it but must not keep
// moving it. If the deadline already fired, a later chunk schedules the usual follow-up.
export function shouldPostponeSearchRebuild(
  structuralRebuildPending: boolean,
  event: Event,
): boolean {
  return !(structuralRebuildPending && isCoalescedHistoryEvent(event));
}

// An account change made elsewhere. Private: it reaches an open dialog's request sequence,
// which nothing outside can see.
const SEARCH_SESSION_CHANGED_EVENT = "unsloth-chat-search-session-changed";

// A history change in another tab or from an API client never reaches this document, so the
// cache would otherwise open onto rows that no longer exist.
if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    // An account switch elsewhere arrives as a storage write alone: the epoch and its events are
    // both this document's. The mark moves on a session boundary, not on an hourly refresh,
    // which must not cost a warm cache.
    if (
      event.key === AUTH_SESSION_MARK_KEY ||
      (event.key === AUTH_TOKEN_KEY && event.newValue === null)
    ) {
      writeCachedIndex(null);
      // Shared by every account on the origin, so it cannot answer for the new one.
      forgetChatSearchHasRows();
      // History first, for the sidebar. Then the account change, which the epoch check cannot see:
      // an open dialog restarts at once, superseding the rebuild just queued.
      window.dispatchEvent(new Event(CHAT_HISTORY_UPDATED_EVENT));
      window.dispatchEvent(new Event(SEARCH_SESSION_CHANGED_EVENT));
      return;
    }
    if (event.key !== CHAT_HISTORY_REVISION_KEY) return;
    writeCachedIndex(null);
    // Dropping the cache alone leaves an open dialog on pre-change rows with nothing scheduled.
    // Re-raised locally so every in-tab listener treats it as a local change.
    window.dispatchEvent(new Event(CHAT_HISTORY_UPDATED_EVENT));
  });
  // The hint outlives the page, so a logout takes it: otherwise the next account to sign in
  // after a reload is sized by the previous one's history.
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    forgetChatSearchHasRows();
  });
}

// Whether to size for rows, readable during render so the dialog picks a height before its
// opening paint. A built cache answers exactly; otherwise the last build's hint does.
// null means genuinely unknown.
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

  // Discarded in the opening render, not in the effect that rebuilds: that runs after the
  // commit, so the invalidated rows would paint first.
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
      // With nothing cached these rows are the last thing holding the conversation text. Released
      // after the exit, not during the closing render: the portal stays mounted for the
      // animation, and emptying it there is the teardown this dialog exists to avoid.
      let release: ReturnType<typeof setTimeout> | null = null;
      const scheduleRelease = () => {
        // Never postponed: a stream invalidates per chunk, and restarting would hold the index for
        // the whole generation.
        if (release !== null) return;
        release = setTimeout(() => {
          release = null;
          if (readCachedIndex() !== null) return;
          // Same reference when there is nothing to release, so no needless re-render.
          setItems((prev) => (prev.length > 0 ? [] : prev));
        }, ROW_RELEASE_DELAY_MS);
      };
      scheduleRelease();
      // History can change while closed, so drop the cache rather than reopening onto chats that
      // no longer exist. Only the cache: clearing state re-renders per streaming chunk.
      const invalidate = () => {
        writeCachedIndex(null);
        // The release above may already have run while the cache was still there.
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
    // Set by a history event, cleared once its rebuild lands, so a close in between knows the cached snapshot is stale.
    let rebuildPending = false;
    let structuralRebuildPending = false;

    const run = () => {
      const seq = ++requestSeqRef.current;
      // A build straddling a logout describes the account it started under.
      const epoch = getAuthSessionEpoch();
      // Only the first build has nothing to show; later ones refresh silently.
      if (readCachedIndex() === null) setLoading(true);
      buildChatSearchIndex()
        .then((build) => {
          // Drop out-of-order responses, and never repopulate a cache already dropped.
          if (cancelled || seq !== requestSeqRef.current) return;
          if (epoch !== getAuthSessionEpoch()) return;
          const result = publishChatSearchBuild(build);
          // A build older than the history event does not satisfy it, so the flag only clears once nothing is queued.
          if (debounceTimer === null) {
            rebuildPending = false;
            structuralRebuildPending = false;
          }
          setItems(result);
        })
        .catch(() => {
          if (cancelled || seq !== requestSeqRef.current) return;
          // A rebuild that failed leaves nothing fresher, and what is cached is the snapshot it was
          // called to replace: keeping it would offer a deleted chat as a live result.
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
      // retires a build that read the history before this change, which would else republish it
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
        // The structural deadline has been honored. Stream events arriving while this rebuild runs
        // return to quiet-window coalescing instead of forcing a rebuild every debounce interval.
        structuralRebuildPending = false;
        if (!cancelled) run();
      }, SEARCH_REBUILD_DEBOUNCE_MS);
    };

    // Not a history change: the rows on screen belong to whoever was signed in a moment ago, so
    // they go now. Rebuilding at once also advances the request sequence, retiring a build
    // still in flight for that account.
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
      // Closing cancels a queued rebuild, so the snapshot left behind is stale and must not survive
      // the next open. The rendered rows stay: clearing them tears the list down inside the exit.
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
