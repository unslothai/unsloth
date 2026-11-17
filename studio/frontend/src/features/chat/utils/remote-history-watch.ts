// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  CHAT_HISTORY_UPDATED_EVENT,
  type ChatHistoryUpdatedDetail,
  getChatHistoryRevision,
} from "../api/chat-api";
import {
  REMOTE_HISTORY_POLL_MS,
  type RemoteHistorySync,
  type RemoteHistoryWatcher,
  createRemoteHistorySync,
} from "./remote-history-sync";

let sync: RemoteHistorySync | null = null;
let timer: ReturnType<typeof setInterval> | null = null;
let detach: (() => void) | null = null;
let lastLocalEventAt = Number.NEGATIVE_INFINITY;
let dispatching = false;

function dispatchRemoteChange(): void {
  // This tab's own writes move the counter too; while it is busy, a chunk-style event is enough.
  const coalesce = Date.now() - lastLocalEventAt < 2 * REMOTE_HISTORY_POLL_MS;
  dispatching = true;
  try {
    window.dispatchEvent(
      new CustomEvent<ChatHistoryUpdatedDetail>(CHAT_HISTORY_UPDATED_EVENT, {
        detail: { coalesce },
      }),
    );
  } finally {
    dispatching = false;
  }
}

function start(current: RemoteHistorySync): void {
  const poll = () => void current.poll();
  const onVisible = () => {
    if (document.visibilityState === "visible") poll();
  };
  const onLocal = () => {
    if (!dispatching) lastLocalEventAt = Date.now();
  };
  window.addEventListener("focus", poll);
  window.addEventListener("online", poll);
  window.addEventListener("pageshow", poll);
  window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, onLocal);
  document.addEventListener("visibilitychange", onVisible);
  timer = setInterval(poll, REMOTE_HISTORY_POLL_MS);
  detach = () => {
    window.removeEventListener("focus", poll);
    window.removeEventListener("online", poll);
    window.removeEventListener("pageshow", poll);
    window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, onLocal);
    document.removeEventListener("visibilitychange", onVisible);
    if (timer !== null) clearInterval(timer);
    timer = null;
  };
  poll();
}

/** Picks up chat writes made on another device: the sidebar refreshes, the open thread re-reads. */
export function watchRemoteHistory(watcher: RemoteHistoryWatcher): () => void {
  if (typeof window === "undefined" || typeof document === "undefined") {
    return () => {};
  }
  sync ??= createRemoteHistorySync({
    fetchRevision: (threadIds) => getChatHistoryRevision(threadIds),
    onHistoryChanged: dispatchRemoteChange,
    isVisible: () => document.visibilityState !== "hidden",
  });
  const current = sync;
  const unwatch = current.watch(watcher);
  if (detach === null) start(current);
  return () => {
    unwatch();
    if (!current.watching()) {
      detach?.();
      detach = null;
    }
  };
}
